"""S7 acceptance (plan 10.1): strategy-interface tests green, and a precision
sanity check on a synthetic corpus with its manifest recorded (no threshold
claimed - the number is recorded, the target is set at S9). Plus: the
quotation bound is enforced and bound-suppressed proposals are surfaced
never dropped (NFR5/NFR9); near-duplicates are grouped never merged away;
extracted candidates are valid staging candidates and travel to the gate;
the run manifest is an NFR2 record; the run is deterministic.
"""

import hashlib

import pytest

from rag4ds.extract.strategy import (
    MIN_TOKENISABLE_FRACTION,
    QUOTATION_MAX_CJK_CHARS,
    QUOTATION_MAX_SOURCE_CHARS,
    QUOTATION_MAX_WORDS,
    SEGMENTED_SCRIPTS,
    Document,
    ExtractError,
    Proposal,
    build_entry_candidate,
    cluster_near_duplicates,
    longest_verbatim_run,
    longest_verbatim_runs,
    longest_verbatim_source_chars,
    measure_precision,
    is_unsegmented,
    quotation_bound_check,
    run_extraction,
    tokenisable_fraction,
    tokenise,
    within_quotation_bound,
)
from rag4ds.extract.strategies import EmbeddingStrategy, RuleStrategy
from rag4ds.staging.store import StagingStore, validate_candidate

def _head_commit() -> str:
    """The commit of the code under test, for records that name it (F11; the
    same rule the S6 e2e artefact follows). Falls back loudly, never silently
    to a stale constant."""
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover - git present
        raise RuntimeError(f"git HEAD unavailable for the record's code_commit: {exc}") from exc


COMMIT = _head_commit()
TS = "2026-08-23 20:10 JST"

SYNTH = (
    "Robotics maintenance notes.\n"
    "Bearing wear is the dominant failure mode of rotary actuators.\n"
    "Torque signature: a rising torque trend that precedes bearing failure.\n"
    "Backlash refers to lost motion in a geartrain.\n"
    "The weather was fine and the team went home early.\n"
    "An encoder is a sensor that reports shaft angle.\n"
)
# gold definienda a correct concept candidate should name (synthetic oracle)
GOLD_TERMS = {"bearing wear", "torque signature", "backlash", "encoder"}


def doc(text=SYNTH, pack="robotics", layer="domain"):
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Document(source_hash=h, text=text, pack=pack, layer=layer,
                    source_reference="corpus:robotics/notes#1")


class FAKE_Embedder:  # noqa: N801 - a clearly named test double (charter section 8)
    model_id = "FAKE-embed"
    revision = "test-double"

    def embed(self, texts):
        import numpy as np
        out = np.zeros((len(texts), 512), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in set(__import__("re").findall(r"\w+", t.lower())):
                out[i, int(hashlib.sha256(w.encode()).hexdigest(), 16) % 512] += 1.0
            if not out[i].any():
                out[i, 0] = 1.0
        return out


# --- the quotation bound (NFR5) ---


def test_longest_verbatim_run_and_bound():
    src = "the quick brown fox jumps over the lazy dog and runs away fast today now"
    assert longest_verbatim_run("quick brown fox", src) == 3
    assert longest_verbatim_run("a quick brown fox indeed", src) == 3  # non-contiguous breaks
    assert longest_verbatim_run("nothing shared here zz", src) == 0
    assert within_quotation_bound("Fox - a small mammal", src) is True
    passage = "the quick brown fox jumps over the lazy dog and runs"
    assert longest_verbatim_run(passage, src) > QUOTATION_MAX_WORDS
    assert within_quotation_bound(passage, src) is False


def test_run_suppresses_bound_violations_and_records_them():
    class PassageStrategy:
        name = "passage"

        def describe(self):
            return {"name": self.name, "model": None, "params": {}}

        def propose(self, document):
            long = " ".join(document.text.split()[:QUOTATION_MAX_WORDS + 4])
            return [Proposal("concept", long, 0, len(long) + 5)]

    result = run_extraction(doc(), [PassageStrategy()], COMMIT, TS)
    assert result.candidates == []
    assert len(result.suppressed) == 1
    sup = result.suppressed[0]
    assert sup["strategy"] == "passage" and sup["word_run"] > QUOTATION_MAX_WORDS
    assert result.manifest["candidates_suppressed_for_bound"] == 1


# --- the rule strategy ---


def test_rule_strategy_proposes_valid_within_bound_candidates(tmp_path):
    result = run_extraction(doc(), [RuleStrategy()], COMMIT, TS)
    assert result.candidates, "the rule strategy found no definitions"
    terms = {c["proposed"]["content"].split(" - ")[0].lower() for c in result.candidates}
    assert {"bearing wear", "torque signature", "backlash", "encoder"} <= terms
    # the non-definitional weather sentence is not proposed
    assert all("weather" not in c["proposed"]["content"].lower() for c in result.candidates)
    for c in result.candidates:
        validate_candidate(c)  # a real staging candidate
        assert c["extractor"] == "rule-v1" and c["source_hash"] == doc().source_hash
        assert within_quotation_bound(c["proposed"]["content"], SYNTH)
        # the span points at real text of the right length
        assert SYNTH[c["span_start"]:c["span_end"]].strip()
    # and they file into staging and reach the gate's queue
    store = StagingStore(tmp_path / "stage.jsonl")
    known = {doc().source_hash}
    for c in result.candidates:
        store.file_candidate(c, TS, known_hashes=known)
    assert all(store.state_of(cid) == "PENDING" for cid in store.candidates())


def test_rule_strategy_span_is_exact_offsets():
    result = run_extraction(doc(), [RuleStrategy()], COMMIT, TS)
    c = next(c for c in result.candidates
             if c["proposed"]["content"].lower().startswith("encoder"))
    assert SYNTH[c["span_start"]:c["span_end"]] == (
        "An encoder is a sensor that reports shaft angle.")


# --- near-duplicate grouping (6.2 / NFR9) ---


def test_near_duplicates_are_grouped_not_dropped():
    text = (
        "Bearing wear is the main failure mode of actuators.\n"
        "Bearing wear is the main failure mode of actuators indeed.\n"
        "Backlash refers to lost motion in a geartrain.\n"
    )
    result = run_extraction(doc(text), [RuleStrategy()], COMMIT, TS, embedder=FAKE_Embedder(),
                            near_duplicate_cosine=0.8)
    # nothing dropped: both bearing-wear sentences are candidates
    bearing = [i for i, c in enumerate(result.candidates)
               if c["proposed"]["content"].lower().startswith("bearing wear")]
    assert len(bearing) == 2
    # and they are grouped as a cluster
    assert any(set(bearing) <= set(cl) for cl in result.clusters)
    assert result.manifest["clusters"] == result.clusters


def test_cluster_near_duplicates_is_deterministic_and_unit():
    contents = ["alpha beta gamma", "alpha beta gamma delta", "wholly different words here"]
    a = cluster_near_duplicates(contents, FAKE_Embedder(), threshold=0.5)
    b = cluster_near_duplicates(contents, FAKE_Embedder(), threshold=0.5)
    assert a == b
    assert cluster_near_duplicates(contents, None) == []  # no embedder -> no grouping


def test_group_head_and_members_file_through_staging(tmp_path):
    text = ("Bearing wear is the main failure mode of actuators.\n"
            "Bearing wear is the primary failure mode of actuators.\n")
    result = run_extraction(doc(text), [RuleStrategy()], COMMIT, TS, embedder=FAKE_Embedder(),
                            near_duplicate_cosine=0.8)
    assert result.clusters, "expected a near-duplicate cluster"
    store = StagingStore(tmp_path / "stage.jsonl")
    ids = [store.file_candidate(c, TS) for c in result.candidates]
    cluster = result.clusters[0]
    head = ids[cluster[0]]
    for m in cluster[1:]:
        store.transition(ids[m], "GROUPED", TS, "extractor", f"near-duplicate of {head}", head=head)
    assert store.members_of(head) == [ids[m] for m in cluster[1:]]


# --- the embedding strategy ---


def test_embedding_strategy_proposes_related_and_honours_seeds():
    seeds = ["failure modes of rotary actuators"]
    strat = EmbeddingStrategy(FAKE_Embedder(), seeds, top_k=2, floor=0.0)
    result = run_extraction(doc(), [strat], COMMIT, TS, embedder=FAKE_Embedder())
    assert result.candidates and all(
        c["proposed"]["content"].startswith("related concept:") for c in result.candidates)
    for c in result.candidates:
        validate_candidate(c)
        assert within_quotation_bound(c["proposed"]["content"], SYNTH)
    strat_names = [d["name"] for d in result.manifest["strategies"]]
    assert strat_names == ["embed-v1"]
    assert result.manifest["strategies"][0]["model"] == {"id": "FAKE-embed",
                                                         "revision": "test-double"}
    # no seeds -> nothing proposed
    empty = run_extraction(doc(), [EmbeddingStrategy(FAKE_Embedder(), [])], COMMIT, TS)
    assert empty.candidates == []


# --- the precision sanity check (S7 exit) ---


def test_precision_measured_and_recorded_no_threshold(tmp_path):
    from rag4ds.auditlog.journal import Journal

    result = run_extraction(doc(), [RuleStrategy()], COMMIT, TS)

    def is_correct(cand):
        return cand["proposed"]["content"].split(" - ")[0].lower() in GOLD_TERMS

    precision = measure_precision(result, is_correct)
    assert precision["proposed"] >= 4 and precision["correct"] >= 3
    assert 0.0 <= precision["precision"] <= 1.0  # a real number, recorded, not asserted against
    # the number and the run manifest are filed (NFR2), no threshold applied
    runs = Journal(tmp_path / "runs" / "extraction.jsonl")
    (tmp_path / "runs").mkdir()
    record = dict(result.manifest, precision_sanity_check=precision, corpus="synthetic")
    runs.append(record)
    filed = runs.read_all()[0]
    assert filed["code_commit"] == COMMIT and filed["source_hash"] == doc().source_hash
    assert filed["precision_sanity_check"]["precision"] == precision["precision"]
    assert filed["corpus"] == "synthetic"
    # AR-rag4ds-022 F8: dump the record where the doc generator can file it into
    # the repository, so the number survives the test run and S9 has a baseline
    import json as _json
    (tmp_path / "s7_precision_check.json").write_text(
        _json.dumps({"record": filed, "corpus_text": SYNTH, "gold_terms": sorted(GOLD_TERMS)},
                    indent=1, ensure_ascii=True), encoding="utf-8")
    # empty run: precision is None, never a misleading 0.0 or 1.0
    empty = measure_precision(run_extraction(doc("Nothing definitional here at all today."),
                                             [RuleStrategy()], COMMIT, TS), is_correct)
    assert empty["precision"] is None


# --- validation / fail-loud ---


def test_document_refuses_wrong_hash_and_bad_layer_pack():
    with pytest.raises(ExtractError, match="does not hash"):
        Document(source_hash="a" * 64, text="x", pack="robotics", layer="domain",
                 source_reference="s")
    good = "some text"
    h = hashlib.sha256(good.encode()).hexdigest()
    with pytest.raises(ExtractError, match="domain-layer documents need a pack code"):
        Document(source_hash=h, text=good, pack="shared", layer="domain", source_reference="s")


def test_run_validates_commit_timestamp_and_span_range():
    with pytest.raises(ExtractError, match="code_commit"):
        run_extraction(doc(), [RuleStrategy()], "nothex", TS)
    with pytest.raises(ExtractError, match="timestamp"):
        run_extraction(doc(), [RuleStrategy()], COMMIT, "whenever")
    with pytest.raises(ExtractError, match="at least one strategy"):
        run_extraction(doc(), [], COMMIT, TS)

    class OverrunStrategy:
        name = "overrun"

        def describe(self):
            return {"name": self.name, "model": None, "params": {}}

        def propose(self, document):
            n = len(document.text)
            return [Proposal("concept", "x", n - 1, n + 50)]

    with pytest.raises(ExtractError, match="outside the document"):
        run_extraction(doc(), [OverrunStrategy()], COMMIT, TS)


def test_run_is_deterministic():
    a = run_extraction(doc(), [RuleStrategy()], COMMIT, TS)
    b = run_extraction(doc(), [RuleStrategy()], COMMIT, TS)
    assert a.candidates == b.candidates and a.manifest == b.manifest


def test_build_entry_candidate_shape():
    p = Proposal("concept", "Bearing wear - dominant failure mode", 28, 90)
    c = build_entry_candidate(doc(), p, "rule-v1", "runs/extraction.jsonl:1")
    validate_candidate(c)
    assert c["proposed"]["source_reference"] == "corpus:robotics/notes#1"


# --- AR-rag4ds-022 r1: F1 (the bound in every script) ---


JP_SOURCE = (
    "\u3053\u308c\u306f\u30ed\u30dc\u30c3\u30c8\u306e\u4fdd\u5b88\u6587\u66f8\u3067\u3059\u3002"
    "\u8ef8\u53d7\u306e\u6469\u8017\u306f\u56de\u8ee2\u30a2\u30af\u30c1\u30e5\u30a8\u30fc\u30bf"
    "\u306e\u4e3b\u306a\u6545\u969c\u30e2\u30fc\u30c9\u3067\u3059\u3002"
)
RU_SOURCE = ("Eto dokument. "
             "\u0418\u0437\u043d\u043e\u0441 \u043f\u043e\u0434\u0448\u0438\u043f"
             "\u043d\u0438\u043a\u0430 \u044f\u0432\u043b\u044f\u0435\u0442\u0441\u044f "
             "\u043e\u0441\u043d\u043e\u0432\u043d\u044b\u043c \u0440\u0435\u0436\u0438\u043c"
             "\u043e\u043c \u043e\u0442\u043a\u0430\u0437\u0430 \u043f\u043e\u0432\u043e\u0440"
             "\u043e\u0442\u043d\u044b\u0445 \u043f\u0440\u0438\u0432\u043e\u0434\u043e\u0432 "
             "\u0438 \u044d\u0442\u043e \u0432\u0430\u0436\u043d\u043e \u0441\u0435\u0433"
             "\u043e\u0434\u043d\u044f \u0437\u0434\u0435\u0441\u044c.")


def test_tokenise_is_unicode_aware():
    assert [t.text for t in tokenise("Bearing wear!")] == ["bearing", "wear"]
    jp = tokenise("\u8ef8\u53d7\u306e\u6469\u8017")
    assert len(jp) == 5 and all(t.unsegmented for t in jp)  # one token per character
    ru = tokenise("\u0418\u0437\u043d\u043e\u0441 "
                  "\u043f\u043e\u0434\u0448\u0438\u043f\u043d\u0438\u043a\u0430")
    assert len(ru) == 2 and not any(t.unsegmented for t in ru)  # Cyrillic words, not chars
    greek_tokens = [t.text for t in tokenise("\u03b1\u03b2\u03b3 \u03b4")]
    assert greek_tokens == ["\u03b1\u03b2\u03b3", "\u03b4"]


def test_f1_a_copied_japanese_passage_is_suppressed_not_passed():
    """The r1 CRITICAL: a verbatim CJK passage measured zero and passed."""
    text = JP_SOURCE
    import hashlib as _h
    d = Document(source_hash=_h.sha256(text.encode()).hexdigest(), text=text,
                 pack="robotics", layer="domain", source_reference="corpus:jp#1")

    class CopyStrategy:
        name = "copy"

        def describe(self):
            return {"name": self.name, "model": None, "params": {}}

        def propose(self, document):
            return [Proposal("concept", document.text[10:40], 10, 40)]

    result = run_extraction(d, [CopyStrategy()], COMMIT, TS)
    assert result.candidates == [], "a copied CJK passage reached a candidate"
    assert result.suppressed[0]["char_run"] > QUOTATION_MAX_CJK_CHARS
    assert "unsegmented script" in result.suppressed[0]["reason"]
    # and the same is true through the shipped embedding strategy
    from rag4ds.extract.strategies import EmbeddingStrategy
    strat = EmbeddingStrategy(FAKE_Embedder(), ["\u6545\u969c"], top_k=3, floor=-1.0)
    r2 = run_extraction(d, [strat], COMMIT, TS)
    for c in r2.candidates:
        assert quotation_bound_check(c["proposed"]["content"], text)["ok"]


def test_f1_a_copied_russian_passage_is_suppressed():
    import hashlib as _h
    d = Document(source_hash=_h.sha256(RU_SOURCE.encode()).hexdigest(), text=RU_SOURCE,
                 pack="robotics", layer="domain", source_reference="corpus:ru#1")

    class CopyStrategy:
        name = "copy"

        def describe(self):
            return {"name": self.name, "model": None, "params": {}}

        def propose(self, document):
            return [Proposal("concept", document.text[14:], 14, len(document.text))]

    result = run_extraction(d, [CopyStrategy()], COMMIT, TS)
    assert result.candidates == []
    assert result.suppressed[0]["word_run"] > QUOTATION_MAX_WORDS


def test_f1_unmeasurable_content_is_refused_not_passed():
    """Content the tokeniser cannot measure must never take the permissive path."""
    check = quotation_bound_check("!!! ??? ... ***", SYNTH)
    assert check["ok"] is False and "unmeasurable" in check["reason"]
    assert check["tokenisable"] < MIN_TOKENISABLE_FRACTION
    assert within_quotation_bound("!!! ??? ... ***", SYNTH) is False


def test_f1_greek_run_lifted_from_an_english_source_is_measured():
    greek = ["\u03b1", "\u03b2", "\u03b3", "\u03b4", "\u03b5",
             "\u03b6", "\u03b7", "\u03b8", "\u03b9", "\u03ba"]
    src = "Constants: " + " ".join(greek) + " end."
    lifted = " ".join(greek)
    w, c = longest_verbatim_runs(lifted, src)
    assert w == 10 and c == 0
    assert within_quotation_bound(lifted, src) is False


# --- F2: the manifest describes every strategy, including its model ---


def test_f2_manifest_records_each_strategys_identity_params_and_model():
    from rag4ds.extract.strategies import EmbeddingStrategy, RuleStrategy
    seeds = ["failure modes"]
    result = run_extraction(doc(), [RuleStrategy(), EmbeddingStrategy(FAKE_Embedder(), seeds,
                                                                     top_k=2, floor=0.0)],
                            COMMIT, TS)  # no runner embedder: the model is the strategy's
    described = {d["name"]: d for d in result.manifest["strategies"]}
    assert set(described) == {"rule-v1", "embed-v1"}
    assert described["embed-v1"]["model"] == {"id": "FAKE-embed", "revision": "test-double"}
    assert described["embed-v1"]["params"]["seeds"] == seeds
    assert described["embed-v1"]["params"]["top_k"] == 2
    assert described["rule-v1"]["params"]["definitions"] is True
    assert result.manifest["clustering_embedder"] is None  # honest: the runner had none
    assert result.manifest["strategy_kept_counts"]["rule-v1"] >= 4


def test_f2_a_strategy_without_describe_is_refused():
    class Bare:
        name = "bare"

        def propose(self, document):
            return []

    with pytest.raises(ExtractError, match="has no describe"):
        run_extraction(doc(), [Bare()], COMMIT, TS)


# --- F3: the three 6.2 capabilities and a non-concept type ---


TABLE_DOC = (
    "# Actuator reference\n"
    "This section lists actuator limits.\n"
    "\n"
    "| Component | Limit | Unit |\n"
    "| --- | --- | --- |\n"
    "| Bearing | 120 | celsius |\n"
    "| Encoder | 5000 | rpm |\n"
)


def test_f3_headings_and_tables_and_a_fact_type():
    from rag4ds.extract.strategies import RuleStrategy
    result = run_extraction(doc(TABLE_DOC), [RuleStrategy()], COMMIT, TS)
    types = {c["proposed"]["type"] for c in result.candidates}
    assert "fact" in types, "table extraction proposes facts"
    contents = [c["proposed"]["content"] for c in result.candidates]
    assert any(c.lower().startswith("actuator reference") for c in contents), "heading heuristic"
    assert any("bearing" in c.lower() and "120" in c for c in contents), "table row"
    for c in result.candidates:
        validate_candidate(c)
        assert within_quotation_bound(c["proposed"]["content"], TABLE_DOC)
        assert TABLE_DOC[c["span_start"]:c["span_end"]].strip()
    # each capability can be switched off, and the switches are in the manifest
    only_defs = run_extraction(doc(TABLE_DOC), [RuleStrategy(headings=False, tables=False)],
                               COMMIT, TS)
    assert all(c["proposed"]["type"] == "concept" for c in only_defs.candidates)
    assert only_defs.manifest["strategies"][0]["params"]["tables"] is False


# --- F4, F5, F6, F7, F9 ---


def test_f4_extraction_id_covers_what_was_produced_and_the_config():
    class OneStrategy:
        name = "one"

        def __init__(self, content):
            self.content = content

        def describe(self):
            return {"name": self.name, "model": None, "params": {"content": self.content}}

        def propose(self, document):
            return [Proposal("concept", self.content, 0, 20)]

    a = run_extraction(doc(), [OneStrategy("Alpha - first thing")], COMMIT, TS)
    b = run_extraction(doc(), [OneStrategy("Beta - second thing")], COMMIT, TS)
    assert a.manifest["extraction_id"] != b.manifest["extraction_id"]
    c = run_extraction(doc(), [OneStrategy("Alpha - first thing")], COMMIT, TS,
                       config={"note": "different"})
    assert a.manifest["extraction_id"] != c.manifest["extraction_id"]
    d = run_extraction(doc(), [OneStrategy("Alpha - first thing")], COMMIT, TS)
    assert a.manifest["extraction_id"] == d.manifest["extraction_id"]  # reproducible


def test_f5_duplicate_strategy_names_are_refused():
    from rag4ds.extract.strategies import RuleStrategy
    with pytest.raises(ExtractError, match="used twice"):
        run_extraction(doc(), [RuleStrategy(), RuleStrategy()], COMMIT, TS)


def test_f6_non_finite_vectors_and_bad_thresholds_raise():
    import numpy as np

    class NanEmbedder:
        model_id, revision = "nan", "t"

        def embed(self, texts):
            m = np.ones((len(texts), 4), dtype=np.float32)
            m[0][0] = np.nan
            return m

    with pytest.raises(ExtractError, match="NaN or inf"):
        cluster_near_duplicates(["a b", "c d"], NanEmbedder())
    for bad in (5.0, -2.0, float("nan"), "0.9", True):
        with pytest.raises(ExtractError, match="must be a number in"):
            cluster_near_duplicates(["a b", "c d"], FAKE_Embedder(), threshold=bad)


def test_f7_suppression_detail_reaches_the_manifest():
    class PassageStrategy:
        name = "passage"

        def describe(self):
            return {"name": self.name, "model": None, "params": {}}

        def propose(self, document):
            long = " ".join(document.text.split()[:QUOTATION_MAX_WORDS + 4])
            return [Proposal("concept", long, 0, len(long) + 5)]

    result = run_extraction(doc(), [PassageStrategy()], COMMIT, TS)
    sup = result.manifest["suppressed"]
    assert len(sup) == 1 and sup == result.suppressed
    assert sup[0]["span_start"] == 0 and sup[0]["word_run"] > QUOTATION_MAX_WORDS
    assert "quotation bound" in sup[0]["reason"]
    # the detail names no source text, so the manifest stays publishable (NFR5)
    import json as _j
    assert "Robotics maintenance" not in _j.dumps(sup)


def test_f9_the_run_manifest_reference_is_stable_and_knowable_before_filing():
    from rag4ds.staging.store import candidate_id
    from rag4ds.extract.strategies import RuleStrategy
    result = run_extraction(doc(), [RuleStrategy()], COMMIT, TS)
    ref = result.manifest["run_manifest_ref"]
    assert ref == f"runs/extraction.jsonl#{result.manifest['extraction_id']}"
    assert all(c["run_manifest"] == ref for c in result.candidates)
    # filing the manifest does not change any candidate's identity
    ids_before = [candidate_id(c) for c in result.candidates]
    again = run_extraction(doc(), [RuleStrategy()], COMMIT, TS)
    assert [candidate_id(c) for c in again.candidates] == ids_before


# --- AR-rag4ds-022 r2: F10, F11, F12 ---


# the auditor's unlisted-block cases, plus scripts nobody enumerated
UNLISTED_SAMPLES = {
    "bopomofo": "\u3105\u3106\u3107\u3108\u3109\u310a\u310b\u310c\u310d\u310e",
    "katakana_phonetic_ext": "\u31f0\u31f1\u31f2\u31f3\u31f4\u31f5\u31f6\u31f7\u31f8\u31f9",
    "hangul_jamo": "\u1100\u1101\u1102\u1103\u1104\u1105\u1106\u1107\u1108\u1109",
    "kana_supplement": "\U0001b000\U0001b001\U0001b002\U0001b003\U0001b004"
             "\U0001b000\U0001b001\U0001b002\U0001b003\U0001b004",
    "tangut": "\U00017000\U00017001\U00017002\U00017003\U00017004"
             "\U00017005\U00017006\U00017007\U00017008\U00017009",
    "nushu": "\U0001b170\U0001b171\U0001b172\U0001b173\U0001b174"
             "\U0001b175\U0001b176\U0001b177\U0001b178\U0001b179",
    "egyptian": "\U00013000\U00013001\U00013002\U00013003\U00013004"
             "\U00013005\U00013006\U00013007\U00013008\U00013009",
}


@pytest.mark.parametrize("name", sorted(UNLISTED_SAMPLES))
def test_f10_unlisted_scripts_are_unsegmented_by_default(name):
    """F10: the class is closed by inversion - a script nobody enumerated is
    one token per character, not one unbounded token."""
    sample = UNLISTED_SAMPLES[name]
    toks = tokenise(sample)
    assert len(toks) == len(sample), f"{name}: {len(toks)} tokens for {len(sample)} characters"
    assert all(t.unsegmented for t in toks)
    assert all(is_unsegmented(ch) for ch in sample)


@pytest.mark.parametrize("name", sorted(UNLISTED_SAMPLES))
def test_f10_a_verbatim_run_in_an_unlisted_script_is_refused(name):
    sample = UNLISTED_SAMPLES[name]
    src = "Reference table. " + sample + " End of table."
    check = quotation_bound_check(sample, src)
    assert check["ok"] is False, f"{name} passed the bound verbatim"
    assert check["char_run"] >= min(len(sample), QUOTATION_MAX_CJK_CHARS + 1)


def test_f10_word_separated_scripts_still_tokenise_as_words():
    """The inversion must not turn Latin, Cyrillic, Greek, Hebrew, Arabic,
    Devanagari or digits into per-character tokens."""
    for script, sample in (
        ("latin", "bearing wear"),
        ("cyrillic", "\u0438\u0437\u043d\u043e\u0441 \u0432\u0430\u043b\u0430"),
        ("greek", "\u03b1\u03bb\u03c6\u03b1 \u03b2\u03b7\u03c4\u03b1"),
        ("hebrew", "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd"),
        ("arabic", "\u0645\u0631\u062d\u0628\u0627 \u0628\u0643"),
        ("devanagari", "\u0928\u092e\u0938\u094d\u0924\u0947 \u091c\u0940"),
        ("digits", "120 5000"),
    ):
        toks = tokenise(sample)
        assert len(toks) == 2, f"{script}: {[t.text for t in toks]}"
        assert not any(t.unsegmented for t in toks), script
    assert "LATIN" in SEGMENTED_SCRIPTS and "CYRILLIC" in SEGMENTED_SCRIPTS


def test_f12_tokenisable_fraction_counts_source_characters():
    """F12: the fraction counted casefolded lengths, so one U+00DF read 2.0 and
    could flip the unmeasurable refusal off."""
    assert tokenisable_fraction("\u00df") == 1.0
    mixed = "\u00df\u00df\u00df\u00df" + "!?.,;:*#@$%^"
    frac = tokenisable_fraction(mixed)
    assert abs(frac - 0.25) < 1e-9, frac
    assert frac < MIN_TOKENISABLE_FRACTION
    assert quotation_bound_check(mixed, SYNTH)["ok"] is False
    assert 0.0 <= tokenisable_fraction("Bearing wear!") <= 1.0


def test_f11_the_record_names_the_code_that_ran():
    """F11: COMMIT is the working tree's HEAD, not a stale constant - so a
    record filed from a run names a commit where this code exists."""
    import subprocess
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    # tautological as written on purpose: it fails the day COMMIT is reverted to a
    # literal, which is exactly the regression this test exists to catch (O17)
    assert COMMIT == head
    tracked = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", head, "src/rag4ds/extract/strategy.py"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert tracked.endswith("src/rag4ds/extract/strategy.py"), (
        "the commit the record names must contain the code that produced it")


# --- AR-rag4ds-022 r3: F13 (marks attach to their base; the character budget
# counts characters, so a mark-bearing copy cannot split across two budgets) ---


# Khmer and Thai: base characters carrying combining marks
KHMER_RUN = ("\u1781\u17d2\u1789\u17bb\u17c6\u1794\u17b6\u1793\u1785\u17bc\u179b"
             "\u1798\u1780\u1793")          # 14 characters, several of them marks
THAI_RUN = "\u0e01\u0e32\u0e23\u0e1a\u0e33\u0e23\u0e38\u0e07\u0e23\u0e31\u0e01\u0e29\u0e32"


@pytest.mark.parametrize("run,label", [(KHMER_RUN, "khmer"), (THAI_RUN, "thai")])
def test_f13_a_mark_bearing_verbatim_copy_is_refused(run, label):
    """r2 regression: combining marks became word tokens inside unsegmented
    scripts, so a 14-character Khmer copy split into 6 words + 8 characters and
    passed BOTH budgets. A mark now extends the character token it modifies and
    the character budget counts characters, so the copy is measured whole."""
    src = "Reference. " + run + " End."
    check = quotation_bound_check(run, src)
    assert check["ok"] is False, f"{label}: a verbatim copy passed the bound"
    assert check["char_run"] >= QUOTATION_MAX_CJK_CHARS + 1, check
    # the mark must not consume the word budget
    assert check["word_run"] == 0, check

    class CopyStrategy:
        name = "copy"

        def describe(self):
            return {"name": self.name, "model": None, "params": {}}

        def propose(self, document):
            i = document.text.index(run)
            return [Proposal("concept", run, i, i + len(run))]

    import hashlib as _h
    d = Document(source_hash=_h.sha256(src.encode()).hexdigest(), text=src,
                 pack="robotics", layer="domain", source_reference="corpus:x#1")
    result = run_extraction(d, [CopyStrategy()], COMMIT, TS)
    assert result.candidates == [], f"{label}: the copy reached a candidate"
    assert result.suppressed[0]["char_run"] >= QUOTATION_MAX_CJK_CHARS + 1


def test_f13_a_mark_extends_its_base_token_not_a_new_word():
    """One base plus its marks is one unsegmented token whose source_len counts
    every character it consumed."""
    cluster = "\u1781\u17d2\u1789"          # Khmer base + coeng + consonant
    toks = tokenise(cluster)
    assert all(t.unsegmented for t in toks), [(t.text, t.unsegmented) for t in toks]
    assert sum(t.source_len for t in toks) == len(cluster)
    # Devanagari (a word-separated script) still tokenises as words, marks included
    deva = "\u0928\u092e\u0938\u094d\u0924\u0947 \u091c\u0940"
    dtoks = tokenise(deva)
    assert len(dtoks) == 2 and not any(t.unsegmented for t in dtoks)
    assert sum(t.source_len for t in dtoks) == len(deva.replace(" ", ""))


def test_o18_nfkc_folds_fullwidth_and_mathematical_before_the_script_is_read():
    """The two dropped allowlist entries: NFKC decomposes these to Latin, so the
    tokeniser sees LATIN and they remain word tokens."""
    assert [t.text for t in tokenise("\uff22\uff45\uff41\uff52")] == ["bear"]
    assert [t.text for t in tokenise("\U0001d400\U0001d401")] == ["ab"]


# --- AR-rag4ds-023 F1: the word budget bounded tokens, not characters ---


def _admitted_by_the_old_ceilings(check):
    """True when BOTH pre-F1 ceilings are satisfied - i.e. the audited pin
    admitted this content. Used to pin every F1 case as a real ok=True ->
    ok=False witness rather than a case that was already refused."""
    return (check["word_run"] <= QUOTATION_MAX_WORDS
            and check["char_run"] <= QUOTATION_MAX_CJK_CHARS)


# eight ordinary bioinformatics terms - the auditor's own case, and the domain
# is one of the project's three declared scenarios
BIO8 = ("chromatin immunoprecipitation polyadenylation oligonucleotide "
        "phosphorylation transcriptomics bioinformatics electrophoresis")
CHEM8 = ("dichlorodiphenyltrichloroethane polytetrafluoroethylene acetylsalicylic "
         "methylcyclopentadienyl tetrahydrocannabinol paracetamol ibuprofen naproxen")


@pytest.mark.parametrize("label,text", [("bioinformatics", BIO8), ("chemistry", CHEM8)])
def test_f1_eight_long_technical_terms_are_refused_though_the_token_count_is_legal(label, text):
    """Eight tokens is exactly the word bound, so the pin admitted these; they
    carry far more verbatim source than a short attributed phrase."""
    check = quotation_bound_check(text, text)
    assert _admitted_by_the_old_ceilings(check), (label, check)   # the pin admitted it
    assert check["word_run"] == 8 and check["char_run"] == 0
    assert check["source_run"] > QUOTATION_MAX_SOURCE_CHARS
    assert check["ok"] is False
    assert "source characters" in check["reason"] and "AR-rag4ds-023 F1" in check["reason"]


@pytest.mark.parametrize("n", [65, 100, 200, 1000, 5000])
def test_f1_one_unbroken_token_is_bounded_at_last(n):
    """A single token was one word whatever its length, so verbatim volume had
    no ceiling at all: a sequence, IUPAC name, URL or base64 blob went through."""
    text = "a" * n
    check = quotation_bound_check(text, text)
    assert _admitted_by_the_old_ceilings(check), check            # the pin admitted it
    assert check["word_run"] == 1 and check["source_run"] == n
    assert check["ok"] is False


def test_f1_the_source_character_ceiling_is_exact_at_its_boundary():
    """The ceiling admits exactly QUOTATION_MAX_SOURCE_CHARS and refuses one more,
    so the bound is a stated number and not an approximation."""
    at = "a" * QUOTATION_MAX_SOURCE_CHARS
    over = "a" * (QUOTATION_MAX_SOURCE_CHARS + 1)
    assert quotation_bound_check(at, at)["ok"] is True
    assert quotation_bound_check(over, over)["ok"] is False
    assert longest_verbatim_source_chars(at, at) == QUOTATION_MAX_SOURCE_CHARS


def test_f1_charges_every_token_whatever_its_script_so_interleaving_closes_too():
    """AR-rag4ds-022 O20/O12/O2: word and unsegmented runs were charged to
    separate budgets, so an interleaved copy passed both. One source-character
    ceiling charges them together, which is why one fix closes both items."""
    interleaved = ("immunoprecipitation \u6a5f immunoprecipitation \u68b0 "
                   "immunoprecipitation \u5b66 immunoprecipitation \u7fd2")
    check = quotation_bound_check(interleaved, interleaved)
    assert _admitted_by_the_old_ceilings(check), check            # the pin admitted it
    assert check["word_run"] <= QUOTATION_MAX_WORDS
    assert check["char_run"] <= QUOTATION_MAX_CJK_CHARS
    assert check["source_run"] > QUOTATION_MAX_SOURCE_CHARS and check["ok"] is False


def test_f1_nothing_became_more_permissive_than_the_audited_pin():
    """The source-character ceiling was ADDED beside the two existing ones, not
    substituted for them, so every input the pin refused is still refused."""
    # unsegmented scripts: refused on the character ceiling, well under 64 chars
    for label, text in [("Japanese", "\u6a5f\u68b0\u5b66\u7fd2\u306f\u30c7\u30fc\u30bf"
                                     "\u304b\u3089\u30d1\u30bf\u30fc\u30f3\u3092\u767a\u898b"),
                        ("Khmer", "\u1781\u17d2\u1789\u1780\u17b6\u179a\u179a\u17c6\u1780")]:
        check = quotation_bound_check(text, text)
        assert check["ok"] is False, label
        assert check["source_run"] <= QUOTATION_MAX_SOURCE_CHARS, label
        assert check["char_run"] > QUOTATION_MAX_CJK_CHARS, label
    # nine short words: refused on the word ceiling, well under 64 chars
    nine = "the cat sat on the mat and ate it"
    check = quotation_bound_check(nine, nine)
    assert check["ok"] is False and check["word_run"] > QUOTATION_MAX_WORDS
    assert check["source_run"] <= QUOTATION_MAX_SOURCE_CHARS


def test_f1_legitimate_extraction_still_passes_the_bound():
    """The ceiling must bound copying without refusing the work: every proposal
    the rule strategy makes on the S7 corpus still reaches a candidate."""
    result = run_extraction(doc(), [RuleStrategy()], COMMIT, TS)
    assert len(result.candidates) >= 4 and result.suppressed == []
    worst = max(longest_verbatim_source_chars(c["proposed"]["content"], SYNTH)
                for c in result.candidates)
    assert worst <= QUOTATION_MAX_SOURCE_CHARS
    # and the headroom is real, not incidental: the worst legitimate run is well
    # inside the ceiling, so the bound is not tuned to just barely admit them
    assert worst < QUOTATION_MAX_SOURCE_CHARS


def test_f1_the_run_manifest_and_suppression_detail_carry_the_new_measure():
    """NFR2/NFR9: the ceiling in force is part of the record, and a suppressed
    proposal states the measurement that refused it."""
    src = SYNTH + "\n" + BIO8 + "\n"

    class CopyStrategy:
        name = "copy-v1"

        def describe(self):
            return {"name": self.name, "params": {"copies": True}, "model": None}

        def propose(self, document):
            i = document.text.index(BIO8)
            return [Proposal("concept", BIO8, i, i + len(BIO8))]

    d = Document(source_hash=hashlib.sha256(src.encode()).hexdigest(), text=src,
                 pack="robotics", layer="domain", source_reference="corpus:x#1")
    result = run_extraction(d, [CopyStrategy()], COMMIT, TS)
    assert result.candidates == []
    assert result.manifest["quotation_max_source_chars"] == QUOTATION_MAX_SOURCE_CHARS
    detail = result.suppressed[0]
    assert detail["source_run"] > QUOTATION_MAX_SOURCE_CHARS
    assert detail["word_run"] == 8 and detail["char_run"] == 0


def test_o19_digits_are_word_characters_only_when_their_unicode_name_says_so():
    """AR-rag4ds-022 O19: the description the code carried was untrue for three
    digit ranges. The corrected description is now pinned by a test, so it
    cannot drift back."""
    # name begins with DIGIT / the script word is not first -> word characters
    for ch in ["0", "\u0966"]:            # ASCII ZERO, DEVANAGARI DIGIT ZERO
        assert is_unsegmented(ch) is False, hex(ord(ch))
    # name begins with an unlisted script -> unsegmented, charged per character
    for ch in ["\u0660", "\u06f0", "\u0e50"]:   # ARABIC-INDIC, EXTENDED, THAI
        assert is_unsegmented(ch) is True, hex(ord(ch))
