"""S9c part 5 - the rule strategy's tuning (opt-in; PI decision A of
2026-09-15): a checklist or pronoun lead, an error-code prefix, a gloss too
short or cut mid-clause, and whitespace variants are refused or merged
BEFORE a proposal is made; every refusal is recorded in the manifest; with
the tuning off nothing changes - the strategy describes itself exactly as
the filed S9c part 1 manifests record, so those runs keep their ids."""
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from rag4ds.auditlog.journal import Journal
from rag4ds.extract.strategies import RuleStrategy
from rag4ds.extract.strategy import Document, ExtractError, run_extraction

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))
import extract_pack  # noqa: E402

TS = "2026-09-15 18:40 JST"
COMMIT = "abcdef0"

TEXT = (
    "Bearing wear is the dominant failure mode of rotary joints. "
    "Check the base mounting bolts are properly tightened. "
    "They are either on or off. "
    "C296A4 Phase 1 is not connected. "
    "Note: be very careful when removing the cover of the joint before service. "
    "Torque ripple is a periodic variation of the output torque, seen at low speed. "
    "Backlash is the play between meshing gears. "
    "Backlash  is   the play between meshing gears. "
    "To remove: 1. "
    "Encoder resolution is 19 bits per revolution.\n"
)


def _doc(text=TEXT):
    return Document(source_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(), text=text,
                    pack="robotics", layer="domain", source_reference="corpus:robotics/t")


def _contents(strat):
    return [p.content for p in strat.propose(_doc())]


def test_off_by_default_and_the_description_is_unchanged():
    off = RuleStrategy(headings=False)
    assert off.tuning is False
    assert "tuning" not in off.describe()["params"]
    on = RuleStrategy(headings=False, tuning=True)
    assert on.describe()["params"]["tuning"]["version"] == "tune-v2"
    # tune-v2 (S9e) names its two structural rules in the description
    assert "citation_line" in on.describe()["params"]["tuning"]
    assert "references_heading" in on.describe()["params"]["tuning"]
    # The filed manifests, in two eras. Until 2026-09-21 no FILED run had ever
    # used the tuning: every tuned run was a dry run, so every manifest
    # described rule-v1 exactly as the untuned strategy does and none carried
    # refused_by_tuning. The intake of the two harmonic-drive papers
    # (S9e, record section 17) was filed under the PI's tuned configuration -
    # option A of 2026-09-15 with option C of 2026-09-17 - and is the first
    # filed run in which the tuning moved what reached staging. The test now
    # says which era each manifest belongs to instead of asserting the old
    # one for all of them: an untuned manifest must still round-trip and
    # carry no refusals, and a tuned one must carry both.
    runs = MODULE_ROOT / "runs" / "extraction.jsonl"
    if runs.is_file():
        recorded = [m for m in Journal(runs).read_all() if m.get("pack") == "robotics"]
        assert recorded, "the pack's extraction manifests are on the record"
        eras = {"untuned": 0, "tuned": 0}
        for m in recorded:
            rule = [s for s in m["strategies"] if s["name"] == "rule-v1"]
            assert rule, m["extraction_id"]
            params = rule[0]["params"]
            if "tuning" not in params:
                eras["untuned"] += 1
                assert rule[0] == RuleStrategy(**{k: params[k] for k in
                                                  ("definitions", "headings", "tables")}
                                               ).describe()
                assert "refused_by_tuning" not in m
            else:
                eras["tuned"] += 1
                # a filed tuned manifest is tune-v1 (the 2026-09-21 run) or, from
                # the S9e fix on, tune-v2; only the CURRENT version must round-trip
                # against the live strategy - an older one keeps its own params
                assert params["tuning"]["version"] in ("tune-v1", "tune-v2")
                if params["tuning"]["version"] == RuleStrategy(tuning=True).describe()[
                        "params"]["tuning"]["version"]:
                    assert rule[0] == RuleStrategy(
                        **{k: params[k] for k in ("definitions", "headings", "tables")},
                        tuning=True).describe()
                # a tuned run records what it refused - and records no source text
                refused = m.get("refused_by_tuning")
                if refused is not None:
                    for cap in refused.values():
                        assert cap["count"] == len(cap["items"])
                        for item in cap["items"]:
                            assert set(item) == {"capability", "reason", "line_sha256",
                                                 "span_start", "span_end"}
        assert eras["untuned"] and eras["tuned"], eras


def test_each_rule_refuses_its_shape_and_records_why():
    off = _contents(RuleStrategy(headings=False))
    on_strat = RuleStrategy(headings=False, tuning=True)
    on = _contents(on_strat)
    # untuned: the checklist line, the pronoun, the error code, the cut note, the
    # whitespace twin and the numeric gloss all come through
    assert any(c.startswith("Check the base") for c in off)
    assert any(c.startswith("They - ") for c in off)
    assert any(c.startswith("C296A4") for c in off)
    assert any(c.startswith("Note - ") for c in off)
    assert sum(1 for c in off if c.startswith("Backlash - ")) == 2
    assert any(c == "To remove - 1" for c in off)
    # tuned: none of them; the real definitions stay, whitespace collapsed, one Backlash
    assert on == ["Bearing wear - dominant failure mode of rotary joints",
                  "Torque ripple - periodic variation of the output torque",
                  "Backlash - play between meshing gears",
                  "Encoder resolution - 19 bits per revolution"]
    reasons = sorted(r["reason"] for r in on_strat.refused)
    assert reasons == ["code-prefix", "exact-duplicate", "lead-stop", "lead-stop", "lead-stop",
                       "lead-stop"]
    by = {r["reason"]: r for r in on_strat.refused}
    assert by["exact-duplicate"]["capability"] == "dedup"
    # a refusal names its line by span and sha256, never by text (AR-rag4ds-038 F2)
    code = by["code-prefix"]
    assert TEXT[code["span_start"]:code["span_end"]].startswith("C296A4")
    assert code["line_sha256"] == hashlib.sha256(
        TEXT[code["span_start"]:code["span_end"]].encode("utf-8")).hexdigest()
    leads = sorted(TEXT[r["span_start"]:r["span_end"]][:5] for r in on_strat.refused
                   if r["reason"] == "lead-stop")
    assert leads == ["Check", "Note:", "They ", "To re"]
    assert all("text" not in r for r in on_strat.refused)
    # a numeric gloss on a real term: too short to say anything
    s = RuleStrategy(headings=False, tuning=True)
    assert [p.content for p in s.propose(_doc("Payload: 1. Reach is 850 mm.\n"))] == [
        "Reach - 850 mm"]
    assert [r["reason"] for r in s.refused] == ["gloss-too-short"]
    # a gloss cut AT a clause boundary is kept: six words then a comma
    text = "Stall torque is the torque at zero speed, measured at the shaft.\n"
    s = RuleStrategy(headings=False, tuning=True)
    # an 8-word gloss with no inner boundary runs to the sentence's end (within 12)
    assert [p.content for p in s.propose(_doc(text))] == [
        "Stall torque - torque at zero speed measured at the shaft"]
    assert s.refused == []
    # a gloss cut mid-clause is EXTENDED to the next clause boundary (up to 12 words) -
    # the shape the validator's own edit of protective-stop v1 produced by hand
    text = ("Protective stops are indications of issues including program or production "
            "problems, not only safety issues.\n")
    s = RuleStrategy(headings=False, tuning=True)
    assert [p.content for p in s.propose(_doc(text))] == [
        "Protective stops - indications of issues including program or production problems"]
    assert s.refused == []
    # no boundary within 12 words: refused
    text = ("Joint drift is a slow change of the measured joint position over hours of "
            "continuous operation under load without any fault being raised anywhere\n")
    s = RuleStrategy(headings=False, tuning=True)
    assert s.propose(_doc(text)) == []
    assert [r["reason"] for r in s.refused] == ["gloss-cut-mid-clause"]


def test_headings_titles_get_the_lead_stop_and_code_rules_too():
    text = ("# Check the brakes\nThe brakes must hold the arm.\n\n# C163 Protective stops\n"
            "Too many stops.\n\n# Joint torque\nThe torque at each joint is measured.\n")
    off = RuleStrategy(definitions=False, tables=False)
    on = RuleStrategy(definitions=False, tables=False, tuning=True)
    assert len(off.propose(_doc(text))) == 3
    assert [p.content for p in on.propose(_doc(text))] == [
        "Joint torque - section topic: The torque at each joint"]
    assert sorted(r["reason"] for r in on.refused) == ["code-prefix", "lead-stop"]


def test_the_refusals_reach_the_manifest_only_when_there_are_any():
    untuned = run_extraction(_doc(), [RuleStrategy(headings=False)], COMMIT, TS)
    assert "refused_by_tuning" not in untuned.manifest
    tuned = run_extraction(_doc(), [RuleStrategy(headings=False, tuning=True)], COMMIT, TS)
    r = tuned.manifest["refused_by_tuning"]["rule-v1"]
    assert r["count"] == 6 and len(r["items"]) == 6
    assert {i["reason"] for i in r["items"]} == {"code-prefix", "exact-duplicate", "lead-stop"}
    assert tuned.manifest["strategies"][0]["params"]["tuning"]["version"] == "tune-v2"
    assert tuned.manifest["extraction_id"] != untuned.manifest["extraction_id"]
    # the manifest is JSON-serialisable with the new block
    json.dumps(tuned.manifest)
    # a tuned run over a text the tuning has nothing to say about carries no block
    clean = run_extraction(_doc("Backlash is the play between meshing gears.\n"),
                           [RuleStrategy(headings=False, tuning=True)], COMMIT, TS)
    assert "refused_by_tuning" not in clean.manifest


def test_the_extraction_tool_passes_the_flag_only_when_asked(monkeypatch):
    seen = {}

    def fake_run(pack, root, names, flags, top_k, floor, dry_run, out=None, clock=None,
                 span_check=True):
        seen.update(flags)
        return 0

    monkeypatch.setattr(extract_pack, "run", fake_run)
    assert extract_pack.main(["robotics", "--dry-run"]) == 0
    assert "tuning" not in seen
    seen.clear()
    assert extract_pack.main(["robotics", "--dry-run", "--tuning"]) == 0
    assert seen["tuning"] is True


def test_the_extension_is_bound_aware_and_a_refusal_record_carries_no_text():
    """AR-rag4ds-038 F1/F2: the longest clause-bounded gloss the quotation
    bound admits is taken - the fallback is the 6-word cut - and a refusal
    record identifies its line by sha256 and span, never by text."""
    import hashlib as _h
    from rag4ds.extract.strategy import QUOTATION_MAX_SOURCE_CHARS, quotation_bound_check
    # the extended gloss (9 words) is a 62-character verbatim run: admitted
    text = ("Protective stops are indications of issues including program or production "
            "problems, not only safety issues.\n")
    s = RuleStrategy(headings=False, tuning=True)
    [p] = s.propose(_doc(text))
    assert p.content == ("Protective stops - indications of issues including program or "
                         "production problems")
    assert quotation_bound_check(p.content, text)["ok"]
    # a longer clause runs over the bound: the rule falls back to the 6-word cut
    text2 = ("Protective stops are indications of issues including program or production "
             "issues not only safety issues, and more.\n")
    s2 = RuleStrategy(headings=False, tuning=True)
    [p2] = s2.propose(_doc(text2))
    assert p2.content == "Protective stops - indications of issues including program or"
    assert quotation_bound_check(p2.content, text2)["ok"] and s2.refused == []
    # when even the cut exceeds the bound: refused as gloss-over-bound, nothing proposed
    long_words = " ".join("x" * 14 for _ in range(6)) + ", and more words here."
    text3 = f"Widget is a {long_words}\n"
    s3 = RuleStrategy(headings=False, tuning=True)
    assert s3.propose(_doc(text3)) == []
    assert [r["reason"] for r in s3.refused] == ["gloss-over-bound"]
    assert len(" ".join("x" * 14 for _ in range(6))) > QUOTATION_MAX_SOURCE_CHARS
    # the refusal record: capability, reason, line sha256, span - no text
    rec = s3.refused[0]
    assert set(rec) == {"capability", "reason", "line_sha256", "span_start", "span_end"}
    line = text3[rec["span_start"]:rec["span_end"]]
    assert rec["line_sha256"] == _h.sha256(line.encode("utf-8")).hexdigest()
    # the runner refuses a strategy whose refusal record smuggles text
    class Leaky:
        name = "rule-v1"
        refused = [{"capability": "definitions", "reason": "lead-stop", "text": "the source",
                    "span_start": 0, "span_end": 5}]

        def propose(self, document):
            return []

        def describe(self):
            return {"name": self.name, "params": {}}

    with pytest.raises(ExtractError, match="no source text reaches the manifest"):
        run_extraction(_doc(), [Leaky()], COMMIT, TS)


def test_live_tuning_keeps_the_approved_span_and_writes_no_text(tmp_path):
    """AR-rag4ds-038 O4: the tuning against the real corpus and the bound -
    every approved span a strategy proposed is still proposed AND filed
    (post-bound) under the tuning, each through the strategy that filed it
    (hand-authored candidates excepted, below), and no refusal record carries
    text. Skips without the corpus."""
    data = Path(os.environ.get("RAG4DS_DATA_DIR") or (MODULE_ROOT / "data"))
    texts = {p.stem: p.read_text(encoding="utf-8")
             for p in (data / "robotics" / "text").glob("*.txt")}
    if not texts:
        pytest.skip("the robotics corpus texts are absent here (set RAG4DS_DATA_DIR)")
    store_path = MODULE_ROOT / "stage" / "candidates.jsonl"
    approved = []
    cands, states = {}, {}
    for line in store_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("record") == "candidate":
            cands[r["id"]] = r["candidate"]
        elif r.get("record") == "transition":
            states[r["id"]] = r["to"]
    for cid, c in cands.items():
        # ENTRY candidates only. A mapping draft carries no source span, so it has
        # no source_hash to look up and no span for the tuning to keep - the same
        # kind guard the extractor's decided-span check needs (AR-rag4ds-040 F2).
        # Before the pack's first mapping was approved on 2026-09-20 this filter
        # was vacuous and its absence could not show; the approval exposed it.
        # A hand-authored candidate ('author:' - S9e part 8's two, approved 2026-09-26 from
        # the PI-signed file) was never proposed by a strategy, so the tuning's promise - to
        # keep the strategies' own approved spans - does not reach it; any other extractor
        # still meets the rule-v1 assertion below and fails loudly.
        if (c.get("pack") == "robotics" and c.get("kind") == "entry"
                and states.get(cid) == "APPROVED"
                and not str(c.get("extractor", "")).startswith("author:")):
            approved.append(c)
    assert approved, "the pack has an approved entry candidate on the record"
    assert all("source_hash" in c and "proposed" in c for c in approved)
    by_hash = {hashlib.sha256(t.encode("utf-8")).hexdigest(): t for t in texts.values()}
    # S9e part 6 (session 5): two approved candidates are embed-v1 proposals, so
    # each candidate is run through the strategy that filed it, in the filed
    # configuration (record 19.5: headings off, floor 0.5, top_k 1, tuning on);
    # the embedder is the pinned model, loaded once, offline
    embed = None
    for c in approved:
        text = by_hash[c["source_hash"]]
        doc = Document(source_hash=c["source_hash"], text=text, pack="robotics", layer="domain",
                       source_reference=c["proposed"]["source_reference"])
        if c["extractor"] == "embed-v1":
            from rag4ds.extract.strategies import EmbeddingStrategy
            from rag4ds.index.build import (PINNED_EMBEDDING_MODEL, PINNED_EMBEDDING_REVISION,
                                            SentenceTransformerEmbedder)
            from rag4ds.model.pack import load_pack_manifest
            if embed is None:
                embed = SentenceTransformerEmbedder(PINNED_EMBEDDING_MODEL,
                                                    PINNED_EMBEDDING_REVISION)
            seeds = list(load_pack_manifest(MODULE_ROOT / "packs" / "robotics").lexicon)
            strat = EmbeddingStrategy(embed, seeds, top_k=1, floor=0.5, tuning=True)
        else:
            assert c["extractor"] == "rule-v1", c["extractor"]
            strat = RuleStrategy(headings=False, tuning=True)
        result = run_extraction(doc, [strat], COMMIT, TS)
        filed = [x for x in result.candidates
                 if (x["span_start"], x["span_end"]) == (c["span_start"], c["span_end"])]
        assert filed, f"the approved span survives the tuning AND the bound ({c['extractor']})"
        assert not any("text" in r for r in strat.refused)
        assert all(set(r) == {"capability", "reason", "line_sha256", "span_start", "span_end"}
                   for r in strat.refused)


def test_a_tuned_strategy_forgets_the_previous_documents_refusals():
    # AR-rag4ds-039 r1 F2: tools/extract_pack.py builds each strategy once and
    # drives every document through it; without the per-document reset the
    # second document's manifest would carry the first document's refusals
    # (and a different extraction id). Witnessed against a fresh strategy.
    shared = RuleStrategy(headings=False, tuning=True)
    first = run_extraction(_doc(), [shared], COMMIT, TS)
    first_hashes = {i["line_sha256"]
                    for i in first.manifest["refused_by_tuning"]["rule-v1"]["items"]}
    assert len(first_hashes) >= 1
    # lines that share nothing with TEXT, so no refusal hash can coincide by content
    second_text = ("Gear backlash is the play between meshing teeth. "
                   "Check the brake pads for wear before service. "
                   "Gear  backlash is the play between meshing teeth.\n")
    second = run_extraction(_doc(second_text), [shared], COMMIT, TS)
    alone = run_extraction(_doc(second_text), [RuleStrategy(headings=False, tuning=True)],
                           COMMIT, TS)
    r_second = second.manifest["refused_by_tuning"]["rule-v1"]
    r_alone = alone.manifest["refused_by_tuning"]["rule-v1"]
    assert r_second["count"] == r_alone["count"] >= 1
    assert r_second["items"] == r_alone["items"]
    assert shared.refused == r_alone["items"]
    assert first_hashes.isdisjoint({i["line_sha256"] for i in r_second["items"]})
    assert second.manifest["extraction_id"] == alone.manifest["extraction_id"]


# ----- tune-v2 (S9e, 2026-09-21): the two rules the first filed tuned run showed were missing


def test_a_citation_line_is_refused_before_any_term_is_read():
    """21 of the 24 candidates of the first filed tuned run were the 2026
    paper's reference list: "https://doi.org/10.36001/..." read as the
    definition "https - doi org 10 36001 ...". A line carrying a URL or a
    DOI is a citation, not a statement, whatever else is on it."""
    text = ("Bearing wear is the dominant failure mode of rotary joints.\n"
            "https://doi.org/10.36001/ijphm.2016.v7i2.2365.\n"
            "Harmonic drives are described at www.example.org/hd.\n"
            "The reference doi 10.1016/j.jmsy.2022.12.001 describes it.\n"
            "Torque ripple is a periodic variation of the output torque.\n")
    s = RuleStrategy(headings=False, tuning=True)
    got = [p.content for p in s.propose(_doc(text))]
    assert got == ["Bearing wear - dominant failure mode of rotary joints",
                   "Torque ripple - periodic variation of the output torque"], got
    reasons = [r["reason"] for r in s.refused]
    assert reasons.count("citation") >= 1, reasons
    assert all(r["capability"] == "definitions" for r in s.refused if r["reason"] == "citation")
    assert all("text" not in r for r in s.refused)
    # untuned, the same lines still come through - the rule is the tuning's
    off = [p.content for p in RuleStrategy(headings=False).propose(_doc(text))]
    assert any("doi" in c for c in off), off


def test_back_matter_is_out_of_scope_for_every_capability():
    """Nothing at or past the last references heading in the second half of
    the document is proposed - definitions, tables and headings alike."""
    body = ("Bearing wear is the dominant failure mode of rotary joints. " * 6 + "\n"
            "Torque ripple is a periodic variation of the output torque.\n")
    # the heading is followed by a sentence-terminated line so the sentence
    # splitter does not glue the heading to the definition after it (the
    # pre-existing splitter behaviour the next test also works around)
    back = ("References\n"
            "See the list below.\n"
            "Stall torque is the torque at zero speed, measured at the shaft.\n"
            "| item | value |\n|---|---|\n| Payload | 5 kg |\n")
    text = body + back
    s = RuleStrategy(headings=False, tables=True, tuning=True)
    got = [p.content for p in s.propose(_doc(text))]
    assert got == ["Bearing wear - dominant failure mode of rotary joints",
                   "Torque ripple - periodic variation of the output torque"], got
    from rag4ds.extract.strategies import back_matter_start
    off = back_matter_start(text)
    assert off == text.index("References")
    refused = [r for r in s.refused if r["reason"] == "back-matter"]
    assert refused and all(r["span_start"] >= off for r in refused), s.refused
    # AR-rag4ds-045 O2: each refusal names the capability that produced the
    # proposal, never a capability called "back-matter"
    assert {r["capability"] for r in refused} == {"definitions", "tables"}, refused
    # the definition past the heading is refused by the definitions capability
    # with the reason; anything else past it is dropped in propose() under the
    # same reason - either way, nothing from back matter is proposed
    assert not any(p.span_start >= off for p in s.propose(_doc(text)))


class FAKE_KeywordEmbedder:  # noqa: N801 - a clearly named test double (charter section 8)
    """Two dimensions: [the row mentions 'torque', 1]. A seed 'torque ripple'
    scores 1.0 against a torque sentence and 0.707 against any other."""
    model_id, revision = "FAKE_keyword", "0"

    def embed(self, rows):
        return [[1.0 if "torque" in r.casefold() else 0.0, 1.0] for r in rows]


def test_back_matter_is_out_of_scope_for_the_embedding_strategy_too():
    """AR-rag4ds-045 F3: the filed run's embed-v1 proposed two reference-list
    lines because the back-matter rule lived in the rule strategy alone.
    Tuned, the embedding strategy refuses a back-matter sentence that would
    have competed for a slot, records it in the contract's shape under its
    own capability, and gives the slot to the document's text; untuned it
    is unchanged, and its description carries the rule only when on."""
    from rag4ds.extract.strategies import EmbeddingStrategy, back_matter_start
    body = ("Bearing wear is the dominant failure mode of rotary joints. " * 6
            + "Torque ripple is a periodic variation of the output torque.\n")
    back = ("References\n"
            "Torque sensing in a harmonic drive, a paper about torque, journal 9(2), 2022.\n"
            "Another reference about bearings, journal 3(1), 2020.\n")
    text = body + back
    off = back_matter_start(text)
    assert off == text.index("References")
    seeds = ["torque ripple"]
    untuned = EmbeddingStrategy(FAKE_KeywordEmbedder(), seeds, top_k=2, floor=0.9)
    got = untuned.propose(_doc(text))
    assert [p.span_start >= off for p in got] == [False, True] and untuned.refused == []
    assert "tuning" not in untuned.describe()["params"]
    tuned = EmbeddingStrategy(FAKE_KeywordEmbedder(), seeds, top_k=2, floor=0.9, tuning=True)
    got = tuned.propose(_doc(text))
    assert len(got) == 1 and got[0].span_start < off and "Torque ripple" in got[0].content
    assert len(tuned.refused) == 1
    rec = tuned.refused[0]
    assert rec["capability"] == "related" and rec["reason"] == "back-matter"
    assert rec["span_start"] >= off and set(rec) == {"capability", "reason", "line_sha256",
                                                     "span_start", "span_end"}
    assert tuned.describe()["params"]["tuning"]["version"] == "tune-v2"
    # the refusal reaches the manifest under this strategy's name; a second
    # document forgets the first's refusals
    result = run_extraction(_doc(text), [tuned], COMMIT, TS)
    assert result.manifest["refused_by_tuning"]["embed-v1"]["count"] == 1
    assert tuned.propose(_doc(body)) and tuned.refused == []


def test_the_extraction_tool_tunes_the_embedding_strategy_only_when_asked(monkeypatch):
    """The --tuning flag reaches EmbeddingStrategy through tools/extract_pack.py."""
    from rag4ds.extract import strategies as mod
    seen = {}

    class FAKE_Embedder:  # noqa: N801
        def __init__(self, *a, **k):
            pass

        def embed(self, rows):
            return [[1.0, 1.0] for _ in rows]

    real = mod.EmbeddingStrategy

    def spy(embedder, seeds, top_k=3, floor=0.4, tuning=False):
        seen["tuning"] = tuning
        return real(embedder, seeds, top_k=top_k, floor=floor, tuning=tuning)

    monkeypatch.setattr(extract_pack, "EmbeddingStrategy", spy)
    import rag4ds.index.build as build
    monkeypatch.setattr(build, "SentenceTransformerEmbedder", FAKE_Embedder)
    monkeypatch.setattr(extract_pack, "load_documents", lambda pack, root: ([], []))
    for tuning in (False, True):
        seen.clear()
        rc = extract_pack.run("robotics", MODULE_ROOT, ["embed-v1"],
                              {"definitions": True, "headings": False, "tables": True,
                               "tuning": tuning}, 1, 0.5, True, out=open(os.devnull, "w"),
                              clock=lambda: TS)
        assert rc == 0 and seen["tuning"] is tuning, (tuning, seen)


def test_a_references_heading_in_the_first_half_does_not_end_the_document():
    """The design limit, stated: a manual that names a references section
    early keeps everything after it; only the citation rule bites there."""
    # the heading is followed by a sentence-terminated line: the sentence
    # splitter joins a bare heading to the sentence after it (pre-existing
    # behaviour, not tune-v2's), so the fixture keeps them apart
    text = ("References\n"
            "See the list below.\n"
            "Bearing wear is the dominant failure mode of rotary joints. " * 4 + "\n"
            "https://doi.org/10.36001/ijphm.2016.v7i2.2365.\n"
            "Torque ripple is a periodic variation of the output torque. " * 4 + "\n")
    from rag4ds.extract.strategies import back_matter_start
    assert back_matter_start(text) is None
    s = RuleStrategy(headings=False, tuning=True)
    got = [p.content for p in s.propose(_doc(text))]
    assert "Bearing wear - dominant failure mode of rotary joints" in got
    assert "Torque ripple - periodic variation of the output torque" in got
    # the repeated sentences are exact duplicates and the DOI line splits into
    # several citation refusals; what matters is that NOTHING was refused as
    # back matter, because the heading sits in the first half
    reasons = {r["reason"] for r in s.refused}
    assert "citation" in reasons and "back-matter" not in reasons, s.refused
    assert reasons <= {"citation", "exact-duplicate"}, reasons


def test_the_references_heading_shapes_that_count_and_those_that_do_not():
    from rag4ds.extract.strategies import back_matter_start
    filler = "x " * 200 + "\n"
    for head in ("References", "REFERENCES", "7 References", "7.1 References:",
                 "Bibliography", "Reference list", "Works Cited"):
        t = filler + head + "\n" + "y " * 20
        assert back_matter_start(t) == len(filler), head
    for not_head in ("References are listed below.", "See the references section",
                     "References and Notes on Methods"):
        t = filler + not_head + "\n" + "y " * 20
        assert back_matter_start(t) is None, not_head
    # the LAST heading in the second half wins
    t = filler + "References\n" + "a " * 50 + "\nReferences\n" + "b " * 5
    assert back_matter_start(t) == t.rindex("References")

