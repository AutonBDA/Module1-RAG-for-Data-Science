"""S9c part 6 (the PI's option C, 2026-09-17) as amended by S9e part 4 (the
PI's YES, 2026-09-21): the extraction tool's span-level "already held"
check. A proposal whose source span (document hash, start, end) staging
already holds - PENDING or GROUPED as well as APPROVED, REJECTED or
REJECTED-DUPLICATE - is withheld from filing, whatever its candidate id; so
is a second proposal of the same run on a span the run has just filed
(AR-rag4ds-046 r1 F3); the run manifest records what was withheld, citing
the held candidate and its state, a terminal state winning the citation
over a pending one (AR-rag4ds-046 r1 F2); the check can be switched off for
measurement, in which case the manifest records what it would have
withheld.
"""

import re
import shutil
import sys
from pathlib import Path

from rag4ds.auditlog.journal import Journal
from rag4ds.staging.store import StagingStore

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))
sys.path.insert(0, str(MODULE_ROOT / "tests"))
import extract_pack  # noqa: E402
from test_s9c_extract import CLOCK, FLAGS, _scratch  # noqa: E402

TUNED = dict(FLAGS, tuning=True)


def _span(rec):
    c = rec["candidate"]
    return (c["source_hash"], c["span_start"], c["span_end"])


def _file_untuned(root):
    assert extract_pack.run("bio", root, ["rule-v1"], FLAGS, 1, 0.5, False, clock=CLOCK) == 0
    store = StagingStore(root / "stage" / "candidates.jsonl")
    cands = store.candidates()
    assert len(cands) >= 3, cands
    return store, cands


def test_a_decided_span_is_withheld_and_the_manifest_says_why(tmp_path, capsys):
    root, h = _scratch(tmp_path)
    store, cands = _file_untuned(root)
    capsys.readouterr()
    rejected, approved = sorted(cands)[:2]
    store.transition(rejected, "REJECTED", CLOCK(), "test", "journal:1")
    store.transition(approved, "APPROVED", CLOCK(), "test", "journal:2")
    # a tuned run proposes new candidate ids (the run manifest is in the id)
    # on the same spans: the two decided spans are withheld with their
    # decision, and - S9e part 4 - the pending ones with PENDING; nothing is
    # filed twice
    assert extract_pack.run("bio", root, ["rule-v1"], TUNED, 1, 0.5, True, clock=CLOCK) == 0
    out = capsys.readouterr().out
    # every proposal sits on a held span; a span may carry more than one
    # proposal (the F9 twin case), so the count is at least the candidates'
    n_withheld = int(re.search(r"withheld (\d+) \(span already held\)", out).group(1))
    assert n_withheld >= len(cands) >= 3, out
    assert "would file 0" in out and "DRY RUN" in out
    before = len(store.candidates())
    assert extract_pack.run("bio", root, ["rule-v1"], TUNED, 1, 0.5, False, clock=CLOCK) == 0
    out = capsys.readouterr().out
    assert f"withheld {n_withheld} (span already held)" in out, out
    after = store.candidates()
    filed_now = {cid: rec for cid, rec in after.items() if cid not in cands}
    assert len(after) == before and not filed_now  # nothing re-filed under a new id
    manifests = Journal(root / "runs" / "extraction.jsonl").read_all()
    m = manifests[-1]
    assert m["config"]["rule_flags"] == TUNED
    block = m["span_check"]
    assert block["enabled"] is True and block["count"] == n_withheld == len(block["withheld"])
    # the tuned run refuses some of the untuned lines, so it cites a subset of
    # the filed candidates - always including the two decided ones
    cited = {w["decided_id"] for w in block["withheld"]}
    assert {rejected, approved} <= cited <= set(cands), cited
    by_decided = {w["decided_id"]: w for w in block["withheld"]}
    assert by_decided[rejected]["decision"] == "REJECTED"
    assert by_decided[approved]["decision"] == "APPROVED"
    assert {w["decision"] for w in block["withheld"]
            if w["decided_id"] not in (rejected, approved)} == {"PENDING"}
    for w in block["withheld"]:
        assert set(w) == {"candidate_id", "extractor", "span_start", "span_end",
                          "decided_id", "decision", "decided_pack"}
        assert w["decided_pack"] == "bio"
        assert w["candidate_id"] not in after  # withheld means not filed
        assert (h, w["span_start"], w["span_end"]) == _span(cands[w["decided_id"]])
    # the first, untuned run withheld only its own within-run twins (AR-046 r1
    # F3: the fixture's tables capability proposes two rows on one span), each
    # citing a candidate that run filed, state PENDING
    twins = manifests[0].get("span_check", {}).get("withheld", [])
    assert all(w["decided_id"] in cands and w["decision"] == "PENDING" for w in twins), twins


def test_the_extraction_id_does_not_depend_on_what_staging_holds(tmp_path, capsys):
    # AR-rag4ds-040 r1 F1: the same extraction in two stores - one that holds
    # candidates (some decided), one that holds nothing - yields ONE
    # extraction id and one run_manifest_ref; only the holding store's
    # manifest carries span_check, and the ids it withheld are exactly the
    # ids the empty store filed for those spans. (S9e part 4: the control
    # store is copied BEFORE the untuned filing, since a pending span is
    # now held too.)
    root_a, _ = _scratch(tmp_path)
    root_b = tmp_path / "mod_b"
    shutil.copytree(root_a, root_b)  # the same repo and journal, staging empty
    store_a, cands = _file_untuned(root_a)
    a1, a2 = sorted(cands)[:2]
    store_a.transition(a1, "REJECTED", CLOCK(), "test", "journal:5")
    store_a.transition(a2, "APPROVED", CLOCK(), "test", "journal:6")
    for root in (root_a, root_b):
        assert extract_pack.run("bio", root, ["rule-v1"], TUNED, 1, 0.5, False, clock=CLOCK) == 0
    capsys.readouterr()
    m_a = Journal(root_a / "runs" / "extraction.jsonl").read_all()[-1]
    m_b = Journal(root_b / "runs" / "extraction.jsonl").read_all()[-1]
    assert m_a["extraction_id"] == m_b["extraction_id"]
    assert m_a["run_manifest_ref"] == m_b["run_manifest_ref"]
    assert ({k: v for k, v in m_a.items() if k != "span_check"}
            == {k: v for k, v in m_b.items() if k != "span_check"})
    # the holding store withheld every proposal (citing the earlier candidates);
    # the empty store filed all but its own within-run twins (AR-046 r1 F3) -
    # the same proposal ids, split differently
    filed_b = set(StagingStore(root_b / "stage" / "candidates.jsonl").candidates()) - set(cands)
    twins_b = {w["candidate_id"] for w in m_b.get("span_check", {}).get("withheld", [])}
    withheld_a = {w["candidate_id"] for w in m_a["span_check"]["withheld"]}
    assert withheld_a and withheld_a == filed_b | twins_b and filed_b.isdisjoint(twins_b)
    assert {w["decided_id"] for w in m_a["span_check"]["withheld"]} <= set(cands)
    filed_a = set(store_a.candidates()) - set(cands)
    assert filed_a == set()


def test_a_decided_mapping_draft_in_staging_is_ignored_by_the_check(tmp_path, capsys):
    # AR-rag4ds-040 r1 F2: a mapping-draft candidate has no span; a decided
    # one must neither crash the run nor enter the decided map
    root, _ = _scratch(tmp_path)
    store, cands = _file_untuned(root)
    mid = store.file_candidate({"kind": "mapping-draft", "yaml_draft": "robot: joint\n",
                                "pack": "bio", "standard": "ieee-1872", "author": "author:test"},
                               CLOCK())
    store.transition(mid, "REJECTED", CLOCK(), "test", "journal:7")
    assert store.state_of(mid) == "REJECTED"
    assert mid not in {cid for cid, _ in extract_pack.decided_spans(store, "bio").values()}
    assert mid not in {cid for cid, _ in extract_pack.held_spans(store, "bio").values()}
    capsys.readouterr()
    # S9e part 4: the entry candidates' own spans are held (PENDING) and
    # withheld; the mapping draft is never what withholds them
    assert extract_pack.run("bio", root, ["rule-v1"], TUNED, 1, 0.5, False, clock=CLOCK) == 0
    out = capsys.readouterr().out
    assert "REFUSED" not in out and "withheld" in out, out
    m = Journal(root / "runs" / "extraction.jsonl").read_all()[-1]
    assert m["span_check"]["withheld"] and all(
        w["decided_id"] != mid and w["decision"] == "PENDING" for w in m["span_check"]["withheld"])


def test_when_two_decided_candidates_share_a_span_the_first_filed_is_cited(tmp_path, capsys):
    # AR-rag4ds-040 r1 O1: the tie-break of decided_spans
    root, _ = _scratch(tmp_path)
    store, cands = _file_untuned(root)
    # a definition span (survives a tables-off configuration below)
    first = next(cid for cid, rec in cands.items()
                 if "backlash" in rec["candidate"]["proposed"]["content"].lower())
    span = _span(cands[first])
    store.transition(first, "REJECTED", CLOCK(), "test", "journal:8")
    # a second candidate at the same span, filed later under another id (check off), then approved
    assert extract_pack.run("bio", root, ["rule-v1"], TUNED, 1, 0.5, False, clock=CLOCK,
                            span_check=False) == 0
    later = [cid for cid, rec in store.candidates().items()
             if cid not in cands and _span(rec) == span]
    assert len(later) == 1
    store.transition(later[0], "APPROVED", CLOCK(), "test", "journal:9")
    assert extract_pack.decided_spans(store, "bio")[span] == (first, "REJECTED")
    # a third configuration proposes the span under a third id: the first filed is cited
    capsys.readouterr()
    third = dict(TUNED, tables=False)
    assert extract_pack.run("bio", root, ["rule-v1"], third, 1, 0.5, False, clock=CLOCK) == 0
    m = Journal(root / "runs" / "extraction.jsonl").read_all()[-1]
    assert m["config"]["rule_flags"] == third
    cited = {(w["span_start"], w["span_end"]): (w["decided_id"], w["decision"])
             for w in m["span_check"]["withheld"]}
    assert cited[(span[1], span[2])] == (first, "REJECTED")


def test_pending_and_grouped_spans_are_held_and_withheld_too(tmp_path, capsys):
    """S9e part 4 (the PI's YES of 2026-09-21, reversing one clause of decision
    C): a span staging already HOLDS - pending or grouped, not only decided -
    is not filed again under a new id. Before this, a tuned re-run whose
    manifest differed re-filed every PENDING span (AR-rag4ds-045 20.4:
    would file 107, the 105 pending twice). The manifest names the held
    candidate and its state."""
    root, _ = _scratch(tmp_path)
    store, cands = _file_untuned(root)
    capsys.readouterr()
    head, member = sorted(cands)[:2]
    store.transition(member, "GROUPED", CLOCK(), "test", "journal:3", head=head)
    assert extract_pack.run("bio", root, ["rule-v1"], TUNED, 1, 0.5, True, clock=CLOCK) == 0
    out = capsys.readouterr().out
    assert re.search(r"would file 0, already filed 0; withheld [1-9]\d* \(span already held\)",
                     out), out
    held = extract_pack.held_spans(store, "bio")
    states = {cid: st for cid, st in held.values()}
    assert states[member] == "GROUPED" and states[head] == "PENDING"
    assert set(states.values()) <= {"PENDING", "GROUPED"}
    # the terminal subset is unchanged: nothing here is decided
    assert extract_pack.decided_spans(store, "bio") == {}


def test_a_span_holding_a_pending_and_a_decided_candidate_is_cited_by_its_decision(tmp_path,
                                                                                    capsys):
    """AR-rag4ds-046 r1 F2: the citation is truthful before it is first-filed.
    In either filing order, a span carrying both an undecided and a decided
    candidate is cited by the decided one, held_spans() and decided_spans()
    agree, and the withheld row says REJECTED, never PENDING."""
    for order in ("pending-first", "decided-first"):
        root, _ = _scratch(tmp_path / order)
        store, cands = _file_untuned(root)
        first = sorted(cands)[0]
        span = _span(cands[first])
        # a second candidate on the same span under another id (check off)
        assert extract_pack.run("bio", root, ["rule-v1"], TUNED, 1, 0.5, False, clock=CLOCK,
                                span_check=False) == 0
        later = [cid for cid, rec in store.candidates().items()
                 if cid not in cands and _span(rec) == span]
        assert len(later) == 1
        decided_one = later[0] if order == "pending-first" else first
        store.transition(decided_one, "REJECTED", CLOCK(), "test", "journal:11")
        assert extract_pack.held_spans(store, "bio")[span] == (decided_one, "REJECTED")
        assert extract_pack.decided_spans(store, "bio")[span] == (decided_one, "REJECTED")
        capsys.readouterr()
        third = dict(TUNED, tables=False)
        assert extract_pack.run("bio", root, ["rule-v1"], third, 1, 0.5, False, clock=CLOCK) == 0
        m = Journal(root / "runs" / "extraction.jsonl").read_all()[-1]
        rows = [w for w in m["span_check"]["withheld"]
                if (w["span_start"], w["span_end"]) == span[1:]]
        assert rows and all((w["decided_id"], w["decision"]) == (decided_one, "REJECTED")
                            for w in rows), (order, rows)


def test_one_run_never_files_two_candidates_on_one_span(tmp_path, capsys, monkeypatch):
    """AR-rag4ds-046 r1 F3: the held map learns what the run itself files, so
    a rule-v1/embed-v1 pair on one sentence, or two tables twins, file once;
    the second is withheld like any other, citing the candidate this run
    filed, with state PENDING. Witnessed with a keyword embedder whose seed
    picks the definition sentence rule-v1 also proposes."""
    import rag4ds.index.build as build

    class FAKE_Embedder:  # noqa: N801 - a clearly named test double (charter section 8)
        model_id, revision = "FAKE_keyword", "0"

        def __init__(self, *a, **k):
            pass

        def embed(self, rows):
            return [[1.0 if "backlash" in r.casefold() else 0.0, 1.0] for r in rows]

    monkeypatch.setattr(build, "SentenceTransformerEmbedder", FAKE_Embedder)
    root, _ = _scratch(tmp_path)
    manifest = root / "packs" / "bio" / "manifest.yaml"
    text = manifest.read_text(encoding="utf-8")
    assert "lexicon" in text
    capsys.readouterr()
    assert extract_pack.run("bio", root, ["rule-v1", "embed-v1"], TUNED, 3, 0.9, False,
                            clock=CLOCK) == 0
    out = capsys.readouterr().out
    store = StagingStore(root / "stage" / "candidates.jsonl")
    spans = {}
    for cid, rec in store.candidates().items():
        spans.setdefault(_span(rec), []).append(cid)
    assert spans and all(len(v) == 1 for v in spans.values()), spans
    m = Journal(root / "runs" / "extraction.jsonl").read_all()[-1]
    rows = m["span_check"]["withheld"]
    assert rows, out  # the pair and/or the twins were caught
    filed = set(store.candidates())
    assert all(w["decided_id"] in filed and w["decision"] == "PENDING"
               and w["candidate_id"] not in filed for w in rows), rows
    assert "withheld" in out and re.search(r"withheld [1-9]\d* \(span already held\)", out), out


def test_the_check_can_be_switched_off_and_the_manifest_records_what_it_would_withhold(
        tmp_path, capsys):
    root, _ = _scratch(tmp_path)
    store, cands = _file_untuned(root)
    capsys.readouterr()
    rejected = sorted(cands)[0]
    store.transition(rejected, "REJECTED", CLOCK(), "test", "journal:4")
    assert extract_pack.run("bio", root, ["rule-v1"], TUNED, 1, 0.5, False, clock=CLOCK,
                            span_check=False) == 0
    out = capsys.readouterr().out
    # S9e part 4: every proposal sits on a held span (one decided, the rest
    # pending), so with the check off all of them would have been withheld
    n = int(re.search(r"would withhold (\d+) \(span already held; check off\)", out).group(1))
    assert n >= len(cands) >= 3, out
    assert f"{n} would be withheld (span already held; check off)" in out, out  # the totals (O2)
    after = store.candidates()
    assert any(_span(rec) == _span(cands[rejected]) for cid, rec in after.items()
               if cid not in cands)  # the decided span WAS filed again
    m = Journal(root / "runs" / "extraction.jsonl").read_all()[-1]
    assert m["span_check"]["enabled"] is False and m["span_check"]["count"] == n
    would = {w["decided_id"]: w["decision"] for w in m["span_check"]["would_withhold"]}
    assert would[rejected] == "REJECTED"
    assert set(would.values()) == {"REJECTED", "PENDING"}
    assert "withheld" not in m["span_check"]


def test_a_run_with_nothing_to_withhold_carries_no_block_and_the_flag_is_wired(tmp_path, capsys,
                                                                              monkeypatch):
    root, _ = _scratch(tmp_path)
    # tables off: the fixture's tables capability proposes twins on one span,
    # which the run itself now withholds (AR-046 r1 F3); definitions alone
    # give distinct spans, so this run has nothing to withhold
    assert extract_pack.run("bio", root, ["rule-v1"], dict(FLAGS, tables=False), 1, 0.5, False,
                            clock=CLOCK) == 0
    capsys.readouterr()
    m = Journal(root / "runs" / "extraction.jsonl").read_all()[0]
    assert "span_check" not in m  # an untouched manifest shape when the check bites nothing
    seen = {}

    def fake_run(pack, module_root, names, flags, top_k, floor, dry_run, span_check=True):
        seen.update(pack=pack, span_check=span_check, flags=flags)
        return 0

    monkeypatch.setattr(extract_pack, "run", fake_run)
    assert extract_pack.main(["bio", "--dry-run"]) == 0
    assert seen["span_check"] is True
    assert extract_pack.main(["bio", "--dry-run", "--no-span-check"]) == 0
    assert seen["span_check"] is False


def test_a_decision_of_another_pack_on_the_same_document_does_not_withhold(tmp_path, capsys):
    # AR-rag4ds-041 r1 O1: the decided map is scoped to the pack being run; a
    # document that serves two packs is decided by each pack's validator
    # separately, and the withheld row names the deciding pack
    root, _ = _scratch(tmp_path)
    store, cands = _file_untuned(root)
    first = sorted(cands)[0]
    twin = dict(cands[first]["candidate"], pack="other")  # the same document and span, another pack
    other = store.file_candidate(twin, CLOCK())
    store.transition(other, "REJECTED", CLOCK(), "test", "journal:10")
    assert _span(store.candidates()[other]) == _span(cands[first])
    assert extract_pack.decided_spans(store, "bio") == {}
    assert extract_pack.decided_spans(store, "other") == {_span(cands[first]): (other, "REJECTED")}
    assert other not in {cid for cid, _ in extract_pack.held_spans(store, "bio").values()}
    capsys.readouterr()
    # S9e part 4: bio's own pending candidates hold their spans and are
    # withheld under bio's name; the other pack's decision is never cited
    assert extract_pack.run("bio", root, ["rule-v1"], TUNED, 1, 0.5, False, clock=CLOCK) == 0
    m = Journal(root / "runs" / "extraction.jsonl").read_all()[-1]
    rows = m["span_check"]["withheld"]
    assert rows and all(w["decided_pack"] == "bio" and w["decided_id"] != other
                        and w["decision"] == "PENDING" for w in rows), rows
