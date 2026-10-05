"""S9 demo road, parts A and B (PI review list 5, D1): the evaluation hook's
demo-flow runner (src/rag4ds/evalhook/demo.py, tools/run_demo.py), the
provenance chain of an entry version (src/rag4ds/auditlog/provenance.py),
the robotics pack's three demo flows - bound in part B from the validator's
certified list (FR9) - and acceptance case A4 on part 8's real correction of
the two harmonic-drive entries."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from rag4ds.auditlog.journal import Journal
from rag4ds.auditlog.provenance import ProvenanceError, approvals, resolve
from rag4ds.evalhook.demo import (
    FAIL, PASS, UNBOUND, DemoFlow, DemoFlowError, Expected, check, load_flows, run, unserved,
)
from rag4ds.gate.identities import IDENTITIES_FILE, IdentityRegistry
from rag4ds.index.build import build_index, load_index, tree_entries
from rag4ds.retrieve.contract import GroundingRequest, GroundingRequestError, shown
from rag4ds.retrieve.service import Retriever
from rag4ds.staging.store import StagingStore

MODULE_ROOT = Path(__file__).resolve().parents[1]
ROBOTICS = MODULE_ROOT / "packs" / "robotics"
TS = "2026-09-27 21:00:00 JST"
CORRECTED = ("harmonic-drive-use", "harmonic-drive-torque")   # part 8, journal 374-375


class FAKE_BagOfWordsEmbedder:  # noqa: N801 - a clearly named test double (charter section 8)
    model_id = "FAKE-bag-of-words"
    revision = "test-double"
    dim = 4096

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace("-", " ").split():
                out[i, int(hashlib.sha256(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            if not out[i].any():
                out[i, 0] = 1.0
        return out


def _tool():
    spec = importlib.util.spec_from_file_location("run_demo_tool",
                                                  MODULE_ROOT / "tools" / "run_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _live_index(tmp_path, min_relevance=0.05):
    """An index over the live gated tree, built with the fake embedder into
    scratch - the served index and the real model are never touched."""
    out = tmp_path / "index"
    build_index(MODULE_ROOT, out, FAKE_BagOfWordsEmbedder(), {"robotics": "0.0.0"},
                min_relevance, "0000000", TS)
    return out


def _journal():
    return Journal(MODULE_ROOT / "journal" / "decisions.jsonl").read_all()


def _v1_content(entry_id):
    return [r["entry"]["content"] for r in _journal() if r.get("type") in (
        "approve", "edit-then-approve") and r.get("entry", {}).get("id") == entry_id
        and r["entry"]["version"] == 1][0]


# ----- the flows file as written ----------------------------------------------------------------

CERTIFIED = "authoring/robotics_s9_demo_naruse_list_2_certified.txt"
CERTIFIED_SHA256 = "c882c94acca9c2bc916d3b2b395d562e82d38114639e375fe8b4f6f0147c4a58"
CERTIFIED3 = "authoring/robotics_s9_demo_naruse_list_3_certified.txt"
CERTIFIED3_SHA256 = "6a86dc40c56cd97ca7a0caa2a9e6df244c0d8d38d201fc881191f5521518703d"


def _list3_mark(query):
    """NARUSE LIST 3, the one row put back to him: (entry id, the letter of the
    box he ticked), read from his certified list itself, which must be the one
    pinned when it came back and must ask about this question."""
    import re
    path = MODULE_ROOT / CERTIFIED3
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CERTIFIED3_SHA256
    text = path.read_text(encoding="ascii")
    assert f"\n  Question 2   \"{query}\"\n" in text
    entry = re.search(r"\n  Entry 10     (\S+)\n", text).group(1)
    block = text.split("YOUR MARK for entry 10 in question 2:\n", 1)[1].split("\n\n", 1)[0]
    ticked = re.findall(r"\[([ARX]) \] ([ARX])\s", block)
    assert len(ticked) == 1 and ticked[0][0] == ticked[0][1], ticked
    assert block.count("[ ]") == 2
    return entry, ticked[0][0]
_MARK_ROW = (r"^\s+(\d+)\s+(\S+)\s+\*?\s+([ARX])\s+\[(A?|R?|X?) ?\]\s+\[(A?|R?|X?) ?\]\s+"
             r"\[(A?|R?|X?) ?\]$")


def _certified_marks():
    """The validator's marks, read from his certified list itself: for each of
    questions 1 and 2, entry id -> the letter in the box he filled."""
    import re
    rows = [re.match(_MARK_ROW, line) for line in
            (MODULE_ROOT / CERTIFIED).read_text(encoding="ascii").split("\n")]
    rows = [m.groups() for m in rows if m]
    assert len(rows) == 24, len(rows)
    marks = []
    for n, entry, _sa, a, r, x in rows:
        filled = [(col, v) for col, v in zip("ARX", (a, r, x), strict=True) if v]
        # one box filled, holding its own column's letter
        assert len(filled) == 1 and filled[0][0] == filled[0][1], (n, entry, filled)
        marks.append((entry, filled[0][0]))
    return dict(marks[:12]), dict(marks[12:])


def test_the_robotics_flows_bind_the_validators_certified_list():
    """Plan 5.2's questions 1 and 2 word for word, and the team's question 3
    filtered to the domain layer, bound from K. Naruse's certified list (the
    S9 demo road's part B): every flow's expected outcome is what the marked
    list says - read from the list itself, not restated - and every id a
    binding names is an entry of the live tree (AR-rag4ds-057 r1 O3)."""
    pack, flows = load_flows(ROBOTICS)
    assert pack == "robotics"
    assert [f.id for f in flows] == ["torque-before-failure", "failure-prediction-data",
                                     "out-of-domain"]
    plan = " ".join((MODULE_ROOT / "docs" / "plan_module1.md").read_text(encoding="utf-8").split())
    for f in flows[:2]:
        assert f"\"{f.query}\"" in plan, f.query
    assert flows[2].request == {"layer": "domain", "domain": "robotics", "top_k": 5}
    text = (MODULE_ROOT / CERTIFIED).read_text(encoding="ascii")
    # AR-rag4ds-058 r1 O4: the list read here is the one pinned when it came back
    assert hashlib.sha256((MODULE_ROOT / CERTIFIED).read_bytes()).hexdigest() == CERTIFIED_SHA256
    # AR-rag4ds-058 r1 F1: each flow asks what the list showed him - its question
    # word for word, and questions 1 and 2 on the terms the list stated ("Asked of
    # the robotics library; at most the five best-matching entries come back")
    for n, f in enumerate(flows, 1):
        assert f"\nQUESTION {n}   \"{f.query}\"\n" in text, f.id
    assert text.count("\n  Asked of the robotics library; at most the five best-matching "
                      "entries\n  come back.") == 2
    assert flows[0].request == flows[1].request == {"domain": "robotics", "top_k": 5}
    assert "     [A ] A  keep this question: nothing may come back" in text
    assert "\n  " + "_" * 69 + "\n" in text   # no general entry named for questions 1 and 2
    assert "Date: __2026-09-28" in text
    q1, q2 = _certified_marks()
    # AR-rag4ds-058 r1 F2 (PI review list 7 D2): one row of question 2 went back
    # to him in list 3; his mark there replaces his list-2 mark for that row
    entry, mark = _list3_mark(flows[1].query)
    assert entry in q2 and (q2[entry], mark) == ("X", "R"), (entry, q2.get(entry), mark)
    q2[entry] = mark
    for flow, marks in ((flows[0], q1), (flows[1], q2)):
        assert flow.expected.status == "OK"
        assert flow.expected.present == tuple(e for e, m in marks.items() if m == "A"), flow.id
        assert flow.expected.absent == tuple(e for e, m in marks.items() if m == "R"), flow.id
    assert (flows[2].expected.status, flows[2].expected.present, flows[2].expected.absent) == (
        "NO_GROUNDING", (), ())
    for f, record in zip(flows, (CERTIFIED, f"{CERTIFIED}; {CERTIFIED3}", CERTIFIED), strict=True):
        assert (f.expected.bound_by, f.expected.bound_on, f.expected.record) == (
            "K. Naruse", "2026-09-28", record), f.id
    live = {e.id for e in tree_entries(MODULE_ROOT)}
    assert unserved(flows, live) == []
    assert all(check(f, {"status": "OK", "entries": []}).verdict != UNBOUND for f in flows)


# ----- the mechanical check ----------------------------------------------------------------------

def _resp(status, *ids):
    return {"status": status,
            "entries": [{"id": i, "version": 1, "layer": "domain"} for i in ids]}


def _flow(expected):
    return DemoFlow("f", "q", {}, expected)


def test_check_is_mechanical_status_present_and_absent():
    ok = Expected("OK", ("a", "b"), ("z",), "K. Naruse", "2026-09-28", "list")
    assert check(_flow(ok), _resp("OK", "b", "x", "a")).verdict == PASS
    miss = check(_flow(ok), _resp("OK", "a", "x"))
    assert miss.verdict == FAIL and miss.reasons == ("expected entry b not returned",)
    extra = check(_flow(ok), _resp("OK", "a", "b", "z"))
    assert extra.verdict == FAIL and extra.reasons == ("entry z returned, expected absent",)
    status = check(_flow(ok), _resp("NO_GROUNDING"))
    assert status.verdict == FAIL and status.reasons[0] == "status NO_GROUNDING, expected OK"
    none = Expected("NO_GROUNDING", (), (), "K. Naruse", "2026-09-28", "list")
    assert check(_flow(none), _resp("NO_GROUNDING")).verdict == PASS
    assert check(_flow(none), _resp("OK", "a")).verdict == FAIL
    assert check(_flow(None), _resp("OK", "a", "b")).verdict == UNBOUND


class _Recorder:
    def __init__(self):
        self.asked = []

    def ground(self, req):
        self.asked.append(req)

        class R:
            def to_dict(self_inner):
                return _resp("OK", "a")
        return R()


def test_run_asks_the_retrieval_library_each_flows_own_request_in_order():
    flows = [DemoFlow("one", "first question", {"domain": "robotics", "top_k": 3}, None),
             DemoFlow("two", "second question", {"layer": "domain"}, None)]
    rec = _Recorder()
    results = run(rec, flows)
    assert [r.flow_id for r in results] == ["one", "two"]
    assert rec.asked == [GroundingRequest("first question", domain="robotics", top_k=3),
                         GroundingRequest("second question", layer="domain")]


def _write_flows(pack_dir: Path, flows: list, pack: str | None = None):
    (pack_dir / "demo").mkdir(parents=True, exist_ok=True)
    doc = {"schema": 1, "pack": pack or pack_dir.name, "flows": flows}
    (pack_dir / "demo" / "flows.yaml").write_text(yaml.safe_dump(doc, sort_keys=False),
                                                  encoding="utf-8")


BOUND = {"status": "OK", "present": ["a"], "absent": [], "bound_by": "K. Naruse",
         "bound_on": "2026-09-28", "record": "authoring/list.txt"}


@pytest.mark.parametrize("flows,words", [
    ([{"id": "f", "query": "q", "request": {}, "expected": None, "extra": 1}], "exactly id"),
    ([{"id": "F!", "query": "q", "request": {}, "expected": None}], "malformed or repeated"),
    ([{"id": "f", "query": "q", "request": {}, "expected": None}] * 2, "malformed or repeated"),
    ([{"id": "f", "query": "q", "request": {"pack": "x"}, "expected": None}], "may carry only"),
    ([{"id": "f", "query": " ", "request": {}, "expected": None}], "not a valid grounding"),
    ([{"id": "f", "query": "q", "request": {"layer": "nope"}, "expected": None}],
     "not a valid grounding"),
    ([{"id": "f", "query": "q", "request": {}, "expected": {**BOUND, "status": "MAYBE"}}],
     "status must be one of"),
    ([{"id": "f", "query": "q", "request": {}, "expected": {**BOUND, "present": []}}],
     "names the entries it expects"),
    ([{"id": "f", "query": "q", "request": {},
       "expected": {**BOUND, "status": "NO_GROUNDING"}}], "expects no entries present"),
    ([{"id": "f", "query": "q", "request": {}, "expected": {**BOUND, "absent": ["a"]}}],
     "both present and absent"),
    ([{"id": "f", "query": "q", "request": {}, "expected": {**BOUND, "bound_by": ""}}],
     "bound_by must be"),
    ([{"id": "f", "query": "q", "request": {}, "expected": {**BOUND, "bound_on": "28 Sep"}}],
     "bound_on must be"),
    ([{"id": "f", "query": "q", "request": {}, "expected": {**BOUND, "record": None}}],
     "record must be"),
    ([{"id": "f", "query": "q", "request": {},
       "expected": {k: v for k, v in BOUND.items() if k != "record"}}], "must be null"),
    # AR-rag4ds-057 F2: the other direction - a seventh key (a misspelt
    # "absent" beside the real one) is refused, not ignored
    ([{"id": "f", "query": "q", "request": {}, "expected": {**BOUND, "abesnt": ["b"]}}],
     "must be null"),
    ([{"id": "f", "query": "q", "request": {}, "expected": {**BOUND, "present": ["Bad Id"]}}],
     "holds a malformed or repeated entry id"),
    ([{"id": "f", "query": "q", "request": {}, "expected": {**BOUND, "absent": ["b", "b"]}}],
     "holds a malformed or repeated entry id"),
    ([], "non-empty list"),
])
def test_a_flows_file_that_is_not_unambiguous_is_refused(tmp_path, flows, words):
    pack_dir = tmp_path / "packs" / "demo-pack"
    _write_flows(pack_dir, flows)
    with pytest.raises(DemoFlowError, match=words):
        load_flows(pack_dir)


def test_a_flows_file_naming_another_pack_or_schema_is_refused(tmp_path):
    pack_dir = tmp_path / "packs" / "demo-pack"
    _write_flows(pack_dir, [{"id": "f", "query": "q", "request": {}, "expected": None}],
                 pack="other")
    with pytest.raises(DemoFlowError, match="pack directory's own code"):
        load_flows(pack_dir)
    (pack_dir / "demo" / "flows.yaml").write_text("schema: 2\npack: demo-pack\nflows: []\n",
                                                  encoding="utf-8")
    with pytest.raises(DemoFlowError, match="schema must be 1"):
        load_flows(pack_dir)
    # AR-rag4ds-057 F2: a key the file does not know is refused, not ignored
    (pack_dir / "demo" / "flows.yaml").write_text(
        "schema: 1\npack: demo-pack\nflows:\n- {id: f, query: q, request: {}, expected: null}\n"
        "notes: x\n", encoding="utf-8")
    with pytest.raises(DemoFlowError, match="must carry exactly schema, pack and flows"):
        load_flows(pack_dir)
    (pack_dir / "demo" / "flows.yaml").unlink()
    with pytest.raises(DemoFlowError, match="not readable YAML: "):  # r5 O14: the named refusal
        load_flows(pack_dir)


def _flows_text(extra: str = "") -> str:
    return ("schema: 1\npack: demo-pack\nflows:\n- id: f\n  query: q\n  request: {}\n"
            "  expected:\n    status: OK\n    present: [a]\n    absent: [b]\n" + extra
            + "    bound_by: K. Naruse\n    bound_on: '2026-09-28'\n"
            "    record: authoring/list.txt\n")


def test_a_flows_file_giving_a_key_twice_is_refused(tmp_path):
    """AR-rag4ds-057 F1: the file is human-edited YAML, read as the project's
    other human-edited YAML is - a key given twice is refused, never resolved
    to its last value. An `absent:` typed twice would otherwise let a bound
    flow pass on the entry it excludes, and a second `flows:` would silently
    replace every flow above it."""
    pack_dir = tmp_path / "packs" / "demo-pack"
    (pack_dir / "demo").mkdir(parents=True)
    flows = pack_dir / "demo" / "flows.yaml"
    # without the repeat the file loads: the refusal is the repeat's alone
    flows.write_text(_flows_text(), encoding="utf-8")
    exp = load_flows(pack_dir)[1][0].expected
    assert (exp.present, exp.absent) == (("a",), ("b",))
    for text in (_flows_text("    absent: []\n"),
                 _flows_text("    status: NO_GROUNDING\n"),
                 # AR-rag4ds-057 r6 F9: the same repeat inside a merge block
                 _flows_text().replace("    absent: [b]\n",
                                       "    <<:\n      absent: [b]\n      absent: []\n"),
                 # AR-rag4ds-057 r8 F11: the merge key itself given twice, the
                 # second block unsaying the first's absent list
                 _flows_text().replace("    absent: [b]\n",
                                       "    <<: {absent: [b]}\n    <<: {absent: []}\n"),
                 _flows_text() + "flows:\n- {id: g, query: q, request: {}, expected: null}\n",
                 "schema: 1\npack: demo-pack\nflows:\n- id: f\n  id: g\n  query: q\n"
                 "  request: {}\n  expected: null\n"):
        flows.write_text(text, encoding="utf-8")
        # AR-rag4ds-057 r5 O14: the named refusal's own words, which the last
        # line would not print
        with pytest.raises(DemoFlowError, match="duplicate mapping key .*; a key given twice says "
                                                "two things"):
            load_flows(pack_dir)
    # the same two blocks as one merge list are YAML's form for several sources:
    # they load, the earlier source winning, as safe_load reads them
    flows.write_text(_flows_text().replace("    absent: [b]\n",
                                           "    <<: [{absent: [b]}, {absent: []}]\n"),
                     encoding="utf-8")
    assert load_flows(pack_dir)[1][0].expected.absent == ("b",)
    # AR-rag4ds-057 r7 F10: bindings that share one block through a merge chain
    # - each flow merging the one before and overriding a key of it - are YAML's
    # own meaning and load as written, never "a key given twice"
    chained = ("schema: 1\npack: demo-pack\nflows:\n"
               "- id: f\n  query: q\n  request: {}\n  expected: &e1\n    status: OK\n"
               "    present: [a]\n    absent: [b]\n" + "    bound_by: K. Naruse\n"
               "    bound_on: '2026-09-28'\n    record: authoring/list.txt\n"
               "- id: g\n  query: q\n  request: {}\n  expected: &e2\n    <<: *e1\n"
               "    present: [c]\n"
               "- id: h\n  query: q\n  request: {}\n  expected:\n    <<: *e2\n"
               "    status: NO_GROUNDING\n    present: []\n    absent: []\n")
    flows.write_text(chained, encoding="utf-8")
    got = [(x.expected.status, x.expected.present, x.expected.absent)
           for x in load_flows(pack_dir)[1]]
    assert got == [("OK", ("a",), ("b",)), ("OK", ("c",), ("b",)), ("NO_GROUNDING", (), ())]


def test_merge_chains_in_a_flows_file_load_as_yaml_reads_them(tmp_path):
    """AR-rag4ds-057 r7 F10: report r7's merge chains C1-C6, written into a
    flows file - a source that overrides a key it merges, or merges two
    sources giving one key, merged again or built again through an alias -
    load through load_flows exactly as yaml.safe_load reads the same text."""
    pack_dir = tmp_path / "packs" / "demo-pack"
    (pack_dir / "demo").mkdir(parents=True)
    flows = pack_dir / "demo" / "flows.yaml"
    head = ("schema: 1\npack: demo-pack\nflows:\n- id: f\n  query: q\n  request: {}\n"
            "  expected: &b\n    status: OK\n    present: [a]\n    absent: [b]\n"
            "    bound_by: K. Naruse\n    bound_on: '2026-09-28'\n    record: authoring/list.txt\n")

    def item(fid, expected):
        return f"- {{id: {fid}, query: q, request: {{}}, expected: {expected}}}\n"

    shapes = {
        "C1": item("g", "&m {<<: *b, present: [c]}") + item("h", "{<<: *m}"),
        "C2": item("g", "{<<: &s {<<: *b, present: [c]}}") + item("h", "{<<: *s}"),
        "C3": ("- {<<: &s {<<: {id: g0, query: q}, id: g, request: {}, expected: *b}}\n"
               "- {<<: *s, id: h}\n"),
        "C4": (item("g", "&c {<<: *b, present: [c]}") + item("h", "&m {<<: [*b, *c]}")
               + item("i", "{<<: *m}")),
        "C5": item("g", "{<<: &s {<<: *b, present: [c]}}") + item("h", "*s"),
        "C6": item("g", "&m {<<: *b, present: [c]}")
              + item("h", "{<<: [*m, {record: authoring/other.txt}]}"),
    }
    for name, tail in shapes.items():
        text = head + tail
        flows.write_text(text, encoding="utf-8")
        got = [(x.id, x.expected.status, x.expected.present, x.expected.absent, x.expected.record)
               for x in load_flows(pack_dir)[1]]
        want = [(r["id"], r["expected"]["status"], tuple(r["expected"]["present"]),
                 tuple(r["expected"]["absent"]), r["expected"]["record"])
                for r in yaml.safe_load(text)["flows"]]
        assert got == want, name
        # the override is read, not dropped: flow g expects c, not f's a
        assert got[1][2] == (("a",) if name == "C3" else ("c",)), name


def test_an_unreadable_flows_file_is_refused_never_a_traceback(tmp_path, capsys):
    """AR-rag4ds-057 r2 F5: a key that is a sequence or a mapping, and bytes
    that are not UTF-8, are refused as a DemoFlowError - and the tool exits 2,
    "a flows file that does not load", never a traceback's 1."""
    pack_dir = tmp_path / "packs" / "demo-pack"
    (pack_dir / "demo").mkdir(parents=True)
    flows = pack_dir / "demo" / "flows.yaml"
    for text in (_flows_text().replace("    absent: [b]\n", "    [b]: absent\n"),
                 "schema: 1\npack: demo-pack\n[flows]: []\n",
                 "schema: 1\npack: demo-pack\n? {flows: 1}\n: []\n"):
        flows.write_text(text, encoding="utf-8")
        with pytest.raises(DemoFlowError, match="not readable YAML"):
            load_flows(pack_dir)
    latin = _flows_text().replace("query: q", "query: caf\xe9").encode("latin-1")
    flows.write_bytes(latin)
    with pytest.raises(DemoFlowError, match="not UTF-8 text"):
        load_flows(pack_dir)
    root = tmp_path / "mod"
    (root / "packs" / "robotics" / "demo").mkdir(parents=True)
    (root / "packs" / "robotics" / "demo" / "flows.yaml").write_bytes(
        latin.replace(b"demo-pack", b"robotics"))
    assert _tool().execute(root, "robotics", tmp_path / "no-index",
                           embedder=FAKE_BagOfWordsEmbedder()) == 2
    assert "not UTF-8 text" in capsys.readouterr().err
    # AR-rag4ds-057 r3 F6: an impossible date, unquoted - exit 2, not a traceback
    impossible = _flows_text().replace(BOUND_ON, "    bound_on: 2026-09-31\n")
    (root / "packs" / "robotics" / "demo" / "flows.yaml").write_text(
        impossible.replace("demo-pack", "robotics"), encoding="utf-8")
    assert _tool().execute(root, "robotics", tmp_path / "no-index",
                           embedder=FAKE_BagOfWordsEmbedder()) == 2
    assert "a value that cannot be built (ValueError" in capsys.readouterr().err
    # AR-rag4ds-057 r4 F7: a pack code too large to print - exit 2, named by its size
    (root / "packs" / "robotics" / "demo" / "flows.yaml").write_text(
        _flows_text().replace("pack: demo-pack\n", f"pack: {HUGE}\n"), encoding="utf-8")
    assert _tool().execute(root, "robotics", tmp_path / "no-index",
                           embedder=FAKE_BagOfWordsEmbedder()) == 2
    assert "pack <an integer of 14288 bits> must be" in capsys.readouterr().err
    # AR-rag4ds-057 r5 F8: a base-60 float too large for a float - exit 2, by the
    # named "not readable YAML" refusal, not the last line
    (root / "packs" / "robotics" / "demo" / "flows.yaml").write_text(
        impossible.replace("    bound_on: 2026-09-31\n", "    bound_on: 1" + ":00" * 199 + ".5\n")
        .replace("demo-pack", "robotics"), encoding="utf-8")
    assert _tool().execute(root, "robotics", tmp_path / "no-index",
                           embedder=FAKE_BagOfWordsEmbedder()) == 2
    assert "not readable YAML: a value that cannot be built (OverflowError" in \
        capsys.readouterr().err
    # AR-rag4ds-057 r6 F9: a key given twice inside a merge block, which loaded
    # and let a flow pass on the entry its binding excludes - refused, exit 2
    merged = _flows_text().replace("    absent: [b]\n",
                                   "    <<:\n      absent: [b]\n      absent: []\n")
    (root / "packs" / "robotics" / "demo" / "flows.yaml").write_text(
        merged.replace("demo-pack", "robotics"), encoding="utf-8")
    assert _tool().execute(root, "robotics", tmp_path / "no-index",
                           embedder=FAKE_BagOfWordsEmbedder()) == 2
    assert "duplicate mapping key 'absent'" in capsys.readouterr().err
    # AR-rag4ds-057 r8 F11: the merge key itself given twice, which loaded and
    # let a flow pass on the entry its first block excludes - refused, exit 2
    twice = _flows_text().replace("    absent: [b]\n",
                                  "    <<: {absent: [b]}\n    <<: {absent: []}\n")
    (root / "packs" / "robotics" / "demo" / "flows.yaml").write_text(
        twice.replace("demo-pack", "robotics"), encoding="utf-8")
    assert _tool().execute(root, "robotics", tmp_path / "no-index",
                           embedder=FAKE_BagOfWordsEmbedder()) == 2
    assert "duplicate mapping key '<<' at line" in capsys.readouterr().err


BOUND_ON = "    bound_on: '2026-09-28'\n"
RECORD = "    record: authoring/list.txt\n"


@pytest.mark.parametrize("old, new, words", [
    (BOUND_ON, "    bound_on: 2026-09-31\n", "ValueError: day is out of range"),
    (BOUND_ON, "    bound_on: 2026-02-29\n", "ValueError: day is out of range"),
    (RECORD, "    record: !!int q\n", "ValueError: invalid literal"),
    (RECORD, "    record: " + "7" * 5000 + "\n", "ValueError: Exceeds the limit"),
    (RECORD, "    record: !!bool q\n", "KeyError"),
    (RECORD, "    record: !!timestamp q\n", "AttributeError"),
    (RECORD, "    record: !!int ''\n", "IndexError"),
    (RECORD, "    record: !!map q\n", "expected a mapping node, but found scalar"),
    (RECORD, "    record: !!set [a, b]\n", "expected a mapping node, but found sequence"),
    (RECORD, "    record: " + "[" * 1000 + "]" * 1000 + "\n", "RecursionError"),
    # AR-rag4ds-057 r5 F8: a base-60 float of 175 parts, tagged and untagged
    (RECORD, "    record: !!float " + ":".join(["1"] * 175) + "\n", "OverflowError"),
    (BOUND_ON, "    bound_on: 1" + ":00" * 199 + ".5\n", "OverflowError"),
], ids=["impossible-date", "no-feb-29", "int-tag", "5000-digits", "bool-tag", "timestamp-tag",
        "empty-int", "map-tag-on-scalar", "set-tag-on-sequence", "nesting", "float-overflow",
        "base60-overflow"])
def test_a_flows_file_yaml_cannot_build_is_refused_never_a_traceback(tmp_path, old, new, words):
    """AR-rag4ds-057 r3 F6: every class of text PyYAML parses but cannot turn
    into data - an impossible date, a tagged number, boolean or timestamp it
    cannot read, a mapping tag on a node that is not a mapping, nesting past
    the recursion limit - is refused as a DemoFlowError, never a Python error."""
    pack_dir = tmp_path / "packs" / "demo-pack"
    (pack_dir / "demo").mkdir(parents=True)
    text = _flows_text()
    assert text.count(old) == 1
    (pack_dir / "demo" / "flows.yaml").write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(DemoFlowError, match="not readable YAML") as exc:
        load_flows(pack_dir)
    assert words in str(exc.value), str(exc.value)[:200]


HUGE ="0x" + "f" * 3572          # the smallest hex literal Python will not print (report r4)


@pytest.mark.parametrize("old, new, words", [
    ("pack: demo-pack\n", f"pack: {HUGE}\n", "pack <an integer of 14288 bits> must be the pack"),
    ("pack: demo-pack\n", "pack: 0" + "7" * 6000 + "\n", "pack <an integer of"),
    ("pack: demo-pack\n", "pack: 0b" + "1" * 20000 + "\n", "pack <an integer of 20000 bits>"),
    ("pack: demo-pack\n", "pack: " + "1:" * 3000 + "1\n", "pack <an integer of"),
    ("- id: f\n", f"- id: {HUGE}\n", "id <an integer of 14288 bits> malformed or repeated"),
    ("  request: {}\n", f"  request: {{top_k: {HUGE}}}\n",
     "top_k must be an integer in 1..1000, got <an"),
    ("  request: {}\n", f"  request: {{domain: {HUGE}}}\n", "domain <an integer of"),
    ("  request: {}\n", f"  request: {{layer: {HUGE}}}\n", "layer <an integer of"),
    ("  request: {}\n", f"  request: {{phase: {HUGE}}}\n", "phase <an integer of"),
    ("  request: {}\n", f"  request: {{layer: [{HUGE}]}}\n",
     "layer <a list that cannot be printed>"),
], ids=["pack-hex", "pack-octal", "pack-binary", "pack-base60", "id-hex", "top_k", "domain",
        "layer", "phase", "layer-list"])
def test_a_value_too_large_to_print_is_refused_by_name(tmp_path, old, new, words):
    """AR-rag4ds-057 r4 F7: Python will not print an integer of more than 4300
    decimal digits, and YAML builds one from a hex, octal, binary or base-60
    literal; where such a value stands in for the pack code, a flow id or a
    request field, the refusal names it by its size - the refusal the value
    earns, never a Python error from printing it."""
    _check_refused(tmp_path, old, new, words)


def _check_refused(tmp_path, old, new, words):
    pack_dir = tmp_path / "packs" / "demo-pack"
    (pack_dir / "demo").mkdir(parents=True)
    text = _flows_text()
    assert text.count(old) == 1
    (pack_dir / "demo" / "flows.yaml").write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(DemoFlowError) as exc:
        load_flows(pack_dir)
    assert words in str(exc.value), str(exc.value)[:300]


def test_the_contract_prints_any_value_it_refuses():
    """AR-rag4ds-057 r4 F7, at the contract: shown() is repr wherever repr
    works, so no message changes for an ordinary value, and names a value
    repr cannot print by its type and size."""
    big = int("f" * 3572, 16)
    assert shown("robotics") == "'robotics'" and shown(5) == "5" and shown(None) == "None"
    assert shown(big) == "<an integer of 14288 bits>"
    assert shown([big]) == "<a list that cannot be printed>"
    deep = []
    for _ in range(100000):   # nested past repr's recursion limit
        deep = [deep]
    assert shown(deep) == "<a list that cannot be printed>"
    with pytest.raises(GroundingRequestError, match="got <an integer of 14288 bits>"):
        GroundingRequest("q", top_k=big)
    # AR-rag4ds-057 r5 O16: the pin message, which a flows file cannot reach
    with pytest.raises(GroundingRequestError, match="pin 'robotics': <an integer of 14288 bits>"):
        GroundingRequest("q", pins={"robotics": big})
    with pytest.raises(GroundingRequestError, match="pin <an integer of 14288 bits>: '1.0.0'"):
        GroundingRequest("q", pins={big: "1.0.0"})


def test_a_binding_is_dated_and_a_flow_asks_its_own_pack(tmp_path):
    """AR-rag4ds-057 r1 O3: bound_on must be a calendar date, not only its
    shape (2026-99-99 loaded before); and a flow's request may name no pack
    but the file's own."""
    pack_dir = tmp_path / "packs" / "demo-pack"
    for day in ("2026-99-99", "2026-02-30", "2026-13-01"):
        _write_flows(pack_dir, [{"id": "f", "query": "q", "request": {},
                                 "expected": {**BOUND, "bound_on": day}}])
        with pytest.raises(DemoFlowError, match=f"bound_on {day} is not a calendar date"):
            load_flows(pack_dir)
    _write_flows(pack_dir, [{"id": "f", "query": "q", "request": {},
                             "expected": {**BOUND, "bound_on": "2024-02-29"}}])
    assert load_flows(pack_dir)[1][0].expected.bound_on == "2024-02-29"
    _write_flows(pack_dir, [{"id": "f", "query": "q", "request": {"domain": "robotics"},
                             "expected": None}])
    with pytest.raises(DemoFlowError, match="request.domain 'robotics' must be the file's own "
                                            "pack, demo-pack"):
        load_flows(pack_dir)
    _write_flows(pack_dir, [{"id": "f", "query": "q", "request": {"domain": "demo-pack"},
                             "expected": None}])
    assert load_flows(pack_dir)[1][0].request == {"domain": "demo-pack"}


def test_the_services_error_is_carried_printed_and_recorded(tmp_path, capsys, monkeypatch):
    """AR-rag4ds-057 r1 O2: when the service answers ERROR its reason travels
    with the verdict, and the tool prints it and records it - a failed run
    stays diagnosable."""
    r = check(_flow(Expected("OK", ("a",), (), "K. Naruse", "2026-09-28", "x")),
              {"status": "ERROR", "entries": [], "meta": {"error": "RuntimeError: down"}})
    assert (r.verdict, r.error) == (FAIL, "RuntimeError: down")
    assert check(_flow(None), {"status": "OK", "entries": []}).error == ""
    tool = _tool()
    idx = _live_index(tmp_path)

    class Broken(FAKE_BagOfWordsEmbedder):
        def embed(self, texts):
            raise RuntimeError("the embedder is down")

    served = load_index(idx).rows[0]["id"]
    flow = {"id": "probe", "query": "harmonic drive torque", "request": {"domain": "robotics"},
            "expected": {**BOUND, "present": [served]}}
    root = _bound_module(tmp_path / "err", [flow])
    (root / "runs").mkdir()
    monkeypatch.setattr(tool, "head_commit", lambda root_: "abcdef0")
    assert tool.execute(root, "robotics", idx, embedder=Broken(), record=True,
                        clock=lambda: TS) == 1
    out = capsys.readouterr().out
    assert "FAIL     probe: ERROR; returned (none) - status ERROR, expected OK" in out
    assert "the service's error: RuntimeError: the embedder is down" in out
    rec = Journal(root / "runs" / "demo-flows.jsonl").read_all()[0]
    assert rec["results"][0]["error"] == "RuntimeError: the embedder is down"


def test_a_bound_flow_that_expects_no_entries_loads_and_checks(tmp_path):
    """AR-rag4ds-057 r5 O15: Part B will bind question 3 as NO_GROUNDING. A
    bound flow whose expected status is not OK - NO_GROUNDING with nothing
    present and an id absent, ERROR with both lists empty, NO_GROUNDING with
    both lists null - loads, and passes only on the response it expects."""
    for i, (status, present, absent) in enumerate((("NO_GROUNDING", "[]", "[b]"),
                                                    ("ERROR", "[]", "[]"),
                                                    ("NO_GROUNDING", "null", "null"))):
        text = (_flows_text().replace("    status: OK\n", f"    status: {status}\n")
                .replace("    present: [a]\n", f"    present: {present}\n")
                .replace("    absent: [b]\n", f"    absent: {absent}\n"))
        pack_dir = tmp_path / str(i) / "packs" / "demo-pack"
        (pack_dir / "demo").mkdir(parents=True)
        (pack_dir / "demo" / "flows.yaml").write_text(text, encoding="utf-8")
        flow = load_flows(pack_dir)[1][0]
        assert flow.expected.status == status and flow.expected.present == ()
        assert check(flow, _resp(status)).verdict == PASS, status
        assert check(flow, _resp("OK", "a")).verdict == FAIL, status
    ng = flow   # the last: NO_GROUNDING, both null
    assert check(ng, _resp("NO_GROUNDING", "b")).verdict == PASS  # nothing is listed absent


def test_a_failure_nobody_foresaw_is_still_a_refusal(tmp_path, monkeypatch):
    """AR-rag4ds-057 r4 F7, the last line of load_flows: any other error while
    the file is read or checked is a DemoFlowError naming it - the tool exits 2,
    never in a traceback - driven here by errors no input is known to cause."""
    import rag4ds.evalhook.demo as demo_mod
    pack_dir = tmp_path / "packs" / "demo-pack"
    (pack_dir / "demo").mkdir(parents=True)
    (pack_dir / "demo" / "flows.yaml").write_text(_flows_text(), encoding="utf-8")
    assert load_flows(pack_dir)[0] == "demo-pack"
    # a named refusal passes through the last line as it is, not re-wrapped
    (pack_dir / "demo" / "flows.yaml").write_text(_flows_text() + "notes: x\n", encoding="utf-8")
    with pytest.raises(DemoFlowError) as named:
        load_flows(pack_dir)
    assert str(named.value) == f"{pack_dir / 'demo' / 'flows.yaml'}: must carry exactly schema, " \
                               f"pack and flows"
    (pack_dir / "demo" / "flows.yaml").write_text(_flows_text(), encoding="utf-8")

    def boom(*a, **k):
        raise RuntimeError("an error nobody foresaw")

    class Boom:
        fullmatch = staticmethod(boom)

    with monkeypatch.context() as m:
        m.setattr(demo_mod, "strict_yaml_load", boom)
        with pytest.raises(DemoFlowError, match="not readable - RuntimeError: an error nobody"):
            load_flows(pack_dir)
    with monkeypatch.context() as m:
        m.setattr(demo_mod, "ENTRY_ID_RE", Boom())
        with pytest.raises(DemoFlowError, match="not a valid flows file - RuntimeError while "
                                                "checking it: an error nobody foresaw"):
            load_flows(pack_dir)


# ----- the tool, end to end over the live tree ---------------------------------------------------

def _bound_module(tmp_path, flows):
    root = tmp_path / "mod"
    _write_flows(root / "packs" / "robotics", flows)
    return root


def test_run_demo_passes_fails_and_refuses_by_exit_code(tmp_path, capsys, monkeypatch):
    tool = _tool()
    idx = _live_index(tmp_path)
    r = Retriever(load_index(idx), FAKE_BagOfWordsEmbedder())
    q = "harmonic drive torque"
    got = [e["id"] for e in r.ground(GroundingRequest(q, domain="robotics")).to_dict()["entries"]]
    assert got, "the fake index answers the probe"
    flow = {"id": "probe", "query": q, "request": {"domain": "robotics"},
            "expected": {**BOUND, "present": [got[0]]}}
    root = _bound_module(tmp_path / "pass", [flow])
    assert tool.execute(root, "robotics", idx, embedder=FAKE_BagOfWordsEmbedder()) == 0
    assert "PASS     probe" in capsys.readouterr().out
    # a FAIL: an entry the index holds but the question does not bring back
    elsewhere = next(row["id"] for row in load_index(idx).rows if row["id"] not in got)
    wrong = {**flow, "expected": {**BOUND, "present": [elsewhere]}}
    root = _bound_module(tmp_path / "fail", [wrong])
    assert tool.execute(root, "robotics", idx, embedder=FAKE_BagOfWordsEmbedder()) == 1
    assert f"expected entry {elsewhere} not returned" in capsys.readouterr().out
    # AR-rag4ds-057 r1 O3: an id the index does not hold - present, or absent - is
    # a constraint that could never be met or never fire: refused, exit 2, named
    for name, expected in (("typo-present", {**BOUND, "present": ["no-such-entry"]}),
                           ("typo-absent", {**BOUND, "present": [got[0]],
                                            "absent": ["no-such-entry"]})):
        root = _bound_module(tmp_path / name, [{**flow, "expected": expected}])
        assert tool.execute(root, "robotics", idx, embedder=FAKE_BagOfWordsEmbedder()) == 2
        assert "does not hold - probe: no-such-entry" in capsys.readouterr().err, name
    # the live robotics flows, bound in part B: every flow gets a verdict, none
    # UNBOUND (the fake embedder's verdicts are this test's own, not the served
    # index's), and the exit code follows them
    rc = tool.execute(MODULE_ROOT, "robotics", idx, embedder=FAKE_BagOfWordsEmbedder())
    lines = [x for x in capsys.readouterr().out.splitlines()
             if x.split(" ")[0] in (PASS, FAIL, UNBOUND)]
    assert len(lines) == 3 and not any(x.startswith(UNBOUND) for x in lines)
    assert rc == (0 if all(x.startswith(PASS) for x in lines) else 1)
    # a pack without a flows file, an index that does not serve the pack (with a
    # valid flows file, so the pack check itself is what refuses), and a wrong
    # embedder: refused, exit 2
    assert tool.execute(MODULE_ROOT, "grocery", idx, embedder=FAKE_BagOfWordsEmbedder()) == 2
    other = tmp_path / "other" / "mod"
    _write_flows(other / "packs" / "grocery", [{**flow, "request": {"domain": "grocery"},
                                                "expected": None}])
    assert tool.execute(other, "grocery", idx, embedder=FAKE_BagOfWordsEmbedder()) == 2
    assert "does not serve pack 'grocery'" in capsys.readouterr().err

    class Other(FAKE_BagOfWordsEmbedder):
        revision = "another"
    assert tool.execute(MODULE_ROOT, "robotics", idx, embedder=Other()) == 2
    assert "REFUSED" in capsys.readouterr().err
    # AR-rag4ds-057 F2: an index that does not load is refused with 2 - not a
    # traceback's exit 1, which the release checklist would read as a failed flow
    assert tool.execute(MODULE_ROOT, "robotics", tmp_path / "no-index",
                        embedder=FAKE_BagOfWordsEmbedder()) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_run_demo_writes_its_record_only_when_asked(tmp_path, monkeypatch):
    tool = _tool()
    idx = _live_index(tmp_path)
    flow = {"id": "probe", "query": "harmonic drive torque", "request": {"domain": "robotics"},
            "expected": None}
    root = _bound_module(tmp_path / "rec", [flow])
    assert tool.execute(root, "robotics", idx, embedder=FAKE_BagOfWordsEmbedder()) == 1
    assert not (root / "runs").exists()
    monkeypatch.setattr(tool, "head_commit", lambda root_: "abcdef0")
    # a module without runs/ is the wrong root: the record is refused, loudly, not
    # written somewhere new
    assert tool.execute(root, "robotics", idx, embedder=FAKE_BagOfWordsEmbedder(), record=True,
                        clock=lambda: TS) == 2
    assert not (root / "runs").exists()
    (root / "runs").mkdir()
    assert tool.execute(root, "robotics", idx, embedder=FAKE_BagOfWordsEmbedder(), record=True,
                        clock=lambda: TS) == 1
    recs = Journal(root / "runs" / "demo-flows.jsonl").read_all()
    assert len(recs) == 1
    rec = recs[0]
    flows_bytes = (root / "packs" / "robotics" / "demo" / "flows.yaml").read_bytes()
    assert rec["flows_sha256"] == hashlib.sha256(flows_bytes).hexdigest()
    assert (rec["pack"], rec["code_commit"], rec["timestamp"]) == ("robotics", "abcdef0", TS)
    assert rec["pack_versions"] == {"robotics": "0.0.0"} and rec["passed"] is False
    assert [x["verdict"] for x in rec["results"]] == [UNBOUND]
    # schema 2 (AR-rag4ds-057 r1 O2): every result carries the service's error, empty
    # when it answered
    assert rec["schema"] == 2 and [x["error"] for x in rec["results"]] == [""]
    assert rec["index_version"] == json.loads((idx / "manifest.json").read_text())["index_version"]

    # AR-rag4ds-057 F2: a git call that fails while recording refuses the record
    # with 2 and writes nothing
    def no_git(root_):
        raise subprocess.CalledProcessError(128, ["git", "rev-parse", "--short", "HEAD"])
    monkeypatch.setattr(tool, "head_commit", no_git)
    assert tool.execute(root, "robotics", idx, embedder=FAKE_BagOfWordsEmbedder(), record=True,
                        clock=lambda: TS) == 2
    assert len(Journal(root / "runs" / "demo-flows.jsonl").read_all()) == 1


# ----- the provenance chain ----------------------------------------------------------------------

def test_the_chain_of_a_corrected_entry_resolves_to_its_document_and_validator():
    registry = IdentityRegistry.load(MODULE_ROOT / IDENTITIES_FILE)
    for eid in CORRECTED:
        v2 = resolve(MODULE_ROOT, eid, 2)
        assert (v2.approver, v2.candidate_state, v2.supersedes_version) == ("K. Naruse",
                                                                             "APPROVED", 1)
        assert registry.resolve(v2.approver).validates("robotics")
        assert v2.document["hash"] == v2.source_hash and v2.superseded_by_line is None


def _scratch_module(tmp_path) -> Path:
    root = tmp_path / "mod"
    for rel in ("journal/decisions.jsonl", "stage/candidates.jsonl", IDENTITIES_FILE,
                "packs/robotics/corpus/catalog.yaml", "shared/corpus/catalog.yaml"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(MODULE_ROOT / rel, root / rel)
    return root


def _rewrite_jsonl(path: Path, change):
    rows = [json.loads(x) for x in path.read_text(encoding="ascii").split("\n") if x]
    path.unlink()
    j = Journal(path)
    for row in rows:
        j.append(change(row))


def test_every_approved_entry_version_on_the_journal_resolves():
    """AR-rag4ds-057 F3, the other direction: the chain's checks refuse nothing
    that is right - every entry version the live journal approves resolves,
    its candidate approved by that decision and carrying the text it records
    (the mapping draft's approval carries no entry and is not a chain)."""
    records = _journal()
    keys = sorted(approvals(records))
    assert len(keys) >= 43, len(keys)
    for eid, v in keys:
        chain = resolve(MODULE_ROOT, eid, v, records=records)
        assert chain.candidate_state == "APPROVED" and chain.decision_line >= 1, (eid, v)


def _is_use_v2(r) -> bool:
    return r.get("entry", {}).get("id") == "harmonic-drive-use" and r["entry"]["version"] == 2


def test_every_link_of_the_chain_is_looked_up_and_a_broken_one_raises(tmp_path):
    base = _scratch_module(tmp_path / "ok")
    assert resolve(base, "harmonic-drive-use", 2).decision_line == 374
    with pytest.raises(ProvenanceError, match="0 approving decisions"):
        resolve(base, "harmonic-drive-use", 3)
    # two approving decisions for one version: ambiguous, refused - and the old
    # version they both supersede has two successors (AR-rag4ds-057 F2)
    root = _scratch_module(tmp_path / "twice")
    rows = [json.loads(x) for x in (root / "journal/decisions.jsonl").read_text(
        encoding="ascii").split("\n") if x]
    Journal(root / "journal/decisions.jsonl").append(rows[373])
    with pytest.raises(ProvenanceError, match="2 approving decisions"):
        resolve(root, "harmonic-drive-use", 2)
    with pytest.raises(ProvenanceError, match="superseded by 2 decisions"):
        resolve(root, "harmonic-drive-use", 1)
    # AR-rag4ds-057 F3: a decision naming a candidate that ANOTHER decision
    # approved (line 375's), and a decision whose candidate_text is not its
    # candidate's text - each refused
    root = _scratch_module(tmp_path / "crossed")
    _rewrite_jsonl(root / "journal/decisions.jsonl",
                   lambda r: {**r, "candidate_id": "cand-eb6ec5f158ac2cc4"} if _is_use_v2(r) else r)
    with pytest.raises(ProvenanceError, match=r"approved by \['journal/decisions.jsonl:375'\], "
                                              r"not by this decision"):
        resolve(root, "harmonic-drive-use", 2)
    root = _scratch_module(tmp_path / "text")
    _rewrite_jsonl(root / "journal/decisions.jsonl",
                   lambda r: {**r, "candidate_text": r["candidate_text"] + " (altered)"}
                   if _is_use_v2(r) else r)
    with pytest.raises(ProvenanceError, match="candidate_text is not"):
        resolve(root, "harmonic-drive-use", 2)
    # AR-rag4ds-057 F2: the decision's entry without its layer; version 2 with
    # no superseded version; a candidate of another layer - each refused
    root = _scratch_module(tmp_path / "nolayer")
    _rewrite_jsonl(root / "journal/decisions.jsonl",
                   lambda r: {**r, "entry": {k: v for k, v in r["entry"].items() if k != "layer"}}
                   if _is_use_v2(r) else r)
    with pytest.raises(ProvenanceError, match="names no pack or layer"):
        resolve(root, "harmonic-drive-use", 2)
    root = _scratch_module(tmp_path / "nosup")
    _rewrite_jsonl(root / "journal/decisions.jsonl",
                   lambda r: {k: v for k, v in r.items() if k != "supersedes"}
                   if _is_use_v2(r) else r)
    with pytest.raises(ProvenanceError, match="names no superseded version"):
        resolve(root, "harmonic-drive-use", 2)
    # a candidate of another layer, kind or pack (AR-rag4ds-057 r2 F2: each of
    # the arm's three conditions witnessed on its own)
    for field, value in (("layer", "task"), ("kind", "mapping-draft"), ("pack", "grocery")):
        root = _scratch_module(tmp_path / f"cand-{field}")
        _rewrite_jsonl(root / "stage/candidates.jsonl",
                       lambda r, f=field, v=value: {**r, "candidate": {**r["candidate"], f: v}}
                       if r.get("record") == "candidate" and r.get("id") == "cand-34dd352217968739"
                       else r)
        with pytest.raises(ProvenanceError, match="is not an entry candidate of pack"):
            resolve(root, "harmonic-drive-use", 2)
    # AR-rag4ds-057 r2 O9: a candidate grouped, detached and then approved -
    # the chain reads the APPROVED transition's authority alone, and resolves
    root = _scratch_module(tmp_path / "detached")
    path = root / "stage/candidates.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="ascii").split("\n") if x]
    path.unlink()
    staging = Journal(path)
    head = "cand-eb6ec5f158ac2cc4"
    for row in rows:
        if (row.get("record") == "transition" and row.get("id") == "cand-34dd352217968739"
                and row.get("to") == "APPROVED"):
            step = {"schema": 1, "record": "transition", "id": row["id"],
                    "timestamp": row["timestamp"], "actor": "K. Naruse"}
            staging.append({**step, "from": "PENDING", "to": "GROUPED", "head": head,
                            "authority": f"gate:K. Naruse:grouped-under:{head}"})
            staging.append({**step, "from": "GROUPED", "to": "PENDING",
                            "authority": f"detached from {head}"})
        staging.append(row)
    assert StagingStore(path).state_of("cand-34dd352217968739") == "APPROVED"
    assert resolve(root, "harmonic-drive-use", 2).decision_line == 374
    # the approver not registered
    root = _scratch_module(tmp_path / "who")
    _rewrite_jsonl(root / "journal/decisions.jsonl",
                   lambda r: {**r, "approver": "Somebody Else"} if r.get("entry", {}).get(
                       "id") == "harmonic-drive-use" and r["entry"]["version"] == 2 else r)
    with pytest.raises(ProvenanceError, match="does not resolve"):
        resolve(root, "harmonic-drive-use", 2)
    # the candidate not in staging
    root = _scratch_module(tmp_path / "cand")
    _rewrite_jsonl(root / "stage/candidates.jsonl",
                   lambda r: {**r, "id": "cand-0000000000000000"}
                   if r.get("id") == "cand-34dd352217968739" else r)
    with pytest.raises(ProvenanceError, match="not in staging"):
        resolve(root, "harmonic-drive-use", 2)
    # the candidate not APPROVED
    root = _scratch_module(tmp_path / "state")
    _rewrite_jsonl(root / "stage/candidates.jsonl",
                   lambda r: {**r, "to": "REJECTED"} if r.get("record") == "transition"
                   and r.get("id") == "cand-34dd352217968739" else r)
    with pytest.raises(ProvenanceError, match="not APPROVED"):
        resolve(root, "harmonic-drive-use", 2)
    # the cited document not catalogued
    root = _scratch_module(tmp_path / "doc")
    cat = root / "packs/robotics/corpus/catalog.yaml"
    source = resolve(base, "harmonic-drive-use", 2).source_hash
    data = yaml.safe_load(cat.read_text(encoding="utf-8"))
    data["documents"] = [d for d in data["documents"] if d["hash"] != source]
    cat.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ProvenanceError, match="does not resolve"):
        resolve(root, "harmonic-drive-use", 2)
    # a supersession that names the wrong version
    root = _scratch_module(tmp_path / "sup")
    _rewrite_jsonl(root / "journal/decisions.jsonl",
                   lambda r: {**r, "supersedes": {"id": "harmonic-drive-use", "version": 7}}
                   if r.get("entry", {}).get("id") == "harmonic-drive-use"
                   and r["entry"]["version"] == 2 else r)
    with pytest.raises(ProvenanceError, match="not version 1 of the same id"):
        resolve(root, "harmonic-drive-use", 2)


# ----- acceptance case A4 ------------------------------------------------------------------------

def test_A4_a_correction_serves_only_the_new_version_and_keeps_the_old_on_the_record(tmp_path):
    """Plan section 2, A4: after a gate decision supersedes an approved
    robotics entry, retrieval returns only the new version; the old version
    remains stored in the audit log, marked superseded by the superseding
    decision that names it; the audit shows both decisions, and the
    provenance chain of the new version resolves fully. Witnessed on part
    8's real correction (journal lines 374-375)."""
    registry = IdentityRegistry.load(MODULE_ROOT / IDENTITIES_FILE)
    r = Retriever(load_index(_live_index(tmp_path)), FAKE_BagOfWordsEmbedder())
    for eid in CORRECTED:
        old, new = resolve(MODULE_ROOT, eid, 1), resolve(MODULE_ROOT, eid, 2)
        # the audit shows both decisions; the old one is marked by the one that names it
        assert old.decision_line < new.decision_line
        assert old.superseded_by_line == new.decision_line and new.supersedes_version == 1
        # the new version's chain resolves fully: the validator of the pack, an
        # APPROVED candidate, a catalogued document
        assert registry.resolve(new.approver).validates("robotics")
        assert new.candidate_state == "APPROVED" and new.document["hash"] == new.source_hash
        # the tree holds the new version only
        tree = json.loads((ROBOTICS / "entries" / f"{eid}.json").read_text(encoding="utf-8"))
        assert (tree["version"], tree["status"]) == (2, "current")
        # retrieval, asked with the old version's own words, returns only the new version
        got = r.ground(GroundingRequest(_v1_content(eid), domain="robotics", top_k=24)).to_dict()
        hits = [e for e in got["entries"] if e["id"] == eid]
        assert got["status"] == "OK" and [e["version"] for e in hits] == [2], got
