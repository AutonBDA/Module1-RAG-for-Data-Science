"""S9c part 3 - the re-file road (tools/refile_candidates.py): rejected
candidates filed again as fresh PENDING candidates under a re-file
manifest, for the validator's live session (PI decision (b),
AR-rag4ds-032 r1 F2). Every refusal witnessed; the real run on a scratch
module through the real staging store; the fresh candidates reach the
real gate's queue in filing order, members before heads."""
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
from rich.console import Console

from rag4ds.auditlog.journal import Journal
from rag4ds.author import drafts as au
from rag4ds.gate.cli import DECISION_JOURNAL
from rag4ds.gate.identities import IDENTITIES_FILE
from rag4ds.staging.store import StagingStore, candidate_id

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import gate_sheet  # noqa: E402
import refile_candidates as rf  # noqa: E402

TS = "2026-09-14 21:40:00 JST"
DECISION = "audit/rag4ds/responses/PI_decision_test.md"


def CLOCK():
    return TS


def _module(tmp_path, n=4):
    """A scratch repository shaped like the real one (module two levels
    below the repo root, so the decision record resolves), with n bio
    candidates: the first n-1 REJECTED by a sheet drive, the last PENDING."""
    repo = tmp_path / "repo"
    root = repo / "AutonBDA" / "mod"
    for d in ("journal", "stage", "runs", "shared/general", "shared/tasks", "packs/bio/entries",
              "data/transcripts", "audit/rag4ds/responses"):
        (repo / d if d.startswith("audit") else root / d).mkdir(parents=True)
    (repo / DECISION).write_text("# PI decision - test\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
                    "--allow-empty", "-m", "init"], cwd=repo, check=True)
    (root / ".gitignore").write_text("data/\n", encoding="utf-8")
    (root / IDENTITIES_FILE).write_text(
        "schema: 1\nidentities:\n  - id: v\n    name: V. Alidator\n    aliases: []\n"
        "    roles: [validator]\n    packs: [bio]\n  - id: sa\n    name: Draft author - DA\n"
        "    aliases: []\n    roles: [author]\n", encoding="utf-8")
    store = StagingStore(root / "stage" / "candidates.jsonl")
    ids = []
    for i in range(n):
        ids.append(store.file_candidate({
            "kind": "entry",
            "proposed": {"type": "concept", "content": f"Gene {i} - a unit of heredity number {i}.",
                         "source_reference": "corpus:bio/doc"},
            "layer": "domain", "pack": "bio", "source_hash": "c" * 64, "span_start": 0,
            "span_end": 10, "extractor": au.author_extractor("Draft author - DA"),
            "run_manifest": "authoring.jsonl#r"}, TS))
    sheet = tmp_path / "s.csv"
    sheet.write_text("n,id,key,reason\n" + "".join(f"{k + 1},{c},r,junk\n" for k, c in
                                                   enumerate(ids[:-1]))
                     + f"{n},{ids[-1]},s,\n", encoding="utf-8")
    assert gate_sheet.run(root, sheet, "V. Alidator", "abcdef0", None, root, "bio", None,
                          dry_run=False, clock=CLOCK) == 0
    assert [store.state_of(c) for c in ids] == ["REJECTED"] * (n - 1) + ["PENDING"]
    return root, ids


def _ids_file(tmp_path, ids, name="ids.txt"):
    p = tmp_path / name
    p.write_text("# comment\n" + "\n".join(f"{c}  # note" for c in ids) + "\n", encoding="utf-8")
    return p


def _run(root, ids_path, dry_run=False, reason="PI decision (b)", decision=DECISION, clock=CLOCK):
    return rf.run(root, "bio", ids_path, decision, reason, dry_run, clock=clock)


def test_every_refusal_is_loud_and_writes_nothing(tmp_path, capsys, monkeypatch):
    root, ids = _module(tmp_path)
    # another pack's rejected candidate, filed up front
    store = StagingStore(root / "stage" / "candidates.jsonl")
    other = store.file_candidate({
        "kind": "entry", "proposed": {"type": "concept", "content": "Torque - a twist.",
                                      "source_reference": "corpus:robo/doc"},
        "layer": "domain", "pack": "robo", "source_hash": "d" * 64, "span_start": 0,
        "span_end": 5, "extractor": au.author_extractor("Draft author - DA"),
        "run_manifest": "authoring.jsonl#r"}, TS)
    store.transition(other, "REJECTED", TS, "test", "journal:1")
    before = (root / "stage" / "candidates.jsonl").read_bytes()
    cases = [
        (_ids_file(tmp_path, ["cand-0000000000000000"], "a.txt"), {}, "is not a filed candidate"),
        (_ids_file(tmp_path, [ids[-1]], "b.txt"), {}, "is PENDING, not REJECTED"),
        (_ids_file(tmp_path, [ids[0], ids[0]], "c.txt"), {}, "is listed twice"),
        (_ids_file(tmp_path, [ids[0]], "d.txt"), {"decision": "audit/nope.md"}, "does not exist"),
        (_ids_file(tmp_path, [ids[0]], "e.txt"), {"reason": "  "}, "a reason is required"),
        (_ids_file(tmp_path, [], "f.txt"), {}, "no candidate ids"),
    ]
    p = tmp_path / "bad.txt"
    p.write_text("not-an-id\n", encoding="utf-8")
    cases.append((p, {}, "is not a candidate id"))
    for path, kw, needle in cases:
        assert _run(root, path, **kw) == 1
        assert needle in capsys.readouterr().err, needle
    assert _run(root, _ids_file(tmp_path, [other])) == 1
    assert "belongs to pack 'robo', not 'bio'" in capsys.readouterr().err
    # a non-JST machine clock without an injected clock
    monkeypatch.setattr(rf.time, "localtime", lambda: types.SimpleNamespace(tm_gmtoff=0))
    assert _run(root, _ids_file(tmp_path, [ids[0]]), clock=None) == 1
    assert "local clock is not JST" in capsys.readouterr().err
    assert not (root / rf.REFILE_JOURNAL).exists()
    assert (root / "stage" / "candidates.jsonl").read_bytes() == before


def test_a_dry_run_files_nothing_and_a_real_run_files_candidates_before_the_manifest(
        tmp_path, capsys, monkeypatch):
    root, ids = _module(tmp_path)
    p = _ids_file(tmp_path, ids[:3])
    before = (root / "stage" / "candidates.jsonl").read_bytes()
    assert _run(root, p, dry_run=True) == 0
    out = capsys.readouterr().out
    assert "would file 3" in out and "DRY RUN" in out
    assert (root / "stage" / "candidates.jsonl").read_bytes() == before
    assert not (root / rf.REFILE_JOURNAL).exists()
    # the real run: a store whose manifest append is made to fail after the candidates
    # would leave the candidates filed and no manifest (candidates before manifest)
    assert _run(root, p) == 0
    out = capsys.readouterr().out
    assert "filed 3; manifest runs/refile.jsonl#" in out
    store = StagingStore(root / "stage" / "candidates.jsonl")
    manifests = Journal(root / rf.REFILE_JOURNAL).read_all()
    assert len(manifests) == 1 and manifests[0]["count"] == 3
    pairs = manifests[0]["pairs"]
    assert [q["from"] for q in pairs] == ids[:3]
    fresh = [q["to"] for q in pairs]
    assert len(set(fresh)) == 3 and not set(fresh) & set(ids)
    cands = store.candidates()
    for old, new in zip(ids[:3], fresh, strict=True):
        o, n_ = cands[old]["candidate"], cands[new]["candidate"]
        assert n_["run_manifest"] == manifests[0]["run_manifest_ref"]
        assert {k: v for k, v in o.items() if k != "run_manifest"} == \
            {k: v for k, v in n_.items() if k != "run_manifest"}
        assert candidate_id(n_) == new
        assert store.state_of(old) == "REJECTED" and store.state_of(new) == "PENDING"
    # the filing order is the ids file's order, which is the gate's queue order
    filed_order = [json.loads(line)["id"] for line in
                   (root / "stage" / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
                   if json.loads(line).get("record") == "candidate"]
    assert filed_order[-3:] == fresh
    # the same run again is refused as already on the journal; a listed id already
    # re-filed is refused by name; nothing more written
    after = (root / "stage" / "candidates.jsonl").read_bytes()
    assert _run(root, p) == 1
    assert "is already on runs/refile.jsonl" in capsys.readouterr().err
    assert _run(root, _ids_file(tmp_path, [ids[0]], "one.txt"), reason="again") == 1
    assert f"{ids[0]} was already re-filed as {fresh[0]}" in capsys.readouterr().err
    assert (root / "stage" / "candidates.jsonl").read_bytes() == after
    assert len(Journal(root / rf.REFILE_JOURNAL).read_all()) == 1
    # the session-1 decisions are untouched
    assert len(Journal(root / DECISION_JOURNAL).read_all()) == 3


def test_fresh_candidates_reach_the_real_gate_in_filing_order_and_a_member_groups_under_its_head(
        tmp_path, capsys):
    """Members are filed before their heads, so at the live session 'g' finds
    the head PENDING; the head's decision then closes the member
    mechanically citing it."""
    from rag4ds.gate.cli import GateSession
    from rag4ds.gate.identities import IdentityRegistry
    root, ids = _module(tmp_path)
    member, head = ids[1], ids[0]
    assert _run(root, _ids_file(tmp_path, [member, head])) == 0
    pairs = Journal(root / rf.REFILE_JOURNAL).read_all()[0]["pairs"]
    new_member, new_head = pairs[0]["to"], pairs[1]["to"]
    console = Console(record=True, markup=False, highlight=False, width=100)
    # the queue in filing order: the original still-PENDING candidate first (skipped),
    # then the fresh member (grouped under the fresh head), then the fresh head
    # (rejected - its member closes mechanically); the queue is then empty
    answers = iter(["s", "g", new_head, "r", "not a concept"])
    ctx = gate_sheet.build_context(root, "V. Alidator", "abcdef0", None, root,
                                   IdentityRegistry.load(root / IDENTITIES_FILE), clock=CLOCK)
    session = GateSession(ctx, console, lambda prompt: next(answers), pack="bio")
    session.run()
    assert session.queue() == [ids[-1]]  # only the skipped original is still PENDING
    store = StagingStore(root / "stage" / "candidates.jsonl")
    assert store.state_of(new_member) == "REJECTED-DUPLICATE"
    assert store.state_of(new_head) == "REJECTED"
    assert store.state_of(ids[-1]) == "PENDING"  # the skipped original
    records = Journal(root / DECISION_JOURNAL).read_all()
    assert [r["type"] for r in records[3:]] == ["reject", "reject-duplicate"]
    assert records[3]["candidate_id"] == new_head and records[4]["candidate_id"] == new_member
    counts = session.summary.counts
    assert counts["grouped"] == 1 and counts["reject-duplicate"] == 1


def test_two_rejected_candidates_that_would_collapse_to_one_fresh_id_are_refused_whole(
        tmp_path, capsys):
    """AR-rag4ds-034 F2: the same content filed under two run manifests is two
    rejected candidates but ONE fresh candidate; refused before anything is
    written, naming both."""
    root, ids = _module(tmp_path)
    store = StagingStore(root / "stage" / "candidates.jsonl")
    twin_src = dict(store.candidates()[ids[0]]["candidate"], run_manifest="authoring.jsonl#other")
    twin = store.file_candidate(twin_src, TS)
    store.transition(twin, "REJECTED", TS, "test", "journal:9")
    before = (root / "stage" / "candidates.jsonl").read_bytes()
    assert _run(root, _ids_file(tmp_path, [ids[1], ids[0], twin])) == 1
    err = capsys.readouterr().err
    assert f"{twin} and {ids[0]} would re-file to the same fresh candidate" in err
    assert (root / "stage" / "candidates.jsonl").read_bytes() == before
    assert not (root / rf.REFILE_JOURNAL).exists()


def test_a_failure_between_the_candidates_and_the_manifest_is_completed_by_a_retry(
        tmp_path, capsys, monkeypatch):
    """AR-rag4ds-034 F6: candidates are filed before the manifest; if the manifest
    append fails, the fresh candidates exist and no manifest names them - the
    same command again counts them as filed by this run and appends the
    manifest naming every pair."""
    root, ids = _module(tmp_path)
    p = _ids_file(tmp_path, ids[:3])
    real_append = rf.Journal.append

    def failing_append(self, record):
        if record.get("kind") == "refile":
            raise OSError("disk full (witness)")
        return real_append(self, record)

    monkeypatch.setattr(rf.Journal, "append", failing_append)
    assert _run(root, p) == 1
    assert "disk full" in capsys.readouterr().err
    store = StagingStore(root / "stage" / "candidates.jsonl")
    fresh = [c for c, r in store.candidates().items()
             if r["candidate"]["run_manifest"].startswith("runs/refile.jsonl#")]
    assert len(fresh) == 3 and all(store.state_of(c) == "PENDING" for c in fresh)
    assert not (root / rf.REFILE_JOURNAL).exists()
    monkeypatch.setattr(rf.Journal, "append", real_append)
    assert _run(root, p) == 0
    out = capsys.readouterr().out
    assert "filed 0 (3 already filed by an earlier attempt of this run); manifest" in out
    manifests = Journal(root / rf.REFILE_JOURNAL).read_all()
    assert len(manifests) == 1 and manifests[0]["count"] == 3
    assert sorted(q["to"] for q in manifests[0]["pairs"]) == sorted(fresh)
    assert len([c for c in store.candidates()]) == 4 + 3  # no duplicate filings


def _fail_manifest_once(monkeypatch):
    real_append = rf.Journal.append

    def failing_append(self, record):
        if record.get("kind") == "refile":
            monkeypatch.setattr(rf.Journal, "append", real_append)
            raise OSError("disk full (witness)")
        return real_append(self, record)

    monkeypatch.setattr(rf.Journal, "append", failing_append)


def test_a_retry_after_a_commit_completes_the_same_run(tmp_path, capsys, monkeypatch):
    """AR-rag4ds-035 F1: the commit is not part of the run's identity - a
    failed re-file retried after a commit completes, filing nothing twice."""
    root, ids = _module(tmp_path)
    p = _ids_file(tmp_path, ids[:3])
    _fail_manifest_once(monkeypatch)
    assert _run(root, p) == 1
    capsys.readouterr()
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
                    "--allow-empty", "-m", "between"], cwd=root, check=True)
    assert _run(root, p) == 0
    assert "filed 0 (3 already filed by an earlier attempt of this run); manifest" in \
        capsys.readouterr().out
    store = StagingStore(root / "stage" / "candidates.jsonl")
    fresh = [c for c, r in store.candidates().items()
             if r["candidate"]["run_manifest"].startswith("runs/refile.jsonl#")]
    assert len(fresh) == 3
    manifests = Journal(root / rf.REFILE_JOURNAL).read_all()
    assert len(manifests) == 1 and sorted(q["to"] for q in manifests[0]["pairs"]) == sorted(fresh)
    assert not rf.stranded(store, Journal(root / rf.REFILE_JOURNAL))


def test_a_different_re_file_is_refused_while_a_failed_one_is_stranded(tmp_path, capsys,
                                                                       monkeypatch):
    """AR-rag4ds-035 F1: after a failure, a run with a changed list is refused
    before anything is written, naming the stranded candidates; the original
    run then completes them."""
    root, ids = _module(tmp_path)
    p3 = _ids_file(tmp_path, ids[:3], "three.txt")
    _fail_manifest_once(monkeypatch)
    assert _run(root, p3) == 1
    capsys.readouterr()
    store = StagingStore(root / "stage" / "candidates.jsonl")
    before = (root / "stage" / "candidates.jsonl").read_bytes()
    strand = rf.stranded(store, Journal(root / rf.REFILE_JOURNAL))
    assert len(strand) == 1 and len(next(iter(strand.values()))) == 3
    p2 = _ids_file(tmp_path, ids[:2], "two.txt")
    assert _run(root, p2) == 1
    err = capsys.readouterr().err
    assert "3 fresh candidate(s) whose run_manifest runs/refile.jsonl#" in err
    assert "re-run THAT re-file with its original ids file and decision" in err
    for cid in next(iter(strand.values())):
        assert cid in err
    assert (root / "stage" / "candidates.jsonl").read_bytes() == before
    assert not (root / rf.REFILE_JOURNAL).exists()
    assert _run(root, p3) == 0  # the original run completes
    assert not rf.stranded(store, Journal(root / rf.REFILE_JOURNAL))


def test_the_live_ids_file_names_73_rejected_robotics_candidates_members_first():
    """The committed list for the validator's session: 73 ids, every one a
    REJECTED robotics candidate of session 1, members before heads."""
    root = Path(__file__).resolve().parents[1]
    ids = rf.load_ids(root / "authoring" / "robotics_s9_refile_ids.txt")
    assert len(ids) == 73 and len(set(ids)) == 73
    store = StagingStore(root / "stage" / "candidates.jsonl")
    cands = store.candidates()
    for c in ids:
        assert cands[c]["candidate"]["pack"] == "robotics"
        assert store.state_of(c) == "REJECTED"
    text = (root / "authoring" / "robotics_s9_refile_ids.txt").read_text(encoding="utf-8")
    heads = [line.split()[0] for line in text.splitlines() if "head of a cluster" in line]
    members = [line for line in text.splitlines() if ": g under" in line]
    assert len(heads) == 10 and len(members) == 25
    first_head = min(ids.index(h) for h in heads)
    assert all(ids.index(line.split()[0]) < first_head for line in members)


def test_the_tool_refuses_a_non_jst_clock_by_default(tmp_path, monkeypatch, capsys):
    root, ids = _module(tmp_path)
    monkeypatch.setattr(rf.time, "localtime", lambda: types.SimpleNamespace(tm_gmtoff=0))
    assert _run(root, _ids_file(tmp_path, [ids[0]]), clock=None) == 1
    assert "local clock is not JST" in capsys.readouterr().err
    assert not (root / rf.REFILE_JOURNAL).exists()


@pytest.mark.parametrize("bad", ["", "   "])
def test_an_empty_reason_is_refused(tmp_path, capsys, bad):
    root, ids = _module(tmp_path)
    assert _run(root, _ids_file(tmp_path, [ids[0]]), reason=bad) == 1
    assert "a reason is required" in capsys.readouterr().err
