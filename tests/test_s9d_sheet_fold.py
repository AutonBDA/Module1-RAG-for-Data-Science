"""S9c part 2: the sheet-driven gate road (tools/gate_sheet.py - the
validator's written decision sheet drives the real GateSession; keys r and
s only; the sheet must equal the queue in order; any prompt beyond key and
reason aborts; a gate-sheets manifest per run) and the committed fold tool
(tools/fold_tree.py - compile_tree + verify_reconstruction + a fold
manifest, AR-rag4ds-026 O5)."""

import json
import sys
from pathlib import Path

import pytest
from rich.console import Console

from rag4ds.auditlog.journal import Journal
from rag4ds.author import drafts as au
from rag4ds.gate.cli import DECISION_JOURNAL, GateError
from rag4ds.gate.identities import IDENTITIES_FILE
from rag4ds.staging.store import StagingStore

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))
sys.path.insert(0, str(MODULE_ROOT / "tests"))
import fold_tree  # noqa: E402
import gate_sheet  # noqa: E402
import test_s4_fold  # noqa: E402

TS = "2026-09-14 18:00:00 JST"


def CLOCK():
    return TS


def _module(tmp_path, n_candidates=3):
    """A scratch module: registry with a validator of pack 'bio', three
    PENDING bio candidates in staging, empty journal."""
    root = tmp_path / "mod"
    for d in ("journal", "stage", "runs", "shared/general", "shared/tasks", "packs/bio/entries",
              "packs/bio/mappings", "data/transcripts"):
        (root / d).mkdir(parents=True)
    (root / ".gitignore").write_text("data/\n", encoding="utf-8")
    (root / IDENTITIES_FILE).write_text(
        "schema: 1\nidentities:\n  - id: v\n    name: V. Alidator\n    aliases: []\n"
        "    roles: [validator]\n    packs: [bio]\n  - id: sa\n    name: Draft author - DA\n"
        "    aliases: []\n    roles: [author]\n", encoding="utf-8")
    store = StagingStore(root / "stage" / "candidates.jsonl")
    ids = []
    for i in range(n_candidates):
        ids.append(store.file_candidate({
            "kind": "entry",
            "proposed": {"type": "concept", "content": f"Gene {i} - a unit of heredity number {i}.",
                         "source_reference": "corpus:bio/doc"},
            "layer": "domain", "pack": "bio", "source_hash": "c" * 64, "span_start": 0,
            "span_end": 10, "extractor": au.author_extractor("Draft author - DA"),
            "run_manifest": "authoring.jsonl#r"}, TS))
    return root, ids


def _sheet(path: Path, rows):
    lines = ["n,id,key,reason"] + [f"{n},{cid},{k},{r}" for n, cid, k, r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_sheet_validation_refuses_every_bad_shape(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("id,key\nx,r\n", encoding="utf-8")
    with pytest.raises(gate_sheet.SheetError, match="columns must be n,id,key,reason"):
        gate_sheet.load_sheet(p)
    for key, needle in (("a", "refused on a sheet"), ("e", "refused on a sheet"),
                        ("x", "key must be r, s or g"), ("", "key must be r, s or g")):
        _sheet(p, [(1, "cand-1", key, "why")])
        with pytest.raises(gate_sheet.SheetError, match=needle):
            gate_sheet.load_sheet(p)
    # S9c part 3: a grouping row carries its head id in the reason column
    for reason in ("", "why", "cand-1"):
        _sheet(p, [(1, "cand-1", "g", reason)])
        with pytest.raises(gate_sheet.SheetError, match="a grouping needs the head candidate id"):
            gate_sheet.load_sheet(p)
    assert gate_sheet.load_sheet(_sheet(p, [(1, "cand-1", "g", "cand-2")]))[0]["reason"] == "cand-2"
    _sheet(p, [(1, "cand-1", "r", "")])
    with pytest.raises(gate_sheet.SheetError, match="a rejection needs a reason"):
        gate_sheet.load_sheet(p)
    _sheet(p, [(2, "cand-1", "r", "why")])
    with pytest.raises(gate_sheet.SheetError, match="n must be 1"):
        gate_sheet.load_sheet(p)
    _sheet(p, [])
    with pytest.raises(gate_sheet.SheetError, match="no rows"):
        gate_sheet.load_sheet(p)
    rows = gate_sheet.load_sheet(_sheet(p, [(1, "cand-1", "r", "junk"), (2, "cand-2", "s", "")]))
    assert [(r["key"], r["reason"]) for r in rows] == [("r", "junk"), ("s", "")]
    # AR-rag4ds-032 F5: a bare comma in a reason is refused, never cut; so is a short row
    _sheet(p, [(1, "cand-1", "r", "fragment, cut mid-clause")])
    with pytest.raises(gate_sheet.SheetError, match="row 1: 5 fields, expected 4"):
        gate_sheet.load_sheet(p)
    p.write_text("n,id,key,reason\n1,cand-1,r\n", encoding="utf-8")
    with pytest.raises(gate_sheet.SheetError, match="row 1: 3 fields, expected 4"):
        gate_sheet.load_sheet(p)
    _sheet(p, [(1, "cand-1", "r", '"fragment, cut mid-clause"')])
    assert gate_sheet.load_sheet(p)[0]["reason"] == "fragment, cut mid-clause"


def test_sheet_must_equal_the_queue_in_order_or_nothing_is_decided(tmp_path, capsys):
    root, ids = _module(tmp_path)
    p = _sheet(tmp_path / "s.csv", [(1, ids[1], "r", "junk"), (2, ids[0], "r", "junk"),
                                    (3, ids[2], "r", "junk")])
    assert gate_sheet.run(root, p, "V. Alidator", "abcdef0", None, root, "bio", None,
                          dry_run=False, clock=CLOCK) == 1
    err = capsys.readouterr().err
    assert "does not match the gate's queue" in err and "first mismatch at row 1" in err
    assert not (root / DECISION_JOURNAL).exists()
    assert not (root / "runs" / "gate-sheets.jsonl").exists()
    p = _sheet(tmp_path / "s2.csv", [(1, ids[0], "r", "junk"), (2, ids[1], "r", "junk")])
    assert gate_sheet.run(root, p, "V. Alidator", "abcdef0", None, root, "bio", None,
                          dry_run=True, clock=CLOCK) == 1
    assert "sheet 2 rows, queue 3 candidates" in capsys.readouterr().err


def test_a_matching_sheet_drives_the_real_gate_and_writes_the_manifests(tmp_path, capsys,
                                                                       monkeypatch):
    root, ids = _module(tmp_path)
    p = _sheet(tmp_path / "s.csv", [(1, ids[0], "r", "junk"), (2, ids[1], "s", ""),
                                    (3, ids[2], "r", "dup")])
    # the dry run reports and writes nothing
    assert gate_sheet.run(root, p, "V. Alidator", "abcdef0", None, root, "bio", None,
                          dry_run=True, clock=CLOCK) == 0
    out = capsys.readouterr().out
    assert "3 rows match the queue: 2 reject, 0 group, 1 skip" in out and "DRY RUN" in out
    assert not (root / DECISION_JOURNAL).exists()
    # the drive: the machine clock must be JST for the gate (the gate reads it)
    transcript = root / "data" / "transcripts" / "t.txt"
    rc = gate_sheet.run(root, p, "V. Alidator", "abcdef0", None, root, "bio", transcript,
                        dry_run=False, clock=CLOCK)
    out = capsys.readouterr().out
    assert rc == 0, out
    store = StagingStore(root / "stage" / "candidates.jsonl")
    assert [store.state_of(i) for i in ids] == ["REJECTED", "PENDING", "REJECTED"]
    records = Journal(root / DECISION_JOURNAL).read_all()
    assert len(records) == 2 and all(r["approver"] == "V. Alidator" for r in records)
    assert [r["reason"] for r in records] == ["junk", "dup"]
    sessions = Journal(root / "runs" / "gate-sessions.jsonl").read_all()
    assert len(sessions) == 1 and sessions[0]["counts"]["reject"] == 2
    assert sessions[0]["counts"]["skipped"] == 1
    sheets = Journal(root / "runs" / "gate-sheets.jsonl").read_all()
    assert len(sheets) == 1 and sheets[0]["rows"] == 3 and sheets[0]["reject_rows"] == 2
    assert sheets[0]["rows_answered"] == 3 and sheets[0]["session_failed"] is False
    assert sheets[0]["sheet_sha256"] == __import__("hashlib").sha256(p.read_bytes()).hexdigest()
    text = transcript.read_text(encoding="utf-8")
    assert "SHEET-DRIVEN SESSION" in text and "sheet: row 1" in text
    # a second drive with the same sheet: the queue is now one candidate, so the
    # sheet no longer matches - refused, nothing more decided
    assert gate_sheet.run(root, p, "V. Alidator", "abcdef0", None, root, "bio", None,
                          dry_run=False, clock=CLOCK) == 1
    assert len(Journal(root / DECISION_JOURNAL).read_all()) == 2


class _Shown:
    """A stand-in for the session: only the candidate the gate is deciding."""

    def __init__(self, cid):
        self.current_cid = cid


def test_the_reader_answers_only_key_and_reason_prompts():
    console = Console(record=True, markup=False, highlight=False, width=100)
    rows = [{"n": 1, "id": "cand-1", "key": "r", "reason": "junk"},
            {"n": 2, "id": "cand-2", "key": "s", "reason": ""}]
    reader = gate_sheet.SheetReader(rows, console)
    reader.session = _Shown("cand-1")
    assert reader(gate_sheet.KEY_PROMPT) == "r"
    assert reader(gate_sheet.REASON_PROMPT) == "junk"
    reader.session = _Shown("cand-2")
    assert reader(gate_sheet.KEY_PROMPT) == "s"
    assert reader(gate_sheet.KEY_PROMPT) == "q"  # rows exhausted
    reader2 = gate_sheet.SheetReader(rows, console)
    reader2.session = _Shown("cand-1")
    reader2(gate_sheet.KEY_PROMPT)
    with pytest.raises(GateError, match="unexpected prompt 'entry id"):
        reader2("entry id [x]: ")
    # AR-rag4ds-032 F1: the reader answers only for the candidate the gate is deciding
    reader3 = gate_sheet.SheetReader(rows, console)
    reader3.session = _Shown("cand-2")
    with pytest.raises(GateError, match="deciding 'cand-2' but the sheet's next row 1 is 'cand-1'"):
        reader3(gate_sheet.KEY_PROMPT)
    reader3.session = None  # no session at all: nothing is shown, nothing is answered
    with pytest.raises(GateError, match="deciding None"):
        reader3(gate_sheet.KEY_PROMPT)


def _drive(root, sheet, transcript, clock=CLOCK):
    return gate_sheet.run(root, sheet, "V. Alidator", "abcdef0", None, root, "bio", transcript,
                          dry_run=False, clock=clock)


def _after_abort(root, transcript, rows, answered):
    """The F8 half of the manifest: session_failed True, rows_answered short of
    rows, the transcript written, the decision journal untouched."""
    sheets = Journal(root / "runs" / "gate-sheets.jsonl").read_all()
    assert len(sheets) == 1 and sheets[0]["session_failed"] is True
    assert sheets[0]["rows"] == rows and sheets[0]["rows_answered"] == answered
    assert transcript.exists() and "SHEET-DRIVEN SESSION" in transcript.read_text(encoding="utf-8")
    assert not (root / DECISION_JOURNAL).exists()
    sessions = Journal(root / "runs" / "gate-sessions.jsonl").read_all()
    assert len(sessions) == 1 and sessions[0]["aborted"].startswith("GateError: sheet road")
    return sessions[0]


def test_an_unrenderable_candidate_desyncs_the_sheet_and_the_drive_aborts(tmp_path, capsys,
                                                                          monkeypatch):
    """AR-rag4ds-032 F1, the UNRENDERABLE arm: the gate parks a candidate it
    cannot render and shows the next one WITHOUT consuming a key; the sheet's
    row for the parked candidate must not be spent on the one now shown."""
    from rag4ds.gate import cli as gate_cli
    root, ids = _module(tmp_path)
    p = _sheet(tmp_path / "s.csv", [(1, ids[0], "r", "junk"), (2, ids[1], "r", "junk"),
                                    (3, ids[2], "r", "junk")])
    real_show = gate_cli.GateSession.show

    def show(self, cid, rec):
        if cid == ids[0]:
            raise GateError("source text unavailable (witness)")
        return real_show(self, cid, rec)

    monkeypatch.setattr(gate_cli.GateSession, "show", show)
    transcript = root / "data" / "transcripts" / "t.txt"
    assert _drive(root, p, transcript) == 1
    err = capsys.readouterr().err
    assert f"the gate is deciding '{ids[1]}' but the sheet's next row 1 is '{ids[0]}'" in err
    session = _after_abort(root, transcript, rows=3, answered=0)
    assert session["counts"]["unrenderable"] == 1 and session["counts"]["reject"] == 0
    store = StagingStore(root / "stage" / "candidates.jsonl")
    assert [store.state_of(i) for i in ids] == ["PENDING", "PENDING", "PENDING"]


def test_a_refused_decision_desyncs_the_sheet_and_the_drive_aborts(tmp_path, capsys,
                                                                   monkeypatch):
    """AR-rag4ds-032 F1, the REFUSED arm: a refused decision re-shows the SAME
    candidate and asks for a key again; the sheet's next row belongs to the
    next candidate and must not be spent on the re-shown one."""
    from rag4ds.gate import cli as gate_cli
    root, ids = _module(tmp_path)
    p = _sheet(tmp_path / "s.csv", [(1, ids[0], "r", "junk"), (2, ids[1], "r", "junk"),
                                    (3, ids[2], "r", "junk")])

    def reject(self, cid, c, span_state, display, related):
        raise GateError("rejection refused (witness)")

    monkeypatch.setattr(gate_cli.GateSession, "_reject", reject)
    transcript = root / "data" / "transcripts" / "t.txt"
    assert _drive(root, p, transcript) == 1
    err = capsys.readouterr().err
    assert f"the gate is deciding '{ids[0]}' but the sheet's next row 2 is '{ids[1]}'" in err
    session = _after_abort(root, transcript, rows=3, answered=1)
    assert session["counts"]["refused"] == 1 and session["counts"]["reject"] == 0


def test_a_drive_that_ends_with_rows_unanswered_is_not_a_success(tmp_path, capsys,
                                                                 monkeypatch):
    """AR-rag4ds-032 F1/F8: a session that returns cleanly with sheet rows
    still unanswered exits 1 and its manifest says how many were answered."""
    from rag4ds.gate import cli as gate_cli
    root, ids = _module(tmp_path)
    p = _sheet(tmp_path / "s.csv", [(1, ids[0], "r", "junk"), (2, ids[1], "r", "junk"),
                                    (3, ids[2], "r", "junk")])

    def run_nothing(self):
        self.end()  # the loop never asked for a key
        return self.summary

    monkeypatch.setattr(gate_cli.GateSession, "run", run_nothing)
    transcript = root / "data" / "transcripts" / "t.txt"
    assert _drive(root, p, transcript) == 1
    assert "session ended with 0 of 3 rows answered - NOT complete" in capsys.readouterr().err
    sheets = Journal(root / "runs" / "gate-sheets.jsonl").read_all()
    assert sheets[0]["rows_answered"] == 0 and sheets[0]["session_failed"] is False
    assert not (root / DECISION_JOURNAL).exists()


def test_a_grouping_row_groups_under_its_head_and_closes_on_the_head_decision(tmp_path, capsys):
    """S9c part 3: 'g' on a sheet with the head id in the reason column; the
    member becomes GROUPED, the head's rejection closes it as
    REJECTED-DUPLICATE citing that decision; the manifest counts the group
    row. A head that is not PENDING is refused by the gate itself, which
    re-shows the member without a key - the reader then refuses to answer
    for it (the F1 guard) and nothing further is decided."""
    root, ids = _module(tmp_path)
    p = _sheet(tmp_path / "s.csv", [(1, ids[0], "g", ids[2]), (2, ids[1], "r", "junk"),
                                    (3, ids[2], "r", "dup head")])
    transcript = root / "data" / "transcripts" / "t.txt"
    assert _drive(root, p, transcript) == 0
    out = capsys.readouterr().out
    assert "1 group" in out
    store = StagingStore(root / "stage" / "candidates.jsonl")
    assert [store.state_of(i) for i in ids] == ["REJECTED-DUPLICATE", "REJECTED", "REJECTED"]
    records = Journal(root / DECISION_JOURNAL).read_all()
    assert [r["type"] for r in records] == ["reject", "reject", "reject-duplicate"]
    assert records[2]["candidate_id"] == ids[0] and records[2]["head_decision"].endswith(":2")
    sheets = Journal(root / "runs" / "gate-sheets.jsonl").read_all()
    assert sheets[0]["group_rows"] == 1 and sheets[0]["reject_rows"] == 2
    assert sheets[0]["rows_answered"] == 3 and sheets[0]["session_failed"] is False
    # a head that is already decided: the gate refuses the grouping, re-shows the
    # member, and the sheet road aborts rather than spend row 2 on it
    root2, ids2 = _module(tmp_path / "two")
    p2 = _sheet(tmp_path / "s2.csv", [(1, ids2[0], "r", "junk"), (2, ids2[1], "g", ids2[0]),
                                      (3, ids2[2], "r", "junk")])
    t2 = root2 / "data" / "transcripts" / "t.txt"
    assert _drive(root2, p2, t2) == 1
    err = capsys.readouterr().err
    assert f"the gate is deciding '{ids2[1]}' but the sheet's next row 3 is '{ids2[2]}'" in err
    session = _after_abort_partial(root2, t2, rows=3, answered=2)
    assert session["counts"]["refused"] == 1 and session["counts"]["reject"] == 1
    store2 = StagingStore(root2 / "stage" / "candidates.jsonl")
    assert [store2.state_of(i) for i in ids2] == ["REJECTED", "PENDING", "PENDING"]


def test_the_transcript_witnesses_the_reasons_and_head_ids_the_sheet_supplied(tmp_path):
    """AR-rag4ds-034 F3: not only the keys - every answer the sheet gives at a
    prompt is in the transcript."""
    root, ids = _module(tmp_path)
    p = _sheet(tmp_path / "s.csv", [(1, ids[0], "g", ids[2]), (2, ids[1], "r", "junk-words"),
                                    (3, ids[2], "r", "dup-head")])
    transcript = root / "data" / "transcripts" / "t.txt"
    assert _drive(root, p, transcript) == 0
    text = transcript.read_text(encoding="utf-8")
    assert f"sheet: {gate_sheet.HEAD_PROMPT}{ids[2]}" in text
    assert f"sheet: {gate_sheet.REASON_PROMPT}junk-words" in text
    assert f"sheet: {gate_sheet.REASON_PROMPT}dup-head" in text


def test_the_live_gate_reader_echoes_what_was_typed(capsys):
    """AR-rag4ds-034 F4 (AR-033 r1 O1): the live gate's read callback puts the
    prompt and the typed answer into the recorded console."""
    from rag4ds.gate.cli import make_echo_reader
    console = Console(record=True, markup=False, highlight=False, width=200)
    typed = iter(["r", "not a concept", "cand-9"])
    read = make_echo_reader(console, lambda prompt: next(typed))
    assert read(gate_sheet.KEY_PROMPT) == "r"
    assert read(gate_sheet.REASON_PROMPT) == "not a concept"
    assert read("head candidate id: ") == "cand-9"
    text = console.export_text()
    assert f"{gate_sheet.KEY_PROMPT}r" in text
    assert "reason: not a concept" in text and "head candidate id: cand-9" in text


def _after_abort_partial(root, transcript, rows, answered):
    sheets = Journal(root / "runs" / "gate-sheets.jsonl").read_all()
    assert len(sheets) == 1 and sheets[0]["session_failed"] is True
    assert sheets[0]["rows"] == rows and sheets[0]["rows_answered"] == answered
    assert transcript.exists()
    sessions = Journal(root / "runs" / "gate-sessions.jsonl").read_all()
    assert len(sessions) == 1 and sessions[0]["aborted"].startswith("GateError: sheet road")
    return sessions[0]


def test_an_interrupted_drive_records_session_failed_and_re_raises(tmp_path, capsys,
                                                                   monkeypatch):
    """AR-rag4ds-033 F1: Ctrl+C during a drive is not an Exception; the
    sheet manifest must still record the failure, and the interrupt
    propagates."""
    root, ids = _module(tmp_path)
    p = _sheet(tmp_path / "s.csv", [(1, ids[0], "r", "junk"), (2, ids[1], "r", "junk"),
                                    (3, ids[2], "r", "junk")])
    real_call = gate_sheet.SheetReader.__call__
    seen = []

    def call(self, prompt):
        if prompt == gate_sheet.KEY_PROMPT:
            seen.append(prompt)
            if len(seen) == 2:
                raise KeyboardInterrupt
        return real_call(self, prompt)

    monkeypatch.setattr(gate_sheet.SheetReader, "__call__", call)
    transcript = root / "data" / "transcripts" / "t.txt"
    with pytest.raises(KeyboardInterrupt):
        _drive(root, p, transcript)
    sheets = Journal(root / "runs" / "gate-sheets.jsonl").read_all()
    assert len(sheets) == 1 and sheets[0]["session_failed"] is True
    assert sheets[0]["rows_answered"] == 1 and sheets[0]["rows"] == 3
    assert transcript.exists()
    assert len(Journal(root / DECISION_JOURNAL).read_all()) == 1  # row 1 was decided, no more
    sessions = Journal(root / "runs" / "gate-sessions.jsonl").read_all()
    assert sessions[0]["aborted"].startswith("KeyboardInterrupt")


def test_the_tools_refuse_a_non_jst_machine_clock(tmp_path, monkeypatch, capsys):
    """AR-rag4ds-033 F2: the default clock is the machine's and must be JST;
    without an injected clock, a non-JST offset is refused with nothing
    written - the sheet tool and the fold tool alike."""
    import types
    root, ids = _module(tmp_path)
    p = _sheet(tmp_path / "s.csv", [(1, ids[0], "r", "junk"), (2, ids[1], "r", "junk"),
                                    (3, ids[2], "r", "junk")])
    # AR-rag4ds-033 r2 O1: not only a zero offset - UTC+8 (one hour off) is refused too,
    # so a loosened check (any positive offset) cannot hide
    for offset in (0, 8 * 3600):
        monkeypatch.setattr(gate_sheet.time, "localtime",
                            lambda offset=offset: types.SimpleNamespace(tm_gmtoff=offset))
        assert _drive(root, p, root / "data" / "transcripts" / "t.txt", clock=None) == 1
        assert "local clock is not JST" in capsys.readouterr().err
    assert not (root / DECISION_JOURNAL).exists()
    assert not (root / "runs" / "gate-sheets.jsonl").exists()
    assert not (root / "data" / "transcripts" / "t.txt").exists()
    froot = tmp_path / "fold"
    (froot / "runs").mkdir(parents=True)
    (froot / "journal").mkdir()
    monkeypatch.setattr(fold_tree.time, "localtime", lambda: types.SimpleNamespace(tm_gmtoff=0))
    assert fold_tree.run(froot, dry_run=False) == 1
    assert "local clock is not JST" in capsys.readouterr().err
    assert not (froot / "runs" / "fold.jsonl").exists()


def test_fold_tool_folds_verifies_and_writes_a_manifest(tmp_path, capsys):
    root = tmp_path / "mod"
    (root / "runs").mkdir(parents=True)
    (root / "journal").mkdir()
    j = test_s4_fold.full_journal(root / "journal" / "decisions.jsonl")
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@x", "-c", "user.name=t", "commit", "-q",
                    "--allow-empty", "-m", "init"], cwd=root, check=True)
    # dry run on an empty tree: not the fold of the journal, nothing written
    assert fold_tree.run(root, dry_run=True, clock=CLOCK) == 1
    err = capsys.readouterr().err
    assert "FAILED" in err and "MISSING" in err
    assert not (root / "runs" / "fold.jsonl").exists()
    # the fold: tree built, reconstruction OK, manifest written and serialisable
    assert fold_tree.run(root, dry_run=False, clock=CLOCK) == 0
    assert "reconstruction OK" in capsys.readouterr().out
    manifests = Journal(root / "runs" / "fold.jsonl").read_all()
    assert len(manifests) == 1
    m = manifests[0]
    assert m["kind"] == "fold" and m["journal"] == "journal/decisions.jsonl"
    assert m["journal_records"] == len(j.read_all()) and m["reconstruction_problems"] == []
    assert m["live_entries"] == 3 and m["entries_per_dir"]["packs/robotics/entries"] == 1
    json.dumps(m)  # canonical already; the WindowsPath slip of the first run cannot recur
    assert (root / "packs/robotics/entries/actuator-failure-modes.json").is_file()
    # a hand-edited entry: the dry run reports it
    p = root / "packs/robotics/entries/actuator-failure-modes.json"
    p.write_bytes(p.read_bytes() + b"\n")
    assert fold_tree.run(root, dry_run=True, clock=CLOCK) == 1
    assert "FAILED" in capsys.readouterr().err


def test_live_tree_is_the_fold_of_the_live_journal_after_the_validator_sessions():
    """The robotics pack's first gated entry (protective-stop v1, K. Naruse,
    2026-09-14) folds from the live journal; the live tree matches."""
    from rag4ds.kb.fold import verify_reconstruction
    assert verify_reconstruction(Journal(MODULE_ROOT / DECISION_JOURNAL), MODULE_ROOT) == []
    entry = MODULE_ROOT / "packs" / "robotics" / "entries" / "protective-stop.json"
    if not entry.is_file():
        pytest.skip("the robotics pack holds no gated entry yet")
    e = json.loads(entry.read_text(encoding="utf-8"))
    assert e["pack"] == "robotics" and e["layer"] == "domain" and e["status"] == "current"


def test_the_gate_forces_a_utf8_console_so_a_ligature_cannot_abort_a_session(monkeypatch,
                                                                              capsys):
    """AR-rag4ds-049 F4: session 5's first drive aborted after 32 rows when
    the gate printed a candidate carrying U+FB00 (a PDF's 'ff' ligature) to a
    cp1252 console. The fix belongs in the code the wrappers call, or the
    documented `python tools/gate_sheet.py ...` road still dies on the next
    ligature. Here: the abort reproduced, the helper's cure, and BOTH entry
    points applying it before they render anything."""
    import io

    from rag4ds.gate import cli as gate_cli

    lig = "\ufb00"  # the character in the candidate that aborted the drive
    raw = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="")
    console = Console(file=raw, record=True, markup=False, highlight=False, width=80)
    with pytest.raises(UnicodeEncodeError):
        console.print("sheet: row 34 " + lig)
        raw.flush()
    gate_cli.use_utf8_console(raw)
    console.print("sheet: row 34 " + lig)  # the same write, no longer fatal
    raw.flush()
    assert raw.encoding == "utf-8"
    # a stream that cannot be reconfigured is left alone, never raised on -
    # and says so on stderr rather than leaving the operator to find out at
    # the next ligature (r2 O9)
    gate_cli.use_utf8_console(object())
    assert "cannot be set to UTF-8" in capsys.readouterr().err
    # both entry points: argparse refuses the empty argv, but only after the
    # console has been hardened - the reconfigure is the first thing main does
    for entry in (gate_cli.main, gate_sheet.main):
        out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="")
        monkeypatch.setattr(sys, "stdout", out)
        with pytest.raises(SystemExit):
            entry([])
        assert out.encoding == "utf-8", entry
        out.write(lig)  # what aborted the session is now writable
