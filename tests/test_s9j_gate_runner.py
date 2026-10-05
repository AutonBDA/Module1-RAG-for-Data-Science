"""The gate-session runner's checks (AR-rag4ds-044 r1 F3, O4).

tools/s9d_mapping_gate.ps1 carried these checks as PowerShell here-strings,
where no test could reach them. The auditor showed what that costs: he
filed a probe mapping draft the recommendation sheet never names, and the
pre-flight passed it while printing that the sheet was unchanged - the
sheet was bound by hash, the candidate by nothing, and the pair held only
because two script defaults agreed. The checks now live in
tools/gate_run_checks.py and are driven here.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))
import gate_run_checks as grc  # noqa: E402

from rag4ds.gate.identities import IDENTITIES_FILE  # noqa: E402
from rag4ds.staging.store import StagingStore  # noqa: E402

TS = "2026-09-20 22:00:00 JST"
VALIDATOR = "K. Naruse"
DRAFT = ("standard: cora-ieee-1872\nrows:\n- entry: protective-stop\n"
         "  class: corax:RobotMotion\n  relation: related\n  note: n\n")


def _candidate(author="Draft author - DA", draft=DRAFT, pack="robotics"):
    return {"kind": "mapping-draft", "pack": pack, "standard": "cora-ieee-1872",
            "author": author, "yaml_draft": draft}


def _module(tmp_path, candidate=None, sheet_names=True, state=None):
    """A scratch module: the shipped identities, one filed mapping draft,
    and a sheet that names it (or does not)."""
    root = tmp_path / "module"
    for d in ("journal", "stage", "runs", "data/transcripts"):
        (root / d).mkdir(parents=True, exist_ok=True)
    shutil.copy(MODULE_ROOT / IDENTITIES_FILE, root / IDENTITIES_FILE)
    store = StagingStore(root / grc.STAGING)
    cid = store.file_candidate(candidate or _candidate(), TS)
    if state:
        store.transition(cid, state, TS, "test", "journal:1")
    sheet = root / "data" / "transcripts" / "sheet.md"
    body = "# recommendations\n\nApprove the row.\n"
    if sheet_names:
        body += f"\n| 1 | {cid} | approve |\n"
    sheet.write_bytes(body.encode("utf-8"))
    sha = hashlib.sha256(sheet.read_bytes()).hexdigest()
    return root, cid, sheet, sha


def _pre(root, cid, sha, sheet="data/transcripts/sheet.md", approver=VALIDATOR, pack="robotics",
         transcript="data/transcripts/t.txt", skip=False):
    return grc.preflight(root, cid, pack, approver, Path(sheet), sha, Path(transcript), skip)[0]


# ----- pre-flight -----


def test_a_clean_session_passes_preflight(tmp_path):
    root, cid, _sheet, sha = _module(tmp_path)
    assert _pre(root, cid, sha) == []


def test_the_sheet_must_name_the_candidate(tmp_path):
    """AR-rag4ds-044 r1 F3, the auditor's probe: a candidate the sheet
    never names is refused even though the sheet's own hash is right."""
    root, cid, _sheet, sha = _module(tmp_path, sheet_names=False)
    problems = _pre(root, cid, sha)
    assert any("does not name" in p and cid in p for p in problems), problems
    # and the hash check has NOT fired - the sheet is genuinely unchanged
    assert not any("has changed" in p for p in problems), problems


def test_a_second_candidate_cannot_ride_the_first_ones_sheet(tmp_path):
    """The same attack from the other side: the sheet names candidate A,
    the runner is pointed at candidate B, both PENDING."""
    root, cid_a, _sheet, sha = _module(tmp_path)
    store = StagingStore(root / grc.STAGING)
    cid_b = store.file_candidate(_candidate(draft=DRAFT.replace("note: n", "note: other")), TS)
    problems = _pre(root, cid_b, sha)
    assert any("does not name" in p and cid_b in p for p in problems), problems


def test_a_changed_sheet_is_refused(tmp_path):
    root, cid, sheet, sha = _module(tmp_path)
    sheet.write_bytes(sheet.read_bytes() + b"and one more row\n")
    problems = _pre(root, cid, sha)
    assert any("has changed since the validator reviewed it" in p for p in problems), problems


@pytest.mark.parametrize("state, expected", [
    ("APPROVED", "not PENDING"),
    ("REJECTED", "not PENDING"),
])
def test_an_already_decided_candidate_is_refused(tmp_path, state, expected):
    root, cid, _sheet, sha = _module(tmp_path, state=state)
    assert any(expected in p for p in _pre(root, cid, sha))


def test_skip_gate_refuses_a_candidate_still_pending(tmp_path):
    root, cid, _sheet, sha = _module(tmp_path)
    assert any("still PENDING" in p for p in _pre(root, cid, sha, skip=True))
    root, cid, _sheet, sha = _module(tmp_path / "b", state="APPROVED")
    assert _pre(root, cid, sha, skip=True) == []


def test_the_author_may_not_be_the_approver_and_the_approver_must_hold_the_role(tmp_path):
    root, cid, _sheet, sha = _module(tmp_path, candidate=_candidate(author=VALIDATOR))
    assert any("names K. Naruse as its author" in p for p in _pre(root, cid, sha))
    root, cid, _sheet, sha = _module(tmp_path / "b")
    assert any("does not validate pack" in p for p in _pre(root, cid, sha, approver="I. Paik"))
    assert any("approver refused" in p for p in _pre(root, cid, sha, approver="Nobody At All"))


def test_a_used_transcript_name_and_a_busy_queue_are_refused(tmp_path):
    root, cid, _sheet, sha = _module(tmp_path)
    (root / "data" / "transcripts" / "t.txt").write_text("x", encoding="utf-8")
    assert any("already exists" in p for p in _pre(root, cid, sha))
    root, cid, _sheet, sha = _module(tmp_path / "b")
    StagingStore(root / grc.STAGING).file_candidate(
        _candidate(draft=DRAFT.replace("note: n", "note: second")), TS)
    assert any("not exactly" in p for p in _pre(root, cid, sha))


def test_a_draft_of_another_pack_is_refused(tmp_path):
    root, cid, _sheet, sha = _module(tmp_path, candidate=_candidate(pack="bio"))
    assert any("is not a robotics mapping draft" in p for p in _pre(root, cid, sha))


# ----- what the validator decided -----


def _decide(root, cid, decision, approver=VALIDATOR, artifact=True, reason=None):
    rec = {"schema": 1, "candidate_id": cid, "approver": approver, "timestamp": TS,
           "type": decision, "pack": "robotics", "source_reference": f"staging:{cid}"}
    if reason:
        rec["reason"] = reason
    if artifact and decision == "approve":
        rec["mapping_artifact"] = {"standard": "cora-ieee-1872", "rows": [
            {"entry": "protective-stop", "class": "corax:RobotMotion",
             "relation": "related", "note": "n"}]}
    (root / grc.DECISIONS).write_text(json.dumps(rec, sort_keys=True) + "\n", encoding="utf-8")
    StagingStore(root / grc.STAGING).transition(
        cid, "APPROVED" if decision == "approve" else "REJECTED", TS, approver, "journal:1")


def test_a_session_that_decided_nothing_is_refused(tmp_path):
    """The case both empty sessions of 2026-09-20 hit in use."""
    root, cid, _sheet, _sha = _module(tmp_path)
    with pytest.raises(ValueError, match="without a decision"):
        grc.verify_decision(root, cid, VALIDATOR)


def test_an_approval_is_verified_and_its_artifact_required(tmp_path):
    root, cid, _sheet, _sha = _module(tmp_path)
    _decide(root, cid, "approve")
    out = grc.verify_decision(root, cid, VALIDATOR)
    assert out["state"] == "APPROVED"
    assert out["record"]["mapping_artifact"]["rows"][0]["entry"] == "protective-stop"
    root, cid, _sheet, _sha = _module(tmp_path / "b")
    _decide(root, cid, "approve", artifact=False)
    with pytest.raises(ValueError, match="embeds no mapping artifact"):
        grc.verify_decision(root, cid, VALIDATOR)


def test_a_rejection_is_verified_and_needs_no_artifact(tmp_path):
    root, cid, _sheet, _sha = _module(tmp_path)
    _decide(root, cid, "reject", artifact=False, reason="no counterpart in CORA")
    out = grc.verify_decision(root, cid, VALIDATOR)
    assert out["state"] == "REJECTED" and out["record"]["reason"] == "no counterpart in CORA"


def test_a_decision_by_someone_else_is_refused(tmp_path):
    root, cid, _sheet, _sha = _module(tmp_path)
    _decide(root, cid, "approve", approver="T. A. Siriweera")
    with pytest.raises(ValueError, match="names 'T. A. Siriweera', not 'K. Naruse'"):
        grc.verify_decision(root, cid, VALIDATOR)


def test_two_decisions_on_one_candidate_are_refused(tmp_path):
    root, cid, _sheet, _sha = _module(tmp_path)
    _decide(root, cid, "approve")
    with (root / grc.DECISIONS).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"schema": 1, "candidate_id": cid, "approver": VALIDATOR,
                             "timestamp": TS, "type": "approve"}, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="2 decisions cite"):
        grc.verify_decision(root, cid, VALIDATOR)


# ----- main(): the printed lines against the checks actually made -----
# AR-rag4ds-044 r2 F4 and O7. Both defects of the "states what it did not
# check" class lived in main(), which nothing drove. These drive it.


def _env(monkeypatch, root, cid, sha, transcript="data/transcripts/t.txt", skip=False,
         sheet="data/transcripts/sheet.md", approver=VALIDATOR):
    for k, v in {"S9D_ROOT": str(root), "S9D_CANDIDATE": cid, "S9D_PACK": "robotics",
                 "S9D_APPROVER": approver, "S9D_SHEET": sheet, "S9D_SHEET_SHA": sha,
                 "S9D_TRANSCRIPT": transcript, "S9D_COMMIT": "abcdef0",
                 "S9D_OUTCOME": "data/transcripts/outcome.json"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("S9D_SKIPGATE", "1" if skip else "")


def test_main_prints_only_lines_its_checks_established(tmp_path, monkeypatch, capsys):
    """AR-rag4ds-044 r3 O9: the EXACT set, not a subset. Checking only that
    the expected lines appear leaves room for a confirmation that is simply
    untrue - the auditor appended an unconditional false line and all
    twenty tests still passed."""
    root, cid, _sheet, sha = _module(tmp_path)
    _env(monkeypatch, root, cid, sha)
    assert grc.main(["preflight"]) == 0
    lines = [ln.strip() for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert lines == [
        f"the sheet is unchanged ({sha[:12]}...) and names this candidate",
        f"transcript {root / 'data/transcripts/t.txt'} is free",
        "approver K. Naruse holds the validator role for robotics",
        "the draft's author is 'Draft author - DA', not the approver",
        f"candidate {cid} is PENDING",
        "it is the only candidate queued for robotics",
        "code commit abcdef0",
    ], lines


def test_main_under_skip_gate_never_claims_a_transcript_it_did_not_check(tmp_path, monkeypatch,
                                                                         capsys):
    """r2 F4: the transcript line was printed unconditionally while its own
    check was gated on not skip_gate - so with skip-gate and an EXISTING
    transcript the run asserted 'is free' about a file that was not."""
    root, cid, _sheet, sha = _module(tmp_path, state="APPROVED")
    used = root / "data" / "transcripts" / "used.txt"
    used.write_text("a previous session", encoding="utf-8")
    _env(monkeypatch, root, cid, sha, transcript="data/transcripts/used.txt", skip=True)
    assert grc.main(["preflight"]) == 0
    out = capsys.readouterr().out
    assert "is free" not in out, out
    # the exact set again (r3 O9): under skip-gate the transcript line is
    # replaced, not merely dropped, and the queue line does not exist
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    assert lines == [
        f"the sheet is unchanged ({sha[:12]}...) and names this candidate",
        "no transcript is needed: the gate is not opened (skip-gate)",
        "approver K. Naruse holds the validator role for robotics",
        "the draft's author is 'Draft author - DA', not the approver",
        f"candidate {cid} is APPROVED",
        "code commit abcdef0",
    ], lines


def test_main_refuses_on_stderr_and_prints_no_confirmation(tmp_path, monkeypatch, capsys):
    root, cid, _sheet, sha = _module(tmp_path, sheet_names=False)
    _env(monkeypatch, root, cid, sha)
    assert grc.main(["preflight"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "", captured.out
    assert "REFUSED: the sheet does not name" in captured.err


def test_main_verify_writes_the_outcome_only_on_a_real_decision(tmp_path, monkeypatch, capsys):
    root, cid, _sheet, sha = _module(tmp_path)
    _env(monkeypatch, root, cid, sha)
    outcome = root / "data" / "transcripts" / "outcome.json"
    assert grc.main(["verify"]) == 1
    assert "REFUSED: the session ended without a decision" in capsys.readouterr().err
    assert not outcome.exists()
    _decide(root, cid, "approve")
    assert grc.main(["verify"]) == 0
    out = capsys.readouterr().out
    assert f"{cid} is now APPROVED" in out and "protective-stop related corax:RobotMotion" in out
    assert json.loads(outcome.read_text(encoding="utf-8"))["state"] == "APPROVED"


def test_main_rejects_an_unknown_subcommand(capsys):
    assert grc.main(["fold"]) == 2
    assert "usage:" in capsys.readouterr().err
