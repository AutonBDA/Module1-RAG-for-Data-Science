"""AR-rag4ds-008 F10/F12 acceptance: framework_scan's delta sections (4-5)
and the stamp guard (3b) fire on real mutations, proven in scratch git
repos - not on empty commit ranges.
"""

import subprocess
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parents[1]
SCAN = REPO_ROOT / "audit" / "tools" / "framework_scan.py"

LEDGER_STRUCTURE = (
    "| AR ID | JST date | Type | Scope |\n"
    "counts NUMBERED findings\n"
    "| Finding | Severity | Adjudication | Credit | Source |\n"
    "CRITICAL 5, MAJOR 3, MINOR 1\n"
    "| Net daily reward | Cumulative | Citations |\n"
    "Auditor confirmation (report)\n"
)

BRIEFING_OK = (
    "# briefing\n\n2026-08-21 12:00 JST\n\n"
    "plan (as of 2026-08-21 12:00 JST: stub)\n\n"
    "## Change log\n\n"
    "- 2026-08-21 11:00 JST - entry.\n"
)

MAINTAINED_STUB = "# stub\n\n## Change log\n\n- 2026-08-21 10:00 JST - initial.\n"


def git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd, check=True, capture_output=True,
    )


def make_repo(tmp_path):
    (tmp_path / "Z_Audit_Protocol.md").write_text("## 1. one\n", encoding="utf-8")
    audit = tmp_path / "audit"
    (audit / "rag4ds" / "reports").mkdir(parents=True)
    (audit / "LEDGER.md").write_text(LEDGER_STRUCTURE, encoding="utf-8")
    (audit / "BRIEFING.md").write_text(BRIEFING_OK, encoding="utf-8")
    for name in ("README.md", "SCOPES.md"):
        (audit / name).write_text(MAINTAINED_STUB, encoding="utf-8")
    report = audit / "rag4ds" / "reports" / "AR-x_report_r1.md"
    report.write_text("# report\nline one\nline two\nline three\n", encoding="utf-8")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "base")
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return base, report


def run_scan(cwd, *args):
    return subprocess.run(
        [sys.executable, str(SCAN), *args], capture_output=True, text=True, cwd=cwd
    )


def test_deleting_report_lines_is_a_violation(tmp_path):
    base, report = make_repo(tmp_path)
    report.write_text("# report\nline one\n", encoding="utf-8")
    git(tmp_path, "commit", "-aqm", "audit: mutate report")
    result = run_scan(tmp_path, base)
    assert result.returncode == 1, result.stdout
    assert "VIOLATION" in result.stdout, result.stdout


def test_maintained_edit_without_changelog_line_is_delinquent(tmp_path):
    base, _ = make_repo(tmp_path)
    readme = tmp_path / "audit" / "README.md"
    readme.write_text(MAINTAINED_STUB + "edited without a log line\n", encoding="utf-8")
    git(tmp_path, "commit", "-aqm", "audit: silent maintained edit")
    result = run_scan(tmp_path, base)
    assert result.returncode == 1, result.stdout
    assert "DELINQUENT" in result.stdout, result.stdout


def test_stale_briefing_stamp_is_flagged(tmp_path):
    make_repo(tmp_path)
    briefing = tmp_path / "audit" / "BRIEFING.md"
    briefing.write_text(
        BRIEFING_OK + "- 2026-08-21 23:59 JST - newer entry, stamps not moved.\n",
        encoding="utf-8",
    )
    result = run_scan(tmp_path)
    assert result.returncode == 1, result.stdout
    assert "BEHIND newest change-log entry" in result.stdout, result.stdout


def test_clean_delta_passes(tmp_path):
    base, _ = make_repo(tmp_path)
    (tmp_path / "audit" / "rag4ds" / "reports" / "AR-x_report_r2.md").write_text(
        "# second report, append-only growth\n", encoding="utf-8"
    )
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "audit: append new report")
    result = run_scan(tmp_path, base)
    assert result.returncode == 0, result.stdout
    assert "violations: 0" in result.stdout, result.stdout


def test_midfile_insertion_into_committed_report_is_a_violation(tmp_path):
    # AR-rag4ds-010 r3 F6: reports follow the append-only prefix rule -
    # mid-file tampering violates; an appended correction does not.
    base, report = make_repo(tmp_path)
    report.write_text(
        "# report\nline one\nFORGED mid-file\nline two\nline three\n",
        encoding="utf-8",
    )
    git(tmp_path, "commit", "-aqm", "audit: forge mid-file")
    result = run_scan(tmp_path, base)
    assert result.returncode == 1, result.stdout
    assert "non-append modification" in result.stdout, result.stdout


def test_end_appended_report_correction_is_clean(tmp_path):
    # Protocol 4.1/8 permits an auditor to append a correction to his own
    # report; the scanner must not redden it (the AR-rag4ds-010 F6 case).
    base, report = make_repo(tmp_path)
    report.write_text(
        report.read_text(encoding="utf-8") + "\n## Correction (appended)\nfixed.\n",
        encoding="utf-8",
    )
    git(tmp_path, "commit", "-aqm", "audit: append correction to own report")
    result = run_scan(tmp_path, base)
    assert result.returncode == 0, result.stdout
    assert "violations: 0" in result.stdout, result.stdout


def test_renaming_a_report_is_a_violation(tmp_path):
    base, report = make_repo(tmp_path)
    git(tmp_path, "mv", str(report), str(report.parent / "AR-x_renamed.md"))
    git(tmp_path, "commit", "-qm", "audit: rename report")
    result = run_scan(tmp_path, base)
    assert result.returncode == 1, result.stdout
    assert "VIOLATION" in result.stdout, result.stdout


def make_response(tmp_path):
    resp_dir = tmp_path / "audit" / "rag4ds" / "responses"
    resp_dir.mkdir(parents=True)
    resp = resp_dir / "AR-x_response_r1.md"
    resp.write_text("# response\nfirst line\nsecond line\n", encoding="utf-8")
    return resp


def test_midfile_insertion_into_response_is_a_violation(tmp_path):
    base, _ = make_repo(tmp_path)
    resp = make_response(tmp_path)
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "audit: file response")
    base2 = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    resp.write_text("# response\nfirst line\nINSERTED mid-file\nsecond line\n", encoding="utf-8")
    git(tmp_path, "commit", "-aqm", "audit: sneak insertion")
    result = run_scan(tmp_path, base2)
    assert result.returncode == 1, result.stdout
    assert "non-append modification" in result.stdout, result.stdout


def test_end_append_to_response_is_clean(tmp_path):
    base, _ = make_repo(tmp_path)
    resp = make_response(tmp_path)
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "audit: file response")
    base2 = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    resp.write_text(
        resp.read_text(encoding="utf-8") + "\n## Correction (appended)\nlegit append\n",
        encoding="utf-8",
    )
    git(tmp_path, "commit", "-aqm", "audit: legit appended correction")
    result = run_scan(tmp_path, base2)
    assert result.returncode == 0, result.stdout
    assert "violations: 0" in result.stdout, result.stdout


def test_evil_merge_smuggling_is_a_violation(tmp_path):
    # AR-rag4ds-008 F19: a change introduced IN a merge commit itself (in
    # neither parent) must still face the record gates.
    base, report = make_repo(tmp_path)
    git(tmp_path, "checkout", "-qb", "side")
    (tmp_path / "harmless.txt").write_text("side work\n", encoding="utf-8")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "side: harmless")
    git(tmp_path, "checkout", "-q", "master")
    git(tmp_path, "merge", "--no-commit", "--no-ff", "-q", "side")
    # Mid-file forgery: an end-append inside a merge is as legal as a
    # direct one (AR-rag4ds-010 F6); the smuggle that must still fire is
    # prefix-breaking tampering carried by the merge commit.
    report.write_text(
        "# report\nline one\nFORGED via merge\nline two\nline three\n",
        encoding="utf-8",
    )
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "merge side (evil)")
    result = run_scan(tmp_path, base)
    assert result.returncode == 1, result.stdout
    assert "VIOLATION" in result.stdout, result.stdout


def test_ledger_row_erasure_inside_merge_is_a_violation(tmp_path):
    # AR-rag4ds-008 F20: merge-carried ledger row deletion must fire at row
    # level, not be downgraded by the combined-diff rendering.
    base, _ = make_repo(tmp_path)
    ledger = tmp_path / "audit" / "LEDGER.md"
    ledger.write_text(
        LEDGER_STRUCTURE + "| AR-x-F1 | MINOR | CREDITED | +1 | src | note |\n",
        encoding="utf-8",
    )
    git(tmp_path, "commit", "-aqm", "audit: book a row")
    base2 = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    git(tmp_path, "checkout", "-qb", "side2")
    (tmp_path / "harmless2.txt").write_text("x\n", encoding="utf-8")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "side: harmless")
    git(tmp_path, "checkout", "-q", "master")
    git(tmp_path, "merge", "--no-commit", "--no-ff", "-q", "side2")
    ledger.write_text(LEDGER_STRUCTURE, encoding="utf-8")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "merge side2 (erases row)")
    result = run_scan(tmp_path, base2)
    assert result.returncode == 1, result.stdout
    assert "deleted ledger ROW" in result.stdout, result.stdout


def test_same_commit_heading_rename_cannot_self_exempt(tmp_path):
    # AR-rag4ds-008 F21: the pre-log exemption's alibi is the PARENT state;
    # renaming the change-log heading in the delinquent commit itself must
    # not rescue it.
    base, _ = make_repo(tmp_path)
    readme = tmp_path / "audit" / "README.md"
    readme.write_text(
        MAINTAINED_STUB.replace("## Change log", "## History (renamed)")
        + "silent edit\n",
        encoding="utf-8",
    )
    git(tmp_path, "commit", "-aqm", "audit: rename heading and edit silently")
    result = run_scan(tmp_path, base)
    assert result.returncode == 1, result.stdout
    assert "DELINQUENT" in result.stdout, result.stdout


def test_record_file_named_ledger_gets_strict_gates(tmp_path):
    # AR-rag4ds-008 F22: a record file merely NAMED *LEDGER.md must face the
    # strict record gates, not the real ledger's soft row-classification.
    base, _ = make_repo(tmp_path)
    req_dir = tmp_path / "audit" / "meta" / "requests"
    req_dir.mkdir(parents=True)
    decoy = req_dir / "AR-x_LEDGER.md"
    decoy.write_text("# request\nclaim one\nclaim two\n", encoding="utf-8")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "audit: file request with decoy name")
    base2 = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    decoy.write_text("# request\nclaim one\n", encoding="utf-8")
    git(tmp_path, "commit", "-aqm", "audit: delete a claim from the decoy")
    result = run_scan(tmp_path, base2)
    assert result.returncode == 1, result.stdout
    assert "VIOLATION" in result.stdout, result.stdout
    assert "structure-only deletion" not in result.stdout, result.stdout
