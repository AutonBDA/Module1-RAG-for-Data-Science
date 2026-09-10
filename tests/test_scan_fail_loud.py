"""AR-rag4ds-008 F2 acceptance: the scan tools FAIL on violations.

Mutation tests: each builds a doctored mini-tree in tmp_path and asserts
the scanner exits non-zero and names the defect. Green-path exit codes
are covered by test_s0_scaffold.py against the real tree.
"""

import subprocess
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parents[1]


def run(tool: Path, cwd: Path):
    return subprocess.run(
        [sys.executable, str(tool)], capture_output=True, text=True, cwd=cwd
    )


def test_plan_scan_fails_on_non_ascii_and_naming(tmp_path):
    docs = tmp_path / "AutonBDA" / "module1_rag_for_ds" / "docs"
    docs.mkdir(parents=True)
    (docs / "plan_module1.md").write_text(
        "# doctored plan—with an em dash\n\n"
        "## 4. x\n## 5. y\n\nModule 3 is referenced here without a declared map.\n",
        encoding="utf-8",
    )
    result = run(MODULE_ROOT / "tools" / "plan_scan.py", tmp_path)
    assert result.returncode == 1, result.stdout
    assert "plan_scan: FAIL" in result.stdout
    assert "non-ASCII" in result.stdout
    assert "undeclared other-module naming" in result.stdout


def test_plan_scan_passes_clean_doctored_tree(tmp_path):
    docs = tmp_path / "AutonBDA" / "module1_rag_for_ds" / "docs"
    docs.mkdir(parents=True)
    (docs / "plan_module1.md").write_text(
        "# clean plan\n\n## 4. x\n## 5. y\n\nplain ASCII only.\n", encoding="utf-8"
    )
    result = run(MODULE_ROOT / "tools" / "plan_scan.py", tmp_path)
    assert result.returncode == 0, result.stdout
    assert "plan_scan: OK" in result.stdout


def test_framework_scan_fails_on_unresolved_ref_and_missing_tables(tmp_path):
    (tmp_path / "Z_Audit_Protocol.md").write_text(
        "## 1. one\n\nsee section 42 for nothing.\n", encoding="utf-8"
    )
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "LEDGER.md").write_text("# ledger with no tables\n", encoding="utf-8")
    for name in ("README.md", "BRIEFING.md", "SCOPES.md"):
        (audit / name).write_text("# stub\n", encoding="utf-8")
    result = run(REPO_ROOT / "audit" / "tools" / "framework_scan.py", tmp_path)
    assert result.returncode == 1, result.stdout
    assert "framework_scan: FAIL" in result.stdout
    assert "unresolved: ['42']" in result.stdout
    assert "MISSING" in result.stdout


def test_framework_scan_fails_on_non_ascii_flag(tmp_path):
    (tmp_path / "Z_Audit_Protocol.md").write_text("## 1. one\n", encoding="utf-8")
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "LEDGER.md").write_text(
        "| AR ID | JST date | Type | Scope |\ncounts NUMBERED findings\n"
        "| Finding | Severity | Adjudication | Credit | Source |\n"
        "CRITICAL 5, MAJOR 3, MINOR 1\n"
        "| Net daily reward | Cumulative | Citations |\n"
        "Auditor confirmation (report)\n",
        encoding="utf-8",
    )
    for name in ("README.md", "BRIEFING.md", "SCOPES.md"):
        (audit / name).write_text("# stub\n", encoding="utf-8")
    (audit / "flagged.md").write_text("smørrebrød\n", encoding="utf-8")
    result = run(REPO_ROOT / "audit" / "tools" / "framework_scan.py", tmp_path)
    assert result.returncode == 1, result.stdout
    assert "FLAG" in result.stdout
    assert "framework_scan: FAIL" in result.stdout
