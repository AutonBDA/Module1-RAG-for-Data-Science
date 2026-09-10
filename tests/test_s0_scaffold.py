"""S0 acceptance: the 9.1 layout exists and both scan tools run clean.

These are real tests of S0's deliverable (the scaffold), not placeholders:
they fail if a 9.1 directory disappears or a scan tool stops running.
"""

import subprocess
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parents[1]

LAYOUT_9_1 = [
    "docs",
    "src/rag4ds/model",
    "src/rag4ds/corpus",
    "src/rag4ds/extract",
    "src/rag4ds/staging",
    "src/rag4ds/gate",
    "src/rag4ds/kb",
    "src/rag4ds/index",
    "src/rag4ds/retrieve",
    "src/rag4ds/evalhook",
    "src/rag4ds/auditlog",
    "packs",
    "shared/general",
    "shared/tasks",
    "journal",
    "stage",
    "eval/gold",
    "runs",
    "tests",
    "tools",
]


def test_layout_9_1_directories_exist():
    missing = [d for d in LAYOUT_9_1 if not (MODULE_ROOT / d).is_dir()]
    assert not missing, f"9.1 directories missing: {missing}"


def test_plan_scan_runs_clean():
    result = subprocess.run(
        [sys.executable, str(MODULE_ROOT / "tools" / "plan_scan.py")],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "non-ASCII: 0" in result.stdout, result.stdout


def test_framework_scan_runs_clean():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "audit" / "tools" / "framework_scan.py")],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "non-exempt flags: 0" in result.stdout, result.stdout


def test_framework_scan_delta_mode_runs():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "audit" / "tools" / "framework_scan.py"), "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout
    assert "== 4." in result.stdout and "== 5." in result.stdout, result.stdout


def test_rag4ds_package_is_installed():
    import rag4ds  # noqa: F401  - AR-rag4ds-008 F6: the increment installs its own package


def test_locks_agree_on_common_packages():
    # AR-rag4ds-008 F17: the Windows lock (VT-3 pins) and the Linux lock must
    # agree wherever they overlap.
    def parse(name):
        pins = {}
        for line in (MODULE_ROOT / name).read_text(encoding="utf-8").splitlines():
            if "==" in line:
                pkg, ver = line.strip().split("==", 1)
                pins[pkg.lower()] = ver
        return pins

    win, linux = parse("requirements.lock"), parse("requirements-linux.lock")
    common = set(win) & set(linux)
    assert len(common) > 50, f"suspiciously small overlap: {len(common)}"
    disagree = {p: (win[p], linux[p]) for p in common if win[p] != linux[p]}
    assert not disagree, f"locks disagree on: {disagree}"
