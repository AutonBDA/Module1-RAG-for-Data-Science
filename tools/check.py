"""The one check entry point (9.2): every CI job is this script, and this
script runs identically on the local machine. Runs ruff, pytest, and both
scan tools; exits non-zero on the first failure. No step is optional and
no failure is swallowed.
"""

import subprocess
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parents[1]

STEPS = [
    (
        "ruff",
        [sys.executable, "-m", "ruff", "check", "src", "tests", "tools"],
        MODULE_ROOT,
    ),
    (
        "pytest",
        [sys.executable, "-m", "pytest"],
        MODULE_ROOT,
    ),
    (
        "plan_scan",
        [sys.executable, str(MODULE_ROOT / "tools" / "plan_scan.py")],
        REPO_ROOT,
    ),
    (
        "framework_scan",
        [sys.executable, str(REPO_ROOT / "audit" / "tools" / "framework_scan.py")],
        REPO_ROOT,
    ),
]


def main() -> int:
    print(f"check.py invocation: {sys.executable} {' '.join(sys.argv)}", flush=True)
    base = sys.argv[1] if len(sys.argv) > 1 else None
    for name, cmd, cwd in STEPS:
        if name == "framework_scan" and base:
            cmd = cmd + [base]  # sections 4-5: record append-only + change-log coverage
        print(f"\n== {name}: {' '.join(str(c) for c in cmd)} (cwd={cwd}) ==", flush=True)
        result = subprocess.run(cmd, cwd=cwd)
        if result.returncode != 0:
            print(f"FAILED: {name} exited {result.returncode}", file=sys.stderr, flush=True)
            return result.returncode
    print("\nall checks green", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
