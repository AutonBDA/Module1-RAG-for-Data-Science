"""evidence.py - capture audit-request evidence from a CLEAN worktree at a
pin, hermetically, or fail loudly. The class-level remedy for the
unguarded-chain failure (AR-rag4ds-015 O2): evidence can no longer be
"captured" from a missing directory or a failed step and filed anyway,
because this tool either prints a complete block or exits non-zero with
nothing printed.

Usage, from the repository root:
    python AutonBDA/module1_rag_for_ds/tools/evidence.py <pin> [--venv-python PATH] [--base REV]

It creates (or refreshes) a detached worktree of <pin> under the
scratchpad path given by RAG4DS_EVIDENCE_WORKTREE (or a sibling
'_evidence_wt' directory next to the repository if unset), then runs
pytest and tools/check.py there with the given interpreter and prints the
evidence block that requests paste verbatim.

Hermetic (AR-rag4ds-016 F3): the module venv's editable install resolves
`import rag4ds` to the MAIN working tree, so cwd alone does not test the
pin. Every subprocess runs with PYTHONPATH=<worktree>/src prepended, and
before any figure is taken the tool asks the interpreter where
rag4ds.__file__ resolves and refuses (exit 1, nothing on stdout) unless
that is inside the worktree. The resolved path is printed inside the
block, so the identity of the code tested is part of the evidence.
Any failure - missing pin, worktree error, identity mismatch, failing
pytest or empty closing line, non-zero check - exits 1 and prints NOTHING
to stdout (diagnostics go to stderr), so a chain can never carry a
half-captured block forward.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

MODULE_REL = Path("AutonBDA") / "module1_rag_for_ds"


def run(cmd, cwd, env=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)


def hermetic_env(module_root: Path) -> dict:
    """os.environ with <module_root>/src FIRST on PYTHONPATH, so the pinned
    package shadows the venv's editable (.pth) install of the main tree."""
    env = dict(os.environ)
    src = str(Path(module_root) / "src")
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src + (os.pathsep + prior if prior else "")
    return env


def import_identity(py: str, module_root: Path) -> tuple[bool, str]:
    """Ask the interpreter (under hermetic_env) where rag4ds resolves.
    Returns (ok, resolved_path_or_error): ok only if the resolved package
    file is <module_root>/src/rag4ds/__init__.py."""
    module_root = Path(module_root)
    probe = run(
        [py, "-c", "import rag4ds, pathlib; print(pathlib.Path(rag4ds.__file__).resolve())"],
        module_root, hermetic_env(module_root),
    )
    if probe.returncode != 0:
        return False, f"import probe failed: {probe.stderr.strip()}"
    resolved = probe.stdout.strip()
    expected = (module_root / "src" / "rag4ds" / "__init__.py").resolve()
    try:
        same = Path(resolved).resolve() == expected
    except OSError:
        same = False
    return same, resolved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pin")
    ap.add_argument("--venv-python", default=os.environ.get("RAG4DS_VENV_PYTHON", sys.executable))
    ap.add_argument("--base", default=None,
                    help="delta base for check.py (framework_scan sections 4-5)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[3]
    wt = Path(os.environ.get("RAG4DS_EVIDENCE_WORKTREE") or (repo.parent / "_evidence_wt"))

    rev = run(["git", "rev-parse", "--verify", "--quiet", args.pin + "^{commit}"], repo)
    if rev.returncode != 0 or not rev.stdout.strip():
        print(f"evidence: pin {args.pin!r} does not resolve to a commit", file=sys.stderr)
        return 1
    full = rev.stdout.strip()

    run(["git", "worktree", "prune"], repo)
    if wt.exists():
        rm = run(["git", "worktree", "remove", "--force", str(wt)], repo)
        if rm.returncode != 0 and wt.exists():
            print(f"evidence: could not remove stale worktree {wt}: {rm.stderr}", file=sys.stderr)
            return 1
    add = run(["git", "worktree", "add", "--detach", str(wt), full], repo)
    if add.returncode != 0 or not (wt / MODULE_REL).is_dir():
        print(f"evidence: worktree creation failed: {add.stderr}", file=sys.stderr)
        return 1

    mod = wt / MODULE_REL
    py = args.venv_python
    env = hermetic_env(mod)
    ok, resolved = import_identity(py, mod)
    if not ok:
        print(f"evidence: rag4ds does not resolve inside the worktree: {resolved}",
              file=sys.stderr)
        return 1

    pt = run([py, "-m", "pytest", "-q"], mod, env)
    pt_line = pt.stdout.strip().splitlines()[-1] if pt.stdout.strip() else ""
    if pt.returncode != 0 or "passed" not in pt_line:
        print(f"evidence: pytest did not pass cleanly (exit {pt.returncode}): {pt_line!r}",
              file=sys.stderr)
        return 1
    ck_cmd = [py, str(mod / "tools" / "check.py")] + ([args.base] if args.base else [])
    ck = run(ck_cmd, mod, env)
    if ck.returncode != 0:
        print(f"evidence: check.py exited {ck.returncode}", file=sys.stderr)
        return 1

    short = run(["git", "rev-parse", "--short", "HEAD"], wt).stdout.strip()
    base_note = f" {args.base}" if args.base else "   (no delta base: scan sections 4-5 skipped)"
    print(f"    $ git rev-parse --short HEAD   (clean worktree at {wt})")
    print(f"    {short}")
    print()
    print("    $ PYTHONPATH=<worktree>/src python -c 'import rag4ds; print(rag4ds.__file__)'")
    print(f"    {resolved}")
    print()
    print(f"    $ {py} -m pytest -q   (closing line; PYTHONPATH as above)")
    print(f"    {pt_line}")
    print()
    print(f"    $ {py} tools/check.py{base_note}")
    print("    exit=0   (returncode captured via subprocess; PYTHONPATH as above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
