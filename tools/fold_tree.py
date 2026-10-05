"""Fold the decision journal into the live tree and verify the fold (plan
component 4, NFR1; the S4 road, now a committed tool - AR-rag4ds-026 O5
asked for a fold that writes a run manifest). Run from the module root with
the module venv, after a gate session:

    python tools/fold_tree.py [--dry-run]

compile_tree rebuilds every gated directory (shared/general, shared/tasks,
packs/<code>/entries and mappings) from journal/decisions.jsonl - off-tree
first, then swapped in; a pack's ungated content is never touched - and
verify_reconstruction folds the journal again from empty and compares with
the live tree byte for byte. Any difference is a loud failure (exit 1)
after the manifest is written. The manifest (runs/fold.jsonl, NFR2) carries
the journal's record count and sha256, the code commit, the timestamp, the
number of live entries per gated directory and the reconstruction result.
--dry-run verifies the CURRENT tree against the journal and writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))
from rag4ds.auditlog.journal import Journal  # noqa: E402
from rag4ds.gate.cli import DECISION_JOURNAL  # noqa: E402
from rag4ds.kb.fold import FoldError, compile_tree, gated_dirs, verify_reconstruction  # noqa: E402


def jst_now() -> str:
    if time.localtime().tm_gmtoff != 9 * 3600:
        raise FoldError("this machine's local clock is not JST (UTC+9); refusing")
    return time.strftime("%Y-%m-%d %H:%M:%S JST")


def head_commit(module_root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=module_root,
                          capture_output=True, text=True, check=True).stdout.strip()


def _counts(root: Path) -> dict:
    out = {}
    for rel in gated_dirs(root):
        d = root / rel
        out[str(rel).replace("\\", "/")] = (sum(1 for p in d.iterdir()
                                                if p.is_file() and p.suffix == ".json")
                                            if d.is_dir() else 0)
    return out


def run(module_root: Path, dry_run: bool, clock=None, out=None) -> int:
    out = out or sys.stdout
    module_root = Path(module_root)
    journal = Journal(module_root / DECISION_JOURNAL)
    try:
        ts = (clock or jst_now)()
        commit = head_commit(module_root)
        raw = journal.path.read_bytes() if journal.path.exists() else b""
        records = journal.read_all()
        if not dry_run:
            result = compile_tree(journal, module_root)
            live = len(result.versions)
        else:
            live = None
        problems = verify_reconstruction(journal, module_root)
        manifest = {
            "kind": "fold", "schema": 1, "journal": str(DECISION_JOURNAL).replace("\\", "/"),
            "journal_records": len(records), "journal_sha256": hashlib.sha256(raw).hexdigest(),
            "live_entries": live, "entries_per_dir": _counts(module_root),
            "reconstruction_problems": problems, "dry_run": dry_run,
            "code_commit": commit, "timestamp": ts, "tool": "tools/fold_tree.py",
        }
        if not dry_run:
            Journal(module_root / "runs" / "fold.jsonl").append(manifest)
    except FoldError as exc:
        print(f"fold_tree: REFUSED - {exc}", file=sys.stderr)
        return 1
    print(f"fold_tree: {'dry run - ' if dry_run else ''}{len(records)} journal records; "
          f"entries per gated dir {manifest['entries_per_dir']}; reconstruction "
          f"{'OK' if not problems else 'FAILED'}; at {commit}, {ts}", file=out)
    if problems:
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("fold_tree: FAILED - the live tree is not the fold of the journal (NFR1)",
              file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-root", default=str(MODULE_ROOT))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return run(Path(a.module_root), a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
