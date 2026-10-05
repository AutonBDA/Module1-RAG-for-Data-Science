"""Fill the empty keys of a decision sheet at the keyboard (S9c part 3).

The validator is asked, for each row whose key is empty, one question:
the row's candidate content is shown (from staging, the same text the gate
renders as "content"), and he types r + a reason, or s (see it live -
the only road to an approval). Nothing else is offered: no
recommendation, no default. The sheet is rewritten after every answer,
so a Ctrl+C keeps what was answered; rows already keyed are never
touched. The gate never runs here - tools/gate_sheet.py does that, from
the filled sheet, after its own checks.

    python tools/fill_sheet.py --sheet authoring/<sheet>.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE_ROOT = HERE.parent
sys.path.insert(0, str(MODULE_ROOT / "src"))

from rag4ds.staging.store import StagingStore  # noqa: E402

FIELDS = ["n", "id", "key", "reason"]


class FillError(RuntimeError):
    """A refusal - printed and exit 1."""


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, restkey="_extra")
        if reader.fieldnames != FIELDS:
            raise FillError(f"{path}: columns must be n,id,key,reason (got {reader.fieldnames})")
        rows = []
        for i, row in enumerate(reader, 1):
            if "_extra" in row or any(v is None for v in row.values()):
                raise FillError(f"{path}: row {i}: expected 4 fields")
            rows.append({k: (row[k] or "").strip() for k in FIELDS})
    if not rows:
        raise FillError(f"{path}: no rows")
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def ask_row(row: dict, content: str, read, out) -> None:
    """One question per empty row; validated, re-asked until it is r + a
    reason or s. The row is updated in place."""
    print(f"\nrow {row['n']}  {row['id']}\n  content: {content}", file=out)
    while True:
        key = read("  r = reject (then a reason)   s = see it live (also for approve)   key: ")
        key = key.strip().lower()
        if key == "s":
            row["key"], row["reason"] = "s", ""
            return
        if key == "r":
            reason = read("  reason (one word or a few, no comma): ").strip()
            if not reason:
                print("  a rejection needs a reason", file=out)
                continue
            if "," in reason or '"' in reason:
                print("  no comma or quote in the reason", file=out)
                continue
            row["key"], row["reason"] = "r", reason
            return
        if key in ("a", "e"):
            print("  an approval needs the passage on screen: type s and it is shown live",
                  file=out)
            continue
        print("  type r or s", file=out)


def run(module_root: Path, sheet: Path, read=input, out=None) -> int:
    out = out or sys.stdout
    try:
        rows = read_rows(sheet)
        store = StagingStore(Path(module_root) / "stage" / "candidates.jsonl")
        cands = store.candidates()
        empty = [r for r in rows if not r["key"]]
        print(f"fill_sheet: {sheet} - {len(rows)} rows, {len(empty)} to fill", file=out)
        for r in empty:
            rec = cands.get(r["id"])
            if rec is None:
                raise FillError(f"row {r['n']}: {r['id']} is not a filed candidate")
            c = rec["candidate"]
            content = c["proposed"]["content"] if c["kind"] == "entry" else c["yaml_draft"]
            ask_row(r, content, read, out)
            write_rows(sheet, rows)  # every answer saved at once
        left = sum(1 for r in rows if not r["key"])
        print(f"fill_sheet: done - {len(rows) - left} of {len(rows)} rows keyed"
              + ("" if left == 0 else f"; {left} still empty"), file=out)
        return 0 if left == 0 else 1
    except (FillError, OSError, KeyError) as exc:
        print(f"fill_sheet: REFUSED - {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sheet", required=True, type=Path)
    ap.add_argument("--module-root", type=Path, default=MODULE_ROOT)
    a = ap.parse_args(argv)
    try:
        return run(a.module_root, a.sheet)
    except KeyboardInterrupt:
        print("\nfill_sheet: stopped - answered rows are saved; run again to continue",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
