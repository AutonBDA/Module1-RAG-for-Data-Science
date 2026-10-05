"""Sheet-driven gate session (S9c, the validator's request of 2026-09-14):
the registered validator decides on a WRITTEN decision sheet - one row per
PENDING candidate of the pack, in the gate's own queue order, with the key
and the reason - and this tool drives the real gate (rag4ds.gate.cli
GateSession, the same journal, staging transitions, session manifest and
transcript) from that sheet, answering only the two prompts a rejection or
a skip needs. Nothing is decided here: the sheet is the validator's
decision, endorsed in writing; the tool is the keyboard.

    python tools/gate_sheet.py --sheet authoring/<sheet>.csv --approver "K. Naruse"
                               --code-commit <HEAD> --index-dir data/index --data-dir .
                               --pack robotics --transcript data/transcripts/<name>.txt
                               [--dry-run]

Sheet: CSV with the columns n,id,key,reason. Keys admitted: r (reject, the
reason column is typed at the gate's reason prompt), s (skip - stays
PENDING for a live screen) and, since S9c part 3, g (group - the reason
column carries the head candidate id, typed at the gate's head prompt; the
gate itself refuses a head that is not a PENDING candidate of the same
pack). a and e are REFUSED here: an approval needs the validator in front
of the rendered passage (the gate's evidence rule, plan 6.4); those
candidates are skipped by the sheet and decided at a live session. The
sheet's ids must equal the gate's queue for the pack, in order, or the
tool refuses before any decision is written; the reader answers a key
prompt only for the candidate the gate is deciding; any prompt other than
the key, reason and head prompts aborts the session loudly (the gate
writes its manifest either way). What the sheet supplies at every prompt
is echoed into the transcript.

Every run appends a manifest to runs/gate-sheets.jsonl: the sheet's path and
sha256, the approver, the code commit, the timestamp, the row counts - so
the record can say which sheet drove which session. The gate's own session
manifest (runs/gate-sessions.jsonl) records elapsed seconds of the DRIVE,
not of the validator's reading; the validator's review time is reported
separately in the record. --dry-run verifies the sheet against the queue
and prints what would be sent; it opens no session.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))
from rich.console import Console  # noqa: E402

from rag4ds.auditlog.journal import Journal  # noqa: E402
from rag4ds.gate.cli import (  # noqa: E402
    DECISION_JOURNAL, HEAD_PROMPT, KEY_PROMPT, REASON_PROMPT,  # the prompts: one literal each
    GateContext, GateError, GateSession, display_provenance_of,
    make_related_finder, make_source_resolver, transcript_path_is_publishable, use_utf8_console,
    write_transcript,
)
from rag4ds.gate.identities import IDENTITIES_FILE, IdentityError, IdentityRegistry  # noqa: E402
from rag4ds.staging.store import StagingStore  # noqa: E402

SHEET_KEYS = ("r", "s", "g")  # a 'g' row's reason column carries the head id


class SheetError(RuntimeError):
    """A refusal before any decision is written - printed and exit 1."""


def load_sheet(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, restkey="_extra")
        if reader.fieldnames != ["n", "id", "key", "reason"]:
            raise SheetError(f"{path}: columns must be n,id,key,reason (got {reader.fieldnames})")
        for i, row in enumerate(reader, 1):
            extra = row.pop("_extra", [])
            n_fields = sum(v is not None for v in row.values()) + len(extra)
            if n_fields != 4:
                # AR-rag4ds-032 F5: a bare comma in a reason would otherwise be cut
                # silently at the comma (csv fills the rest key); a short row would
                # read None for the missing columns
                raise SheetError(f"{path}: row {i}: {n_fields} fields, expected 4 - quote a "
                                 f"reason that contains a comma")
            key = (row["key"] or "").strip().lower()
            if key in ("a", "e"):
                raise SheetError(f"{path}: row {i} ({row['id']}): key {key!r} is refused on a "
                                 f"sheet - an approval or an edit needs the validator at the "
                                 f"rendered passage; mark it s and decide it at a live session")
            if key not in SHEET_KEYS:
                raise SheetError(f"{path}: row {i} ({row['id']}): key must be r, s or g, got "
                                 f"{key!r}")
            if key == "r" and not (row["reason"] or "").strip():
                raise SheetError(f"{path}: row {i} ({row['id']}): a rejection needs a reason")
            if key == "g":
                # S9c part 3: a grouping row's reason column IS the head id; the gate
                # itself refuses a head that is not a PENDING candidate of the same pack
                head = (row["reason"] or "").strip()
                if not head.startswith("cand-") or head == row["id"].strip():
                    raise SheetError(f"{path}: row {i} ({row['id']}): a grouping needs the head "
                                     f"candidate id in the reason column (got {head!r})")
            if str(i) != (row["n"] or "").strip():
                raise SheetError(f"{path}: row {i}: n must be {i}, got {row['n']!r}")
            rows.append({"n": i, "id": row["id"].strip(), "key": key,
                         "reason": (row["reason"] or "").strip()})
    if not rows:
        raise SheetError(f"{path}: no rows")
    return rows


class SheetReader:
    """The gate's read callback: answers the key prompt with the next row's
    key - only if the candidate the gate is deciding (session.current_cid,
    AR-rag4ds-032 F1) is that row's id; a mismatch aborts the session, so a
    candidate the loop skipped or re-showed without consuming a key can
    never receive another row's decision - and the reason prompt with its
    reason; anything else aborts."""

    def __init__(self, rows: list[dict], console: Console):
        self.rows, self.console, self.pos, self.current = rows, console, 0, None
        self.session = None  # set by run() once the session exists

    def __call__(self, prompt: str) -> str:
        if prompt == KEY_PROMPT:
            if self.pos >= len(self.rows):
                self.console.print("sheet: every row answered - quitting")
                return "q"
            row = self.rows[self.pos]
            shown = getattr(self.session, "current_cid", None)
            if shown != row["id"]:
                raise GateError(f"sheet road: the gate is deciding {shown!r} but the sheet's "
                                f"next row {row['n']} is {row['id']!r} - a candidate was skipped "
                                f"or re-shown without a key; the session is aborted, nothing "
                                f"further is decided")
            self.current = row
            self.pos += 1
            self.console.print(f"sheet: row {self.current['n']} {self.current['id']} -> "
                               f"{self.current['key']}")
            return self.current["key"]
        if prompt == REASON_PROMPT and self.current and self.current["key"] == "r":
            # AR-rag4ds-034 F3: the transcript witnesses the reason and the head id
            # the sheet supplied, not only the key
            self.console.print(f"sheet: {prompt}{self.current['reason']}")
            return self.current["reason"]
        if prompt == HEAD_PROMPT and self.current and self.current["key"] == "g":
            self.console.print(f"sheet: {prompt}{self.current['reason']}")
            return self.current["reason"]
        raise GateError(f"sheet road: unexpected prompt {prompt!r} at row "
                        f"{self.current['n'] if self.current else '?'} - the sheet answers only "
                        f"the key, reason and head prompts; the session is aborted, nothing "
                        f"further is decided")


def build_context(root: Path, approver: str, code_commit: str, index_dir: Path | None,
                  data_dir: Path, identities: IdentityRegistry, clock=None) -> GateContext:
    related = provenance = None
    if index_dir is not None:
        from rag4ds.index.build import SentenceTransformerEmbedder, load_index
        from rag4ds.retrieve.service import Retriever
        idx = load_index(index_dir)
        pin = idx.manifest["embedding_model"]
        retriever = Retriever(idx, SentenceTransformerEmbedder(pin["id"], pin["revision"]))
        related, provenance = make_related_finder(retriever), display_provenance_of(retriever)
    return GateContext(
        decisions=Journal(root / DECISION_JOURNAL),
        staging=StagingStore(root / "stage" / "candidates.jsonl"),
        runs=Journal(root / "runs" / "gate-sessions.jsonl"),
        approver=approver, code_commit=code_commit,
        source_resolver=make_source_resolver(Journal(root / "runs" / "ingest.jsonl"), data_dir),
        related_finder=related, display_provenance=provenance,
        allow_unchecked_approval=False, identities=identities,
        **({"clock": clock} if clock is not None else {}),
    )


def run(root: Path, sheet: Path, approver: str, code_commit: str, index_dir: Path | None,
        data_dir: Path, pack: str, transcript: Path | None, dry_run: bool,
        out=None, clock=None) -> int:
    """clock: a callable returning the NFR2 timestamp for the gate's records;
    the machine's JST clock by default (refused when not JST), injectable so
    the suite runs on a UTC runner."""
    out = out or sys.stdout
    root = Path(root)
    try:
        if clock is None and time.localtime().tm_gmtoff != 9 * 3600:
            raise SheetError("this machine's local clock is not JST (UTC+9); refusing")
        rows = load_sheet(sheet)
        sheet_sha = hashlib.sha256(sheet.read_bytes()).hexdigest()
        if transcript is not None:
            transcript.parent.mkdir(parents=True, exist_ok=True)
            refused = transcript_path_is_publishable(transcript)
            if refused:
                raise SheetError(refused)
        identities = IdentityRegistry.load(root / IDENTITIES_FILE)
        identities.resolve(approver)
        ctx = build_context(root, approver, code_commit, None if dry_run else index_dir,
                            data_dir, identities, clock)
        console = Console(record=True, markup=False, highlight=False)
        reader = SheetReader(rows, console)
        session = GateSession(ctx, console, reader, pack=pack)
        reader.session = session
        queue = session.queue()
        sheet_ids = [r["id"] for r in rows]
        if sheet_ids != queue:
            first = next((i for i, (a, b) in enumerate(zip(sheet_ids, queue, strict=False))
                          if a != b), min(len(sheet_ids), len(queue)))
            raise SheetError(f"sheet does not match the gate's queue for pack {pack!r}: sheet "
                             f"{len(sheet_ids)} rows, queue {len(queue)} candidates; first "
                             f"mismatch at row {first + 1} (sheet "
                             f"{sheet_ids[first] if first < len(sheet_ids) else '-'}, queue "
                             f"{queue[first] if first < len(queue) else '-'}); nothing decided")
        n_r = sum(1 for r in rows if r["key"] == "r")
        n_g = sum(1 for r in rows if r["key"] == "g")
        n_s = len(rows) - n_r - n_g
        print(f"gate_sheet: sheet {sheet} sha256 {sheet_sha[:16]}... - {len(rows)} rows match "
              f"the queue: {n_r} reject, {n_g} group, {n_s} skip; approver {approver!r}; "
              f"{'DRY RUN - no session' if dry_run else 'driving the gate'}", file=out)
        if dry_run:
            for r in rows:
                print(f"  {r['n']:3d} {r['id']} {r['key']} {r['reason']}", file=out)
            return 0
    except (SheetError, GateError, IdentityError, OSError) as exc:
        print(f"gate_sheet: REFUSED - {exc}", file=sys.stderr)
        return 1
    console.print(f"SHEET-DRIVEN SESSION: sheet {sheet.as_posix()} sha256 {sheet_sha}; the "
                  f"validator {approver!r} decided every row in writing; tools/gate_sheet.py "
                  f"at {code_commit} is the keyboard")
    rc = 0
    failed = None
    try:
        session.run()
    except Exception as exc:  # noqa: BLE001 - recorded in the manifest and reported (exit 1)
        failed = exc
    except BaseException as exc:
        # AR-rag4ds-033 F1: a KeyboardInterrupt or SystemExit is not an Exception -
        # the manifest must still say the session failed; recorded, then re-raised
        failed = exc
        raise
    finally:
        if transcript is not None:
            rc = write_transcript(console, transcript, session_failed=failed is not None)
        Journal(root / "runs" / "gate-sheets.jsonl").append({
            "kind": "gate-sheet", "schema": 1, "sheet": sheet.as_posix(), "sheet_sha256": sheet_sha,
            "approver": approver, "code_commit": code_commit, "pack": pack,
            "timestamp": (clock() if clock is not None else time.strftime("%Y-%m-%d %H:%M:%S JST")),
            "rows": len(rows),
            "reject_rows": n_r, "group_rows": n_g, "skip_rows": n_s, "rows_answered": reader.pos,
            "session_failed": failed is not None,
            "transcript": transcript.as_posix() if transcript else None,
        })
    if failed is not None:
        print(f"gate_sheet: session aborted after row {reader.pos}: {failed}", file=sys.stderr)
        return 1
    if reader.pos != len(rows):
        # AR-rag4ds-032 F1: a session that ended with rows unanswered is not a success
        print(f"gate_sheet: session ended with {reader.pos} of {len(rows)} rows answered - "
              f"NOT complete", file=sys.stderr)
        return 1
    print(f"gate_sheet: done - {reader.pos} rows answered; counts {session.summary.counts}",
          file=out)
    return rc


def main(argv: list[str] | None = None) -> int:
    use_utf8_console()  # AR-rag4ds-049 F4: before anything is rendered
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-root", default=str(MODULE_ROOT))
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--approver", required=True)
    ap.add_argument("--code-commit", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--transcript")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return run(Path(a.module_root), Path(a.sheet), a.approver, a.code_commit, Path(a.index_dir),
               Path(a.data_dir), a.pack, Path(a.transcript) if a.transcript else None, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
