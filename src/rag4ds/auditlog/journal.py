"""The append-only journal (plan S2; 6.5, section 8, NFR1).

Machine-written records are JSON Lines of canonical-JSON objects, one
record per line - the reconstruction fold reads exactly this, so the
format determinism the byte-equality test needs is the format's own
definition. The writer can only append (one atomic line per record,
flushed to disk before returning, per 6.4's crash contract); there is no
rewrite, truncate, or delete API. The reader refuses a line that is not
the canonical byte form of its own parse, and refuses a torn final line,
loudly (I1, NFR9). Append-only here is an API property, not a file
property: this module cannot detect a record deleted by an editor. The
catch for that is the fold's byte-equality of the journal against the
built knowledge base (rag4ds.kb.fold, live since S4 and asserted on the
repository's own journal and tree by the suite) - AR-rag4ds-013 O1.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class JournalError(ValueError):
    """Raised loudly for any journal integrity violation."""


def canonical_record_bytes(record: dict) -> bytes:
    """One canonical JSON Line: sorted keys, fixed separators,
    ASCII-escaped, LF terminator. The single byte form per record."""
    if not isinstance(record, dict):
        raise JournalError(f"a journal record must be a JSON object, got {type(record).__name__}")
    try:
        text = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError) as exc:
        raise JournalError(f"record is not canonically serializable: {exc}") from exc
    if "\n" in text or "\r" in text:
        raise JournalError("serialized record contains a line break - impossible for a JSON Line")
    return (text + "\n").encode("ascii")


class Journal:
    """Append-only JSON Lines journal over one file."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, record: dict) -> None:
        """Append one record atomically: a single canonical line, written
        in one os.write and fsynced before returning - a crash loses at
        most the in-flight record and corrupts nothing (6.4)."""
        line = canonical_record_bytes(record)
        creating = not self.path.exists()
        # O_BINARY: Windows os.open defaults to text mode and would translate
        # the LF terminator to CRLF - silent corruption the read-side
        # canonical check catches, so the writer must never produce it.
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
        fd = os.open(self.path, flags, 0o644)
        try:
            written = os.write(fd, line)
            # AR-rag4ds-012 F6: a short write must never let the session
            # believe the decision was recorded (6.6); the file may now
            # hold a torn record, which the reader refuses.
            if written != len(line):
                raise JournalError(
                    f"short write: {written} of {len(line)} bytes reached "
                    f"{self.path.name}; the decision is NOT recorded and the "
                    "journal tail may be torn"
                )
            os.fsync(fd)
        finally:
            os.close(fd)
        if creating and hasattr(os, "O_DIRECTORY"):
            # AR-rag4ds-013 O2: the first record creates the file; flushing
            # the parent directory makes the creation itself durable, so a
            # crash loses at most the in-flight record, never the file
            # (6.4). POSIX only - Windows has no directory fd semantics.
            dfd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)

    def read_all(self) -> list[dict]:
        """Read and validate every record. A line that is not byte-canonical,
        not an object, or torn (no trailing LF) fails the read loudly."""
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        if raw == b"":
            return []
        if not raw.endswith(b"\n"):
            raise JournalError(
                f"{self.path.name} ends in a torn record (no trailing newline); "
                "refusing to read a corrupt journal"
            )
        records = []
        for lineno, line in enumerate(raw.split(b"\n")[:-1], 1):
            try:
                record = json.loads(line.decode("ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise JournalError(
                    f"{self.path.name}:{lineno} is not readable as a canonical record: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise JournalError(f"{self.path.name}:{lineno} is not a JSON object")
            if line + b"\n" != canonical_record_bytes(record):
                raise JournalError(
                    f"{self.path.name}:{lineno} is not in canonical byte form; "
                    "refusing to read a non-canonical journal line (I1)"
                )
            records.append(record)
        return records
