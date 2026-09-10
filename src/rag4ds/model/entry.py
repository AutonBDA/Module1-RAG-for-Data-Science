"""The entry model and canonical serialization (plan S1; FR1, section 8).

An entry is one unit of approved knowledge. This module defines its
mandatory fields, the entry-id rule, and the canonical-JSON byte form -
the single byte representation per entry that the reconstruction test
(plan 6.5) depends on. An entry missing or violating any mandatory field
fails loudly (FR1); a stored file that is not byte-canonical fails loudly
on read (I1). Deliberately out of scope here: case-insensitive KB-wide id
UNIQUENESS, which is the gate's job at id assignment (FR5, S2).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Bumps only on FR1 / manifest field changes (plan 9.3, 4.1).
CORE_SCHEMA_VERSION = 1

# Section 8's entry-id rule: lowercase [a-z0-9-], 1 to 64 characters.
# Case-insensitive uniqueness across the knowledge base is the gate's job
# at id assignment (FR5, S2); the format rule lives here. fullmatch, never
# $-anchored: Python's $ matches before a trailing newline, which admitted
# an id with a trailing newline and a 65th character (AR-rag4ds-010 F1).
ENTRY_ID_RE = re.compile(r"[a-z0-9-]{1,64}")

ENTRY_TYPES = ("concept", "relation", "fact", "task-pattern")
LAYERS = ("general", "domain", "task")
STATUSES = ("current", "retracted")

# Reserved pack value for general- and task-layer entries (FR1, 4.1):
# never a pack code.
SHARED_PACK = "shared"

FIELDS = (
    "id",
    "type",
    "layer",
    "pack",
    "content",
    "source_reference",
    "version",
    "status",
)


class EntryValidationError(ValueError):
    """Raised with the complete list of violations - never one at a time."""


@dataclass(frozen=True)
class Entry:
    id: str
    type: str
    layer: str
    pack: str
    content: str
    source_reference: str
    version: int
    status: str

    def __post_init__(self) -> None:
        problems = []
        if not isinstance(self.id, str) or not ENTRY_ID_RE.fullmatch(self.id):
            problems.append(
                f"id {self.id!r} violates the entry-id rule (lowercase [a-z0-9-], 1-64 chars)"
            )
        if self.type not in ENTRY_TYPES:
            problems.append(f"type {self.type!r} not one of {ENTRY_TYPES}")
        if self.layer not in LAYERS:
            problems.append(f"layer {self.layer!r} not one of {LAYERS}")
        if not isinstance(self.pack, str) or not ENTRY_ID_RE.fullmatch(self.pack):
            # AR-rag4ds-016 F1: the pack is also a tree directory name
            # (9.1), so it obeys the id charset - no separators, no '..',
            # no drive letters, no case variants across platforms.
            problems.append(
                f"pack {self.pack!r} must be a pack code or {SHARED_PACK!r} "
                "(lowercase [a-z0-9-], 1-64 chars - it names a tree directory, 9.1)"
            )
        elif self.layer == "domain" and self.pack == SHARED_PACK:
            problems.append(
                f"domain-layer entries need a pack code; {SHARED_PACK!r} is reserved (FR1)"
            )
        elif self.layer in ("general", "task") and self.pack != SHARED_PACK:
            problems.append(
                f"{self.layer}-layer entries must use the reserved pack {SHARED_PACK!r}, "
                f"got {self.pack!r} (FR1)"
            )
        if not isinstance(self.content, str) or not self.content.strip():
            problems.append("content is mandatory and must be a non-blank string")
        if not isinstance(self.source_reference, str) or not self.source_reference.strip():
            problems.append("source_reference is mandatory and must be a non-blank string")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            problems.append(f"version must be an integer >= 1, got {self.version!r}")
        if self.status not in STATUSES:
            problems.append(f"status {self.status!r} not one of {STATUSES}")
        if problems:
            raise EntryValidationError(
                f"entry invalid ({len(problems)} problem(s)): " + "; ".join(problems)
            )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "layer": self.layer,
            "pack": self.pack,
            "content": self.content,
            "source_reference": self.source_reference,
            "version": self.version,
            "status": self.status,
        }


def entry_from_dict(data: dict) -> Entry:
    """Strict constructor: unknown or missing keys fail loudly by name."""
    unknown = sorted(set(data) - set(FIELDS))
    missing = sorted(set(FIELDS) - set(data))
    problems = []
    if unknown:
        problems.append(f"unknown field(s): {unknown}")
    if missing:
        problems.append(f"missing mandatory field(s): {missing}")
    if problems:
        raise EntryValidationError("entry invalid: " + "; ".join(problems))
    return Entry(**data)


def canonical_bytes(entry: Entry) -> bytes:
    """Section 8's canonical JSON: sorted keys, fixed separators, LF, one
    trailing newline, ASCII-escaped. The one byte form per entry (I1)."""
    text = json.dumps(
        entry.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return (text + "\n").encode("ascii")


def entry_filename(entry: Entry) -> str:
    return f"{entry.id}.json"


def write_entry(entry: Entry, directory: Path) -> Path:
    """Write the entry's canonical byte form to <id>.json under directory."""
    path = Path(directory) / entry_filename(entry)
    path.write_bytes(canonical_bytes(entry))
    return path


def read_entry(path: Path) -> Entry:
    """Read and validate an entry file. Fails loudly if the file's bytes
    are not the entry's canonical form, or if the filename does not match
    the id - a non-canonical tree file is a defect, never silently
    normalized (I1)."""
    path = Path(path)
    raw = path.read_bytes()
    try:
        data = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntryValidationError(
            f"{path.name} is not readable as canonical entry JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise EntryValidationError(
            f"{path.name} does not contain a JSON object"
        )
    entry = entry_from_dict(data)
    if raw != canonical_bytes(entry):
        raise EntryValidationError(
            f"{path.name} is not in canonical byte form; refusing to read a "
            "non-canonical tree file (I1)"
        )
    if path.name != entry_filename(entry):
        raise EntryValidationError(
            f"filename {path.name!r} does not match entry id {entry.id!r} "
            "(section 8: filename = <entry id>.json)"
        )
    return entry
