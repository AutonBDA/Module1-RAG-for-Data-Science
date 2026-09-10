"""Compile and fold (plan S4; 6.5, 9.1 tree-path rule, FR5, FR8, NFR1).

The fold applies each journal decision in order by its type: approvals -
supersessions included - write the entry's tree file (and remove the
previous version's file when a supersession moves the entry to another
pack or layer, so the tree holds each entry's CURRENT version only -
section 8); mapping-file approvals write the embedded artifact under the
pack's mappings/; retractions remove the entry's file; rejections of
either kind change nothing. The tree-path rule, total over gated
artifacts: an entry whose pack is the reserved value shared serializes
under shared/general/ or shared/tasks/ by its layer; every other entry
under packs/<pack>/entries/; an approved mapping artifact under
packs/<pack>/mappings/<standard-id>.json. Compile IS the fold applied to
the live tree; the reconstruction test folds the journal from empty into
a scratch tree and demands byte-equality with the live tree - any file
that differs, is missing, or is extra fails loudly (I1, NFR1).

The fold validates what it replays and where it writes. Every record is
shape-checked before it is read (FoldError with the record index, never
a bare KeyError); every computed tree path is checked to lie inside the
root (pack and standard ids obey the id charset, and the resolved path is
contained - AR-rag4ds-016 F1); and the chain is validated (the log-side
twin of the gate's known_versions / retired check): a fresh approval's id
must be unknown and not retired, at version 1; a supersession must target
the current version and be version+1; a retraction must name the current
version. A retracted id is RETIRED for the life of the log - never reused
(section 8 KB-wide uniqueness; AR-rag4ds-016 F4) - and FoldResult carries
both maps for the gate. A log that violates any of this never folds
silently.

Journal files: this fold reads the gate-decision journal
(journal/decisions.jsonl). Pack lifecycle events (component 8) are a
different record family and live in their own file under journal/; a
non-decision record in the decision journal is refused, not skipped
(AR-rag4ds-016 O3).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rag4ds.auditlog.journal import Journal
from rag4ds.gate.decisions import APPROVAL_TYPES, DECISION_TYPES
from rag4ds.model.entry import (
    ENTRY_ID_RE,
    SHARED_PACK,
    EntryValidationError,
    canonical_bytes,
    entry_filename,
    entry_from_dict,
)

GATED_SHARED = (Path("shared") / "general", Path("shared") / "tasks")
GATED_PACK_SUBDIRS = ("entries", "mappings")
DECISION_JOURNAL = Path("journal") / "decisions.jsonl"


class FoldError(ValueError):
    """Raised loudly for any fold violation, naming the record index."""


@dataclass(frozen=True)
class FoldResult:
    """The fold's final view of the log: id -> current version for live
    entries, and the ids retired by retraction (reserved forever). Both
    feed the gate (approve_entry's known_versions and retired)."""

    versions: dict
    retired: frozenset


def gated_dirs(root: Path) -> list[Path]:
    """The directories the fold owns, relative to the tree root: shared/general,
    shared/tasks, and every pack's entries/ and mappings/. Nothing else under
    packs/<code>/ (manifest, corpus/, validation/, releases/, demo/ - 4.1) is
    touched by compile or compared by the reconstruction test."""
    out = list(GATED_SHARED)
    packs = root / "packs"
    if packs.is_dir():
        for pack in sorted(packs.iterdir()):
            if pack.is_dir():
                out.extend(Path("packs") / pack.name / sub for sub in GATED_PACK_SUBDIRS)
    return out


def _pack_code(pack, where: str) -> str:
    if not isinstance(pack, str) or not ENTRY_ID_RE.fullmatch(pack) or pack == SHARED_PACK:
        raise FoldError(f"{where}: pack {pack!r} must be a pack code (id charset, never shared)")
    return pack


def entry_tree_path(entry_dict: dict) -> Path:
    """The 9.1 tree-path rule for entries (relative to the tree root).
    Validates the entry (EntryValidationError -> FoldError), which pins
    the pack to the id charset - the path can carry no separators."""
    try:
        entry = entry_from_dict(entry_dict)
    except EntryValidationError as exc:
        raise FoldError(f"entry is not a valid FR1 entry: {exc}") from exc
    name = entry_filename(entry)
    if entry.pack == SHARED_PACK:
        if entry.layer == "general":
            return Path("shared") / "general" / name
        if entry.layer == "task":
            return Path("shared") / "tasks" / name
        # Unreachable by construction (Entry forbids shared with a domain
        # layer); kept as defence, like the retraction missing-file branch.
        raise FoldError(f"shared pack with layer {entry.layer!r} has no tree home")
    return Path("packs") / _pack_code(entry.pack, "entry") / "entries" / name


def mapping_tree_path(record: dict) -> Path:
    """packs/<pack>/mappings/<standard-id>.json; pack and standard obey the
    id charset (checked here independently of the gate)."""
    pack = _pack_code(record.get("pack"), "mapping approval")
    standard = record["mapping_artifact"].get("standard")
    if not isinstance(standard, str) or not ENTRY_ID_RE.fullmatch(standard):
        raise FoldError(f"mapping approval: standard {standard!r} must obey the id charset")
    return Path("packs") / pack / "mappings" / (standard + ".json")


def mapping_artifact_bytes(artifact: dict) -> bytes:
    """Canonical JSON for a gated mapping artifact: the same rules as an
    entry (sorted keys, compact separators, ASCII-escaped, one LF)."""
    text = json.dumps(artifact, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return (text + "\n").encode("ascii")


def _contained(root: Path, rel: Path, where: str) -> Path:
    """Defence in depth behind the charset checks: the computed path must be
    relative, dot-free, and resolve inside root (AR-rag4ds-016 F1)."""
    if rel.is_absolute() or ".." in rel.parts or "." in rel.parts:
        raise FoldError(f"{where}: tree path {rel.as_posix()!r} is not a plain relative path")
    full = (root / rel).resolve()
    try:
        full.relative_to(root.resolve())
    except ValueError as exc:
        raise FoldError(f"{where}: tree path {rel.as_posix()!r} escapes the tree root") from exc
    return root / rel


def _write(root: Path, rel: Path, data: bytes, where: str) -> None:
    path = _contained(root, rel, where)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _shape(i: int, rec) -> str:
    """Shape-check one record before anything subscripts it (AR-rag4ds-016
    F7): returns the decision type. Every violation is a FoldError naming
    the record index."""
    where = f"record {i}"
    if not isinstance(rec, dict):
        raise FoldError(f"{where}: a decision record must be an object, got {type(rec).__name__}")
    kind = rec.get("type")
    if kind not in DECISION_TYPES:
        raise FoldError(f"{where}: unknown decision type {kind!r}")
    if kind in APPROVAL_TYPES:
        has_entry, has_map = "entry" in rec, "mapping_artifact" in rec
        if has_entry == has_map:
            raise FoldError(f"{where}: an approval carries exactly one of entry, mapping_artifact")
        if has_entry:
            e = rec["entry"]
            if not isinstance(e, dict):
                raise FoldError(f"{where}: entry must be an object, got {type(e).__name__}")
            if not isinstance(e.get("id"), str) or not _is_int(e.get("version")):
                raise FoldError(f"{where}: entry needs a string id and an integer version")
            if "supersedes" in rec:
                s = rec["supersedes"]
                if (
                    not isinstance(s, dict)
                    or set(s) != {"id", "version"}
                    or not isinstance(s["id"], str)
                    or not _is_int(s["version"])
                ):
                    raise FoldError(f"{where}: supersedes must be {{'id': str, 'version': int}}")
                if s["id"].lower() != e["id"].lower():
                    raise FoldError(f"{where}: supersedes id {s['id']!r} is not the entry id")
        else:
            if not isinstance(rec["mapping_artifact"], dict):
                raise FoldError(f"{where}: mapping_artifact must be an object")
    elif kind == "retract":
        if not isinstance(rec.get("target_id"), str) or not _is_int(rec.get("target_version")):
            raise FoldError(
                f"{where}: retract needs a string target_id and an integer target_version"
            )
    return kind


def fold(records: list, root: Path) -> FoldResult:
    """Apply the decisions in order into root (which must be an empty or
    non-existent directory - the fold starts from empty). Returns the
    FoldResult (current versions + retired ids). Fails loudly on any shape,
    path, or chain violation."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FoldError(f"fold target {root} is not empty; the fold starts from empty")
    root.mkdir(parents=True, exist_ok=True)
    versions: dict[str, int] = {}
    paths: dict[str, Path] = {}
    retired: set[str] = set()
    for i, rec in enumerate(records, 1):
        kind = _shape(i, rec)
        where = f"record {i}"
        if kind in APPROVAL_TYPES:
            if "entry" in rec:
                e = rec["entry"]
                key = e["id"].lower()
                if key in retired:
                    raise FoldError(
                        f"{where}: id {e['id']!r} was retracted and stays reserved "
                        "(section 8 uniqueness)"
                    )
                if "supersedes" in rec:
                    cur = versions.get(key)
                    sv = rec["supersedes"]["version"]
                    if cur is None or sv != cur or e["version"] != cur + 1:
                        raise FoldError(
                            f"{where}: supersession of {e['id']!r} v{sv} does not match "
                            f"the chain (current {cur}) - the log is inconsistent"
                        )
                else:
                    if key in versions:
                        raise FoldError(
                            f"{where}: fresh approval of an already-known id {e['id']!r}"
                        )
                    if e["version"] != 1:
                        raise FoldError(f"{where}: fresh approval at version {e['version']}")
                try:
                    rel = entry_tree_path(e)
                except FoldError as exc:
                    raise FoldError(f"{where}: {exc}") from exc
                old = paths.get(key)
                if old is not None and old != rel:
                    # AR-rag4ds-016 F2: the supersession moved the entry
                    # (pack or layer changed) - the tree holds the CURRENT
                    # version only, so the previous file leaves.
                    old_path = _contained(root, old, where)  # AR-rag4ds-016 O13
                    if not old_path.is_file():  # unreachable by construction; defence
                        raise FoldError(f"{where}: previous file {old.as_posix()} is missing")
                    old_path.unlink()
                _write(root, rel, canonical_bytes(entry_from_dict(e)), where)
                versions[key] = e["version"]
                paths[key] = rel
            else:
                try:
                    rel = mapping_tree_path(rec)
                except FoldError as exc:
                    raise FoldError(f"{where}: {exc}") from exc
                _write(root, rel, mapping_artifact_bytes(rec["mapping_artifact"]), where)
        elif kind == "retract":
            key = rec["target_id"].lower()
            cur = versions.get(key)
            if cur is None or rec["target_version"] != cur:
                raise FoldError(
                    f"{where}: retraction of {rec['target_id']!r} v{rec['target_version']} "
                    f"does not match the chain (current {cur})"
                )
            target = _contained(root, paths[key], where)  # AR-rag4ds-016 O13
            if not target.is_file():
                # Unreachable through the fold's own control flow (it owns
                # the directory from empty); kept as defence against a
                # tampered scratch dir and reached by injection in tests.
                raise FoldError(f"{where}: retraction target file {target} is missing")
            target.unlink()
            del versions[key]
            del paths[key]
            retired.add(key)
        # reject / reject-duplicate: nothing.
    return FoldResult(versions=versions, retired=frozenset(retired))


def compile_tree(journal: Journal, tree_root: Path) -> FoldResult:
    """Compile: fold the journal into a scratch dir, then replace the live
    tree's gated directories (shared/general, shared/tasks, every pack's
    entries/ and mappings/) with the result; a pack's ungated content
    (manifest, corpus catalog, validation, releases, demo) is never
    touched. The fold itself happens off-tree, so a failing fold leaves
    the live tree untouched; the final swap (remove then copy per gated
    dir) is not atomic - an interruption inside it leaves gated dirs
    removed and not yet restored, which the reconstruction test then
    reports as MISSING (AR-rag4ds-016 O4). Returns the FoldResult."""
    tree_root = Path(tree_root)
    records = journal.read_all()
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "fold"
        result = fold(records, scratch)
        for rel in gated_dirs(tree_root):  # clear every gated dir the live tree has
            live = tree_root / rel
            if live.exists():
                shutil.rmtree(live)
        for rel in gated_dirs(scratch):  # then install the fold's gated dirs
            src = scratch / rel
            if src.exists():
                shutil.copytree(src, tree_root / rel)
    return result


def _listing(root: Path) -> dict[str, bytes]:
    root = Path(root)
    out = {}
    for rel in gated_dirs(root):
        base = root / rel
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file():
                out[p.relative_to(root).as_posix()] = p.read_bytes()
    return out


def verify_reconstruction(journal: Journal, tree_root: Path) -> list[str]:
    """The NFR1 test: fold the journal from empty and compare with the live
    tree, byte for byte, over the gated directories. Returns the list of
    differences (empty means reconstructable); callers fail the build on
    any entry. A file named exactly .gitkeep (basename match - nothing
    else, AR-rag4ds-016 F5) is ignored on the tree side."""
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "fold"
        fold(journal.read_all(), scratch)
        want = _listing(scratch)
        have = {
            k: v for k, v in _listing(Path(tree_root)).items() if Path(k).name != ".gitkeep"
        }
    diffs = []
    for rel in sorted(set(want) | set(have)):
        if rel not in have:
            diffs.append(f"MISSING in tree: {rel}")
        elif rel not in want:
            diffs.append(f"EXTRA in tree (no decision produces it): {rel}")
        elif want[rel] != have[rel]:
            diffs.append(f"DIFFERS: {rel}")
    return diffs


def assert_reconstructable(journal: Journal, tree_root: Path) -> None:
    diffs = verify_reconstruction(journal, tree_root)
    if diffs:
        raise FoldError(
            f"the tree is NOT reconstructable from the journal ({len(diffs)} difference(s)): "
            + "; ".join(diffs)
        )


__all__ = [
    "DECISION_JOURNAL", "FoldError", "FoldResult", "fold", "compile_tree",
    "verify_reconstruction", "assert_reconstructable", "entry_tree_path",
    "mapping_tree_path", "mapping_artifact_bytes", "gated_dirs",
]
