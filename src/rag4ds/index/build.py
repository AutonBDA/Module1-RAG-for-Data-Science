"""Index build (plan S5; component 5, section 8, I1, NFR2, NFR9, FR3).

Two derived, never-authoritative indexes are built from the compiled
tree: a structural index (SQLite via the standard library: typed entries
with layer and pack, for the FR6 pre-filter) and a vector index (exact
brute-force cosine over a numpy matrix, embeddings of entry content). The
build reads ONLY the three kinds of gated entry directory of the tree
(shared/general, shared/tasks, packs/<code>/entries - the same set the
fold owns), so staged candidates and anything outside the tree are
unreachable by construction (FR4, FR5). Every file in those directories
must be a canonical, current entry whose layer and pack agree with its
directory, and packs/ may hold only pack-code directories; anything else
fails the build loudly. The build trusts the tree it is given to be the
gated tree: pass the decision journal and the build first asserts the
tree is reconstructable from it (6.5), refusing otherwise.

The build's pack set and versions are configuration handed in at build
time, never code (FR3): every pack DIRECTORY present in the tree must be
named in the configuration (presence is read from the filesystem, not
from harvested entries), and every configured pack must exist in the
tree. The index manifest (canonical JSON, build_dir/manifest.json) has
two parts. The reproducible CORE - schema, embedding model id and
revision, pack versions served, relevance floor, entry count, dimension,
tree_sha256 (over the canonical entry bytes in row order), and
vectors_sha256 (over the vector file's bytes) - is digested into
index_version, which retrieval echoes in meta (component 7); a rebuild of
the same tree by the same pinned model reproduces the core and
index_version exactly. The RUN fields - code_commit and timestamp (NFR2,
validated as catalog.py validates them) - sit outside the digest, so two
builds of one tree are distinguishable in the manifest while
index_version stays a pure function of what was indexed (AR-rag4ds-018
F2).

The build is atomic: everything is written into a temporary directory
beside the target and moved into place with one rename, so an
interrupted build leaves no half-written build directory (AR-rag4ds-018
F1). load_index verifies every part against the manifest - the manifest's
canonical form and index_version, each row's canonical sha256, the tree
hash recomputed from the rows, the vector file's digest, the shapes -
and refuses a missing or disagreeing part with IndexBuildError, so a
mixed-generation, altered, or truncated build never loads (NFR9).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

import numpy as np

from rag4ds.gate.decisions import _timestamp_valid
from rag4ds.kb.fold import GATED_PACK_SUBDIRS, GATED_SHARED, assert_reconstructable
from rag4ds.model.entry import (
    ENTRY_ID_RE,
    SHARED_PACK,
    Entry,
    EntryValidationError,
    canonical_bytes,
    entry_from_dict,
    read_entry,
)

INDEX_SCHEMA_VERSION = 1
STRUCTURAL_FILE = "structural.sqlite"
VECTORS_FILE = "vectors.npy"
MANIFEST_FILE = "manifest.json"
CORE_FIELDS = ("schema", "embedding_model", "pack_versions", "min_relevance", "entry_count",
               "dimension", "tree_sha256", "vectors_sha256")
RUN_FIELDS = ("code_commit", "timestamp")

# Name-keyed binding to the fold's gated directories (AR-rag4ds-018 O3):
# asserted against kb.fold's constants so a change there fails here loudly.
SHARED_LAYER_DIRS = {"general": Path("shared") / "general", "task": Path("shared") / "tasks"}
PACK_ENTRY_SUBDIR = "entries"
# A raise, not an assert: python -O strips asserts (AR-rag4ds-018 O11).
if (
    set(SHARED_LAYER_DIRS.values()) != set(GATED_SHARED)
    or PACK_ENTRY_SUBDIR not in GATED_PACK_SUBDIRS
):
    raise ImportError(
        "rag4ds.index.build: kb.fold's gated directory constants changed; rebind here"
    )


class IndexBuildError(ValueError):
    """Raised loudly for any build or load violation."""


class Embedder(Protocol):
    """What the build and the retriever need from an embedding model: its
    pinned identity and a function from texts to L2-normalized float32
    row vectors. The real implementation is SentenceTransformerEmbedder;
    tests use a clearly named fake that never reaches a result."""

    model_id: str
    revision: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


# The VT-3 pin (docs/vt3_pins.md): the embedding model and the exact Hub
# revision every index build, clustering and embedding strategy uses. Until
# S9c the pin lived only in the tests and the record; the tools import it
# from here and the suite asserts the copies agree.
PINNED_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PINNED_EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class SentenceTransformerEmbedder:
    """The pinned sentence-transformers model (VT-3: all-MiniLM-L6-v2 at
    its recorded revision). Loaded lazily; embeddings are L2-normalized so
    the dot product is the cosine."""

    def __init__(self, model_id: str, revision: str):
        if not isinstance(model_id, str) or not model_id.strip():
            raise IndexBuildError("model_id must be a non-blank string")
        if not isinstance(revision, str) or not revision.strip():
            raise IndexBuildError("revision must be a non-blank string (the pin, NFR2)")
        self.model_id, self.revision = model_id, revision
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, revision=self.revision, device="cpu")
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise IndexBuildError("embed() takes a list of strings")
        if not texts:
            return np.zeros((0, self._load().get_sentence_embedding_dimension()), dtype=np.float32)
        vecs = self._load().encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim != 2:
        raise IndexBuildError(f"embeddings must be a 2-D array, got shape {m.shape}")
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise IndexBuildError("an embedding has zero norm; cosine is undefined")
    return m / norms


def pack_dirs(tree_root: Path) -> list[str]:
    """The pack codes present in the tree, read from the filesystem: every
    child of packs/ must be a pack-code directory (AR-rag4ds-018 F4)."""
    packs = Path(tree_root) / "packs"
    if not packs.is_dir():
        return []
    codes = []
    for child in sorted(packs.iterdir()):
        if child.name == ".gitkeep" and child.is_file():
            continue
        if not child.is_dir():
            raise IndexBuildError(f"{child} is not a pack directory; packs/ holds packs only")
        if not ENTRY_ID_RE.fullmatch(child.name) or child.name == SHARED_PACK:
            raise IndexBuildError(f"packs/{child.name} is not a pack code directory")
        codes.append(child.name)
    return codes


def _entry_dirs(tree_root: Path) -> list[tuple[Path, str, str]]:
    """(dir, layer, pack) for every gated ENTRY directory of the tree.
    Mapping artifacts (packs/<code>/mappings) are gated content but not
    retrievable entries."""
    tree_root = Path(tree_root)
    out = [(tree_root / SHARED_LAYER_DIRS["general"], "general", SHARED_PACK),
           (tree_root / SHARED_LAYER_DIRS["task"], "task", SHARED_PACK)]
    for code in pack_dirs(tree_root):
        out.append((tree_root / "packs" / code / PACK_ENTRY_SUBDIR, "domain", code))
    return out


def tree_entries(tree_root: Path) -> list[Entry]:
    """Every current entry in the tree, in a deterministic order (layer,
    pack, id). A non-canonical file, a wrong directory, a non-entry file,
    or a non-current status fails the build."""
    tree_root = Path(tree_root)
    entries = []
    for d, layer, pack in _entry_dirs(tree_root):
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.name == ".gitkeep":
                continue
            if not p.is_file() or p.suffix != ".json":
                raise IndexBuildError(f"{p} is not an entry file (only <id>.json belongs here)")
            try:
                e = read_entry(p)
            except EntryValidationError as exc:
                raise IndexBuildError(f"{p}: {exc}") from exc
            if e.layer != layer or e.pack != pack:
                raise IndexBuildError(
                    f"{p}: entry layer/pack {e.layer}/{e.pack} disagree with its directory "
                    f"{layer}/{pack} (9.1 tree-path rule)"
                )
            if e.status != "current":
                raise IndexBuildError(f"{p}: the tree holds current entries only (section 8)")
            entries.append(e)
    entries.sort(key=lambda e: (e.layer, e.pack, e.id))
    ids = [e.id for e in entries]
    if len(set(ids)) != len(ids):
        raise IndexBuildError("duplicate entry id across the tree (section 8 uniqueness)")
    return entries


def _validate_pack_versions(pack_versions, tree_root: Path) -> dict:
    if not isinstance(pack_versions, dict):
        raise IndexBuildError("pack_versions must be a mapping of pack code -> version (FR3)")
    present = pack_dirs(tree_root)
    for k, v in pack_versions.items():
        if not isinstance(k, str) or not ENTRY_ID_RE.fullmatch(k) or k == SHARED_PACK:
            raise IndexBuildError(f"configured pack {k!r} is not a pack code")
        if not isinstance(v, str) or not v.strip():
            raise IndexBuildError(f"configured pack {k!r} needs a non-blank version string")
        if k not in present:
            raise IndexBuildError(f"configured pack {k!r} is not in the tree")
    unconfigured = [p for p in present if p not in pack_versions]
    if unconfigured:
        raise IndexBuildError(
            f"pack(s) present in the tree but not configured for this build: {unconfigured} "
            "(FR3: the active pack set is configuration)"
        )
    return {k: pack_versions[k] for k in sorted(pack_versions)}


def _canonical(obj) -> bytes:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return (text + "\n").encode("ascii")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def index_version_of(core: dict) -> str:
    return _digest(_canonical({k: core[k] for k in CORE_FIELDS}))[:16]


def _sqlite_ro_uri(resolved: Path) -> str:
    """A read-only sqlite URI for an already-resolved path: percent-encoded
    so '#', '%', '?' and spaces are data (F8); a UNC path (//server/share)
    takes sqlite's file:////server/share form (AR-rag4ds-018 O15)."""
    posix = Path(resolved).as_posix()
    quoted = quote(posix, safe="/:")
    if posix.startswith("//"):
        return "file:" + "//" + quoted + "?mode=ro"
    return "file:" + quoted + "?mode=ro"


def build_index(
    tree_root: Path,
    build_dir: Path,
    embedder: Embedder,
    pack_versions: dict,
    min_relevance: float,
    code_commit: str,
    timestamp: str,
    journal=None,
) -> dict:
    """Build both indexes from the tree into build_dir, atomically (a
    previous build at build_dir is replaced only once the new one is
    complete), and return the manifest. code_commit (7-40 hex) and
    timestamp ('YYYY-MM-DD HH:MM[:SS] JST') are the NFR2 run fields. If
    journal is given, the tree must first be reconstructable from it.
    Concurrent builds into one target are NOT supported (the scratch
    reaper assumes one builder per target - AR-rag4ds-018 O14)."""
    tree_root, build_dir = Path(tree_root), Path(build_dir)
    if not isinstance(min_relevance, (int, float)) or isinstance(min_relevance, bool):
        raise IndexBuildError("min_relevance must be a number in [-1, 1]")
    if not -1.0 <= float(min_relevance) <= 1.0:
        raise IndexBuildError("min_relevance must be a number in [-1, 1]")
    if not isinstance(code_commit, str) or not re.fullmatch(r"[0-9a-f]{7,40}", code_commit):
        raise IndexBuildError("code_commit must be a 7-40 hex git commit id (NFR2)")
    if not _timestamp_valid(timestamp):
        raise IndexBuildError(f"timestamp {timestamp!r} must be 'YYYY-MM-DD HH:MM[:SS] JST' (NFR2)")
    if journal is not None:
        assert_reconstructable(journal, tree_root)  # FoldError propagates: not the gated tree
    entries = tree_entries(tree_root)
    versions = _validate_pack_versions(pack_versions, tree_root)

    build_dir.parent.mkdir(parents=True, exist_ok=True)
    # AR-rag4ds-018 O8/F9: reap scratch directories a killed earlier build of
    # THIS target left - matched by literal prefix, never as a glob pattern.
    prefix = build_dir.name + ".building-"
    for stale in build_dir.parent.iterdir():
        if stale.name.startswith(prefix) and stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
    tmp = Path(tempfile.mkdtemp(prefix=build_dir.name + ".building-", dir=build_dir.parent))
    try:
        con = sqlite3.connect(tmp / STRUCTURAL_FILE)
        try:
            con.execute(
                "CREATE TABLE entries (row INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, "
                "version INTEGER NOT NULL, type TEXT NOT NULL, layer TEXT NOT NULL, "
                "pack TEXT NOT NULL, content TEXT NOT NULL, source_reference TEXT NOT NULL, "
                "status TEXT NOT NULL, sha256 TEXT NOT NULL)"
            )
            con.execute("CREATE INDEX entries_layer ON entries(layer)")
            con.execute("CREATE INDEX entries_pack ON entries(pack)")
            for i, e in enumerate(entries):
                con.execute(
                    "INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (i, e.id, e.version, e.type, e.layer, e.pack, e.content,
                     e.source_reference, e.status, _digest(canonical_bytes(e))),
                )
            con.commit()
        finally:
            con.close()

        if entries:
            vecs = _normalize_rows(embedder.embed([e.content for e in entries]))
            if vecs.shape[0] != len(entries):
                raise IndexBuildError(
                    f"embedder returned {vecs.shape[0]} rows for {len(entries)} entries"
                )
        else:
            vecs = np.zeros((0, 0), dtype=np.float32)
        np.save(tmp / VECTORS_FILE, vecs, allow_pickle=False)

        core = {
            "schema": INDEX_SCHEMA_VERSION,
            "embedding_model": {"id": embedder.model_id, "revision": embedder.revision},
            "pack_versions": versions,
            "min_relevance": float(min_relevance),
            "entry_count": len(entries),
            "dimension": int(vecs.shape[1]) if entries else 0,
            "tree_sha256": _digest(b"".join(canonical_bytes(e) for e in entries)),
            "vectors_sha256": _digest((tmp / VECTORS_FILE).read_bytes()),
        }
        manifest = dict(core, index_version=index_version_of(core),
                        code_commit=code_commit, timestamp=timestamp)
        (tmp / MANIFEST_FILE).write_bytes(_canonical(manifest))

        # atomic install: the finished directory replaces the old one in one rename
        if build_dir.exists():
            old = Path(tempfile.mkdtemp(prefix=build_dir.name + ".old-", dir=build_dir.parent))
            old.rmdir()
            os.rename(build_dir, old)
            try:
                os.rename(tmp, build_dir)
            except OSError:
                os.rename(old, build_dir)
                raise
            shutil.rmtree(old, ignore_errors=True)
        else:
            os.rename(tmp, build_dir)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return manifest


@dataclass(frozen=True)
class LoadedIndex:
    manifest: dict
    rows: tuple  # of dicts, in row order
    vectors: np.ndarray


def load_index(build_dir: Path) -> LoadedIndex:
    """Load a build, verifying every part against its manifest; refuses a
    missing, mixed-generation, altered, or truncated build (F1, F7)."""
    build_dir = Path(build_dir)
    for name in (MANIFEST_FILE, STRUCTURAL_FILE, VECTORS_FILE):
        if not (build_dir / name).is_file():
            raise IndexBuildError(f"index part {name} is missing from {build_dir}")
    raw = (build_dir / MANIFEST_FILE).read_bytes()
    try:
        manifest = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IndexBuildError(f"{MANIFEST_FILE} is not readable canonical JSON: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or _canonical(manifest) != raw
        or manifest.get("schema") != INDEX_SCHEMA_VERSION
        or set(manifest) != set(CORE_FIELDS) | set(RUN_FIELDS) | {"index_version"}
    ):
        raise IndexBuildError(
            f"{MANIFEST_FILE} is not a canonical schema-{INDEX_SCHEMA_VERSION} index manifest"
        )
    if index_version_of(manifest) != manifest["index_version"]:
        raise IndexBuildError("index_version does not match the manifest core (tampered?)")

    vec_bytes = (build_dir / VECTORS_FILE).read_bytes()
    if _digest(vec_bytes) != manifest["vectors_sha256"]:
        raise IndexBuildError("vectors.npy does not match the manifest's vectors_sha256")
    vectors = np.load(build_dir / VECTORS_FILE, allow_pickle=False)

    # Read-only by URI; the path is percent-encoded so '#', '%' or '?' in a
    # directory name cannot be parsed as URI syntax (AR-rag4ds-018 F8), and
    # connect() sits inside the try so no raw sqlite error escapes.
    uri = _sqlite_ro_uri((build_dir / STRUCTURAL_FILE).resolve())
    con = None
    try:
        con = sqlite3.connect(uri, uri=True)
        cur = con.execute(
            "SELECT row,id,version,type,layer,pack,content,source_reference,status,sha256 "
            "FROM entries ORDER BY row"
        )
        cols = [c[0] for c in cur.description]
        rows = tuple(dict(zip(cols, r, strict=True)) for r in cur.fetchall())
    except sqlite3.Error as exc:
        raise IndexBuildError(
            f"{STRUCTURAL_FILE} is not a readable structural index: {exc}"
        ) from exc
    finally:
        if con is not None:
            con.close()

    # AR-rag4ds-018 O9: the run fields are provenance echoed in every
    # response; validate them on load exactly as the build did.
    if not isinstance(manifest["code_commit"], str) or not re.fullmatch(
        r"[0-9a-f]{7,40}", manifest["code_commit"]
    ):
        raise IndexBuildError("manifest code_commit is not a 7-40 hex git commit id (NFR2)")
    if not _timestamp_valid(manifest["timestamp"]):
        raise IndexBuildError("manifest timestamp is not 'YYYY-MM-DD HH:MM[:SS] JST' (NFR2)")

    if len(rows) != manifest["entry_count"]:
        raise IndexBuildError(
            f"structural index holds {len(rows)} rows, manifest says {manifest['entry_count']}"
        )
    expected_shape = (len(rows), manifest["dimension"]) if rows else (0, 0)
    if tuple(vectors.shape) != expected_shape:
        raise IndexBuildError(f"vectors shape {vectors.shape} != manifest {expected_shape}")
    blob = b""
    for i, r in enumerate(rows):
        if r["row"] != i:
            raise IndexBuildError("structural rows are not contiguous in row order")
        fields = {k: r[k] for k in ("id", "type", "layer", "pack", "content",
                                   "source_reference", "version", "status")}
        try:
            cb = canonical_bytes(entry_from_dict(fields))
        except EntryValidationError as exc:
            raise IndexBuildError(f"row {i} ({r['id']!r}) is not a valid entry: {exc}") from exc
        if _digest(cb) != r["sha256"]:
            raise IndexBuildError(f"row {i} ({r['id']!r}) does not match its stored sha256")
        blob += cb
    if _digest(blob) != manifest["tree_sha256"]:
        raise IndexBuildError("structural rows do not match the manifest's tree_sha256")
    return LoadedIndex(manifest=manifest, rows=rows, vectors=vectors)


__all__ = [
    "INDEX_SCHEMA_VERSION", "CORE_FIELDS", "RUN_FIELDS", "Embedder", "SentenceTransformerEmbedder",
    "IndexBuildError", "LoadedIndex", "build_index", "load_index", "tree_entries", "pack_dirs",
    "index_version_of",
]
