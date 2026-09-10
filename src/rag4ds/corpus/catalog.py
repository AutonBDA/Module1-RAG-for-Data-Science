"""The corpus catalog and ingest (plan S3; component 1, NFR6, NFR7, FR7).

Git holds the catalog only: per document its content hash, license,
fetch pointer, redistribution permission, and metadata (NFR6). Raw
documents live outside git in a local data directory. A document is
never edited - new content is a new document under a new hash. Every
ingest writes a run manifest (hash, license, source, timestamp) to the
runs journal, so the audit trail starts at the document (FR7). The
catalog file is YAML - human-edited by convention (section 8) - written
deterministically here; the manifest is a canonical JSON Line (NFR2).
Reconstructability (S3 exit): catalog plus manifest locate the raw file
and re-derive its hash; a mismatch fails loudly.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

from rag4ds.auditlog.journal import Journal
from rag4ds.gate.decisions import _timestamp_valid
from rag4ds.model.entry import ENTRY_ID_RE, SHARED_PACK

CATALOG_SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"[0-9a-f]{64}")
DOCUMENT_FIELDS = ("hash", "license", "fetch_pointer", "redistributable", "metadata")


class CatalogError(ValueError):
    """Raised loudly for any catalog or ingest violation."""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_document(doc: dict) -> None:
    problems = []
    if not isinstance(doc, dict):
        raise CatalogError(f"a catalog document must be a mapping, got {type(doc).__name__}")
    unknown = sorted(set(doc) - set(DOCUMENT_FIELDS))
    missing = sorted(set(DOCUMENT_FIELDS) - set(doc))
    if unknown:
        problems.append(f"unknown field(s): {unknown}")
    if missing:
        problems.append(f"missing mandatory field(s): {missing}")
    if problems:
        raise CatalogError("catalog document invalid: " + "; ".join(problems))
    if not isinstance(doc["hash"], str) or not SHA256_RE.fullmatch(doc["hash"]):
        problems.append("hash must be a 64-hex sha256")
    if not isinstance(doc["license"], str) or not doc["license"].strip():
        problems.append("license is mandatory and must be a non-blank string (NFR6)")
    if not isinstance(doc["fetch_pointer"], str) or not doc["fetch_pointer"].strip():
        problems.append("fetch_pointer is mandatory and must be a non-blank string")
    if not isinstance(doc["redistributable"], bool):
        problems.append("redistributable must be a boolean (NFR6 vendoring permission)")
    if not isinstance(doc["metadata"], dict):
        problems.append("metadata must be a mapping")
    if problems:
        raise CatalogError(
            f"catalog document invalid ({len(problems)} problem(s)): " + "; ".join(problems)
        )


class Catalog:
    """One pack's corpus catalog: packs/<pack>/corpus/catalog.yaml."""

    def __init__(self, pack: str, pack_root: Path):
        if not isinstance(pack, str) or not ENTRY_ID_RE.fullmatch(pack) or pack == SHARED_PACK:
            raise CatalogError(f"pack {pack!r} must be a pack code (never {SHARED_PACK!r})")
        self.pack = pack
        self.path = Path(pack_root) / "corpus" / "catalog.yaml"

    def load(self) -> dict:
        if not self.path.exists():
            return {"schema": CATALOG_SCHEMA_VERSION, "pack": self.pack, "documents": []}
        try:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise CatalogError(f"{self.path} is not readable YAML: {exc}") from exc
        if (
            not isinstance(data, dict)
            or data.get("schema") != CATALOG_SCHEMA_VERSION
            or data.get("pack") != self.pack
            or not isinstance(data.get("documents"), list)
        ):
            raise CatalogError(
                f"{self.path} is not a schema-{CATALOG_SCHEMA_VERSION} catalog "
                f"for pack {self.pack!r}"
            )
        seen = set()
        for doc in data["documents"]:
            validate_document(doc)
            if doc["hash"] in seen:
                raise CatalogError(f"{self.path} lists hash {doc['hash']} twice")
            seen.add(doc["hash"])
        return data

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(data, sort_keys=True, allow_unicode=False, default_flow_style=False)
        self.path.write_text(text, encoding="utf-8", newline="\n")

    def register(self, doc: dict) -> None:
        """Add one document record. Registering a hash already present is an
        error: documents are immutable and never re-registered."""
        validate_document(doc)
        data = self.load()
        if any(d["hash"] == doc["hash"] for d in data["documents"]):
            raise CatalogError(
                f"document {doc['hash']} is already registered in pack {self.pack!r}"
            )
        data["documents"].append(dict(doc))
        self._save(data)

    def get(self, content_hash: str) -> dict:
        for d in self.load()["documents"]:
            if d["hash"] == content_hash:
                return d
        raise CatalogError(f"document {content_hash} is not in the {self.pack!r} catalog")


def ingest(
    catalog: Catalog,
    raw_path: Path,
    data_dir: Path,
    license_: str,
    fetch_pointer: str,
    redistributable: bool,
    metadata: dict,
    timestamp: str,
    runs_journal: Journal,
    code_commit: str,
    config: dict,
    tool: str = "rag4ds.corpus.ingest",
) -> dict:
    """Content-hash a raw document that lives under data_dir (outside git),
    write the ingest manifest, then register the document. Returns the
    manifest. The manifest carries the code commit and the run
    configuration (NFR2, AR-rag4ds-014 F4). Order (F5): the pre-checks
    (document validity, not already registered) run first, the manifest
    is appended, and only then is the catalog written - so a failure
    never leaves a catalogued document without its manifest; a failure
    after the manifest leaves a manifest whose document is not yet
    catalogued, and a retry appends a second manifest and registers
    (append logs tolerate that; the catalog never holds an unmanifested
    document)."""
    raw_path, data_dir = Path(raw_path), Path(data_dir)
    if not raw_path.is_file():
        raise CatalogError(f"raw document {raw_path} does not exist")
    try:
        raw_path.resolve().relative_to(data_dir.resolve())
    except ValueError as exc:
        raise CatalogError(
            f"raw document {raw_path} is outside the data directory {data_dir} (NFR6)"
        ) from exc
    if not _timestamp_valid(timestamp):
        raise CatalogError(f"timestamp {timestamp!r} must be 'YYYY-MM-DD HH:MM[:SS] JST'")
    if not isinstance(code_commit, str) or not re.fullmatch(r"[0-9a-f]{7,40}", code_commit):
        raise CatalogError("code_commit must be a 7-40 hex git commit id (NFR2)")
    if not isinstance(config, dict):
        raise CatalogError("config must be a mapping (NFR2)")
    content_hash = sha256_of(raw_path)
    doc = {
        "hash": content_hash,
        "license": license_,
        "fetch_pointer": fetch_pointer,
        "redistributable": redistributable,
        "metadata": dict(metadata),
    }
    validate_document(doc)
    if any(d["hash"] == content_hash for d in catalog.load()["documents"]):
        raise CatalogError(
            f"document {content_hash} is already registered in pack {catalog.pack!r}"
        )
    manifest = {
        "kind": "ingest",
        "schema": CATALOG_SCHEMA_VERSION,
        "pack": catalog.pack,
        "hash": content_hash,
        "license": license_,
        "source": fetch_pointer,
        "local_path": raw_path.resolve().relative_to(data_dir.resolve()).as_posix(),
        "size_bytes": raw_path.stat().st_size,
        "timestamp": timestamp,
        "tool": tool,
        "code_commit": code_commit,
        "config": dict(config),
    }
    runs_journal.append(manifest)
    catalog.register(doc)
    return manifest


def reconstruct(catalog: Catalog, manifest: dict, data_dir: Path) -> Path:
    """The S3 exit property: catalog plus manifest locate the raw document
    and its hash re-derives exactly; anything else fails loudly."""
    # AR-rag4ds-015 O1: the runs journal is heterogeneous; a non-ingest or
    # malformed manifest must refuse with the module exception, never KeyError.
    if not isinstance(manifest, dict) or manifest.get("kind") != "ingest":
        raise CatalogError(
            f"reconstruct needs an ingest manifest, got kind={manifest.get('kind')!r}"
            if isinstance(manifest, dict) else "reconstruct needs a manifest object"
        )
    missing = [k for k in ("hash", "local_path", "source", "license") if k not in manifest]
    if missing:
        raise CatalogError(f"ingest manifest missing field(s): {missing}")
    doc = catalog.get(manifest["hash"])
    local = manifest["local_path"]
    if not isinstance(local, str) or Path(local).is_absolute() or ".." in Path(local).parts:
        raise CatalogError(
            f"manifest local_path {local!r} must be a relative path inside the data directory"
        )
    path = Path(data_dir) / local
    try:
        path.resolve().relative_to(Path(data_dir).resolve())
    except ValueError as exc:
        raise CatalogError(f"manifest local_path {local!r} escapes the data directory") from exc
    if not path.is_file():
        raise CatalogError(f"manifest points at {path}, which does not exist")
    actual = sha256_of(path)
    if actual != doc["hash"] or actual != manifest["hash"]:
        raise CatalogError(
            f"{path} hashes to {actual}, not the recorded {doc['hash']} - the "
            "document is not the one catalogued (documents are immutable)"
        )
    if doc["fetch_pointer"] != manifest["source"] or doc["license"] != manifest["license"]:
        raise CatalogError("catalog and manifest disagree on source or license")
    return path
