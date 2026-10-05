"""The provenance chain of a gated entry version (FR7; plan section 3,
component 8 - readable by the future integrity layer; acceptance cases A1 and
A4).

For one (entry id, version) the chain is: the journal decision that approved
exactly that version; the approver, resolved in the identity registry; the
staged candidate the decision names, which must be APPROVED by that same
decision (its APPROVED transition cites the decision's journal line) and
whose proposed text the decision records as its candidate_text
(AR-rag4ds-057 F3); and the
catalogued corpus document that candidate cites by hash, in the catalog of
the entry's own pack (or the shared corpus, for the shared layers). The
supersession links come from the journal too: the version this decision
superseded, and the later decision that superseded this version, if any
(FR8 - a superseded version stays in the audit log, marked by the decision
that names it).

Every link is looked up, none is assumed: a link that does not resolve
raises ProvenanceError (NFR9). Judgements about the links - whether the
approver validates the entry's pack, whether a version is the one served -
are the caller's; this module only resolves and reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag4ds.auditlog.journal import Journal
from rag4ds.corpus.catalog import Catalog, CatalogError
from rag4ds.gate.identities import IDENTITIES_FILE, IdentityError, IdentityRegistry
from rag4ds.staging.store import StagingError, StagingStore

DECISION_JOURNAL = Path("journal") / "decisions.jsonl"
STAGING_FILE = Path("stage") / "candidates.jsonl"
SHARED_PACK = "shared"
APPROVING = ("approve", "edit-then-approve")


class ProvenanceError(ValueError):
    """A link of the chain does not resolve."""


@dataclass(frozen=True)
class Chain:
    entry_id: str
    version: int
    layer: str
    pack: str
    decision_line: int           # 1-based line of the approving decision in the journal
    decision_type: str
    decision_timestamp: str
    approver: str
    approver_id: str
    approver_roles: tuple[str, ...]
    approver_packs: tuple[str, ...]
    candidate_id: str
    candidate_state: str
    source_hash: str
    document: dict               # the catalog record of the cited document
    supersedes_version: int | None
    superseded_by_line: int | None


def approvals(records: list[dict]) -> dict[tuple[str, int], list[tuple[int, dict]]]:
    """(entry id, version) -> every approving decision for it, with its line."""
    out: dict[tuple[str, int], list[tuple[int, dict]]] = {}
    for line, rec in enumerate(records, 1):
        entry = rec.get("entry")
        if rec.get("type") in APPROVING and isinstance(entry, dict):
            out.setdefault((entry.get("id"), entry.get("version")), []).append((line, rec))
    return out


def _corpus_root(module_root: Path, pack: str) -> Path:
    return module_root / (SHARED_PACK if pack == SHARED_PACK else Path("packs") / pack)


def resolve(module_root: Path, entry_id: str, version: int, *, records: list[dict] | None = None,
            registry: IdentityRegistry | None = None,
            store: StagingStore | None = None) -> Chain:
    """Resolve the chain of one entry version under module_root. records,
    registry and store may be passed in to resolve many versions from one
    read; otherwise they are read from the module's own files."""
    module_root = Path(module_root)
    if records is None:
        records = Journal(module_root / DECISION_JOURNAL).read_all()
    found = approvals(records).get((entry_id, version), [])
    if len(found) != 1:
        raise ProvenanceError(f"{entry_id} v{version}: {len(found)} approving decisions on the "
                              f"journal, not exactly one")
    line, rec = found[0]
    entry = rec["entry"]
    pack, layer = entry.get("pack"), entry.get("layer")
    if not isinstance(pack, str) or not isinstance(layer, str):
        raise ProvenanceError(f"{entry_id} v{version}: the decision's entry names no pack or layer")
    try:
        registry = registry or IdentityRegistry.load(module_root / IDENTITIES_FILE)
        ident = registry.resolve(rec.get("approver", ""))
    except IdentityError as exc:
        raise ProvenanceError(f"{entry_id} v{version}: approver {rec.get('approver')!r} does not "
                              f"resolve: {exc}") from exc
    cid = rec.get("candidate_id")
    store = store or StagingStore(module_root / STAGING_FILE)
    filed = store.candidates()
    if not isinstance(cid, str) or cid not in filed:
        raise ProvenanceError(f"{entry_id} v{version}: candidate {cid!r} is not in staging")
    try:
        state = store.state_of(cid)
    except StagingError as exc:
        raise ProvenanceError(f"{entry_id} v{version}: {exc}") from exc
    if state != "APPROVED":
        raise ProvenanceError(f"{entry_id} v{version}: candidate {cid} is {state}, not APPROVED")
    # AR-rag4ds-057 F3: the candidate link is walked both ways, not taken on the
    # decision's word - staging records on the APPROVED transition the decision
    # that authorised it, and that must be this decision's own journal line
    own = f"{DECISION_JOURNAL.as_posix()}:{line}"
    authority = [r.get("authority") for r in store.journal.read_all()
                 if r.get("record") == "transition" and r.get("id") == cid
                 and r.get("to") == "APPROVED"]
    if authority != [own]:
        raise ProvenanceError(f"{entry_id} v{version}: candidate {cid} was approved by "
                              f"{authority or 'no decision'}, not by this decision ({own})")
    cand = filed[cid]["candidate"]
    if cand.get("kind") != "entry" or cand.get("pack") != pack or cand.get("layer") != layer:
        raise ProvenanceError(f"{entry_id} v{version}: candidate {cid} is not an entry candidate "
                              f"of pack {pack!r}, layer {layer!r}")
    # and the decision records the text it decided on: the candidate's own
    if (cand.get("proposed") or {}).get("content") != rec.get("candidate_text"):
        raise ProvenanceError(f"{entry_id} v{version}: the decision's candidate_text is not "
                              f"candidate {cid}'s proposed content")
    source_hash = cand.get("source_hash")
    try:
        document = Catalog(pack, _corpus_root(module_root, pack)).get(source_hash)
    except CatalogError as exc:
        raise ProvenanceError(f"{entry_id} v{version}: candidate {cid} cites a document that "
                              f"does not resolve: {exc}") from exc
    supersedes = rec.get("supersedes")
    sup_version = None
    if supersedes is not None:
        if (not isinstance(supersedes, dict) or supersedes.get("id") != entry_id
                or supersedes.get("version") != version - 1):
            raise ProvenanceError(f"{entry_id} v{version}: the decision supersedes {supersedes!r}, "
                                  f"not version {version - 1} of the same id")
        sup_version = supersedes["version"]
    elif version != 1:
        raise ProvenanceError(f"{entry_id} v{version}: version {version} names no superseded "
                              f"version")
    later = [i for i, r in enumerate(records, 1)
             if r.get("type") in APPROVING and r.get("supersedes") == {"id": entry_id,
                                                                       "version": version}]
    if len(later) > 1:
        raise ProvenanceError(f"{entry_id} v{version}: superseded by {len(later)} decisions")
    return Chain(entry_id=entry_id, version=version, layer=layer, pack=pack, decision_line=line,
                 decision_type=rec["type"], decision_timestamp=rec.get("timestamp", ""),
                 approver=rec["approver"], approver_id=ident.id,
                 approver_roles=tuple(ident.roles), approver_packs=tuple(ident.packs),
                 candidate_id=cid, candidate_state=state, source_hash=source_hash,
                 document=dict(document), supersedes_version=sup_version,
                 superseded_by_line=later[0] if later else None)
