"""The grounding contract, v1 (plan component 7; FR6, FR10, NFR2, NFR9).

GroundingRequest: query text, optional layer / domain (pack) / CRISP-DM
phase filters, top_k, and optional pack-version pins. GroundingResponse:
status (OK | NO_GROUNDING | ERROR), entries (id, version, type, layer,
pack, content, relevance_score, source, status), and meta (schema
version, index version and the index build's run fields, embedding-model
pin, pack versions served, the filters applied, and on ERROR the reason).
Scores state relevance to the query, never
confidence in an entry's truth - the field is named relevance_score and
no field is named confidence (FR6). The schema is versioned from day one
(CONTRACT_SCHEMA_VERSION); docs/grounding_contract_v1.md is the published
form. Requests are validated at construction and a malformed request is
a caller error (GroundingRequestError), not a service ERROR - ERROR is
reserved for the service failing on a well-formed request (FR6).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from types import MappingProxyType

from rag4ds.model.entry import ENTRY_ID_RE, LAYERS, SHARED_PACK
from rag4ds.model.entry import STATUSES as ENTRY_STATUSES

CONTRACT_SCHEMA_VERSION = 1
STATUS_OK = "OK"
STATUS_NO_GROUNDING = "NO_GROUNDING"
STATUS_ERROR = "ERROR"
STATUSES = (STATUS_OK, STATUS_NO_GROUNDING, STATUS_ERROR)

# The CRISP-DM phases, as slugs (general-layer vocabulary anchor, plan
# section 3 layer semantics).
CRISP_DM_PHASES = (
    "business-understanding",
    "data-understanding",
    "data-preparation",
    "modeling",
    "evaluation",
    "deployment",
)
MAX_TOP_K = 1000


class GroundingRequestError(ValueError):
    """A malformed request - the caller's error, raised before any lookup."""


@dataclass(frozen=True)
class GroundingRequest:
    query: str
    layer: str | None = None
    domain: str | None = None
    phase: str | None = None
    top_k: int = 5
    pins: dict | None = None

    def __post_init__(self) -> None:
        problems = []
        if not isinstance(self.query, str) or not self.query.strip():
            problems.append("query must be a non-blank string")
        if self.layer is not None and self.layer not in LAYERS:
            problems.append(f"layer {self.layer!r} not one of {LAYERS}")
        if self.domain is not None and (
            not isinstance(self.domain, str)
            or not ENTRY_ID_RE.fullmatch(self.domain)
            or self.domain == SHARED_PACK
        ):
            problems.append(f"domain {self.domain!r} must be a pack code (never {SHARED_PACK!r})")
        if self.phase is not None and self.phase not in CRISP_DM_PHASES:
            problems.append(f"phase {self.phase!r} not one of {CRISP_DM_PHASES}")
        if (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or not 1 <= self.top_k <= MAX_TOP_K
        ):
            problems.append(f"top_k must be an integer in 1..{MAX_TOP_K}, got {self.top_k!r}")
        if self.pins is not None:
            if not isinstance(self.pins, dict):
                problems.append("pins must be a mapping of pack code -> version string")
            else:
                for k, v in self.pins.items():
                    if (
                        not isinstance(k, str)
                        or not ENTRY_ID_RE.fullmatch(k)
                        or k == SHARED_PACK
                        or not isinstance(v, str)
                        or not v.strip()
                    ):
                        problems.append(f"pin {k!r}: {v!r} must be pack code -> non-blank version")
        if problems:
            raise GroundingRequestError(
                f"request invalid ({len(problems)} problem(s)): " + "; ".join(problems)
            )
        if self.pins is not None:
            # AR-rag4ds-018 O5/O12: hold our own read-only copy so neither the
            # caller's later mutation nor mutation through the request can
            # bypass the validation above.
            object.__setattr__(self, "pins", MappingProxyType(dict(self.pins)))

    def filters(self) -> dict:
        return {"layer": self.layer, "domain": self.domain, "phase": self.phase,
                "top_k": self.top_k, "pins": dict(self.pins) if self.pins else {}}


@dataclass(frozen=True)
class GroundedEntry:
    id: str
    version: int
    type: str
    layer: str
    pack: str
    content: str
    relevance_score: float
    source: str
    status: str  # component 6 names status among the results (F6); required and
    # validated (O10) - the served set is current-only (section 8) so it reads current today

    def __post_init__(self) -> None:
        if self.status not in ENTRY_STATUSES:
            raise ValueError(f"status {self.status!r} not one of {ENTRY_STATUSES}")

    def to_dict(self) -> dict:
        return {
            "id": self.id, "version": self.version, "type": self.type, "layer": self.layer,
            "pack": self.pack, "content": self.content,
            "relevance_score": self.relevance_score, "source": self.source,
            "status": self.status,
        }


@dataclass(frozen=True)
class GroundingResponse:
    status: str
    entries: tuple = ()
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status {self.status!r} not one of {STATUSES}")
        if self.status != STATUS_OK and self.entries:
            raise ValueError(f"a {self.status} response carries no entries")
        if self.status == STATUS_OK and not self.entries:
            raise ValueError("an OK response carries at least one entry (FR6)")
        if self.status == STATUS_ERROR and not self.meta.get("error"):
            raise ValueError("an ERROR response names its error in meta")

    def to_dict(self) -> dict:
        return {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "status": self.status,
            "entries": [e.to_dict() for e in self.entries],
            "meta": dict(self.meta),
        }

    def canonical_json(self) -> bytes:
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True)
        return (text + "\n").encode("ascii")


__all__ = [
    "CONTRACT_SCHEMA_VERSION", "CRISP_DM_PHASES", "MAX_TOP_K", "STATUSES", "STATUS_OK",
    "STATUS_NO_GROUNDING", "STATUS_ERROR", "GroundingRequest", "GroundingRequestError",
    "GroundedEntry", "GroundingResponse",
]
