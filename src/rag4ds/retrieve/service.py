"""The retrieval library (plan S5; component 6; FR6, FR9, NFR9).

Query -> embed -> structural pre-filter (layer, domain, phase) -> exact
cosine over the filtered set -> relevance floor -> top_k. The FR6 rules:
the domain filter narrows DOMAIN-layer entries to the named pack while
general- and task-layer entries stay eligible (they ground every
domain) and are excluded only by an explicit layer filter; an empty or
exhausted store answers NO_GROUNDING, never a silent empty list; a
failure answers ERROR, never NO_GROUNDING; NO_GROUNDING is answered only
after the store is exhausted - the structural filter is applied to the
WHOLE store (never to a shortlist), so a filtered-out shortlist can
never produce it. Pins are binding (component 7): a pin naming a version
the build does not serve is ERROR with the served versions in meta; a
domain filter naming a pack the build does not serve is ERROR the same
way (AR-rag4ds-018 F5) - silence is the one option NFR9 does not allow.

The CRISP-DM phase filter in v1 is lexical (declared): an entry is
eligible under a phase if its id or content names that phase as a whole
word or hyphenated slug, case-insensitive - never inside a longer word
("remodeling" is not "modeling", AR-rag4ds-018 F3). FR1 carries no phase
field; S8 seeds the general-layer phase vocabulary, and a structural
phase link is a later increment. The relevance floor (min_relevance) is
build configuration, recorded in the index manifest and echoed in meta;
its default value is [ASSUMED] until S9 sets it with the validator.
Scores are rounded to six decimals and the ranking is computed on the
rounded values (score desc, then id), so the order a response shows is
the order its own numbers explain (AR-rag4ds-018 O4).
"""

from __future__ import annotations

import re

import numpy as np

from rag4ds.index.build import Embedder, LoadedIndex
from rag4ds.retrieve.contract import (
    CONTRACT_SCHEMA_VERSION,
    STATUS_ERROR,
    STATUS_NO_GROUNDING,
    STATUS_OK,
    GroundedEntry,
    GroundingRequest,
    GroundingResponse,
)

SCORE_DECIMALS = 6


class RetrievalError(RuntimeError):
    """A backend failure surfaced as an ERROR response by Retriever.ground."""


def _phase_matches(phase: str, row: dict) -> bool:
    """Whole-word match of the phase (slug or spaced form) in id + content;
    hyphens and underscores are treated as spaces so 'data-preparation',
    'data preparation' and 'data_preparation' all name the phase, while
    'remodeling' does not name 'modeling'."""
    spaced = phase.lower().replace("-", " ")
    hay = re.sub(r"[-_]", " ", (row["id"] + " " + row["content"]).lower())
    return re.search(rf"(?<![a-z0-9]){re.escape(spaced)}(?![a-z0-9])", hay) is not None


def eligible_rows(index: LoadedIndex, req: GroundingRequest) -> list[int]:
    """The FR6 structural pre-filter over the WHOLE store: row indexes that
    satisfy layer, domain (pack) and phase. Domain narrows domain-layer
    entries to the named pack; shared layers stay unless a layer filter
    excludes them."""
    out = []
    for r in index.rows:
        if req.layer is not None and r["layer"] != req.layer:
            continue
        if req.domain is not None and r["layer"] == "domain" and r["pack"] != req.domain:
            continue
        if req.phase is not None and not _phase_matches(req.phase, r):
            continue
        out.append(r["row"])
    return out


class Retriever:
    """The grounding service as a library (FR10); S12's HTTP endpoint
    wraps this same object."""

    def __init__(self, index: LoadedIndex, embedder: Embedder):
        self.index = index
        self.embedder = embedder
        pin = index.manifest["embedding_model"]
        if (embedder.model_id, embedder.revision) != (pin["id"], pin["revision"]):
            raise RetrievalError(
                f"embedder {embedder.model_id}@{embedder.revision} is not the index's pinned "
                f"model {pin['id']}@{pin['revision']} (NFR2)"
            )

    def _meta(self, req: GroundingRequest, **extra) -> dict:
        m = self.index.manifest
        meta = {
            "contract_schema_version": CONTRACT_SCHEMA_VERSION,
            "index_version": m["index_version"],
            "index_schema": m["schema"],
            "index_build": {"code_commit": m["code_commit"], "timestamp": m["timestamp"]},
            "embedding_model": dict(m["embedding_model"]),
            "pack_versions": dict(m["pack_versions"]),
            "min_relevance": m["min_relevance"],
            "entry_count": m["entry_count"],
            "filters": req.filters(),
        }
        meta.update(extra)
        return meta

    def ground(self, req: GroundingRequest) -> GroundingResponse:
        if not isinstance(req, GroundingRequest):
            raise TypeError("ground() takes a validated GroundingRequest")
        try:
            served = self.index.manifest["pack_versions"]
            if req.domain is not None and req.domain not in served:
                return GroundingResponse(
                    STATUS_ERROR,
                    meta=self._meta(
                        req,
                        error=f"domain {req.domain!r} is not a pack this build serves; served "
                        f"packs {sorted(served)} (NFR9)",
                    ),
                )
            if req.pins:
                bad = {p: v for p, v in req.pins.items() if served.get(p) != v}
                if bad:
                    return GroundingResponse(
                        STATUS_ERROR,
                        meta=self._meta(
                            req,
                            error=f"pin(s) not served by this build: {bad}; served {served} "
                            "(replaying against other versions is a rebuild, NFR9)",
                        ),
                    )
            rows = eligible_rows(self.index, req)
            if not rows:
                return GroundingResponse(
                    STATUS_NO_GROUNDING,
                    meta=self._meta(req, candidates_considered=0,
                                    reason="store exhausted under the filters: no eligible entry"),
                )
            q = self.embedder.embed([req.query])
            q = np.asarray(q, dtype=np.float32)
            if q.shape != (1, self.index.vectors.shape[1]):
                raise RetrievalError(
                    f"query embedding shape {q.shape} does not match the index dimension "
                    f"{self.index.vectors.shape[1]}"
                )
            qn = float(np.linalg.norm(q[0]))
            if qn == 0:
                raise RetrievalError("query embedding has zero norm")
            qv = q[0] / qn
            idx = np.asarray(rows, dtype=np.int64)
            scores = self.index.vectors[idx] @ qv
            floor = float(self.index.manifest["min_relevance"])
            # the floor is compared against the ROUNDED score the response shows (O13)
            ranked = sorted(
                ((round(float(s), SCORE_DECIMALS), int(i))
                 for s, i in zip(scores, idx, strict=True)
                 if round(float(s), SCORE_DECIMALS) >= floor),
                key=lambda t: (-t[0], self.index.rows[t[1]]["id"]),
            )[: req.top_k]
            if not ranked:
                return GroundingResponse(
                    STATUS_NO_GROUNDING,
                    meta=self._meta(req, candidates_considered=len(rows),
                                    reason="store exhausted under the filters: nothing at or "
                                           f"above the relevance floor {floor}"),
                )
            entries = tuple(
                GroundedEntry(
                    id=self.index.rows[i]["id"], version=self.index.rows[i]["version"],
                    type=self.index.rows[i]["type"], layer=self.index.rows[i]["layer"],
                    pack=self.index.rows[i]["pack"], content=self.index.rows[i]["content"],
                    relevance_score=s, source=self.index.rows[i]["source_reference"],
                    status=self.index.rows[i]["status"],
                )
                for s, i in ranked
            )
            return GroundingResponse(
                STATUS_OK, entries=entries, meta=self._meta(req, candidates_considered=len(rows))
            )
        except Exception as exc:  # the service failed: ERROR, never NO_GROUNDING (FR6)
            return GroundingResponse(
                STATUS_ERROR, meta=self._meta(req, error=f"{type(exc).__name__}: {exc}")
            )


__all__ = ["Retriever", "RetrievalError", "eligible_rows", "SCORE_DECIMALS"]
