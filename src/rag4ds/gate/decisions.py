"""Gate decision records (plan S2; FR5, 6.4, 6.5, 4.5).

The five decision types, complete: approve, edit-then-approve, reject,
reject-duplicate (mechanical, citing its head decision as authority),
and retract. A supersession is not a separate type: it is an approval
whose embedded entry carries an existing id at version+1 and names what
it supersedes. Every decision records the approver, the timestamp, and
its type; the rest is conditional on the type (FR5). Approvals embed the
COMPLETE approved artifact - an FR1 entry with its id assigned at gate
time, or a canonical mapping artifact (4.5) - so the fold invents
nothing (6.5). Id assignment enforces the entry-id format and
case-insensitive uniqueness across the knowledge base (FR5, section 8),
AND the version chain (AR-rag4ds-013 F1): the gate is handed each known
id's CURRENT version, so a supersession must target exactly that version
and a retraction must name it - two conflicting supersessions of one
version, or a supersession of a version never recorded, are refused here,
at the gate, which is where this check is placed and recorded.
Every builder validates loudly and returns a record the journal accepts.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime

from rag4ds.model.entry import ENTRY_ID_RE, SHARED_PACK, Entry

DECISION_TYPES = (
    "approve",
    "edit-then-approve",
    "reject",
    "reject-duplicate",
    "retract",
)

APPROVAL_TYPES = ("approve", "edit-then-approve")

# The record schema version for journal decisions (bumps with FR5 changes).
DECISION_SCHEMA_VERSION = 1

TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}(:\d{2})? JST\Z")


def _timestamp_valid(ts) -> bool:
    """Shape AND calendar validity (AR-rag4ds-012 F5): the shape regex
    alone accepted 0000-99-99 99:99 JST."""
    if not isinstance(ts, str) or not TIMESTAMP_RE.fullmatch(ts):
        return False
    body = ts[: -len(" JST")]
    fmt = "%Y-%m-%d %H:%M:%S" if body.count(":") == 2 else "%Y-%m-%d %H:%M"
    try:
        datetime.strptime(body, fmt)
    except ValueError:
        return False
    return True


class DecisionError(ValueError):
    """Raised loudly with every violation named."""


def _base(kind: str, approver, timestamp) -> tuple[dict, list]:
    problems = []
    if not isinstance(approver, str) or not approver.strip():
        problems.append("approver is mandatory and must be a non-blank string")
    if not _timestamp_valid(timestamp):
        problems.append(
            f"timestamp {timestamp!r} must be a real calendar time "
            "'YYYY-MM-DD HH:MM[:SS] JST'"
        )
    return (
        {"schema": DECISION_SCHEMA_VERSION, "type": kind,
         "approver": approver, "timestamp": timestamp},
        problems,
    )


def _raise(problems: list) -> None:
    if problems:
        raise DecisionError(
            f"decision invalid ({len(problems)} problem(s)): " + "; ".join(problems)
        )


# The staging candidate id form (staging.store.candidate_id): the optional
# back-link from a decision to the candidate it decides (S6 gate CLI).
CANDIDATE_ID_RE = re.compile(r"cand-[0-9a-f]{16}")


def _check_candidate_id(candidate_id, problems) -> None:
    if candidate_id is not None and (
        not isinstance(candidate_id, str) or not CANDIDATE_ID_RE.fullmatch(candidate_id)
    ):
        problems.append(f"candidate_id {candidate_id!r} must be cand-<16 hex> (the staging id)")


# The gate's review context recorded ON the decision (AR-rag4ds-020 F5/F8/F10):
# what the conflict display showed when the expert decided, which shown ids
# the expert flagged as deliberate coexistence, whether the source span was
# rendered, and whether the quotation bound was confirmed. Closed key set.
REVIEW_FIELDS = {
    "conflict_display", "related_shown", "conflicts", "source_span", "quotation_bound_confirmed",
}
CONFLICT_DISPLAY_STATES = (
    "checked-clean", "checked-with-results", "lookup-failed", "no-finder", "not-applicable",
)
SOURCE_SPAN_STATES = ("rendered", "unavailable", "invalid", "not-applicable")


def _check_review(review, problems, candidate_id=None) -> None:
    if review is None:
        if candidate_id is not None:
            # AR-rag4ds-020 O10: a gate-written decision (it names its candidate)
            # must carry its review context - nothing below the gate may drop it.
            problems.append("a decision that names its candidate_id must carry a review context")
        return
    if not isinstance(review, dict) or set(review) != REVIEW_FIELDS:
        problems.append(f"review must be a mapping with exactly the keys {sorted(REVIEW_FIELDS)}")
        return
    if review["conflict_display"] not in CONFLICT_DISPLAY_STATES:
        problems.append(f"review.conflict_display not one of {CONFLICT_DISPLAY_STATES}")
    if review["source_span"] not in SOURCE_SPAN_STATES:
        problems.append(f"review.source_span not one of {SOURCE_SPAN_STATES}")
    for key in ("related_shown", "conflicts"):
        v = review[key]
        if not isinstance(v, list) or not all(
            isinstance(i, str) and ENTRY_ID_RE.fullmatch(i) for i in v
        ):
            problems.append(f"review.{key} must be a list of entry ids")
    if not isinstance(review["quotation_bound_confirmed"], bool):
        problems.append("review.quotation_bound_confirmed must be a boolean")
    if (
        isinstance(review.get("conflicts"), list)
        and isinstance(review.get("related_shown"), list)
        and not set(review["conflicts"]) <= set(review["related_shown"])
    ):
        problems.append("review.conflicts must be a subset of review.related_shown")


def _normalize_known(known_versions, problems) -> dict | None:
    """known_versions: the gate's view of the knowledge base - each known
    id mapped to its CURRENT version. AR-rag4ds-012 F2 (no bare strings)
    and AR-rag4ds-013 F1 (ids alone cannot validate the version chain)."""
    if not isinstance(known_versions, dict):
        problems.append(
            "known_versions must be a dict of id -> current version, "
            f"got {type(known_versions).__name__}"
        )
        return None
    normalized = {}
    for k, v in known_versions.items():
        if not isinstance(k, str) or not isinstance(v, int) or isinstance(v, bool) or v < 1:
            problems.append(
                f"known_versions entries must be str -> int >= 1, got {k!r}: {v!r}"
            )
            return None
        if k.lower() in normalized:
            problems.append(
                f"known_versions holds case-variant duplicates of {k.lower()!r}"
            )
            return None
        normalized[k.lower()] = v
    return normalized


def _normalize_retired(retired, problems) -> frozenset | None:
    """retired: ids whose entries were retracted. A retracted id stays
    reserved for the life of the log (section 8: ids unique across the
    whole knowledge base; AR-rag4ds-016 F4) - never reused, never
    superseded."""
    if (
        isinstance(retired, (str, bytes, Mapping))  # a mapping is known_versions' shape, O8
        or not hasattr(retired, "__iter__")
    ):
        problems.append(f"retired must be a set of ids, got {type(retired).__name__}")
        return None
    out = set()
    for r in retired:
        if not isinstance(r, str):
            problems.append(f"retired ids must be strings, got {r!r}")
            return None
        out.add(r.lower())
    return frozenset(out)


def _check_id_assignment(
    entry: Entry, known_versions, supersedes, problems, retired=frozenset()
) -> None:
    known = _normalize_known(known_versions, problems)
    gone = _normalize_retired(retired, problems)
    if known is None or gone is None:
        return
    if entry.id.lower() in gone:
        problems.append(
            f"id {entry.id!r} was retracted and stays reserved - never reused or "
            "superseded (section 8 uniqueness, AR-rag4ds-016 F4)"
        )
    if supersedes is None:
        if entry.id.lower() in known:
            problems.append(
                f"id {entry.id!r} is already taken (case-insensitive uniqueness, FR5)"
            )
        if entry.version != 1:
            problems.append(
                f"a fresh approval starts at version 1, got {entry.version}"
            )
    else:
        if not isinstance(supersedes, dict) or set(supersedes) != {"id", "version"}:
            problems.append("supersedes must be {'id': ..., 'version': ...}")
            return
        if supersedes["id"] != entry.id:
            problems.append(
                f"a supersession carries the EXISTING id: entry id {entry.id!r} "
                f"vs supersedes id {supersedes['id']!r} (FR5)"
            )
        current = known.get(entry.id.lower())
        if current is None:
            problems.append(
                f"supersession target id {entry.id!r} is not a known id"
            )
        elif (
            not isinstance(supersedes["version"], int)
            or isinstance(supersedes["version"], bool)
            or supersedes["version"] != current
        ):
            # AR-rag4ds-013 F1: the chain - only the CURRENT version can be
            # superseded, so two conflicting successors of one version, or
            # a successor of a version never recorded, are refused.
            problems.append(
                f"a supersession must target the CURRENT version of {entry.id!r}, "
                f"which is {current}; got supersedes version {supersedes['version']!r} "
                "(FR5/FR8 chain integrity)"
            )
        if (
            isinstance(supersedes["version"], int)
            and not isinstance(supersedes["version"], bool)
            and entry.version != supersedes["version"] + 1
        ):
            problems.append(
                f"a supersession is version+1: entry version {entry.version} "
                f"vs superseded version {supersedes['version']!r} (FR5)"
            )


def approve_entry(
    kind: str,
    approver: str,
    timestamp: str,
    candidate_text: str,
    source_reference: str,
    entry: Entry,
    known_versions: dict,
    supersedes: dict | None = None,
    conflict_with: str | None = None,
    retired=frozenset(),
    candidate_id: str | None = None,
    review: dict | None = None,
) -> dict:
    """approve / edit-then-approve of an entry. Embeds the complete FR1
    entry as it will serialize; a supersession names its target and is
    version+1 on the existing id (FR5). conflict_with records the 6.4
    conflict flag linking the two entries when the expert approves
    despite a shown conflict. retired: the ids retracted so far (the
    fold's FoldResult.retired) - reserved forever, refused for fresh
    approval and supersession alike (AR-rag4ds-016 F4)."""
    record, problems = _base(kind, approver, timestamp)
    if kind not in APPROVAL_TYPES:
        problems.append(f"kind {kind!r} not one of {APPROVAL_TYPES}")
    if not isinstance(candidate_text, str) or not candidate_text.strip():
        problems.append("candidate_text is mandatory and must be a non-blank string")
    if not isinstance(source_reference, str) or not source_reference.strip():
        problems.append("source_reference is mandatory and must be a non-blank string")
    if not isinstance(entry, Entry):
        problems.append("entry must be a validated Entry")
        _raise(problems)
    _check_id_assignment(entry, known_versions, supersedes, problems, retired)
    if entry.status != "current":
        problems.append(f"an approved entry is status current, got {entry.status!r}")
    if conflict_with is not None and (
        not isinstance(conflict_with, str) or not ENTRY_ID_RE.fullmatch(conflict_with)
    ):
        problems.append(f"conflict_with {conflict_with!r} must be a valid entry id")
    _check_candidate_id(candidate_id, problems)
    _check_review(review, problems, candidate_id)
    _raise(problems)
    record.update(
        candidate_text=candidate_text,
        source_reference=source_reference,
        entry=entry.to_dict(),
    )
    if candidate_id is not None:
        record["candidate_id"] = candidate_id
    if review is not None:
        record["review"] = {k: review[k] for k in sorted(review)}
    if supersedes is not None:
        record["supersedes"] = {"id": supersedes["id"], "version": supersedes["version"]}
    if conflict_with is not None:
        record["conflict_with"] = conflict_with
    return record


def approve_mapping(
    kind: str,
    approver: str,
    timestamp: str,
    candidate_content_hash: str,
    source_reference: str,
    mapping_artifact: dict,
    pack: str,
    candidate_id: str | None = None,
    review: dict | None = None,
) -> dict:
    """The FR5 payload variant: a mapping-file approval (4.5) embeds the
    canonical-JSON mapping artifact in place of an entry and carries the
    draft's content hash as its candidate reference in place of candidate
    text; the remaining fields are the same. The record also carries its
    PACK CODE (AR-rag4ds-012 F1): the 6.5 fold writes the artifact under
    packs/<pack>/mappings/<standard-id>.json, and nothing else in the
    record determines the directory half of that path."""
    record, problems = _base(kind, approver, timestamp)
    if kind not in APPROVAL_TYPES:
        problems.append(f"kind {kind!r} not one of {APPROVAL_TYPES}")
    if (
        not isinstance(pack, str)
        or not ENTRY_ID_RE.fullmatch(pack or "")
        or pack == SHARED_PACK
    ):
        problems.append(
            f"pack {pack!r} must be a pack code (id charset, never {SHARED_PACK!r}) - "
            "mappings belong to a pack (4.5)"
        )
    if not isinstance(candidate_content_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", candidate_content_hash or ""
    ):
        problems.append(
            "candidate_content_hash must be a 64-hex sha256 of the draft (4.5)"
        )
    if not isinstance(source_reference, str) or not source_reference.strip():
        problems.append("source_reference is mandatory and must be a non-blank string")
    if not isinstance(mapping_artifact, dict) or not mapping_artifact:
        problems.append("mapping_artifact must be a non-empty JSON object (4.5)")
    else:
        # AR-rag4ds-012 F8: the artifact's top level is closed, exactly as
        # an entry's field set is - unknown keys are journaled permanently
        # and the fold writes them, so they are refused by name ("same
        # gate as entries", 4.5).
        unknown = sorted(set(mapping_artifact) - {"standard", "rows"})
        if unknown:
            problems.append(f"mapping_artifact has unknown top-level key(s): {unknown}")
        std = mapping_artifact.get("standard")
        if not isinstance(std, str) or not ENTRY_ID_RE.fullmatch(std):
            problems.append(
                "mapping_artifact must record its standard id, obeying the "
                "entry-id character rule (4.5)"
            )
        # AR-rag4ds-012 F4: 4.5's row shape - each row maps one pack entry
        # to one external class with a relation from the closed set and a
        # note.
        rows = mapping_artifact.get("rows")
        if not isinstance(rows, list) or not rows:
            problems.append("mapping_artifact.rows must be a non-empty list (4.5)")
        else:
            for i, row in enumerate(rows):
                if not isinstance(row, dict) or set(row) != {
                    "entry", "class", "relation", "note",
                }:
                    problems.append(
                        f"rows[{i}] must be an object with exactly "
                        "entry/class/relation/note (4.5)"
                    )
                    continue
                if not isinstance(row["entry"], str) or not ENTRY_ID_RE.fullmatch(row["entry"]):
                    problems.append(f"rows[{i}].entry must be a valid entry id")
                if not isinstance(row["class"], str) or not row["class"].strip():
                    problems.append(f"rows[{i}].class must be a non-blank string")
                if row["relation"] not in ("equivalent", "narrower-than", "related"):
                    problems.append(
                        f"rows[{i}].relation {row['relation']!r} not one of "
                        "('equivalent', 'narrower-than', 'related') (4.5)"
                    )
                if not isinstance(row["note"], str):
                    problems.append(f"rows[{i}].note must be a string")
    _check_candidate_id(candidate_id, problems)
    _check_review(review, problems, candidate_id)
    _raise(problems)
    if candidate_id is not None:
        record["candidate_id"] = candidate_id
    if review is not None:
        record["review"] = {k: review[k] for k in sorted(review)}
    record.update(
        candidate_content_hash=candidate_content_hash,
        source_reference=source_reference,
        mapping_artifact=mapping_artifact,
        pack=pack,
    )
    return record


def reject(
    kind: str,
    approver: str,
    timestamp: str,
    candidate_text: str,
    source_reference: str,
    reason: str,
    head_decision: str | None = None,
    candidate_id: str | None = None,
    review: dict | None = None,
) -> dict:
    """reject / reject-duplicate: candidate, source reference, reason -
    no entry (FR5). reject-duplicate is mechanical and MUST cite the head
    decision as its authority; plain reject must not."""
    record, problems = _base(kind, approver, timestamp)
    if kind not in ("reject", "reject-duplicate"):
        problems.append(f"kind {kind!r} not one of ('reject', 'reject-duplicate')")
    if not isinstance(candidate_text, str) or not candidate_text.strip():
        problems.append("candidate_text is mandatory and must be a non-blank string")
    if not isinstance(source_reference, str) or not source_reference.strip():
        problems.append("source_reference is mandatory and must be a non-blank string")
    if not isinstance(reason, str) or not reason.strip():
        problems.append("reason is mandatory and must be a non-blank string")
    if kind == "reject-duplicate":
        if not isinstance(head_decision, str) or not head_decision.strip():
            problems.append(
                "reject-duplicate must cite its head decision as authority (FR5)"
            )
    elif head_decision is not None:
        problems.append("head_decision belongs only to reject-duplicate")
    _check_candidate_id(candidate_id, problems)
    _check_review(review, problems, candidate_id)
    _raise(problems)
    record.update(
        candidate_text=candidate_text, source_reference=source_reference, reason=reason
    )
    if kind == "reject-duplicate":
        record["head_decision"] = head_decision
    if candidate_id is not None:
        record["candidate_id"] = candidate_id
    if review is not None:
        record["review"] = {k: review[k] for k in sorted(review)}
    return record


def retract(
    approver: str,
    timestamp: str,
    target_id: str,
    target_version: int,
    final_form: Entry,
    known_versions: dict,
    review: dict | None = None,
) -> dict:
    """retract: carries the target entry id and version and embeds the
    entry's final form marked status retracted - the marking FR8 relies
    on - and directs the fold to remove the entry's tree file (FR5). The
    target must be a known id at its CURRENT version (AR-rag4ds-013 F1:
    chain integrity applies to retraction too)."""
    record, problems = _base("retract", approver, timestamp)
    if not isinstance(final_form, Entry):
        problems.append("final_form must be a validated Entry")
        _raise(problems)
    known = _normalize_known(known_versions, problems)
    if known is not None:
        current = known.get(str(target_id).lower()) if isinstance(target_id, str) else None
        if current is None:
            problems.append(f"retract target id {target_id!r} is not a known id")
        elif target_version != current:
            problems.append(
                f"retract must name the CURRENT version of {target_id!r}, which is "
                f"{current}; got {target_version!r} (FR8 chain integrity)"
            )
    if final_form.status != "retracted":
        problems.append(
            f"the embedded final form must be status retracted, got {final_form.status!r}"
        )
    if final_form.id != target_id:
        problems.append(
            f"final form id {final_form.id!r} does not match target id {target_id!r}"
        )
    if not isinstance(target_version, int) or isinstance(target_version, bool):
        problems.append(f"target_version must be an integer, got {target_version!r}")
    elif final_form.version != target_version:
        problems.append(
            f"final form version {final_form.version} does not match target "
            f"version {target_version!r}"
        )
    _check_review(review, problems)
    _raise(problems)
    record.update(
        target_id=target_id, target_version=target_version, entry=final_form.to_dict()
    )
    if review is not None:
        record["review"] = {k: review[k] for k in sorted(review)}
    return record
