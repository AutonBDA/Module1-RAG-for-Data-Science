"""The staging store (plan S3; component 3, FR4, 6.3, section 8).

Candidates await review here. A candidate, once filed, is immutable;
decisions reference it by id. Two kinds (FR4): an entry candidate
(proposed entry, layer, target pack, source document and span, extractor
identity and run manifest) and a mapping-draft candidate (the YAML draft
in place of a proposed entry, its target pack and external standard, its
author identity). The five states (6.3): PENDING, APPROVED, REJECTED,
REJECTED-DUPLICATE, GROUPED - GROUPED has two exits: when the head
receives ANY decision its members close mechanically as
REJECTED-DUPLICATE, each with a record citing the head's decision; or
the expert detaches a member before the head is decided, returning it
to PENDING. Rejected candidates remain in staging history forever.

Storage: one append-only JSON Lines journal per store; records are
"candidate" (the immutable filing) and "transition" (a state change
with its authority). State is the fold of transitions - nothing is ever
rewritten. Source spans are document hash plus character offsets, never
copied text (the quotation bound, component 2 / NFR5).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from rag4ds.auditlog.journal import Journal
from rag4ds.gate.decisions import _timestamp_valid
from rag4ds.model.entry import ENTRY_ID_RE, ENTRY_TYPES, LAYERS, SHARED_PACK

STAGING_SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"[0-9a-f]{64}")

STATES = ("PENDING", "APPROVED", "REJECTED", "REJECTED-DUPLICATE", "GROUPED")
TERMINAL = ("APPROVED", "REJECTED", "REJECTED-DUPLICATE")
ALLOWED = {
    ("PENDING", "APPROVED"),
    ("PENDING", "REJECTED"),
    ("PENDING", "REJECTED-DUPLICATE"),
    ("PENDING", "GROUPED"),
    ("GROUPED", "REJECTED-DUPLICATE"),  # head decided: mechanical close
    ("GROUPED", "PENDING"),  # detached before the head was decided
}

ENTRY_CANDIDATE_FIELDS = {
    "kind", "proposed", "layer", "pack", "source_hash", "span_start", "span_end",
    "extractor", "run_manifest",
}
MAPPING_CANDIDATE_FIELDS = {"kind", "yaml_draft", "pack", "standard", "author"}
# AR-rag4ds-027 r2 O3 / r3 F9: the machine extractor identities the module ships -
# exactly the names the strategies in extract/strategies.py call themselves. A
# candidate's extractor is one of these or 'author:<name>'; any other token
# ("siriweera", "t.a.siriweera", "designee") is refused, so nothing reaches the
# gate as a machine proposal while naming a person. A new strategy registers its
# name here (the test suite checks every strategy's name is listed).
KNOWN_EXTRACTORS = frozenset({"rule-v1", "embed-v1"})


class StagingError(ValueError):
    """Raised loudly for any staging violation."""


def _nonblank(v) -> bool:
    return isinstance(v, str) and bool(v.strip())


def candidate_id(candidate: dict) -> str:
    """Deterministic id from the candidate's canonical content: the same
    filing twice is the same candidate (immutability by construction)."""
    text = json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "cand-" + hashlib.sha256(text.encode("ascii")).hexdigest()[:16]


def validate_candidate(candidate: dict) -> None:
    problems = []
    if not isinstance(candidate, dict):
        raise StagingError("a candidate must be a JSON object")
    kind = candidate.get("kind")
    if kind == "entry":
        fields = ENTRY_CANDIDATE_FIELDS
    elif kind == "mapping-draft":
        fields = MAPPING_CANDIDATE_FIELDS
    else:
        raise StagingError(f"candidate kind {kind!r} not one of ('entry', 'mapping-draft')")
    unknown = sorted(set(candidate) - fields)
    missing = sorted(fields - set(candidate))
    if unknown:
        problems.append(
            f"unknown field(s): {unknown} (source spans travel as hash+offsets, never text)"
        )
    if missing:
        problems.append(f"missing mandatory field(s): {missing}")
    if problems:
        raise StagingError("candidate invalid: " + "; ".join(problems))
    pack = candidate["pack"]
    if kind == "entry":
        p = candidate["proposed"]
        if not isinstance(p, dict) or set(p) != {"type", "content", "source_reference"}:
            problems.append("proposed must carry exactly type/content/source_reference "
                            "(id, version, status are assigned at the gate)")
        else:
            if p["type"] not in ENTRY_TYPES:
                problems.append(f"proposed.type {p['type']!r} not one of {ENTRY_TYPES}")
            if not _nonblank(p["content"]):
                problems.append("proposed.content must be a non-blank string")
            if not _nonblank(p["source_reference"]):
                problems.append("proposed.source_reference must be a non-blank string")
        layer = candidate["layer"]
        if layer not in LAYERS:
            problems.append(f"layer {layer!r} not one of {LAYERS}")
        if not isinstance(pack, str) or not ENTRY_ID_RE.fullmatch(pack or ""):
            problems.append(
                f"pack {pack!r} must be a pack code in the id charset (or {SHARED_PACK!r})"
            )
        elif layer == "domain" and pack == SHARED_PACK:
            problems.append(
                f"domain-layer candidates need a pack code; {SHARED_PACK!r} is reserved"
            )
        elif layer in ("general", "task") and pack != SHARED_PACK:
            problems.append(f"{layer}-layer candidates must use pack {SHARED_PACK!r}")
        sh = candidate["source_hash"]
        if not isinstance(sh, str) or not SHA256_RE.fullmatch(sh):
            problems.append("source_hash must be the 64-hex sha256 of the catalogued document")
        s, e = candidate["span_start"], candidate["span_end"]
        if (not isinstance(s, int) or not isinstance(e, int) or isinstance(s, bool)
                or isinstance(e, bool) or s < 0 or e <= s):
            problems.append("span_start/span_end must be character offsets with 0 <= start < end")
        if not _nonblank(candidate["extractor"]):
            problems.append("extractor identity must be a non-blank string")
        else:
            ex = candidate["extractor"]
            # AR-rag4ds-027 F3, r2 O3, r3 F9: an extractor is one of exactly two forms - a
            # registered machine strategy id (KNOWN_EXTRACTORS) or 'author:<name>' with a
            # non-blank name. Any other token ('Author:', a look-alike in another script,
            # 'by:', a bare person's name) is refused here.
            is_author = ex.startswith("author:") and bool(ex[len("author:"):].strip())
            if not is_author and ex not in KNOWN_EXTRACTORS:
                problems.append(f"extractor {ex!r} is neither a registered machine strategy id "
                                f"({', '.join(sorted(KNOWN_EXTRACTORS))}) nor 'author:<name>' "
                                f"with a non-blank name")
        if not _nonblank(candidate["run_manifest"]):
            problems.append("run_manifest reference must be a non-blank string")
    else:
        if not _nonblank(candidate["yaml_draft"]):
            problems.append("yaml_draft must be a non-blank string")
        if (not isinstance(pack, str) or not ENTRY_ID_RE.fullmatch(pack or "")
                or pack == SHARED_PACK):
            problems.append(
                f"pack {pack!r} must be a pack code (never {SHARED_PACK!r}); "
                "mappings belong to a pack"
            )
        std = candidate["standard"]
        if not isinstance(std, str) or not ENTRY_ID_RE.fullmatch(std or ""):
            problems.append("standard must obey the entry-id character rule (4.5)")
        if not _nonblank(candidate["author"]):
            problems.append("author identity must be a non-blank string")
    if problems:
        raise StagingError(
            f"candidate invalid ({len(problems)} problem(s)): " + "; ".join(problems)
        )


class StagingStore:
    """Append-only candidate store over one JSON Lines journal."""

    def __init__(self, path: Path):
        self.journal = Journal(Path(path))

    # ----- filing -----

    def file_candidate(self, candidate: dict, timestamp: str,
                       known_hashes=None) -> str:
        """File an immutable candidate; returns its id. Filing an id that
        already exists is refused - candidates are never re-filed. When
        known_hashes (the catalogued document hashes) is given, an entry
        candidate's source_hash must be among them - FR7's first link
        (document -> candidate) is checked HERE, by the filer (AR-rag4ds-014
        observation); the S6 CLI supplies the catalog's hashes."""
        validate_candidate(candidate)
        if not _timestamp_valid(timestamp):
            raise StagingError(f"timestamp {timestamp!r} must be 'YYYY-MM-DD HH:MM[:SS] JST'")
        if known_hashes is not None:
            if isinstance(known_hashes, (str, bytes)) or not isinstance(
                known_hashes, (set, frozenset, list, tuple)
            ):
                raise StagingError("known_hashes must be a set/list/tuple of sha256 strings")
            if candidate["kind"] == "entry" and candidate["source_hash"] not in known_hashes:
                raise StagingError(
                    f"source_hash {candidate['source_hash']} is not a catalogued document (FR7)"
                )
        cid = candidate_id(candidate)
        if cid in self.candidates():
            raise StagingError(f"candidate {cid} is already filed; candidates are immutable")
        self.journal.append({
            "schema": STAGING_SCHEMA_VERSION, "record": "candidate", "id": cid,
            "filed": timestamp, "candidate": candidate,
        })
        return cid

    # ----- state -----

    def candidates(self) -> dict:
        """id -> filed candidate record, from the journal."""
        out = {}
        for rec in self.journal.read_all():
            if rec.get("record") == "candidate":
                out[rec["id"]] = rec
        return out

    def state_of(self, cid: str) -> str:
        """Fold the transitions: the latest wins; PENDING if none."""
        if cid not in self.candidates():
            raise StagingError(f"unknown candidate {cid}")
        state = "PENDING"
        for rec in self.journal.read_all():
            if rec.get("record") == "transition" and rec["id"] == cid:
                if rec.get("from") != state:
                    raise StagingError(
                        f"journal transition for {cid} claims from={rec.get('from')!r} but the "
                        f"folded state is {state!r}; refusing a forged or disordered line"
                    )
                state = rec["to"]
        return state

    def states(self) -> dict:
        """One pass over the journal: id -> (state, head-or-None) for every
        candidate, with the same from-check state_of applies (AR-rag4ds-020
        F7: a session reads the journal once, not once per candidate)."""
        out = {}
        for rec in self.journal.read_all():
            if rec.get("record") == "candidate":
                out[rec["id"]] = ("PENDING", None)
            elif rec.get("record") == "transition":
                cid = rec["id"]
                if cid not in out:
                    raise StagingError(f"transition for unknown candidate {cid}")
                state, _ = out[cid]
                if rec.get("from") != state:
                    raise StagingError(
                        f"journal transition for {cid} claims from={rec.get('from')!r} but the "
                        f"folded state is {state!r}; refusing a forged or disordered line"
                    )
                out[cid] = (rec["to"], rec.get("head") if rec["to"] == "GROUPED" else None)
        return out

    def group_head_of(self, cid: str) -> str | None:
        head = None
        for rec in self.journal.read_all():
            if rec.get("record") == "transition" and rec["id"] == cid:
                head = rec.get("head") if rec["to"] == "GROUPED" else None
        return head

    def transition(self, cid: str, to: str, timestamp: str, actor: str,
                   authority: str, head: str | None = None) -> None:
        """One state change, validated against the 6.3 machine. authority
        names what licenses it: a journal decision reference, a detach, or
        the head's decision for a mechanical duplicate close."""
        if to not in STATES:
            raise StagingError(f"state {to!r} not one of {STATES}")
        current = self.state_of(cid)
        if (current, to) not in ALLOWED:
            raise StagingError(
                f"transition {current} -> {to} is not allowed for {cid} (6.3 state machine)"
            )
        if not _timestamp_valid(timestamp):
            raise StagingError(f"timestamp {timestamp!r} must be 'YYYY-MM-DD HH:MM[:SS] JST'")
        if not _nonblank(actor) or not _nonblank(authority):
            raise StagingError("actor and authority must be non-blank strings")
        if to == "GROUPED":
            if head is None or head == cid or head not in self.candidates():
                raise StagingError("GROUPED needs a known head candidate other than itself")
            head_state = self.state_of(head)
            if head_state in TERMINAL:
                raise StagingError(f"cannot group under {head}: it is already decided")
            if head_state != "PENDING":
                # AR-rag4ds-014 F1: a head is a PENDING candidate, never itself a
                # member - no chains, no cycles, no stranded members.
                raise StagingError(
                    f"cannot group under {head}: a head must be PENDING, it is {head_state}"
                )
            if self.members_of(cid):
                # AR-rag4ds-014 r2 F1: the other order - a candidate that is
                # already the head of a cluster cannot itself be grouped, or
                # its members would be stranded one level down.
                raise StagingError(
                    f"cannot group {cid}: it is the head of a cluster "
                    f"({len(self.members_of(cid))} member(s)); clusters are one level deep"
                )
        elif head is not None:
            raise StagingError("head belongs only to GROUPED transitions")
        rec = {"schema": STAGING_SCHEMA_VERSION, "record": "transition", "id": cid,
               "from": current, "to": to, "timestamp": timestamp, "actor": actor,
               "authority": authority}
        if head is not None:
            rec["head"] = head
        self.journal.append(rec)

    def members_of(self, head: str) -> list[str]:
        return [cid for cid in self.candidates()
                if self.state_of(cid) == "GROUPED" and self.group_head_of(cid) == head]

    def head_decision_of(self, head: str) -> str:
        """The authority recorded on the head's own terminal transition."""
        if self.state_of(head) not in TERMINAL:
            raise StagingError(f"head {head} has no decision yet; members cannot close")
        authority = None
        for rec in self.journal.read_all():
            if rec.get("record") == "transition" and rec["id"] == head and rec["to"] in TERMINAL:
                authority = rec["authority"]
        return authority

    def close_cluster(self, head: str, timestamp: str) -> list[str]:
        """The mechanical GROUPED exit: the head has received a decision, so
        every member closes as REJECTED-DUPLICATE citing that decision. The
        citation is taken from the head's own terminal record in this
        journal, never from the caller (AR-rag4ds-014 F2)."""
        head_decision = self.head_decision_of(head)
        closed = []
        for cid in self.members_of(head):
            self.transition(cid, "REJECTED-DUPLICATE", timestamp, "gate", head_decision)
            closed.append(cid)
        return closed

    def detach(self, cid: str, timestamp: str, actor: str) -> None:
        """The expert's GROUPED exit: back to PENDING for its own review,
        only while the head is undecided."""
        head = self.group_head_of(cid)
        if self.state_of(cid) != "GROUPED" or head is None:
            raise StagingError(f"{cid} is not grouped")
        if self.state_of(head) in TERMINAL:
            raise StagingError(f"{cid} cannot detach: head {head} is already decided")
        self.transition(cid, "PENDING", timestamp, actor, f"detached from {head}")
