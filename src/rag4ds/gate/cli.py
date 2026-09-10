"""The expert gate, day to day - the CLI (plan S6; 6.4, D-1; FR5, FR8, 6.3).

A gate session: the expert starts the tool under their recorded identity
(single writer), takes a queue filtered by pack and layer, and for each
PENDING candidate sees the proposed entry beside its rendered source span
from the local corpus copy and beside the related existing entries the
retriever finds (conflict display). Decisions are keyboard-driven:
  a  approve            e  edit-then-approve        r  reject
  g  group under a head (GROUPED; the head's decision later closes the
     member mechanically as reject-duplicate citing that decision)
  s  skip               q  quit the session
Every decision is appended to the decision journal atomically BEFORE the
staging transition is written and before the next candidate is shown; a
crashed session loses at most the in-flight decision and corrupts
nothing, and the next session start repairs a decision whose staging
transition never landed (the decision carries the candidate id) AND runs
the mechanical cluster close that decision owed. The gate never writes
the knowledge base: approval marks the candidate, the fold compiles
(I2).

Review context on the record: every decision the gate writes - the
expert's approve / edit-then-approve / reject, the mechanical
reject-duplicate, and retract - carries a `review` context (what the
conflict display showed, which shown ids the expert flagged as deliberate
coexistence, whether the source span rendered, the quotation-bound
confirmation; mechanical and retract records carry the not-applicable
values), so unrecorded coexistence cannot exist in the trail
(AR-rag4ds-020 F5/F8/F10/F13), and the decisions layer refuses a
candidate-naming decision without one (O10). The evidence rule: an ENTRY
APPROVAL needs both evidences - a working conflict display AND a rendered
source span - and is REFUSED when either is absent or failed, unless the
session was started with allow_unchecked_approval, which the manifest
and the decision's review then say (O6: the rule is deliberate and
symmetric); an INVALID span (the local document contradicts the record)
is never approvable under any flag (O14). Rejecting, grouping and
skipping never need them.

A candidate whose source span cannot be rendered (no local document,
hash mismatch, not UTF-8, offsets outside the document) is NOT fatal: the
span panel carries the loud reason, the review records the span state
(unavailable / invalid), and the expert can reject, group or skip it
(F11); only a failure of the candidate panel itself refuses the candidate
for the session, parks it, counts it (counts.unrenderable) and prints
why. The session ends with a written summary - counts by decision (per
candidate and per decision), time spent, the index and model behind the
display - appended to the runs journal as an NFR2 run manifest; it is
written on every exit, including an abort, with the abort reason.
Rendering is plain rich with markup OFF for every untrusted string,
recorded, no TUI framework (D-1).

Source spans are CHARACTER offsets into the UTF-8 text of the document
(plan component 2, staging): the gate decodes the document strictly once
and slices the str (F1/F2). Id assignment happens here (FR5): the expert
accepts or types the entry id; uniqueness and the version chain are
checked against the fold of the decision journal, computed once per
session and updated per decision (F7). Retraction takes the entry's
final form from the journal's current approval, never from the derived
tree (F9). Concurrency: one session per journal at a time.
"""

from __future__ import annotations

import hashlib
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rag4ds.auditlog.journal import Journal
from rag4ds.gate.decisions import (
    APPROVAL_TYPES,
    DecisionError,
    _timestamp_valid,
    approve_entry,
    approve_mapping,
    reject,
    retract,
)
from rag4ds.kb.fold import DECISION_JOURNAL, fold
from rag4ds.model.entry import ENTRY_ID_RE, SHARED_PACK, Entry, EntryValidationError
from rag4ds.staging.store import StagingError, StagingStore

GATE_SESSION_SCHEMA_VERSION = 1
KEYS = ("a", "e", "r", "g", "s", "q")
QUOTATION_BOUND = ("the entry states the fact in the project's own words; verbatim quotation "
                   "is bounded to a short attributed phrase, never a passage (component 2, NFR5)")
def not_applicable_review() -> dict:
    """A fresh not-applicable review context (fresh lists each call - O12: no
    record may share a mutable object with a module constant)."""
    return {"conflict_display": "not-applicable", "related_shown": [], "conflicts": [],
            "source_span": "not-applicable", "quotation_bound_confirmed": False}


class GateError(RuntimeError):
    """Raised loudly for any gate-session violation."""


def jst_now() -> str:
    """The local clock, which on the project machines IS JST; main() refuses
    to run on a machine whose UTC offset is not +9 h."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S JST")


def suggest_id(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = "-".join(words[:6])[:64].strip("-")
    return out or "entry"


@dataclass
class GateContext:
    decisions: Journal          # journal/decisions.jsonl - THE audit log of gate decisions
    staging: StagingStore       # stage/candidates.jsonl
    runs: Journal               # runs/gate-sessions.jsonl - session summaries (NFR2)
    approver: str
    code_commit: str
    clock: Callable[[], str] = jst_now
    source_resolver: Callable[[str], Path | None] | None = None  # source_hash -> raw file
    # (content, layer, pack) -> related approved entries. CONTRACT (O13/O16): it
    # may raise GateError (recorded as lookup-failed); an OSError from it is
    # treated like an unrenderable candidate panel (parked for the session,
    # counted, nothing recorded); any other exception is a programming error
    # and aborts the session with a manifest.
    related_finder: Callable[[str, str, str], list[dict]] | None = None
    display_provenance: dict | None = None  # index_version + embedding model behind the finder
    # approve an entry whose evidence is UNAVAILABLE (no display / no span), recorded on the
    # record; never covers INVALID evidence - a span that contradicts the record is always
    # refused (O14)
    allow_unchecked_approval: bool = False
    journal_ref: str = DECISION_JOURNAL.as_posix()

    def __post_init__(self) -> None:
        if not isinstance(self.approver, str) or not self.approver.strip():
            raise GateError("approver (the expert's recorded identity) must be a non-blank string")
        if not isinstance(self.code_commit, str) or not re.fullmatch(
            r"[0-9a-f]{7,40}", self.code_commit
        ):
            raise GateError("code_commit must be a 7-40 hex git commit id (NFR2)")
        # AR-rag4ds-020 O1: the citation names the file that holds the decisions
        if not self.decisions.path.as_posix().endswith(self.journal_ref):
            raise GateError(
                f"journal_ref {self.journal_ref!r} does not name the decision journal "
                f"{self.decisions.path.as_posix()!r}"
            )
        # O11: a finder without its provenance would be recorded like no finder at all
        if self.related_finder is not None and not isinstance(self.display_provenance, dict):
            raise GateError("a related_finder needs its display_provenance (index + model pin)")


@dataclass
class Summary:
    approver: str
    started: str
    filters: dict
    counts: dict = field(default_factory=lambda: {
        "approve": 0, "edit-then-approve": 0, "reject": 0, "reject-duplicate": 0,
        "grouped": 0, "skipped": 0, "retract": 0, "refused": 0, "unrenderable": 0,
        "candidates_source_unavailable": 0, "candidates_display_unavailable": 0,
        "decisions_without_source": 0, "decisions_without_display": 0})
    decisions_written: int = 0
    repaired: int = 0
    ended: str = ""
    elapsed_seconds: float = 0.0
    aborted: str | None = None


def _plain(s) -> Text:
    """Untrusted text rendered verbatim - never as rich markup (F4)."""
    return Text(str(s))


class GateSession:
    """One expert session. `read(prompt) -> str` supplies keystrokes/lines
    (input() in the CLI, a scripted list in tests); `console` is a rich
    Console, recorded so the session transcript can be saved. State is read
    once per session and updated incrementally (F7)."""

    def __init__(self, ctx: GateContext, console: Console, read: Callable[[str], str],
                 pack: str | None = None, layer: str | None = None):
        self.ctx, self.console, self.read = ctx, console, read
        if pack is not None and (not ENTRY_ID_RE.fullmatch(pack)):
            raise GateError(f"pack filter {pack!r} is not a pack code")
        if layer is not None and layer not in ("general", "domain", "task"):
            raise GateError(f"layer filter {layer!r} is not a layer")
        self.pack, self.layer = pack, layer
        self._t0 = time.monotonic()
        self._skipped: set[str] = set()
        self._no_source: set[str] = set()   # candidates whose span was not rendered (F12)
        self._no_display: set[str] = set()  # candidates shown without a working display
        self.summary = Summary(ctx.approver, ctx.clock(), {"pack": pack, "layer": layer})
        # one read of each journal; updated as the session writes
        self._cands = ctx.staging.candidates()
        self._states = ctx.staging.states()            # cid -> (state, head)
        self._decisions = ctx.decisions.read_all()
        with tempfile.TemporaryDirectory() as tmp:
            st = fold(self._decisions, Path(tmp) / "fold")
        self._versions: dict = dict(st.versions)
        self._retired: set = set(st.retired)
        self._ended = False

    # ----- journal / state helpers -----

    def _append_decision(self, record: dict) -> str:
        """Append one decision atomically and return its journal reference
        (file:line, 1-based) - the authority staging transitions cite."""
        self.ctx.decisions.append(record)
        self._decisions.append(record)
        self.summary.decisions_written += 1
        return f"{self.ctx.journal_ref}:{len(self._decisions)}"

    def _transition(self, cid: str, to: str, authority: str, head: str | None = None) -> None:
        self.ctx.staging.transition(cid, to, self.ctx.clock(), self.ctx.approver, authority,
                                    head=head)
        self._states[cid] = (to, head if to == "GROUPED" else None)

    def _members_of(self, head: str) -> list[str]:
        return [cid for cid, (st, h) in self._states.items() if st == "GROUPED" and h == head]

    def _current_entry(self, entry_id: str) -> dict | None:
        """The entry's current form from the journal: the latest approval of
        that id at its current version (F9 - never from the derived tree)."""
        cur = self._versions.get(entry_id.lower())
        if cur is None:
            return None
        for rec in reversed(self._decisions):
            if rec.get("type") in APPROVAL_TYPES and "entry" in rec:
                e = rec["entry"]
                if e["id"].lower() == entry_id.lower() and e["version"] == cur:
                    return dict(e)
        raise GateError(f"journal inconsistency: no approval record for {entry_id!r} v{cur}")

    def repair(self) -> int:
        """Start-of-session repair (6.4 crash contract): a decision whose
        staging transition never landed is applied now, loudly, AND the
        mechanical cluster close that decision owed is run (F3); a GROUPED
        member whose head is already decided with no decision of its own
        is an inconsistency the session refuses to hide."""
        open_ids = {cid for cid, (st, _) in self._states.items() if st in ("PENDING", "GROUPED")}
        repaired = 0
        if open_ids:
            for n, rec in enumerate(list(self._decisions), 1):
                cid = rec.get("candidate_id")
                if cid not in open_ids:
                    continue
                if self._states.get(cid, ("PENDING", None))[0] not in ("PENDING", "GROUPED"):
                    open_ids.discard(cid)  # closed meanwhile (a repaired head's cluster)
                    continue
                ref = f"{self.ctx.journal_ref}:{n}"
                to = {"approve": "APPROVED", "edit-then-approve": "APPROVED", "reject": "REJECTED",
                      "reject-duplicate": "REJECTED-DUPLICATE"}.get(rec["type"])
                if to is None:
                    continue
                self._transition(cid, to, ref)
                self.console.print(_plain(f"REPAIRED {cid}: decision {ref} had no staging "
                                          f"transition; applied -> {to}"), style="bold red")
                open_ids.discard(cid)
                repaired += 1
                if to in ("APPROVED", "REJECTED"):
                    repaired += self._close_members(cid, ref, repairing=True)
        for cid, (st, head) in self._states.items():
            if st == "GROUPED" and head is not None:
                hstate = self._states.get(head, ("?", None))[0]
                if hstate in ("APPROVED", "REJECTED", "REJECTED-DUPLICATE"):
                    raise GateError(
                        f"stranded member {cid}: head {head} is {hstate} but the member is still "
                        "GROUPED with no decision - the journal and staging disagree"
                    )
        self.summary.repaired = repaired
        return repaired

    def queue(self) -> list[str]:
        out = []
        for cid, rec in self._cands.items():
            if self._states.get(cid, ("PENDING", None))[0] != "PENDING":
                continue
            c = rec["candidate"]
            if self.pack is not None and c.get("pack") != self.pack:
                continue
            if self.layer is not None and c.get("layer") != self.layer:
                continue
            out.append(cid)
        return out

    # ----- rendering -----

    def _source_span(self, cid: str, c: dict) -> tuple[str, str]:
        """(state, text): the span as CHARACTERS of the document's UTF-8 text,
        or a loud unavailability / invalidity - never an exception (F11), so
        the candidate stays rejectable; out-of-range spans are 'invalid'."""
        if c["kind"] != "entry":
            return "not-applicable", "(mapping draft: no source span)"
        if self.ctx.source_resolver is None:
            return ("unavailable",
                    "SOURCE UNAVAILABLE: no source resolver configured for this session")
        path = self.ctx.source_resolver(c["source_hash"])
        if path is None:
            return ("unavailable",
                    f"SOURCE UNAVAILABLE: no local document for hash {c['source_hash'][:16]}...")
        try:
            raw = Path(path).read_bytes()
        except OSError as exc:  # unreadable file (permissions, a vanished path) - F14
            return "unavailable", f"SOURCE UNAVAILABLE: local document {path} cannot be read: {exc}"
        if hashlib.sha256(raw).hexdigest() != c["source_hash"]:
            return ("invalid", f"SOURCE INVALID: local document {path} does not hash to the "
                               "candidate's source_hash (tampered or wrong file)")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            return "invalid", f"SOURCE INVALID: local document {path} is not UTF-8 text: {exc}"
        s, e = c["span_start"], c["span_end"]
        if not (0 <= s < e <= len(text)):
            return ("invalid", f"SOURCE INVALID: candidate {cid} span [{s}, {e}) is outside the "
                               f"document ({len(text)} characters)")
        return "rendered", text[s:e]

    def _related(self, c: dict) -> tuple[str, list[dict]]:
        """(conflict-display state, rows). The state is recorded on the decision."""
        if c["kind"] != "entry":
            return "not-applicable", []
        if self.ctx.related_finder is None:
            self.console.print(_plain("NO CONFLICT DISPLAY: no related-entry finder configured; "
                                      "related approved entries cannot be shown."),
                               style="bold red")
            return "no-finder", []
        try:
            rows = list(self.ctx.related_finder(c["proposed"]["content"], c["layer"], c["pack"]))
        except GateError as exc:  # the finder's own loud failure (an ERROR status, say)
            self.console.print(_plain(f"RELATED LOOKUP FAILED: {exc}"), style="bold red")
            return "lookup-failed", []
        return ("checked-with-results" if rows else "checked-clean"), rows

    def show(self, cid: str, rec: dict) -> tuple[str, str, list[dict]]:
        c = rec["candidate"]
        t = Table(show_header=False, box=None)
        t.add_row("candidate", _plain(cid))
        t.add_row("kind", _plain(c["kind"]))
        t.add_row("pack / layer", _plain(f"{c.get('pack')} / {c.get('layer', '-')}"))
        if c["kind"] == "entry":
            p = c["proposed"]
            t.add_row("type", _plain(p["type"]))
            t.add_row("content", _plain(p["content"]))
            t.add_row("source_reference", _plain(p["source_reference"]))
            t.add_row("extractor", _plain(f"{c['extractor']} ({c['run_manifest']})"))
        else:
            t.add_row("standard", _plain(c["standard"]))
            t.add_row("author", _plain(c["author"]))
            t.add_row("draft", _plain(c["yaml_draft"]))
        self.console.print(Panel(t, title="candidate"))
        span_state, span_text = self._source_span(cid, c)
        style = "bold red" if span_state in ("unavailable", "invalid") else None
        self.console.print(Panel(_plain(span_text), title="source span (local corpus copy)",
                                 style=style or ""))
        if span_state in ("unavailable", "invalid"):
            self._no_source.add(cid)
        display, related = self._related(c)
        if related:
            r = Table(title="related approved entries (conflicts / overlaps)")
            r.add_column("id")
            r.add_column("relevance")
            r.add_column("content")
            for e in related:
                r.add_row(_plain(e["id"]), _plain(f"{e['relevance_score']:.3f}"),
                          _plain(e["content"]))
            self.console.print(r)
        elif display == "checked-clean":
            self.console.print("related approved entries: none found")
        if display in ("no-finder", "lookup-failed"):
            self._no_display.add(cid)
        return span_state, display, related

    # ----- decisions -----

    def _review(self, display: str, related: list[dict], conflicts: list[str],
                span_state: str, quotation_confirmed: bool) -> dict:
        return {
            "conflict_display": display,
            "related_shown": [e["id"] for e in related],
            "conflicts": list(conflicts),
            "source_span": span_state,
            "quotation_bound_confirmed": quotation_confirmed,
        }

    def _count_evidence(self, span_state: str, display: str) -> None:
        if span_state in ("unavailable", "invalid"):
            self.summary.counts["decisions_without_source"] += 1
        if display in ("no-finder", "lookup-failed"):
            self.summary.counts["decisions_without_display"] += 1

    def _decide_entry(self, cid: str, c: dict, kind: str, span_state: str, display: str,
                      related: list[dict]) -> None:
        p = c["proposed"]
        if span_state == "invalid":
            raise GateError(
                "approval refused: the source span is INVALID (the local document contradicts "
                "the candidate's record - wrong file, tampered, not UTF-8, or offsets outside "
                "it); this is never approvable, even with allow_unchecked_approval (O14) - "
                "reject it or fix the corpus copy"
            )
        missing = []
        if display in ("no-finder", "lookup-failed"):
            missing.append(f"the conflict display is {display}")
        if span_state == "unavailable":
            missing.append("the source span is unavailable")
        if missing and not self.ctx.allow_unchecked_approval:
            raise GateError(
                "approval refused: " + " and ".join(missing) + " - an entry needs both evidences "
                "(plan 6.4: unrecorded coexistence is not allowed; the span is what the claim is "
                "checked against); reject/group/skip it, fix the evidence, or start the session "
                "with allow_unchecked_approval, which is recorded"
            )
        content = p["content"]
        if kind == "edit-then-approve":
            content = self.read("edited content: ").strip()
            if not content:
                raise GateError("edit-then-approve needs the edited content")
        self.console.print(Panel(_plain(QUOTATION_BOUND), title="quotation bound (component 2)"))
        if self.read("the content honors the quotation bound? [y/N]: ").strip().lower() != "y":
            raise GateError("approval refused: the quotation bound was not confirmed")
        default = suggest_id(content)
        entry_id = (self.read(f"entry id [{default}]: ").strip() or default)
        sup = self.read("supersedes existing id (blank = new entry): ").strip() or None
        conflicts: list[str] = []
        if related:
            shown = {e["id"] for e in related}
            typed = self.read("approve despite shown related entries? their ids, comma-separated "
                              "(blank = none): ").strip()
            conflicts = [x.strip() for x in typed.split(",") if x.strip()]
            bad = [x for x in conflicts if x not in shown]
            if bad:
                raise GateError(f"conflict id(s) {bad} were not among the shown related entries "
                                f"{sorted(shown)} (F8)")
        if sup:
            if sup != entry_id:
                raise GateError("a supersession keeps the existing id: type the same id")
            cur = self._versions.get(entry_id.lower())
            if cur is None:
                raise GateError(f"supersession target {entry_id!r} is not a current entry")
            version, supersedes = cur + 1, {"id": entry_id, "version": cur}
        else:
            version, supersedes = 1, None
        review = self._review(display, related, conflicts, span_state, True)
        try:
            entry = Entry(id=entry_id, type=p["type"], layer=c["layer"], pack=c["pack"],
                          content=content, source_reference=p["source_reference"],
                          version=version, status="current")
            record = approve_entry(kind, self.ctx.approver, self.ctx.clock(), p["content"],
                                   p["source_reference"], entry, self._versions,
                                   supersedes=supersedes,
                                   conflict_with=conflicts[0] if conflicts else None,
                                   retired=frozenset(self._retired), candidate_id=cid,
                                   review=review)
        except (EntryValidationError, DecisionError) as exc:
            raise GateError(f"decision refused: {exc}") from exc
        ref = self._append_decision(record)
        self._transition(cid, "APPROVED", ref)
        self._versions[entry_id.lower()] = version
        self.summary.counts[kind] += 1
        self._count_evidence(span_state, display)
        self.console.print(_plain(f"{kind} {entry_id} v{version} -> {ref}"), style="green")
        self._close_members(cid, ref)

    def _decide_mapping(self, cid: str, c: dict, kind: str) -> None:
        try:
            artifact = yaml.safe_load(c["yaml_draft"])
        except yaml.YAMLError as exc:
            raise GateError(f"mapping draft is not readable YAML: {exc}") from exc
        content_hash = hashlib.sha256(c["yaml_draft"].encode("utf-8")).hexdigest()
        try:
            record = approve_mapping(kind, self.ctx.approver, self.ctx.clock(), content_hash,
                                     f"staging:{cid}", artifact, c["pack"], candidate_id=cid,
                                     review=not_applicable_review())
        except DecisionError as exc:
            raise GateError(f"decision refused: {exc}") from exc
        ref = self._append_decision(record)
        self._transition(cid, "APPROVED", ref)
        self.summary.counts[kind] += 1
        self.console.print(_plain(f"{kind} mapping {artifact.get('standard')} -> {ref}"),
                           style="green")
        self._close_members(cid, ref)

    def _reject(self, cid: str, c: dict, span_state: str, display: str,
                related: list[dict]) -> None:
        reason = self.read("reason: ").strip()
        text = c["proposed"]["content"] if c["kind"] == "entry" else c["yaml_draft"]
        src = c["proposed"]["source_reference"] if c["kind"] == "entry" else f"staging:{cid}"
        review = self._review(display, related, [], span_state, False)
        try:
            record = reject("reject", self.ctx.approver, self.ctx.clock(), text, src, reason,
                            candidate_id=cid, review=review)
        except DecisionError as exc:
            raise GateError(f"decision refused: {exc}") from exc
        ref = self._append_decision(record)
        self._transition(cid, "REJECTED", ref)
        self.summary.counts["reject"] += 1
        self._count_evidence(span_state, display)
        self.console.print(_plain(f"reject {cid} -> {ref}"), style="yellow")
        self._close_members(cid, ref)

    def _close_members(self, head: str, head_ref: str, repairing: bool = False) -> int:
        """The head was decided: every GROUPED member gets a mechanical
        reject-duplicate decision citing the head decision (review:
        not-applicable - no display is consulted for a mechanical close),
        then the staging store closes the cluster from the head's own
        terminal record."""
        members = self._members_of(head)
        if not members:
            return 0
        for m in members:
            mc = self._cands[m]["candidate"]
            text = mc["proposed"]["content"] if mc["kind"] == "entry" else mc["yaml_draft"]
            src = (mc["proposed"]["source_reference"] if mc["kind"] == "entry"
                   else f"staging:{m}")
            record = reject("reject-duplicate", self.ctx.approver, self.ctx.clock(), text, src,
                            f"duplicate of {head}", head_decision=head_ref, candidate_id=m,
                            review=not_applicable_review())
            self._append_decision(record)
        closed = self.ctx.staging.close_cluster(head, self.ctx.clock())
        for m in closed:
            self._states[m] = ("REJECTED-DUPLICATE", None)
        self.summary.counts["reject-duplicate"] += len(closed)
        self.console.print(_plain(f"reject-duplicate {len(closed)} member(s) of {head} closed "
                                  f"citing {head_ref}{' (repair)' if repairing else ''}"),
                           style="yellow")
        return len(closed)

    def _group(self, cid: str) -> None:
        head = self.read("head candidate id: ").strip()
        try:
            self._transition(cid, "GROUPED", f"gate:{self.ctx.approver}:grouped-under:{head}",
                             head=head)
        except StagingError as exc:
            raise GateError(f"grouping refused: {exc}") from exc
        self.summary.counts["grouped"] += 1
        self.console.print(_plain(f"grouped {cid} under {head}"), style="cyan")

    def retract_entry(self, entry_id: str) -> str:
        """Retract a current entry (FR5/FR8): its final form is the journal's
        current approval marked retracted (never the derived tree, F9); the
        decision directs the fold to remove the file. The review context is
        not-applicable (no display is consulted); the id must be typed
        exactly as recorded (O9)."""
        cur_form = self._current_entry(entry_id)
        if cur_form is None:
            raise GateError(f"{entry_id!r} is not a current entry")
        if cur_form["id"] != entry_id:
            raise GateError(f"type the entry id exactly as recorded: {cur_form['id']!r} "
                            f"(ids are case-insensitively unique; {entry_id!r} names it)")
        final = Entry(**dict(cur_form, status="retracted"))
        try:
            record = retract(self.ctx.approver, self.ctx.clock(), entry_id, final.version, final,
                             self._versions, review=not_applicable_review())
        except DecisionError as exc:
            raise GateError(f"retraction refused: {exc}") from exc
        ref = self._append_decision(record)
        del self._versions[entry_id.lower()]
        self._retired.add(entry_id.lower())
        self.summary.counts["retract"] += 1
        self.console.print(_plain(f"retract {entry_id} v{final.version} -> {ref}"), style="red")
        return ref

    # ----- the loop -----

    def run(self) -> Summary:
        """The session loop; end() runs on every exit (F6)."""
        try:
            self.repair()
            while True:
                q = [c for c in self.queue() if c not in self._skipped]
                if not q:
                    self.console.print(f"queue empty ({len(self._skipped)} skipped this session)")
                    break
                cid = q[0]
                rec = self._cands[cid]
                try:
                    span_state, display, related = self.show(cid, rec)
                except (GateError, OSError) as exc:
                    # the candidate panel itself cannot be rendered (F11/F14): park it for
                    # this session, say why, and move on - never freeze the queue; any
                    # other exception is a programming error and aborts (manifest written)
                    self.summary.counts["unrenderable"] += 1
                    self._skipped.add(cid)
                    self.console.print(_plain(f"UNRENDERABLE {cid}: {exc}"), style="bold red")
                    continue
                key = self.read("[a]pprove [e]dit-then-approve [r]eject [g]roup [s]kip [q]uit: ")
                key = key.strip().lower()[:1]
                if key not in KEYS:
                    self.console.print(_plain(f"unknown key {key!r}"))
                    continue
                c = rec["candidate"]
                if key == "q":
                    break
                if key == "s":
                    self.summary.counts["skipped"] += 1
                    self._skipped.add(cid)  # stays PENDING; not shown again this session
                    continue
                try:
                    if key in ("a", "e"):
                        kind = "approve" if key == "a" else "edit-then-approve"
                        if c["kind"] == "entry":
                            self._decide_entry(cid, c, kind, span_state, display, related)
                        elif key == "e":
                            raise GateError("edit-then-approve applies to entry candidates; "
                                            "edit the draft and re-file it")
                        else:
                            self._decide_mapping(cid, c, kind)
                    elif key == "r":
                        self._reject(cid, c, span_state, display, related)
                    elif key == "g":
                        self._group(cid)
                except GateError as exc:
                    # a refused decision is shown loudly and the same candidate is re-shown (F6)
                    self.summary.counts["refused"] += 1
                    self.console.print(_plain(f"REFUSED: {exc}"), style="bold red")
                    continue
        except BaseException as exc:
            self.summary.aborted = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.end()
        return self.summary

    def end(self) -> Summary:
        if self._ended:
            return self.summary
        self._ended = True
        self.summary.ended = self.ctx.clock()
        self.summary.elapsed_seconds = round(time.monotonic() - self._t0, 3)
        self.summary.counts["candidates_source_unavailable"] = len(self._no_source)
        self.summary.counts["candidates_display_unavailable"] = len(self._no_display)
        record = {
            "kind": "gate-session",
            "schema": GATE_SESSION_SCHEMA_VERSION,
            "approver": self.summary.approver,
            "started": self.summary.started,
            "ended": self.summary.ended,
            "elapsed_seconds": self.summary.elapsed_seconds,
            "filters": self.summary.filters,
            "counts": dict(self.summary.counts),
            "decisions_written": self.summary.decisions_written,
            "repaired": self.summary.repaired,
            "aborted": self.summary.aborted,
            "code_commit": self.ctx.code_commit,
            "config": {
                "journal": self.ctx.journal_ref,
                "tool": "rag4ds.gate.cli",
                "conflict_display": self.ctx.display_provenance
                if self.ctx.related_finder is not None else None,
                "allow_unchecked_approval": self.ctx.allow_unchecked_approval,
                "source_resolver": self.ctx.source_resolver is not None,
            },
        }
        self.ctx.runs.append(record)
        t = Table(title="session summary")
        t.add_column("decision")
        t.add_column("count")
        for k, v in self.summary.counts.items():
            t.add_row(k, str(v))
        t.add_row("elapsed seconds", str(self.summary.elapsed_seconds))
        if self.summary.aborted:
            t.add_row("aborted", _plain(self.summary.aborted))
        self.console.print(t)
        return self.summary


# ----- wiring helpers -----


def make_source_resolver(runs_journal: Journal, data_dir: Path) -> Callable[[str], Path | None]:
    """source_hash -> local raw document path, via the ingest manifests in
    the runs journal (corpus.ingest wrote them); the gate re-hashes on read."""
    def resolve(source_hash: str) -> Path | None:
        for rec in runs_journal.read_all():
            if rec.get("kind") == "ingest" and rec.get("hash") == source_hash:
                local = rec["local_path"]
                if Path(local).is_absolute() or ".." in Path(local).parts:
                    raise GateError(f"ingest manifest local_path {local!r} is not inside data_dir")
                p = Path(data_dir) / local
                return p if p.is_file() else None
        return None
    return resolve


def make_related_finder(retriever, top_k: int = 5) -> Callable[[str, str, str], list[dict]]:
    """Conflict display via the S5 retriever: related approved entries for a
    candidate's content (domain filter for pack candidates). An ERROR
    status raises GateError - the session records lookup-failed."""
    from rag4ds.retrieve.contract import GroundingRequest

    def find(content: str, layer: str, pack: str) -> list[dict]:
        domain = None if pack == SHARED_PACK else pack
        resp = retriever.ground(GroundingRequest(query=content, domain=domain, top_k=top_k))
        if resp.status == "ERROR":
            raise GateError(f"retriever ERROR: {resp.meta.get('error')}")
        return [e.to_dict() for e in resp.entries]
    return find


def display_provenance_of(retriever) -> dict:
    m = retriever.index.manifest
    return {"index_version": m["index_version"], "embedding_model": dict(m["embedding_model"]),
            "min_relevance": m["min_relevance"]}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="rag4ds expert gate (plan 6.4, D-1)")
    ap.add_argument("--module-root", required=True)
    ap.add_argument("--approver", required=True)
    ap.add_argument("--code-commit", required=True)
    ap.add_argument("--index-dir", required=True, help="a built index (conflict display)")
    ap.add_argument("--data-dir", required=True, help="local raw corpus copy")
    ap.add_argument("--pack")
    ap.add_argument("--layer")
    ap.add_argument("--transcript", help="write the recorded session here")
    ap.add_argument("--retract", help="retract this entry id instead of walking the queue")
    ap.add_argument("--allow-unchecked-approval", action="store_true",
                    help="approve entries without both evidences (recorded on every record)")
    args = ap.parse_args(argv)

    if time.localtime().tm_gmtoff != 9 * 3600:
        print("gate: this machine's local clock is not JST (UTC+9); refusing", file=sys.stderr)
        return 1
    root = Path(args.module_root)
    from rag4ds.index.build import SentenceTransformerEmbedder, load_index
    from rag4ds.retrieve.service import Retriever

    idx = load_index(args.index_dir)
    pin = idx.manifest["embedding_model"]
    retriever = Retriever(idx, SentenceTransformerEmbedder(pin["id"], pin["revision"]))
    ctx = GateContext(
        decisions=Journal(root / DECISION_JOURNAL),
        staging=StagingStore(root / "stage" / "candidates.jsonl"),
        runs=Journal(root / "runs" / "gate-sessions.jsonl"),
        approver=args.approver, code_commit=args.code_commit,
        source_resolver=make_source_resolver(Journal(root / "runs" / "ingest.jsonl"),
                                             Path(args.data_dir)),
        related_finder=make_related_finder(retriever),
        display_provenance=display_provenance_of(retriever),
        allow_unchecked_approval=args.allow_unchecked_approval,
    )
    console = Console(record=True, markup=False, highlight=False)
    session = GateSession(ctx, console, input, pack=args.pack, layer=args.layer)
    rc = 0
    try:
        if args.retract:
            try:
                session.retract_entry(args.retract)
            finally:
                session.end()
        else:
            session.run()
    finally:
        if args.transcript:
            # O20/F16: a failed transcript write never masks a session error, and
            # never hides behind a successful session either - it is then the failure
            rc = write_transcript(console, Path(args.transcript),
                                  session_failed=sys.exc_info()[0] is not None)
    return rc


def write_transcript(console: Console, path: Path, session_failed: bool) -> int:
    """Write the recorded session to path. Returns 0 on success; on OSError it
    prints the reason to stderr and returns 2 when the session itself
    succeeded (the write failure is the failure, F16) or 0 when a session
    exception is already in flight (that exception wins, O20)."""
    try:
        Path(path).write_text(console.export_text(), encoding="utf-8")
        return 0
    except OSError as exc:
        print(f"gate: transcript not written to {path}: {exc}", file=sys.stderr)
        return 0 if session_failed else 2


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "GateContext", "GateError", "GateSession", "Summary", "QUOTATION_BOUND",
    "not_applicable_review", "jst_now", "suggest_id", "make_source_resolver",
    "make_related_finder", "display_provenance_of", "write_transcript", "main",
    "_timestamp_valid",
]
