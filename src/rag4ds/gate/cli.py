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

import difflib
import hashlib
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from io import UnsupportedOperation
from pathlib import Path
from typing import Callable

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rag4ds.auditlog.journal import Journal
from rag4ds.author.drafts import MIN_PASSAGE_COVERAGE, author_of, passage_coverage
from rag4ds.extract.strategy import quotation_bound_check
from rag4ds.gate.identities import (
    GATING_ROLES,
    IDENTITIES_FILE,
    IdentityError,
    IdentityRegistry,
)
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
# the three prompts a scripted reader answers (tools/gate_sheet.py imports them -
# AR-rag4ds-032 O2, AR-rag4ds-034 O4: one literal each, no drift)
KEY_PROMPT = "[a]pprove [e]dit-then-approve [r]eject [g]roup [s]kip [q]uit: "
REASON_PROMPT = "reason: "
HEAD_PROMPT = "head candidate id: "
# AR-rag4ds-027 F2/F7 - the edit line, in the measure it uses. Two shares, from
# difflib's matching blocks between the candidate's text and the edited text:
#   kept = matched characters / len(candidate)   - how much of the candidate survives
#   own  = 1 - matched characters / len(edited)  - how much of the entry the editor wrote
# An edit keeps at least MIN_EDIT_KEPT of the candidate and the editor's own words
# are at most MAX_EDIT_OWN of the entry. The two cases the external auditor drove
# at r2: a candidate cut to its first 43% (kept 0.43) is refused; an entry more
# than half of which the editor appended (own 0.55) is refused; a one-word fix or a
# short clause added to the whole candidate passes. Set by the PI on 2026-09-13
# (audit/rag4ds/responses/PI_decision_edit-line_2026-09-13.md); the human still
# judges the meaning - no lexical line sees a reversed claim (r2 O2).
MIN_EDIT_KEPT = 0.7
MAX_EDIT_OWN = 0.5
QUOTATION_BOUND = ("the entry states the fact in the project's own words; verbatim quotation "
                   "is bounded to a short attributed phrase, never a passage (component 2, NFR5)")


def use_utf8_console(stream=None) -> None:
    """Render to UTF-8 whatever the console's codepage is (AR-rag4ds-049 F4).

    Session 5 of 2026-09-21 aborted after 32 rows when the gate printed a
    candidate carrying U+FB00 (a PDF's "ff" ligature) to a cp1252 console:
    the corpus is full of ligatures and Greek letters, and a validator's
    session must not die on one. Set on the gate's own entry points, not
    only in the shell wrappers that call them, so the documented direct
    invocation is protected too. A stream that cannot be reconfigured (a
    capture object, a pipe already wrapped) is left alone - this hardens
    the console, it never fails the session - but it SAYS SO on stderr
    (AR-rag4ds-049 r2 O9): a console left on its locale codec is a session
    that can still abort on a ligature, and the operator is told."""
    stream = sys.stdout if stream is None else stream
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        print(f"gate: stdout ({type(stream).__name__}) cannot be set to UTF-8; a candidate "
              f"carrying a ligature may abort this session", file=sys.stderr)
        return
    try:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    except (ValueError, OSError, UnsupportedOperation) as exc:  # pragma: no cover
        print(f"gate: stdout refused UTF-8 ({exc}); a candidate carrying a ligature may abort "
              f"this session", file=sys.stderr)


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
    # AR-rag4ds-026 F3: the approver and every recorded author are REGISTERED
    # identities (identities.yaml); OD-1 compares identities, never spellings
    identities: IdentityRegistry | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.approver, str) or not self.approver.strip():
            raise GateError("approver (the expert's recorded identity) must be a non-blank string")
        if self.identities is None:
            raise GateError("an identity registry is required (identities.yaml): the approver "
                            "must be a registered identity (AR-rag4ds-026 F3)")
        try:
            ident = self.identities.resolve(self.approver)
        except IdentityError as exc:
            raise GateError(f"approver refused: {exc}") from exc
        # F12: a registered identity is not thereby an approver - it must hold a
        # gating role (the per-candidate check then matches the role to the home)
        if not any(r in GATING_ROLES for r in ident.roles):
            raise GateError(f"approver refused: {self.approver!r} holds no gating role "
                            f"(has {list(ident.roles)}; gating roles are {GATING_ROLES}) - "
                            f"an authoring identity does not approve (identities.yaml)")
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
        self.current_cid: str | None = None  # the candidate the pending key prompt decides
        self._skipped: set[str] = set()
        self._no_source: set[str] = set()   # candidates whose span was not rendered (F12)
        self._no_display: set[str] = set()  # candidates shown without a working display
        self.summary = Summary(ctx.approver, ctx.clock(), {"pack": pack, "layer": layer})
        self._last_span_text: str | None = None
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
        mechanical cluster close that decision owed is run (F3) - checked
        against the members' home roles BEFORE the transition is applied, so
        a refusal leaves the crash state as found (AR-rag4ds-026 r5 F19). A
        GROUPED member whose head is already decided with no decision of its
        own (a stranded cluster) is recovered here by an identity holding
        every member's home role, provided the head's staging authority
        names the head's own journal decision; otherwise, and for anyone
        else, it is refused by name - an inconsistency the session never
        hides (r6 F21)."""
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
                if to in ("APPROVED", "REJECTED"):
                    # r5 F19: the close this repair owes is checked BEFORE the transition
                    # is applied, so a refusal leaves the crash state exactly as found -
                    # repairable by an identity holding every member's home role
                    self._check_members_home(cid)
                self._transition(cid, to, ref)
                self.console.print(_plain(f"REPAIRED {cid}: decision {ref} had no staging "
                                          f"transition; applied -> {to}"), style="bold red")
                open_ids.discard(cid)
                repaired += 1
                if to in ("APPROVED", "REJECTED"):
                    repaired += self._close_members(cid, ref, repairing=True)
        stranded_heads = sorted({head for cid, (st, head) in self._states.items()
                                 if st == "GROUPED" and head is not None
                                 and self._states.get(head, ("?", None))[0]
                                 in ("APPROVED", "REJECTED", "REJECTED-DUPLICATE")})
        for head in stranded_heads:
            # r5 F19/O25: a head decided with its cluster still open is the state a
            # refused close (or a crash inside the close) leaves behind; it is
            # recovered here by an identity holding every member's home role -
            # _close_members refuses anyone else by name - never hidden, never fatal
            hstate = self._states[head][0]
            try:
                head_ref = self.ctx.staging.head_decision_of(head)
            except StagingError as exc:
                raise GateError(f"stranded member(s) of head {head} ({hstate}): {exc}") from exc
            if hstate == "REJECTED-DUPLICATE":
                raise GateError(
                    f"stranded member(s) under head {head}, itself REJECTED-DUPLICATE: a member "
                    f"of a member - the journal and staging disagree")
            # the head's cited decision must exist and be the head's own: a forged or
            # dangling authority is the inconsistency the crash contract refuses to hide
            n = self._decision_index(head_ref)
            in_range = n is not None and 0 < n <= len(self._decisions)
            cited = self._decisions[n - 1] if in_range else None
            if cited is None or cited.get("candidate_id") != head:
                raise GateError(
                    f"stranded member(s) of head {head} ({hstate}): its staging authority "
                    f"{head_ref!r} does not name a journal decision for {head} - the journal and "
                    f"staging disagree")
            self._check_members_home(head)  # r6 O27: allowed before it is announced
            self.console.print(_plain(f"REPAIR: head {head} is {hstate} with GROUPED members "
                                      f"and no close; closing them now citing {head_ref}"),
                               style="bold red")
            repaired += self._close_members(head, head_ref, repairing=True)
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
        # the rendered passage is what an edit is checked against (AR-rag4ds-027 F2)
        self._last_span_text = span_text if span_state == "rendered" else None
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
            # AR-rag4ds-026 F5: the record itself names the index and model pin
            # behind its conflict display (None when there was no finder)
            "display_provenance": (dict(self.ctx.display_provenance)
                                   if self.ctx.related_finder is not None else None),
        }

    def _count_evidence(self, span_state: str, display: str) -> None:
        if span_state in ("unavailable", "invalid"):
            self.summary.counts["decisions_without_source"] += 1
        if display in ("no-finder", "lookup-failed"):
            self.summary.counts["decisions_without_display"] += 1

    def _decide_entry(self, cid: str, c: dict, kind: str, span_state: str, display: str,
                      related: list[dict]) -> None:
        p = c["proposed"]
        # OD-1 (PI decision 2026-09-12): an entry authored by the approver is gated
        # by the alternate - the author is just another proposer (plan section 3)
        self._refuse_self_approval(cid, c)
        self._check_members_home(cid)  # r4 F18: the close this decision owes
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
        edit: dict | None = None
        if kind == "edit-then-approve":
            content = self.read("edited content: ").strip()
            if not content:
                raise GateError("edit-then-approve needs the edited content")
            edit = self._check_edit(p["content"], content, span_state)
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
            # AR-rag4ds-051 O2: the entry being superseded is replaced, not kept beside
            # the new version, so it cannot be a coexisting conflict of it
            if entry_id in conflicts:
                raise GateError(f"{entry_id!r} is the entry this approval supersedes, not a "
                                f"conflict that coexists with it; leave it out of the "
                                f"related-entries answer")
            cur = self._versions.get(entry_id.lower())
            if cur is None:
                raise GateError(f"supersession target {entry_id!r} is not a current entry")
            # r4 F18: a supersession keeps the entry's HOME - moving an entry between
            # the shared trees and a pack would let a pack validator displace a
            # shared-layer entry; retract it in its home and file anew instead
            prev = self._current_entry(entry_id)
            if prev is not None and prev.get("pack") != c["pack"]:
                raise GateError(
                    f"supersession refused: {entry_id!r} lives in {_home_name(prev.get('pack'))} "
                    f"and the candidate is for {_home_name(c['pack'])} - a supersession keeps "
                    f"the entry's home; retract it there and file anew")
            version, supersedes = cur + 1, {"id": entry_id, "version": cur}
        else:
            version, supersedes = 1, None
        review = self._review(display, related, conflicts, span_state, True)
        if edit is not None:
            review["edit"] = edit
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

    def _check_edit(self, original: str, edited: str, span_state: str) -> dict:
        """AR-rag4ds-027 F2/F7: an edit is an edit, not an authoring road. The
        edited text must keep at least MIN_EDIT_KEPT of the candidate's
        characters (matched / len(candidate)) and the editor's own characters
        must be at most MAX_EDIT_OWN of the entry (1 - matched / len(edited)) -
        beyond either it is a rewrite and is filed as a candidate under the
        editor's name; it must pass the mechanical quotation bound against
        the rendered passage and cover it like any candidate; and the edit is
        recorded on the decision (who, kept, own, the longest verbatim source
        run). The two shares are the record's meaning of the line; the human
        judges the meaning of the words."""
        kept, own = edit_shares(original, edited)
        if kept < MIN_EDIT_KEPT:
            raise GateError(
                f"edit refused: the edited text keeps only {kept:.0%} of the candidate's "
                f"characters (minimum {MIN_EDIT_KEPT:.0%}) - that is a rewrite, not an edit; file "
                f"it as a candidate under your own name and let the gate decide it")
        if own > MAX_EDIT_OWN:
            raise GateError(
                f"edit refused: {own:.0%} of the edited text is the editor's own (maximum "
                f"{MAX_EDIT_OWN:.0%}) - that is authoring, not editing; file it as a candidate "
                f"under your own name and let the gate decide it")
        if span_state != "rendered" or self._last_span_text is None:
            raise GateError("edit refused: the source passage is not rendered, so the edited "
                            "content cannot be checked against it; approve or reject as filed")
        check = quotation_bound_check(edited, self._last_span_text)
        if not check["ok"]:
            raise GateError(f"edit refused: the edited content fails the quotation bound against "
                            f"the passage - {check['reason']}")
        cov = passage_coverage(edited, self._last_span_text)
        if cov < MIN_PASSAGE_COVERAGE:
            raise GateError(f"edit refused: the passage supports only {cov:.0%} of the edited "
                            f"content's words (minimum {MIN_PASSAGE_COVERAGE:.0%})")
        return {"editor": self.ctx.approver, "kept": round(kept, 3), "own": round(own, 3),
                "source_run": int(check["source_run"])}

    def _refuse_self_approval(self, cid: str, c: dict) -> None:
        """OD-1 (PI decision 2026-09-12) and the gating roles (AR-rag4ds-026
        F12), enforced per candidate against REGISTERED identities:
        - the approver must hold the gating role the candidate's home
          requires: od1-designee or od1-alternate for the shared layers,
          validator for a pack;
        - a candidate authored by the approver is refused (the author is
          just another proposer - plan section 3);
        - a candidate authored by the OD-1 designee is approved by the
          od1-alternate and nobody else;
        - an author the registry does not know is refused by name (F3)."""
        registry = self.ctx.identities
        approver = self._check_home_role(c.get("pack"), "approval")
        shared = c.get("pack") == SHARED_PACK
        if c.get("kind") == "mapping-draft":
            author = c.get("author")  # a mapping draft names its author directly
        else:
            extractor = c.get("extractor") or ""
            author = author_of(extractor)
            # AR-rag4ds-027 F3: a candidate that names an author in any other spelling of
            # the prefix is refused, never read as machine-proposed (staging refuses it too)
            if author is None and extractor.strip().casefold().startswith("author"):
                raise GateError(f"approval refused: candidate {cid} carries a malformed author "
                                f"extractor {extractor!r}; an author is written 'author:<name>' "
                                f"exactly")
        if author is None:
            return
        try:
            author_id = registry.resolve(author)
        except IdentityError as exc:
            raise GateError(f"approval refused: candidate {cid} names an author the identity "
                            f"registry does not know - {exc}") from exc
        if author_id.id == approver.id:
            who = "the alternate" if shared else "another validator"
            raise GateError(
                f"approval refused: candidate {cid} was authored by the approver "
                f"({author!r}); the author is just another proposer - skip it and let "
                f"{who} decide it{' (OD-1)' if shared else ''}"
            )
        # AR-rag4ds-026 r3 F15: the designee-needs-alternate arm is OD-1's, and OD-1
        # speaks of the shared layers; a pack candidate the designee authored is
        # gated by any validator who is not its author
        if shared and "od1-designee" in author_id.roles and "od1-alternate" not in approver.roles:
            raise GateError(
                f"approval refused: candidate {cid} was authored by the OD-1 designee "
                f"({author!r}) and only the alternate gates it (OD-1); "
                f"{self.ctx.approver!r} is not the alternate - skip it")

    def _check_home_role(self, pack: str | None, what: str):
        """The approver must hold the gating role of the candidate's or
        entry's home: od1-designee or od1-alternate for the shared layers, a
        validator for a pack (AR-rag4ds-026 F12; on every write path that
        disposes of knowledge - approval, rejection, retraction - r3 F16).
        Returns the approver's identity."""
        approver = self.ctx.identities.resolve(self.ctx.approver)
        shared = pack == SHARED_PACK
        needed = ("od1-designee", "od1-alternate") if shared else ("validator",)
        if not any(r in approver.roles for r in needed):
            raise GateError(
                f"{what} refused: approver {self.ctx.approver!r} holds no gating role for "
                f"{'the shared layers' if shared else 'pack ' + repr(pack)} "
                f"(needs one of {needed}; has {list(approver.roles)}) - identities.yaml")
        # S9 (AR-rag4ds-026 O17/O22): the validator role is scoped to its pack(s)
        if not shared and not approver.validates(pack):
            raise GateError(
                f"{what} refused: approver {self.ctx.approver!r} validates "
                f"{list(approver.packs)}, not pack {pack!r} - identities.yaml")
        return approver

    def _decide_mapping(self, cid: str, c: dict, kind: str) -> None:
        self._refuse_self_approval(cid, c)  # AR-rag4ds-026 O6: the rule covers every kind
        self._check_members_home(cid)  # r5 F19: before the write, like an entry head
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
        self._check_home_role(c.get("pack"), "rejection")  # r3 F16
        self._check_members_home(cid)  # r4 F18: the close this decision owes
        reason = self.read(REASON_PROMPT).strip()
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
        self._check_members_home(head)  # r4 F18 (also on crash repair)
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

    def _decision_index(self, ref: str | None) -> int | None:
        """The 1-based journal line a decision reference names, or None."""
        if not isinstance(ref, str) or not ref.startswith(self.ctx.journal_ref + ":"):
            return None
        tail = ref[len(self.ctx.journal_ref) + 1:]
        return int(tail) if tail.isdigit() else None

    def _check_members_home(self, head: str) -> None:
        """r4 F18: a head decision mechanically closes every GROUPED member
        with a reject-duplicate decision the approver signs, so the approver
        must hold the gating role of each member's own home - checked
        BEFORE the head decision is written, so a refusal leaves no head
        decided with its cluster open. A mixed cluster can only be a legacy
        one: _group refuses to build it."""
        for m in self._members_of(head):
            mc = self._cands[m]["candidate"]
            self._check_home_role(mc.get("pack"), f"the mechanical close of member {m}")

    def _group(self, cid: str) -> None:
        head = self.read(HEAD_PROMPT).strip()
        c = self._cands[cid]["candidate"]
        # r4 F18: grouping is what makes the cluster path reach a candidate, so
        # it takes the member's home role; and a duplicate lives in the same
        # home as its head - a cross-home cluster would let one home's approver
        # dispose of another home's candidate through the head's decision
        self._check_home_role(c.get("pack"), "grouping")
        head_rec = self._cands.get(head)
        if head_rec is not None and head_rec["candidate"].get("pack") != c.get("pack"):
            raise GateError(
                f"grouping refused: {cid} (home {c.get('pack')!r}) and head {head} (home "
                f"{head_rec['candidate'].get('pack')!r}) live in different homes; a duplicate "
                f"is a duplicate within one tree")
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
        self._check_home_role(cur_form.get("pack"), "retraction")  # r3 F16: the fold deletes
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
                # AR-rag4ds-032 F1: the candidate the next key decides, exposed so a
                # scripted reader (tools/gate_sheet.py) can refuse to answer for any
                # other - the loop's continue arms (unrenderable, refused) re-show or
                # skip a candidate without consuming a key
                self.current_cid = cid
                key = self.read(KEY_PROMPT)
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


def edit_shares(original: str, edited: str) -> tuple[float, float]:
    """(kept, own): the share of the candidate's characters that survive in
    the edited text, and the share of the edited text that is not the
    candidate's - both from difflib's matching blocks, so a pasted clause
    and a deletion are each visible."""
    # autojunk=False (AR-rag4ds-027 r3 F7): the default drops frequent characters
    # from matching in any text of 200 characters or more, so a five-word change in
    # a 499-character candidate scored kept 0.56 - a rewrite - where the true share
    # is 0.95; the shares must be the ones the record and the PI's decision state
    matcher = difflib.SequenceMatcher(None, original, edited, autojunk=False)
    matched = sum(b.size for b in matcher.get_matching_blocks())
    kept = matched / len(original) if original else 0.0
    own = 1.0 - (matched / len(edited) if edited else 1.0)
    return kept, max(0.0, own)


def _home_name(pack: str | None) -> str:
    return "the shared layers" if pack == SHARED_PACK else f"pack {pack!r}"


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
    # entry_count (AR-rag4ds-026 O3): a record can say the display was consulted
    # over an empty index, which is indistinguishable from a clean check otherwise
    return {"index_version": m["index_version"], "embedding_model": dict(m["embedding_model"]),
            "min_relevance": m["min_relevance"], "entry_count": m["entry_count"]}


def transcript_path_is_publishable(path: Path) -> str | None:
    """A transcript renders source passages, so it may never land where git
    would commit it (AR-rag4ds-026 F1: the S8 transcript carried 626
    characters of a non-redistributable document into the tree). Returns the
    reason a path is refused, or None when git ignores it or it lies outside
    any work tree."""
    import subprocess

    path = Path(path).resolve()
    probe = subprocess.run(["git", "check-ignore", "-q", "--", str(path)], cwd=str(path.parent),
                           capture_output=True, text=True)
    if probe.returncode == 0:
        return None
    if probe.returncode == 128 and "not a git repository" in probe.stderr:
        return None
    if probe.returncode == 1:
        return (f"transcript path {path} is inside a git work tree and not ignored - it would "
                f"be committed; write it under a gitignored directory such as data/transcripts/")
    return f"git check-ignore failed for {path} (exit {probe.returncode}): {probe.stderr.strip()}"


def main(argv: list[str] | None = None) -> int:
    import argparse

    use_utf8_console()  # AR-rag4ds-049 F4: before anything is rendered
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
    if args.transcript:
        Path(args.transcript).parent.mkdir(parents=True, exist_ok=True)
        refused = transcript_path_is_publishable(Path(args.transcript))
        if refused:
            print(f"gate: {refused}", file=sys.stderr)
            return 1
    try:
        identities = IdentityRegistry.load(root / IDENTITIES_FILE)
        identities.resolve(args.approver)
    except IdentityError as exc:
        print(f"gate: refused - {exc}", file=sys.stderr)
        return 1
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
        identities=identities,
    )
    console = Console(record=True, markup=False, highlight=False)
    session = GateSession(ctx, console, make_echo_reader(console, input), pack=args.pack,
                          layer=args.layer)
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


def make_echo_reader(console: Console, read: Callable[[str], str]) -> Callable[[str], str]:
    """The live gate's read callback: what is typed at every prompt (keys,
    reasons, edited text, ids) is echoed into the recorded console, so the
    transcript witnesses the input and not only the outcome (AR-rag4ds-033
    r1 O1; AR-rag4ds-034 F4)."""
    def read_and_echo(prompt: str) -> str:
        answer = read(prompt)
        console.print(_plain(f"{prompt}{answer}"))
        return answer
    return read_and_echo


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
    "transcript_path_is_publishable", "_timestamp_valid", "use_utf8_console",
]
