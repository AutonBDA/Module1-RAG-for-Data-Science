"""Hand-authored candidates (plan S8; section 3 - "the author is just another
proposer"; I2 has no exception).

A draft is a human-edited YAML file (section 8) listing entries an author
proposes for the shared layers - or, when the draft names a top-level
``pack``, for that one domain pack's domain layer (S9e part 7, 2026-09-24:
the plan never confined authorship to the shared layers - section 3, 6.4
item 3, and R4's "more hand-authoring through the same gate" - the S8 code
did, because S8 was the shared-layer increment). A pack draft's items are
all domain-layer, a shared draft's are all general or task, which is the
entry model's own FR1 rule; the caller supplies the sources, and for a pack
draft those are the pack's own catalogued documents, so an item can cite
nothing outside its pack. Each draft item names the catalogued source
document it restates and LOCATES the passage it restates in one of two ways:

- ``anchor`` - a short verbatim excerpt (at most ANCHOR_MAX_CHARS) of the
  document that the tool turns into the character offsets of the WHOLE
  PARAGRAPH the excerpt sits in (blank-line delimited): an anchor locates a
  passage, it does not delimit one (AR-rag4ds-027 F1). Allowed ONLY for a
  document the catalog records as redistributable, because the draft is a
  versioned file and an anchor puts the document's own words into git
  (NFR5, NFR6; AR-rag4ds-026 F1).
- ``span`` - the offsets of the passage plus the sha256 of the text they
  delimit (``{start, end, sha256}``); the tool re-derives the passage from
  the local document and refuses when it does not hash to the record. This
  is the only locator a non-redistributable document admits: nothing of the
  document is copied anywhere.

Whichever locator, the located text must BE the passage the content
restates: at least MIN_PASSAGE_COVERAGE of the content's words (NFKC,
casefolded, four letters or more, counted once each) must occur in it, or
the item is refused as a fragment (AR-rag4ds-027 F1: the S8 spans covered a
third of their entries' words; the passages they sit in cover four fifths);
and it is at most MAX_PASSAGE_CHARS long, or the item is refused as a
section rather than a passage (F6).

Either way the candidate carries the span as hash + offsets, never text; the
proposed content is checked against that passage by the same quotation bound
extraction uses (component 2, NFR5), and the candidate is filed into staging
with extractor "author:<name>" and a run-manifest reference, so it reaches
the gate with the same evidence an extracted candidate carries. An anchor
that is absent, ambiguous, over-long or mid-word, a span that does not hash,
content over the bound, an unregistered author, or a draft that does not
validate fails loudly and files nothing.

The authoring run writes one manifest (NFR2): code commit, timestamp, the
draft file's hash, the author, the source documents, the count filed, and
the candidate ids - so every hand-authored candidate names the run that
proposed it. Order (AR-rag4ds-026 F8): every item is built and checked, every
candidate is checked against what staging already holds, then the
candidates are filed, then the manifest is appended - a refusal at any point
leaves nothing behind, and a draft whose candidates are already in staging
(same source, span, content and author) is refused rather than filed twice.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml

from rag4ds.auditlog.journal import Journal
from rag4ds.extract.strategy import quotation_bound_check
from rag4ds.gate.decisions import _timestamp_valid
from rag4ds.gate.identities import IdentityError, IdentityRegistry
from rag4ds.model.entry import ENTRY_ID_RE, ENTRY_TYPES, LAYERS, SHARED_PACK
from rag4ds.staging.store import StagingStore, candidate_id, validate_candidate

DRAFT_SCHEMA_VERSION = 1
ITEM_FIELDS = {"type", "layer", "content", "source_reference", "source"}
LOCATOR_FIELDS = {"anchor", "span"}
SPAN_FIELDS = {"start", "end", "sha256"}
ANCHOR_MAX_CHARS = 200  # AR-rag4ds-026 O4: an anchor locates a passage, it does not carry one
MIN_PASSAGE_COVERAGE = 0.5  # AR-rag4ds-027 F1: the located text must be the passage restated
# AR-rag4ds-027 F6: a passage is a paragraph, not a section or a document - the
# largest live one is 926 characters; a whole-document span or an anchor in a
# document without blank lines would otherwise pass every check
MAX_PASSAGE_CHARS = 1500
AUTHOR_RE = re.compile(r"[A-Za-z][A-Za-z .'-]{1,79}")


class DraftError(ValueError):
    """The draft, an item, a locator or a bound check failed - nothing is filed."""


@dataclass(frozen=True)
class SourceDoc:
    """A catalogued document as the draft road sees it: its content hash, its
    UTF-8 text, whether the catalog records it as redistributable (the fact
    that decides which locator an item may use), and the pack whose catalogue
    it is in - "shared" for the shared corpus. The pack is REQUIRED, never
    defaulted (AR-rag4ds-051 F1): the road refuses an item citing a document
    of another pack, and a caller that forgot to say where a document came
    from must fail, not be taken for the shared road."""
    hash: str
    text: str
    redistributable: bool
    pack: str


def author_extractor(author: str) -> str:
    """The extractor identity a hand-authored candidate carries."""
    return f"author:{author}"


def author_of(extractor: str) -> str | None:
    """The author named by an extractor identity, or None for a machine strategy."""
    return extractor[len("author:"):] if extractor.startswith("author:") else None


def passage_sha256(passage: str) -> str:
    return hashlib.sha256(passage.encode("utf-8")).hexdigest()


def content_words(text: str) -> set[str]:
    """The words a coverage measure counts: NFKC-normalised (a PDF's "fi"
    ligature is "fi"), casefolded, letters only, four or more, each once."""
    norm = unicodedata.normalize("NFKC", text).casefold()
    return {w for w in re.findall(r"[^\W\d_]+", norm) if len(w) >= 4}


def passage_coverage(content: str, passage: str) -> float:
    """The share of the content's words that occur in the passage (types, not
    tokens); 0.0 when the content has no countable word."""
    words = content_words(content)
    if not words:
        return 0.0
    have = content_words(passage)
    return len(words & have) / len(words)


def paragraph_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """The blank-line delimited paragraph containing [start, end), with
    leading and trailing whitespace trimmed."""
    ps = text.rfind("\n\n", 0, start)
    ps = 0 if ps < 0 else ps + 2
    pe = text.find("\n\n", end)
    pe = len(text) if pe < 0 else pe
    while ps < pe and text[ps].isspace():
        ps += 1
    while pe > ps and text[pe - 1].isspace():
        pe -= 1
    return ps, pe


def load_draft(path: Path, identities: IdentityRegistry | None = None) -> dict:
    """Read and validate a draft. With a registry, the author must be a
    registered identity (F3): an unregistered spelling is refused here, so it
    can never be recorded as an extractor."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DraftError(f"{path}: not a readable YAML draft: {exc}") from exc
    problems = []
    if not isinstance(data, dict):
        raise DraftError(f"{path}: a draft is a mapping")
    if data.get("schema") != DRAFT_SCHEMA_VERSION:
        problems.append(f"schema must be {DRAFT_SCHEMA_VERSION}")
    author = data.get("author")
    if not isinstance(author, str) or not AUTHOR_RE.fullmatch(author.strip()):
        problems.append("author must be a person's name (letters, spaces, . ' -; 2-80 chars)")
    elif identities is not None:
        try:
            identities.resolve(author.strip())
        except IdentityError as exc:
            problems.append(f"author: {exc}")
    # S9e part 7: a draft may name ONE domain pack; absent, it is a shared draft
    pack = data.get("pack", SHARED_PACK)
    if not isinstance(pack, str) or not ENTRY_ID_RE.fullmatch(pack):
        # the entry model's own rule for a pack code: it names a tree directory (9.1)
        problems.append(f"pack {pack!r} must be a pack code (lowercase [a-z0-9-], 1-64 chars)")
    elif "pack" in data and pack == SHARED_PACK:
        problems.append(f"pack {SHARED_PACK!r} is the default and is not named: omit the field "
                        f"for a shared draft")
    item_layers = ("general", "task") if pack == SHARED_PACK else ("domain",)
    items = data.get("items")
    if not isinstance(items, list) or not items:
        problems.append("items must be a non-empty list")
        items = []
    for i, item in enumerate(items):
        where = f"items[{i}]"
        if not isinstance(item, dict):
            problems.append(f"{where}: must be a mapping")
            continue
        missing = sorted(ITEM_FIELDS - set(item))
        unknown = sorted(set(item) - ITEM_FIELDS - LOCATOR_FIELDS)
        locators = sorted(set(item) & LOCATOR_FIELDS)
        if missing:
            problems.append(f"{where}: missing {missing}")
        if unknown:
            problems.append(f"{where}: unknown field(s) {unknown}")
        if len(locators) != 1:
            problems.append(f"{where}: exactly one locator (anchor or span) is required, "
                            f"got {locators}")
        if missing or unknown or len(locators) != 1:
            continue
        if item["type"] not in ENTRY_TYPES:
            problems.append(f"{where}: type {item['type']!r} not one of {ENTRY_TYPES}")
        if item["layer"] not in item_layers:
            problems.append(
                f"{where}: layer {item['layer']!r} must be general or task (domain knowledge "
                f"lives in a pack: name the pack in the draft)" if pack == SHARED_PACK else
                f"{where}: layer {item['layer']!r} must be domain in a draft for pack "
                f"{pack!r} (general and task knowledge lives in the shared layers, FR1)")
        for key in ("content", "source_reference", "source"):
            if not isinstance(item[key], str) or not item[key].strip():
                problems.append(f"{where}: {key} must be a non-blank string")
        if "anchor" in item:
            a = item["anchor"]
            if not isinstance(a, str) or not a.strip():
                problems.append(f"{where}: anchor must be a non-blank string")
            elif len(a) > ANCHOR_MAX_CHARS:
                problems.append(f"{where}: anchor is {len(a)} chars; at most {ANCHOR_MAX_CHARS} "
                                f"- an anchor locates a passage, it does not carry one")
        else:
            problems.extend(_span_problems(item["span"], where))
    if problems:
        raise DraftError(
            f"{path}: draft invalid ({len(problems)} problem(s)): " + "; ".join(problems))
    assert set(LAYERS) >= {"general", "task"}
    return data


def _span_problems(span, where: str) -> list[str]:
    if not isinstance(span, dict) or set(span) != SPAN_FIELDS:
        return [f"{where}: span must be a mapping with exactly the keys {sorted(SPAN_FIELDS)}"]
    out = []
    start, end, digest = span["start"], span["end"], span["sha256"]
    if (not isinstance(start, int) or not isinstance(end, int) or isinstance(start, bool)
            or isinstance(end, bool) or start < 0 or end <= start):
        out.append(f"{where}: span.start/end must be integers with 0 <= start < end")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        out.append(f"{where}: span.sha256 must be the 64-hex sha256 of the passage's UTF-8 bytes")
    return out


def locate_anchor(text: str, anchor: str, where: str) -> tuple[int, int]:
    """Character offsets of the anchor in the document text. Matching is
    exact on characters, whole-word at both ends (an anchor never matches
    inside a longer word - AR-rag4ds-026 F6) and tolerant of whitespace: any
    run of whitespace in the anchor matches any run in the document,
    newlines included, because text extracted from a PDF wraps lines
    mid-sentence. Exactly one occurrence is required: none and the passage
    is not in this document; two and the span would be ambiguous. The
    offsets returned are those of the document's own text, so the span
    renders verbatim at the gate."""
    tokens = anchor.split()
    if not tokens:
        raise DraftError(f"{where}: anchor is blank")
    pattern = r"\s+".join(re.escape(tok) for tok in tokens)
    if re.match(r"\w", tokens[0]):
        pattern = r"(?<!\w)" + pattern
    if re.search(r"\w\Z", tokens[-1]):
        pattern = pattern + r"(?!\w)"
    matches = list(re.finditer(pattern, text))
    if not matches:
        raise DraftError(f"{where}: anchor not found in the source document as whole words: "
                         f"{anchor[:60]!r}")
    if len(matches) > 1:
        raise DraftError(f"{where}: anchor occurs {len(matches)} times in the source document "
                         f"- make it unique: {anchor[:60]!r}")
    return matches[0].start(), matches[0].end()


def locate_passage(text: str, anchor: str, where: str) -> tuple[int, int]:
    """The paragraph an anchor sits in (AR-rag4ds-027 F1): the anchor is
    located exactly as before, then widened to the enclosing blank-line
    delimited paragraph, which is the passage the gate renders and the bound
    and coverage are checked against."""
    s, e = locate_anchor(text, anchor, where)
    return paragraph_bounds(text, s, e)


def locate_span(text: str, span: dict, where: str) -> tuple[int, int]:
    """Offsets of a span locator, admitted only when the document's own text
    at those offsets hashes to the recorded passage digest - so a span is as
    tied to its document as an anchor is, without copying a character."""
    start, end = span["start"], span["end"]
    if end > len(text):
        raise DraftError(f"{where}: span [{start}, {end}) runs past the end of the document "
                         f"({len(text)} characters)")
    got = passage_sha256(text[start:end])
    if got != span["sha256"]:
        raise DraftError(f"{where}: the document's text at [{start}, {end}) hashes to "
                         f"{got[:16]}..., not the recorded {span['sha256'][:16]}... "
                         f"(wrong offsets, or a different document)")
    return start, end


def build_candidates(draft: dict, sources: dict[str, SourceDoc], run_ref: str) -> list[dict]:
    """Entry candidates for every draft item. sources maps a draft's source
    key to its SourceDoc. The quotation bound is checked against the located
    passage - not the whole document - because that passage is what the
    content restates; a refusal names the item. An anchor on a
    non-redistributable document is refused (NFR6): a span is the locator
    for those."""
    out = []
    extractor = author_extractor(draft["author"].strip())
    pack = draft.get("pack", SHARED_PACK)  # S9e part 7: load_draft validated it
    for i, item in enumerate(draft["items"]):
        where = f"items[{i}] ({item['content'][:40]!r})"
        if item["source"] not in sources:
            raise DraftError(f"{where}: source {item['source']!r} is not a catalogued document "
                             f"of this run (known: {sorted(sources)})")
        doc = sources[item["source"]]
        # AR-rag4ds-051 F1: the pack boundary is the road's, not one caller's - a
        # draft's items cite only documents of the draft's own pack (FR3: a pack is a
        # self-contained set of domain entries and its corpus references), and a
        # shared draft cites only the shared corpus
        if doc.pack != pack:
            raise DraftError(
                f"{where}: source {item['source']!r} is a document of pack {doc.pack!r}, and "
                f"this is a draft for pack {pack!r}; an item cites only its own pack's "
                f"catalogued documents (FR3)")
        # O3 of the same report: a pack entry's source reference names its own
        # document, in the form every pack entry already uses
        own_ref = f"corpus:{pack}/{item['source']}"
        if pack != SHARED_PACK and item["source_reference"].strip() != own_ref:
            raise DraftError(
                f"{where}: source_reference {item['source_reference']!r} must be "
                f"'corpus:{pack}/{item['source']}', the document the item cites")
        if "anchor" in item:
            if not doc.redistributable:
                raise DraftError(
                    f"{where}: an anchor copies the document's words into this versioned draft, "
                    f"and source {item['source']!r} is catalogued as NOT redistributable (NFR6); "
                    f"locate the passage with a span (start, end, sha256) instead")
            start, end = locate_passage(doc.text, item["anchor"], where)
        else:
            start, end = locate_span(doc.text, item["span"], where)
        passage = doc.text[start:end]
        if len(passage) > MAX_PASSAGE_CHARS:
            raise DraftError(
                f"{where}: the located text is {len(passage)} characters, more than "
                f"MAX_PASSAGE_CHARS ({MAX_PASSAGE_CHARS}) - a passage is a paragraph, not a "
                f"section (AR-rag4ds-027 F6); narrow the span, or add the paragraph break the "
                f"document lacks")
        check = quotation_bound_check(item["content"], passage)
        if not check["ok"]:
            raise DraftError(f"{where}: content fails the quotation bound against its passage: "
                             f"{check['reason']}")
        cov = passage_coverage(item["content"], passage)
        if cov < MIN_PASSAGE_COVERAGE:
            raise DraftError(
                f"{where}: the located text supports only {cov:.0%} of the content's words "
                f"(minimum {MIN_PASSAGE_COVERAGE:.0%}) - the locator names a fragment, not the "
                f"passage the content restates (AR-rag4ds-027 F1); widen the span")
        candidate = {
            "kind": "entry",
            "proposed": {"type": item["type"], "content": item["content"].strip(),
                         "source_reference": item["source_reference"].strip()},
            "layer": item["layer"],
            "pack": pack,
            "source_hash": doc.hash,
            "span_start": start,
            "span_end": end,
            "extractor": extractor,
            "run_manifest": run_ref,
        }
        validate_candidate(candidate)
        out.append(candidate)
    return out


def _content_key(c: dict) -> tuple:
    return (c["source_hash"], c["span_start"], c["span_end"], c["extractor"],
            c["proposed"]["type"], c["proposed"]["content"], c["proposed"]["source_reference"])


def already_filed(candidates: list[dict], store: StagingStore) -> list[str]:
    """For each candidate that staging already holds with the same source,
    span, author and proposed content (any run): a line naming both."""
    return [f"{c['proposed']['content'][:40]!r} is already filed as {prior}"
            for c, prior in filed_before(candidates, store)]


def filed_before(candidates: list[dict], store: StagingStore) -> list[tuple[dict, str]]:
    """(candidate, prior id) for every candidate staging already holds with
    the same source, span, author and proposed content."""
    existing: dict[tuple, str] = {}
    for cid, rec in store.candidates().items():
        c = rec["candidate"]
        # a REJECTED proposal may be proposed again (rejected history stays, the
        # re-filing is a new proposal - PI decision O9, 2026-09-13); anything
        # pending, grouped or approved blocks a duplicate
        if c.get("kind") == "entry" and not store.state_of(cid).startswith("REJECTED"):
            existing[_content_key(c)] = cid
    out = []
    for c in candidates:
        prior = existing.get(_content_key(c))
        if prior is not None:
            out.append((c, prior))
    return out


def file_draft(draft_path: Path, sources: dict[str, SourceDoc], store: StagingStore,
               runs: Journal, code_commit: str, timestamp: str,
               identities: IdentityRegistry | None = None,
               allow_partial: bool = False, pack: str = SHARED_PACK) -> dict:
    """Load, build, check and file a draft; write the authoring manifest.
    Returns the manifest. Builds and checks EVERY item, and checks every
    candidate against what staging already holds, before filing any - so a
    draft with one bad or already-filed item files nothing and writes no
    manifest; the manifest is appended only after the last candidate is
    filed (F8). With allow_partial (a re-filing after the locators changed,
    AR-rag4ds-027 F1), items staging already holds unchanged are skipped by
    id - recorded in the manifest's skipped_already_filed - and only the
    new items are filed; a draft with nothing new still files nothing."""
    if not _timestamp_valid(timestamp):
        raise DraftError(f"timestamp {timestamp!r} must be 'YYYY-MM-DD HH:MM[:SS] JST'")
    if not isinstance(code_commit, str) or not re.fullmatch(r"[0-9a-f]{7,40}", code_commit):
        raise DraftError("code_commit must be a 7-40 hex git commit id (NFR2)")
    draft_path = Path(draft_path)
    draft = load_draft(draft_path, identities)
    # AR-rag4ds-051 F1: each road files ONE pack - the shared seed (the default
    # caller) the shared layers, tools/pack_author.py its named pack - and a
    # draft for another is refused here, before anything is built
    named = draft.get("pack", SHARED_PACK)
    if named != pack:
        raise DraftError(
            f"{draft_path.name} is a draft for pack {named!r} and this road files pack {pack!r}; "
            + ("a pack draft is filed by tools/pack_author.py" if pack == SHARED_PACK else
               "a shared draft is filed by tools/s8_seed.py"))
    draft_hash = hashlib.sha256(draft_path.read_bytes()).hexdigest()
    # the run id carries the journal's sequence number as well as draft, commit and
    # timestamp: two runs of one draft in one minute at one commit once shared a
    # run id (2026-09-13 16:05 JST, manifests 8 and 9 of runs/authoring.jsonl), and
    # a candidate's run_manifest reference must name one manifest
    prior_runs = runs.read_all() if runs.path.exists() else []
    sequence = len(prior_runs)
    run_id = hashlib.sha256(
        f"{draft_hash}:{code_commit}:{timestamp}:{sequence}".encode()).hexdigest()[:16]
    if any(r.get("run_id") == run_id for r in prior_runs):
        raise DraftError(f"{draft_path.name}: run id {run_id} is already in {runs.path.name}; "
                         f"refusing to write a second manifest under it")
    run_ref = f"{runs.path.as_posix().rsplit('/', 1)[-1]}#{run_id}"
    candidates = build_candidates(draft, sources, run_ref)
    prior = filed_before(candidates, store)
    skipped: list[str] = []
    if prior and not allow_partial:
        dupes = already_filed(candidates, store)
        raise DraftError(f"{draft_path.name}: {len(dupes)} item(s) already in staging - a draft "
                         f"files once; nothing filed: " + "; ".join(dupes))
    if prior:
        skipped = [pid for _, pid in prior]
        held = {id(c) for c, _ in prior}
        candidates = [c for c in candidates if id(c) not in held]
        if not candidates:
            raise DraftError(f"{draft_path.name}: every item is already in staging "
                             f"({', '.join(skipped)}); nothing filed")
    ids = [candidate_id(c) for c in candidates]
    manifest = {
        "kind": "authoring",
        "schema": DRAFT_SCHEMA_VERSION,
        "run_id": run_id,
        "author": draft["author"].strip(),
        "draft": draft_path.name,
        "draft_sha256": draft_hash,
        "sources": {k: v.hash for k, v in sorted(sources.items())},
        "candidates": ids,
        "count": len(ids),
        "sequence": sequence,
        "skipped_already_filed": skipped,
        "code_commit": code_commit,
        "timestamp": timestamp,
        "tool": "rag4ds.author.file_draft",
    }
    for c in candidates:
        store.file_candidate(c, timestamp)
    runs.append(manifest)
    return manifest
