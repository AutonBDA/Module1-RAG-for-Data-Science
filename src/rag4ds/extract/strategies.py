"""The two v1 extraction strategies (plan S7; 6.2).

RuleStrategy ships the three capabilities 6.2 names (AR-rag4ds-022 F3):

  definitions   "<term> is/are ...", "<term>: ...", "<term> refers to /
                means / denotes ..." over sentence-split text -> concept
  headings      a heading line (markdown ATX, or a short title-cased line
                with no terminal punctuation) plus the first sentence
                under it -> concept, scoped by the section it heads
  tables        pipe-delimited rows under a header row -> fact, one per
                data row, composed as "<row label> <column>: <value>"

and proposes two entry types (concept and fact) rather than one. Its
sentence splitting is spaCy's blank English pipeline with the sentencizer,
which needs no language model and runs offline inside the hardware
envelope (NFR4). A language model IS pinned since VT-3b
(extract/language_model.py: en_core_web_sm 3.8.0, hash-verified at load)
but the rule strategy does not split with it, by measurement: on the S7
corpus its boundaries equal the sentencizer's and on a table sample they
are worse (docs/vt3_pins.md). POS, noun chunks and NER are therefore
available to the next extraction increment and unused here - declared,
not hidden. Precision over recall (6.2): only clear shapes are
matched, and every composed content is kept short so the runner's
quotation bound passes it on merit rather than by luck.

EmbeddingStrategy proposes related-concept candidates by similarity to
caller-supplied seeds, using the pinned embedding model - the same model
family the runner uses to group near-duplicates. Its describe() carries
that model's pin, so a run whose only model lives inside the strategy is
still reconstructable from the manifest (F2).

Neither strategy writes the knowledge base (FR5); both return proposals.
"""

from __future__ import annotations

import hashlib
import re

from rag4ds.extract.strategy import (
    Document, ExtractError, Proposal, SourceIndex, quotation_bound_check,
)

_GLOSS_WORDS = 6  # a short gloss, well inside the quotation bound with the term
_MAX_CELL_WORDS = 5

# "<Term> is/are [a/an/the] ..."  |  "<Term>:  ..."  |  "<Term> refers to/means ..."
_DEF_PATTERNS = (
    re.compile(r"^(?P<term>[^\W\d_][\w -]{1,48}?)\s+(?:is|are)\s+(?:a|an|the\s+)?(?P<gloss>.+)$"),
    re.compile(r"^(?P<term>[^\W\d_][\w -]{1,48}?)\s*:\s*(?P<gloss>.+)$"),
    re.compile(r"^(?P<term>[^\W\d_][\w -]{1,48}?)\s+(?:refers to|means|denotes)\s+(?P<gloss>.+)$"),
)
_ATX_HEADING = re.compile(r"^(#{1,6})\s+(?P<title>.+?)\s*#*$")
_TABLE_ROW = re.compile(r"^\s*\|(?P<cells>.+)\|\s*$")
_TABLE_RULE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")

# ----- tuning (S9c part 5; opt-in, recorded in the manifest only when on) -----
# The shapes the robotics validator's rejection reasons pointed at
# (docs/s9_robotics_pack.md 10.4, 11.x; PI_decision_headings-A_2026-09-15.md):
# a checklist or procedure line read as a definition ("Check the bolts -
# properly tightened"), a pronoun or adverb read as a term ("They - either on
# or off"), an error code read as a term ("C296A4 Phase 1 - not connected"),
# a gloss cut mid-clause ("Note - Be very careful when removing the"), and
# whitespace variants of one string filed as distinct proposals.
# tune-v2 (S9e, 2026-09-21): two rules a new document class exposed - the
# first filed tuned run over a conference paper proposed 21 of its 24
# candidates from the paper's reference list, bare DOIs read as
# definitions ("https - doi org 10 ..."). (1) CITATION: a line carrying a
# URL or a DOI is a citation, not a statement the pack restates. (2) BACK
# MATTER: nothing at or past the paper's references heading is proposed at
# all. tune-v1 manifests keep their ids; a tune-v2 manifest names its
# version, so the filed record says which rules produced it.
_TUNING_VERSION = "tune-v2"
_CITATION_LINE = re.compile(r"https?://|\bwww\.\S|\bdoi\b|\b10\.\d{4,9}/\S", re.IGNORECASE)
# a references heading is a line that is nothing but the word, optionally
# numbered ("7 References", "REFERENCES", "Bibliography:"), and it marks
# back matter only in the second half of the document - a manual may cite
# a references section early, and a heading there is not the end of the
# text. The LAST such line wins. Design limit, stated: a paper whose
# reference list begins before its midpoint keeps its citations, which the
# CITATION rule still refuses line by line.
_REFERENCES_HEADING = re.compile(
    r"^[ \t]*(?:\d+(?:\.\d+)*[.)]?\s+)?(?:references?|bibliography|reference list|"
    r"literature cited|works cited)[ \t]*:?[ \t]*$", re.IGNORECASE | re.MULTILINE)
_LEAD_STOP = frozenset((
    "check", "inspect", "ensure", "verify", "make", "note", "notes", "warning", "caution",
    "danger", "remove", "replace", "install", "power", "press", "turn", "use", "see", "refer",
    "tighten", "loosen", "connect", "disconnect", "confirm", "click", "select", "open", "close",
    "wait", "repeat", "perform", "follow", "read", "do", "there", "they", "it", "this", "these",
    "those", "we", "you", "to", "more", "most", "not", "all", "example", "symbols", "also", "then",
    "when", "if", "as", "for", "after", "before",
))
_CODE_PREFIX = re.compile(r"^[A-Z]{1,3}\d{1,4}[A-Z]?\d*\b")
_MIN_GLOSS_WORDS = 2
_MAX_GLOSS_WORDS = 12  # a cut gloss may extend to the next clause boundary up to here
_CLAUSE_BOUNDARY = re.compile(r"^\s*[,;:.)\]]")


def _collapse(text: str) -> str:
    return " ".join(text.split())


def back_matter_start(text: str) -> int | None:
    """The offset at which a document's back matter begins: the LAST
    references heading that lies in the second half of the text, or None.
    Public so the extraction tool and the tests can state, per document,
    where the rule bites."""
    half = len(text) // 2
    starts = [m.start() for m in _REFERENCES_HEADING.finditer(text) if m.start() >= half]
    return starts[-1] if starts else None


def _term_refused(term: str) -> str | None:
    """The tuning's verdict on a term: None (kept) or the reason."""
    words = term.split()
    if not words:
        return "empty"
    if words[0].casefold() in _LEAD_STOP:
        return "lead-stop"
    if _CODE_PREFIX.match(term):
        return "code-prefix"
    return None


def _tune_gloss(gloss_raw: str, limit: int) -> tuple[str | None, list[str]]:
    """The tuning's verdict on a gloss that _short() would cut to `limit`
    words: (reason, candidate glosses). Too short to say anything ->
    refused. Within the limit -> the gloss itself. Longer -> every clause
    boundary at or after the limit up to _MAX_GLOSS_WORDS words yields a
    candidate, longest first, and the `limit`-word cut comes last as the
    fallback (the shape the validator's edit of the pack's first approved
    entry started from); the caller keeps the first that the quotation
    bound admits (AR-rag4ds-038 F1). No boundary by _MAX_GLOSS_WORDS ->
    refused as cut mid-clause."""
    words = list(re.finditer(r"[\w-]+", gloss_raw))
    alpha = [m.group() for m in words if any(ch.isalpha() for ch in m.group())]
    if len(words) < _MIN_GLOSS_WORDS or not alpha:
        return "gloss-too-short", []
    if len(words) <= limit:
        return None, [" ".join(m.group() for m in words)]
    ends = [n for n in range(limit, min(len(words), _MAX_GLOSS_WORDS) + 1)
            if n == len(words) or _CLAUSE_BOUNDARY.match(gloss_raw[words[n - 1].end():])]
    if not ends:
        return "gloss-cut-mid-clause", []
    candidates = [" ".join(m.group() for m in words[:n]) for n in sorted(ends, reverse=True)]
    candidates.append(" ".join(m.group() for m in words[:limit]))
    return None, candidates


def _blank_sentencizer():
    import spacy
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def _short(text: str, n: int) -> str:
    return " ".join(re.findall(r"[\w-]+", text)[:n])


def _drop_article(term: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", term.strip(), flags=re.IGNORECASE)


def _line_spans(text: str):
    """(line, start, end) for every line, offsets into text."""
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        lead = len(stripped) - len(stripped.lstrip())
        s = pos + lead
        e = pos + len(stripped.rstrip()) if stripped.strip() else s
        yield stripped.strip(), s, max(e, s)
        pos += len(line)


class RuleStrategy:
    """Definitions, heading/section heuristics and table extraction (6.2)."""

    name = "rule-v1"

    def __init__(self, definitions: bool = True, headings: bool = True, tables: bool = True,
                 tuning: bool = False):
        self.definitions, self.headings, self.tables = definitions, headings, tables
        self.tuning = bool(tuning)
        self._nlp = None
        self._index = None
        self._back_matter: int | None = None  # per document, when tuning (tune-v2)
        self.refused: list[dict] = []  # what the tuning refused in the last propose() (NFR9)

    def describe(self) -> dict:
        params = {"definitions": self.definitions, "headings": self.headings,
                  "tables": self.tables, "gloss_words": _GLOSS_WORDS,
                  "max_cell_words": _MAX_CELL_WORDS,
                  "sentencizer": "spacy.blank('en') + sentencizer (the VT-3b "
                                 "language model is pinned but not used for "
                                 "splitting, by measurement)"}
        if self.tuning:
            # only when on, so a run without it describes itself exactly as before
            # and the filed runs of S9c part 1 keep their extraction ids
            params["tuning"] = {"version": _TUNING_VERSION, "lead_stop": sorted(_LEAD_STOP),
                                "code_prefix": _CODE_PREFIX.pattern,
                                "min_gloss_words": _MIN_GLOSS_WORDS,
                                "max_gloss_words": _MAX_GLOSS_WORDS,
                                "cut_gloss": "a gloss longer than gloss_words runs to the first "
                                             "clause boundary at or after it, up to "
                                             "max_gloss_words; none by then -> refused",
                                "whitespace": "collapsed in every composed content; exact "
                                              "duplicates after collapse dropped per document",
                                # tune-v2
                                "citation_line": _CITATION_LINE.pattern,
                                "back_matter": "nothing is proposed at or past the last "
                                               "references heading in the second half of the "
                                               "document (regex references_heading)",
                                "references_heading": _REFERENCES_HEADING.pattern}
        return {"name": self.name, "model": None, "params": params}

    def _structural_refusal(self, raw: str, s: int) -> str | None:
        """tune-v2: the two rules that apply before any term or gloss is
        read - back matter first (the whole line is out of scope), then a
        citation line (a URL or a DOI on it)."""
        if self._back_matter is not None and s >= self._back_matter:
            return "back-matter"
        if _CITATION_LINE.search(raw):
            return "citation"
        return None

    def _refuse(self, capability: str, why: str, text: str, s: int, e: int) -> None:
        # AR-rag4ds-038 F2: no source text reaches the manifest - the refusal is
        # identified by its span and the sha256 of the line it refused, the shape
        # the runner's suppression record already uses
        self.refused.append({"capability": capability, "reason": why,
                             "line_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                             "span_start": s, "span_end": e})

    def _pipe(self):
        if self._nlp is None:
            self._nlp = _blank_sentencizer()
        return self._nlp

    # ----- capability 1: definitions -----

    def _definitions(self, document: Document) -> list[Proposal]:
        out, seen = [], set()
        for sent in self._pipe()(document.text).sents:
            raw = sent.text.strip()
            if not raw or raw.startswith("#") or raw.startswith("|"):
                continue
            lead = len(sent.text) - len(sent.text.lstrip())
            s = sent.start_char + lead
            e = s + len(raw)
            body = raw.rstrip(".")
            for pat in _DEF_PATTERNS:
                m = pat.match(body)
                if not m:
                    continue
                term = _drop_article(m.group("term"))
                gloss = _short(m.group("gloss"), _GLOSS_WORDS)
                if not term or not gloss or (s, e) in seen:
                    break
                seen.add((s, e))
                if self.tuning:
                    term = _collapse(term)
                    why = self._structural_refusal(raw, s) or _term_refused(term)
                    if not why:
                        why, options = _tune_gloss(m.group("gloss"), _GLOSS_WORDS)
                    if not why:
                        # bound-aware (AR-rag4ds-038 F1): the longest clause-bounded gloss
                        # the quotation bound admits; none admitted -> refused
                        gloss = next((g for g in options
                                      if quotation_bound_check(f"{term} - {g}", self._index)["ok"]),
                                     None)
                        if gloss is None:
                            why = "gloss-over-bound"
                    if why:
                        self._refuse("definitions", why, raw, s, e)
                        break
                out.append(Proposal("concept", f"{term} - {gloss}", s, e))
                break
        return out

    # ----- capability 2: heading and section heuristics -----

    def _headings(self, document: Document) -> list[Proposal]:
        lines = list(_line_spans(document.text))
        out = []
        for i, (line, s, e) in enumerate(lines):
            if not line:
                continue
            m = _ATX_HEADING.match(line)
            if m:
                title = m.group("title").strip()
            elif (len(line.split()) <= 6 and not line.endswith((".", ":", "|"))
                  and line[:1].isupper() and not _TABLE_ROW.match(line)
                  and any(c.isalpha() for c in line)):
                title = line
            else:
                continue
            # the first non-empty, non-heading, non-table line under it
            body = ""
            body_end = e
            for nline, _ns, ne in lines[i + 1:]:
                if not nline:
                    continue
                if _ATX_HEADING.match(nline) or _TABLE_ROW.match(nline):
                    break
                body, body_end = nline, ne
                break
            if not body:
                continue
            if self.tuning:
                why = _term_refused(_collapse(title))
                if why:
                    self._refuse("headings", why, line, s, body_end)
                    continue
            out.append(Proposal("concept",
                                f"{_short(title, 6)} - section topic: {_short(body, 5)}",
                                s, body_end))
        return out

    # ----- capability 3: table extraction -----

    def _tables(self, document: Document) -> list[Proposal]:
        lines = list(_line_spans(document.text))
        out = []
        header: list[str] | None = None
        for idx, (line, s, e) in enumerate(lines):
            if not _TABLE_ROW.match(line):
                header = None
                continue
            if _TABLE_RULE.match(line):
                continue
            cells = [c.strip() for c in _TABLE_ROW.match(line).group("cells").split("|")]
            cells = [c for c in cells if c != ""]
            if header is None:
                nxt = lines[idx + 1][0] if idx + 1 < len(lines) else ""
                if _TABLE_RULE.match(nxt) or (len(cells) >= 2 and all(
                        c[:1].isupper() for c in cells if c)):
                    header = cells
                continue
            if len(cells) < 2 or not header or len(cells) > len(header):
                continue
            label = cells[0]
            for col, value in zip(header[1:], cells[1:], strict=False):
                if not value or not col:
                    continue
                out.append(Proposal(
                    "fact",
                    f"{_short(label, 4)} {_short(col, 3)}: {_short(value, _MAX_CELL_WORDS)}",
                    s, e))
        return out

    def propose(self, document: Document) -> list[Proposal]:
        self.refused = []
        self._index = SourceIndex.build(document.text) if self.tuning else None
        self._back_matter = back_matter_start(document.text) if self.tuning else None
        out: list[Proposal] = []
        produced: list[tuple[str, list[Proposal]]] = []
        if self.definitions:
            produced.append(("definitions", self._definitions(document)))
        if self.headings:
            produced.append(("headings", self._headings(document)))
        if self.tables:
            produced.append(("tables", self._tables(document)))
        for _, proposals in produced:
            out += proposals
        if self.tuning and self._back_matter is not None:
            # tune-v2: the headings and tables capabilities do not read the
            # line's text through _structural_refusal, so back matter is
            # enforced once here for every capability; the definitions
            # capability has already refused its own with the reason recorded.
            # AR-rag4ds-045 O2: the refusal names the capability that produced
            # the proposal, as every other refusal record does
            kept = []
            for capability, proposals in produced:
                for p in proposals:
                    if p.span_start >= self._back_matter:
                        self._refuse(capability, "back-matter", p.content, p.span_start,
                                     p.span_end)
                        continue
                    kept.append(p)
            out = kept
        out = sorted(out, key=lambda p: (p.span_start, p.span_end, p.type, p.content))
        if self.tuning:
            # whitespace collapsed in every composed content, then exact duplicates
            # (after collapse) dropped per document, the first in span order kept -
            # the near-duplicate clusterer downstream sees one of each
            kept, seen_text = [], set()
            for p in out:
                content = _collapse(p.content)
                if content in seen_text:
                    self._refuse("dedup", "exact-duplicate", content, p.span_start, p.span_end)
                    continue
                seen_text.add(content)
                kept.append(Proposal(p.type, content, p.span_start, p.span_end))
            out = kept
        return out


class EmbeddingStrategy:
    """Related-concept proposals: the sentences most similar to the caller's
    seed concepts, above a floor, as concept candidates. With no seeds it
    proposes nothing."""

    name = "embed-v1"

    def __init__(self, embedder, seeds: list[str], top_k: int = 3, floor: float = 0.4,
                 tuning: bool = False):
        if not hasattr(embedder, "embed"):
            raise ExtractError("EmbeddingStrategy needs an embedder with .embed()")
        self.embedder = embedder
        self.seeds = list(seeds)
        # untuned, this strategy refuses nothing (the Strategy contract); tuned
        # (AR-rag4ds-045 F3), it refuses back matter like the rule strategy does
        self.refused: list[dict] = []
        self.top_k = top_k
        self.floor = floor
        self.tuning = tuning

    def describe(self) -> dict:
        params = {"seeds": list(self.seeds), "top_k": self.top_k, "floor": self.floor}
        if self.tuning:
            # only when on, so an untuned run's manifest core and extraction id are
            # unchanged (the S9c part 5 rule)
            params["tuning"] = {"version": _TUNING_VERSION,
                                "back_matter": "no sentence at or past the last references "
                                               "heading in the second half of the document is "
                                               "proposed; those above the floor are refused and "
                                               "recorded"}
        return {"name": self.name,
                "model": {"id": getattr(self.embedder, "model_id", None),
                          "revision": getattr(self.embedder, "revision", None)},
                "params": params}

    def propose(self, document: Document) -> list[Proposal]:
        self.refused = []
        if not self.seeds:
            return []
        back = back_matter_start(document.text) if self.tuning else None
        import numpy as np
        nlp = _blank_sentencizer()
        sents, spans = [], []
        for sent in nlp(document.text).sents:
            raw = sent.text.strip()
            if len(raw.split()) < 4 or raw.startswith(("#", "|")):
                continue
            lead = len(sent.text) - len(sent.text.lstrip())
            s = sent.start_char + lead
            sents.append(raw)
            spans.append((s, s + len(raw)))
        if not sents:
            return []

        def unit(rows):
            m = np.asarray(self.embedder.embed(rows), dtype=np.float32)
            if not np.all(np.isfinite(m)):
                raise ExtractError("a non-finite embedding in EmbeddingStrategy")
            n = np.linalg.norm(m, axis=1, keepdims=True)
            if np.any(n == 0):
                raise ExtractError("a zero-norm embedding in EmbeddingStrategy")
            return m / n

        sv, qv = unit(sents), unit(self.seeds)
        best = (sv @ qv.T).max(axis=1)
        eligible = [i for i in range(len(sents)) if float(best[i]) >= self.floor]
        if back is not None:
            # tune-v2 for this strategy (AR-rag4ds-045 F3): a sentence in the back
            # matter that would have competed for a slot is refused and recorded
            # in the rule strategy's record shape, before ranking, so the slots go
            # to the document's own text
            refused_ix = [i for i in eligible if spans[i][0] >= back]
            for i in refused_ix:
                self.refused.append({"capability": "related", "reason": "back-matter",
                                     "line_sha256": hashlib.sha256(
                                         sents[i].encode("utf-8")).hexdigest(),
                                     "span_start": spans[i][0], "span_end": spans[i][1]})
            eligible = [i for i in eligible if spans[i][0] < back]
        ranked = sorted(eligible, key=lambda i: (-float(best[i]), spans[i][0]))[: self.top_k]
        # AR-rag4ds-049 F5: what the strategy proposes is the source's own
        # words. The label "related concept: " this line used to prepend was
        # the strategy's own name for its capability, present in no document,
        # and it reached the served knowledge base as the opening of the two
        # entries session 5 approved. What proposed a span is already recorded
        # on the candidate's `extractor` field and in the run manifest's
        # strategy block - NOT on the proposal's type, which is "concept" here
        # and for the rule strategy's definitions alike (AR-rag4ds-049 r2 F8);
        # either way it is not content.
        return [Proposal("concept", _short(sents[i], _GLOSS_WORDS + 1),
                         spans[i][0], spans[i][1])
                for i in sorted(ranked, key=lambda i: spans[i][0])]


__all__ = ["RuleStrategy", "EmbeddingStrategy"]
