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
which needs no downloaded language model and runs offline inside the
hardware envelope (NFR4) - VT-3 pins spacy but no model, so no POS, no
noun chunks and no NER are available; that limitation is declared, not
worked around. Precision over recall (6.2): only clear shapes are
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

import re

from rag4ds.extract.strategy import Document, ExtractError, Proposal

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

    def __init__(self, definitions: bool = True, headings: bool = True, tables: bool = True):
        self.definitions, self.headings, self.tables = definitions, headings, tables
        self._nlp = None

    def describe(self) -> dict:
        return {"name": self.name, "model": None,
                "params": {"definitions": self.definitions, "headings": self.headings,
                           "tables": self.tables, "gloss_words": _GLOSS_WORDS,
                           "max_cell_words": _MAX_CELL_WORDS,
                           "sentencizer": "spacy.blank('en') + sentencizer (no language model "
                                          "pinned at VT-3)"}}

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
        out: list[Proposal] = []
        if self.definitions:
            out += self._definitions(document)
        if self.headings:
            out += self._headings(document)
        if self.tables:
            out += self._tables(document)
        return sorted(out, key=lambda p: (p.span_start, p.span_end, p.type, p.content))


class EmbeddingStrategy:
    """Related-concept proposals: the sentences most similar to the caller's
    seed concepts, above a floor, as concept candidates. With no seeds it
    proposes nothing."""

    name = "embed-v1"

    def __init__(self, embedder, seeds: list[str], top_k: int = 3, floor: float = 0.4):
        if not hasattr(embedder, "embed"):
            raise ExtractError("EmbeddingStrategy needs an embedder with .embed()")
        self.embedder = embedder
        self.seeds = list(seeds)
        self.top_k = top_k
        self.floor = floor

    def describe(self) -> dict:
        return {"name": self.name,
                "model": {"id": getattr(self.embedder, "model_id", None),
                          "revision": getattr(self.embedder, "revision", None)},
                "params": {"seeds": list(self.seeds), "top_k": self.top_k, "floor": self.floor}}

    def propose(self, document: Document) -> list[Proposal]:
        if not self.seeds:
            return []
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
        ranked = sorted((i for i in range(len(sents)) if float(best[i]) >= self.floor),
                        key=lambda i: (-float(best[i]), spans[i][0]))[: self.top_k]
        return [Proposal("concept", "related concept: " + _short(sents[i], _GLOSS_WORDS + 1),
                         spans[i][0], spans[i][1])
                for i in sorted(ranked, key=lambda i: spans[i][0])]


__all__ = ["RuleStrategy", "EmbeddingStrategy"]
