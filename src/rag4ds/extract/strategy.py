"""Extraction v1 - the strategy interface and the run (plan S7; 6.2, FR4,
NFR2, NFR5, NFR9).

A strategy implements one interface: a registered Document in, typed
Proposals out - each a proposed entry (type, content), a source span as
character offsets into the document's text - plus describe(), which
returns the strategy's identity, its parameters and any model pin it
uses, so the run manifest can reconstruct the run (NFR2). The runner
(run_extraction) turns proposals into staging candidates, enforces the
quotation bound, groups near-duplicates, and assembles the manifest. It
files nothing itself (FR5): the gate is the only path into the knowledge
base.

The quotation bound (NFR5), enforced mechanically and in EVERY script.
Candidate content is committed to git, so it must be our own words with
at most a short attributed phrase quoted. The measure is the longest run
of consecutive tokens of the content that also appears, contiguously, in
the source. Tokenisation is Unicode-aware (AR-rag4ds-022 F1) and its script class is
CLOSED by inversion (F10): letters and digits of the scripts that
separate words with spaces - SEGMENTED_SCRIPTS, read from each
character's Unicode name - form word tokens, and characters of EVERY
other script, listed or not, are one token each. A copied Japanese
passage therefore measures its true length instead of zero, and a script
nobody thought of gets the conservative treatment rather than becoming
one unbounded token. THREE ceilings apply to the same matched run, and a
run must satisfy all of them: QUOTATION_MAX_SOURCE_CHARS source characters
charged across every token whatever its script (AR-rag4ds-023 F1),
QUOTATION_MAX_WORDS word tokens, and QUOTATION_MAX_CJK_CHARS source
characters contributed by unsegmented scripts. The source-character ceiling
is the one that bounds VOLUME: counting tokens alone bounded neither eight
long technical terms nor a single unbroken token of any length, so verbatim
character volume was unbounded whenever the copy was few tokens - the hole
under plan 9.1's rationale for committing stage/ to git. Content the
tokeniser cannot
measure - too small a fraction of its characters tokenise - is REFUSED,
not passed: "I cannot measure this" and "this is within the bound" must
never return the same answer.

Two things this module refuses to do silently (NFR9): a proposal over the
bound (or unmeasurable) is suppressed WITH ITS DETAIL in the manifest -
strategy, type, offsets, measured runs, reason - so the expert can see
what was withheld and why; and near-duplicates are grouped, never merged
away, with the clustering threshold recorded and non-finite vectors or an
out-of-range threshold raising rather than silently un-grouping.

The run manifest (NFR2) records the code commit, the configuration, every
strategy's describe() - identity, parameters, model pin - the input
document hash, the timestamp, the suppression detail, the clusters, and
an extraction_id that digests what the run produced (the proposals'
contents and spans) as well as how it was configured. The candidates'
run_manifest reference is derived from that id, so it is knowable before
filing and does not change when the manifest lands (F9). Precision is
measured on a synthetic corpus (measure_precision); the number is
recorded, no threshold is claimed - the target is set with the validator
at S9.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from rag4ds.model.entry import ENTRY_ID_RE, ENTRY_TYPES, LAYERS, SHARED_PACK

# AR-rag4ds-023 F1: the ceiling that bounds verbatim VOLUME - the source
# characters of every token in a matched run, summed whatever the token's
# script. Token counts alone bound neither long tokens nor long words: eight
# ordinary bioinformatics terms carried 117 source characters and a single
# unbroken token carried as many as it contained, with no ceiling at all.
# 64 = QUOTATION_MAX_WORDS words at eight source characters each, which is
# what "a short attributed phrase" costs in technical prose; the longest run
# any legitimate proposal reaches on the S7 corpus is 45. Separators between
# the matched tokens are NOT counted - the measure is the characters of the
# tokens themselves.
QUOTATION_MAX_SOURCE_CHARS = 64
# A short attributed phrase (NFR5), in word tokens of word-separated scripts.
# Retained beside the source-character ceiling rather than replaced by it, so
# that no input becomes MORE permissive than it was before F1 was fixed.
QUOTATION_MAX_WORDS = 8
# The same bound for scripts that do not separate words, counted in characters
# of the source (a base plus its combining marks is one token but counts its
# characters - AR-rag4ds-022 F13).
QUOTATION_MAX_CJK_CHARS = 8
# Content whose characters tokenise below this fraction cannot be measured and
# is refused rather than passed (F1).
MIN_TOKENISABLE_FRACTION = 0.5
# Near-duplicate cosine threshold for grouping (embedding-assisted, 6.2).
NEAR_DUPLICATE_COSINE = 0.92
EXTRACTION_SCHEMA_VERSION = 1

# AR-rag4ds-022 F10: the class is CLOSED by inversion. We enumerate the scripts
# that separate words with spaces - where counting words is meaningful - and treat
# EVERY other script as unsegmented, one token per character. An unfamiliar or
# newly-encoded script therefore gets the conservative treatment by default
# instead of becoming one unbounded token. The script of a character is read from
# its Unicode name, whose first words are the script (e.g. "CYRILLIC SMALL LETTER
# A", "BOPOMOFO LETTER B"); a character with no name is unsegmented.
SEGMENTED_SCRIPTS = frozenset({
    "LATIN", "GREEK", "COPTIC", "CYRILLIC", "ARMENIAN", "HEBREW", "ARABIC", "SYRIAC",
    "THAANA", "NKO", "SAMARITAN", "MANDAIC", "DEVANAGARI", "BENGALI", "GURMUKHI",
    "GUJARATI", "ORIYA", "TAMIL", "TELUGU", "KANNADA", "MALAYALAM", "SINHALA",
    "GEORGIAN", "ETHIOPIC", "CHEROKEE", "OGHAM", "RUNIC", "MONGOLIAN", "GLAGOLITIC",
    "GOTHIC", "DESERET", "SHAVIAN", "OSAGE", "ADLAM", "VAI", "TIFINAGH", "BAMUM",
    "MODIFIER",
})
# (FULLWIDTH and MATHEMATICAL are not listed: NFKC decomposes those characters to
# their Latin/Greek forms before the script is read - AR-rag4ds-022 O18.)
# AR-rag4ds-022 O19: digits are word characters only when their Unicode name
# BEGINS with one of these words. That is true of ASCII "DIGIT ZERO" and of
# "DEVANAGARI DIGIT ZERO" (the script word is not first), but NOT of
# "ARABIC-INDIC DIGIT ZERO", "EXTENDED ARABIC-INDIC DIGIT ZERO" or "THAI DIGIT
# ZERO", whose names begin with a script this module does not list, so they are
# treated as unsegmented - one token per character. The direction is
# conservative (those digits are charged more, never less), which is why this
# is a description to correct and not a behaviour to change.
_NUMERIC_NAME_PREFIXES = ("DIGIT", "NUMBER", "SUPERSCRIPT", "SUBSCRIPT")


class ExtractError(ValueError):
    """Raised loudly for any extraction violation."""


def script_of(ch: str) -> str:
    """The script name from the character's Unicode name (its leading words), or
    "" when the character is unnamed."""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return ""
    return name.split()[0]


def is_unsegmented(ch: str) -> bool:
    """True when the character belongs to a script that does not separate words -
    which, by inversion (F10), is every script not in SEGMENTED_SCRIPTS. Digits
    and numerals are word characters when their Unicode name begins with DIGIT,
    NUMBER, SUPERSCRIPT or SUBSCRIPT; digits whose name begins with an unlisted
    script instead - Arabic-Indic, Extended Arabic-Indic, Thai - are unsegmented
    and charged per character (AR-rag4ds-022 O19: conservative, so no verdict
    moves)."""
    if ch.isspace() or not ch.isalnum():
        return False
    script = script_of(ch)
    if not script:
        return True
    if script in SEGMENTED_SCRIPTS or script.startswith(_NUMERIC_NAME_PREFIXES):
        return False
    return True


@dataclass(frozen=True)
class Token:
    text: str          # the comparison form (casefolded for word tokens)
    unsegmented: bool
    source_len: int = 0  # characters of the normalised source this token consumed (F12)


def tokenise(text: str) -> list[Token]:
    """Unicode-aware tokens (F1): a run of letters/digits of a word-separated
    script is one word token (casefolded, accent-normalised); a character of an
    unsegmented script is one token of its own. Punctuation and whitespace are
    separators and produce no tokens."""
    out: list[Token] = []
    buf: list[str] = []

    def flush():
        if buf:
            raw = "".join(buf)
            out.append(Token(raw.casefold(), False, len(raw)))
            buf.clear()

    for ch in unicodedata.normalize("NFKC", text):
        if is_unsegmented(ch):
            flush()
            out.append(Token(ch, True, 1))
        elif unicodedata.category(ch).startswith("M"):
            # AR-rag4ds-022 F13: a combining mark attaches to the token it
            # modifies. Inside a word it continues the word; after an
            # unsegmented character it EXTENDS that character token (one
            # cluster, one token, source_len + 1) instead of opening a word
            # token, which would split one copied run across two budgets.
            if buf:
                buf.append(ch)
            elif out and out[-1].unsegmented:
                last = out[-1]
                out[-1] = Token(last.text + ch, True, last.source_len + 1)
            # a leading mark with no base contributes nothing
        elif ch.isalnum():
            buf.append(ch)
        else:
            flush()
    flush()
    return out


def tokenisable_fraction(text: str) -> float:
    """The fraction of the content's non-space characters the tokeniser
    accounts for. Below MIN_TOKENISABLE_FRACTION the bound cannot be
    measured and the content is refused (F1)."""
    chars = [c for c in unicodedata.normalize("NFKC", text) if not c.isspace()]
    if not chars:
        return 0.0
    covered = sum(t.source_len for t in tokenise(text))
    return min(covered / len(chars), 1.0)


def _longest_runs(content: str, source_text: str) -> tuple[int, int, int]:
    """(word tokens, unsegmented source characters, TOTAL source characters) of
    the runs of content that appear contiguously in the source, each maximised
    over the same single scan.

    The third number is the one AR-rag4ds-023 F1 added and it is the one that
    bounds volume: EVERY token in the run is charged the source characters it
    consumed, word tokens included, so eight long technical terms and one
    unbroken 5000-character token are both measured in the unit that matters.
    The first two are retained and still enforced, so nothing loosens."""
    c = tokenise(content)
    s = tokenise(source_text)
    if not c or not s:
        return 0, 0, 0
    index: dict[str, list[int]] = {}
    for j, t in enumerate(s):
        index.setdefault(t.text, []).append(j)
    best_words = best_chars = best_source = 0
    n, m = len(c), len(s)
    for i in range(n):
        for j in index.get(c[i].text, ()):
            run = 0
            words = chars = source = 0
            while i + run < n and j + run < m and c[i + run].text == s[j + run].text:
                tok = c[i + run]
                source += tok.source_len          # every token, every script (F1)
                if tok.unsegmented:
                    # count the CHARACTERS the cluster consumed, not the token (F13)
                    chars += tok.source_len
                else:
                    words += 1
                run += 1
            best_words = max(best_words, words)
            best_chars = max(best_chars, chars)
            best_source = max(best_source, source)
    return best_words, best_chars, best_source


def longest_verbatim_runs(content: str, source_text: str) -> tuple[int, int]:
    """(longest word-token run, longest unsegmented-character run) of content
    that appears contiguously in the source. Both are measured over the same
    matched runs: a run's length counts in whichever unit its tokens are.
    See longest_verbatim_source_chars for the volume measure (F1)."""
    w, c, _ = _longest_runs(content, source_text)
    return w, c


def longest_verbatim_source_chars(content: str, source_text: str) -> int:
    """The longest verbatim run measured in SOURCE CHARACTERS across every
    token whatever its script (AR-rag4ds-023 F1). Separators between the
    matched tokens are not counted."""
    return _longest_runs(content, source_text)[2]


def longest_verbatim_run(content: str, source_text: str) -> int:
    """The longest verbatim run in token units (words plus unsegmented
    characters), for callers that want one number."""
    w, c = longest_verbatim_runs(content, source_text)
    return max(w, c)


def quotation_bound_check(content: str, source_text: str) -> dict:
    """Measure the bound. Returns {'ok', 'reason', 'word_run', 'char_run',
    'source_run', 'tokenisable'}; ok is False when the content quotes too much
    by ANY of the three ceilings AND when it cannot be measured - the cases are
    distinguished by reason, and none is silently allowed (AR-rag4ds-022 F1,
    AR-rag4ds-023 F1)."""
    frac = tokenisable_fraction(content)
    if frac < MIN_TOKENISABLE_FRACTION:
        return {"ok": False, "reason": "unmeasurable: only "
                f"{frac:.2f} of the content's characters tokenise (below "
                f"{MIN_TOKENISABLE_FRACTION}); the quotation bound cannot be "
                "measured, so the content is refused rather than passed",
                "word_run": None, "char_run": None, "source_run": None,
                "tokenisable": frac}
    w, c, src = _longest_runs(content, source_text)
    base = {"word_run": w, "char_run": c, "source_run": src, "tokenisable": frac}
    if src > QUOTATION_MAX_SOURCE_CHARS:
        return {"ok": False, "reason": f"verbatim run of {src} source characters exceeds the "
                f"quotation bound ({QUOTATION_MAX_SOURCE_CHARS}) (NFR5); the run is "
                f"{w} word token(s) and {c} unsegmented character(s), so a token count "
                "alone would not have caught it (AR-rag4ds-023 F1)", **base}
    if w > QUOTATION_MAX_WORDS:
        return {"ok": False, "reason": f"verbatim run of {w} source words exceeds the "
                f"quotation bound ({QUOTATION_MAX_WORDS}) (NFR5)", **base}
    if c > QUOTATION_MAX_CJK_CHARS:
        return {"ok": False, "reason": f"verbatim run of {c} source characters in an "
                f"unsegmented script exceeds the quotation bound "
                f"({QUOTATION_MAX_CJK_CHARS}) (NFR5)", **base}
    return {"ok": True, "reason": None, **base}


def within_quotation_bound(content: str, source_text: str) -> bool:
    return bool(quotation_bound_check(content, source_text)["ok"])


@dataclass(frozen=True)
class Document:
    """A registered corpus document as the extractor sees it: the content
    hash it is catalogued under (NFR6), its UTF-8 text, the target pack and
    layer, and the source_reference the candidates cite. The text must hash
    to source_hash before any strategy reads it."""

    source_hash: str
    text: str
    pack: str
    layer: str
    source_reference: str

    def __post_init__(self) -> None:
        problems = []
        if not isinstance(self.source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}",
                                                                     self.source_hash):
            problems.append("source_hash must be a 64-hex sha256")
        elif hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.source_hash:
            problems.append("text does not hash to source_hash (wrong document)")
        if not isinstance(self.text, str) or not self.text.strip():
            problems.append("text must be a non-blank string")
        if self.layer not in LAYERS:
            problems.append(f"layer {self.layer!r} not one of {LAYERS}")
        if not isinstance(self.pack, str) or not ENTRY_ID_RE.fullmatch(self.pack or ""):
            problems.append(f"pack {self.pack!r} must be a pack code or {SHARED_PACK!r}")
        elif self.layer == "domain" and self.pack == SHARED_PACK:
            problems.append("domain-layer documents need a pack code")
        elif self.layer in ("general", "task") and self.pack != SHARED_PACK:
            problems.append(f"{self.layer}-layer documents use the reserved pack {SHARED_PACK!r}")
        if not isinstance(self.source_reference, str) or not self.source_reference.strip():
            problems.append("source_reference must be a non-blank string")
        if problems:
            raise ExtractError("document invalid: " + "; ".join(problems))


@dataclass(frozen=True)
class Proposal:
    """What a strategy proposes: a typed entry content and the source span it
    came from (character offsets into Document.text). The runner assigns the
    extractor identity and the run-manifest reference; id/version/status are
    the gate's (FR5)."""

    type: str
    content: str
    span_start: int
    span_end: int

    def __post_init__(self) -> None:
        problems = []
        if self.type not in ENTRY_TYPES:
            problems.append(f"type {self.type!r} not one of {ENTRY_TYPES}")
        if not isinstance(self.content, str) or not self.content.strip():
            problems.append("content must be a non-blank string")
        if (not isinstance(self.span_start, int) or not isinstance(self.span_end, int)
                or isinstance(self.span_start, bool) or isinstance(self.span_end, bool)
                or self.span_start < 0 or self.span_end <= self.span_start):
            problems.append("span offsets must be integers with 0 <= start < end")
        if problems:
            raise ExtractError("proposal invalid: " + "; ".join(problems))


class Strategy(Protocol):
    name: str

    def propose(self, document: Document) -> list[Proposal]: ...

    def describe(self) -> dict:
        """Identity, parameters and any model pin this strategy used - the
        manifest's per-strategy record (NFR2, F2)."""
        ...


@dataclass
class ExtractionResult:
    candidates: list[dict]                # staging entry-candidate dicts, filing order
    clusters: list[list[int]]             # near-duplicate groups as indices into candidates
    manifest: dict                        # the NFR2 run manifest
    suppressed: list[dict] = field(default_factory=list)  # bound-suppressed proposals


def _canonical_bytes(obj) -> bytes:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return (text + "\n").encode("ascii")


def build_entry_candidate(document: Document, proposal: Proposal, extractor: str,
                          run_manifest: str) -> dict:
    """A staging entry-candidate (staging.store.ENTRY_CANDIDATE_FIELDS) - the
    span travels as hash + offsets, never text."""
    return {
        "kind": "entry",
        "proposed": {"type": proposal.type, "content": proposal.content,
                     "source_reference": document.source_reference},
        "layer": document.layer,
        "pack": document.pack,
        "source_hash": document.source_hash,
        "span_start": proposal.span_start,
        "span_end": proposal.span_end,
        "extractor": extractor,
        "run_manifest": run_manifest,
    }


def cluster_near_duplicates(contents: list[str], embedder=None,
                            threshold: float = NEAR_DUPLICATE_COSINE) -> list[list[int]]:
    """Group near-duplicate contents (6.2) by cosine over the embedder's
    vectors; returns clusters of size >= 2 as index lists. Deterministic:
    union-find over pairs in index order. A non-finite vector or an
    out-of-range threshold raises rather than silently un-grouping (F6)."""
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) \
            or not -1.0 <= float(threshold) <= 1.0 or math.isnan(float(threshold)):
        raise ExtractError(f"near-duplicate threshold {threshold!r} must be a number in [-1, 1]")
    n = len(contents)
    if embedder is None or n < 2:
        return []
    vecs = np.asarray(embedder.embed(list(contents)), dtype=np.float32)
    if not np.all(np.isfinite(vecs)):
        raise ExtractError("a content embedding holds NaN or inf; cannot cluster")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ExtractError("a content embedding has zero norm; cannot cluster")
    unit = vecs / norms
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if float(unit[i] @ unit[j]) >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in (groups[k] for k in sorted(groups)) if len(g) >= 2]


def _describe(strat: Strategy) -> dict:
    d = getattr(strat, "describe", None)
    if not callable(d):
        raise ExtractError(f"strategy {strat.name!r} has no describe() - the manifest cannot "
                           "record how it ran (NFR2)")
    out = d()
    if not isinstance(out, dict) or "name" not in out or "params" not in out:
        raise ExtractError(f"strategy {strat.name!r}: describe() must return "
                           "{'name', 'params', 'model'(optional)}")
    if out["name"] != strat.name:
        raise ExtractError(f"strategy describe().name {out['name']!r} != .name {strat.name!r}")
    return out


def run_extraction(document: Document, strategies: list[Strategy], code_commit: str,
                   timestamp: str, config: dict | None = None, embedder=None,
                   near_duplicate_cosine: float = NEAR_DUPLICATE_COSINE) -> ExtractionResult:
    """Run every strategy over the document, enforce the quotation bound,
    build staging candidates, group near-duplicates, and assemble the NFR2
    run manifest. Files nothing (FR5). Deterministic given the inputs."""
    from rag4ds.gate.decisions import _timestamp_valid

    if not isinstance(code_commit, str) or not re.fullmatch(r"[0-9a-f]{7,40}", code_commit):
        raise ExtractError("code_commit must be a 7-40 hex git commit id (NFR2)")
    if not _timestamp_valid(timestamp):
        raise ExtractError("timestamp must be 'YYYY-MM-DD HH:MM[:SS] JST' (NFR2)")
    if not strategies:
        raise ExtractError("run_extraction needs at least one strategy")
    names = [s.name for s in strategies]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ExtractError(f"strategy name(s) {dupes} used twice; the name is the extractor "
                           "provenance field and the manifest's key, so it must be unique (F5)")
    described = [_describe(s) for s in strategies]

    kept: list[tuple[str, Proposal]] = []
    suppressed: list[dict] = []
    per_strategy: dict[str, int] = {}
    for strat in strategies:
        proposals = strat.propose(document)
        n_kept = 0
        for p in proposals:
            if not (0 <= p.span_start < p.span_end <= len(document.text)):
                raise ExtractError(f"{strat.name}: proposal span [{p.span_start},{p.span_end}) "
                                   f"outside the document ({len(document.text)} chars)")
            check = quotation_bound_check(p.content, document.text)
            if not check["ok"]:
                suppressed.append({"strategy": strat.name, "type": p.type,
                                   "span_start": p.span_start, "span_end": p.span_end,
                                   "word_run": check["word_run"], "char_run": check["char_run"],
                                   "source_run": check["source_run"],
                                   "tokenisable_fraction": round(check["tokenisable"], 4),
                                   "reason": check["reason"]})
                continue
            kept.append((strat.name, p))
            n_kept += 1
        per_strategy[strat.name] = n_kept

    # the identity of what this run produced and how it was configured (F4)
    produced = [{"strategy": name, "type": p.type, "content": p.content,
                 "span_start": p.span_start, "span_end": p.span_end} for name, p in kept]
    core = {
        "kind": "extraction",
        "schema": EXTRACTION_SCHEMA_VERSION,
        "source_hash": document.source_hash,
        "pack": document.pack,
        "layer": document.layer,
        "strategies": described,
        "strategy_kept_counts": dict(per_strategy),
        "clustering_embedder": ({"id": embedder.model_id, "revision": embedder.revision}
                                if embedder is not None else None),
        "near_duplicate_cosine": float(near_duplicate_cosine) if embedder is not None else None,
        "quotation_max_source_chars": QUOTATION_MAX_SOURCE_CHARS,
        "quotation_max_words": QUOTATION_MAX_WORDS,
        "quotation_max_cjk_chars": QUOTATION_MAX_CJK_CHARS,
        "min_tokenisable_fraction": MIN_TOKENISABLE_FRACTION,
        "candidates_proposed": len(kept),
        "candidates_suppressed_for_bound": len(suppressed),
        "suppressed": suppressed,
        "config": dict(config or {}),
        "produced_sha256": hashlib.sha256(_canonical_bytes(produced)).hexdigest(),
    }
    extraction_id = hashlib.sha256(_canonical_bytes(core)).hexdigest()[:16]
    run_ref = f"runs/extraction.jsonl#{extraction_id}"  # knowable before filing (F9)
    candidates = [build_entry_candidate(document, p, name, run_ref) for name, p in kept]
    clusters = cluster_near_duplicates([c["proposed"]["content"] for c in candidates], embedder,
                                       threshold=near_duplicate_cosine)
    manifest = dict(core, extraction_id=extraction_id, run_manifest_ref=run_ref,
                    clusters=clusters, code_commit=code_commit, timestamp=timestamp)
    return ExtractionResult(candidates=candidates, clusters=clusters, manifest=manifest,
                            suppressed=suppressed)


def measure_precision(result: ExtractionResult, is_correct: Callable[[dict], bool]) -> dict:
    """A precision sanity check on a synthetic corpus (S7 exit): the fraction
    of proposed candidates a caller-supplied oracle judges correct. NO
    threshold is applied - the number is returned and recorded; the target is
    set with the validator at S9. precision is None when nothing was proposed,
    so an empty run is not reported as 0.0 or 1.0 - both would mislead."""
    total = len(result.candidates)
    correct = sum(1 for c in result.candidates if is_correct(c))
    return {"proposed": total, "correct": correct,
            "precision": (correct / total) if total else None,
            "suppressed_for_bound": len(result.suppressed)}


__all__ = [
    "QUOTATION_MAX_SOURCE_CHARS", "QUOTATION_MAX_WORDS", "QUOTATION_MAX_CJK_CHARS",
    "MIN_TOKENISABLE_FRACTION",
    "NEAR_DUPLICATE_COSINE", "EXTRACTION_SCHEMA_VERSION", "SEGMENTED_SCRIPTS", "ExtractError",
    "Document", "Proposal", "Strategy", "Token", "ExtractionResult", "is_unsegmented",
    "script_of",
    "tokenise", "tokenisable_fraction", "longest_verbatim_run", "longest_verbatim_runs",
    "longest_verbatim_source_chars",
    "quotation_bound_check", "within_quotation_bound", "build_entry_candidate",
    "cluster_near_duplicates", "run_extraction", "measure_precision",
]
