"""NFR6 scan (AR-rag4ds-026 F1): no passage of a non-redistributable corpus
document may be in git. For every catalogued document whose catalog entry
says redistributable: false, the scan reads the local raw copy the ingest
manifest names, and looks in EVERY git-tracked file of the module for a
verbatim run of that document at least QUOTATION_MAX_SOURCE_CHARS long -
the same ceiling the quotation bound applies to entry content (component
2, NFR5), measured on whitespace-collapsed, casefolded text. Any hit fails
the scan and is printed with the file, the run length and its first
characters, so the rule that used to be a sentence is now a check.

Two rules added at S9 (docs/s9_robotics_pack.md section 6): the scan reads
every file git tracks OR would add (untracked and not ignored), so a local
run sees what the next commit would carry - the hermetic capture at
f22e486 caught a title in a not-yet-added catalog that the local run had
not seen; and a document's own catalogued title (metadata.title, at most
TITLE_MAX_CHARS, present at the head of the document) is a citation, not a passage:
each bare occurrence is blanked out of a scanned file before windowing, so a
record may name the document it cites - but an occurrence inside a longer
verbatim run of the document (more than TITLE_SLACK characters of the
document continuing on either side) is a copy and is reported as one
(AR-rag4ds-028 F1: the earlier blind replace cut such a run in half).

A document whose local copy is absent cannot be scanned: the tool then
exits 2 and says so by hash - it never reads absence as compliance. The
data directory is the module's data/ unless RAG4DS_DATA_DIR names another
(the hermetic evidence capture runs from a fresh worktree that has no
data/, and points here at the real one).

    python tools/nfr6_scan.py            (from the module root; exit 0 = clean)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import yaml

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))
from rag4ds.extract.strategy import QUOTATION_MAX_SOURCE_CHARS  # noqa: E402

WINDOW = QUOTATION_MAX_SOURCE_CHARS
# a dot leader is ONE repeated character; a genomic sequence has four, a binary
# matrix three - those are text and must be found (AR-rag4ds-026 r3 F17)
MIN_DISTINCT_CHARS = 3
# a document's own title is a citation (NFR7), never a passage - up to this many
# normalised characters, and only when the document itself carries it AT ITS
# HEAD (within the first TITLE_HEAD_CHARS): a "title" that is a passage from
# the body is not exempt, and the catalog naming it is itself a hit
# AR-rag4ds-028 ruling: the cap sits just above the longest catalogued title
# (106 then -> 120). 2026-09-21 (S9e part 3, record section 19): the validator's
# eleven documents brought titles of 143, 152 and 157 characters, so the same
# rule gives 170; the suite asserts the constant and every catalogued title
TITLE_MAX_CHARS = 170
TITLE_HEAD_CHARS = 3000
# AR-rag4ds-048 F1 (external): a title is bound by SHAPE as the venue is - it must
# START within the first TITLE_START_CHARS normalised characters of its document.
# Measured 2026-09-21 over every catalogued document whose text carries its
# title: the latest start is 257 (a review with a long journal banner), so 300 -
# the cap rule of AR-rag4ds-028, just above the observed maximum.
# TITLE_HEAD_CHARS stays the window a citable copy is read for.
TITLE_START_CHARS = 300
# AR-rag4ds-049 F1: position alone is not shape. Five of the twenty-three
# catalogued documents whose text can be read print their first structural
# marker INSIDE that window (introduction at 146 and 287, abstract at 199, 238
# and 265), so a sentence of those abstracts could be exempted by a
# title-field mistake. The second test, the one the occurrence count is to the
# venue: the document's FIRST structural marker must begin at or after the END
# of the declared title - a title block precedes the abstract, the keywords and
# the introduction, so a title neither follows a marker (r1) nor swallows one
# (r2: a passage beginning at the marker word was exempt while the same passage
# nine characters later was two hits). A document printing no marker at all
# has no title block to bound, so there the title must be the FIRST text the
# document prints - nothing but punctuation and spaces before it (AR-rag4ds-048
# r3 O1; the auditor's words were "start at 0", and the one no-marker document
# whose title is found, an open one, prints "# " before it - measured
# 2026-09-27, record section 55).
# Measured over the live corpus before the rule was written: no catalogued
# title contains any marker word, and all twenty exempt titles end before their
# own document's first marker, so the rule costs no honest exemption today.
# Its reach, stated: it binds only where the first marker falls inside the
# window (five of the twenty-three; four of the twelve restricted), three
# documents print no marker at all and are bound by the first-text rule, and
# everywhere else the position bound still decides alone.
# The trade: a document whose real title contains one of these words loses its
# exemption and its title is reported as a possible copy. That is the safe
# direction - the scan says more, never less - and no catalogued title has the
# shape today.
# The residual, stated as the venue's is (AR-rag4ds-045 r2 O7): a string of at
# most TITLE_MAX_CHARS characters that the document prints starting within its
# first TITLE_START_CHARS characters and ENDING before its first structural
# marker - that is, inside the title block itself, where the author line and
# the affiliations sit - is exempt whatever it says; and in a document that
# prints no marker, so is such a string when nothing but punctuation and
# spaces precedes it (AR-rag4ds-049 r3 O14).
_STRUCTURAL_MARKER = re.compile(r"\b(?:abstract|keywords|introduction)\b")
# AR-rag4ds-045 F1: a venue is bound by SHAPE, not only by length - a journal or
# conference name is short and is printed as a running header, so it occurs
# more than once in the document; a sentence from the abstract does neither.
# The two venues exempt today are 39 and 74 characters and print 10+ times.
VENUE_MAX_CHARS = 100
VENUE_MIN_OCCURRENCES = 2
# AR-rag4ds-028 F1: an occurrence of the title is a bare citation only when the
# verbatim run of the document around it extends at most this far on either
# side (a space, a quote, a colon); any longer run around the title is a copy
TITLE_SLACK = 8


class Nfr6Error(RuntimeError):
    """The scan could not run to completion - reported, never read as clean."""


_LINE_BREAK_HYPHEN = re.compile(r"(?<=[^\W\d_])- (?=[^\W\d_])")


def normalise(text: str) -> str:
    """NFKC first (AR-rag4ds-027 F4: a PDF's ligatures are letters), then
    casefold, collapse whitespace, and rejoin a hyphenated word a PDF broke
    at a line end ('self- healing' -> 'self-healing'; S9e part 3, record
    section 19: a catalogued title was not found at its own document's head
    for this). Only a hyphen between two letters with a space after it; a
    spaced dash or a hyphen by a digit is untouched. The trade, stated
    (AR-rag4ds-045 F6): the rejoin finds runs the old normalisation missed
    when one side was line-broken and the other not, and it DELETES one
    character per break on both sides, so a run of 64 to 64+k characters
    carrying k breaks falls under the window and is no longer reported -
    the effective window on densely broken text is 64 plus the breaks."""
    collapsed = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return _LINE_BREAK_HYPHEN.sub("-", collapsed)


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def readable_text(path: Path) -> str:
    """A tracked file as the text a reader would see: JSON and JSON Lines
    files are parsed and their string values joined, so the module's own
    escaping (a newline as two characters, a non-ASCII letter as six) cannot
    hide a passage from the scan (AR-rag4ds-027 F4); anything else is read
    as UTF-8 text. A JSON file that does not parse is a loud failure."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        out = []
        for n, line in enumerate(raw.splitlines(), 1):
            if line.strip():
                try:
                    out.extend(_strings(json.loads(line)))
                except json.JSONDecodeError as exc:
                    raise Nfr6Error(f"{path}:{n}: not a JSON record: {exc}") from exc
        return " ".join(out)
    if path.suffix == ".json":
        try:
            return " ".join(_strings(json.loads(raw)))
        except json.JSONDecodeError as exc:
            raise Nfr6Error(f"{path}: not JSON: {exc}") from exc
    return raw


def catalogued_documents(module_root: Path) -> list[dict]:
    """Every catalogued document across the shared catalog and every pack
    catalog, with its redistributable flag and (when an ingest manifest names
    one) its local path."""
    catalogs = [module_root / "shared" / "corpus" / "catalog.yaml"]
    catalogs += sorted((module_root / "packs").glob("*/corpus/catalog.yaml"))
    manifests: dict[str, str] = {}
    runs = module_root / "runs" / "ingest.jsonl"
    if runs.is_file():
        for line in runs.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("kind") == "ingest":
                    manifests[r["hash"]] = r["local_path"]
    out = []
    for cat in catalogs:
        if not cat.is_file():
            continue
        data = yaml.safe_load(cat.read_text(encoding="utf-8"))
        for d in data.get("documents", []):
            meta = d.get("metadata") or {}
            out.append({"hash": d["hash"], "catalog": str(cat.relative_to(module_root)),
                        "local_path": manifests.get(d["hash"]),
                        "redistributable": d.get("redistributable"),
                        "title": meta.get("title", ""), "venue": meta.get("venue", "")})
    return out


def restricted_documents(module_root: Path) -> list[dict]:
    """Every catalogued document with redistributable false, across the
    shared catalog and every pack catalog, with its manifested local path."""
    out = []
    for d in catalogued_documents(module_root):
        if d["redistributable"] is False:
            if d["local_path"] is None:
                raise Nfr6Error(f"{d['catalog']}: document {d['hash']} has no ingest manifest, "
                                f"so its local copy cannot be located")
            out.append(d)
    return out


def citable_titles(module_root: Path, data_dir: Path | None) -> list[str]:
    """S9e part 3 (2026-09-21, record section 19): the papers of a corpus
    cite one another, so a restricted document's reference list carries
    OTHER catalogued documents' titles verbatim, and naming one of those in a
    record tripped the scan against the paper that cites it. A catalogued
    document's title is a citation (NFR7) whichever document also prints it -
    under the same two conditions as its own: at most TITLE_MAX_CHARS
    normalised characters, and present at the HEAD of the document it names
    (checked against that document's own local copy; a document whose copy is
    absent here contributes nothing - the scan then sees its title as text).
    Returned normalised, longest first, so a title that contains another is
    blanked whole before the shorter one is looked for."""
    titles = []
    for d in catalogued_documents(module_root):
        title = normalise(d["title"])
        if not (0 < len(title) <= TITLE_MAX_CHARS) or d["local_path"] is None:
            continue
        raw = locate(d, module_root, data_dir)
        if raw is None:
            continue
        # AR-rag4ds-045 F2: the copy whose head decides what is exempt IS the
        # catalogued document, or the scan does not run - the AR-rag4ds-026 F10
        # rule, which run() applies to the restricted copy, applied here too
        got = hashlib.sha256(raw.read_bytes()).hexdigest()
        if got != d["hash"]:
            raise Nfr6Error(f"{raw} hashes to {got[:16]}..., not the catalogued "
                            f"{d['hash'][:16]}... ({d['title']!r}); a copy that is not the "
                            f"document decides no exemption")
        try:
            head = normalise(raw.read_text(encoding="utf-8")[:TITLE_HEAD_CHARS * 2])
        except UnicodeDecodeError:
            continue  # an archive or binary document has no head to carry a title
        if title_at_head(title, head) and title not in titles:
            titles.append(title)
    return sorted(titles, key=len, reverse=True)


def title_at_head(title: str, text: str) -> bool:
    """The tests a catalogued title must pass to be a citation: non-empty,
    within the cap, STARTING within the first TITLE_START_CHARS of the
    document's normalised text (AR-rag4ds-048 F1), and ENDING before the
    document's first structural marker where it prints one (AR-rag4ds-049 F1,
    r1 and r2) - the shape of a title block, which a sentence of the abstract,
    the keywords or the body does not have, and which a string that begins at
    the marker word does not have either. Where the document prints no
    marker, the title must be the first text it prints: no letter or digit
    before it (AR-rag4ds-048 r3 O1)."""
    if not (0 < len(title) <= TITLE_MAX_CHARS):
        return False
    i = text.find(title)
    if not (0 <= i < TITLE_START_CHARS):
        return False
    marker = _STRUCTURAL_MARKER.search(text)
    if marker is None:
        return not any(c.isalnum() for c in text[:i])
    return i + len(title) <= marker.start()


def locate(doc: dict, module_root: Path, data_dir: Path | None) -> Path | None:
    rel = Path(doc["local_path"])
    candidates = []
    if data_dir is not None:
        candidates.append(data_dir / rel)
        if rel.parts and rel.parts[0] == "data":  # manifested relative to the module root
            candidates.append(data_dir / rel.relative_to("data"))
    candidates.append(module_root / rel)
    for p in candidates:
        if p.is_file():
            return p
    return None


def tracked_files(module_root: Path) -> list[Path]:
    """Every file git tracks or would add: tracked, plus untracked files that
    no ignore rule covers (git ls-files --others --exclude-standard) - so the
    scan sees a new catalog or record BEFORE it is committed, not only after.
    An ignored file (data/) is never scanned; it is not headed for git."""
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                         cwd=module_root, capture_output=True, check=True).stdout
    seen, files = set(), []
    for p in out.split(b"\0"):
        if p and p not in seen:
            seen.add(p)
            f = module_root / p.decode("utf-8")
            if f.is_file():  # a deleted-but-still-indexed path has nothing to scan
                files.append(f)
    return files


def blank_citations(hay: str, title: str, text: str) -> tuple[str, tuple[int, str] | None]:
    """AR-rag4ds-028 F1: every occurrence of the document's title in hay is
    either a bare citation - the verbatim run of the document that contains
    it extends at most TITLE_SLACK characters on either side - or part of a
    copy. Citations are blanked (spaces of the same length, so offsets and
    the windows around them are unchanged); the first copy found is returned
    as (run length, excerpt) and the caller reports it. The run is extended
    against the whole document, not the title's own neighbourhood, so a
    passage that continues past the title on either side, by however
    little, is a run and not a name."""
    out, copy, start, L = hay, None, 0, len(title)
    while True:
        i = out.find(title, start)
        if i < 0:
            break
        a, b = i, i + L
        while i - a < TITLE_SLACK + 1 and a > 0 and out[a - 1:b] in text:
            a -= 1
        while b - (i + L) < TITLE_SLACK + 1 and b < len(out) and out[a:b + 1] in text:
            b += 1
        if i - a > TITLE_SLACK or b - (i + L) > TITLE_SLACK:
            if copy is None:
                copy = (b - a, out[a:a + 80])
            start = i + L
            continue
        out = out[:i] + " " * L + out[i + L:]
        start = i + L
    return out, copy


def scan_text(hay: str, windows: set[str]) -> tuple[int, str] | None:
    """The longest verbatim run of the document in hay (normalised), or
    None when no WINDOW-length run exists. Returns (length, excerpt)."""
    best: tuple[int, str] | None = None
    i, n = 0, len(hay)
    while i + WINDOW <= n:
        if hay[i:i + WINDOW] in windows:
            j = i + WINDOW
            while j < n and hay[i:j + 1][-WINDOW:] in windows:
                j += 1
            run = hay[i:j]
            # AR-rag4ds-026 O11: a run of one repeated character (pytest's progress
            # dots against the guide's dot-leader table of contents) is not text
            if len(set(run)) >= MIN_DISTINCT_CHARS and (best is None or j - i > best[0]):
                best = (j - i, hay[i:i + 80])
            i = j
        else:
            i += 1
    return best


def run(module_root: Path, data_dir: Path | None) -> tuple[list[str], list[str]]:
    """(hits, notes). hits is non-empty when the scan FAILS."""
    docs = restricted_documents(module_root)
    hits, notes = [], []
    files = tracked_files(module_root)
    if not files:
        raise Nfr6Error("git ls-files returned nothing; not a tracked module")
    citable = citable_titles(module_root, data_dir)
    for doc in docs:
        raw = locate(doc, module_root, data_dir)
        if raw is None:
            raise Nfr6Error(f"the local copy of {doc['hash'][:16]}... ({doc['title']!r}, "
                            f"manifested at {doc['local_path']}) is absent; cannot scan")
        # AR-rag4ds-026 F10: the copy scanned against IS the catalogued document,
        # or the scan does not run - a decoy at the manifested path is not a pass
        got = hashlib.sha256(raw.read_bytes()).hexdigest()
        if got != doc["hash"]:
            raise Nfr6Error(f"{raw} hashes to {got[:16]}..., not the catalogued "
                            f"{doc['hash'][:16]}... ({doc['title']!r}); the scan will not run "
                            f"against a copy that is not the document")
        text = normalise(raw.read_text(encoding="utf-8"))
        if len(text) < WINDOW:
            raise Nfr6Error(f"{raw}: shorter than one window ({len(text)} < {WINDOW})")
        windows = {text[i:i + WINDOW] for i in range(len(text) - WINDOW + 1)}
        shown = raw.relative_to(module_root) if raw.is_relative_to(module_root) else raw
        notes.append(f"{doc['hash'][:16]}... {doc['title']!r}: {len(windows)} windows of "
                     f"{WINDOW} chars from {shown}")
        # the document's own name may be cited (NFR7): blanked before windowing.
        # S9e part 3: so may its catalogued venue - a conference or journal name
        # the document prints as a running header - under the same head check
        own = []
        title = normalise(doc["title"])
        if title_at_head(title, text):
            own.append(title)
            notes.append(f"title ({len(title)} chars, starting at {text.find(title)}) is a "
                         f"citation, exempt")
        # AR-rag4ds-045 F1: the venue under its own, tighter shape - at most
        # VENUE_MAX_CHARS, at the head, AND printed at least VENUE_MIN_OCCURRENCES
        # times in the document (a running header); a passage from the head that
        # is declared the venue occurs once and is not exempt
        venue = normalise(doc.get("venue") or "")
        printed = text.count(venue) if venue else 0
        if (0 < len(venue) <= VENUE_MAX_CHARS and venue in text[:TITLE_HEAD_CHARS]
                and printed >= VENUE_MIN_OCCURRENCES and venue not in own):
            own.append(venue)
            notes.append(f"venue ({len(venue)} chars, printed {printed} times) is a citation, "
                         f"exempt")
        exempt = title if own and own[0] == title else ""
        # and so are the names of the OTHER catalogued documents, whether this one
        # prints them whole (its reference list) or shares a run with a title it
        # cites differently (two papers by the same authors with the same tail):
        # each blanked under the same slack rule, so a run of THIS document
        # continuing past a cited title is still a copy
        cited = [t for t in citable if t not in own]
        if cited:
            notes.append(f"{len(cited)} other catalogued title(s) are citations, exempt")
        for f in files:
            if f.resolve() == raw.resolve():
                hits.append(f"{f.relative_to(module_root)}: IS the restricted document (tracked)")
                continue
            try:
                hay = normalise(readable_text(f))
            except UnicodeDecodeError as exc:
                raise Nfr6Error(f"{f}: not UTF-8, cannot be scanned: {exc}") from exc
            copied = None
            for t in own + cited:
                hay, copied = blank_citations(hay, t, text)
                if copied:
                    what = ("its title" if t == exempt else "its venue" if t in own
                            else "a title it cites")
                    hits.append(f"{f.relative_to(module_root).as_posix()}: verbatim run of "
                                f"{copied[0]} chars of {doc['title']!r} around {what} (a "
                                f"copy, not a citation): {copied[1]!r}...")
                    break
            if copied:
                continue
            found = scan_text(hay, windows)
            if found:
                hits.append(f"{f.relative_to(module_root).as_posix()}: verbatim run of "
                            f"{found[0]} chars of {doc['title']!r}: {found[1]!r}...")
        notes.append(f"scanned {len(files)} files (tracked or not yet added)")
    if not docs:
        notes.append("no non-redistributable document catalogued; nothing to scan")
    return hits, notes


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    skip_absent = "--skip-absent" in argv
    data_env = os.environ.get("RAG4DS_DATA_DIR")
    data_dir = Path(data_env) if data_env else None
    try:
        hits, notes = run(MODULE_ROOT, data_dir)
    except Nfr6Error as exc:
        if skip_absent and " is absent; cannot scan" in str(exc):
            # AR-rag4ds-026 O13: check.py runs this step everywhere; a machine
            # without the restricted document (CI) says so loudly instead of
            # failing - an absent copy is a named skip, a wrong copy is exit 2
            print(f"nfr6_scan: SKIPPED (not a pass) - {exc}")
            return 0
        print(f"nfr6_scan: CANNOT SCAN - {exc}", file=sys.stderr)
        return 2
    for n in notes:
        print(f"nfr6_scan: {n}")
    if hits:
        print(f"nfr6_scan: FAILED - {len(hits)} tracked file(s) carry a verbatim run of at least "
              f"{WINDOW} characters of a non-redistributable document:", file=sys.stderr)
        for h in hits:
            print(f"  {h}", file=sys.stderr)
        return 1
    print("nfr6_scan: OK - no tracked file carries a passage of a non-redistributable document")
    return 0


if __name__ == "__main__":
    sys.exit(main())
