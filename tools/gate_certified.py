"""Certified-text approvals from a PI-signed file (S9e part 8; plan 6.4 as
amended 2026-09-24 on the PI's signed review list 2).

A candidate whose EXACT text a pack's validator certified in a confirm list
pinned in git, and which tools/pack_author.py filed, may be approved without
keys typed at the screen: this tool drives the real gate (rag4ds.gate.cli
GateSession - the same journal, staging transitions, session manifest and
transcript) from an approvals file the PI signed. On this road no person
reads the passage at the moment of approval, so the program's checks are the
approval's evidence, and every one runs before any decision is written, in
the dry run and the live run alike (AR-rag4ds-053 F1):

- the code: the module root is this tool's own tree, --code-commit is its
  HEAD, and git holds, unmodified, everything the process loaded or could
  still import from inside the root (tools/pack_author.py's code_paths;
  AR-rag4ds-053 F5); every rag4ds module and every road tool this process
  imported comes from that root's src/ and tools/ (r2 F6); and a module run
  from a bytecode cache inside the root ran its committed source, compiled
  (r2 O1, AR-rag4ds-051 r3 O14). The modules the road would import lazily
  (rag4ds.retrieve) are imported first, so the check covers them, and once
  it has passed no rag4ds module or road tool may load at all (r3 F8);
- the approvals file, the validator's certified list, the list as sent to
  the validator and the PI's signed review list are tracked by git and
  unmodified; the list as sent was committed before the certified list and
  has never changed - one commit touches it, the one that added it (r6
  F13, r7 F15) - and it is blank where the validator writes - empty boxes,
  an empty R field over the whole region the reader reads, no date - so an
  unchanged copy certifies nothing (r6 F14, r7 F16);
- the signed review list carries the approvals file's, the certified
  list's and the sent list's sha256, every "Your mark:" line on it reads A,
  and it carries exactly one sign-off, which names a reviewer and is dated;
- the reviewer is not the approver, and is the PI - the registered identity
  holding the OD-1 designee role - judged by identity, not spelling: the road
  is two people, the validator and the PI (AR-rag4ds-061 F1, F4);
- the certified list is the list as sent, line for line, changed only in
  its writing places, found from the list as sent's structure exactly as
  the reader reads it (r8 F17): in each item its one mark line, the R line
  directly under it and the rest of its R field to the item's end (written
  indented, so no ITEM or CERTIFICATION line can be made there), and the
  one Name/Date line after the one CERTIFICATION line. A writing line
  anywhere else is refused; nothing is added, removed or moved, so no note
  can stand where it is not read (r5 O1, F11, F12). The name and date are
  read from the one Name/Date writing place and no other line (r9); each
  entry id once;
- the certified list is dated and signed by the approver, and each approved
  entry id is an item of it with exactly ONE mark meaning as-written - its A
  box ticked with nothing on the R line, or no box ticked and the letter A
  alone on the R line (record 28.2's reading); an X tick, two ticks, a
  tick beside the validator's own wording, or any mark in a box other than
  x is refused by name (F2, r2 F2). The R field is read to the item's end -
  the next ITEM line, or CERTIFICATION after the last item - at any
  indentation (r3 F9, r4 F9); the mark line must hold the two boxes and
  nothing else, with the R line directly under it, and a box that appears
  twice in the item makes it unreadable (r3 O1, F10);
- each row's candidate is a PENDING entry candidate of the pack, in the
  gate's own queue order, hand-authored (an 'author:' extractor), named by
  the authoring manifest its run_manifest cites, whose draft is the file
  that manifest filed; its text is BYTE-equal to that draft item's text, and
  that text is the certified text with the list's line wraps joined (F3);
- its local source document hashes to its source_hash, and its span is the
  draft item's span and hashes to the draft's span sha256; and its text
  passes the quotation bound against that passage, measured here, since
  this road answers the gate's bound question for the person
  (AR-rag4ds-054 F1);
- each row's entry id is a valid id, used once in the file; a supersession
  names a current entry of the same pack, and a new entry an id that is
  neither current nor retired (F1); and the gate's own OD-1 and home-role
  rule admits the approver for it - a validator never approves a text the
  validator authored (r2 F7). The dry run reads the index as the live run does (r2 O3).
  A candidate that heads GROUPED members is refused: approving it would also
  close them, a decision the signed file does not name (r3 O2).

At the screen it answers only the approval's own prompts - a; y at the
quotation bound; the entry id; the supersession the file fixes; a BLANK
conflict answer - and only for the candidate the gate is showing; anything
else aborts. A transcript is required, and every run appends a manifest to
runs/gate-certified.jsonl naming the road and the files' hashes, and what
the run did: the session's counts, the rows approved, and whether the file
was applied whole (AR-rag4ds-054 F2). A run stopped part way is never
re-run on the same signed file: the queue has changed, so a new approvals
file and signature are needed.
Extraction proposals and edits are never approved here.

    python tools/gate_certified.py --approvals authoring/<approvals>.csv
        --certified-list authoring/<list>_certified.txt --sent-list authoring/<list>.txt
        --signed authoring/<review>_signed.txt
        --approver "K. Naruse" --code-commit <HEAD> --index-dir data/index --data-dir .
        --pack robotics --transcript data/transcripts/<name>.txt [--dry-run]

Approvals file: CSV with the columns n,id,entry_id,supersedes (supersedes
blank for a new entry, else the same id as entry_id). --index-dir is
required: the gate approves only beside its related-entry display, since
this road, like the live screen, never allows an unchecked approval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.abc
import importlib.machinery
import marshal
import re
import subprocess
import sys
import time
import types
from pathlib import Path

import yaml

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))
sys.path.insert(0, str(MODULE_ROOT / "tools"))
from gate_sheet import build_context  # noqa: E402
from pack_author import code_paths  # noqa: E402
from rich.console import Console  # noqa: E402

from rag4ds.auditlog.journal import Journal  # noqa: E402
from rag4ds.author.drafts import passage_sha256  # noqa: E402
from rag4ds.extract.strategy import quotation_bound_check  # noqa: E402
from rag4ds.gate.cli import (  # noqa: E402
    KEY_PROMPT, GateError, GateSession, make_source_resolver, transcript_path_is_publishable,
    use_utf8_console, write_transcript,
)
from rag4ds.gate.identities import IDENTITIES_FILE, IdentityError, IdentityRegistry  # noqa: E402
from rag4ds.index.build import IndexBuildError  # noqa: E402
from rag4ds.model.entry import ENTRY_ID_RE  # noqa: E402

# r3 F8: the modules the road otherwise imports lazily (gate_sheet.build_context,
# cli.make_related_finder), loaded here so the code check sees them; after it no
# rag4ds module or road tool may load at all (_LateImportSeal)
import rag4ds.retrieve.contract  # noqa: E402, F401
import rag4ds.retrieve.service  # noqa: E402, F401

BOUND_PROMPT = "the content honors the quotation bound? [y/N]: "
ID_PROMPT_START = "entry id ["
SUP_PROMPT = "supersedes existing id (blank = new entry): "
ROAD_TOOLS = ("gate_certified", "gate_sheet", "pack_author", "extract_pack")
CONFLICT_PROMPT = ("approve despite shown related entries? their ids, comma-separated "
                   "(blank = none): ")


class CertifiedError(RuntimeError):
    """A refusal before any decision is written - printed and exit 1."""


class _LateImportSeal(importlib.abc.MetaPathFinder):
    """r3 F8: armed once the code check has passed. The check covers only the
    modules already loaded, so any rag4ds module or road tool that would load
    after it is refused instead of run unchecked."""

    def find_spec(self, name, path=None, target=None):
        if name == "rag4ds" or name.startswith("rag4ds.") or name in ROAD_TOOLS:
            raise ImportError(f"certified road: {name} would load after the code check, so no "
                              f"check covers it; refusing")
        return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def code_binding(root: Path, code_commit: str) -> None:
    """The recorded code commit holds the code that decides (F5): this tool's
    tree, at HEAD, with everything it loaded or could import committed."""
    code_root = Path(__file__).resolve().parents[1]
    if root.resolve() != code_root:
        raise CertifiedError(f"the module root {root} is not the tree this tool's code comes from "
                             f"({code_root}); run the module's own tools/gate_certified.py")
    head = _git(root, "rev-parse", "HEAD")
    if (head.returncode != 0 or len(code_commit) < 7
            or not head.stdout.strip().startswith(code_commit)):
        raise CertifiedError(f"--code-commit {code_commit!r} is not this tree's HEAD "
                             f"({head.stdout.strip()[:12] or head.stderr.strip()})")
    loaded, entries = code_paths(root)
    status = _git(root, "--literal-pathspecs", "status", "--porcelain", "--", "src", "tools",
                  IDENTITIES_FILE, *entries, *loaded)
    if status.returncode != 0 or status.stdout.strip():
        raise CertifiedError("the code or the registry has uncommitted changes, so no commit "
                             "holds what would decide; commit first:\n" + status.stdout.strip())
    must_track = sorted(set(loaded) | {e for e in entries if (root / e).is_file()})
    listed = _git(root, "--literal-pathspecs", "ls-files", "-z", "--", *must_track)
    untracked = sorted(set(must_track) - set(listed.stdout.split("\0")))
    if listed.returncode != 0 or untracked:
        raise CertifiedError(f"code this process loaded from inside the module root is not "
                             f"tracked by git: {', '.join(untracked) or listed.stderr.strip()}")
    if entries:
        others = _git(root, "--literal-pathspecs", "ls-files", "--others", "-z", "--", *entries)
        suffixes = tuple(importlib.machinery.all_suffixes())
        stray = sorted(p for p in others.stdout.split("\0")
                       if p.endswith(suffixes) and "__pycache__" not in p.split("/"))
        if others.returncode != 0 or stray:
            raise CertifiedError(f"files Python could import from the module's import path are "
                                 f"not tracked by git: {', '.join(stray) or others.stderr.strip()}")
    src, tools = (root / "src").resolve(), (root / "tools").resolve()
    for name, mod in list(sys.modules.items()):
        if not isinstance(mod, types.ModuleType):
            continue
        f = mod.__dict__.get("__file__")
        if not isinstance(f, str):
            continue
        path = Path(f).resolve()
        # AR-rag4ds-053 r2 F6: the road's code this process imported is this tree's
        home = (src if name == "rag4ds" or name.startswith("rag4ds.")
                else tools if name in ROAD_TOOLS else None)
        if home is not None and not path.is_relative_to(home):
            raise CertifiedError(f"the road's code ({name}) was loaded from {path}, not from this "
                                 f"module root's {home.name}/; run the module's own "
                                 f"tools/gate_certified.py as a program")
        # r2 O1 (AR-rag4ds-051 r3 O14): a module run from a cache inside the root must be
        # its committed source, compiled - a stale cache would decide with other code
        cached = mod.__dict__.get("__cached__")
        if (path.is_relative_to(root.resolve()) and path.suffix == ".py"
                and isinstance(cached, str) and Path(cached).is_file()):
            code = marshal.loads(Path(cached).read_bytes()[16:])
            if code != compile(path.read_bytes(), f, "exec", dont_inherit=True):
                raise CertifiedError(f"the bytecode cache {cached} is not {path.name} as "
                                     f"committed; clear the __pycache__ directories or run "
                                     f"with -X pycache_prefix at a fresh empty directory")


def committed(root: Path, path: Path) -> None:
    """The file is tracked by git and unmodified - so a commit reproduces it."""
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        raise CertifiedError(f"{path} is outside the module root") from None
    if _git(root, "ls-files", "--error-unmatch", "--", rel).returncode != 0:
        raise CertifiedError(f"{rel} is not tracked by git; commit it first")
    status = _git(root, "status", "--porcelain", "--", rel)
    if status.returncode != 0 or status.stdout.strip():
        raise CertifiedError(f"{rel} has uncommitted changes; commit it first")


def load_approvals(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, restkey="_extra")
        if reader.fieldnames != ["n", "id", "entry_id", "supersedes"]:
            raise CertifiedError(f"{path}: columns must be n,id,entry_id,supersedes "
                                 f"(got {reader.fieldnames})")
        for i, row in enumerate(reader, 1):
            if row.pop("_extra", None) or any(v is None for v in row.values()):
                raise CertifiedError(f"{path}: row {i}: expected exactly 4 fields")
            if str(i) != row["n"].strip():
                raise CertifiedError(f"{path}: row {i}: n must be {i}, got {row['n']!r}")
            cid, eid, sup = row["id"].strip(), row["entry_id"].strip(), row["supersedes"].strip()
            if not cid.startswith("cand-") or not ENTRY_ID_RE.fullmatch(eid):
                raise CertifiedError(f"{path}: row {i}: a candidate id and a valid entry id "
                                     f"are needed")
            if sup and sup != eid:
                raise CertifiedError(f"{path}: row {i}: a supersession keeps the entry id "
                                     f"({eid!r}), got {sup!r}")
            rows.append({"n": i, "id": cid, "entry_id": eid, "supersedes": sup})
    if not rows:
        raise CertifiedError(f"{path}: no rows")
    for key in ("id", "entry_id"):
        seen = [r[key] for r in rows]
        if len(set(seen)) != len(seen):
            raise CertifiedError(f"{path}: a {key} appears on more than one row")
    return rows


MARK_RE = re.compile(r"^([ \t]*Your mark[ \t]+)\[[^\]\n]*\]([ \t]*A[ \t]+)\[[^\]\n]*\]([ \t]*X)"
                     r"[ \t]*$")
R_RE = re.compile(r"^([ \t]*)\[[^\]\n]*\]([ \t]*R[ \t]+your text:)")
DATE_RE = re.compile(r"^([ \t]*Name:.*?Date:)")
ISO_DATE = "[0-9]{4}-[0-9]{2}-[0-9]{2}"  # ASCII digits only (r6 O2: \d takes fullwidth ones)


def _history(root: Path, path: Path, *opts: str) -> list[str]:
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    return _git(root, "log", *opts, "--format=%H", "--", rel).stdout.split()


def sent_before(root: Path, sent: Path, cert: Path) -> None:
    """The list as sent was committed before the certified list came back,
    and has never changed (r6 F13, r7 F15): one commit touches it, followed
    through renames and copies (r8 F18), and the same one commit in its full
    history, merges included - and committed() holds the working file to
    HEAD. Git cannot
    see a list rewritten from scratch under a new path; the PI's signed
    review list, carrying its sha256, is what binds that file."""
    # the order first, on each file's own adding commit - not followed, since the returned
    # list is rightly a near-copy of the list as sent and git's copy detection follows it
    a_own, b_own = _history(root, sent, "--diff-filter=A"), _history(root, cert, "--diff-filter=A")
    a, b = (a_own[-1] if a_own else None), (b_own[-1] if b_own else None)
    if (a is None or b is None or a == b
            or _git(root, "merge-base", "--is-ancestor", a, b).returncode != 0):
        raise CertifiedError("the list as sent must be committed in an earlier commit than the "
                             "certified list")
    # followed through renames and copies, one commit; and the full history, which --follow
    # does not give (a merge commit that edits the file is missed by --follow), the same one
    touched = _history(root, sent, "--follow")
    full = _history(root, sent, "--full-history")
    if len(touched) != 1 or full != touched:
        raise CertifiedError(f"the list as sent was changed after the commit that added it, or "
                             f"came from another file ({len(touched)} commits touch it followed, "
                             f"{len(full)} in its full history); it must be the list exactly as "
                             f"it was sent")


ITEM_RE = re.compile(r"ITEM \d+(\s|$)")


def writing_places(sent: str) -> dict[int, str]:
    """r8 F17: the lines of the list as sent where the validator may write,
    found from the form's structure exactly as the reader reads it. Each
    item (from its ITEM line to the next, or to the one CERTIFICATION line
    after the last item) holds one mark line, the R line directly under it,
    and the rest of its R field to the item's end; after the CERTIFICATION
    line stands the one Name/Date line. A mark line, R line or Name/Date line
    anywhere else - in the header, the certification paragraph, another
    item's text - is refused, so nothing is written where the road does not
    read. Returns line index -> 'mark', 'r', 'rmore' or 'date'."""
    lines = sent.split("\n")
    items = [i for i, s in enumerate(lines) if ITEM_RE.match(s)]
    certs = [i for i, s in enumerate(lines) if re.match(r"CERTIFICATION\b", s)]
    if len(certs) != 1 or not items or certs[0] < items[-1]:
        raise CertifiedError("the list as sent must hold its items and then exactly one "
                             "CERTIFICATION line")
    places = {}
    bounds = items + certs
    for start, end in zip(bounds, bounds[1:], strict=False):
        marks = [i for i in range(start, end) if MARK_RE.match(lines[i])]
        if len(marks) != 1 or marks[0] + 1 >= end or not R_RE.match(lines[marks[0] + 1]):
            raise CertifiedError(f"the item at line {start + 1} of the list as sent has no one "
                                 f"mark line with its R line directly under it")
        places[marks[0]], places[marks[0] + 1] = "mark", "r"
        places.update((i, "rmore") for i in range(marks[0] + 2, end))
    dates = [i for i in range(certs[0] + 1, len(lines)) if DATE_RE.match(lines[i])]
    if len(dates) != 1:
        raise CertifiedError("the list as sent must hold one Name/Date line after its "
                             "CERTIFICATION line")
    places[dates[0]] = "date"
    for i, s in enumerate(lines):
        if i not in places and (MARK_RE.match(s) or R_RE.match(s) or DATE_RE.match(s)):
            raise CertifiedError(f"line {i + 1} of the list as sent is a writing line outside "
                                 f"the places the road reads: {s.strip()!r}")
    return places


def sent_is_blank(sent: str) -> None:
    """r6 F14: the list as sent carries no mark - every box empty, the R
    field underscores and blanks only, the date unfilled - so a certified
    list equal to it certifies nothing. r7 F16, r8 F17: checked over exactly
    the writing places, which are the places the reader reads."""
    lines = sent.split("\n")
    for i, kind in sorted(writing_places(sent).items()):
        s = lines[i]
        if kind in ("mark", "r"):
            bad = any(box.strip() for box in re.findall(r"\[([^\]\n]*)\]", s))
            if kind == "r":
                bad = bad or s[R_RE.match(s).end():].replace("_", "").strip() != ""
        elif kind == "rmore":
            bad = re.fullmatch(r"[ \t_]*", s) is None
        else:
            bad = s[DATE_RE.match(s).end():].replace("_", "").strip() != ""
        if bad:
            raise CertifiedError(f"line {i + 1} of the list as sent already carries a mark: "
                                 f"{s.strip()!r}; the list as sent is blank where the "
                                 f"validator writes")


def compare_with_sent(sent: str, cert: str) -> None:
    """r5 F11, F12 and O1: the certified list is the list as sent, changed
    only in its writing places (r8 F17) - inside the three boxes, in the R
    field (the rest of the "your text:" line, and the lines under it to the
    item's end, written indented so no ITEM or CERTIFICATION line can be
    made there) and the date. Nothing may be added, removed or moved."""
    s_lines, c_lines = sent.split("\n"), cert.split("\n")
    if len(s_lines) != len(c_lines):
        raise CertifiedError(f"the certified list has {len(c_lines)} lines and the list as sent "
                             f"{len(s_lines)}; nothing may be added to or removed from it")
    places = writing_places(sent)
    for i, (s, c) in enumerate(zip(s_lines, c_lines, strict=True), 1):
        kind = places.get(i - 1)
        ms, mr, md = MARK_RE.match(s), R_RE.match(s), DATE_RE.match(s)
        if s == c:
            continue
        if kind == "mark":
            mc = MARK_RE.match(c)
            ok = mc is not None and mc.groups() == ms.groups()
        elif kind == "r":
            mc = R_RE.match(c)
            ok = mc is not None and mc.groups() == mr.groups()
        elif kind == "rmore":
            ok = c[:1] in (" ", "\t")  # under the R field, indented: never a new ITEM line
        elif kind == "date":
            ok = (c.startswith(md.group(1))
                  and re.fullmatch(rf"[ \t]*_*{ISO_DATE}_*[ \t]*", c[len(md.group(1)):])
                  is not None)
        else:
            ok = False
        if not ok:
            raise CertifiedError(f"line {i} of the certified list differs from the list as sent "
                                 f"outside a box, the R field and the date: {c.strip()!r}")


def certified_items(text: str, date_line: int) -> tuple[str, dict[str, dict]]:
    """The validator's certified list: (the signer's name, entry id ->
    proposed replacement and whether it bears exactly one as-written mark).
    date_line is the index of the one Name/Date writing place, found by
    writing_places() on the list as sent; the signer and date are read from
    that line and no other (r8 F17, r9)."""
    if len(re.findall(r"^CERTIFICATION\b", text, re.M)) != 1:  # r5 F12
        raise CertifiedError("the certified list must hold exactly one CERTIFICATION line")
    lines = text.split("\n")
    cert = (re.fullmatch(rf"[ \t]*Name:[ \t]*(\S.*?)[ \t]+Date:[ \t]*_*({ISO_DATE})_*[ \t]*",
                         lines[date_line]) if 0 <= date_line < len(lines) else None)
    if not cert:
        raise CertifiedError("the certified list's CERTIFICATION line carries no name and date")
    blocks = re.split(r"^ITEM \d+\s+", text, flags=re.M)[1:]
    out = {}
    for n, block in enumerate(blocks, 1):
        eid = block.split(None, 1)[0]
        if eid in out:  # r5 F11: one reading per entry, never the last copy's
            raise CertifiedError(f"entry {eid!r} heads more than one ITEM of the certified list")
        m = re.search(r"^\s*Proposed replacement\s*$(.*?)^\s*Your mark", block, re.M | re.S)
        if not m:
            raise CertifiedError(f"certified list item {eid}: no proposed replacement found")

        boxes = {}
        for letter in "AXR":
            found = re.findall(rf"\[([^\]\n]*)\]\s*{letter}\b", block)
            # r2 F2: a box is empty or x/X; any other mark (a tick, a v) is unreadable
            boxes[letter] = found[0].strip() if len(found) == 1 else None
        unreadable = [k for k, v in boxes.items() if v not in ("", "x", "X")]
        # r3 F9 and O1, r4 F9, r5 F12: an item ends at the next ITEM line (the split);
        # only the last ends at the form's one CERTIFICATION line. The mark line holds
        # the two boxes and nothing else, the R line follows it directly, and the R
        # field is every character after "your text:" to the item's end
        item_text = (re.split(r"^CERTIFICATION\b", block, maxsplit=1, flags=re.M)[0]
                     if n == len(blocks) else block)
        tail = re.search(r"^[ \t]*Your mark[ \t]+\[[^\]\n]*\][ \t]*A[ \t]+\[[^\]\n]*\][ \t]*X"
                         r"[ \t]*\r?\n[ \t]*\[[^\]\n]*\][ \t]*R[ \t]+your text:(.*)\Z",
                         item_text, re.M | re.S)
        if not tail:
            unreadable.append("the mark lines")
        r_text = " ".join(tail.group(1).replace("_", " ").split()) if tail else ""
        a, x, r = (boxes[k] in ("x", "X") for k in "AXR")
        # exactly one mark, meaning as written (F2): anything else is not certified
        as_written = (not unreadable and not x and not r
                      and ((a and r_text == "") or (not a and r_text == "A")))
        out[eid] = {"proposed": _norm(m.group(1)), "as_written": as_written,
                    "marks": f"boxes {boxes}, R line {r_text!r}"}
    return cert.group(1).strip(), out


def check_signed(text: str, approvals_sha: str, list_sha: str, sent_sha: str) -> str:
    """Checks the PI's signed review list; returns the reviewer's name as signed."""
    for need, what in ((approvals_sha, "approvals file"), (list_sha, "certified list"),
                       (sent_sha, "list as sent")):
        if need not in text:
            raise CertifiedError(f"the signed review list does not carry the {what}'s sha256 "
                                 f"{need}")
    marks = re.findall(r"Your mark:\s*([^\n]*)", text)
    if not marks:
        raise CertifiedError("the signed review list has no 'Your mark:' line")
    bad = [m for m in marks if m.replace("_", "").strip() != "A"]
    if bad:
        raise CertifiedError(f"every decision on the signed review list must be marked A; got "
                             f"{[b.replace('_', '').strip() for b in bad]}")
    # AR-rag4ds-061 r2 F4: exactly one sign-off line - a second one would be ignored
    if len(re.findall(r"Reviewed and marked by:", text)) != 1:
        raise CertifiedError("the signed review list must carry exactly one 'Reviewed and marked "
                             "by:' sign-off line")
    sign_off = re.search(rf"Reviewed and marked by:\s*([^_\s].*?)\s+Date:\s*_*{ISO_DATE}", text)
    if not sign_off:
        raise CertifiedError("the signed review list's sign-off carries no reviewer and date")
    return sign_off.group(1).strip()


class CertifiedReader:
    """The gate's read callback: answers only an approval's own prompts, for
    the row whose candidate the gate is showing; anything else aborts."""

    def __init__(self, rows: list[dict], console: Console):
        self.rows, self.console, self.pos, self.current = rows, console, 0, None
        self.session = None

    def _say(self, prompt: str, answer: str) -> str:
        self.console.print(f"certified: {prompt}{answer}")
        return answer

    def __call__(self, prompt: str) -> str:
        if prompt == KEY_PROMPT:
            if self.pos >= len(self.rows):
                self.console.print("certified: every row answered - quitting")
                return "q"
            row = self.rows[self.pos]
            shown = getattr(self.session, "current_cid", None)
            if shown != row["id"]:
                raise GateError(f"certified road: the gate is deciding {shown!r} but the next row "
                                f"{row['n']} is {row['id']!r}; the session is aborted")
            self.current = row
            self.pos += 1
            return self._say(prompt, "a")
        if self.current is None:
            raise GateError(f"certified road: prompt {prompt!r} before any row")
        if prompt == BOUND_PROMPT:
            return self._say(prompt, "y")
        if prompt.startswith(ID_PROMPT_START):
            return self._say(prompt, self.current["entry_id"])
        if prompt == SUP_PROMPT:
            return self._say(prompt, self.current["supersedes"])
        if prompt == CONFLICT_PROMPT:
            return self._say(prompt, "")  # a certified approval never records a conflict
        raise GateError(f"certified road: unexpected prompt {prompt!r} at row {self.current['n']}; "
                        f"the session is aborted")


def check_rows(root: Path, rows: list[dict], items: dict, ctx, pack: str, data_dir: Path) -> None:
    store = ctx.staging
    cands = store.candidates()
    runs = root / "runs" / "authoring.jsonl"
    authoring = ({m["run_id"]: m for m in Journal(runs).read_all() if m.get("kind") == "authoring"}
                 if runs.is_file() else {})
    resolve = make_source_resolver(Journal(root / "runs" / "ingest.jsonl"), data_dir)
    for row in rows:
        n, cid = row["n"], row["id"]
        rec = cands.get(cid)
        if rec is None or store.state_of(cid) != "PENDING":
            raise CertifiedError(f"row {n}: {cid} is not a PENDING candidate")
        c = rec["candidate"]
        if c.get("kind") != "entry" or c.get("pack") != pack:
            raise CertifiedError(f"row {n}: {cid} is not an entry candidate of pack {pack!r}")
        if not str(c.get("extractor", "")).startswith("author:"):
            raise CertifiedError(f"row {n}: {cid} was proposed by {c.get('extractor')!r}; only a "
                                 f"hand-authored candidate may be approved from a file")
        item = items.get(row["entry_id"])
        if item is None:
            raise CertifiedError(f"row {n}: entry {row['entry_id']!r} is not an item of the "
                                 f"certified list")
        if not item["as_written"]:
            raise CertifiedError(f"row {n}: entry {row['entry_id']!r} is not certified as written "
                                 f"by exactly one mark ({item['marks']})")
        ref = str(c.get("run_manifest", ""))
        man = authoring.get(ref.split("#", 1)[1]) if ref.startswith("authoring.jsonl#") else None
        if man is None or cid not in man.get("candidates", []):
            raise CertifiedError(f"row {n}: {cid} is not named by an authoring manifest")
        draft = root / "authoring" / man["draft"]
        if (not draft.is_file()
                or hashlib.sha256(draft.read_bytes()).hexdigest() != man["draft_sha256"]):
            raise CertifiedError(f"row {n}: the draft {man['draft']} is not the one filed")
        content = c["proposed"]["content"]
        mine = [it for it in yaml.safe_load(draft.read_text(encoding="utf-8"))["items"]
                if it["content"] == content]
        # byte-equal to the draft item, which is the certified text, wraps joined (F3)
        if len(mine) != 1 or content != item["proposed"]:
            raise CertifiedError(f"row {n}: {cid}'s text is not byte-equal to the certified text "
                                 f"of {row['entry_id']!r} (the list's line wraps joined)")
        sp = mine[0]["span"]
        path = resolve(c["source_hash"])
        raw = path.read_bytes() if path is not None else None
        if raw is None or hashlib.sha256(raw).hexdigest() != c["source_hash"]:
            raise CertifiedError(f"row {n}: {cid}'s source passage cannot be checked: its local "
                                 f"document is missing or does not hash to its source_hash")
        passage = raw.decode("utf-8")[c["span_start"]:c["span_end"]]
        if ((sp["start"], sp["end"]) != (c["span_start"], c["span_end"])
                or passage_sha256(passage) != sp["sha256"]):
            raise CertifiedError(f"row {n}: {cid}'s source passage does not match its recorded "
                                 f"hash")
        # AR-rag4ds-054 F1: the gate's quotation-bound question is answered "y" on this road,
        # so the bound is measured here, on the passage, as the filing measured it - never
        # taken on trust from a filing run at another commit
        bound = quotation_bound_check(content, passage)
        if not bound["ok"]:
            raise CertifiedError(f"row {n}: {cid}'s text fails the quotation bound against its "
                                 f"passage (NFR5): {bound['reason']}")


def check_entries(rows: list[dict], session: GateSession, ctx, pack: str) -> None:
    """What the gate would refuse inside the session, refused before it (F1):
    a supersession names a current entry of this pack; a new entry an id
    that is neither current nor retired."""
    cands = ctx.staging.candidates()
    for row in rows:
        eid = row["entry_id"].lower()
        current = session._versions.get(eid)  # the session's own journal fold, read only
        if row["supersedes"]:
            prev = session._current_entry(eid)
            if current is None or prev is None:
                raise CertifiedError(f"row {row['n']}: supersession target {row['entry_id']!r} "
                                     f"is not a current entry")
            if prev.get("pack") != cands[row["id"]]["candidate"]["pack"]:
                raise CertifiedError(f"row {row['n']}: {row['entry_id']!r} lives outside pack "
                                     f"{pack!r}; a supersession keeps the entry's home")
        elif current is not None or eid in session._retired:
            raise CertifiedError(f"row {row['n']}: {row['entry_id']!r} already exists (current "
                                 f"or retired); a new entry needs a new id")
        try:  # r2 F7: the gate's own OD-1 and home-role rule, run before any decision
            session._refuse_self_approval(row["id"], cands[row["id"]]["candidate"])
        except GateError as exc:
            raise CertifiedError(f"row {row['n']}: {exc}") from None
        # r3 O2: a head decision also closes its GROUPED members (reject-duplicate),
        # a decision the signed file does not name - such a head is decided at the screen
        members = session._members_of(row["id"])
        if members:
            raise CertifiedError(f"row {row['n']}: {row['id']} heads grouped member(s) "
                                 f"{', '.join(members)}; approving it would also close them, "
                                 f"which the signed file does not name - decide it at the screen")


def run(root: Path, approvals: Path, certified_list: Path, signed: Path, approver: str,
        code_commit: str, index_dir: Path | None, data_dir: Path, pack: str,
        transcript: Path | None, dry_run: bool, out=None, clock=None,
        sent_list: Path | None = None) -> int:
    seal = _LateImportSeal()
    try:
        return _run(root, approvals, certified_list, signed, approver, code_commit, index_dir,
                    data_dir, pack, transcript, dry_run, out, clock, seal, sent_list)
    finally:
        if seal in sys.meta_path:
            sys.meta_path.remove(seal)


def _run(root, approvals, certified_list, signed, approver, code_commit, index_dir, data_dir,
         pack, transcript, dry_run, out, clock, seal, sent_list) -> int:
    out = out or sys.stdout
    root = Path(root)
    try:
        if clock is None and time.localtime().tm_gmtoff != 9 * 3600:
            raise CertifiedError("this machine's local clock is not JST (UTC+9); refusing")
        if transcript is None:
            raise CertifiedError("a transcript is required on this road (--transcript)")
        if sent_list is None:
            raise CertifiedError("the list as sent is required on this road (--sent-list)")
        code_binding(root, code_commit)
        sys.meta_path.insert(0, seal)  # r3 F8: nothing of the road's loads unchecked from here
        for p in (approvals, certified_list, signed, sent_list):
            committed(root, p)
        rows = load_approvals(approvals)
        shas = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (approvals, certified_list, signed, sent_list)}
        reviewer = check_signed(signed.read_text(encoding="utf-8"), shas[approvals],
                                shas[certified_list], shas[sent_list])
        sent_before(root, sent_list, certified_list)
        compare_with_sent(sent_list.read_text(encoding="utf-8"),
                          certified_list.read_text(encoding="utf-8"))
        sent_is_blank(sent_list.read_text(encoding="utf-8"))
        date_line, = (i for i, kind in writing_places(sent_list.read_text(encoding="utf-8"))
                      .items() if kind == "date")
        signer, items = certified_items(certified_list.read_text(encoding="utf-8"), date_line)
        if signer != approver:
            raise CertifiedError(f"the certified list is signed by {signer!r}, not the approver "
                                 f"{approver!r}")
        transcript.parent.mkdir(parents=True, exist_ok=True)
        refused = transcript_path_is_publishable(transcript)
        if refused:
            raise CertifiedError(refused)
        identities = IdentityRegistry.load(root / IDENTITIES_FILE)
        identities.resolve(approver)
        # AR-rag4ds-061 F1: this road is two people - the validator who certified the text
        # and is the approver of record, and the reviewer whose signed review list
        # countersigns it (the PI's review list 2, D3; plan 6.4). Since 2026-10-05 the PI is
        # also a robotics validator; one registered person may not fill both places,
        # judged by identity, not by spelling. An unregistered reviewer is refused too.
        if identities.same_person(reviewer, approver):
            raise CertifiedError(f"the signed review list is signed by {reviewer!r}, who is the "
                                 f"approver {approver!r}: this road needs a second person")
        # AR-rag4ds-061 r2 F4: the countersignature is the PI's (plan 6.4: "an approvals file
        # the PI signed"); the PI is the identity holding the OD-1 designee role. Anyone else -
        # another validator, the alternate, the drafting author of the text - is refused.
        if "od1-designee" not in identities.resolve(reviewer).roles:
            raise CertifiedError(f"the signed review list is signed by {reviewer!r}, not the PI: "
                                 f"only the PI countersigns this road")
        ctx = build_context(root, approver, code_commit, index_dir,  # the dry run too (r2 O3)
                            data_dir, identities, clock)
        check_rows(root, rows, items, ctx, pack, data_dir)
        console = Console(record=True, markup=False, highlight=False)
        reader = CertifiedReader(rows, console)
        session = GateSession(ctx, console, reader, pack=pack)
        reader.session = session
        queue = session.queue()
        if [r["id"] for r in rows] != queue:
            raise CertifiedError(f"the approvals file's rows {[r['id'] for r in rows]} are not the "
                                 f"gate's queue for pack {pack!r} {queue}; nothing decided")
        check_entries(rows, session, ctx, pack)
        print(f"gate_certified: {len(rows)} certified approval(s) checked against the queue and "
              f"the journal; approver {approver!r}; code {code_commit}; "
              f"{'DRY RUN - no session' if dry_run else 'driving the gate'}", file=out)
        for r in rows:
            print(f"  {r['n']} {r['id']} -> {r['entry_id']}"
                  f"{' supersedes ' + r['supersedes'] if r['supersedes'] else ' (new)'}, "
                  f"no conflict", file=out)
        if dry_run:
            return 0
    except (CertifiedError, GateError, IdentityError, IndexBuildError, ImportError,
            OSError) as exc:
        print(f"gate_certified: REFUSED - {exc}", file=sys.stderr)
        return 1
    console.print(f"CERTIFIED-FILE SESSION: approvals {approvals.as_posix()} sha256 "
                  f"{shas[approvals]}; certified list {certified_list.as_posix()} sha256 "
                  f"{shas[certified_list]}; PI's signed review ({reviewer}) "
                  f"{signed.as_posix()} sha256 {shas[signed]}; tools/gate_certified.py at "
                  f"{code_commit} is the keyboard")
    rc = 0
    failed = None
    try:
        session.run()
    except Exception as exc:  # noqa: BLE001 - recorded in the manifest and reported (exit 1)
        failed = exc
    except BaseException as exc:
        failed = exc
        raise
    finally:
        rc = write_transcript(console, transcript, session_failed=failed is not None)
        # AR-rag4ds-054 F2: the manifest says what the run did - the session's counts, the
        # rows whose candidate ended APPROVED, and whether the file was applied whole: every
        # row's candidate APPROVED (a failed session is recorded apart, as session_failed)
        counts = dict(session.summary.counts)
        approved = [r["id"] for r in rows if ctx.staging.state_of(r["id"]) == "APPROVED"]
        complete = approved == [r["id"] for r in rows]
        Journal(root / "runs" / "gate-certified.jsonl").append({
            # schema 3 (AR-rag4ds-061 r2 F4): the reviewer who countersigned is recorded
            "kind": "gate-certified", "schema": 3, "road": "signed file", "reviewer": reviewer,
            "approvals": approvals.as_posix(), "approvals_sha256": shas[approvals],
            "certified_list": certified_list.as_posix(),
            "certified_list_sha256": shas[certified_list],
            "sent_list": sent_list.as_posix(), "sent_list_sha256": shas[sent_list],
            "signed": signed.as_posix(), "signed_sha256": shas[signed],
            "approver": approver, "code_commit": code_commit, "pack": pack,
            "timestamp": (clock() if clock is not None else time.strftime("%Y-%m-%d %H:%M:%S JST")),
            "rows": [{"id": r["id"], "entry_id": r["entry_id"], "supersedes": r["supersedes"]}
                     for r in rows],
            "rows_answered": reader.pos, "session_failed": failed is not None,
            "counts": counts, "approved": approved, "complete": complete,
            "transcript": transcript.as_posix(),
        })
    if failed is not None:
        print(f"gate_certified: session aborted after row {reader.pos}: {failed}", file=sys.stderr)
        return 1
    if not complete:
        print(f"gate_certified: {reader.pos} of {len(rows)} rows answered, counts "
              f"{session.summary.counts} - NOT complete", file=sys.stderr)
        return 1
    print(f"gate_certified: done - {reader.pos} approvals; counts {session.summary.counts}",
          file=out)
    return rc


def main(argv: list[str] | None = None) -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-root", default=str(MODULE_ROOT))
    for flag in ("--approvals", "--certified-list", "--sent-list", "--signed", "--approver",
                 "--code-commit", "--index-dir", "--data-dir", "--pack", "--transcript"):
        ap.add_argument(flag, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return run(Path(a.module_root), Path(a.approvals), Path(a.certified_list), Path(a.signed),
               a.approver, a.code_commit, Path(a.index_dir), Path(a.data_dir), a.pack,
               Path(a.transcript), a.dry_run, sent_list=Path(a.sent_list))


if __name__ == "__main__":
    sys.exit(main())
