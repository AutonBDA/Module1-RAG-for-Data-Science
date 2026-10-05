"""File a hand-authored draft into ONE domain pack's staging (S9e part 7).

The pack counterpart of tools/s8_seed.py's filing step. Plan section 3 ("the
author is just another proposer"), 6.4 item 3 and R4 ("more hand-authoring
through the same gate") never confined authorship to the shared layers; the
S8 code did, because S8 was the shared-layer increment. A pack draft names
its pack at the top (rag4ds.author.drafts: its items are then all
domain-layer, the entry model's FR1 rule), and each item cites one of the
pack's OWN catalogued documents by the local copy's file stem - the name the
pack's entries already use in their source reference, corpus:<pack>/<stem>.
The documents are loaded through tools/extract_pack.load_documents, which
hash-checks every local copy against its catalogue record before its text is
used; nothing outside the pack can be cited, because nothing outside the pack
is offered as a source.

    python tools/pack_author.py robotics authoring/<draft>.yaml --dry-run
    python tools/pack_author.py robotics authoring/<draft>.yaml

--dry-run builds and checks every candidate exactly as the live run would -
the author's registration, the span's hash, the quotation bound, the passage
coverage, and what staging already holds - prints each, writes nothing, and
states the live run's verdict: if staging already holds any item, the live
run refuses the whole draft, and so does the dry run (AR-rag4ds-051 F3).
Both refuse a machine whose clock is not JST. A live run also refuses unless
it runs from the module's own tree and everything that decides what it files
is committed and tracked - the code, the identity registry, the pack's
catalogue and the draft - so the manifest's code_commit names what ran
(AR-rag4ds-045 F4, AR-rag4ds-049 r1 O5, AR-rag4ds-051 F2). The code is every
module file the process loaded from inside the module root and every file it
could still import from an import-path entry there, ignored ones included,
since git's status is silent about an ignored path (AR-rag4ds-052 F1); it files through
rag4ds.author.drafts.file_draft for this pack, whose authoring manifest goes
to runs/authoring.jsonl like the S8 seed's. The pack boundary itself is the
draft road's (AR-rag4ds-051 F1): each source carries its pack, and an item
citing another pack's document is refused whoever the caller is.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import subprocess
import sys
import time
import types
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))
sys.path.insert(0, str(MODULE_ROOT / "tools"))
from extract_pack import ExtractPackError, load_documents  # noqa: E402

from rag4ds.auditlog.journal import Journal  # noqa: E402
from rag4ds.author.drafts import (  # noqa: E402
    DraftError, SourceDoc, build_candidates, file_draft, filed_before, load_draft,
    passage_coverage,
)
from rag4ds.gate.identities import IDENTITIES_FILE, IdentityError, IdentityRegistry  # noqa: E402
from rag4ds.model.entry import SHARED_PACK  # noqa: E402
from rag4ds.staging.store import StagingStore  # noqa: E402


class PackAuthorError(RuntimeError):
    """A refusal of the pack authoring run, named."""


def jst_now() -> str:
    """The run's timestamp; a machine whose clock is not JST is refused, since
    the manifest's timestamp claims JST (the repository's operating rules, section 5; the guard is
    witnessed over a table of wrong offsets in tests/test_jst_guards.py)."""
    if time.localtime().tm_gmtoff != 9 * 3600:
        raise PackAuthorError("this machine's local clock is not JST (UTC+9); refusing")
    return time.strftime("%Y-%m-%d %H:%M:%S JST")


def pack_sources(pack: str, module_root: Path) -> dict[str, SourceDoc]:
    """The pack's own catalogued text documents, keyed by the local copy's
    file stem; each is hash-checked by load_documents before it is used."""
    docs, _skipped = load_documents(pack, module_root)
    out: dict[str, SourceDoc] = {}
    for rec, path, text in docs:
        stem = path.stem
        if stem in out:
            raise PackAuthorError(f"two catalogued documents of pack {pack!r} share the file stem "
                                  f"{stem!r}; a draft could not name one of them unambiguously")
        # the document's pack travels with it, so the road itself can refuse an item
        # citing outside the draft's pack (AR-rag4ds-051 F1)
        out[stem] = SourceDoc(rec["hash"], text, bool(rec.get("redistributable")), pack)
    return out


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def _rel(path: Path, root: Path) -> str | None:
    """`path` as a git path relative to the resolved `root`, or None outside it."""
    p = path.resolve()
    return p.relative_to(root).as_posix() if p.is_relative_to(root) else None


def code_paths(module_root: Path) -> tuple[list[str], list[str]]:
    """The code this process ran, or could still run, from inside the module
    root (AR-rag4ds-052 F1), as git paths relative to it: the file of every
    module already loaded from inside the root, and every import-path entry
    inside the root, whose files a later import could still load. Only module
    objects are read, through their own __file__; a module with none (built
    in, frozen, a namespace package) was not loaded from a file."""
    root = module_root.resolve()
    loaded = set()
    for mod in list(sys.modules.values()):
        if isinstance(mod, types.ModuleType):
            f = mod.__dict__.get("__file__")
            if isinstance(f, str) and (rel := _rel(Path(f), root)) is not None:
                loaded.add(rel)
    # an empty entry is the working directory, which Python searches too
    entries = {rel for e in list(sys.path) if isinstance(e, str)
               and (rel := _rel(Path(e or "."), root)) is not None}
    return sorted(loaded), sorted(entries)


def uncommitted(module_root: Path, pack: str, draft_path: Path) -> str:
    """Why a live run may not file, or "" when it may (AR-rag4ds-051 F2, F7).
    The recorded commit must hold the code that ran, so this file's tree must
    BE the module root, AND the road's code this process actually imported -
    the rag4ds package and extract_pack - must come from that root's src/ and
    tools/: a run from one worktree into another, or from a process that
    imported the road from elsewhere, would record a commit for code it never
    ran. Run as a program (python tools/pack_author.py), the tool imports its
    own tree's code, which is the supported road. Everything that decides what
    is filed must then be committed: the code (src, tools, and every import-path
    entry and loaded module file inside the root), the identity registry that
    admits the author, the pack's catalogue that admits the sources, and the
    draft - which must also be TRACKED, because git's status is silent about an
    ignored path, and inside the module root. The same silence covers code
    (AR-rag4ds-052 F1): the module ignores *.pyc, Python imports a sourceless
    X.pyc wherever no X.py stands earlier on the path, and tools/ heads that
    path - so every module file loaded from inside the root must be tracked,
    and no untracked file Python could import may stand on an import-path
    entry inside it, loaded yet or not. Bytecode caches under __pycache__ are
    the one exception: a stale cache is AR-rag4ds-051 r3 O14's, answered by
    running with -X pycache_prefix at a fresh empty directory."""
    code_root = Path(__file__).resolve().parents[1]
    if module_root.resolve() != code_root:
        return (f"the module root {module_root} is not the tree this tool's code comes from "
                f"({code_root}); run the module's own tools/pack_author.py")
    # AR-rag4ds-051 F7: the road's code is whatever this process imported, which
    # is not necessarily this file's tree - a process that imported rag4ds or
    # extract_pack from elsewhere would file that code under this tree's commit
    for name, home in (("rag4ds", "src"), ("extract_pack", "tools")):
        loaded = Path(sys.modules[name].__file__).resolve()
        if not loaded.is_relative_to((module_root / home).resolve()):
            return (f"the road's code ({name}) was loaded from {loaded}, not from this module "
                    f"root's {home}/; run the module's own tools/pack_author.py as a program")
    try:
        rel = str(Path(draft_path).resolve().relative_to(module_root.resolve()))
    except ValueError:  # AR-rag4ds-051 r2 O11: named, not a traceback
        return f"the draft {draft_path} is outside the module root {module_root}"
    if _git(module_root, "ls-files", "--error-unmatch", "--", rel).returncode != 0:
        return f"the draft {rel} is not tracked by git, so no commit could reproduce it"
    loaded, entries = code_paths(module_root)
    status = _git(module_root, "--literal-pathspecs", "status", "--porcelain", "--", "src",
                  "tools", IDENTITIES_FILE, f"packs/{pack}/corpus", rel, *entries, *loaded)
    if status.returncode != 0 or status.stdout.strip():
        return ("the code, the registry, the pack's catalogue or the draft has uncommitted "
                "changes, so no manifest could name what ran; commit first:\n"
                + status.stdout.strip())
    # AR-rag4ds-052 F1: git's status says nothing of an ignored file, so what it
    # cannot see is asked of git directly - first the module files this process
    # has loaded, with any import-path entry that is itself a file (an archive
    # Python imports from), then everything it could still import from the path
    must_track = sorted(set(loaded) | {e for e in entries if (module_root / e).is_file()})
    listed = _git(module_root, "--literal-pathspecs", "ls-files", "-z", "--", *must_track)
    untracked = sorted(set(must_track) - set(listed.stdout.split("\0")))
    if listed.returncode != 0 or untracked:
        return (f"code this process loaded, or can import, from inside the module root is not "
                f"tracked by git, so no commit holds it: "
                f"{', '.join(untracked) or listed.stderr.strip()}")
    if entries:
        others = _git(module_root, "--literal-pathspecs", "ls-files", "--others", "-z", "--",
                      *entries)
        suffixes = tuple(importlib.machinery.all_suffixes())
        stray = sorted(p for p in others.stdout.split("\0")
                       if p.endswith(suffixes) and "__pycache__" not in p.split("/"))
        if others.returncode != 0 or stray:
            return (f"files Python could import from the module's import path are not tracked "
                    f"by git, so a live run could load them with no commit holding them: "
                    f"{', '.join(stray) or others.stderr.strip()}; remove them first")
    return ""


def run(pack: str, draft_path: Path, module_root: Path, dry_run: bool, out=None) -> int:
    out = out or sys.stdout
    try:
        stamp = jst_now()  # refused before anything is read, on a dry run as on a live one
        if pack == SHARED_PACK:
            raise PackAuthorError(f"pack {SHARED_PACK!r} is filed by tools/s8_seed.py; this tool "
                                  f"files a domain pack's draft")
        registry = IdentityRegistry.load(module_root / IDENTITIES_FILE)
        draft = load_draft(draft_path, registry)
        named = draft.get("pack", SHARED_PACK)
        if named != pack:
            raise PackAuthorError(f"{draft_path.name} is a draft for pack {named!r}, not {pack!r}")
        sources = pack_sources(pack, module_root)
        store = StagingStore(module_root / "stage" / "candidates.jsonl")
        if dry_run:
            cands = build_candidates(draft, sources, "authoring.jsonl#DRY-RUN")
            prior = {id(c): pid for c, pid in filed_before(cands, store)}
            for i, (c, item) in enumerate(zip(cands, draft["items"], strict=True), 1):
                passage = sources[item["source"]].text[c["span_start"]:c["span_end"]]
                state = f"ALREADY FILED as {prior[id(c)]}" if id(c) in prior else "would file"
                cov = passage_coverage(c["proposed"]["content"], passage)
                print(f"  {i}. {state}: pack {c['pack']}, layer {c['layer']}, "
                      f"{c['proposed']['type']}, source {item['source']} "
                      f"[{c['span_start']}:{c['span_end']}], coverage {cov:.2f}, "
                      f"extractor {c['extractor']}", file=out)
                print(f"     {c['proposed']['content']}", file=out)
            if prior:
                # the live run files a draft once or not at all (file_draft, no
                # allow_partial): the dry run states that verdict (AR-rag4ds-051 F3)
                print(f"pack_author: DRY RUN - the live run would REFUSE this draft: "
                      f"{len(prior)} item(s) already in staging "
                      f"({', '.join(sorted(prior.values()))}); a draft files once, so nothing "
                      f"would be filed; nothing written", file=out)
                return 1
            print(f"pack_author: DRY RUN - {len(cands)} candidate(s) would be filed into "
                  f"{pack!r}; nothing written", file=out)
            return 0
        pending = uncommitted(module_root, pack, draft_path)
        if pending:
            raise PackAuthorError(pending)
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=module_root,
                                capture_output=True, text=True, check=True).stdout.strip()
        manifest = file_draft(draft_path, sources, store,
                              Journal(module_root / "runs" / "authoring.jsonl"), commit, stamp,
                              registry, pack=pack)
    except (PackAuthorError, DraftError, IdentityError, ExtractPackError) as exc:
        print(f"pack_author: REFUSED - {exc}", file=sys.stderr)
        return 1
    print(f"pack_author: filed {manifest['count']} candidate(s) into {pack!r} at {commit}, run "
          f"{manifest['run_id']}: {', '.join(manifest['candidates'])}", file=out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pack")
    ap.add_argument("draft")
    ap.add_argument("--module-root", default=str(MODULE_ROOT))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return run(a.pack, Path(a.draft), Path(a.module_root), a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
