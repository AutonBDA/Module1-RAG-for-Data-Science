"""S8 seed run (plan S8): catalogue the shared-corpus sources and file the
hand-authored drafts into staging, at the current git HEAD, with every run
manifest the pipeline requires (NFR2). Run from the module root:

    python tools/s8_seed.py                        the two S8 drafts
    python tools/s8_seed.py shared_tasks_2.yaml    only the drafts named

A draft named must be one of DRAFTS below. Every run catalogues every source
(a catalogued one is reused), then files the drafts it names - so a later
draft is filed without re-filing the S8 drafts, whose items staging holds and
which, filed again, would stop the run before the later draft (file_draft
refuses a draft with nothing new).

Sources (all located relative to the module root, which is the gate
session's --data-dir for the shared layers):
  data/shared/crisp-dm-1.0.txt      the CRISP-DM 1.0 guide's text, extracted
                                    from the PDF with pypdf - gitignored (NFR6:
                                    the guide is copyrighted, not redistributable;
                                    ingest refuses a non-redistributable document
                                    that git would commit - AR-rag4ds-026 F2);
                                    fetch and extraction instructions are printed
                                    when the file is absent, and its sha256 is
                                    checked against the value recorded below
  authoring/sources/shared-task-patterns.md
                                    the team's own source note (Apache-2.0),
                                    versioned in git
  authoring/sources/shared-task-patterns-2.md
                                    its part 2 (2026-09-29, PI review list 5
                                    D2): note 1's anomaly-detection paragraph
                                    word for word, with one sentence added on
                                    machine condition monitoring - a document
                                    of its own, so note 1 stays as catalogued

The drafts locate the guide's passages by span (offsets + passage hash) and
the note's by anchor; an anchor on the guide is refused by the draft road
(NFR6). The drafts' author must be a registered identity (identities.yaml).

Idempotency (AR-rag4ds-026 F8, AR-rag4ds-027 F1): the catalog refuses a
hash already registered (a catalogued source is reused), and the draft
road skips items staging already holds unchanged - same source, span,
content and author, whatever the run - recording them in the manifest, and
files only the new ones; a run with nothing new files nothing and exits
non-zero. The seed prints the manifests it wrote and what it skipped.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from pathlib import Path

from rag4ds.auditlog.journal import Journal
from rag4ds.author.drafts import DraftError, SourceDoc, file_draft
from rag4ds.corpus.catalog import Catalog, CatalogError, ingest
from rag4ds.gate.identities import IDENTITIES_FILE, IdentityError, IdentityRegistry
from rag4ds.model.entry import SHARED_PACK
from rag4ds.staging.store import StagingStore

MODULE_ROOT = Path(__file__).resolve().parents[1]

CRISPDM_PDF_URL = ("https://www.kde.cs.uni-kassel.de/wp-content/uploads/lehre/ws2012-13/kdd/"
                   "files/CRISPWP-0800.pdf")
CRISPDM_PDF_SHA256 = "1dd1f0b01e74777d265b3f59ecd6f697a7799df5fc13528bba4da6d661ec5804"
CRISPDM_TEXT_SHA256 = "aee4b0cba8797a19e37acc76a82abb4fbfa26195b7960592b0c13778bab346a5"
CRISPDM_TEXT = MODULE_ROOT / "data" / "shared" / "crisp-dm-1.0.txt"
NOTE = MODULE_ROOT / "authoring" / "sources" / "shared-task-patterns.md"
NOTE2 = MODULE_ROOT / "authoring" / "sources" / "shared-task-patterns-2.md"
# the drafts this seed files; with no argument, the two of S8
DRAFTS = ("shared_general.yaml", "shared_tasks.yaml", "shared_tasks_2.yaml")
S8_DRAFTS = DRAFTS[:2]

SOURCES = {
    "crispdm": {
        "path": CRISPDM_TEXT,
        "license": ("proprietary - copyright 1999, 2000 the CRISP-DM consortium (NCR Systems "
                    "Engineering Copenhagen, DaimlerChrysler AG, SPSS Inc., OHRA); publicly "
                    "distributed, NOT redistributable; raw text kept outside git (NFR6)"),
        "fetch_pointer": (f"{CRISPDM_PDF_URL} (PDF sha256 {CRISPDM_PDF_SHA256}; text extracted "
                          f"with pypdf 6.16.1: PdfReader pages joined by two newlines)"),
        "redistributable": False,
        "metadata": {"title": "CRISP-DM 1.0 Step-by-step data mining guide",
                     "authors": "Chapman, Clinton, Kerber, Khabaza, Reinartz, Shearer, Wirth",
                     "year": "2000", "pages": 78, "pdf_sha256": CRISPDM_PDF_SHA256},
    },
    "note": {
        "path": NOTE,
        "license": "Apache-2.0 (the module's own licence; the team's original text)",
        "fetch_pointer": "authoring/sources/shared-task-patterns.md (this repository)",
        "redistributable": True,
        "metadata": {"title": "Shared-layer source note: task patterns and core vocabulary",
                     "authors": "Module 1 team (T. A. Siriweera)",
                     "year": "2026"},
    },
    "note2": {
        "path": NOTE2,
        "license": "Apache-2.0 (the module's own licence; the team's original text)",
        "fetch_pointer": "authoring/sources/shared-task-patterns-2.md (this repository)",
        "redistributable": True,
        "metadata": {"title": ("Shared-layer source note, part 2: anomaly detection in machine "
                               "condition monitoring"),
                     "authors": ("Module 1 team (T. A. Siriweera; the added "
                                 "sentence by the team)"),
                     "year": "2026"},
    },
}


def jst_now() -> str:
    if time.localtime().tm_gmtoff != 9 * 3600:
        raise SystemExit("s8_seed: this machine's local clock is not JST (UTC+9); refusing")
    return time.strftime("%Y-%m-%d %H:%M:%S JST")  # seconds: two runs in a minute stay apart


def head_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=MODULE_ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()


def drafts_named(argv: list[str]) -> list[str]:
    """The drafts a run files: the two of S8 when none is named; otherwise the
    names given, each one of DRAFTS, each once."""
    names = list(argv) or list(S8_DRAFTS)
    unknown = [n for n in names if n not in DRAFTS]
    if unknown or len(set(names)) != len(names):
        raise ValueError(f"drafts must be distinct names from {list(DRAFTS)}; got {names}")
    return names


def main(argv: list[str] | None = None) -> int:
    try:
        names = drafts_named(sys.argv[1:] if argv is None else argv)
    except ValueError as exc:
        print(f"s8_seed: {exc}", file=sys.stderr)
        return 1
    if not CRISPDM_TEXT.is_file():
        print(f"s8_seed: {CRISPDM_TEXT} is absent. Fetch the PDF from {CRISPDM_PDF_URL} "
              f"(sha256 {CRISPDM_PDF_SHA256}), extract its text with pypdf 6.16.1 "
              f"(PdfReader; pages joined by two newlines; UTF-8, LF), and place it there. "
              f"Expected text sha256: {CRISPDM_TEXT_SHA256}", file=sys.stderr)
        return 1
    got = hashlib.sha256(CRISPDM_TEXT.read_bytes()).hexdigest()
    if got != CRISPDM_TEXT_SHA256:
        print(f"s8_seed: {CRISPDM_TEXT} hashes to {got}, expected {CRISPDM_TEXT_SHA256}; "
              f"refusing to catalogue a document that is not the recorded one", file=sys.stderr)
        return 1
    try:
        identities = IdentityRegistry.load(MODULE_ROOT / IDENTITIES_FILE)
    except IdentityError as exc:
        print(f"s8_seed: {exc}", file=sys.stderr)
        return 1
    commit, ts = head_commit(), jst_now()
    catalog = Catalog("shared", MODULE_ROOT / "shared")
    ingest_runs = Journal(MODULE_ROOT / "runs" / "ingest.jsonl")
    texts: dict[str, SourceDoc] = {}
    already = {d["hash"]: d for d in catalog.load()["documents"]}
    for key, src in SOURCES.items():
        h = hashlib.sha256(src["path"].read_bytes()).hexdigest()
        if h in already:
            # a previous seed run catalogued it (documents are immutable and never
            # re-registered); reuse it, with the catalog's own redistributable flag
            texts[key] = SourceDoc(h, src["path"].read_text(encoding="utf-8"),
                                   bool(already[h]["redistributable"]), SHARED_PACK)
            print(f"{key}: already catalogued as {h[:16]}..., reused")
            continue
        try:
            m = ingest(catalog, src["path"], MODULE_ROOT, src["license"], src["fetch_pointer"],
                       src["redistributable"], src["metadata"], ts, ingest_runs, commit,
                       {"seed": "tools/s8_seed.py", "source_key": key})
        except CatalogError as exc:
            print(f"s8_seed: ingest of {key} refused: {exc}", file=sys.stderr)
            return 1
        texts[key] = SourceDoc(m["hash"], src["path"].read_text(encoding="utf-8"),
                               src["redistributable"], SHARED_PACK)
        print(f"ingested {key}: hash {m['hash'][:16]}... local_path {m['local_path']}")
    store = StagingStore(MODULE_ROOT / "stage" / "candidates.jsonl")
    authoring_runs = Journal(MODULE_ROOT / "runs" / "authoring.jsonl")
    for draft in names:
        try:
            m = file_draft(MODULE_ROOT / "authoring" / draft, texts, store, authoring_runs,
                           commit, ts, identities, allow_partial=True)
        except DraftError as exc:
            print(f"s8_seed: {draft} refused: {exc}", file=sys.stderr)
            return 1
        print(f"filed {draft}: {m['count']} candidates, run {m['run_id']}, author {m['author']}"
              + (f"; skipped, already in staging unchanged: {', '.join(m['skipped_already_filed'])}"
                 if m["skipped_already_filed"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
