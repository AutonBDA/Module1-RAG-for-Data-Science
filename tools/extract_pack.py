"""Pack extraction run (plan S9c, component 2, FR5, NFR2): the S7 extraction
strategies over every catalogued TEXT document of a pack, the surviving
proposals filed into staging as entry candidates for the pack's validator,
one NFR2 extraction manifest per document. Generic - knows no domain; the
pack's data (catalog, sources, lexicon) is the only input. Run from the
module root with the module venv:

    python tools/extract_pack.py <pack code> [--dry-run] [--strategies rule-v1,embed-v1]
                                             [--no-definitions] [--no-headings] [--no-tables]
                                             [--top-k 3] [--floor 0.4] [--tuning]
                                             [--no-span-check]

S9c part 6 (the PI's option C of 2026-09-17): before filing, every proposal
whose source span (document hash, start, end) already carries a TERMINAL
decision in staging - APPROVED, REJECTED or REJECTED-DUPLICATE - is
withheld, whatever its candidate id (a tuned run's ids differ because the
run manifest is inside the id, so the id check alone cannot see a decided
span). S9e part 4 (the PI's YES of 2026-09-21, reversing one clause of
option C after AR-rag4ds-045 20.4): a span staging still holds PENDING or
GROUPED is withheld too - it is already before the validator, and a re-run
whose manifest differs would otherwise file it again under a new id. The
manifest records what was withheld, with the held candidate's id and state,
outside its core, so the extraction id is the same whatever staging holds.
--no-span-check files them anyway, for measurement, and the manifest then
records what it would have withheld.

Road (FR5: extraction proposes, the gate decides; nothing here is
retrievable): for each document in packs/<code>/corpus/catalog.yaml whose
local copy (via the ingest manifest in runs/ingest.jsonl, under the
module's data/) is a UTF-8 text file hashing to the catalogued hash, build
the extractor's Document (pack, layer "domain"), run the requested
strategies - rule-v1 (definitions, heading/section heuristics, tables) and
embed-v1 (sentences nearest the pack's own lexicon terms, the seeds, under
the pinned embedding model) - through run_extraction (the quotation bound
suppresses any proposal copying more than the bound), then file the
candidates (candidates before the manifest, AR-rag4ds-026 F8; a candidate
already in staging is skipped and counted, never re-filed) and append the
manifest to runs/extraction.jsonl. A binary document (an archive) is
skipped with its reason. --dry-run runs everything and files nothing.

No cap is applied to the volume: population is incremental and the gate is
the rate limiter by design (plan 4.3); the validator decides as many as a
session allows and the rest stay PENDING. The counts per document and
strategy are printed so the team can choose the strategy configuration
for a corpus before filing (and record it).
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))
sys.path.insert(0, str(MODULE_ROOT / "tools"))
from rag4ds.auditlog.journal import Journal  # noqa: E402
from rag4ds.corpus.catalog import Catalog, CatalogError  # noqa: E402
from rag4ds.extract.strategies import EmbeddingStrategy, RuleStrategy  # noqa: E402
from rag4ds.extract.strategy import Document, ExtractError, run_extraction  # noqa: E402
from rag4ds.gate.cli import make_source_resolver  # noqa: E402
from rag4ds.model.entry import ENTRY_ID_RE, SHARED_PACK  # noqa: E402
from rag4ds.model.pack import PackManifestError, load_pack_manifest  # noqa: E402
from rag4ds.staging.store import StagingError, StagingStore, candidate_id, TERMINAL  # noqa: E402

KNOWN_STRATEGIES = ("rule-v1", "embed-v1")


class ExtractPackError(RuntimeError):
    """A refusal raised before any filing - printed and exit 1. A failure
    raised by the staging store, the extractor or the journal on document N
    of a run leaves documents 1..N-1 fully filed (candidates and manifest)
    and document N with candidates but no manifest - the AR-rag4ds-026 F8
    direction; a re-run files the rest (AR-rag4ds-030 O4)."""


def jst_now() -> str:
    if time.localtime().tm_gmtoff != 9 * 3600:
        raise ExtractPackError("this machine's local clock is not JST (UTC+9); refusing")
    return time.strftime("%Y-%m-%d %H:%M:%S JST")


def head_commit(module_root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=module_root,
                          capture_output=True, text=True, check=True).stdout.strip()


def load_documents(pack: str, module_root: Path) -> tuple[list[tuple[dict, Path, str]], list[str]]:
    """(documents, skipped): each document is (catalog record, local path,
    text); a catalogued document whose local copy is absent is a loud
    failure; a binary one is skipped with a reason."""
    catalog = Catalog(pack, module_root / "packs" / pack)
    records = catalog.load()["documents"]
    if not records:
        raise ExtractPackError(f"pack {pack!r} has no catalogued document")
    resolve = make_source_resolver(Journal(module_root / "runs" / "ingest.jsonl"), module_root)
    docs, skipped = [], []
    for rec in records:
        title = rec["metadata"].get("title", rec["hash"][:16])
        path = resolve(rec["hash"])
        if path is None:
            raise ExtractPackError(f"{title!r}: local copy of {rec['hash'][:16]}... is absent "
                                   f"(the ingest manifest names it under data/)")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != rec["hash"]:
            raise ExtractPackError(f"{path}: does not hash to the catalogued {rec['hash'][:16]}...")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped.append(f"{title!r} ({path.name}): not a UTF-8 text document (an archive or "
                           f"binary) - data for demo flows, not text entries restate")
            continue
        if not text.strip():
            skipped.append(f"{title!r} ({path.name}): empty text")
            continue
        docs.append((rec, path, text))
    return docs, skipped


def held_spans(store: StagingStore, pack: str, terminal_only: bool = False) -> dict:
    """(source_hash, span_start, span_end) -> (candidate id, state) for every
    entry candidate OF THIS PACK that staging holds - in any state (S9e part
    4, the PI's YES of 2026-09-21: a pending span is already before the
    validator, and filing it again under a new id puts the same text in his
    queue twice), or only in a terminal one when terminal_only. One pass over
    the journal; when two candidates share a span the first filed is the one
    cited. Scoped to the pack (AR-rag4ds-041 r1 O1): a document that serves
    two packs is decided by each pack's validator separately, so one pack's
    candidate never withholds the other pack's proposal."""
    states = store.states()
    cands = store.candidates()
    out = {}
    for cid, (state, _head) in states.items():
        if terminal_only and state not in TERMINAL:
            continue
        c = cands[cid]["candidate"]
        if c.get("kind") != "entry" or c.get("pack") != pack:
            continue
        key = (c["source_hash"], c["span_start"], c["span_end"])
        cur = out.get(key)
        # AR-rag4ds-046 r1 F2: the citation is truthful before it is first-filed
        # - a span that carries a terminal decision is cited by that decision,
        # whichever candidate came first; among candidates of the same class
        # the first filed is cited
        if cur is None or (cur[1] not in TERMINAL and state in TERMINAL):
            out[key] = (cid, state)
    return out


def decided_spans(store: StagingStore, pack: str) -> dict:
    """The terminal subset of held_spans(): the S9c part 6 guard as the PI's
    decision C first set it; kept for the readers that ask about decisions."""
    return held_spans(store, pack, terminal_only=True)


def run(pack: str, module_root: Path, strategy_names: list[str], rule_flags: dict,
        top_k: int, floor: float, dry_run: bool, out=None, clock=None,
        span_check: bool = True) -> int:
    """clock: a callable returning the NFR2 timestamp; the machine's JST
    clock by default (refused when the clock is not JST), injectable so the
    suite runs on a UTC runner (the CI lesson of 911241b)."""
    out = out or sys.stdout  # resolved at call time (pytest's capture replaces sys.stdout)
    module_root = Path(module_root)
    if not ENTRY_ID_RE.fullmatch(pack or "") or pack == SHARED_PACK:
        print(f"extract_pack: {pack!r} is not a pack code", file=sys.stderr)
        return 2
    try:
        unknown = [n for n in strategy_names if n not in KNOWN_STRATEGIES]
        if unknown:
            raise ExtractPackError(f"unknown strategy {unknown[0]!r}; known: {KNOWN_STRATEGIES}")
        if not strategy_names:
            raise ExtractPackError("no strategy requested")
        manifest = load_pack_manifest(module_root / "packs" / pack)
        docs, skipped = load_documents(pack, module_root)
        commit, ts = head_commit(module_root), (clock or jst_now)()
        strategies, embedder = [], None
        for n in strategy_names:
            if n == "rule-v1":
                strategies.append(RuleStrategy(**rule_flags))
            elif n == "embed-v1":
                from rag4ds.index.build import (PINNED_EMBEDDING_MODEL,
                                                PINNED_EMBEDDING_REVISION,
                                                SentenceTransformerEmbedder)
                embedder = SentenceTransformerEmbedder(PINNED_EMBEDDING_MODEL,
                                                       PINNED_EMBEDDING_REVISION)
                # AR-rag4ds-045 F3: the tuning flag reaches this strategy too, so
                # back matter is out of scope for every strategy of a tuned run
                strategies.append(EmbeddingStrategy(embedder, list(manifest.lexicon),
                                                    top_k=top_k, floor=floor,
                                                    tuning=bool(rule_flags.get("tuning"))))
        store = StagingStore(module_root / "stage" / "candidates.jsonl")
        runs = Journal(module_root / "runs" / "extraction.jsonl")
        # FR7's document -> candidate link is enforced in load_documents: every
        # document comes from the catalog and its local copy is re-hashed against
        # the catalogued hash (AR-rag4ds-030 F1: a known_hashes argument built from
        # a second read of the same catalog could never refuse anything)
        already = set(store.candidates())
        # S9c part 6 withheld this pack's DECIDED spans; S9e part 4 (the PI's YES
        # of 2026-09-21) withholds every span staging holds, pending or decided
        decided = held_spans(store, pack)
        manifest_ids = {m.get("extraction_id") for m in runs.read_all()
                        if m.get("kind") == "extraction"}
        print(f"extract_pack: pack {pack!r} at {commit}, {ts}; strategies "
              f"{[s.name for s in strategies]}; rule flags {rule_flags}; "
              f"{'DRY RUN - nothing filed' if dry_run else 'filing'}", file=out)
        totals = {"proposed": 0, "suppressed": 0, "filed": 0, "would_file": 0, "already": 0,
                  "withheld": 0, "would_withhold": 0}
        for rec, path, text in docs:
            doc = Document(source_hash=rec["hash"], text=text, pack=pack, layer="domain",
                           source_reference=f"corpus:{pack}/{path.stem}")
            result = run_extraction(doc, strategies, commit, ts,
                                    config={"tool": "tools/extract_pack.py", "pack": pack,
                                            "document": path.stem, "rule_flags": rule_flags,
                                            "seeds": "packs/%s/manifest.yaml lexicon" % pack
                                            if embedder is not None else None},
                                    embedder=embedder)
            per = result.manifest["strategy_kept_counts"]
            filed = already_n = 0
            # AR-rag4ds-030 r2 F9: two proposals of ONE document may share a candidate
            # id (the tables capability, a repeated column); the second is "already
            # filed" once the first is, so the id set grows as the list is walked
            new, seen, withheld = [], set(already), []
            for c in result.candidates:
                cid = candidate_id(c)
                if cid in seen:
                    already_n += 1
                    continue
                seen.add(cid)
                # S9c part 6: a span staging has already decided is not put to the
                # validator again under a new id; recorded, not filed (option C).
                # S9e part 4: nor is a span staging still holds PENDING or GROUPED
                # - the "decision" recorded is then that state
                hit = decided.get((c["source_hash"], c["span_start"], c["span_end"]))
                if hit is not None:
                    withheld.append({"candidate_id": cid, "extractor": c["extractor"],
                                     "span_start": c["span_start"], "span_end": c["span_end"],
                                     "decided_id": hit[0], "decision": hit[1],
                                     "decided_pack": pack})
                    if span_check:
                        continue
                new.append(c)
                # AR-rag4ds-046 r1 F3: the run's own filing holds the span from
                # here on, so a second proposal of this run on the same span (a
                # rule-v1/embed-v1 pair, a tables twin) is withheld like any
                # other and cites the candidate this run files
                decided.setdefault((c["source_hash"], c["span_start"], c["span_end"]),
                                   (cid, "PENDING"))
            if withheld:
                # outside the manifest core: the extraction id does not depend on
                # what staging holds; the record of the withholding does (NFR9)
                result.manifest["span_check"] = {
                    "enabled": span_check, "count": len(withheld),
                    ("withheld" if span_check else "would_withhold"): withheld}
            manifest_new = result.manifest["extraction_id"] not in manifest_ids
            if not dry_run:
                for c in new:
                    store.file_candidate(c, ts)
                    already.add(candidate_id(c))
                    filed += 1
                # AR-rag4ds-030 r1 O3 / r2 F10: the manifest is appended after the
                # candidates (F8) whenever its extraction id is not yet on the
                # journal - so a first run over a document whose every proposal the
                # bound suppressed still records the suppression detail (NFR9), and a
                # no-op re-run never repeats an id
                if manifest_new:
                    runs.append(result.manifest)
                    manifest_ids.add(result.manifest["extraction_id"])
            print(f"  {path.stem}: {len(text)} chars; proposed {len(result.candidates)} "
                  f"({', '.join(f'{k} {v}' for k, v in per.items())}); suppressed by the bound "
                  f"{len(result.suppressed)}; near-duplicate clusters "
                  f"{sum(1 for c in result.clusters if len(c) > 1)}"
                  + (f"; would file {len(new)}, already filed {already_n}" if dry_run else
                     f"; filed {filed}, already filed {already_n}")
                  + (("; withheld %d (span already held)" if span_check else
                      "; would withhold %d (span already held; check off)") % len(withheld)
                     if withheld else "")
                  + ("; manifest " + ("new" if manifest_new else "already on the journal")
                     if dry_run else
                     "; manifest " + (result.manifest['run_manifest_ref'] if manifest_new
                                      else "already on the journal (not repeated)")), file=out)
            totals["proposed"] += len(result.candidates)
            totals["suppressed"] += len(result.suppressed)
            totals["filed"] += filed
            totals["would_file"] += len(new)
            totals["already"] += already_n
            totals["withheld" if span_check else "would_withhold"] += len(withheld)
        for s in skipped:
            print(f"  skipped {s}", file=out)
    except (ExtractPackError, CatalogError, ExtractError, StagingError, PackManifestError,
            FileNotFoundError) as exc:
        print(f"extract_pack: REFUSED - {exc}", file=sys.stderr)
        return 1
    print(f"extract_pack: {len(docs)} documents, {totals['proposed']} proposed, "
          f"{totals['suppressed']} suppressed, "
          + (f"would file {totals['would_file']}" if dry_run else f"{totals['filed']} filed")
          + f", {totals['already']} already filed"
          + (f", {totals['withheld']} withheld (span already held)" if totals["withheld"]
             else "")
          + (f", {totals['would_withhold']} would be withheld (span already held; check off)"
             if totals["would_withhold"] else "")
          + f", {len(skipped)} skipped"
          + (" (dry run)" if dry_run else ""), file=out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--strategies", default="rule-v1,embed-v1")
    ap.add_argument("--no-definitions", action="store_true")
    ap.add_argument("--no-headings", action="store_true")
    ap.add_argument("--no-tables", action="store_true")
    ap.add_argument("--tuning", action="store_true",
                    help="S9c part 5: the rule strategy's tuning (lead-stop, code-prefix, gloss "
                         "and whitespace rules), recorded in the manifest when on")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--floor", type=float, default=0.4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-span-check", action="store_true",
                    help="S9c part 6 / S9e part 4: file proposals on spans staging already holds "
                         "(for measurement; the manifest records what would have been withheld)")
    args = ap.parse_args(argv)
    names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    flags = {"definitions": not args.no_definitions, "headings": not args.no_headings,
             "tables": not args.no_tables}
    if args.tuning:
        flags["tuning"] = True  # only when on: the manifest config of an untuned run is unchanged
    return run(args.pack, MODULE_ROOT, names, flags, args.top_k, args.floor, args.dry_run,
               span_check=not args.no_span_check)


if __name__ == "__main__":
    sys.exit(main())
