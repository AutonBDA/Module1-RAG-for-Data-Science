"""Run a pack's demo flows (plan section 3, component 9; FR9) against a built
index: each flow's question goes through the retrieval library and is checked
mechanically against its expected outcome (src/rag4ds/evalhook/demo.py). Run
from the module root with the module venv:

    python tools/run_demo.py --pack <code> [--index-dir data/index] [--record]

Prints one line per flow - PASS, FAIL or UNBOUND, the entries returned, the
reasons, and the service's own error when it answers ERROR - and exits 0
only when every flow passes; a FAIL or an UNBOUND flow exits 1; a flows file
or an index that does not load, or a bound flow naming an entry id the index
does not hold, exits 2 (NFR9).
With --record the run's manifest is appended to runs/demo-flows.jsonl: the
code commit, the index version and the pack versions it served, the flows
file's sha256 and every verdict - the record the release checklist's
demo-flow check cites (plan 4.3). Without --record nothing is written.
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
from rag4ds.auditlog.journal import Journal  # noqa: E402
from rag4ds.evalhook.demo import (  # noqa: E402
    FLOWS_FILE, PASS, DemoFlowError, load_flows, run, unserved,
)
from rag4ds.index.build import (  # noqa: E402
    IndexBuildError, SentenceTransformerEmbedder, load_index,
)
from rag4ds.retrieve.service import RetrievalError, Retriever  # noqa: E402

RUN_RECORDS = Path("runs") / "demo-flows.jsonl"
# schema 2 (AR-rag4ds-057 r1 O2): each result carries the service's error reason;
# no schema-1 record was ever written
RECORD_SCHEMA = 2


class RunDemoError(RuntimeError):
    pass


def jst_now() -> str:
    if time.localtime().tm_gmtoff != 9 * 3600:
        raise RunDemoError("this machine's local clock is not JST (UTC+9); refusing")
    return time.strftime("%Y-%m-%d %H:%M:%S JST")


def head_commit(module_root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=module_root,
                          capture_output=True, text=True, check=True).stdout.strip()


def execute(module_root: Path, pack: str, index_dir: Path, embedder=None, record: bool = False,
            stream=None, clock=None) -> int:
    stream = stream or sys.stdout
    module_root = Path(module_root)
    pack_dir = module_root / "packs" / pack
    try:
        code, flows = load_flows(pack_dir)
        index = load_index(Path(index_dir))
        if pack not in index.manifest["pack_versions"]:
            raise RunDemoError(f"the index at {index_dir} does not serve pack {pack!r}")
        # AR-rag4ds-057 r1 O3: every id a bound flow names must be one this index holds
        missing = unserved(flows, {row["id"] for row in index.rows})
        if missing:
            raise RunDemoError(f"the flows name entry ids the index at {index_dir} does not hold "
                               f"- {'; '.join(missing)}")
        pin = index.manifest["embedding_model"]
        retriever = Retriever(index, embedder or SentenceTransformerEmbedder(pin["id"],
                                                                             pin["revision"]))
        results = run(retriever, flows)
    except (DemoFlowError, IndexBuildError, RetrievalError, RunDemoError) as exc:
        print(f"run_demo: REFUSED - {exc}", file=sys.stderr)
        return 2
    for r in results:
        got = ", ".join(f"{i} v{v}" for i, v, _ in r.returned) or "(none)"
        why = f" - {'; '.join(r.reasons)}" if r.reasons else ""
        err = f" - the service's error: {r.error}" if r.error else ""
        print(f"{r.verdict:8} {r.flow_id}: {r.status}; returned {got}{why}{err}", file=stream)
    passed = all(r.verdict == PASS for r in results)
    m = index.manifest
    print(f"run_demo: {sum(r.verdict == PASS for r in results)} of {len(results)} flows pass; "
          f"pack {pack} at {m['pack_versions'][pack]}, index {m['index_version']}", file=stream)
    if record:
        try:
            rec = {
                "schema": RECORD_SCHEMA, "kind": "demo-flows", "pack": code,
                "flows_file": (Path("packs") / pack / FLOWS_FILE).as_posix(),
                "flows_sha256": hashlib.sha256((pack_dir / FLOWS_FILE).read_bytes()).hexdigest(),
                "code_commit": head_commit(module_root), "timestamp": (clock or jst_now)(),
                "index_version": m["index_version"], "pack_versions": dict(m["pack_versions"]),
                "results": [{"flow": r.flow_id, "verdict": r.verdict, "status": r.status,
                             "returned": [list(x) for x in r.returned],
                             "reasons": list(r.reasons), "error": r.error} for r in results],
                "passed": passed,
            }
            Journal(module_root / RUN_RECORDS).append(rec)
        except (RunDemoError, OSError, subprocess.CalledProcessError) as exc:
            print(f"run_demo: the run record was NOT written - {exc}", file=sys.stderr)
            return 2
        print(f"run_demo: run record appended to {RUN_RECORDS.as_posix()}", file=stream)
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--index-dir", default=str(MODULE_ROOT / "data" / "index"))
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args(argv)
    return execute(MODULE_ROOT, args.pack, Path(args.index_dir), record=args.record)


if __name__ == "__main__":
    sys.exit(main())
