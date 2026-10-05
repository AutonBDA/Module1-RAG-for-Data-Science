"""Re-file rejected candidates as fresh PENDING candidates (S9c part 3).

Why this exists. AR-rag4ds-032 r1 F2: the sheet that drove the robotics
pack's session 1 was built from the recommendation table the
validator reviewed, and it resolved 69 rows the table had not - 44 rows
referred to the validator ("JUDGE") and 25 groupings - as rejections he
signed. The PI chose option (b) (audit/rag4ds/responses/
PI_decision_44-referred-rows_2026-09-14.md): put them to him properly.

Why a re-file and not an un-reject. Staging candidates are immutable and a
rejection is final in the 6.3 state machine (rag4ds.staging.store: no
transition leaves REJECTED). So each candidate is filed AGAIN as a fresh
PENDING candidate with the same content, span, source and extractor and a
`run_manifest` that points at the re-file manifest this tool appends to
runs/refile.jsonl - that reference is what gives the fresh candidate its
new id (AR-rag4ds-033 r1 O2 i; no schema change), and the manifest names
the rejected candidate each fresh one re-files, the decision record that
licenses it and the reason. The rejected candidates stay in staging
history as what they were; the journal's session-1 decisions stand.

Rules (each refused loudly, nothing written): an id that is not filed; a
candidate whose state is not REJECTED; a candidate of another pack; an id
listed twice; an id already re-filed by an earlier manifest; a decision
record that does not exist; a run whose re-file id is already on the
journal (the same set under the same decision at the same commit is the
same run); a machine clock that is not JST unless a clock is injected.
Two rejected candidates that differ only in their run_manifest would
collapse to one fresh candidate: refused whole, naming both (AR-rag4ds-034
F2). Candidates are filed before the manifest (AR-rag4ds-026 F8); a failure
between the two leaves fresh PENDING candidates whose run_manifest names a
manifest not yet on the journal - the operator re-runs the same command
(same ids file and decision; the commit is NOT part of the run's identity,
so a commit in between does not matter - AR-rag4ds-035 F1): the retry
counts those as already filed by this run, files the rest and appends the
manifest naming every pair (AR-rag4ds-034 F6). While such stranded
candidates exist, any OTHER re-file is refused, naming them. --dry-run
files nothing and prints what a real run would file.

    python tools/refile_candidates.py --pack robotics --ids authoring/refile_ids.txt
        --decision audit/rag4ds/responses/PI_decision_44-referred-rows_2026-09-14.md
        --reason "..." [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE_ROOT = HERE.parent
sys.path.insert(0, str(MODULE_ROOT / "src"))

from rag4ds.auditlog.journal import Journal  # noqa: E402
from rag4ds.staging.store import StagingError, StagingStore, candidate_id  # noqa: E402

REFILE_JOURNAL = "runs/refile.jsonl"


class RefileError(RuntimeError):
    """A refusal before anything is written - printed and exit 1."""


def jst_now() -> str:
    if time.localtime().tm_gmtoff != 9 * 3600:
        raise RefileError("this machine's local clock is not JST (UTC+9); refusing")
    return time.strftime("%Y-%m-%d %H:%M:%S JST")


def head_commit(module_root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=module_root,
                          capture_output=True, text=True, check=True).stdout.strip()


def load_ids(path: Path) -> list[str]:
    """One candidate id per line; blank lines and '#' comments ignored; an
    id listed twice is refused."""
    ids: list[str] = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if not line.startswith("cand-"):
            raise RefileError(f"{path}: line {n}: {line!r} is not a candidate id")
        if line in ids:
            raise RefileError(f"{path}: line {n}: {line} is listed twice")
        ids.append(line)
    if not ids:
        raise RefileError(f"{path}: no candidate ids")
    return ids


def refile_id_for(ids: list[str], decision: str) -> str:
    """The run's identity: the ids and the decision - NOT the commit, so a
    retry after a commit is the same run (AR-rag4ds-035 F1); the commit is
    recorded in the manifest."""
    core = "\n".join([*sorted(ids), decision])
    return hashlib.sha256(core.encode("utf-8")).hexdigest()[:16]


def stranded(store: StagingStore, runs: Journal) -> dict[str, list[str]]:
    """re-file ref -> fresh candidate ids filed under it while no manifest on
    the journal carries that ref: the trace a run that failed between its
    candidates and its manifest leaves (AR-rag4ds-035 F1)."""
    on_journal = {m.get("run_manifest_ref") for m in runs.read_all() if m.get("kind") == "refile"}
    out: dict[str, list[str]] = {}
    for cid, rec in store.candidates().items():
        ref = rec["candidate"].get("run_manifest", "")
        if ref.startswith(f"{REFILE_JOURNAL}#") and ref not in on_journal:
            out.setdefault(ref, []).append(cid)
    return out


def already_refiled(runs: Journal) -> dict[str, str]:
    """rejected id -> fresh id, from every re-file manifest on the journal."""
    done: dict[str, str] = {}
    for m in runs.read_all():
        if m.get("kind") == "refile":
            for pair in m.get("pairs", []):
                done[pair["from"]] = pair["to"]
    return done


def plan(module_root: Path, pack: str, ids: list[str], decision: str, reason: str,
         commit: str) -> tuple[list[tuple[str, dict]], str, str]:
    """Validate every rule and return (pairs of (old id, fresh candidate),
    refile id, manifest ref) - nothing written."""
    if not reason.strip():
        raise RefileError("a reason is required (--reason)")
    repo_root = module_root.parent.parent
    if not (repo_root / decision).is_file():
        raise RefileError(f"decision record {decision!r} does not exist under {repo_root}")
    store = StagingStore(module_root / "stage" / "candidates.jsonl")
    runs = Journal(module_root / REFILE_JOURNAL)
    refile_id = refile_id_for(ids, decision)
    if any(m.get("kind") == "refile" and m.get("refile_id") == refile_id
           for m in runs.read_all()):
        raise RefileError(f"re-file {refile_id} is already on {REFILE_JOURNAL} (the same ids "
                          f"under the same decision); nothing to do")
    ref = f"{REFILE_JOURNAL}#{refile_id}"
    # AR-rag4ds-035 F1: a run that failed between its candidates and its manifest
    # leaves fresh candidates naming a manifest that does not exist; only the
    # SAME run (same ids, same decision) may proceed - it completes them; any
    # other list is refused, naming the stranded candidates
    for other_ref, cids in stranded(store, runs).items():
        if other_ref != ref:
            raise RefileError(f"staging holds {len(cids)} fresh candidate(s) whose run_manifest "
                              f"{other_ref} is on no manifest ({', '.join(cids)}): a re-file "
                              f"failed between its candidates and its manifest; re-run THAT "
                              f"re-file with its original ids file and decision to complete "
                              f"it before any other re-file")
    filed = store.candidates()
    done = already_refiled(runs)
    pairs: list[tuple[str, dict]] = []
    fresh_of: dict[str, str] = {}  # fresh id -> the rejected id it comes from (this batch)
    for cid in ids:
        rec = filed.get(cid)
        if rec is None:
            raise RefileError(f"{cid} is not a filed candidate")
        if cid in done:
            raise RefileError(f"{cid} was already re-filed as {done[cid]}")
        state = store.state_of(cid)
        if state != "REJECTED":
            raise RefileError(f"{cid} is {state}, not REJECTED; only a rejected candidate is "
                              f"re-filed")
        old = rec["candidate"]
        if old.get("pack") != pack:
            raise RefileError(f"{cid} belongs to pack {old.get('pack')!r}, not {pack!r}")
        fresh = dict(old, run_manifest=ref)
        fid = candidate_id(fresh)
        if fid in fresh_of:
            # AR-rag4ds-034 F2: two rejected candidates that differ only in their
            # run_manifest collapse to ONE fresh candidate; refused whole, before
            # anything is written, naming both
            raise RefileError(f"{cid} and {fresh_of[fid]} would re-file to the same fresh "
                              f"candidate {fid} (same content, span, source and extractor); "
                              f"list one of them")
        fresh_of[fid] = cid
        if fid in filed:
            # AR-rag4ds-034 F6: the fresh id hashes run_manifest, so a filed twin can
            # only be this run's own earlier attempt, which failed before the
            # manifest; the retry counts it as filed and completes the run
            continue
        pairs.append((cid, fresh))
    return pairs, refile_id, ref


def run(module_root: Path, pack: str, ids_path: Path, decision: str, reason: str,
        dry_run: bool, clock=None, out=None) -> int:
    """clock: a callable returning the NFR2 timestamp; the machine's JST
    clock by default (refused when not JST), injectable for the suite."""
    out = out or sys.stdout
    module_root = Path(module_root)
    try:
        ts = (clock or jst_now)()
        commit = head_commit(module_root)
        ids = load_ids(Path(ids_path))
        pairs, refile_id, ref = plan(module_root, pack, ids, decision, reason, commit)
        mode = "DRY RUN - nothing filed" if dry_run else "filing"
        print(f"refile_candidates: pack {pack!r} at {commit}, {ts}; {len(pairs)} rejected "
              f"candidate(s) under {decision}; {mode}", file=out)
        for old, fresh in pairs:
            print(f"  {old} -> {candidate_id(fresh)}", file=out)
        if dry_run:
            print(f"refile_candidates: would file {len(pairs)} and manifest {ref} (dry run)",
                  file=out)
            return 0
        store = StagingStore(module_root / "stage" / "candidates.jsonl")
        filed = []
        for old, fresh in pairs:
            filed.append({"from": old, "to": store.file_candidate(fresh, ts)})
        # the manifest names EVERY pair of the run, including those an earlier
        # attempt of the same run filed before failing (AR-rag4ds-034 F6); the
        # journal is read once (r2 O6) and what was written this run is a subset
        # of what the manifest records (r2 O7)
        cands = store.candidates()
        all_pairs = [{"from": cid, "to": candidate_id(dict(cands[cid]["candidate"],
                                                           run_manifest=ref))}
                     for cid in ids]
        assert {q["to"] for q in filed} <= {q["to"] for q in all_pairs}
        assert all(q["to"] in cands for q in all_pairs)
        Journal(module_root / REFILE_JOURNAL).append({
            "kind": "refile", "schema": 1, "refile_id": refile_id, "run_manifest_ref": ref,
            "timestamp": ts, "code_commit": commit, "pack": pack, "decision": decision,
            "reason": reason, "count": len(all_pairs), "pairs": all_pairs,
        })
        resumed = len(all_pairs) - len(filed)
        print(f"refile_candidates: filed {len(filed)}"
              + (f" ({resumed} already filed by an earlier attempt of this run)" if resumed else "")
              + f"; manifest {ref}", file=out)
        return 0
    except (RefileError, StagingError, OSError, subprocess.CalledProcessError) as exc:
        print(f"refile_candidates: REFUSED - {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ids", required=True, type=Path,
                    help="file: one rejected candidate id per line")
    ap.add_argument("--decision", required=True,
                    help="the PI decision record that licenses the re-file, repo-relative")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--module-root", type=Path, default=MODULE_ROOT)
    a = ap.parse_args(argv)
    return run(a.module_root, a.pack, a.ids, a.decision, a.reason, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
