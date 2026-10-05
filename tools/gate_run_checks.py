"""gate_run_checks.py - the checks a gate-session runner makes around a live
session, as functions a test can drive (AR-rag4ds-044 r1 F3 and O4).

They were embedded in tools/s9d_mapping_gate.ps1 as here-strings, where
nothing could witness them: the auditor filed a probe candidate the
recommendation sheet never names and the pre-flight passed it, still
printing that the sheet was unchanged. The sheet was bound by hash and
the candidate by nothing, so the pair held only because two script
defaults happened to agree.

Two entry points, both fail-loud and neither writing anything:

- `preflight` - before the gate opens: the candidate is the only one
  queued for the pack and is PENDING; the approver holds the pack's
  validator role and is not the draft's author; the sheet the validator
  reviewed still hashes to what he saw AND NAMES THIS CANDIDATE; the
  transcript name is free.
- `verify` - after the gate closes: exactly one decision cites the
  candidate, it names the approver, and an approval embeds an artifact.
  It writes the outcome file the rest of the run reads.

    python tools/gate_run_checks.py preflight
    python tools/gate_run_checks.py verify

Both read their arguments from the environment (S9D_*), so a PowerShell
wrapper passes them without quoting gymnastics.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))

from rag4ds.gate.identities import IDENTITIES_FILE, IdentityError, IdentityRegistry  # noqa: E402
from rag4ds.staging.store import StagingStore  # noqa: E402

DECISIONS = Path("journal") / "decisions.jsonl"
STAGING = Path("stage") / "candidates.jsonl"


def preflight(root: Path, candidate: str, pack: str, approver: str, sheet: Path,
              sheet_sha256: str, transcript: Path,
              skip_gate: bool = False) -> tuple[list[str], list[str]]:
    """(problems, confirmations). The second list is what this call actually
    established - the caller prints THAT and nothing else, so a success line
    can never claim a check the run skipped (AR-rag4ds-044 r2 F4: the
    transcript line was printed unconditionally while its own check was
    gated on not skip_gate)."""
    root = Path(root)
    problems: list[str] = []
    confirmed: list[str] = []

    sheet = root / sheet if not Path(sheet).is_absolute() else Path(sheet)
    sheet_text = ""
    if not sheet.is_file():
        problems.append(f"the reviewed sheet {sheet} is absent")
    else:
        raw = sheet.read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        if got != sheet_sha256:
            problems.append(f"the sheet has changed since the validator reviewed it: "
                            f"{got} != {sheet_sha256}")
        sheet_text = raw.decode("utf-8", "replace")
        # AR-rag4ds-044 r1 F3: the sheet is bound by hash, so bind the
        # candidate to the sheet too - otherwise a candidate the validator
        # never saw passes under a sheet that is genuinely unchanged. The
        # candidate id is content-addressed, so its presence in the text he
        # reviewed is what ties his agreement to THIS draft.
        if candidate not in sheet_text:
            problems.append(f"the sheet does not name {candidate}; the validator reviewed a "
                            f"different candidate, so his agreement does not reach this one")
        elif got == sheet_sha256:
            confirmed.append(f"the sheet is unchanged ({sheet_sha256[:12]}...) and names "
                             f"this candidate")

    transcript = root / transcript if not Path(transcript).is_absolute() else Path(transcript)
    if skip_gate:
        confirmed.append("no transcript is needed: the gate is not opened (skip-gate)")
    elif transcript.exists():
        problems.append(f"{transcript} already exists; name a new transcript")
    else:
        confirmed.append(f"transcript {transcript} is free")

    try:
        ident = IdentityRegistry.load(root / IDENTITIES_FILE).resolve(approver)
    except (IdentityError, OSError) as exc:
        problems.append(f"approver refused: {exc}")
    else:
        if "validator" not in ident.roles or not ident.validates(pack):
            problems.append(f"{approver} does not validate pack {pack}: roles {list(ident.roles)}")
        else:
            confirmed.append(f"approver {approver} holds the validator role for {pack}")

    store = StagingStore(root / STAGING)
    cands = store.candidates()
    if candidate not in cands:
        problems.append(f"{candidate} is not in staging")
    else:
        c = cands[candidate]["candidate"]
        if c.get("kind") != "mapping-draft" or c.get("pack") != pack:
            problems.append(f"{candidate} is not a {pack} mapping draft: "
                            f"{c.get('kind')}, {c.get('pack')}")
        if c.get("author") == approver:
            problems.append(f"{candidate} names {approver} as its author; the author may not "
                            f"approve it")
        else:
            confirmed.append(f"the draft's author is {c.get('author')!r}, not the approver")
        state = store.state_of(candidate)
        confirmed.append(f"candidate {candidate} is {state}")
        if skip_gate and state == "PENDING":
            problems.append(f"skip-gate given but {candidate} is still PENDING; nothing decided")
        if not skip_gate and state != "PENDING":
            problems.append(f"{candidate} is {state}, not PENDING; nothing to decide")

    if not skip_gate:
        pending = sorted(k for k in cands if store.state_of(k) == "PENDING"
                         and cands[k]["candidate"].get("pack") == pack)
        if pending != [candidate]:
            problems.append(f"the {pack} queue is {pending}, not exactly [{candidate}] - this "
                            f"runner drives one known decision, not a queue")
        else:
            confirmed.append(f"it is the only candidate queued for {pack}")
    return problems, confirmed


def verify_decision(root: Path, candidate: str, approver: str) -> dict:
    """What the validator actually decided. Raises ValueError with the
    reason when the session decided nothing, when the decision is not his,
    or when an approval carries no artifact."""
    root = Path(root)
    state = StagingStore(root / STAGING).state_of(candidate)
    if state == "PENDING":
        raise ValueError(f"the session ended without a decision on {candidate}; nothing folded")
    records = [json.loads(line) for line
               in (root / DECISIONS).read_text(encoding="utf-8").splitlines() if line.strip()]
    mine = [r for r in records if r.get("candidate_id") == candidate]
    if len(mine) != 1:
        raise ValueError(f"{len(mine)} decisions cite {candidate}; expected exactly one")
    rec = mine[0]
    if rec.get("approver") != approver:
        raise ValueError(f"the decision names {rec.get('approver')!r}, not {approver!r}")
    if not state.startswith("REJECTED"):
        art = rec.get("mapping_artifact")
        if not isinstance(art, dict) or "rows" not in art:
            raise ValueError("the approval embeds no mapping artifact")
    return {"state": state, "record": rec}


def _env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"gate_run_checks: {name} is not set")
    return v


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0] not in ("preflight", "verify"):
        print("usage: gate_run_checks.py preflight|verify", file=sys.stderr)
        return 2
    root = Path(os.environ.get("S9D_ROOT") or MODULE_ROOT)
    candidate, approver = _env("S9D_CANDIDATE"), _env("S9D_APPROVER")

    if argv[0] == "preflight":
        problems, confirmed = preflight(
            root, candidate, _env("S9D_PACK"), approver, Path(_env("S9D_SHEET")),
            _env("S9D_SHEET_SHA"), Path(_env("S9D_TRANSCRIPT")),
            bool(os.environ.get("S9D_SKIPGATE")))
        if problems:
            for p in problems:
                print(f"  REFUSED: {p}", file=sys.stderr)
            return 1
        # every line below was established by the call above, and nothing else
        # is printed: a success line can no longer outrun its check (r2 F4)
        for line in confirmed:
            print(f"  {line}")
        print(f"  code commit {os.environ.get('S9D_COMMIT', '?')}")
        return 0

    try:
        out = verify_decision(root, candidate, approver)
    except ValueError as exc:
        print(f"  REFUSED: {exc}", file=sys.stderr)
        return 1
    rec, state = out["record"], out["state"]
    print(f"  {candidate} is now {state}")
    print(f"  decision: {rec.get('type')} by {rec['approver']} at {rec['timestamp']}")
    if state.startswith("REJECTED"):
        print(f"  he REJECTED it. Reason as recorded: {rec.get('reason')!r}")
    else:
        art = rec["mapping_artifact"]
        print(f"  artifact: standard {art['standard']}, {len(art['rows'])} row(s) - "
              + "; ".join(f"{r['entry']} {r['relation']} {r['class']}" for r in art["rows"]))
    outcome = root / os.environ.get("S9D_OUTCOME", "data/transcripts/gate-run-outcome.json")
    outcome.parent.mkdir(parents=True, exist_ok=True)
    outcome.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
