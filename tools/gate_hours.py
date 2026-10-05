"""Measured gate hours (plan S8 exit; 6.4's session summary "feeds the
section 10 estimates"): read runs/gate-sessions.jsonl and report, per
approver and per layer/pack filter, the sessions, wall-clock seconds,
decisions written and decisions per hour. Nothing is estimated and nothing
is defaulted (AR-rag4ds-026 F7): every figure is summed from the session
manifests the gate itself wrote, a row that is not a gate-session manifest
is refused, and a manifest missing or mistyping a summed field stops the
tool by name rather than contributing a zero. Run from the module root:

    python tools/gate_hours.py [runs/gate-sessions.jsonl]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


class GateHoursError(ValueError):
    """A manifest the tool will not sum - named, never defaulted."""


def _number(row: dict, key: str, where: str, kind) -> float | int:
    if key not in row:
        raise GateHoursError(f"{where}: missing {key!r}; a session manifest that does not "
                             f"record it cannot be summed")
    v = row[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (kind is int and v != int(v)):
        raise GateHoursError(f"{where}: {key!r} must be a {kind.__name__}, got {v!r}")
    if v < 0:
        raise GateHoursError(f"{where}: {key!r} is negative ({v!r})")
    return kind(v)


def summarise(path: Path) -> dict:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    out: dict = defaultdict(lambda: {"sessions": 0, "seconds": 0.0, "decisions": 0,
                                     "counts": defaultdict(int)})
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        where = f"{path}:{n}"
        try:
            r = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GateHoursError(f"{where}: not a JSON record: {exc}") from exc
        if not isinstance(r, dict) or r.get("kind") != "gate-session":
            kind = r.get("kind") if isinstance(r, dict) else None
            raise GateHoursError(f"{where}: kind {kind!r} is not 'gate-session'; only the gate's "
                                 f"own manifests are summed")
        approver = r.get("approver")
        if not isinstance(approver, str) or not approver.strip():
            raise GateHoursError(f"{where}: approver must be a non-blank string")
        filters = r.get("filters")
        if not isinstance(filters, dict):
            raise GateHoursError(f"{where}: filters must be a mapping")
        counts = r.get("counts")
        if not isinstance(counts, dict):
            raise GateHoursError(f"{where}: counts must be a mapping")
        seconds = _number(r, "elapsed_seconds", where, float)
        decisions = _number(r, "decisions_written", where, int)
        if decisions > 0 and seconds <= 0:
            # AR-rag4ds-026 O15: decisions with no time would inflate the rate
            raise GateHoursError(f"{where}: {decisions} decisions written in "
                                 f"{seconds} seconds - not a session the gate can write")
        k = out[(approver, json.dumps(filters, sort_keys=True))]
        k["sessions"] += 1
        k["seconds"] += seconds
        k["decisions"] += decisions
        for name, c in counts.items():
            if isinstance(c, bool) or not isinstance(c, int) or c < 0:
                raise GateHoursError(f"{where}: counts[{name!r}] must be a non-negative integer")
            k["counts"][name] += c
    result = {}
    for (approver, filters), k in out.items():
        hours = k["seconds"] / 3600.0
        result[f"{approver} {filters}"] = {
            "sessions": k["sessions"], "hours": round(hours, 4), "decisions": k["decisions"],
            "decisions_per_hour": round(k["decisions"] / hours, 2) if hours > 0 else None,
            "counts": dict(k["counts"]),
        }
    return result


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path("runs") / "gate-sessions.jsonl"
    if not path.is_file():
        print(f"no session manifests at {path}", file=sys.stderr)
        return 1
    try:
        result = summarise(path)
    except GateHoursError as exc:
        print(f"gate_hours: refused - {exc}", file=sys.stderr)
        return 1
    if not result:
        print(f"{path}: no sessions recorded", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
