"""The evaluation hook's demo-flow runner (plan section 3, component 9;
FR9; section 4.1's demo/).

A pack's demo flows are pack-authored queries with expected outcomes, kept
in packs/<pack>/demo/flows.yaml (human-edited YAML, plan section 8). Each
flow runs through the same retrieval library the reasoner uses, and is
checked mechanically - FR9 makes "passing" a check, not a judgment:

- the response status must equal the expected status;
- every expected entry id must be among the returned entries;
- no entry id the flow lists as absent may be returned.

A flow's expected outcome is bound at gating time by the pack's validator
(plan 5.2, FR9); until it is bound the flow is UNBOUND, and an UNBOUND flow
never passes. The binding records who bound it, when, and where the
binding is on the record.

These are the development smoke tests of plan section 7; the separately
authored gold answer sets of step S10 are a different thing and do not
live here."""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rag4ds.model.entry import ENTRY_ID_RE
from rag4ds.model.yamlstrict import DuplicateKeyError, strict_yaml_load
from rag4ds.retrieve.contract import GroundingRequest, GroundingRequestError, shown

FLOWS_FILE = Path("demo") / "flows.yaml"
FLOWS_SCHEMA = 1
STATUSES = ("OK", "NO_GROUNDING", "ERROR")
REQUEST_KEYS = ("layer", "domain", "phase", "top_k")
EXPECTED_KEYS = {"status", "present", "absent", "bound_by", "bound_on", "record"}
ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
PASS, FAIL, UNBOUND = "PASS", "FAIL", "UNBOUND"


class DemoFlowError(ValueError):
    """A flows file that does not say one thing unambiguously."""


@dataclass(frozen=True)
class Expected:
    status: str
    present: tuple[str, ...]
    absent: tuple[str, ...]
    bound_by: str
    bound_on: str
    record: str


@dataclass(frozen=True)
class DemoFlow:
    id: str
    query: str
    request: dict
    expected: Expected | None     # None: not yet bound by the pack's validator

    def grounding_request(self) -> GroundingRequest:
        return GroundingRequest(query=self.query, **self.request)


@dataclass(frozen=True)
class FlowResult:
    flow_id: str
    verdict: str                  # PASS | FAIL | UNBOUND
    status: str
    returned: tuple[tuple[str, int, str], ...]   # (entry id, version, layer), in rank order
    reasons: tuple[str, ...] = field(default=())
    # AR-rag4ds-057 r1 O2: the service's own reason when it answers ERROR, so a
    # failed run stays diagnosable (printed and recorded by tools/run_demo.py)
    error: str = ""


def _ids(value, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise DemoFlowError(f"{where} must be a list of entry ids")
    bad = [v for v in value if not ENTRY_ID_RE.fullmatch(v)]
    if bad or len(set(value)) != len(value):
        raise DemoFlowError(f"{where} holds a malformed or repeated entry id: {bad or value}")
    return tuple(value)


def _expected(raw, where: str) -> Expected | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != EXPECTED_KEYS:
        raise DemoFlowError(f"{where}.expected must be null (unbound) or carry exactly "
                            f"{sorted(EXPECTED_KEYS)}")
    if raw["status"] not in STATUSES:
        raise DemoFlowError(f"{where}.expected.status must be one of {STATUSES}")
    present = _ids(raw["present"], f"{where}.expected.present")
    absent = _ids(raw["absent"], f"{where}.expected.absent")
    if set(present) & set(absent):
        raise DemoFlowError(f"{where}.expected lists an id both present and absent")
    if raw["status"] != "OK" and present:
        raise DemoFlowError(f"{where}.expected: a {raw['status']} flow expects no entries present")
    if raw["status"] == "OK" and not present:
        raise DemoFlowError(f"{where}.expected: an OK flow names the entries it expects")
    for key in ("bound_by", "record"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise DemoFlowError(f"{where}.expected.{key} must be a non-empty string")
    if not isinstance(raw["bound_on"], str) or not ISO_DATE.fullmatch(raw["bound_on"]):
        raise DemoFlowError(f"{where}.expected.bound_on must be a YYYY-MM-DD date string")
    # AR-rag4ds-057 r1 O3: the shape alone let 2026-99-99 load; a binding is dated
    try:
        datetime.date.fromisoformat(raw["bound_on"])
    except ValueError:
        raise DemoFlowError(f"{where}.expected.bound_on {raw['bound_on']} is not a calendar "
                            f"date") from None
    return Expected(raw["status"], present, absent, raw["bound_by"], raw["bound_on"],
                    raw["record"])


def load_flows(pack_dir: Path) -> tuple[str, list[DemoFlow]]:
    """Read and validate packs/<pack>/demo/flows.yaml; returns (pack code, flows).
    Every way the file can fail to be read or to be valid is a DemoFlowError,
    never a Python error (claim 2 of AR-rag4ds-057): the named refusals below
    and, as the last line of both steps, any other error, named by its type
    (AR-rag4ds-057 r4 F7 - four audit rounds found a new corner of this one
    promise; the named refusals keep the messages exact, the last line keeps
    the promise for a corner not yet found)."""
    path = Path(pack_dir) / FLOWS_FILE
    # AR-rag4ds-057 F1: the file is human-edited, so it is read as the project's
    # other human-edited YAML is (AR-rag4ds-028 O1) - a key given twice is
    # refused, never resolved silently to its last value
    try:
        data = strict_yaml_load(path.read_text(encoding="utf-8"))
    except DuplicateKeyError as exc:
        raise DemoFlowError(f"{path}: {exc}; a key given twice says two things") from exc
    except UnicodeDecodeError as exc:
        # AR-rag4ds-057 r2 F5: an editor that saved the file in another encoding
        raise DemoFlowError(f"{path}: not UTF-8 text ({exc}); save it as UTF-8") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise DemoFlowError(f"{path}: not readable YAML: {exc}") from exc
    except Exception as exc:  # the last line: a refusal naming the error, never a traceback
        raise DemoFlowError(f"{path}: not readable - {type(exc).__name__}: {exc}") from exc
    try:
        return _validated(path, Path(pack_dir), data)
    except DemoFlowError:
        raise
    except Exception as exc:  # the last line, as above
        raise DemoFlowError(f"{path}: not a valid flows file - {type(exc).__name__} while "
                            f"checking it: {exc}") from exc


def _validated(path: Path, pack_dir: Path, data) -> tuple[str, list[DemoFlow]]:
    if not isinstance(data, dict) or set(data) != {"schema", "pack", "flows"}:
        raise DemoFlowError(f"{path}: must carry exactly schema, pack and flows")
    if data["schema"] != FLOWS_SCHEMA:
        raise DemoFlowError(f"{path}: schema must be {FLOWS_SCHEMA}")
    pack = data["pack"]
    if not isinstance(pack, str) or not ENTRY_ID_RE.fullmatch(pack) or pack != Path(pack_dir).name:
        raise DemoFlowError(f"{path}: pack {shown(pack)} must be the pack directory's own code")
    if not isinstance(data["flows"], list) or not data["flows"]:
        raise DemoFlowError(f"{path}: flows must be a non-empty list")
    flows, seen = [], set()
    for i, raw in enumerate(data["flows"]):
        where = f"{path.name}: flows[{i}]"
        if not isinstance(raw, dict) or set(raw) != {"id", "query", "request", "expected"}:
            raise DemoFlowError(f"{where} must carry exactly id, query, request and expected")
        fid = raw["id"]
        if not isinstance(fid, str) or not ENTRY_ID_RE.fullmatch(fid) or fid in seen:
            raise DemoFlowError(f"{where}: id {shown(fid)} malformed or repeated")
        seen.add(fid)
        request = raw["request"]
        if not isinstance(request, dict) or not set(request) <= set(REQUEST_KEYS):
            raise DemoFlowError(f"{where}.request may carry only {REQUEST_KEYS}")
        # AR-rag4ds-057 r1 O3: a pack's flows ask its own pack, never another's
        if "domain" in request and request["domain"] != pack:
            raise DemoFlowError(f"{where}.request.domain {shown(request['domain'])} must be the "
                                f"file's own pack, {pack}")
        flow = DemoFlow(fid, raw["query"], dict(request), _expected(raw["expected"], where))
        try:
            flow.grounding_request()
        except (GroundingRequestError, TypeError) as exc:
            raise DemoFlowError(f"{where}: not a valid grounding request: {exc}") from exc
        flows.append(flow)
    return pack, flows


def check(flow: DemoFlow, response: dict) -> FlowResult:
    """The mechanical verdict of one flow against one retrieval response."""
    returned = tuple((e["id"], e["version"], e["layer"]) for e in response.get("entries", []))
    status = response["status"]
    error = str((response.get("meta") or {}).get("error") or "")
    if flow.expected is None:
        return FlowResult(flow.id, UNBOUND, status, returned,
                          ("the expected outcome is not yet bound by the pack's validator",),
                          error)
    ids = [r[0] for r in returned]
    reasons = []
    if status != flow.expected.status:
        reasons.append(f"status {status}, expected {flow.expected.status}")
    reasons += [f"expected entry {x} not returned" for x in flow.expected.present if x not in ids]
    reasons += [f"entry {x} returned, expected absent" for x in flow.expected.absent if x in ids]
    return FlowResult(flow.id, FAIL if reasons else PASS, status, returned, tuple(reasons), error)


def unserved(flows: list[DemoFlow], served: set[str]) -> list[str]:
    """AR-rag4ds-057 r1 O3: the entry ids each bound flow names that the index
    does not hold, as "flow: id, id" - a mistyped id is a constraint that can
    never fire (an absent one) or can never be met (a present one)."""
    out = []
    for f in flows:
        if f.expected is None:
            continue
        missing = [x for x in f.expected.present + f.expected.absent if x not in served]
        if missing:
            out.append(f"{f.id}: {', '.join(missing)}")
    return out


def run(retriever, flows: list[DemoFlow]) -> list[FlowResult]:
    """Run every flow through the retrieval library, in file order."""
    return [check(f, retriever.ground(f.grounding_request()).to_dict()) for f in flows]
