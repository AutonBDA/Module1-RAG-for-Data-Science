"""S9d: the external-standard mapping road on real data (plan 4.5, FR11,
VT-1). packs/<code>/standards.yaml registers each standard the pack maps
to (file pinned by sha256, the verified class list, the relations a row
may not target); tools/file_mapping_draft.py refuses a draft that names an
unregistered standard, an unverified class, a relation, an entry the tree
does not hold, a bad relation word, a blank note or a duplicate row, and
files ONE mapping-draft candidate (the draft's bytes verbatim) plus a run
manifest; the validator approves it at the gate; the fold writes
packs/<code>/mappings/<standard>.json.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from rich.console import Console

from rag4ds.auditlog.journal import Journal
from rag4ds.gate.cli import DECISION_JOURNAL, GateContext, GateSession
from rag4ds.gate.identities import IDENTITIES_FILE, IdentityRegistry
from rag4ds.kb.fold import fold, mapping_artifact_bytes
from rag4ds.staging.store import StagingStore

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))
import file_mapping_draft as fmd  # noqa: E402

TS = "2026-09-20 15:00:00 JST"
SHIPPED_PACK = MODULE_ROOT / "packs" / "robotics"
SHIPPED_DRAFT = MODULE_ROOT / "authoring" / "robotics_cora-ieee-1872_mapping.yaml"
STAND_IN = b"a stand-in for the standard's file, pinned by its hash - never the text\n"

DRAFT = "\n".join([
    "# a comment the gate's parser drops and the candidate keeps: the draft travels verbatim",
    "standard: cora-ieee-1872",
    "rows:",
    "  - entry: protective-stop",
    "    class: corax:RobotMotion",
    "    relation: related",
    "    note: the process a stop interrupts",
    "",
])


def _registry(sha, **over):
    std = {"title": "t", "publisher": "p", "year": 2015, "file": "std.pdf", "sha256": sha,
           "pages": 1, "license": "l", "verified": "v",
           "classes": [{"name": "corax:RobotMotion", "clause": "6.7"},
                       {"name": "cora:Robot", "clause": "7.1"},
                       {"name": "rparts:Controller", "clause": "8.5", "example": True}],
           "relations": [{"name": "cora:robotPart", "clause": "7.2"}]}
    std.update(over)
    return {"schema": 1, "pack": "robotics", "raw_dir": "data/standards",
            "standards": {"cora-ieee-1872": std}}


def _scratch(tmp_path, registry=None, entries=("protective-stop",)):
    root = tmp_path / "module"
    pack = root / "packs" / "robotics"
    (pack / "entries").mkdir(parents=True)
    (pack / "manifest.yaml").write_text("schema: 1\n", encoding="utf-8")
    shutil.copy(MODULE_ROOT / IDENTITIES_FILE, root / IDENTITIES_FILE)
    std_dir = root / "data" / "standards"
    std_dir.mkdir(parents=True)
    (std_dir / "std.pdf").write_bytes(STAND_IN)
    sha = hashlib.sha256(STAND_IN).hexdigest()
    reg = registry if registry is not None else _registry(sha)
    (pack / "standards.yaml").write_text(yaml.safe_dump(reg, sort_keys=False), encoding="utf-8")
    for eid in entries:
        (pack / "entries" / f"{eid}.json").write_text(json.dumps(
            {"id": eid, "status": "current", "pack": "robotics", "layer": "domain",
             "type": "concept", "content": "c", "source_reference": "s", "version": 1}),
            encoding="utf-8")
    for d in ("journal", "stage", "runs"):
        (root / d).mkdir()
    (root / "authoring").mkdir()
    (root / "authoring" / "draft.yaml").write_bytes(DRAFT.encode("utf-8"))
    return root, sha


TS2 = "2026-09-20 16:30:00 JST"


def _run(root, draft="authoring/draft.yaml", author="Draft author - DA", dry_run=False,
         clock=lambda: TS, code_commit="0000000"):
    lines = []
    rc = fmd.run(root, "robotics", Path(draft), author, dry_run=dry_run, clock=clock,
                 code_commit=code_commit, out=lines.append)
    return rc, "\n".join(lines)


# ----- the shipped registry and draft -----


def test_the_shipped_registry_loads_and_the_shipped_draft_validates_against_the_tree():
    """The real registry: IEEE 1872-2015's verified class list (and the
    relations a row may not target); the AuR standard registered too. The
    real draft names only current entries of the shipped tree and verified
    classes. The pinned PDFs are not in CI, so the file check is NOT run
    here (it is exercised on the scratch standard below)."""
    reg = fmd.load_registry(SHIPPED_PACK)
    cora = reg["standards"]["cora-ieee-1872"]
    assert "corax:RobotMotion" in cora["class_names"]
    assert "cora:Robot" in cora["class_names"]
    assert "cora:robotPart" in cora["relation_names"]
    assert "cora:fullyAutonomousRobot" in cora["relation_names"]
    assert "rparts:robotProcessingPart" in cora["relation_names"]
    assert not (cora["class_names"] & cora["relation_names"])
    # AR-rag4ds-042 r1 F1: every relation clauses 6-9 declare, by subrelation or
    # by a relation-typing axiom - 42, the twelve orientation/pose ones included
    assert len(cora["class_names"]) == 35 and len(cora["relation_names"]) == 42
    for name in ("pos:extensionPos", "pos:overlapsPosition", "pos:refOCS", "pos:inOCS",
                 "pos:relOrientation", "pos:mapsOCS", "pos:oCompose", "pos:parentOCS",
                 "pos:inOPR", "pos:oRGenerated", "pos:posePosition", "pos:poseOrientation"):
        assert name in cora["relation_names"], name
    examples = {t["name"] for t in cora["classes"] if t.get("example") is True}
    assert examples == {"rparts:LaserSensor", "rparts:EndEffector", "rparts:RadioReceiver",
                        "rparts:Controller"}
    aur = reg["standards"]["aur-ieee-1872-2"]
    assert "aur:FunctionExecution" in aur["class_names"] and not aur["relation_names"]
    assert not (cora["class_names"] & aur["class_names"])  # prefixes keep the standards apart
    text = SHIPPED_DRAFT.read_bytes()
    assert b"\r" not in text
    artifact = fmd.validate_draft(text.decode("utf-8"), SHIPPED_PACK, reg)
    assert artifact["standard"] == "cora-ieee-1872"
    for row in artifact["rows"]:
        assert (SHIPPED_PACK / "entries" / f"{row['entry']}.json").is_file()
        assert row["class"] in cora["class_names"]
        assert row["relation"] in fmd.RELATIONS
        assert row["note"].strip()


# ----- refusals, by name -----


@pytest.mark.parametrize("edit, message", [
    (("standard: cora-ieee-1872", "standard: cora-ieee-9999"), "not registered"),
    (("class: corax:RobotMotion", "class: cora:robotPart"), "registered as a RELATION"),
    (("class: corax:RobotMotion", "class: corax:Robotmotion"), "not in the verified class list"),
    (("class: corax:RobotMotion", "class: RobotMotion"), "not in the verified class list"),
    (("entry: protective-stop", "entry: emergency-stop"), "not a current entry"),
    (("entry: protective-stop", "entry: Protective_Stop"), "not a valid entry id"),
    (("relation: related", "relation: broader-than"), "not one of"),
    (("note: the process a stop interrupts", "note: '  '"), "note must say why"),
    (("standard: cora-ieee-1872", "standard: cora-ieee-1872\nauthor: me"), "exactly the keys"),
])
def test_a_wrong_draft_is_refused_and_nothing_is_filed(tmp_path, edit, message):
    root, _ = _scratch(tmp_path)
    old, new = edit
    assert old in DRAFT
    (root / "authoring" / "draft.yaml").write_bytes(DRAFT.replace(old, new).encode("utf-8"))
    with pytest.raises(fmd.MappingDraftError, match=message):
        _run(root)
    assert not (root / "stage" / "candidates.jsonl").exists()
    assert not (root / "runs" / "authoring.jsonl").exists()


def test_a_duplicate_row_and_an_unregistered_author_are_refused(tmp_path):
    root, _ = _scratch(tmp_path)
    twice = DRAFT + DRAFT[DRAFT.index("  - entry"):]
    (root / "authoring" / "draft.yaml").write_bytes(twice.encode("utf-8"))
    with pytest.raises(fmd.MappingDraftError, match="mapped twice"):
        _run(root)
    (root / "authoring" / "draft.yaml").write_bytes(DRAFT.encode("utf-8"))
    with pytest.raises(fmd.MappingDraftError, match="author refused"):
        _run(root, author="Some Body")
    assert not (root / "stage" / "candidates.jsonl").exists()


def test_a_retired_or_foreign_entry_is_not_current(tmp_path):
    """The tree holds current entries only, but the check reads the file:
    a file whose id or status disagrees with its name is not an entry."""
    root, _ = _scratch(tmp_path, entries=())
    p = root / "packs" / "robotics" / "entries" / "protective-stop.json"
    p.write_text(json.dumps({"id": "other", "status": "current"}), encoding="utf-8")
    with pytest.raises(fmd.MappingDraftError, match="not a current entry"):
        _run(root)
    p.write_text(json.dumps({"id": "protective-stop", "status": "retracted"}), encoding="utf-8")
    with pytest.raises(fmd.MappingDraftError, match="not a current entry"):
        _run(root)


def test_the_standard_file_must_be_present_and_hash_to_its_pin(tmp_path):
    root, sha = _scratch(tmp_path)
    std = root / "data" / "standards" / "std.pdf"
    std.write_bytes(STAND_IN + b"another edition\n")
    with pytest.raises(fmd.MappingDraftError, match="not the pinned"):
        _run(root)
    std.unlink()
    with pytest.raises(fmd.MappingDraftError, match="is absent"):
        _run(root)
    assert not (root / "stage" / "candidates.jsonl").exists()


def test_a_standard_without_a_verified_class_list_takes_no_row(tmp_path):
    root, sha = _scratch(tmp_path, registry=_registry(sha_of(STAND_IN), classes=[]))
    with pytest.raises(fmd.MappingDraftError, match="registers no verified class"):
        _run(root)


def sha_of(b):
    return hashlib.sha256(b).hexdigest()


@pytest.mark.parametrize("over, message", [
    ({"classes": [{"name": "cora:Robot", "clause": "7.1"}],
      "relations": [{"name": "cora:Robot", "clause": "7.1"}]}, "class and relation"),
    ({"classes": [{"name": "Robot", "clause": "7.1"}]}, "must be prefix:Term"),
    ({"classes": [{"name": "cora:Robot", "clause": "7.1"},
                  {"name": "cora:Robot", "clause": "7.5"}]}, "listed twice"),
    ({"sha256": "abc"}, "64 lowercase hex"),
    ({"file": "../std.pdf"}, "bare file name"),
    ({"extra": 1}, "unknown"),
    # AR-rag4ds-042 r1 F4: the spelling rule - prefix lowercase letters, a colon,
    # the term letters and digits starting with a letter
    ({"classes": [{"name": "CORA:Robot", "clause": "7.1"}]}, "must be prefix:Term"),
    ({"classes": [{"name": "cora:9Robot", "clause": "7.1"}]}, "must be prefix:Term"),
    ({"classes": [{"name": "c0ra:Robot", "clause": "7.1"}]}, "must be prefix:Term"),
    ({"classes": [{"name": "cora:Robot:Arm", "clause": "7.1"}]}, "must be prefix:Term"),
    ({"relations": [{"name": "cora:robot part", "clause": "7.2"}]}, "must be prefix:Term"),
])
def test_a_wrong_registry_is_refused(tmp_path, over, message):
    root, sha = _scratch(tmp_path, registry=_registry(sha_of(STAND_IN), **over))
    with pytest.raises(fmd.MappingDraftError, match=message):
        fmd.load_registry(root / "packs" / "robotics")


def test_one_hash_may_pin_only_one_standard(tmp_path):
    """AR-rag4ds-042 r1 F4: a registry pinning one file for two standards
    would let a row attest the wrong document; refused by name."""
    reg = _registry(sha_of(STAND_IN))
    reg["standards"]["aur-x"] = dict(reg["standards"]["cora-ieee-1872"])
    root, _ = _scratch(tmp_path, registry=reg)
    with pytest.raises(fmd.MappingDraftError,
                       match="aur-x: sha256 already pinned by 'cora-ieee-1872'"):
        fmd.load_registry(root / "packs" / "robotics")


@pytest.mark.parametrize("raw_dir", ["../outside", "data/../../outside", "..",
                                     "data/standards/../../.."])
def test_raw_dir_must_stay_inside_the_module_root(tmp_path, raw_dir):
    """AR-rag4ds-042 r1 F3: `file` is a bare name, so the standard's copy
    lives inside the module; raw_dir may not walk out of it."""
    reg = _registry(sha_of(STAND_IN))
    reg["raw_dir"] = raw_dir
    root, _ = _scratch(tmp_path, registry=reg)
    outside = tmp_path / "outside"
    outside.mkdir(exist_ok=True)
    (outside / "std.pdf").write_bytes(STAND_IN)
    with pytest.raises(fmd.MappingDraftError, match="must stay inside the module root"):
        fmd.load_registry(root / "packs" / "robotics")
    with pytest.raises(fmd.MappingDraftError, match="must stay inside the module root"):
        _run(root)
    assert not (root / "stage" / "candidates.jsonl").exists()


@pytest.mark.parametrize("schema", [None, 2, "1", 0])
def test_a_registry_of_another_schema_is_refused(tmp_path, schema):
    """AR-rag4ds-043 r1 F2: the registry's own schema version is checked -
    absent, a later version, a string or zero are all refused by name, and
    nothing is filed against a file this code does not know how to read."""
    reg = _registry(sha_of(STAND_IN))
    if schema is None:
        del reg["schema"]
    else:
        reg["schema"] = schema
    root, _ = _scratch(tmp_path, registry=reg)
    with pytest.raises(fmd.MappingDraftError, match="schema must be 1"):
        fmd.load_registry(root / "packs" / "robotics")
    with pytest.raises(fmd.MappingDraftError, match="schema must be 1"):
        _run(root)
    assert not (root / "stage" / "candidates.jsonl").exists()
    assert not (root / "runs" / "authoring.jsonl").exists()


def test_the_default_clock_refuses_a_machine_that_is_not_on_jst(tmp_path, monkeypatch):
    """AR-rag4ds-043 r1 F2 (and the AR-rag4ds-033 F2 line): every other test
    injects a clock, so the shipped one - which refuses a machine whose
    local time is not JST, because every record's timestamp claims JST -
    was never taken. Here it is, with nothing injected."""
    root, _ = _scratch(tmp_path)

    class NotJST:
        tm_gmtoff = 0  # UTC

    monkeypatch.setattr(fmd.time, "localtime", lambda *a: NotJST())
    with pytest.raises(fmd.MappingDraftError, match="not JST"):
        fmd.jst_now()
    with pytest.raises(fmd.MappingDraftError, match="not JST"):
        fmd.run(root, "robotics", Path("authoring/draft.yaml"), "Draft author - DA",
                code_commit="0000000", out=lambda _m: None)
    assert not (root / "stage" / "candidates.jsonl").exists()
    assert not (root / "runs" / "authoring.jsonl").exists()
    # the same refusal guards the recovery path, which writes a manifest of
    # its own; a dry run needs no timestamp and still works
    assert fmd.run(root, "robotics", Path("authoring/draft.yaml"), "Draft author - DA",
                   dry_run=True, code_commit="0000000", out=lambda _m: None) == 0
    jst = type("JST", (), {"tm_gmtoff": 9 * 3600})
    monkeypatch.setattr(fmd.time, "localtime", lambda *a: jst())
    assert fmd.jst_now().endswith(" JST")


def test_a_raw_dir_inside_the_module_in_another_spelling_is_fine(tmp_path):
    reg = _registry(sha_of(STAND_IN))
    reg["raw_dir"] = "data/./standards"
    root, _ = _scratch(tmp_path, registry=reg)
    assert _run(root, dry_run=True)[0] == 0


def test_a_missing_draft_is_a_named_refusal(tmp_path):
    """AR-rag4ds-042 r1 O1."""
    root, _ = _scratch(tmp_path)
    with pytest.raises(fmd.MappingDraftError, match="draft .* is absent"):
        _run(root, draft="authoring/nowhere.yaml")


def test_a_worked_example_class_is_named_to_the_operator_and_in_the_manifest(tmp_path):
    """AR-rag4ds-042 r1 O2: the candidate carries only the draft, so the
    registry's example flag is surfaced where the filing is recorded."""
    root, _ = _scratch(tmp_path)
    (root / "authoring" / "draft.yaml").write_bytes(
        DRAFT.replace("class: corax:RobotMotion", "class: rparts:Controller").encode("utf-8"))
    rc, out = _run(root)
    assert rc == 0 and "worked-example class(es) targeted: ['rparts:Controller']" in out
    m = Journal(root / "runs" / "authoring.jsonl").read_all()[-1]
    assert m["example_classes"] == ["rparts:Controller"]
    (root / "authoring" / "draft.yaml").write_bytes(DRAFT.encode("utf-8"))
    rc, out = _run(root)
    assert rc == 0 and "worked-example" not in out
    assert Journal(root / "runs" / "authoring.jsonl").read_all()[-1]["example_classes"] == []


def test_an_interrupted_run_is_completed_by_the_next_run_not_refused_forever(tmp_path, monkeypatch):
    """AR-rag4ds-042 r1 F2: a crash between the candidate and its manifest
    left a PENDING candidate without provenance and a tool that refused on
    every re-run. Now the re-run appends the missing manifest, marked
    recovered, and the run after that is refused as before."""
    root, sha = _scratch(tmp_path)
    real = fmd.Journal

    class CrashOnAppend(real):
        def append(self, record):
            raise OSError("simulated crash while appending the run manifest")

    monkeypatch.setattr(fmd, "Journal", CrashOnAppend)
    with pytest.raises(OSError, match="simulated crash"):
        _run(root)
    monkeypatch.setattr(fmd, "Journal", real)
    store = StagingStore(root / "stage" / "candidates.jsonl")
    (cid, rec), = store.candidates().items()
    assert not (root / "runs" / "authoring.jsonl").exists()
    rc, out = _run(root, dry_run=True, clock=lambda: TS2)
    assert rc == 0 and "NO run manifest" in out and cid in out
    assert not (root / "runs" / "authoring.jsonl").exists()
    # AR-rag4ds-042 r2 F6: the recovery runs on a LATER clock and under another
    # commit, so each field is checked against its own source
    rc, out = _run(root, clock=lambda: TS2, code_commit="1111111")
    assert rc == 0 and "without a run manifest" in out and "marked recovered" in out
    runs = real(root / "runs" / "authoring.jsonl").read_all()
    assert len(runs) == 1
    m = runs[0]
    assert m["candidate"] == cid and m["recovered_after_interrupted_run"] is True
    assert m["candidate_filed"] == rec["filed"] == TS
    assert m["timestamp"] == TS2 and TS2 != TS
    assert m["code_commit"] == "1111111" and m["code_commit_is_recovery_run"] is True
    assert m["draft_sha256"] == sha_of(DRAFT.encode("utf-8"))
    assert m["standard_file"] == {"file": "std.pdf", "sha256": sha}
    assert len(store.candidates()) == 1 and store.state_of(cid) == "PENDING"
    rc, out = _run(root)
    assert rc == 2 and "already filed" in out
    assert len(real(root / "runs" / "authoring.jsonl").read_all()) == 1


def test_a_dry_run_needs_no_git_checkout_and_a_live_run_without_one_is_a_named_refusal(tmp_path):
    """AR-rag4ds-042 r2 O5: the code commit is resolved only when a manifest
    is about to be written; a scratch module under the temp directory is
    not inside a checkout."""
    root, _ = _scratch(tmp_path)
    rc, out = _run(root, dry_run=True, code_commit=None)
    assert rc == 0 and "DRY RUN" in out
    with pytest.raises(fmd.MappingDraftError, match="code commit unknown"):
        _run(root, code_commit=None)
    assert not (root / "stage" / "candidates.jsonl").exists()
    assert not (root / "runs" / "authoring.jsonl").exists()


def test_a_registry_naming_another_pack_or_a_bad_standard_id_is_refused(tmp_path):
    root, sha = _scratch(tmp_path)
    reg = _registry(sha)
    reg["pack"] = "bio"
    p = root / "packs" / "robotics" / "standards.yaml"
    p.write_text(yaml.safe_dump(reg), encoding="utf-8")
    with pytest.raises(fmd.MappingDraftError, match="is not the directory's"):
        fmd.load_registry(root / "packs" / "robotics")
    reg = _registry(sha)
    reg["standards"] = {"CORA 1872": reg["standards"]["cora-ieee-1872"]}
    p.write_text(yaml.safe_dump(reg), encoding="utf-8")
    with pytest.raises(fmd.MappingDraftError, match="entry-id character rule"):
        fmd.load_registry(root / "packs" / "robotics")


# ----- the road end to end: file once, the validator approves, the fold writes -----


def _gate(root, approver, keys):
    store = StagingStore(root / "stage" / "candidates.jsonl")
    ctx = GateContext(decisions=Journal(root / DECISION_JOURNAL), staging=store,
                      runs=Journal(root / "runs" / "gate-sessions.jsonl"), approver=approver,
                      code_commit="0000000", clock=lambda: TS, source_resolver=lambda h: None,
                      identities=IdentityRegistry.load(root / IDENTITIES_FILE))
    console = Console(record=True, markup=False, highlight=False, width=100)
    lines = list(keys)
    session = GateSession(ctx, console, lambda prompt: lines.pop(0) if lines else "q")
    session.run()
    return store, session, ctx


def test_the_draft_is_filed_once_verbatim_then_approved_then_folded(tmp_path):
    root, sha = _scratch(tmp_path)
    rc, out = _run(root, dry_run=True)
    assert rc == 0 and "DRY RUN" in out and "1 row(s)" in out
    assert not (root / "stage" / "candidates.jsonl").exists()
    rc, out = _run(root)
    assert rc == 0 and "filed cand-" in out
    store = StagingStore(root / "stage" / "candidates.jsonl")
    cands = store.candidates()
    assert len(cands) == 1
    cid, rec = next(iter(cands.items()))
    c = rec["candidate"]
    assert c == {"kind": "mapping-draft", "pack": "robotics", "standard": "cora-ieee-1872",
                 "author": "Draft author - DA", "yaml_draft": DRAFT}
    assert store.state_of(cid) == "PENDING"
    runs = Journal(root / "runs" / "authoring.jsonl").read_all()
    assert len(runs) == 1
    m = runs[0]
    assert m["kind"] == "mapping-authoring" and m["candidate"] == cid and m["rows"] == 1
    assert m["example_classes"] == [] and "recovered_after_interrupted_run" not in m
    assert "code_commit_is_recovery_run" not in m and "candidate_filed" not in m
    assert m["draft_sha256"] == sha_of(DRAFT.encode("utf-8"))
    assert m["standard_file"] == {"file": "std.pdf", "sha256": sha}
    assert m["timestamp"] == TS and m["code_commit"] == "0000000"
    # the same bytes by the same author: refused, nothing appended
    rc, out = _run(root)
    assert rc == 2 and "already filed" in out
    assert len(store.candidates()) == 1
    assert len(Journal(root / "runs" / "authoring.jsonl").read_all()) == 1
    # someone without the pack's validator role may not approve it (the alternate; the
    # designee is a robotics validator since 2026-10-05, refinement A); the validator may
    store, session, _ = _gate(root, "I. Paik", ["a", "q"])
    assert store.state_of(cid) == "PENDING"
    store, session, ctx = _gate(root, "K. Naruse", ["a", "q"])
    assert store.state_of(cid) == "APPROVED"
    assert session.summary.counts["approve"] == 1
    records = ctx.decisions.read_all()
    approval = [r for r in records if r.get("candidate_id") == cid]
    assert len(approval) == 1
    art = approval[0]["mapping_artifact"]
    assert art == yaml.safe_load(DRAFT)
    assert approval[0]["candidate_content_hash"] == sha_of(DRAFT.encode("utf-8"))
    # the fold writes the canonical artifact at the 4.5 path
    tree = tmp_path / "tree"
    fold(records, tree)
    path = tree / "packs" / "robotics" / "mappings" / "cora-ieee-1872.json"
    assert path.read_bytes() == mapping_artifact_bytes(art)


def test_a_changed_draft_is_a_new_candidate_and_the_author_is_in_the_id(tmp_path):
    root, _ = _scratch(tmp_path)
    assert _run(root)[0] == 0
    (root / "authoring" / "draft.yaml").write_bytes(
        DRAFT.replace("a stop interrupts", "a stop halts").encode("utf-8"))
    assert _run(root)[0] == 0
    store = StagingStore(root / "stage" / "candidates.jsonl")
    assert len(store.candidates()) == 2
    (root / "authoring" / "draft.yaml").write_bytes(DRAFT.encode("utf-8"))
    assert _run(root, author="K. Naruse")[0] == 0
    assert len(store.candidates()) == 3


def test_a_draft_with_a_carriage_return_is_refused(tmp_path):
    root, _ = _scratch(tmp_path)
    (root / "authoring" / "draft.yaml").write_bytes(DRAFT.replace("\n", "\r\n").encode("utf-8"))
    with pytest.raises(fmd.MappingDraftError, match="carriage return"):
        _run(root)
