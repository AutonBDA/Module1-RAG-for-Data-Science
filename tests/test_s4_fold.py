"""S4 acceptance (plan 10.1): 6.5 byte-equality green on synthetic fixtures
covering every decision type, both artifact kinds, and a retraction - the
reconstruction test, live. Plus fold-side shape, path, and chain
validation (NFR1/I1; AR-rag4ds-016 F1-F7) and the live-tree gate (9.2).
"""

from pathlib import Path

import pytest

import rag4ds.kb.fold as fold_mod
from rag4ds.auditlog.journal import Journal
from rag4ds.gate.decisions import DecisionError, approve_entry, approve_mapping, reject, retract
from rag4ds.kb.fold import (
    DECISION_JOURNAL,
    FoldError,
    assert_reconstructable,
    compile_tree,
    entry_tree_path,
    fold,
    mapping_artifact_bytes,
    mapping_tree_path,
    verify_reconstruction,
)
from rag4ds.model.entry import Entry, EntryValidationError, canonical_bytes

MODULE_ROOT = Path(__file__).resolve().parents[1]
TS = "2026-08-22 21:10 JST"
WHO = "k-naruse"
SHA = "c" * 64
KNOWN4 = {"actuator-failure-modes": 1, "encoder-drift": 1, "crisp-dm": 1, "anomaly-detection": 1}


def ent(**o):
    f = dict(id="actuator-failure-modes", type="concept", layer="domain", pack="robotics",
             content="Bearing wear dominates actuator failure.", source_reference="corpus:r#1",
             version=1, status="current")
    f.update(o)
    return Entry(**f)


def full_journal(path):
    """Every decision type, both artifact kinds, a supersession, a retraction."""
    j = Journal(path)
    j.append(approve_entry("approve", WHO, TS, "cand text", "corpus:r#1", ent(), {}))
    j.append(approve_entry("edit-then-approve", WHO, TS, "cand text 2", "corpus:r#2",
                           ent(id="encoder-drift", content="Encoder drift precedes failure."), {}))
    j.append(approve_entry("approve", WHO, TS, "general", "corpus:g#1",
                           ent(id="crisp-dm", layer="general", pack="shared",
                               content="CRISP-DM has six phases."), {}))
    j.append(approve_entry("approve", WHO, TS, "task", "corpus:t#1",
                           ent(id="anomaly-detection", layer="task", pack="shared",
                               type="task-pattern",
                               content="Anomaly detection flags rare events."), {}))
    j.append(approve_mapping("approve", WHO, TS, SHA, "corpus:cora#1",
                             {"standard": "cora-ieee-1872", "rows": [
                                 {"entry": "actuator-failure-modes", "class": "cora:Failure",
                                  "relation": "narrower-than", "note": "n"}]}, "robotics"))
    j.append(reject("reject", WHO, TS, "bad cand", "corpus:r#9", "unsupported"))
    j.append(reject("reject-duplicate", WHO, TS, "dup cand", "corpus:r#9", "dup",
                    "journal.jsonl:1"))
    # supersession of actuator-failure-modes v1 -> v2
    j.append(approve_entry("approve", WHO, TS, "better text", "corpus:r#3",
                           ent(version=2, content="Bearing wear and encoder drift dominate."),
                           KNOWN4, supersedes={"id": "actuator-failure-modes", "version": 1}))
    # retraction of encoder-drift v1
    j.append(retract(WHO, TS, "encoder-drift", 1,
                     ent(id="encoder-drift", status="retracted",
                         content="Encoder drift precedes failure."),
                     dict(KNOWN4, **{"actuator-failure-modes": 2})))
    return j


# --- the S4 exit criterion ---


def test_compile_then_reconstruction_is_byte_equal(tmp_path):
    j = full_journal(tmp_path / "journal.jsonl")
    tree = tmp_path / "tree"
    result = compile_tree(j, tree)
    assert result.versions == {"actuator-failure-modes": 2, "crisp-dm": 1, "anomaly-detection": 1}
    assert result.retired == frozenset({"encoder-drift"})
    assert verify_reconstruction(j, tree) == []
    assert_reconstructable(j, tree)  # does not raise


def test_live_tree_is_reconstructable_from_the_live_journal():
    """Plan 9.2: the reconstruction test gates the repository's own journal
    and tree (AR-rag4ds-016 O1). Trivially true today (empty journal,
    .gitkeep-only trees); it catches the first hand-placed file."""
    assert_reconstructable(Journal(MODULE_ROOT / DECISION_JOURNAL), MODULE_ROOT)


def test_tree_layout_follows_the_9_1_rule(tmp_path):
    j = full_journal(tmp_path / "journal.jsonl")
    tree = tmp_path / "tree"
    compile_tree(j, tree)
    assert (tree / "packs/robotics/entries/actuator-failure-modes.json").is_file()
    assert not (tree / "packs/robotics/entries/encoder-drift.json").exists()  # retracted
    assert (tree / "shared/general/crisp-dm.json").is_file()
    assert (tree / "shared/tasks/anomaly-detection.json").is_file()
    assert (tree / "packs/robotics/mappings/cora-ieee-1872.json").is_file()
    # the superseded version is overwritten: the file is v2's canonical bytes
    v2 = ent(version=2, content="Bearing wear and encoder drift dominate.")
    live = (tree / "packs/robotics/entries/actuator-failure-modes.json").read_bytes()
    assert live == canonical_bytes(v2)


def test_fold_is_deterministic(tmp_path):
    j = full_journal(tmp_path / "journal.jsonl")
    a, b = tmp_path / "a", tmp_path / "b"
    fold(j.read_all(), a)
    fold(j.read_all(), b)
    la = {p.relative_to(a): p.read_bytes() for p in a.rglob("*") if p.is_file()}
    lb = {p.relative_to(b): p.read_bytes() for p in b.rglob("*") if p.is_file()}
    assert la == lb and len(la) == 4


def test_reconstruction_detects_edited_missing_and_extra_files(tmp_path):
    j = full_journal(tmp_path / "journal.jsonl")
    tree = tmp_path / "tree"
    compile_tree(j, tree)
    f = tree / "shared/general/crisp-dm.json"
    f.write_bytes(f.read_bytes().replace(b"six", b"seven"))
    assert any(d.startswith("DIFFERS") for d in verify_reconstruction(j, tree))
    f.unlink()
    assert any(d.startswith("MISSING") for d in verify_reconstruction(j, tree))
    compile_tree(j, tree)
    (tree / "packs/robotics/entries/hand-made.json").write_bytes(b"{}\n")
    diffs = verify_reconstruction(j, tree)
    assert any(d.startswith("EXTRA") for d in diffs)
    with pytest.raises(FoldError, match="NOT reconstructable"):
        assert_reconstructable(j, tree)


def test_gitkeep_is_ignored_only_by_exact_basename_inside_gated_dirs(tmp_path):
    """AR-rag4ds-016 F5: the live tree commits shared/general/.gitkeep INSIDE a
    gated dir - that is what the exemption is for; a *.gitkeep-suffixed file
    is not a placeholder and is EXTRA."""
    j = full_journal(tmp_path / "journal.jsonl")
    tree = tmp_path / "tree"
    compile_tree(j, tree)
    (tree / "shared/general/.gitkeep").write_bytes(b"")
    (tree / "packs/robotics/entries/.gitkeep").write_bytes(b"")
    assert verify_reconstruction(j, tree) == []
    (tree / "shared/general/smuggled.gitkeep").write_bytes(b'{"x":1}\n')
    diffs = verify_reconstruction(j, tree)
    assert diffs == ["EXTRA in tree (no decision produces it): shared/general/smuggled.gitkeep"]


def test_mapping_artifact_bytes_are_canonical():
    raw = mapping_artifact_bytes({"standard": "x", "rows": [{"b": 1, "a": 2}]})
    assert raw == b'{"rows":[{"a":2,"b":1}],"standard":"x"}\n'


def test_entry_tree_path_rule():
    e = ent().to_dict()
    assert entry_tree_path(e).as_posix() == "packs/robotics/entries/actuator-failure-modes.json"
    e = ent(id="crisp-dm", layer="general", pack="shared").to_dict()
    assert entry_tree_path(e).as_posix() == "shared/general/crisp-dm.json"
    e = ent(id="t", layer="task", pack="shared", type="task-pattern").to_dict()
    assert entry_tree_path(e).as_posix() == "shared/tasks/t.json"


# --- F1: the pack names a directory; it is pinned at creation AND at the fold ---


@pytest.mark.parametrize("pack", ["../../escaped", "C:/abs", "/abs", "Robotics", "a/b", "a b"])
def test_entry_refuses_packs_that_are_not_pack_codes(pack):
    with pytest.raises(EntryValidationError, match="pack .* must be a pack code"):
        ent(pack=pack)


@pytest.mark.parametrize("pack", ["../../escaped", "Robotics", "a/b"])
def test_fold_refuses_forged_entry_packs_and_writes_nothing_outside(tmp_path, pack):
    """Construct a gate-legal approval, then forge the pack in the record
    (the gate would refuse it now; the log is attacked directly)."""
    rec = approve_entry("approve", WHO, TS, "c", "s", ent(id="pwned"), {})
    rec["entry"] = dict(rec["entry"], pack=pack)
    root = tmp_path / "deep" / "deeper" / "root"
    with pytest.raises(FoldError, match="record 1: .*pack"):
        fold([rec], root)
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


@pytest.mark.parametrize("field,value", [("pack", "../../x"), ("pack", "Robotics"),
                                         ("standard", "../esc"), ("standard", "Cora")])
def test_fold_refuses_forged_mapping_paths(tmp_path, field, value):
    rows = [{"entry": "x", "class": "c:X", "relation": "equivalent", "note": "n"}]
    rec = approve_mapping("approve", WHO, TS, SHA, "s", {"standard": "cora", "rows": rows},
                          "robotics")
    if field == "pack":
        rec["pack"] = value
    else:
        rec["mapping_artifact"] = dict(rec["mapping_artifact"], standard=value)
    with pytest.raises(FoldError, match="record 1: mapping approval"):
        fold([rec], tmp_path / "root")
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []
    with pytest.raises(FoldError):
        mapping_tree_path(rec)


def test_contained_path_guard_is_independent_of_the_charset(tmp_path):
    with pytest.raises(FoldError, match="not a plain relative path"):
        fold_mod._contained(tmp_path, Path("packs/../x.json"), "t")
    with pytest.raises(FoldError, match="not a plain relative path"):
        fold_mod._contained(tmp_path, Path(tmp_path / "abs.json"), "t")
    assert fold_mod._contained(tmp_path, Path("packs/p/entries/x.json"), "t") == (
        tmp_path / "packs/p/entries/x.json"
    )


# --- F2: a supersession that moves the entry leaves one file, and retraction clears it ---


@pytest.mark.parametrize("move", ["pack", "layer"])
def test_supersession_that_moves_the_entry_removes_the_old_file(tmp_path, move):
    j = Journal(tmp_path / "journal.jsonl")
    j.append(approve_entry("approve", WHO, TS, "c", "s", ent(id="drift", content="v1"), {}))
    if move == "pack":
        v2 = ent(id="drift", version=2, content="v2", pack="grocery")
        old, new = "packs/robotics/entries/drift.json", "packs/grocery/entries/drift.json"
        final = dict(pack="grocery")
    else:
        v2 = ent(id="drift", version=2, content="v2", layer="general", pack="shared")
        old, new = "packs/robotics/entries/drift.json", "shared/general/drift.json"
        final = dict(layer="general", pack="shared")
    j.append(approve_entry("approve", WHO, TS, "c", "s", v2, {"drift": 1},
                           supersedes={"id": "drift", "version": 1}))
    tree = tmp_path / "tree"
    compile_tree(j, tree)
    assert (tree / new).read_bytes() == canonical_bytes(v2)
    assert not (tree / old).exists()
    j.append(retract(WHO, TS, "drift", 2,
                     ent(id="drift", version=2, content="v2", status="retracted", **final),
                     {"drift": 2}))
    compile_tree(j, tree)
    assert list(tree.rglob("*.json")) == []  # the retracted entry is gone everywhere
    assert verify_reconstruction(j, tree) == []


# --- F4: a retracted id is retired for the life of the log ---


def test_retracted_id_is_retired_at_fold_and_gate(tmp_path):
    j = full_journal(tmp_path / "journal.jsonl")
    result = fold(j.read_all(), tmp_path / "f")
    assert "encoder-drift" in result.retired
    # the gate, fed the fold's view, refuses reuse and supersession
    with pytest.raises(DecisionError, match="retracted and stays reserved"):
        approve_entry("approve", WHO, TS, "c", "s", ent(id="encoder-drift"), result.versions,
                      retired=result.retired)
    with pytest.raises(DecisionError, match="retracted and stays reserved"):
        approve_entry("approve", WHO, TS, "c", "s", ent(id="encoder-drift", version=2),
                      result.versions, supersedes={"id": "encoder-drift", "version": 1},
                      retired=result.retired)
    with pytest.raises(DecisionError, match="retracted and stays reserved"):  # case-insensitive
        approve_entry("approve", WHO, TS, "c", "s", ent(id="encoder-drift"), {},
                      retired=frozenset({"ENCODER-DRIFT"}))
    with pytest.raises(DecisionError, match="retired must be a set"):
        approve_entry("approve", WHO, TS, "c", "s", ent(id="fresh"), {}, retired="encoder-drift")
    with pytest.raises(DecisionError, match="retired must be a set"):  # a mapping is not a set
        approve_entry("approve", WHO, TS, "c", "s", ent(id="fresh"), {}, retired={"a": 1})
    # and the fold refuses a forged reuse
    reuse = approve_entry("approve", WHO, TS, "c", "s", ent(id="encoder-drift"), {})
    with pytest.raises(FoldError, match="record 10: .*retracted and stays reserved"):
        fold(j.read_all() + [reuse], tmp_path / "g")


# --- fold-side chain validation: a corrupt log never folds silently ---


def test_fold_refuses_conflicting_supersession(tmp_path):
    recs = [
        approve_entry("approve", WHO, TS, "c", "s", ent(), {}),
        approve_entry("approve", WHO, TS, "c", "s", ent(version=2),
                      {"actuator-failure-modes": 1},
                      supersedes={"id": "actuator-failure-modes", "version": 1}),
    ]
    # hand-forge a second v2 superseding v1 (the gate would refuse; the log is attacked)
    forged = dict(recs[1])
    forged["entry"] = dict(forged["entry"], content="conflicting")
    with pytest.raises(FoldError, match="does not match the chain"):
        fold(recs + [forged], tmp_path / "f")


def test_fold_refuses_fresh_approval_of_known_id_and_phantom_retraction(tmp_path):
    a = approve_entry("approve", WHO, TS, "c", "s", ent(), {})
    with pytest.raises(FoldError, match="already-known id"):
        fold([a, a], tmp_path / "f1")
    r = retract(WHO, TS, "actuator-failure-modes", 1,
                ent(status="retracted"), {"actuator-failure-modes": 1})
    with pytest.raises(FoldError, match="does not match the chain"):
        fold([r], tmp_path / "f2")


def test_fold_refuses_fresh_approval_not_at_version_1(tmp_path):
    """AR-rag4ds-016 F6 (first half)."""
    a = approve_entry("approve", WHO, TS, "c", "s", ent(), {})
    a["entry"] = dict(a["entry"], version=7)
    with pytest.raises(FoldError, match="record 1: fresh approval at version 7"):
        fold([a], tmp_path / "f")


def test_fold_refuses_retraction_whose_file_is_missing_by_injection(tmp_path, monkeypatch):
    """AR-rag4ds-016 F6 (second half): the branch is unreachable through the
    fold's own control flow; reach it by making the writer skip one file."""
    real_write = fold_mod._write

    def skipping_write(root, rel, data, where):
        if rel.name == "encoder-drift.json":
            return
        real_write(root, rel, data, where)

    monkeypatch.setattr(fold_mod, "_write", skipping_write)
    j = full_journal(tmp_path / "journal.jsonl")
    with pytest.raises(FoldError, match="record 9: retraction target file .* is missing"):
        fold(j.read_all(), tmp_path / "f")


def test_fold_refuses_non_empty_target_and_unknown_type(tmp_path):
    target = tmp_path / "f"
    target.mkdir()
    (target / "stray").write_bytes(b"x")
    with pytest.raises(FoldError, match="not empty"):
        fold([], target)
    with pytest.raises(FoldError, match="record 1: unknown decision type"):
        fold([{"type": "bless"}], tmp_path / "g")


@pytest.mark.parametrize("rec,msg", [
    ("not-a-dict", "record 1: a decision record must be an object"),
    ({"type": "approve"}, "exactly one of entry, mapping_artifact"),
    ({"type": "approve", "entry": {}, "mapping_artifact": {}}, "exactly one of"),
    ({"type": "approve", "entry": "string"}, "entry must be an object"),
    ({"type": "approve", "entry": {"version": 1}}, "string id and an integer version"),
    ({"type": "approve", "entry": {"id": "x", "version": True}}, "integer version"),
    ({"type": "approve", "entry": {"id": "x", "version": 2}, "supersedes": {"id": "x"}},
     "supersedes must be"),
    ({"type": "approve", "entry": {"id": "x", "version": 2},
      "supersedes": {"id": "y", "version": 1}}, "is not the entry id"),
    ({"type": "approve", "mapping_artifact": "nope", "pack": "p"}, "mapping_artifact must be"),
    ({"type": "approve", "entry": {"id": "x", "version": 1}}, "not a valid FR1 entry"),
    ({"type": "retract"}, "string target_id and an integer target_version"),
    ({"type": "retract", "target_id": 3, "target_version": 1}, "string target_id"),
])
def test_malformed_records_raise_fold_error_with_the_record_index(tmp_path, rec, msg):
    """AR-rag4ds-016 F7: never KeyError/TypeError/AttributeError."""
    with pytest.raises(FoldError, match=msg):
        fold([rec], tmp_path / "f")


def test_compile_clears_stale_gated_content(tmp_path):
    j = full_journal(tmp_path / "journal.jsonl")
    tree = tmp_path / "tree"
    compile_tree(j, tree)
    (tree / "packs/robotics/entries/stale.json").write_bytes(b"{}\n")
    compile_tree(j, tree)  # recompiling replaces the gated dirs wholesale
    assert not (tree / "packs/robotics/entries/stale.json").exists()
    assert verify_reconstruction(j, tree) == []


def test_compile_and_verify_leave_ungated_pack_content_alone(tmp_path):
    """4.1: packs/<code>/ also holds manifest, corpus/, validation/, releases/,
    demo/. Compile must replace only entries/ and mappings/; the reconstruction
    test must neither flag that content as EXTRA nor require it."""
    j = full_journal(tmp_path / "journal.jsonl")
    tree = tmp_path / "tree"
    cat = tree / "packs/robotics/corpus/catalog.yaml"
    cat.parent.mkdir(parents=True)
    cat.write_bytes(b"schema: 1\n")
    (tree / "packs/robotics/manifest").write_bytes(b"code: robotics\n")
    compile_tree(j, tree)
    assert cat.read_bytes() == b"schema: 1\n"
    assert (tree / "packs/robotics/manifest").read_bytes() == b"code: robotics\n"
    assert verify_reconstruction(j, tree) == []
    # a pack dir that exists in the live tree with stale gated content is cleared
    stale = tree / "packs/orphan/entries/x.json"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"{}\n")
    assert any("orphan" in d for d in verify_reconstruction(j, tree))
    compile_tree(j, tree)
    assert not stale.exists()
    assert verify_reconstruction(j, tree) == []


# --- AR-rag4ds-016 F8: the second layer of the F1 remedy, pinned directly ---


def test_contained_resolve_branch_refuses_a_path_that_resolves_outside(tmp_path):
    """AR-rag4ds-016 F8 / O11: reach _contained's RESOLVE branch (the one after
    the parts test). Construction actually used: (1) a directory symlink
    (junction fallback on Windows) from inside the root to outside it, so a
    dot-free relative path resolves outside; (2) on Windows additionally a
    drive-less path like /abs/x.json, which is NOT absolute-by-parts there
    and resolves to the current drive's root. A '..' path is shown first as
    the contrast: it is caught by the parts test, not by resolve."""
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    rel = Path("sub") / ".." / "x.json"  # has '..' - caught by the parts test
    with pytest.raises(FoldError, match="not a plain relative path"):
        fold_mod._contained(root, rel, "t")
    link = root / "sub" / "out"
    target = tmp_path / "outside"
    target.mkdir()
    made = False
    try:
        link.symlink_to(target, target_is_directory=True)
        made = True
    except (OSError, NotImplementedError):
        import subprocess
        import sys
        if sys.platform == "win32":
            r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                               capture_output=True)
            made = r.returncode == 0
    if made:
        with pytest.raises(FoldError, match="escapes the tree root"):
            fold_mod._contained(root, Path("sub") / "out" / "x.json", "t")
    import sys
    if sys.platform == "win32":
        # a drive-less path is not absolute-by-parts on Windows but resolves
        # to the current drive's root - the resolve branch, with no symlink
        with pytest.raises(FoldError, match="escapes the tree root"):
            fold_mod._contained(root, Path("/abs/x.json"), "t")
    elif not made:
        pytest.skip("no symlink support on this platform/account")


def test_write_goes_through_contained_even_when_the_path_rule_is_subverted(tmp_path, monkeypatch):
    """Pin the wiring: if entry_tree_path / mapping_tree_path ever return an
    escaping path, the writer refuses and nothing lands anywhere."""
    with pytest.raises(FoldError, match="not a plain relative path"):
        fold_mod._write(tmp_path / "root", Path("../x.json"), b"", "t")
    assert list(tmp_path.rglob("*")) == []
    monkeypatch.setattr(fold_mod, "entry_tree_path", lambda e: Path("..") / "pwned.json")
    rec = approve_entry("approve", WHO, TS, "c", "s", ent(id="pwned"), {})
    with pytest.raises(FoldError, match="record 1: tree path '../pwned.json' is not a plain"):
        fold([rec], tmp_path / "deep" / "root")
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []
    monkeypatch.undo()
    monkeypatch.setattr(fold_mod, "mapping_tree_path", lambda r: Path("/abs/pwned.json"))
    rows = [{"entry": "x", "class": "c:X", "relation": "equivalent", "note": "n"}]
    mrec = approve_mapping("approve", WHO, TS, SHA, "s", {"standard": "cora", "rows": rows},
                           "robotics")
    # '/abs/...' is absolute-by-parts on POSIX and drive-less (resolve branch) on Windows
    with pytest.raises(FoldError, match="record 1: tree path .* (is not a plain|escapes)"):
        fold([mrec], tmp_path / "deep" / "root2")
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []
