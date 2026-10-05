"""S9 demo road, part C: tools/release_pack.py - the release checklist (plan
4.3), the release manifest with the gated-content hash, the validation
snapshot of the gate decisions behind the shipped content, and the pack
manifest's version and state. The checklist's items are injected where a test
is about what release() does with their verdicts; the real items are driven
too (AR-rag4ds-059 F2), with the expensive parts patched at module level - the
index build and the demo run, the NFR3 scan - so that each real item is shown
to be called and to refuse. The live tree's licences and snapshot are checked
read-only."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))
import release_pack as rp  # noqa: E402

TS = "2026-09-29 03:00:00 JST"
OK = {"gated": lambda: [], "licences": lambda: [], "lexicon": lambda: [],
      "demo": lambda: ([], {"index_version": "x", "exit": 0, "lines": ["run_demo: 3 of 3"]})}


def _tree(tmp_path):
    """The pieces a release reads, copied: the pack, the journal, the ingest runs."""
    for rel in ("packs/robotics/entries", "packs/robotics/mappings", "packs/robotics/corpus"):
        shutil.copytree(MODULE_ROOT / rel, tmp_path / rel)
    for rel in ("packs/robotics/manifest.yaml", "journal/decisions.jsonl", "runs/ingest.jsonl"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MODULE_ROOT / rel, tmp_path / rel)
    # the copy stands for a pack BEFORE its first release, whatever the live pack's
    # state now is (robotics 1.0.0 was released on 2026-09-29, record 69)
    import re
    man = tmp_path / "packs/robotics/manifest.yaml"
    text = re.sub(r"(?m)^version: .*$", "version: 0.0.0", man.read_text(encoding="utf-8"))
    man.write_text(re.sub(r"(?m)^state: .*$", "state: POPULATING", text), encoding="utf-8")
    return tmp_path


def _release(root, version="1.0.0", dry_run=False, checks=None):
    return rp.release(root, "robotics", version, dry_run, clock=lambda: TS, commit="abcdef0",
                      checks=OK if checks is None else checks)


def _written(root):
    return (root / "packs/robotics/releases").exists() or \
        (root / "packs/robotics/validation").exists()


def test_the_gated_content_hash_moves_with_any_byte_file_or_name(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text("{}", encoding="utf-8")
    b.write_text("[]", encoding="utf-8")
    h1, listing = rp.gated_content_hash(tmp_path, [a, b])
    assert [f["path"] for f in listing] == ["a.json", "b.json"]
    assert rp.gated_content_hash(tmp_path, [a, b])[0] == h1
    b.write_text("[ ]", encoding="utf-8")
    assert rp.gated_content_hash(tmp_path, [a, b])[0] != h1
    b.write_text("[]", encoding="utf-8")
    assert rp.gated_content_hash(tmp_path, [a])[0] != h1
    c = tmp_path / "c.json"
    b.rename(c)
    assert rp.gated_content_hash(tmp_path, [a, c])[0] != h1


def test_the_live_pack_cites_licensed_documents_and_every_shipped_file_has_its_approval():
    assert rp.check_licences(MODULE_ROOT, "robotics") == []
    files = rp.shipped_files(MODULE_ROOT, "robotics")
    lines = rp.snapshot(MODULE_ROOT, "robotics", files)
    journal = (MODULE_ROOT / "journal" / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(files) >= 1 and all(ln in journal for ln in lines)


def test_a_licence_missing_from_the_catalogue_is_a_failure(tmp_path):
    root = _tree(tmp_path)
    cat = root / "packs/robotics/corpus/catalog.yaml"
    import yaml
    data = yaml.safe_load(cat.read_text(encoding="utf-8"))
    for d in data["documents"]:
        d["license"] = ""
    cat.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert rp.check_licences(root, "robotics")


def test_a_release_writes_its_manifest_snapshot_and_version(tmp_path):
    root = _tree(tmp_path)
    m = _release(root)
    files = rp.shipped_files(root, "robotics")
    assert m["gated_content_sha256"] == rp.gated_content_hash(root, files)[0]
    rel = root / "packs/robotics/releases/robotics-1.0.0.json"
    out = json.loads(rel.read_text(encoding="utf-8"))
    assert out == json.loads(json.dumps(m)) and out["tag"] == "pack-robotics-v1.0.0"
    val = root / "packs/robotics/validation/1.0.0"
    snap = (val / "decisions.jsonl").read_bytes()
    assert (val / "SHA256SUMS").read_text(encoding="utf-8") == \
        f"{hashlib.sha256(snap).hexdigest()}  decisions.jsonl\n"
    assert out["validation_snapshot"]["decisions"] == len(files) == len(snap.splitlines())
    man = (root / "packs/robotics/manifest.yaml").read_text(encoding="utf-8")
    assert "\nversion: 1.0.0\n" in man and "\nstate: RELEASED\n" in man
    with pytest.raises(rp.ReleaseError, match="released before|not above"):
        _release(root)


@pytest.mark.parametrize("item", ["gated", "licences", "lexicon", "demo"])
def test_any_failing_checklist_item_refuses_and_writes_nothing(tmp_path, item):
    root = _tree(tmp_path)
    before = (root / "packs/robotics/manifest.yaml").read_bytes()
    checks = dict(OK)
    checks[item] = (lambda: (["demo flows: run_demo exit 1"], {})) if item == "demo" \
        else (lambda: ["broken"])
    with pytest.raises(rp.ReleaseError, match="checklist failed"):
        _release(root, checks=checks)
    assert not _written(root)
    assert (root / "packs/robotics/manifest.yaml").read_bytes() == before


def test_a_dry_run_and_a_bad_version_write_nothing(tmp_path):
    root = _tree(tmp_path)
    before = (root / "packs/robotics/manifest.yaml").read_bytes()
    assert _release(root, dry_run=True)["version"] == "1.0.0"
    for bad in ("1.0", "01.0.0", "0.0.0"):
        with pytest.raises(rp.ReleaseError):
            _release(root, version=bad)
    assert not _written(root)
    assert (root / "packs/robotics/manifest.yaml").read_bytes() == before


def test_a_shipped_file_without_its_approval_refuses(tmp_path):
    root = _tree(tmp_path)
    p = sorted((root / "packs/robotics/entries").glob("*.json"))[0]
    e = json.loads(p.read_text(encoding="utf-8"))
    e["version"] += 1
    p.write_text(json.dumps(e), encoding="utf-8")
    with pytest.raises(rp.ReleaseError, match="no approval in the journal ships"):
        _release(root)
    assert not _written(root)


# ----- AR-rag4ds-059 F2: the REAL checklist items, each called and each refusing -----

def _real_but(root, **real):
    """OK for every item except the named ones, which run for real (checks=None
    for them): release() must call them itself."""
    return {k: v for k, v in OK.items() if k not in real}


def test_the_real_demo_item_refuses_a_demo_run_that_does_not_pass(tmp_path, monkeypatch):
    import build_index
    import run_demo
    root = _tree(tmp_path)
    calls = []

    def fake_build(module_root, out_dir, min_relevance, embedder=None, stream=None, clock=None):
        calls.append("build")
        return 0

    def fake_demo(module_root, pack, index_dir, embedder=None, stream=None, **kw):
        calls.append("demo")
        print("FAIL     torque-before-failure: ...", file=stream)
        return 1

    monkeypatch.setattr(build_index, "run", fake_build)
    monkeypatch.setattr(run_demo, "execute", fake_demo)
    monkeypatch.setattr("rag4ds.index.build.load_index",
                        lambda p: type("I", (), {"manifest": {"index_version": "v"}})())
    with pytest.raises(rp.ReleaseError, match="run_demo exit 1"):
        _release(root, checks=_real_but(root, demo=True))
    assert calls == ["build", "demo"] and not _written(root)


def test_the_real_demo_item_refuses_an_index_that_does_not_build(tmp_path, monkeypatch):
    import build_index
    root = _tree(tmp_path)
    monkeypatch.setattr(build_index, "run", lambda *a, **k: 1)
    with pytest.raises(rp.ReleaseError, match="did not build"):
        _release(root, checks=_real_but(root, demo=True))
    assert not _written(root)


def test_the_real_lexicon_item_refuses_an_nfr3_hit_and_a_scan_that_cannot_run(tmp_path,
                                                                           monkeypatch):
    import nfr3_scan
    root = _tree(tmp_path)
    monkeypatch.setattr(nfr3_scan, "run", lambda r: (["shared/tasks/x.json:1: term"], ["n"]))
    with pytest.raises(rp.ReleaseError, match="NFR3: shared/tasks/x.json"):
        _release(root, checks=_real_but(root, lexicon=True))

    def broken(r):
        raise nfr3_scan.Nfr3Error("unreadable")

    monkeypatch.setattr(nfr3_scan, "run", broken)
    with pytest.raises(rp.ReleaseError, match="could not run"):
        _release(root, checks=_real_but(root, lexicon=True))
    assert not _written(root)


def test_the_real_lexicon_item_refuses_a_pack_with_no_lexicon(tmp_path, monkeypatch):
    import nfr3_scan
    root = _tree(tmp_path)
    monkeypatch.setattr(nfr3_scan, "run", lambda r: ([], ["n"]))
    man = root / "packs/robotics/manifest.yaml"
    text = man.read_text(encoding="utf-8")
    head, tail = text.split("lexicon:\n", 1)
    rest = "\n".join(x for x in tail.split("\n") if not x.startswith("  - "))
    man.write_text(head + "lexicon: []\n" + rest, encoding="utf-8")
    # the manifest loader refuses an empty lexicon before the item runs; the release
    # turns that into its own refusal, with nothing written
    with pytest.raises(rp.ReleaseError, match="does not validate|declares no lexicon"):
        _release(root, checks=_real_but(root, lexicon=True))
    assert not _written(root)


def test_the_real_gated_item_refuses_a_tree_that_is_not_the_fold(tmp_path):
    root = _tree(tmp_path)
    p = sorted((root / "packs/robotics/entries").glob("*.json"))[0]
    p.write_text(p.read_text(encoding="utf-8").replace('"current"', '"retired"'), encoding="utf-8")
    with pytest.raises(rp.ReleaseError, match="gated:"):
        _release(root, checks=_real_but(root, gated=True))
    assert not _written(root)


def test_a_release_refuses_uncommitted_gated_content(tmp_path, monkeypatch):
    """AR-rag4ds-059 O2: the manifest names a commit, so the content it hashes must be
    that commit's - a dirty packs/<pack>, shared/ or journal/ refuses the release."""
    root = _tree(tmp_path)
    monkeypatch.setattr(rp, "dirty_paths", lambda r, pack: ["packs/robotics/entries/x.json"])
    with pytest.raises(rp.ReleaseError, match="uncommitted"):
        rp.release(root, "robotics", "1.0.0", False, clock=lambda: TS, checks=OK)
    assert not _written(root)


# ----- AR-rag4ds-059 r2 F5 and O8: the git query itself, and the licences item's wiring -----

GIT_ID = ("-c", "user.email=t@t", "-c", "user.name=t")  # CI has no git identity of its own


def _git(repo, *args):
    import subprocess
    return subprocess.run(["git", *GIT_ID, *args], cwd=repo, capture_output=True, text=True,
                          check=True).stdout


def test_dirty_paths_reads_git_for_everything_a_release_reads_and_names(tmp_path):
    """dirty_paths against a real repository: nothing on a clean tree; a modified,
    staged or untracked file under packs/<pack>, shared/, journal/, runs/ingest.jsonl,
    src/ or tools/ is listed; another pack, other runs files and other folders are not."""
    repo = tmp_path / "repo"
    # every kind of file under packs/<pack> a release reads (AR-rag4ds-059 r3 O10): the
    # entries, the mapping, the manifest, the catalogue (the licences item) and the flows
    mine = ["packs/robotics/entries/a.json", "packs/robotics/mappings/m.json",
            "packs/robotics/manifest.yaml", "packs/robotics/corpus/catalog.yaml",
            "packs/robotics/demo/flows.yaml"]
    files = mine + ["packs/other/entries/b.json", "shared/tasks/c.json",
                    "journal/decisions.jsonl", "runs/ingest.jsonl", "runs/fold.jsonl", "src/m.py",
                    "tools/t.py", "docs/d.md"]
    for f in files:
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        (repo / f).write_text("x\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    assert rp.dirty_paths(repo, "robotics") == []
    for f in files:
        (repo / f).write_text("changed\n", encoding="utf-8")
    (repo / "shared/general").mkdir()
    (repo / "shared/general/new.json").write_text("{}\n", encoding="utf-8")
    _git(repo, "add", "src/m.py")  # staged counts too
    listed = sorted(rp.dirty_paths(repo, "robotics"))
    assert listed == sorted(mine + ["shared/tasks/c.json", "shared/general/new.json",
                                    "journal/decisions.jsonl", "runs/ingest.jsonl", "src/m.py",
                                    "tools/t.py"]), listed


def test_the_real_licences_item_refuses_and_writes_nothing(tmp_path):
    """release() calls check_licences itself when no licences item is injected: a
    catalogue that has lost its licences refuses the release, nothing written."""
    root = _tree(tmp_path)
    cat = root / "packs/robotics/corpus/catalog.yaml"
    import yaml
    data = yaml.safe_load(cat.read_text(encoding="utf-8"))
    for d in data["documents"]:
        d["license"] = ""
    cat.write_text(yaml.safe_dump(data), encoding="utf-8")
    before = (root / "packs/robotics/manifest.yaml").read_bytes()
    with pytest.raises(rp.ReleaseError, match="licences: .*no licence in the pack's catalogue"):
        _release(root, checks=_real_but(root, licences=True))
    assert not _written(root)
    assert (root / "packs/robotics/manifest.yaml").read_bytes() == before


def test_one_missing_licence_is_found_and_named_wherever_its_entry_sorts(tmp_path):
    """AR-rag4ds-059 r3 O11: blank the licence of ONE document only - the one cited by
    the last entry in file order - and the item names that entry and no other."""
    import yaml
    root = _tree(tmp_path)
    entries = sorted((root / "packs/robotics/entries").glob("*.json"))
    last = entries[-1]
    stem = json.loads(last.read_text(encoding="utf-8"))["source_reference"].split("/", 1)[1]
    import re
    ingest = [json.loads(x) for x in (root / "runs/ingest.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    h = [r["hash"] for r in ingest if r.get("kind") == "ingest" and r.get("pack") == "robotics"
         and Path(r["local_path"]).stem == stem]
    assert len(h) == 1, (stem, h)
    cat = root / "packs/robotics/corpus/catalog.yaml"
    data = yaml.safe_load(cat.read_text(encoding="utf-8"))
    hit = [d for d in data["documents"] if d["hash"] == h[0]]
    assert len(hit) == 1
    hit[0]["license"] = ""
    cat.write_text(yaml.safe_dump(data), encoding="utf-8")
    problems = rp.check_licences(root, "robotics")
    assert len(problems) == 1 and problems[0].startswith(last.name), problems
    assert re.search(r"no licence in the pack's catalogue", problems[0])
