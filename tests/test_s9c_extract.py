"""S9c (the robotics pack, part 2): the generic pack extraction run -
catalogued text documents through the S7 strategies into staging for the
pack's validator, one NFR2 manifest per document, candidates filed before
the manifest, a re-run filing nothing twice, a dry run filing nothing at
all; the reproducible index build for the gate; and the VT-3 embedding pin
living in code.
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from rag4ds.auditlog.journal import Journal
from rag4ds.index.build import PINNED_EMBEDDING_MODEL, PINNED_EMBEDDING_REVISION
from rag4ds.staging.store import StagingStore

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))
sys.path.insert(0, str(MODULE_ROOT / "tests"))
import build_index as build_index_tool  # noqa: E402
import conftest  # noqa: E402
import extract_pack  # noqa: E402

TEXT = (
    "Actuator Faults\n\n"
    "A harmonic drive is a strain wave gear used in robot joints.\n"
    "Backlash: lost motion in a geartrain when the direction reverses.\n"
    "The joint temperature rose steadily before the failure.\n"
    "Encoder faults refer to wrong shaft-angle readings from the joint sensor.\n"
    "| Joint | Torque limit | Temperature limit |\n"
    "|---|---|---|\n"
    "| Base | 150 Nm | 80 C |\n"
    "| Wrist | 28 Nm | 75 C |\n"
)


def _scratch(tmp_path, text=TEXT, manifest_extra=""):
    """A scratch module: a git repo with one pack, one catalogued text
    document under data/, its ingest manifest, and the shared/ dirs."""
    root = tmp_path / "mod"
    for d in ("packs/bio/corpus", "packs/bio/entries", "shared/general", "shared/tasks",
              "stage", "runs", "journal", "data/bio/text"):
        (root / d).mkdir(parents=True)
    (root / ".gitignore").write_text("data/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@x", "-c", "user.name=t", "commit", "-q",
                    "--allow-empty", "-m", "init"], cwd=root, check=True)
    (root / "packs" / "bio" / "manifest.yaml").write_text(
        "schema: 1\ncode: bio\nname: Bio\nversion: 0.1.0\nstate: POPULATING\n"
        "validator: {role: validator, designee: K. Naruse}\nlicense: Apache-2.0\n"
        "ethics: none\ndescription: scratch\nlexicon: [harmonic drive, backlash]\n"
        "min_core_schema_version: 1\n" + manifest_extra, encoding="utf-8")
    data = text.encode("utf-8")
    h = hashlib.sha256(data).hexdigest()
    (root / "data" / "bio" / "text" / "doc.txt").write_bytes(data)
    (root / "packs" / "bio" / "corpus" / "catalog.yaml").write_text(yaml.safe_dump({
        "schema": 1, "pack": "bio", "documents": [{
            "hash": h, "license": "CC BY 4.0", "fetch_pointer": "https://example.org/d",
            "redistributable": True, "metadata": {"title": "A scratch document"}}]}),
        encoding="utf-8")
    Journal(root / "runs" / "ingest.jsonl").append({
        "kind": "ingest", "hash": h, "local_path": "data/bio/text/doc.txt", "pack": "bio"})
    return root, h


FLAGS = {"definitions": True, "headings": False, "tables": True}


def CLOCK():
    """A fixed NFR2 timestamp: the tools refuse a non-JST machine clock, and the
    CI runner is UTC (the 911241b lesson)."""
    return "2026-09-14 12:00:00 JST"


def test_dry_run_files_nothing_and_a_real_run_files_candidates_before_the_manifest(tmp_path,
                                                                                  capsys):
    root, h = _scratch(tmp_path)
    assert extract_pack.run("bio", root, ["rule-v1"], FLAGS, 1, 0.5, dry_run=True,
                            clock=CLOCK) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "proposed" in out and not re.search(r"\d+ filed", out)
    assert re.search(r"would file [1-9]\d*, already filed 0", out), out  # O5: says what is new
    assert not (root / "stage" / "candidates.jsonl").exists()
    assert not (root / "runs" / "extraction.jsonl").exists()
    assert extract_pack.run("bio", root, ["rule-v1"], FLAGS, 1, 0.5, dry_run=False,
                            clock=CLOCK) == 0
    out = capsys.readouterr().out
    store = StagingStore(root / "stage" / "candidates.jsonl")
    cands = store.candidates()
    assert cands, out
    manifests = Journal(root / "runs" / "extraction.jsonl").read_all()
    assert len(manifests) == 1 and manifests[0]["kind"] == "extraction"
    m = manifests[0]
    assert m["source_hash"] == h and m["pack"] == "bio" and m["layer"] == "domain"
    assert m["config"]["tool"] == "tools/extract_pack.py" and m["config"]["rule_flags"] == FLAGS
    for rec in cands.values():
        c = rec["candidate"]
        assert c["pack"] == "bio" and c["layer"] == "domain" and c["source_hash"] == h
        assert c["run_manifest"] == m["run_manifest_ref"]
        assert c["extractor"] == "rule-v1"
    # every proposal was filed, except the fixture's own tables twins - a second
    # proposal on a span the run has just filed is withheld (AR-rag4ds-046 r1 F3)
    twins = m.get("span_check", {}).get("withheld", [])
    assert m["candidates_proposed"] == len(cands) + len(twins)
    assert all(w["decided_id"] in cands and w["decision"] == "PENDING" for w in twins), twins
    # the candidates were filed BEFORE the manifest (AR-rag4ds-026 F8): the
    # staging file's last modification precedes the journal's creation
    assert (root / "stage" / "candidates.jsonl").stat().st_mtime_ns <= \
        (root / "runs" / "extraction.jsonl").stat().st_mtime_ns
    # a re-run files nothing twice and appends NO second manifest (AR-rag4ds-030 O3);
    # a dry run over staged candidates says what a real run would file (O5)
    assert extract_pack.run("bio", root, ["rule-v1"], FLAGS, 1, 0.5, dry_run=True,
                            clock=CLOCK) == 0
    assert f"would file 0, already filed {len(cands)}" in capsys.readouterr().out
    assert extract_pack.run("bio", root, ["rule-v1"], FLAGS, 1, 0.5, dry_run=False,
                            clock=CLOCK) == 0
    out = capsys.readouterr().out
    assert f"filed 0, already filed {len(cands)}" in out
    assert "already on the journal (not repeated)" in out
    assert len(store.candidates()) == len(cands)
    assert len(Journal(root / "runs" / "extraction.jsonl").read_all()) == 1


def test_two_identical_proposals_from_one_document_file_once_and_the_run_completes(tmp_path,
                                                                                     capsys):
    """AR-rag4ds-030 r2 F9: a pipe table whose header repeats a column name
    and whose row repeats the value yields two byte-identical proposals at
    one span, so one candidate id twice; the run files it once, counts the
    twin as already filed, appends the manifest and exits 0."""
    text = ("Joint limits table.\n\n| Joint | Limit | Limit |\n|---|---|---|\n"
            "| Base | 5 Nm | 5 Nm |\n")
    root, _ = _scratch(tmp_path, text=text)
    assert extract_pack.run("bio", root, ["rule-v1"], FLAGS, 1, 0.5, False, clock=CLOCK) == 0
    out = capsys.readouterr().out
    assert "proposed 2" in out and "filed 1, already filed 1" in out, out
    assert len(StagingStore(root / "stage" / "candidates.jsonl").candidates()) == 1
    assert len(Journal(root / "runs" / "extraction.jsonl").read_all()) == 1


def test_a_suppressed_only_document_still_files_its_manifest_once(tmp_path, capsys):
    """AR-rag4ds-030 r2 F10: every proposal of this document is suppressed
    by the quotation bound, so nothing is filed - and the NFR2 manifest with
    the suppression detail (NFR9) is appended on the first run, and never
    again on a re-run (no repeated extraction id - r1 O3)."""
    long_words = ("pneumonoultramicroscopicsilicovolcanoconiosis "
                  "floccinaucinihilipilification antidisestablishmentarianism "
                  "hippopotomonstrosesquippedaliophobia supercalifragilisticexpialidocious "
                  "honorificabilitudinitatibus")
    text = f"Alpha: {long_words} here.\n\nBeta: {long_words} again.\n"
    root, _ = _scratch(tmp_path, text=text)
    assert extract_pack.run("bio", root, ["rule-v1"], FLAGS, 1, 0.5, False, clock=CLOCK) == 0
    out = capsys.readouterr().out
    assert "proposed 0" in out and "suppressed by the bound 2" in out, out
    assert not (root / "stage" / "candidates.jsonl").exists()
    manifests = Journal(root / "runs" / "extraction.jsonl").read_all()
    assert len(manifests) == 1 and manifests[0]["candidates_suppressed_for_bound"] == 2
    assert len(manifests[0]["suppressed"]) == 2 and manifests[0]["suppressed"][0]["reason"]
    assert extract_pack.run("bio", root, ["rule-v1"], FLAGS, 1, 0.5, False, clock=CLOCK) == 0
    assert "already on the journal (not repeated)" in capsys.readouterr().out
    assert len(Journal(root / "runs" / "extraction.jsonl").read_all()) == 1
    # the dry run's summary line carries the would-file total (O11)
    root2, _ = _scratch(tmp_path / "s")
    assert extract_pack.run("bio", root2, ["rule-v1"], FLAGS, 1, 0.5, True, clock=CLOCK) == 0
    assert re.search(r"documents, \d+ proposed, \d+ suppressed, would file [1-9]\d*,",
                     capsys.readouterr().out)


def test_an_empty_catalog_is_refused_and_an_empty_text_is_skipped(tmp_path, capsys):
    """AR-rag4ds-030 F3: the two reachable arms of load_documents."""
    root, h = _scratch(tmp_path)
    cat = root / "packs" / "bio" / "corpus" / "catalog.yaml"
    cat.write_text("schema: 1\npack: bio\ndocuments: []\n", encoding="utf-8")
    assert extract_pack.run("bio", root, ["rule-v1"], FLAGS, 1, 0.5, True, clock=CLOCK) == 1
    assert "has no catalogued document" in capsys.readouterr().err
    root2, _ = _scratch(tmp_path / "e", text="   \n\n  \n")
    assert extract_pack.run("bio", root2, ["rule-v1"], FLAGS, 1, 0.5, True, clock=CLOCK) == 0
    out = capsys.readouterr().out
    assert "empty text" in out and "0 documents" in out and "1 skipped" in out


def test_refusals_are_loud(tmp_path, capsys):
    root, h = _scratch(tmp_path)
    assert extract_pack.run("shared", root, ["rule-v1"], FLAGS, 1, 0.5, True, clock=CLOCK) == 2
    assert extract_pack.run("bio", root, ["magic-v9"], FLAGS, 1, 0.5, True, clock=CLOCK) == 1
    assert "unknown strategy 'magic-v9'" in capsys.readouterr().err
    assert extract_pack.run("bio", root, [], FLAGS, 1, 0.5, True, clock=CLOCK) == 1
    assert "no strategy requested" in capsys.readouterr().err
    # the local copy edited: not the catalogued document
    p = root / "data" / "bio" / "text" / "doc.txt"
    p.write_bytes(p.read_bytes() + b"x")
    assert extract_pack.run("bio", root, ["rule-v1"], FLAGS, 1, 0.5, True, clock=CLOCK) == 1
    assert "does not hash to the catalogued" in capsys.readouterr().err
    # the local copy absent
    p.unlink()
    assert extract_pack.run("bio", root, ["rule-v1"], FLAGS, 1, 0.5, True, clock=CLOCK) == 1
    assert "is absent" in capsys.readouterr().err
    # a binary document is skipped with its reason, not refused
    root2, _ = _scratch(tmp_path / "b", text="\x00\xff binary" + "\n" * 5)
    (root2 / "data" / "bio" / "text" / "doc.txt").write_bytes(b"PK\x03\x04\xff\xfe" * 20)
    cat = root2 / "packs" / "bio" / "corpus" / "catalog.yaml"
    data = yaml.safe_load(cat.read_text(encoding="utf-8"))
    data["documents"][0]["hash"] = hashlib.sha256(b"PK\x03\x04\xff\xfe" * 20).hexdigest()
    cat.write_text(yaml.safe_dump(data), encoding="utf-8")
    (root2 / "runs" / "ingest.jsonl").unlink()
    Journal(root2 / "runs" / "ingest.jsonl").append({
        "kind": "ingest", "hash": data["documents"][0]["hash"],
        "local_path": "data/bio/text/doc.txt", "pack": "bio"})
    assert extract_pack.run("bio", root2, ["rule-v1"], FLAGS, 1, 0.5, True, clock=CLOCK) == 0
    out = capsys.readouterr().out
    assert "skipped" in out and "not a UTF-8 text document" in out and "0 documents" in out


def test_embedding_pin_lives_in_code_and_the_copies_agree():
    assert PINNED_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert PINNED_EMBEDDING_REVISION == conftest.PINNED_REV
    pins = (MODULE_ROOT / "docs" / "vt3_pins.md").read_text(encoding="utf-8")
    assert PINNED_EMBEDDING_REVISION in pins


class _FakeEmbedder:
    model_id, revision = "FAKE_embedder", "FAKE_rev"

    def embed(self, texts):
        import numpy as np
        out = np.zeros((len(texts), 4), dtype="float32")
        for i, t in enumerate(texts):
            out[i, hash(t) % 4] = 1.0
        return out


def test_build_index_tool_configures_every_pack_at_its_manifest_version(tmp_path, capsys):
    root, _ = _scratch(tmp_path)
    assert build_index_tool.pack_versions_from_manifests(root) == {"bio": "0.1.0"}
    assert build_index_tool.run(root, tmp_path / "idx", 0.2, embedder=_FakeEmbedder(),
                                clock=CLOCK) == 0
    out = capsys.readouterr().out
    manifest = json.loads(out.splitlines()[0])
    assert manifest["pack_versions"] == {"bio": "0.1.0"} and manifest["entry_count"] == 0
    assert (tmp_path / "idx" / "manifest.json").is_file()
    # a pack without a valid manifest: refused, never built
    (root / "packs" / "bio" / "manifest.yaml").write_text("schema: 1\n", encoding="utf-8")
    assert build_index_tool.run(root, tmp_path / "idx2", 0.2, embedder=_FakeEmbedder(),
                                clock=CLOCK) == 1
    assert "REFUSED" in capsys.readouterr().err


def test_live_extraction_manifests_match_the_filed_candidates():
    """The robotics run on the record: every extraction manifest names a
    catalogued document, and every robotics entry candidate in staging cites
    an extraction manifest that exists. Skips by name until the run is filed."""
    runs = MODULE_ROOT / "runs" / "extraction.jsonl"
    if not runs.is_file():
        pytest.skip("no extraction run filed yet (S9c not run)")
    manifests = [m for m in Journal(runs).read_all() if m.get("pack") == "robotics"]
    assert manifests
    cat = yaml.safe_load((MODULE_ROOT / "packs/robotics/corpus/catalog.yaml").read_text("utf-8"))
    hashes = {d["hash"] for d in cat["documents"]}
    refs = {m["run_manifest_ref"] for m in manifests}
    assert all(m["source_hash"] in hashes for m in manifests)
    store = StagingStore(MODULE_ROOT / "stage" / "candidates.jsonl")
    ours = {cid: r["candidate"] for cid, r in store.candidates().items()
            if r["candidate"].get("kind") == "entry" and r["candidate"].get("pack") == "robotics"}
    assert ours
    # S9c part 3: a re-filed candidate cites the re-file manifest that names it
    # (runs/refile.jsonl), whose pairs point back at a rejected extraction candidate
    refile_runs = MODULE_ROOT / "runs" / "refile.jsonl"
    refiled = {}
    if refile_runs.is_file():
        for m in Journal(refile_runs).read_all():
            if m.get("kind") == "refile" and m.get("pack") == "robotics":
                for pair in m["pairs"]:
                    refiled[pair["to"]] = (m["run_manifest_ref"], pair["from"])
    # S9e part 8: the pack now also holds hand-authored candidates, filed by
    # tools/pack_author.py; each cites the authoring manifest that names it
    authored = {}
    author_runs = MODULE_ROOT / "runs" / "authoring.jsonl"
    if author_runs.is_file():
        for m in Journal(author_runs).read_all():
            # a robotics authoring run offers only this pack's catalogued documents
            if m.get("kind") == "authoring" and set(m["sources"].values()) <= hashes:
                for cand in m["candidates"]:
                    authored[cand] = f"authoring.jsonl#{m['run_id']}"
    for cid, c in ours.items():
        assert c["source_hash"] in hashes
        if c["extractor"].startswith("author:"):
            assert c["run_manifest"] == authored[cid], cid
            continue
        assert c["extractor"] in ("rule-v1", "embed-v1")
        if cid in refiled:
            ref, old = refiled[cid]
            assert c["run_manifest"] == ref and old in ours
            assert ours[old]["run_manifest"] in refs
            assert {k: v for k, v in ours[old].items() if k != "run_manifest"} == \
                {k: v for k, v in c.items() if k != "run_manifest"}
        else:
            assert c["run_manifest"] in refs


def test_the_tools_refuse_a_non_jst_machine_clock(tmp_path, monkeypatch, capsys):
    """The default clock is the machine's and must be JST; on a UTC runner
    the tools refuse rather than stamp a wrong timestamp (the repository's
    operating rules, section 5). The suite itself injects a fixed clock."""
    import types
    for mod in (extract_pack, build_index_tool):
        monkeypatch.setattr(mod.time, "localtime", lambda: types.SimpleNamespace(tm_gmtoff=0))
    root, _ = _scratch(tmp_path)
    assert extract_pack.run("bio", root, ["rule-v1"], FLAGS, 1, 0.5, True) == 1
    assert "local clock is not JST" in capsys.readouterr().err
    assert build_index_tool.run(root, tmp_path / "idx", 0.2, embedder=_FakeEmbedder()) == 1
    assert "local clock is not JST" in capsys.readouterr().err
