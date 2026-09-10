"""S5 acceptance (plan 10.1): A0 complete - all three clauses (layer-filtered
irrelevant query -> NO_GROUNDING; broken backend -> ERROR; an unapproved
staged candidate never appears), the staged-candidate clause included;
FR6 semantics tests green; the grounding contract schema versioned and
documented. Plus index build/rebuild exactness, load-time integrity
(AR-rag4ds-018 F1), NFR2 run fields (F2), pins, and the pinned real model.

FAKE_BagOfWordsEmbedder is a test double (integrity charter section 8):
deterministic hashed bag-of-words vectors (4096 buckets) so relevance is
controllable (shared words -> positive cosine; disjoint words -> 0 unless
two distinct words collide, which the fixtures' word sets do not). It
never reaches a result; the real embedder is SentenceTransformerEmbedder.
"""

import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

from rag4ds.auditlog.journal import Journal
from rag4ds.gate.decisions import approve_entry, approve_mapping, reject, retract
from rag4ds.index.build import (
    CORE_FIELDS,
    MANIFEST_FILE,
    RUN_FIELDS,
    STRUCTURAL_FILE,
    VECTORS_FILE,
    IndexBuildError,
    SentenceTransformerEmbedder,
    build_index,
    load_index,
    tree_entries,
)
from rag4ds.kb.fold import FoldError, compile_tree
from rag4ds.model.entry import Entry, canonical_bytes
from rag4ds.retrieve.contract import (
    CONTRACT_SCHEMA_VERSION,
    CRISP_DM_PHASES,
    GroundingRequest,
    GroundingRequestError,
    GroundingResponse,
)
from rag4ds.retrieve.service import RetrievalError, Retriever
from rag4ds.staging.store import StagingStore

MODULE_ROOT = Path(__file__).resolve().parents[1]
TS = "2026-08-23 11:00 JST"
TS2 = "2026-08-23 11:30 JST"
COMMIT = "99e34e8"
WHO = "k-naruse"
SHA = "c" * 64
PINNED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PINNED_REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class FAKE_BagOfWordsEmbedder:  # noqa: N801 - the FAKE_ prefix is the point
    model_id = "FAKE-bag-of-words"
    revision = "test-double"
    dim = 4096

    def __init__(self, fail=False):
        self.fail = fail

    def embed(self, texts):
        if self.fail:
            raise RuntimeError("embedder backend down (test double)")
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace("-", " ").split():
                h = int(hashlib.sha256(w.encode()).hexdigest(), 16) % self.dim
                out[i, h] += 1.0
            if not out[i].any():
                out[i, 0] = 1.0
        return out


def ent(**o):
    f = dict(id="actuator-failure-modes", type="concept", layer="domain", pack="robotics",
             content="Bearing wear dominates actuator failure in rotary joints.",
             source_reference="corpus:r#1", version=1, status="current")
    f.update(o)
    return Entry(**f)


def build_tree(tmp_path):
    """A small gated tree: two robotics entries (one superseded to v2), one
    grocery entry, one general CRISP-DM entry, one task pattern, one mapping
    artifact, one reject, one retraction."""
    j = Journal(tmp_path / "journal.jsonl")
    j.append(approve_entry("approve", WHO, TS, "c", "s", ent(), {}))
    j.append(approve_entry("approve", WHO, TS, "c", "s",
                           ent(id="torque-signature", content="Torque signature precedes bearing "
                                                             "failure in rotary joints."), {}))
    j.append(approve_entry("approve", WHO, TS, "c", "s",
                           ent(id="demand-pattern", pack="grocery",
                               content="Weekly demand pattern of perishable groceries; "
                                       "remodeling the store floor."), {}))
    j.append(approve_entry("approve", WHO, TS, "c", "s",
                           ent(id="crisp-dm-modeling", layer="general", pack="shared",
                               content="CRISP-DM modeling phase: select technique, build model."),
                           {}))
    j.append(approve_entry("approve", WHO, TS, "c", "s",
                           ent(id="anomaly-detection", layer="task", pack="shared",
                               type="task-pattern",
                               content="Anomaly detection on multivariate time series flags "
                                       "rare events; evaluation by precision."), {}))
    j.append(approve_mapping("approve", WHO, TS, SHA, "s",
                             {"standard": "cora-ieee-1872", "rows": [
                                 {"entry": "actuator-failure-modes", "class": "cora:Failure",
                                  "relation": "narrower-than", "note": "n"}]}, "robotics"))
    j.append(reject("reject", WHO, TS, "Unapproved claim about actuator lubrication.", "s",
                    "unsupported"))
    known = {"actuator-failure-modes": 1, "torque-signature": 1, "demand-pattern": 1,
             "crisp-dm-modeling": 1, "anomaly-detection": 1}
    j.append(approve_entry("approve", WHO, TS, "c", "s",
                           ent(version=2, content="Bearing wear and misalignment dominate "
                                                  "actuator failure in rotary joints."),
                           known, supersedes={"id": "actuator-failure-modes", "version": 1}))
    known["actuator-failure-modes"] = 2
    j.append(retract(WHO, TS, "torque-signature", 1,
                     ent(id="torque-signature", status="retracted",
                         content="Torque signature precedes bearing failure in rotary joints."),
                     known))
    tree = tmp_path / "tree"
    compile_tree(j, tree)
    return j, tree


PACKS = {"robotics": "0.1.0", "grocery": "0.1.0"}


def bi(tree, build_dir, emb=None, packs=PACKS, floor=0.05, commit=COMMIT, ts=TS, journal=None):
    return build_index(tree, build_dir, emb or FAKE_BagOfWordsEmbedder(), packs, floor,
                       commit, ts, journal=journal)


def build(tmp_path, emb=None, min_relevance=0.05, packs=PACKS):
    j, tree = build_tree(tmp_path)
    emb = emb or FAKE_BagOfWordsEmbedder()
    manifest = bi(tree, tmp_path / "build", emb, packs, min_relevance, journal=j)
    return tree, manifest, Retriever(load_index(tmp_path / "build"), emb)


# --- build ---


def test_build_indexes_only_current_gated_entries_and_pins_the_model(tmp_path):
    tree, manifest, _ = build(tmp_path)
    ids = [e.id for e in tree_entries(tree)]
    assert ids == ["demand-pattern", "actuator-failure-modes", "crisp-dm-modeling",
                   "anomaly-detection"]  # sorted by (layer, pack, id); retracted absent
    assert manifest["entry_count"] == 4 and manifest["dimension"] == 4096
    assert manifest["embedding_model"] == {"id": "FAKE-bag-of-words", "revision": "test-double"}
    assert manifest["pack_versions"] == PACKS and manifest["min_relevance"] == 0.05
    assert len(manifest["index_version"]) == 16
    assert manifest["code_commit"] == COMMIT and manifest["timestamp"] == TS
    assert set(manifest) == set(CORE_FIELDS) | set(RUN_FIELDS) | {"index_version"}
    con = sqlite3.connect(tmp_path / "build" / STRUCTURAL_FILE)
    rows = con.execute("SELECT id, version, layer, pack FROM entries ORDER BY row").fetchall()
    con.close()
    assert ("actuator-failure-modes", 2, "domain", "robotics") in rows
    assert all(r[0] != "torque-signature" for r in rows)
    # no build scratch directory survives
    assert [p.name for p in tmp_path.iterdir() if ".building-" in p.name or ".old-" in p.name] == []


def test_rebuild_reproduces_the_core_and_index_version_while_run_fields_differ(tmp_path):
    """AR-rag4ds-018 F2: index_version digests the reproducible core only; the
    NFR2 run fields (code_commit, timestamp) sit outside it."""
    tree, m1, _ = build(tmp_path)
    b1 = tmp_path / "build"
    m2 = bi(tree, tmp_path / "build2", commit="abcdef0", ts=TS2)
    b2 = tmp_path / "build2"
    assert {k: m1[k] for k in CORE_FIELDS} == {k: m2[k] for k in CORE_FIELDS}
    assert m1["index_version"] == m2["index_version"]
    assert (m1["code_commit"], m1["timestamp"]) != (m2["code_commit"], m2["timestamp"])
    assert (b1 / MANIFEST_FILE).read_bytes() != (b2 / MANIFEST_FILE).read_bytes()
    assert (b1 / VECTORS_FILE).read_bytes() == (b2 / VECTORS_FILE).read_bytes()
    d1 = "\n".join(sqlite3.connect(b1 / STRUCTURAL_FILE).iterdump())
    d2 = "\n".join(sqlite3.connect(b2 / STRUCTURAL_FILE).iterdump())
    assert d1 == d2
    # a different tree -> different index_version
    (tree / "shared/general/crisp-dm-modeling.json").unlink()
    m3 = bi(tree, tmp_path / "build3")
    assert m3["index_version"] != m1["index_version"]


def test_run_fields_are_validated_like_the_ingest_manifest(tmp_path):
    _, tree = build_tree(tmp_path)
    with pytest.raises(IndexBuildError, match="code_commit"):
        bi(tree, tmp_path / "b", commit="not-hex")
    with pytest.raises(IndexBuildError, match="timestamp"):
        bi(tree, tmp_path / "b", ts="yesterday")
    assert not (tmp_path / "b").exists()


def test_build_refuses_unconfigured_or_missing_packs_and_bad_tree_files(tmp_path):
    _, tree = build_tree(tmp_path)
    with pytest.raises(IndexBuildError, match="not configured for this build"):
        bi(tree, tmp_path / "b", packs={"robotics": "0.1.0"})
    with pytest.raises(IndexBuildError, match="is not in the tree"):
        bi(tree, tmp_path / "b", packs=dict(PACKS, bio="1.0.0"))
    with pytest.raises(IndexBuildError, match="min_relevance"):
        bi(tree, tmp_path / "b", floor=2.0)
    (tree / "packs/robotics/entries/notes.txt").write_bytes(b"hello")
    with pytest.raises(IndexBuildError, match="not an entry file"):
        bi(tree, tmp_path / "b")
    (tree / "packs/robotics/entries/notes.txt").unlink()
    f = tree / "packs/robotics/entries/actuator-failure-modes.json"
    f.write_bytes(f.read_bytes().replace(b'"robotics"', b'"grocery"'))
    with pytest.raises(IndexBuildError, match="disagree with its directory|not the entry"):
        bi(tree, tmp_path / "b")


def test_f4_pack_presence_is_read_from_the_filesystem(tmp_path):
    """AR-rag4ds-018 F4: an entry-less pack directory is present and must be
    configured; a stray file or non-pack directory under packs/ is refused."""
    _, tree = build_tree(tmp_path)
    (tree / "packs" / "bio" / "entries").mkdir(parents=True)
    with pytest.raises(IndexBuildError, match=r"not configured.*\['bio'\]"):
        bi(tree, tmp_path / "b")
    bi(tree, tmp_path / "b", packs=dict(PACKS, bio="0.0.1"))  # configured: fine
    (tree / "packs" / "rogue.json").write_bytes(b"{}\n")
    with pytest.raises(IndexBuildError, match="packs/ holds packs only"):
        bi(tree, tmp_path / "b2", packs=dict(PACKS, bio="0.0.1"))
    (tree / "packs" / "rogue.json").unlink()
    (tree / "packs" / "Bad Name").mkdir()
    with pytest.raises(IndexBuildError, match="not a pack code directory"):
        bi(tree, tmp_path / "b2", packs=dict(PACKS, bio="0.0.1"))


def test_build_with_journal_refuses_an_unreconstructable_tree(tmp_path):
    """AR-rag4ds-018 O1: pass the journal and the build asserts 6.5 first."""
    j, tree = build_tree(tmp_path)
    bi(tree, tmp_path / "b", journal=j)
    planted = ent(id="hand-planted", content="Ungated hand-planted claim.")
    (tree / "packs/robotics/entries/hand-planted.json").write_bytes(canonical_bytes(planted))
    with pytest.raises(FoldError, match="NOT reconstructable"):
        bi(tree, tmp_path / "b2", journal=j)
    assert not (tmp_path / "b2").exists()


# --- F1: load-time integrity and atomic build ---


def test_f1_mixed_generation_build_is_refused(tmp_path):
    """The crash-between-writes state: generation B's structural index over
    generation A's manifest and vectors."""
    tree, _, _ = build(tmp_path)
    f = tree / "packs/robotics/entries/actuator-failure-modes.json"
    e2 = ent(version=2, content="Coolant loss dominates actuator failure in linear joints.")
    f.write_bytes(canonical_bytes(e2))
    bi(tree, tmp_path / "gen-b")
    shutil.copy(tmp_path / "gen-b" / STRUCTURAL_FILE, tmp_path / "build" / STRUCTURAL_FILE)
    with pytest.raises(IndexBuildError, match="tree_sha256|stored sha256"):
        load_index(tmp_path / "build")


def test_f1_mutated_row_and_swapped_vectors_are_refused(tmp_path):
    _, _, _ = build(tmp_path)
    b = tmp_path / "build"
    con = sqlite3.connect(b / STRUCTURAL_FILE)
    con.execute("UPDATE entries SET content='Lubricant brand X prevents all actuator failure.', "
                "source_reference='corpus:FORGED#1' WHERE id='actuator-failure-modes'")
    con.commit()
    con.close()
    with pytest.raises(IndexBuildError, match="does not match its stored sha256"):
        load_index(b)
    # restore the structural index from a fresh build, then swap the vectors
    tree = tmp_path / "tree"
    bi(tree, tmp_path / "b2")
    shutil.copy(tmp_path / "b2" / STRUCTURAL_FILE, b / STRUCTURAL_FILE)
    load_index(b)  # healthy again
    vecs = np.load(b / VECTORS_FILE)
    np.save(b / VECTORS_FILE, np.roll(vecs, 1, axis=0), allow_pickle=False)
    with pytest.raises(IndexBuildError, match="vectors_sha256"):
        load_index(b)


def test_f1_tampered_manifest_core_is_refused(tmp_path):
    _, _, _ = build(tmp_path)
    mp = tmp_path / "build" / MANIFEST_FILE
    m = json.loads(mp.read_bytes())
    m["min_relevance"] = 0.9
    mp.write_bytes((json.dumps(m, sort_keys=True, separators=(",", ":")) + "\n").encode())
    with pytest.raises(IndexBuildError, match="index_version does not match"):
        load_index(tmp_path / "build")


def test_f7_missing_parts_are_refused_without_side_effects(tmp_path):
    _, _, _ = build(tmp_path)
    b = tmp_path / "build"
    (b / STRUCTURAL_FILE).unlink()
    with pytest.raises(IndexBuildError, match="structural.sqlite is missing"):
        load_index(b)
    assert not (b / STRUCTURAL_FILE).exists()  # no empty database created by the read
    bi(tmp_path / "tree", tmp_path / "b2")
    (tmp_path / "b2" / VECTORS_FILE).unlink()
    with pytest.raises(IndexBuildError, match="vectors.npy is missing"):
        load_index(tmp_path / "b2")


def test_build_is_atomic_a_failing_embed_leaves_the_previous_build_intact(tmp_path):
    tree, m1, _ = build(tmp_path)
    with pytest.raises(RuntimeError, match="backend down"):
        bi(tree, tmp_path / "build", emb=FAKE_BagOfWordsEmbedder(fail=True))
    assert load_index(tmp_path / "build").manifest == m1
    assert [p.name for p in tmp_path.iterdir() if ".building-" in p.name] == []


def test_retriever_refuses_mismatched_embedder(tmp_path):
    class Other(FAKE_BagOfWordsEmbedder):
        revision = "other"

    build(tmp_path)
    with pytest.raises(RetrievalError, match="not the index's pinned model"):
        Retriever(load_index(tmp_path / "build"), Other())


# --- FR6 semantics ---


def test_ok_response_carries_scores_sources_status_and_echoes_meta(tmp_path):
    _, manifest, r = build(tmp_path)
    resp = r.ground(GroundingRequest("bearing wear actuator failure", domain="robotics", top_k=3))
    assert resp.status == "OK"
    assert resp.entries[0].id == "actuator-failure-modes" and resp.entries[0].version == 2
    assert resp.entries[0].source == "corpus:r#1" and resp.entries[0].relevance_score > 0
    assert resp.entries[0].status == "current"  # F6
    assert resp.meta["index_version"] == manifest["index_version"]
    assert resp.meta["index_build"] == {"code_commit": COMMIT, "timestamp": TS}
    assert resp.meta["pack_versions"] == PACKS
    assert resp.meta["embedding_model"]["id"].startswith("FAKE")
    d = resp.to_dict()
    assert d["schema_version"] == CONTRACT_SCHEMA_VERSION and d["entries"][0]["status"] == "current"
    assert "confidence" not in json.dumps(d).lower()
    assert resp.canonical_json().endswith(b"\n")


def test_domain_filter_keeps_shared_layers_and_layer_filter_excludes_them(tmp_path):
    _, _, r = build(tmp_path, min_relevance=-1.0)
    ids = lambda resp: {e.id for e in resp.entries}  # noqa: E731
    resp = r.ground(GroundingRequest("modeling anomaly demand bearing", domain="robotics",
                                     top_k=10))
    assert resp.status == "OK"
    assert ids(resp) == {"actuator-failure-modes", "crisp-dm-modeling", "anomaly-detection"}
    resp = r.ground(GroundingRequest("modeling anomaly demand bearing", domain="robotics",
                                     layer="domain", top_k=10))
    assert ids(resp) == {"actuator-failure-modes"}
    resp = r.ground(GroundingRequest("anything", layer="general", top_k=10))
    assert ids(resp) == {"crisp-dm-modeling"}
    resp = r.ground(GroundingRequest("anything", top_k=10))
    assert ids(resp) == {"actuator-failure-modes", "crisp-dm-modeling", "anomaly-detection",
                         "demand-pattern"}


def test_f5_unserved_domain_is_error_naming_served_packs(tmp_path):
    _, _, r = build(tmp_path)
    resp = r.ground(GroundingRequest("bearing wear", domain="robotcs"))
    assert resp.status == "ERROR" and "robotcs" in resp.meta["error"]
    assert "'grocery', 'robotics'" in resp.meta["error"]


def test_phase_filter_is_lexical_whole_word_over_id_and_content(tmp_path):
    _, _, r = build(tmp_path, min_relevance=-1.0)
    resp = r.ground(GroundingRequest("anything", phase="modeling", top_k=10))
    # F3: "remodeling the store floor" (demand-pattern) does NOT name the phase
    assert {e.id for e in resp.entries} == {"crisp-dm-modeling"}
    resp = r.ground(GroundingRequest("anything", phase="evaluation", top_k=10))
    assert {e.id for e in resp.entries} == {"anomaly-detection"}
    resp = r.ground(GroundingRequest("anything", phase="deployment", top_k=10))
    assert resp.status == "NO_GROUNDING"
    assert set(CRISP_DM_PHASES) == {"business-understanding", "data-understanding",
                                    "data-preparation", "modeling", "evaluation", "deployment"}


@pytest.mark.parametrize("content,phase,hit", [
    ("Remodeling the plant floor.", "modeling", False),
    ("A reevaluation of the plan.", "evaluation", False),
    ("Redeployment of the fleet.", "deployment", False),
    ("Data preparation comes before modeling.", "data-preparation", True),
    ("data_preparation step", "data-preparation", True),
    ("The MODELING phase.", "modeling", True),
    ("submodeling-ish", "modeling", False),
])
def test_f3_phase_matching_cases(content, phase, hit):
    from rag4ds.retrieve.service import _phase_matches
    assert _phase_matches(phase, {"id": "x", "content": content}) is hit


def test_top_k_and_deterministic_ranking_on_the_shown_scores(tmp_path):
    _, _, r = build(tmp_path, min_relevance=-1.0)
    a = r.ground(GroundingRequest("rotary joints failure", top_k=2))
    b = r.ground(GroundingRequest("rotary joints failure", top_k=2))
    assert a == b and len(a.entries) == 2
    assert a.entries[0].relevance_score >= a.entries[1].relevance_score
    assert all(round(e.relevance_score, 6) == e.relevance_score for e in a.entries)


# --- A0, all three clauses ---


def test_a0_layer_filtered_query_against_empty_or_irrelevant_pack_is_no_grounding(tmp_path):
    _, _, r = build(tmp_path)
    resp = r.ground(GroundingRequest("actuator bearing wear", domain="grocery", layer="domain"))
    assert resp.status == "NO_GROUNDING" and resp.entries == ()
    assert "relevance floor" in resp.meta["reason"] and resp.meta["candidates_considered"] == 1
    tree = tmp_path / "tree"
    (tree / "packs" / "bio" / "entries").mkdir(parents=True)
    bi(tree, tmp_path / "b2", packs=dict(PACKS, bio="0.0.1"))
    r2 = Retriever(load_index(tmp_path / "b2"), FAKE_BagOfWordsEmbedder())
    resp = r2.ground(GroundingRequest("actuator bearing wear", domain="bio", layer="domain"))
    assert resp.status == "NO_GROUNDING" and resp.meta["candidates_considered"] == 0
    empty = tmp_path / "empty-tree"
    (empty / "shared" / "general").mkdir(parents=True)
    bi(empty, tmp_path / "b3", packs={})
    r3 = Retriever(load_index(tmp_path / "b3"), FAKE_BagOfWordsEmbedder())
    assert r3.ground(GroundingRequest("anything")).status == "NO_GROUNDING"


def test_a0_broken_backend_is_error_never_no_grounding(tmp_path):
    build(tmp_path)
    r = Retriever(load_index(tmp_path / "build"), FAKE_BagOfWordsEmbedder(fail=True))
    resp = r.ground(GroundingRequest("actuator bearing wear"))
    assert resp.status == "ERROR" and resp.entries == ()
    assert "embedder backend down" in resp.meta["error"]


def test_a0_unapproved_staged_candidate_and_rejected_text_never_appear(tmp_path):
    _, _, r = build(tmp_path, min_relevance=-1.0)
    store = StagingStore(tmp_path / "stage.jsonl")
    store.file_candidate({
        "kind": "entry",
        "proposed": {"type": "concept",
                     "content": "Unapproved lubrication schedule prevents actuator failure.",
                     "source_reference": "corpus:r#77"},
        "layer": "domain", "pack": "robotics", "source_hash": "a" * 64,
        "span_start": 1, "span_end": 9, "extractor": "rule-v1", "run_manifest": "runs/x.jsonl:1",
    }, TS)
    resp = r.ground(GroundingRequest("lubrication schedule actuator", top_k=10))
    texts = " ".join(e.content for e in resp.entries)
    assert "lubrication" not in texts  # staged candidate unreachable (FR4)
    assert "Unapproved claim" not in texts  # rejected candidate unreachable (FR5)
    assert all(e.id != "torque-signature" for e in resp.entries)  # retracted gone (FR8)


# --- pins (component 7) ---


def test_pins_are_binding_and_copied(tmp_path):
    _, _, r = build(tmp_path)
    pins = {"robotics": "0.1.0"}
    req = GroundingRequest("bearing wear actuator", pins=pins)
    pins["shared"] = ""  # O5: a later mutation of the caller's dict must not leak in
    assert req.pins == {"robotics": "0.1.0"}
    assert r.ground(req).status == "OK"
    bad = r.ground(GroundingRequest("bearing wear actuator", pins={"robotics": "0.2.0"}))
    assert bad.status == "ERROR" and "not served" in bad.meta["error"]
    assert bad.meta["pack_versions"] == PACKS


# --- contract ---


@pytest.mark.parametrize("kw", [dict(query=""), dict(query="q", layer="core"),
                                dict(query="q", domain="shared"), dict(query="q", domain="A/B"),
                                dict(query="q", phase="training"), dict(query="q", top_k=0),
                                dict(query="q", top_k=True), dict(query="q", pins={"shared": "1"}),
                                dict(query="q", pins={"robotics": ""})])
def test_malformed_requests_are_caller_errors(kw):
    with pytest.raises(GroundingRequestError):
        GroundingRequest(**kw)


def test_response_invariants():
    with pytest.raises(ValueError, match="carries no entries"):
        GroundingResponse("NO_GROUNDING", entries=(1,))
    with pytest.raises(ValueError, match="at least one entry"):
        GroundingResponse("OK")
    with pytest.raises(ValueError, match="names its error"):
        GroundingResponse("ERROR")


def test_contract_is_documented_and_versioned():
    doc = (MODULE_ROOT / "docs" / "grounding_contract_v1.md").read_text(encoding="utf-8")
    assert f"Schema version: **{CONTRACT_SCHEMA_VERSION}**" in doc
    for field in ("query", "layer", "domain", "phase", "top_k", "pins", "relevance_score",
                  "NO_GROUNDING", "ERROR", "index_version", "status", "code_commit",
                  "index_build", "vectors_sha256"):
        assert field in doc


# --- the pinned real model (VT-3) ---


def test_pinned_sentence_transformer_embeds_normalized_and_deterministically():
    emb = SentenceTransformerEmbedder(PINNED_MODEL, PINNED_REV)
    a = emb.embed(["Bearing wear dominates actuator failure.", "Weekly demand of groceries."])
    b = emb.embed(["Bearing wear dominates actuator failure.", "Weekly demand of groceries."])
    assert a.shape == (2, 384) and a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(a, b)
    q = emb.embed(["actuator bearing failure"])[0]
    assert float(a[0] @ q) > float(a[1] @ q)  # the relevant sentence scores higher


def test_real_model_rebuild_is_byte_identical(tmp_path):
    """AR-rag4ds-018 O2: component 5's requirement under the suite - the
    vector index rebuilds reproducibly given the pinned model."""
    _, tree = build_tree(tmp_path)
    emb = SentenceTransformerEmbedder(PINNED_MODEL, PINNED_REV)
    m1 = bi(tree, tmp_path / "r1", emb=emb)
    m2 = bi(tree, tmp_path / "r2", emb=SentenceTransformerEmbedder(PINNED_MODEL, PINNED_REV))
    assert m1["index_version"] == m2["index_version"] and m1["dimension"] == 384
    v1 = (tmp_path / "r1" / VECTORS_FILE).read_bytes()
    assert v1 == (tmp_path / "r2" / VECTORS_FILE).read_bytes()
    r = Retriever(load_index(tmp_path / "r1"), emb)
    resp = r.ground(GroundingRequest("which torque patterns precede actuator failure",
                                     domain="robotics", top_k=2))
    assert resp.status == "OK" and resp.entries[0].id == "actuator-failure-modes"


# --- AR-rag4ds-018 r2: F8 and observations O8-O12 ---


# "?" is illegal in Windows file names, so that case runs on POSIX (CI) only
URI_NAMES = ["b#1", "b%20x", "b 1"] + ([] if sys.platform == "win32" else ["b?q"])


@pytest.mark.parametrize("name", URI_NAMES)
def test_f8_build_dirs_with_uri_special_characters_load_read_only(tmp_path, name):
    _, tree = build_tree(tmp_path)
    bd = tmp_path / name
    bi(tree, bd)
    before = sorted(p.name for p in tmp_path.iterdir())
    idx = load_index(bd)
    assert idx.manifest["entry_count"] == 4
    assert sorted(p.name for p in tmp_path.iterdir()) == before  # no file created by the read
    # and the structural index is genuinely opened read-only: a write through the
    # same URI form is refused
    from urllib.parse import quote
    uri = "file:" + quote((bd / STRUCTURAL_FILE).resolve().as_posix(), safe="/:") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        con.execute("CREATE TABLE t (x)")
    con.close()


def test_f8_unreadable_structural_index_is_index_build_error(tmp_path):
    _, _, _ = build(tmp_path)
    (tmp_path / "build" / STRUCTURAL_FILE).write_bytes(b"not a database at all" * 100)
    with pytest.raises(IndexBuildError, match="not a readable structural index"):
        load_index(tmp_path / "build")


def test_o8_stale_scratch_dirs_are_reaped_by_the_next_build(tmp_path):
    _, tree = build_tree(tmp_path)
    stale = tmp_path / "build.building-stale"
    stale.mkdir()
    (stale / STRUCTURAL_FILE).write_bytes(b"x")
    bi(tree, tmp_path / "build")
    assert not stale.exists()


def test_o9_run_fields_are_validated_on_load(tmp_path):
    _, _, _ = build(tmp_path)
    mp = tmp_path / "build" / MANIFEST_FILE
    m = json.loads(mp.read_bytes())
    m["code_commit"] = "NOT A COMMIT"
    mp.write_bytes((json.dumps(m, sort_keys=True, separators=(",", ":")) + "\n").encode())
    with pytest.raises(IndexBuildError, match="code_commit"):
        load_index(tmp_path / "build")
    m["code_commit"] = COMMIT
    m["timestamp"] = "sometime"
    mp.write_bytes((json.dumps(m, sort_keys=True, separators=(",", ":")) + "\n").encode())
    with pytest.raises(IndexBuildError, match="timestamp"):
        load_index(tmp_path / "build")


def test_o10_grounded_entry_status_is_required_and_validated():
    from rag4ds.retrieve.contract import GroundedEntry
    with pytest.raises(TypeError):
        GroundedEntry("x", 1, "concept", "domain", "p", "c", 0.5, "s")
    with pytest.raises(ValueError, match="status"):
        GroundedEntry("x", 1, "concept", "domain", "p", "c", 0.5, "s", status="not-a-status")


def test_o12_request_pins_are_read_only_through_the_request():
    req = GroundingRequest("q", pins={"robotics": "0.1.0"})
    with pytest.raises(TypeError):
        req.pins["injected"] = "1"
    assert req.filters()["pins"] == {"robotics": "0.1.0"}


# --- AR-rag4ds-018 r3: F9 and O15 ---


def test_f9_scratch_reaper_matches_the_literal_prefix_not_a_glob(tmp_path):
    """Building into 'b[1]' must reap 'b[1].building-*' and leave 'b1.building-*'
    (another target's scratch) alone - the glob reading inverted both."""
    _, tree = build_tree(tmp_path)
    own_stale = tmp_path / "b[1].building-stale"
    other = tmp_path / "b1.building-victim"
    own_stale.mkdir()
    other.mkdir()
    (other / "keep").write_bytes(b"x")
    bi(tree, tmp_path / "b[1]")
    assert not own_stale.exists()
    assert (other / "keep").exists()
    assert load_index(tmp_path / "b[1]").manifest["entry_count"] == 4


def test_o15_sqlite_uri_builder_handles_unc_and_special_characters():
    from rag4ds.index.build import _sqlite_ro_uri
    if sys.platform == "win32":
        assert _sqlite_ro_uri(Path(r"\\server\share\b#1\structural.sqlite")) == (
            "file:////server/share/b%231/structural.sqlite?mode=ro"
        )
        win = _sqlite_ro_uri(Path(r"C:\x y\b%1\s.sqlite"))
        assert win == "file:C:/x%20y/b%251/s.sqlite?mode=ro"
    assert _sqlite_ro_uri(Path("/tmp/b#1/s.sqlite")).startswith("file:/tmp/b%231/s.sqlite?mode=ro")
