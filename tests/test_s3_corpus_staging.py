"""S3 acceptance (plan 10.1): state-machine tests green (all five states,
both GROUPED exits, both candidate kinds); an ingested document is
reconstructable from catalog plus manifest. Plus NFR6/FR7/quotation-bound
disciplines and fail-loud rules.
"""

import pytest
import yaml

from rag4ds.auditlog.journal import Journal
from rag4ds.corpus.catalog import (
    Catalog,
    CatalogError,
    ingest,
    reconstruct,
    sha256_of,
)
from rag4ds.staging.store import (
    StagingError,
    StagingStore,
    candidate_id,
    validate_candidate,
)

TS = "2026-08-22 19:30 JST"
DOC_HASH = "b" * 64


# ---------------- catalog + ingest ----------------


def make_raw(data_dir, name="actuator-handbook.txt", text="Actuators wear at the bearings.\n"):
    data_dir.mkdir(parents=True, exist_ok=True)
    p = data_dir / name
    p.write_text(text, encoding="utf-8")
    return p


def do_ingest(tmp_path, **overrides):
    data_dir = tmp_path / "data"
    raw = make_raw(data_dir)
    catalog = Catalog("robotics", tmp_path / "packs" / "robotics")
    runs = Journal(tmp_path / "runs" / "ingest.jsonl")
    (tmp_path / "runs").mkdir(exist_ok=True)
    kwargs = dict(
        catalog=catalog, raw_path=raw, data_dir=data_dir, license_="CC-BY-4.0",
        fetch_pointer="https://example.org/actuator-handbook.txt", redistributable=True,
        metadata={"title": "Actuator handbook", "year": 2024}, timestamp=TS,
        runs_journal=runs, code_commit="f38d364", config={"hasher": "sha256"},
    )
    kwargs.update(overrides)
    return catalog, runs, data_dir, raw, ingest(**kwargs)


def test_ingest_registers_hash_and_writes_manifest(tmp_path):
    catalog, runs, data_dir, raw, manifest = do_ingest(tmp_path)
    assert manifest["hash"] == sha256_of(raw)
    assert manifest["kind"] == "ingest" and manifest["pack"] == "robotics"
    assert manifest["code_commit"] == "f38d364" and manifest["config"] == {"hasher": "sha256"}
    assert runs.read_all() == [manifest]  # canonical JSON Line, read back equal
    doc = catalog.get(manifest["hash"])
    assert doc["license"] == "CC-BY-4.0" and doc["redistributable"] is True
    assert catalog.path.exists() and catalog.path.suffix == ".yaml"


def test_ingested_document_is_reconstructable_from_catalog_plus_manifest(tmp_path):
    # the S3 exit property
    catalog, runs, data_dir, raw, manifest = do_ingest(tmp_path)
    located = reconstruct(catalog, runs.read_all()[0], data_dir)
    assert located.resolve() == raw.resolve()
    raw.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(CatalogError, match="not the one catalogued"):
        reconstruct(catalog, manifest, data_dir)


def test_documents_are_immutable_and_never_reregistered(tmp_path):
    catalog, runs, data_dir, raw, manifest = do_ingest(tmp_path)
    with pytest.raises(CatalogError, match="already registered"):
        ingest(catalog, raw, data_dir, "CC-BY-4.0", "https://example.org/x", True, {}, TS, runs,
               "f38d364", {})
    assert len(runs.read_all()) == 1  # the refused retry wrote no second manifest


def test_manifest_precedes_catalog_and_carries_provenance(tmp_path, monkeypatch):
    # AR-rag4ds-014 F4/F5: the manifest carries code commit + config, and a
    # failure at catalog write leaves no catalogued-but-unmanifested document.
    data_dir = tmp_path / "data"
    raw = make_raw(data_dir)
    catalog = Catalog("robotics", tmp_path / "packs" / "robotics")
    runs = Journal(tmp_path / "ingest.jsonl")
    import rag4ds.corpus.catalog as mod

    def boom(self, data):
        raise OSError("disk full")
    monkeypatch.setattr(mod.Catalog, "_save", boom)
    with pytest.raises(OSError):
        ingest(catalog, raw, data_dir, "MIT", "ptr", False, {}, TS, runs, "abcdef1", {"k": 1})
    monkeypatch.undo()
    assert len(runs.read_all()) == 1  # manifest written
    assert catalog.load()["documents"] == []  # nothing catalogued without it
    with pytest.raises(CatalogError, match="code_commit"):
        ingest(catalog, raw, data_dir, "MIT", "ptr", False, {}, TS, runs, "not-hex", {})


def test_reconstruct_refuses_escaping_local_path(tmp_path):
    catalog, runs, data_dir, raw, manifest = do_ingest(tmp_path)
    bad = dict(manifest, local_path="../elsewhere.txt")
    with pytest.raises(CatalogError, match="relative path inside"):
        reconstruct(catalog, bad, data_dir)
    absolute = dict(manifest, local_path=str(raw.resolve()))
    with pytest.raises(CatalogError, match="relative path inside"):
        reconstruct(catalog, absolute, data_dir)


def test_raw_document_must_live_in_the_data_directory(tmp_path):
    catalog = Catalog("robotics", tmp_path / "packs" / "robotics")
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("x\n", encoding="utf-8")
    runs = Journal(tmp_path / "ingest.jsonl")
    with pytest.raises(CatalogError, match="outside the data directory"):
        ingest(catalog, outside, tmp_path / "data", "MIT", "ptr", False, {}, TS, runs,
               "f38d364", {})


def test_catalog_yaml_is_deterministic_and_validated(tmp_path):
    catalog, *_ = do_ingest(tmp_path)
    first = catalog.path.read_bytes()
    catalog._save(catalog.load())
    assert catalog.path.read_bytes() == first  # sorted keys, stable dump
    text = yaml.safe_load(first.decode("utf-8"))
    assert text["pack"] == "robotics" and len(text["documents"]) == 1
    catalog.path.write_text("schema: 1\npack: other\ndocuments: []\n", encoding="utf-8")
    with pytest.raises(CatalogError, match="not a schema-1 catalog for pack"):
        catalog.load()
    catalog.path.write_text(": not: yaml: [", encoding="utf-8")
    with pytest.raises(CatalogError, match="not readable YAML"):
        catalog.load()


def test_catalog_document_fields_are_closed_and_mandatory(tmp_path):
    catalog = Catalog("robotics", tmp_path / "packs" / "robotics")
    with pytest.raises(CatalogError, match="missing mandatory field"):
        catalog.register({"hash": DOC_HASH})
    with pytest.raises(CatalogError, match="unknown field"):
        catalog.register({"hash": DOC_HASH, "license": "MIT", "fetch_pointer": "p",
                          "redistributable": False, "metadata": {}, "text": "copied!"})
    with pytest.raises(CatalogError, match="license is mandatory"):
        catalog.register({"hash": DOC_HASH, "license": " ", "fetch_pointer": "p",
                          "redistributable": False, "metadata": {}})


def test_catalog_pack_code_is_validated(tmp_path):
    with pytest.raises(CatalogError, match="pack code"):
        Catalog("shared", tmp_path)
    with pytest.raises(CatalogError, match="pack code"):
        Catalog("Robotics", tmp_path)


# ---------------- staging: candidates ----------------


def entry_candidate(**overrides):
    c = {
        "kind": "entry",
        "proposed": {"type": "concept",
                     "content": "Bearing wear is a dominant actuator failure mode.",
                     "source_reference": "corpus:robotics/actuator-handbook#s3"},
        "layer": "domain", "pack": "robotics",
        "source_hash": DOC_HASH, "span_start": 120, "span_end": 188,
        "extractor": "rule-v1", "run_manifest": "runs/extract.jsonl:1",
    }
    c.update(overrides)
    return c


def mapping_candidate(**overrides):
    c = {"kind": "mapping-draft", "yaml_draft": "standard: cora-ieee-1872\nrows: []\n",
         "pack": "robotics", "standard": "cora-ieee-1872", "author": "k-naruse"}
    c.update(overrides)
    return c


def test_both_candidate_kinds_file_and_read_back(tmp_path):
    store = StagingStore(tmp_path / "stage.jsonl")
    e = store.file_candidate(entry_candidate(), TS)
    m = store.file_candidate(mapping_candidate(), TS)
    cands = store.candidates()
    assert cands[e]["candidate"] == entry_candidate()
    assert cands[m]["candidate"] == mapping_candidate()
    assert store.state_of(e) == "PENDING" and store.state_of(m) == "PENDING"


def test_candidate_is_immutable_and_id_is_deterministic(tmp_path):
    store = StagingStore(tmp_path / "stage.jsonl")
    cid = store.file_candidate(entry_candidate(), TS)
    assert cid == candidate_id(entry_candidate())
    with pytest.raises(StagingError, match="already filed"):
        store.file_candidate(entry_candidate(), TS)
    assert store.file_candidate(entry_candidate(span_end=200), TS) != cid


def test_source_span_is_hash_plus_offsets_never_text():
    with pytest.raises(StagingError, match="never text"):
        validate_candidate(entry_candidate(span_text="copied passage"))
    with pytest.raises(StagingError, match="offsets"):
        validate_candidate(entry_candidate(span_start=10, span_end=10))
    with pytest.raises(StagingError, match="sha256"):
        validate_candidate(entry_candidate(source_hash="nothex"))


def test_entry_candidate_rules():
    with pytest.raises(StagingError, match="assigned at the gate"):
        validate_candidate(entry_candidate(proposed={"type": "concept", "content": "c",
                                                     "source_reference": "s", "id": "x"}))
    with pytest.raises(StagingError, match="reserved"):
        validate_candidate(entry_candidate(pack="shared"))
    with pytest.raises(StagingError, match="must use pack 'shared'"):
        validate_candidate(entry_candidate(layer="general"))
    with pytest.raises(StagingError, match="pack code in the id charset"):
        validate_candidate(entry_candidate(pack="Robotics Pack!! / ../etc"))  # AR-014 F6
    with pytest.raises(StagingError, match="proposed.type"):
        validate_candidate(entry_candidate(proposed={"type": "idea", "content": "c",
                                                     "source_reference": "s"}))


def test_mapping_candidate_rules():
    with pytest.raises(StagingError, match="mappings belong to a pack"):
        validate_candidate(mapping_candidate(pack="shared"))
    with pytest.raises(StagingError, match="entry-id character rule"):
        validate_candidate(mapping_candidate(standard="CORA 1872"))
    with pytest.raises(StagingError, match="yaml_draft"):
        validate_candidate(mapping_candidate(yaml_draft="  "))


def test_unknown_kind_and_field_sets_fail_loudly():
    with pytest.raises(StagingError, match="kind"):
        validate_candidate({"kind": "note"})
    with pytest.raises(StagingError, match="missing mandatory"):
        validate_candidate({"kind": "mapping-draft", "pack": "robotics"})


# ---------------- staging: the 6.3 state machine ----------------


def test_pending_to_each_terminal_state(tmp_path):
    store = StagingStore(tmp_path / "stage.jsonl")
    for i, to in enumerate(("APPROVED", "REJECTED", "REJECTED-DUPLICATE")):
        cid = store.file_candidate(entry_candidate(span_start=i), TS)
        store.transition(cid, to, TS, "k-naruse", "journal.jsonl:7")
        assert store.state_of(cid) == to
        with pytest.raises(StagingError, match="not allowed"):
            store.transition(cid, "PENDING", TS, "k-naruse", "x")  # terminal is final


def test_grouped_exit_one_head_decided_closes_members_mechanically(tmp_path):
    store = StagingStore(tmp_path / "stage.jsonl")
    head = store.file_candidate(entry_candidate(), TS)
    m1 = store.file_candidate(entry_candidate(span_start=1), TS)
    m2 = store.file_candidate(entry_candidate(span_start=2), TS)
    store.transition(m1, "GROUPED", TS, "cluster", "near-duplicate", head=head)
    store.transition(m2, "GROUPED", TS, "cluster", "near-duplicate", head=head)
    with pytest.raises(StagingError, match="no decision yet"):
        store.close_cluster(head, TS)
    store.transition(head, "REJECTED", TS, "k-naruse", "journal.jsonl:9")  # ANY decision
    closed = store.close_cluster(head, TS)  # AR-014 F2: citation derived, not supplied
    assert sorted(closed) == sorted([m1, m2])
    assert store.state_of(m1) == store.state_of(m2) == "REJECTED-DUPLICATE"
    recs = [r for r in store.journal.read_all() if r.get("record") == "transition"
            and r["to"] == "REJECTED-DUPLICATE"]
    assert all(r["authority"] == "journal.jsonl:9" for r in recs)  # cites the head decision


def test_grouped_exit_two_detach_returns_to_pending_only_while_head_undecided(tmp_path):
    store = StagingStore(tmp_path / "stage.jsonl")
    head = store.file_candidate(entry_candidate(), TS)
    m = store.file_candidate(entry_candidate(span_start=1), TS)
    store.transition(m, "GROUPED", TS, "cluster", "near-duplicate", head=head)
    store.detach(m, TS, "k-naruse")
    assert store.state_of(m) == "PENDING"
    store.transition(m, "GROUPED", TS, "cluster", "near-duplicate", head=head)
    store.transition(head, "APPROVED", TS, "k-naruse", "journal.jsonl:10")
    with pytest.raises(StagingError, match="already decided"):
        store.detach(m, TS, "k-naruse")


def test_grouping_rules(tmp_path):
    store = StagingStore(tmp_path / "stage.jsonl")
    head = store.file_candidate(entry_candidate(), TS)
    m = store.file_candidate(entry_candidate(span_start=1), TS)
    with pytest.raises(StagingError, match="known head"):
        store.transition(m, "GROUPED", TS, "cluster", "dup")  # no head
    with pytest.raises(StagingError, match="other than itself"):
        store.transition(m, "GROUPED", TS, "cluster", "dup", head=m)
    store.transition(head, "REJECTED", TS, "k-naruse", "journal.jsonl:1")
    with pytest.raises(StagingError, match="already decided"):
        store.transition(m, "GROUPED", TS, "cluster", "dup", head=head)
    with pytest.raises(StagingError, match="belongs only to GROUPED"):
        store.transition(m, "REJECTED", TS, "k-naruse", "journal.jsonl:2", head=head)


def test_head_must_be_pending_no_chains_or_cycles(tmp_path):
    # AR-rag4ds-014 F1: A->B->C stranded A; a head must itself be PENDING.
    store = StagingStore(tmp_path / "stage.jsonl")
    a = store.file_candidate(entry_candidate(span_start=1), TS)
    b = store.file_candidate(entry_candidate(span_start=2), TS)
    c = store.file_candidate(entry_candidate(span_start=3), TS)
    store.transition(b, "GROUPED", TS, "cluster", "dup", head=c)
    with pytest.raises(StagingError, match="must be PENDING"):
        store.transition(a, "GROUPED", TS, "cluster", "dup", head=b)


def test_a_head_cannot_itself_be_grouped_either_order(tmp_path):
    # AR-rag4ds-014 r2 F1: the order the r1 test missed - group the member
    # first, then try to group its head under a third candidate.
    store = StagingStore(tmp_path / "stage.jsonl")
    a = store.file_candidate(entry_candidate(span_start=1), TS)
    b = store.file_candidate(entry_candidate(span_start=2), TS)
    c = store.file_candidate(entry_candidate(span_start=3), TS)
    store.transition(a, "GROUPED", TS, "cluster", "dup", head=b)  # b is now a head
    with pytest.raises(StagingError, match="head of a cluster"):
        store.transition(b, "GROUPED", TS, "cluster", "dup", head=c)
    # and the stranding scenario cannot be built: c decided, a still reachable
    store.transition(c, "REJECTED", TS, "k-naruse", "journal.jsonl:1")
    assert store.state_of(a) == "GROUPED" and store.group_head_of(a) == b
    store.detach(a, TS, "k-naruse")  # b is undecided, so a can still leave
    assert store.state_of(a) == "PENDING"


def test_mapping_candidates_travel_the_state_machine_too(tmp_path):
    # AR-rag4ds-014 F3: both candidate kinds through every exit.
    store = StagingStore(tmp_path / "stage.jsonl")
    head = store.file_candidate(mapping_candidate(), TS)
    m = store.file_candidate(mapping_candidate(author="other"), TS)
    store.transition(m, "GROUPED", TS, "cluster", "dup", head=head)
    store.detach(m, TS, "k-naruse")
    assert store.state_of(m) == "PENDING"
    store.transition(m, "GROUPED", TS, "cluster", "dup", head=head)
    store.transition(head, "APPROVED", TS, "k-naruse", "journal.jsonl:4")
    assert store.close_cluster(head, TS) == [m]
    assert store.state_of(m) == "REJECTED-DUPLICATE"
    r = store.file_candidate(mapping_candidate(author="third"), TS)
    store.transition(r, "REJECTED", TS, "k-naruse", "journal.jsonl:5")
    assert store.state_of(r) == "REJECTED"


def test_fold_refuses_forged_from_state(tmp_path):
    # observation: a transition line whose `from` contradicts the fold
    store = StagingStore(tmp_path / "stage.jsonl")
    cid = store.file_candidate(entry_candidate(), TS)
    store.journal.append({"schema": 1, "record": "transition", "id": cid, "from": "GROUPED",
                          "to": "PENDING", "timestamp": TS, "actor": "x", "authority": "forged"})
    with pytest.raises(StagingError, match="forged or disordered"):
        store.state_of(cid)


def test_source_hash_is_checked_against_known_hashes_when_given(tmp_path):
    # observation: FR7's first link, assigned to the filer
    store = StagingStore(tmp_path / "stage.jsonl")
    with pytest.raises(StagingError, match="not a catalogued document"):
        store.file_candidate(entry_candidate(), TS, known_hashes={"c" * 64})
    assert store.file_candidate(entry_candidate(), TS, known_hashes={DOC_HASH})
    with pytest.raises(StagingError, match="known_hashes must be"):
        store.file_candidate(entry_candidate(span_start=9), TS, known_hashes=DOC_HASH)


def test_rejected_history_stays_forever_and_store_is_append_only(tmp_path):
    store = StagingStore(tmp_path / "stage.jsonl")
    cid = store.file_candidate(entry_candidate(), TS)
    store.transition(cid, "REJECTED", TS, "k-naruse", "journal.jsonl:3")
    before = store.journal.path.read_bytes()
    assert not hasattr(store, "delete") and not hasattr(store, "remove")
    assert cid in store.candidates() and store.state_of(cid) == "REJECTED"
    store.file_candidate(entry_candidate(span_start=5), TS)
    assert store.journal.path.read_bytes().startswith(before)  # pure append


def test_unknown_candidate_and_bad_transition_inputs_fail_loudly(tmp_path):
    store = StagingStore(tmp_path / "stage.jsonl")
    with pytest.raises(StagingError, match="unknown candidate"):
        store.state_of("cand-0000000000000000")
    cid = store.file_candidate(entry_candidate(), TS)
    with pytest.raises(StagingError, match="not one of"):
        store.transition(cid, "LIMBO", TS, "a", "b")
    with pytest.raises(StagingError, match="timestamp"):
        store.transition(cid, "REJECTED", "now", "a", "b")
    with pytest.raises(StagingError, match="non-blank"):
        store.transition(cid, "REJECTED", TS, " ", "b")


def test_reconstruct_refuses_non_ingest_or_malformed_manifest(tmp_path):
    """AR-rag4ds-015 O1: the runs journal is heterogeneous; reconstruct must
    refuse a non-ingest record or a manifest missing its fields with
    CatalogError, never a bare KeyError."""
    catalog, runs, data_dir, raw, manifest = do_ingest(tmp_path)
    with pytest.raises(CatalogError, match="needs an ingest manifest"):
        reconstruct(catalog, {"kind": "extract", "hash": manifest["hash"]}, data_dir)
    with pytest.raises(CatalogError, match="needs a manifest object"):
        reconstruct(catalog, ["not", "a", "dict"], data_dir)
    broken = {k: v for k, v in manifest.items() if k != "local_path"}
    with pytest.raises(CatalogError, match="missing field"):
        reconstruct(catalog, broken, data_dir)
    assert reconstruct(catalog, manifest, data_dir) == raw.resolve()
