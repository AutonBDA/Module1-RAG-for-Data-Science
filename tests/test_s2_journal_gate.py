"""S2 acceptance (plan 10.1): decision-write tests green for every type
and both payload variants - plus the journal's append-only, atomicity,
and canonical-line disciplines (6.4, 6.5, section 8, FR5).
"""

import pytest

from rag4ds.auditlog.journal import Journal, JournalError, canonical_record_bytes
from rag4ds.gate.decisions import (
    DecisionError,
    approve_entry,
    approve_mapping,
    reject,
    retract,
)
from rag4ds.model.entry import Entry

TS = "2026-08-22 00:30 JST"
APPROVER = "k-naruse"
SHA = "a" * 64


def make_entry(**overrides):
    fields = dict(
        id="actuator-failure-modes",
        type="concept",
        layer="domain",
        pack="robotics",
        content="Common actuator failure modes: bearing wear, encoder drift.",
        source_reference="corpus:robotics/actuator-handbook#s3",
        version=1,
        status="current",
    )
    fields.update(overrides)
    return Entry(**fields)


def approve_kwargs(**overrides):
    kwargs = dict(
        kind="approve",
        approver=APPROVER,
        timestamp=TS,
        candidate_text="Actuators commonly fail by bearing wear or encoder drift.",
        source_reference="corpus:robotics/actuator-handbook#s3",
        entry=make_entry(),
        known_versions={},
    )
    kwargs.update(overrides)
    return kwargs


KNOWN = {"actuator-failure-modes": 1}


# --- the S2 exit criterion: every type and both payload variants write ---


@pytest.mark.parametrize("kind", ["approve", "edit-then-approve"])
def test_entry_approval_writes_and_reads_back(tmp_path, kind):
    journal = Journal(tmp_path / "journal.jsonl")
    record = approve_entry(**approve_kwargs(kind=kind))
    journal.append(record)
    assert journal.read_all() == [record]
    assert record["entry"] == make_entry().to_dict()  # complete FR1 artifact embedded


@pytest.mark.parametrize("kind", ["approve", "edit-then-approve"])
def test_mapping_approval_variant_writes_and_reads_back(tmp_path, kind):
    journal = Journal(tmp_path / "journal.jsonl")
    artifact = {"standard": "cora-ieee-1872", "rows": [
        {"entry": "actuator-failure-modes", "class": "cora:Failure", "relation": "narrower-than",
         "note": "maps the failure-mode concept"}]}
    record = approve_mapping(
        kind=kind, approver=APPROVER, timestamp=TS,
        candidate_content_hash=SHA,
        source_reference="corpus:robotics/cora-draft#v1",
        mapping_artifact=artifact, pack="robotics",
    )
    journal.append(record)
    assert journal.read_all() == [record]
    assert record["mapping_artifact"] == artifact
    assert record["pack"] == "robotics"  # the fold's directory half (F1)
    assert "candidate_text" not in record  # the hash stands in (4.5)


def test_reject_writes_and_reads_back(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    record = reject(
        kind="reject", approver=APPROVER, timestamp=TS,
        candidate_text="Motors are magic.", source_reference="corpus:robotics/x#1",
        reason="unsupported by the source span",
    )
    journal.append(record)
    assert journal.read_all() == [record]
    assert "entry" not in record  # no entry on rejection (FR5)


def test_reject_duplicate_cites_its_head(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    record = reject(
        kind="reject-duplicate", approver=APPROVER, timestamp=TS,
        candidate_text="Actuators fail via bearings.", source_reference="corpus:robotics/x#2",
        reason="duplicate of cluster head", head_decision="journal.jsonl:1",
    )
    journal.append(record)
    assert journal.read_all()[0]["head_decision"] == "journal.jsonl:1"
    with pytest.raises(DecisionError, match="head decision"):
        reject(
            kind="reject-duplicate", approver=APPROVER, timestamp=TS,
            candidate_text="x", source_reference="y", reason="dup",
        )
    with pytest.raises(DecisionError, match="belongs only to reject-duplicate"):
        reject(
            kind="reject", approver=APPROVER, timestamp=TS,
            candidate_text="x", source_reference="y", reason="r",
            head_decision="journal.jsonl:1",
        )


def test_retract_embeds_the_retracted_final_form(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    final = make_entry(status="retracted", version=3)
    record = retract(APPROVER, TS, "actuator-failure-modes", 3, final,
                     {"actuator-failure-modes": 3})
    journal.append(record)
    got = journal.read_all()[0]
    assert got["entry"]["status"] == "retracted"
    assert (got["target_id"], got["target_version"]) == ("actuator-failure-modes", 3)


# --- id assignment (FR5) ---


def test_fresh_approval_rejects_taken_id_case_insensitively():
    with pytest.raises(DecisionError, match="already taken"):
        approve_entry(**approve_kwargs(known_versions={"ACTUATOR-FAILURE-MODES": 1}))


def test_fresh_approval_must_start_at_version_one():
    with pytest.raises(DecisionError, match="version 1"):
        approve_entry(**approve_kwargs(entry=make_entry(version=2)))


def test_supersession_is_existing_id_at_version_plus_one():
    record = approve_entry(**approve_kwargs(
        entry=make_entry(version=2),
        known_versions=KNOWN,
        supersedes={"id": "actuator-failure-modes", "version": 1},
    ))
    assert record["supersedes"] == {"id": "actuator-failure-modes", "version": 1}


def test_supersession_wrong_version_or_unknown_target_fails():
    with pytest.raises(DecisionError, match="version\\+1"):
        approve_entry(**approve_kwargs(
            entry=make_entry(version=3),
            known_versions=KNOWN,
            supersedes={"id": "actuator-failure-modes", "version": 1},
        ))
    with pytest.raises(DecisionError, match="not a known id"):
        approve_entry(**approve_kwargs(
            entry=make_entry(version=2),
            known_versions={},
            supersedes={"id": "actuator-failure-modes", "version": 1},
        ))


def test_chain_integrity_refuses_conflicting_successors():
    # AR-rag4ds-013 F1 (external): once v2 is current, a SECOND v2
    # superseding v1 must be refused - only the current version can be
    # superseded.
    with pytest.raises(DecisionError, match="CURRENT version"):
        approve_entry(**approve_kwargs(
            entry=make_entry(version=2),
            known_versions={"actuator-failure-modes": 2},
            supersedes={"id": "actuator-failure-modes", "version": 1},
        ))


def test_chain_integrity_refuses_phantom_predecessor():
    # AR-rag4ds-013 F1 (external): a v8 superseding a never-recorded v7
    # (current is 3) must be refused.
    with pytest.raises(DecisionError, match="CURRENT version"):
        approve_entry(**approve_kwargs(
            entry=make_entry(version=8),
            known_versions={"actuator-failure-modes": 3},
            supersedes={"id": "actuator-failure-modes", "version": 7},
        ))


def test_retract_must_name_the_current_version():
    final = make_entry(status="retracted", version=2)
    with pytest.raises(DecisionError, match="CURRENT version"):
        retract(APPROVER, TS, "actuator-failure-modes", 2, final,
                {"actuator-failure-modes": 3})
    with pytest.raises(DecisionError, match="not a known id"):
        retract(APPROVER, TS, "actuator-failure-modes", 2, final, {})


def test_known_versions_shape_is_enforced():
    with pytest.raises(DecisionError, match="known_versions must be a dict"):
        approve_entry(**approve_kwargs(known_versions={"a-b"}))
    with pytest.raises(DecisionError, match="int >= 1"):
        approve_entry(**approve_kwargs(known_versions={"a-b": 0}))
    with pytest.raises(DecisionError, match="case-variant duplicates"):
        approve_entry(**approve_kwargs(known_versions={"a-b": 1, "A-B": 1}))


def test_conflict_flag_is_recorded_and_validated():
    record = approve_entry(**approve_kwargs(conflict_with="encoder-drift"))
    assert record["conflict_with"] == "encoder-drift"
    with pytest.raises(DecisionError, match="conflict_with"):
        approve_entry(**approve_kwargs(conflict_with="BAD ID"))


# --- fail-loud validation ---


def test_base_fields_validated_together():
    with pytest.raises(DecisionError, match=r"2 problem"):
        reject(kind="reject", approver="  ", timestamp="today",
               candidate_text="x", source_reference="y", reason="r")


@pytest.mark.parametrize("bad_ts", ["2026-08-22 00:30", "00:30 JST", "2026/08/22 00:30 JST", ""])
def test_timestamp_format_is_enforced(bad_ts):
    with pytest.raises(DecisionError, match="timestamp"):
        approve_entry(**approve_kwargs(timestamp=bad_ts))


GOOD_ROW = {"entry": "actuator-failure-modes", "class": "cora:Failure",
            "relation": "narrower-than", "note": "n"}


def mapping_kwargs(**overrides):
    kwargs = dict(
        kind="approve", approver=APPROVER, timestamp=TS,
        candidate_content_hash=SHA, source_reference="s",
        mapping_artifact={"standard": "cora-ieee-1872", "rows": [dict(GOOD_ROW)]},
        pack="robotics",
    )
    kwargs.update(overrides)
    return kwargs


def test_mapping_artifact_must_carry_its_standard_id():
    with pytest.raises(DecisionError, match="standard id"):
        approve_mapping(**mapping_kwargs(mapping_artifact={"rows": [dict(GOOD_ROW)]}))
    with pytest.raises(DecisionError, match="sha256"):
        approve_mapping(**mapping_kwargs(candidate_content_hash="nothex"))


def test_mapping_approval_requires_a_real_pack_code():
    # AR-rag4ds-012 F1: without a pack the fold cannot place the artifact.
    for bad in ("", "shared", "UPPER", None):
        with pytest.raises(DecisionError, match="pack"):
            approve_mapping(**mapping_kwargs(pack=bad))


def test_mapping_rows_shape_and_relation_enum_enforced():
    # AR-rag4ds-012 F4: 4.5's row shape and three-value relation enum.
    with pytest.raises(DecisionError, match="rows must be a non-empty list"):
        approve_mapping(**mapping_kwargs(
            mapping_artifact={"standard": "cora-ieee-1872", "rows": "not-a-list"}))
    with pytest.raises(DecisionError, match="relation"):
        approve_mapping(**mapping_kwargs(mapping_artifact={
            "standard": "cora-ieee-1872",
            "rows": [dict(GOOD_ROW, relation="invented-relation")]}))
    with pytest.raises(DecisionError, match="exactly"):
        approve_mapping(**mapping_kwargs(mapping_artifact={
            "standard": "cora-ieee-1872", "rows": [{"entry": "x"}]}))
    with pytest.raises(DecisionError, match="entry id"):
        approve_mapping(**mapping_kwargs(mapping_artifact={
            "standard": "cora-ieee-1872",
            "rows": [dict(GOOD_ROW, entry="BAD ID")]}))


def test_known_versions_must_be_a_dict_of_str_to_int():
    # AR-rag4ds-012 F2: a bare string iterated as characters and reported
    # every id unique.
    with pytest.raises(DecisionError, match="known_versions must be a dict"):
        approve_entry(**approve_kwargs(known_versions="actuator-failure-modes"))
    with pytest.raises(DecisionError, match="str -> int"):
        approve_entry(**approve_kwargs(known_versions={1: 1}))


def test_bool_versions_are_rejected_everywhere():
    # AR-rag4ds-012 F3: bool must not serialize as a version number.
    with pytest.raises(DecisionError, match="version"):
        approve_entry(**approve_kwargs(
            entry=make_entry(version=2),
            known_versions=KNOWN,
            supersedes={"id": "actuator-failure-modes", "version": True},
        ))
    with pytest.raises(DecisionError, match="target_version"):
        retract(APPROVER, TS, "actuator-failure-modes", True,
                make_entry(status="retracted", version=1), KNOWN)


def test_timestamp_must_be_a_real_calendar_time():
    # AR-rag4ds-012 F5: shape alone accepted 0000-99-99 99:99 JST.
    with pytest.raises(DecisionError, match="timestamp"):
        approve_entry(**approve_kwargs(timestamp="0000-99-99 99:99 JST"))
    with pytest.raises(DecisionError, match="timestamp"):
        approve_entry(**approve_kwargs(timestamp="2026-02-30 12:00 JST"))


def test_retract_final_form_must_match_and_be_retracted():
    with pytest.raises(DecisionError, match="status retracted"):
        retract(APPROVER, TS, "actuator-failure-modes", 1, make_entry(), KNOWN)
    with pytest.raises(DecisionError, match="does not match target id"):
        retract(APPROVER, TS, "other-id", 3, make_entry(status="retracted", version=3),
                {"other-id": 3})
    with pytest.raises(DecisionError, match="does not match target"):
        retract(APPROVER, TS, "actuator-failure-modes", 2,
                make_entry(status="retracted", version=3), {"actuator-failure-modes": 2})


# --- journal disciplines (6.4, 6.5, section 8) ---


def test_appends_grow_the_file_by_exactly_one_canonical_line(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    r1 = reject(kind="reject", approver=APPROVER, timestamp=TS,
                candidate_text="a", source_reference="b", reason="c")
    r2 = approve_entry(**approve_kwargs())
    journal.append(r1)
    first = journal.path.read_bytes()
    journal.append(r2)
    second = journal.path.read_bytes()
    assert second.startswith(first)  # pure append: prior bytes untouched
    assert second[len(first):] == canonical_record_bytes(r2)
    assert journal.read_all() == [r1, r2]


def test_torn_final_line_fails_the_read_loudly(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    journal.append(approve_entry(**approve_kwargs()))
    with open(journal.path, "ab") as f:
        f.write(b'{"torn":')
    with pytest.raises(JournalError, match="torn"):
        journal.read_all()


def test_non_canonical_line_fails_the_read_loudly(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    journal.path.write_bytes(b'{"b": 1, "a": 2}\n')  # spaces + unsorted keys
    with pytest.raises(JournalError, match="canonical"):
        journal.read_all()
    journal.path.write_bytes(b"not json at all\n")
    with pytest.raises(JournalError, match="not readable"):
        journal.read_all()
    journal.path.write_bytes(b'["a-list"]\n')
    with pytest.raises(JournalError, match="JSON object"):
        journal.read_all()


def test_unserializable_record_is_refused_before_touching_the_file(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    with pytest.raises(JournalError, match="serializable"):
        journal.append({"bad": object()})
    assert not journal.path.exists()


def test_append_is_one_write_then_fsync(tmp_path, monkeypatch):
    # AR-rag4ds-012 F7: the two mechanisms claim 3 rests on, pinned -
    # exactly one os.write of the full line, and an fsync on the same fd
    # after it, per append.
    import os as _os
    calls = []
    real_write, real_fsync = _os.write, _os.fsync
    monkeypatch.setattr(
        _os, "write",
        lambda fd, data: (calls.append(("write", len(data))), real_write(fd, data))[1],
    )
    monkeypatch.setattr(_os, "fsync", lambda fd: (calls.append(("fsync", None)), real_fsync(fd))[1])
    journal = Journal(tmp_path / "journal.jsonl")
    record = reject(kind="reject", approver=APPROVER, timestamp=TS,
                    candidate_text="a", source_reference="b", reason="c")
    journal.append(record)
    write_calls = [c for c in calls if c[0] == "write"]
    assert len(write_calls) == 1, calls
    assert write_calls[0][1] == len(canonical_record_bytes(record))
    assert calls[-1] == ("fsync", None), calls


def test_short_write_raises_and_is_never_presumed_recorded(tmp_path, monkeypatch):
    # AR-rag4ds-012 F6: a short write must fail loudly (6.6).
    import os as _os
    real_write = _os.write
    monkeypatch.setattr(_os, "write", lambda fd, data: real_write(fd, data[:5]))
    journal = Journal(tmp_path / "journal.jsonl")
    record = reject(kind="reject", approver=APPROVER, timestamp=TS,
                    candidate_text="a", source_reference="b", reason="c")
    with pytest.raises(JournalError, match="NOT recorded"):
        journal.append(record)
    with pytest.raises(JournalError, match="torn"):
        journal.read_all()


# --- AR-rag4ds-012 F7: one killing test per named surviving mutant ---


def test_approved_entry_must_be_status_current():
    # kills r1 M4 / r2 N21
    with pytest.raises(DecisionError, match="status current"):
        approve_entry(**approve_kwargs(entry=make_entry(status="retracted")))


def test_approval_kind_whitelist_is_enforced():
    # kills r1 M12 / r2 N22, both builders
    with pytest.raises(DecisionError, match="not one of"):
        approve_entry(**approve_kwargs(kind="retract"))
    with pytest.raises(DecisionError, match="not one of"):
        approve_mapping(**mapping_kwargs(kind="reject"))


def test_approval_candidate_text_is_mandatory():
    # kills r1 M20 / r2 N23
    with pytest.raises(DecisionError, match="candidate_text"):
        approve_entry(**approve_kwargs(candidate_text="   "))


def test_mapping_artifact_must_be_a_non_empty_object():
    # kills r1 M22 / r2 N24
    with pytest.raises(DecisionError, match="non-empty JSON object"):
        approve_mapping(**mapping_kwargs(mapping_artifact={}))
    with pytest.raises(DecisionError, match="non-empty JSON object"):
        approve_mapping(**mapping_kwargs(mapping_artifact=["not", "a", "dict"]))


def test_reject_reason_is_mandatory():
    # kills r1 M25 / r2 N25
    with pytest.raises(DecisionError, match="reason"):
        reject(kind="reject", approver=APPROVER, timestamp=TS,
               candidate_text="x", source_reference="y", reason="  ")


def test_reject_source_reference_is_mandatory():
    # kills r1 M26 / r2 N26
    with pytest.raises(DecisionError, match="source_reference"):
        reject(kind="reject", approver=APPROVER, timestamp=TS,
               candidate_text="x", source_reference="", reason="r")


def test_mapping_row_class_must_be_non_blank():
    # kills r2 N18
    with pytest.raises(DecisionError, match="class"):
        approve_mapping(**mapping_kwargs(mapping_artifact={
            "standard": "cora-ieee-1872",
            "rows": [dict(GOOD_ROW, **{"class": "   "})]}))


def test_mapping_row_note_must_be_a_string():
    # kills r2 N19
    with pytest.raises(DecisionError, match="note"):
        approve_mapping(**mapping_kwargs(mapping_artifact={
            "standard": "cora-ieee-1872",
            "rows": [dict(GOOD_ROW, note=5)]}))


def test_mapping_artifact_top_level_is_closed():
    # AR-rag4ds-012 F8: unknown top-level keys refused by name, like an
    # entry's unknown fields - including the r1 probe's stray relation.
    with pytest.raises(DecisionError, match="unknown top-level key"):
        approve_mapping(**mapping_kwargs(mapping_artifact={
            "standard": "cora-ieee-1872", "rows": [dict(GOOD_ROW)],
            "junk": 1}))
    with pytest.raises(DecisionError, match=r"\['relation', 'rowz'\]"):
        approve_mapping(**mapping_kwargs(mapping_artifact={
            "standard": "cora-ieee-1872", "rowz": [], "rows": [dict(GOOD_ROW)],
            "relation": "invented-relation"}))


@pytest.mark.skipif(not hasattr(__import__("os"), "O_DIRECTORY"),
                    reason="directory fsync is POSIX-only")
def test_first_record_fsyncs_the_directory(tmp_path, monkeypatch):
    # AR-rag4ds-013 O2: creating the file flushes the parent directory too.
    import os as _os
    fsyncs = []
    real_fsync = _os.fsync
    monkeypatch.setattr(_os, "fsync", lambda fd: (fsyncs.append(fd), real_fsync(fd))[1])
    journal = Journal(tmp_path / "journal.jsonl")
    record = reject(kind="reject", approver=APPROVER, timestamp=TS,
                    candidate_text="a", source_reference="b", reason="c")
    journal.append(record)
    assert len(fsyncs) == 2  # file, then directory, on creation
    journal.append(record)
    assert len(fsyncs) == 3  # file only afterwards


def test_empty_and_missing_journal_read_as_empty(tmp_path):
    journal = Journal(tmp_path / "missing.jsonl")
    assert journal.read_all() == []
    journal.path.write_bytes(b"")
    assert journal.read_all() == []
