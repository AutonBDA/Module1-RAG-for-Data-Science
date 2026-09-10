"""S1 acceptance (plan 10.1): a round-trip test proves one byte form per
artifact, repeatably - plus the FR1 mandatory-field, entry-id, layer/pack,
and fail-loud rules.
"""

import json

import pytest

from rag4ds.model.entry import (
    SHARED_PACK,
    Entry,
    EntryValidationError,
    canonical_bytes,
    entry_from_dict,
    entry_filename,
    read_entry,
    write_entry,
)


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


# --- the S1 exit criterion: one byte form per artifact, repeatably ---


def test_round_trip_is_byte_identical_and_repeatable(tmp_path):
    entry = make_entry()
    first = write_entry(entry, tmp_path)
    round_tripped = read_entry(first)
    assert round_tripped == entry
    second = write_entry(round_tripped, tmp_path)
    assert first.read_bytes() == second.read_bytes()
    assert canonical_bytes(entry) == canonical_bytes(round_tripped)


def test_serialization_is_deterministic_regardless_of_field_order():
    d = make_entry().to_dict()
    reordered = dict(reversed(list(d.items())))
    assert canonical_bytes(entry_from_dict(reordered)) == canonical_bytes(entry_from_dict(d))


def test_canonical_form_properties():
    raw = canonical_bytes(make_entry(content="café forecasting rules"))
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b"\r" not in raw
    raw.decode("ascii")  # ASCII-escaped: must not raise
    assert b"\\u00e9" in raw
    parsed = json.loads(raw)
    assert list(parsed) == sorted(parsed)  # sorted keys
    assert b": " not in raw and b", " not in raw  # fixed separators


def test_filename_is_id_dot_json(tmp_path):
    entry = make_entry()
    assert entry_filename(entry) == "actuator-failure-modes.json"
    assert write_entry(entry, tmp_path).name == "actuator-failure-modes.json"


# --- fail-loud rules (FR1, I1) ---


def test_every_missing_mandatory_field_is_named():
    complete = make_entry().to_dict()
    for field in complete:
        partial = {k: v for k, v in complete.items() if k != field}
        with pytest.raises(
            EntryValidationError, match=rf"missing mandatory field\(s\): \[.*'{field}'.*\]"
        ):
            entry_from_dict(partial)


@pytest.mark.parametrize("bad", ["idea", "Concept", "task pattern", "", None])
def test_type_enum_is_closed(bad):
    with pytest.raises(EntryValidationError, match="type"):
        make_entry(type=bad)


@pytest.mark.parametrize("bad", ["shared", "Domain", "core", "", None])
def test_layer_enum_is_closed(bad):
    # "shared" is a reserved PACK value, never a layer (FR1/FR2) - the
    # confusable case AR-rag4ds-010 F3 named.
    with pytest.raises(EntryValidationError, match="layer"):
        make_entry(layer=bad)


@pytest.mark.parametrize("field", ["content", "source_reference"])
@pytest.mark.parametrize("bad", ["", "   ", "\n\t", None, 7])
def test_content_and_source_must_be_non_blank_strings(field, bad):
    with pytest.raises(EntryValidationError, match=field):
        make_entry(**{field: bad})


@pytest.mark.parametrize("bad", ["", None, 3])
def test_pack_must_be_non_empty_string(bad):
    with pytest.raises(EntryValidationError, match="pack"):
        make_entry(pack=bad)


def test_task_pattern_wire_spelling_is_pinned():
    raw = canonical_bytes(make_entry(layer="task", pack=SHARED_PACK, type="task-pattern"))
    assert b'"type":"task-pattern"' in raw


def test_unknown_field_is_rejected_by_name():
    d = make_entry().to_dict()
    d["confidence"] = 0.9
    with pytest.raises(EntryValidationError, match="confidence"):
        entry_from_dict(d)


@pytest.mark.parametrize(
    "bad_id",
    [
        "",
        "UPPER-case",
        "under_score",
        "space id",
        "a" * 65,
        "café",
        "dot.json",
        "trailing-newline\n",
        "a" * 64 + "\n",
        "\nleading",
    ],
)
def test_entry_id_rule_rejects(bad_id):
    with pytest.raises(EntryValidationError, match="entry-id rule"):
        make_entry(id=bad_id)


@pytest.mark.parametrize("good_id", ["a", "a" * 64, "actuator-failure-modes", "x9-y"])
def test_entry_id_rule_accepts(good_id):
    assert make_entry(id=good_id).id == good_id


def test_domain_layer_rejects_reserved_shared_pack():
    with pytest.raises(EntryValidationError, match="reserved"):
        make_entry(pack=SHARED_PACK)


@pytest.mark.parametrize("layer", ["general", "task"])
def test_general_and_task_layers_require_shared_pack(layer):
    with pytest.raises(EntryValidationError, match="shared"):
        make_entry(layer=layer, pack="robotics")
    assert make_entry(layer=layer, pack=SHARED_PACK).pack == SHARED_PACK


@pytest.mark.parametrize("bad", ["draft", "superseded", "", "CURRENT"])
def test_status_enum_is_closed(bad):
    with pytest.raises(EntryValidationError, match="status"):
        make_entry(status=bad)


@pytest.mark.parametrize("bad", [0, -1, "1", 1.5, True])
def test_version_must_be_integer_at_least_one(bad):
    with pytest.raises(EntryValidationError, match="version"):
        make_entry(version=bad)


def test_multiple_problems_reported_together():
    with pytest.raises(EntryValidationError) as excinfo:
        make_entry(id="BAD ID", status="draft", version=0)
    message = str(excinfo.value)
    assert "3 problem" in message
    assert "entry-id rule" in message
    assert "status 'draft'" in message
    assert "version must be an integer >= 1" in message


def test_read_rejects_non_canonical_file(tmp_path):
    entry = make_entry()
    pretty = tmp_path / entry_filename(entry)
    pretty.write_bytes(json.dumps(entry.to_dict(), indent=2).encode("ascii"))
    with pytest.raises(EntryValidationError, match="canonical"):
        read_entry(pretty)


def test_read_rejects_garbage_with_the_module_exception(tmp_path):
    bad = tmp_path / "garbage.json"
    bad.write_bytes(b"\xff\xfenot json")
    with pytest.raises(EntryValidationError, match="not readable"):
        read_entry(bad)
    notdict = tmp_path / "list.json"
    notdict.write_bytes(b'["a"]\n')
    with pytest.raises(EntryValidationError, match="JSON object"):
        read_entry(notdict)


def test_read_rejects_mismatched_filename(tmp_path):
    entry = make_entry()
    wrong = tmp_path / "other-name.json"
    wrong.write_bytes(canonical_bytes(entry))
    with pytest.raises(EntryValidationError, match="does not match entry id"):
        read_entry(wrong)


def test_unknown_and_missing_fields_reported_together():
    # AR-rag4ds-010 r2 F2 remainder: pins the entry_from_dict assembly site -
    # one call with BOTH an unknown and a missing field must name both.
    d = make_entry().to_dict()
    del d["status"]
    d["confidence"] = 0.9
    with pytest.raises(EntryValidationError) as excinfo:
        entry_from_dict(d)
    message = str(excinfo.value)
    assert "unknown field(s): ['confidence']" in message
    assert "missing mandatory field(s): ['status']" in message


@pytest.mark.parametrize("bad", [7, None, b"bytes-id", ["list"]])
def test_non_string_id_fails_loudly_with_the_module_exception(bad):
    # AR-rag4ds-010 r2 F5: the isinstance guard, pinned - a non-string id
    # must raise EntryValidationError, never a bare TypeError.
    with pytest.raises(EntryValidationError, match="entry-id rule"):
        make_entry(id=bad)


def test_read_wraps_json_parse_errors(tmp_path):
    # AR-rag4ds-010 r2 F5: the JSONDecodeError branch, pinned - valid ASCII
    # that is not JSON must refuse via the module exception with context.
    bad = tmp_path / "notjson.json"
    bad.write_bytes(b"{this is not json\n")
    with pytest.raises(EntryValidationError, match="not readable"):
        read_entry(bad)
