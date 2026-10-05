"""VT-3b (plan section 12, re-opened per the PI decision of 2026-08-23): a
spaCy language model is pinned - one version, one wheel hash - verified at
load against the ENVIRONMENT as well as the pin (AR-rag4ds-024 F2: the
refusal witnesses below shape fake distributions on sys.path, so every guard
whose trigger lives in the installed environment is reached), coherent with
the lockfile and the pins record, loadable offline, and demonstrably
unlocking POS / dependencies / noun chunks / NER (F3: a real entity is
asserted); and S7's rule strategy still splits with the blank sentencizer on
the measured basis that tools/splitter_measurement.py records (F1: the test
asserts the measurement, so a future model that changes it turns the suite
red and the decision is re-taken instead of a stale justification standing).
"""

import base64
import hashlib
import importlib
import importlib.util
import json
import re
from pathlib import Path

import pytest
import spacy

from rag4ds.extract import language_model as lm
from rag4ds.extract.strategies import RuleStrategy

MODULE_ROOT = Path(__file__).resolve().parents[1]
SYNTH = (
    "Robotics maintenance notes.\n"
    "Bearing wear is the dominant failure mode of rotary actuators.\n"
    "Torque signature: a rising torque trend that precedes bearing failure.\n"
    "Backlash refers to lost motion in a geartrain.\n"
    "The weather was fine and the team went home early.\n"
    "An encoder is a sensor that reports shaft angle.\n"
)


@pytest.fixture(scope="module")
def nlp():
    return lm.load_pinned_language_model()


def _measurement_module():
    spec = importlib.util.spec_from_file_location(
        "splitter_measurement", MODULE_ROOT / "tools" / "splitter_measurement.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the pin, verified against the pin AND the environment ---


def test_pinned_model_loads_offline_and_is_the_pin(nlp):
    pin = lm.PINNED_LANGUAGE_MODEL
    assert (nlp.meta["lang"], nlp.meta["name"], nlp.meta["version"]) == (
        pin["lang"], pin["name"], pin["version"])
    assert lm.installed_wheel_sha256() == pin["sha256"]
    assert lm.verify_installed_files() >= 1
    assert spacy.util.is_compatible_version(spacy.__version__, pin["spacy_version"])
    for component in ("tok2vec", "tagger", "parser", "ner", "lemmatizer"):
        assert component in nlp.pipe_names, nlp.pipe_names
    assert lm.describe(nlp) == {"id": "en_core_web_sm", "version": "3.8.0",
                                "sha256": pin["sha256"]}


def test_pin_is_read_only():
    with pytest.raises(TypeError):
        lm.PINNED_LANGUAGE_MODEL["version"] = "9.9.9"  # type: ignore[index]


def test_pin_is_coherent_with_the_lockfile_and_the_record():
    pin = lm.PINNED_LANGUAGE_MODEL
    lock = (MODULE_ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines()
    line = [ln for ln in lock if ln.startswith("en_core_web_sm @ ")]
    assert line == [f"en_core_web_sm @ {pin['url']}#sha256={pin['sha256']}"], line
    record = (MODULE_ROOT / "docs" / "vt3_pins.md").read_text(encoding="utf-8")
    assert pin["sha256"] in record and "en_core_web_sm 3.8.0" in record
    # pyproject carries the same direct reference, so the relock job resolves the same wheel
    pyproject = (MODULE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f"{pin['url']}#sha256={pin['sha256']}" in pyproject


def test_describe_refuses_a_pipeline_that_is_not_the_pin():
    blank = spacy.blank("en")
    with pytest.raises(lm.LanguageModelError, match="not the pin"):
        lm.describe(blank)


# --- refusal witnesses that mutate the PIN (the comparisons whose expected value is the pin) ---


def test_loader_refuses_a_wheel_hash_mismatch():
    pin = dict(lm.PINNED_LANGUAGE_MODEL, sha256="0" * 64)
    with pytest.raises(lm.LanguageModelError, match="sha256"):
        lm.load_pinned_language_model(pin)


def test_loader_refuses_a_version_mismatch():
    pin = dict(lm.PINNED_LANGUAGE_MODEL, version="9.9.9")
    with pytest.raises(lm.LanguageModelError, match="pipeline meta"):
        lm.load_pinned_language_model(pin)


def test_loader_refuses_a_declared_range_mismatch():
    pin = dict(lm.PINNED_LANGUAGE_MODEL, spacy_version=">=4.0.0,<5.0.0")
    with pytest.raises(lm.LanguageModelError, match="spacy_version"):
        lm.load_pinned_language_model(pin)


def test_loader_refuses_an_absent_package_loudly():
    pin = dict(lm.PINNED_LANGUAGE_MODEL, package="en_core_web_NOT_INSTALLED")
    with pytest.raises(lm.LanguageModelError, match="not installed"):
        lm.load_pinned_language_model(pin)


# --- refusal witnesses that shape the ENVIRONMENT (AR-rag4ds-024 F2, F4) ---


def _record_hash(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")


def _fake_dist(tmp_path, monkeypatch, name, *, direct_url=None, files=None,
               direct_url_as_directory=False, version="3.8.0", hashed=True):
    """A synthetic <name>-<version>.dist-info on sys.path, shaped like a pip
    install: METADATA, RECORD with sha256 digests (or, with hashed=False, a
    RECORD whose rows carry no digest at all), and the direct_url.json the
    case calls for (a dict, raw bytes, absent, or a directory that reads as
    'present but unreadable')."""
    site = tmp_path / "site"
    site.mkdir(exist_ok=True)
    info_rel = f"{name}-{version}.dist-info"
    info = site / info_rel
    info.mkdir()
    record = []

    def add(rel, data: bytes):
        path = site / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        record.append(f"{rel},sha256={_record_hash(data)},{len(data)}" if hashed
                      else f"{rel},,")

    add(f"{info_rel}/METADATA",
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode())
    if direct_url_as_directory:
        (info / "direct_url.json").mkdir()
        record.append(f"{info_rel}/direct_url.json,,")
    elif direct_url is not None:
        data = direct_url if isinstance(direct_url, bytes) else json.dumps(direct_url).encode()
        add(f"{info_rel}/direct_url.json", data)
    for rel, data in (files or {}).items():
        add(rel, data)
    record.append(f"{info_rel}/RECORD,,")
    (info / "RECORD").write_text("\n".join(record) + "\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()
    return site


def _pinned_direct_url(pin=lm.PINNED_LANGUAGE_MODEL):
    return {"url": pin["url"], "archive_info": {"hash": f"sha256={pin['sha256']}",
                                               "hashes": {"sha256": pin["sha256"]}}}


def test_refuses_a_distribution_installed_from_an_index(tmp_path, monkeypatch):
    """No direct_url.json at all - the shape of `pip install <name>` from an index."""
    _fake_dist(tmp_path, monkeypatch, "en_core_web_fakeindex")
    pin = dict(lm.PINNED_LANGUAGE_MODEL, package="en_core_web_fakeindex")
    with pytest.raises(lm.LanguageModelError, match="absent"):
        lm.load_pinned_language_model(pin)


def test_refuses_a_direct_url_without_a_hash(tmp_path, monkeypatch):
    """direct_url.json with no hashes - the shape of `spacy download` / a plain URL install."""
    _fake_dist(tmp_path, monkeypatch, "en_core_web_fakeroute",
               direct_url={"url": lm.PINNED_LANGUAGE_MODEL["url"], "archive_info": {}})
    pin = dict(lm.PINNED_LANGUAGE_MODEL, package="en_core_web_fakeroute")
    with pytest.raises(lm.LanguageModelError, match="no sha256"):
        lm.load_pinned_language_model(pin)


def test_refuses_a_corrupt_direct_url_json(tmp_path, monkeypatch):
    _fake_dist(tmp_path, monkeypatch, "en_core_web_fakecorrupt", direct_url=b"{not json")
    pin = dict(lm.PINNED_LANGUAGE_MODEL, package="en_core_web_fakecorrupt")
    with pytest.raises(lm.LanguageModelError, match="not valid JSON"):
        lm.load_pinned_language_model(pin)


def test_refuses_a_present_but_unreadable_direct_url_json(tmp_path, monkeypatch):
    """AR-rag4ds-024 O5: read_text swallows permission errors; the message must
    say 'present but unreadable', not misdiagnose the file as absent."""
    _fake_dist(tmp_path, monkeypatch, "en_core_web_fakeunread", direct_url_as_directory=True)
    pin = dict(lm.PINNED_LANGUAGE_MODEL, package="en_core_web_fakeunread")
    with pytest.raises(lm.LanguageModelError, match="present but unreadable"):
        lm.load_pinned_language_model(pin)


def test_refuses_installed_bytes_that_differ_from_record(tmp_path, monkeypatch):
    """AR-rag4ds-024 O2: direct_url.json says which wheel pip fetched; RECORD
    says what bytes it wrote. A file changed after install is refused."""
    site = _fake_dist(tmp_path, monkeypatch, "en_core_web_fakebytes",
                      direct_url=_pinned_direct_url(),
                      files={"en_core_web_fakebytes/weights.bin": b"original weights"})
    pin = dict(lm.PINNED_LANGUAGE_MODEL, package="en_core_web_fakebytes")
    assert lm.verify_installed_files(pin) >= 2  # METADATA, direct_url.json, weights.bin
    (site / "en_core_web_fakebytes" / "weights.bin").write_bytes(b"tampered weights")
    with pytest.raises(lm.LanguageModelError, match="does not match its RECORD digest"):
        lm.verify_installed_files(pin)
    with pytest.raises(lm.LanguageModelError, match="does not match its RECORD digest"):
        lm.load_pinned_language_model(pin)


def test_refuses_a_record_that_lists_no_digest_at_all(tmp_path, monkeypatch):
    """AR-rag4ds-024 F4 (A6): a RECORD whose rows carry no digest verifies
    nothing, and 'nothing verified' must be a refusal, not a pass."""
    _fake_dist(tmp_path, monkeypatch, "en_core_web_fakenodigest",
               direct_url=_pinned_direct_url(), hashed=False)
    pin = dict(lm.PINNED_LANGUAGE_MODEL, package="en_core_web_fakenodigest")
    with pytest.raises(lm.LanguageModelError, match="no digest at all"):
        lm.verify_installed_files(pin)
    with pytest.raises(lm.LanguageModelError, match="no digest at all"):
        lm.load_pinned_language_model(pin)


def test_refuses_a_listed_file_that_cannot_be_read(tmp_path, monkeypatch):
    """AR-rag4ds-024 F4 (A7): a file RECORD lists with a digest but which is
    missing on disk is a refusal, never a silent skip."""
    site = _fake_dist(tmp_path, monkeypatch, "en_core_web_fakemissing",
                      direct_url=_pinned_direct_url(),
                      files={"en_core_web_fakemissing/weights.bin": b"weights"})
    (site / "en_core_web_fakemissing" / "weights.bin").unlink()
    pin = dict(lm.PINNED_LANGUAGE_MODEL, package="en_core_web_fakemissing")
    with pytest.raises(lm.LanguageModelError, match="cannot be read"):
        lm.verify_installed_files(pin)
    with pytest.raises(lm.LanguageModelError, match="cannot be read"):
        lm.load_pinned_language_model(pin)


def test_describe_refuses_tampered_installed_bytes(tmp_path, monkeypatch, nlp):
    """AR-rag4ds-025 F1: describe() must run the loader's environment checks.
    A model whose bytes fail RECORD, behind a direct_url.json that still
    names the pinned wheel, was certified by the old describe() while the
    loader refused it. The real pipeline's meta is the pin's, so only the
    environment half decides here."""
    site = _fake_dist(tmp_path, monkeypatch, "en_core_web_faketamper",
                      direct_url=_pinned_direct_url(),
                      files={"en_core_web_faketamper/tok2vec/model": b"weights"})
    pin = dict(lm.PINNED_LANGUAGE_MODEL, package="en_core_web_faketamper")
    assert lm.describe(nlp, pin)["sha256"] == pin["sha256"]  # intact: admitted
    (site / "en_core_web_faketamper" / "tok2vec" / "model").write_bytes(b"weightz")
    with pytest.raises(lm.LanguageModelError, match="does not match its RECORD digest"):
        lm.describe(nlp, pin)


def test_describe_refuses_a_wheel_hash_that_is_not_the_pin(tmp_path, monkeypatch, nlp):
    """AR-rag4ds-025 F1, the other direction: direct_url.json names a
    different wheel; the old describe() returned THAT hash as the pin."""
    other = {"url": lm.PINNED_LANGUAGE_MODEL["url"],
             "archive_info": {"hash": "sha256=" + "f" * 64, "hashes": {"sha256": "f" * 64}}}
    _fake_dist(tmp_path, monkeypatch, "en_core_web_fakeother", direct_url=other)
    pin = dict(lm.PINNED_LANGUAGE_MODEL, package="en_core_web_fakeother")
    with pytest.raises(lm.LanguageModelError, match="!= pinned"):
        lm.describe(nlp, pin)


def test_reports_a_listed_but_deleted_direct_url_as_absent(tmp_path, monkeypatch):
    """AR-rag4ds-025 F3: RECORD still lists direct_url.json but the file is
    gone from disk - the partial-removal drift the docstring names. The
    filesystem, not the listing, decides: this is 'absent'."""
    site = _fake_dist(tmp_path, monkeypatch, "en_core_web_fakegone",
                      direct_url=_pinned_direct_url())
    (site / "en_core_web_fakegone-3.8.0.dist-info" / "direct_url.json").unlink()
    pin = dict(lm.PINNED_LANGUAGE_MODEL, package="en_core_web_fakegone")
    with pytest.raises(lm.LanguageModelError, match="is absent"):
        lm.load_pinned_language_model(pin)


def test_refuses_a_distribution_that_does_not_import(tmp_path, monkeypatch):
    """Every metadata check passes, but there is no importable package behind
    it: the ImportError arm must wrap, not leak."""
    _fake_dist(tmp_path, monkeypatch, "en_core_web_fakeghost", direct_url=_pinned_direct_url())
    pin = dict(lm.PINNED_LANGUAGE_MODEL, package="en_core_web_fakeghost")
    with pytest.raises(lm.LanguageModelError, match="does not import"):
        lm.load_pinned_language_model(pin)


def test_refuses_a_running_spacy_outside_the_declared_range(monkeypatch):
    """The pin is untouched; only the running spaCy is forced past the model's
    range - the shape of a dependency-bump commit."""
    monkeypatch.setattr(spacy, "__version__", "4.0.0")
    with pytest.raises(lm.LanguageModelError, match="outside the model's range"):
        lm.load_pinned_language_model()


# --- the unlock is real (F3: a real entity, not hasattr) ---


def test_the_unlock_is_real_pos_dependencies_chunks_and_entities(nlp):
    doc = nlp("Bearing wear is the dominant failure mode of rotary actuators.")
    assert all(tok.pos_ for tok in doc)
    assert all(tok.dep_ for tok in doc)
    assert [c.text for c in doc.noun_chunks] == ["wear", "the dominant failure mode",
                                                 "rotary actuators"]
    dated = nlp("The report was filed on 3 March 2024 in Tokyo by Explosion.")
    assert [(e.text, e.label_) for e in dated.ents] == [("3 March 2024", "DATE"),
                                                        ("Tokyo", "GPE")]


def test_model_split_is_deterministic(nlp):
    first = [(s.text, [t.pos_ for t in s]) for s in nlp(SYNTH).sents]
    again = [(s.text, [t.pos_ for t in s]) for s in lm.load_pinned_language_model()(SYNTH).sents]
    assert first == again


# --- the measured basis for keeping the sentencizer (F1) ---


def test_rule_strategy_keeps_the_sentencizer_on_the_measured_basis(nlp):
    """The decision recorded in docs/vt3_pins.md, asserted from the same
    procedure that produced it (tools/splitter_measurement.py): equal on the
    S7 corpus, worse on the heading-and-table sample. If a future model
    changes either fact this goes red and the decision is re-taken."""
    sm = _measurement_module()
    assert sm.SYNTH == SYNTH  # byte-identical to the S7 corpus
    m = sm.measure(nlp)
    synth = m["SYNTH"]
    assert synth["parser_equals_sentencizer_non_empty"]
    assert len(sm.non_empty(synth["parser"])) == 6
    table = m["HEADING_TABLE"]
    assert not table["parser_equals_sentencizer_non_empty"]
    assert table["parser_splits_inside_a_table_row"]
    assert table["senter_equals_sentencizer_non_empty"]  # measured: no gain from senter
    # AR-rag4ds-025 F2: abbreviation-adjacent prose, where the record had
    # generalised "neither is better". Measured: each splitter wins and loses
    # different boundaries, so neither dominates and the decision stands.
    prose = m["PROSE_ABBREVIATIONS"]
    assert not prose["parser_equals_sentencizer_non_empty"]
    assert not prose["senter_equals_sentencizer_non_empty"]
    sent, par, snt = (sm.non_empty(prose[k]) for k in ("sentencizer", "parser", "senter"))
    assert "Paik et al." in sent and "2019)." in sent      # sentencizer breaks the citation
    assert "Paik et al. 2019)." in par and "Paik et al. 2019)." in snt  # the model keeps it
    assert "See Fig." in sent and "See Fig. 3 for the torque signature." in par
    assert "The method follows Smith et al." in sent and "al." in par  # parser breaks et al.
    assert "Vibration peaked at 3.2 g. The controller logged every event." in sent
    assert "Vibration peaked at 3.2 g." in par                # parser finds the unit boundary
    described = RuleStrategy().describe()
    assert described["model"] is None
    assert re.search(r"sentencizer .*pinned but not used", described["params"]["sentencizer"])
