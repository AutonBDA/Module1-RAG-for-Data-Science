"""S6 end to end (AR-rag4ds-020 O15): main() run for real - a real corpus
document ingested through corpus.ingest, a real index built with the pinned
sentence-transformers model over a compiled tree, the rag4ds-gate CLI driven
through its argparse entry with scripted stdin, then the --retract branch -
and the fold compiles what the session approved. The transcript and the
session manifests are written where the test can show them.
"""

import builtins
import json
import os
import sys
import time

import pytest

import rag4ds.gate.cli as cli
from rag4ds.auditlog.journal import Journal
from rag4ds.corpus.catalog import Catalog, ingest
from rag4ds.gate.decisions import approve_entry
from rag4ds.index.build import SentenceTransformerEmbedder, build_index
from rag4ds.kb.fold import DECISION_JOURNAL, compile_tree
from rag4ds.model.entry import Entry
from rag4ds.staging.store import StagingStore

PINNED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PINNED_REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
WHO = "k-naruse"
TS = "2026-08-23 15:40 JST"


def tree_identity() -> dict:
    """The real commit of the code under test (AR-rag4ds-020 O17: the manifests
    name the code that ran, not a fixture constant) AND whether the module's
    src/tests were dirty at the time (O21: the artefact says so itself, so a
    reader need not trust a sentence). Skips if git is absent."""
    import subprocess
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", "src", "tests"],
                                capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git unavailable for the manifest's code_commit: {exc}")
    return {"head": head, "dirty": bool(status.strip()),
            "dirty_paths": [line[3:] for line in status.splitlines()]}


@pytest.fixture
def jst_process_clock():
    """main() refuses a non-JST clock by design. On POSIX the process clock is
    settable (TZ + tzset), so the e2e runs on the Linux CI runner too; the
    fixture restores TZ and re-runs tzset() on teardown (O22). Yields whether
    the clock is JST now."""
    if time.localtime().tm_gmtoff == 9 * 3600:
        yield True
        return
    if not hasattr(time, "tzset"):
        yield False
        return
    old = os.environ.get("TZ")
    os.environ["TZ"] = "Asia/Tokyo"
    time.tzset()
    try:
        yield time.localtime().tm_gmtoff == 9 * 3600
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


def test_main_refuses_a_non_jst_clock(monkeypatch, capsys):
    """The JST guard in main() (F15): a non-JST local clock is refused before
    anything is read or written."""
    class NotJst:
        tm_gmtoff = 0

    monkeypatch.setattr(cli.time, "localtime", lambda *a: NotJst())
    assert cli.main(["--module-root", "x", "--approver", WHO, "--code-commit", "abcdef0",
                     "--index-dir", "x", "--data-dir", "x"]) == 1
    assert "not JST" in capsys.readouterr().err


DOC = ("Section 3. Bearing wear dominates actuator failure in rotary joints. "
       "A rising torque signature precedes most bearing failures. End.")


def test_main_end_to_end_with_real_index_model_and_corpus(tmp_path, monkeypatch,
                                                          jst_process_clock):
    if not jst_process_clock:
        pytest.skip("main() refuses a non-JST clock by design and this machine's clock cannot "
                    "be set to JST for the test; the guard is covered by "
                    "test_main_refuses_a_non_jst_clock; main()'s end-to-end coverage runs on "
                    "JST machines and wherever TZ+tzset can set the process clock")
    tree = tree_identity()
    COMMIT = tree["head"]
    (tmp_path / "tree.json").write_text(json.dumps(tree, indent=1), encoding="utf-8")
    root = tmp_path / "module"
    for d in ("journal", "stage", "runs", "shared/general", "shared/tasks",
              "packs/robotics/entries", "packs/robotics/mappings"):
        (root / d).mkdir(parents=True)
    # AR-rag4ds-026 F3: main() loads the module's identity registry and refuses
    # an approver it does not know
    (root / "identities.yaml").write_text(
        f"schema: 1\nidentities:\n  - id: v\n    name: {WHO}\n    aliases: []\n"
        f"    roles: [validator]\n    packs: [robotics]\n", encoding="utf-8")
    data = tmp_path / "data"
    (data / "robotics").mkdir(parents=True)
    raw = data / "robotics" / "handbook.txt"
    raw.write_text(DOC, encoding="utf-8")

    # 1. the corpus: catalog + ingest manifest (the source resolver reads it)
    catalog = Catalog("robotics", root / "packs" / "robotics")
    ingest_runs = Journal(root / "runs" / "ingest.jsonl")
    manifest = ingest(catalog, raw, data, "CC-BY-4.0", "https://example.org/handbook", True,
                      {"title": "handbook"}, TS, ingest_runs, COMMIT, {"strategy": "manual"})
    doc_hash = manifest["hash"]

    # 2. one already-approved entry, so the conflict display has something to find
    decisions = Journal(root / DECISION_JOURNAL)
    prior = Entry(id="torque-signature", type="concept", layer="domain", pack="robotics",
                  content="A rising torque signature precedes bearing failure.",
                  source_reference="corpus:robotics/handbook#s3", version=1, status="current")
    decisions.append(approve_entry("approve", WHO, TS, "prior", "corpus:robotics/handbook#s3",
                                   prior, {}))
    compile_tree(decisions, root)

    # 3. the real index with the pinned model over the compiled tree
    emb = SentenceTransformerEmbedder(PINNED_MODEL, PINNED_REV)
    build_index(root, tmp_path / "index", emb, {"robotics": "0.1.0"}, 0.2, COMMIT, TS)

    # 4. two candidates in staging
    st = StagingStore(root / "stage" / "candidates.jsonl")
    s = DOC.index("Bearing wear")
    e = DOC.index("joints.") + len("joints.")
    c1 = st.file_candidate({"kind": "entry", "proposed": {
        "type": "concept", "content": "Bearing wear dominates actuator failure in rotary joints.",
        "source_reference": "corpus:robotics/handbook#s3"},
        "layer": "domain", "pack": "robotics", "source_hash": doc_hash,
        "span_start": s, "span_end": e, "extractor": "rule-v1",
        "run_manifest": "runs/extract.jsonl:1"}, TS)
    st.file_candidate({"kind": "entry", "proposed": {
        "type": "fact", "content": "Lubrication schedules are irrelevant to bearing wear.",
        "source_reference": "corpus:robotics/handbook#s3"},
        "layer": "domain", "pack": "robotics", "source_hash": doc_hash,
        "span_start": 0, "span_end": 10, "extractor": "rule-v1",
        "run_manifest": "runs/extract.jsonl:1"}, TS)

    # 5. the CLI, for real: argparse entry + scripted stdin
    answers = iter([
        "a", "y", "bearing-wear", "", "torque-signature",   # approve c1, flag the shown conflict
        "r", "the source does not support it",              # reject c2
    ])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    transcript = tmp_path / "session.txt"
    rc = cli.main(["--module-root", str(root), "--approver", WHO, "--code-commit", COMMIT,
                   "--index-dir", str(tmp_path / "index"), "--data-dir", str(data),
                   "--transcript", str(transcript)])
    assert rc == 0
    text = transcript.read_text(encoding="utf-8")
    assert "Bearing wear dominates actuator failure in rotary joints." in text
    assert "related approved entries" in text and "torque-signature" in text
    assert "session summary" in text
    # AR-rag4ds-034 F9 (AR-033 r1 O1): the live gate's transcript witnesses what was
    # typed at the prompts, at the call site - main() wraps input with the echo reader
    assert "reason: the source does not support it" in text
    assert "entry id [bearing-wear]: bearing-wear" in text or "]: bearing-wear" in text
    assert text.count("[q]uit: a") == 1 and text.count("[q]uit: r") == 1
    decs = decisions.read_all()
    assert [d["type"] for d in decs] == ["approve", "approve", "reject"]
    assert decs[1]["entry"]["id"] == "bearing-wear"
    assert decs[1]["conflict_with"] == "torque-signature"
    assert decs[1]["review"]["conflict_display"] == "checked-with-results"
    assert decs[1]["review"]["source_span"] == "rendered"
    assert decs[1]["review"]["related_shown"] == ["torque-signature"]
    assert st.state_of(c1) == "APPROVED"
    runs = Journal(root / "runs" / "gate-sessions.jsonl").read_all()
    assert len(runs) == 1 and runs[0]["counts"]["approve"] == 1 and runs[0]["counts"]["reject"] == 1
    assert runs[0]["config"]["conflict_display"]["embedding_model"]["id"] == PINNED_MODEL
    assert runs[0]["config"]["conflict_display"]["embedding_model"]["revision"] == PINNED_REV

    # 6. the --retract branch, end to end
    rc = cli.main(["--module-root", str(root), "--approver", WHO, "--code-commit", COMMIT,
                   "--index-dir", str(tmp_path / "index"), "--data-dir", str(data),
                   "--retract", "bearing-wear"])
    assert rc == 0
    decs = decisions.read_all()
    assert decs[-1]["type"] == "retract"
    assert decs[-1]["review"]["conflict_display"] == "not-applicable"
    assert len(Journal(root / "runs" / "gate-sessions.jsonl").read_all()) == 2

    # 7. the fold compiles what the sessions decided
    result = compile_tree(decisions, root)
    assert result.versions == {"torque-signature": 1}
    assert result.retired == frozenset({"bearing-wear"})
    assert not (root / "packs/robotics/entries/bearing-wear.json").exists()
    # keep the artefacts readable for the record
    (tmp_path / "manifests.json").write_text(
        json.dumps(Journal(root / "runs" / "gate-sessions.jsonl").read_all(), indent=1),
        encoding="utf-8")
    sys.stdout.write(f"\n[e2e transcript at {transcript}]\n")


def test_f16_transcript_write_failure_is_a_failure_unless_a_session_error_wins(tmp_path,
                                                                                capsys):
    """AR-rag4ds-020 F16/O20: write_transcript returns 0 on success; on OSError
    it reports to stderr and returns 2 after a successful session (the write
    failure IS the failure - never rc 0 with no file) or 0 when a session
    exception is in flight (that exception wins)."""
    from rich.console import Console
    console = Console(record=True, markup=False, file=open(os.devnull, "w"))
    console.print("hello transcript")
    ok = tmp_path / "t.txt"
    assert cli.write_transcript(console, ok, session_failed=False) == 0
    assert "hello transcript" in ok.read_text(encoding="utf-8")
    unwritable = tmp_path  # a directory: write_text raises OSError
    assert cli.write_transcript(console, unwritable, session_failed=False) == 2
    assert "transcript not written" in capsys.readouterr().err
    assert cli.write_transcript(console, unwritable, session_failed=True) == 0
    assert "transcript not written" in capsys.readouterr().err
