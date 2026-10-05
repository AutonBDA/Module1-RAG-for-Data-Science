"""NFR6, mechanically (AR-rag4ds-026 F1): no git-tracked file of the module
carries a verbatim run of a non-redistributable corpus document at or over
the quotation bound's source-character ceiling. The scan runs against the
live tree when the restricted document's local copy is present (the module's
data/ or RAG4DS_DATA_DIR); when it is absent the test SKIPS BY NAME - it never
passes on absence. Its sensitivity is witnessed on a scratch module with a
planted passage.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))
import nfr6_scan  # noqa: E402


def _data_dir():
    env = os.environ.get("RAG4DS_DATA_DIR")
    return Path(env) if env else None


def test_live_tree_carries_no_passage_of_a_restricted_document():
    docs = nfr6_scan.restricted_documents(MODULE_ROOT)
    assert docs, "the CRISP-DM guide is catalogued as not redistributable"
    for d in docs:
        if nfr6_scan.locate(d, MODULE_ROOT, _data_dir()) is None:
            pytest.skip(f"local copy of restricted document {d['hash'][:16]}... "
                        f"({d['title']!r}) is absent here; the NFR6 scan cannot run "
                        f"(set RAG4DS_DATA_DIR)")
    hits, notes = nfr6_scan.run(MODULE_ROOT, _data_dir())
    assert any("windows" in n for n in notes)
    assert hits == [], "\n".join(hits)


def _scratch_module(tmp_path, planted: str | None):
    root = tmp_path / "mod"
    (root / "shared" / "corpus").mkdir(parents=True)
    (root / "runs").mkdir()
    (root / "data").mkdir()
    (root / "docs").mkdir()
    doc = ("The guide. " + " ".join(f"word{i} sentence of the restricted text" for i in range(40))
           + " end.")
    (root / "data" / "guide.txt").write_text(doc, encoding="utf-8")
    import hashlib
    import json
    h = hashlib.sha256(doc.encode()).hexdigest()
    (root / "shared" / "corpus" / "catalog.yaml").write_text(
        f"schema: 1\npack: shared\ndocuments:\n- hash: {h}\n  license: proprietary\n"
        f"  fetch_pointer: x\n  redistributable: false\n  metadata:\n    title: guide\n",
        encoding="utf-8")
    (root / "runs" / "ingest.jsonl").write_text(
        json.dumps({"kind": "ingest", "hash": h, "local_path": "data/guide.txt"}) + "\n",
        encoding="utf-8")
    (root / ".gitignore").write_text("data/\n", encoding="utf-8")
    text = "A note that restates the guide in its own words.\n"
    if planted:
        text += planted + "\n"
    (root / "docs" / "note.md").write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root, doc


def test_scan_finds_a_planted_passage_and_passes_a_clean_tree(tmp_path):
    root, doc = _scratch_module(tmp_path, None)
    hits, notes = nfr6_scan.run(root, None)
    assert hits == [] and any("windows" in n for n in notes)
    # a run one character UNDER the ceiling passes: that is the bound's own rule
    short = doc[100:100 + nfr6_scan.WINDOW - 1]
    root2, _ = _scratch_module(tmp_path / "b", short)
    assert nfr6_scan.run(root2, None)[0] == []
    # at the ceiling, whitespace-wrapped and re-cased, it is found and named
    planted = doc[100:100 + nfr6_scan.WINDOW].upper().replace(" ", "\n  ")
    root3, _ = _scratch_module(tmp_path / "c", planted)
    hits, _ = nfr6_scan.run(root3, None)
    assert len(hits) == 1 and hits[0].startswith("docs/note.md: verbatim run of "), hits


def test_scan_refuses_to_run_without_the_local_copy(tmp_path):
    root, _ = _scratch_module(tmp_path, None)
    (root / "data" / "guide.txt").unlink()
    with pytest.raises(nfr6_scan.Nfr6Error, match="is absent; cannot scan"):
        nfr6_scan.run(root, None)


def test_scan_flags_a_tracked_restricted_document_itself(tmp_path):
    root, _ = _scratch_module(tmp_path, None)
    (root / ".gitignore").write_text("", encoding="utf-8")
    subprocess.run(["git", "add", "-f", "data/guide.txt"], cwd=root, check=True)
    hits, _ = nfr6_scan.run(root, None)
    assert any("IS the restricted document" in h for h in hits), hits


def test_scan_refuses_a_copy_that_is_not_the_catalogued_document(tmp_path):
    """AR-rag4ds-026 r2 F10: the auditor's decoy - a different file at the
    manifested path under RAG4DS_DATA_DIR turned a 722-character leak into
    OK. The scan now hashes the copy against the catalog and will not run
    against anything else."""
    _, doc = _scratch_module(tmp_path / "ref", None)
    planted = doc[100:100 + nfr6_scan.WINDOW + 10]
    root, _ = _scratch_module(tmp_path / "leak", planted)
    hits, _ = nfr6_scan.run(root, None)
    assert len(hits) == 1
    decoy = tmp_path / "decoy"
    (decoy / "data").mkdir(parents=True)
    (decoy / "data" / "guide.txt").write_text("lorem ipsum dolor sit amet " * 200,
                                              encoding="utf-8")
    with pytest.raises(nfr6_scan.Nfr6Error, match="not the catalogued"):
        nfr6_scan.run(root, decoy)


def test_a_dot_leader_is_not_a_passage(tmp_path):
    """AR-rag4ds-026 r2 O11: pytest progress dots against the guide's
    dot-leader contents table matched as a 74-character run; a run needs
    MIN_DISTINCT_CHARS distinct characters to count."""
    root, _ = _scratch_module(tmp_path, "." * 80)
    # give the scratch document a dot leader too
    import hashlib
    import json
    d = root / "data" / "guide.txt"
    text = d.read_text(encoding="utf-8") + " contents " + "." * 90 + " 7"
    d.write_text(text, encoding="utf-8")
    h = hashlib.sha256(text.encode()).hexdigest()
    cat = root / "shared" / "corpus" / "catalog.yaml"
    cat.write_text(cat.read_text(encoding="utf-8").replace(
        cat.read_text(encoding="utf-8").split("hash: ")[1].split("\n")[0], h), encoding="utf-8")
    (root / "runs" / "ingest.jsonl").write_text(
        json.dumps({"kind": "ingest", "hash": h, "local_path": "data/guide.txt"}) + "\n",
        encoding="utf-8")
    hits, _ = nfr6_scan.run(root, None)
    assert hits == [], hits


def test_check_py_runs_the_scan_and_skips_by_name_without_the_copy(tmp_path, monkeypatch,
                                                                    capsys):
    """AR-rag4ds-026 r2 O13: the scan is a check.py step; --skip-absent makes a
    missing local copy a NAMED skip (exit 0, the document named), while a
    present-but-wrong copy stays exit 2 with or without the flag."""
    steps = (MODULE_ROOT / "tools" / "check.py").read_text(encoding="utf-8")
    assert '"nfr6_scan"' in steps and "--skip-absent" in steps
    root, _ = _scratch_module(tmp_path, None)
    monkeypatch.setattr(nfr6_scan, "MODULE_ROOT", root)
    monkeypatch.delenv("RAG4DS_DATA_DIR", raising=False)
    (root / "data" / "guide.txt").unlink()
    assert nfr6_scan.main([]) == 2
    assert "CANNOT SCAN" in capsys.readouterr().err
    assert nfr6_scan.main(["--skip-absent"]) == 0
    out = capsys.readouterr().out
    assert "SKIPPED (not a pass)" in out and "is absent; cannot scan" in out
    (root / "data" / "guide.txt").write_text("not the document " * 20, encoding="utf-8")
    assert nfr6_scan.main(["--skip-absent"]) == 2
    assert "not the catalogued" in capsys.readouterr().err


def test_a_low_alphabet_run_is_still_a_passage(tmp_path):
    """AR-rag4ds-026 r3 F17: MIN_DISTINCT_CHARS removes only the
    repeated-single-character class; a genomic sequence (four letters) or a
    binary matrix (three characters) at the ceiling is found."""
    import hashlib
    import json
    for name, alphabet in (("dna", "ACGT"), ("binary", "01 ")):
        root, _ = _scratch_module(tmp_path / name, None)
        seq = "".join(alphabet[(i * 7 + i // 3) % len(alphabet)] for i in range(400))
        d = root / "data" / "guide.txt"
        text = d.read_text(encoding="utf-8") + " appendix " + seq
        d.write_text(text, encoding="utf-8")
        h = hashlib.sha256(text.encode()).hexdigest()
        cat = root / "shared" / "corpus" / "catalog.yaml"
        old = cat.read_text(encoding="utf-8")
        cat.write_text(old.replace(old.split("hash: ")[1].split("\n")[0], h), encoding="utf-8")
        (root / "runs" / "ingest.jsonl").write_text(
            json.dumps({"kind": "ingest", "hash": h, "local_path": "data/guide.txt"}) + "\n",
            encoding="utf-8")
        (root / "docs" / "note.md").write_text("planted: " + seq + "\n", encoding="utf-8")
        assert len(set(seq)) <= 4
        hits, _ = nfr6_scan.run(root, None)
        assert len(hits) == 1 and "docs/note.md" in hits[0], (name, hits)


def test_json_escaping_and_ligatures_do_not_hide_a_passage(tmp_path):
    """AR-rag4ds-027 F4: the module writes knowledge as canonical JSON with
    ensure_ascii, so a newline is two characters and a non-ASCII letter six;
    the guide carries ligatures. The scan reads JSON string values unescaped
    and NFKC-normalises both sides, so the auditor's evading case is found."""
    import hashlib
    import json
    root, _ = _scratch_module(tmp_path, None)
    d = root / "data" / "guide.txt"
    passage = ("The de\ufb01nition of a \ufb01eld in the \ufb02ow of a data\nmining project "
               "is \u201cfixed\u201d before the modeling phase begins, and the con\ufb01dence "
               "in it is measured.")
    text = d.read_text(encoding="utf-8") + "\n\n" + passage
    d.write_bytes(text.encode("utf-8"))  # bytes: no newline translation, the hash must hold
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cat = root / "shared" / "corpus" / "catalog.yaml"
    old = cat.read_text(encoding="utf-8")
    cat.write_text(old.replace(old.split("hash: ")[1].split("\n")[0], h), encoding="utf-8")
    (root / "runs" / "ingest.jsonl").write_text(
        json.dumps({"kind": "ingest", "hash": h, "local_path": "data/guide.txt"}) + "\n",
        encoding="utf-8")
    (root / "shared").joinpath("general").mkdir()
    # the module's own form: canonical JSON, ensure_ascii, the passage as an entry content
    entry = {"id": "phase-note", "content": passage.replace("\ufb01", "fi").replace("\ufb02", "fl")}
    (root / "shared" / "general" / "phase-note.json").write_text(
        json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8")
    (root / "journal").mkdir()
    (root / "journal" / "decisions.jsonl").write_text(
        json.dumps({"type": "approve", "entry": entry}, ensure_ascii=True) + "\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    # the NFKC arm on its own: casefold already folds the fi/fl ligatures, so the witness
    # is a compatibility form casefold leaves alone - fullwidth letters
    assert nfr6_scan.normalise("con\ufb01dence in the \ufb02ow") == "confidence in the flow"
    assert nfr6_scan.normalise("\uff46\uff49\uff45\uff4c\uff44 note") == "field note"
    hits, _ = nfr6_scan.run(root, None)
    files = sorted(h.split(":")[0] for h in hits)
    assert files == ["journal/decisions.jsonl", "shared/general/phase-note.json"], hits
    # a tracked JSON file that does not parse is a loud failure, not a skipped file
    (root / "docs" / "broken.json").write_text("{not json", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    with pytest.raises(nfr6_scan.Nfr6Error, match="not JSON"):
        nfr6_scan.run(root, None)


def test_the_json_lines_branch_finds_what_a_raw_read_misses(tmp_path):
    """AR-rag4ds-027 F8: a decision record whose passage carries an escaped
    newline every forty characters has no 64-character run in the raw file
    (the two-character escape breaks every window); the branch that parses
    the record finds it, and deleting the branch turns this red."""
    import hashlib
    import json
    root, _ = _scratch_module(tmp_path, None)
    words = "these are the words of the restricted document restated here in a long single line"
    passage = " ".join(words.split() * 3)
    lined = "\n".join(passage[i:i + 40] for i in range(0, len(passage), 40))
    d = root / "data" / "guide.txt"
    text = d.read_text(encoding="utf-8") + "\n\n" + lined
    d.write_bytes(text.encode("utf-8"))
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cat = root / "shared" / "corpus" / "catalog.yaml"
    old = cat.read_text(encoding="utf-8")
    cat.write_text(old.replace(old.split("hash: ")[1].split("\n")[0], h), encoding="utf-8")
    (root / "runs" / "ingest.jsonl").write_text(
        json.dumps({"kind": "ingest", "hash": h, "local_path": "data/guide.txt"}) + "\n",
        encoding="utf-8")
    (root / "journal").mkdir()
    rec = json.dumps({"type": "approve", "entry": {"content": lined}}, ensure_ascii=True)
    assert "\\n" in rec and "\n" not in rec.strip()  # the newline is the two-character escape
    (root / "journal" / "decisions.jsonl").write_text(rec + "\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    raw_hay = nfr6_scan.normalise((root / "journal" / "decisions.jsonl").read_text("utf-8"))
    doc = nfr6_scan.normalise(text)
    windows = {doc[i:i + nfr6_scan.WINDOW] for i in range(len(doc) - nfr6_scan.WINDOW + 1)}
    assert nfr6_scan.scan_text(raw_hay, windows) is None  # a raw read misses it
    hits, _ = nfr6_scan.run(root, None)
    assert [h.split(":")[0] for h in hits] == ["journal/decisions.jsonl"], hits
