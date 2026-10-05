"""S9c part 3 - tools/fill_sheet.py: the validator keys the empty rows of a
sheet one question at a time; keyed rows are never touched; every answer
is saved at once; a, e and bad answers are re-asked; the filled sheet is
what tools/gate_sheet.py then drives."""
import sys
from pathlib import Path

from rag4ds.staging.store import StagingStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import fill_sheet  # noqa: E402
import gate_sheet  # noqa: E402
import test_s9d_sheet_fold as s9d  # noqa: E402


def test_empty_rows_are_asked_keyed_rows_are_kept_and_the_result_drives_the_gate(tmp_path,
                                                                                capsys):
    root, ids = s9d._module(tmp_path)
    sheet = tmp_path / "s.csv"
    sheet.write_text(f"n,id,key,reason\n1,{ids[0]},r,junk\n2,{ids[1]},,\n3,{ids[2]},,\n",
                     encoding="utf-8")
    answers = iter(["a",            # refused: needs the screen
                    "x",            # refused: r or s
                    "r", "",        # refused: a reason
                    "r", "a, b",    # refused: comma
                    "r", "step",    # row 2 keyed
                    "S"])           # row 3 keyed (case-insensitive)
    assert fill_sheet.run(root, sheet, read=lambda prompt: next(answers)) == 0
    out = capsys.readouterr().out
    assert "2 to fill" in out and "Gene 1 - a unit of heredity number 1." in out
    assert "needs the passage on screen" in out and "type r or s" in out
    assert "a rejection needs a reason" in out and "no comma or quote" in out
    assert "done - 3 of 3 rows keyed" in out
    rows = gate_sheet.load_sheet(sheet)
    assert [(r["key"], r["reason"]) for r in rows] == [("r", "junk"), ("r", "step"), ("s", "")]
    # the filled sheet drives the real gate
    assert gate_sheet.run(root, sheet, "V. Alidator", "abcdef0", None, root, "bio", None,
                          dry_run=False, clock=s9d.CLOCK) == 0
    store = StagingStore(root / "stage" / "candidates.jsonl")
    assert [store.state_of(i) for i in ids] == ["REJECTED", "REJECTED", "PENDING"]


def test_an_interrupted_fill_keeps_what_was_answered(tmp_path, capsys):
    root, ids = s9d._module(tmp_path)
    sheet = tmp_path / "s.csv"
    sheet.write_text(f"n,id,key,reason\n1,{ids[0]},,\n2,{ids[1]},,\n3,{ids[2]},g,{ids[0]}\n",
                     encoding="utf-8")

    def read(prompt):
        # the key prompt mentions "a reason" too, so match the prompts by their start
        if prompt.startswith("  reason"):
            return "junk"
        if read.n == 0:
            read.n += 1
            return "r"
        raise KeyboardInterrupt
    read.n = 0
    try:
        fill_sheet.run(root, sheet, read=read)
    except KeyboardInterrupt:
        pass
    text = sheet.read_text(encoding="utf-8")
    assert f"1,{ids[0]},r,junk\n2,{ids[1]},,\n3,{ids[2]},g,{ids[0]}\n" in text
    # a second run asks only the one row still empty
    answers = iter(["s"])
    assert fill_sheet.run(root, sheet, read=lambda prompt: next(answers)) == 0
    assert "1 to fill" in capsys.readouterr().out


def test_refusals(tmp_path, capsys):
    root, ids = s9d._module(tmp_path)
    bad = tmp_path / "b.csv"
    bad.write_text("id,key\nx,r\n", encoding="utf-8")
    assert fill_sheet.run(root, bad, read=lambda p: "s") == 1
    assert "columns must be" in capsys.readouterr().err
    bad.write_text("n,id,key,reason\n1,cand-0000000000000000,,\n", encoding="utf-8")
    assert fill_sheet.run(root, bad, read=lambda p: "s") == 1
    assert "is not a filed candidate" in capsys.readouterr().err
    bad.write_text("n,id,key,reason\n1,cand-1,r\n", encoding="utf-8")
    assert fill_sheet.run(root, bad, read=lambda p: "s") == 1
    assert "expected 4 fields" in capsys.readouterr().err
