"""Every JST clock guard in the module, in one table (AR-rag4ds-043 r2 O1).

Each tool and the gate refuse to run on a machine whose local clock is not
JST, because every timestamp they write claims JST and a wrong one is a
fabricated record (the repository's operating rules, section 5). The guard has been found weak
three times - AR-rag4ds-033 F2 (the sheet and fold tools had none),
AR-rag4ds-042 F4 and AR-rag4ds-043 F2 (the mapping tool's was never taken
because every test injects a clock), and AR-rag4ds-043 r2 O1 (the
witnesses pinned one offset, so loosening "exactly UTC+9" to "any positive
offset" stayed green). This file closes the class:

- every guard is driven over a table of WRONG offsets, not just UTC, so no
  loosening of the comparison can hide, and over UTC+9 itself, so the
  guard cannot be turned into a refusal of everything;
- a coverage test scans src/ and tools/ for the guard's own expression and
  fails if any occurrence is not accounted for here, so a guard added
  later cannot arrive unwitnessed.
"""

import re
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))

import build_index  # noqa: E402
import extract_pack  # noqa: E402
import file_mapping_draft  # noqa: E402
import fold_tree  # noqa: E402
import pack_author  # noqa: E402
import pack_intake  # noqa: E402
import refile_candidates  # noqa: E402
import release_pack  # noqa: E402
import run_demo  # noqa: E402
import s8_seed  # noqa: E402
from rag4ds.gate import cli as gate_cli  # noqa: E402

# offsets that are NOT JST. UTC+8 and UTC+9:30 are the ones a loosened
# comparison would wave through; the negative and the off-by-one second
# cover the other directions.
WRONG = [
    pytest.param(0, id="utc"),
    pytest.param(8 * 3600, id="utc+8"),
    pytest.param(9 * 3600 + 1800, id="utc+9:30"),
    pytest.param(-5 * 3600, id="utc-5"),
    pytest.param(9 * 3600 - 1, id="one-second-short"),
]
JST = 9 * 3600

# the guards reached through a jst_now() of their own: (name, module, error)
CLOCKS = [
    ("tools/build_index.py", build_index, build_index.IndexBuildError),
    ("tools/extract_pack.py", extract_pack, extract_pack.ExtractPackError),
    ("tools/file_mapping_draft.py", file_mapping_draft, file_mapping_draft.MappingDraftError),
    ("tools/fold_tree.py", fold_tree, fold_tree.FoldError),
    ("tools/pack_author.py", pack_author, pack_author.PackAuthorError),  # S9e part 7
    ("tools/pack_intake.py", pack_intake, pack_intake.IntakeError),
    ("tools/refile_candidates.py", refile_candidates, refile_candidates.RefileError),
    ("tools/release_pack.py", release_pack, SystemExit),  # S9 demo road part C
    ("tools/run_demo.py", run_demo, run_demo.RunDemoError),  # S9 demo road part A
    ("tools/s8_seed.py", s8_seed, SystemExit),
]
# the two that test their own clock inline, witnessed in their own files
# over the same offsets: gate_sheet.py (test_s9d_sheet_fold.py) and the
# gate CLI's main() (test_s6_gate_e2e.py)
INLINE = {
    "tools/gate_sheet.py": "tests/test_s9d_sheet_fold.py",
    "tools/gate_certified.py": "tests/test_s9e_gate_certified.py",  # S9e part 8
    "src/rag4ds/gate/cli.py": "tests/test_s6_gate_e2e.py",
}


def _offset(monkeypatch, module, seconds):
    monkeypatch.setattr(module.time, "localtime",
                        lambda *a: types.SimpleNamespace(tm_gmtoff=seconds))


@pytest.mark.parametrize("where, module, error", CLOCKS, ids=[c[0] for c in CLOCKS])
@pytest.mark.parametrize("offset", WRONG)
def test_a_clock_that_is_not_jst_is_refused(monkeypatch, where, module, error, offset):
    _offset(monkeypatch, module, offset)
    with pytest.raises(error, match="not JST"):
        module.jst_now()


@pytest.mark.parametrize("where, module, error", CLOCKS, ids=[c[0] for c in CLOCKS])
def test_the_jst_clock_itself_is_accepted_and_stamped_jst(monkeypatch, where, module, error):
    """The other half: a guard that refuses everything would pass the test
    above and break every tool."""
    _offset(monkeypatch, module, JST)
    stamp = module.jst_now()
    assert stamp.endswith(" JST")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}(:\d{2})? JST", stamp), stamp


def test_the_gate_cli_refuses_a_non_jst_clock_over_the_same_offsets(monkeypatch, capsys):
    """test_s6_gate_e2e.py drives main() on UTC; the other offsets belong
    here with the rest of the table."""
    for offset in (8 * 3600, 9 * 3600 + 1800, -5 * 3600):
        _offset(monkeypatch, gate_cli, offset)
        assert gate_cli.main(["--module-root", "x", "--approver", "K. Naruse",
                              "--code-commit", "abcdef0", "--index-dir", "x",
                              "--data-dir", "x"]) == 1
        assert "not JST" in capsys.readouterr().err


def test_every_jst_guard_in_the_module_is_accounted_for_here():
    """A guard added later without a witness fails this test by name. The
    scan is over git's own file list, so an untracked scratch copy cannot
    add or hide one."""
    tracked = subprocess.run(["git", "ls-files", "-z", "--", "src", "tools"],
                             cwd=MODULE_ROOT, capture_output=True, check=True)
    found = {}
    for rel in tracked.stdout.decode().split("\0"):
        if not rel.endswith(".py"):
            continue
        text = (MODULE_ROOT / rel).read_text(encoding="utf-8")
        n = len(re.findall(r"tm_gmtoff", text))
        if n:
            found[rel] = n
    accounted = {where for where, _m, _e in CLOCKS} | set(INLINE)
    assert set(found) == accounted, (
        f"JST guards not accounted for in this table: {sorted(set(found) - accounted)}; "
        f"listed but no longer present: {sorted(accounted - set(found))}")
    assert all(n == 1 for n in found.values()), found
    for witness in INLINE.values():
        assert (MODULE_ROOT / witness).is_file()
        assert "tm_gmtoff" in (MODULE_ROOT / witness).read_text(encoding="utf-8"), witness


def test_on_a_jst_machine_the_uninjected_clock_stamps_todays_date():
    """The real path, nothing patched: on the team's machine (JST) every
    guard passes and stamps the day the machine is on - which is what a
    live run writes into a record. Skipped on a CI runner (UTC), where no
    tool is run live."""
    if time.localtime().tm_gmtoff != JST:
        pytest.skip("this runner is not on JST; no tool is run live here")
    today = time.strftime("%Y-%m-%d")
    for where, module, _error in CLOCKS:
        stamp = module.jst_now()
        assert stamp.startswith(today) and stamp.endswith(" JST"), (where, stamp)
