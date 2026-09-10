"""tools/evidence.py (AR-rag4ds-016 F3): the identity guard must fire when
rag4ds does not resolve inside the module root under test, and pass when it
does - proving PYTHONPATH beats the venv's editable install."""

import importlib.util
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    path = MODULE_ROOT / "tools" / "evidence.py"
    spec = importlib.util.spec_from_file_location("evidence_tool", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hermetic_env_puts_module_src_first(monkeypatch):
    ev = load_tool()
    monkeypatch.setenv("PYTHONPATH", "elsewhere")
    env = ev.hermetic_env(MODULE_ROOT)
    assert env["PYTHONPATH"].split(ev.os.pathsep)[0] == str(MODULE_ROOT / "src")
    assert env["PYTHONPATH"].endswith("elsewhere")


def test_identity_guard_fires_for_a_root_without_the_package(tmp_path):
    ev = load_tool()
    ok, resolved = ev.import_identity(sys.executable, tmp_path)
    assert ok is False
    # whatever resolved (the editable main tree, or an import error), it is not tmp_path
    assert str(tmp_path) not in resolved


def test_identity_guard_passes_when_the_root_shadows_the_install(tmp_path):
    """A fake rag4ds under tmp_path/src must win over the venv's editable
    install - that is the whole hermeticity claim."""
    ev = load_tool()
    pkg = tmp_path / "src" / "rag4ds"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    ok, resolved = ev.import_identity(sys.executable, tmp_path)
    assert ok is True, resolved
    assert Path(resolved).resolve() == (pkg / "__init__.py").resolve()


def test_identity_guard_passes_for_the_real_module_root():
    ev = load_tool()
    ok, resolved = ev.import_identity(sys.executable, MODULE_ROOT)
    assert ok is True, resolved


# --- AR-rag4ds-016 F9: the TOOL's contract (main()), not just the detector ---


class _R:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _fake_run_factory(pytest_line="210 passed, 1 skipped in 1.00s", pytest_rc=0, check_rc=0):
    def fake_run(cmd, cwd, env=None):
        s = " ".join(str(c) for c in cmd)
        if "rev-parse --verify" in s:
            return _R(0, "0123456789abcdef0123456789abcdef01234567\n")
        if "rev-parse --short" in s:
            return _R(0, "0123456\n")
        if "-m pytest" in s:
            return _R(pytest_rc, pytest_line + "\n")
        if "check.py" in s:
            return _R(check_rc, "")
        return _R(0, "")
    return fake_run


def _arm(ev, monkeypatch, tmp_path, identity_ok, resolved="/elsewhere/rag4ds/__init__.py",
         **run_kw):
    wt = tmp_path / "wt"
    (wt / "AutonBDA" / "module1_rag_for_ds").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("RAG4DS_EVIDENCE_WORKTREE", str(wt))
    monkeypatch.setattr(ev, "run", _fake_run_factory(**run_kw))
    monkeypatch.setattr(ev, "import_identity", lambda *a: (identity_ok, resolved))
    monkeypatch.setattr(sys, "argv", ["evidence.py", "deadbeef", "--venv-python", "py"])


def test_main_refuses_when_identity_is_wrong_and_prints_nothing(tmp_path, monkeypatch, capsys):
    ev = load_tool()
    _arm(ev, monkeypatch, tmp_path, identity_ok=False)
    assert ev.main() == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert "does not resolve inside the worktree" in err


def test_main_refuses_when_pytest_is_not_clean_and_prints_nothing(tmp_path, monkeypatch, capsys):
    ev = load_tool()
    _arm(ev, monkeypatch, tmp_path, identity_ok=True, resolved="x",
         pytest_line="2 failed, 208 passed in 1.00s", pytest_rc=1)
    assert ev.main() == 1
    out, err = capsys.readouterr()
    assert out == "" and "pytest did not pass cleanly" in err
    # a green-looking line with a non-zero exit is still refused
    _arm(ev, monkeypatch, tmp_path, identity_ok=True, resolved="x",
         pytest_line="210 passed in 1.00s", pytest_rc=1)
    assert ev.main() == 1 and capsys.readouterr().out == ""
    # an empty closing line is refused
    _arm(ev, monkeypatch, tmp_path, identity_ok=True, resolved="x", pytest_line="", pytest_rc=0)
    assert ev.main() == 1 and capsys.readouterr().out == ""


def test_main_refuses_when_check_fails_and_prints_nothing(tmp_path, monkeypatch, capsys):
    ev = load_tool()
    _arm(ev, monkeypatch, tmp_path, identity_ok=True, resolved="x", check_rc=3)
    assert ev.main() == 1
    out, err = capsys.readouterr()
    assert out == "" and "check.py exited 3" in err


def test_main_prints_the_complete_block_only_when_every_gate_passes(tmp_path, monkeypatch, capsys):
    ev = load_tool()
    _arm(ev, monkeypatch, tmp_path, identity_ok=True, resolved="/wt/src/rag4ds/__init__.py")
    assert ev.main() == 0
    out = capsys.readouterr().out
    assert "0123456" in out and "/wt/src/rag4ds/__init__.py" in out
    assert "210 passed, 1 skipped in 1.00s" in out and "exit=0" in out
