"""S0 acceptance: the 9.1 layout exists and both scan tools run clean.

These are real tests of S0's deliverable (the scaffold), not placeholders:
they fail if a 9.1 directory disappears or a scan tool stops running.
"""

import copy
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parents[1]

LAYOUT_9_1 = [
    "docs",
    "src/rag4ds/model",
    "src/rag4ds/corpus",
    "src/rag4ds/extract",
    "src/rag4ds/staging",
    "src/rag4ds/gate",
    "src/rag4ds/kb",
    "src/rag4ds/index",
    "src/rag4ds/retrieve",
    "src/rag4ds/evalhook",
    "src/rag4ds/auditlog",
    "packs",
    "shared/general",
    "shared/tasks",
    "journal",
    "stage",
    "eval/gold",
    "runs",
    "tests",
    "tools",
]


def test_layout_9_1_directories_exist():
    missing = [d for d in LAYOUT_9_1 if not (MODULE_ROOT / d).is_dir()]
    assert not missing, f"9.1 directories missing: {missing}"


def test_plan_scan_runs_clean():
    result = subprocess.run(
        [sys.executable, str(MODULE_ROOT / "tools" / "plan_scan.py")],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "non-ASCII: 0" in result.stdout, result.stdout


def test_framework_scan_runs_clean():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "audit" / "tools" / "framework_scan.py")],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "non-exempt flags: 0" in result.stdout, result.stdout


def test_framework_scan_delta_mode_runs():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "audit" / "tools" / "framework_scan.py"), "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout
    assert "== 4." in result.stdout and "== 5." in result.stdout, result.stdout


PROBE = """import os


def test_one(request):
    # by what the plugin does, not by the name it was loaded under (r3 F3):
    # the cache provider gives the session a config.cache, whatever its name
    plugins = request.config.pluginmanager.get_plugins()
    active = hasattr(request.config, "cache") or any(
        getattr(p, "__name__", "") == "_pytest.cacheprovider" for p in plugins)
    with open(os.environ["CACHE_PROBE_OUT"], "w", encoding="utf-8") as f:
        f.write(str(active))
"""


def _tree(root, recursive):
    paths = root.rglob("*") if recursive else root.iterdir()
    return sorted(p.relative_to(root).as_posix() for p in paths if "__pycache__" not in p.parts)


def test_check_py_runs_pytest_with_its_cache_provider_off(tmp_path, monkeypatch):
    """AR-rag4ds-055 r1 O6 to r7 F3 and F5: tools/check.py's steps, as its
    main() makes them through the module's subprocess.run, keep no pytest or
    ruff cache (the repository's operating rules, section 3). What this
    witness binds, and only this:

    (1) STEPS names exactly its six steps, each pinned word for word with its
        cwd; the module root holds no pytest configuration file but
        pyproject.toml.
    (2) Before check.py is loaded, the witness takes os.environ, the bytes (or
        absence) of the held files - the refused configuration names at the
        module root, pyproject.toml and tests/conftest.py - sys.executable,
        and snapshots of the module tree (__pycache__ aside) and of the top of
        the repository root. main(), with subprocess.run replaced by a
        recorder (nothing is executed) and called without and with a base
        argument, must make exactly the calls listed by a copy of STEPS taken
        before it ran, in order, with no keyword but cwd (the base only
        extends framework_scan's command). Every call, and the state after
        each main(), must see the environment and the held files as they were
        before the load; sys.executable and STEPS must be unchanged.
    (3) The pinned pytest command runs on a one-test scratch tree under byte
        copies of the module's pyproject.toml and tests/conftest.py as held; the
        test there records, by what the plugin does rather than its name,
        whether the cache is active. It must be inactive, the scratch tree
        hold nothing beyond its own files, and both trees match the pre-load
        snapshots; a control run of the bare command must show the cache
        active and written, so the probe cannot pass vacuously.

    Outside what this binds (r6 O16; report r7 section 2.3 and F5): a call
    around the module's subprocess.run; a change to a child's environment
    around os.environ (os.putenv, ctypes, Python start-up files); code that
    detects the test or makes the witness's own reads lie; changes to other
    files the steps execute (the scan scripts, modules under src/); and
    pytest configuration from below tests/ - the conftest.py of a directory
    pytest collects, or a test module's pytest_plugins. The net for all of
    these is the CI step after check.py, which fails on any pytest or ruff
    cache left in the tree (.github/workflows/module1.yml)."""
    names = ("pytest.toml", ".pytest.toml", "pytest.ini", ".pytest.ini", "tox.ini", "setup.cfg",
             "conftest.py")
    held = [MODULE_ROOT / n for n in names + ("pyproject.toml", "tests/conftest.py")]

    def state():
        return dict(os.environ), [p.read_bytes() if p.exists() else None for p in held]

    # (2) every baseline before check.py is loaded (r7 F3: I1, I2, I5, X3, L1)
    loaded = state()
    exe = sys.executable
    module_before, root_before = _tree(MODULE_ROOT, True), _tree(REPO_ROOT, False)
    assert not [n for n in names if (MODULE_ROOT / n).exists()]  # r5 O14, r6 F4
    spec = importlib.util.spec_from_file_location("check_tool",
                                                  MODULE_ROOT / "tools" / "check.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    assert sys.executable == exe and state() == loaded
    # (1) all six steps pinned word for word (r7 F3: N1)
    tools = MODULE_ROOT / "tools"
    step = [exe, "-m", "pytest", "-p", "no:cacheprovider"]
    assert list(check.STEPS) == [
        ("ruff", [exe, "-m", "ruff", "check", "--no-cache", "src", "tests", "tools"],
         MODULE_ROOT),
        ("pytest", step, MODULE_ROOT),
        ("nfr6_scan", [exe, str(tools / "nfr6_scan.py"), "--skip-absent"], MODULE_ROOT),
        ("nfr3_scan", [exe, str(tools / "nfr3_scan.py")], MODULE_ROOT),
        ("plan_scan", [exe, str(tools / "plan_scan.py")], REPO_ROOT),
        ("framework_scan", [exe, str(REPO_ROOT / "audit" / "tools" / "framework_scan.py")],
         REPO_ROOT),
    ]
    pinned = copy.deepcopy(check.STEPS)  # r5 F3: STEPS edited in place cannot hide

    class _Done:
        returncode = 0

    for argv in (["check.py"], ["check.py", "78858bd"]):
        ran, seen = [], []

        def record(cmd, *args, _ran=ran, _seen=seen, **kw):
            _ran.append((list(cmd), args, kw))
            _seen.append(state())
            return _Done()

        monkeypatch.setattr(check, "subprocess", type("S", (), {"run": staticmethod(record)}))
        monkeypatch.setattr(sys, "argv", argv)
        assert check.main() == 0
        # r5 F3 (E2), r6 F3 (E4, E5, C3b), r7 F3 (P3b, X3): at every call and after
        assert all(s == loaded for s in seen), argv
        assert state() == loaded and sys.executable == exe, argv
        assert check.STEPS == pinned, argv
        want = [(cmd + (argv[1:] if name == "framework_scan" else []), (), {"cwd": cwd})
                for name, cmd, cwd in pinned]
        assert ran == want, (argv, ran)

    # (3) the probe, under the configuration as held before the load
    env = {k: v for k, v in loaded[0].items() if k != "PYTEST_ADDOPTS"}
    conftest = loaded[1][held.index(MODULE_ROOT / "tests/conftest.py")]
    pyproject = loaded[1][held.index(MODULE_ROOT / "pyproject.toml")]
    for name, cmd, active in (("step", step, "False"), ("control", step[:3], "True")):
        tree = tmp_path / name
        tree.mkdir()
        (tree / "test_one.py").write_text(PROBE, encoding="utf-8")
        (tree / "conftest.py").write_bytes(conftest)
        # the module's pyproject.toml as held, byte for byte, beside the probe: with -c
        # naming the module's own file, pytest collected from the common ancestor of the
        # module and this scratch tree - the user's profile folder - and listed every
        # folder under AppData, so a temp folder another program removed mid-walk failed
        # the run (2026-10-05, record 71); the copy keeps collection inside the tree
        (tree / "pyproject.toml").write_bytes(pyproject)
        config = ["-c", str(tree / "pyproject.toml")]
        out = tmp_path / f"{name}.probe"
        run = subprocess.run(cmd + config + ["--rootdir", str(tree), "-q", "test_one.py"],
                             cwd=tree, capture_output=True, text=True,
                             env={**env, "CACHE_PROBE_OUT": str(out)})
        assert run.returncode == 0, run.stdout + run.stderr
        assert out.read_text(encoding="utf-8") == active, (name, run.stdout)
        extra = set(_tree(tree, False)) - {"test_one.py", "conftest.py", "pyproject.toml"}
        assert extra == ({".pytest_cache"} if name == "control" else set()), (name, extra)
        assert _tree(MODULE_ROOT, True) == module_before, name
        assert _tree(REPO_ROOT, False) == root_before, name
        assert state() == loaded, name


def test_rag4ds_package_is_installed():
    import rag4ds  # noqa: F401  - AR-rag4ds-008 F6: the increment installs its own package


def test_locks_agree_on_common_packages():
    # AR-rag4ds-008 F17: the Windows lock (VT-3 pins) and the Linux lock must
    # agree wherever they overlap.
    def parse(name):
        pins = {}
        for line in (MODULE_ROOT / name).read_text(encoding="utf-8").splitlines():
            if "==" in line:
                pkg, ver = line.strip().split("==", 1)
                pins[pkg.lower()] = ver
        return pins

    win, linux = parse("requirements.lock"), parse("requirements-linux.lock")
    common = set(win) & set(linux)
    assert len(common) > 50, f"suspiciously small overlap: {len(common)}"
    disagree = {p: (win[p], linux[p]) for p in common if win[p] != linux[p]}
    assert not disagree, f"locks disagree on: {disagree}"

    # VT-3b: direct-URL pins (name @ url#sha256=...) are invisible to the "=="
    # parse above, so they are compared separately - byte for byte, both ways.
    def urls(name):
        return {line.split(" @ ", 1)[0].lower(): line.strip()
                for line in (MODULE_ROOT / name).read_text(encoding="utf-8").splitlines()
                if " @ " in line}

    win_urls, linux_urls = urls("requirements.lock"), urls("requirements-linux.lock")
    assert win_urls == linux_urls, f"direct-URL pins differ: {win_urls} vs {linux_urls}"
    assert "en_core_web_sm" in win_urls and "#sha256=" in win_urls["en_core_web_sm"]


def test_module_docs_and_code_carry_no_control_bytes():
    """AR-rag4ds-024 F6: a vertical tab and a carriage return once reached a
    committed record through an interpreted backslash path; the ASCII
    census cannot see bytes below 128, so this test can. TAB and LF are
    text; CR only as part of a CRLF line ending (autocrlf checkouts) - a
    bare CR inside a line is exactly F6's shape."""
    files = sorted((MODULE_ROOT / "docs").glob("*.md"))
    for sub in ("src", "tests", "tools"):
        files += sorted((MODULE_ROOT / sub).rglob("*.py"))
    offenders = {}
    for path in files:
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        data = path.read_bytes()
        hits = [i for i, c in enumerate(data)
                if (c < 32 and c not in (9, 10, 13)) or c == 127
                or (c == 13 and data[i + 1:i + 2] != b"\n")]
        if hits:
            offenders[str(path.relative_to(MODULE_ROOT))] = hits[:5]
    assert files, "nothing scanned"
    assert not offenders, f"control bytes in: {offenders}"


def test_canonical_artefacts_are_never_eol_converted():
    """I1: the journals and the gated trees are byte-canonical and are checked
    out unchanged on every platform - .gitattributes marks them -text. A fresh
    autocrlf checkout on Windows once turned journal/decisions.jsonl into CRLF
    and the live reconstruction test refused it (S8, 2026-09-12); the same
    checkout rewrote the catalogued source note under authoring/ and the gate
    refused every candidate anchored to it (AR-rag4ds-026 F4)."""
    paths = ["journal/decisions.jsonl", "stage/candidates.jsonl", "runs/ingest.jsonl",
             "shared/general/x.json", "shared/tasks/x.json", "packs/robotics/entries/x.json",
             "authoring/shared_general.yaml", "authoring/sources/shared-task-patterns.md"]
    out = subprocess.run(["git", "check-attr", "text", "--", *paths], cwd=MODULE_ROOT,
                         capture_output=True, text=True, check=True).stdout
    lines = out.strip().splitlines()
    assert len(lines) == len(paths), out
    for line in lines:
        assert line.endswith(": text: unset"), line


def _git_bytes(rel: str) -> bytes:
    """The bytes git holds for a tracked file (its index blob), conversion-free."""
    out = subprocess.run(["git", "ls-files", "-s", "--", rel], cwd=MODULE_ROOT,
                         capture_output=True, text=True, check=True).stdout.split()
    assert out, f"{rel} is not tracked"
    return subprocess.run(["git", "cat-file", "blob", out[1]], cwd=MODULE_ROOT,
                          capture_output=True, check=True).stdout


def test_in_git_corpus_documents_check_out_with_their_catalogued_bytes():
    """AR-rag4ds-026 F4: a catalogued document that lives IN git (the team's
    own source note) is manifested and cited by hash; the working-tree bytes
    must equal git's blob and hash to the catalog entry, on every checkout.
    Every tracked file under authoring/ is held to the same byte identity,
    because the drafts are hashed into the authoring manifests."""
    import hashlib
    import json

    import yaml

    catalog = yaml.safe_load((MODULE_ROOT / "shared/corpus/catalog.yaml").read_text("utf-8"))
    by_hash = {d["hash"]: d for d in catalog["documents"]}
    manifests = [json.loads(line) for line in
                 (MODULE_ROOT / "runs/ingest.jsonl").read_text("utf-8").splitlines()
                 if line.strip()]
    in_git = []
    for m in manifests:
        if m["pack"] != "shared":
            continue
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", m["local_path"]],
                                 cwd=MODULE_ROOT, capture_output=True).returncode == 0
        if tracked:
            in_git.append(m)
            working = (MODULE_ROOT / m["local_path"]).read_bytes()
            assert working == _git_bytes(m["local_path"]), (
                f"{m['local_path']}: working-tree bytes differ from git's blob (end-of-line "
                f"conversion?)")
            assert hashlib.sha256(working).hexdigest() == m["hash"] == by_hash[m["hash"]]["hash"]
    assert in_git, "the in-git source note is expected to be catalogued and manifested"
    tracked = subprocess.run(["git", "ls-files", "-z", "--", "authoring"], cwd=MODULE_ROOT,
                             capture_output=True, check=True).stdout.decode().split("\0")
    tracked = [t for t in tracked if t]
    assert tracked
    for rel in tracked:
        # a draft may be legitimately edited before its commit, so the working copy is
        # held to LF canonical form (git's blob is LF), not to blob identity
        assert b"\r" not in (MODULE_ROOT / rel).read_bytes(), (
            f"{rel}: carriage return in the working copy (end-of-line conversion?)")
        assert b"\r" not in _git_bytes(rel), f"{rel}: carriage return in git's blob"
