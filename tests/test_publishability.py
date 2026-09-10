"""NFR5 publishability check (9.2): the AutonBDA subtree stays free of
secrets at every commit. Grep-based, covering the obvious credential
patterns; the corpus rule (NFR6) handles restricted-license content.

Scans only git-tracked files, so .venv/ and data/ (gitignored) are out of
scope by construction.
"""

import re
import subprocess
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
SUBTREE_ROOT = MODULE_ROOT.parent  # AutonBDA/

SECRET_PATTERNS = [
    re.compile(rb"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(rb"ghp_[A-Za-z0-9]{36}"),  # GitHub PAT
    re.compile(rb"github_pat_[A-Za-z0-9_]{22,}"),
    re.compile(rb"hf_[A-Za-z0-9]{34}"),  # Hugging Face token
    re.compile(rb"sk-[A-Za-z0-9]{20,}T3BlbkFJ[A-Za-z0-9]{20,}"),  # OpenAI key
    re.compile(rb"xox[baprs]-[A-Za-z0-9-]{10,}"),  # Slack token
]


def tracked_files():
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", str(SUBTREE_ROOT)],
        capture_output=True,
        check=True,
        cwd=SUBTREE_ROOT,
    ).stdout
    return [SUBTREE_ROOT / p.decode("utf-8") for p in out.split(b"\0") if p]


def test_sweep_is_not_vacuous():
    files = tracked_files()
    assert len(files) > 10, f"suspiciously few tracked files: {len(files)}"


def test_no_secret_patterns_in_tracked_subtree():
    hits = []
    for path in tracked_files():
        data = path.read_bytes()
        for pat in SECRET_PATTERNS:
            if pat.search(data):
                hits.append((str(path), pat.pattern.decode()))
    assert not hits, f"secret-pattern hits in AutonBDA subtree: {hits}"
