"""Build the gate's index over the current tree (plan component 5; the S6
gate's --index-dir for its conflict display; S9c). Generic: every pack
present in the tree is configured at the version its manifest declares, the
shared layers are always in, the embedder is the VT-3 pin. Run from the
module root with the module venv:

    python tools/build_index.py [--out data/index] [--min-relevance 0.2]

The index is DERIVED (never authoritative, rebuilt from the tree) and lives
under the gitignored data/ by default; its NFR2 manifest (index version,
model pin, entry count, tree digest, code commit, timestamp) is printed and
the gate records the index version on every decision (AR-rag4ds-026 F5).
Until S9c the S8 sessions' index was built by an uncommitted snippet; this
tool is the reproducible road.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))
sys.path.insert(0, str(MODULE_ROOT / "tools"))  # release_pack, for the dev-load marking
from rag4ds.index.build import (  # noqa: E402
    PINNED_EMBEDDING_MODEL, PINNED_EMBEDDING_REVISION, IndexBuildError,
    SentenceTransformerEmbedder, build_index, pack_dirs,
)
from rag4ds.model.pack import PackManifestError, load_pack_manifest  # noqa: E402

DEFAULT_MIN_RELEVANCE = 0.2  # the S8 sessions' value (data/index/manifest.json at 0b434b9)


def jst_now() -> str:
    if time.localtime().tm_gmtoff != 9 * 3600:
        raise IndexBuildError("this machine's local clock is not JST (UTC+9); refusing")
    return time.strftime("%Y-%m-%d %H:%M:%S JST")


def head_commit(module_root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=module_root,
                          capture_output=True, text=True, check=True).stdout.strip()


UNRELEASED = "+unreleased"


def pack_versions_from_manifests(module_root: Path) -> dict:
    """Every pack the index build sees in the tree, at its manifest's version - and a
    RELEASED pack whose gated content (its entries and mapping artifacts) is not the
    content its release manifest hashed is labelled "<version>+unreleased" (the dev-load
    marking, AR-rag4ds-059 r1 O1): an entry gated after a release is never served under
    the released version's label. A RELEASED pack with no release manifest is refused."""
    from release_pack import gated_content_hash, shipped_files  # tools/, beside this file

    module_root = Path(module_root)
    out = {}
    for code in pack_dirs(module_root):
        m = load_pack_manifest(module_root / "packs" / code)
        out[code] = m.version
        if m.state != "RELEASED":
            continue
        rel = module_root / "packs" / code / "releases" / f"{code}-{m.version}.json"
        if not rel.is_file():
            raise IndexBuildError(f"pack {code!r} is RELEASED at {m.version}, but its release "
                                  f"manifest {rel.relative_to(module_root).as_posix()} is missing")
        released = json.loads(rel.read_text(encoding="utf-8"))["gated_content_sha256"]
        if gated_content_hash(module_root, shipped_files(module_root, code))[0] != released:
            out[code] = m.version + UNRELEASED
    return out


def run(module_root: Path, out_dir: Path, min_relevance: float, embedder=None,
        stream=None, clock=None) -> int:
    """clock: a callable returning the NFR2 timestamp; the machine's JST
    clock by default (refused when not JST), injectable for a UTC runner."""
    stream = stream or sys.stdout  # resolved at call time (pytest's capture)
    module_root = Path(module_root)
    try:
        versions = pack_versions_from_manifests(module_root)
        emb = embedder or SentenceTransformerEmbedder(PINNED_EMBEDDING_MODEL,
                                                      PINNED_EMBEDDING_REVISION)
        manifest = build_index(module_root, Path(out_dir), emb, versions, min_relevance,
                               head_commit(module_root), (clock or jst_now)())
    except (IndexBuildError, PackManifestError) as exc:
        print(f"build_index: REFUSED - {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, sort_keys=True), file=stream)
    for code, v in versions.items():
        if v.endswith(UNRELEASED):
            print(f"build_index: pack {code!r} differs from its release "
                  f"{v[:-len(UNRELEASED)]} - served as {v}", file=stream)
    print(f"build_index: {manifest['entry_count']} entries, packs {versions}, index version "
          f"{manifest['index_version']} at {manifest['code_commit']}, {manifest['timestamp']} "
          f"-> {out_dir}", file=stream)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(MODULE_ROOT / "data" / "index"))
    ap.add_argument("--min-relevance", type=float, default=DEFAULT_MIN_RELEVANCE)
    args = ap.parse_args(argv)
    return run(MODULE_ROOT, Path(args.out), args.min_relevance)


if __name__ == "__main__":
    sys.exit(main())
