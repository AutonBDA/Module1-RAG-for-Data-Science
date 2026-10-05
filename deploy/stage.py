"""Write the build folder for the public demo container (deploy/README.md): the
demo's committed files at one commit, taken with git archive, plus the served
index from data/index - checked against its own manifest - and nothing else. No
working-tree file, no corpus text, no key. Run from the module root:

    python deploy/stage.py --out C:/dev/rag4ds-deploy [--commit HEAD]

The folder is emptied first; it must lie outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parents[1]
MODULE = "AutonBDA/module1_rag_for_ds"
# what the demo reads (tools/demo_web.py): the library, its two tools, the robotics
# pack's manifest and demo questions, and the Linux lock for the image
FILES = ["src", "tools/demo_web.py", "tools/answer_writer.py", "packs/robotics/manifest.yaml",
         "packs/robotics/demo", "requirements-linux.lock", "deploy/Dockerfile"]
INDEX_FILES = ["manifest.json", "structural.sqlite", "vectors.npy"]


class StageError(RuntimeError):
    pass


def stage(out: Path, commit: str, index_dir: Path) -> dict:
    out = out.resolve()
    if out == REPO_ROOT or REPO_ROOT in out.parents:
        raise StageError(f"{out} lies inside the repository; stage outside it")
    sha = subprocess.run(["git", "rev-parse", commit], cwd=REPO_ROOT, capture_output=True,
                         text=True, check=True).stdout.strip()
    tar = subprocess.run(["git", "archive", "--format=tar", f"{sha}:{MODULE}", *FILES],
                         cwd=REPO_ROOT, capture_output=True, check=True).stdout
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        tf.extractall(out, filter="data")
    (out / "deploy" / "Dockerfile").replace(out / "Dockerfile")
    (out / "deploy").rmdir()
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    vec = hashlib.sha256((index_dir / "vectors.npy").read_bytes()).hexdigest()
    if vec != manifest["vectors_sha256"]:
        raise StageError("the served index's vectors do not match its manifest")
    (out / "data" / "index").mkdir(parents=True)
    for name in INDEX_FILES:
        shutil.copy2(index_dir / name, out / "data" / "index" / name)
    names = [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()]
    bad = [n for n in names if "conf.info" in n or n.startswith(("data/robotics", "data/corpus"))]
    if bad:
        raise StageError(f"refused to stage {bad}")
    return {"commit": sha, "files": len(names), "index_version": manifest["index_version"],
            "pack_versions": manifest["pack_versions"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--index-dir", default=str(MODULE_ROOT / "data" / "index"))
    args = ap.parse_args(argv)
    try:
        info = stage(Path(args.out), args.commit, Path(args.index_dir))
    except (StageError, subprocess.CalledProcessError, OSError, KeyError) as exc:
        print(f"stage: REFUSED - {exc}", file=sys.stderr)
        return 2
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
