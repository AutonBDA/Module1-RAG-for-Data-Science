"""Pack release (plan 4.3 and 9.3; the S9 demo road's Part C): the release
checklist, the release manifest and the validation snapshot for one version of
one pack. Run from the module root:

    python tools/release_pack.py --pack robotics --version 1.0.0 --dry-run
    python tools/release_pack.py --pack robotics --version 1.0.0

The checklist (plan 4.3), every item required - one failure refuses the
release and writes nothing:
  gated     the live tree is the fold of the decision journal: every entry and
            mapping artifact the release ships was approved at the gate (NFR1)
  licences  every entry of the pack cites a document of the pack's own
            catalogue (corpus:<pack>/<stem>), whose record carries its licence
  lexicon   the pack's lexicon is declared, and the NFR3 independence scan is
            clean over the core and both shared layers
  demo      an index built from the working tree - the dev load - answers every
            demo flow of the pack as bound (tools/run_demo.py's verdicts, all PASS)
  version   the version is X.Y.Z, above the manifest's, and not released before
The release writes, in one step, after every item holds:
  packs/<pack>/releases/<pack>-<version>.json  the release manifest: the
            gated-content hash (entries and mapping artifacts together), every
            shipped file's sha256, the checklist's results, the demo's index and
            verdicts, the snapshot's hash, the journal's hash, commit and time
  packs/<pack>/validation/<version>/decisions.jsonl  the gate decisions behind
            the shipped entries and mapping artifacts, as the journal holds them
  packs/<pack>/validation/<version>/SHA256SUMS  the snapshot's checksum
  packs/<pack>/manifest.yaml  version -> <version>, state -> RELEASED
The git tag pack-<pack>-v<version> goes on the commit that holds those files -
a separate step after that commit (plan 9.3); this tool does not tag.
Vector indexes are not release artifacts (4.3): the demo's index is built into
a scratch directory and only its version is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))
sys.path.insert(0, str(MODULE_ROOT / "tools"))

from rag4ds.auditlog.journal import Journal  # noqa: E402
from rag4ds.gate.cli import DECISION_JOURNAL  # noqa: E402
from rag4ds.gate.decisions import APPROVAL_TYPES  # noqa: E402
from rag4ds.kb.fold import mapping_artifact_bytes, mapping_tree_path, verify_reconstruction  # noqa: E402
from rag4ds.model.pack import PackManifestError, load_all_pack_manifests  # noqa: E402

VERSION_RE = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


class ReleaseError(RuntimeError):
    """The release is refused; nothing was written."""


def jst_now() -> str:
    if time.localtime().tm_gmtoff != 9 * 3600:
        raise SystemExit("release_pack: this machine's local clock is not JST (UTC+9); refusing")
    return time.strftime("%Y-%m-%d %H:%M:%S JST")


def head_commit(module_root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=module_root,
                          capture_output=True, text=True, check=True).stdout.strip()


# what a release reads and names, so what must be committed when it runs: the gated
# content (AR-rag4ds-059 r1 O2), and the ingest record the licences item reads and the
# code the manifest's code_commit names (r2 O8)
RELEASE_INPUTS = ("shared", "journal", "runs/ingest.jsonl", "src", "tools")


def dirty_paths(module_root: Path, pack: str) -> list[str]:
    """Paths with changes git has not committed - staged, unstaged or untracked -
    under packs/<pack> and RELEASE_INPUTS: the manifest names HEAD, so what it
    hashes, reads and runs must be HEAD's."""
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--",
                          f"packs/{pack}", *RELEASE_INPUTS], cwd=module_root,
                         capture_output=True, text=True, check=True).stdout
    return [line[3:] for line in out.splitlines() if line.strip()]


def _semver(v: str) -> tuple[int, int, int]:
    m = VERSION_RE.fullmatch(v)
    if not m:
        raise ReleaseError(f"version {v!r} is not X.Y.Z")
    return tuple(int(x) for x in m.groups())


def shipped_files(module_root: Path, pack: str) -> list[Path]:
    """The gated content a release ships: the pack's entries and mapping artifacts."""
    root = Path(module_root) / "packs" / pack
    return sorted([*(root / "entries").glob("*.json"), *(root / "mappings").glob("*.json")])


def gated_content_hash(module_root: Path, files: list[Path]) -> tuple[str, list[dict]]:
    """sha256 over each shipped file's path and sha256, in path order - entries and
    mapping artifacts together (4.3); any byte of any file, or a file added or
    removed, changes it."""
    listing = [{"path": p.relative_to(module_root).as_posix(),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    body = "".join(f"{f['path']}\0{f['sha256']}\n" for f in listing)
    return hashlib.sha256(body.encode("utf-8")).hexdigest(), listing


def check_licences(module_root: Path, pack: str) -> list[str]:
    """Every entry cites corpus:<pack>/<stem>; the stem is a document this pack
    ingested (runs/ingest.jsonl), and the pack's catalogue records its licence."""
    import yaml

    problems = []
    ingested = {}
    for rec in Journal(Path(module_root) / "runs" / "ingest.jsonl").read_all():
        if rec.get("kind") == "ingest" and rec.get("pack") == pack:
            ingested[Path(rec["local_path"]).stem] = rec["hash"]
    cat = yaml.safe_load((Path(module_root) / "packs" / pack / "corpus" / "catalog.yaml")
                         .read_text(encoding="utf-8"))
    licence = {d["hash"]: (d.get("license") or "").strip() for d in cat["documents"]}
    for p in sorted((Path(module_root) / "packs" / pack / "entries").glob("*.json")):
        ref = json.loads(p.read_text(encoding="utf-8"))["source_reference"]
        prefix = f"corpus:{pack}/"
        if not ref.startswith(prefix):
            problems.append(f"{p.name}: source_reference {ref!r} does not cite this pack's corpus")
            continue
        h = ingested.get(ref[len(prefix):])
        if h is None:
            problems.append(f"{p.name}: {ref!r} names no document this pack ingested")
        elif not licence.get(h):
            problems.append(f"{p.name}: {ref!r} has no licence in the pack's catalogue")
    return problems


def check_lexicon(module_root: Path, pack: str) -> list[str]:
    import nfr3_scan

    man = load_all_pack_manifests(Path(module_root)).get(pack)
    if man is None:
        return [f"no manifest for pack {pack!r}"]
    problems = [] if man.lexicon else [f"pack {pack!r} declares no lexicon"]
    try:
        hits, _notes = nfr3_scan.run(Path(module_root))  # (hits, per-pack summary notes)
    except nfr3_scan.Nfr3Error as exc:  # a scan that cannot run is a failure, never a pass
        return problems + [f"NFR3 scan could not run: {exc}"]
    return problems + [f"NFR3: {h}" for h in hits]


def check_demo(module_root: Path, pack: str, embedder=None) -> tuple[list[str], dict]:
    """Build the dev load into a scratch directory and run the pack's demo flows."""
    import io

    import build_index
    import run_demo
    from rag4ds.index.build import load_index

    with tempfile.TemporaryDirectory(prefix="release-demo-") as tmp:
        out = io.StringIO()
        rc = build_index.run(Path(module_root), Path(tmp), build_index.DEFAULT_MIN_RELEVANCE,
                             embedder=embedder, stream=out)
        if rc != 0:
            return [f"the dev-load index did not build (exit {rc}): {out.getvalue().strip()}"], {}
        version = load_index(Path(tmp)).manifest["index_version"]
        demo_out = io.StringIO()
        rc = run_demo.execute(Path(module_root), pack, Path(tmp), embedder=embedder,
                              stream=demo_out)
    lines = [x for x in demo_out.getvalue().splitlines() if x.strip()]
    info = {"index_version": version, "exit": rc, "lines": lines}
    return ([] if rc == 0 else [f"demo flows: run_demo exit {rc} - not every flow PASSES"]), info


def snapshot(module_root: Path, pack: str, files: list[Path]) -> list[str]:
    """The journal lines of the approvals behind the shipped files: for an entry,
    the approval of its id at its version; for a mapping artifact, the approval
    whose artifact is the file's bytes. One line per shipped file, or refuse."""
    lines = (Path(module_root) / DECISION_JOURNAL).read_text(encoding="utf-8").splitlines()
    recs = [(ln, json.loads(ln)) for ln in lines if ln.strip()]
    out = []
    for p in files:
        rel = p.relative_to(Path(module_root) / "packs" / pack)
        data = p.read_bytes()
        found = []
        if rel.parts[0] == "entries":
            e = json.loads(data)
            found = [ln for ln, r in recs if r.get("type") in APPROVAL_TYPES
                     and isinstance(r.get("entry"), dict) and r["entry"].get("id") == e["id"]
                     and r["entry"].get("version") == e["version"]
                     and r["entry"].get("pack") == pack]
        else:
            found = [ln for ln, r in recs if r.get("type") in APPROVAL_TYPES
                     and isinstance(r.get("mapping_artifact"), dict)
                     and mapping_tree_path(r).name == p.name
                     and mapping_artifact_bytes(r["mapping_artifact"]) == data]
        if not found:
            raise ReleaseError(f"no approval in the journal ships {rel.as_posix()}")
        out.append(found[-1])
    return out


def release(module_root: Path, pack: str, version: str, dry_run: bool, clock=None,
            commit=None, embedder=None, checks=None, stream=None) -> dict:
    """Run the checklist; unless dry_run, write the release. Returns the manifest.
    checks: {name: callable} to replace a checklist item (tests)."""
    stream = stream or sys.stdout
    module_root = Path(module_root)
    ts = (clock or jst_now)()
    if commit is None:
        # a real run: refuse gated content git has not committed, then name HEAD (a
        # commit passed in is a test's stand-in for a tree that is not a repository)
        dirty = dirty_paths(module_root, pack)
        if dirty:
            raise ReleaseError(f"uncommitted changes in the gated content - commit them "
                               f"first: {', '.join(dirty[:5])}")
        commit = head_commit(module_root)
    man_path = module_root / "packs" / pack / "manifest.yaml"
    try:
        manifests = load_all_pack_manifests(module_root)
    except PackManifestError as exc:  # e.g. an empty lexicon - the loader refuses it first
        raise ReleaseError(f"a pack manifest does not validate: {exc}") from exc
    if pack not in manifests:
        raise ReleaseError(f"no pack {pack!r} under packs/")
    current = manifests[pack].version
    if _semver(version) <= _semver(current):
        raise ReleaseError(f"version {version} is not above the manifest's {current}")
    rel_dir = module_root / "packs" / pack / "releases"
    out_manifest = rel_dir / f"{pack}-{version}.json"
    val_dir = module_root / "packs" / pack / "validation" / version
    if out_manifest.exists() or val_dir.exists():
        raise ReleaseError(f"{pack} {version} was released before ({out_manifest.name} or "
                           f"validation/{version} exists)")
    run = dict(checks or {})
    results, problems = {}, []
    gated = run.get("gated", lambda: verify_reconstruction(
        Journal(module_root / DECISION_JOURNAL), module_root))()
    results["gated"] = not gated
    problems += [f"gated: {p}" for p in gated]
    lic = run.get("licences", lambda: check_licences(module_root, pack))()
    results["licences"] = not lic
    problems += [f"licences: {p}" for p in lic]
    lex = run.get("lexicon", lambda: check_lexicon(module_root, pack))()
    results["lexicon"] = not lex
    problems += [f"lexicon: {p}" for p in lex]
    demo_problems, demo = run.get("demo", lambda: check_demo(module_root, pack, embedder))()
    results["demo"] = not demo_problems
    problems += demo_problems
    files = shipped_files(module_root, pack)
    if not files:
        problems.append("the pack ships no entry and no mapping artifact")
    for name, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'} {name}", file=stream)
    if problems:
        raise ReleaseError("checklist failed - nothing written:\n  " + "\n  ".join(problems))
    snap_lines = snapshot(module_root, pack, files)
    snap_bytes = ("\n".join(snap_lines) + "\n").encode("utf-8")
    snap_sha = hashlib.sha256(snap_bytes).hexdigest()
    content_sha, listing = gated_content_hash(module_root, files)
    journal_raw = (module_root / DECISION_JOURNAL).read_bytes()
    manifest = {
        "kind": "pack-release", "schema": 1, "pack": pack, "version": version,
        "previous_version": current, "tag": f"pack-{pack}-v{version}",
        "gated_content_sha256": content_sha, "files": listing,
        "checklist": results, "demo": demo,
        "validation_snapshot": {"path": f"packs/{pack}/validation/{version}/decisions.jsonl",
                                "sha256": snap_sha, "decisions": len(snap_lines)},
        "journal_records": sum(1 for x in journal_raw.decode("utf-8").splitlines() if x.strip()),
        "journal_sha256": hashlib.sha256(journal_raw).hexdigest(),
        "evaluation_run": "none - the evaluation hook is S10, not yet active (4.3)",
        "code_commit": commit, "timestamp": ts, "tool": "tools/release_pack.py",
    }
    if dry_run:
        print(f"release_pack: dry run - {pack} {version} would ship {len(files)} files, gated "
              f"content {content_sha[:16]}...; nothing written", file=stream)
        return manifest
    text = man_path.read_text(encoding="utf-8")
    new_text, n1 = re.subn(rf"(?m)^version: {re.escape(current)}$", f"version: {version}", text)
    new_text, n2 = re.subn(r"(?m)^state: [A-Z]+$", "state: RELEASED", new_text)
    if (n1, n2) != (1, 1):
        raise ReleaseError(f"{man_path.name}: expected one version line and one state line")
    rel_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True)
    (val_dir / "decisions.jsonl").write_bytes(snap_bytes)
    (val_dir / "SHA256SUMS").write_text(f"{snap_sha}  decisions.jsonl\n", encoding="utf-8",
                                         newline="\n")
    out_manifest.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n",
                            encoding="utf-8", newline="\n")
    man_path.write_text(new_text, encoding="utf-8", newline="\n")
    print(f"release_pack: {pack} {version} released in the tree - {len(files)} files, gated "
          f"content {content_sha[:16]}...; commit these files, then tag the commit "
          f"{manifest['tag']}", file=stream)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    try:
        release(MODULE_ROOT, args.pack, args.version, args.dry_run)
    except ReleaseError as exc:
        print(f"release_pack: REFUSED - {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
