"""file_mapping_draft.py - file a pack's external-standard mapping draft as a
mapping-draft candidate for the pack's validator (plan 4.5, FR11; S9d).

Generic: the pack is data. The draft is a human-edited YAML file (exactly
the artifact the gate will canonicalise: `standard` and `rows`, each row
entry/class/relation/note); the pack's packs/<code>/standards.yaml names
the standards it may map to, pins each standard's file by sha256 and lists
the classes a row may name (verified against the standard's text - VT-1).
Before anything is filed the tool refuses, by name:

- a draft whose top-level keys are not exactly standard and rows;
- a standard the registry does not list;
- a row whose entry is not a CURRENT entry of the pack's tree
  (packs/<code>/entries/<entry>.json);
- a row whose class is not in the standard's registered class list (a
  registered RELATION is refused with that word - a row maps to a class);
- a relation outside equivalent | narrower-than | related, a blank note,
  two rows on one (entry, class) pair;
- an author the identity registry does not know;
- a standard file absent from the pinned path (raw_dir inside the module
  root) or hashing differently; a draft path that does not exist.

Then it files ONE candidate {kind: mapping-draft, pack, standard, author,
yaml_draft} - the draft's text verbatim, so the gate hashes and parses the
bytes the author wrote - and appends a run manifest (kind
mapping-authoring) to runs/authoring.jsonl, which names any targeted class
the standard declares only inside a worked example. A draft already filed
(same bytes, same author) is refused: candidates are immutable, and a
changed draft is a new candidate - unless the earlier run was interrupted
between the candidate and its manifest, in which case the re-run appends
the missing manifest, marked recovered, and says so (AR-rag4ds-042 r1 F2).
Nothing here approves anything: the validator
decides the candidate at the gate (key a in a live session; a sheet
refuses approvals), the fold then writes packs/<code>/mappings/
<standard>.json from the decision.

    python tools/file_mapping_draft.py --pack robotics \
        --draft authoring/robotics_cora-ieee-1872_mapping.yaml \
        --author "<drafting author>" [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import subprocess
import sys
import time
from pathlib import Path

import yaml

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))

from rag4ds.auditlog.journal import Journal  # noqa: E402
from rag4ds.gate.identities import IDENTITIES_FILE, IdentityError, IdentityRegistry  # noqa: E402
from rag4ds.model.entry import ENTRY_ID_RE  # noqa: E402
from rag4ds.staging.store import StagingError, StagingStore, candidate_id  # noqa: E402

REGISTRY_FILE = "standards.yaml"
REGISTRY_SCHEMA_VERSION = 1
STANDARD_REQUIRED = {"title", "publisher", "year", "file", "sha256", "pages", "license",
                     "verified", "classes"}
STANDARD_OPTIONAL = {"relations", "sub_ontologies", "concepts", "note"}
TERM_REQUIRED = {"name", "clause"}
TERM_OPTIONAL = {"parent", "spelled", "example", "note"}
ROW_FIELDS = {"entry", "class", "relation", "note"}
RELATIONS = ("equivalent", "narrower-than", "related")
HEX = set("0123456789abcdef")


class MappingDraftError(ValueError):
    """Raised loudly for anything that would file a wrong candidate."""


def _bare_name(v) -> bool:
    return isinstance(v, str) and bool(v) and "/" not in v and chr(92) not in v and v not in (
        ".", "..") and ":" not in v


def _term_name_ok(name) -> bool:
    """<prefix>:<Name> - a lowercase sub-ontology prefix, a colon, then the
    term as the standard spells it (letters and digits)."""
    if not isinstance(name, str) or name.count(":") != 1:
        return False
    prefix, term = name.split(":")
    return prefix.isalpha() and prefix.islower() and bool(term) and term.isalnum() and (
        term[0].isalpha())


def _sha_ok(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and set(v) <= HEX


def _terms(problems: list, std_id: str, key: str, items) -> dict:
    """Validate a classes/relations list; return name -> term."""
    out: dict = {}
    if not isinstance(items, list):
        problems.append(f"{std_id}: {key} must be a list")
        return out
    for i, t in enumerate(items):
        where = f"{std_id}: {key}[{i}]"
        if not isinstance(t, dict):
            problems.append(f"{where} must be a mapping")
            continue
        missing = sorted(TERM_REQUIRED - set(t))
        unknown = sorted(set(t) - TERM_REQUIRED - TERM_OPTIONAL)
        if missing or unknown:
            problems.append(f"{where}: missing {missing}, unknown {unknown}")
            continue
        if not _term_name_ok(t["name"]):
            problems.append(f"{where}: name {t['name']!r} must be prefix:Term")
            continue
        if not isinstance(t["clause"], str) or not t["clause"].strip():
            problems.append(f"{where}: clause must be a non-blank string")
        if t["name"] in out:
            problems.append(f"{where}: name {t['name']!r} listed twice")
        out[t["name"]] = t
    return out


def load_registry(pack_root: Path) -> dict:
    """Read and validate packs/<code>/standards.yaml. Returns the parsed
    registry with, per standard, `class_names` and `relation_names` sets."""
    pack_root = Path(pack_root)
    path = pack_root / REGISTRY_FILE
    if not path.is_file():
        raise MappingDraftError(f"{path} is absent: the pack registers no external standard")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MappingDraftError(f"{path}: not readable YAML: {exc}") from exc
    problems: list[str] = []
    if not isinstance(data, dict):
        raise MappingDraftError(f"{path}: the registry is a mapping")
    if data.get("schema") != REGISTRY_SCHEMA_VERSION:
        problems.append(f"schema must be {REGISTRY_SCHEMA_VERSION}")
    if data.get("pack") != pack_root.name:
        problems.append(f"pack {data.get('pack')!r} is not the directory's {pack_root.name!r}")
    raw_dir = data.get("raw_dir")
    if not isinstance(raw_dir, str) or not raw_dir.strip() or raw_dir.startswith(("/", chr(92))):
        problems.append("raw_dir must be a relative directory path")
    else:
        # AR-rag4ds-042 r1 F3: the standard's copy lives INSIDE the module
        # (packs/<code>/ is two levels below the module root), like `file`
        module_root = pack_root.resolve().parents[1]
        target = (module_root / raw_dir).resolve()
        if target != module_root and module_root not in target.parents:
            problems.append(f"raw_dir {raw_dir!r} must stay inside the module root "
                            f"(resolves to {target})")
    stds = data.get("standards")
    if not isinstance(stds, dict) or not stds:
        problems.append("standards must be a non-empty mapping of standard id -> standard")
        stds = {}
    hashes: dict[str, str] = {}
    for std_id, std in stds.items():
        if not isinstance(std_id, str) or not ENTRY_ID_RE.fullmatch(std_id):
            problems.append(f"standard id {std_id!r} must obey the entry-id character rule (4.5)")
            continue
        if not isinstance(std, dict):
            problems.append(f"{std_id}: must be a mapping")
            continue
        missing = sorted(STANDARD_REQUIRED - set(std))
        unknown = sorted(set(std) - STANDARD_REQUIRED - STANDARD_OPTIONAL)
        if missing or unknown:
            problems.append(f"{std_id}: missing {missing}, unknown {unknown}")
            continue
        for key in ("title", "publisher", "license", "verified"):
            if not isinstance(std[key], str) or not std[key].strip():
                problems.append(f"{std_id}: {key} must be a non-blank string")
        if not isinstance(std["year"], int) or not isinstance(std["pages"], int) or (
                std["pages"] <= 0):
            problems.append(f"{std_id}: year and pages must be integers (pages positive)")
        if not _bare_name(std["file"]):
            problems.append(f"{std_id}: file must be a bare file name")
        if not _sha_ok(std["sha256"]):
            problems.append(f"{std_id}: sha256 must be 64 lowercase hex")
        elif std["sha256"] in hashes:
            problems.append(f"{std_id}: sha256 already pinned by {hashes[std['sha256']]!r}")
        else:
            hashes[std["sha256"]] = std_id
        classes = _terms(problems, std_id, "classes", std["classes"])
        relations = _terms(problems, std_id, "relations", std.get("relations", []))
        both = sorted(set(classes) & set(relations))
        if both:
            problems.append(f"{std_id}: listed as class and relation: {both}")
        std["class_names"] = set(classes)
        std["relation_names"] = set(relations)
    if problems:
        raise MappingDraftError(f"{path}: registry invalid ({len(problems)} problem(s)): "
                                + "; ".join(problems))
    return data


def _current_entry_ids(pack_root: Path) -> set[str]:
    """The ids of the pack's CURRENT entries: the tree holds the current
    version of each gated entry and nothing else (plan section 8)."""
    out = set()
    entries = Path(pack_root) / "entries"
    if not entries.is_dir():
        return out
    for p in sorted(entries.glob("*.json")):
        try:
            e = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise MappingDraftError(f"{p}: not a readable entry: {exc}") from exc
        if isinstance(e, dict) and e.get("id") == p.stem and e.get("status") == "current":
            out.add(p.stem)
    return out


def validate_draft(text: str, pack_root: Path, registry: dict) -> dict:
    """Parse the draft text and check it against the registry and the pack's
    tree. Returns the artifact (what the gate will canonicalise)."""
    try:
        artifact = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise MappingDraftError(f"draft is not readable YAML: {exc}") from exc
    problems: list[str] = []
    if not isinstance(artifact, dict) or set(artifact) != {"standard", "rows"}:
        raise MappingDraftError("draft must be a mapping with exactly the keys standard and "
                                "rows (4.5) - author and pack are given to this tool, not "
                                "written into the artifact")
    std_id = artifact["standard"]
    std = registry["standards"].get(std_id) if isinstance(std_id, str) else None
    if std is None:
        raise MappingDraftError(f"standard {std_id!r} is not registered in "
                                f"{Path(pack_root).name}/{REGISTRY_FILE}: "
                                f"{sorted(registry['standards'])}")
    if not std["class_names"]:
        raise MappingDraftError(f"standard {std_id!r} registers no verified class; no row may "
                                f"target it ({std['verified']})")
    rows = artifact["rows"]
    if not isinstance(rows, list) or not rows:
        raise MappingDraftError("rows must be a non-empty list (4.5)")
    current = _current_entry_ids(pack_root)
    seen: set[tuple[str, str]] = set()
    for i, row in enumerate(rows):
        where = f"rows[{i}]"
        if not isinstance(row, dict) or set(row) != ROW_FIELDS:
            problems.append(f"{where}: must carry exactly entry/class/relation/note (4.5)")
            continue
        entry, cls, rel, note = row["entry"], row["class"], row["relation"], row["note"]
        if not isinstance(entry, str) or not ENTRY_ID_RE.fullmatch(entry):
            problems.append(f"{where}: entry {entry!r} is not a valid entry id")
        elif entry not in current:
            problems.append(f"{where}: entry {entry!r} is not a current entry of the pack's tree "
                            f"(packs/{Path(pack_root).name}/entries/); a row maps a gated entry")
        if not isinstance(cls, str):
            problems.append(f"{where}: class must be a string")
        elif cls in std["relation_names"]:
            problems.append(f"{where}: {cls!r} is registered as a RELATION of {std_id}; a row "
                            f"maps an entry to a class")
        elif cls not in std["class_names"]:
            problems.append(f"{where}: class {cls!r} is not in the verified class list of "
                            f"{std_id} (the registry's spelling, with its prefix)")
        if rel not in RELATIONS:
            problems.append(f"{where}: relation {rel!r} not one of {RELATIONS} (4.5)")
        if not isinstance(note, str) or not note.strip():
            problems.append(f"{where}: note must say why (a non-blank string)")
        if isinstance(entry, str) and isinstance(cls, str):
            if (entry, cls) in seen:
                problems.append(f"{where}: ({entry!r}, {cls!r}) is mapped twice")
            seen.add((entry, cls))
    if problems:
        raise MappingDraftError(f"draft invalid ({len(problems)} problem(s)): "
                                + "; ".join(problems))
    return artifact


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_standard_file(registry: dict, std_id: str, module_root: Path) -> dict:
    """The standard's pinned file must be present at raw_dir/<file> and hash
    to the pin - the mapping is made against THAT text, not any edition."""
    std = registry["standards"][std_id]
    raw = Path(module_root) / registry["raw_dir"] / std["file"]
    if not raw.is_file():
        raise MappingDraftError(f"{std_id}: {raw} is absent; the standard's pinned copy must be "
                                f"in place (it is never fetched or committed)")
    got = sha256_of(raw)
    if got != std["sha256"]:
        raise MappingDraftError(f"{std_id}: {raw.name} hashes to {got}, not the pinned "
                                f"{std['sha256']}; this is not the verified copy")
    return {"file": std["file"], "sha256": got}


def jst_now() -> str:
    if time.localtime().tm_gmtoff != 9 * 3600:
        raise MappingDraftError("this machine's local clock is not JST (UTC+9); refusing")
    return time.strftime("%Y-%m-%d %H:%M:%S JST")


def head_commit(module_root: Path) -> str:
    """The commit a manifest records; resolved only when a manifest is about
    to be written (AR-rag4ds-042 r2 O5: a dry run needs no checkout)."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=module_root,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError) as exc:
        raise MappingDraftError(f"code commit unknown: {module_root} is not inside a git "
                                f"checkout ({exc}); nothing filed") from exc


def run(module_root: Path, pack: str, draft: Path, author: str, dry_run: bool = False,
        clock=jst_now, code_commit: str | None = None, out=print) -> int:
    """Validate, then file the one candidate and its run manifest. Exit 0
    filed (or dry run), 2 refused (nothing written)."""
    module_root = Path(module_root)
    if not ENTRY_ID_RE.fullmatch(pack or ""):
        raise MappingDraftError(f"pack {pack!r} must be a pack code")
    pack_root = module_root / "packs" / pack
    if not (pack_root / "manifest.yaml").is_file():
        raise MappingDraftError(f"{pack_root} is not a pack (no manifest.yaml)")
    identities = IdentityRegistry.load(module_root / IDENTITIES_FILE)
    try:
        identities.resolve(author)
    except IdentityError as exc:
        raise MappingDraftError(f"author refused: {exc}") from exc
    draft = Path(draft)
    if not draft.is_absolute():
        draft = module_root / draft
    if not draft.is_file():
        raise MappingDraftError(f"draft {draft} is absent")  # AR-rag4ds-042 r1 O1
    raw = draft.read_bytes()
    if b"\r" in raw:
        raise MappingDraftError(f"{draft}: carriage return in the draft; drafts are LF")
    text = raw.decode("utf-8")
    registry = load_registry(pack_root)
    artifact = validate_draft(text, pack_root, registry)
    std_id = artifact["standard"]
    std = registry["standards"][std_id]
    std_file = check_standard_file(registry, std_id, module_root)
    candidate = {"kind": "mapping-draft", "pack": pack, "standard": std_id,
                 "author": author.strip(), "yaml_draft": text}
    cid = candidate_id(candidate)
    store = StagingStore(module_root / "stage" / "candidates.jsonl")
    runs = Journal(module_root / "runs" / "authoring.jsonl")
    rel_draft = draft.relative_to(module_root).as_posix() if draft.is_relative_to(
        module_root) else draft.name
    # AR-rag4ds-042 r1 O2: a class declared inside a worked example of the
    # standard is named to the operator and in the manifest, since the
    # candidate the validator sees carries only the draft
    classes = {t["name"]: t for t in std["classes"]}
    examples = sorted({r["class"] for r in artifact["rows"]
                       if classes[r["class"]].get("example") is True})
    manifest = {
        "kind": "mapping-authoring",
        "schema": 1,
        "run_id": secrets.token_hex(8),
        "author": author.strip(),
        "pack": pack,
        "standard": std_id,
        "standard_file": std_file,
        "draft": rel_draft,
        "draft_sha256": hashlib.sha256(raw).hexdigest(),
        "rows": len(artifact["rows"]),
        "example_classes": examples,
        "candidate": cid,
        "code_commit": code_commit,  # resolved just before a manifest is written
        "timestamp": None,
        "tool": "tools/file_mapping_draft.py",
    }
    if cid in store.candidates():
        # AR-rag4ds-042 r1 F2: a run interrupted between the candidate and its
        # manifest left a PENDING candidate without provenance; the re-run
        # completes the missing half instead of refusing forever
        cited = [m for m in runs.read_all() if m.get("kind") == "mapping-authoring"
                 and m.get("candidate") == cid] if runs.path.exists() else []
        if cited:
            out(f"REFUSED: this draft (these bytes, this author) is already filed as {cid} "
                f"(state {store.state_of(cid)}); a changed draft is a new candidate")
            return 2
        if dry_run:
            out(f"DRY RUN - {cid} is filed (state {store.state_of(cid)}) and has NO run "
                f"manifest (an interrupted run); a live run appends it")
            return 0
        manifest["code_commit"] = code_commit or head_commit(module_root)
        manifest["timestamp"] = clock()
        manifest["recovered_after_interrupted_run"] = True
        # AR-rag4ds-042 r2 O7: the record is a reconstruction - the commit, the
        # draft path and its hash are the recovery run's; only candidate_filed
        # is the interrupted run's (from the staging record itself)
        manifest["code_commit_is_recovery_run"] = True
        manifest["candidate_filed"] = store.candidates()[cid]["filed"]
        runs.append(manifest)
        out(f"{cid} was filed (state {store.state_of(cid)}) without a run manifest - an "
            f"interrupted run; manifest run {manifest['run_id']} appended now to "
            f"runs/authoring.jsonl, marked recovered")
        return 0
    out(f"{pack} -> {std_id}: {len(artifact['rows'])} row(s); standard file {std_file['file']} "
        f"verified against its pin; entries "
        f"{sorted({r['entry'] for r in artifact['rows']})}; candidate {cid}"
        + (f"; worked-example class(es) targeted: {examples}" if examples else ""))
    if dry_run:
        out("DRY RUN - nothing filed")
        return 0
    manifest["code_commit"] = code_commit or head_commit(module_root)  # before any write
    ts = clock()
    try:
        filed = store.file_candidate(candidate, ts)
    except StagingError as exc:
        raise MappingDraftError(f"staging refused the candidate: {exc}") from exc
    assert filed == cid
    manifest["timestamp"] = ts
    runs.append(manifest)
    out(f"filed {cid} at {ts}; manifest run {manifest['run_id']} appended to "
        f"runs/authoring.jsonl")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--draft", required=True, help="the YAML draft (standard + rows)")
    ap.add_argument("--author", required=True, help="a registered identity")
    ap.add_argument("--module-root", default=str(MODULE_ROOT))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        return run(Path(a.module_root), a.pack, Path(a.draft), a.author, a.dry_run)
    except MappingDraftError as exc:
        print(f"REFUSED: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
