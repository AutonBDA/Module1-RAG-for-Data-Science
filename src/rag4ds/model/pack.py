"""The pack manifest (plan 4.1, 4.4 - the whole coupling surface between a
domain pack and the core): packs/<code>/manifest.yaml, human-edited YAML
(section 8), read and validated here and nowhere else.

A manifest names the pack (code, name, semver version, lifecycle state - 4.3),
its validator (the role and the current designee, who must be a registered
identity holding the validator role FOR THIS PACK - identities.yaml), its
licence and ethics declaration (NFR7), its domain lexicon (NFR3: every
domain term the pack owns, the input of tools/nfr3_scan.py), and the minimum
core schema version it requires (the build refuses a pack the running core
cannot honour, loudly - NFR9). A pack is data, never code; nothing here
executes anything a pack provides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from rag4ds.model.entry import CORE_SCHEMA_VERSION, ENTRY_ID_RE, SHARED_PACK
from rag4ds.model.yamlstrict import DuplicateKeyError, strict_yaml_load

PACK_MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILE = "manifest.yaml"
PACK_STATES = ("DRAFT", "POPULATING", "RELEASED", "RETIRED")
MANIFEST_FIELDS = {
    "schema", "code", "name", "version", "state", "validator", "license", "ethics",
    "lexicon", "min_core_schema_version", "description",
}
SEMVER_RE = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")
LEXICON_TERM_RE = re.compile(r"[a-z0-9][a-z0-9 \-']{0,59}")  # lowercase words and phrases


class PackManifestError(ValueError):
    """A manifest that does not validate - refused with every problem named."""


@dataclass(frozen=True)
class PackManifest:
    code: str
    name: str
    version: str
    state: str
    validator_role: str
    validator_designee: str
    license: str
    ethics: str
    lexicon: tuple[str, ...]
    min_core_schema_version: int
    description: str
    path: Path


def manifest_path(pack_root: Path) -> Path:
    return Path(pack_root) / MANIFEST_FILE


def load_pack_manifest(pack_root: Path) -> PackManifest:
    """Read packs/<code>/manifest.yaml and validate it against the running
    core. The directory name must equal the manifest's code."""
    pack_root = Path(pack_root)
    path = manifest_path(pack_root)
    try:
        data = strict_yaml_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, DuplicateKeyError) as exc:
        raise PackManifestError(f"{path}: not a readable YAML manifest: {exc}") from exc
    if not isinstance(data, dict):
        raise PackManifestError(f"{path}: a manifest is a mapping")
    problems: list[str] = []
    if data.get("schema") != PACK_MANIFEST_SCHEMA_VERSION:
        problems.append(f"schema must be {PACK_MANIFEST_SCHEMA_VERSION}")
    missing = sorted(MANIFEST_FIELDS - set(data))
    unknown = sorted(set(data) - MANIFEST_FIELDS)
    if missing:
        problems.append(f"missing {missing}")
    if unknown:
        problems.append(f"unknown field(s) {unknown}")
    if problems:
        raise PackManifestError(f"{path}: manifest invalid: " + "; ".join(problems))

    code = data["code"]
    if not isinstance(code, str) or not ENTRY_ID_RE.fullmatch(code) or code == SHARED_PACK:
        problems.append(f"code {code!r} must be a pack code (id charset, never {SHARED_PACK!r})")
    elif code != pack_root.name:
        problems.append(f"code {code!r} must equal the directory name {pack_root.name!r}")
    for key in ("name", "license", "ethics", "description"):
        if not isinstance(data[key], str) or not data[key].strip():
            problems.append(f"{key} must be a non-blank string")
    if not isinstance(data["version"], str) or not SEMVER_RE.fullmatch(data["version"]):
        problems.append(f"version {data['version']!r} must be semver MAJOR.MINOR.PATCH")
    if data["state"] not in PACK_STATES:
        problems.append(f"state {data['state']!r} not one of {PACK_STATES}")
    v = data["validator"]
    if (not isinstance(v, dict) or set(v) != {"role", "designee"}
            or not isinstance(v.get("role"), str) or not v["role"].strip()
            or not isinstance(v.get("designee"), str) or not v["designee"].strip()):
        problems.append("validator must be a mapping with exactly role and designee, both "
                        "non-blank strings")
    lex = data["lexicon"]
    if not isinstance(lex, list) or not lex:
        problems.append("lexicon must be a non-empty list of the pack's domain terms (NFR3)")
    else:
        seen = set()
        for i, term in enumerate(lex):
            if not isinstance(term, str) or not LEXICON_TERM_RE.fullmatch(term):
                problems.append(f"lexicon[{i}] {term!r} must be a lowercase word or phrase "
                                f"(letters, digits, spaces, hyphens, apostrophes; 1-60 chars)")
            elif term in seen:
                problems.append(f"lexicon[{i}] {term!r} is listed twice")
            else:
                seen.add(term)
    mcs = data["min_core_schema_version"]
    if isinstance(mcs, bool) or not isinstance(mcs, int) or mcs < 1:
        problems.append("min_core_schema_version must be a positive integer")
    elif mcs > CORE_SCHEMA_VERSION:
        problems.append(f"min_core_schema_version {mcs} exceeds the running core's "
                        f"{CORE_SCHEMA_VERSION}; this core cannot honour the pack (NFR9)")
    if problems:
        raise PackManifestError(f"{path}: manifest invalid ({len(problems)} problem(s)): "
                                + "; ".join(problems))
    return PackManifest(
        code=code, name=data["name"].strip(), version=data["version"], state=data["state"],
        validator_role=v["role"].strip(), validator_designee=v["designee"].strip(),
        license=data["license"].strip(), ethics=data["ethics"].strip(),
        lexicon=tuple(lex), min_core_schema_version=mcs,
        description=data["description"].strip(), path=path,
    )


def load_all_pack_manifests(tree_root: Path) -> dict[str, PackManifest]:
    """Every pack under packs/ by code; a pack directory without a valid
    manifest is a loud failure, never skipped."""
    packs = Path(tree_root) / "packs"
    out: dict[str, PackManifest] = {}
    if not packs.is_dir():
        return out
    for child in sorted(packs.iterdir()):
        if child.name == ".gitkeep" and child.is_file():
            continue
        if not child.is_dir():
            raise PackManifestError(f"{child} is not a pack directory; packs/ holds packs only")
        out[child.name] = load_pack_manifest(child)
    return out


def check_validator(manifest: PackManifest, registry) -> None:
    """The manifest's designee must be a registered identity holding the
    validator role for this very pack (identities.yaml, scoped per pack -
    AR-rag4ds-026 O17/O22). Raises PackManifestError otherwise."""
    from rag4ds.gate.identities import IdentityError

    try:
        ident = registry.resolve(manifest.validator_designee)
    except IdentityError as exc:
        raise PackManifestError(f"{manifest.path}: validator designee: {exc}") from exc
    if manifest.validator_role != "validator":
        raise PackManifestError(f"{manifest.path}: validator role must be 'validator', got "
                                f"{manifest.validator_role!r}")
    if not ident.validates(manifest.code):
        raise PackManifestError(
            f"{manifest.path}: {manifest.validator_designee!r} is not registered as the "
            f"validator of pack {manifest.code!r} (identities.yaml: roles {list(ident.roles)}, "
            f"packs {list(ident.packs)})")
