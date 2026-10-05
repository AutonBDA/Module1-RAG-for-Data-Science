"""The pinned spaCy language model (VT-3 re-opened 2026-09-11 per the PI
decision of 2026-08-23; the record is docs/vt3_pins.md, section VT-3b).

One model, one version, one hash - and the loader refuses anything else.
Before handing the pipeline out it verifies, in this order: that the
installed distribution came from the pinned wheel (pip records in
direct_url.json the sha256 it verified at install, and that hash must equal
the pin); that every installed file still carries the digest pip wrote
into RECORD at install, so the bytes on disk are the wheel's bytes
(AR-rag4ds-024 O2 - direct_url.json alone says which wheel pip fetched, not
what is on disk now); that the package imports; that the loaded pipeline's
own meta names the pinned language, name and version; that the range the
model declares is the one the pin expects; and that the running spaCy
satisfies it. Any mismatch raises LanguageModelError (NFR9; charter
section 4). There is no fallback to a blank pipeline: a wrong model
silently substituted for the pinned one is exactly the failure the pin
exists to make impossible.

describe(nlp) - the route by which a run manifest records the pin (NFR2) -
runs the SAME environment verification as the loader before it returns
anything (AR-rag4ds-025 F1: an earlier version re-read the hash without
comparing it to the pin and never checked the bytes, so it certified a
tampered model the loader refused). A manifest can therefore carry the pin
only when the installed wheel is the pin, its bytes match RECORD, and the
pipeline's meta is the pin's.

What the check is not: direct_url.json and RECORD are ordinary files in
site-packages, so a deliberate, consistent edit of both together would
pass. The guard is against accident and drift - a model installed by
`spacy download`, from a local file, from an index, a spaCy bump past the
model's range, a partially overwritten install - not against an adversary
with write access to the environment.

At this pin nothing in extraction USES the model. S7's rule strategy keeps
the blank sentencizer, by measurement (tools/splitter_measurement.py; the
output is recorded in docs/vt3_pins.md and asserted by the tests). The pin
is here so that the next extraction increment - POS/dependency patterns and
relation-type proposals - depends on a verified resource, which is VT-3's
own rule: pin before the first line of code depends on it.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.metadata
import json
import types
from pathlib import Path

# Read-only (AR-rag4ds-024 O3): nothing can redefine the pin in place.
PINNED_LANGUAGE_MODEL = types.MappingProxyType({
    "package": "en_core_web_sm",
    "lang": "en",
    "name": "core_web_sm",
    "version": "3.8.0",
    "sha256": "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85",
    "spacy_version": ">=3.8.0,<3.9.0",
    "license": "MIT",
    "url": ("https://github.com/explosion/spacy-models/releases/download/"
            "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"),
})


class LanguageModelError(RuntimeError):
    """The pinned language model is absent, unverifiable, or not the pin."""


def pinned_install_command(pin=PINNED_LANGUAGE_MODEL) -> str:
    return f'pip install --no-deps "{pin["url"]}#sha256={pin["sha256"]}"'


def _distribution(pin) -> importlib.metadata.Distribution:
    try:
        return importlib.metadata.distribution(pin["package"])
    except importlib.metadata.PackageNotFoundError as exc:
        raise LanguageModelError(
            f"{pin['package']} is not installed in this environment; install the pinned "
            f"wheel: {pinned_install_command(pin)}") from exc


def _direct_url_path(dist: importlib.metadata.Distribution):
    """The on-disk path of the distribution's direct_url.json as RECORD
    locates it, or None when RECORD does not list one."""
    for entry in dist.files or []:
        if Path(str(entry)).name == "direct_url.json":
            return Path(str(dist.locate_file(entry)))
    return None


def installed_wheel_sha256(pin=PINNED_LANGUAGE_MODEL) -> str:
    """The sha256 pip verified when it installed the model's wheel, read from
    the distribution's direct_url.json. Absent, unreadable, corrupt or
    hash-less metadata is an error, not an unknown: a model that arrived by
    any other route cannot be tied to the pin."""
    package = pin["package"]
    dist = _distribution(pin)
    raw = dist.read_text("direct_url.json")
    if raw is None:
        # read_text also swallows permission and is-a-directory errors (O5):
        # ask the FILESYSTEM whether the file is there (AR-rag4ds-025 F3 - the
        # RECORD listing alone misdiagnoses a listed-but-deleted file as
        # unreadable), so the message names the case that actually holds
        path = _direct_url_path(dist)
        present = path is not None and path.exists()
        why = ("is present but unreadable" if present
               else "is absent - the package was not installed from a URL")
        raise LanguageModelError(
            f"{package}: direct_url.json {why}, so its wheel hash cannot be verified; "
            f"reinstall it: {pinned_install_command(pin)}")
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LanguageModelError(
            f"{package}: direct_url.json is not valid JSON ({exc}); reinstall it: "
            f"{pinned_install_command(pin)}") from exc
    try:
        return info["archive_info"]["hashes"]["sha256"]
    except (KeyError, TypeError) as exc:
        raise LanguageModelError(
            f"{package}: direct_url.json carries no sha256 (installed without a hash fragment): "
            f"{raw}; reinstall it: {pinned_install_command(pin)}") from exc


def verify_installed_files(pin=PINNED_LANGUAGE_MODEL) -> int:
    """Hash every installed file the distribution's RECORD lists against the
    digest pip wrote there at install, so the bytes on disk are the wheel's
    bytes (AR-rag4ds-024 O2). Entries without a digest - RECORD itself and
    byte-compiled files - are skipped. Returns the number of files verified;
    zero is an error."""
    package = pin["package"]
    dist = _distribution(pin)
    files = dist.files
    if not files:
        raise LanguageModelError(
            f"{package}: no RECORD, so the installed files cannot be verified; reinstall it: "
            f"{pinned_install_command(pin)}")
    verified = 0
    for entry in files:
        if entry.hash is None:
            continue
        if entry.hash.mode != "sha256":
            raise LanguageModelError(
                f"{package}: RECORD digest for {entry} uses {entry.hash.mode}, not sha256")
        path = Path(str(dist.locate_file(entry)))
        try:
            digest = hashlib.sha256(path.read_bytes()).digest()
        except OSError as exc:
            raise LanguageModelError(
                f"{package}: installed file {entry} cannot be read ({exc}); reinstall it: "
                f"{pinned_install_command(pin)}") from exc
        got = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        if got != entry.hash.value:
            raise LanguageModelError(
                f"{package}: installed file {entry} does not match its RECORD digest - the "
                f"bytes changed since install; reinstall it: {pinned_install_command(pin)}")
        verified += 1
    if verified == 0:
        raise LanguageModelError(
            f"{package}: RECORD lists no digest at all, so nothing could be verified; "
            f"reinstall it: {pinned_install_command(pin)}")
    return verified


def verify_installed_environment(pin=PINNED_LANGUAGE_MODEL) -> str:
    """The environment half of the guarantee, shared by the loader and by
    describe() (AR-rag4ds-025 F1): the installed wheel's hash equals the pin
    and every installed file matches RECORD. Returns the verified sha256."""
    got = installed_wheel_sha256(pin)
    if got != pin["sha256"]:
        raise LanguageModelError(
            f"{pin['package']}: installed wheel sha256 {got} != pinned {pin['sha256']}; "
            f"reinstall it: {pinned_install_command(pin)}")
    verify_installed_files(pin)
    return got


def _check_pipeline_meta(nlp, pin) -> None:
    meta = nlp.meta
    found = (meta.get("lang"), meta.get("name"), meta.get("version"))
    expected = (pin["lang"], pin["name"], pin["version"])
    if found != expected:
        raise LanguageModelError(f"pipeline meta {found} is not the pin {expected}")


def describe(nlp, pin=PINNED_LANGUAGE_MODEL) -> dict:
    """The pin as a manifest records it (NFR2). Runs the same environment
    verification as the loader - wheel hash equal to the pin, installed
    bytes matching RECORD - and re-checks the pipeline's meta against the
    pin before returning, so no manifest can record the pin without the
    pinned model, byte for byte, behind it (AR-rag4ds-024 O1 as
    AR-rag4ds-025 F1 required it to be met)."""
    sha = verify_installed_environment(pin)
    _check_pipeline_meta(nlp, pin)
    return {"id": pin["package"], "version": pin["version"], "sha256": sha}


def load_pinned_language_model(pin=PINNED_LANGUAGE_MODEL):
    """Load the pinned model, verifying wheel hash, installed bytes, pipeline
    meta and spaCy compatibility first; raise LanguageModelError on any
    mismatch."""
    import spacy

    verify_installed_environment(pin)
    try:
        module = importlib.import_module(pin["package"])
    except ImportError as exc:
        raise LanguageModelError(
            f"{pin['package']} is installed but does not import: {exc}") from exc
    nlp = module.load()
    _check_pipeline_meta(nlp, pin)
    declared = nlp.meta.get("spacy_version")
    if declared != pin["spacy_version"]:
        raise LanguageModelError(
            f"model declares spacy_version {declared!r}, pin expects {pin['spacy_version']!r}")
    if not spacy.util.is_compatible_version(spacy.__version__, declared):
        raise LanguageModelError(
            f"running spaCy {spacy.__version__} is outside the model's range {declared}")
    return nlp
