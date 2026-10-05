"""Pack corpus intake (plan S9, component 1, NFR6, NFR2) - generic: the
tool knows no domain (I3, AR-rag4ds-028 F6); everything about a pack's
corpus is DATA in packs/<code>/corpus/sources.yaml, which this tool reads.
Run from the module root with the module venv:

    python tools/pack_intake.py <pack code>

sources.yaml pins every raw download by sha256 (a different file at the
path is refused), names each document's licence, redistributable flag,
fetch pointer, metadata and - for a PDF - its pypdf extraction mode
(layout | plain) and a licence MARKER that must be found in the extracted
text (normalised) as evidence of the recorded licence; `marker: null` says
the licence rests on the publisher's page and the licence string must say
so. A PDF's text is extracted to <text_dir>/<stem>.txt (UTF-8, LF, CR
removed, pages joined by two newlines) and THE TEXT FILE is the catalogued
document, the PDF's hash, page count, mode, pypdf version and warning
counts recorded in its metadata; an archive is catalogued as itself, its
expected members checked. Entries locate their spans in the text file and
the NFR6 scan reads it. Raw files and text live under the gitignored
data/ and never enter git (ingest refuses a non-redistributable document
git would commit).

Idempotent: a text file already present must re-extract byte-identical or
the run refuses (a stale or edited copy is not the document); a hash
already catalogued is reused, never re-registered; an archive listed under
`skipped` is reported with its reason and never catalogued.
"""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import yaml

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))
from rag4ds.auditlog.journal import Journal  # noqa: E402
from rag4ds.corpus.catalog import Catalog, CatalogError, git_would_commit, ingest  # noqa: E402
from rag4ds.model.entry import ENTRY_ID_RE  # noqa: E402
from rag4ds.model.yamlstrict import DuplicateKeyError, strict_yaml_load  # noqa: E402

SOURCES_SCHEMA_VERSION = 1
SOURCES_FILE = "sources.yaml"
SOURCE_FIELDS = {"file", "kind", "sha256", "license", "redistributable", "marker",
                 "fetch_pointer", "metadata", "mode", "members", "sha256_also"}
SOURCE_REQUIRED = {"file", "kind", "sha256", "license", "redistributable", "marker",
                   "fetch_pointer", "metadata"}
MODES = ("layout", "plain")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
MIN_TEXT_CHARS = 1000


class IntakeError(RuntimeError):
    """A refusal - printed and exit 1; nothing half-catalogued."""


def _bare_file_name(name) -> bool:
    """A file name and nothing else: no separator, no drive, no root, not a
    dot segment (AR-rag4ds-028 F7: 'C:evil.pdf' and '..' passed before)."""
    if not isinstance(name, str) or not name or name in (".", ".."):
        return False
    # a separator, a drive colon or a root: refused by character - every drive
    # or anchor spelling contains one of these three characters
    return not ("/" in name or "\\" in name or ":" in name)


def _under_data(module_root: Path, rel) -> str | None:
    """Why rel is not a directory under module_root/data, or None when it is:
    relative (no drive, no root), no '..' or '.' segment, first segment
    'data' (AR-rag4ds-028 F7: 'data/../out' satisfied a startswith check).
    Judged on the text, segment by segment - pathlib drops '.' segments and
    a resolve() adds nothing these three rules do not already refuse."""
    if not isinstance(rel, str) or not rel.strip():
        return "must be a non-blank relative path under data/"
    p = Path(rel)
    if p.is_absolute() or p.drive or p.anchor:
        return f"must be a relative path under data/ (got {rel!r})"
    segments = rel.replace("\\", "/").split("/")
    if ".." in segments or "." in segments or "" in segments[1:]:
        return f"must not carry a '..' or '.' segment (got {rel!r})"
    if segments[0] != "data":
        return f"must lie under data/ (gitignored; NFR6), got {rel!r}"
    return None


def load_sources(pack_root: Path, module_root: Path | None = None) -> dict:
    """packs/<code>/corpus/sources.yaml, validated: every problem named.
    module_root defaults to two levels above the pack (packs/<code>)."""
    pack_root = Path(pack_root)
    module_root = Path(module_root) if module_root else pack_root.parents[1]
    path = pack_root / "corpus" / SOURCES_FILE
    try:
        data = strict_yaml_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, DuplicateKeyError) as exc:
        raise IntakeError(f"{path}: not a readable sources file: {exc}") from exc
    problems: list[str] = []
    if not isinstance(data, dict) or data.get("schema") != SOURCES_SCHEMA_VERSION:
        raise IntakeError(f"{path}: schema must be {SOURCES_SCHEMA_VERSION}")
    if data.get("pack") != pack_root.name:
        problems.append(f"pack {data.get('pack')!r} must equal the directory name "
                        f"{pack_root.name!r}")
    for key in ("raw_dir", "text_dir"):
        why = _under_data(module_root, data.get(key))
        if why:
            problems.append(f"{key} {why}")
    sources = data.get("sources")
    if not isinstance(sources, dict) or not sources:
        problems.append("sources must be a non-empty mapping of key -> source")
        sources = {}
    hashes: dict[str, str] = {}
    for key, src in sources.items():
        if not isinstance(key, str) or not ENTRY_ID_RE.fullmatch(key):
            problems.append(f"source key {key!r} must use the id charset")
            continue
        if not isinstance(src, dict):
            problems.append(f"{key}: a source is a mapping")
            continue
        missing = sorted(SOURCE_REQUIRED - set(src))
        unknown = sorted(set(src) - SOURCE_FIELDS)
        if missing or unknown:
            problems.append(f"{key}: missing {missing}, unknown {unknown}")
            continue
        if not _bare_file_name(src["file"]):
            problems.append(f"{key}: file must be a bare file name (no separator, drive, root "
                            f"or dot segment), got {src['file']!r}")
        if not isinstance(src["sha256"], str) or not SHA256_RE.fullmatch(src["sha256"]):
            problems.append(f"{key}: sha256 must be 64 lowercase hex")
        elif src["sha256"] in hashes:
            problems.append(f"{key}: sha256 already pinned by {hashes[src['sha256']]!r}")
        else:
            hashes[src["sha256"]] = key
        # 2026-09-15 (docs/data_loss_2026-09-15.md): a publisher that stamps each
        # download yields a different raw file for the same document; the pin of
        # record stays, and further renditions are accepted at the path by name
        also = src.get("sha256_also", [])
        if not isinstance(also, list) or not also:
            if "sha256_also" in src:
                problems.append(f"{key}: sha256_also must be a non-empty list")
        else:
            for h in also:
                if not isinstance(h, str) or not SHA256_RE.fullmatch(h):
                    problems.append(f"{key}: sha256_also entries must be 64 lowercase hex")
                elif h == src.get("sha256") or h in hashes:
                    problems.append(f"{key}: sha256_also {h[:12]}... already pinned by "
                                    f"{hashes.get(h, key)!r}")
                else:
                    hashes[h] = key
        for f in ("license", "fetch_pointer"):
            if not isinstance(src[f], str) or not src[f].strip():
                problems.append(f"{key}: {f} must be a non-blank string")
        if not isinstance(src["redistributable"], bool):
            problems.append(f"{key}: redistributable must be a boolean")
        if src["marker"] is not None and (not isinstance(src["marker"], str)
                                          or not src["marker"].strip()):
            problems.append(f"{key}: marker must be a non-blank string or null")
        if src["marker"] is None and "page" not in src["license"]:
            problems.append(f"{key}: a null marker needs a licence string that names the "
                            f"publisher's page it rests on")
        if not isinstance(src["metadata"], dict) or not src["metadata"].get("title"):
            problems.append(f"{key}: metadata must be a mapping with a title")
        if src["kind"] == "pdf":
            if src.get("mode") not in MODES:
                problems.append(f"{key}: a pdf source needs mode in {MODES}")
            if "members" in src:
                problems.append(f"{key}: members belong to an archive, not a pdf")
        elif src["kind"] == "zip":
            m = src.get("members")
            if not isinstance(m, list) or not m or not all(isinstance(x, str) for x in m):
                problems.append(f"{key}: a zip source needs a non-empty members list")
            if "mode" in src:
                problems.append(f"{key}: mode belongs to a pdf, not an archive")
        else:
            problems.append(f"{key}: kind must be pdf or zip, got {src['kind']!r}")
    skipped = data.get("skipped") or {}
    if not isinstance(skipped, dict):
        problems.append("skipped must be a mapping")
        skipped = {}
    for key, sk in skipped.items():
        if (not isinstance(sk, dict) or set(sk) != {"file", "sha256", "reason"}
                or not _bare_file_name(sk.get("file"))
                or not isinstance(sk.get("sha256"), str) or not SHA256_RE.fullmatch(sk["sha256"])
                or not isinstance(sk.get("reason"), str) or len(sk["reason"].strip()) < 40):
            problems.append(f"skipped {key!r}: needs file (a bare file name), a 64-hex sha256 "
                            f"and a reason of at least 40 characters")
        elif sk["sha256"] in hashes:
            problems.append(f"skipped {key!r}: sha256 is also a source ({hashes[sk['sha256']]!r})")
    unknown_top = sorted(set(data) - {"schema", "pack", "raw_dir", "text_dir", "sources",
                                      "skipped"})
    if unknown_top:
        problems.append(f"unknown top-level field(s) {unknown_top}")
    if problems:
        raise IntakeError(f"{path}: sources invalid ({len(problems)} problem(s)): "
                          + "; ".join(problems))
    data["skipped"] = skipped
    return data


class _WarningCounter(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.rotated = 0
        self.other = 0

    def emit(self, record):
        if "Rotated text discovered" in record.getMessage():
            self.rotated += 1
        else:
            self.other += 1


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def scrub_lone_surrogates(text: str) -> tuple[str, int]:
    """pypdf can hand back half of a surrogate pair for a glyph a font maps
    beyond the BMP (2026-09-21: 23 maths-italic symbols in one paper's
    formulas) - a code point no UTF-8 file can hold. Each becomes U+FFFD, the
    replacement character, and the COUNT is returned to be recorded with the
    document's other extraction facts: the loss is small, local to formula
    symbols, and stated, never silent."""
    n = len(_LONE_SURROGATE.findall(text))
    return (_LONE_SURROGATE.sub("\ufffd", text) if n else text), n


def extract_text(pdf: Path, mode: str) -> tuple[str, dict]:
    """The document text: every page's extract_text (layout or plain mode),
    CR removed, pages joined by two newlines, lone surrogates replaced by
    U+FFFD. Returns (text, facts) with the page count, the pypdf version, its
    warning counts and the lone-surrogate count."""
    import pypdf

    if mode not in MODES:
        raise IntakeError(f"{pdf.name}: extraction mode {mode!r} is not layout or plain")
    counter = _WarningCounter()
    log = logging.getLogger("pypdf")
    log.addHandler(counter)
    try:
        reader = pypdf.PdfReader(pdf)
        pages = []
        for page in reader.pages:
            if mode == "layout":
                t = page.extract_text(extraction_mode="layout")
            else:
                t = page.extract_text()
            pages.append((t or "").replace("\r", ""))
    finally:
        log.removeHandler(counter)
    text, lone = scrub_lone_surrogates("\n\n".join(pages))
    return text, {"pages": len(pages), "rotated_text_warnings": counter.rotated,
                  "other_pypdf_warnings": counter.other, "lone_surrogates": lone,
                  "pypdf": pypdf.__version__,
                  "extraction": f"pypdf {pypdf.__version__} PdfReader extract_text("
                                f"{'extraction_mode=layout' if mode == 'layout' else 'plain'}), "
                                f"CR removed, pages joined by two newlines, lone surrogates "
                                f"replaced by U+FFFD (count recorded), UTF-8, LF"}


def normalise(text: str) -> str:
    return " ".join(text.casefold().split())


def jst_now() -> str:
    if time.localtime().tm_gmtoff != 9 * 3600:
        raise IntakeError("this machine's local clock is not JST (UTC+9); refusing")
    return time.strftime("%Y-%m-%d %H:%M:%S JST")


def head_commit(module_root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=module_root,
                          capture_output=True, text=True, check=True).stdout.strip()


# the sources whose code produces an ingest manifest: when any is modified or
# untracked at run time, HEAD is not the code that ran (AR-rag4ds-045 F4; the
# class AR-rag4ds-030 r1 O2 asked a guard for)
TOOL_SOURCES = ("tools/pack_intake.py", "src/rag4ds/corpus/catalog.py",
                # AR-rag4ds-045 r2 O8: the modules this tool imports that shape a
                # manifest or a catalogue entry are the code that produced them too
                "src/rag4ds/auditlog/journal.py", "src/rag4ds/model/entry.py",
                "src/rag4ds/model/yamlstrict.py")
# the metadata keys the intake MEASURES (prepare() writes them); sources.yaml
# may not carry one, so a refresh can never rewrite a measured fact through the
# tool (AR-rag4ds-045 r2 O9)
INTAKE_FACT_KEYS = frozenset({"pdf_sha256", "pdf_file", "extraction_mode", "chars", "pages",
                              "rotated_text_warnings", "other_pypdf_warnings", "lone_surrogates",
                              "pypdf", "extraction", "licence_evidence", "archive_members"})


def tool_sources_dirty(module_root: Path) -> list[str]:
    """The tool's own source files that differ from HEAD (modified, staged or
    untracked), as git names them; empty when the recorded commit is the
    code that runs."""
    out = subprocess.run(["git", "status", "--porcelain", "--", *TOOL_SOURCES], cwd=module_root,
                         capture_output=True, text=True, check=True).stdout
    return [line[3:].strip() for line in out.splitlines() if line.strip()]


def prepare(key: str, src: dict, raw_dir: Path, text_dir: Path) -> tuple[Path, dict]:
    """Verify the download against its pin, produce the document to be
    catalogued (the text file for a PDF, the archive itself for a zip) and
    the metadata to record. Raises IntakeError on any mismatch."""
    raw = Path(raw_dir) / src["file"]
    if not raw.is_file():
        raise IntakeError(f"{key}: {raw} is absent (fetch from {src['fetch_pointer']})")
    got = sha256_of(raw)
    accepted = {src["sha256"], *src.get("sha256_also", [])}
    if got not in accepted:
        raise IntakeError(f"{key}: {raw.name} hashes to {got}, not the pinned {src['sha256']}"
                          + (f" (nor any of {len(accepted) - 1} accepted rendition(s))"
                             if len(accepted) > 1 else "")
                          + "; this is not the pinned download")
    meta = dict(src["metadata"])
    if src["kind"] == "pdf":
        text, facts = extract_text(raw, src["mode"])
        if len(text.strip()) < MIN_TEXT_CHARS:
            raise IntakeError(f"{key}: extracted only {len(text)} chars; not a text PDF")
        marker = src["marker"]
        if marker and normalise(marker) not in normalise(text):
            raise IntakeError(f"{key}: licence marker {marker!r} not found in the extracted "
                              f"text; the recorded licence is not evidenced")
        out = Path(text_dir) / (raw.stem + ".txt")
        data = text.encode("utf-8")
        if out.exists():
            if out.read_bytes() != data:
                raise IntakeError(f"{key}: {out} exists and differs from a fresh extraction; "
                                  f"a stale or edited copy is not the document - remove it")
        else:
            # AR-rag4ds-028 F7: nothing is written where git would commit it -
            # asked of git BEFORE the write, for every document (not only the
            # non-redistributable ones ingest() later refuses)
            out.parent.mkdir(parents=True, exist_ok=True)
            refused = git_would_commit(out)
            if refused:
                raise IntakeError(f"{key}: refusing to write the extracted text to {out}: it "
                                  f"{refused} (NFR6: text lives outside git, in an ignored "
                                  f"directory)")
            out.write_bytes(data)
        # pdf_sha256 is the file's OWN hash (an accepted rendition may differ from the pin)
        meta.update({"pdf_sha256": got, "pdf_file": raw.name,
                     "extraction_mode": src["mode"], "chars": len(text), **facts,
                     "licence_evidence": (f"marker {marker!r} found in the text" if marker
                                          else "publisher page (see license)")})
        return out, meta
    if src["kind"] == "zip":
        with zipfile.ZipFile(raw) as z:
            names = set(z.namelist())
        missing = [m for m in src["members"] if m not in names]
        if missing:
            raise IntakeError(f"{key}: archive lacks expected members {missing}")
        meta.update({"archive_members": sorted(names),
                     "licence_evidence": "publisher page (see license)"})
        return raw, meta
    raise IntakeError(f"{key}: unknown kind {src['kind']!r}")


def refresh_metadata(catalog: Catalog, catalogued: dict, source_meta: dict, runs: Journal,
                     ts: str, commit: str, dirty: list[str], key: str) -> list[str]:
    """AR-rag4ds-045 F8: a catalogued document's source-provided metadata
    (title, authors, role, ...) follows sources.yaml. When the catalogue's
    copy differs, the changed keys are rewritten in the catalogue - the
    extraction facts the intake recorded stay - and a `metadata-refresh`
    manifest names them (NFR2, NFR9). Returns the changed keys, [] when the
    catalogue already agrees. Nothing but metadata is ever touched."""
    current = catalogued["metadata"]
    measured = sorted(set(source_meta) & INTAKE_FACT_KEYS)
    if measured:
        raise IntakeError(f"{key}: sources.yaml metadata carries {measured}, which the intake "
                          f"measures; a refresh rewrites source-provided strings only")
    changed = sorted(k for k, v in source_meta.items() if current.get(k) != v)
    if not changed:
        return []
    runs.append({"kind": "metadata-refresh", "schema": 1, "pack": catalog.pack,
                 "hash": catalogued["hash"], "source_key": key, "changed": changed,
                 "timestamp": ts, "tool": "tools/pack_intake.py", "code_commit": commit,
                 "code_commit_dirty": bool(dirty), "dirty_sources": list(dirty)})
    catalog.update_metadata(catalogued["hash"], {**current, **source_meta})
    return changed


def run(pack: str, module_root: Path, out=sys.stdout) -> int:
    import pypdf

    module_root = Path(module_root)
    if not ENTRY_ID_RE.fullmatch(pack or "") or pack == "shared":
        print(f"pack_intake: {pack!r} is not a pack code", file=sys.stderr)
        return 2
    pack_root = module_root / "packs" / pack
    print(f"pack_intake: pack {pack!r}, python {sys.version.split()[0]}, pypdf "
          f"{pypdf.__version__}", file=out)
    try:
        data = load_sources(pack_root, module_root)
        raw_dir, text_dir = module_root / data["raw_dir"], module_root / data["text_dir"]
        commit, ts = head_commit(module_root), jst_now()
        dirty = tool_sources_dirty(module_root)
        if dirty:
            # recorded, never silent: the manifest names HEAD, and says HEAD is
            # not the code that ran
            print(f"pack_intake: WARNING - {', '.join(dirty)} differ(s) from {commit}; every "
                  f"manifest of this run carries code_commit_dirty: true", file=out)
        catalog = Catalog(pack, pack_root)
        runs = Journal(module_root / "runs" / "ingest.jsonl")
        already = {d["hash"]: d for d in catalog.load()["documents"]}
        ingested = reused = 0
        for key, src in data["sources"].items():
            doc_path, meta = prepare(key, src, raw_dir, text_dir)
            h = sha256_of(doc_path)
            if h in already:
                changed = refresh_metadata(catalog, already[h], src["metadata"], runs, ts, commit,
                                           dirty, key)
                print(f"{key}: already catalogued as {h[:16]}..., reused"
                      + (f"; metadata refreshed from sources.yaml ({', '.join(changed)})"
                         if changed else ""), file=out)
                reused += 1
                continue
            m = ingest(catalog, doc_path, module_root, src["license"], src["fetch_pointer"],
                       src["redistributable"], meta, ts, runs, commit,
                       {"tool": "tools/pack_intake.py", "sources": f"packs/{pack}/corpus/"
                        f"{SOURCES_FILE}", "source_key": key, "kind": src["kind"],
                        "extraction_mode": src.get("mode"),
                        "code_commit_dirty": bool(dirty), "dirty_sources": dirty,
                        "pypdf": pypdf.__version__ if src["kind"] == "pdf" else None,
                        "pinned_sha256": src["sha256"]})
            print(f"ingested {key}: document {m['hash'][:16]}... local_path {m['local_path']} "
                  f"({meta.get('pages', '-')} pages, {meta.get('chars', m['size_bytes'])} "
                  f"chars/bytes, rotated-text warnings {meta.get('rotated_text_warnings', '-')}, "
                  f"lone surrogates {meta.get('lone_surrogates', '-')})",
                  file=out)
            ingested += 1
        for key, skip in data["skipped"].items():
            raw = raw_dir / skip["file"]
            state = "absent" if not raw.is_file() else (
                "pinned" if sha256_of(raw) == skip["sha256"] else "DIFFERENT HASH")
            print(f"skipped {key}: {skip['file']} ({state}) - {skip['reason']}", file=out)
    except (IntakeError, CatalogError) as exc:
        print(f"pack_intake: REFUSED - {exc}", file=sys.stderr)
        return 1
    print(f"pack_intake: {ingested} ingested, {reused} reused, {len(data['skipped'])} skipped; "
          f"catalog {catalog.path.relative_to(module_root).as_posix()} at {commit}, {ts}",
          file=out)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python tools/pack_intake.py <pack code>", file=sys.stderr)
        return 2
    return run(argv[0], MODULE_ROOT)


if __name__ == "__main__":
    sys.exit(main())
