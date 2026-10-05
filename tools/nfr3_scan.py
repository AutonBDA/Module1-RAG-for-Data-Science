"""NFR3 scan (plan 4.2, S9): every pack declares its own lexicon - the domain
terms it owns - in its manifest, and the core must stay domain-blind (I3).
This tool reads every packs/<code>/manifest.yaml, and scans the core code
(src/rag4ds, every .py file) and the two shared layers
(shared/general, shared/tasks - every gated entry's JSON) for any lexicon
term, matching whole words and phrases - over the whole file text, so a
phrase wrapped across a line break is found - never substrings ("omics"
must not hit "economics"; "joint" does not hit "join"), case-insensitively.
Any hit fails the scan and
is printed with the file, the term and the line. A pack's own directory is
never scanned against its own lexicon; another pack's directory is (packs do
not share terms in v1, 4.2).

    python tools/nfr3_scan.py            (from the module root; exit 0 = clean)

Exit 2 when a manifest does not validate or a scanned file cannot be read -
never a pass.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))
from rag4ds.model.pack import PackManifestError, load_all_pack_manifests  # noqa: E402

# the core package - the plan's scan scope (4.2). tools/ is NOT scanned: the
# VT-3b splitter measurement fixture (tools/splitter_measurement.py) carries
# sample sentences with domain words; the pack intake itself is generic
# (tools/pack_intake.py reads packs/<code>/corpus/sources.yaml - F6)
SCANNED_CODE = ("src/rag4ds",)
SCANNED_SHARED = ("shared/general", "shared/tasks")


class Nfr3Error(RuntimeError):
    """The scan could not run to completion - reported, never read as clean."""


def term_pattern(term: str) -> re.Pattern:
    """Whole-word, whole-phrase, case-insensitive; a hyphen or any whitespace
    run inside the term matches a hyphen or whitespace run in the text."""
    parts = [re.escape(p) for p in re.split(r"[\s-]+", term.strip()) if p]
    body = r"[\s\-]+".join(parts)
    return re.compile(r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9])", re.IGNORECASE)


def _text_of(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise Nfr3Error(f"{path}: not JSON: {exc}") from exc

        def strings(v):
            if isinstance(v, str):
                return [v]
            if isinstance(v, dict):
                return [s for x in v.values() for s in strings(x)]
            if isinstance(v, list):
                return [s for x in v for s in strings(x)]
            return []
        return "\n".join(strings(obj))
    return raw


def scanned_files(module_root: Path, exclude_pack: str | None = None) -> list[Path]:
    files: list[Path] = []
    for rel in SCANNED_CODE:
        files += sorted(p for p in (module_root / rel).rglob("*.py")
                        if "__pycache__" not in p.parts)
    for rel in SCANNED_SHARED:
        d = module_root / rel
        if d.is_dir():
            files += sorted(d.glob("*.json"))
    packs = module_root / "packs"
    if packs.is_dir():
        for other in sorted(packs.iterdir()):
            if other.is_dir() and other.name != exclude_pack:
                files += sorted(p for p in other.rglob("*") if p.is_file()
                                and p.suffix in (".json", ".yaml", ".md"))
    return files


def run(module_root: Path) -> tuple[list[str], list[str]]:
    """(hits, notes). hits non-empty = the scan FAILS."""
    try:
        manifests = load_all_pack_manifests(module_root)
    except PackManifestError as exc:
        raise Nfr3Error(str(exc)) from exc
    hits, notes = [], []
    if not manifests:
        notes.append("no pack manifest under packs/; nothing to scan against")
        return hits, notes
    for code, m in manifests.items():
        patterns = [(t, term_pattern(t)) for t in m.lexicon]
        files = scanned_files(module_root, exclude_pack=code)
        n = 0
        for f in files:
            try:
                text = _text_of(f)
            except UnicodeDecodeError as exc:
                raise Nfr3Error(f"{f}: not UTF-8: {exc}") from exc
            # AR-rag4ds-028 F5: matched over the WHOLE text, so a phrase broken
            # across a line break is seen; the line number is derived from the
            # match offset, and one hit is reported per term per line
            seen: set[tuple[str, int]] = set()
            for term, pat in patterns:
                for mt in pat.finditer(text):
                    ln = text.count("\n", 0, mt.start()) + 1
                    if (term, ln) in seen:
                        continue
                    seen.add((term, ln))
                    n += 1
                    line = text.split("\n")[ln - 1]
                    hits.append(f"{f.relative_to(module_root).as_posix()}:{ln}: pack "
                                f"{code!r} term {term!r} in: {line.strip()[:80]!r}")
        notes.append(f"pack {code!r}: {len(m.lexicon)} lexicon terms scanned over "
                     f"{len(files)} files, {n} hit(s)")
    return hits, notes


def main() -> int:
    try:
        hits, notes = run(MODULE_ROOT)
    except Nfr3Error as exc:
        print(f"nfr3_scan: CANNOT SCAN - {exc}", file=sys.stderr)
        return 2
    for n in notes:
        print(f"nfr3_scan: {n}")
    if hits:
        print(f"nfr3_scan: FAILED - {len(hits)} domain term(s) found outside their pack:",
              file=sys.stderr)
        for h in hits:
            print(f"  {h}", file=sys.stderr)
        return 1
    print("nfr3_scan: OK - the core and the shared layers carry no pack's domain term")
    return 0


if __name__ == "__main__":
    sys.exit(main())
