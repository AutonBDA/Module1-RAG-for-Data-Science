"""Extraction-fidelity probe (AR-rag4ds-028 F2): the measurements behind
a pack's per-document choice of pypdf extraction mode, reproducible from
the raw PDFs by anyone holding them. Generic - it reads the pack's
sources.yaml (every pdf source) and knows no domain.

    python tools/extraction_probe.py <pack code>

Two measures per document, both modes (layout, plain), every page:

1. FUSED-WORD RATE: alphabetic tokens ([A-Za-z]+) of the whole document
   text; a token is "fused" when it contains an internal lowercase-to-
   uppercase boundary ([a-z][A-Z]) or is longer than 18 letters (English
   words practically never are). Printed as fused/total and percent.
2. WORDS ABSENT FROM THE OTHER MODE: per page, every alphabetic token of at
   least 4 letters in mode A whose casefolded form does not occur as a
   substring of the same page's mode-B text with ALL whitespace removed
   (so that fusion in either mode cannot create a spurious difference),
   unioned over pages, counted and listed (first 20, sorted). Computed in
   both directions: plain-not-in-layout says what layout mode DROPS or
   re-orders across a page; layout-not-in-plain says the reverse. A token
   is a substring test, so a word re-ordered within the page is NOT
   counted absent; a word that vanished from the page is.

Neither measure inspects meaning; they bound the extractor's damage, they
do not prove fidelity. The docstring IS the method: the output must be
re-derivable from it alone.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "tools"))
import pack_intake  # noqa: E402

FUSED_RE = re.compile(r"[a-z][A-Z]")
MAX_WORD_LETTERS = 18
MIN_WORD_LETTERS = 4


def fused_rate(text: str) -> tuple[int, int]:
    toks = re.findall(r"[A-Za-z]+", text)
    fused = [t for t in toks if FUSED_RE.search(t) or len(t) > MAX_WORD_LETTERS]
    return len(fused), len(toks)


def absent_words(pages_a: list[str], pages_b: list[str]) -> set[str]:
    out: set[str] = set()
    for a, b in zip(pages_a, pages_b, strict=True):
        squashed = "".join(b.casefold().split())
        for w in re.findall(r"[A-Za-z]{%d,}" % MIN_WORD_LETTERS, a):
            if w.casefold() not in squashed:
                out.add(w)
    return out


def page_texts(pdf: Path) -> dict[str, list[str]]:
    import pypdf

    reader = pypdf.PdfReader(pdf)
    out = {"plain": [], "layout": []}
    for page in reader.pages:
        out["plain"].append((page.extract_text() or "").replace("\r", ""))
        out["layout"].append((page.extract_text(extraction_mode="layout") or "").replace("\r", ""))
    return out


def run(pack: str, module_root: Path, out=sys.stdout) -> int:
    import pypdf

    module_root = Path(module_root)
    try:
        data = pack_intake.load_sources(module_root / "packs" / pack)
    except pack_intake.IntakeError as exc:
        print(f"extraction_probe: {exc}", file=sys.stderr)
        return 2
    raw_dir = module_root / data["raw_dir"]
    print(f"extraction_probe: pack {pack!r}, pypdf {pypdf.__version__}; method in the "
          f"docstring of tools/extraction_probe.py", file=out)
    print("key | chosen mode | plain fused/total (rate) | layout fused/total (rate) | "
          "words in plain absent from layout | words in layout absent from plain", file=out)
    for key, src in data["sources"].items():
        if src["kind"] != "pdf":
            continue
        pdf = raw_dir / src["file"]
        if not pdf.is_file():
            print(f"{key}: {pdf} absent; cannot probe", file=sys.stderr)
            return 1
        # the SAME acceptance set the intake uses (pack_intake.prepare): a
        # publisher that serves a slightly different file per download is
        # pinned by sha256 plus any sha256_also rendition. Checking only
        # sha256 here made the probe refuse a download the intake had
        # accepted and catalogued - and refuse the whole pack with it,
        # since the loop returns on the first refusal (found by use,
        # 2026-09-21, on rie-review).
        accepted = {src["sha256"], *src.get("sha256_also", [])}
        got = pack_intake.sha256_of(pdf)
        if got not in accepted:
            print(f"{key}: {pdf} hashes to {got}, not the pinned {src['sha256']}"
                  + (f" (nor any of {len(accepted) - 1} accepted rendition(s))"
                     if len(accepted) > 1 else "")
                  + "; refusing", file=sys.stderr)
            return 1
        texts = page_texts(pdf)
        fp, tp = fused_rate("\n".join(texts["plain"]))
        fl, tl = fused_rate("\n".join(texts["layout"]))
        p_not_l = absent_words(texts["plain"], texts["layout"])
        l_not_p = absent_words(texts["layout"], texts["plain"])
        print(f"{key} | {src['mode']} | {fp}/{tp} ({100 * fp / max(tp, 1):.2f}%) | "
              f"{fl}/{tl} ({100 * fl / max(tl, 1):.2f}%) | {len(p_not_l)} | {len(l_not_p)}",
              file=out)
        print(f"  plain-not-in-layout ({len(p_not_l)}): {sorted(p_not_l)[:20]}", file=out)
        print(f"  layout-not-in-plain ({len(l_not_p)}): {sorted(l_not_p)[:20]}", file=out)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python tools/extraction_probe.py <pack code>", file=sys.stderr)
        return 2
    return run(argv[0], MODULE_ROOT)


if __name__ == "__main__":
    sys.exit(main())
