"""plan_scan.py - the exact evidence scan for plan_module1.md audits.

Run from the repository root:
    python AutonBDA/module1_rag_for_ds/tools/plan_scan.py

Prints the scan lines audit requests paste as evidence (AR-rag4ds series).
Whitespace-split word counts; byte-level non-ASCII; section-4 reference
extraction; acceptance-case enumeration; the other-module naming grep
(self-reference "Module 1" is permitted by the plan's declared naming map).
"""

import re
import sys
from pathlib import Path

PLAN = Path("AutonBDA/module1_rag_for_ds/docs/plan_module1.md")


def main() -> int:
    if not PLAN.exists():
        print(f"ERROR: {PLAN} not found (run from the repository root)")
        return 2
    text = PLAN.read_text(encoding="utf-8")

    words = len(re.findall(r"\S+", text))
    non_ascii = len([c for c in text if ord(c) > 127])
    print(f"words: {words} | non-ASCII: {non_ascii}")

    s4 = text[text.find("## 4.") : text.find("## 5.")]
    s4_words = len(re.findall(r"\S+", s4))
    print(f"s4 words: {s4_words}")
    refs = sorted(set(re.findall(r"FR\d+|NFR\d+|C\d+\b|I\d\b|A\d\b|OD-\d+|D-\d+|VT-\d+", s4)))
    print(f"s4 refs: {refs}")

    cases = re.findall(r"^- (A\d) ", text, re.M)
    print(f"acceptance cases: {len(cases)} {cases}")

    naming = [
        f"{i}:{line}"
        for i, line in enumerate(text.splitlines(), 1)
        if re.search(r"Module [2-9]|\bM[2-9]\b", line)
    ]
    print(f"other-module naming hits: {len(naming)}")
    for hit in naming:
        print(f"  {hit}")

    # Verdict (AR-rag4ds-008 F2): violations exit non-zero so CI can fail.
    # A naming hit is exempt only on the plan's own declared-map sentence.
    failures = []
    if non_ascii:
        failures.append(f"non-ASCII: {non_ascii}")
    bad_naming = [h for h in naming if "Under that declared map" not in h]
    if bad_naming:
        failures.append(f"undeclared other-module naming: {len(bad_naming)}")
    if failures:
        print(f"plan_scan: FAIL ({'; '.join(failures)})")
        return 1
    print("plan_scan: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
