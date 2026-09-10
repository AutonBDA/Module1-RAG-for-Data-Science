# Module 1 — RAG for Data Science

The expert-validated knowledge framework of the **Auton-BDA** service: a three-layer
knowledge base (general concepts / domain knowledge / task patterns), populated
semi-automatically behind an expert validation gate, grounding AI reasoning through
retrieval, with every entry carrying its source and a complete audit trail.

## Status — v0.1, 2026-09-10

| | |
|---|---|
| Implementation | steps S0–S7 of the module plan complete: entry model and canonical serialization; decision journal and gate core; corpus catalog and staging; compile-and-fold with a byte-exact reconstruction test; indexes and retrieval (grounding contract v1); the interactive expert-gate CLI; extraction v1 with a Unicode quotation bound |
| Tests | 349 passed, 1 skipped (POSIX-only), run hermetically from a clean worktree |
| Verification | every step passed an independent internal audit before the next began; steps S0–S6 additionally passed external checkpoint audits; an external re-check of step S7 is in progress |
| Content | code only — the knowledge layers ship empty; the first domain pack (robotics) is the next milestone |
| Provenance | snapshot of the development repository at commit `16997ac` |

## Layout

- `docs/plan_module1.md` — the module plan (start here)
- `src/rag4ds/` — the library: `model/`, `auditlog/`, `gate/`, `corpus/`, `staging/`, `kb/`, `index/`, `retrieve/`, `extract/`
- `tests/` — the full suite, one file per implementation step
- `tools/` — `check.py` (lint + tests + scans) and `evidence.py` (hermetic evidence capture)
- `requirements.lock` / `requirements-linux.lock` — pinned environments (Python 3.11)

Pinned embedding model: `sentence-transformers/all-MiniLM-L6-v2`
@ `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

## Funding

CRF2026 — University of Aizu Competitive Research Funds, FY2026 — pilots this module.

## License

Apache-2.0. Knowledge-base content, when released, will carry its own source attributions.
