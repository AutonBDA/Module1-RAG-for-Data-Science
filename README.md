# Module 1 - RAG for Data Science

The expert-validated knowledge framework of the **Auton-BDA** service: a three-layer
knowledge base (general concepts / domain knowledge / task patterns), populated
semi-automatically behind an expert validation gate, grounding reasoning through
retrieval, with every entry carrying its source and a complete audit trail.

## Status - v0.2, 2026-10-06

| | |
|---|---|
| Implementation | steps S0-S9 of the module plan: entry model and canonical serialization; decision journal and gate core; corpus catalog and staging; compile-and-fold with a byte-exact reconstruction test; indexes and retrieval (grounding contract v1); the expert-gate CLI; extraction; the shared layers; the first domain pack (robotics) with its release tooling, a demonstration page, and written answers that must cite the retrieved entries |
| Knowledge pack | robotics 1.0.1, released after expert validation; the served index holds 24 entries (the pack and the shared layers) |
| Live demonstration | https://rag4ds-robotics-181763845696.asia-northeast1.run.app (Google Cloud Run, Tokyo) |
| Tests | 654 passed, 117 skipped (112 need the development repository's private records or layout and are marked so in tests/conftest.py; 5 depend on the machine) in this repository; tests that need the development repository's decision records, its people registry or the local corpus files are left out or skip |
| Verification | every development step up to this release passed an independent internal audit before the next began; the deployment files were self-tested before and after going live |
| Provenance | snapshot of the development repository at commit `9319e7c` |

## Layout

- `docs/plan_module1.md` - the module plan (start here); `docs/grounding_contract_v1.md` - the retrieval contract
- `src/rag4ds/` - the library: `model/`, `auditlog/`, `gate/`, `corpus/`, `staging/`, `kb/`, `index/`, `retrieve/`, `extract/`, `author/`, `evalhook/`
- `packs/robotics/` - the robotics pack: entries, mappings, corpus catalog with sources and licences, demonstration questions, releases and validation records
- `shared/` - the shared layers (general concepts, task patterns)
- `tools/` - `check.py` (lint, tests, scans), `build_index.py`, `demo_web.py` (the demonstration page), `answer_writer.py` (written answers), release, extraction and gate tools
- `deploy/` - the Cloud Run container
- `tests/` - the test suite
- `requirements.lock` / `requirements-linux.lock` - pinned environments (Python 3.11)

Pinned embedding model: `sentence-transformers/all-MiniLM-L6-v2`
@ `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

## Running the demonstration page

```
pip install -r requirements.lock
python tools/build_index.py --out data/index
python tools/demo_web.py --no-answers
```

Then open http://127.0.0.1:8765/. The tools stamp their records in Japan time and refuse
to run on a machine whose clock is not JST (`TZ=Asia/Tokyo` on Linux). Written answers
need a Gemini API key in `GEMINI_API_KEY`; without one the page runs retrieval only.

## Funding

CRF2026 - University of Aizu Competitive Research Funds, FY2026 - pilots this module.

## License

Apache-2.0. Knowledge-base content carries its own source attributions
(`packs/robotics/corpus/sources.yaml`, `shared/corpus/catalog.yaml`).
