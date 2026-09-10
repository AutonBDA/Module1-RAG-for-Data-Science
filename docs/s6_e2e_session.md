# S6 end-to-end gate session (AR-rag4ds-020 O15) - recorded transcript and manifests

Produced by tests/test_s6_gate_e2e.py: a real corpus document ingested through
corpus.ingest, a real index built with the pinned model (all-MiniLM-L6-v2 @
1110a243fdf4706b3f48f1d95db1a4f5529b4d41) over a compiled tree, rag4ds-gate main()
driven through argparse with scripted stdin (approve with a flagged conflict, reject),
then the --retract branch. Synthetic candidates; FUNCTION only, no throughput figure.
Regenerated on 2026-08-23 (JST) in the team venv (offline: the model is cached) from git
HEAD a87e07d; the tree identity the test itself recorded (HEAD, dirty flag, dirty paths) is
reproduced below (AR-rag4ds-020 O17/O21) - the manifests' code_commit names that HEAD, and
the commit that files this document is the next one.

## Tree identity recorded by the test (tree.json)

```json
{
 "head": "a87e07d",
 "dirty": false,
 "dirty_paths": []
}
```

## Transcript (rich export_text, markup off)

```
┌───────────────────────────────── candidate ─────────────────────────────────┐
│  candidate         cand-1977bc7757ebb93c                                    │
│  kind              entry                                                    │
│  pack / layer      robotics / domain                                        │
│  type              concept                                                  │
│  content           Bearing wear dominates actuator failure in rotary        │
│                    joints.                                                  │
│  source_reference  corpus:robotics/handbook#s3                              │
│  extractor         rule-v1 (runs/extract.jsonl:1)                           │
└─────────────────────────────────────────────────────────────────────────────┘
┌────────────────────── source span (local corpus copy) ──────────────────────┐
│ Bearing wear dominates actuator failure in rotary joints.                   │
└─────────────────────────────────────────────────────────────────────────────┘
                related approved entries (conflicts / overlaps)                
┌──────────────────┬───────────┬──────────────────────────────────────────────┐
│ id               │ relevance │ content                                      │
├──────────────────┼───────────┼──────────────────────────────────────────────┤
│ torque-signature │ 0.542     │ A rising torque signature precedes bearing   │
│                  │           │ failure.                                     │
└──────────────────┴───────────┴──────────────────────────────────────────────┘
┌─────────────────────── quotation bound (component 2) ───────────────────────┐
│ the entry states the fact in the project's own words; verbatim quotation is │
│ bounded to a short attributed phrase, never a passage (component 2, NFR5)   │
└─────────────────────────────────────────────────────────────────────────────┘
approve bearing-wear v1 -> journal/decisions.jsonl:2
┌───────────────────────────────── candidate ─────────────────────────────────┐
│  candidate         cand-c945f0167b534f16                                    │
│  kind              entry                                                    │
│  pack / layer      robotics / domain                                        │
│  type              fact                                                     │
│  content           Lubrication schedules are irrelevant to bearing wear.    │
│  source_reference  corpus:robotics/handbook#s3                              │
│  extractor         rule-v1 (runs/extract.jsonl:1)                           │
└─────────────────────────────────────────────────────────────────────────────┘
┌────────────────────── source span (local corpus copy) ──────────────────────┐
│ Section 3.                                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                related approved entries (conflicts / overlaps)                
┌──────────────────┬───────────┬──────────────────────────────────────────────┐
│ id               │ relevance │ content                                      │
├──────────────────┼───────────┼──────────────────────────────────────────────┤
│ torque-signature │ 0.424     │ A rising torque signature precedes bearing   │
│                  │           │ failure.                                     │
└──────────────────┴───────────┴──────────────────────────────────────────────┘
reject cand-c945f0167b534f16 -> journal/decisions.jsonl:3
queue empty (0 skipped this session)
             session summary              
┌────────────────────────────────┬───────┐
│ decision                       │ count │
├────────────────────────────────┼───────┤
│ approve                        │ 1     │
│ edit-then-approve              │ 0     │
│ reject                         │ 1     │
│ reject-duplicate               │ 0     │
│ grouped                        │ 0     │
│ skipped                        │ 0     │
│ retract                        │ 0     │
│ refused                        │ 0     │
│ unrenderable                   │ 0     │
│ candidates_source_unavailable  │ 0     │
│ candidates_display_unavailable │ 0     │
│ decisions_without_source       │ 0     │
│ decisions_without_display      │ 0     │
│ elapsed seconds                │ 0.172 │
└────────────────────────────────┴───────┘

```

## Session manifests (runs/gate-sessions.jsonl)

```json
[
 {
  "aborted": null,
  "approver": "k-naruse",
  "code_commit": "a87e07d",
  "config": {
   "allow_unchecked_approval": false,
   "conflict_display": {
    "embedding_model": {
     "id": "sentence-transformers/all-MiniLM-L6-v2",
     "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    },
    "index_version": "aef5a4c49bfd3d74",
    "min_relevance": 0.2
   },
   "journal": "journal/decisions.jsonl",
   "source_resolver": true,
   "tool": "rag4ds.gate.cli"
  },
  "counts": {
   "approve": 1,
   "candidates_display_unavailable": 0,
   "candidates_source_unavailable": 0,
   "decisions_without_display": 0,
   "decisions_without_source": 0,
   "edit-then-approve": 0,
   "grouped": 0,
   "refused": 0,
   "reject": 1,
   "reject-duplicate": 0,
   "retract": 0,
   "skipped": 0,
   "unrenderable": 0
  },
  "decisions_written": 2,
  "elapsed_seconds": 0.172,
  "ended": "2026-08-23 17:16:08 JST",
  "filters": {
   "layer": null,
   "pack": null
  },
  "kind": "gate-session",
  "repaired": 0,
  "schema": 1,
  "started": "2026-08-23 17:16:08 JST"
 },
 {
  "aborted": null,
  "approver": "k-naruse",
  "code_commit": "a87e07d",
  "config": {
   "allow_unchecked_approval": false,
   "conflict_display": {
    "embedding_model": {
     "id": "sentence-transformers/all-MiniLM-L6-v2",
     "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    },
    "index_version": "aef5a4c49bfd3d74",
    "min_relevance": 0.2
   },
   "journal": "journal/decisions.jsonl",
   "source_resolver": true,
   "tool": "rag4ds.gate.cli"
  },
  "counts": {
   "approve": 0,
   "candidates_display_unavailable": 0,
   "candidates_source_unavailable": 0,
   "decisions_without_display": 0,
   "decisions_without_source": 0,
   "edit-then-approve": 0,
   "grouped": 0,
   "refused": 0,
   "reject": 0,
   "reject-duplicate": 0,
   "retract": 1,
   "skipped": 0,
   "unrenderable": 0
  },
  "decisions_written": 1,
  "elapsed_seconds": 0.0,
  "ended": "2026-08-23 17:16:08 JST",
  "filters": {
   "layer": null,
   "pack": null
  },
  "kind": "gate-session",
  "repaired": 0,
  "schema": 1,
  "started": "2026-08-23 17:16:08 JST"
 }
]
```
