# S7 precision sanity check (plan 10.1) - the recorded number and its run manifest

Produced by tests/test_s7_extraction.py::test_precision_measured_and_recorded_no_threshold,
regenerated from a clean working tree at git HEAD 0a419d3 (AR-rag4ds-022 F8: the S7 exit criterion
asks for the number to be RECORDED, so it lives here and not only in a deleted pytest tmpdir).
The code_commit inside the record is read from git at run time, so it names the code that
produced these figures (F11) - it is this same HEAD, and the commit filing this document is the
next one. The `timestamp` field is a FIXED TEST CONSTANT, not the wall-clock of this
regeneration: the run must be deterministic, so the record's NFR2 timestamp is pinned in the
test rather than read from the clock. Every other field is measured.

REGENERATED for AR-rag4ds-023 F1. The run manifest now records
`quotation_max_source_chars`, the ceiling that charges every token in a matched run in source
characters, so both `extraction_id` and the `run_manifest_ref` derived from it differ from the
record filed at the previous pin. The measured figures did not move: the same four candidates
are proposed and none is suppressed, so the new ceiling bounds copying without refusing any
of the work the rule strategy legitimately does (the longest verbatim run any of the four
reaches is 45 source characters, against the ceiling of 64).

NO THRESHOLD IS CLAIMED. The plan defers the extraction-precision target to S9, where it is set
with the validator against real corpus intake; this number is a sanity check of the rule strategy
against a SYNTHETIC six-sentence corpus written for the test, and is not an estimate of field
precision. The oracle is the gold definienda listed below.

## The record (run manifest + measured precision)

```json
{
 "record": {
  "candidates_proposed": 4,
  "candidates_suppressed_for_bound": 0,
  "clustering_embedder": null,
  "clusters": [],
  "code_commit": "0a419d3",
  "config": {},
  "corpus": "synthetic",
  "extraction_id": "3bc2d451445cfc66",
  "kind": "extraction",
  "layer": "domain",
  "min_tokenisable_fraction": 0.5,
  "near_duplicate_cosine": null,
  "pack": "robotics",
  "precision_sanity_check": {
   "correct": 4,
   "precision": 1.0,
   "proposed": 4,
   "suppressed_for_bound": 0
  },
  "produced_sha256": "265bac8d31ccb891c17e156ba6eb738eda989e030c290eff8be08ab08ffeaf35",
  "quotation_max_cjk_chars": 8,
  "quotation_max_source_chars": 64,
  "quotation_max_words": 8,
  "run_manifest_ref": "runs/extraction.jsonl#3bc2d451445cfc66",
  "schema": 1,
  "source_hash": "964d382539a13a326fa4063eb685c1caf599fd48c5cb144a75941ebb1b47a095",
  "strategies": [
   {
    "model": null,
    "name": "rule-v1",
    "params": {
     "definitions": true,
     "gloss_words": 6,
     "headings": true,
     "max_cell_words": 5,
     "sentencizer": "spacy.blank('en') + sentencizer (no language model pinned at VT-3)",
     "tables": true
    }
   }
  ],
  "strategy_kept_counts": {
   "rule-v1": 4
  },
  "suppressed": [],
  "timestamp": "2026-08-23 20:10 JST"
 },
 "corpus_text": "Robotics maintenance notes.\nBearing wear is the dominant failure mode of rotary actuators.\nTorque signature: a rising torque trend that precedes bearing failure.\nBacklash refers to lost motion in a geartrain.\nThe weather was fine and the team went home early.\nAn encoder is a sensor that reports shaft angle.\n",
 "gold_terms": [
  "backlash",
  "bearing wear",
  "encoder",
  "torque signature"
 ]
}
```
