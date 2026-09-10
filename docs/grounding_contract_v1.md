# Grounding contract v1 (plan component 7; FR6, FR10, NFR2, NFR9)

Schema version: **1** (`rag4ds.retrieve.contract.CONTRACT_SCHEMA_VERSION`).
The contract is versioned from day one; a field change bumps the version
and this document moves with it. The library object is
`rag4ds.retrieve.service.Retriever.ground(GroundingRequest) -> GroundingResponse`;
S12's thin HTTP endpoint serializes exactly these objects (`to_dict()` /
`canonical_json()`).

## GroundingRequest

| field | type | rule |
|---|---|---|
| `query` | string | non-blank |
| `layer` | `general` \| `domain` \| `task` \| null | optional layer filter |
| `domain` | pack code \| null | optional domain filter: narrows DOMAIN-layer entries to this pack; general- and task-layer entries stay eligible (they ground every domain) unless `layer` excludes them. A domain naming a pack the build does not serve is `ERROR` with the served packs in `meta` |
| `phase` | one of `business-understanding`, `data-understanding`, `data-preparation`, `modeling`, `evaluation`, `deployment` \| null | optional CRISP-DM phase filter (v1: lexical, whole-word; see below) |
| `top_k` | integer 1..1000 | default 5 |
| `pins` | `{pack code: version string}` \| null | optional pack-version pins (copied at construction); a pin naming a version the build does not serve is `ERROR` with the served versions in `meta.pack_versions` |

A malformed request raises `GroundingRequestError` at construction (the
caller's error); `ERROR` is reserved for the service failing on a
well-formed request.

## GroundingResponse

| field | content |
|---|---|
| `schema_version` | 1 |
| `status` | `OK` \| `NO_GROUNDING` \| `ERROR` |
| `entries` | ranked list; each `{id, version, type, layer, pack, content, relevance_score, source, status}`; non-empty iff `OK`. `status` is the entry's FR1 status, required and validated against the FR1 enum (`current` \| `retracted`); the served set is current-only (section 8), so it reads `current` today |
| `meta` | `contract_schema_version`, `index_version`, `index_schema`, `index_build {code_commit, timestamp}` (the build's NFR2 run fields), `embedding_model {id, revision}`, `pack_versions` served, `min_relevance`, `entry_count`, `filters` (the request's, echoed), `candidates_considered`; `reason` on `NO_GROUNDING`; `error` on `ERROR` |

Status rules (FR6): an empty store or a filter set under which no entry
is eligible answers `NO_GROUNDING` - never a silent empty list; nothing
at or above the relevance floor answers `NO_GROUNDING`; a failure of the
service (embedder, index, pin mismatch, unserved domain) answers `ERROR` -
never `NO_GROUNDING`. `NO_GROUNDING` is answered only after the whole
store is exhausted under the filters: the structural filter runs over the
whole store, never over a shortlist.

`relevance_score` states relevance of the entry to the query (cosine in
[-1, 1], exact, over the pinned embedding model, rounded to six
decimals); it never states confidence in an entry's truth, and no field
is named confidence. Ranking is by the rounded score descending, ties
broken by entry id (deterministic, and explained by the numbers shown).

## The index manifest behind a response

`index_version` is a 16-hex digest of the build's reproducible CORE
(schema, embedding model id+revision, pack versions, relevance floor,
entry count, dimension, `tree_sha256` over the indexed entries'
canonical bytes, `vectors_sha256` over the vector file). A rebuild of the
same tree with the same pinned model reproduces the core and
`index_version` exactly. The build's RUN fields - `code_commit` and
`timestamp` (NFR2) - sit outside that digest and are echoed in
`meta.index_build`, so two builds of one tree are distinguishable while
`index_version` stays a pure function of what was indexed. `load_index`
verifies every part (manifest form and digest, each row's canonical
hash, the tree hash over the rows, the vector file digest, shapes) and
refuses a missing, altered, mixed-generation, or truncated build.

## Declared v1 limits

- Phase filter: lexical, whole-word - an entry is eligible under a phase
  if its id or content names the phase as a whole word or hyphenated
  slug, case-insensitive (never inside a longer word). FR1 carries no
  phase field; S8 seeds the general-layer phase vocabulary.
- Relevance floor (`min_relevance`): build configuration recorded in the
  index manifest; its value is [ASSUMED] until S9 sets it with the
  validator.
- Pack versions served come from build configuration (FR3) until pack
  manifests (4.1) exist at S8/S9.
