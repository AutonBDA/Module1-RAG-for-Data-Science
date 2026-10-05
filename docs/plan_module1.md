# Module 1 (RAG for Data Science) - Comprehensive Plan

Status: sections 1-4 audit-certified (AR-rag4ds-001 PASSED, 2026-08-20; FR5
extended post-certification by round 5, for the reconstruction test); sections 5-6
revised through review rounds 4-6 (round 6 covered section 7, with one fix
touching section 6.4's conflict flags); section 8 revised through
review round 7; AR-rag4ds-002 r1 fixes F1-F6 verified in r2; r2 fixes F7-F8 applied
2026-08-20 (FR5 decision vocabulary completed; certified-item changes remain
flagged for the sweep); section 9 revised through the AR-rag4ds-003 cycle (r1 fixes F1-F6; r2 fixes
F7-F10) - certified sections amended and flagged for the sweep: FR1 (status enum,
shared pack value), FR5 (marked-carrier wording), FR6 (domain-filter semantics),
FR8/A4 (by-reference supersession), section 4.1 (releases/ added to the anatomy),
section 8's current-only bullet, and per r3: A0 (layer-filtered first clause),
4.5 (mapping drafts YAML, approved artifacts canonical JSON), 6.5 (fold and
preconditions cover gated artifacts), FR5 (mapping-approval payload variant),
section 8's format-split bullet, 9.1 (three fold targets), 5.2 probe wording,
and per r5: FR4 (non-entry candidate authorized), component 3 (mapping-draft
payload), FR5 (candidate-reference-by-hash), 4.3 (snapshot covers mapping
decisions, O17), section 8 and 4.5 filename rules (O18, closing old O2);
section 10 revised through the AR-rag4ds-004 cycle (r1 fixes F1-F9; r2 fixes
F4, F10, F12-F14; r3 fixes F11, F15-F18 - the r2 response falsely claimed four
of these fixed when unasserted script replaces had silently no-opped, raised
and owned as F19; steps S0-S12, shared-layers step S8, per-step figure
replacement, coverage answered, evaluation gate in 4.3 -
4.1/4.3/9.3/section 8 flagged); section 10's effort paragraph amended
2026-08-21 per external checkpoint AR-rag4ds-005 F1 (each figure now names
its measuring step explicitly); six prose lines escaped 2026-08-21
(PI-reported rendering defect: angle-bracket placeholders parsed as HTML
tags by markdown renderers; backslash escapes added in 4.5, 9.1's prose,
and 9.2 - formatting only, no wording change);
section 11 drafted 2026-08-21 and revised the same day per AR-rag4ds-006
r1 (F1 R7 lever stated; F2 R2 blocker list agrees with 10.2; F3 R9's
unrecorded-incident claim deleted; F4 three-word likelihood scale defined
and applied to all nine risks; observations O2-O4 folded in), certified
2026-08-21 (AR-rag4ds-006 PASSED r3); section 8's compile-time RDF clause
struck 2026-08-21 per external checkpoint AR-rag4ds-007 F1 (no step built
it; FR11 is discharged by the 4.5 gate path and VT-1), with I4 gaining its
enforcement pointer and 9.1 housing S12's endpoint (O2, O3). Certification
summary, closing the internal-rounds ambiguity (AR-rag4ds-006 R1 advice,
AR-rag4ds-007 O4): sections 1-4 certified by AR-rag4ds-001, 5-8 by
AR-rag4ds-002 (its r4 coverage reads "sections 5-8 in full" - the internal
review rounds cited above verified nothing; the AR did), 9 by
AR-rag4ds-003, 10 by AR-rag4ds-004 plus external AR-rag4ds-005, 11 by
AR-rag4ds-006; the amendment history above stands as written; section
12's VT-3 marked DISCHARGED 2026-08-21 at S0 (docs/vt3_pins.md), flagged, and RE-OPENED then DISCHARGED again 2026-09-11 for the language model (VT-3b, docs/vt3_pins.md), flagged; section 12 records D-1, OD-1..OD-4, VT-1..VT-3; OD-1 DECIDED 2026-09-12 (PI), flagged; VT-1 DISCHARGED 2026-09-20 at S9d (packs/robotics/standards.yaml; docs/s9_robotics_pack.md section 15), flagged; 6.4 AMENDED 2026-09-24 at S9e part 8 for certified-text approvals from a PI-signed file (PI decision, authoring/robotics_s9e_part8_PI_review_2_signed.txt), flagged. Written from scratch; the July plan
(AutoAI_Cluster/z_module_2_plan.md) is reference material only. No deadline shapes this
document; scope is the whole module.

## 1. Purpose and scope

Module 1 is the expert-validated knowledge framework of the Auton-BDA service. Its job,
in one sentence: hold domain knowledge that human experts have approved, and hand the
right pieces of it to an AI at the moment of reasoning, with every piece traceable to
its source.

The module exists because the service's central hypothesis demands it: an AI can conduct
the full CRISP-DM lifecycle at expert-comparable quality only when its reasoning is
grounded in expert-validated domain knowledge. Module 1 is the grounding half of that
sentence. Everything else in the service (reasoner, orchestration, integrity, the
unified service) consumes what this module holds.

Naming: in the KAKENHI FY2027 proposal this is Module 1; in CRF2026's internal numbering
the same work is Module 2. Under that declared map, this document may call itself
Module 1 (its directory name does the same); every OTHER module, and any cross-track
reference, is named in prose - the knowledge framework, the reasoner - never by bare
number, because the two tracks number the same modules differently.

What this module IS:
- a three-layer knowledge base: general concepts, domain knowledge, task patterns
- a population pipeline in which extraction proposes and a domain expert approves
- a retrieval service that grounds queries in approved knowledge, with sources attached
- the domain-pack mechanism that lets new domains plug in without changing the core

What this module is NOT (boundaries):
- not the reasoner: it answers "what is known", never "what should be done next"
- not orchestration: it does not run pipelines
- not the integrity ledger: it keeps its own audit trail, but the service-wide
  provenance chain is the integrity layer's job; this module exposes its
  records to it
- not the benchmark: AutonBDA-Bench is a consumer of this module, built beside it

Sentences this module must make true (the specification, quoted or abridged from the
funding documents; each carries into the tagged requirements of section 2):

| # | Claim (abridged) | Source |
|---|---|---|
| C1 | a three-layer knowledge base holds general concepts, domain knowledge, and task patterns | KAKENHI (4) |
| C2 | population is semi-automatic: extraction suggests entries, and the domain expert approves each via a validation gate | KAKENHI (4) |
| C3 | every entry in the knowledge base passes the expert validation gate | KAKENHI (2),(5) |
| C4 | retrieval grounds each reasoning step | KAKENHI (4) |
| C5 | answers are scored against expert-checked sets by different authors | KAKENHI (4) |
| C6 | domains plug in; the service does not change | Figure 2 |
| C7 | the robotics knowledge aligns with the IEEE 1872 family of robot ontology standards | KAKENHI (6) |
| C8 | the knowledge framework runs with repository, releases, and demonstration public | KAKENHI (2),(5) |
| C9 | the framework is ontology-driven, grounds AI answers through retrieval with full auditability, and every entry carries its source | SUMMARY; Figure 2 |
| C10 | the service is demonstrated in three scenarios - robotics (industry), bioinformatics (research), and grocery (retail) - which together exercise DM, ML, and DL | SUMMARY; KAKENHI (2) |

Note on C7: an agreed strengthening of this sentence (reuse of the CORA classes of
IEEE 1872) is drafted but not yet applied to the master. FR11 targets the stronger
form; the weaker locked sentence is satisfied by the same work.

## 2. Requirements

Functional requirements. Each is testable; each traces to a claim above.

- FR1 (C1, C3, C9). Entry model. An entry is one unit of approved knowledge. Types:
  concept, relation, fact, task pattern. Mandatory fields: id (format per
  section 8's entry-id rule), type, layer, pack (a pack code for domain-layer
  entries; the reserved value shared - never a pack code, section 4.1 - for
  general- and task-layer entries), content, source reference, version, status
  (current | retracted; a superseded version is marked by the superseding
  decision that names it, never by a stored status - section 9.1). An entry
  missing any mandatory field fails the build, loudly.
- FR2 (C1). The knowledge base stores entries in exactly three layers: general concepts
  (CRISP-DM and data science vocabulary, shared by all domains), domain knowledge
  (entities, relations, and facts of one domain), and task patterns (analysis task
  types such as classification, forecasting, anomaly detection, clustering, and their
  requirements). Every entry declares its layer.
- FR3 (C6). A domain is a pack: a self-contained set of domain-layer entries plus its
  corpus references, validation records, external-standard mappings, and demo query
  flows (full anatomy: section 4.1). Packs load at configuration time without any
  core code change; the core contains no domain-specific logic or vocabulary.
- FR4 (C2). Population is semi-automatic. An extractor reads corpus documents and
  proposes candidate entries into a staging area. Candidates are visibly candidates;
  they are unreachable by retrieval. A candidate is usually a proposed entry; a
  mapping draft (4.5) is the one non-entry candidate kind, taking the same
  staging road to the same gate.
- FR5 (C3). The expert gate is the only path from staging into the knowledge base.
  Decision types, the complete set: approve, edit-then-approve, reject,
  reject-duplicate (mechanical - written by the gate when a cluster head is
  decided, citing the head decision as its authority), and retract. A
  supersession is not a separate type: it is an approval whose embedded entry
  carries an existing id at version+1 and names what it supersedes. Every
  decision records the approver, the timestamp, and its type; the rest is
  conditional on the type. Approve and edit-then-approve records carry the
  original candidate text, its source reference, and the complete approved entry
  with every FR1 field as it will serialize - id (assigned at gate time), type,
  layer, pack, content, source reference, version, status - so the audit log
  alone can rebuild the knowledge base byte for byte (section 6.5). Reject and
  reject-duplicate records carry the candidate, its source reference, and the
  reason - no entry. Retract records carry the target entry id and version and
  embed the entry's final form marked status retracted - the retracted marking
  FR8 relies on - and direct the fold to remove the entry's tree file. One
  payload variant: a mapping-file approval (4.5) embeds the canonical-JSON
  mapping artifact in place of an entry and carries the draft's content hash as
  its candidate reference in place of candidate text; the remaining fields are
  the same. The
  audit record is append-only. A test proves that an unapproved entry cannot be
  retrieved.
- FR6 (C4). Retrieval accepts a query with optional filters (layer, domain, CRISP-DM
  phase, top-k) and returns approved entries with relevance scores and full source
  attribution. Responses carry one of three statuses: OK, NO_GROUNDING (nothing
  relevant exists), or ERROR (the service failed). An empty store answers NO_GROUNDING,
  never a silent empty list; a failure answers ERROR, never NO_GROUNDING. The domain
  filter narrows domain-layer entries to the named pack; general- and task-layer
  entries stay eligible under a domain filter (they ground every domain) and are
  excluded only by an explicit layer filter. Scores state relevance to the query,
  never confidence in an entry's truth; field names say relevance, not confidence.
- FR7 (C3, C9). Provenance is a chain, walkable in both directions: source document ->
  candidate -> gate decision(s) -> knowledge entry and its versions -> retrieval answer.
- FR8 (C3, C9). Corrections. An approved entry changes only through a new gate decision
  that supersedes or retracts it. A superseded version remains stored complete in
  the append-only audit log as its original approval record, marked superseded BY
  the superseding decision that names its id and version - never by rewriting the
  record; a retracted entry's final form is embedded, marked retracted, in its
  retract record (FR5). The source tree holds current versions only (section 8);
  retrieval serves current versions only; nothing is ever silently deleted. The
  audit shows the full version history of every entry.
- FR9 (C10). Each of the three scenarios has a demo query flow that retrieval
  answers from its pack: actuator-failure knowledge (robotics), treatment-outcome
  knowledge (bioinformatics), demand-and-basket knowledge (grocery). Each flow
  records its expected outcomes (status, entry ids), making it a mechanical test.
- FR10 (C5, future modules). The module exposes: a grounding interface that the
  future reasoner calls; read access to audit records for the future integrity layer;
  and an evaluation hook through which answer sets (authored by people other than the
  knowledge authors) score retrieval quality.
- FR11 (C7). The robotics pack's schema maps its classes to the CORA classes of
  IEEE 1872 where equivalents exist, and records the mapping explicitly.

Non-functional requirements.

- NFR1 Auditability: every state change of the knowledge base is reconstructable from
  the audit trail alone.
- NFR2 Reproducibility: every pipeline run (extraction, indexing, evaluation) writes a
  run manifest: code commit, configuration, model identifiers and versions, input
  hashes, timestamp.
- NFR3 Domain independence, mechanically enforced: every pack declares its own lexicon
  (its domain terms) in pack metadata; a test scans core code, the general layer, and
  the task layer for any declared lexicon term, matching whole words and phrases,
  never substrings ("omics" must not hit "economics"). Domain words may appear only
  inside their own pack. Two rules keep the scan honest instead of hollow: shared-layer
  entries are named domain-neutrally (the scan over the shared layers enforces
  exactly that), and a collision is resolved by renaming the shared entry or by
  narrowing the lexicon term to a domain-specific phrase - ruled at the gate by the
  shared-layer designee (OD-1), never by silently dropping the term from the
  lexicon.
- NFR4 Hardware: the whole module runs on existing hardware (CPU-class); embedding
  models are chosen CPU-viable. Nothing in Module 1 requires GPU training.
- NFR5 Openness (carries C8): Apache-2.0; the repository is publishable at any
  moment: no secrets, no personal data, no restricted-license corpus content in
  git. Claims may be carried by NFRs as well as FRs; C8 is the one that is.
  Standing rule (PI decision 2026-09-13, flagged post-certification;
  audit/rag4ds/responses/PI_decision_history_2026-09-13.md): "in git" is the
  tracked tree at a release; this repository's HISTORY is never published or
  transferred, and a release is an export of the AutonBDA/ subtree at a tagged
  commit - never a push, mirror or fork of the repository. Twelve commits of
  2026-09-12 carry six sentences of a restricted document (AR-rag4ds-026 F1,
  AR-rag4ds-027 F5) and are not rewritten.
- NFR6 Corpus licensing: git stores the corpus catalog only (content hash, license,
  fetch pointer, and metadata per document). Raw documents live outside git in a local
  data directory; a document is vendored into the repository only when its license
  permits redistribution.
- NFR7 Sensitive data: no personal data anywhere in the module (corpus, staging,
  entries, logs). The bioinformatics pack uses public, de-identified datasets only.
  Every pack declares the license and ethics status of its sources.
- NFR8 Versioning: knowledge releases are versioned per pack (semantic versioning);
  a released version is immutable.
- NFR9 Fail-loud: no silent fallbacks anywhere; every degraded path raises or returns
  ERROR with a reason.

Acceptance cases (the three scenarios as tests).

- A1 robotics: given the robotics pack at some released version, when retrieval is
  asked about abnormal joint-torque patterns before actuator failure, it returns
  approved robotics entries with sources, status OK, and the provenance chain of each
  entry resolves to a corpus document and a gate decision by the designated robotics
  expert (currently K. Naruse).
- A2 bioinformatics: same shape, treatment-outcome query, bioinformatics pack.
- A3 grocery: same shape, demand-forecast and basket-grouping query, grocery pack.
- A0 negative: a query filtered to the domain layer against an empty or
  irrelevant pack returns NO_GROUNDING - the layer filter makes the clause
  meaningful under FR6, since shared-layer entries otherwise stay eligible by
  design; a deliberately broken backend returns ERROR; an unapproved staged
  candidate never appears in any answer.
- A4 correction: after a gate decision supersedes an approved robotics entry,
  retrieval returns only the new version; the old version remains stored in the
  audit log, marked superseded by the superseding decision that names it; the
  audit shows both decisions, and the provenance chain of the new version
  resolves fully.
- A5 new domain (the C6 test): given the core at a fixed version, a pack not
  previously known to the system, with a valid manifest, loads from configuration
  alone; its demo flows answer with status OK against their expected outcomes; the
  NFR3 scan stays clean; and git history shows zero core commits in between.

Assumptions and non-goals (v1). Stated now so nobody discovers them later:

- English-only: corpus, entries, and queries are English in v1.
- One expert per pack; the gate is single-writer; no concurrent review sessions.
- Single-user service; no authentication or multi-tenancy in v1.
- Latency is measured and reported, not gated; interactive response on CPU is the
  aim, not a contract.
- Scale envelope for v1: hundreds to a few thousand entries per pack.
- Packs are mutually independent: no pack references another pack's entries;
  cross-domain links go through the shared layers (section 4.2).

Each of these can be lifted later without breaking the contracts above.

## 3. Architecture

Overview (data flows left to right; audit underneath everything):

    corpus docs        staging           gate              knowledge base        consumers
    (per domain)   ->  candidate     ->  expert         -> three layers      ->  retrieval API
    PDF/text/CSV       entries           approve/          general | domain      -> reasoner
                       (extractor        reject/           | task, as packs      -> benchmark
                       proposes)         edit              versioned             -> demos
         |                  |               |                  |                     |
         +------------------+---------------+------------------+---------------------+
                                append-only audit + run manifests

Components, each with its single responsibility:

1. Corpus store (satisfies FR7; NFR6, NFR7). Immutable source documents per domain, content-hashed on ingest. Git
   holds the catalog: hash, license, fetch pointer, and metadata per document (NFR6);
   raw files live in a local data directory outside git, vendored only when their
   license permits redistribution. A document is never edited; new versions are new
   documents. Every ingest writes a run manifest (hash, license, source, timestamp),
   so the audit trail starts at the document, not at staging.
2. Extractor (satisfies FR4; NFR2, NFR5). Reads corpus documents, proposes candidate entries (concept, relation,
   fact, or task pattern) with the source span attached. Candidate content states the
   fact in the project's own words: verbatim quotation is bounded (a short attributed
   phrase, never passages), and source spans are referenced by document hash plus
   character offsets, not by copied text - so restricted-license corpus content never
   enters git (NFR5, NFR6). Pluggable strategies: rule and embedding based first; an
   LLM-assisted strategy can be added later behind the same interface. The extractor
   has no write access to the knowledge base; it writes only to staging.
3. Staging area (satisfies FR4, FR5). Holds candidates awaiting review. An entry
   candidate carries: proposed entry, layer, target pack, source document and
   span, extractor identity and run manifest. A mapping-draft candidate carries,
   mirroring FR5's payload variant: the YAML draft in place of a proposed entry,
   its target pack and external standard, and its author identity.
4. Expert gate (satisfies FR5, FR8). A review tool (command line first; recorded as
   decision D-1, section 12) that walks an
   expert through pending candidates. Decisions append to the audit log. Approving an
   entry includes confirming its content honors the quotation bound (component 2).
   Approved entries compile into the pack; rejected ones remain in staging history.
   Corrections and retractions of approved entries pass through this same gate -
   a correction as a superseding approval at version+1, a retraction as its own
   retract decision (FR5, FR8).
5. Knowledge base (satisfies FR1, FR2, FR3; NFR8). The three layers. Source of truth is versioned files in the
   repository (human-diffable); a build step compiles approved entries into the
   serving indexes. Two indexes are derived, never authoritative: a structural index
   (typed entries, relations, layer and pack filters) and a vector index (embeddings
   for similarity search). If the derived indexes are deleted, the structural index
   rebuilds exactly from the source files; the vector index rebuilds reproducibly
   given the embedding-model version pinned in its index manifest. The build step
   doubles as the pack loader: which packs, at which released versions, are active
   is configuration read at build time - never code (FR3).
6. Retrieval service (satisfies FR6, FR9). Query -> embed -> structural pre-filter (layer, pack, phase)
   where the index supports it, then vector search within the filtered set; where
   pre-filtering is not supported, over-fetch the shortlist and refill until the
   filtered set satisfies top-k or the store is exhausted. NO_GROUNDING is answered
   only after the store is exhausted - a filtered-out shortlist never produces it
   (FR6). Results: ranked entries with relevance scores, sources, and status. Exposed
   as a library first and a thin HTTP endpoint second; both share one implementation.
7. Grounding interface (the reasoner contract, defined now, consumed later; satisfies FR10, NFR2).
   GroundingRequest: query text, optional CRISP-DM phase, optional domain, top_k,
   and optional pack-version pins - a replayed reasoning run can demand the exact
   pack releases it used before (NFR2); default is the build's configured versions,
   and the versions actually served are always echoed in meta. Pins are binding: a
   pin naming a version the current build does not serve returns ERROR with the
   served versions in meta (NFR9) - replaying against other versions is a rebuild,
   never a request parameter.
   GroundingResponse: status (OK | NO_GROUNDING | ERROR), entries (each with id,
   layer, pack, content, relevance_score, source), and meta (pack versions, index
   version, run identifiers). The schema is versioned from day one.
8. Audit and manifests (satisfies NFR1, NFR2; FR10). Append-only log of gate
   decisions and pack lifecycle transitions (creation, release, retirement -
   section 4.3); run manifests for every pipeline execution. Readable by the
   future integrity layer.
9. Evaluation hook (satisfies FR10, FR9, C5). A harness that feeds expert-checked answer sets - authored by
   people other than the knowledge authors (FR10) - through the same retrieval
   library and scores the results, writing a run manifest like any pipeline run.
   The same harness executes pack demo flows (FR9), which are pack-authored and
   thus distinct from the separately-authored answer sets. AutonBDA-Bench attaches
   here later without redesign; the "benchmark" consumer in the diagram is this
   hook.

Layer semantics (what lives where):

- General concepts: the CRISP-DM process model (phases, their inputs and outputs),
  core data science vocabulary (dataset, feature, model, metric, evaluation,
  deployment). Hand-authored, small, stable, shared by every domain. This layer is
  where the reasoner's phase vocabulary anchors.
- Domain knowledge: per-pack entities, relations, and facts (a robot joint, its
  torque signal, the failure modes of an actuator; a treatment, an outcome measure,
  a biomarker; a product category, a demand pattern). Populated through the pipeline;
  the only layer that grows continuously.
- Task patterns: analysis task types and their requirements (what a forecasting task
  needs from data, what anomaly detection produces, which metrics fit which task).
  Hand-curated with extraction assistance; shared across domains; this layer is what
  lets the same service serve DM, ML, and DL requirements.

Hand-authored and hand-curated entries take the same road as extracted ones: they
enter staging as candidates - the author is just another proposer - and pass the
expert gate. Invariant I2 has no exception. Their source reference (FR1) cites the
authoritative text they restate (for CRISP-DM, its published process guide) or, for
original knowledge, names the author as the source.
Mechanically (S8, flagged amendment 2026-09-12): the text restated is a catalogued
document of the shared corpus (9.1), the candidate spans the passage it restates
(made mechanical at AR-rag4ds-027 r1, 2026-09-13, flagged: an anchor locates its
whole paragraph and a locator whose text supports under half of the content's
words is refused as a fragment - the S8 spans were sentence-long locator
extents until then), and the quotation bound is checked against that passage
at filing - so a hand-
authored candidate carries the same evidence an extracted one does; original
knowledge is a team-authored source note, catalogued the same way, whose author is
named in the source reference.

Gate ownership: each domain pack is gated by its domain validator (robotics:
currently K. Naruse; T. A. Siriweera is registered as a second robotics
validator, PI decision 2026-10-05, flagged post-certification - each decision
records its approver, so the records say which robotics approvals are his,
and nobody approves a candidate they authored). The general and task layers are gated by a designated data
science expert from the team, recorded per gate decision; the designee is
T. A. Siriweera with I. Paik as alternate, the alternate gating any entry the
designee authored (OD-1, decided 2026-09-12, flagged post-certification).

Design invariants (the rules everything above obeys):

- I1. Versioned source files are the single source of truth; serving indexes are
  disposable derivations.
- I2. No path into the knowledge base bypasses the gate; enforced by construction
  (the compiler only reads approved entries) and by test (A0).
- I3. The core is domain-blind (NFR3); everything domain-specific lives in a pack.
- I4. Every answer is explainable: entry -> approval -> source, no exceptions
  (exercised by A1 and A4's citation checks and 7.1's grounding-integrity
  metric; 9.2 requires a failing test per invariant).
- I5. Interfaces are versioned contracts; consumers depend on the contract, not the
  implementation.

## 4. The domain-pack contract

This section carries the scaling promise of the whole service (C6): a new domain is a
new pack, and a pack is data - the core never changes. What follows is the contract
that makes that sentence mechanical rather than aspirational.

### 4.1 What a pack is

A pack is a self-contained directory of data files - never code - holding everything
one domain contributes to the service:

    packs/<code>/
      manifest      pack identity: code (shared is reserved and never a pack
                    code, FR1), name, semver version, the domain validator
                    (role and current designee), license and ethics declaration
                    (NFR7), the domain lexicon (NFR3), minimum core schema version
      corpus/       the corpus catalog: per-document content hash, license, fetch
                    pointer, metadata (NFR6); raw documents stay outside git
      entries/      approved domain-layer entries, versioned source files (I1),
                    each carrying the FR1 mandatory fields
      mappings/     approved mapping artifacts, canonical JSON (4.5); drafts
                    travel through staging like any candidate
      validation/   release-time snapshot of the gate decisions behind the
                    shipped gated content (4.3)
      releases/     release manifests, one per release (4.3)
      demo/         the scenario demo flows (FR9): queries with expected outcomes
                    (expected status and entry ids), so "passing" is a mechanical
                    check, not a judgment

File formats here are illustrative; the concrete choices are section 8 decisions. The
contract is structural: these parts, these obligations.

The core schema version, referenced by the manifest: the core publishes a single
version number covering the two schemas packs depend on - the entry model (FR1) and
the manifest schema itself. It bumps only when those fields change. A pack declares
the minimum core schema version it requires; the build refuses a pack the running
core cannot honor, loudly (NFR9).

Two hard rules follow from section 3 and are restated as pack law:

- A pack contains only domain-layer entries. General concepts and task patterns live
  in the shared layers; a pack links to them, never redefines them.
- A pack cannot introduce new entry types, new code, or schema extensions. Domain
  structure is expressed as relations among typed entries (FR1). A domain that
  appears to need a new type is a core design question and goes to section 12, not
  into a pack.

### 4.2 Pack independence, mechanically enforced

- The manifest lexicon is the NFR3 scan input: every domain term the pack owns. Core
  code, the general layer, and the task layer are scanned against every pack's
  lexicon; a hit fails the build.
- Packs do not reference each other's entries in v1. Cross-domain connections go
  through the shared layers or not at all; this keeps unloading safe and domains
  independently releasable. (Liftable later; the assumptions block in section 2
  applies.)
- The new-domain test is acceptance case A5 (section 2): a new pack loads from
  configuration alone, answers its demo flows, and scans clean - with zero core
  commits.

### 4.3 Lifecycle: from corpus to versioned release

A pack moves through four states; every transition leaves audit records.

1. DRAFT: manifest exists, validator named, licenses declared. Nothing retrievable.
2. POPULATING: corpus documents registered into the catalog (ingest manifests,
   component 1); extraction proposes candidates into staging; the domain validator
   works the gate (FR5). Population is incremental - there is no big-bang import;
   the gate is the rate limiter by design.
3. RELEASED: a semver release freezes an immutable set of gated content -
   entries and mapping artifacts (NFR8). Release
   semantics, by structure rather than by judgment calls, over both gated-artifact
   kinds (entries and mapping artifacts, addressing observation O22 of
   AR-rag4ds-003): PATCH supersedes existing entries or mapping artifacts with
   corrected content (FR8); MINOR adds entries or mapping artifacts; MAJOR
   retracts either, or restructures the manifest or lexicon.
   The release checklist: every entry and mapping artifact gated; licenses verified against the catalog;
   lexicon current; demo flows pass against their expected outcomes; independence
   scan clean; from the evaluation hook's activation onward (section 10, S10),
   the release's evaluation run filed against the agreed targets; the release
   manifest records the gated-content hash - entries and mapping artifacts
   together. Vector indexes are not release artifacts: they are
   derived at build time from the configured pack set, and the embedding-model pin
   lives in the build's index manifest (section 3, component 5). At release time the
   build copies the gate decisions behind the
   shipped entries and mapping artifacts into validation/ as a checksummed snapshot - the global audit log
   stays the single source (component 8), but a released pack carries its own
   provenance and can be published or moved as one unit.
4. RETIRED: unloaded from configuration; the pack and its history remain in the
   repository forever - nothing is deleted (FR8 spirit at pack scale).

Loading and unloading are configuration acts (FR3, component 5): the configuration
names pairs of (pack, released version); the build step compiles exactly those. No
load-order effects exist, because packs are independent (4.2).

Pre-release loading (the bootstrap path): a pack in POPULATING may be loaded
locally from the working tree, so its demo flows can run before any release
exists - otherwise the release checklist could never be satisfied for a first
release. A dev load is marked: retrieval meta labels the pack version unreleased,
the build accepts it only in a local development configuration, and nothing
unreleased ever serves the reasoner or the evaluation of released packs. The
release checklist's demo-flow check runs against exactly this dev load.

### 4.4 What the core knows (the whole coupling surface)

The core knows exactly four things about any pack, and nothing else:

1. the entry model (FR1) - types and mandatory fields;
2. the layer semantics (FR2) - a pack populates the domain layer only;
3. the manifest schema (4.1) - identity, lexicon, licenses, validator;
4. the retrieval filters (FR6) - a pack code is a filter value.

Everything else is the pack's private business. This list is the whole scaling
argument: if adding a domain requires touching anything beyond a new pack directory
and one configuration line, the contract is broken and that is a defect (I3).

Demo flows are executed by the evaluation-hook harness (section 3, component 9),
which already speaks the retrieval library; the core itself never reads demo/ -
so the list above stays four items.

### 4.5 External-standard mappings (FR11)

Any pack may declare mappings from its entries to an external standard or ontology;
the robotics pack must. The mapping is a data file per standard: each row maps one
pack entry to one external class with a relation (equivalent | narrower-than |
related) and a note. Mapping files pass the same gate as entries - the domain
validator approves them - and releases snapshot them. The author drafts in YAML
(a human-edited file, section 8); at approval the gate canonicalizes the draft
to canonical JSON, and THAT artifact is what the decision embeds and what lives
under packs/\<code>/mappings/, named \<standard-id>.json - the standard id obeys
section 8's entry-id character rule and is recorded in the artifact itself, so
the fold's filenames are fully determined. Gate granularity: one decision per mapping-file
version, recorded with the FR5 fields, the draft's content hash as the
candidate reference, and the canonical artifact embedded in full - so the fold
rebuilds packs/\<code>/mappings/ from the log byte for byte, exactly as it
rebuilds entries (6.5).

For robotics the target is the CORA ontology family of IEEE 1872, per C7 and FR11.
[UNVERIFIED] The concrete class list to map against must be verified against the
actual IEEE 1872 text before the robotics pack's first release; recorded as VT-1 in
section 12 (the proposal-side C7 note tracks the same dependency).
VT-1 DISCHARGED 2026-09-20 at S9d, flagged: the class list was verified against
the text of IEEE Std 1872-2015 (the PI's own copy, pinned by sha256 in the pack,
kept outside git) and is recorded as pack DATA in packs/robotics/standards.yaml -
the classes by clause under their sub-ontology prefix (corax, cora, rparts, pos),
and separately every relation the same clauses declare, by a subrelation axiom or
by an axiom typing the term as a relation, predicate or function (the 7.2 part
role and the 7.4 autonomy roles are relations in the text, not classes), so that a
mapping row can name only a verified class; the verification itself is
docs/s9_robotics_pack.md section 15 (the relation list completed to all 42 after
AR-rag4ds-042 r1 F1, flagged). IEEE Std 1872.2-2021 (the AuR ontology) is
registered beside it as the extension reference. The draft road is
tools/file_mapping_draft.py (generic; the pack is data): it refuses a row naming
an unverified class, a relation, or an entry the tree does not hold, and files
the draft's bytes verbatim as the mapping-draft candidate this section's gate
path decides.

### 4.6 How DM, ML, and DL requirements map to the task-pattern layer

The task-pattern layer (shared, not in packs) is where the "across DM, ML, and DL"
claim becomes concrete. Each task-pattern entry declares: the task type, the data
properties it requires, what it produces, the technique families that implement it
(data mining, machine learning, deep learning), and the metrics that measure it.
Domain entries link to task patterns - "actuator failure modes" links to "anomaly
detection on multivariate time series" - and that link is the retrieval join the
future reasoner walks from a domain need to a grounded analysis approach.

The three scenarios jointly exercise all three families:

| Scenario | Task patterns exercised | Family |
|----------|-------------------------|--------|
| Robotics: actuator-failure prediction | anomaly detection and failure prediction on multivariate time series | ML, DL |
| Bioinformatics: treatment outcome | classification on high-dimensional tabular data | ML, DL |
| Grocery: demand and baskets | time-series forecasting; association-rule (co-occurrence) analysis | ML, DM |

New task patterns are added like all shared-layer entries: through staging and the
gate, owned by the shared-layer designee (OD-1).

### 4.7 What adding a domain costs (honest scaling economics)

A new domain costs: corpus gathering, extraction runs, the domain expert's hours at
the gate, demo-query authoring, and a release cycle. It costs zero core engineering -
that is the contract. The expert hours are the true critical path and are estimated
per domain in section 10; the contract makes them the only scarce ingredient, which
is exactly the scaling property the service claims.

One honesty note: the zero-core-engineering property is provable only from the
second domain onward. Robotics, as the first pack, co-forges the core alongside
this contract; bioinformatics and grocery are the proof that the contract holds.

## 5. The three scenarios

Each scenario is one domain pack (section 4) plus the demo flow that proves it
(FR9, A1-A3). This section fixes, per scenario: the use case, the data, the corpus
sources, the validating expert, the target depth, and the demo queries. C10 is the
governing claim; section 4.6's table already maps each scenario to its task
patterns.

### 5.1 Why these three

The trio is chosen so that no pair of scenarios shares a sector, a data shape,
or a validator archetype: industry, research, and commerce; multivariate time
series, high-dimensional tabular, and transactions; an engineer, a researcher,
and a manager - so the expert gate is exercised by genuinely different kinds of
expert, not one persona three times. Jointly the three cover all three
technique families (ML, DL, DM; the section 4.6 table). A knowledge framework that serves
all three without core changes has demonstrated the domain-independence claim
(C6) about as hard as three domains can.

Depth vocabulary, used below: a SEEDED pack holds enough gated entries to answer
its demo flows (order of tens of entries); a DEEP pack is systematically
populated from its corpus (the section 2 scale envelope: hundreds to a few
thousand). Depth targets here are planning assumptions [ASSUMED]; the expert
hours they cost are estimated in section 10.

### 5.2 Robotics (pack code: robotics) - the deep, first pack

- Use case: actuator-failure prediction - grounding an analysis that predicts
  failure from joint and actuator telemetry.
- Data shape: multivariate time series (joint torque, actuator readings).
- Corpus: the validator-provided robotics corpus (maintenance documentation and
  failure-mode material; the proposal names the validator as its provider), plus
  published robotics-maintenance literature where licenses permit catalog entry
  (NFR6). [ASSUMED] the validator corpus arrives in usable text form; the intake
  path is the ordinary one (catalog, hash, license per document).
- Validator: the designated robotics expert (currently K. Naruse), with
  T. A. Siriweera as a second robotics validator from 2026-10-05 (PI decision,
  flagged post-certification; section 3, gate ownership).
- Depth: DEEP. Robotics is the first pack and forges the core alongside the
  contract (4.7); it also carries the CORA mapping obligation (FR11, VT-1).
- Demo flow (expected outcomes bound at gating time, per FR9): (1) "which torque
  patterns precede actuator failure in rotary joints" - expects domain entries
  (failure modes, torque signatures) plus the anomaly-detection task pattern,
  status OK; (2) "what data does failure prediction need from a robot arm" -
  expects the failure-prediction task pattern with its data-requirement fields
  and linked domain entries; (3) a deliberately out-of-domain query filtered to
  the domain layer - expects NO_GROUNDING (this probe covers A0's
  irrelevant-query clause only, in A0's layer-filtered form; A0's
  broken-backend and staged-candidate clauses are separate tests).

### 5.3 Bioinformatics (pack code: bioinformatics) - seeded, deepened second

- Use case: treatment-outcome classification - grounding an analysis that
  predicts patient outcome classes from clinical and omics-derived features.
- Data shape: high-dimensional tabular (public, de-identified only; NFR7 is
  absolute here).
- Corpus: published methods and review literature for outcome prediction, plus
  the documentation of public de-identified datasets. Candidate dataset families
  [UNVERIFIED]: TCGA-class public omics resources and UCI-class clinical
  benchmarks; the concrete choice and its license and ethics verification are
  VT-2 (section 12) and must close before corpus registration.
- Validator: OPEN - no bioinformatics validator is named yet (OD-2, section 12).
  Because every domain entry passes the domain validator's gate (I2, no
  exceptions), OD-2 blocks seeding too, not just deepening - acceptance case A2
  depends on OD-2 closing. Until then only preparation proceeds: corpus
  candidates into staging (unreachable by retrieval, so no gate is needed) and
  shared task patterns via the OD-1 designee.
- Depth: SEEDED first, deepened after robotics has hardened the pipeline.
- Demo flow: (1) "which patient features are established predictors of treatment
  outcome" - expects domain entries with literature sources; (2) "what does
  classification need from high-dimensional tabular data" - expects the
  classification task pattern; (3) the out-of-domain NO_GROUNDING query.

### 5.4 Grocery (pack code: grocery) - seeded

- Use case: demand forecasting and basket analysis - grounding analyses of what
  sells when, and what sells together.
- Data shape: transactions (timestamped purchase records).
- Corpus: retail-analytics literature and the documentation of public
  transaction datasets. Candidate dataset families [UNVERIFIED]: Instacart-class
  and UCI Online-Retail-class public sets; concrete choice, license, and any
  personal-data screening (NFR7) are part of VT-2.
- Validator: OPEN - no grocery-domain validator is named yet (OD-3, section 12).
  Same rule as OD-2: seeding itself waits for the validator (I2), so acceptance
  case A3 depends on OD-3 closing; staging preparation may proceed.
- Depth: SEEDED. Grocery is a breadth prover, not a depth prover: bioinformatics
  and grocery are jointly the two proof points for A5's zero-core-commits claim,
  and their creation order is fixed in section 10's increment sequence, not here.
- Demo flow: (1) "which demand patterns matter for forecasting perishables" -
  expects domain entries; (2) "which analysis tasks fit basket data" - expects
  the association-rule task pattern; (3) the out-of-domain NO_GROUNDING query.

### 5.5 What the three jointly prove

Robotics proves depth: a real expert gates real corpus knowledge to answerable
density, with a standards mapping. Grocery proves breadth: a second domain loads
and answers with zero core commits (A5). Bioinformatics proves the sensitive
edge: the framework operates where data ethics are strictest, on public
de-identified material only. Together they discharge C10 and give A1-A3 their
concrete content; what none of them proves alone - and the plan does not claim -
is expert-comparable lifecycle quality, which is the reasoner's hypothesis to
test, not the knowledge framework's.

## 6. Population pipeline

This section makes C2 operational: what "semi-automatic" means precisely, how
candidates are made, and how the gate works day to day. The architecture is fixed
in section 3 (components 1-4); nothing here changes it.

### 6.1 What "semi-automatic" means, precisely

The machine proposes; the human disposes. Exactly four rules:

1. The extractor may read any registered corpus document and may propose anything
   into staging. Proposals have zero effect on the knowledge base.
2. Nothing becomes knowledge except through a gate decision by the responsible
   expert (I2). There is no auto-approval, no confidence threshold that bypasses
   the gate, no batch-accept - not now, not as a later optimization. The moment a
   proposal can become knowledge without a human decision, the central hypothesis
   is no longer being tested.
3. The expert may approve, reject, or edit-then-approve (FR5), and may author
   candidates directly - authorship is just another proposer (section 3).
4. The automated share of the work is an outcome, not a target. Quality is set by
   the gate; throughput is bounded by expert hours, which section 10 treats as
   the true critical path.

### 6.2 Extraction strategies

A strategy implements one interface: registered document in; typed candidates out,
each carrying the proposed entry, layer, target pack, source span (document hash
plus character offsets, quotation-bounded per section 3, component 2), strategy
identity, and a run manifest (NFR2).

v1 ships two strategies: rule-based (definition and glossary patterns, heading
and section heuristics, table extraction for structured facts) and
embedding-assisted (similarity clustering to propose related-concept candidates;
near-duplicate detection). An LLM-assisted strategy can come later behind the
same interface; it must run within the hardware envelope (NFR4), honor the
quotation bound, record model identity and version in its manifests, and respect
data egress: local-first by default, corpus content leaves the machine only
where the specific document's license permits it, and never for anything in
NFR7's sensitive scope - the concrete model choice is a section 8 decision.

The tuning bias is precision over recall: gate hours are the scarce resource, so
fewer, better candidates beat exhaustive noise. Near-duplicates are grouped and
shown as a cluster, never silently dropped (NFR9) - suppression the expert cannot
see is a silent fallback by another name.

### 6.3 Staging semantics

A candidate, once filed, is immutable; decisions reference it. Candidate states,
all five: PENDING, APPROVED, REJECTED, REJECTED-DUPLICATE, GROUPED (member of a
duplicate cluster whose head is under review). GROUPED has two exits: when the
head receives ANY decision - approve, reject, or edit-then-approve - members
close mechanically as REJECTED-DUPLICATE, each with an audit record referencing
the head's decision - a recorded mechanical consequence that creates no
knowledge, so I2 is untouched; and the expert may detach a member from its
cluster before the head is decided, returning it to PENDING for its own review.
Rejected candidates remain in staging history forever (section 3).
Nothing in staging is reachable by retrieval (FR4) - enforced by construction,
tested by A0.

### 6.4 The expert gate, day to day (CLI first, D-1)

A gate session: the expert starts the tool under their recorded identity
(single-writer, section 2 assumptions), takes a queue filtered by pack and layer,
and for each candidate sees the proposed entry beside its rendered source span
from the local corpus copy. Decisions are keyboard-driven; each decision is
appended to the audit log atomically before the next candidate is shown - a
crashed session loses at most the in-flight decision and corrupts nothing.
Crash contract as amended (S8, AR-rag4ds-026 r5-r7, 2026-09-13, flagged): the
start-of-session repair applies a decision whose staging transition never
landed only after checking the approver holds every grouped member's home
role; a head found decided with its cluster still open is closed by such an
identity when the head's staging authority names its own journal decision,
and refused by name otherwise - never hidden, never fatal for everyone. The
session ends with a written summary (counts by decision, time spent), which
feeds the section 10 estimates. Throughput planning figure [ASSUMED]: order of
tens of decisions per expert-hour, to be replaced by measured numbers from the
first robotics sessions.
First measurement (S8, 2026-09-12, flagged): 64.7 decisions per designee-hour on
hand-authored shared-layer candidates with an empty conflict display - an upper
bound for the robotics figure, not its estimate.
Certified-text approvals (S9e part 8, PI decision 2026-09-24, flagged;
authoring/robotics_s9e_part8_PI_review_2_signed.txt): a second road beside the
screen, for one case only. A hand-authored candidate whose exact text the pack's
validator certified as written in a confirm list pinned in git may be approved
from an approvals file the PI signed, through tools/gate_certified.py, which
drives the same gate session and refuses before any decision - in its dry run
and its live run alike - unless the tool runs from its own tree at the
recorded HEAD with its code committed; the three files are committed; the
certified item bears exactly one as-written mark and the list is signed by the
approver; the candidate came from an authoring run and its text is
byte-equal to the draft item, itself the certified text with the list's line
wraps joined; its source passage still hashes to the draft's span; and every
row's supersession or new id agrees with the journal (amended at AR-rag4ds-053
r1, flagged). The approval is recorded under the validator; the road and the
three files' hashes go to runs/gate-certified.jsonl, with a required
transcript; no conflict can be recorded this way. Extraction proposals and edits are never approved from a file: for them the
expert at the rendered span stays the rule.

When a candidate touches ground an approved entry already covers, the gate shows
the related existing entries beside it: contradictions and overlaps are resolved
by the expert at review time (reject, edit, or supersede via FR8), never by
automation. If the expert approves despite a shown conflict, the decision writes
a conflict flag into the audit trail linking the two entries; the flag stands
until supersession or retraction (FR8) resolves the pair. Deliberate coexistence
is allowed - unrecorded coexistence is not.

The gate never writes the knowledge base. Approval marks the candidate; the
build step compiles approved candidates into the pack (I2: the compiler reads
approved entries only).

### 6.5 Audit-trail mechanics and the reconstruction test

Every gate decision appends the FR5 record; every pipeline run (ingest,
extraction, build, evaluation) writes its manifest (NFR2); ingest manifests
start the trail at the document (component 1). The concrete test for NFR1, run
in CI: fold the audit log from empty and compare the result with the built
knowledge base - byte-equal source files or the build fails. The fold applies
each decision in order by its type (FR5): approvals - supersessions included -
write or overwrite the entry's tree file; mapping-file approvals write the
embedded file under the pack's mappings/ (4.5); retractions remove the entry's
file; rejections of either kind change nothing. Two preconditions
make byte-equality achievable, and both are binding: the gate decision embeds
the complete approved artifact - an entry with every FR1 field and its id
assigned at gate time, or a canonical mapping artifact (4.5) - nothing invented
by the compiler (FR5); and gated artifacts serialize canonically - exactly one
byte form each, the concrete format being a section 8 decision. Auditability
is not a promise; it is a passing test.

### 6.6 Failure paths (NFR9 applied)

- A document the extractor cannot parse: the run manifest records the failure,
  the catalog marks the document unprocessed - never silently skipped.
- A gate write that fails: the session halts with the error; no silent retry,
  no decision presumed recorded.
- A compile-time defect (missing FR1 field, identifier collision): the build
  fails loudly and names the entry.
- An extraction run that produces zero candidates: reported as zero with its
  manifest, distinguishable from a run that never happened.

## 7. Evaluation and benchmark hooks

This section defines how the knowledge framework's quality is measured (C5), the
firewall that keeps the measurement honest, and where AutonBDA-Bench attaches
later without redesign. It defines metrics and rules only; no number in this
section is a result, and no target threshold is invented here - thresholds
belong to section 10's exit criteria, agreed before evaluation begins.

### 7.1 What quality means for a knowledge framework

Three measurable dimensions, all produced by the evaluation hook (component 9):

1. Retrieval quality: against gold answer sets (7.2), standard ranked-retrieval
   metrics - precision and recall at k, mean reciprocal rank - computed per
   query and reported as full distributions, never as a lone mean. Gold marks
   relevance; the metrics measure agreement with gold; truth confidence is
   measured nowhere (FR6's rule extends to evaluation).
2. Coverage: the fraction of gold queries answered OK with at least one
   gold-relevant entry, per pack and per layer; and, separately, population
   progress - the share of registered corpus documents that have yielded gated
   entries. Progress is a workload measure, never presented as quality.
3. Grounding integrity: returned sources resolve end to end - FR7 walked both
   directions on every evaluation run's returned set, sampling only above a
   stated size recorded in the run manifest - zero unapproved leakage (the A0
   class), and
   the count of open conflict flags (6.4: written when an expert approves
   despite a shown conflict, cleared only by FR8 resolution).

### 7.2 The answer-set firewall (C5)

Gold answer sets are authored by people other than the knowledge authors - C5's
own words - with the separation enforced structurally where possible and
procedurally where not:

- A gold set is authored from the corpus and domain expertise, blind to the
  entry set: its author may read any registered document, never the pack's
  entries. Blindness is a recorded obligation on the author, not a technical
  control; the structural guards are the freeze, the hash, and the recorded
  authorship below. Anchoring gold to what the knowledge base happens to
  contain would inflate every metric the set exists to measure.
- Gold sets live outside the packs (location fixed in section 9), with named
  authors and reviewers recorded per set, and are frozen by hash before their
  first evaluation run. A frozen set is never edited; extension is a new
  version, re-frozen.
- No tuning on gold: gold results never directly drive extraction tuning,
  lexicon edits, or entry edits. Failure analysis may motivate new corpus work
  or new candidates - through the ordinary gate - and the gold set rotates or
  extends rather than being fit (the integrity charter's leakage rules, applied
  to retrieval). Demo flows (pack-authored, section 5) are the development
  smoke tests; gold is held out.
- Small-team honesty: with today's team the author-separation is thin - roles,
  not departments [ASSUMED: one team member plus domain validators]. The
  mitigations are structural: authorship recorded per set, freezing by hash
  before first use, and the charter's no-peeking rules binding everyone. And
  one hard consequence stated plainly: for robotics, the assumed staffing
  provides NO qualifying gold author - the team member proposes candidates and
  the validator gates every entry, so both fail the blindness rule above.
  Gold authoring for a domain cannot start, and 7.3's evaluation runs cannot
  exist for it, until OD-4 (section 12) names an author who qualifies - or the
  PI records a relaxation of the blindness rule with its risk accepted.

### 7.3 Evaluation runs

Every run is reproducible or it does not count: its manifest records code
commit, the pinned pack versions actually served (the binding pins of
component 7), embedding-model version, gold-set version and hash, and
timestamp (NFR2). Results are filed as run artifacts beside their manifests;
a number quoted anywhere without its manifest reference does not exist. Each
pack release gets an evaluation run recorded against the then-current gold
version; a decline against the prior release is flagged loudly in the run
summary - releases are gated by the section 4.3 checklist, and evaluation
informs rather than blocks, until section 10's exit criteria say otherwise.

### 7.4 Where AutonBDA-Bench attaches

The benchmark is the public packaging of exactly this harness: tasks are query
sets with gold and metrics per domain; the runner is component 9; publication
and task authorship belong to the bench scope, not to this module. The
proposal's figure of 50 tasks per domain is the bench's commitment, referenced
here, not owned here. Because authorship, freezing, and manifests are already
recorded, the firewall provenance the benchmark must publish exists by
construction - nothing in this module changes when the benchmark arrives (I5).

### 7.5 What evaluation here does not claim

Retrieval quality is not lifecycle quality. High marks on every metric above
say the knowledge framework retrieves the right approved knowledge; whether
grounded reasoning reaches expert-comparable analysis quality is the central
hypothesis, and it is tested where the reasoner is tested - not here (5.5).

## 8. Technology stack

One decision per component, each with its reason. These choices become binding
when this section's review round closes. No version numbers are invented here:
exact pins are recorded in the lockfile at environment creation, and verifying
each library's license and current stable release at setup is VT-3 (section 12).
Library names below are stable, widely used tools known to the team
[UNVERIFIED at the pin: current versions and license texts - that is VT-3's job].

- Language and runtime: Python 3.11-class. Reason: the extraction and embedding
  ecosystem lives there, it is CPU-friendly, and nothing in this module needs
  anything faster than its bottleneck - the expert gate.
- Entry source format (the I1 truth, and the biggest call in this section):
  one file per entry, canonical JSON - sorted keys, fixed separators, LF, one
  trailing newline, ASCII-escaped - filename = <entry id>.json. Entry-id format,
  defined here once: lowercase [a-z0-9-], 1 to 64 characters, unique
  case-insensitively across the whole knowledge base - safe as a filename on
  case-insensitive filesystems (this repository lives on Windows; CI runs on
  Linux) and in URLs; the gate enforces format and uniqueness at assignment
  (FR5). The tree holds each entry's CURRENT version only - a tree file's
  status is always current; the audit log holds the full history - superseded
  versions as their original approval records, marked by the superseding
  decision that names them, and retracted entries' final forms embedded in
  their retract records (FR8) - and a retracted entry's file leaves the tree
  while its history stays in the log. The reconstruction fold (6.5) therefore rebuilds
  exactly the current-state tree. Reason: the
  reconstruction test (6.5) needs exactly one byte form per entry, and
  canonical JSON is the simplest serialization that is deterministic,
  git-diffable, and boring. The "ontology-driven" claim (C9) is carried by the
  schema, not the file format: entries are typed classes with typed relations
  (FR1) - that IS the ontology. No RDF artifact is built in this plan: the
  CORA mapping is produced and validated by the 4.5 gate path and VT-1,
  which is what discharges FR11, and no step, layout entry, or exit
  criterion carries an RDF view - so none is promised (the original
  compile-time RDF clause was struck per AR-rag4ds-007 F1; an RDF export
  can be added later by flagged plan amendment if a concrete consumer
  appears). This interpretation of C9 is deliberate and visible:
  schema-as-ontology, not an RDF-native store, because the gate, the audit
  trail, and the byte-equality test all get simpler and nothing the claims
  require is lost.
- File formats for everything else, honoring section 4.1's deferral:
  machine-written records - the append-only audit log, staging candidates, run
  manifests, index manifests - are JSON Lines of canonical-JSON objects, one
  record per line; the reconstruction fold (6.5) reads exactly this, so the
  format determinism the test needs is the format's own definition.
  Human-edited files - the pack manifest, corpus catalog, demo flows, and
  external-standard mapping DRAFTS - are YAML: comment-friendly for the humans
  who maintain them, and none of them participates in byte-equality. Approved
  mapping artifacts are the exception that proves the split: the gate
  canonicalizes an approved mapping draft to canonical JSON (4.5), and that
  gated artifact - machine-written like every gated record - does participate
  in byte-equality. Corpus intake includes
  PDF text extraction via pypdf-class tooling; the concrete library is verified
  and pinned at VT-3 like everything else.
- Structural index: SQLite via the standard library. Reason: serverless,
  single file, robust, already everywhere; a disposable derivation (I1) should
  not bring a server with it.
- Vector search: exact brute-force cosine over numpy arrays in v1 - no
  approximate-nearest-neighbor library at all. Reason: the scale envelope is
  hundreds to a few thousand entries per pack (section 2); at that size exact
  search is instant, fully deterministic (which reproducibility loves), and
  dependency-free. Revisit trigger, recorded now: only if the loaded entry
  count approaches fifty thousand does an ANN index become a question.
- Embeddings: sentence-transformers with a small CPU-viable model; the exact
  model id and version are pinned in the index manifest (section 3) and chosen
  at VT-3 time. Reason: standard, local, no service dependency, runs on
  existing hardware (NFR4). The library's license and the model's own license
  are both settled at VT-3, like every license in this section - model
  licenses in particular vary and are not assumed here.
- Extraction: spaCy for the rule and pattern strategies (its language model
  pinned at VT-3b, 2026-09-11, flagged: en_core_web_sm 3.8.0, MIT,
  hash-verified at load; S7's rule strategy keeps the blank sentencizer by
  measurement, and the model serves the POS/dependency and relation-type
  work of the next extraction increment); the embedding model
  above for similarity clustering and near-duplicate grouping (6.2). The
  LLM-assisted strategy stays deferred (6.2); if it ever calls an external
  model, it uses the provider's own SDK directly - litellm is banned in this
  project under a standing rule, with no exceptions.
- Service layer: the retrieval library is the product (component 6); a thin
  FastAPI endpoint is added only when the first non-library consumer exists,
  as its own increment (section 10). Reason: an HTTP server no one calls is
  maintenance without a user.
- Gate CLI: plain Python with rich for rendering - keyboard-driven loop, no
  TUI framework. Reason: D-1 says CLI first; the tool's job is showing a
  candidate beside its source and taking one keystroke, and expert time is
  too expensive to spend on interface novelty.
- Testing and quality: pytest for everything including the acceptance cases
  (A0-A5) and the reconstruction test (6.5); ruff for lint and format;
  coverage measured and reported, never gated (section 10 answers the
  once-deferred threshold question: the acceptance record is named tests,
  not a percentage). CI runs on the private repository's GitHub
  Actions [ASSUMED: runner minutes available]; every CI job is a script that
  runs identically on the local machine, so CI is a runner, never the
  definition.
- Hardware split, stated once: everything above runs on existing CPU-class
  hardware - that is NFR4 and it holds with zero exceptions in this module.
  When the GPU server arrives it may accelerate embedding runs and host
  LLM-assisted extraction experiments; nothing in Module 1 waits for it.

## 9. Repository layout and engineering practice

This section fixes where everything lives and how the code is worked on. It
implements decisions already made in sections 3, 4, 6, 7 and 8; nothing here
changes them.

### 9.1 Layout inside AutonBDA/module1_rag_for_ds/

    docs/               this plan; later design notes, one file per decision
    src/rag4ds/         the core package - domain-blind (NFR3, I3)
      model/            entry model, manifest schemas, core schema version (FR1, 4.1)
      corpus/           catalog and ingest with manifests (component 1)
      extract/          strategy interface and the two v1 strategies (6.2)
      staging/          candidate store and states (6.3)
      gate/             the review CLI and decision writing (6.4, D-1)
      kb/               compile step: approved entries -> canonical tree (6.5)
      index/            structural (SQLite) and vector (numpy) builds (section 8)
      retrieve/         the retrieval library and grounding contract
                        (components 6-7; S12's thin HTTP endpoint lives
                        here too, sharing the library implementation)
      evalhook/         the harness of component 9 (7.1-7.3)
      auditlog/         append-only decision log and run manifests (component 8)
    packs/<code>/       domain packs, exactly the 4.1 anatomy; each pack's
                        releases/ holds its release manifests (4.3)
    shared/general/     the general-concept entry tree; shared/tasks/ the
    shared/tasks/       task-pattern tree - same canonical entry form as
                        packs, gated by the OD-1 designee, never in a pack
    shared/corpus/      the shared corpus catalog (S8, flagged amendment
                        2026-09-12): the authoritative texts the general and
                        task layers restate - the CRISP-DM guide, the team's
                        task-pattern source note - catalogued under the same
                        NFR6 rules as a pack's corpus/; raw copies in data/,
                        except the team's own Apache-2.0 source notes, which
                        are versioned under authoring/sources/
    journal/            THE append-only audit log data: gate decisions,
                        lifecycle events, JSON Lines (section 8); committed
                        (src/rag4ds/auditlog/ is the code that writes it)
    stage/              the staging candidate store, committed - candidate
                        content is our-own-words by the quotation bound, so
                        nothing restricted-license enters git (NFR5, NFR6);
                        rejected history stays forever (6.3)
    eval/gold/          gold answer sets - outside packs by design (7.2)
    runs/               run manifests and evaluation artifacts, committed
                        (results without their manifests do not exist, 7.3)
    tests/              pytest: unit tests, A0-A5, the reconstruction test,
                        the NFR3 lexicon scan as a test
    tools/              committed evidence and maintenance scripts
    data/               raw corpus documents - gitignored (NFR6); catalog
                        lives in packs/<code>/corpus/ (or shared/corpus/)
    pyproject + lockfile  pins recorded at VT-3

One clarification this layout forces, closing observation O8 of
AR-rag4ds-002 honestly: FR1's status enum is current | retracted. Tree
files (packs/ and shared/) are always current; a retracted entry's final
form is embedded, marked retracted, in its retract record (FR5); and a
SUPERSEDED version is marked by reference - the superseding decision
names the id and version it supersedes - never by rewriting the
append-only journal. No record ever carries a superseded status field;
supersession is a property of the chain, which is exactly what FR7's
walkable provenance provides. The fold's tree-path rule, total over gated artifacts: an entry
whose pack field is the reserved value shared serializes under
shared/general/ or shared/tasks/ by its layer; every other entry serializes
under packs/\<pack>/entries/ (FR1); an approved mapping artifact serializes
under packs/\<pack>/mappings/ (4.5).

### 9.2 Engineering practice

- Fail-loud is enforced, not hoped for: ruff configuration bans bare
  excepts; no silent fallback survives review; every invariant the plan
  names (I1-I5) has at least one test that fails when it breaks.
- Tests are the acceptance record: A0-A5 exist as named tests from the
  first increment they apply to. CI runs the full suite on every push,
  each job a script that runs identically locally (section 8); the
  reconstruction test (6.5) additionally gates every commit touching
  kb/, gate/, auditlog/, model/ (where canonical serialization lives),
  packs/ or shared/ (the trees it compares against), or journal/.
- Work lands on main in small commits; nothing is squashed or
  force-pushed (the audit rules already forbid it repo-wide). Each
  increment ends with the scan tools clean and an audit submission
  through the independent channel (the audit protocol's section 6
  increment gate) before the next increment begins.
- Run manifests are committed beside their artifacts in runs/; a number
  quoted anywhere without its manifest reference does not exist (7.3).
- Definition of done, per increment: code + tests green + docs updated +
  scans clean + audit submitted. Not one of the five is optional.
- Publishability discipline (NFR5): the AutonBDA subtree stays free of
  secrets, personal data, and restricted-license content at every commit,
  not just at release - a grep-based check in tests/ enforces the obvious
  patterns, and the corpus rule (NFR6) handles the rest.

### 9.3 Versioning and releases

Five version streams, none conflated: the core package (semver; its
schema version per section 4.1 bumps only on FR1/manifest field changes),
each pack (semver per section 4.3), the grounding contract schema
(versioned from day one, component 7), each entry's own version (FR1;
version+1 on supersession, FR5), and the index version echoed in
retrieval meta (component 7). The lockfile pins dependencies; the index
manifest pins the embedding model (section 8).

Release mechanics, completing the section's brief: a pack release is a
git tag pack-\<code>-vX.Y.Z on the commit whose tree holds the released
gated content, with the release manifest filed in packs/\<code>/releases/
(gated-content hash, section 4.3); core releases tag core-vX.Y.Z.
Immutability (NFR8) rests on the tag plus the append-only journal plus
the repo-wide no-force-push rule - a release, once tagged, can be
superseded by the next one but never rewritten.

## 10. Increments and milestones

The build order, dependency-ordered, with exit criteria per increment. No
calendar dates appear here; calendars are applied later as filters over this
sequence. Every deferred threshold in sections 7-9 lands in an exit criterion
below, and the standing rule for all of them: thresholds are agreed with the
validating experts BEFORE the measurement they judge begins - never set after
seeing results.

### 10.1 The step sequence (S-numbered, S0-S12; I-numbers are the section 3 invariants, and bare M-numbers never appear - section 1 naming rule)

- S0 Environment. Execute VT-3 (verify licenses and releases, pin everything,
  choose the embedding model); scaffold the 9.1 layout; CI runner live.
  Exit: lockfile committed; CI green on the empty suite; both scan tools run
  in CI.
- S1 Entry model and canonical serialization (model/). FR1 fields and types,
  the entry-id rule, the canonical-JSON writer. Exit: a round-trip test
  proves one byte form per artifact, repeatably.
- S2 Journal and gate core (auditlog/, the decision-writing half of gate/).
  All five FR5 decision types with their conditional payloads, id assignment,
  append-only journal. Exit: decision-write tests green for every type and
  both payload variants. (A0's staged-candidate clause becomes testable only
  at S5, where retrieval exists; it belongs to S5's exit.)
- S3 Corpus catalog and staging (corpus/, staging/). NFR6 catalog with ingest
  manifests; candidate states including both GROUPED exits; both candidate
  kinds. Exit: state-machine tests green; an ingested document is
  reconstructable from catalog plus manifest.
- S4 Compile and fold (kb/). Tree build from approved decisions; the
  reconstruction test goes LIVE. Exit: 6.5 byte-equality green on synthetic
  fixtures covering every decision type, both artifact kinds, and a
  retraction.
- S5 Indexes and retrieval (index/, retrieve/). Structural SQLite, exact
  vector search, FR6 semantics including the filter rules and both failure
  statuses; grounding contract v1 published. Exit: A0 complete - all three
  clauses, the staged-candidate clause included; FR6 semantics tests green;
  contract schema versioned and documented.
- S6 Gate CLI (the interactive half of gate/). D-1 tool with conflict display
  and flags, session summaries. Exit: a recorded session over synthetic
  candidates proving FUNCTION only - explicitly not a throughput figure;
  6.4's [ASSUMED] decisions-per-hour is replaced at S9, by real validator
  sessions, exactly as 6.4 says.
- S7 Extraction v1 (extract/). Rule and embedding strategies, duplicate
  grouping. Exit: strategy-interface tests green; a precision sanity check on
  a synthetic corpus, its manifest recorded (no threshold claimed - the
  number is recorded, the target is set with the validator at S9).
- S8 Shared layers seeded (shared/). General-concept and task-pattern entries
  gated by the OD-1 designee - the task patterns S9's demo flows and A1
  depend on, which no pack may carry (4.1). Blocks on OD-1. Exit: the shared
  trees hold gated entries covering CRISP-DM phases and the section 4.6 task
  patterns; the designee's measured hours recorded, replacing the
  shared-layer [ASSUMED] figure. (The NFR3 scan first bites at S9, when a
  real pack lexicon exists - it is listed there.)
  DONE 2026-09-12 (flagged post-certification): shared/general holds 12 and
  shared/tasks 5 gated entries; the designee's session measured 0.53 h for
  34 decisions (64.7 per hour); docs/s8_shared_layers.md is the record;
  submitted as AR-rag4ds-026 (its verdict is on the ledger, not here -
  AR-rag4ds-026 O8: this line once said "audited" before the audit ran).
- S9 Robotics pack, seeded then deep. Real corpus intake; gated population by
  the validator; VT-1 closes BEFORE the first release; the CORA mapping
  artifact; demo flows. Entry criterion (AR-rag4ds-026 r3 O18, r7 carve-out
  7, flagged): the PI registers the robotics validator identity in
  identities.yaml, and decides whether the validator role carries its pack
  (O17/O22), before the first pack session - the shipped registry holds no
  validator, so no pack candidate can be approved or retracted by anyone
  until then. Exit: robotics v1.0.0 tagged with its release
  manifest; A1 and A4 green; the NFR3 scan - now with a real pack lexicon -
  clean over core and both shared layers; the validator's measured hours and
  decisions-per-hour recorded, replacing 6.4's figure and the robotics line
  of the effort paragraph below; the extraction-precision target set with
  the validator (S7's deferral) and the measured value recorded against it.
- S10 Evaluation hook and gold (evalhook/, eval/gold/). The 7.1 metrics, the
  firewall (OD-4 must close first), run manifests. Exit: metric AND coverage
  targets (7.1) agreed with the validators before the first run and recorded
  in its manifest; the first evaluation run filed - including the
  RETROACTIVE run for robotics v1.0.0, recorded as retroactive, which
  reconciles 7.3's per-release rule with S9 preceding this hook; from S10
  onward a release requires its evaluation run filed and measured against
  the agreed targets (7.3's "until section 10 says otherwise" ends here).
- S11 Breadth packs, seeded - bioinformatics FIRST, then grocery (the order
  5.4 defers here: bioinformatics's stricter data regime makes it the harder
  proof, worth the earlier start). VT-2 closes before corpus registration;
  OD-2 and OD-3 close before any gating; A2, A3, and A5 - the
  zero-core-commits proof, with the git evidence in the A5 test's record.
  Exit: both packs released at seeded depth; A2, A3, A5 green; validator
  hours measured per pack, replacing the bioinformatics and grocery
  [ASSUMED] figures.
- S12 HTTP endpoint. Built only when the first non-library consumer exists
  (section 8); explicitly NOT part of module completion.

Module completion, defined: every exit of S0-S11 met. S12 is excluded by
design - it waits for a consumer. Milestones, satisfying this section's
title: MILESTONE A, core proven = S0-S5 (fold, retrieval and contracts green
on synthetic data); MILESTONE B, first domain live = S9 (robotics v1.0.0 and
A1); MILESTONE C, claims discharged = S11 (A0-A5 all green; C6 and C10
demonstrated).

Effort planning figures, each [ASSUMED] until its measuring step replaces
it - shared layers at S8, robotics at S9, bioinformatics and grocery both
at S11:
robotics (deep) - order of tens of validator hours; bioinformatics (seeded) -
order of a few validator hours; grocery (seeded) - order of a few validator
hours; shared layers (S8) - order of a few designee hours. The validator's
hours are the module's critical path (4.7), and S9 exists partly to turn
these assumptions into measured facts.
MEASURED at S8 (2026-09-12, flagged): the shared layers took 0.53 designee hours -
one session, 34 decisions (17 approvals; 17 rejections of a misfiled batch),
64.7 decisions per hour on hand-authored, conflict-free candidates
(runs/gate-sessions.jsonl; docs/s8_shared_layers.md) - inside the assumed 'a few',
and not a proxy for validator sessions over extracted robotics candidates.

Section 8's code-coverage question, answered here where it was deferred:
pytest coverage is measured and reported at every step's exit and never
gated - the acceptance record is the named tests (A0-A5, the
reconstruction test, the invariant tests), not a percentage.

### 10.2 Gate and decision dependencies, restated in one place

Nothing blocks S0-S7 except the work itself. OD-1 blocks S8 (the first
shared-layer gating); VT-1 blocks S9's release; OD-4 blocks S10's gold
and, through S10's per-release evaluation rule and 7.2's per-domain gold
blockade, every release from S10 onward - S11's included; VT-2, OD-2 and
OD-3 also block S11. VT-3 is S0.

## 11. Risks and mitigations

Only risks that are real for this project, each with its mitigation and the
lever that gets pulled if it fires. Likelihood is rated on a three-word
scale - HAS-OCCURRED (on this project's record already), EXPECTED (fires
unless acted on), POSSIBLE (could fire) - and nothing finer: inventing
probabilities would be a number without a run.

- R1 Validator hours (the critical path, 4.7). EXPECTED. The whole schedule
  hangs on expert time at the gate. Mitigation: seeded-before-deep ordering,
  batched gate sessions, S9 measuring the true rate early. Lever if it
  fires: depth is the variable, never breadth - a shallower robotics pack
  still proves the pipeline; dropping a domain would break C10 and is not a
  lever. Outright loss of a validator is the extreme case: the same lever,
  plus PI re-designation of the role - the pack manifest names the role and
  its current designee (4.1) precisely so the person is replaceable.
- R2 Open decisions and verification tasks stalling steps (OD-1..OD-4 and
  VT-1..VT-2 block S8-S11 per 10.2; VT-3 gates S0 and therefore
  everything). EXPECTED. Mitigation: the blockers are named, mapped to
  steps, and visible; once S0 discharges its own VT-3, S0-S7 proceed
  regardless of the rest. Lever: escalation to the PI is the only unblock
  for OD items - the plan cannot decide people.
- R3 Corpus, dataset, and toolchain licensing and quality. POSSIBLE. The
  robotics corpus may arrive in poor machine-readable shape; VT-2 may
  reject candidate datasets; a library or embedding model may carry a
  license section 8 refuses to assume acceptable (its verification is
  VT-3, at S0). Mitigation:
  catalog-first intake with licensing checked at registration (NFR6);
  candidate dataset families named in 5.3-5.4 with alternatives re-verified
  under the same VT-2. Lever: corpus breadth is adjustable; the gate and
  provenance machinery are corpus-agnostic.
- R4 Extraction precision too low, ballooning gate hours. POSSIBLE.
  Mitigation: the
  precision-over-recall bias (6.2), the S7 sanity check, the S9 target set
  with the validator, and pluggable strategies behind one interface. Lever:
  more hand-authoring through the same gate - slower but identical in
  provenance.
- R5 Retrieval quality on small knowledge bases - a small embedding model
  may rank domain terms poorly. POSSIBLE. Mitigation: exact search plus structural
  filters carry the v1 scale; the model is one pinned swap (section 8);
  quality is measured at S10 against targets agreed beforehand. Lever:
  model swap, then filter-weighting - never threshold-lowering after the
  fact (integrity charter, section 5).
- R6 Single-developer process decay. HAS-OCCURRED: the audit record of
  2026-08-20/21 documents evidence filed from stale trees, silent edit
  no-ops behind false claims, and premature timestamps - each caught by
  audit, each with a standing remedy now in the protocol or the team's
  memory. Mitigation: the increment gate (at most one unaudited
  increment), per-increment independent audits within the protocol's
  disclosed limit (independence of process and record, not of substrate,
  Z_Audit_Protocol.md section 1), external checkpoints through the PI, and
  everything on an append-only record. Lever: when
  audits find process decay recurring, the cadence tightens - smaller
  increments, not looser checks.
- R7 Downstream pressure from the reasoner pulling features early or
  bending the boundary. POSSIBLE. Mitigation: the grounding contract is
  versioned (I5); the module boundary is written (section 1); changes
  arrive as plan amendments through the audit channel or not at all.
  Lever if it fires: the contract version is the pressure valve - the
  reasoner pins the version it was built against while the proposed change
  goes through amendment and audit; no unversioned accommodation ships.
- R8 Gold authorship (7.2): the blindness rule needs a person today's
  staffing does not supply. HAS-OCCURRED - the gap is present fact, OD-4
  is it on the record. Mitigation: OD-4 names the gap loudly and
  blocks what depends on it; the PI-recorded relaxation path exists with
  its risk named. Lever: the PI's call, on the record.
- R9 Repository environment: the working tree lives under OneDrive, a
  known git-interference source - stated here as the generic hazard it is,
  no project incident being on the record - and the private repository
  holds confidential KAKENHI material. POSSIBLE. Mitigation: push after every working session (the
  remote is the backup); the NFR5 publishability discipline plus the
  AutonBDA-subtree checks keep the public/private boundary clean. Lever:
  relocating the working tree off OneDrive is available whenever the PI
  wants it; until then, push-always stands.

## 12. Open decisions for the PI

Decisions already made (recorded so nothing rests on memory):

- D-1: the expert gate ships as a command-line tool first; a web UI is a later
  increment. Decided by the PI during Module 1 planning, 2026-08.

Open:

- OD-1: who is the designated expert gating the general and task layers
  (section 3, gate ownership). Domain packs have named validators; these two
  shared layers need one.
  DECIDED 2026-09-12 by the PI (audit/rag4ds/responses/PI_decision_OD-1_2026-09-12.md),
  flagged post-certification: "OD-1: Siriweera (designee); alternate Paik; entries authored by the designee are gated by the alternate".
  T. A. Siriweera gates the general and task layers; I. Paik is the alternate;
  an entry authored by the designee is gated by the alternate. S8 opens.
- OD-2: who validates the bioinformatics pack (section 5.3). Seeding and
  deepening both wait on this - every domain entry passes the domain gate
  (I2) - so acceptance case A2 depends on it; staging preparation may proceed.
- OD-3: who validates the grocery pack (section 5.4). Same rule; acceptance
  case A3 depends on it.
- OD-4: who authors and who reviews the gold answer sets per domain
  (section 7.2). The firewall requires them to be other than the knowledge
  authors, blind to the entry set - which today's assumed staffing cannot
  supply for robotics (both available people fail the blindness rule), so
  gold authoring and evaluation runs for a domain are BLOCKED until OD-4
  closes for it, or the PI records a relaxation with its risk accepted.

Verification tasks (must close before the dependent work ships):

- VT-1: verify the concrete CORA class list against the actual IEEE 1872 text
  before the robotics pack's first release (section 4.5; C7 note in section 1).
  DISCHARGED 2026-09-20 at S9d, flagged: the list is verified against IEEE Std
  1872-2015 itself (the PI's copy, sha256-pinned; nothing of its text in git)
  and recorded as data in packs/robotics/standards.yaml, with IEEE Std
  1872.2-2021 (AuR) registered as the extension reference; the verification
  and the corrections it made to the team's earlier assumed list (part and
  autonomy roles are relations, not classes) are docs/s9_robotics_pack.md
  section 15; the relation list (declared by subrelation or by a
  relation-typing axiom) was completed to all 42 after AR-rag4ds-042 r1 F1,
  flagged. The first mapping draft (one row per gated entry of the pack)
  is filed for the validator by the 4.5 gate path.
- VT-2: verify the concrete public datasets for the bioinformatics and grocery
  corpora - existence, license terms, and absence of personal data (NFR6, NFR7) -
  before those corpora are registered (sections 5.3, 5.4).
- VT-3: at environment creation, verify each section 8 library's license and
  current stable release, choose and pin the embedding model, and record the
  exact pins in the lockfile - before the first line of code depends on them.
  DISCHARGED 2026-08-21 at S0: docs/vt3_pins.md records the verification
  (every license permissive, litellm absent), the lockfile
  (requirements.lock, 68 pins, Python 3.11.9), the model choice
  (sentence-transformers/all-MiniLM-L6-v2, apache-2.0, revision
  1110a243fdf4706b3f48f1d95db1a4f5529b4d41), and the recorded
  environment incident (WinError 206 under the deep OneDrive path;
  venv relocated to a short path outside the tree).
  RE-OPENED 2026-09-11 per the PI decision of 2026-08-23 (a spaCy language
  model to be pinned) and DISCHARGED again the same day, flagged post-
  certification (VT-3b in docs/vt3_pins.md): en_core_web_sm 3.8.0, MIT,
  wheel sha256-pinned in pyproject and both locks and verified at load by
  extract/language_model.py (fail-loud, no blank-pipeline fallback); NOT
  used for S7's sentence splitting, by measurement - identical boundaries
  on the S7 corpus, worse on a table sample - so S7's behaviour is
  unchanged and the model serves the next extraction increment.
