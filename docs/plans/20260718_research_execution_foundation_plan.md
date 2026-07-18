<!-- /autoplan restore point: /Users/davidgraf/.gstack/projects/raisesquad-langres/main-autoplan-restore-20260718-171721.md -->
# Research Execution Foundation

> **Status:** APPROVED for implementation on 2026-07-18.
> **Date:** 2026-07-18
> **Target:** integration branch based on `main@3dbd13d`
> **Primary tracking context:** epic #193 (W1-W3 merged through PR #217; W4-W5
> and research-platform integration remain)

## Goal

Make langres ready to implement, compare, reproduce, and share entity-resolution
architectures without writing a bespoke execution path for each combination.

A researcher should be able to choose:

- an `Embedder`;
- an optional `Reranker`;
- an optional `LLM`;
- datasets, splits, and seeds;
- a tracker such as Trackio;

and run one of four named recipes:

- `Retrieve`;
- `RetrieveRerank`;
- `RetrieveLLM`;
- `RetrieveRerankLLM`.

The run must produce comparable quality, funnel, latency, resource, token, and
cost measurements, and the complete architecture must be saveable locally and
shareable through Hugging Face without executing arbitrary remote code.

## Why this is the next milestone

The refactor merged in PR #217 established the `Pairs` carrier, `Op` algebra,
explicit linear topologies, and v2 persistence. The remaining problem is that the
research loop is still fragmented:

- the standardized benchmark runner reaches through `ERModel` into legacy
  `.module` and `.clusterer` slots;
- topology models have gaps in schema/binding, logging, index lifecycle, and
  selection semantics;
- architecture execution, benchmark evaluation, run persistence, and Trackio
  tracking are separate user workflows;
- result identity has one `backbone`, while realistic recipes have multiple
  model/resource slots;
- cost and aggregate latency exist, but tokens, model/vector facts, stage
  timings, hardware, and unknown-value semantics are not one stable contract;
- a complete architecture can be saved locally but not published to or loaded
  from Hugging Face using familiar `save_pretrained` / `from_pretrained`
  semantics.

Adding many embedding, reranker, and LLM implementations before closing these
gaps would create incomparable experiments and duplicate integration code.

## Product premises

These premises must be confirmed during the CEO phase of `/autoplan`:

1. **The experiment seam comes before a broad model matrix.** We will implement
   only enough concrete resources to prove all four recipes end to end.
2. **Resources are named for what they are, not where they are used.** The same
   `Reranker` resource may score pairs before `Select(top_k=...)` or before
   `Select(threshold=...)`; there will be no `RerankerBlocker` and
   `RerankerMatcher` duplication.
3. **Topology owns role.** Operations such as `Retrieve`, `Rerank`, `Generate`,
   `Parse`, `Select`, and `Cluster` describe where a resource is used.
4. **`LLM` is the resource name.** This follows LiteLLM's terminology. `Generate`
   and `Parse` make the use of the model explicit; `Matcher` remains a
   compatibility concept while the migration is additive.
5. **Measurements are facts first, prices second.** Raw token/resource facts are
   durable. USD cost is a derived value with provider/model/pricing-snapshot
   provenance and may be repriced later.
6. **Missing measurements are unknown, not zero.** Unsupported counters use
   `None`; zero means the provider or runtime measured zero.
7. **Trackio is optional at runtime but required for official reproducibility.**
   Local runs still work without Trackio. An official benchmark record must have
   a durable run record and declared tracking/provenance state.
8. **Clustering remains downstream but is not the first research focus.** Every
   deduplication recipe can terminate in `Cluster`; this milestone does not add
   new clustering algorithms or transitivity research.
9. **The leaderboard data contract comes before the leaderboard UI.** This
   milestone makes comparable rows possible. A polished MTEB-like docs
   leaderboard is a follow-up.
10. **Hugging Face sharing is configuration/artifact based.** Loading a langres
    artifact must not require `trust_remote_code=True`.

## Vocabulary contract

### Resources

Resources are reusable capabilities and configuration/weight owners:

| Resource | Responsibility | Existing leverage |
|---|---|---|
| `Embedder` | text/record features to vectors | embedding protocols and `SentenceTransformerEmbedder` |
| `Reranker` | candidate pairs to relevance/match scores | `Score`, matcher adapters, research CrossEncoder examples |
| `LLM` | prompts/messages to generated responses plus usage | LiteLLM and local Transformers backends |
| `VectorIndex` | vector storage and nearest-neighbour search | FAISS/Qdrant index contracts |
| `Clusterer` | selected match edges to entity groups | existing clusterer contracts |

### Operations

Operations define topology and data movement:

```text
Retrieve → Rerank → Select → Generate → Parse → Select → Cluster
```

Not every recipe uses every operation.

- `Retrieve` produces candidate pairs and retrieval scores/facts.
- `Rerank` replaces or adds the current decision score using a `Reranker`.
- `Generate` invokes an `LLM` and captures response plus usage.
- `Parse` converts a generated response into a typed decision/score/abstention.
- `Select` filters by `top_k`, threshold, or typed decision.
- `Cluster` consumes selected match edges.
- `Evaluate` is an experiment-runner concern, not an inference `Op`, because its
  output is an evaluation record rather than `Pairs`.

### Compatibility

- Existing `Matcher`, `Blocker`, `LLMMatcher`, and four-slot `ERModel`
  construction remain supported through adapters during the 0.x migration.
- New research code must depend on resources, operations, and architecture
  factories, not on `.module`, `.blocker`, or `.clusterer` internals.
- Deprecations require warnings, migration docs, and at least one release of
  overlap. This milestone does not delete the legacy slots.

## Target developer experience

```python
from langres.architectures import RetrieveRerankLLM
from langres.experiments import Experiment

architecture = RetrieveRerankLLM(
    embedder="sentence-transformers/all-MiniLM-L6-v2",
    reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
    llm="openai/gpt-5-mini",
)

report = Experiment(
    architectures=[architecture],
    benchmarks=["amazon_google", "abt_buy"],
    splits=["validation", "test"],
    seeds=[0, 1, 2],
    tracker="trackio",
).run()
```

The exact constructor spelling may change during DX review, but the user-visible
properties may not:

- model/resource references are passed once;
- multiple benchmarks, splits, and seeds are explicit;
- threshold tuning never evaluates on the test split;
- expensive resource inference is reused when only selection thresholds change;
- the returned report contains every run and an aggregate view;
- all configuration needed to reproduce the run is recorded.

## Architecture

```text
ModelRef / runtime configuration
        │
        ▼
resources: Embedder · Reranker · LLM · VectorIndex · Clusterer
        │                         │
        └──────────────┬──────────┘
                       ▼
operations: Retrieve → Rerank → Select → Generate → Parse → Select → Cluster
                       │
                       ▼
named recipe / ERModel explicit topology
                       │
                       ▼
Experiment runner
  ├─ benchmark registry + dataset revision + split
  ├─ score-once / replay-selection evaluation
  ├─ StageMeasurement + resource facts + quality metrics
  ├─ RunStore / capture_run
  └─ ExperimentTracker → Trackio (optional local, required official)
                       │
                       ▼
ExperimentReport / benchmark rows
  ├─ local JSON/JSONL artifacts
  ├─ Trackio / Hugging Face Dataset synchronization
  └─ model card / future docs leaderboard

ERModel.save_pretrained()
  └─ manifest + topology + model refs + sidecars + measurement/protocol metadata
      └─ push_to_hub() / from_pretrained(repo_id, revision=...)
```

## Workstream 0 — topology hardening

The explicit topology must become a reliable public execution contract before
the experiment runner depends on it.

### Changes

- Make explicit-chain `.schema` and `.is_bound` derive from the source
  operation.
- Thread judgement logging through explicit-chain `dedupe` and `compare`.
- Make `compare` honor applicable `Select` semantics instead of skipping all
  selection stages and applying only one final threshold out of band.
- Define and test the vector-index binding/build lifecycle used by explicit
  retrieval topologies.
- Make benchmark and fit-facing metadata available through public `ERModel`
  methods rather than legacy slots.
- Publicly export the supported architecture-authoring contracts.
- Move operator persistence from a hard-coded exact-class whitelist to the
  existing safe registry pattern. Custom loading remains explicit and never
  executes downloaded code.
- Reconcile `AGENTS.md`, component-design rules, technical overview, and
  architecture reference docs with the explicit topology.

### Exit criteria

- A supported explicit topology reports schema/binding correctly, logs all
  scoring calls, compares and deduplicates consistently, and round-trips.
- The benchmark package has no direct reads of `.module`, `.blocker`, or
  `.clusterer`.
- A registered custom `Score` or `Select` operation round-trips without adding
  another serialization `if` branch.

## Workstream 1 — stable measurement contract

### Models

Add a generic, versioned measurement schema owned outside `core` if it is a
consumer of inference:

- `StageMeasurement`
  - stage name and operation kind;
  - wall and CPU duration;
  - item/pair counts in and out;
  - throughput;
  - optional p50/p95 per-item latency;
  - cold/warm marker;
  - resource slot/model identity;
  - usage and runtime facts;
  - structured warnings for unavailable facts.
- `TokenUsage`
  - `input_tokens`;
  - `output_tokens`;
  - `cache_read_input_tokens`;
  - `cache_creation_input_tokens`;
  - `reasoning_output_tokens`;
  - provider-specific extensions under a namespaced dictionary.
- `EmbeddingFacts`
  - dimensions;
  - dtype;
  - quantization;
  - vectors produced;
  - bytes per vector and total vector bytes;
  - model parameter count, artifact bytes, and loaded-memory bytes when known.
- `RuntimeFacts`
  - host/OS/Python/langres version;
  - CPU, RAM;
  - accelerator model/count;
  - runtime/library versions;
  - device, dtype, quantization, batch size, worker count.
- `PriceSnapshot`
  - provider/model;
  - currency;
  - effective/captured timestamp;
  - per-token/request/input-cache rates used;
  - source or user-supplied marker.

Existing `LLMUsage` remains readable and gains a lossless migration into
`TokenUsage`. Missing provider fields must remain `None`, not be coerced to zero.

### Funnel facts

Every run records:

- possible pairs;
- retrieved pairs;
- pairs after each `Select`;
- pairs sent to a reranker;
- pairs sent to an LLM;
- parsed abstentions;
- selected match edges;
- clusters produced.

### Exit criteria

- The four recipes emit the same schema even when some measurements are
  unsupported.
- Token facts can be repriced without rerunning inference.
- A run identifies dimensions/dtype/quantization and vector bytes when the
  embedder exposes them.
- Hardware/runtime configuration is attached to performance measurements.
- Tests distinguish `None` from `0`.

## Workstream 2 — architecture experiment runner

Add one public experiment seam that consumes an architecture or factory rather
than method-registry names or legacy slots.

### Responsibilities

- Accept one or more architectures, benchmark ids, split ids, and seeds.
- Accept an optional spend budget. Ordinary local or paid experiments may omit
  it; when supplied, it is enforced as a cap. The guarded official paid proof
  always supplies the plan's USD 20 stopping threshold.
- Resolve benchmark datasets through the existing registry.
- Record dataset name, fingerprint/revision, protocol version, split, and seed.
- Tune thresholds only on the configured training/validation data.
- Compute expensive retrieval/reranking/LLM outputs once per relevant split and
  replay cheap `Select`/cluster/evaluation steps across threshold candidates.
- Evaluate pairwise and clustering metrics with metric definitions attached.
- Open `capture_run` automatically and persist the result in `RunStore`.
- Resolve `tracker=` once and forward flattened numeric measurements and
  artifacts to Trackio/other trackers.
- Return typed per-run records plus aggregates across seeds.
- Preserve failures and partial measurements instead of silently dropping a
  matrix cell.

### Ranking

This milestone does not declare one universal winner. It provides:

- sorting by a chosen headline metric;
- Pareto-front data for quality, latency, cost, and size;
- constraints such as maximum p95 latency, USD budget, model size, or memory;
- enough metadata to compare only compatible infrastructure/protocol cohorts.

### Exit criteria

- One call runs at least two benchmarks, two splits, and multiple seeds.
- The runner accepts all four named recipes without special cases.
- Threshold sweeps do not repeat expensive resource inference.
- Trackio receives params, metrics, and artifacts automatically when configured.
- Local-only execution works without Trackio or Hugging Face credentials.
- Interrupted/failed cells are visible and resumable by recipe/run identity.

## Workstream 3 — concrete resources and named recipes

Implement only the concrete components required to prove the matrix.

### Resources

- Adapt existing sentence-transformer embeddings to accept and preserve a full
  `ModelRef`, including revision and runtime configuration.
- Add a production `Reranker` resource backed initially by a Hugging Face
  sequence-classification/cross-encoder model.
- Extract an `LLM` resource seam from current LiteLLM/local Transformer
  invocation so generation and parsing are independently testable.
- Preserve the existing `LLMMatcher` as an adapter over `LLM + Generate + Parse`.

### Recipes

- `Retrieve`: retrieve → select by calibrated retrieval score → optional cluster.
- `RetrieveRerank`: retrieve → rerank → select → optional cluster.
- `RetrieveLLM`: retrieve → select candidates → generate → parse → select
  decisions → optional cluster.
- `RetrieveRerankLLM`: retrieve → rerank → select candidates → generate → parse
  → select decisions → optional cluster.

Each recipe exposes all resource slots as a mapping such as
`resources: dict[str, ModelRef]`; the singular `backbone` property remains
compatibility sugar only when exactly one model slot exists.

### Exit criteria

- Reusing the same `Reranker` instance before `Select(top_k=...)` and before
  `Select(threshold=...)` requires no wrapper subclass or duplicated inference
  implementation.
- API and local LLM refs flow through the same recipe contract.
- Each recipe runs on a zero-network fake resource test.
- At least one opt-in integration smoke test exercises real Hugging Face
  embedding/reranker loading.

## Workstream 4 — Hugging Face artifact lifecycle

Build on the existing safe local `ERModel.save/load` manifest.

### API

- `save_pretrained(path, *, measurement_summary=None, model_card=...)`
- `from_pretrained(repo_or_path, *, revision=None, token=None, ...)`
- `push_to_hub(repo_id, *, revision/commit metadata, private=..., ...)`

Final names may live on `ERModel` or a dedicated artifact adapter based on the
engineering review, but the familiar user journey must remain.

### Artifact contents

- artifact/protocol schema version;
- topology and registered operation specs;
- every resource `ModelRef` including revisions;
- selection thresholds and calibration provenance;
- optional local trained sidecars;
- required Python package/extras and langres compatibility range;
- benchmark protocol and measurement summary;
- generated model card with intended use, datasets, quality, cost/token,
  performance/hardware, size, and limitations.

### Safety and portability

- Use the Hugging Face Hub client as an optional dependency.
- Pin downloaded Hub revisions in the loaded artifact/run record.
- Never default to arbitrary remote code execution.
- Reject unknown component types with an actionable error listing the missing
  package/registration step.
- Make clear which resources are references and which weights are bundled.
- Never upload benchmark datasets or user records implicitly.

### Exit criteria

- A complete built-in recipe round-trips local → Hub test double → local with
  identical topology, model refs, and thresholds.
- `from_pretrained(..., revision=...)` forwards and records the revision.
- Unknown remote component types fail before inference with problem, cause, and
  remediation.
- No network or heavy Hub dependency is imported by bare `import langres`.

## Workstream 5 — documentation and migration

- Update architecture and component-design documentation to the agreed
  resource/operation/recipe vocabulary.
- Add a migration table from Blocker/Matcher/Judge-era concepts.
- Add copy-paste examples for:
  - embedding separability;
  - each of the four recipes;
  - multiple benchmarks/splits/seeds;
  - Trackio reproduction;
  - `save_pretrained` / `from_pretrained` / `push_to_hub`;
  - repricing a stored token-usage record.
- Document compatible-comparison cohorts and the effect of infrastructure on
  latency, throughput, memory, and cost.
- Add a docs-generated static leaderboard example only if it consumes the real
  result schema with negligible extra scope. A hosted/live leaderboard is
  deferred.

## Parallel execution plan

All work lands through an integration branch. Sub-branches are created from the
latest green integration commit and merged back through PRs targeting the
integration branch.

### Wave A — foundation

| Sub-PR | Owner | Files/responsibility | Dependencies |
|---|---|---|---|
| A1 topology hardening | agent 1 | `core` topology execution, metadata, persistence, contract exports, focused tests | none |
| A2 measurements | agent 2 | measurement models, usage migration, runtime probes, focused tests | none |
| A3 experiment API design/tests | agent 3 | experiment result/protocol models and test fixtures that do not yet execute topologies | none |

Merge order: A1, A2, A3 after each PR is green and reviewed.

### Wave B — integrated execution

| Sub-PR | Owner | Files/responsibility | Dependencies |
|---|---|---|---|
| B1 experiment runner | agent 1 | benchmark adapter, score-once replay, RunStore/Trackio wiring | A1+A2+A3 |
| B2 resources and recipes | agent 2 | embedder ref, reranker, LLM seam, four architectures | A1+A2 |
| B3 Hugging Face lifecycle | agent 3 | Hub adapter, pretrained APIs, model-card artifact | A1+A2+A3 |

Merge order is chosen by dependency after CI; integration fixes happen on a
dedicated branch/PR, not as unreviewed direct changes.

### Wave C — convergence

- docs/migration/example sub-PR based on all merged code;
- cross-cutting integration-test sub-PR;
- full repository verification;
- GStack `/review`;
- address all actionable GStack and ChatGPT/GitHub review findings;
- open the final PR from the integration branch to `main`.

## PR and review gates

For every sub-PR:

1. branch is based on the latest green integration commit;
2. scope and file ownership are declared in the PR;
3. focused tests, Ruff, and strict MyPy for touched surfaces pass locally;
4. the PR CI is green and not merely cancelled/skipped;
5. all actionable ChatGPT review comments are addressed or explicitly rebutted
   with verified evidence;
6. unresolved conversations are checked before merge;
7. merge uses the repository's normal non-force workflow;
8. the integration branch CI is checked after every merge before the next
   dependent merge.

For the final integration PR:

- full non-slow suite;
- opt-in semantic/Hub smoke tests where credentials/network are available;
- core contract coverage gate;
- import budget;
- artifact round-trip;
- no direct benchmark dependency on legacy slots;
- practical end-to-end matrix run using fakes/local models;
- GStack `/review` clean or all findings resolved;
- GitHub/ChatGPT review comments resolved;
- integration branch up to date with `main`;
- final CI green.

## Test strategy

```text
Resource contract tests
  └─ Embedder / Reranker / LLM fakes
      └─ operation tests
          └─ recipe topology tests
              └─ experiment runner matrix tests
                  ├─ score-once assertions
                  ├─ split leakage assertions
                  ├─ measurement completeness / None-vs-zero
                  ├─ RunStore + Trackio spy
                  └─ failure/resume cells

Persistence registry tests
  └─ local save/load
      └─ save_pretrained/from_pretrained with Hub fake
          └─ revision/safety/unknown-component failures

Compatibility tests
  └─ existing Matcher/Blocker/LLMMatcher APIs
      └─ adapters produce equivalent Pairs/results
```

The main CI remains zero-cost and network-free. Real Hugging Face model tests are
marked slow/integration and use small pinned models. Paid LLM verification is
never required for a PR.

## Failure and rescue requirements

| Failure | Required behavior |
|---|---|
| optional dependency missing | error names the required extra and failing resource |
| Hub offline or revision missing | no partial artifact is treated as valid; local cache guidance is shown |
| provider omits a usage field | field is `None`, not `0`; warning/provenance retained |
| one experiment cell fails | remaining matrix continues when safe; failed cell and error are persisted |
| budget exceeded | partial measurements and judgements survive; no further paid calls |
| process interrupted | running attempt remains visible and resumable |
| incompatible benchmark protocols | comparison is rejected or separated into cohorts |
| custom remote operation unknown | fail before inference; never import downloaded code implicitly |
| index/resource cannot bind | architecture reports unbound with exact missing input |
| tracker unavailable | local run persists; official reproducibility status is incomplete |

## Explicitly not in scope

- implementing the paper backlog (Ditto, Jellyfish, AnyMatch, GLinker variants);
- a new generalized W5 search algorithm over arbitrary topology mutations;
- new clustering or transitivity algorithms;
- deleting legacy four-slot APIs;
- a hosted leaderboard service or polished interactive leaderboard UI;
- implicit upload of user datasets, records, prompts, or judgement logs;
- distributed benchmark execution;
- production batching/serving optimization for every local LLM backend;
- support for arbitrary downloaded Python code.

The experiment result and resource identity contracts must leave room for W4/W5,
paper implementations, clustering research, and a living leaderboard without
implementing them here.

## Completion definition

This milestone is complete when a clean checkout can:

1. instantiate each of the four recipes by changing resource/model references;
2. run them through one experiment API on multiple benchmarks, splits, and seeds;
3. reuse expensive outputs for threshold evaluation;
4. persist and optionally Trackio-publish quality, funnel, latency, hardware,
   model/vector, token, and price-snapshot facts;
5. compare compatible runs by quality and constraints/Pareto dimensions;
6. save and load a complete recipe locally and through a pinned Hugging Face
   revision without remote code execution;
7. pass all compatibility, import-budget, coverage, type, lint, and integration
   gates;
8. present green CI and resolved reviews on the final integration PR.

## GStack CEO review

### Review posture and premise confirmation

Mode: **SELECTIVE EXPANSION**. The existing scope remains the baseline. Additions
are limited to gaps inside the declared experiment, measurement, artifact, and
review blast radius.

The user confirmed all twelve premises on 2026-07-18, with one clarification:
legacy APIs are retained during the additive migration and are not deleted.

The independent Codex CEO voice could not run because the execution environment
blocked exporting this private plan to an external service. The review therefore
ran in GStack's documented **subagent-only** degradation mode.

### Premise challenge

| Premise | Assessment | Decision |
|---|---|---|
| The experiment seam should precede the paper backlog | Valid and supported by the current benchmark/architecture split | Keep |
| Four named recipes are enough to prove the initial matrix | Valid as a product front door, but the runner must also accept one custom topology so it is not recipe-special-cased | Keep and add custom-topology acceptance |
| Trackio is required for an official result | Valid as a publication requirement, not as the source of truth | Require a locally verifiable protocol artifact first, then Trackio publication |
| Rich measurement should ship in the first milestone | Too broad if every field is mandatory | Split into mandatory Tier 0 facts and optional capability extensions |
| Score-once replay is always safe | False without stronger cache identity and stochastic-repeat rules | Add a cache identity and statistical protocol |
| A pinned Hub manifest is reproducible | Incomplete for API resources and mutable external services | Add explicit artifact claim levels |
| Clustering research can wait | Valid for this milestone, provided both pairwise and cluster metrics remain first-class | Keep |
| A public leaderboard UI can wait | Valid; its data contract and a generated static example still need acceptance coverage | Keep |

### What already exists

| Sub-problem | Existing code | Plan action |
|---|---|---|
| Typed pair carrier | `core/pairs.py` | Reuse |
| Explicit linear topology | `core/op.py`, `ERModel.from_topology()` | Harden, do not replace |
| Legacy component adapters | `core/op_adapters.py` | Reuse for migration |
| Safe local artifact manifest | `_model_persist.py`, `_artifacts.py` | Extend to Hub transport |
| Dataset registry and leakage-free splits | `data/benchmark.py`, `data/registry.py` | Reuse |
| Pair and cluster metrics | `benchmarks/runner.py`, `metrics/` | Reuse behind an architecture contract |
| Paid-judge-once evaluation | `benchmarks/judge_eval.py` | Generalize score/replay separation |
| Run identity and durable attempts | `tracking/runs.py` | Reuse; add distinct cache identity |
| Trackio integration | `tracking/trackers/trackio_tracker.py` | Reuse unchanged unless a concrete gap appears |
| Model references | `core/model_ref.py` | Reuse for every resource slot |
| Embedding separability | `data/data_profile/separability.py` | Expose in the first research example |
| Hugging Face model loading | embedding and local-LLM implementations | Preserve revision/runtime identity |

### Dream-state delta

```text
CURRENT
architecture execution · benchmark execution · tracking · artifacts
are useful but separate; the benchmark reaches into legacy slots
        │
        ▼
THIS MILESTONE
one protocol runs explicit architectures, reuses expensive outputs,
records comparable facts, publishes optionally, and round-trips artifacts
        │
        ▼
12-MONTH IDEAL
paper implementations and external adapters register as resources/topologies;
continuous benchmark cohorts publish quality/cost/latency/size Pareto results;
the best reproducible artifacts load locally or from the Hub
```

This plan closes the execution and evidence gap. It intentionally does not build
the generalized W5 search engine or hosted leaderboard.

### Implementation alternatives

| Approach | Shape | Completeness | Effort | Risk | Decision |
|---|---|---:|---:|---:|---|
| A. Thin harness adapter | Make `run_method` accept an `ERModel`, add a few fields, keep method registry and manual tracking | 6/10 | M | Medium: preserves identity and threshold-replay debt | Rejected |
| B. Vertical experiment seam | Harden topology, add a versioned protocol and score cache, prove four recipes plus a custom topology, then add Hub transport | 10/10 | XL | Medium: broad but wave-gated | **Selected** |
| C. Big-bang research platform | Replace legacy APIs, build generalized search, live leaderboard, distributed execution, and Hub lifecycle together | 10/10 | XXL | High: no stable intermediate value | Rejected |

Approach B is the smallest design that achieves the confirmed outcome without
building another benchmark-only abstraction. It must ship in reversible waves,
each ending in a working benchmark proof.

### Scope decisions

Accepted inside the existing blast radius:

1. A versioned statistical evaluation protocol.
2. A separate cache identity for reusable expensive outputs.
3. Tiered measurements: mandatory comparison facts plus capability extensions.
4. Artifact claim levels that distinguish portable configuration from frozen
   local weights and benchmark reproducibility.
5. One custom-topology acceptance test in addition to the four recipes.
6. A generated static result-table example as a schema acceptance test, not a
   leaderboard product.

Deferred:

- adapters for Splink, Dedupe, Zingg, and pyJedAI;
- the hosted/live leaderboard;
- distributed benchmark execution;
- arbitrary-topology W5 search;
- new clustering algorithms;
- paper replication implementations.

### Statistical evaluation protocol

Add `EvaluationProtocol` with a version and these required fields:

- dataset id, fingerprint/revision, protocol, and split ids;
- exact fixed test-set identity shared by compared architectures;
- split seed set, reported as split-instability sensitivity rather than as
  independent population samples;
- deterministic-resource settings where available;
- stochastic repeat count and aggregation rule;
- threshold/calibration train split and untouched test split;
- requested pair and cluster metrics;
- confidence-interval method and level;
- hardware-cohort identity used for performance comparisons;
- benchmark implementation version.

An official comparison must report per-run rows plus aggregate mean, dispersion,
and confidence intervals. Quality uncertainty on a fixed test set uses a paired
cluster/entity bootstrap, preserving the dependency structure instead of
treating pair rows as independent. Missing paired cells remain missing and are
never imputed. A single LLM response may be cached for threshold replay inside
one attempt, but may not stand in for the protocol's independent stochastic
repeats.

The first official proof matrix is deliberately exact:

- five topologies: the four named recipes plus one custom topology;
- two datasets;
- one attempt for deterministic cells;
- three attempts for `RetrieveLLM` and `RetrieveRerankLLM`;
- 18 cells before retries;
- a preflight estimate checked against a USD 20 stopping threshold;
- paid-call concurrency of one.

The same matrix must have a fake/local smoke form that runs in CI. The paid
official matrix is an explicitly invoked publication workflow, not a CI job.

### Identity split

```text
recipe_id
  logical question: same architecture config + data + split + seeds?

evaluation_id
  statistical question: same dataset/splits/protocol/metric definitions and
  comparison cohort?

cache_id
  byte-reuse question: same code/lock + resource revisions + prompt/parser +
  runtime-affecting config + data fingerprint + input rows?

attempt_id
  execution question: which concrete run happened, when, and with what outcome?
```

`recipe_id` remains stable across code revisions for comparison and lineage.
`evaluation_id` prevents rows from different statistical questions from being
ranked together. Official/cache-publishing runs require a clean commit. Dirty
exploratory runs remain allowed, but their cache identity includes a source-tree
and diff hash and they cannot claim an official reproducibility level.

`cache_id` must include code/environment and every input that can change an
expensive output. Cache entries are immutable and content-addressed. A cache hit
is recorded in the stage measurement. Cache semantics are stage-specific:

- deterministic stages are reusable for an identical `cache_id`;
- seeded stages include their seed in `cache_id`;
- stochastic stages include repeat/attempt identity and a reused output cannot
  count as a statistically independent repeat.

### Measurement tiers

**Tier 0, required for comparable rows**

- protocol, architecture, and complete resource-slot identity;
- dataset/split/seed/repeat identity;
- pair and cluster quality metrics;
- funnel counts;
- total wall time and throughput;
- external call counts and token usage;
- observed/derived USD plus pricing provenance;
- hardware cohort, device, dtype, quantization, and batch size;
- status, warnings, and artifact claim level.

**Capability extensions**

- stage p50/p95 latency;
- cold/warm load split;
- CPU time;
- peak CPU/GPU memory;
- parameter, artifact, loaded-memory, and vector-byte facts;
- provider-specific namespaced usage.

An extension becomes Tier 0 only after two independent resource implementations
produce it consistently and it changes a documented comparison decision.

### Artifact claim levels

| Claim | Meaning | Minimum evidence |
|---|---|---|
| `portable_config` | topology and references can be reconstructed | manifest, registered types, pinned refs where possible |
| `frozen_local` | local weights/sidecars are content-addressed and bundled | hashes, environment compatibility, no remote code |
| `reproducible_benchmark` | the benchmark can be rerun under the declared protocol | protocol, dataset fingerprint, code/lock identity, reproduction command, run records |

An API-hosted model may produce `portable_config` and a benchmark record, but
cannot claim frozen behavior that the provider does not expose.

### Architecture review

```text
                          ┌────────────────────────────┐
ModelRef + RuntimeConfig ─▶ resource factories        │
                          │ Embedder · Reranker · LLM │
                          └─────────────┬──────────────┘
                                        ▼
records ─▶ Source/Retrieve ─▶ Score/Rerank ─▶ Select
                                           ├─▶ Generate ─▶ Parse ─▶ Select
                                           └──────────────────────────┐
                                                                      ▼
                                                                   Cluster
                                                                      │
                       ┌──────────────────────────────────────────────┘
                       ▼
ExperimentRunner ─▶ ScoreArtifact/Cache ─▶ replay Select/Cluster/Evaluate
       │                     │
       │                     └─ cache_id, immutable stage outputs
       ├─ RunStore: recipe_id + attempt_id + protocol
       ├─ ExperimentTracker: optional Trackio publication
       └─ ExperimentReport: per-run + aggregate + Pareto/cohorts

ERModel manifest ─▶ PretrainedArtifactAdapter ─▶ local path or HF snapshot
```

The experiment runner depends on public `ERModel`/topology contracts. Core does
not depend on experiments, tracking backends, benchmark datasets, or Hub
transport. Hub transport is an optional adapter around local persistence.

Scaling:

- At 10x pairs, vector materialization and reranker/LLM batches dominate.
- At 100x records, possible-pair calculation must remain arithmetic rather than
  materialized; stage outputs need bounded/chunked persistence.
- The first implementation may be single-process, but the result/cache schema
  must not require all rows in one Python object.

Rollback is a normal git revert of additive APIs. Legacy paths remain available,
so a user can return to the current runner and local `save/load` without artifact
migration.

### Data flow and state machines

```text
INPUT ─▶ VALIDATE ─▶ LOAD/SPLIT ─▶ BIND RESOURCES ─▶ EXECUTE ─▶ EVALUATE ─▶ PERSIST
  │         │             │               │              │           │          │
  ├ nil ────┴─ configuration error        │              │           │          │
  ├ empty benchmark ─ explicit empty-result policy       │           │          │
  ├ invalid split ─ BenchmarkProtocolError               │           │          │
  ├ missing extra ─ ResourceUnavailableError             │           │          │
  ├ load/API failure ─ cell failed, partial facts persist│           │          │
  ├ budget exceeded ─ terminal budget_exceeded + partial artifacts   │          │
  ├ incompatible metrics ─ separate cohort / reject comparison       │          │
  └ tracker failure ─ local result persists, publication_incomplete ─┘          │
                                                                               ▼
                                                                   typed report or error
```

```text
ExperimentCell

pending ─▶ running ─┬─▶ completed
                    ├─▶ failed
                    ├─▶ budget_exceeded
                    └─▶ interrupted  (inferred from stale running attempt)

resume creates a NEW running attempt linked to the prior attempt.
Terminal attempts never transition back to running.
```

Empty inputs are not silently converted into good scores. A valid empty split
produces explicit zero-count measurements and metrics whose mathematical value
is defined by the existing metric contract; an invalid benchmark with no
evaluation examples fails validation.

### Error and rescue registry

| Codepath | Failure | Exception | Rescue | Researcher sees |
|---|---|---|---|---|
| experiment validation | no architectures/benchmarks/seeds or invalid split | `ExperimentConfigurationError` | fail before execution | bad field plus valid values |
| benchmark loading | missing packaged data | existing `BenchmarkDataNotFoundError` | cell does not start | install/checkout remediation |
| protocol comparison | schema/version/cohort incompatible | `IncompatibleProtocolError` | separate cohort or explicit failure | exact incompatible fields |
| resource resolution | optional dependency absent | `ResourceUnavailableError` | fail affected cell | required extra and resource slot |
| resource loading | invalid model/revision/offline cache miss | `ResourceLoadError` | preserve failed attempt | model ref, revision, cache guidance |
| Hub snapshot | network, auth, missing revision, incomplete snapshot | `HubArtifactError` | no partial directory accepted | repo/revision and remediation |
| artifact rebuild | unknown registered type/version | `ArtifactCompatibilityError` | fail before inference | missing package/type/version |
| cache read | hash mismatch/torn artifact | `ScoreCacheError` | quarantine entry and recompute if safe | cache miss/recompute warning |
| LLM generation | timeout/rate limit/provider error | provider-specific error wrapped as `ResourceExecutionError` | configured retry, then failed cell | resource, attempt, retry count |
| response parsing | empty/refusal/malformed response | existing parse policy or `LLMParseError` | abstain or raise as configured | parse reason, no fabricated decision |
| spend enforcement | next call exceeds cap | existing `BudgetExceeded` | stop paid work, finalize partial run | partial report and spend |
| tracker publication | Trackio unavailable/auth/sync failure | `ExperimentPublicationError` | local result remains complete | publication incomplete and retry command |
| matrix orchestration | one cell fails | typed cell failure | continue independent cells | failed row retained |

No new broad `except Exception` rescue is permitted around resource execution.
Matrix orchestration may catch the package's typed cell-boundary base exception;
unexpected exceptions are persisted with a sanitized message and re-raised in
fail-fast mode.

### Security and privacy review

| Threat | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| arbitrary code from Hub artifact | Medium | High | registered built-ins only; never default `trust_remote_code` |
| path traversal in artifact files | Low | High | manifest-relative normalized paths; reject escape/symlink entries |
| accidental dataset/record upload | Medium | High | artifact allowlist; no implicit corpus/judgement-log publication |
| secret leakage in tracker config/errors | Medium | High | redact tokens/headers; never serialize credentials |
| prompt/record PII in telemetry | Medium | High | measurements exclude content by default; explicit opt-in artifacts |
| artifact hash substitution | Low | High | content hashes and pinned Hub commit |
| unbounded downloaded artifact | Medium | Medium | allow patterns, file-count/size validation before load |
| mutable API alias presented as frozen | High | Medium | artifact claim levels and response-model provenance |

The Hub adapter performs no shell execution and does not import downloaded Python
modules. Model cards and manifests are untrusted input and pass Pydantic
validation with bounded strings/collections.

### Code-quality review

- Keep evaluation and Hub transport outside `core`; they consume the inference
  contracts.
- Reuse `RunContext`, `RunStore`, tracker protocols, benchmark loaders, metrics,
  and `ModelRef`. Do not create a second registry or tracker hierarchy.
- Do not add one class per recipe stage when an existing `Source`, `Score`, or
  `Select` with an injected resource expresses the behavior.
- `Generate` and `Parse` may require an internal carrier extension. Prove the
  minimum representation before changing `Pairs`; do not add arbitrary
  multi-score dictionaries to every row.
- Public errors contain problem, cause, fix, and relevant identifiers without
  record content or secrets.

### Test review

```text
TOPOLOGY CONTRACT
├─ schema/is_bound from source                         [unit + regression]
├─ log threading on explicit dedupe/compare            [unit + regression]
├─ selection semantics                                 [unit + behavior]
└─ registered custom stage round-trip                  [integration]

MEASUREMENTS
├─ Tier 0 required / extensions optional               [unit]
├─ None is not zero                                    [unit]
├─ token inclusive/subtype invariants                  [unit]
├─ price snapshot and repricing                        [unit]
└─ hardware cohort normalization                       [unit]

EXPERIMENT PROTOCOL
├─ paired seeds/splits, no leakage                     [unit + integration]
├─ stochastic repeats and confidence intervals         [unit]
├─ recipe_id/cache_id/attempt_id separation            [unit]
├─ score once, replay threshold                        [integration]
├─ corrupt cache quarantine                            [fault injection]
├─ one failed cell does not erase matrix               [integration]
└─ four recipes + one custom topology                  [end-to-end fakes]

TRACKING
├─ local-only path imports no Trackio                   [import budget]
├─ Trackio spy receives params/metrics/artifacts        [integration]
└─ publication failure preserves local result           [fault injection]

HUB
├─ local save_pretrained/from_pretrained                [integration]
├─ Hub fake upload/download with pinned revision        [integration]
├─ unknown component / bad version                     [security failure]
├─ path traversal, symlink, oversize artifact           [security failure]
└─ no remote-code execution/import                      [AST + behavior]

REAL OPTIONAL SMOKES
├─ small pinned embedding/reranker                      [slow semantic]
└─ Hub snapshot cache                                   [slow network]
```

The hostile test is a resumed matrix containing a corrupt score cache, one
missing optional dependency, one malformed LLM response, and a Trackio outage:
unaffected cells complete; no paid call is duplicated; all failures remain
visible.

### Performance review

- Score-once replay is P1 because the current threshold loop rebuilds and
  executes a resolver per threshold.
- Cache rows must be chunked or streamed; do not serialize an unbounded pair
  matrix into one JSON value.
- Stage timing uses monotonic wall time. Percentiles require actual per-batch or
  per-item samples; do not infer p95 from a mean.
- Cold-load time is recorded separately from warm inference.
- Resource probes must be bounded and side-effect free; unavailable facts return
  `None` plus capability metadata.

### Observability and debugging review

Every attempt records:

- experiment, recipe, cache, attempt, and parent ids;
- current cell coordinates;
- stage entry/exit with counts and duration;
- cache hit/miss/recompute reason;
- resource identity and revision;
- budget/spend and token facts;
- terminal status and typed failure.

The local `RunStore` is the audit source. Tracker URLs are publication links, not
the only copy. A `langres reproduce <artifact-or-run>` CLI is not required in
this milestone; the report must emit a copy-paste reproduction command.

### Deployment and rollback review

```text
merge additive contracts
  └─ merge topology hardening
      └─ merge local experiment runner + Tier 0 measurements
          └─ prove recipe/custom-topology matrix
              └─ merge capability measurements + Trackio
                  └─ merge Hub transport
                      └─ full review and final integration PR

rollback:
failing sub-PR ─▶ revert merge on integration ─▶ run integration CI
final regression ─▶ retain legacy runner/save-load ─▶ revert additive facade
```

No database migration or service deployment is involved. Each sub-PR is a
two-way door. Hub and Trackio integration remain lazy optional imports.

### Long-term trajectory review

Reversibility: **4/5**. Additive APIs and retained legacy paths make rollback
easy. The main path-dependency risk is publishing an unstable protocol or
artifact schema. Mitigate with explicit versioning and a supported compatibility
window from the first release.

The plan supports later W4/W5 work by making resource slots, fit provenance, and
experiment identity visible. It does not assume that arbitrary topology mutation
is safe or useful before real benchmark evidence exists.

### Temporal interrogation

| Implementation time | Decision that must already be resolved |
|---|---|
| Foundation | package ownership, public contracts, protocol/cache/artifact versions |
| Core execution | score-artifact shape, selection replay boundary, stochastic-repeat semantics |
| Integration | cell failure policy, Trackio flattening, Hub manifest allowlist |
| Polish/tests | compatibility window, performance cohorts, reproduction command, docs migration |

### Failure modes registry

| Codepath | Failure mode | Rescued | Tested | Visible | Logged |
|---|---|---:|---:|---:|---:|
| protocol validation | incompatible split/metric/cohort | yes | yes | yes | yes |
| score cache | stale/corrupt/torn entry | yes | yes | yes | yes |
| resource load | missing dependency/model/revision | yes | yes | yes | yes |
| LLM call/parse | timeout/rate limit/refusal/malformed | yes | yes | yes | yes |
| budget | cap reached mid-cell | yes | yes | yes | yes |
| matrix | one cell fails | yes | yes | yes | yes |
| Trackio | publication unavailable | yes | yes | yes | yes |
| Hub | auth/network/incomplete snapshot | yes | yes | yes | yes |
| artifact | incompatible/unknown/path escape | yes | yes | yes | yes |
| process | interrupted after running record | yes | yes | yes | yes |

Critical silent gaps after the reviewed additions: **0 planned**.

### CEO implementation tasks

- [ ] **CEO-T1 (P1, human: ~1 day / CC: ~1h)** — protocol — Define
  `EvaluationProtocol`, `evaluation_id`, paired fixed-test-set comparisons,
  split-instability sensitivity, aggregation, and confidence intervals.
  - Verify: paired cluster/entity bootstrap plus deterministic and stochastic
    aggregate tests.
- [ ] **CEO-T2 (P1, human: ~1 day / CC: ~1h)** — identity — Add immutable
  `cache_id` distinct from `recipe_id` and `attempt_id`.
  - Verify: code/prompt/resource revision changes invalidate cache; logical
    recipe comparison remains stable.
- [ ] **CEO-T3 (P1, human: ~1 day / CC: ~1h)** — measurements — Split Tier 0
  comparison facts from resource capability extensions.
  - Verify: two dissimilar resources emit valid comparable records.
- [ ] **CEO-T4 (P1, human: ~1 day / CC: ~1h)** — artifacts — Enforce artifact
  claim levels and safe Hub loading.
  - Verify: reference-only API artifact cannot claim frozen behavior.
- [ ] **CEO-T5 (P1, human: ~2 days / CC: ~2h)** — benchmark proof — Run all
  four recipes plus one custom topology through the same experiment API.
  - Verify: exact two-dataset 18-cell matrix, USD 20 preflight cap, static
    report, and no runner special case.
- [ ] **CEO-T6 (P2, human: ~2h / CC: ~15m)** — reproduction — Emit a
  copy-paste command and attach local protocol artifacts before Trackio publish.
  - Verify: a clean process reconstructs the local result configuration.
- [ ] **CEO-T7 (P3)** — competitor adapters — Benchmark Splink/Dedupe/Zingg/
  pyJedAI through the protocol after the native seam is stable.
  - Destination: `TODOS.md`, separate follow-up.

### CEO completion summary

```text
Mode: SELECTIVE EXPANSION
System audit: W1-W3 landed; benchmark, tracking, and artifacts remain disconnected
Premises: 12 confirmed; 4 hardened without changing user direction
Architecture issues: 4 addressed in plan
Error paths: 13 mapped, 0 planned critical gaps
Security findings: 8 mitigations specified
Data/interaction edge cases: 10 classes covered
Code quality findings: 5 constraints specified
Test gaps: 6 P1 groups added
Performance findings: 5 constraints specified
Observability gaps: source-of-truth and identity clarified
Deployment risks: additive wave sequence and rollback defined
Long-term reversibility: 4/5
Design review: skipped, no UI scope
Outside voice: subagent-only; external Codex blocked by policy
```

## Decision audit trail

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 1 | CEO | Keep the confirmed research-foundation scope | Mechanical | completeness | The user confirmed all premises | scope reduction |
| 2 | CEO | Select vertical experiment seam approach | Taste | explicit + pragmatic | Working wave exits without a big-bang rewrite | thin adapter; big bang |
| 3 | CEO | Use SELECTIVE EXPANSION mode | Mechanical | pragmatic | Existing 0.x system enhancement | expansion; reduction |
| 4 | CEO | Add a versioned statistical protocol | Mechanical | completeness | Comparable stochastic results need repeats and uncertainty | single-run ranking |
| 5 | CEO | Add cache identity distinct from recipe identity | Mechanical | explicit | Logical comparison and byte reuse answer different questions | one overloaded id |
| 6 | CEO | Split required measurements from capabilities | Mechanical | pragmatic | Prevents a sparse ontology from blocking useful rows | all fields mandatory |
| 7 | CEO | Add artifact claim levels | Mechanical | explicit | Configuration portability is not frozen reproducibility | one reproducible flag |
| 8 | CEO | Preserve Trackio as official publication requirement | Mechanical | user direction | Local protocol remains source of truth; Trackio publishes it | Trackio as sole truth |
| 9 | CEO | Require one custom topology acceptance case | Mechanical | DRY | Proves the runner is architecture-generic | recipe-only dispatch |
| 10 | CEO | Defer competitor adapters and live leaderboard | Mechanical | stay in scope | They consume the protocol after it stabilizes | bundle now |
| 11 | CEO | Add `evaluation_id` beside recipe, cache, and attempt identities | Mechanical | explicit | Statistical comparability is independent of topology and byte reuse | rank rows by recipe id |
| 12 | CEO | Tie official cache publication to a clean commit | Mechanical | reproducibility | Dirty exploratory work remains possible without overstating its claim | reject all dirty runs |
| 13 | CEO | Make cache reuse semantics stage-specific | Mechanical | correctness | Seeded and stochastic stages cannot share deterministic reuse semantics | one global cache rule |
| 14 | CEO | Fix the first official proof at 18 cells and USD 20 | Mechanical | pragmatic | A bounded acceptance matrix makes cost and completion testable | open-ended matrix |

## GStack engineering review

### Verdict

**APPROVE WITH REQUIRED CORRECTIONS.** The direction is sound, but the
implementation must be a vertical extension of the current `Pairs`/`Op`,
`ERModel`, `RunRecord`, benchmark, tracker, and safe-manifest seams. It must not
create a second carrier, execution engine, run store, tracker interface, model
registry, or serialization trust model.

The external Codex engineering voice is unavailable under the same private-plan
export policy recorded in the CEO review. The independent repository audit and
the in-repo engineering pass therefore form the outside voice for this phase.

### Scope challenge and corrections

1. **Do not build generic caching.** Implement one local, immutable
   `StageArtifactStore` for declared replay boundaries. Distributed caches,
   eviction policy, and transparent caching of arbitrary Ops remain out of
   scope.
2. **Do not make experiment measurements a core dependency.** Core emits
   lightweight execution events and resource facts through protocols.
   `langres.experiments` owns measurement aggregation, statistics, reports, and
   tracker flattening.
3. **Do not duplicate the Op system with operation-named classes in another
   package.** `Retrieve`, `Rerank`, `Generate`, `Parse`, `Select`, and `Cluster`
   are topology vocabulary implemented by or adapted to the existing
   `Source`/`Score`/`Select`/`ClusterStage` contracts.
4. **Do not claim arbitrary topology replay.** A recipe declares a stable replay
   boundary immediately before its tunable `Select`. Custom topologies are
   benchmarkable without replay unless they explicitly declare a conforming
   boundary.
5. **Keep raw generations private by default.** `Generate` writes a versioned,
   typed generation envelope into pair provenance for the following `Parse`
   stage and local cache. Reports and Hub/Trackio publication contain usage and
   parsed outcomes, not prompts, records, or generated text, unless the caller
   opts in explicitly.
6. **Make Hugging Face Hub a direct optional extra.** Do not rely on Trackio or
   Sentence Transformers to install `huggingface_hub` transitively.
7. **Separate validity from convenience.** The four named recipes are
   convenience factories over the same conforming explicit topology contract;
   the fifth custom-topology cell proves the experiment runner does not dispatch
   on recipe names.

### What already exists and must be reused

| Requirement | Existing seam | Engineering use |
|---|---|---|
| pair carrier | `core.pairs.Pairs` / `PairRow` | stage input/output and local checkpoint payload |
| topology | `core.op` + `ERModel.from_topology` | one inference composition model |
| legacy compatibility | `core.op_adapters` | keep four-slot models running additively |
| local artifacts | `_model_persist` + `_artifacts` + registry | extend safe manifests; no pickle/remote code |
| model identity | `ModelRef` | every resource slot, including revision/runtime config |
| run lineage | `RunContext`, `RunRecord`, `RunStore`, `capture_run` | extend schema; do not add a second store |
| quality evaluation | `benchmarks.runner` and `judge_eval` | extract reusable metric functions and remove slot reach-through |
| token facts | `LLMUsage` | lossless migration into optional-field experiment usage |
| tracking | `ExperimentTracker` + `TrackioTracker` | one resolved tracker per experiment cell |
| datasets | benchmark registry and fingerprints | evaluation identity and matrix expansion |

### Package and dependency boundaries

```text
core leaves
  pairs · op · model_ref · serialization
       ▲
       │
core execution/persistence
  ERModel · op_adapters · registry · _model_run · _model_persist
       ▲                  ▲
       │                  │
resources (lazy heavy backends)     tracking (existing run/tracker seams)
  Embedder · Reranker · LLM              ▲
       ▲                                  │
       └──────────── experiments ─────────┘
                  protocol · identity · measurements · cache
                  statistics · runner · report
                         ▲
                         │
                benchmarks/data registry

hub adapter ──> core persistence + huggingface_hub (lazy, optional)
architectures ──> core contracts + resource Ops
```

Forbidden edges:

- `core -> experiments`;
- `core -> tracking.trackers` or `trackio`;
- eager `core/resources -> torch`, `transformers`, `litellm`, `trackio`, or
  `huggingface_hub`;
- `benchmarks -> resolver.module/blocker/clusterer`;
- Hub loading -> downloaded Python import or `trust_remote_code=True`.

### Public contracts

#### Topology execution

`ERModel` exposes read-only, slot-neutral introspection:

- `execution_plan() -> ExecutionPlan`, with ordered stable stage ids, safe
  serializable specs, resource-slot refs, schema/binding state, and an optional
  declared replay boundary;
- `execute(records, *, observer=None) -> ExecutionResult`, with selected pairs,
  clusters, and stage summaries;
- `execute_from(checkpoint, *, observer=None) -> ExecutionResult`, restricted to
  checkpoints minted by the same plan/cache identity.

The existing `resolve`, `dedupe`, `compare`, and four-slot properties remain.
They delegate to the same execution path where possible; no legacy API is
deleted in this milestone.

`ExecutionObserver` is an import-light callback protocol. It receives stage
start/finish/failure events with counts and durations, but owns no experiment
models and cannot change the result.

#### Resources and operations

- `Embedder.embed(texts) -> EmbeddingBatch`, carrying vectors plus optional
  typed embedding facts.
- `Reranker.rerank(pairs) -> RerankBatch`, carrying one score per input pair and
  optional usage/resource facts.
- `LLM.generate(requests) -> GenerationBatch`, carrying outputs, serving
  identity, and optional token usage.
- Resource instances expose a stable `ModelRef` and runtime config without
  loading weights during configuration or import.
- Resource Ops adapt these capabilities into the existing `Pairs -> Pairs`
  algebra. `Generate` and `Parse` exchange a versioned typed envelope in local
  provenance; parsing validates it and can emit match/no-match/abstain.

The same `Reranker` resource is accepted by any `Rerank` Score. Whether its
output filters candidates or makes the final decision is determined solely by
the following `Select`.

#### Experiment identity

```text
recipe_id      architecture/resources/data/seeds: logical recipe lineage
evaluation_id  protocol/dataset/splits/metrics/cohort: comparable question
cache_id       code/lock/source/resources/stage/input: immutable byte reuse
attempt_id     one concrete execution
```

All four identities are explicit on the extended run record. A publishable
cache entry requires a clean commit. A dirty exploratory cache includes the
source-tree/diff hash and is marked non-official. Deterministic entries may be
reused; seeded entries include their seed; stochastic entries include the
repeat/attempt and never substitute for independent repeats.

#### Results and statistics

`ExperimentReport` contains immutable per-cell `ExperimentRun` records and
derived aggregate/cohort views. It never silently drops or imputes failed or
missing cells.

On the fixed official test set, paired architecture differences use a
cluster/entity-level paired bootstrap. Pair rows are not treated as independent
samples. Re-running alternate split seeds is reported separately as split
instability. Threshold/calibration provenance names the train/validation data;
test data is never used to choose a threshold.

### Data and state flow

```text
Experiment.run
  │
  ├─ expand protocol × architecture × dataset × split × repeat
  ├─ preflight dependencies, identities, clean-state claim, and budget
  └─ for each cell
       ├─ append RunRecord(running)
       ├─ load dataset once + fingerprint
       ├─ build/bind architecture and execution plan
       ├─ cache lookup at declared replay boundary
       │    ├─ miss: execute expensive prefix → atomic checkpoint commit
       │    └─ hit: validate plan/input/cache identity
       ├─ replay Select/Cluster/Evaluate for threshold candidates
       ├─ choose threshold on train/validation only
       ├─ evaluate untouched test set
       ├─ persist measurements/results/artifact pointers locally
       ├─ optionally publish flattened facts to Trackio
       └─ append terminal RunRecord(completed/failed/budget_exceeded)
```

Interrupted attempts retain the running record and any atomically committed
stage checkpoint. Resume creates a new `attempt_id`; it never mutates the
previous attempt and only reuses a checkpoint whose full `cache_id` validates.

### Failure modes and rescue

| Failure | Required engineering behavior |
|---|---|
| corrupt/torn cache entry | checksum/manifest validation fails; quarantine/ignore entry and recompute |
| stale checkpoint | reject before `execute_from`; show the identity fields that differ |
| duplicate pair ids | fail before reranking/cache commit; never last-write-wins silently |
| provider missing usage | preserve `None` plus warning; never manufacture zero |
| generation parse failure/refusal | typed abstention/error measurement; other pairs continue when safe |
| paid timeout/rate limit | bounded retry policy; attempts/calls/tokens recorded; cap checked before retry |
| budget preflight over USD 20 | official matrix does not start |
| budget exceeded in a cell | stop further paid calls; persist partial cell |
| Trackio unavailable | local terminal record survives; official publication state is incomplete |
| Hub partial download/upload | stage in temporary directory; validate before atomic promotion |
| unknown component/resource | fail before inference with missing package/registration guidance |
| dirty source for official run | reject official claim; allow explicitly exploratory run |

### Security and privacy constraints

- Hub manifests accept registered type names and JSON-safe configuration only.
- Path traversal, symlink escape, oversized manifest, checksum mismatch, unknown
  layout version, and unknown component tests are mandatory.
- Authentication tokens are never serialized or logged.
- Upload allowlist contains only the validated model artifact, generated card,
  and explicitly selected measurement summary.
- Dataset rows, prompts, generations, judgement logs, and local caches are
  excluded from upload by default.
- Model cards distinguish reference-only, frozen-weights, and benchmark-
  reproducible claim levels.

### Test coverage diagram

```text
contract/unit
├─ identities: canonicalization + clean/dirty + stage cache semantics
├─ protocol: split leakage + evaluation cohorts + exact matrix expansion
├─ measurements: None != 0 + token subsets + repricing
├─ statistics: paired cluster/entity bootstrap + missing cells
├─ resources: fake Embedder/Reranker/LLM + ModelRef revision/runtime facts
└─ Hub manifest: allowlist + path/checksum/type/version failures
        │
        ▼
core integration
├─ explicit schema/is_bound/logging/index lifecycle
├─ compare/resolve selection parity
├─ registered custom Op persistence
└─ legacy four-slot compatibility
        │
        ▼
experiment integration
├─ score once / threshold replay
├─ RunStore running→terminal + resume
├─ Trackio spy flattening
├─ per-cell failure continuation
└─ five topologies × two datasets fake/local matrix
        │
        ▼
opt-in integration
├─ pinned tiny HF embedder/reranker
├─ Hub upload/download double
└─ paid 18-cell publication workflow (never CI)
```

Regression gates include `ruff check`, `ruff format --check`, strict `mypy src`,
the fast zero-network suite, full non-integration coverage, core coverage,
import-budget tests on a core-only install, wheel-content tests, and strict docs.

### Worktree parallelization strategy

```text
Lane A: topology/observer/persistence hardening ─────┐
Lane B: protocol/identity/measurement/statistics ───┼─> integration contract
Lane C: fake resources + operation adapters ────────┘
                                                     │
       ┌─────────────────────────────────────────────┼────────────────────┐
       ▼                                             ▼                    ▼
Lane D: runner/cache/report                  Lane E: named recipes   Lane F: Hub
       └─────────────────────────────────────────────┼────────────────────┘
                                                     ▼
                                     docs + matrix + migration + review
```

Lanes A-C may start from the same integration baseline with exclusive file
ownership. D-F begin only after their declared foundation PRs are green and
merged. Every later branch starts from the latest green integration commit.

### Engineering implementation tasks

- [ ] **ENG-T1 (P1)** — topology contract — Fix explicit-chain schema/binding,
  logging, selection parity, index binding, public introspection, and benchmark-
  safe execution hooks.
  - Files: `core/_model_state.py`, `core/_model_run.py`, `core/op.py`,
    `core/op_adapters.py`, focused core tests.
  - Depends on: none.
- [ ] **ENG-T2 (P1)** — safe extensible persistence — Register supported Op
  serializers instead of exact-class branching; retain fail-closed loading.
  - Files: `core/registry.py`, `core/serialization.py`, `core/_artifacts.py`,
    `core/_model_persist.py`, artifact tests.
  - Depends on: none.
- [ ] **ENG-T3 (P1)** — protocol and identity — Add `EvaluationProtocol`,
  `evaluation_id`, `cache_id`, source-state identity, exact matrix expansion,
  and extend existing run records compatibly.
  - Files: new `experiments/protocol.py`, `experiments/identity.py`,
    `tracking/runs.py`, tests.
  - Depends on: none.
- [ ] **ENG-T4 (P1)** — measurements/statistics — Add tiered measurements,
  runtime/resource facts, price snapshots, repricing, paired bootstrap, split-
  instability reporting, cohorts, Pareto/constraint views.
  - Files: new `experiments/measurements.py`, `experiments/statistics.py`,
    `experiments/report.py`, tests.
  - Depends on: ENG-T3.
- [ ] **ENG-T5 (P1)** — resource seams and Ops — Implement import-light
  `Embedder`, `Reranker`, and `LLM` protocols, fake resources, lazy concrete
  adapters, and typed Generate/Parse exchange.
  - Files: new `resources/`, resource Op adapters, `core/embeddings.py`,
    transformer/LiteLLM backends, tests.
  - Depends on: ENG-T1.
- [ ] **ENG-T6 (P1)** — experiment runner/cache — Adapt the benchmark harness to
  slot-neutral execution, implement atomic declared-boundary cache/replay,
  automatic `capture_run`/tracker wiring, failure continuation, and resume.
  - Files: new `experiments/cache.py`, `experiments/runner.py`,
    `benchmarks/runner.py`, tests.
  - Depends on: ENG-T1, ENG-T3, ENG-T4.
- [ ] **ENG-T7 (P1)** — named recipe proof — Implement four recipes plus one
  custom-topology conformance case over the same runner; no name dispatch.
  - Files: `architectures/`, experiment integration tests.
  - Depends on: ENG-T5, ENG-T6.
- [ ] **ENG-T8 (P1)** — Hub lifecycle — Add the direct `hub` extra, local
  `save_pretrained`, safe `from_pretrained`, `push_to_hub`, claim levels,
  revision capture, allowlisted card generation, and Hub test doubles.
  - Files: `pyproject.toml`, new `hub.py`/artifact helpers, persistence tests.
  - Depends on: ENG-T2, ENG-T3, ENG-T4.
- [ ] **ENG-T9 (P1)** — acceptance matrix — Run the zero-network fake/local
  five-topology × two-dataset matrix in CI and provide the guarded 18-cell,
  concurrency-one, USD-20 paid publication command outside CI.
  - Files: experiment matrix tests, examples/research command.
  - Depends on: ENG-T7, ENG-T8.
- [ ] **ENG-T10 (P2)** — docs/migration — Update code-coupled rules and docs in
  the same PRs, then add the end-to-end research tutorial, reproduction command,
  repricing example, static result table, and migration map.
  - Files: `AGENTS.md`, `.claude/rules/component-design.md`, `docs/`, examples.
  - Depends on: ENG-T1 through ENG-T9.

### NOT in scope

- arbitrary DAG execution or multi-carrier topology;
- distributed cache/execution or cache eviction policy;
- auto-search over topology mutations (W5);
- generic model training/fine-tuning orchestration (W4 beyond identity/runtime
  slots needed by the recipes);
- raw prompt/generation publication;
- competitor framework adapters;
- new cluster algorithms;
- hosted leaderboard UI;
- removing four-slot compatibility;
- production batching for every local LLM.

### Engineering completion summary

```text
Verdict: APPROVE WITH REQUIRED CORRECTIONS
Existing systems reused: 10
New top-level product seams: experiments, resources, Hub adapter
Forbidden duplicate systems: 6
Parallelization: 6 lanes; 3 foundation lanes parallel, 3 dependent lanes parallel
P1 tasks: 9
P2 tasks: 1
Required fake/local acceptance: 5 topologies × 2 datasets
Paid acceptance: 18 cells, concurrency 1, USD 20 preflight stopping threshold,
not CI; one in-flight provider call may overshoot
Legacy deletion: none
External Codex voice: unavailable by policy; repository-audit voice used
```

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 15 | Eng | Extend current carriers, execution, run, and persistence seams | Mechanical | DRY | Parallel systems would make experiments disagree with production inference | new execution stack |
| 16 | Eng | Use declared replay boundaries | Taste | explicit | Generic transparent caching is unsafe and unnecessary for the first proof | cache every Op |
| 17 | Eng | Keep experiment models outside core | Mechanical | layering | Measurements consume inference; core must remain import-light | core experiment dependency |
| 18 | Eng | Add a direct optional Hub extra | Mechanical | explicit | Transitive optional dependencies are not a public contract | rely on Trackio |
| 19 | Eng | Keep raw generation data local by default | Mechanical | privacy | Reproducibility does not require implicit publication of sensitive text | publish all traces |
| 20 | Eng | Pair-bootstrap fixed test entities; report split instability separately | Mechanical | statistical correctness | Split seeds and pair rows are not independent population samples | seed/pair pseudoreplication |

## GStack developer-experience review

### Mode and verdict

Mode: **POLISH**. Verdict: **APPROVE WITH DX GATES**.

The target persona is a Python ML/entity-resolution researcher who already
understands Hugging Face model ids, Sentence Transformers, LiteLLM model strings,
and benchmark tasks, but should not need to learn langres's historical
Blocker/Matcher/Judge implementation vocabulary before running an experiment.

The independent DX subagent did not return within the bounded review window.
The review therefore used the repository's public docs/examples plus the
ecosystem comparison gathered in the CEO pass. This is recorded as degraded
outside-voice coverage, not silently counted as consensus.

### Developer empathy narrative

The researcher arrives with a concrete question, not a desire to assemble
framework internals:

> “Does this embedding model retrieve the right pairs? Does a reranker improve
> the separation enough to avoid most LLM calls? If I swap the LLM or run the
> same architecture locally, what happens to quality, latency, tokens, memory,
> and cost? Can another researcher reproduce the exact result?”

Their first success must be a real benchmark row, not a configured object or a
diagram. Their second success must be changing one model id without rewriting
the pipeline. Their third must be a reproduction artifact they can hand to
someone else. Every extra registry, wrapper role, implicit split, and manually
wired tracker increases the chance that they abandon the standardized path and
write a one-off notebook.

### Time-to-value targets

- Time from installed core package to first meaningful zero-network experiment:
  **under 3 minutes**.
- Time from installed semantic extra to first pinned embedding/reranker
  experiment, excluding first model download: **under 5 minutes**.
- Model swap: **one constructor value**, no pipeline rewrite.
- Two datasets × three seeds: **one list change**, with split policy visible.
- Local reproduction from a saved run: **one copy-paste command**.
- Official publication: the same run plus `tracker="trackio"` and an explicit
  publication/profile choice; no manual metric flattening.

### Nine-stage developer journey

| Stage | Researcher question | Likely friction | Required DX response |
|---|---|---|---|
| 1. Discover | Is this for linking/deduping and research comparisons? | feature list without a mental model | README shows the four recipes and one result table first |
| 2. Evaluate | Can it compare quality, speed, size, tokens, and cost fairly? | vague “benchmark” claims | link the versioned protocol, cohort rules, and exact stored facts |
| 3. Install | Which extras do I need? | semantic/LLM/Trackio/Hub dependency ambiguity | task-oriented install table and errors naming one exact extra |
| 4. Hello world | Can I get a result without credentials or downloads? | examples start with provider/model setup | built-in tiny dataset + offline baseline through the real `Experiment` API |
| 5. Configure | Where do embedder/reranker/LLM ids go? | model refs repeated across blocker/matcher wrappers | one resource value per named recipe slot; strings normalize to `ModelRef` |
| 6. Run | Which datasets, splits, seeds, thresholds, and budget execute? | hidden defaults and combinatorial surprise | printed/returned preflight matrix with cells, paid calls, cache hits, and cap |
| 7. Debug | Why is a cell missing or incomparable? | backend traceback or silent dropped row | problem/cause/fix errors plus persisted failed-cell record and cohort warning |
| 8. Publish/reproduce | What exactly can I share? | Trackio, Hub, local artifacts conflated | one reproduction bundle; explicit claim level and separate result/model publication |
| 9. Extend/upgrade | How do I add a paper model? | subclassing role-specific Blocker/Matcher types | one resource protocol + Op adapter guide, conformance tests, artifact registration |

### Ecosystem consistency benchmark

| Ecosystem pattern | langres application |
|---|---|
| Transformers `from_pretrained(model_id, revision=...)` | same familiar model/artifact loading shape, with safe registered manifests |
| Sentence Transformers `SentenceTransformer(model_id)` / `CrossEncoder(model_id)` | `Embedder` and `Reranker` resources accept a model id or typed `ModelRef` once |
| LiteLLM model strings | `LLM` is the resource category; concrete LiteLLM resource accepts provider/model ids |
| MTEB model + task evaluation | `Experiment(architectures=..., benchmarks=...).run()` |
| Qdrant explicit client/index configuration | runtime/index config is explicit and measurable, never inferred from role names |
| Trackio init/log | tracker setup is optional and automatic around a run; native tracker remains an escape hatch |

This yields the public hierarchy:

```text
langres.architectures  Retrieve · RetrieveRerank · RetrieveLLM · RetrieveRerankLLM
langres.resources      Embedder · Reranker · LLM · VectorIndex
langres.experiments    Experiment · EvaluationProtocol · ExperimentReport
langres.core           Pairs · Op · Score · Select · ClusterStage (authoring level)
```

To avoid a `Retrieve` recipe/operation import collision, public recipe examples
import from `langres.architectures`; low-level operation authoring uses the
`langres.core.ops` namespace. Documentation may call the operation “Retrieve”
without inventing `VectorBlocker`/`RetrieverMatcher` role names.

### Magical-moment examples

The zero-network path must exercise the real experiment seam:

```python
from langres.architectures import FuzzyString
from langres.experiments import EvaluationProtocol, Experiment

report = Experiment(
    architectures={"fuzzy": FuzzyString()},
    benchmarks=["tiny_fixture"],
    protocol=EvaluationProtocol.smoke(seed=0),
).run()

print(report.to_markdown())
print(report.reproduce_command)
```

Target output:

```text
| architecture | dataset      | pair_f1 | bcubed_f1 | seconds | tokens | usd |
| fuzzy        | tiny_fixture | 1.0000  | 1.0000    | 0.02    | —      | —   |

Reproduce: langres experiments reproduce .langres/runs/<evaluation_id>.json
```

The first resource-composition example changes only the architecture:

```python
from langres.architectures import RetrieveRerank

architecture = RetrieveRerank(
    embedder="sentence-transformers/all-MiniLM-L6-v2",
    reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
)
```

The same `Experiment` call then accepts multiple benchmarks and seeds. The split
policy is part of `EvaluationProtocol`, not another loose list that can conflict
with dataset capabilities.

### Naming and cognitive-load decisions

- Keep `Embedder`, `Reranker`, and `LLM`; do not expose “vector model,” “cross
  encoder,” “language model,” “judge,” or “matcher” as competing generic slot
  names.
- `cross-encoder/...` is a concrete reranker model id, not the resource type.
- Use `benchmarks`, not a mix of `tasks`/`datasets` in the public experiment
  constructor. The protocol record still calls the underlying data a dataset.
- Put splits, seeds, repeats, statistics, thresholds, and cohorts in one
  `EvaluationProtocol`; avoid contradictory top-level knobs.
- Strings are the convenience path. `ModelRef`/resource objects are the
  inspectable advanced path.
- Use `report.runs`, `report.aggregate(...)`, `report.pareto(...)`,
  `report.to_markdown()`, and `report.reproduce_command`; avoid multiple result
  table types.
- Use `None`/em dash for unknown facts and `0` only for measured zero.
- Keep `Matcher`, `Blocker`, and `Judge` only in migration and compatibility
  documentation.

### Preflight and error experience

Before executing, `Experiment.plan()` and `run(dry_run=True)` return the expanded
matrix:

```text
10 cells: 5 topologies × 2 datasets
6 deterministic attempts, 12 stochastic attempts, 18 total
paid concurrency: 1
estimated maximum: USD 12.40 / stopping threshold: USD 20.00
cache: 4 reusable prefixes, 14 misses
publication: local + Trackio; official claim eligible
```

Errors follow one format:

```text
Cannot load reranker 'org/model' (revision 'abc').
Cause: the semantic extra is not installed.
Fix: pip install 'langres[semantic]' and retry, or pass a Reranker resource
that is already available locally.
Cell: RetrieveRerank / amazon_google / test / repeat 0
```

Backend stack traces remain available through exception chaining/debug logging,
but the first message always names the resource slot, cell, cause, and fix.

### Documentation information architecture

1. README: value proposition, four-recipe diagram, offline experiment, result
   table, install chooser.
2. `GETTING_STARTED`: offline baseline → embedding → reranker → optional LLM.
3. `EXPERIMENTS`: protocol, matrix/preflight, score reuse, cohorts, reports,
   repricing, Trackio.
4. `ARCHITECTURES`: resources vs operations vs recipes; custom topology example.
5. `REPRODUCIBILITY`: identities, clean/dirty claims, local bundle, Trackio,
   Hub, privacy defaults.
6. `ADDING_A_METHOD`: migrate to “adding a resource/architecture,” with
   conformance-test checklist.
7. Reference: generated API docs and migration mapping for legacy terms.

### DX implementation tasks

- [ ] **DX-T1 (P1)** — golden public imports/API — Lock the constructor,
  protocol, report, and resource-slot spellings with type/repr/signature tests.
  - Files: package exports, `experiments/__init__.py`,
    `architectures/__init__.py`, API snapshot tests.
  - Depends on: ENG-T3, ENG-T5, ENG-T7.
- [ ] **DX-T2 (P1)** — zero-network first run — Add the tiny-fixture
  `EvaluationProtocol.smoke` path and keep it on the real runner.
  - Files: experiment presets, `examples/research/first_experiment.py`, tests.
  - Depends on: ENG-T6, ENG-T7.
- [ ] **DX-T3 (P1)** — matrix preflight — Implement `Experiment.plan()` and
  `run(dry_run=True)` with cell counts, repeats, cache state, publication
  eligibility, and budget estimate.
  - Files: experiment runner/report, CLI formatting tests.
  - Depends on: ENG-T3, ENG-T6.
- [ ] **DX-T4 (P1)** — actionable errors — Add resource/cell context and exact
  extra/remediation messages for optional dependency, model, cache, tracker,
  budget, and Hub failures.
  - Files: resource loaders, experiment errors, Hub adapter, tests.
  - Depends on: ENG-T5, ENG-T6, ENG-T8.
- [ ] **DX-T5 (P1)** — reproduction handoff — Generate a stable reproduction
  bundle and one copy-paste command; verify it in a clean process.
  - Files: experiment reproduction module/CLI, practical test.
  - Depends on: ENG-T6, ENG-T8.
- [ ] **DX-T6 (P1)** — progressive tutorial/doc migration — Ship the information
  architecture above and update code-coupled rules in the same changes.
  - Files: README, `docs/GETTING_STARTED.md`, `docs/EXPERIMENTS.md`,
    architecture/reproducibility docs, `docs/ADDING_A_METHOD.md`, examples.
  - Depends on: DX-T1 through DX-T5.
- [ ] **DX-T7 (P2)** — static leaderboard proof — Generate the docs table from
  the real fake/local report artifact; no hand-maintained scores.
  - Files: docs data/script/page.
  - Depends on: ENG-T9, DX-T6.

### DX scorecard

| Dimension | Current plan | Required exit |
|---|---:|---:|
| discoverability | 6/10 | 9/10 |
| installation clarity | 6/10 | 9/10 |
| first-run speed | 5/10 | 9/10 |
| naming consistency | 8/10 | 9/10 |
| configuration load | 7/10 | 9/10 |
| error recovery | 5/10 | 9/10 |
| reproducibility handoff | 7/10 | 9/10 |
| extension path | 6/10 | 8/10 |

### DX completion summary

```text
Mode: POLISH
Primary persona: Python ML/entity-resolution researcher
Target first meaningful zero-network run: <3 minutes
Target model swap: one constructor value
Journey stages reviewed: 9
P1 DX tasks: 6
P2 DX tasks: 1
Naming changes: no new role-specific resource types
Legacy deletion: none
Outside voice: timed out; degraded coverage recorded
```

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 21 | DX | Put experimental policy in `EvaluationProtocol` | Mechanical | explicit | Avoids conflicting split/seed/repeat knobs | loose constructor lists |
| 22 | DX | Make the first run offline and real | Mechanical | time-to-value | A fake configuration object is not a meaningful success | provider-first tutorial |
| 23 | DX | Namespace recipe and low-level operation imports | Taste | cognitive load | Keeps the agreed words without Python import collisions | role-specific renaming |
| 24 | DX | Add matrix/budget/cache preflight | Mechanical | cost honesty | Researchers must see combinatorial and paid work before execution | silent expansion |
| 25 | DX | Generate one reproduction command | Mechanical | handoff | Reproduction should not require reconstructing documentation steps | manual checklist |
| 26 | Approval | Make experiment budgets optional | Mechanical | user direction | A cap is valuable when requested but must not be required for every experiment; the official paid proof still supplies one | mandatory budget on all runs |

## Cross-phase synthesis

### Voice consensus

#### CEO consensus

| Question | Plan/root review | Independent CEO voice | Resolution |
|---|---|---|---|
| build broad model matrix now? | no, build experiment seam first | require a vertical proof first | confirmed |
| measurement scope | capture durable facts broadly | tier mandatory facts vs capabilities | adopted tiering |
| statistical validity | repeats and confidence intervals | prevent pseudoreplication | paired entity/cluster bootstrap plus separate split instability |
| cache identity | separate recipe and cache | add source state and stage semantics | adopted |
| reproducibility | Trackio required for official record | local artifact must remain source of truth | adopted |
| competitive differentiation | comparable ER architectures | neutral cross-class comparison is the value | competitor adapters deferred; protocol kept neutral |

Consensus: **6/6 confirmed after corrections**. External Codex execution was
unavailable by policy; the independent CEO subagent supplied the outside voice.

#### Engineering consensus

| Question | Plan/root review | Independent repository audits | Resolution |
|---|---|---|---|
| foundation quality | W1-W3 are usable but incomplete | explicit topology is not yet the universal benchmark contract | hardening Wave A required |
| execution architecture | one experiment seam over `ERModel` | benchmark slot reach-through is the main blocker | slot-neutral execution contract |
| resource naming | capabilities independent of role | current model matrix and multi-slot identity are incomplete | `Embedder`/`Reranker`/`LLM` resources |
| cache design | score once, replay selection | threshold tuning repeats paid inference | declared replay boundary and immutable local store |
| Hub design | extend safe local artifacts | Hub lifecycle absent; schema/revision portability matters | direct optional Hub adapter, fail closed |
| parallel delivery | foundation then integrations | docs/import/persistence constraints cross-cut work | six lanes with green integration gates |

Consensus: **6/6 confirmed after scope narrowing**. External Codex was
unavailable by policy; the two earlier independent codebase audits supplied the
outside engineering evidence.

#### DX consensus

| Question | Plan/root review | Ecosystem/repository evidence | Resolution |
|---|---|---|---|
| first success | one `Experiment.run()` | current systems require manual bridging | offline real-run golden path |
| vocabulary | resources + operations + recipes | HF/LiteLLM/MTEB use model/task/evaluate concepts | agreed vocabulary and namespaces |
| configuration | model refs once, multiple benchmarks explicit | existing APIs sometimes ignore or lose model identity | typed slot identity plus string convenience |
| cost visibility | raw facts and price snapshot | matrix expansion can multiply paid work | dry-run/preflight required |
| reproduction | local + Trackio + Hub | current docs separate these workflows | one reproduction bundle/command |
| extension | reusable resource, no role duplication | current custom persistence is a whitelist | conformance contract plus registered safe artifact |

Consensus: **6/6 based on available ecosystem and repository evidence**.
Independent DX subagent execution timed out, so this is explicitly degraded
rather than counted as a second live model voice.

### Cross-phase themes

1. **Vertical proof before breadth** — CEO, engineering, and DX all require one
   real five-topology benchmark path before adding a large model/paper matrix.
2. **One source of truth** — every phase rejects duplicate execution, identity,
   measurement, tracker, or artifact systems.
3. **Facts and identity before ranking** — quality, tokens, size, hardware,
   price provenance, protocol, and compatible cohorts must exist before a
   leaderboard can be trusted.
4. **Explicit cost and privacy boundaries** — matrix preflight, stage-specific
   cache semantics, clean-source claims, and no implicit raw-data publication
   recur in all phases.
5. **Familiar surface, strict internals** — model-id convenience and
   `from_pretrained` familiarity sit on typed resources, fail-closed manifests,
   versioned protocols, and actionable errors.

### Deferred follow-ups

- competitor adapters for Splink, Dedupe, Zingg, and pyJedAI;
- hosted/live MTEB-style leaderboard;
- arbitrary topology search and W5 mutation logic;
- broader W4 training/fine-tuning orchestration;
- distributed execution/cache;
- new clustering/transitivity algorithms;
- paper implementations and the broad model matrix;
- production local-LLM batching across every backend.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 1 review + 3 spec-review iterations | CLEAR | 7 tasks; protocol, identity, measurement, artifact, and vertical-proof scope locked |
| Codex Review | `/codex review` | Independent 2nd opinion | 1 attempted | UNAVAILABLE | Private workspace export blocked by policy; no workaround used |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR | 10 tasks; 6 lanes; 0 planned critical silent gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | SKIPPED | No UI scope; leaderboard UI explicitly deferred |
| DX Review | `/plan-devex-review` | Developer experience gaps | 1 | CLEAR WITH DEGRADED VOICE | 7 tasks; 8 dimensions; first meaningful run target under 3 minutes |

**CROSS-MODEL:** Available independent CEO and repository-audit voices agree
that the experiment seam, statistical identity, score reuse, and one vertical
proof must precede the broad model/paper matrix. External Codex and the bounded
DX subagent were unavailable; those gaps are disclosed rather than counted as
consensus.

**VERDICT:** CEO + ENG + DX CLEARED — implementation approved by the user on
2026-07-18, with optional ordinary-run budgets. Design correctly skipped. No
legacy API deletion.

NO UNRESOLVED DECISIONS
