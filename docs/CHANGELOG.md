# Changelog

## [Unreleased] - POC Phase

- Designed two-layer API architecture and POC validation plan (3 approaches: classical, semantic, LLM hybrid)
- Implemented core primitives (`Module`, `Blocker`, `Clusterer`) with Pydantic data contracts and 100% test coverage
- Completed Approach 1 (classical baseline): `AllPairsBlocker` + `RapidfuzzModule` end-to-end pipeline

### M4: langres is the seam — DSPy experimentation foundation + first paid signal

Reframed from a distillation-metric chase to **building the composable scorer seam we're
happy to use** — the plumbing to fix M3's cheap-judge precision collapse data-drivenly.
KISS: the smallest seam that proves the plumbing and yields a first honest signal.

- **`DSPyJudge`** — import-safe (`import langres.core` never imports `dspy`),
  `compile(bootstrap|mipro)`, honest per-pair cost, serializable — behind the `Module`
  contract. (`src/langres/core/modules/dspy_judge.py`.)
- **`derive_threshold(scores, labels)`** (Youden / percentile) — data-driven thresholds
  replacing hand-set magic constants; demo lifts held-out AG pair-F1 +0.16 over 0.5.
- **`run_methods(...) -> BenchmarkTable`** experiment facade (`.best()` / `.rank()`) +
  **`langres.clients.openrouter`** (price-pinning, `SpendMonitor` cumulative-spend guard,
  extracted from `examples/m3_race.py`).
- **Proven end-to-end at $0** (DummyLM): a compiled `DSPyJudge` → `compile` →
  `evaluate_judge_on_candidates` (judged-once, pairwise-F1, SOTA-comparable) — the right
  surface for a compiled/paid judge; `run_methods` is the cheap-method race. Getting
  started: `docs/EXPERIMENTS.md`, `examples/m4_experiment_loop.py`.
- **Review fixes on the seam:** Youden `+inf` ROC-sentinel guard, held-out train/test
  calibration split, `run_methods` stamps the requested registry method name, DSPy
  `temperature` forwarded to the LM, `load_state` restores the real compiled flag,
  parse-error branch marks the call billed-but-untrackable.
- **Research-driven direction:** ER SOTA seam audit tracked in
  `docs/research/20260701_er_seam_audit.md` + issue #55; two adjustments — a
  frontier-zero-shot null-baseline gate before paid distillation (C7), and promoting the
  set-wise judgement contract (S1) to M4.5.
- **First paid signal ($2.31/$5 on the 600-pair AG band — `data/benchmarks/m4/M4_RESULTS.md`):**
  a precision-tuned DSPy **signature** lifts the cheap GLM-5.2 judge from pair-F1
  **0.409 → 0.757** (precision 0.264 → 0.671), **beating frontier gpt-4o (0.667) at lower
  cost — uncompiled**. **MIPROv2 compilation did *not* help** (0.757 → 0.746, +$1.63): it
  overfit its 40-example bootstrap metric, confirming the OpenSanctions caveat. **C7
  verdict: the lever is the signature, not compilation — cut distillation.** Honest
  per-pair cost recorded; compiled `Resolver` artifact serialized. Harness
  `examples/m4_race.py` (resumable, per-cell-committed, budget-capped).
- **Deferred to M4.5:** set-wise contract (S1), blocking pair-set algebra + embedder
  sweep, `fit()`-hook trained-judge family (S2).

### M3: The seam — multi-method benchmark race (real-money EXIT)

Raced free scorers (`rapidfuzz`, `weighted_average`, `embedding_cosine`) against an
open-source (**GLM-5.2**) and a frontier (**gpt-4o**) `llm_judge` over an *easy*
(Fodors-Zagat) and a *hard* (Amazon-Google) dataset, under a hard **$15** budget cap.
Pair-level F1 (widened 0.05–0.99 grid) is the primary judge-ranking metric.

- **Total measured spend: $2.1778** / $15.00 cap. Cost is priced from provenance token
  counts (litellm `completion_cost` returns $0 for OpenRouter's provider-less dated
  model ids). Score-extraction failures across all paid calls: **0**.
- **Headline (Amazon-Google hard, pair-F1, 600-pair subsample):** gpt-4o `llm_judge`
  **0.667** (P 0.54 / R 0.87 — SOTA band, beats free) > `embedding_cosine` 0.471 >
  GLM-5.2 `llm_judge` 0.409 (P 0.26 / R 0.90 — high-recall/low-precision, below free) >
  `weighted_average` 0.288 > `rapidfuzz` 0.271. On easy Fodors-Zagat, free embedding
  wins (pair-F1 0.816; pipeline BCubed 0.980 via `weighted_average`) and the GLM judge
  degenerates (0.233, P 0.13 / R 1.0).
- **Reusable primitive:** `evaluate_judge_on_candidates` + `JudgePairEval` in
  `core/benchmark.py` (blocking-free pair-level judge eval). Fixed a grading bug — it now
  restricts gold to candidate-realizable pairs so a subsample isn't penalised for gold
  pairs it never contained (was capping subsample recall artificially).
- **Deferred (M4):** the cascade/hybrid (needs a token-cost source fix + threshold
  calibration to the embedding-score distribution) and the frontier FZ pass.
- Harness `examples/m3_race.py` (resumable, per-cell-committed, budget-capped); results
  `data/benchmarks/m3/M3_RESULTS.md`; decision `docs/M3_DIRECTION_MEMO.md`. The finding
  reshapes M4 toward *making a precise judge cheap* rather than bolting on an LLM.

### M2: Walking skeleton end-to-end + baseline + artifact (Fodors-Zagat)

Wired the existing M0/M1 primitives into one deterministic, zero-spend Resolver
pipeline that reports a held-out BCubed baseline and proves the brainsquad
**artifact consumption contract** end-to-end. This is mechanics + serialization,
not Person-resolution quality (that is M5).

- **Pipeline (compose, no new components)**: `build_restaurant_resolver` wires
  the shared `VectorBlocker` (MiniLM + FAISS-cosine, `k=5`) with the missing-aware
  `Comparator.from_schema(RestaurantSchema)` (auto-excludes `id`, computed
  `embed_text`, and the `source` Literal — comparing `source` would penalise the
  all-cross-source true matches), the registered zero-spend `WeightedAverageJudge`,
  and a connected-components `Clusterer`. `split_restaurant_corpus` is a
  leakage-free stratified split over full records; the threshold is tuned on TRAIN
  only and BCubed is scored on the held-out TEST split against the dataset's TRUE
  `perfectMapping` closed-world partition (NOT the M1 teacher labels). The
  predicted partition is singleton-completed before scoring.
- **Measured baseline (seed=0, test_size=0.3, threshold tuned to 0.8)**: held-out
  TEST BCubed **P/R/F1 = 0.991 / 0.969 / 0.980** vs the merge-nothing sanity floor
  **F1 = 0.932** (Fodors-Zagat is singleton-dominated, so the floor is high by
  construction) — i.e. ~5 honest points of signal over "every record is unique".
  Blocking **Pair-Completeness = 1.0** on the test split (it caps recall) — but
  this is **seed-dependent**, not a system property: the blocker's full-corpus
  Pair-Completeness is **0.9911** (one structurally-missed pair, `f640`/`z325`,
  identical `embed_text` in both sources), and with `seed=0` that pair lands in
  the *train* split, so the *test* split sees 1.0. A different seed would show
  ~0.971 on test. The slow CI gate pins F1 ≥ 0.95 as an informational regression
  floor, not a quality bar — M3 is what improves the baseline. NOTE: BCubed on a
  singleton-heavy corpus over-weights trivially-correct singletons; M3 reports
  **pairwise F1 on true matches** as the honest complement.
- **Artifact contract = `resolve()`-only**: `resolver.save(<dir>)` writes the
  artifact **directory** (a `resolver.json` manifest + FAISS sidecar; no pickle,
  no code execution) and `Resolver.load(<dir>).resolve(records)` is the entire
  consumer call (`records: list[dict]` → `list[set[str]]` of multi-record
  clusters). A fresh-process identity test imports `langres.data.er_benchmarks`
  (which now registers `RestaurantSchema` at import time), reloads the artifact in
  a subprocess, and asserts clusters identical to the in-process run. Bad-input
  contract: empty corpus → `[]`; a record missing a required field → pydantic
  `ValidationError` naming the field (before any embedding). The copy-paste
  consumption snippet lives in `docs/DX_RESOLVER.md`.
- **Deferred to M5**: `Resolver.link()` / `Resolver.stream_against()` (incremental
  linking against a saved corpus) remain `NotImplementedError` stubs and are **not**
  part of the M2 contract — batch dedup via `resolve()` is.

### M1: Cold-start gold-set bootstrapping (LLM-teacher)

Reusable, entity-type-agnostic `langres.bootstrap` package that mines hard-negative
candidate pairs from a blocker, labels them with a budget-capped LLM teacher, and
emits a versioned gold set + coverage/calibration report. Validated on the
**Fodors-Zagat** restaurant benchmark (864 records / 112 cross-source matches).

- **Data contract + adapter**: `GoldPair`/`GoldSet` (versioned Pydantic, JSON
  save/load), `RestaurantSchema` (computed `embed_text`), `load_fodors_zagat`,
  and a blocking k-sweep that pins `DEFAULT_BLOCKING_K=5` (Pair-Completeness 0.9911).
- **Mining + labeling**: `HardNegativeMiner` (three-stratum similarity sampling),
  `TeacherLabeler` (hard $20 budget cap via pre-flight pair cap + per-pair token
  tally + blind-cost abort, `enable_langfuse=False` client), plus `GroundTruth`/`Fake`
  labelers for deterministic, zero-spend CI runs.
- **Metrics + report**: added `cohens_kappa`, `matthews_corrcoef`, `brier_score`,
  `expected_calibration_error` (equal-mass bins), `reliability_bins` to `core.metrics`;
  `BootstrapReport` covers Pair-Completeness, teacher-vs-truth agreement (F1/kappa/MCC),
  calibration (Brier/ECE of P(match) vs is-match), and an agreement-convergence curve.
- **`Bootstrapper`** orchestrator wires blocker → cross-source filter → miner → labeler
  → gold set + report; deterministic real-embedding example + slow CI test.
- **EXIT (real GLM-5.2 teacher run, $1.28)**: 1382-pair gold set committed at
  `data/gold_sets/fodors_zagat/`; Pair-Completeness 0.9911. Teacher-vs-truth over
  the **full** cross-source band (n=1382, closed-world truth): F1 0.446,
  precision 0.288, recall 0.991, kappa 0.368, MCC 0.472; calibration Brier 0.194 /
  ECE 0.195. **Finding**: the raw GLM teacher is high-recall / low-precision and
  overconfident on this band (its 0.999-confidence bin is only ~27% true matches),
  so the report does its job — surfacing that raw teacher labels need a
  precision-raising step (threshold/secondary review) before use as final gold.
  (An earlier draft scored only the 213 pairs whose records both appeared in a
  match cluster, which hid the teacher's false positives and inflated F1 to 0.873;
  the loader now returns the complete closed-world partition so every cross-source
  pair is evaluated.)

### Component Inspection Methods (Progressive Pipeline Building)

**Added exploratory analysis capabilities to core components** - enables parameter tuning WITHOUT ground truth labels:

- **Report Models** (`langres.core.reports`):
  - `CandidateInspectionReport`: Statistics and examples for blocker output
  - `ScoreInspectionReport`: Score distribution analysis for module output
  - `ClusterInspectionReport`: Cluster size distribution and singleton analysis
  - All reports support `.to_markdown()`, `.to_dict()`, and `.stats` property

- **Inspection Methods**:
  - `Blocker.inspect_candidates(candidates, entities, sample_size)`: Explore candidate generation without labels
    - Implemented in `VectorBlocker` with k_neighbors tuning recommendations
  - `Module.inspect_scores(judgements, sample_size)`: Analyze score distributions without labels
    - Implemented in `LLMJudgeModule`, `RapidfuzzModule`, and `CascadeModule`
    - Includes threshold recommendations based on distribution
  - `Clusterer.inspect_clusters(clusters, entities, sample_size)`: Review clustering results without labels
    - Singleton rate analysis and threshold tuning recommendations

- **Example**: exploratory inspect → tune → re-inspect → iterate workflow
  - Demonstrates parameter calibration without expensive labeling
  - All three inspection methods (the standalone `progressive_pipeline_building.py`
    example was removed in M0, superseded by `examples/resolver_company_dedup.py`)

**Key Benefits**:
- **Progressive discovery**: Build pipelines incrementally with feedback at each stage
- **Label-free exploration**: Understand pipeline behavior before expensive labeling
- **Actionable recommendations**: Rule-based parameter tuning suggestions
- **Human + AI readable**: Markdown reports for humans, JSON for agents
- **Type-safe**: Full mypy strict mode compliance with generic SchemaT support
