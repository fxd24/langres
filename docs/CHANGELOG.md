# Changelog

## [Unreleased] — token-usage vector + LLM-judge paper-prompt seams

### ⚠️ Behavior changes

- **`LLMJudge` no longer silently returns `0.5` when it cannot parse a score.**
  The default `response_parser` now *abstains* on an unparseable response: the
  judgement carries `provenance["parse_error"] = True` with `score=0.0` (a
  flagged abstention, distinguishable downstream) instead of a plausible-looking
  mid-confidence `0.5`. `on_parse_error="raise"` turns the same case into an
  immediate `LLMParseError`. The default is `"abstain"` because aborting a long
  paid run on one flaky response is worse than a surfaced, counted abstention —
  and `evaluate()` / `evaluate_judge_on_candidates()` now expose the count as
  `JudgePairEval.n_parse_errors` and warn loudly when it is non-zero.
- **`LLMJudge` default `temperature` changed `1.0` → `0.0`** (deterministic;
  the ER-paper convention, and already the `DSPyJudge` default). Pass
  `temperature=1.0` to restore the old behavior.
- **`LLMJudge.prompt_template` now requires literal `{left}` and `{right}`
  placeholders** (validated at construction) and substitutes them by literal
  replacement rather than `str.format`, so a template containing other braces
  (e.g. a paper's JSON output schema `{"match": true}`) works instead of raising
  `KeyError`.
- **`JudgementLog` schema `"v"` bumped `1` → `2`:** the default (privacy-safe,
  `features=False`) row gained a non-PII `usage` token vector (`null` for
  non-LLM judges). Old `v: 1` rows still read back unchanged.

### Added

- **`langres.core.usage.LLMUsage`** — a frozen Pydantic token-usage vector in the
  OpenTelemetry GenAI vocabulary (snake_case, SUBSET semantics): `input_tokens`
  and `output_tokens` (inclusive totals) with `cache_read_input_tokens`,
  `cache_creation_input_tokens`, `reasoning_tokens` as subsets, plus the serving
  `provider` and `model`. Import-light (pydantic only) so a future pricing layer
  can consume it without core's heavy deps. `LLMJudge` / `DSPyJudge` / `SelectJudge`
  now record it under `provenance["usage"]` (additive — legacy
  `prompt_tokens`/`completion_tokens` unchanged). Pinned against LiteLLM's
  Anthropic normalization (`usage.prompt_tokens` is already the inclusive input
  total, so the cache subsets are never double-counted).
- **`LLMJudge` paper-replication seams** (first-class constructor params, no
  subclass fork): `response_parser` (default `parse_score_response`; shipped
  reusable `parse_binary_yes_no` for the Yes/No prompt family), `record_serializer`
  (default `default_record_serializer`), `system_prompt`, and `on_parse_error`.
  All exported from `langres.core.modules.llm_judge` (`ParsedVerdict`,
  `LLMParseError`, the two parsers, `default_record_serializer`).

## [0.2.0] - 2026-07-06 — the closed flywheel loop

### ⚠️ BREAKING

- **`judge="auto"` (the default for `link`/`dedupe`) now RAISES `NoJudgeAvailableError`
  when no LLM API key is set, instead of silently falling back to fuzzy string
  matching.** Unsupervised string matching over-merges on unlabeled data (in the
  motivating demo it collapsed five distinct entities into one cluster with no
  error), so the library refuses rather than hand back a confidently-wrong answer.
  The unpinned-model-price branch raises the same error.
- **`fallback_reason` removed** from `DedupeResult` / `LinkVerdict` / `ResolvedModule`
  / `ResolvedJudge` — no path could set it after fail-fast, and an always-`None`
  field is anti-self-describing.

  **Migration:**
  - Keyless callers: pass `judge="string"` explicitly to opt into offline fuzzy
    matching (lower quality; pair it with `derive_threshold` on labeled data).
  - Keyed default path: install the `[llm]` extra (`uv sync --extra llm` /
    `pip install 'langres[llm]'`) **and** export `OPENROUTER_API_KEY`; the run is
    spend-capped at `$1` by default (`budget_usd=`).
  - Catch `NoJudgeAvailableError` (now root-exported from `langres`, alongside
    `BudgetExceeded`) on the front door.
  - Replace any `result.fallback_reason` reads with `result.judge_used` /
    `result.score_type` plus the auto-path selection notice.

### Added — the flywheel closed loop (bootstrap → log → review → harvest → train → cascade)

- **`select_for_review` + `ReviewQueue`** (`langres.core.review`, root-exported) —
  pick the judged pairs most worth a human's attention: `uncertainty` (near the
  threshold), `disagreement` (two logs differ), and first-class `audit` (a seeded
  governance sample that catches confident false merges). Snapshot-semantics queue.
- **`langres` CLI** (`langres.cli`) — `review` (terminal y/n/s/q labeler, resumable),
  `export-csv` / `import-csv` (spreadsheet round-trip, the primary review path),
  `--version`. Formula-injection + terminal-control-char hardened; fully stream-injectable.
- **`CascadeJudge`** (`cascade_judge`) — a cheap student everywhere, escalation only
  inside a `(low, high)` band; escalated provenance preserves `cost_usd`/`model`;
  serializes a fitted student through `Resolver.save`/`load`. (Old `CascadeModule`
  deprecated.)
- **Silver-only calibration guard** — `derive_threshold_from_pairs` warns when every
  pair is a judge verdict (circular); overlay human corrections first.
- **`examples/flywheel_closed_loop.py`** — the whole loop end to end at **$0** on a
  committed Fodors-Zagat fixture, with a data-derived escalation band and an honest
  "plumbing not economics" report.

Docs (`docs/GETTING_STARTED.md` + the doc-ladder rewire) are detailed under
[Unreleased] below.

## [Unreleased] - POC Phase

- Designed two-layer API architecture and POC validation plan (3 approaches: classical, semantic, LLM hybrid)
- Implemented core primitives (`Module`, `Blocker`, `Clusterer`) with Pydantic data contracts and 100% test coverage
- Completed Approach 1 (classical baseline): `AllPairsBlocker` + `RapidfuzzModule` end-to-end pipeline

### Experiment tracking & observability — run store, `ExperimentTracker`, LLM trace correlation

The missing spine under langres's otherwise-**ephemeral** benchmark runs (rich results
were printed, then lost — no run id, no config/data snapshot, no cross-run compare):
content-addressed run identity, JSONL persistence, a pluggable tracker seam, and
end-to-end trace correlation — **dependency-free** on the core path (`import langres`
still pulls no `mlflow`/`wandb`).

- **Run store + identity** (`langres.core.runs`, root-exported) — `RunContext` (the
  recipe) + `RunRecord` (recipe + outcomes) with a content-addressed `recipe_id`
  (`sha256` over config/data/seeds, *excluding* code/env provenance so a dirty tree or
  `uv.lock` bump keeps the id) and an `attempt_id` PK; an `fcntl.flock`-guarded,
  append-only JSONL `RunStore` (`read()` collapses `running`+terminal lines
  last-wins-by-`attempt_id`); and `capture_run(context, *, store=None,
  tracker=NoOpTracker())` — writes a `running` line at start and a terminal line on
  exit, and sets the `current_run` contextvar. **`store=None` writes nothing.**
- **`ExperimentTracker` Protocol + adapters** (`langres.core.trackers`) — an
  Accelerate-style seam (`NoOpTracker` null default, `MultiTracker` fan-out to run
  MLflow *and* W&B at once, `resolve_tracker` dispatch) with lazy **MLflow** and **W&B**
  adapters behind the `[mlflow]` / `[wandb]` extras (a missing extra raises a helpful
  `pip install 'langres[<backend>]'` `ImportError`). MLflow defaults to a local file
  store out of the box; W&B supports keyless `offline`/`disabled` runs for CI/no-key use.
- **LLM trace correlation** — `capture_run` sets `current_run`; `JudgementLog` records
  the active `run_id`, and `LLMJudge` injects litellm `metadata` (`langres_attempt_id`
  + pair ids + decision step) on **both** the sync (`forward`) and async
  (`forward_async`) paths, so a Langfuse/OTel trace joins the `RunRecord` and
  `JudgementLog`. Off a run (or a non-litellm client) the calls stay byte-identical
  (no `metadata`).
- **DSPy compile lineage** — `DSPyJudge.compile(...)` records the compilation as a
  first-class optimization run via `capture_run` and stamps `_compile_run_id`, so a
  later eval run threads it into `parent_run_id` (compile → eval lineage).
- **`Settings`** — `RUN_STORE_PATH`, `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT` (the
  MLflow ones consumed by `MlflowTracker`; a zero-config default `store` from
  `RUN_STORE_PATH` is deferred to the benchmark wrap — pass `store=` explicitly today).
  Docs: `docs/EXPERIMENTS.md`; runnable zero-spend
  `examples/research/experiment_tracking_demo.py`.

### Flywheel closed loop — `docs/GETTING_STARTED.md` + doc-ladder rewire

The one entry doc for the closed flywheel (fail-fast `auto`, `select_for_review`
/ `ReviewQueue`, the `langres` review CLI, `CascadeJudge`, and
`examples/flywheel_closed_loop.py`), telling the lifecycle end to end.

- **`docs/GETTING_STARTED.md`** (new, "start here") — the flywheel at altitude:
  LLM bootstrap under a cap (fail-fast `"auto"`: bring an LLM or explicitly opt
  into `judge="string"`) → log from day 1 → review at the margin
  (`select_for_review` + `langres review` / CSV round-trip) → harvest silver +
  gold (with the circularity caveat) → train a cheap trainable student
  (RandomForestJudge, Magellan-style — **not** LLM distillation) + `derive_threshold` →
  `CascadeJudge` → `Resolver.save`/`load`. Every step carries a runnable snippet
  inline; two explicit lanes (keyless `judge="string"` / keyed default `auto`);
  a competitive-positioning section (vs dedupe/Zingg/Splink, "use Splink when… /
  use langres when…"); the audit slice as the governance/trust mechanism; the
  ids-only review mode as the PII privacy posture; and two flywheel operating
  notes (stable ids; one log per run, or dedupe rows before harvest).
- **Snippet-rot guard** (`tests/docs/test_getting_started_snippets.py`) — runs
  the guide's first keyless snippet verbatim at **$0** (hard-refuses any
  non-`judge="string"` block, so a paid snippet can never slip in).
- **Doc-ladder rewire** — `README.md` links GETTING_STARTED first as "start
  here" (+ a quickstart pointer); `docs/TUTORIAL_YOUR_OWN_CSV.md` gains it as
  the big-picture rung and in the calibration tease; `examples/README.md`
  Start-here tier gains `flywheel_closed_loop.py`.

### Wave 3 (W3): experiment DX — docs, the paid-smoke harness + result, examples curation

Making the seam usable and the one substantive paid claim measured. Everything is
zero-spend except the single ≤$10 smoke.

- **DX docs (#75):** three newcomer guides — `docs/ADDING_A_METHOD.md` (register a
  method behind the seam), `docs/TESTING_AT_ZERO_COST.md` (the DummyLM / `budget=0.0`
  zero-spend test surface), `docs/TUTORIAL_YOUR_OWN_CSV.md` (bring-your-own-CSV
  walkthrough).
- **Paid-smoke harness (#76):** `examples/research/w3_paid_smoke.py` — one
  SpendMonitor-capped operator run (hard $10 ceiling) that measures set-wise
  `SelectJudge` vs pairwise on the SAME real model on Amazon-Google, plus the paid
  verb surface (`link`/`dedupe`, a single group call, the signal log). Verified at $0
  with DummyLM in `tests/examples/test_w3_paid_smoke.py`; the cap has a proven
  fires-with-partials test (`BudgetExceeded.partial_judgements`).
- **Paid-smoke result — $4.65 / $10** (`data/benchmarks/w3/w3_smoke_results_*.json` +
  `docs/research/20260703_w3_paid_smoke_results.md`). **Set-wise quality is
  model-dependent, not a clean win:** pairwise wins on gpt-4o-mini (pair-F1 **0.688 vs
  0.620**, −0.068 set-wise) but set-wise wins on gpt-4o (**0.667 vs 0.618**, +0.049) —
  the ComEM Select direction on a strong judge, but **not** its published +16 F1
  magnitude. Set-wise makes 3–5× fewer LLM calls but costs more dollars (token-heavy
  group prompts). The honest U4 "measure before believing the claim" outcome. Other
  deliverables (gpt-4o-mini): `link` match score 0.95; `dedupe` 1 cluster; one group
  call judged 22 members for $0.011; 4-row signal log; verb cost $0.0018.
- **Examples curation (#77):** examples split into a **newcomer tier** kept at
  `examples/` (`quickstart_verbs.py`, `person_resolution.py`, `incremental_assign.py`,
  `canonicalizer_enrichment.py`, `flywheel_threshold_harvest.py`, …) and a **research
  tier** moved to `examples/research/` (the `m3_*` / `m4_*` / `w1_*` / `w2_*` benchmark
  harnesses); doc references updated (`docs/EXPERIMENTS.md`). Adds run-as-a-newcomer DX
  numbers to `docs/FRICTION_LOG.md` — `import langres` **~0.2 s** (lazy heavy imports),
  TTHW **~2.5 s**, cold install **2.3 s** core / **6.8 s** `[semantic]`, all inside
  budget at $0.

### M5 (W2.3): golden records — `Canonicalizer` (survivorship + the enrichment loop)

The Master Data Creation exit (UC4): merge one entity's records into a single
**golden record**, and enrich it as new mentions link in.

- **`Canonicalizer`** (`src/langres/core/canonicalizer.py`) — a thin, composable,
  config-serializable unit. `canonicalize(records) -> golden dict` resolves each
  field independently with a named **survivorship strategy**: `most_complete`
  (default — value from the richest source record), `longest`, `most_frequent`,
  `most_recent` (needs a `timestamp_field`), `first`/`source_priority`, all
  per-field overridable. Dict-in/dict-out (the shape `resolve`/`assign`/
  `AnchorStore` already use); `id` is stamped as the master id, never merged; ties
  break deterministically first-seen; `0`/`False` are present, `None`/`""` missing.
- **`enrich(golden, mention)`** — the enrichment loop: fold a newly-linked sparse
  mention (from `Resolver.assign`) into an existing golden record via the *same*
  survivorship path, filling fields the golden record lacked. Not a parallel code
  path — it is `canonicalize([golden, mention])` with the master id preserved.
- **`save`/`load`** via the config-registry artifact seam (no pickle):
  `canonicalizer.json` carries version + `type_name` + the strategy config.
- End-to-end example (`examples/canonicalizer_enrichment.py`) + tests: a sparse
  mention (name + website) links to an anchored entity, then canonicalization
  fills the `website` the rich anchors never had (golden completeness 3 → 4).
  Per-strategy correctness, edge cases, and a fresh-subprocess config round-trip;
  100% coverage.

### M5 (W2.4): the data flywheel's harvest — verdicts + corrections → labeled pairs → threshold

The harvest half of the flywheel. `JudgementLog` (W0.2) is the inlet; this turns its
logged verdicts, plus human corrections, into labeled pairs that recalibrate a threshold.

- **`langres.core.harvest`** — the outlet, eval/calibration-tier and import-light
  (Pydantic only; scikit-learn stays lazy so the contract models never pull a heavy dep):
  - **`Correction`** — the `corrections.jsonl` line contract an external review queue
    (e.g. brainsquad) writes: `left_id`/`right_id`/`label` required, `"v":1`, plus optional
    `original_score`/`original_verdict`/`reviewer`/`timestamp` audit context.
  - **`CorrectionLog`** — reference JSONL reader/writer, mirroring `JudgementLog`.
  - **`harvest_labeled_pairs(rows, corrections)`** — one `LabeledPair` per judgement row;
    label = logged `verdict` (weak) unless a correction overrides it (matched
    order-independently by id set), with `source` recording the provenance.
  - **`derive_threshold_from_pairs(pairs)`** — `derive_threshold`'s first production caller.
- **`examples/flywheel_threshold_harvest.py`** (D9) + committed Fodors-Zagat fixtures
  (`examples/data/flywheel/`, built at $0 by `generate_fixtures.py`): derives the threshold
  before vs. after 40 corrections and scores both on a **held-out gold** split. Exit criterion
  met — held-out pair-F1 moves 0.558 → 0.708 (+0.150) in the correct direction (precision
  0.39 → 0.56 at held recall), proven on gold the threshold was never fit on. 100% coverage.

### M5 (W2.2): incremental single-record assignment — `AnchorStore` + `Resolver.assign`

The incremental-linking exit (S6): after a batch `resolve()`, answer "here is one NEW
record — which existing entity, or new?" with a **stable** entity id.

- **`AnchorStore`** (`src/langres/core/anchor_store.py`) — a serializable, composable unit
  around a `Resolver`. `AnchorStore.build(resolver, records)` runs a dedicated pass that
  mints a stable, monotonic entity id for **every** record, including the singletons
  `resolve()` drops (clusterer-agnostic). `save`/`load` via the config-registry artifact
  seam (no pickle), delegating the pipeline to `Resolver.save`/`load`.
- **`Resolver.build_anchor_store(records)` + `Resolver.assign(record) -> ClusterDelta`** —
  thin sugar; the reserved cross-source `link`/`stream_against` stubs stay untouched.
  `assign` reuses the vector index single-record kNN (with `similarity_score` + `query_prompt`,
  so `EmbeddingScoreJudge` works incrementally) or all-pairs, and the same Comparator + Module
  judge. Append-only allocator (idempotent per record id); `CompositeBlocker` supported.
- **`ClusterDelta`** — `new` / `link`, with `merge`/`split`/`reject` reserved in the enum so
  the contract stays stable for W2.4/M6.
- Committed-data (Fodors-Zagat) + fresh-subprocess save/load round-trip tests; 100% coverage.
  See `examples/incremental_assign.py`.

### M5 (W2.1): a second entity type, config-only — Person via FEBRL4

The Generalise exit: langres resolves a **person** with **zero new core code** — config
only, the same way a user would add a dataset.

- **Dataset + adapter (#70):** a FEBRL4 Person subset fixture
  (`src/langres/data/datasets/febrl_person/`, 500/side, 500 cross-source matches) + one
  `src/langres/data/febrl_person.py` adapter (`FebrlPersonSchema` / `load_febrl_person`
  / `FebrlPersonBenchmark`), the exact shape of the Fodors-Zagat / Amazon-Google /
  Abt-Buy adapters. **Nothing under `src/langres/core/` changed.** (FEBRL4 is BSD-3
  synthetic, Apache-2.0-compatible; OpenSanctions was CC-BY-NC and could not ship —
  see the dataset `SOURCE.md`.)
- **Measured at $0** (five free local methods raced on the identical blocked candidate
  set, `k=20`): supervised `random_forest` tops pairwise **F1 0.964** (P 0.954 /
  R 0.973); string judges hit **BCubed F1 0.998** at the pipeline level;
  `fellegi_sunter` is high-recall/low-precision (R 1.0 / P 0.75), consistent with the
  W1.2 trained-family finding. Blocking is the recall ceiling (~0.98 Pair-Completeness
  at the cross-platform-honest `k=20` pin).
- **Example + results:** `examples/research/w2_person_benchmark.py`,
  `docs/research/20260703_w2_person_benchmark_results.md`. 100% coverage.

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
  extracted from `examples/research/m3_race.py`).
- **Proven end-to-end at $0** (DummyLM): a compiled `DSPyJudge` → `compile` →
  `evaluate_judge_on_candidates` (judged-once, pairwise-F1, SOTA-comparable) — the right
  surface for a compiled/paid judge; `run_methods` is the cheap-method race. Getting
  started: `docs/EXPERIMENTS.md`, `examples/research/m4_experiment_loop.py`.
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
  `examples/research/m4_race.py` (resumable, per-cell-committed, budget-capped).
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
- Harness `examples/research/m3_race.py` (resumable, per-cell-committed, budget-capped); results
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
