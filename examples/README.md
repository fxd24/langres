# langres Examples

Example scripts demonstrating how to use the langres library, split into two
tiers:

- **Start here** — a clean progression for newcomers, top-level in `examples/`.
- **Research & benchmark reproductions** (`examples/research/`) — the
  milestone/benchmark/internal scripts used to validate the POC and later
  milestones. Not part of the newcomer path; kept for reproducibility.

## Setup

```bash
# Copy the example file and add any API keys you plan to use
cp .env.example .env
```

All environment variables are optional — most examples below run at **$0**
with no key required. Scripts that spend money say so explicitly in their
docstring and are gated behind an explicit flag or a hard budget cap.

## Start here

Run in this order for the intended newcomer progression — models quickstart →
dedupe → resolver → incremental assign → golden record → signal log →
flywheel harvest → closed-loop flywheel → person:

- **`quickstart_models.py`** — dedupe records with zero labels in a handful of
  lines by naming a model: `FuzzyString`, the $0, offline, no-key architecture.
- **`research_recipes.py`** — the research vocabulary in one zero-network
  example: Retrieve → Rerank → LLM → Cluster, with one reusable resource per
  model slot and no blocker/matcher-specific wrappers.
- **`basic_usage.py`** — the foundational data contracts (`CompanySchema`,
  `ERCandidate`, `PairwiseJudgement`).
- **`deduplication_example.py`** — end-to-end entity resolution on a
  real-world dataset of Swiss funder organization names.
- **`resolver_company_dedup.py`** — the `Resolver` north-star: company
  deduplication with `save`/`load`.
- **`incremental_assign.py`** — `AnchorStore` for incremental single-record
  assignment: "does this new record belong to an existing entity, or is it new?"
- **`canonicalizer_enrichment.py`** — the enrichment loop: a sparse mention
  links to an entity and enriches its golden record ($0, deterministic).
- **`judgement_log_demo.py`** — the flywheel inlet: log every judge call
  (`log=` on a model's `.dedupe()`/`.compare()`), then read the log back.
- **`flywheel_threshold_harvest.py`** — the flywheel outlet: logged verdicts +
  human corrections feed `derive_threshold` to re-calibrate a decision threshold.
- **`flywheel_closed_loop.py`** — the whole loop closed end to end at **$0**:
  bootstrap → select_for_review → harvest → train a cheap RandomForestMatcher student →
  `CascadeMatcher` (cheap everywhere, escalate only the margin) → report. The
  runnable twin of [`docs/GETTING_STARTED.md`](../docs/GETTING_STARTED.md)
  (needs the `[trained]` extra).
- **`person_resolution.py`** — the embeddings + LLM "strong path" on a second
  entity type: semantic blocking (MiniLM + FAISS) feeding an LLM judge.
- **`finetune_capstone.py`** — the training-surface capstone: **train your own
  matcher** end to end — fine-tune SmolLM2-135M with LoRA on a real benchmark
  slice, serve the weightless `model_ref` in-process, and evaluate held-out F1,
  reporting the honest cost in GPU-seconds. A REAL (small) fine-tune, slow on
  CPU/MPS; needs the `[finetune,semantic,llm]` extras (see its docstring).
- **`quickstart_mining.py`** — the data-prep substrate that *feeds* training:
  mine AnyMatch hard positives, balance negatives 2:1, attribute-augment, and
  confident-learning denoise a labeled pair set, then print a
  `MiningReadinessSection`. Offline and $0; uses the `[trained]` extra (sklearn)
  for the out-of-fold RandomForest behind the featurizing miners.
- **`recipe_lift_proof.py`** — the data-prep payoff: on abt_buy's all-pairs
  blocked pool a curated training recipe (denoise + hard positives + attribute
  augmentation + 2:1 balance) beats the mean of equal-budget random draws on
  held-out pair F1. Prints baseline vs recipe F1 + margin. Offline and $0
  (`RandomForestMatcher`); self-contained OpenMP guard.
- **`curation_loop.py`** — the failure-mode flywheel end to end: run a matcher,
  log its judgements, build a `FailureModeSection` to find the failing slice,
  mine/augment that slice with the Wave-A miners, retrain, and confirm the
  targeted slice's error rate dropped. Offline and $0; self-contained OpenMP
  guard.

## Example Data

The `data/` directory contains sample datasets used by the examples above and
by several research scripts:
- `companies.json` / `companies_labels.json` — funder organization names + ground truth
- `funder_names_with_ids.json` / `funder_name_deduplicated_groups.json` — labeled funder dedup data
- `flywheel/` — fixtures for the judgement-log harvest demo

## Research & benchmark reproductions (`examples/research/`)

Milestone exit-criteria runs, benchmark harnesses, and exploratory demos.
Several spend real money (OpenRouter) under an explicit hard budget cap — read
each script's docstring before running it. Companion `*_results*.json` /
`*_output.md` files are committed snapshots of past runs.

- **`m1_bootstrap_fodors_zagat.py`** — M1 cold-start bootstrapping end-to-end
  on Fodors-Zagat, deterministic and free.
- **`run_m1_gold_set.py`** — M1's paid EXIT run: labels the Fodors-Zagat
  cross-source band with a real budget-capped GLM teacher.
- **`m2_walking_skeleton_fodors_zagat.py`** — M2's held-out BCubed baseline on
  Fodors-Zagat, zero spend.
- **`m3_race.py`** — M3's paid, resumable multi-method benchmark race (the EXIT run).
- **`m3_zero_spend_race.py`** — the full M3 benchmark harness end-to-end with
  NO LLM spend (the pre-flight gate before `m3_race.py`).
- **`m3_regrade_subsample.py`** — re-grades committed AG subsample cells
  against in-scope gold with no new LLM calls.
- **`m3_report.py`** — renders the M3 race comparison table from committed
  per-cell JSON results.
- **`m4_dspy_judge.py`** — M4 `DSPyMatcher` smoke: Signature → ChainOfThought →
  compile → forward → eval → save/load, at $0 with `DummyLM`.
- **`m4_calibration.py`** — a data-driven threshold beats a hand-set one on AG, at $0.
- **`m4_experiment_loop.py`** — the DSPy-judge experimentation DX loop, end-to-end at $0.
- **`m4_race.py`** — M4's paid, resumable DSPy-scorer benchmark on Amazon-Google.
- **`w1_blocking_algebra.py`** — W1.3 blocking-algebra + clusterer benchmark
  (Fodors-Zagat + Amazon-Google), $0.
- **`w1_select_judge_benchmark.py`** — W1.1 `SelectMatcher` call-count + honest-cost
  reduction benchmark on Amazon-Google.
- **`w1_trained_family_race.py`** — W1.2 trained-family replication
  (`FellegiSunterMatcher` + `RandomForestMatcher`), $0.
- **`w2_person_benchmark.py`** — M5 W2.1: a second entity type (person)
  resolved config-only, at $0.
- **`phase1_blocker_optimization.py`** — POC Phase 1: blocker optimization evaluation.
- **`phase2_full_pipeline.py`** — POC Phase 2: full pipeline evaluation
  (VectorBlocker → LLMMatcher → Clusterer) against the POC success criteria.
- **`deduplication_with_blocker_optimization.py`** — `BlockerOptimizer` +
  Optuna + wandb hyperparameter tuning example (requires Azure/wandb/Langfuse env vars).
- **`deduplication_cached_faiss_simple.py`** — `DiskCachedEmbedder` speedup
  demo with FAISS.
- **`cached_embedder_demo.py`** — `DiskCachedEmbedder` persistent embedding-cache demo.
- **`compare_embedders_for_funders.py`** — compares sentence-transformer
  embedding models for funder-name deduplication.
- **`instruction_embeddings_demo.py`** — instruction-prefixed query embeddings with FAISS search.
- **`blocker_evaluation_demo.py`** — the blocker evaluation architecture, introductory demo.
- **`blocker_evaluation_comprehensive.py`** — comprehensive blocker evaluation
  across configurations.
- **`blocking_evaluation_faiss_vs_qdrant.py`** — FAISS (dense-only) vs Qdrant
  (dense + sparse hybrid) candidate generation comparison.
- **`blocking_evaluation_with_reranking.py`** — FAISS vs Qdrant-hybrid vs
  Qdrant-reranking candidate generation comparison.
- **`blocking_evaluation_with_instructions.py`** — FAISS vs Qdrant-hybrid vs
  CrossEncoder candidate generation, with caching + instruction prefixes.
- **`blocking_evaluation_comprehensive_comparison.py`** — 5-way blocking
  approach, model, and architecture comparison.
- **`debug_pipeline.py`** — debugging an entity resolution pipeline with `PipelineDebugger`.

## Troubleshooting

If you see errors like "environment variable is required":
1. Make sure you've created a `.env` file (copy from `.env.example`).
2. Add the required API keys for the services you're using (Azure/OpenRouter/wandb/Langfuse).
3. Most examples in **Start here** need no keys at all — if one fails on a
   missing key, you likely picked a script from `examples/research/` that needs one.
