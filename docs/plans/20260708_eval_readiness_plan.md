# Evaluation Readiness Plan (reframed)

Date: 2026-07-08
Integration branch: `feat/eval-readiness`
Supersedes: `docs/plans/20260707_research_readiness_eval_plan.md` (Codex draft — greenfield-framed)

## Why this reframe

The Codex draft was written as if the evaluation layer were greenfield ("add a
`langres.core.metrics` module", "add an evaluation harness"). A verification pass
against the actual repo shows **~70–80% of that layer already exists** and is solid.
Building to the original plan would **re-implement working code** — the exact reuse
trap `.claude/rules/component-design.md` and personal-rule #6 warn against.

This plan is therefore **audit-then-fill-the-gaps**, anchored to the authoritative
delta list in `docs/research/20260701_er_seam_audit.md` (§7 metrics/datasets, and
deltas **C1**, **C2**, **C7**) rather than the from-scratch framing.

Scope for this wave: **P0 metrics + P1 benchmarks + P2 harness slicing.**
Hard-case mining (the Codex draft's P3) is the *bridge to training* and is deferred
to its own later wave — and much of its surface already exists (see below).

## What already exists — REUSE, do not rebuild

Verified by two independent code sweeps (file:line).

**Metrics** — `src/langres/core/metrics.py` (39 KB, single reusable module; no metric
math duplicated into experiment scripts):
- BCubed P/R/F1 (`calculate_bcubed_*`), pairwise P/R/F1 + confusion counts
  (`calculate_pairwise_metrics`, `PairMetrics`, `classify_pairs`), cluster→pairs
  (`pairs_from_clusters`), pair PR curve (`pair_pr_curve`).
- Blocking eval (`evaluate_blocking` → `CandidateStats`): already returns
  **`candidate_recall` = Pair Completeness**, `candidate_precision`,
  `missed_matches_count` (FN), `false_positive_candidates_count` (FP). Ranking
  variant `evaluate_blocking_with_ranking` (MAP/MRR/NDCG/Recall@k via `ranx`).
- Agreement: `cohens_kappa`, `matthews_corrcoef`. Calibration: `brier_score`,
  `expected_calibration_error`, `reliability_bins`. Threshold pick:
  `core/calibration.py:derive_threshold`.

**Benchmarks** — `src/langres/data/`:
- `Benchmark` protocol (`core/benchmark.py:351`); `BlockingBenchmark` overlay
  (`methods.py:105`). Loaders + adapters for **Fodors-Zagat, Amazon-Google,
  Abt-Buy, FEBRL4 Person**, each vendored as tiny CSVs under
  `data/datasets/<name>/` with `ATTRIBUTION.md`, loaded via `importlib.resources`.
- `fixed_split_pair_benchmark.py:FixedSplitPairBenchmark` — a **dataset-agnostic**
  bridge that turns DeepMatcher/Magellan `(id_a, id_b, label)` train/valid/test
  rows into `ERCandidate`s (with comparison vectors) + aligned labels. This is the
  adapter the new literature datasets plug into.

**Harness** — `core/benchmark.py` + `methods.py` + `reports.py`:
- `run_method`/`run_methods`/`BenchmarkTable`, `evaluate_judge_on_candidates`,
  `evaluate_resolver_bcubed`, honest-vs-leaky pair eval
  (`evaluate_fixed_split_honest`), `BudgetedModuleRunner`, cost/latency tracks.
- Report models: `BlockerEvaluationReport`, `ScoreInspectionReport`,
  `ClusterInspectionReport`, `RecallCurveStats`, etc. Builders in `analysis.py`.

**Hard-case surface already present** (relevant to the deferred P3 wave):
`select_for_review` (uncertainty/disagreement/audit), `ReviewQueue`,
`harvest_labeled_pairs`, and **FP/FN extraction** already exist
(`analysis.py:extract_false_positives` / `extract_missed_matches`,
`diagnostics.py`).

## Genuine gaps — what this wave ADDS

### P0 — Metrics (pure-Python; must stay import-budget clean)
1. **Reduction Ratio** as a first-class blocking metric — `RR = 1 − C / P_all`,
   handling both dedup (`P_all = n(n−1)/2`) and cross-source linkage
   (`P_all = |A|·|B|`). Surface it on `CandidateStats`/`evaluate_blocking` alongside
   the existing `candidate_recall` (PC) so blocking is scored on **PC *and* RR**,
   not PC alone (seam audit §7).
2. **Generalized Merge Distance (GMD)** — Menestrina et al. 2010, cost-based
   merge/split-asymmetric partition distance; the slice-based algorithm. Add to
   `metrics.py` with tests against hand-computed small partitions. Complements the
   biased Pair-F1 / BCubed pair.

(Deliberately NOT adding a sklearn-style `classification_report` convenience —
low value, F1 is already available inline; simplicity rule.)

### P1 — Benchmark portfolio + registry
1. **Name-keyed benchmark registry** (`name → Benchmark`) mirroring the existing
   *method* registry, so `run_methods` / docs / CI can look benchmarks up by name.
2. **New loaders**, each a tiny vendored fixture under `data/datasets/<name>/` +
   loader module wired through `FixedSplitPairBenchmark`:
   - **WDC Products** — hard product matching + unseen-entity generalization slice.
   - **DBLP-ACM** — ~99 F1 ceiling, clean regression guard.
   - **Walmart-Amazon DIRTY** — dirty/missing-value case (exercises the
     missing-aware Comparator).
   - **OpenSanctions Pairs** — **metadata + published baselines + external-download
     adapter ONLY. NO bundled data** (OpenSanctions is CC-BY-NC, incompatible with
     langres's Apache-2.0 — same reason W2.1 chose FEBRL4). We race *against* its
     baselines (GPT-4o 98.95, DeepSeek-R1-Distill-14B 98.23), we do not vendor it.
3. **Pending:** a benchmark-discovery research pass (citation-following /
   Connected Papers / recent arXiv) may add datasets before the loaders are locked.

### P2 — Harness slicing + scaling curve
1. **Per-pair slice tags + sliced aggregation** (delta **C2**) in
   `evaluate_judge_on_candidates` / `FixedSplitPairBenchmark`, enabling WDC-style
   unseen-entity / corner-case slice reporting.
2. **Scaling-curve output** (delta **C1**) — power-law fit + extrapolation + band
   over a training-size sweep, emitted as a serializable report shape (extend
   `reports.py`, don't invent a parallel style).

## Out of scope (unchanged from Codex draft, plus)
- All training: DSPy compile improvements, embedding fine-tuning, QLoRA/LoRA.
- Hard-case mining wave (FP/FN mining productization, blocking-derived hard
  negatives S3, EL2N/difficulty S4, stratified sampling) — **next** wave.
- Automatic dataset downloads for bundled datasets; new paid benchmark runs.

## Acceptance criteria
1. `uv run pytest -m "not slow and not integration"` green; new code ≥95% coverage
   on the `core` contract (tiered policy, `.claude/rules/testing.md`).
2. Bare `import langres` still leaks none of torch/faiss/litellm/sentence_transformers/
   sklearn (`tests/test_import_budget.py`); RR/GMD are pure-Python.
3. Docs name the benchmark portfolio and *why* each is a first target (regression
   guard / textual-hard / dirty / unseen-entity / north-star).
4. A tiny fixture benchmark runs end-to-end producing pairwise + blocking (incl.
   **RR**) + clustering (incl. **GMD**) metrics.
5. Registry resolves every bundled benchmark by name.

## Execution — waves, branches, PRs

Orchestrated from the main session; implementation in **isolated worktrees**
(rule #5). Each wave → its own branch → PR into `feat/eval-readiness`
(integration). David reviews the integration branch; one final PR to `main`.

| Wave | Content | Depends on | Branch |
|---|---|---|---|
| **R** | Benchmark-discovery research (read-only) → finalize P1 list | — | (no branch; memo) |
| **A** | P0 metrics: Reduction Ratio + GMD + tests | — | `feat/eval-metrics-rr-gmd` |
| **B** | P1 registry (name→Benchmark) | A merged (touches nothing of A) | `feat/eval-benchmark-registry` |
| **C** | P1 loaders: WDC, DBLP-ACM, Walmart-Amazon + OpenSanctions metadata | R (list), B (registry) | `feat/eval-datasets-*` |
| **D** | P2 slice tags + sliced aggregation; scaling-curve report | A (metrics) | `feat/eval-harness-slicing` |

Waves that touch the same files (`metrics.py`, `data/__init__.py`, `benchmark.py`)
are sequenced, not parallelized, to avoid double-writing. R + A run first and in
parallel (R is read-only, A is metrics-only).

## Risks
| Risk | Mitigation |
|---|---|
| Re-implementing existing metrics | This plan is gap-only; each wave cites the existing symbols it extends. |
| GMD algorithm correctness | TDD against hand-computed partitions + published example. |
| Concurrent agents double-write shared files | Sequence waves by file overlap; one integration branch. |
| Bundling CC-BY-NC data | OpenSanctions is metadata/external-only, never vendored. |
| Import-budget regression | RR/GMD pure-Python; any heavy dep stays behind lazy submodule boundary. |
