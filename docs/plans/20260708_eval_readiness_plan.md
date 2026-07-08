# Evaluation Readiness Plan (review-hardened)

Date: 2026-07-08
Integration branch: `feat/eval-readiness`
Status: approved (David) + hardened by a 3-model plan review (Codex + two Claude lenses, 2026-07-08).

## Context

Prepare langres to reproduce/compare SOTA entity resolution by adding the
**evaluation instrument** — the metrics the field uses, a credible benchmark
portfolio, honest per-slice reporting — that must exist before any training.
The eval layer is **~70–80% already built** (`core/metrics.py`,
`core/benchmark.py` harness, `reports.py`, `review`/`harvest`); this wave is
**gap-only**. Vision anchor: langres = the composable ER seam (Retrieve → Judge
→ Resolve), swappable + serializable, whose edge is zero-shot-LLM-judge →
optional cheap distilled student, and which should make it **easy to build ER
systems**. Build the leanest instrument that unlocks the next experiment
(frontier null-baseline gate + set-wise judge), then stop.

Direction: **A — lean, vision-first**. Portfolio finalized by a citation/paper
research pass; design corrected by the review below.

## Portfolio (license-clean)

| Have | Add — bundle-able (full DeepMatcher split, CC-BY 4.0 via Leipzig/matchbench) | External-only | Deferred |
|---|---|---|---|
| FZ, Amazon-Google, Abt-Buy, FEBRL4 | **DBLP-ACM, DBLP-Scholar, Walmart-Amazon (structured), WDC Products** | OpenSanctions (metadata + baselines, CC-BY-NC → never vendored) | Beer/iTunes; scaling-curve C1; real-world person fixture (NCVR/GLEIF) |

## Review gate outcome (why the design below looks like it does)

Three independent reviewers converged on these corrections — all folded in:

1. **Registry must be a static, import-light manifest, not import-all discovery.** Every loader eagerly imports the `[semantic]` stack (`VectorBlocker`/`SentenceTransformer`/`FAISS`). Auto-importing all loaders would break the core-only install and `test_import_budget`, and kill all benchmarks if one loader fails to import.
2. **Dataset-namespaced schema class names are mandatory.** `register_schema_idempotent` *raises* on a name clash; DBLP-ACM/DBLP-Scholar share a schema shape and Walmart-Amazon collides with amazon_google's `ProductSchema`. Parallel agents can't see each other → guaranteed crash.
3. **Slices stay external — no `ERCandidate` / `FixedSplitPairBenchmark` contract change.** `build()` unpacks strict 3-tuples; a 4th tag element would break the shared adapter. Compute the slice via a `slice_fn` closure over a `pair_key→tag` map the WDC benchmark exposes.
4. **Honest per-slice thresholding.** One global best-F1 threshold, then grade each slice at that *fixed* cut. Per-slice argmax would fake the seen→unseen drop.
5. **The "make ER easy" user needs a BYO-data one-liner + the LLM-judge in the flagship example** — two north-star DX gaps the reproduce-SOTA framing missed.
6. **Prefer a generic loader factory over 4 × ~350-line verbatim copies** (≈1.4k duplicated lines = the bloat the seam audit §6 warns against). Cuts the next dataset to ~30 lines.

## Genuine gaps this wave fills

- **P0 metrics** — Reduction Ratio + Generalized Merge Distance. **DONE** on `feat/eval-metrics-rr-gmd` (PR #89): pure-stdlib, 100% cov, RR threaded onto `evaluate_blocking` with `n_left`/`n_right`/`num_records` (handles cross-source `|A|·|B|` correctly).
- **P1** — import-light benchmark **registry/manifest** + generic **loader factory** + 4 loaders + OpenSanctions metadata (non-loadable) + `list_methods()`.
- **P2** — external **slice tags + sliced aggregation** (C2), honest fixed-threshold grading, WDC seen/unseen. C1 scaling curve **deferred** (no consumer; gated behind training).
- **DX** — BYO-data `evaluate(...)` one-liner + "score your own data" tutorial; registry-driven `portfolio_race.py` incl. LLM-judge behind `--paid`.

## Key design decisions (final)

1. **Registry — `src/langres/data/registry.py` (import-light manifest).** A static `name → BenchmarkEntry{task, domain, loadable, module_path, loader_symbol}` map (lightweight metadata, no loader import). `list_benchmarks()` returns metadata without importing any loader; `get_benchmark(name)` imports only the selected module (actionable `pip install langres[semantic]` error on missing extra, like `core/registry.py`). Lives in `langres.data`, **off** `langres/__init__.py`'s eager path, **not** wired into `core.benchmark` (avoids the `core→data→core` cycle). Add `list_methods()` (surface `ALL_METHODS`). Register the existing 4 + the new entries. OpenSanctions entry is `loadable=False`; its `load()` raises an actionable "fetch manually" error.

2. **Loader factory — `src/langres/data/_deepmatcher_loader.py` (central, Wave B).** `make_deepmatcher_benchmark(schema, package, table_files, split_files, constants, ...)` → `(load_<x>, load_<x>_pair_splits, <X>Benchmark)`, reusing the six `_benchmark_utils.py` helpers. Per-dataset module becomes ~30–50 lines: the **dataset-namespaced** schema (`DblpAcmSchema`, `DblpScholarSchema`, `WalmartAmazonSchema`, `WdcProductSchema` — never reuse `ProductSchema`), the factory call, honest constants. Asserts id format at load and remaps to synthetic `<char><int>` if a source violates the `int(rid[1:])` split constraint. Ship a **shared parametrized test template** (id-scheme + gold-count + split-leakage) each loader reuses.

3. **Slice tags (Wave D) — external, `core/benchmark.py`.** `evaluate_judge_on_candidates(..., slice_fn: Callable[[Any], str|None] | None = None)`; compute the **global** best-F1 threshold once, then grade each slice at that fixed threshold via `classify_pairs(slice_judgements, slice_gold, global_threshold)` → optional `slices: dict[str, PairTrack]` on `JudgePairEval`. **No `ERCandidate` or `FixedSplitPairBenchmark` change.** WDC benchmark exposes `slice_map(split) -> dict[frozenset, str]` (seen/half-seen/unseen from train-entity membership); the caller builds `slice_fn` closing over it.

4. **Honest constants (all loaders).** Measure PC once via the slow `sweep_blocking_k`, commit the number as evidence, and pin `DEFAULT_<X>_BLOCKING_K` / `ACHIEVED_PC` / `GATE_MET` to the true values. A **non-slow deterministic fixture test** (id-parse + gold-counts) gates CI. Literature F1 ceilings are **cited**, never asserted as locally measured.

5. **DX (Wave E).** `evaluate(judge_or_resolver, gold_pairs) -> BenchmarkTable`-shaped one-liner wrapping `evaluate_judge_on_candidates`, + a "score your own CSV" tutorial section (build on `FixedSplitPairBenchmark.from_loaders`). `portfolio_race.py` iterates loadable registry entries → `run_methods` → `BenchmarkTable.to_markdown()`, includes the zero-shot-LLM-judge row behind an API-key/`--paid` guard (free by default), skips non-loadable entries. Portfolio doc names each dataset + why. State the registry's real value honestly: **discoverability + serialization** (`run_methods` already killed the racing boilerplate).

## Waves, dependencies, branches

| Wave | Content | Depends on | Branch |
|---|---|---|---|
| **A** ✅ | RR + GMD metrics (PR #89, green) | — | `feat/eval-metrics-rr-gmd` |
| **B** | Registry manifest + loader factory + shared test template + `list_methods`; register existing 4 | A merged | `feat/eval-registry-factory` |
| **C1** | **WDC Products** loader (+ `slice_map`) — critical path for D | **B merged** | `feat/eval-dataset-wdc` |
| **C2** | DBLP-ACM, DBLP-Scholar, Walmart-Amazon (structured) + OpenSanctions metadata — mutually parallel, independently shippable | **B merged** | `feat/eval-dataset-<name>` ×4 |
| **D** | External slice tags + honest fixed-threshold sliced aggregation | A merged; C1/WDC (real slice map) | `feat/eval-slice-tags` |
| **E** | DX: `evaluate()` one-liner + tutorial; `portfolio_race.py`; portfolio doc | B, C | `feat/eval-portfolio-dx` |

**Sequencing is real, per the review:** B must **merge** before any C loader branches (they import `registry`/factory). C loaders add isolated modules + fixtures + tests; the **orchestrator owns manifest entries centrally** (one line each) to avoid collision. Packaging is parallel-safe (hatchling ships committed `data/datasets/<x>/*.csv` with no pyproject edit).

## Access / front-loaded (rule #1)
- Datasets from HuggingFace **matchbench** (`huggingface.co` reachable). WDC from `webdatacommons.org` (not allowlisted → `dangerouslyDisableSandbox` for that fetch); verify WDC id column is integer, else remap.
- Git writes need `dangerouslyDisableSandbox` (repo sandbox blocks `.git/config`).
- No paid LLM in this wave (the `--paid` example row stays off by default). Frontier null-baseline gate is the immediate follow-on, not part of these PRs.

## Verification (before hand-off)
1. `uv run pytest -m "not slow and not integration"` green; new `core` code ≥95% cov; `test_import_budget` green — **incl. `list_benchmarks()` importing no loader / no faiss in a core-only env.**
2. `list_benchmarks()` returns every entry with correct `loadable`; `get_benchmark("<x>").load()` valid for loadable ones; OpenSanctions `.load()` raises the actionable error.
3. A new dataset runs end-to-end → pairwise + blocking (incl. **RR**) + clustering (incl. **GMD**).
4. WDC `slice_fn` → non-empty `slices` with distinct seen/unseen `PairTrack`s graded at the **same** threshold (expect an unseen F1 drop).
5. `portfolio_race.py` runs free by default over loadable entries; `evaluate()` one-liner scores a toy gold set.
6. Pinned constants match a fresh `sweep_blocking_k`; literature ceilings cited, not claimed.

## Out of scope / deferred
- Scaling-curve **C1**, hard-case mining (FP/FN extraction already exists), Beer/iTunes, experiment-tracking adapter (separate follow-on), all training, `langres bench` CLI (note: `cli.py` is stdlib-only → needs a lazy-import carve-out).
- **Real-world bundle-able person/org fixture (NCVR/GLEIF)** — deferred (David, 2026-07-08); tracked as the **explicit precondition** for the multilingual-person north-star experiment.

## Fast-follow (not this wave)
- If the loader factory lands clean, no separate refactor needed. Otherwise extract it next so adding a dataset is ~30 lines.

## Wave A review item
- Confirm `reduction_ratio` was added at the **end** of the `CandidateStats` dataclass (`core/debugging.py`) with a default + a compat test (positional-caller / `asdict` safety).
