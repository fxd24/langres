# W2.1 — Person resolution (FEBRL4), config-only, $0

**Date:** 2026-07-03
**Branch:** `feat/w2-person-febrl`
**Cost:** $0.00 (five free local methods; no LLM calls)
**Script:** `examples/research/w2_person_benchmark.py`
**Dataset:** `src/langres/data/datasets/febrl_person/` (500/side FEBRL4 subset)

## What this proves

langres resolves a **second entity type — a person** — with **zero new core
code**. Everything here is config only: a new dataset fixture plus one
`src/langres/data/febrl_person.py` adapter (`FebrlPersonSchema` + `load_febrl_person`
+ `FebrlPersonBenchmark`), added the exact way a user would add a dataset — the
same shape as the Fodors-Zagat, Amazon-Google, and Abt-Buy adapters. No file
under `src/langres/core/` changed. The person is then scored through the
**existing** harness (`run_method` / `evaluate_judge_on_candidates`) that already
races products and restaurants.

## What this measures

Blocking is a MiniLM VectorBlocker over `embed_text = "{given_name} {surname}
{suburb}"`, pinned at **`k=20`**. Cross-source blocking Pair-Completeness sweep
on the full 1000-record corpus (500 originals + 500 duplicates, 500 gold pairs),
measured on macOS:

| k | 5 | 10 | 20 | 30 | 50 |
|---|---|---|---|---|---|
| cross-source PC (macOS) | 0.966 | 0.968 | **0.978** | 0.982 | 0.986 |

Recall is high everywhere but sits near the 0.95 gate at small `k`, and the exact
value is **platform-dependent**: cross-platform embedding float differences (ARM
vs x86 BLAS) reorder borderline neighbours, so `k=5` clears 0.95 on macOS but
dips just below it on the Linux CI runner. There is no seed/index fix — the FAISS
cosine search is already exact; the divergence is in the MiniLM embeddings, which
are not bit-identical across architectures. So `k` is pinned at **20** — the
smallest neighbourhood that clears the 0.95 gate on the **ship platform (Linux
CI)**, and which also clears comfortably on macOS (0.978) — for a stable,
cross-platform-robust pin rather than the macOS-only min-k of 5.

Five free scorers race on the identical blocked candidate set at `seed=0`:

- **Zero-spend** (`rapidfuzz`, `weighted_average`, `embedding_cosine`) run through
  the full pipeline (`run_method`, `budget=0.0` hard-asserts zero spend), so they
  report both pipeline **BCubed** F1 and pre-clustering **pairwise** P/R/F1.
- **Trained family** (`fellegi_sunter` unsupervised EM, `random_forest`
  supervised) cannot be raced unfit, so they follow the fit seam: fit on the
  train split's own blocked candidates, then grade the TEST split's candidates
  once via `evaluate_judge_on_candidates` (pairwise F1 only — the judged-once
  surface has no clustering step, so BCubed is not defined there, shown `—`).

## Results (seed=0)

| method | family | bcubed_f1 | pair_P | pair_R | pair_F1 | thr | usd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rapidfuzz | zero-spend | 0.9983 | 0.7487 | 0.9933 | 0.8539 | 0.60 | 0.0000 |
| weighted_average | zero-spend | 0.9983 | 0.7487 | 0.9933 | 0.8539 | 0.60 | 0.0000 |
| embedding_cosine | zero-spend | 0.9088 | 0.5819 | 0.9000 | 0.7068 | 0.80 | 0.0000 |
| fellegi_sunter | trained | — | 0.7487 | 1.0000 | 0.8563 | 0.30 | 0.0000 |
| random_forest | trained | — | 0.9539 | 0.9732 | 0.9635 | 0.30 | 0.0000 |

**Total spend across all 5 cells: $0.0000.** (Measured on macOS at the pinned
`k=20`; the committed reference run.)

## Read-out

- **Person resolution is measurable and strong.** The supervised
  `random_forest` judge tops the pairwise field at **F1 0.964** (P 0.954 /
  R 0.973); the string/field judges hit **BCubed F1 0.998** at the pipeline level.
  FEBRL persons are clean multi-field identity data (name + address + DOB +
  SSN-like id), so — like Fodors-Zagat — this is a saturated, high-ceiling
  benchmark, not a hard one.
- **`rapidfuzz` and `weighted_average` are identical here.** On this schema both
  reduce to the same per-field string-similarity signal over the same
  Comparator features at the same tuned threshold, so they converge cell-for-cell
  (expected, not a bug).
- **`fellegi_sunter` is high-recall/low-precision** (R 1.00, P 0.746) — the
  unsupervised EM keeps every blocked candidate above threshold at the low end
  of the grid. Consistent with the W1.2 trained-family finding on the other
  datasets: FS is an honest high-recall labeler, RF is the precision lever.
- **Blocking is the recall ceiling.** Pairwise recall tracks the ~0.98 blocking
  Pair-Completeness (at `k=20`) on the test split; no scorer can recover a pair
  the blocker never surfaced.
- **The pin is cross-platform-honest, not tuned-for-green.** `k=20` is the min-k
  clearing the 0.95 gate on the ship platform (Linux CI). On macOS the same gate
  clears at `k=5`; pinning the higher, platform-robust `k` (rather than the
  macOS-only min-k) is the honest choice given the embeddings are not
  bit-identical across architectures.

## Reproduce

```bash
# Fixture is committed; recordlinkage is NOT a project dependency (only needed
# once to regenerate the fixture — see datasets/febrl_person/SOURCE.md).
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  uv run --extra semantic --extra trained --extra llm \
  python examples/research/w2_person_benchmark.py
```

(The `llm` extra is needed only because importing `langres.methods` pulls the
cascade module's `litellm` import; **no LLM call is made** and total spend is $0.)
