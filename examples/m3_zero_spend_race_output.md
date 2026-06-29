# M3 zero-spend race — reference output

Reference run of `examples/m3_zero_spend_race.py` (seed=0, real MiniLM
embeddings, **zero LLM spend**). Captured so the numbers survive without
re-embedding. Reproduce with:

```
uv run python examples/m3_zero_spend_race.py
```

The three zero-spend scorers (`rapidfuzz`, `weighted_average`,
`embedding_cosine`) are each raced through `run_method` on **both**
`FodorsZagatBenchmark` and `AmazonGoogleBenchmark`. Embedding/FAISS
nondeterminism moves the low-order digits slightly across machines; the headline
conclusions (FZ saturated, AG hard + discriminative, embedding_cosine over-merges
at the grid-capped threshold) are stable.

## Both tracks per (dataset, scorer)

| dataset | scorer | thr | pair_P | pair_R | pair_F1 | bc_P | bc_R | bc_F1 | clus_F1 | floor_F1 | Δ_floor | usd | s/pair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fodors_zagat | RapidfuzzModule | 0.80 | 0.6857 | 0.7273 | 0.7059 | 0.9922 | 0.9651 | 0.9785 | 0.8136 | 0.9317 | +0.0468 | 0.0000 | 0.003731 |
| fodors_zagat | weighted_average_judge | 0.80 | 0.6944 | 0.7576 | 0.7246 | 0.9910 | 0.9690 | 0.9799 | 0.8197 | 0.9317 | +0.0482 | 0.0000 | 0.003739 |
| fodors_zagat | embedding_score_judge | 0.80 | 0.0554 | 1.0000 | 0.1049 | 0.1988 | 1.0000 | 0.3316 | 0.0034 | 0.9317 | -0.6000 | 0.0000 | 0.003655 |
| amazon_google | RapidfuzzModule | 0.80 | 0.0468 | 0.0976 | 0.0633 | 0.7156 | 0.7642 | 0.7391 | 0.0322 | 0.8545 | -0.1154 | 0.0000 | 0.000105 |
| amazon_google | weighted_average_judge | 0.80 | 0.1538 | 0.1857 | 0.1683 | 0.8415 | 0.7902 | 0.8151 | 0.1353 | 0.8545 | -0.0394 | 0.0000 | 0.000106 |
| amazon_google | embedding_score_judge | 0.80 | 0.1358 | 0.9357 | 0.2372 | 0.4538 | 0.9879 | 0.6219 | 0.0248 | 0.8545 | -0.2326 | 0.0000 | 0.000096 |

## Headline table (`BenchmarkTable.to_markdown`)

| method | dataset | seed | threshold | bcubed_f1 | pair_f1 | usd_total | s_per_pair |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RapidfuzzModule | fodors_zagat | 0 | 0.80 | 0.9785 | 0.7059 | 0.0000 | 0.003731 |
| weighted_average_judge | fodors_zagat | 0 | 0.80 | 0.9799 | 0.7246 | 0.0000 | 0.003739 |
| embedding_score_judge | fodors_zagat | 0 | 0.80 | 0.3316 | 0.1049 | 0.0000 | 0.003655 |
| RapidfuzzModule | amazon_google | 0 | 0.80 | 0.7391 | 0.0633 | 0.0000 | 0.000105 |
| weighted_average_judge | amazon_google | 0 | 0.80 | 0.8151 | 0.1683 | 0.0000 | 0.000106 |
| embedding_score_judge | amazon_google | 0 | 0.80 | 0.6219 | 0.2372 | 0.0000 | 0.000096 |

## Amazon-Google read-out

- **Pair-level F1 spread across the 3 scorers: 0.1740** (min 0.0633 `rapidfuzz`,
  max 0.2372 `embedding_cosine`) — **DISCRIMINATES** (well above the 0.05 bar).
- **Best zero-spend pair-level F1: 0.2372 (`embedding_cosine`)** — the bar W4's
  LLM judge must beat on Amazon-Google.
- Pipeline BCubed recall is **blocking-ceiling-limited**: AG blocking
  Pair-Completeness caps ~0.84, and the best AG `bcubed_R` here is 0.9879 only
  because `embedding_cosine` over-merges (it pulls in nearly everything the
  blocker surfaced). The *useful* signal is the pair track, which is exactly why
  the harness reports it alongside BCubed.
- **Total spend across all 6 cells: $0.0000.**

## What the numbers say

- **Fodors-Zagat is saturated and easy.** `rapidfuzz` (0.9785) and
  `weighted_average` (0.9799) both clear the all-singletons floor (0.9317) with
  pipeline BCubed F1 ≈ 0.98 — reproducing the M2 baseline.
- **`embedding_cosine` over-merges at the grid-capped threshold.** Raw cosine
  similarity between blocked neighbours sits above the grid's top threshold
  (0.80) on FZ, so the clusterer merges almost everything (recall 1.0, precision
  0.05 → BCubed F1 0.33, far *below* the floor). This is an honest method
  weakness surfaced by the race, not a wiring bug: passing un-calibrated cosine
  as a match probability needs a higher / calibrated threshold than the shared
  grid provides.
- **Amazon-Google is hard and recall-capped.** Every method's pipeline BCubed F1
  sits at or below the high all-singletons floor (0.8545) — the corpus is mostly
  singletons, so "merge nothing" is a strong baseline. The **pair-level track**
  is where the methods separate (spread 0.174), confirming AG discriminates and
  that the pre-clustering scorer comparison is the right lens for W4.
