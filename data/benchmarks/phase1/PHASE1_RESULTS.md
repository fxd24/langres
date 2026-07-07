# Phase 1 — RandomForestJudge honest pair-level floor ($0)

Single-metric `StringComparator` (one rapidfuzz `token_sort_ratio` per field) + `RandomForestJudge`, graded on the **full standard test split** at a threshold **derived on train** (Youden). `argmax_on_test` is the leaky ceiling (threshold tuned on test) shown only to expose the honesty delta. This is a floor to beat, not a Magellan-class multi-feature replication.

| dataset | honest P | honest R | honest F1 | argmax-on-test F1 | honesty Δ | threshold | Ditto F1 | gap to Ditto |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| amazon_google | 0.3151 | 0.4188 | 0.3596 | 0.4048 | 0.0452 | 0.2524 | 0.756 | 0.3964 |
| abt_buy | 0.3787 | 0.4320 | 0.4036 | 0.4198 | 0.0161 | 0.2803 | 0.893 | 0.4894 |

## Reading

- **honest F1** is the number that matters: no test-label peeking.
- **honesty Δ** = how much an argmax-on-test report would have inflated F1 over the honest cut — the exact leakage this Phase 1 seam removes.
- **gap to Ditto** is the distance a $0, single-metric local baseline leaves for the paid/multi-feature judges the later phases add.

Per-dataset detail (shapes, tp/fp/fn, features) is in the sibling `phase1_rf_floor_<dataset>.json` files.
