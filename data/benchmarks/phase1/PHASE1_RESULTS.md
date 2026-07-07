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

## The seam: swap the judge, lift the number

`PHASE1_LLM_PLACEMENT.md` is the companion **paid** run — a zero-shot `LLMJudge`
(DeepSeek V4-Flash / V4-Pro) graded through the *same* `FixedSplitPairBenchmark`
seam, same full splits, same honest (leakage-free) protocol. Swapping only the
judge lifts honest F1 well above this $0 floor:

| dataset | RF floor F1 | Flash F1 | Pro F1 | Ditto |
| --- | --- | --- | --- | --- |
| amazon_google | 0.360 | 0.575 | 0.614 | 0.756 |
| abt_buy | 0.404 | 0.680 | 0.737 | 0.893 |

Every LLM cell clears the floor by **+0.21 … +0.33** at real, metered OpenRouter
cost ($5.20 total, 100% real-cost). The remaining **0.14 … 0.21** gap to Ditto —
a *fine-tuned* specialist vs. these *zero-shot* judges — is exactly what the
Phase 2 (#81) small-LM student is meant to close.

**These are an untuned floor.** The judges use `LLMJudge`'s generic default
prompt (no DSPy, no prompt/signature tuning, no few-shot) — hence the
high-recall / low-precision profile (they over-call matches). langres's M4 work
showed a **precision-tuned DSPy signature is a large lever** (cheap GLM
0.409 → 0.757 on Amazon-Google, with *no* expensive compilation). So prompt/
signature tuning is the near rung and a fine-tuned student the far rung on the
ladder from this zero-shot number toward Ditto — deliberately left as follow-ups.
