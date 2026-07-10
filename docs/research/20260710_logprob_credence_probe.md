# The logprob-credence probe — does an LLM judge know when it is wrong?

**Date:** 2026-07-10 · **Cost:** $0.0158 (real, provider-billed) · **Status:** GATE PASSED

> This probe is a **gate, not a validation**. It ran *before* `confidence` was added to
> `PairwiseJudgement`, precisely so the evidence could kill the field rather than excuse it.
> Had `roc_auc(correct, credence)` come back ≈ 0.5, PR-2 would have shipped `decision` +
> abstain alone and `confidence` / `confidence_source` would have been dropped entirely.

## The question

`select_for_review(strategy="uncertainty")` exists to surface the pairs a judge probably got
wrong. For a binary LLM judge it currently no-ops: every score is `0.0` or `1.0`, so
`|score − threshold|` is always `0.5` and the uncertainty band is always empty. The loop reports
**exhausted** having never started.

Fixing that needs a real uncertainty signal. The candidate is the model's own first-token
credence, recovered from `logprobs`. Whether it is worth a permanent schema field depends on
**two different questions that must never be conflated**:

1. **`roc_auc(gold, p_yes)`** — does credence rank true matches above non-matches?
   A *task* question. A judge can ace this and still be useless for review.
2. **`roc_auc(answer_was_correct, credence)`** where `credence = max(p_yes, 1 − p_yes)` — does
   credence predict the model's **own errors**? **This is the one the flywheel needs.**

## Setup

- **Model:** `openrouter/openai/gpt-4o-mini-2024-07-18`
- **Data:** all **1206** Abt-Buy pairs from the Peeters/MatchGPT replication, `domain-complex-force`
  prompt, identical to the committed replication run apart from the logprob request.
- **Rows:** `examples/research/results/peeters/abt-buy__domain-complex-force__…__logprobs.jsonl`
  (schema v2). Written to a **separate `__logprobs` path** — the probe is physically incapable of
  overwriting the committed replication rows.
- **Analysis:** `examples/research/analyze_logprob_credence.py` (reads rows, makes no API calls).

`p_yes` is `yes_mass / (yes_mass + no_mass)` over the first non-whitespace content token,
renormalised over the **two-way subspace only**. The discarded mass is recorded as
`leaked_mass`, never normalised away.

## Results (all 1206 pairs)

| | |
|---|---|
| accuracy | **0.9751** (30 errors) |
| `roc_auc(gold, p_yes)` — Q1, task | **0.9947** |
| `average_precision(gold, p_yes)` | 0.9768 |
| **`roc_auc(correct, credence)` — Q2, flywheel** | **0.9500** |
| Brier (credence vs correct) | 0.0239 |
| ECE (10 equal-width bins) | 0.0248 |

**Reviewing the K% least-confident pairs:**

| K | reviewed | errors caught | of | capture | lift vs random |
|---|---|---|---|---|---|
| 1% | 12 | 2 | 30 | 6.7% | 6.7× |
| 2% | 24 | 6 | 30 | 20.0% | 10.1× |
| **5%** | **60** | **18** | **30** | **60.0%** | **12.1×** |
| 10% | 121 | 25 | 30 | 83.3% | 8.3× |
| 20% | 241 | 30 | 30 | **100.0%** | 5.0× |

## Verdict — `confidence` is earned

`roc_auc(correct, credence) = 0.9500`. Credence carries strong signal about the model's *own*
errors, which is the property `select_for_review` needs and the property Q1 does **not** establish.

Operationally: **reviewing the least-confident 5% of pairs catches 60% of the judge's errors**, a
12.1× lift over random sampling. Reviewing 20% catches *all thirty*. That is a working flywheel.

Calibration is good out of the box (ECE 0.0248) — the credence is close to an honest probability,
not just a usable ranking.

**PR-2 ships `confidence` + `confidence_source`, with this evidence.**

## Cost — confirmed free in output tokens

Total output tokens across 1206 pairs: **1206** — exactly **1.00 per pair**. Requesting
`top_logprobs=20` added **zero** output tokens, so the probe cost `$0.0158`: the same as the
replication run it mirrors. This confirms the plan's cost premise by measurement, not assumption.

Note the precise claim: confidence is free **in output tokens, at `explain=False`, on a
logprob-returning model.** It is *not* free in general — reasoning costs 3.75× on this same data.
Never write "confidence is free" in the API docs.

## Integrity checks (each is a way the numbers could have been quietly wrong)

- **Leaked mass** — mean `1.63e-09`, max `1.31e-07`. The Yes/No token matcher is not silently
  dropping BPE casing/whitespace variants. (A dropped variant would give `yes_mass = 0` on a real
  "Yes": wrong `p_yes`, no crash.)
- **`p_yes_is_bound`** — `0/1206`. Both masses were present on every pair, so no `p_yes` is a
  one-sided lower bound. Nothing was averaged into ECE as if exact.
- **`argmax(p_yes)` vs parsed verdict** — disagrees on `3/1206` (0.25%). All three are genuine
  near-ties (`p_yes = 0.50000001…`, model wrote `No`), not a `0.5` fallback: **zero** rows have
  `p_yes` exactly `0.5`. Verified at full float precision. These are the maximally-uncertain pairs
  a review queue *should* surface.
- **No duplicates** — 1206 unique `(left_id, right_id)`; resume never double-charged a pair.
- **`cost_is_real = True`** on every row — provider-billed, not a price-table estimate.
- **The `$0` replication regression still reproduces exactly** after these changes: F1 **92.09**
  (gpt-4o-mini) / **90.71** (gpt-4o), 99.25% per-pair agreement, reading the committed v1 rows.

## Limits — what this does NOT establish

- **One model, one dataset, one prompt design.** gpt-4o-mini on Abt-Buy with
  `domain-complex-force`. Nothing here shows credence is informative for gpt-4o, for a different
  prompt, or on a different dataset. The 30-error denominator is small; the 1% and 2% rows of the
  capture table rest on 2 and 6 errors respectively and should not be over-read.
- **Logprobs are an OpenAI-family feature.** Our own paid experiments run GLM / DeepSeek / Qwen
  via OpenRouter. Those judges may return no logprobs at all, in which case
  `confidence_source="none"` and the flywheel falls back to `strategy="disagreement"`. This is
  exactly why `confidence_source` must distinguish `"none"` (this judge structurally has none)
  from `"unrequested"` (it could, you didn't ask). **Unverified for non-OpenAI models.**
- **Calibration ≠ causation.** ECE 0.0248 says the credence is well-calibrated *on this data*.
  It does not license using `p_yes` as a probability elsewhere.
- The probe changed **no schema**. `PairwiseJudgement` was untouched; `p_yes`, `leaked_mass` and
  `p_yes_is_bound` rode in `provenance`.

## Reproduce ($0, from the committed rows)

```bash
uv run python examples/research/analyze_logprob_credence.py \
  examples/research/results/peeters/abt-buy__domain-complex-force__openrouter_openai_gpt-4o-mini-2024-07-18__logprobs.jsonl
```

The paid run itself (already done; a re-run resumes at `$0` because every judged pair is
committed):

```bash
uv run python examples/research/peeters_llm_em_replication.py \
  --mode live --logprobs --yes-spend-money \
  --model openrouter/openai/gpt-4o-mini-2024-07-18 \
  --results-dir examples/research/results/peeters --budget 0.05
```
