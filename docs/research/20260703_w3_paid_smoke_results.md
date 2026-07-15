# W3 paid smoke — SelectMatcher (set-wise) vs pairwise, measured

**Date:** 2026-07-03
**Branch:** `feat/w3-results-and-docs`
**Cost:** **$4.65 total** across both model points (SpendMonitor-capped at ≤$10)
**Script:** `examples/research/w3_paid_smoke.py`
**Results:** `data/benchmarks/w3/w3_smoke_results_gpt4o-mini.json`,
`data/benchmarks/w3/w3_smoke_results_gpt4o.json` (+ the emitted signal logs
`w3_smoke_judgements_*.jsonl`)
**Dataset:** the fixed Amazon-Google literature pair split
(`load_amazon_google_pair_splits`), grouped by Amazon anchor.

## What this proves (U4: measure before believing the claim)

The ER field's headline set-wise result — **ComEM Select: ~+16 F1 at ~⅓ cost**
by scoring an anchor against a *set* of candidates in one LLM call instead of
pair-by-pair — is the single biggest cost *and* quality lever we deferred to
M4.5 (S1). W3 is the ≤$10 paid smoke that **measures whether it replicates on our
data**, using langres' own `SelectMatcher` set-wise contract vs. an ordinary
pairwise judge, **same real model on both arms**, graded side by side on
Amazon-Google via `evaluate_judge_on_candidates` / `pair_pr_curve`.

The finding is a **useful nuanced/negative result**: the set-wise quality
advantage is **model-capability-dependent**, and the published +16 F1 magnitude
does **not** replicate here.

## What this measures

Both arms run the SAME real model over the SAME candidate population, graded once
(no blocking-recall ceiling, no clustering amplification — directly pairwise-F1):

- **Pairwise:** a `DSPyMatcher` scoring **one LLM call per pair**, graded via
  `evaluate_judge_on_candidates` (under a `BudgetedModuleRunner` pre-flight cap).
- **Set-wise:** a `SelectMatcher` scoring **one LLM call per anchor group** (the AG
  pairs re-shaped into per-Amazon-anchor groups via `derive_groups_from_pairs`),
  still yielding `PairwiseJudgement`s, graded with the same `pair_pr_curve`.

Two model points were run. **The two runs used different AG subset sizes** (300
anchor-groups for gpt-4o-mini, 120 for gpt-4o, to keep the frontier arm inside
budget), so **cross-model absolute F1 is NOT comparable** — only the *within-model*
set-vs-pairwise gap is meaningful.

## Results

| model | arm | pair-F1 | P | R | thr | LLM calls | cost (USD) |
|---|---|---|---|---|---|---|---|
| gpt-4o-mini (300 groups) | pairwise | **0.688** | 0.674 | 0.703 | 0.35 | 1421 | 0.264 |
| gpt-4o-mini (300 groups) | set-wise | 0.620 | 0.536 | 0.736 | 0.05 | 300 | 0.403 |
| gpt-4o (120 groups) | pairwise | 0.618 | 0.750 | 0.525 | 0.90 | 385 | 0.942 |
| gpt-4o (120 groups) | set-wise | **0.667** | 0.571 | 0.800 | 0.05 | 120 | 2.858 |

- **gpt-4o-mini: pairwise wins by +0.068 F1** (0.688 vs 0.620).
- **gpt-4o: set-wise wins by +0.049 F1** (0.667 vs 0.618) — the ComEM direction,
  at a fraction of its claimed magnitude.

## Read-out

- **The set-wise advantage is model-capability-dependent.** On the frontier model
  (gpt-4o) set-wise edges *ahead* of pairwise (+0.049) — matching the *direction*
  of the ComEM Select claim. On the mid-tier model (gpt-4o-mini) it falls *behind*
  (−0.068). The likely mechanism (a hypothesis consistent with the P/R split, not
  a proven cause): one-call-per-group is a **harder multi-candidate reasoning
  task** — the model must weigh several candidates against the anchor at once — and
  a weaker judge handles that less reliably than a sequence of simple pair
  decisions.
- **Set-wise consistently trades precision for recall.** In both runs set-wise has
  *higher recall* (0.736 vs 0.703; 0.800 vs 0.525) but *lower precision* (0.536 vs
  0.674; 0.571 vs 0.750) than pairwise. On the stronger model the recall gain
  outweighs the precision loss (net win); on the weaker model it does not (net
  loss). Set-wise scores also concentrate low, so its best threshold sits at the
  bottom of the grid (0.05) in both runs.
- **Fewer LLM *calls*, but more *dollars*.** Set-wise makes **3–5× fewer calls**
  (1421→300, i.e. 4.7×, on the mini run; 385→120, i.e. 3.2×, on the gpt-4o run —
  the ratio is just the mean group size of the scored anchors). But it costs *more
  dollars*: gpt-4o-mini $0.403 vs $0.264 (1.5×), gpt-4o $2.858 vs $0.942 (3.0×).
  Each group prompt packs every candidate's text into one call, so the token-heavy
  input outweighs the call-count saving at these group sizes. **The call-count
  lever and the dollar-cost lever point in opposite directions here** — the "⅓
  cost" ComEM headline is a *call-count* saving, not a dollar saving on these
  prompts.
- **ComEM's +16 F1 does not replicate at that magnitude.** The effect here is
  smaller (best case +0.049 on gpt-4o) and model-dependent (negative on
  gpt-4o-mini). This is the honest U4 outcome: the claim is real in *direction* on
  a strong model, but neither the magnitude nor the "cheaper" framing carries over
  unqualified to our data and models.

## Other smoke deliverables (gpt-4o-mini run)

The same script exercises the paid verb surface end-to-end under the one budget
cap:

- **`link()`** on one pair — match, score **0.95** (`judge_used="custom"`).
- **`dedupe()`** on a small record set — **1 cluster** (`[d1, d2]`).
- **A single `SelectMatcher` group call** — **1 LLM call judging 22 members** →
  22 `PairwiseJudgement`s for **$0.011**. This is the raw set-wise cost lever on a
  real model: one call where pairwise would have made 22.
- **Signal log** (`JudgementLog`, the flywheel inlet) — **4 rows** emitted by the
  verb calls, read back with honest per-call cost.
- **Verb cost** (link + dedupe) — **$0.0018**.

## Reproduce

Both runs are SpendMonitor-capped so they structurally cannot cross `--budget`
(hard ceiling $10). Needs a real `OPENROUTER_API_KEY` (loaded from `.env`) and the
`semantic` + `llm` extras (`langres.data.amazon_google` pulls the embedding stack;
`DSPyMatcher` pulls dspy/litellm).

```bash
# gpt-4o-mini (default: 300 AG anchor-groups) — the primary run
uv run python examples/research/w3_paid_smoke.py \
    --budget 9.0 --model openrouter/openai/gpt-4o-mini

# gpt-4o frontier point (bounded to 120 anchor-groups to stay inside budget)
uv run python examples/research/w3_paid_smoke.py \
    --budget 9.0 --model openrouter/openai/gpt-4o --ag-groups 120
```

The whole flow is verified at **$0** with DSPy `DummyLM` in
`tests/examples/test_w3_paid_smoke.py` (`run_smoke` takes injectable LMs — that
test never makes a real call); the harness has a proven cap-fires-with-partials
test (the `BudgetExceeded.partial_judgements` contract). Only the single paid
execution above spends money.
