# Is there a fixed per-score-family match threshold that beats 0.5?

> **Generated document — do not edit in place.** The prose lives in
> `docs/research/_20260728_threshold_constant.body.md`; every table below is
> printed by `examples/research/threshold_constant_sweep.py --render` and spliced
> in by `tools/render_threshold_constant_writeup.py`. Edit the body, re-run the
> tool. Three PRs on 2026-07-28 shipped factual errors and **every one was in
> hand-typed prose sitting beside a correct generated table**, which is why
> **no number produced by this study is typed by hand anywhere in this
> document** — every one arrives in a generated table. The prose does quote
> figures from *other* documents (`45 of 54` and the `0.174–0.695` derived-cut
> span, both from PR #250); each was re-read in its source before being repeated
> here.

- **Date**: 2026-07-28 (checkpoint-transfer run, §7, completed 2026-07-29)
- **Harness**: `examples/research/threshold_constant_sweep.py` (committed before the run)
- **Artifacts**: `examples/research/results/threshold_constant_sweep.json` (the
  portfolio) and `…_sweep.e5.json` (the checkpoint-transfer variant, §7)
- **Spend**: $0. Every scorer swept here is free and offline.

## 1. The question, and why it was still open

`langres` ships six front doors that hard-code a `0.5` match cut (`FuzzyString`,
`VectorLLMCascade`, and the four `architectures/retrieval.py` recipes) — though
strictly only four of them *cut* at it: `RetrieveLLM` and `RetrieveRerankLLM`
declare the number and never read it, which is its own bug and is dealt with
separately (§9). `MethodSpec.default_threshold` has existed alongside all six the
whole time — declared per method, documented, and read by **nothing**.

PR #250 (`docs/research/20260728_threshold_default.md`) measured that *deriving*
the cut from labels beats `0.5` in 45 of 54 cells, and recommended fixing the
constant next. But its §5.2 is explicit about what it did **not** do: it scored
F1 at `0.5` and at each dataset's own derived cut, and nowhere else. It never
evaluated a *shared replacement constant*. So the numbers a reader would be
tempted to reach for — the medians of those derived cuts — are a **documented
prior, not a measurement of the constant**. Shipping them would be the exact
error this repo keeps re-learning: treating your own earlier summary as a
finding.

This study measures the thing you would actually ship: **one fixed number per
score family, chosen without access to the dataset it is graded on.**

## 2. What can be measured for free, and what cannot

| family | methods | measurable at $0? | why |
|---|---|---|---|
| `heuristic` | rapidfuzz, string, weighted_average | yes | pure string similarity -- no model, no spend |
| `prob_fs` | fellegi_sunter | no | a FITTED matcher (EM) -- a label-free user cannot run it at all |
| `prob_group_llm` | select_judge | no | every score costs a paid completion |
| `prob_llm` | cascade, dspy_judge, llm_judge, prompt_llm, zero_shot_llm | no | every score costs a paid completion |
| `prob_rf` | random_forest | no | a FITTED matcher (labels required) -- same |
| `sim_cos` | embedding, embedding_cosine | yes | local sentence-transformer -- free after one download |

Only two families are in scope, and the boundary is not arbitrary. `prob_llm` /
`prob_group_llm` bill a paid completion for every score, so a portfolio-wide grid
sweep over them is a real invoice, not a free one. `prob_fs` / `prob_rf` are
*fitted* matchers: a user with no labels cannot run them at all, so "the best
out-of-the-box constant" is not even the same question for them. **Those four
families are left alone, and this document is not evidence about them.**

## 3. Protocol

Each cell is one `(benchmark, method, seed)`.

1. **Split by corpus, not by pairs.** `Benchmark.split(seed=...)` assigns whole
   gold clusters to one side, so no entity and no match pair straddles the
   boundary. The label-derived cut (Youden's J) is fitted on the **train**
   corpus; every number reported here is from the **disjoint test** corpus.
2. **Score once, sweep for free.** Thresholding does not change scores, so a
   single blocking + scoring pass yields the exact F1 at all 99 grid constants,
   at `0.5`, at the derived cut, and at the oracle.
3. **The oracle is exact.** It sweeps every distinct score, not the grid — so
   "how much of the achievable gain does a constant capture?" is measured against
   the true ceiling rather than the best point of an arbitrary grid.
4. **Leave-one-benchmark-out selection.** For each held-out benchmark the
   constant is re-selected from the *other* benchmarks only, then graded on the
   held-out one. This is the number that answers "what does shipping this do on a
   dataset I have never seen?" The in-sample argmax is also printed, labeled as
   the optimistic one.
5. **Paired cluster bootstrap for the intervals.** Candidate pair rows are not
   independent — every pair touching one entity rises and falls together — so
   resampling pair rows yields intervals that are far too narrow. The resampling
   unit is the **gold cluster**; each judged pair is assigned to `min` of its two
   endpoints' units, so every pair is counted exactly once. The same resample
   grades both the candidate constant and `0.5`, and the interval is on their
   difference.

   **What that `min` does and does not buy** — an earlier draft of this section
   claimed more than it should, and review caught it. A *gold* pair has both
   endpoints in the same cluster by definition, so `min` is a no-op for it: the
   true-positive numerator and the recall denominator are attributed exactly, and
   an entity's gold pairs do move as a block. A *non-gold candidate* pair spans
   two clusters, and attributing it to one endpoint means resampling the other
   endpoint's cluster does not move it — so the model is exact on the recall side
   and approximate on the precision side, biasing intervals slightly **too
   narrow**. It is disclosed rather than fixed because the alternative (requiring
   both clusters to be drawn) makes inclusion quadratic and measures a different
   quantity, and because **neither verdict is marginal**: `sim_cos` ships on
   point estimates positive in every eligible cell, which no interval width can
   reverse, and `heuristic`'s veto — the one direction that could be sensitive —
   rests on deltas several half-widths below zero on three independent seeds and
   fails safe, since its effect is to keep the incumbent.
6. **No averaging across benchmarks in any reported number.** Aggregation appears
   in exactly one place — the LOBO *selection* criterion, which has to reduce the
   other benchmarks to a single ordering — and it is a median over benchmarks of
   a mean over seeds, labeled a selection statistic and never reported as
   performance.
7. **The ship rule was pre-registered** in the harness (`SHIP_RULE`) before the
   numbers were seen, and §6's verdict table is computed from the artifact by
   applying it.

Two conventions are computed for every cell, because the choice of denominator
can flip a conclusion and it should be visible if it does:

- **blocked-gold** — F1 over the gold pairs the blocker actually surfaced. This
  isolates the *threshold's* effect from the blocker's recall ceiling, and is the
  convention the headline tables use.
- **all-gold** — F1 against the full gold set, i.e. end-to-end, with the
  blocker's misses counted as false negatives. §5 repeats the comparison under
  it.

### Verification the harness is measuring what it claims

The sweep computes F1 from its own reverse-cumulative sufficient statistics
rather than by re-running the metric 99 times, which is fast but easy to get
subtly wrong. So it checks itself: `_assert_matches_classify_pairs` recomputes
three grid points per cell through the library's own `classify_pairs` and aborts
the run on any disagreement above `1e-9`. Every cell in the artifact passed it.

## 4. Results

Every cell measured, no aggregation:

| benchmark | family | method | seed | test pairs | held-out gold | units | F1@0.50 | best grid t | F1@best | oracle t | oracle F1 | derived t | derived F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abt_buy | `heuristic` | rapidfuzz | 0 | 9,003 | 310 | 341 | 0.0625 | 0.29 | 0.0772 | 0.2928 | 0.0775 | 0.1744 | 0.0730 |
| abt_buy | `heuristic` | rapidfuzz | 1 | 9,143 | 309 | 341 | 0.0630 | 0.21 | 0.0726 | 0.2943 | 0.0735 | 0.1949 | 0.0716 |
| abt_buy | `heuristic` | rapidfuzz | 2 | 8,816 | 309 | 341 | 0.0642 | 0.25 | 0.0813 | 0.2501 | 0.0814 | 0.1874 | 0.0754 |
| abt_buy | `sim_cos` | embedding_cosine | 0 | 9,003 | 310 | 341 | 0.0666 | 0.88 | 0.2453 | 0.8918 | 0.2479 | 0.8361 | 0.2301 |
| abt_buy | `sim_cos` | embedding_cosine | 1 | 9,143 | 309 | 341 | 0.0654 | 0.87 | 0.2299 | 0.8753 | 0.2359 | 0.8304 | 0.2255 |
| abt_buy | `sim_cos` | embedding_cosine | 2 | 8,816 | 309 | 341 | 0.0677 | 0.88 | 0.3251 | 0.8833 | 0.3281 | 0.8294 | 0.2567 |
| amazon_google | `heuristic` | rapidfuzz | 0 | 48,811 | 420 | 1,025 | 0.0149 | 0.91 | 0.0809 | 0.9116 | 0.0814 | 0.3136 | 0.0231 |
| amazon_google | `heuristic` | rapidfuzz | 1 | 49,194 | 420 | 1,025 | 0.0149 | 0.93 | 0.0864 | 0.9333 | 0.0873 | 0.3136 | 0.0225 |
| amazon_google | `heuristic` | rapidfuzz | 2 | 49,093 | 420 | 1,025 | 0.0159 | 0.89 | 0.0794 | 0.8981 | 0.0821 | 0.3085 | 0.0230 |
| amazon_google | `sim_cos` | embedding_cosine | 0 | 48,811 | 420 | 1,025 | 0.0171 | 0.91 | 0.4415 | 0.9088 | 0.4463 | 0.8139 | 0.2803 |
| amazon_google | `sim_cos` | embedding_cosine | 1 | 49,194 | 420 | 1,025 | 0.0169 | 0.91 | 0.4664 | 0.9084 | 0.4722 | 0.8125 | 0.2456 |
| amazon_google | `sim_cos` | embedding_cosine | 2 | 49,093 | 420 | 1,025 | 0.0170 | 0.92 | 0.4567 | 0.9297 | 0.4623 | 0.8125 | 0.2502 |
| dblp_acm | `heuristic` | rapidfuzz | 0 | 5,381 | 665 | 807 | 0.3972 | 0.70 | 0.8241 | 0.7002 | 0.8246 | 0.6952 | 0.8163 |
| dblp_acm | `heuristic` | rapidfuzz | 1 | 5,466 | 665 | 807 | 0.3910 | 0.70 | 0.8018 | 0.7131 | 0.8053 | 0.6917 | 0.7899 |
| dblp_acm | `heuristic` | rapidfuzz | 2 | 5,477 | 662 | 807 | 0.3972 | 0.71 | 0.7951 | 0.7050 | 0.8021 | 0.6954 | 0.7885 |
| dblp_acm | `sim_cos` | embedding_cosine | 0 | 5,381 | 665 | 807 | 0.2200 | 0.92 | 0.8928 | 0.9166 | 0.8974 | 0.9018 | 0.8825 |
| dblp_acm | `sim_cos` | embedding_cosine | 1 | 5,466 | 665 | 807 | 0.2169 | 0.91 | 0.8845 | 0.9120 | 0.8886 | 0.9041 | 0.8802 |
| dblp_acm | `sim_cos` | embedding_cosine | 2 | 5,477 | 662 | 807 | 0.2157 | 0.92 | 0.8693 | 0.9248 | 0.8714 | 0.9010 | 0.8537 |
| dblp_scholar | `heuristic` | rapidfuzz | 0 | 730,714 | 4,729 | 18,478 | 0.0344 | 0.74 | 0.2908 | 0.7371 | 0.2941 | 0.4858 | 0.0324 |
| dblp_scholar | `heuristic` | rapidfuzz | 1 | 731,910 | 4,670 | 18,478 | 0.0332 | 0.76 | 0.2793 | 0.7611 | 0.2806 | 0.4891 | 0.0317 |
| dblp_scholar | `heuristic` | rapidfuzz | 2 | 733,264 | 4,689 | 18,478 | 0.0335 | 0.74 | 0.2851 | 0.7393 | 0.2873 | 0.4870 | 0.0316 |
| dblp_scholar | `sim_cos` | embedding_cosine | 0 | 730,714 | 4,729 | 18,478 | 0.0129 | 0.90 | 0.6945 | 0.9027 | 0.6983 | 0.8360 | 0.3245 |
| dblp_scholar | `sim_cos` | embedding_cosine | 1 | 731,910 | 4,670 | 18,478 | 0.0127 | 0.91 | 0.6903 | 0.9026 | 0.6951 | 0.8390 | 0.3276 |
| dblp_scholar | `sim_cos` | embedding_cosine | 2 | 733,264 | 4,689 | 18,478 | 0.0127 | 0.90 | 0.6695 | 0.9027 | 0.6727 | 0.8390 | 0.3286 |
| febrl_dedup | `heuristic` | rapidfuzz | 0 | 52,350 | 1,869 | 597 | 0.9585 | 0.58 | 0.9979 | 0.5742 | 0.9981 | 0.5286 | 0.9860 |
| febrl_dedup | `heuristic` | rapidfuzz | 1 | 52,151 | 1,885 | 597 | 0.9505 | 0.58 | 0.9963 | 0.5793 | 0.9965 | 0.5417 | 0.9874 |
| febrl_dedup | `heuristic` | rapidfuzz | 2 | 52,849 | 1,893 | 597 | 0.9580 | 0.57 | 0.9968 | 0.5742 | 0.9971 | 0.5289 | 0.9846 |
| febrl_dedup | `sim_cos` | embedding_cosine | 0 | 52,350 | 1,869 | 597 | 0.0689 | 0.84 | 0.8320 | 0.8334 | 0.8366 | 0.8029 | 0.7567 |
| febrl_dedup | `sim_cos` | embedding_cosine | 1 | 52,151 | 1,885 | 597 | 0.0698 | 0.85 | 0.8012 | 0.8514 | 0.8033 | 0.8029 | 0.7101 |
| febrl_dedup | `sim_cos` | embedding_cosine | 2 | 52,849 | 1,893 | 597 | 0.0692 | 0.84 | 0.8363 | 0.8358 | 0.8377 | 0.8030 | 0.7271 |
| febrl_person | `heuristic` | rapidfuzz | 0 | 4,180 | 149 | 150 | 0.9521 | 0.57 | 1.0000 | 0.7434 | 1.0000 | 0.5963 | 1.0000 |
| febrl_person | `heuristic` | rapidfuzz | 1 | 4,186 | 150 | 150 | 0.9434 | 0.55 | 1.0000 | 0.6070 | 1.0000 | 0.5963 | 1.0000 |
| febrl_person | `heuristic` | rapidfuzz | 2 | 4,231 | 149 | 150 | 0.9460 | 0.55 | 1.0000 | 0.6419 | 1.0000 | 0.5963 | 1.0000 |
| febrl_person | `sim_cos` | embedding_cosine | 0 | 4,180 | 149 | 150 | 0.0688 | 0.82 | 0.9028 | 0.8141 | 0.9078 | 0.8090 | 0.8926 |
| febrl_person | `sim_cos` | embedding_cosine | 1 | 4,186 | 150 | 150 | 0.0692 | 0.84 | 0.9258 | 0.8410 | 0.9258 | 0.8107 | 0.8846 |
| febrl_person | `sim_cos` | embedding_cosine | 2 | 4,231 | 149 | 150 | 0.0680 | 0.83 | 0.9141 | 0.8410 | 0.9253 | 0.8107 | 0.8734 |
| fodors_zagat | `heuristic` | rapidfuzz | 0 | 1,014 | 33 | 225 | 0.1164 | 0.81 | 0.8276 | 0.8175 | 0.8421 | 0.6725 | 0.5299 |
| fodors_zagat | `heuristic` | rapidfuzz | 1 | 1,004 | 33 | 225 | 0.1107 | 0.76 | 0.7812 | 0.7649 | 0.7812 | 0.6725 | 0.5688 |
| fodors_zagat | `heuristic` | rapidfuzz | 2 | 977 | 33 | 225 | 0.1121 | 0.79 | 0.7500 | 0.7665 | 0.7541 | 0.6762 | 0.5357 |
| fodors_zagat | `sim_cos` | embedding_cosine | 0 | 1,014 | 33 | 225 | 0.0630 | 0.90 | 0.9254 | 0.9153 | 0.9394 | 0.9018 | 0.9254 |
| fodors_zagat | `sim_cos` | embedding_cosine | 1 | 1,004 | 33 | 225 | 0.0636 | 0.91 | 0.9524 | 0.9153 | 0.9524 | 0.9018 | 0.9524 |
| fodors_zagat | `sim_cos` | embedding_cosine | 2 | 977 | 33 | 225 | 0.0653 | 0.90 | 0.9355 | 0.9055 | 0.9355 | 0.9018 | 0.9355 |
| tiny_fixture | `heuristic` | rapidfuzz | 0 | 3 | 1 | 2 | 1.0000 | 0.31 | 1.0000 | 0.6214 | 1.0000 | 0.6319 | 0.0000 |
| tiny_fixture | `heuristic` | rapidfuzz | 1 | 3 | 1 | 2 | 1.0000 | 0.39 | 1.0000 | 0.6214 | 1.0000 | 0.6319 | 0.0000 |
| tiny_fixture | `heuristic` | rapidfuzz | 2 | 3 | 1 | 2 | 1.0000 | 0.29 | 1.0000 | 0.6639 | 1.0000 | 0.6214 | 1.0000 |
| tiny_fixture | `sim_cos` | embedding_cosine | 0 | 3 | 1 | 2 | 0.5000 | 0.54 | 1.0000 | 0.9647 | 1.0000 | 0.9451 | 1.0000 |
| tiny_fixture | `sim_cos` | embedding_cosine | 1 | 3 | 1 | 2 | 0.5000 | 0.61 | 1.0000 | 0.9647 | 1.0000 | 0.9451 | 1.0000 |
| tiny_fixture | `sim_cos` | embedding_cosine | 2 | 3 | 1 | 2 | 0.5000 | 0.65 | 1.0000 | 0.9718 | 1.0000 | 0.9451 | 1.0000 |
| walmart_amazon | `heuristic` | rapidfuzz | 0 | 259,587 | 325 | 7,099 | 0.0083 | 0.66 | 0.0163 | 0.6395 | 0.0163 | 0.5590 | 0.0112 |
| walmart_amazon | `heuristic` | rapidfuzz | 1 | 260,526 | 325 | 7,099 | 0.0087 | 0.97 | 0.0201 | 0.9718 | 0.0203 | 0.5206 | 0.0097 |
| walmart_amazon | `heuristic` | rapidfuzz | 2 | 260,465 | 324 | 7,099 | 0.0091 | 0.84 | 0.0198 | 0.8360 | 0.0209 | 0.5538 | 0.0119 |
| walmart_amazon | `sim_cos` | embedding_cosine | 0 | 259,587 | 325 | 7,099 | 0.0025 | 0.92 | 0.0692 | 0.9659 | 0.0701 | 0.8634 | 0.0336 |
| walmart_amazon | `sim_cos` | embedding_cosine | 1 | 260,526 | 325 | 7,099 | 0.0025 | 0.95 | 0.0759 | 0.9349 | 0.0766 | 0.8633 | 0.0332 |
| walmart_amazon | `sim_cos` | embedding_cosine | 2 | 260,465 | 324 | 7,099 | 0.0025 | 0.94 | 0.0689 | 0.9438 | 0.0716 | 0.8633 | 0.0323 |
| wdc_computers | `heuristic` | rapidfuzz | 0 | 47,359 | 308 | 1,097 | 0.0272 | 0.81 | 0.0631 | 0.8103 | 0.0634 | 0.4142 | 0.0218 |
| wdc_computers | `heuristic` | rapidfuzz | 1 | 47,639 | 310 | 1,097 | 0.0291 | 0.83 | 0.0612 | 0.8254 | 0.0618 | 0.4056 | 0.0222 |
| wdc_computers | `heuristic` | rapidfuzz | 2 | 47,421 | 312 | 1,097 | 0.0283 | 0.81 | 0.0711 | 0.7882 | 0.0726 | 0.4045 | 0.0216 |
| wdc_computers | `sim_cos` | embedding_cosine | 0 | 47,359 | 308 | 1,097 | 0.0129 | 0.97 | 0.0749 | 0.9704 | 0.0757 | 0.8725 | 0.0485 |
| wdc_computers | `sim_cos` | embedding_cosine | 1 | 47,639 | 310 | 1,097 | 0.0129 | 0.93 | 0.0721 | 0.9249 | 0.0757 | 0.8671 | 0.0417 |
| wdc_computers | `sim_cos` | embedding_cosine | 2 | 47,421 | 312 | 1,097 | 0.0131 | 0.96 | 0.0830 | 0.9640 | 0.0862 | 0.8730 | 0.0471 |

Blocker checkpoint(s) in this artifact: all-MiniLM-L6-v2 (benchmark pin).

The headline: a constant chosen **without** the benchmark it is graded on.

| family | benchmark | seed | LOBO t | F1@0.50 | F1@LOBO | Δ | 95% CI | oracle F1 | capture |
|---|---|---|---|---|---|---|---|---|---|
| `heuristic` | abt_buy | 0 | 0.74 | 0.0625 | 0.0083 | -0.0542 | [-0.0682, -0.0420] | 0.0775 | -360% |
| `heuristic` | abt_buy | 1 | 0.74 | 0.0630 | 0.0145 | -0.0486 | [-0.0633, -0.0318] | 0.0735 | -465% |
| `heuristic` | abt_buy | 2 | 0.74 | 0.0642 | 0.0227 | -0.0416 | [-0.0592, -0.0219] | 0.0814 | -243% |
| `heuristic` | amazon_google | 0 | 0.74 | 0.0149 | 0.0463 | +0.0314 | [+0.0157, +0.0501] | 0.0814 | 47% |
| `heuristic` | amazon_google | 1 | 0.74 | 0.0149 | 0.0506 | +0.0358 | [+0.0205, +0.0553] | 0.0873 | 49% |
| `heuristic` | amazon_google | 2 | 0.74 | 0.0159 | 0.0504 | +0.0345 | [+0.0147, +0.0594] | 0.0821 | 52% |
| `heuristic` | dblp_acm | 0 | 0.76 | 0.3972 | 0.6880 | +0.2907 | [+0.2521, +0.3261] | 0.8246 | 68% |
| `heuristic` | dblp_acm | 1 | 0.76 | 0.3910 | 0.6701 | +0.2791 | [+0.2382, +0.3144] | 0.8053 | 67% |
| `heuristic` | dblp_acm | 2 | 0.76 | 0.3972 | 0.6546 | +0.2573 | [+0.2195, +0.2920] | 0.8021 | 64% |
| `heuristic` | dblp_scholar | 0 | 0.73 | 0.0344 | 0.2869 | +0.2525 | [+0.2223, +0.2835] | 0.2941 | 97% |
| `heuristic` | dblp_scholar | 1 | 0.73 | 0.0332 | 0.2728 | +0.2396 | [+0.2091, +0.2699] | 0.2806 | 97% |
| `heuristic` | dblp_scholar | 2 | 0.73 | 0.0335 | 0.2775 | +0.2440 | [+0.2165, +0.2727] | 0.2873 | 96% |
| `heuristic` | febrl_dedup | 0 | 0.76 | 0.9585 | 0.9426 | -0.0159 | [-0.0323, +0.0037] | 0.9981 | -40% |
| `heuristic` | febrl_dedup | 1 | 0.76 | 0.9505 | 0.9329 | -0.0176 | [-0.0347, +0.0021] | 0.9965 | -38% |
| `heuristic` | febrl_dedup | 2 | 0.76 | 0.9580 | 0.9365 | -0.0215 | [-0.0399, -0.0033] | 0.9971 | -55% |
| `heuristic` | febrl_person | 0 | 0.76 | 0.9521 | 0.9932 | +0.0412 | [+0.0127, +0.0735] | 1.0000 | 86% |
| `heuristic` | febrl_person | 1 | 0.76 | 0.9434 | 0.9761 | +0.0327 | [-0.0050, +0.0719] | 1.0000 | 58% |
| `heuristic` | febrl_person | 2 | 0.76 | 0.9460 | 0.9898 | +0.0438 | [+0.0084, +0.0863] | 1.0000 | 81% |
| `heuristic` | fodors_zagat | 0 | 0.76 | 0.1164 | 0.8125 | +0.6961 | [+0.5833, +0.7954] | 0.8421 | 96% |
| `heuristic` | fodors_zagat | 1 | 0.76 | 0.1107 | 0.7812 | +0.6705 | [+0.5538, +0.7708] | 0.7812 | 100% |
| `heuristic` | fodors_zagat | 2 | 0.76 | 0.1121 | 0.7419 | +0.6299 | [+0.4977, +0.7402] | 0.7541 | 98% |
| `heuristic` | walmart_amazon | 0 | 0.74 | 0.0083 | 0.0156 | +0.0073 | [+0.0032, +0.0118] | 0.0163 | 91% |
| `heuristic` | walmart_amazon | 1 | 0.74 | 0.0087 | 0.0145 | +0.0058 | [+0.0023, +0.0097] | 0.0203 | 50% |
| `heuristic` | walmart_amazon | 2 | 0.74 | 0.0091 | 0.0158 | +0.0068 | [+0.0026, +0.0106] | 0.0209 | 57% |
| `heuristic` | wdc_computers | 0 | 0.74 | 0.0272 | 0.0524 | +0.0252 | [+0.0109, +0.0394] | 0.0634 | 70% |
| `heuristic` | wdc_computers | 1 | 0.74 | 0.0291 | 0.0514 | +0.0223 | [+0.0086, +0.0370] | 0.0618 | 68% |
| `heuristic` | wdc_computers | 2 | 0.74 | 0.0283 | 0.0565 | +0.0282 | [+0.0147, +0.0425] | 0.0726 | 64% |
| `sim_cos` | abt_buy | 0 | 0.89 | 0.0666 | 0.2446 | +0.1781 | [+0.1423, +0.2208] | 0.2479 | 98% |
| `sim_cos` | abt_buy | 1 | 0.89 | 0.0654 | 0.2191 | +0.1537 | [+0.1139, +0.1976] | 0.2359 | 90% |
| `sim_cos` | abt_buy | 2 | 0.89 | 0.0677 | 0.3190 | +0.2513 | [+0.2107, +0.2980] | 0.3281 | 97% |
| `sim_cos` | amazon_google | 0 | 0.89 | 0.0171 | 0.4252 | +0.4082 | [+0.3596, +0.4529] | 0.4463 | 95% |
| `sim_cos` | amazon_google | 1 | 0.89 | 0.0169 | 0.4469 | +0.4300 | [+0.3901, +0.4771] | 0.4722 | 94% |
| `sim_cos` | amazon_google | 2 | 0.89 | 0.0170 | 0.4337 | +0.4167 | [+0.3677, +0.4689] | 0.4623 | 94% |
| `sim_cos` | dblp_acm | 0 | 0.90 | 0.2200 | 0.8808 | +0.6609 | [+0.6338, +0.6861] | 0.8974 | 98% |
| `sim_cos` | dblp_acm | 1 | 0.90 | 0.2169 | 0.8802 | +0.6632 | [+0.6369, +0.6885] | 0.8886 | 99% |
| `sim_cos` | dblp_acm | 2 | 0.90 | 0.2157 | 0.8537 | +0.6380 | [+0.6060, +0.6692] | 0.8714 | 97% |
| `sim_cos` | dblp_scholar | 0 | 0.88 | 0.0129 | 0.6377 | +0.6249 | [+0.5784, +0.6664] | 0.6983 | 91% |
| `sim_cos` | dblp_scholar | 1 | 0.88 | 0.0127 | 0.6248 | +0.6121 | [+0.5606, +0.6566] | 0.6951 | 90% |
| `sim_cos` | dblp_scholar | 2 | 0.88 | 0.0127 | 0.6211 | +0.6084 | [+0.5549, +0.6539] | 0.6727 | 92% |
| `sim_cos` | febrl_dedup | 0 | 0.91 | 0.0689 | 0.6559 | +0.5870 | [+0.5511, +0.6217] | 0.8366 | 76% |
| `sim_cos` | febrl_dedup | 1 | 0.91 | 0.0698 | 0.6391 | +0.5693 | [+0.5359, +0.6011] | 0.8033 | 78% |
| `sim_cos` | febrl_dedup | 2 | 0.91 | 0.0692 | 0.6655 | +0.5963 | [+0.5647, +0.6263] | 0.8377 | 78% |
| `sim_cos` | febrl_person | 0 | 0.90 | 0.0688 | 0.7935 | +0.7247 | [+0.6694, +0.7771] | 0.9078 | 86% |
| `sim_cos` | febrl_person | 1 | 0.90 | 0.0692 | 0.8142 | +0.7450 | [+0.6907, +0.7944] | 0.9258 | 87% |
| `sim_cos` | febrl_person | 2 | 0.90 | 0.0680 | 0.8080 | +0.7400 | [+0.6847, +0.7902] | 0.9253 | 86% |
| `sim_cos` | fodors_zagat | 0 | 0.90 | 0.0630 | 0.9254 | +0.8623 | [+0.7811, +0.9310] | 0.9394 | 98% |
| `sim_cos` | fodors_zagat | 1 | 0.90 | 0.0636 | 0.9231 | +0.8594 | [+0.7775, +0.9303] | 0.9524 | 97% |
| `sim_cos` | fodors_zagat | 2 | 0.90 | 0.0653 | 0.9355 | +0.8701 | [+0.7969, +0.9300] | 0.9355 | 100% |
| `sim_cos` | walmart_amazon | 0 | 0.89 | 0.0025 | 0.0517 | +0.0492 | [+0.0409, +0.0585] | 0.0701 | 73% |
| `sim_cos` | walmart_amazon | 1 | 0.89 | 0.0025 | 0.0546 | +0.0521 | [+0.0438, +0.0602] | 0.0766 | 70% |
| `sim_cos` | walmart_amazon | 2 | 0.89 | 0.0025 | 0.0526 | +0.0502 | [+0.0424, +0.0581] | 0.0716 | 73% |
| `sim_cos` | wdc_computers | 0 | 0.89 | 0.0129 | 0.0601 | +0.0471 | [+0.0366, +0.0579] | 0.0757 | 75% |
| `sim_cos` | wdc_computers | 1 | 0.89 | 0.0129 | 0.0567 | +0.0438 | [+0.0349, +0.0536] | 0.0757 | 70% |
| `sim_cos` | wdc_computers | 2 | 0.89 | 0.0131 | 0.0591 | +0.0461 | [+0.0368, +0.0566] | 0.0862 | 63% |

Selection-ineligible and therefore absent above: `tiny_fixture` (fewer than 10 blocked gold pairs held out, so one pair swings the metric). They are still measured -- see the per-cell table.

## 5. The same comparison, end-to-end

If the blocked-gold and all-gold conventions disagreed about whether a constant
helps, that disagreement would be the finding. They are reported side by side so
nobody has to take the denominator on trust:

| family | benchmark | seed | LOBO t | F1@0.50 | F1@LOBO | Δ | 95% CI |
|---|---|---|---|---|---|---|---|
| `heuristic` | abt_buy | 0 | 0.74 | 0.0625 | 0.0083 | -0.0542 | [-0.0666, -0.0418] |
| `heuristic` | abt_buy | 1 | 0.74 | 0.0630 | 0.0145 | -0.0486 | [-0.0635, -0.0327] |
| `heuristic` | abt_buy | 2 | 0.74 | 0.0642 | 0.0226 | -0.0416 | [-0.0584, -0.0255] |
| `heuristic` | amazon_google | 0 | 0.74 | 0.0149 | 0.0463 | +0.0314 | [+0.0157, +0.0493] |
| `heuristic` | amazon_google | 1 | 0.74 | 0.0149 | 0.0506 | +0.0358 | [+0.0208, +0.0524] |
| `heuristic` | amazon_google | 2 | 0.74 | 0.0159 | 0.0504 | +0.0345 | [+0.0158, +0.0567] |
| `heuristic` | dblp_acm | 0 | 0.76 | 0.3971 | 0.6874 | +0.2903 | [+0.2540, +0.3288] |
| `heuristic` | dblp_acm | 1 | 0.76 | 0.3909 | 0.6696 | +0.2786 | [+0.2398, +0.3158] |
| `heuristic` | dblp_acm | 2 | 0.76 | 0.3968 | 0.6523 | +0.2556 | [+0.2179, +0.2906] |
| `heuristic` | dblp_scholar | 0 | 0.73 | 0.0344 | 0.2853 | +0.2509 | [+0.2195, +0.2834] |
| `heuristic` | dblp_scholar | 1 | 0.73 | 0.0332 | 0.2696 | +0.2364 | [+0.2068, +0.2651] |
| `heuristic` | dblp_scholar | 2 | 0.73 | 0.0335 | 0.2748 | +0.2413 | [+0.2148, +0.2683] |
| `heuristic` | febrl_dedup | 0 | 0.76 | 0.9399 | 0.9225 | -0.0174 | [-0.0349, +0.0021] |
| `heuristic` | febrl_dedup | 1 | 0.76 | 0.9361 | 0.9171 | -0.0190 | [-0.0355, -0.0002] |
| `heuristic` | febrl_dedup | 2 | 0.76 | 0.9453 | 0.9228 | -0.0225 | [-0.0410, -0.0050] |
| `heuristic` | febrl_person | 0 | 0.76 | 0.9490 | 0.9899 | +0.0409 | [+0.0121, +0.0712] |
| `heuristic` | febrl_person | 1 | 0.76 | 0.9434 | 0.9761 | +0.0327 | [-0.0050, +0.0736] |
| `heuristic` | febrl_person | 2 | 0.76 | 0.9430 | 0.9865 | +0.0434 | [+0.0064, +0.0836] |
| `heuristic` | fodors_zagat | 0 | 0.76 | 0.1164 | 0.8125 | +0.6961 | [+0.5751, +0.7922] |
| `heuristic` | fodors_zagat | 1 | 0.76 | 0.1107 | 0.7812 | +0.6705 | [+0.5560, +0.7687] |
| `heuristic` | fodors_zagat | 2 | 0.76 | 0.1121 | 0.7419 | +0.6299 | [+0.4990, +0.7366] |
| `heuristic` | walmart_amazon | 0 | 0.74 | 0.0083 | 0.0156 | +0.0073 | [+0.0035, +0.0111] |
| `heuristic` | walmart_amazon | 1 | 0.74 | 0.0087 | 0.0145 | +0.0058 | [+0.0023, +0.0100] |
| `heuristic` | walmart_amazon | 2 | 0.74 | 0.0091 | 0.0158 | +0.0068 | [+0.0028, +0.0108] |
| `heuristic` | wdc_computers | 0 | 0.74 | 0.0271 | 0.0518 | +0.0247 | [+0.0103, +0.0395] |
| `heuristic` | wdc_computers | 1 | 0.74 | 0.0291 | 0.0508 | +0.0218 | [+0.0080, +0.0360] |
| `heuristic` | wdc_computers | 2 | 0.74 | 0.0283 | 0.0560 | +0.0277 | [+0.0150, +0.0430] |
| `sim_cos` | abt_buy | 0 | 0.89 | 0.0666 | 0.2446 | +0.1781 | [+0.1434, +0.2210] |
| `sim_cos` | abt_buy | 1 | 0.89 | 0.0654 | 0.2189 | +0.1535 | [+0.1166, +0.1939] |
| `sim_cos` | abt_buy | 2 | 0.89 | 0.0677 | 0.3186 | +0.2509 | [+0.2072, +0.2969] |
| `sim_cos` | amazon_google | 0 | 0.89 | 0.0171 | 0.4252 | +0.4082 | [+0.3619, +0.4581] |
| `sim_cos` | amazon_google | 1 | 0.89 | 0.0169 | 0.4469 | +0.4300 | [+0.3863, +0.4751] |
| `sim_cos` | amazon_google | 2 | 0.89 | 0.0170 | 0.4337 | +0.4167 | [+0.3685, +0.4642] |
| `sim_cos` | dblp_acm | 0 | 0.90 | 0.2199 | 0.8802 | +0.6603 | [+0.6348, +0.6857] |
| `sim_cos` | dblp_acm | 1 | 0.90 | 0.2169 | 0.8796 | +0.6627 | [+0.6369, +0.6891] |
| `sim_cos` | dblp_acm | 2 | 0.90 | 0.2155 | 0.8514 | +0.6358 | [+0.6042, +0.6680] |
| `sim_cos` | dblp_scholar | 0 | 0.88 | 0.0129 | 0.6349 | +0.6220 | [+0.5792, +0.6602] |
| `sim_cos` | dblp_scholar | 1 | 0.88 | 0.0127 | 0.6189 | +0.6062 | [+0.5552, +0.6488] |
| `sim_cos` | dblp_scholar | 2 | 0.88 | 0.0127 | 0.6162 | +0.6035 | [+0.5556, +0.6510] |
| `sim_cos` | febrl_dedup | 0 | 0.91 | 0.0688 | 0.6383 | +0.5694 | [+0.5348, +0.6043] |
| `sim_cos` | febrl_dedup | 1 | 0.91 | 0.0697 | 0.6254 | +0.5557 | [+0.5222, +0.5860] |
| `sim_cos` | febrl_dedup | 2 | 0.91 | 0.0691 | 0.6533 | +0.5842 | [+0.5527, +0.6161] |
| `sim_cos` | febrl_person | 0 | 0.90 | 0.0688 | 0.7903 | +0.7215 | [+0.6683, +0.7747] |
| `sim_cos` | febrl_person | 1 | 0.90 | 0.0692 | 0.8142 | +0.7450 | [+0.6903, +0.7951] |
| `sim_cos` | febrl_person | 2 | 0.90 | 0.0680 | 0.8048 | +0.7368 | [+0.6808, +0.7894] |
| `sim_cos` | fodors_zagat | 0 | 0.90 | 0.0630 | 0.9254 | +0.8623 | [+0.7775, +0.9273] |
| `sim_cos` | fodors_zagat | 1 | 0.90 | 0.0636 | 0.9231 | +0.8594 | [+0.7745, +0.9313] |
| `sim_cos` | fodors_zagat | 2 | 0.90 | 0.0653 | 0.9355 | +0.8701 | [+0.7985, +0.9303] |
| `sim_cos` | walmart_amazon | 0 | 0.89 | 0.0025 | 0.0517 | +0.0492 | [+0.0407, +0.0573] |
| `sim_cos` | walmart_amazon | 1 | 0.89 | 0.0025 | 0.0546 | +0.0521 | [+0.0445, +0.0602] |
| `sim_cos` | walmart_amazon | 2 | 0.89 | 0.0025 | 0.0526 | +0.0502 | [+0.0425, +0.0580] |
| `sim_cos` | wdc_computers | 0 | 0.89 | 0.0129 | 0.0598 | +0.0469 | [+0.0359, +0.0573] |
| `sim_cos` | wdc_computers | 1 | 0.89 | 0.0129 | 0.0565 | +0.0436 | [+0.0351, +0.0532] |
| `sim_cos` | wdc_computers | 2 | 0.89 | 0.0131 | 0.0589 | +0.0459 | [+0.0361, +0.0566] |

## 6. Is it actually *one* constant?

This is the question that decides whether a "default" is a default or a
per-dataset tuning wearing a constant's clothes:

| family | eligible benchmarks | in-sample argmax | LOBO min | LOBO max | spread |
|---|---|---|---|---|---|
| `heuristic` | 9 | 0.74 | 0.73 | 0.76 | 0.03 |
| `sim_cos` | 9 | 0.90 | 0.88 | 0.91 | 0.03 |

And the decision framed as a ladder — what `0.5` gets you, what a free shipped
constant gets you, what labels get you, and what is left on the table after all
three:

| family | benchmark | F1@0.50 | F1@LOBO | F1@derived (labels) | oracle F1 |
|---|---|---|---|---|---|
| `heuristic` | abt_buy | 0.0632 | 0.0151 | 0.0734 | 0.0774 |
| `heuristic` | amazon_google | 0.0152 | 0.0491 | 0.0229 | 0.0836 |
| `heuristic` | dblp_acm | 0.3952 | 0.6709 | 0.7983 | 0.8107 |
| `heuristic` | dblp_scholar | 0.0337 | 0.2791 | 0.0319 | 0.2874 |
| `heuristic` | febrl_dedup | 0.9556 | 0.9373 | 0.9860 | 0.9973 |
| `heuristic` | febrl_person | 0.9472 | 0.9864 | 1.0000 | 1.0000 |
| `heuristic` | fodors_zagat | 0.1131 | 0.7786 | 0.5448 | 0.7925 |
| `heuristic` | walmart_amazon | 0.0087 | 0.0153 | 0.0109 | 0.0192 |
| `heuristic` | wdc_computers | 0.0282 | 0.0534 | 0.0218 | 0.0659 |
| `sim_cos` | abt_buy | 0.0666 | 0.2609 | 0.2374 | 0.2706 |
| `sim_cos` | amazon_google | 0.0170 | 0.4353 | 0.2587 | 0.4602 |
| `sim_cos` | dblp_acm | 0.2175 | 0.8716 | 0.8721 | 0.8858 |
| `sim_cos` | dblp_scholar | 0.0127 | 0.6279 | 0.3269 | 0.6887 |
| `sim_cos` | febrl_dedup | 0.0693 | 0.6535 | 0.7313 | 0.8258 |
| `sim_cos` | febrl_person | 0.0687 | 0.8053 | 0.8836 | 0.9196 |
| `sim_cos` | fodors_zagat | 0.0640 | 0.9280 | 0.9377 | 0.9424 |
| `sim_cos` | walmart_amazon | 0.0025 | 0.0530 | 0.0330 | 0.0728 |
| `sim_cos` | wdc_computers | 0.0130 | 0.0586 | 0.0457 | 0.0792 |

Seed-mean per benchmark, never pooled across benchmarks. `F1@derived` is what a user **with full labels** gets (PR #250's seam); `F1@LOBO` is what a user with **no labels at all** would get from a shipped constant.

## 7. Does a `sim_cos` constant even mean anything? (the checkpoint test)

A `heuristic` cut is a cut on rapidfuzz's ratio, which is fixed. A `sim_cos` cut
is a cut on a **cosine scale**, and the scale belongs to the *encoder*, not to
the family tag — two models can both emit "cosine similarity" and disagree about
what `0.9` means.

This matters concretely here, and it is easy to miss: every benchmark loader in
this repo pins `all-MiniLM-L6-v2` for its blocker, but `DEFAULT_EMBEDDING_MODEL`
moved to `intfloat/e5-base-v2` (the embedder-ladder PR). So the constant §4
selects is a constant **for MiniLM cosine**, while the number it would be written
into is read by users running e5. Assuming it transfers is exactly the
"documented prior treated as a measured finding" error this study was written to
avoid, so it is measured instead: the identical protocol, re-run with the
blocker re-pointed at the shipped default (`--embedder`), and the baseline's
constant graded on those cells.

| family | benchmark | seed | baseline t | F1@0.50 | F1@baseline t | Δ | 95% CI | variant's own argmax |
|---|---|---|---|---|---|---|---|---|
| `sim_cos` | abt_buy | 0 | 0.90 | 0.0688 | 0.0704 | +0.0017 | [+0.0010, +0.0024] | 0.96 |
| `sim_cos` | abt_buy | 1 | 0.90 | 0.0684 | 0.0718 | +0.0034 | [+0.0026, +0.0045] | 0.96 |
| `sim_cos` | abt_buy | 2 | 0.90 | 0.0712 | 0.0731 | +0.0018 | [+0.0011, +0.0028] | 0.96 |
| `sim_cos` | amazon_google | 0 | 0.90 | 0.0166 | 0.0217 | +0.0051 | [+0.0042, +0.0061] | 0.96 |
| `sim_cos` | amazon_google | 1 | 0.90 | 0.0165 | 0.0210 | +0.0046 | [+0.0038, +0.0054] | 0.96 |
| `sim_cos` | amazon_google | 2 | 0.90 | 0.0165 | 0.0209 | +0.0044 | [+0.0037, +0.0052] | 0.96 |
| `sim_cos` | dblp_acm | 0 | 0.90 | 0.2167 | 0.2167 | +0.0000 | [+0.0000, +0.0000] | 0.96 |
| `sim_cos` | dblp_acm | 1 | 0.90 | 0.2190 | 0.2190 | +0.0000 | [+0.0000, +0.0000] | 0.96 |
| `sim_cos` | dblp_acm | 2 | 0.90 | 0.2178 | 0.2178 | +0.0000 | [+0.0000, +0.0000] | 0.96 |
| `sim_cos` | dblp_scholar | 0 | 0.90 | 0.0123 | 0.0127 | +0.0004 | [+0.0003, +0.0006] | 0.96 |
| `sim_cos` | dblp_scholar | 1 | 0.90 | 0.0122 | 0.0126 | +0.0004 | [+0.0003, +0.0005] | 0.96 |
| `sim_cos` | dblp_scholar | 2 | 0.90 | 0.0122 | 0.0127 | +0.0004 | [+0.0003, +0.0006] | 0.96 |
| `sim_cos` | febrl_dedup | 0 | 0.90 | 0.0726 | 0.1071 | +0.0346 | [+0.0297, +0.0397] | 0.96 |
| `sim_cos` | febrl_dedup | 1 | 0.90 | 0.0736 | 0.1054 | +0.0318 | [+0.0276, +0.0364] | 0.96 |
| `sim_cos` | febrl_dedup | 2 | 0.90 | 0.0735 | 0.1081 | +0.0346 | [+0.0302, +0.0394] | 0.96 |
| `sim_cos` | febrl_person | 0 | 0.90 | 0.0702 | 0.1357 | +0.0655 | [+0.0548, +0.0790] | 0.96 |
| `sim_cos` | febrl_person | 1 | 0.90 | 0.0716 | 0.1612 | +0.0896 | [+0.0754, +0.1076] | 0.96 |
| `sim_cos` | febrl_person | 2 | 0.90 | 0.0708 | 0.1473 | +0.0766 | [+0.0648, +0.0937] | 0.96 |
| `sim_cos` | fodors_zagat | 0 | 0.90 | 0.0653 | 0.0653 | +0.0000 | [+0.0000, +0.0000] | 0.96 |
| `sim_cos` | fodors_zagat | 1 | 0.90 | 0.0679 | 0.0679 | +0.0000 | [+0.0000, +0.0000] | 0.96 |
| `sim_cos` | fodors_zagat | 2 | 0.90 | 0.0670 | 0.0670 | +0.0000 | [+0.0000, +0.0000] | 0.96 |
| `sim_cos` | tiny_fixture | 0 | 0.90 | 0.5000 | 1.0000 | +0.5000 | [+0.0000, +0.5000] | 0.96 |
| `sim_cos` | tiny_fixture | 1 | 0.90 | 0.5000 | 0.5000 | +0.0000 | [+0.0000, +0.0000] | 0.96 |
| `sim_cos` | tiny_fixture | 2 | 0.90 | 0.5000 | 0.5000 | +0.0000 | [+0.0000, +0.0000] | 0.96 |
| `sim_cos` | walmart_amazon | 0 | 0.90 | 0.0026 | 0.0026 | +0.0000 | [+0.0000, +0.0000] | 0.96 |
| `sim_cos` | walmart_amazon | 1 | 0.90 | 0.0026 | 0.0026 | +0.0000 | [+0.0000, +0.0000] | 0.96 |
| `sim_cos` | walmart_amazon | 2 | 0.90 | 0.0026 | 0.0026 | +0.0000 | [+0.0000, +0.0000] | 0.96 |
| `sim_cos` | wdc_computers | 0 | 0.90 | 0.0136 | 0.0136 | +0.0000 | [+0.0000, +0.0000] | 0.96 |
| `sim_cos` | wdc_computers | 1 | 0.90 | 0.0134 | 0.0134 | +0.0000 | [+0.0000, +0.0000] | 0.96 |
| `sim_cos` | wdc_computers | 2 | 0.90 | 0.0138 | 0.0138 | +0.0000 | [+0.0000, +0.0000] | 0.96 |

Variant checkpoint(s): intfloat/e5-base-v2.

Pre-registered rule: **transfers iff, on the variant checkpoint, (1) no benchmark is significantly worse than 0.5 in a majority of its seeds and (2) the median per-benchmark mean delta-F1 is > 0**.

- `sim_cos`: **TRANSFERS**. Median per-benchmark Δ on the variant checkpoint +0.0004; significantly worse than 0.5 on no benchmark (0 of 27 eligible cells). Diagnostic, not a veto: the argmax moves 0.06 across checkpoints (0.90 -> 0.96).

### The same test in the other direction

Run one way only, a transfer test is a cherry-pick. "Does *my* constant survive
elsewhere?" is the flattering question; the one that actually constrains the
choice is "would the *other* checkpoint's constant have been safe **here**?" The
e5 run selects its own constant by the same leave-one-out procedure, so that
question is free to answer — and it is the one that decides what ships:

| family | benchmark | seed | baseline t | F1@0.50 | F1@baseline t | Δ | 95% CI | variant's own argmax |
|---|---|---|---|---|---|---|---|---|
| `sim_cos` | abt_buy | 0 | 0.96 | 0.0666 | 0.0104 | -0.0562 | [-0.0688, -0.0416] | 0.90 |
| `sim_cos` | abt_buy | 1 | 0.96 | 0.0654 | 0.0103 | -0.0551 | [-0.0661, -0.0415] | 0.90 |
| `sim_cos` | abt_buy | 2 | 0.96 | 0.0677 | 0.0267 | -0.0410 | [-0.0620, -0.0174] | 0.90 |
| `sim_cos` | amazon_google | 0 | 0.96 | 0.0171 | 0.3375 | +0.3205 | [+0.2635, +0.3804] | 0.90 |
| `sim_cos` | amazon_google | 1 | 0.96 | 0.0169 | 0.3189 | +0.3020 | [+0.2474, +0.3556] | 0.90 |
| `sim_cos` | amazon_google | 2 | 0.96 | 0.0170 | 0.3748 | +0.3578 | [+0.2936, +0.4179] | 0.90 |
| `sim_cos` | dblp_acm | 0 | 0.96 | 0.2200 | 0.7661 | +0.5461 | [+0.5171, +0.5755] | 0.90 |
| `sim_cos` | dblp_acm | 1 | 0.96 | 0.2169 | 0.7210 | +0.5041 | [+0.4731, +0.5352] | 0.90 |
| `sim_cos` | dblp_acm | 2 | 0.96 | 0.2157 | 0.7498 | +0.5341 | [+0.5013, +0.5629] | 0.90 |
| `sim_cos` | dblp_scholar | 0 | 0.96 | 0.0129 | 0.4516 | +0.4387 | [+0.3814, +0.5000] | 0.90 |
| `sim_cos` | dblp_scholar | 1 | 0.96 | 0.0127 | 0.4556 | +0.4430 | [+0.3925, +0.5031] | 0.90 |
| `sim_cos` | dblp_scholar | 2 | 0.96 | 0.0127 | 0.4454 | +0.4327 | [+0.3773, +0.4926] | 0.90 |
| `sim_cos` | febrl_dedup | 0 | 0.96 | 0.0689 | 0.4793 | +0.4103 | [+0.3746, +0.4456] | 0.90 |
| `sim_cos` | febrl_dedup | 1 | 0.96 | 0.0698 | 0.4731 | +0.4033 | [+0.3647, +0.4419] | 0.90 |
| `sim_cos` | febrl_dedup | 2 | 0.96 | 0.0692 | 0.5071 | +0.4379 | [+0.3983, +0.4771] | 0.90 |
| `sim_cos` | febrl_person | 0 | 0.96 | 0.0688 | 0.6330 | +0.5642 | [+0.4862, +0.6315] | 0.90 |
| `sim_cos` | febrl_person | 1 | 0.96 | 0.0692 | 0.6486 | +0.5795 | [+0.5027, +0.6495] | 0.90 |
| `sim_cos` | febrl_person | 2 | 0.96 | 0.0680 | 0.6267 | +0.5587 | [+0.4800, +0.6306] | 0.90 |
| `sim_cos` | fodors_zagat | 0 | 0.96 | 0.0630 | 0.8667 | +0.8036 | [+0.6949, +0.8913] | 0.90 |
| `sim_cos` | fodors_zagat | 1 | 0.96 | 0.0636 | 0.8421 | +0.7785 | [+0.6641, +0.8744] | 0.90 |
| `sim_cos` | fodors_zagat | 2 | 0.96 | 0.0653 | 0.8214 | +0.7561 | [+0.6316, +0.8517] | 0.90 |
| `sim_cos` | tiny_fixture | 0 | 0.96 | 0.5000 | 1.0000 | +0.5000 | [+0.0000, +0.5000] | 0.90 |
| `sim_cos` | tiny_fixture | 1 | 0.96 | 0.5000 | 1.0000 | +0.5000 | [+0.0000, +0.5000] | 0.90 |
| `sim_cos` | tiny_fixture | 2 | 0.96 | 0.5000 | 1.0000 | +0.5000 | [+0.0000, +0.5000] | 0.90 |
| `sim_cos` | walmart_amazon | 0 | 0.96 | 0.0025 | 0.0630 | +0.0605 | [+0.0409, +0.0849] | 0.90 |
| `sim_cos` | walmart_amazon | 1 | 0.96 | 0.0025 | 0.0590 | +0.0565 | [+0.0393, +0.0748] | 0.90 |
| `sim_cos` | walmart_amazon | 2 | 0.96 | 0.0025 | 0.0614 | +0.0589 | [+0.0415, +0.0779] | 0.90 |
| `sim_cos` | wdc_computers | 0 | 0.96 | 0.0129 | 0.0673 | +0.0544 | [+0.0314, +0.0790] | 0.90 |
| `sim_cos` | wdc_computers | 1 | 0.96 | 0.0129 | 0.0694 | +0.0565 | [+0.0360, +0.0784] | 0.90 |
| `sim_cos` | wdc_computers | 2 | 0.96 | 0.0131 | 0.0830 | +0.0699 | [+0.0438, +0.0971] | 0.90 |

Variant checkpoint(s): all-MiniLM-L6-v2 (pin).

Pre-registered rule: **transfers iff, on the variant checkpoint, (1) no benchmark is significantly worse than 0.5 in a majority of its seeds and (2) the median per-benchmark mean delta-F1 is > 0**.

- `sim_cos`: **DOES NOT TRANSFER**. Median per-benchmark Δ on the variant checkpoint +0.4172; significantly worse than 0.5 on abt_buy (3 of 27 eligible cells). Diagnostic, not a veto: the argmax moves 0.06 across checkpoints (0.96 -> 0.90).

### Reading both directions together

The two runs agree on the finding that matters and disagree on the number, and
both halves are load-bearing:

- **They agree that `0.5` is the wrong order of magnitude for a cosine.** On
  *both* checkpoints, on every benchmark, cutting at `0.5` leaves F1 in the low
  hundredths. Whatever the right constant is, it is nowhere near `0.5`.
- **They disagree about how high.** Each checkpoint's own leave-one-out
  selection is internally rock-solid — and they land in different places. That is
  not noise between two unstable estimates; it is two stable estimates of two
  different scales, which is precisely what "the cosine scale belongs to the
  encoder" predicts.
- **The asymmetry is the verdict.** The lower constant is safe on both
  checkpoints. The higher one is not: carried back, it lands **significantly
  below `0.5` on `abt_buy`, on every seed, with intervals entirely below zero** —
  the same clause-(2) failure that disqualifies `heuristic` in §8. So the higher
  constant is not "the better number measured on the more relevant checkpoint";
  it is a number with a demonstrated victim.
- **What the safe constant is worth is honestly asymmetric.** On the pinned
  checkpoint it is transformative. On the shipped default it is *safe but close
  to inert* — several benchmarks score identically to `0.5`, because on that
  encoder almost every candidate pair already sits above the cut. Its Δ there is
  positive and never negative, which is what the pre-registered rule asks, but
  the honest word for the magnitude is "small", and the table says so rather
  than the prose.

That combination — a real floor, safe everywhere, worth a great deal on one
scale and little on another — is exactly what a *default* should be, and exactly
why it is not a substitute for tuning. A shipped constant's job is to stop the
out-of-the-box path being catastrophically wrong. It is not to be optimal on
your encoder, and this table is the evidence that it cannot be.

## 8. Verdict

Pre-registered rule: **ship iff (1) LOBO constants span <= 0.05, (2) no eligible benchmark is significantly worse (95% CI entirely < 0) in a majority of its seeds, and (3) the median per-benchmark mean delta-F1 is > 0**.

| family | LOBO spread | benchmarks significantly worse | median per-benchmark Δ | verdict | constant |
|---|---|---|---|---|---|
| `heuristic` | 0.03 | abt_buy | +0.0339 | **DO NOT SHIP** | 0.74 |
| `sim_cos` | 0.03 | none | +0.5842 | **SHIP** | 0.90 |

`constant` is the in-sample argmax -- the value to ship **only** when the verdict is SHIP. It is deliberately printed for both outcomes so a DO NOT SHIP row cannot be read as 'no number was found'; the number exists, it just does not generalize.

### Reading the `sim_cos` row: the incumbent was the bug

For cosine similarity the result is not marginal. In the leave-one-out table
(§4), **every selection-eligible cell improves, and every 95% interval sits
entirely above zero** — there is no benchmark, and no seed, where the constant
loses. The `capture` column shows it recovers most of the headroom an *oracle
with test labels* could reach.

The reason is visible in the `F1@0.50` column: `0.5` is not a mildly
mistuned cut on a cosine scale, it is a catastrophic one. Normalized embeddings
put almost every candidate pair above `0.5`, so the threshold accepts nearly
everything the blocker proposed and precision collapses. Several benchmarks score
in the low hundredths. This is less "we found a better constant" than "the shipped
constant was never on the right scale", which is exactly the failure mode a
per-family default exists to prevent.

That is also why this is the family where wiring the default mattered most: the
same `0.5` is defensible for a `heuristic` score and indefensible for a cosine,
and a single global constant cannot be both.

**What ships is the *safe* constant, not the best one.** §7 measured a second
checkpoint and the two directions are not symmetric: the constant selected here
is harmless on the other encoder, while that encoder's own (higher) constant
predictably damages `abt_buy` when carried back. Where the two disagree this
study ships the one with no demonstrated victim, which is the same standard
clause (2) applies to `heuristic` — applied to our own preferred answer rather
than only to the one we were ready to reject.

### Reading the `heuristic` row: it is not instability

The obvious guess — that rapidfuzz has no single constant because every dataset
wants a different one — is **wrong here, and the table says so**. Its
leave-one-out constants agree closely (§6): drop any one dataset and the choice
barely moves. PR #250's derived cuts spanned 0.174–0.695, which made
non-generalization the expected outcome; the *fixed*-constant question turns out
to have a different answer than the *derived*-cut question did.

What kills it is clause (2), not clause (1). One constant, chosen without seeing
the dataset it is graded on, **helps on most of the portfolio and reliably
damages `abt_buy`** — on every seed, with the 95% interval entirely below zero.
That is not noise and it is not a tie; it is a predictable loss for a whole class
of data (short, noisy product titles where a high string-similarity bar
throws away true matches faster than it removes false ones).

**The conventions agree, so this does not rest on the denominator.** The
end-to-end (all-gold) table in §5 shows `abt_buy` losing by the same margins with
intervals just as clearly below zero, and `febrl_dedup`'s mild losses harden
there rather than softening. If anything the end-to-end view is *less* favourable
to shipping a constant than the blocked-gold view the verdict is computed on.

A default that improves the median while predictably harming a known dataset
class is not a default — it is a recommendation with an undisclosed victim.
Clause (2) was pre-registered precisely so this case could not be argued away
after the fact by pointing at a healthy median. So `heuristic` keeps `0.5`, and
the honest guidance for it is unchanged: **derive the cut from labels** (PR
#250's seam, `derive_threshold`).

**But read the ladder in §6 carefully before concluding that labels simply
dominate — they do not, and an earlier draft of this sentence said they did.**
Compare its `F1@LOBO` and `F1@derived` columns: the *rejected* constant scores
**higher than the label-derived cut on 5 of the 9 benchmarks**, for both
families. Neither approach dominates this portfolio. What makes deriving the
right advice here is not that it wins on average — it is *where* it wins: on
`abt_buy`, the single benchmark whose harm vetoes the constant, the derived cut
beats both the constant and `0.5`. Labels buy you the case a constant cannot
cover, which is a different and more useful claim than "labels are better".

Note also what the ladder shows about the incumbent: on the harder product
benchmarks `0.5` is not a mild compromise, it is close to worthless. The reason
to leave it in place is that nothing free and *safe* beats it, not that it is
good.

## 9. What was measured vs. what was inferred

**Measured** (this artifact, held-out, multi-seed, with intervals):

- Per-cell F1 at all 99 grid constants, at `0.5`, at the label-derived cut, and
  at the exact oracle, for the `heuristic` and `sim_cos` families across the
  loadable portfolio.
- The LOBO-selected constant's Δ-F1 against `0.5` per benchmark and seed, with
  95% paired cluster-bootstrap intervals.
- The stability of that selection under dropping one dataset.
- Whether a `sim_cos` constant survives a change of embedding checkpoint (§7),
  measured on `intfloat/e5-base-v2` — the library's own default — under the same
  protocol, **in both directions**: this study's constant graded on that
  checkpoint's cells, and that checkpoint's own leave-one-out constant graded
  back on these.

**Inferred, not measured** — flagged because it would be easy to read this
document as covering it:

- **`prob_llm`'s `0.7`.** The registry has declared `0.7` for the LLM families
  since the field was introduced; wiring `threshold=None` to the registry makes
  `VectorLLMCascade` actually *use* it, changing its out-of-the-box cut from
  `0.5` to `0.7`. **That number is not measured here** — it is the value the
  codebase already declared, now honoured instead of ignored. Sweeping it costs
  paid completions on every benchmark and is a separate study.
- **`prob_fs` / `prob_rf`.** Untouched, for the reason in §2.
- **The emitted `score_type` tags do not match the families used to resolve the
  defaults**, and review caught this study leaning on the tag as if it did.
  `RetrieveRerank` cuts a *cross-encoder's* score but is tagged `heuristic`
  (`Rerank`'s `out_space` default). Worse for this document: the `Retrieve` op
  stamps its rows `heuristic` too (`resources/retrieve.py:174`, hard-coded), even
  though they are cosines and `Retrieve` resolves its default against `sim_cos`.
  The constant is justified by **what the score is** — these really are embedding
  cosines, which is what was swept — not by the tag, and an earlier draft of the
  `Retrieve` docstring wrongly claimed the op emitted `sim_cos`.

  So one coarse tag currently covers three different scales (rapidfuzz ratios,
  cross-encoder scores, cosines). Nothing here justifies a *cross-encoder*
  constant, and none was set. Fixing the tags means changing a resource's emitted
  contract, which is a separate change from a threshold default; it is recorded
  here and in both class docstrings rather than quietly worked around.

## 10. Reproducing

```bash
OMP_NUM_THREADS=1 uv run --env-file .env python \
    examples/research/threshold_constant_sweep.py
```

`OMP_NUM_THREADS=1` is not optional on macOS: faiss plus a torch model in one
process deadlock silently without it — no error, no output, no CPU, forever.
(`KMP_DUPLICATE_LIB_OK` suppresses the OpenMP *abort*, not the *deadlock*.) The
module sets it via `os.environ.setdefault` before any import that could pull
either in, so a run is protected however it was launched.

Each benchmark runs in its own subprocess to bound torch's non-releasing MPS
allocator, and every benchmark is checkpointed — an interrupted sweep continues
with `--resume`. To re-print every table from the committed artifact without
measuring anything:

```bash
uv run --env-file .env python examples/research/threshold_constant_sweep.py \
    --render examples/research/results/threshold_constant_sweep.json
```

The §7 checkpoint variant, and the two transfer directions it feeds:

```bash
OMP_NUM_THREADS=1 uv run --env-file .env python \
    examples/research/threshold_constant_sweep.py \
    --methods embedding_cosine --embedder intfloat/e5-base-v2 \
    --out examples/research/results/threshold_constant_sweep.e5.json

# --compare BASELINE VARIANT: select on BASELINE, grade on VARIANT's cells.
uv run --env-file .env python examples/research/threshold_constant_sweep.py \
    --compare examples/research/results/threshold_constant_sweep.json \
              examples/research/results/threshold_constant_sweep.e5.json
```

To regenerate this whole document — every table above — from the committed
artifacts:

```bash
uv run --env-file .env python tools/render_threshold_constant_writeup.py
```
