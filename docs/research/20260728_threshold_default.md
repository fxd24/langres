# The threshold-default study — should `fit(derive_threshold=True)` be the default?

**Date:** 2026-07-28 · **Cost:** $0 in spend (rapidfuzz / embedding_cosine, no paid
call) · **Harness:** `examples/research/threshold_default_study.py` ·
**Raw data:** `examples/research/results/threshold_default_study.json`

> Not dependency-free or offline on a cold cache: blocking is always the
> benchmark's own `VectorBlocker`, so a run needs the `[semantic]` extra and
> downloads `all-MiniLM-L6-v2` once; `derive_threshold` needs `[trained]`
> (scikit-learn). No paid model is ever called.

PR #241 landed `ERModel.fit(..., derive_threshold=True)`: it derives the match cut
from labels (Youden's J) and **races it against the incumbent** on `train`,
keeping the incumbent unless the candidate *strictly* wins. It is off by default.
Meanwhile the front door hard-codes `threshold: float = 0.5` in six places
(`architectures/fuzzy_string.py`, `architectures/retrieval.py` ×4,
`architectures/vector_llm_cascade.py`), so out of the box langres resolves at a
constant nobody measured.

**Should the flag default to `True`?** This measures it. Nothing here changes a
default — following the precedent set by the closure diagnostic: *the measurement
is the deliverable; changing a default is a separate PR.*

---

## 0. TL;DR

**54 cells** — 9 loadable benchmarks × 2 $0 scorers × seeds 0/1/2 — every number
graded on a corpus-disjoint held-out split.

- **(a) `fit(derive_threshold=True)` beats `0.5` almost everywhere.** 45 cells
  improve, 7 tie exactly (the race declined, so the threshold did not move),
  **2 lose**. The two losses are both `tiny_fixture` + `rapidfuzz`, whose
  held-out corpus is **3 records containing 1 gold pair**, so *recall* is binary
  and one pair decides the whole metric. (F1 itself is not binary — false
  positives still move precision, and the artifact shows it: the
  `tiny_fixture`/`embedding_cosine` rows sit at **0.5000** on that same single
  gold pair. What makes these cells uninformative is the sample size, not a
  two-valued metric.) On the **8 real benchmarks the shipped outcome never loses
  at any seed for either scorer**, and the wins are not marginal: +0.42
  (dblp_acm/rapidfuzz), +0.66 (dblp_acm/cosine), +0.82 (febrl_person/cosine),
  +0.86 (fodors_zagat/cosine) — all at **seed 0**, so the four sit on one basis.
  (The largest single cell anywhere is +0.8887, fodors_zagat/cosine at seed 1.)

  **These are outcomes of the whole `fit` seam — derive *then* race — not of the
  derived cut alone**, and the difference is the entire point of (b). Scored on
  its own, the raw Youden candidate is better in 45 cells, tied in 1, and
  **worse in 8** — six of them on real benchmarks (`dblp_scholar`/`rapidfuzz`
  ×3, `wdc_computers`/`rapidfuzz` ×3). Those six appear above as ties only
  because the race refused them. A derive-and-apply implementation would have
  shipped all six as regressions.
- **(b) The race earns its seat.** It declined 7 times, and in **6 of those 7 the
  derived cut really was worse held-out** (the 7th was an exact tie, which the
  tie-keeps-incumbent rule handles). It has never once declined a cut that would
  have helped. It is not infallible in the other direction — it *kept* two cuts
  that lost — but both are the 1-gold-pair fixture, and #241 already documents
  that selecting on `train` "bounds the risk, it does not remove it". This is
  that residual, observed.
- **(c) So: yes to deriving — but "flip the default" is not a one-character
  change, and this PR does not make it.** `derive_threshold: bool = False →
  True` makes **three verified call shapes raise** — the plain `fit(records)`
  no-op, `fit(records, labels=…)`, and every `fit(…, method=…)`; the repo's own
  suite would go red on the first test. The measured answer and the API change
  are different decisions — see §3.
- **(d) The most actionable finding is not about the flag at all.** For
  `embedding_cosine` the derived cut lands in **0.809–0.945 (median 0.863) across
  all nine datasets** — a spread of 0.14. `0.5` is not merely unmeasured there,
  it is *nowhere near* right, and a per-family default in that band would help
  every user **with no labels at all**. For `rapidfuzz` the same spread is
  **0.174–0.695** (0.52, a 4× range), so a shared constant is a much weaker bet
  there — though this study cannot rule one out, having measured F1 only at `0.5`
  and at each dataset's own derived cut (§5.2). One number is unlikely to serve
  both families — which is exactly what `MethodSpec.default_threshold` already
  exists to express, and exactly what the six architectures bypass.
- **(e) A separate defect, found by trying the obvious design first: the split
  trap.** `align_pairs(split=0.3)` returns an **empty `valid` in 26 of 27
  (benchmark, seed) rows** on this label set — 0 held-out pairs, not 30 %. So
  `FitReport.metrics` / `ThresholdCandidate.held_out_f1` are not trustworthy on a
  dense label set, and nothing warns. §4.

---

## 1. Method

For every registered, **loadable** benchmark (9 of the 10 entries; `opensanctions`
is external-only and never vendored), for two $0 scorers and three seeds:

1. Split the corpus with the benchmark's own `Benchmark.split(seed=…)` —
   `stratified_corpus_split`, which assigns **whole gold clusters** to one side,
   so no entity and no match pair straddles the boundary (70 / 30).
2. Block the **train** corpus, label *every* blocked candidate by closed-world
   gold membership, and call
   `fit(train_data, pairs=labels, split=None, derive_threshold=True)` on a model
   constructed at the front door's `0.5`.
3. Read the race's own verdict off `fit_report_.threshold_fit`: the incumbent
   cut, the derived cut, and `source` (`derived` = kept, `declined` = incumbent
   held).
4. Block and score the **held-out** corpus once, and grade *both* cuts on it with
   `metrics.classify_pairs` — the same function `fit` itself grades with.

### Why the held-out estimate comes from a disjoint corpus, not from `split=0.3`

The obvious design — label every blocked candidate and let
`fit(pairs=…, split=0.3)` hold out 30 % — was tried first and **degenerates**.
`align_pairs`' entity-disjoint split assigns whole union-find components, and a
k-NN candidate graph over a real corpus is essentially *one* component, so there
is nothing to split. See §4; the harness records what that split *would* have
held out in every cell, so this is data rather than a claim.

**Selection never reads `valid`** — `_select_threshold` scores both candidate
cuts on `train` only. But `split=` is not therefore inert: `resolver.py:1276`
builds `aligned` *before* selection, so a nonempty `valid` set removes those
pairs from `train` and the cut is then derived from a smaller sample. `split=`
changes *which pairs the race sees*, not *which side it grades on*. (An earlier
draft of this section claimed `split=` "only changes what `fit` reports". That is
true only in the degenerate case where `valid` comes back empty — which, per §4,
is 26 of 27 rows here, so it happened to hold for every number in this study. It
is not true in general, and the distinction matters for anyone reading this as
guidance rather than as a description of this run.)

Passing `split=None` therefore gives the race the *whole* label set — exactly
what a user handing over their labels would give it — and grading on the disjoint
corpus is a *stronger* held-out estimate than the one `fit` would have printed.

### Choices that make this the strong form of the test

- **Labels are every blocked candidate on the train corpus** — full supervision
  over the exact distribution the cut operates on. This is the most generous
  labelling budget deriving could ask for; a cut that cannot win here will not
  win from a handful of corrections.
- **The incumbent is `0.5`**, the constant the six architectures hard-code — not
  each benchmark's tuned operating point. The question is about the *default*.
- **`incumbent F1` / `derived F1` restrict gold to pairs blocking proposed**,
  which is exactly the convention `fit`'s own `held_out_f1` uses (its gold comes
  from the aligned valid candidates). The JSON also carries the end-to-end
  variant (`*_f1_all_gold`), which charges blocking's misses to recall.
- **Seeds 0, 1, 2**, reported per cell — never averaged. An average hides the one
  cell that motivated the race.

### What this does *not* cover

- **`weighted_average` was not raced.** The two $0 scorers here are `rapidfuzz`
  (a heuristic string family) and `embedding_cosine` (a cosine family) — the two
  score families the six defaults sit on. But `FuzzyString`'s matcher is
  literally `WeightedAverageMatcher`, i.e. the third $0 method
  (`weighted_average`), not `rapidfuzz`. The two score the *same* fields with the
  same weights (`methods.py` builds both off `StringComparator.from_schema`) and
  both carry `score_type="heuristic"`, so `rapidfuzz` is a close stand-in — but
  it is a stand-in, and the arm covering `FuzzyString`'s exact matcher is
  missing. Re-run with `--methods weighted_average` to close it.
- **No paid method.** `VectorLLMCascade` and `RetrieveLLM`/`RetrieveRerankLLM`
  also default to `0.5`, on an LLM-derived score family the method registry
  already defaults to `0.7`. Nothing here measures them; that is a paid run.
- **Every benchmark in the registry is *linkage*.** There is no registered dedup
  benchmark (see `20260727_portfolio_annotation.md`), so none of this speaks to
  single-source deduplication.

**Status: this is an exploratory run, published as such.** `docs/REPRODUCIBILITY.md`
reserves a *clean* claim for one carrying committed source plus the relevant lock,
environment, dataset/test and model revisions; the tracked artifact here records
cell metrics only. In particular the blocker loads the mutable alias
`all-MiniLM-L6-v2` without pinning a resolved Hub revision, so if a later re-run
disagrees with this table there is no way to separate model drift from code, data
or environment drift. That is a real limit on what these numbers can settle, and
it is stated rather than papered over — the same doc says exploratory runs are
useful but must not be relabelled clean after the fact. It is proportionate here
because the conclusion rests on a direction — the derived cut never losing on any
of the 8 real benchmarks, at any seed, for either scorer — plus a two-cell
exception with an identified cause, not on any single decimal being reproducible.

---

## 2. The result

Seeds 0, 1, 2. `incumbent t` is the shipped `0.5`; `derived t` is Youden's J on
the train corpus's labeled pairs; `kept` is `threshold_fit.source`; `Δ F1` is the
held-out pair-F1 of the cut that *won* minus that of `0.5` — i.e. exactly what
switching would buy. Regenerate with
`--render examples/research/results/threshold_default_study.json`.

| benchmark | method | seed | train pairs | test pairs | held-out gold | incumbent t | incumbent F1 | derived t | derived F1 | kept | Δ F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| abt_buy | rapidfuzz | 0 | 21,233 | 9,003 | 310 | 0.50 | 0.0625 | 0.1744 | 0.0730 | derived | +0.0106 |
| abt_buy | rapidfuzz | 1 | 21,118 | 9,143 | 309 | 0.50 | 0.0630 | 0.1949 | 0.0716 | derived | +0.0086 |
| abt_buy | rapidfuzz | 2 | 21,365 | 8,816 | 309 | 0.50 | 0.0642 | 0.1874 | 0.0754 | derived | +0.0112 |
| abt_buy | embedding_cosine | 0 | 21,233 | 9,003 | 310 | 0.50 | 0.0666 | 0.8361 | 0.2301 | derived | +0.1635 |
| abt_buy | embedding_cosine | 1 | 21,118 | 9,143 | 309 | 0.50 | 0.0654 | 0.8304 | 0.2255 | derived | +0.1602 |
| abt_buy | embedding_cosine | 2 | 21,365 | 8,816 | 309 | 0.50 | 0.0677 | 0.8294 | 0.2567 | derived | +0.1890 |
| amazon_google | rapidfuzz | 0 | 117,039 | 48,811 | 420 | 0.50 | 0.0149 | 0.3136 | 0.0231 | derived | +0.0081 |
| amazon_google | rapidfuzz | 1 | 116,391 | 49,194 | 420 | 0.50 | 0.0149 | 0.3136 | 0.0225 | derived | +0.0077 |
| amazon_google | rapidfuzz | 2 | 116,482 | 49,093 | 420 | 0.50 | 0.0159 | 0.3085 | 0.0230 | derived | +0.0071 |
| amazon_google | embedding_cosine | 0 | 117,039 | 48,811 | 420 | 0.50 | 0.0171 | 0.8139 | 0.2803 | derived | +0.2633 |
| amazon_google | embedding_cosine | 1 | 116,391 | 49,194 | 420 | 0.50 | 0.0169 | 0.8125 | 0.2456 | derived | +0.2286 |
| amazon_google | embedding_cosine | 2 | 116,482 | 49,093 | 420 | 0.50 | 0.0170 | 0.8125 | 0.2502 | derived | +0.2333 |
| dblp_acm | rapidfuzz | 0 | 12,746 | 5,381 | 665 | 0.50 | 0.3972 | 0.6952 | 0.8163 | derived | +0.4191 |
| dblp_acm | rapidfuzz | 1 | 12,681 | 5,466 | 665 | 0.50 | 0.3910 | 0.6917 | 0.7899 | derived | +0.3989 |
| dblp_acm | rapidfuzz | 2 | 12,708 | 5,477 | 662 | 0.50 | 0.3972 | 0.6954 | 0.7885 | derived | +0.3913 |
| dblp_acm | embedding_cosine | 0 | 12,746 | 5,381 | 665 | 0.50 | 0.2200 | 0.9018 | 0.8825 | derived | +0.6625 |
| dblp_acm | embedding_cosine | 1 | 12,681 | 5,466 | 665 | 0.50 | 0.2169 | 0.9041 | 0.8802 | derived | +0.6633 |
| dblp_acm | embedding_cosine | 2 | 12,708 | 5,477 | 662 | 0.50 | 0.2157 | 0.9010 | 0.8537 | derived | +0.6380 |
| dblp_scholar | rapidfuzz | 0 | 1,738,080 | 730,714 | 4,729 | 0.50 | 0.0344 | 0.4858 | 0.0324 | **declined** | +0.0000 |
| dblp_scholar | rapidfuzz | 1 | 1,735,460 | 731,910 | 4,670 | 0.50 | 0.0332 | 0.4891 | 0.0317 | **declined** | +0.0000 |
| dblp_scholar | rapidfuzz | 2 | 1,734,987 | 733,264 | 4,689 | 0.50 | 0.0335 | 0.4870 | 0.0316 | **declined** | +0.0000 |
| dblp_scholar | embedding_cosine | 0 | 1,738,080 | 730,714 | 4,729 | 0.50 | 0.0129 | 0.8360 | 0.3245 | derived | +0.3117 |
| dblp_scholar | embedding_cosine | 1 | 1,735,460 | 731,910 | 4,670 | 0.50 | 0.0127 | 0.8390 | 0.3276 | derived | +0.3149 |
| dblp_scholar | embedding_cosine | 2 | 1,734,987 | 733,264 | 4,689 | 0.50 | 0.0127 | 0.8390 | 0.3286 | derived | +0.3159 |
| febrl_person | rapidfuzz | 0 | 9,990 | 4,180 | 149 | 0.50 | 0.9521 | 0.5963 | 1.0000 | derived | +0.0479 |
| febrl_person | rapidfuzz | 1 | 9,920 | 4,186 | 150 | 0.50 | 0.9434 | 0.5963 | 1.0000 | derived | +0.0566 |
| febrl_person | rapidfuzz | 2 | 9,970 | 4,231 | 149 | 0.50 | 0.9460 | 0.5963 | 1.0000 | derived | +0.0540 |
| febrl_person | embedding_cosine | 0 | 9,990 | 4,180 | 149 | 0.50 | 0.0688 | 0.8090 | 0.8926 | derived | +0.8238 |
| febrl_person | embedding_cosine | 1 | 9,920 | 4,186 | 150 | 0.50 | 0.0692 | 0.8107 | 0.8846 | derived | +0.8154 |
| febrl_person | embedding_cosine | 2 | 9,970 | 4,231 | 149 | 0.50 | 0.0680 | 0.8107 | 0.8734 | derived | +0.8054 |
| fodors_zagat | rapidfuzz | 0 | 2,422 | 1,014 | 33 | 0.50 | 0.1164 | 0.6725 | 0.5299 | derived | +0.4135 |
| fodors_zagat | rapidfuzz | 1 | 2,431 | 1,004 | 33 | 0.50 | 0.1107 | 0.6725 | 0.5688 | derived | +0.4581 |
| fodors_zagat | rapidfuzz | 2 | 2,454 | 977 | 33 | 0.50 | 0.1121 | 0.6762 | 0.5357 | derived | +0.4237 |
| fodors_zagat | embedding_cosine | 0 | 2,422 | 1,014 | 33 | 0.50 | 0.0630 | 0.9018 | 0.9254 | derived | +0.8623 |
| fodors_zagat | embedding_cosine | 1 | 2,431 | 1,004 | 33 | 0.50 | 0.0636 | 0.9018 | 0.9524 | derived | +0.8887 |
| fodors_zagat | embedding_cosine | 2 | 2,454 | 977 | 33 | 0.50 | 0.0653 | 0.9018 | 0.9355 | derived | +0.8701 |
| ⚠ tiny_fixture | rapidfuzz | 0 | 28 | 3 | **1** | 0.50 | 1.0000 | 0.6319 | 0.0000 | derived | **−1.0000** |
| ⚠ tiny_fixture | rapidfuzz | 1 | 28 | 3 | **1** | 0.50 | 1.0000 | 0.6319 | 0.0000 | derived | **−1.0000** |
| ⚠ tiny_fixture | rapidfuzz | 2 | 26 | 3 | **1** | 0.50 | 1.0000 | 0.6214 | 1.0000 | **declined** | +0.0000 |
| ⚠ tiny_fixture | embedding_cosine | 0 | 28 | 3 | **1** | 0.50 | 0.5000 | 0.9451 | 1.0000 | derived | +0.5000 |
| ⚠ tiny_fixture | embedding_cosine | 1 | 28 | 3 | **1** | 0.50 | 0.5000 | 0.9451 | 1.0000 | derived | +0.5000 |
| ⚠ tiny_fixture | embedding_cosine | 2 | 26 | 3 | **1** | 0.50 | 0.5000 | 0.9451 | 1.0000 | derived | +0.5000 |
| walmart_amazon | rapidfuzz | 0 | 615,690 | 259,587 | 325 | 0.50 | 0.0083 | 0.5590 | 0.0112 | derived | +0.0028 |
| walmart_amazon | rapidfuzz | 1 | 614,841 | 260,526 | 325 | 0.50 | 0.0087 | 0.5206 | 0.0097 | derived | +0.0011 |
| walmart_amazon | rapidfuzz | 2 | 614,615 | 260,465 | 324 | 0.50 | 0.0091 | 0.5538 | 0.0119 | derived | +0.0029 |
| walmart_amazon | embedding_cosine | 0 | 615,690 | 259,587 | 325 | 0.50 | 0.0025 | 0.8634 | 0.0336 | derived | +0.0311 |
| walmart_amazon | embedding_cosine | 1 | 614,841 | 260,526 | 325 | 0.50 | 0.0025 | 0.8633 | 0.0332 | derived | +0.0307 |
| walmart_amazon | embedding_cosine | 2 | 614,615 | 260,465 | 324 | 0.50 | 0.0025 | 0.8633 | 0.0323 | derived | +0.0299 |
| wdc_computers | rapidfuzz | 0 | 112,037 | 47,359 | 308 | 0.50 | 0.0272 | 0.4142 | 0.0218 | **declined** | +0.0000 |
| wdc_computers | rapidfuzz | 1 | 111,377 | 47,639 | 310 | 0.50 | 0.0291 | 0.4056 | 0.0222 | **declined** | +0.0000 |
| wdc_computers | rapidfuzz | 2 | 111,850 | 47,421 | 312 | 0.50 | 0.0283 | 0.4045 | 0.0216 | **declined** | +0.0000 |
| wdc_computers | embedding_cosine | 0 | 112,037 | 47,359 | 308 | 0.50 | 0.0129 | 0.8725 | 0.0485 | derived | +0.0356 |
| wdc_computers | embedding_cosine | 1 | 111,377 | 47,639 | 310 | 0.50 | 0.0129 | 0.8671 | 0.0417 | derived | +0.0287 |
| wdc_computers | embedding_cosine | 2 | 111,850 | 47,421 | 312 | 0.50 | 0.0131 | 0.8730 | 0.0471 | derived | +0.0340 |

⚠ `tiny_fixture` is a **12-record smoke fixture**, not an ER benchmark. Its
held-out corpus is 3 records with **one** gold pair, so a single pair decides
recall outright and its ±1.0000 deltas carry no evidential weight in either
direction. (F1 there is not itself two-valued — false positives still move
precision, which is why the `embedding_cosine` rows read `0.5000` on the same
one gold pair. The problem is the sample size, not the metric's range.) It is kept in the table rather than dropped, because dropping
the only cells that lose would be exactly the wrong instinct — but it should not
be read as a benchmark result. (The harness refuses a cell with **zero** blocked
gold pairs; a corpus with one is the smallest thing it will still report.)

### 2.1 Counted

Generated by `--render`, both framings together — the first row is what `fit`
ships, the second is what a plain derive-and-apply would have shipped:

| framing | better | tied | worse |
|---|---|---|---|
| `fit(derive_threshold=True)` (derive **then race**) | 45 | 7 | 2 |
| the raw derived candidate alone | 45 | 1 | 8 |

Race win span: +0.0011 .. +0.8887. Of the 8 raw-candidate losses, **6** were
caught by the race (kept the incumbent) and 2 were applied anyway.

Where they fall: the 45 wins are every scorer on all 8 real benchmarks at every
seed, except the 7 declines; the declines are `dblp_scholar`/`rapidfuzz` ×3,
`wdc_computers`/`rapidfuzz` ×3 and `tiny_fixture`/`rapidfuzz` seed 2; the 2
losses are `tiny_fixture`/`rapidfuzz` seeds 0–1. The 6 raw losses the race caught
are the `dblp_scholar` and `wdc_computers` rows — they appear as ties in the
first row precisely because the race refused them.

**Seed stability.** No benchmark×scorer flips its `kept` verdict across seeds
except `tiny_fixture`/`rapidfuzz` (derived, derived, declined) — and that is the
1-gold-pair fixture again. The *cut itself* is less stable than the verdict, and
how much depends on the scorer. Per-cell spread across the three seeds:

| scorer | min spread | max spread | cells > 0.01 |
|---|---|---|---|
| `embedding_cosine` | 0.0000 (`fodors_zagat`, `tiny_fixture`) | **0.0067** (`abt_buy`) | 0 of 9 |
| `rapidfuzz` | 0.0000 (`febrl_person` — identical at all 3 seeds) | **0.0384** (`walmart_amazon`, 0.5206 / 0.5538 / 0.5590) | 3 of 9 |

So the cosine family is reproducible to well under ±0.005, and most `rapidfuzz`
cells are too — but three exceed 0.01: `tiny_fixture` 0.0105 (the 1-gold-pair
fixture), `abt_buy` 0.0205 and `walmart_amazon` 0.0384, with `wdc_computers`
next at 0.0097. "Derived cuts are stable to ~±0.005" was in an earlier draft of
this section, generalised from one benchmark; it is wrong by ~8× at the
tail. That matters for anyone tempted to lift a single derived number and
hard-code it — the thing §5 argues against on other grounds too.

### 2.2 Was the race right when it declined?

| declined cell | incumbent F1 | derived F1 (not applied) | declining was… |
|---|---|---|---|
| dblp_scholar/rapidfuzz s0 | 0.0344 | 0.0324 | correct |
| dblp_scholar/rapidfuzz s1 | 0.0332 | 0.0317 | correct |
| dblp_scholar/rapidfuzz s2 | 0.0335 | 0.0316 | correct |
| wdc_computers/rapidfuzz s0 | 0.0272 | 0.0218 | correct |
| wdc_computers/rapidfuzz s1 | 0.0291 | 0.0222 | correct |
| wdc_computers/rapidfuzz s2 | 0.0283 | 0.0216 | correct |
| tiny_fixture/rapidfuzz s2 | 1.0000 | 1.0000 | a tie — incumbent kept by rule |

**7 declines, 0 mistakes.** The race never refused a cut that would have helped.
That is the number that matters for #241's design: a plain derive-and-apply would
have shipped six real held-out regressions on two benchmarks as "improvements".

---

## 3. Should the default flip?

### 3.1 The measured half: yes, deriving earns its keep

On the evidence above, a user who has id-keyed labels and does **not** derive is
leaving a lot on the table — up to +0.89 pair-F1, and never less than zero on any
real benchmark at any seed. The two negative cells are a 12-record fixture with a
single held-out gold pair; treating them as a portfolio-level "it loses
sometimes" would be the same vacuous-zero mistake the closure diagnostic warns
about (`20260727_closure_diagnostic.md` §3.1). **If the question is "when asked,
should `fit` derive?", the answer is an unambiguous yes, and #241's race is what
makes it safe** — it declined 7 times and was right 7 times.

### 3.2 "Flip the default" is not a one-character change

Independently of the measurement, `derive_threshold: bool = False` →
`bool = True` cannot be done as written. The flag is not a preference read late;
it is a **precondition checked early**, and **three** call shapes that work today
raise the moment it is on — every one of them executed, not inferred:

| call shape | today | with `derive_threshold=True` | evidence |
|---|---|---|---|
| `fit(records)` — the sklearn-style no-op | **OK** | `ValueError: fit(derive_threshold=True) needs pairs=…` | **run** |
| `fit(records, labels=[…])` on a `SupervisedFitMixin` matcher | **OK** | same `ValueError` — `labels=` carries no split | **run** |
| `fit(records, pairs=…, method=Platt())` | **OK** | `ValueError: fit(method=…, derive_threshold=True) is not supported` — unconditional, no argument satisfies it | **run** |

> **Correction.** An earlier version of this table listed **six** shapes, adding
> three reached by code reading: `fit(pairs=…)` on a core-only install, on a
> *decider* matcher, and on an `_ops` chain with no `ThresholdSelect`. Those three
> are **not** regressions from the flip, because they do not work today either.
> `_fit_from_pairs` refuses `pairs=` outright when the matcher is not a
> `SupervisedFitMixin` *and* `derive_threshold` is false (`resolver.py:1267`), so
> for exactly those matchers the call already raises. Run both ways:
>
> ```
> TODAY (derive_threshold=False):
>   ValueError   fit(pairs=...) on a non-supervised matcher: WeightedAverageMatcher
>                does not support fit(pairs=...)
> UNDER THE FLIP (derive_threshold=True):
>   OK           fit(pairs=...) on a non-supervised matcher
> ```
>
> The flip *legalizes* that shape rather than breaking it. The three rows above
> stand, and they are enough — the first is the plain no-op `fit(records)`, which
> the repo's own suite calls — but the honest count is three, not six.

The three were **executed**, not inferred — each is `OK` today and raises
under the flip:

```
with derive_threshold=True (what a flipped default would do to each shape):
  ValueError  fit(records)
  ValueError  fit(records, labels=[...]) [supervised matcher]
  ValueError  fit(records, pairs=PAIRS, method=Platt())

today's default (derive_threshold=False), same shapes:
  OK          fit(records)
  OK          fit(records, labels=[...]) [supervised matcher]
  OK          fit(records, pairs=PAIRS, method=Platt())
```

This is not breakage imagined against hypothetical users. `grep -rn "\.fit("
tests examples docs src` finds dozens of call sites in exactly these shapes —
`.fit(records)`, `.fit(records, labels=…)`, and `.fit(…, method=…)` across
`Platt`, `Bootstrap` and the finetune methods — inside this repo alone.

So the honest framing of the question is **two** questions, and they have
different answers:

1. *When a user has id-keyed labels and asks for a cut, is the derived cut better
   than `0.5`?* — answered by §2.
2. *Should `fit` derive one without being asked?* — an API change, not a
   measurement consequence. The only non-breaking shape is a tri-state
   (`bool | None = None`, "derive when `pairs=` is given"), which still (a) makes
   `fit(pairs=…)` require `[trained]`, (b) starts raising for decider matchers,
   and (c) **silently moves the threshold of every existing supervised-fit
   caller**. And it is the same *shape* of implicit behaviour W4 deleted when it
   removed `matcher="auto"` — "naming a model is the user's job, not a
   heuristic's".

### 3.3 The recommendation, and what was *not* done

**`derive_threshold` was left at `False`. This PR changes no default and no
behavior.** Not because the measurement was weak — it is the strongest result
this study could have produced — but because the two questions in §3.2 have
different answers and the second one is a product decision. Three options, with
the costs each actually carries:

| option | what a user gets | cost / risk | reversal |
|---|---|---|---|
| **A. Leave it opt-in** (what this PR does) | Nothing changes. The measurement now exists to point users at, and `docs/EXPERIMENTS.md` can say "measured: up to +0.89, never worse on a real benchmark". | Users who never read the docs keep resolving at `0.5`. The out-of-the-box case (no labels at all) is untouched either way — `derive_threshold` cannot help it. | n/a |
| **B. Tri-state** `bool \| None = None` — derive when `pairs=` is given | Every labelled `fit` gets a measured cut without asking. | Three residual costs, all verified: `fit(pairs=…)` starts **requiring `[trained]`**; it starts **raising** for decider matchers; and it **silently moves the threshold** of every existing supervised-fit caller. Also the shape of implicit behaviour W4 deleted with `matcher="auto"`. | One-line revert, but any model saved in between carries the moved cut. |
| **C. Fix the constant instead** (§5) | Every user, **including those with no labels**, starts from a cut that is at least in the right decade for their score family. | Needs its own measurement per family (this study measures the *derived* cut, not a fixed replacement — see the caveat in §5.2). Changes output for existing no-argument users. | One-line revert. |

**Recommended: A now, C next, B only as a deliberate 0.4 API change.** C is where
the leverage is — it is the only one of the three that touches the case the
opening complaint is actually about ("out of the box, langres uses a magic
constant"), because `derive_threshold` by construction only ever helps a user who
already has labels.

This also follows the precedent this repo set two days earlier, in the same area:
*"The measurement is the deliverable. Changing the default is a separate PR."*

---

## 4. The split trap — a finding about #241's *reporting* path

This study's first design fed every blocked candidate to
`fit(pairs=…, split=0.3)` and read the held-out numbers `fit` printed. Those
numbers were unusable, and the reason generalises past this harness.

`align_pairs`' entity-disjoint split is a union-find over the two ids of each
**labeled** pair, and it assigns *whole components* to `valid` (a row-random
split would leak an entity across the boundary; assigning components cannot). It
also refuses to empty `train`, skipping any component that would swallow every
pair. Both rules are right. But a k-NN candidate graph over a real corpus is
essentially **one giant component** — every record is a neighbour of a neighbour
— so there is nothing left to assign:

| benchmark | seed | labeled pairs | align train | align valid | valid share (asked for 0.30) |
|---|---|---|---|---|---|
| abt_buy | 0 | 21,233 | 21,233 | 0 | 0.0000 |
| abt_buy | 1 | 21,118 | 21,118 | 0 | 0.0000 |
| abt_buy | 2 | 21,365 | 21,365 | 0 | 0.0000 |
| amazon_google | 0 | 117,039 | 117,039 | 0 | 0.0000 |
| amazon_google | 1 | 116,391 | 116,391 | 0 | 0.0000 |
| amazon_google | 2 | 116,482 | 116,482 | 0 | 0.0000 |
| dblp_acm | 0 | 12,746 | 12,746 | 0 | 0.0000 |
| dblp_acm | 1 | 12,681 | 12,681 | 0 | 0.0000 |
| dblp_acm | 2 | 12,708 | 12,679 | 29 | 0.0023 |
| dblp_scholar | 0 | 1,738,080 | 1,738,080 | 0 | 0.0000 |
| dblp_scholar | 1 | 1,735,460 | 1,735,460 | 0 | 0.0000 |
| dblp_scholar | 2 | 1,734,987 | 1,734,987 | 0 | 0.0000 |
| febrl_person | 0 | 9,990 | 9,990 | 0 | 0.0000 |
| febrl_person | 1 | 9,920 | 9,920 | 0 | 0.0000 |
| febrl_person | 2 | 9,970 | 9,970 | 0 | 0.0000 |
| fodors_zagat | 0 | 2,422 | 2,422 | 0 | 0.0000 |
| fodors_zagat | 1 | 2,431 | 2,431 | 0 | 0.0000 |
| fodors_zagat | 2 | 2,454 | 2,454 | 0 | 0.0000 |
| tiny_fixture | 0 | 28 | 28 | 0 | 0.0000 |
| tiny_fixture | 1 | 28 | 28 | 0 | 0.0000 |
| tiny_fixture | 2 | 26 | 26 | 0 | 0.0000 |
| walmart_amazon | 0 | 615,690 | 615,690 | 0 | 0.0000 |
| walmart_amazon | 1 | 614,841 | 614,841 | 0 | 0.0000 |
| walmart_amazon | 2 | 614,615 | 614,615 | 0 | 0.0000 |
| wdc_computers | 0 | 112,037 | 112,037 | 0 | 0.0000 |
| wdc_computers | 1 | 111,377 | 111,377 | 0 | 0.0000 |
| wdc_computers | 2 | 111,850 | 111,850 | 0 | 0.0000 |

**26 of 27 rows hold out nothing at all.** The single exception, `dblp_acm`
seed 2, holds out 29 pairs of 12,708 — 0.23 % where 30 % was requested. The
effect is not scale-dependent: it fires identically at 26 labeled pairs
(`tiny_fixture`) and at 1.7 million (`dblp_scholar`). It is about **connectivity**,
not size.

An earlier by-hand probe on the *full* corpora (before this study switched to
grading on a disjoint corpus) also produced the inverted case — `fit(split=0.3)`
returning `n_train=58, n_valid=18,127` on `dblp_acm` seed 0, i.e. 99.7 % held
out. That probe is not in the tracked artifact (different corpus scope: full vs.
the 70 % train split measured above), so treat it as an observation, not as one
of the numbers in the table.

Measured directly rather than inferred: every cell records what
`align_pairs(split=0.3)` *would* have held out of the same label set the race
used (`align_split_train` / `align_split_valid` in the JSON).

**What this does and does not mean.**

- It does **not** invalidate the race. Selection happens on `train` and never
  reads `valid`, so no held-out pair ever votes on which cut wins. What `split=`
  *does* change is how many labeled pairs the derivation saw — pairs moved into
  `valid` leave `train` — so a cut derived at `split=0.3` and one derived at
  `split=None` are fit on different samples. (This bullet previously said
  `split=` "changes only what `FitReport` reports", which contradicted its own
  next clause. It is only report-only in the degenerate case where `valid` comes
  back empty — which is 26 of these 27 rows, hence the confusion.)
- It **does** mean `FitReport.metrics` and `ThresholdCandidate.held_out_f1` are
  not trustworthy at this label scale — not because they are computed wrongly,
  but because the split that feeds them can silently return an empty, a
  three-pair, or an *inverted* `valid`. `FitReport` already reports `n_valid`, so
  the evidence is there; nothing warns.
- The feature's own docstrings scope it to "a handful of corrections out of a
  review loop", where components are sparse and the split behaves. The trap
  appears when the label set is dense — which is exactly what a user gets if they
  label a blocked candidate dump, and exactly what the fully-supervised regime in
  §1 is.

Worth its own follow-up: `align_pairs` could report the achieved valid fraction
(or warn when it lands far from the requested `split`), the same way
`GoldCoverage` makes blocking's leak visible instead of letting held-out metrics
quietly absorb it.

---

## 5. The six hard-coded `0.5`s — a proposal, not a change

### 5.1 It is not six sites, it is seventeen

The count needs care, because the constant is spelled three different ways and no
single search finds all of it. `grep -rn "threshold.*= 0\.5" src/langres` returns
**17** hits, but that set is simultaneously too wide and too narrow:

- **Two hits are a different concept** and are excluded by name, not by pattern:
  `core/adapters/glinker.py:47` is a *blocking* cut ("minimum entity-match
  confidence to emit a **candidate**" — an earlier pipeline stage), and
  `core/matchers/fellegi_sunter.py:146` is `agreement_threshold`, a per-*field*
  agreement cut feeding the EM model rather than a pair decision.
- **Two sites the grep misses**: `core/method_registry.py:538,545` spell it
  `default_threshold=0.5` with no space, which the pattern above cannot see.
  These are the `"string"` and `"embedding"` method specs.

What remains is **seventeen declared defaults spelled `0.5` for the match
decision** (15 of the grep's hits, plus the 2 it missed) — though, as the last
column records, not all seventeen are *effective* cuts:

| where | sites | what it defaults | effective? |
|---|---|---|---|
| `architectures/retrieval.py` | 4 | `Retrieve`, `RetrieveRerank`, `RetrieveLLM`, `RetrieveRerankLLM` | yes for the 2 score-carrying recipes; **inert** for `RetrieveLLM`/`RetrieveRerankLLM` (see below) |
| `architectures/fuzzy_string.py` | 1 | `FuzzyString` | yes |
| `architectures/vector_llm_cascade.py` | 1 | `VectorLLMCascade` | yes |
| `core/clusterer.py` | 1 | `Clusterer(threshold=0.5)` | yes |
| `core/matchers/rapidfuzz.py` | 1 | `RapidfuzzMatcher` | **no** — its own docstring says "stored for compatibility with Optimizer, but not used in `forward()`" |
| `core/matchers/embedding_score.py` | 1 | `EmbeddingScoreMatcher` | **no** — only picks the `decision_step` label and provenance; the emitted `score` is the raw similarity and the caller's threshold still decides |
| `core/method_registry.py` | 3 | `MethodSpec.default_threshold` field default + the `"string"` and `"embedding"` specs | yes |
| `curation/labelers.py` | 3 | `FakeLabeler`, `TeacherLabeler.__init__`, `TeacherLabeler.from_env` | yes |
| `report/eval_report.py` | 2 | the tearsheet's operating point | yes (display) |

**Two of the seventeen are decorative**, which an earlier version of this section
asserted they were not ("every one of them means 'a pair whose score reaches this
is a match'"). `RapidfuzzMatcher.threshold` is never read by `forward`, and
`EmbeddingScoreMatcher.threshold` changes only a provenance label. That does not
shrink the consolidation case — a constructor parameter named `threshold`
defaulting to `0.5` and doing *nothing* is arguably worse than one that works,
because it silently invites the caller to tune it — but the inventory should say
which is which rather than flatten them.

Docstring examples that merely *show* `threshold=0.5` (`core/clusterer.py:128`,
`core/resolver.py:491`, `core/matchers/cascade_judge.py:129`) are not counted:
they illustrate a default rather than declare one.

That the inventory takes this much care is itself part of the argument for naming
the constant. Three spellings across nine files is precisely the state in which a
value gets updated in eight places and missed in the ninth — and this section got
it wrong twice before the search was run properly.

`architectures/reranker.py` is the instructive exception: `Reranker.for_schema`
takes `threshold: float` with **no default at all**. One architecture in the same
package already treats "we do not know your score scale" as "you must tell us".

### 5.2 What the measurement says about the value

The optimum is not one number, and it is not near `0.5`:

| method | benchmarks | min derived t | median | max derived t | spread |
|---|---|---|---|---|---|
| embedding_cosine | 9 | 0.8090 | 0.8634 | 0.9451 | **0.1361** |
| rapidfuzz | 9 | 0.1744 | 0.5590 | 0.6952 | **0.5209** |

(One value per dataset — the lowest seed's cut — so the spread is measured
*between* datasets, not across seeds of the same one. Regenerate with `--render`.)

This is the answer to the strongest objection this study faces: *"the derived cut
only wins because `0.5` is a bad constant; a better constant per score family
would capture the same gain with no labels."* The objection is **half right, and
the halves split by family**:

- **For `embedding_cosine` it is right.** Nine datasets, spread 0.14, median
  0.863 — a fixed default in that band would land close to the derived cut
  everywhere, with no labels. `0.5` is not merely unmeasured here; it is off by
  ~0.35 on a scale where the discriminating region is ~0.1 wide, which is why the
  incumbent F1s in §2 are 0.0025–0.22 while the derived ones are 0.03–0.95.
- **For `rapidfuzz` it is much weaker.** Spread 0.52 across a 0.17–0.70 range — a
  4× ratio — so *if* a single constant does serve this family, it is nowhere near
  obvious what it is, and `0.5` sits in the middle of the range rather than near
  any dataset's optimum. A heuristic string score's scale depends on the schema's
  field count and fill rate, not just on the metric.

  **Stated precisely, because the stronger version is not supported:** an earlier
  draft concluded "there is no constant that is simultaneously right for
  `abt_buy` (0.17) and `dblp_acm` (0.70)". The artifact does not show that. It
  measures F1 at `0.5` and at each dataset's *own* derived cut — never at a
  shared alternative constant. Youden's J is flat across a separating gap (the
  very property that motivates #241's race), so two datasets can have far-apart
  argmaxes and still both score well at some third value. Ruling a shared
  constant out requires sweeping F1 over a grid of fixed thresholds per dataset,
  which this study did not run. What the spread supports is: the derived cut
  varies far more here than for cosine, and a per-family constant is a *less*
  promising fix for the string family than for the cosine one.

**Caveat, stated because it is the kind that gets skipped:** this study measured
the *derived* cut, not a fixed replacement constant. "A default near 0.86 would
land close to the derived cut" is an inference from the spread, not a measured
arm. Before changing any default, run the fixed-constant arm.

That is not noise between benchmarks — it is a **score-family** effect, and the
codebase already knows it. `MethodSpec.default_threshold` exists precisely because
"score scales differ per family, so each method carries its own sane default"
(the E12 comment), and the four LLM specs already override to `0.7`. The six
architectures bypass that seam entirely and re-declare `0.5` by hand.

**But not `RetrieveLLM` / `RetrieveRerankLLM` — correcting an earlier claim in
this section.** Their `ThresholdSelect` does sit after a `Parse()` of an LLM
response, which is what that claim rested on; the mistake was concluding they
should therefore default to `0.7`. `Parse` emits `ParsedGeneration(decision=True/
False)` — a *decider*, no score — and `predicted_match` gives `decision`
precedence, documented in `core/models.py` as "a judge that both decided and
ranked already made its call, so the threshold never overrides it". So on those
two recipes the threshold is **inert**, and moving it from `0.5` to `0.7` would
change nothing at all. The per-family routing argument below applies to the
score-carrying recipes (`Retrieve`, `RetrieveRerank`, `FuzzyString`,
`VectorLLMCascade`); for the two LLM recipes the real question is why a decider
pipeline exposes a `threshold` parameter that cannot affect its output.

### 5.3 The proposal

**Yes, name it — but name it for what it is, and do not stop there.**

1. **One constant, one meaning.** Add a stdlib-only leaf constant beside
   `predicted_match` (`core/models.py` — the one place that answers "is this pair
   a match"), e.g.:

   ```python
   #: The no-information match cut: the midpoint of a [0, 1] score, used when
   #: nothing has been measured. NOT a calibrated value, and unlikely to suit
   #: every score family — measured across 9 benchmarks, the cut derived from
   #: labels lands at 0.81–0.95 for embedding cosine but 0.17–0.70 for rapidfuzz,
   #: so 0.5 is far from both families' measured operating points and no single
   #: number sits near both ranges.
   #: Measurements: docs/research/20260728_threshold_default.md §5.2.
   #: Derive yours: fit(pairs=..., derive_threshold=True).
   DEFAULT_MATCH_THRESHOLD = 0.5
   ```

   Then have the six architectures, `Clusterer`, the two ranker matchers and
   `MethodSpec.default_threshold` reference it. The win is not DRY — it is that
   the caveat gets stated **once, where the value lives**, instead of seventeen
   times or (as today) nowhere.

2. **Do not sweep in what is not a match cut by the same authority.** The
   `report/eval_report.py` pair is a *display* operating point and the three
   `curation/labelers.py` cuts belong to the labeling loop; they are the same
   concept but a different owner, and collapsing them into one import would tie
   the report and curation packages to a core constant for no benefit. Leave
   them, with a comment pointing at the constant. The two genuinely different
   cuts found in §5.1 — GLiNER's candidate-emission confidence and
   Fellegi–Sunter's field-agreement cut — stay out entirely.

3. **The constant is the floor, not the fix.** The data says the real defect is
   that *one* number serves two score families. The seam for per-family defaults
   already exists (`MethodSpec.default_threshold`); the architectures just do not
   use it. Routing them through it — so a cosine-scored `Retrieve` and a
   heuristic-scored `FuzzyString` do not start from the same cut — is a bigger,
   better change than the constant, and belongs in its own PR with its own
   measurement.

All three are proposals. This PR changes no default and no behavior.
