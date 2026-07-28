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

- **(a) The derived cut beats `0.5` almost everywhere.** 45 cells improve, 7 tie
  exactly (the race declined, so the threshold did not move), **2 lose**. The two
  losses are both `tiny_fixture` + `rapidfuzz`, whose held-out corpus is **3
  records containing 1 gold pair** — an F1 that can only be `0.0` or `1.0`. On
  the **8 real benchmarks the derived cut never loses at any seed for either
  scorer**, and the wins are not marginal: +0.42 (dblp_acm/rapidfuzz), +0.66
  (dblp_acm/cosine), +0.82 (febrl_person/cosine), +0.89 (fodors_zagat/cosine).
- **(b) The race earns its seat.** It declined 7 times, and in **6 of those 7 the
  derived cut really was worse held-out** (the 7th was an exact tie, which the
  tie-keeps-incumbent rule handles). It has never once declined a cut that would
  have helped. It is not infallible in the other direction — it *kept* two cuts
  that lost — but both are the 1-gold-pair fixture, and #241 already documents
  that selecting on `train` "bounds the risk, it does not remove it". This is
  that residual, observed.
- **(c) So: yes to deriving — but "flip the default" is not a one-character
  change, and this PR does not make it.** `derive_threshold: bool = False →
  True` makes **six documented call shapes raise**, starting with the plain
  `fit(records)` no-op and every `fit(…, method=…)`; the repo's own suite would
  go red on the first test. The measured answer and the API change are different
  decisions — see §3.
- **(d) The most actionable finding is not about the flag at all.** For
  `embedding_cosine` the derived cut lands in **0.809–0.945 (median 0.863) across
  all nine datasets** — a spread of 0.14. `0.5` is not merely unmeasured there,
  it is *nowhere near* right, and a per-family default in that band would help
  every user **with no labels at all**. For `rapidfuzz` the same spread is
  **0.174–0.695** (0.52, a 4× range), so no constant works and that family
  genuinely needs derivation. One number cannot serve both — which is exactly
  what `MethodSpec.default_threshold` already exists to express, and exactly what
  the six architectures bypass.
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

`split=` does **not** affect the race — `_select_threshold` selects on `train`
and never reads `valid`; the argument only changes what `fit` *reports*. So
`split=None` gives the race exactly the label set a user would give it, and
grading on the disjoint corpus is a *stronger* held-out estimate than the one
`fit` would have printed.

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
held-out corpus is 3 records with **one** gold pair, so its F1 is a coin flip
between `0.0` and `1.0` and its ±1.0000 deltas carry no evidential weight in
either direction. It is kept in the table rather than dropped, because dropping
the only cells that lose would be exactly the wrong instinct — but it should not
be read as a benchmark result. (The harness refuses a cell with **zero** blocked
gold pairs; a corpus with one is the smallest thing it will still report.)

### 2.1 Counted

| outcome | cells | where |
|---|---|---|
| derived cut kept **and** better held-out | **45** | every scorer on all 8 real benchmarks, at every seed, except the 7 declines below |
| race declined → threshold unmoved, Δ exactly 0 | **7** | dblp_scholar/rapidfuzz ×3, wdc_computers/rapidfuzz ×3, tiny_fixture/rapidfuzz seed 2 |
| derived cut kept **and** worse held-out | **2** | tiny_fixture/rapidfuzz seeds 0–1 (1 gold pair) |

**Seed stability.** No benchmark×scorer flips its `kept` verdict across seeds
except `tiny_fixture`/`rapidfuzz` (derived, derived, declined) — and that is the
1-gold-pair fixture again. Derived cuts are stable to ~±0.005 between seeds
(e.g. dblp_acm/rapidfuzz 0.6952 / 0.6917 / 0.6954).

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

<!-- DECISION -->

### 3.2 "Flip the default" is not a one-character change

Independently of the measurement, `derive_threshold: bool = False` →
`bool = True` cannot be done as written. The flag is not a preference read late;
it is a **precondition checked early**, and six documented paths raise the moment
it is on. Each is a code read, not a guess:

| call shape | what raises with `derive_threshold=True` | where |
|---|---|---|
| `fit(records)` (the sklearn-style no-op) | `ValueError: fit(derive_threshold=True) needs pairs=…` | `resolver.py` (`if derive_threshold and pairs is None`) |
| `fit(records, labels=[…])` (the pre-existing positional contract) | same `ValueError` — `labels=` carries no split | same guard |
| `fit(records, method=Bootstrap()/Platt()/…)` | `ValueError: fit(method=…, derive_threshold=True) is not supported` | `resolver.py`, first branch of `fit` |
| `fit(records, pairs=…)` on a **core-only** install | `ImportError` — `derive_threshold_from_pairs` lazily imports `training.calibration`, which imports scikit-learn at module scope | `curation/harvest.py` → `training/calibration.py` |
| `fit(records, pairs=…)` with a **decider** matcher (`LLMMatcher(response_parser="binary_yes_no")`) | `ValueError` from `_refuse_deciders` | `resolver.py` |
| `fit(records, pairs=…)` on an explicit `_ops` chain with no `ThresholdSelect` | `ValueError: found no decision threshold to fit` | `resolver.py::_fit_chain_threshold` |

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

<!-- SPLIT TRAP TABLE -->

Measured directly rather than inferred: every cell records what
`align_pairs(split=0.3)` *would* have held out of the same label set the race
used (`align_split_train` / `align_split_valid` in the JSON).

**What this does and does not mean.**

- It does **not** invalidate the race. Selection happens on `train`; `split=`
  changes only what `FitReport` reports. A cut derived with `split=0.3` and one
  derived with `split=None` differ only in how many labeled pairs the derivation
  saw.
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

### 5.1 It is not six sites, it is fourteen

`grep -rn "threshold.*= 0\.5" src/langres` finds the same decision cut, spelled as
the same literal, at **fourteen** independent places — and every one of them means
"a pair whose score reaches this is a match":

| where | sites | what it defaults |
|---|---|---|
| `architectures/retrieval.py` | 4 | `Retrieve`, `RetrieveRerank`, `RetrieveLLM`, `RetrieveRerankLLM` |
| `architectures/fuzzy_string.py` | 1 | `FuzzyString` |
| `architectures/vector_llm_cascade.py` | 1 | `VectorLLMCascade` |
| `core/clusterer.py` | 1 | `Clusterer(threshold=0.5)` |
| `core/matchers/rapidfuzz.py` | 1 | `RapidfuzzMatcher` |
| `core/matchers/embedding_score.py` | 1 | `EmbeddingScoreMatcher` |
| `core/method_registry.py` | 3 | `MethodSpec.default_threshold` field default + `"string"` + `"embedding"` |
| `curation/labelers.py` | 2 | the fake labeler's cut, the LLM teacher's cut |
| `report/eval_report.py` | 2 | the tearsheet's operating point |

`architectures/reranker.py` is the instructive exception: `Reranker.for_schema`
takes `threshold: float` with **no default at all**. One architecture in the same
package already treats "we do not know your score scale" as "you must tell us".

### 5.2 What the measurement says about the value

The optimum is not one number, and it is not near `0.5`:

<!-- DERIVED RANGE -->

That is not noise between benchmarks — it is a **score-family** effect, and the
codebase already knows it. `MethodSpec.default_threshold` exists precisely because
"score scales differ per family, so each method carries its own sane default"
(the E12 comment), and the four LLM specs already override to `0.7`. The six
architectures bypass that seam entirely and re-declare `0.5` by hand — including
`RetrieveLLM` / `RetrieveRerankLLM`, whose `ThresholdSelect` sits *after* a
`Parse()` of an LLM response, i.e. on the very score family the registry says
should default to `0.7`.

### 5.3 The proposal

**Yes, name it — but name it for what it is, and do not stop there.**

1. **One constant, one meaning.** Add a stdlib-only leaf constant beside
   `predicted_match` (`core/models.py` — the one place that answers "is this pair
   a match"), e.g.:

   ```python
   #: The no-information match cut: the midpoint of a [0, 1] score, used when
   #: nothing has been measured. NOT a calibrated value — see
   #: docs/research/20260728_threshold_default.md, where the measured optimum is
   #: 0.62–0.70 for rapidfuzz and 0.83–0.96 for cosine on every benchmark tested.
   #: Derive yours: fit(pairs=..., derive_threshold=True).
   DEFAULT_MATCH_THRESHOLD = 0.5
   ```

   Then have the six architectures, `Clusterer`, the two ranker matchers and
   `MethodSpec.default_threshold` reference it. The win is not DRY — it is that
   the caveat gets stated **once, where the value lives**, instead of fourteen
   times or (as today) nowhere.

2. **Do not sweep in what is not a match cut by the same authority.** The
   `report/eval_report.py` pair is a *display* operating point and the
   `curation/labelers.py` pair belongs to the labeling loop; they are the same
   concept but a different owner, and collapsing them into one import would tie
   the report and curation packages to a core constant for no benefit. Leave
   them, with a comment pointing at the constant.

3. **The constant is the floor, not the fix.** The data says the real defect is
   that *one* number serves two score families. The seam for per-family defaults
   already exists (`MethodSpec.default_threshold`); the architectures just do not
   use it. Routing them through it — so a cosine-scored `Retrieve` and a
   heuristic-scored `FuzzyString` do not start from the same cut — is a bigger,
   better change than the constant, and belongs in its own PR with its own
   measurement.

All three are proposals. This PR changes no default and no behavior.
