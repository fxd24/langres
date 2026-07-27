# B1 — The closure diagnostic: does an output cluster contain a pair we judged and *rejected*?

**Date:** 2026-07-27 · **Cost:** $0 in spend (rapidfuzz scorer, no paid call) ·
**Harness:** `examples/research/closure_diagnostic.py` ·
**Raw data:** `examples/research/results/closure_diagnostic.json`

> Not dependency-free or offline on a cold cache: blocking is always the
> benchmark's own `VectorBlocker`, so a run needs the `[semantic]` extra and
> downloads `all-MiniLM-L6-v2` once. No paid model is ever called.

`Clusterer` — transitive closure over the accepted edges — is what an `ERModel`
uses unless you say otherwise. `docs/THEORY.md` §7 prices it as correlation
clustering with **+∞ on every observed positive edge and 0 on everything else**:
the negatives we paid a matcher to produce are simply not in the objective. So
closure can, in principle, put a pair inside one output cluster *after the
matcher looked at that exact pair and said no*.

This measures whether it actually does, on our data, with our default.
The measurement is the deliverable. **Changing the default is a separate PR.**

---

## 0. TL;DR

- **(a) It is not ~0.** At each benchmark's tuned operating point, closure puts
  **3,776 judged-and-rejected pairs inside its own output clusters** across the
  portfolio — 0.10 %–2.69 % of all rejected pairs, and **10.5 %–39.8 % of every
  pair that shares an output cluster**. Three benchmarks report exactly 0, and
  all three are *vacuous* zeros (§3.1) — not evidence that closure is safe.
- **(b) The hypothesis "they concentrate in the largest clusters" is half right,
  and the half it gets wrong is the interesting one.** The *count* does
  concentrate — 57 %–97 % of them sit in clusters of size ≥ 5, which are only
  3 %–16 % of clusters. But the *rate* does not keep climbing: contamination is
  0 % at size 2 (structurally), jumps to **11 %–21 % the instant a cluster
  reaches size 3**, and then stays roughly flat. **The discontinuity is the
  first chained merge, not bigness.**
- **(c) `CorrelationClusterer` cuts it 3.0×–7.4× at the tuned point (3,776 →
  676) and never scores worse.** Over all 54 grid points measured, its BCubed F1
  is strictly higher at 36, tied at 9, and lower at **0** (9 unscored — closure's
  giant component was too large to score).
- **The largest effect is not at the tuned point at all.** One grid step below
  it, closure collapses: on `walmart_amazon` at t = 0.50 it swallows **173,470**
  rejected pairs (largest cluster: the entire 7,386-record test split) against
  correlation's 4,615. Closure's quality is a **cliff**; correlation's is a
  slope.

---

## 1. How the pairs were reconstructed (and the two ways to get this wrong)

The naive version of this experiment is worse than not running it, because both
of the obvious shortcuts manufacture the finding they are looking for.

**Trap 1 — do not scan `score < threshold`.** The `JudgementLog` v4 row schema
(`src/langres/tracking/judgement_log.py`) persists
`score, decision, verdict, confidence, …` and **no `threshold`**. A scan would
have to re-guess the threshold the run actually used. Instead the harness
rebuilds each edge from **`verdict == True`**, which is the value
`LoggingMatcher` obtained from `predicted_match(judgement, clusterer.threshold)`
— *the same predicate `Clusterer.cluster()` calls*. Rows from earlier
retrieval/reranking stages carry `verdict = null` and are excluded.

**Trap 2 — `predicted_match` gives `decision` precedence over `score`**
(`src/langres/core/models.py:176-190`). For a *decider* matcher, `verdict` can
be `True` while `score` is `None` or below threshold, and a score-vs-threshold
scan mis-flags exactly those rows. This is not hypothetical:
`src/langres/core/matchers/llm_judge.py:165` returns
`ParsedVerdict(decision=..., score=None)` for the binary parser, so on an LLM run
a `score < threshold` scan has no score to compare at all.

**Both were counted, not assumed.** Two instrument checks run on every
benchmark:

| check | what it proves | result |
|---|---|---|
| `verdict_agreement` — recompute `predicted_match(j, t)` from each row's own `score`/`decision` and compare to the logged `verdict` | the reconstruction is the *same* predicate the clusterer used | **1.000 on all 9** (730,721 rows on `dblp_scholar` alone) |
| `reconstruction_exact` — re-cluster the rebuilt edges and compare the partition to the pipeline's own output | no edge was lost or invented in the rebuild | **`true` on all 9** |
| `decider_override_rows` — rows where `decision` overrode `score` | how much Trap 2 would have cost | **0** — see below |

`decider_override_rows = 0` because the `rapidfuzz` scorer emits
`decision: null` on every row (verified in the raw log). **That makes Trap 2
latent here, not absent.** The counter is what establishes that, rather than
luck; on the LLM path it goes from latent to live.

---

## 2. The measurement

`stratified_corpus_split` → derive the threshold on **train** by BCubed F1 over
the `{0.3 … 0.8}` grid → run **test** at that threshold → rebuild the edges from
the log → count, for each output cluster, the pairs inside it whose
`predicted_match` is `False`. Both clusterers consume the **identical**
judgement set, so (c) is a re-clustering, not a re-run: nothing is re-scored and
nothing is re-sampled.

Grid points whose largest train cluster exceeds 2,000 records are
**disqualified** from threshold tuning, and BCubed/pairwise scoring is skipped
above the same bound. This is not cosmetic: `calculate_bcubed_precision` is
O(Σ size²) and `calculate_pairwise_metrics` materializes C(size, 2) pairs, so a
20,091-record giant component does not terminate. The **rejected-inside count
is still exact at those points** — only the F1 is withheld (`giant` below).

### 2.1 At the tuned operating point

| benchmark | t | judged | rejected | **closure: rejected-inside** | rate | closure contamination | **corr: rejected-inside** | corr contamination | closure ÷ corr | closure BCubed F1 | corr BCubed F1 | closure largest | corr largest | largest **gold** cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `abt_buy` | 0.80 | 9,004 | 8,796 | **57** | 0.65 % | 11.0 % | **19** | 8.8 % | 3.0× | 0.6455 | 0.6511 | 27 | 11 | 3 |
| `amazon_google` | 0.80 | 48,854 | 47,978 | **944** | 1.97 % | **39.8 %** | **128** | 14.8 % | 7.4× | 0.7391 | 0.7641 | 32 | 15 | 6 |
| `dblp_acm` | 0.70 | 5,387 | 4,534 | **122** | 2.69 % | 10.5 % | **30** | 3.9 % | 4.1× | 0.9131 | 0.9357 | 12 | 6 | 2 |
| `dblp_scholar` | 0.80 | 730,721 | 728,899 | **747** | 0.10 % | 20.6 % | **173** | 9.6 % | 4.3× | 0.9502 | 0.9512 | 44 | 13 | 37 |
| `walmart_amazon` | 0.80 | 259,589 | 256,390 | **1,502** | 0.59 % | 28.3 % | **263** | 9.0 % | 5.7× | 0.8832 | 0.8967 | 37 | 31 | 4 |
| `wdc_computers` | 0.80 | 47,481 | 46,482 | **404** | 0.87 % | **31.5 %** | **63** | 7.4 % | 6.4× | 0.7242 | 0.7426 | 14 | 9 | 4 |
| `febrl_person` | 0.60 | 4,229 | 4,031 | 0 | 0 | 0 | 0 | 0 | — | 0.9983 | 0.9983 | 2 | 2 | 2 |
| `fodors_zagat` | 0.80 | 1,023 | 988 | 0 | 0 | 0 | 0 | 0 | — | 0.9785 | 0.9785 | 2 | 2 | 2 |
| `tiny_fixture` | 0.40 | 3 | 2 | 0 | 0 | 0 | 0 | 0 | — | 1.0000 | 1.0000 | 2 | 2 | 2 |
| **portfolio** | | | | **3,776** | | | **676** | | **5.6×** | | | | | |

*rate* = rejected-inside ÷ all rejected pairs. *contamination* = rejected-inside
÷ all pairs sharing an output cluster — the share of "this cluster says these
two are the same entity" that our own matcher contradicted.
*largest gold cluster* is over the whole corpus, from
[`20260727_portfolio_annotation.md`](20260727_portfolio_annotation.md).

`opensanctions` is `loadable=False` (external-only, CC-BY-NC — not vendored), so
it is skipped by the registry-driven selection; the other **9 of 10 registered
benchmarks all ran**.

---

## 3. Reading the three answers

### 3.1 (a) — it is not ~0, and the three zeros are vacuous

Six of the nine benchmarks put judged-and-rejected pairs inside their own
clusters, at contamination between 10.5 % and 39.8 %. On `amazon_google`, **two
of every five pairs that share an output cluster are pairs our matcher looked at
and rejected.**

The three zeros deserve their own sentence, because they are the trap this
harness could most easily have fallen into. `febrl_person`, `fodors_zagat` and
`tiny_fixture` produce **only size-2 clusters at their tuned threshold**. A
size-2 cluster's single in-cluster pair *is* the accepted edge, so 0 is
arithmetically forced — the experiment could not have found anything else. That
is why the harness sweeps the whole grid instead of reporting the tuned point
alone: `fodors_zagat` at t = 0.50 has **224** rejected-inside pairs, and
`febrl_person` at t = 0.30 has **2,145**. Reporting their tuned-point 0 as
"closure is fine here" would have been a measurement artifact presented as a
finding.

Also worth noting from the last three columns of §2.1: **closure's largest output
cluster exceeds the largest cluster in the entire gold labeling on every one of
the six** — 27 vs 3, 32 vs 6, 12 vs 2, 44 vs 37, 37 vs 4, 14 vs 4. Closure is
not just absorbing rejected pairs; it is building entities of a size the ground
truth never contains.

### 3.2 (b) — the discontinuity is at size 3, not at "large"

**Contamination density by output-cluster size (closure, tuned point):**

| benchmark | size 2 | size 3–4 | size 5–9 | size ≥ 10 |
|---|---|---|---|---|
| `abt_buy` | 0.0 % | 11.1 % | 18.8 % | 10.0 % |
| `amazon_google` | 0.0 % | 16.1 % | 34.2 % | 44.1 % |
| `dblp_acm` | 0.0 % | 18.4 % | 19.0 % | 16.7 % |
| `dblp_scholar` | 0.0 % | 21.0 % | 30.4 % | 25.4 % |
| `walmart_amazon` | 0.0 % | 19.7 % | 35.8 % | 31.3 % |
| `wdc_computers` | 0.0 % | 11.4 % | 32.5 % | 52.7 % |

The **count** does concentrate as the hypothesis predicted: clusters of size ≥ 5
are 3.5 %–16.0 % of clusters but carry **57 %–97 %** of all rejected-inside
pairs. Read alone, that number would look like a strong confirmation.

It mostly is not. Large clusters hold quadratically more pairs, so most of that
concentration is pair-counting, not chaining. The **rate** is the honest test,
and it says something different: 0 % at size 2 (structural), then a jump to
11 %–21 % at size 3–4, and then broadly flat — rising on `amazon_google` and
`wdc_computers`, falling on `abt_buy` and `dblp_acm`. The evidence loss is
**switched on by the very first chained merge** and does not need a giant
component to be substantial. A "cap the cluster size" mitigation would therefore
not address it.

### 3.3 (c) — correlation clustering strictly dominates, everywhere measured

Same judgements, same threshold, different `π`. `CorrelationClusterer`
(`src/langres/core/clusterers/correlation.py`) is a drop-in with the same
constructor and config: a node joins a cluster only on a **direct** edge to that
cluster's pivot.

One correction while citing it, because it matters for how these numbers may be
read: the shipped pivot order is **deterministic** — `sorted` by highest incident
score, ties by node id (`_pivot_priority`) — not the *uniformly random* pivot of
Ailon–Charikar–Newman that the class docstring names. ACN's 3-approximation
guarantee is a property of that randomization and does **not** transfer to this
implementation. For this harness the determinism is a feature (the correlation
column is exactly reproducible, no seed required), but nothing below is evidence
for an approximation bound. *(The overclaim is in the source docstring, which is
outside this PR's scope to edit — flagged as a follow-up.)*

Over the **54 grid points** across 9 benchmarks:

- rejected-inside is **never higher** for correlation — at any point, on any
  benchmark;
- BCubed F1 is **strictly higher at 36 points, tied at 9, lower at 0** (9
  unscored — closure's component was too large to score, which is itself the
  result).

And the tuned point is where closure looks *best*. One grid step down:

| benchmark | t | closure rejected-inside | closure largest cluster | corr rejected-inside | corr largest | closure BCubed | corr BCubed |
|---|---|---|---|---|---|---|---|
| `walmart_amazon` | 0.50 | **173,470** | 6,798 | 4,615 | 66 | `giant` | 0.4329 |
| `dblp_scholar` | 0.50 | **522,514** | 19,857 | 5,875 | 111 | `giant` | 0.3711 |
| `wdc_computers` | 0.50 | **27,643** | 1,243 | 1,133 | 67 | 0.1405 | 0.3890 |
| `amazon_google` | 0.50 | **25,340** | 1,344 | 315 | 100 | 0.0422 | 0.2789 |
| `abt_buy` | 0.40 | **4,747** | 642 | 146 | 29 | 0.0182 | 0.3216 |
| `dblp_acm` | 0.40 | **1,228** | 1,466 | 49 | 16 | 0.0120 | 0.5894 |

At t = 0.30, closure returns **exactly one cluster on all nine benchmarks**, and
on eight of them that cluster is the *entire* test split — so *every* rejected
pair is inside it (`dblp_scholar`: 255,015 of 255,015; `walmart_amazon`: 76,020
of 76,020). The ninth, `dblp_acm`, holds 1,472 of 1,473 records. Correlation at
the same point returns 56–1,782 clusters on those same nine (2 on the
12-record `tiny_fixture`), with a largest cluster of 14–146.

That is the practically important form of the finding. It is not that closure
is 5.6× worse at a well-tuned threshold — it is that **closure's quality is a
cliff and correlation's is a slope**, so closure makes the resolver's output a
sharp function of a threshold that has to be derived from data. Correlation is
the more forgiving default for exactly the users least able to tune it.

---

## 4. What this does and does not license

**Does:**

- The tuned-point numbers are exact, from a reconstruction proven identical to
  the clusterer's own predicate on all 730k+ rows.
- The comparison in (c) is like-for-like: one scored pass, two `π`s.

**Does not:**

- **This is not a default change.** Per the brief, the measurement ships alone;
  swapping `Clusterer` for `CorrelationClusterer` is a separate PR with its own
  blast radius (serialized `resolver.json` manifests name the clusterer class,
  and pivot clustering is randomized where closure is deterministic).
- It is measured with **one $0 scorer** (`rapidfuzz`) on **one split seed** (0).
  A matcher with a different error profile — in particular a decider LLM, where
  Trap 2 goes live — could move the numbers. It is unlikely to move the
  *ordering*, since (c) is a property of the two objectives rather than of the
  scores, but that is a hypothesis, not a measurement.
- The three vacuous zeros mean the portfolio contains **no benchmark that
  demonstrates closure is safe**; it contains three that cannot answer the
  question.

## 5. Reproducing

```bash
# all 9 loadable entries; 639 s of measured compute (dblp_scholar alone is 405 s)
uv run python examples/research/closure_diagnostic.py
# fodors_zagat + dblp_acm + tiny_fixture only, 27 s
uv run python examples/research/closure_diagnostic.py --fast
```

`$0` throughout. Judgement logs land in `tmp/closure_diagnostic/` (gitignored,
regenerable, ~600 MB for `dblp_scholar`); the findings land at the tracked
`examples/research/results/closure_diagnostic.json`.
