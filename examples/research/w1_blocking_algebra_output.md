# W1.3 blocking algebra + C6 clusterer — reference output

Reference run of `examples/w1_blocking_algebra.py` on Fodors-Zagat (FZ) and
Amazon-Google (AG), **zero LLM spend** (blocking is `KeyBlocker`/`VectorBlocker`,
scoring is `WeightedAverageJudge` — no LLM calls anywhere). Embedding/FAISS
nondeterminism moves the low-order digits slightly across machines; the
headline conclusions are stable. Reproduce with:

```
uv run python examples/w1_blocking_algebra.py
```

## 1. Composite blocking: Pair-Completeness (PC) / Reduction-Ratio (RR)

PC = cross-source recall of the blocking stage (`evaluate_blocking(...).candidate_recall`
after filtering to cross-source pairs, matching the methodology already
pinned in `er_benchmarks.py`/`amazon_google.py`). RR = `1 - n_candidates /
(n_source_a * n_source_b)` (cross-source product denominator — both
benchmarks are linkage tasks whose true matches are all cross-source).

| dataset | blocker | n_candidates | PC | RR |
| --- | --- | --- | --- | --- |
| fodors_zagat | VectorBlocker only (k=5, pinned) | 1,381 | 0.9911 | 0.9922 |
| fodors_zagat | **KeyBlocker(city) UNION VectorBlocker** | 10,901 | **1.0000** | 0.9382 |
| amazon_google | VectorBlocker only (k=50, pinned) | 60,502 | 0.8388 | 0.9862 |
| amazon_google | KeyBlocker(manufacturer) UNION VectorBlocker | 61,439 | 0.8388 | 0.9860 |

### Read-out

- **Fodors-Zagat: composite blocking closes the recall gap.** The pinned
  `VectorBlocker` alone misses exactly one gold pair (`f640`/`z325`,
  "masa's") — a documented, structural edge case where both sources have
  *identical* `embed_text`, so `VectorBlocker`'s position-based self-skip
  drops the one neighbour that would have surfaced it (see
  `er_benchmarks.py`'s `DEFAULT_BLOCKING_K` comment). `KeyBlocker(city)`
  recovers it: two records with identical text obviously share a city, so
  the UNION surfaces the pair the vector search structurally cannot. PC
  goes to a clean 1.0000, at a real RR cost (0.9922 → 0.9382 — still a
  93.8% reduction vs. brute-force `C(864,2)`, just less aggressive).
- **Amazon-Google: composite blocking does NOT lift the documented 0.8388
  ceiling.** `manufacturer` is missing on **62.5% of AG records**
  (2,870/4,589) — `KeyBlocker`'s standard "exclude records with no key"
  semantics means most of the corpus never gets a chance to contribute a
  candidate through this key, so the union adds ~937 extra candidates
  without recovering any *new* true match the vector search hadn't already
  found. This is an honest negative result for this specific key choice: a
  field-selectivity problem, not a `KeyBlocker`/`CompositeBlocker`
  correctness problem. A denser or composite key (e.g. normalized title
  n-grams, or `manufacturer` OR normalized-price) would likely do better —
  left for future work / #55, not required by this exit criterion (PC/RR
  *measured*, not *maximized*).
- **Recall-first union does no *harm*** in either case: PC never drops
  below the `VectorBlocker`-alone baseline (`union` is provably a
  pairs-superset per `CompositeBlocker`'s own test suite), consistent with
  its "recall-maximizing default" design intent.

## 2. Clusterer comparison: base `Clusterer` vs `CorrelationClusterer` (C6)

Same blocking (`VectorBlocker`, pinned k) + same `WeightedAverageJudge` +
same threshold (0.80, reused from `examples/m3_zero_spend_race_output.md`)
for both rows — **only the clusterer differs**, isolating its effect.

| dataset | clusterer | bc_P | bc_R | bc_F1 |
| --- | --- | --- | --- | --- |
| fodors_zagat | Clusterer (default, transitive closure) | 0.9888 | 0.9583 | 0.9733 |
| fodors_zagat | CorrelationClusterer (C6, pivot algorithm) | 0.9911 | 0.9572 | 0.9739 |
| amazon_google | Clusterer (default, transitive closure) | 0.6746 | 0.7978 | 0.7311 |
| amazon_google | **CorrelationClusterer (C6, pivot algorithm)** | **0.7461** | 0.7818 | **0.7635** |

### Read-out

- **Fodors-Zagat: a wash.** ΔbcF1 = +0.0006 (0.9733 → 0.9739) — both land
  within measurement noise of each other. FZ is small, near-saturated, and
  its match clusters are almost all size-2 (one Fodor's + one Zagat record),
  so there's little chaining structure for the base `Clusterer`'s transitive
  closure to over-merge in the first place.
- **Amazon-Google: C6 clearly wins.** ΔbcF1 = **+0.0324** (0.7311 → 0.7635,
  a 4.4% relative improvement), driven by **+0.0715 precision** (0.6746 →
  0.7461) at a modest **-0.0160 recall** cost (0.7978 → 0.7818). This is
  exactly the over-merge signature the base `Clusterer`'s transitive closure
  is known to produce on the harder, chattier dataset (M3's documented
  −0.63 BCubed regression on `embedding_cosine` was the extreme version of
  this same failure mode) — `CorrelationClusterer` requires a *direct* edge
  to a cluster's pivot rather than any transitive path, so it declines to
  chain-merge borderline pairs the base `Clusterer` would.
- **The synthetic unit tests** (`tests/core/clusterers/test_correlation_clusterer.py`)
  show the mechanism directly: on a 3-node chain (A-B, B-C, no direct A-C
  edge) the base `Clusterer` merges `{A, B, C}` into one cluster;
  `CorrelationClusterer` produces `{A, B}, {C}` — while a fully-connected
  triangle (every pair directly compared and matched) still merges fully
  under both, so C6 is not simply "more conservative everywhere," only where
  the evidence is actually indirect.

### Default-flip decision: **NOT flipped — C6 stays opt-in**

Per the exit criterion ("only flip the clusterer default if C6 clearly wins
on the benchmark; otherwise keep it opt-in and report why"): C6 wins clearly
on Amazon-Google (the harder, more chain-prone dataset) but is a wash on
Fodors-Zagat. Two datasets, one showing a clear win and one showing no
differentiation, is not the "clearly wins" bar a *global default* change
needs — flipping the default would be extrapolating a hard-dataset win onto
every user, including the easy/well-behaved cases where it buys nothing.
`CorrelationClusterer` (registered as `"correlation_clusterer"`) is shipped
as a pluggable, opt-in alternative:

```python
from langres.core.clusterers.correlation import CorrelationClusterer

resolver = Resolver(
    blocker=...,
    module=...,
    clusterer=CorrelationClusterer(threshold=0.7),  # opt-in, not the default
)
```

**Recommendation:** prefer `CorrelationClusterer` for harder/messier
entity-resolution problems (long, noisy fields; many borderline-similarity
records — Amazon-Google-shaped data) where the base `Clusterer`'s transitive
closure has real chaining risk. Keep the base `Clusterer` for small,
well-behaved, mostly-clean datasets (Fodors-Zagat-shaped data) where the two
are equivalent and the base clusterer's simplicity is one less thing to
reason about.
