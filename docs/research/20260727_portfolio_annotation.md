# B2 — Portfolio annotation: is the benchmark portfolio trustworthy?

**Date:** 2026-07-27 · **Cost:** $0 in spend (rapidfuzz + the profiler; no paid
call) · **Harness:** `examples/research/portfolio_profile.py` ·
**Raw data:** `examples/research/results/portfolio_profile.json`

Every claim langres makes is measured on this portfolio. If a benchmark is
**saturated** (a free method already ties the literature, so it has no headroom
to rank anything) or **structurally unrepresentative** (an artifact in the gold
labels makes the metric describe the labeling), then the numbers we report are
about the benchmark, not the method. This is the calibration of our own
instrument, and it gates the reading of every model comparison.

**Nothing is retired on this basis.** A saturated set is still a fine regression
guard; it just cannot be evidence *for* a method. The deliverable is the
annotation below.

---

## 1. The headline

**No registered benchmark is saturated for the free string method.** On every
dataset where the comparison can be made honestly, `rapidfuzz` sits **0.09 to
0.69 F1 below** the published number. The portfolio's problem is not a lack of
headroom — it is the opposite: a free lexical baseline is nowhere near the
ceiling, so there is plenty of room to rank methods.

The one genuinely saturated set is **`fodors_zagat`**, and it fails a *different*
metric (see §4) — which is exactly why the verdict has to name its metric.

---

## 2. The annotation table

| benchmark | task | wheel-loadable | records | gold pairs | prevalence | max gold cluster | sep AUC | vocab Jaccard | min token coverage | **x-src recall ceiling** | rapidfuzz F1 (lit. split) | published F1 | **saturated** | **structural caveats** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `abt_buy` | linkage | no | 2,173 | 1,044 | 4.42e-04 | 3 | 0.957 | 0.322 | 0.796 | 0.9847 | 0.2018 | 0.893 (Ditto) | **no** | — |
| `amazon_google` | linkage | no | 4,589 | 1,390 | 1.32e-04 | 6 | 0.955 | 0.338 | 0.791 | **0.8396** | 0.2881 | 0.756 (Ditto) | **no** | **capped-recall** |
| `dblp_acm` | linkage | no | 4,910 | 2,220 | 1.84e-04 | 2 | 1.000 | 0.842 | 0.967 | 1.0000 | 0.8905 | ~0.98 | **no** (gap 0.09) | † strictly 1:1 labels |
| `dblp_scholar` | linkage | no | 66,879 | 13,763 | 6.15e-06 | **37** | 0.993 | **0.061** | 0.651 | **0.3977** | 0.6588 | not recorded | **?** | **capped-recall**, **large-component** |
| `walmart_amazon` | linkage | no | 24,628 | 1,092 | 3.60e-06 | 4 | 0.996 | 0.133 | 0.806 | **0.8837** | 0.3051 | not recorded | **?** | **capped-recall** |
| `wdc_computers` | linkage | no | 4,647 | 1,111 | 1.03e-04 | 4 | 0.929 | 0.720 | 0.978 | **0.9001** | 0.4447 | not recorded | **?** | **capped-recall** |
| `fodors_zagat` | linkage | **yes** | 864 | **112** | 3.00e-04 | 2 | 0.998 | 0.285 | 0.708 | 1.0000 | no literature split | 1.00 (ZeroER) | † **YES** — see §4 | **tiny-gold** |
| `febrl_person` | linkage | **yes** | 1,000 | 500 | 1.00e-03 | 2 | 1.000 | 0.682 | 0.851 | 1.0000 | no literature split | not recorded | **?** | **one-to-one by construction** |
| `tiny_fixture` | linkage | **yes** | 12 | 3 | 4.55e-02 | 2 | 0.989 | 0.269 | 0.447 | 1.0000 | 0.0000 | n/a — not a benchmark | n/a | tiny-gold, lexical-gap |
| `febrl_dedup` | **dedup** | **yes** | 5,000 | 6,538 | 5.23e-04 | 6 | 1.000 | n/a | n/a | n/a — single source | no literature split | not recorded | **?** | — |
| `opensanctions` | linkage | not vendored | — | — | — | — | — | — | — | — | not loadable | † 98.95 (GPT-4o, 0–100) | **?** | external-only (CC-BY-NC) |

**`febrl_dedup` was added after this document's first publication** (2026-07-28);
its row is rendered from the same regenerated `portfolio_profile.json`. Its four
`n/a` cells are structural, not missing measurements: vocabulary overlap and the
cross-source recall ceiling are both defined *between two sources*, and this
benchmark has one. See §7.

**† = written by hand, not produced by the harness.** Four cells, and they are
marked because everything else in this table is a rendering of
`portfolio_profile.json` and a reader is entitled to know which is which:

- `fodors_zagat` **saturated** — the harness computes `null` (`?`) because the
  dataset ships no literature split for `rapidfuzz` to be graded on, so the
  automated rule cannot fire. The **YES** is the §4 argument, on a different
  metric, made explicitly.
- `dblp_acm` **published F1 `~0.98`** — the cited source writes it with a tilde;
  it is an approximation used as an exact comparand. The verdict survives
  comfortably (gap 0.0895 against a 0.02 margin), but the `~` belongs in the cell.
- `dblp_acm` **"strictly 1:1 labels"** — true and worth saying, but the computed
  `one-to-one` rule requires zero singletons and `dblp_acm` has 470, so the
  harness emits no tag. The rule was fixed before looking and is deliberately
  left alone rather than widened post-hoc to make this cell computed.
- `opensanctions` **published F1** — recorded in
  `docs/research/20260701_er_seam_audit.md:182` and the registry, but it is not
  in `PUBLISHED_SOTA` (nothing loadable to compare) and it is on a 0–100 scale in
  a column of 0–1 F1s.

**Reading the columns.**

- **x-src recall ceiling** — the share of gold pairs a **cross-source** candidate
  set can ever contain. A two-source linkage pipeline pairs A-records against
  B-records, so it can never emit a pair whose two records come from the same
  side; any gold cluster spanning 3+ records emits exactly such pairs, and every
  one of them is **unreachable by any blocker, however good**. This is a hard cap
  on recall and Pair Completeness, and it is a property of the *labeling*, not of
  the method. It caps F1 too, but **at a higher number, not at this one**: with
  perfect precision and recall capped at `r`, F1 reaches `2r/(1+r)`, so
  `amazon_google`'s 0.8396 still permits an F1 of 0.9128. Read this column as a
  recall/PC ceiling; treating it as an F1 bound would mark achievable scores
  impossible. Four of the ten loadable
  benchmarks are capped: `amazon_google` at 0.8396, `wdc_computers` 0.9001,
  `walmart_amazon` 0.8837, and `dblp_scholar` at **0.3977**. A "0.84 recall" on
  `amazon_google` is therefore a **perfect** score, not an 84% one — see §8.

- **wheel-loadable** — can a `pip install langres` user load it? `True` only when
  the dataset vendors data files and `pyproject.toml`'s
  `[tool.hatch.build] exclude` drops **none** of them. `abt_buy`/`amazon_google`
  ship their `peeters_sampled_test.csv` pair set but *not* their corpus tables,
  so they are partially shipped and still raise `BenchmarkDataNotFoundError`.
  This is reproduction status, not quality — but it does mean six of ten
  loadable benchmarks exist only in a git checkout.
- **gold pairs** — within-cluster pairs of the **closure** gold partition, which
  is larger than the literature's positive count wherever the labels chain:
  `abt_buy` 1028 → 1044, `amazon_google` 1167 → 1390, `walmart_amazon` 962 →
  1092. Closure *adds* pairs, in the labels, before we run anything.
- **rapidfuzz F1** — `rapidfuzz` on the dataset's own fixed *literature*
  train/valid/test split at a threshold derived on train
  (`evaluate_fixed_split_honest`). This is the only pair-F1 that is
  apples-to-apples with a published DeepMatcher/Ditto number. Blank where the
  dataset ships no literature split.
- **published F1** — only numbers already recorded in this repository, each with
  a file citation in `PUBLISHED_SOTA` (`examples/research/portfolio_profile.py`).
  Five loadable sets have **no** published number written down anywhere here, so
  they get no verdict — reported as `?`, never silently read as "unsaturated".
- **vocab Jaccard / min token coverage** — the new `VocabularyOverlapSection`
  (§6). Jaccard is over token *types*; coverage is the share of a side's token
  *occurrences* whose type also occurs on the other side.

---

## 3. The rules, fixed before looking

Both are the research agenda's, stated before any number was computed, and both
are executable code (`examples/research/portfolio_profile.py`), not prose:

- **Saturated** := the free `rapidfuzz` method scores within **0.02** of the
  published SOTA on the same split. If a free method ties the literature, the set
  cannot rank methods.
- **Structurally caveated** := the gold labels carry an artifact that makes the
  metric describe the labeling rather than the task:
  - `tiny-gold` — fewer than 200 positive pairs, so one pair moves F1 by ~1%;
  - `one-to-one` — every gold cluster is exactly a pair and nothing is a
    singleton: an assignment problem, not the many-to-many ER task;
  - `large-component` — a gold cluster of 10+ records in a two-source linkage
    set: the label is itself a transitive closure, so the metric rewards
    chaining;
  - `lexical-gap` — the two sources share under 50% of their token occurrences,
    so a string method measures the encoding gap, not entity resolution.

---

## 4. `fodors_zagat` — confirmed saturated, but not by this rule

The suspicion was right, the stated rule cannot reach it, and the distinction
matters.

**Confirmed:** the dataset has **112 gold pairs** over 864 records
(`src/langres/data/datasets/fodors_zagat/SOURCE.md:14` — reproduced exactly by
the profiler). At 112 pairs, a single pair is worth ~0.9% of F1, so the set
cannot resolve a method difference smaller than about a point. It is flagged
`tiny-gold`.

**But it ships no fixed literature split** (a perfect mapping instead), so the
agenda's pair-F1 rule has nothing to evaluate. The saturation verdict therefore
rests on numbers already recorded in this repo, on the *clustering* metric:

- `rapidfuzz` BCubed F1 **0.9785** against an all-singletons sanity floor of
  **0.9317** (`examples/research/m3_zero_spend_race_output.md:22`). **Merging
  nothing already scores 0.93** — the entire measurable range is 0.047 wide.
- `random_forest` pair F1 **0.985** (`docs/research/20260702_w1_trained_family_results.md:45`)
  against ZeroER's published **1.00** — inside the 0.02 margin, on a protocol
  (best-of-grid threshold on the blocked candidate band) that `docs/BENCHMARKS.md`
  itself flags as optimistically biased, i.e. biased *toward* this verdict.

**Verdict: saturated on BCubed; keep as a regression guard, never quote it as
evidence for a method.** And note the general lesson: *saturation is
metric-dependent*. FZ is saturated on BCubed and unmeasurable on pair-F1; the
DeepMatcher sets are unsaturated on pair-F1. A verdict without its metric is not
a verdict.

---

## 5. `dblp_scholar` — confirmed transitive-closure labeling artifact

**Confirmed, exactly.** The profiler independently reproduces
`src/langres/data/datasets/dblp_scholar/ATTRIBUTION.md`: **66,879 records**,
**2,351 match clusters**, **13,763 closure gold pairs**, and a **largest gold
cluster of 37 records**.

A 37-record "entity" in a bibliographic two-source set is a transitive closure of
the labeling, not 37 genuinely identical papers — and the composition proves it.
Re-derived by an independent path (stdlib `csv` + union-find over the raw
`train`/`valid`/`test` label files, importing no langres code, which reproduces
2,351 components and the largest at 37 exactly), that component holds **2 records
from table A and 35 from table B**. It is not 37 papers: it is two DBLP entries
chained together through 35 Scholar records. The next-largest component is 20.

Two measurable consequences:

1. **60% of the gold pairs are intra-source.** Of 13,763 within-cluster gold
   pairs only 5,473 (39.77%) are cross-source, which caps cross-source blocking
   Pair-Completeness at **0.3977** *regardless of blocking quality* — the
   achieved 0.3945 is 99.2% of that ceiling. The famous "PC 0.39" is a labeling
   artifact, not a blocking failure. **This is the extreme case of a
   portfolio-wide property**, now computed for every entry as the
   `x-src recall ceiling` column — see §5.1.
2. **Our own output over-merges here hardest.** At its tuned threshold
   `dblp_scholar` produces the portfolio's largest closure output cluster —
   44 records, above even its 37-record gold component (B1, §2.1). It is *not*
   unique in collapsing at lower thresholds: the completed B1 sweep shows **every
   one of the nine benchmarks** returns a single cluster at t = 0.30, so the
   giant component is a property of transitive closure, not of this dataset.
   What `dblp_scholar` is, is the dataset where the same phenomenon is visible
   from both ends — in the gold labels *and* in our output.

`dblp_scholar` also has by far the **lowest vocabulary overlap** in the portfolio
(Jaccard 0.061) — unsurprising for a 66k-record corpus against a 2.6k-record one,
and a reminder that Jaccard is type-weighted and size-sensitive; its token
coverage (0.651) is mid-portfolio.

### 5.1 The same cap, portfolio-wide: four benchmarks cannot reach recall 1.0

`dblp_scholar` is the extreme, not the exception. A two-source linkage pipeline
pairs A against B, so **no candidate set it can produce contains an intra-source
pair** — and every gold cluster spanning 3+ records emits some. Those gold pairs
are unreachable by construction, so recall is capped below 1.0 before any method
runs:

| benchmark | x-src ceiling | unreachable gold pairs | max gold cluster |
|---|---|---|---|
| `amazon_google` | **0.8396** | 16.0 % | 6 |
| `walmart_amazon` | **0.8837** | 11.6 % | 4 |
| `wdc_computers` | **0.9001** | 10.0 % | 4 |
| `abt_buy` | 0.9847 | 1.5 % | 3 |
| `dblp_scholar` | **0.3977** | **60.2 %** | 37 |
| `dblp_acm` / `febrl_person` / `fodors_zagat` / `tiny_fixture` | 1.0000 | 0 % | 2 |

The four sets at ≤ 0.95 carry the `capped-recall` tag. Note the exact
correspondence with the last column: **a ceiling below 1.0 is precisely the
signature of gold clusters larger than 2**, which is why the four 1:1 sets are
uncapped. It is the same fact the `max gold cluster` column already reported,
converted into the unit that bounds a result.

**What this changes in how a number is read.** A blocker scoring 0.84 recall on
`amazon_google` is not at 84 % of the task — it is at **100.0 % of what is
reachable**. Reporting it against a 1.0 target invents a 16-point gap that no
amount of engineering can close, and any optimizer maximizing recall there will
spend its whole budget chasing it. The ceiling belongs beside the score.

**It bounds F1 too, but not at this number.** Recall and Pair Completeness are
capped at `r` directly; F1 is capped at `2r/(1+r)`, which is strictly higher for
every `r < 1`. Carrying the recall ceiling over to F1 unchanged would mark
reachable scores impossible — the opposite error to the one this section fixes:

| benchmark | recall/PC ceiling `r` | F1 ceiling `2r/(1+r)` |
|---|---|---|
| `amazon_google` | 0.8396 | 0.9128 |
| `walmart_amazon` | 0.8837 | 0.9383 |
| `wdc_computers` | 0.9001 | 0.9474 |
| `abt_buy` | 0.9847 | 0.9923 |
| `dblp_scholar` | 0.3977 | 0.5691 |

So `amazon_google`'s published 0.756 Ditto F1 sits under a 0.9128 ceiling, not a
0.8396 one — there is real headroom, just less than 1.0 of it.

This also flags a live claim in shipped source: `src/langres/optimize.py:135-139`
asserts that all gold matches are inter-source. On four of the ten loadable
benchmarks that is false, and on `dblp_scholar` it is false for the *majority* of
gold pairs. `febrl_dedup` makes it a fifth, and the starkest: it is single-source,
so **every** one of its 6,538 gold pairs is intra-source and none is inter-source. Out of scope for this PR (a source fix with its own blast radius),
recorded here as the measurement that settles it.

Independently corroborated: stream A derived the same four ceilings
(0.9847 / 0.8396 / 0.8837 / 0.9001) from a completely separate candidate-recall
harness, and this measurement reproduces them exactly from the gold labels alone.

---

## 6. What was missing and is now measured: vocabulary overlap

Everything else in the B2 list was already computed by `DataProfileReport` — this
was re-verified against the code, not inherited from the agenda:
`LabelStructureSection` (class balance, prevalence, imbalance, cluster-size
distribution), `CorpusFieldSection` (field sparsity, string length),
`SeparabilitySection` (the pair-difficulty proxy), `HeroSection`.

**Vocabulary overlap was genuinely absent** and now ships as
`VocabularyOverlapSection` (`src/langres/data/data_profile/vocabulary.py`),
composed automatically whenever a corpus carries a two-valued `source` field. It
reports two numbers that answer different questions:

- **type Jaccard** — how much of the combined dictionary is common. Dominated by
  the long tail of one-off tokens (model numbers, OCR noise), and sensitive to a
  size imbalance between the sides.
- **token coverage** per side — the fraction of that side's token *occurrences*
  whose type also occurs on the other side. This is what a string comparator
  actually experiences while reading a record.

The two come apart, and the gap is informative: `wdc_computers` has both high
(J 0.720 / cover 0.978 — the two sides are near lexical twins), while
`walmart_amazon` has J 0.133 but coverage 0.806 — a shared common core plus two
large disjoint tails. `tiny_fixture` is the only entry that trips the
`lexical-gap` rule (0.447), which is fine: it is a 12-record toy, explicitly
"not a real benchmark" per its own `ATTRIBUTION.md`.

**What the tokenizer counts, and one bug it had.** Cross-model review caught that
the first cut of this section double-counted the blocking text. A benchmark
corpus reaches the profiler as `model_dump()`, and pydantic v2 **includes
computed fields** — every vendored ER schema defines
`embed_text` as a concatenation of other fields on the same record
(`AbtBuySchema.embed_text == name + " " + description`). Tokenizing the dump
as-is therefore counted `name` and `description` twice, over-weighting exactly
the fields a blocker reads, and disagreeing with the separability signal in the
column beside it, which iterates `model_fields` and never sees a computed field
at all. The section now excludes the schema's computed fields.

The correction is small and moves nothing structural — **type Jaccard is
unaffected** (a derived field's tokens are a subset of the fields it
concatenates, so the type set is identical), and coverage moved only where
`embed_text` joins more than one field: `abt_buy` 0.795→0.796, `amazon_google`
0.793→0.791, `febrl_person` 0.838→0.851, `fodors_zagat` 0.702→0.708,
`walmart_amazon` 0.798→0.806, with `dblp_acm`, `dblp_scholar`, `wdc_computers`
and `tiny_fixture` unchanged. No verdict changes. It is recorded here rather than
quietly fixed because the published numbers moved, and because the failure mode —
a measurement that silently disagreed with the metric printed next to it — is
the one this whole document exists to catch.

---

## 7. Portfolio-level gaps

> **Update (2026-07-28): this gap is now closed.** `febrl_dedup` (FEBRL3) is
> registered as `task="dedup"` and is in the regenerated raw data above (11
> entries, tasks `['dedup', 'linkage']`). The finding below is kept as written —
> it is the rationale the benchmark was added for — but it no longer describes
> the portfolio. See `docs/BENCHMARKS.md` §2a for the measured dedup run.

**All 10 registered entries are `task="linkage"`. There is no registered dedup
benchmark** — although `dedupe()` is the primary shipped verb and `"dedup"` is a
declared-but-unused value of `BenchmarkTask`. Every number langres reports about
its front door is measured on a cross-source linkage task with a `source` field
and prefixed ids. This is the largest single gap in the portfolio.

**Benchmarks named in our own docs but never registered:** WDC Products, Beer,
iTunes-Amazon, Alaska, Machamp (all `docs/research/20260701_er_seam_audit.md:57`;
WDC Products and Beer/iTunes also in `docs/plans/20260708_eval_readiness_plan.md`).
Two clarifications worth recording so they are not mistaken for gaps:

- **`wdc_computers` is not WDC Products.** Its own `ATTRIBUTION.md:47-48` says
  so: it is the *computers* category of the older WDC product corpus, not the
  newer WDC Products generalization benchmark (arXiv:2301.09521) whose headline
  finding is that *every* SOTA matcher degrades badly on unseen entities.
- **"matchbench" is a Hugging Face mirror org, not a benchmark.** It is the
  provenance of several vendored datasets, not a missing entry.

**Reproduction:** only 3 of 9 loadable datasets (`fodors_zagat`, `febrl_person`,
`tiny_fixture`) survive the PyPI wheel's exclude list. The other six exist only
in a git checkout. This is deliberate and licence-driven (the vendored
DeepMatcher/Magellan corpora carry attribution but no explicit redistribution
licence) — but it means a `pip` user cannot reproduce most of the portfolio, and
"failed to load" from a wheel means "excluded from this install", not "broken".

---

## 8. How the instrument was validated before any verdict was published

An instrument that has never been checked against a known value is not an
instrument. Six numbers were reproduced **exactly** from values already recorded
in this repository, none of them fed to the profiler:

| quantity | recorded | computed | source |
|---|---|---|---|
| `fodors_zagat` gold pairs | 112 | 112 | `datasets/fodors_zagat/SOURCE.md:14` |
| `dblp_scholar` records / clusters / closure pairs / largest component | 66879 / 2351 / 13763 / 37 | same | `datasets/dblp_scholar/ATTRIBUTION.md` |
| `walmart_amazon` match clusters / singletons / closure pairs | 846 / 22820 / 1092 | same | `datasets/walmart_amazon/ATTRIBUTION.md` |
| `wdc_computers` clusters / closure pairs | 877 / 1111 | same | `datasets/wdc_computers/ATTRIBUTION.md` |
| `dblp_acm` clusters (strictly 1:1) | 2220 | 2220 | `datasets/dblp_acm/ATTRIBUTION.md` |
| `febrl_person` gold pairs (1:1) | 500 | 500 | `datasets/febrl_person/SOURCE.md` |

The **wheel status** is cross-checked by a different route entirely: this harness
derives it from `pyproject.toml`'s exclude globs, while
`tests/test_wheel_contents.py::SHIPPED_NON_PY_FILES` is asserted against the
*real built artifacts*. `tests/examples/test_portfolio_profile.py` fails if the
two disagree — which is the failure mode `pyproject.toml`'s own comment warns
about ("path literals that fail silently").

**Sanity on the saturation numbers themselves:** `rapidfuzz`'s honest F1 on the
literature splits (`abt_buy` 0.2018, `amazon_google` 0.2881) sits just *below*
the recorded `RandomForestJudge` floor measured with the same instrument on the
same splits (0.4036 / 0.3596, `data/benchmarks/phase1/PHASE1_RESULTS.md`) — which
is the expected ordering: a supervised judge over the whole comparison vector
should beat one unsupervised combined similarity.

---

## 9. What this changes

- **A model ladder on `amazon_google` / `abt_buy` / `walmart_amazon` /
  `wdc_computers` is safe from saturation** — none is saturated, so there is real
  headroom to measure. **Three of the four are structurally caveated, though**:
  `amazon_google`, `walmart_amazon` and `wdc_computers` all carry `capped-recall`
  (§5.1), so their recall/PC ceilings are 0.8396 / 0.8837 / 0.9001 and their F1
  ceilings 0.9128 / 0.9383 / 0.9474 — not 1.0. `abt_buy` is the only one of the
  four that is effectively uncapped (0.9847). Headroom is real on all four; on
  three it is bounded well below 1.0, and a ladder that reports against 1.0 will
  read a labeling artifact as a model plateau.
- **…but on three of those four, blocking recall is the binding ceiling**, not
  the model: pinned Pair-Completeness is `amazon_google` 0.8388,
  `walmart_amazon` 0.8773, `wdc_computers` 0.7237 — all below the 0.90 gate
  (`abt_buy` 0.9301 is the only one that clears it). Any end-to-end number on
  those three is capped by the blocker, so a ladder that shares the pinned `k`
  must report recall alongside F1 or it will read a blocker ceiling as a model
  plateau.
- **`fodors_zagat` is a regression guard, not evidence.** Free methods reach
  within 0.047 of the whole measurable BCubed range.
- **`dblp_scholar`'s gold labels are a transitive closure**; a metric on it
  partly measures the labeling. Report cross-source recall next to PC or the 0.39
  reads as a blocking failure it is not.
- ~~**The portfolio has no dedup benchmark at all**, which is the gap most worth
  closing next given what `dedupe()` is.~~ **Closed 2026-07-28** by
  `febrl_dedup` (FEBRL3, single-source, membership gold) — see §7's update note.
