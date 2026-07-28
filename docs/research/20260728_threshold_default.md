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

<!-- TLDR -->

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

---

## 2. The result

<!-- MAIN TABLE -->

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

This is not hypothetical breakage measured against imagined users: the repo's own
tests, examples and docs are full of exactly these call shapes — `.fit(records)`
(6), `.fit(records, labels=…)` (4), `.fit(…, method=…)` (20+ across `Platt`,
`Bootstrap` and the finetune methods).

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
