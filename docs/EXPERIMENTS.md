# Running experiments in langres

This is the getting-started for **experimenting on entity-resolution scorers** in
langres: racing cheap methods, and iterating on a DSPy LLM judge. Everything below
runs at **$0** with DSPy's `DummyLM` — no API key, no network. See the full
runnable script:

- **`examples/m4_experiment_loop.py`** — the whole loop end-to-end, zero-spend.
  Run it: `uv run python examples/m4_experiment_loop.py`.

## Two experiment surfaces — and when to use each

langres gives you two measurement surfaces. Pick by **what you are measuring** and
**how expensive the scorer is**.

### (a) `run_methods` — full-pipeline race, for cheap zero-spend methods

```python
from langres.core.benchmark import run_methods
from langres.data.amazon_google import AmazonGoogleBenchmark

table = run_methods(AmazonGoogleBenchmark(), ["rapidfuzz", "embedding_cosine"], budget=0.0)
print(table.best().method)          # winner by pair-F1
print(table.to_markdown())          # BCubed-F1 + pair-F1 + cost per method
```

- **What it measures:** the *full pipeline* — block → judge → tune threshold →
  cluster → grade **BCubed F1** (post-clustering) *and* pair-F1.
- **How it works:** for each method it builds a resolver factory
  (`make_resolver_factory`) and runs `run_method`, which calls
  `resolver_factory(threshold)` **repeatedly** — once per grid threshold while
  tuning, again for the pair curve, again for test. Each call rebuilds the module
  and **re-judges** the candidates.
- **When to use:** **cheap, zero-spend** methods (`rapidfuzz`, `embedding_cosine`,
  `weighted_average`, …) where rebuild-and-re-judge-per-threshold is free.
- **`budget=0.0`** asserts genuine zero spend — it raises if any method is charged.

### (b) `evaluate_judge_on_candidates` — pairwise-F1 for a compiled/paid judge, judged once

```python
from langres.core.benchmark import evaluate_judge_on_candidates

result, judgements = evaluate_judge_on_candidates(
    compiled_judge,        # a Module instance (e.g. a compiled DSPyJudge)
    candidates,            # a FIXED candidate set (a pair split, or a blocked band)
    gold_pairs,            # set[frozenset[str]] — order-independent true matches
    grid=(0.1, 0.3, 0.5, 0.7, 0.9),
)
print(result.pair.f1, result.pair.precision, result.pair.recall, result.best_threshold)
```

- **What it measures:** **pairwise precision/recall/F1** on a *fixed* candidate set
  at the best-F1 grid threshold, plus the full PR curve. No blocking, no
  clustering — so the number is directly **comparable to pairwise-F1 SOTA** with no
  blocking-recall ceiling or clustering amplification.
- **How it works:** takes a **module instance**, judges the candidate set **exactly
  once**, and grades it. Returns `(JudgePairEval, list[PairwiseJudgement])` — the
  graded summary plus the raw judgements (kept in-process for error-map analysis).
- **When to use:** a **compiled and/or paid** scorer — this is the **DSPy
  experimentation surface** and the SOTA-comparable precision measurement. For a
  paid judge, pass a `BudgetedModuleRunner` via `runner=` to hard-cap spend.

> **KISS warning — do NOT race a compiled/paid LLM judge through `run_methods`.**
> `run_methods`/`run_method` call the resolver factory *per grid threshold* and
> **rebuild the module uncompiled, then re-judge every time**. For a compiled DSPy
> judge that throws away your compilation; for a paid judge it multiplies spend by
> the grid size for an identical set of judgements. Use
> `evaluate_judge_on_candidates` (judged **once**) for anything compiled or paid.

## The DSPy loop: build → compile → evaluate

```python
from dspy.utils.dummies import DummyLM        # zero-spend; swap for dspy.LM when paid
from langres.core.modules.dspy_judge import DSPyJudge
from langres.core.benchmark import evaluate_judge_on_candidates

# 1. build
judge = DSPyJudge(lm=DummyLM([...]), model="dummy", entity_noun="product")

# 2. compile  (BootstrapFewShot is the zero-spend path; "mipro" for MIPROv2)
judge.compile(trainset, optimizer="bootstrap")   # trainset = list[dspy.Example]

# 3. evaluate the COMPILED judge, once
result, judgements = evaluate_judge_on_candidates(judge, test_candidates, test_gold, grid)
```

- **Candidates** come from a fixed pair split via
  `langres.data.amazon_google.load_amazon_google_pair_splits()` — it returns
  `{"train"/"valid"/"test": [(amazon_id, google_id, label), ...]}`. Look each id up
  in the corpus (`load_amazon_google()`) to build `ERCandidate`s and the
  `gold_pairs` frozenset (see `build_candidates` in the example).
- **`trainset`** is `list[dspy.Example]` with `left`/`right` inputs (the same
  rendering `forward` uses) and a boolean `match` — see `to_trainset` in the
  example.
- **Paid runs:** swap `DummyLM` for a real `dspy.LM`. Keep the same three steps; add
  a `runner=BudgetedModuleRunner(...)` to `evaluate_judge_on_candidates` and a
  `SpendMonitor` (below) so spend stays capped and observed.

## Data-driven thresholds (kill the magic constants)

Don't hand-set `0.5`. Derive the operating point from the score distribution:

```python
from langres.core.calibration import derive_threshold

scores = [j.score for j in judgements]
labels = [frozenset({j.left_id, j.right_id}) in gold_pairs for j in judgements]
threshold = derive_threshold(scores, labels, method="youden")   # or "percentile"
```

`derive_threshold` maximizes Youden's J over the ROC curve (needs both classes in
`labels`) and clamps the result into the observed score range. Derive on a **train**
band and report on **held-out test** so the threshold isn't tuned on the pairs it's
measured on — see `examples/m4_calibration.py` for the honest held-out version.

## Budget monitoring (`SpendMonitor`, ≤ $5)

```python
from langres.clients.openrouter import SpendMonitor

monitor = SpendMonitor(budget_usd=5.0)     # warns at 80%, raises past the cap
monitor.add(result.cost.usd_total)         # accumulate honest per-run spend
monitor.check()                            # warn / raise on cumulative spend
print(f"${monitor.spent:.2f} spent, ${monitor.remaining:.2f} left")
```

`SpendMonitor` is a KISS cumulative-cost **ledger** — it observes and warns/raises,
it does not throttle the LM. For a hard pre-flight cap on the run itself, wrap the
judge in a `BudgetedModuleRunner` and pass it to `evaluate_judge_on_candidates`. On
the zero-spend `DummyLM` path both report **$0.00**.

## See also

- `examples/m4_experiment_loop.py` — the runnable zero-spend loop documented here.
- `examples/m4_dspy_judge.py` — DSPyJudge compile + save/load round-trip.
- `examples/m4_calibration.py` — honest held-out `derive_threshold` lift on AG.
- `examples/m3_race.py` / `examples/m3_zero_spend_race.py` — multi-method races.
