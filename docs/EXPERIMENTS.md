# Running experiments in langres

This is the getting-started for **experimenting on entity-resolution scorers** in
langres: racing cheap methods, and iterating on a DSPy LLM judge. Everything below
runs at **$0** with DSPy's `DummyLM` — no API key, no network. See the full
runnable script:

- **`examples/research/m4_experiment_loop.py`** — the whole loop end-to-end, zero-spend.
  Run it: `uv run python examples/research/m4_experiment_loop.py`.

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
- **`budget=0.0`** asserts genuine zero spend — a **post-hoc** guard that raises
  *after* a method runs if its measured spend was charged, not a pre-flight block.

### (b) `evaluate_judge_on_candidates` — pairwise-F1 for a compiled/paid judge, judged once

```python
from langres.core.benchmark import evaluate_judge_on_candidates

result, judgements = evaluate_judge_on_candidates(
    compiled_judge,        # a Matcher instance (e.g. a compiled DSPyMatcher)
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
  paid judge, pass a `BudgetedModuleRunner` via `runner=` to cap spend (enforced
  between calls — one in-flight call can overrun the cap by its own cost); this
  surface keeps the full cost knobs (`runner=` / `price_per_token_or_pair=` /
  `cost_track_fn=`) and hands you the raw judgements back. The `evaluate()`
  one-liner below is spend-capped too, but takes the simpler `budget_usd=` and
  builds the runner for you — a one-liner shouldn't make you construct one.

> **One-liner for the common case.** When you just want honest pair-level P/R/F1
> for a judge over a fixed candidate set — spend-capped by default, no raw
> judgements back — `evaluate(judge, candidates, gold_pairs)` — import it from the
> curated `langres.eval` facade (`from langres.eval import evaluate`; re-exported
> from `langres.core.benchmark`) — is the thin wrapper over
> `evaluate_judge_on_candidates` that returns only the `JudgePairEval` (default
> grid `DEFAULT_PAIR_GRID`; `slice_fn=` forwarded). The spend cap is the
> **default**, not an opt-in: `budget_usd=` resolves to `DEFAULT_BUDGET_USD` =
> $1.00 (free string/embedding judges never reach it). And with the default
> `threshold=None` it sweeps the grid for `best_threshold` while emitting a
> one-shot `UserWarning` that the number is argmax-fitted to the same gold it
> reports on (optimistically biased) — pass `threshold=<float>` to grade once at
> a fixed, honest cut. Reach for `evaluate_judge_on_candidates` when you need the
> raw judgements back or a caller-owned runner / custom cost. See
> [`docs/BENCHMARKS.md`](BENCHMARKS.md#3-score-your-own-data) for the
> bring-your-own-data walkthrough.

> **KISS warning — do NOT race a compiled/paid LLM judge through `run_methods`.**
> `run_methods`/`run_method` call the resolver factory *per grid threshold* and
> **rebuild the module uncompiled, then re-judge every time**. For a compiled DSPy
> judge that throws away your compilation; for a paid judge it multiplies spend by
> the grid size for an identical set of judgements. Use
> `evaluate_judge_on_candidates` (judged **once**) for anything compiled or paid.
>
> The same warning applies to the **trained family** (`fellegi_sunter` /
> `random_forest`, W1.2): `run_methods`/`run_method` rebuild an *unfit* module
> per grid threshold, and both judges raise `ValueError` from `forward()` until
> fit — so racing them through `run_methods` always crashes. That is exactly why
> neither name is in `ZERO_SPEND_METHODS`/`ALL_METHODS`; see "The fit seam"
> below.

## The fit seam: judges that need `Resolver.fit(...)` before scoring

Some judges are *learned*, not just configured — they need to see data before
`forward()` can score anything. `Resolver.fit(records, labels=...)` is the one
seam for this, and it dispatches on which of two runtime-checkable protocols
(`langres.core.fit`) the judge implements:

- **`UnsupervisedFitMixin.fit_unlabeled(candidates)`** — learns with **no**
  labels. `FellegiSunterMatcher` (classical Fellegi-Sunter EM) is the first
  example: it binarizes each `ComparisonVector`'s similarities into
  agree/disagree itself (never asking the comparator to emit `MISMATCH` — that
  would change `combine_present` scoring for every other judge), estimates
  u-probabilities from **random pairs** of the entities it saw (not the
  blocked candidates themselves, which are match-enriched and would bias u
  upward), and learns m-probabilities + the match prior via log-space EM.
  Called with `resolver.fit(records)` — no `labels=`.
- **`SupervisedFitMixin.fit(candidates, labels)`** — learns **with** labels.
  `RandomForestMatcher` (a Magellan-style sklearn `RandomForestClassifier` over
  `ComparisonVector.similarities`) is the example: `resolver.fit(records,
  labels=[...])`, positionally aligned with the blocked candidates. Omitting
  `labels=` raises — a trainable module that silently never trains is exactly
  the footgun this hook exists to prevent.

```python
from langres.core.resolver import Resolver
from langres.methods import make_resolver_factory

# fellegi_sunter: unsupervised (no labels)
resolver = make_resolver_factory("fellegi_sunter", benchmark)(0.5)
resolver.fit(train_records)                     # fit_unlabeled under the hood
judgements = resolver.predict(test_records)      # score_type="prob_fs"

# random_forest: supervised (labels positionally aligned with candidates)
resolver = make_resolver_factory("random_forest", benchmark)(0.5)
candidates = list(resolver._candidates(train_records))
labels = [is_match(c) for c in candidates]       # your gold lookup
resolver.fit(train_records, labels=labels)        # fit under the hood
judgements = resolver.predict(test_records)      # score_type="prob_rf"
```

Once fit, evaluate either judge the same way as a compiled DSPy judge — judged
**once** via `evaluate_judge_on_candidates(resolver.module, candidates,
gold_pairs, grid)` — never via `run_methods`.

RF's fitted forest is **not pickled**: `Resolver.save` persists it as a strict
per-tree JSON array representation (an sklearn-version guard refuses to load
across a minor-version boundary). FS's fitted state (`prior`/`m_prob`/`u_prob`)
is plain JSON floats — no sidecar file needed. Both round-trip through a
fresh-process `Resolver.load` (see `tests/core/judges/test_fellegi_sunter_judge.py`
/ `tests/core/modules/test_random_forest_judge.py`).

## The DSPy loop: build → compile → evaluate

```python
from dspy.utils.dummies import DummyLM        # zero-spend; swap for dspy.LM when paid
from langres.core.matchers.dspy_judge import DSPyMatcher
from langres.core.benchmark import evaluate_judge_on_candidates

# 1. build
judge = DSPyMatcher(lm=DummyLM([...]), model="dummy", entity_noun="product")

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
from langres.training.calibration import derive_threshold

scores = [j.score for j in judgements]
labels = [frozenset({j.left_id, j.right_id}) in gold_pairs for j in judgements]
threshold = derive_threshold(scores, labels, method="youden")   # or "percentile"
```

`derive_threshold` maximizes Youden's J over the ROC curve (needs both classes in
`labels`) and clamps the result into the observed score range. Derive on a **train**
band and report on **held-out test** so the threshold isn't tuned on the pairs it's
measured on — see `examples/research/m4_calibration.py` for the honest held-out version.

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

## Persisting & comparing runs (`capture_run` + `RunStore`)

Every surface above scores a run, prints it, and forgets it — no run id, no
config/data snapshot, no way to compare against last week's run. `capture_run`
adds the missing spine: it wraps a run, gives it a content-addressed identity, and
persists one JSONL line you can read back and diff across sessions. It is
**dependency-free** (stdlib + pydantic) and writes nothing unless you pass a `store`.

```python
from langres.architectures import FuzzyString
from langres.core import RunContext, RunStore, capture_run, compute_recipe_id

context = RunContext(
    experiment="fuzzy-string-sweep",
    resolver_config={"architecture": "FuzzyString", "threshold": 0.6},   # config snapshot (hashed)
    dataset_name="toy-companies",
    seeds={"split": 13},                                     # named seeds (hashed)
)

with capture_run(context, store=RunStore("runs/langres_runs.jsonl")) as run:
    result = FuzzyString(threshold=0.6).dedupe(records)
    run.log_metrics({"f1": 0.75}, metric_definition="pair_f1", headline_metric=0.75)
    run.record_cost(0.0)               # = SpendMonitor.spent for a paid architecture

runs = RunStore("runs/langres_runs.jsonl").read()            # list[RunRecord]
```

**What a `RunRecord` captures** — one frozen line per attempt:

- **Identity.** `recipe_id` = `sha256` over the *recipe fields* (`resolver_config`,
  `dataset_name`, `dataset_fingerprint`, `llm_model`, `seeds`, …); `attempt_id` =
  `f"{recipe_id}-{started_at}"` is the record PK.
- **Config snapshot** — `resolver_config` (best-effort; `None` for a bespoke run
  with no registered config) plus `llm_model` / `blocking_k` / `method` / `budget_usd`.
- **Provenance, recorded but *not* hashed** — `git_sha` + `git_dirty`,
  `lockfile_hash`, `langres_version`, `python_version`, `platform`. So a dirty tree
  or a `uv.lock` bump does **not** mint a new `recipe_id`: it stays a dedup key over
  the *logical* experiment (config + data + seeds), stable across code churn.
- **Metrics** (`metrics` opaque dict + `metric_definition` + `headline_metric` +
  `per_seed_metrics`), **cost** (`spend_usd`, `budget_exceeded`), **artifacts**, and
  **status** (`running` / `completed` / `failed` / `budget_exceeded` — `running` is
  written at *start*, so a crashed run leaves a visible lone line).

**The API.** `capture_run(context, *, store=None, tracker=NoOpTracker())` computes
the identity, writes the `running` line, yields a handle (`log_metrics` /
`record_cost` / `log_artifact` / `set_status`), then finalizes the terminal record
on exit. `store` accepts a path or a `RunStore`; **`store=None` writes nothing**.
`RunStore.read()` collapses each attempt's `running`+terminal lines
**last-wins-by-`attempt_id`** and takes an `fcntl.flock` per append, so several
agents can write one file safely. Pass `tracker=` (an `ExperimentTracker`) to *also*
mirror params/metrics into MLflow or W&B; omit it for the JSONL-only path.
(`git_sha()` and `dataset_fingerprint()` live in `langres.tracking.runs`.)

**How the tracking `Settings` take effect (today).** `Settings` reads
`RUN_STORE_PATH`, `MLFLOW_TRACKING_URI`, and `MLFLOW_EXPERIMENT`, but they are
*not* auto-applied by `capture_run`:

- `RUN_STORE_PATH` is **not** wired as a default `store` yet — `capture_run`'s
  `store` still defaults to `None` (writes nothing), so you pass it explicitly:
  `capture_run(context, store=Settings().run_store_path)`. Threading it as the
  zero-config default is deferred to the benchmark/harness wrap.
- `MLFLOW_TRACKING_URI` / `MLFLOW_EXPERIMENT` are consumed by the **MLflow
  tracker** (`resolve_tracker("mlflow")` / `MlflowTracker`), not by `capture_run`
  — they take effect only when you pass that tracker. Likewise `WANDB_*` is read
  by `WandbTracker`. With no tracker (`NoOpTracker`) none of these are read.

**Idempotent replay — the agent move.** LLM runs are nondeterministic, so identity
is *same recipe → same `recipe_id`*, **not** same metrics. An agent re-running a
sweep skips a config it already paid for and checks its budget in two lines:

```python
completed = [r for r in RunStore("runs/langres_runs.jsonl").read() if r.status == "completed"]
already_ran = compute_recipe_id(context) in {r.recipe_id for r in completed}   # skip if True
remaining = 5.0 - sum(r.spend_usd for r in completed)                          # budget left
```

`RunContext.parent_run_id` threads lineage — a sweep parents its per-seed children;
a DSPy-compile run parents the eval runs that reuse its compiled program.

Runnable, zero-spend end to end: **`examples/research/experiment_tracking_demo.py`**
captures a two-threshold sweep, reads it back, and prints the two-run metric diff
plus the agent two-liner. Run it:
`uv run python examples/research/experiment_tracking_demo.py`.

## Self-tuning: the autoresearch loop (`langres.optimize`)

Every surface above measures *one* configuration you chose. The **autoresearch
loop** closes the outer loop: it **propose**s configs from a search space,
**run**s and **evaluate**s each into metrics, and **keep**s the one an
`Objective` prefers — a small, deterministic hill-climber. Because ER F1
saturates near 99%, the loop steers on a **loss-like signal, not a thresholded
F1**: `candidate_recall@budget`, `log_loss`, or a quality×cost Pareto front.

`langres.optimize` is the one-call facade; it is **import-light** (`optimize` and
`score_blocking` are root exports, but every heavy import — faiss, the benchmark
loader — is lazy inside the call, so a bare `import langres` never pulls the
[semantic] stack; `tests/test_import_budget.py` guards this).

```python
from langres import optimize
from langres.autoresearch.objective import Objective
from langres.autoresearch.search_space import SearchSpace

# 1. A SearchSpace is a declarative Cartesian grid of blocker configs. k_neighbors
#    is the INNERMOST axis, so one vector index is built per
#    (embedding_model, metric, text_field) group and reused across every k.
space = SearchSpace(
    blocker=("vector",),
    embedding_model=("all-MiniLM-L6-v2",),   # local + free ($0, offline)
    metric=("cosine", "L2"),
    text_field=("embed_text", "title"),
    k_neighbors=(5, 10, 20, 40, 80),         # ascending: recall climbs with k
)

# 2. An Objective is the immutable keep-if-better rule. Maximize a continuous
#    recall signal subject to a reduction-ratio floor (>=98.5% of the |A|x|B|
#    comparisons eliminated) — a real recall@budget tradeoff, not a saturated F1.
objective = Objective.maximize(
    "candidate_recall", subject_to=[("reduction_ratio", ">=", 0.985)]
)

# 3. optimize() loads the benchmark once, drives the loop, and persists EVERY
#    trial (accepted + rejected) to the local JSONL at store=.
result = optimize(
    space, objective, "amazon_google",
    store="tmp/autoresearch/ag_blocking.jsonl",   # gitignored; store=None writes nothing
    seed=0,
)
print(result.best_config, result.best_metrics)     # the winning feasible config
```

**Defining the `Objective` — three ergonomic constructors.** An `Objective`
bundles one or more goals with zero or more feasibility constraints
(`(metric, op, threshold)` triples, `op` in `>= <= > <`). A candidate that
violates any constraint is never kept; among feasible candidates the incumbent is
displaced only by a strict Pareto win (`>=` on every goal, `>` on at least one) —
ties and incomparable trade-offs keep the incumbent, so the loop is monotone.

```python
# single maximize goal, optionally constrained
Objective.maximize("candidate_recall", subject_to=[("reduction_ratio", ">=", 0.985)])

# single minimize goal (e.g. log_loss, cost) — the matching vertical's steering signal
Objective.minimize("log_loss")

# multi-objective Pareto front — never scalarized (recall up, work down)
Objective.pareto([("candidate_recall", "maximize"), ("reduction_ratio", "maximize")])
```

**Where results are stored — local JSONL only, today.** `optimize(store=...)`
appends **every** trial — accepted, over-budget rejects, and scorer failures — as
one line to a local `RunStore` JSONL (the same `tracking.runs` spine as
`capture_run` above), so the full audit trail is durable off-git. `store=None`
persists **nothing** (the offline path); the same `LoopResult` is returned either
way. Read the trail back with `RunStore(path).read()`:

```python
from langres.tracking.runs import RunStore

records = RunStore("tmp/autoresearch/ag_blocking.jsonl").read()   # last-wins per attempt
accepted = [r for r in records if (r.metrics or {}).get("accepted") == 1.0]
```

The `RunStore` JSONL is the durable local record; `store=None` skips it entirely
(the offline path). For a durable, off-laptop **dashboard**, pass a `tracker=`
**spec** — a backend name, an already-built instance, a sequence of either
(fan-out), or `None` (the default, no-op) — and `optimize` resolves it internally
via `resolve_tracker`, mirroring the `matcher="..."` string dispatch:

```python
# Local-first: no credentials, no network -- writes to a local trackio SQLite store.
result = optimize(space, objective, "amazon_google", tracker="trackio")

# An instance configures the backend (e.g. HF-Space sync); a sequence fans out
# to several backends at once.
from langres.tracking.trackers import TrackioTracker
result = optimize(space, objective, "amazon_google",
                   tracker=[TrackioTracker(space_id="user/space"), "mlflow"])
```

Set `HF_TOKEN` and configure a `space_id` (`TrackioTracker(space_id="user/space")` or
`TRACKIO_SPACE_ID`) to additionally sync the dashboard to a Hugging Face Space (and
`TRACKIO_DATASET_ID`/`dataset_id` for a persistent HF Dataset) — omitted, this stays
local-only. Every trial the loop already logs to `RunStore` (accepted, over-budget
rejects, and scorer failures alike) also streams to the tracker via the same
`capture_run` call, so nothing needs to be wired twice. Also deferred: swapping the
exhaustive grid proposer for an Optuna/LLAMBO one, and the **matching vertical**
(steering a judge on `log_loss` / AUC-PR) plus small-LM fine-tuning. **M1 proves the
loop on the blocking vertical only** (epic #145).

### Worked proof — climbing blocking recall@budget on amazon_google

`examples/research/blocking_recall_autoresearch.py` runs exactly the loop above
over the 20-config grid at **$0, offline** (local MiniLM embeddings, no LLM; the
benchmark is vendored). amazon_google is a genuinely hard, *unsaturated* two-source
linkage benchmark — blocking recall plateaus ~0.84, never reaching a 0.90 gate —
which is what makes the climb and the budget tradeoff honest rather than a
foregone conclusion. A measured run:

- The incumbent `candidate_recall` **climbs** as `k` grows within the winning
  `(metric, text_field)` group: **0.7568 → 0.8129 → 0.8317 → 0.8388** for
  `k = 5 → 10 → 20 → 40`.
- `k = 80` is **rejected as over-budget** — it spends ~2× the comparisons for no
  extra recall, so its `reduction_ratio` (0.9776) breaches the 0.985 floor. The
  loop keeps the best *feasible* incumbent. The budget is a real gate, not
  decoration.
- **20 trials logged (4 accepted / 16 rejected)** to the gitignored
  `tmp/autoresearch/` JSONL; ~41 s wall-clock, **$0**.

```bash
uv run --env-file .env python examples/research/blocking_recall_autoresearch.py
# needs the [semantic] extra: uv sync --all-extras --no-extra finetune
```

## W3 paid smoke — SelectMatcher vs pairwise, measured

`examples/research/w3_paid_smoke.py` is the ≤$10, SpendMonitor-capped operator run
that puts both surfaces above on a real model at once: it grades a **set-wise
`SelectMatcher`** (one LLM call per anchor group) against an ordinary **pairwise
judge** (one call per pair), same model on both arms, side by side on
Amazon-Google via `evaluate_judge_on_candidates` / `pair_pr_curve`. Verified at $0
with `DummyLM` in `tests/examples/test_w3_paid_smoke.py`; the single paid run cost
**$4.65 total** across two model points.

**The measured finding is honest and nuanced — set-wise is *not* a clean win.** It
edges *ahead* of pairwise on the frontier model (gpt-4o, +0.049 F1) but falls
*behind* on the mid-tier model (gpt-4o-mini, −0.068 F1): the one-call-per-group
task is harder multi-candidate reasoning, and a weaker judge handles it less well.
Set-wise consistently trades precision for recall, makes **3–5× fewer LLM calls**
but costs **more dollars** (token-heavy group prompts), and ComEM's published +16
F1 does **not** replicate at that magnitude here. Full two-point table, P/R split,
and reproduction commands:
[`docs/research/20260703_w3_paid_smoke_results.md`](research/20260703_w3_paid_smoke_results.md).

## Signal log — the flywheel inlet (`JudgementLog`)

Every `ERModel`'s `.compare()`/`.dedupe()` take an opt-in, keyword-only `log=` (a
`langres.JudgementLog` or a path — `None` by default, zero overhead):

```python
from langres import JudgementLog
from langres.architectures import FuzzyString

log = JudgementLog("runs/judgements.jsonl")
result = FuzzyString(threshold=0.6).dedupe(records, log=log)

rows = log.read()  # round-trips every line written
```

Every judge call appends one JSON line: pair ids, `score`, the judge's own
`decision` plus the caller's `verdict` (its `predicted_match` — a decider's
`decision`, else `score >= threshold`), the judge's `confidence` /
`confidence_source`, `model`, `cost_usd`, `decision_step`, optional explicit
topology `stage_id`, `timestamp`, and a schema-version field `"v": 4` (bumped
1→2→3 for the decision contract and 3→4 for stage-aware logging; `read()`
backfills `decision` from `verdict` for older v1/v2 rows). A retrieval or
reranking stage that does not feed a binary threshold directly records
`verdict=null`, rather than being mislabeled with a later match cut. Record content is **off by
default** — pass `JudgementLog(path, features=True)` to additionally log
`reasoning` and the judge's raw `provenance` (comparison levels,
similarities, token counts, ...): this may contain PII (the record content a
judge reasoned over), and JSONL is plaintext on disk.

Implementation note: `JudgementLog` is a plain file sink, not a `Matcher`.
`log=` wraps the resolved matcher in a `LoggingMatcher` — a small boundary
component (the same pattern `SpendCappedMatcher` uses) that logs each
`PairwiseJudgement` as it streams past without materializing the whole
judgement stream. It is a transient, per-call wrapper built inside `_scorer()`
around the model's current `matcher` slot — it never touches `self.module`, so
it is intentionally excluded from a saved artifact: `.save()` persists the
model's components, never a `log=` wrapper from one past call.

This is the flywheel's inlet; the harvest half is below.

## Flywheel harvest — verdicts + corrections → a better threshold (`langres.core.harvest`)

The outlet of the flywheel (W2.4): turn a `judgements.jsonl` log plus a
`corrections.jsonl` review-queue export into **labeled pairs**, and feed them to
`derive_threshold` — its first production caller. langres owns the contract and
the harvest; the human-review UX (the queue a reviewer clicks) stays downstream.

```python
from langres import JudgementLog
from langres.core.harvest import (
    CorrectionLog, harvest_labeled_pairs, derive_threshold_from_pairs,
)

rows = JudgementLog("runs/judgements.jsonl").read()      # the inlet's output
corrections = CorrectionLog("runs/corrections.jsonl").read()

pairs = harvest_labeled_pairs(rows, corrections)  # verdicts as weak labels,
                                                  # corrections overriding them
threshold = derive_threshold_from_pairs(pairs)    # data-driven, not hand-set
```

`harvest_labeled_pairs` emits one `LabeledPair` per judgement row; its label is
the logged `verdict` (a weak label) unless a `Correction` covers the same pair
(matched order-independently by id set), in which case the human label wins and
`source="correction"` records the override. Deriving from verdicts **alone** just
recovers the judge's own cut — self-training on your own labels teaches nothing;
the human corrections are what carry new signal.

**The two flywheel JSONL schemas:**

- `judgements.jsonl` (written by `JudgementLog`) — `{"v":1, "left_id", "right_id",
  "score", "verdict", "model", "cost_usd", "decision_step", "timestamp"}`.
- `corrections.jsonl` (the `Correction` contract, written by a review tool) —
  `{"v":1, "left_id", "right_id", "label"}` required, plus optional audit fields
  `original_score` / `original_verdict` / `reviewer` / `timestamp`.

Runnable demo: `examples/flywheel_threshold_harvest.py` reads committed
Fodors-Zagat fixtures, derives the threshold before vs. after corrections, and
scores both on a **held-out gold** split (never used to derive the threshold). 40
simulated corrections move held-out pair-F1 from ~0.56 to ~0.71 — a real gain in
the correct direction, not circular self-training. Regenerate the fixtures at $0
with `examples/data/flywheel/generate_fixtures.py`.

## See also

- [`docs/BENCHMARKS.md`](BENCHMARKS.md) — the benchmark portfolio + registry
  (`list_benchmarks` / `get_benchmark`), and the `evaluate()` bring-your-own-data
  scoring walkthrough.
- `examples/research/portfolio_race.py` — registry-driven race over every loadable
  benchmark (offline by default; optional capped LLM row).
- `examples/research/blocking_recall_autoresearch.py` — the autoresearch loop
  (`langres.optimize`) hill-climbing blocking recall@budget on amazon_google, $0/offline.
- `examples/research/m4_experiment_loop.py` — the runnable zero-spend loop documented here.
- `examples/research/m4_dspy_judge.py` — DSPyMatcher compile + save/load round-trip.
- `examples/research/m4_calibration.py` — honest held-out `derive_threshold` lift on AG.
- `examples/judgement_log_demo.py` — `JudgementLog` write-then-read round-trip.
- `examples/flywheel_threshold_harvest.py` — harvest verdicts + corrections → a
  re-derived threshold, with before/after held-out gold F1.
- `examples/research/experiment_tracking_demo.py` — persist runs with `capture_run`
  + `RunStore`, diff two runs across sessions, and the agent idempotency/budget two-liner.
- `examples/research/m3_race.py` / `examples/research/m3_zero_spend_race.py` — multi-method races.
