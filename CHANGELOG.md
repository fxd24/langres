# Changelog

## [Unreleased]

### The model is explicit: `ERModel` + named architectures — **breaking**

Nothing in langres named a *whole* ER pipeline. `link()`/`dedupe()` took a
`matcher=` string × a `model=` string, and the default `matcher="auto"` sniffed
`OPENROUTER_API_KEY`/`OPENAI_API_KEY` out of your environment (or a `.env` file
you had forgotten about) and **spent real money** on whatever it found. You could
not tell what model was underneath without reading the source.

Now you name the model, and it is a class you can read:

```python
from langres.architectures import FuzzyString, VectorLLMCascade

FuzzyString().dedupe(records)                                  # $0, offline, no key
VectorLLMCascade(llm="openrouter/openai/gpt-4o-mini").dedupe(records)   # paid, because you said so
```

`FuzzyString` costs nothing because it has no paid model slot — not because a
heuristic guessed well or a cap held.

#### Added

- **`langres.architectures`** — named ER architectures. `FuzzyString` (all-pairs
  + string similarity; $0, offline, deterministic, no network) and
  `VectorLLMCascade` (vector blocking + a free embedding student + an LLM
  escalated only at the uncertain margin). One self-contained file each.
- **`ERModel`** — the reshaped `Resolver` (`Resolver` remains as a plain alias;
  `Resolver is ERModel`). New: `.dedupe(records)`, `.compare(a, b)`,
  `.from_components(...)`, `.backbone`, `.schema`, `.is_bound`.
- **`ERModel.compare(a, b) -> LinkVerdict`** — replaces the verb `link(a, b)`
  (one pair). Deliberately not named `link`: `ERModel.link(left, right)` remains
  a reserved cross-source stub, and two incompatible things were called `link`
  before this.
- `langres.core.inputs` (schema inference / record normalization) and
  `langres.core.results` (`LinkVerdict`, `DedupeResult`) — lifted out of the
  verbs as reusable contracts. **`DedupeResult` is now root-exported**; it never
  was, despite being what `dedupe()` returned.
- `log=` on `.dedupe()`/`.compare()` — the flywheel inlet, now per call and
  composed *inside* the spend cap rather than by mutating the matcher slot.

#### Removed — **breaking, no shim**

- **`langres.link` / `langres.dedupe`** (the module-level verbs) and
  `langres.verbs`. Use a model's `.dedupe()`/`.compare()`.
- **`matcher="auto"` / `judge="auto"`**, `choose_auto_judge`,
  `DEFAULT_AUTO_MODEL`, `NoMatcherAvailableError`, and all of
  **`langres.core.presets`** (`build_judge`, `resolve_judge`, `build_resolver`,
  `default_threshold_for`, `PAID_JUDGE_NAMES`, `MatcherName`). Deleted, not
  relocated. `Resolver.from_schema(matcher="auto")` now raises and names a
  concrete architecture instead.
- **`LANGRES_OFFLINE` / `Settings.langres_offline`** — deleted outright, not just
  disconnected. It only ever gated the `auto` path, so once that went it was a
  *documented spend guard that did nothing* — the worst failure mode a safety
  switch has. There is no auto-decision left to disarm: naming a paid
  architecture is now the only way to spend. `Settings` uses `extra="ignore"`,
  so a leftover `LANGRES_OFFLINE=1` in an existing `.env` is ignored rather than
  a `ValidationError`.
- `default_threshold_for` has no replacement, deliberately: a bound model carries
  its own cut in its `Clusterer`, so `compare()` reads `self.clusterer.threshold`.
  The name→threshold table existed only because the verbs had no model to ask.

#### Changed

- **`LinkVerdict`/`DedupeResult` fields**: `judge_used` → **`architecture`** (the
  model class that ran) and `model` → **`backbone`** (the LLM id / embedder name,
  or `None` when nothing with weights ran). The two were conflated before; the
  whole point of an architecture is that swapping a backbone does not change it.
- **Schema inference with an *injected* matcher is gone.**
  `dedupe(records, matcher=MyMatcher())` inferred a schema; the equivalent,
  `ERModel.from_schema(MySchema, matcher=MyMatcher())`, requires a real one. A
  named architecture (`FuzzyString()`) is still schema-optional. This is the
  production path regardless: an inferred schema is an ephemeral class a fresh
  process cannot import, so it could never round-trip through `save()`/`load()`.
- `examples/quickstart_verbs.py` → `examples/quickstart_models.py`.

#### Moved out of `langres.core` — **breaking for deep imports, no shim**

`langres.core` had grown to 67% of the source tree by carrying code that is not
entity-resolution modelling. These moves are **relocations, not deletions**: every
symbol still exists, and every *supported* import path still works. Only the deep
`langres.core.*` paths break. No compatibility shims: a re-export relay would
re-create the exact import edges these moves exist to remove (measured — see
`tests/test_import_tangle.py`, which documents why a relay makes the graph worse
while making the metric look better).

- **The HTML/SVG render seam** → **`langres.report`**. `langres.core.eval_report`
  → `langres.report.eval_report`; `langres.core._svg`/`langres.core._report_html`
  (both private) → `langres.report.*`. **`langres.EvalReport` and
  `langres.eval.EvalReport` are unchanged** — those are the supported paths, and
  `EvalReport` was never in `langres.core.__all__`. Only the deep path advertised
  in the old README breaks.
- **The autoresearch/optimizer engine** → **`langres.autoresearch`**, behind the
  unchanged **`langres.optimize`** facade.
  `langres.core.optimizers` → `langres.autoresearch.blocker_optimizer`;
  `langres.core.autoresearch.*` → `langres.autoresearch.*`. **`langres.optimize()`
  and `score_blocking()` are unchanged** — the facade is the supported surface.
  `BlockerOptimizer` was in `langres.core.optimizers.__all__`, but never in
  `langres.__all__` or `langres.core.__all__`, so it was reachable only by deep
  import — the path documented in `docs/TECHNICAL_OVERVIEW.md`, which breaks.

  **Both import forms work**, as they did on the old `langres.core.optimizers`:

  ```python
  from langres.autoresearch.blocker_optimizer import BlockerOptimizer   # works

  import langres.autoresearch.blocker_optimizer as bo                   # works
  import langres.autoresearch.blocker_optimizer
  langres.autoresearch.blocker_optimizer.BlockerOptimizer               # works
  ```

  The engine gets its own top-level name because **`langres.optimize` is a
  callable**: that is the front door, and the root package binds the name to the
  function. Anything under that name is therefore unreachable by attribute
  traversal — traversal finds the function and stops. An interim layout put the
  engine there and silently broke the two dotted forms above (`ImportError` /
  `AttributeError`) while the `from` form kept working; no test caught it,
  because every in-repo call site uses the `from` form. Splitting the names apart
  fixes it: the engine is the `langres.autoresearch` package, the facade stays a
  *module* at `langres/optimize.py` (so plain `import langres.optimize` keeps
  working too), and `tests/test_import_budget.py` now asserts all three forms.

- **The human-in-the-loop labelling + cold-start seam** → **`langres.curation`**.
  `review`, `harvest`, `anchor_store` and `canonicalizer` left `langres.core`, and
  **all of `langres.bootstrap` dissolved in** (`langres.bootstrap` no longer
  exists). *Supported surfaces are unchanged*: `langres.core.Correction`,
  `CorrectionLog`, `harvest_labeled_pairs`, `derive_threshold_from_pairs`,
  `select_for_review`, `ReviewQueue`, `align_pairs`, `LabeledPair` are still on the
  `langres.core` facade (re-exported from `langres.curation.*`), so
  `len(langres.core.__all__)` is unchanged. `AnchorStore`/`Canonicalizer` were
  already deep-import-only (never on the facade) — import them from
  `langres.curation.anchor_store` / `langres.curation.canonicalizer`.

  Unlike the two relocations above (no shim — a `core` relay recreates the very
  edges the move removes), the four deep module paths `langres.core.{review,
  harvest,anchor_store,canonicalizer}` keep **temporary re-export shims** marked
  `# TEMPORARY: deleted by the W2 sweep`. They are acyclic leaf modules nothing on
  the eager path imports, so they add no cycle — `tests/test_import_tangle.py` is
  unchanged at `[9, 3, 2, 2, 2]` / 18. **`langres.bootstrap` gets no shim**
  (breaking for deep `langres.bootstrap` imports): import its contents from
  `langres.curation` (`Bootstrapper`, `HardNegativeMiner`, `GoldPair`, `GoldSet`,
  `GroundTruthLabeler`, `TeacherLabeler`, …), all still lazy where they were.
  `data/mining.py` deliberately **stays** in `langres.data`: moving it would add a
  new eager `langres.data → langres.curation` edge (its functions are re-exported
  from `langres.data`) while it imports no `data/` internals to relocate.

#### Observability moved to `langres.tracking` — non-breaking (shims kept)

Run identity/persistence and experiment tracking are **observability, not ER
modelling**, so they now live in a sibling `langres.tracking` package instead of
inside `langres.core`:

- `langres.core.runs` → `langres.tracking.runs` (`RunContext`/`RunRecord`/`RunStore`,
  `capture_run`, `compute_recipe_id`, `git_sha`, `dataset_fingerprint`)
- `langres.core.judgement_log` → `langres.tracking.judgement_log`
  (`JudgementLog`, `LoggingMatcher`)
- `langres.core.trackers` → `langres.tracking.trackers` (the `ExperimentTracker`
  Protocol, `NoOpTracker`/`MultiTracker`, `resolve_tracker`, and the lazy
  `MlflowTracker`/`WandbTracker`/`TrackioTracker` adapters)
- `langres.clients.tracking` → `langres.tracking.factories`
  (`create_wandb_tracker`, `create_trackio_tracker`)

Unlike the `langres.core` moves above, **this one keeps back-compat**: every old
path is a re-export shim (marked `# TEMPORARY: deleted by the W2 sweep`), the
`langres.core` facade still re-exports the primitives (`from langres.core import
RunStore, JudgementLog, resolve_tracker` unchanged), and
`langres.clients.create_wandb_tracker`/`create_trackio_tracker` are unchanged.
Prefer the new `langres.tracking.*` paths; the shims exist only until the W2
sweep. The backends stay lazy — a bare `import langres` still pulls no
`mlflow`/`wandb`/`trackio`/`huggingface_hub` — and `optimize(tracker="trackio")`
still resolves the backend by string internally.

Measured: the package import graph is unchanged from before the move — 7
`core → tracking` edges (6 toplevel), all pre-existing (the `langres.core` facade
re-exporting the primitives, plus the `dspy`/`llm` matchers that capture runs),
and the module-level SCCs are identical (`tests/test_import_tangle.py`).

#### Moved to `langres.metrics` — internal, back-compat **shims kept**

The evaluation/diagnostics metrics left `langres.core`: `core/metrics.py`
(BCubed/pairwise/ranking ER metrics), `core/analysis.py` (blocker-analysis),
`core/debugging.py` (`PipelineDebugger`) and `core/diagnostics.py` (error-case
models) now live in the new **`langres.metrics`** package. Metrics *score* a
resolution; they are not the modelling contract, so they sit beside `langres.core`,
not in it.

Unlike the `report`/`autoresearch` moves above, this wave **keeps back-compat
shims** at every old path (`langres.core.metrics`/`.analysis`/`.debugging`/
`.diagnostics` still import and re-export their full public surface), so no
supported *or* deep import breaks. The `langres.eval` facade (`reduction_ratio`,
`classify_pairs`, `roc_auc_score`, …) is unchanged. Shims carry a
`# TEMPORARY: deleted by the W2 sweep` marker; the sweep repoints the remaining
in-repo callers and removes them.

Measured (`tools/import_graph.py kinds`): the runtime SCC stays **empty** and the
all-edges tangle is **unchanged at `[9, 3, 2, 2, 2]` / 18** — the shims did not
re-knot the graph, because metrics is not in the eager `import langres` path and
`core/reports.py`'s one lazy edge into `analysis` was repointed at
`langres.metrics.analysis` (leaving it on the shim would have grown the
`{analysis, reports, plotting.blockers}` knot from 3 to 4). That knot is
**relocated, not removed**: it is now `{metrics.analysis, core.reports,
plotting.blockers}`. The CI core-contract coverage gate's `--include` glob does
not yet cover `src/langres/metrics/*`; those files measure ~99% and should be
added to it so the contract tier stays gated.

#### Moved to `langres.training` — deep-import shims kept (temporarily)

Fitting/calibrating a matcher is what *produces* a tuned model, not
entity-resolution modelling itself, so the fit family moved out of
`langres.core` into a new **`langres.training`** package beside it (the same cut
that moved `langres.report` out): `core.finetune` → `training.finetune`,
`core.calibration` → `training.calibration`, `core.fit_report` →
`training.fit_report`, `core.methods_prompt` → `training.methods_prompt`,
`core.methods_calibrate` → `training.methods_calibrate`.

**Every supported path is unchanged** — `langres.QLoRA` / `run_finetune` /
`finetune` / `derive_threshold`, and `langres.core.FitReport` / `Bootstrap` /
`MIPRO` / `GEPA` / `Platt` / `Isotonic` / `Calibrator` all still resolve
(`langres.__all__` = 36, `langres.core.__all__` = 76, both unchanged). Unlike the
`report`/`autoresearch` moves above, the deep `langres.core.*` paths here **also
keep working**, through back-compat shims at each old path marked
`# TEMPORARY: deleted by the W2 sweep`. The shims are fan-in-0 leaves (nothing
in-repo imports them), so they do **not** re-knot the import graph — the tangle
is unchanged (`[9, 3, 2, 2, 2]` all-edges, `[]` runtime; measured). `core →
training` is non-zero by design (`ERModel.fit` dispatches into `training` via
function-local imports, plus one toplevel `fit_report` for the `fit_report_`
attribute type, and the `core/_exports/_training` public surface) — see
`tests/test_import_tangle.py`.

#### Logger names follow the code that moved

Every logger here is `getLogger(__name__)`, so a module that moves takes its
logger name with it. Several moves in this release rename an emitting logger —
including the whole `langres.curation` extraction above (`review`, `harvest`,
`anchor_store`, `canonicalizer`, `bootstrapper`, `labelers` now emit under
`langres.curation.*` instead of `langres.core.*` / `langres.bootstrap.*`). The
table below lists the `ERModel`/engine moves; the same guidance applies to all of
them:

| records | was (0.3.0) | now |
|---|---|---|
| "Saved Resolver artifact to %s", the `langres_version` mismatch warning | `langres.core.resolver` | `langres.core._model_persist` |
| "Embedding %d records…" | `langres.core.resolver` | `langres.core._model_run` |
| "Optimization complete. Best parameters: %s", "Best value: %.4f", the wandb notice | `langres.core.optimizers.blocker_optimizer` | `langres.autoresearch.blocker_optimizer` |
| git-unavailable / malformed-run-record warnings | `langres.core.runs` | `langres.tracking.runs` |
| the created-tracker notice | `langres.clients.tracking` | `langres.tracking.factories` |
| the tracker-error `logger.exception` | `langres.core.trackers` | `langres.tracking.trackers` |

Nothing in the codebase filters on a logger name, and anyone configuring at
`langres` — or at `langres.core` for the first two — is unaffected via the
hierarchy. But a caller who pinned the exact module —
`logging.getLogger("langres.core.resolver").setLevel(INFO)` to watch saves, or
`…("langres.core.optimizers.blocker_optimizer")` to watch a study finish —
silently stops seeing those records. Configure at `langres` instead. This is
inherent to moving code between modules, not a change of intent.

#### Known limitation — `VectorLLMCascade` cannot `save()`

`VectorLLMCascade(...).save(path)` raises `NotImplementedError`, by design,
naming the real gap. `FuzzyString` saves and loads today.

Its `VectorBlocker` is built with a `text_field_extractor` **closure** (blocking
text = every comparable field concatenated), and a callable cannot round-trip
through a JSON config. This is **inherited, not new**: the deleted `presets`
path built the same closure — it simply never called `save()`, so nothing ever
raised. Naming the architecture is what made the gap *visible*, because a class
that looks persistable is expected to persist.

To persist an embedding pipeline today, build the `ERModel` directly with a
serializable single named field instead of the closure:

```python
ERModel.from_components(
    blocker=VectorBlocker(
        vector_index=FAISSIndex(embedder=SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")),
        schema=MySchema,
        text_field="name",   # a named field serializes; a closure does not
    ),
    comparator=StringComparator(feature_specs=[FeatureSpec(name="name", kind="string")]),
    matcher=EmbeddingScoreMatcher(threshold=0.7),
    clusterer=Clusterer(threshold=0.7),
)
```

**Open design decision (not made here):** the fix is a named-extractor seam
mirroring `LLMMatcher(response_parser=...)` — a registry of named extractors so
the *name* serializes instead of the closure. That changes what `VectorBlocker`
accepts on the paid path, so it is an architecture call, deliberately left to
its owner rather than settled at the end of a wave.

**Scope note for the HF-readiness gate:** the weightless-round-trip proof is
green only for architectures **without a closure-bearing blocker**. It
round-trips `FuzzyString`. It does not — and currently cannot — cover
`VectorLLMCascade`.

### `Resolver` is spend-capped (B1) — **behavior change**

`Resolver` — the whole low-level public API — had **no budget guard at all**. The
spend cap lived in `langres.core.presets`, which `Resolver` is architecturally
forbidden to import, so only `langres.link` / `langres.dedupe` were ever capped;
`from_schema`'s own docstring advertised the hole ("runs **UNCAPPED** … nothing
stops a runaway bill"). A paid `Matcher` on a `Resolver` could bill without limit.

#### Added

- **`budget_usd=` on `Resolver(...)` and `Resolver.from_schema(...)`.** Defaults to
  `DEFAULT_BUDGET_USD` ($1.00) — the same default the verbs have always had.
- **`langres.core.spend_cap`** — `SpendCappedMatcher` (the one enforcer),
  `DEFAULT_BUDGET_USD`, `UNCAPPED_BUDGET_USD`, `effective_budget`.
- **`langres.core.spend`** — `SpendMonitor` / `BudgetExceeded`, moved out of
  `langres.clients.openrouter` (a pure USD ledger with nothing OpenRouter-specific
  about it, which `core` must be able to reach). **Both names remain importable
  from `langres.clients.openrouter`**, and `langres.BudgetExceeded` is unchanged.

#### Fixed

- **The cap's `SpendMonitor` is now built once per instance, not per `forward()`
  call.** It was rebuilt on every call, so `r.resolve(a); r.resolve(b)` spent **2x**
  the budget and N resolves spent N x. Harmless for a one-shot verb call; wrong for
  a long-lived object. Two `resolve()` calls on one `Resolver` now share one budget.
- **A cap that has already tripped now costs $0 on the next call** instead of paying
  for one more judgement before refusing.

- **Every seam that scores through the matcher now shares ONE ledger.**
  `AnchorStore._judge` reached through the Resolver's public `.module` attribute and
  scored with the **raw** matcher — no cap, no ledger — so a paid `LLMMatcher` billed
  without limit on every `assign()`, and a long-lived store is exactly the
  many-`assign()` object that turns that into a real bill. The cause was structural:
  the capped scorer was built inline at one call site, so every *other* caller
  silently got the raw matcher. There is now a single internal accessor
  (`Resolver._scorer()`); `resolve` / `predict` / `fit` / `AnchorStore.assign` all
  route through it, and an AST sweep in the test suite bans
  `<anything>.module.forward(...)` in `src/` so the next caller cannot reopen it.

#### Changed / migration

- **A paid `Matcher` scoring through `resolve()` / `predict()` on a `Resolver` that
  previously ran unbounded now raises `BudgetExceeded` past $1.00.** Free matchers
  (`"string"`, `"embedding"`) meter $0 and never trip. Pass `budget_usd=` to raise
  it, or `langres.core.spend_cap.UNCAPPED_BUDGET_USD` (`float("inf")`) to opt out
  deliberately. `budget_usd=None` means "the default", **never** "uncapped" — you
  cannot disable the cap by forgetting to pass it. `BudgetExceeded.partial_judgements`
  still carries everything already paid for.
- **`AnchorStore.assign()` now bills against its Resolver's budget** and raises
  `BudgetExceeded` once that budget is gone (it previously spent unbounded). Build the
  store from a `Resolver` with a `budget_usd=` sized for the store's whole lifetime,
  not for one batch — `AnchorStore.build()`'s own `resolve()` pass draws down the same
  ledger every later `assign()` draws from.
- **`fit(method=Platt()/Isotonic())` is now capped too** — previously disclosed as a
  known gap, now closed. Its scoring pass **is** its entire bill (the sklearn
  calibrator itself is $0), so metering it is complete protection, not the partial
  kind rejected for `distil()` below. `fit(..., pairs=...)` and
  `fit(method=Finetune())` validation passes route through the same accessor; both
  are $0 paths in practice (sklearn / a local fine-tuned LM), so the cap is a no-op
  there — they go through it for uniformity, because "every `.forward()` on the
  matcher is metered" is a rule that can be enforced, and "some are" is what caused
  the `AnchorStore` hole.

#### Known gap (scoped deliberately, not fixed here)

- **`distil()` / `fit(method=MIPRO())` is not bound by the Resolver's `budget_usd`.**
  DSPy's compile calls never reach `self.module.forward`, so the scoring accessor
  cannot see them — a cap there would read as protection while covering nothing. It
  keeps its own separate `method.budget_usd` monitor, which today observes **$0**
  because DSPy-compile spend capture is itself deferred (issue #100). So a paid
  `MIPROv2` compile is **effectively unbounded** until #100 lands. Left visibly open
  rather than half-closed.

**The guarantee, stated honestly:** spend is bounded by `budget_usd` **plus the cost
of at most one further call**. An LLM call's cost is not knowable until it has been
made, so "check before" can only mean "am I already at/over budget? then refuse the
next call". No doc here claims more.

### Autoresearch: the self-tuning loop over blocking (epic #145, M1)

A `propose → run → evaluate → keep-if-better` hill-climber that tunes a pipeline
against a **loss-like** objective instead of a saturated F1. M1 proves the loop on
the **blocking** vertical; the matching vertical (`log_loss` / AUC-PR steering) and
small-LM fine-tuning are deferred, as is an Optuna/LLAMBO proposer swap. A durable
off-laptop dashboard is no longer deferred: `tracker="trackio"` (local-first; an HF
Space/Dataset sync is a `space_id`/`HF_TOKEN` opt-in) plugs straight into
`optimize`/`run_loop` via the existing `tracker=` hook (see below).

#### Added

- **`langres.optimize(space, objective, benchmark, *, seed=None, store=None,
  dedup=True, split="full", embedder=None, tracker=None) -> LoopResult`** — the
  one-call autoresearch facade over blocking search. Loads the benchmark once,
  wraps an index-caching scorer (one vector index per
  `(embedding_model, metric, text_field)` group, reused across every `k`), drives
  the loop over `space.configs()`, keeps the incumbent the `objective` prefers, and
  persists **every** trial (accepted, over-budget reject, and scorer failure) to a
  local `RunStore` JSONL at `store=` (`store=None` writes nothing). **Local-only
  persistence today.**
- **DX: `tracker=` on `optimize`/`run_loop` takes a spec, not a resolved
  instance** — a backend name (`tracker="trackio"`), an already-built
  `ExperimentTracker`, a sequence of either (fan-out), or `None` (default,
  no-op). Both resolve it internally via `tracking.trackers.resolve_tracker`
  (mirrors the `matcher="..."` preset convention), so the boilerplate
  `tracker=resolve_tracker("trackio")` is no longer needed on the happy path;
  `resolve_tracker` stays public for advanced/explicit use.
- **`langres.score_blocking(config, benchmark, *, embedder=None, index=None) ->
  dict[str, float]`** — the concrete one-config blocking scorer `optimize` wraps,
  returning `candidate_recall` / `reduction_ratio` / `candidate_precision` /
  `total_candidates`. Both `optimize` and `score_blocking` are **root exports and
  import-light** (heavy imports are lazy inside the call, so a bare `import langres`
  never pulls faiss; `tests/test_import_budget.py` guards it).
- **`Objective`** (`langres.autoresearch.objective`) — the immutable
  keep-if-better scorer: `Objective.maximize` / `minimize` / `pareto`, feasibility
  `subject_to=[(metric, op, threshold), ...]` constraints, feasibility-first then
  strict-Pareto-dominance `is_better` (ties keep the incumbent). Pure-stdlib,
  metric-source-agnostic.
- **`SearchSpace`** (`langres.autoresearch.search_space`) — a frozen,
  declarative Cartesian grid of blocker configs; `configs()` yields config dicts
  with `k_neighbors` as the **innermost** axis (the index-reuse ordering contract).
  Pure-stdlib.
- **`langres.core.metrics.log_loss(confidences, outcomes)`** — binary
  cross-entropy, the strictly-proper loss-like steering signal for the loop
  (penalizes confident mistakes far more than `brier_score`; per-item penalty
  clamped near `-log(1e-15)` so a confident-and-wrong `p=0/1` is large but finite).
- **`examples/research/blocking_recall_autoresearch.py`** — the E1 proof: the loop
  hill-climbs blocking recall@budget on amazon_google at **$0, offline** (measured
  climb + budget-rejection numbers in `docs/EXPERIMENTS.md`).

### Model identity + one method registry (v0.3 slice, closes #103)

The maintainer complaint this fixes: *"`langres.dedupe` is too simple — you
have no idea what's running behind. What is the default model? And it makes
changing the default model hard."* Design note:
`docs/research/20260713_model_identity_and_hub.md`.

#### ⚠️ Behavior changes

- **`LinkVerdict.model` and `DedupeResult.model` are new required fields**
  carrying the resolved underlying model id: the LLM id (e.g.
  `"openrouter/openai/gpt-4o-mini"`) for the LLM judges, the
  sentence-transformers embedder name for `judge="embedding"`, an injected
  `Module`'s own `model` attribute for `judge_used="custom"` (identity is no
  longer erased at the escape hatch), and `None` for pure-string similarity.
  External code constructing these models directly must now pass the field
  (pre-1.0; same contract as the 0.3.0 `threshold` fields). `JudgementLog`
  rows backfill their existing `model` column from the same resolved value
  whenever a judge doesn't stamp `provenance["model"]` itself, so log and
  result can't drift.
- **The `judge="auto"` default model is now a pinned, documented constant:
  `langres.DEFAULT_AUTO_MODEL`** (aliasing
  `clients.openrouter.DEFAULT_OPENROUTER_MODEL`; the direct-OpenAI fallback
  route when only `OPENAI_API_KEY` is set remains `openai/gpt-5-mini`). The
  values are unchanged — what's new is the policy: **changing the default
  model is a user-facing behavior change and requires a changelog entry.**

#### Added

- **`judge="prompt_llm"` — the bring-your-own-prompt LLM judge, by name**
  (closes #103). `link`/`dedupe`/`Resolver.from_schema` accept
  `prompt_template=`, `system_prompt=`, and `response_parser=` (a *registered*
  parser name: `"score"` or `"binary_yes_no"` — see
  `llm_judge.RESPONSE_PARSERS`/`RECORD_SERIALIZERS`). Named parsers serialize
  in `LLMJudge.config`, closing the round-trip gap where a saved
  paper-replication judge silently reverted to the default parser on
  `Resolver.load`. Passing a prompt-seam kwarg with any other judge raises —
  never silently ignored. `zero_shot_llm` keeps backing `DSPyJudge`
  (no behavior change for existing callers).
- **One method registry** (`langres.core.method_registry`): every
  name-selectable judge/method is a `MethodSpec` (builder + `score_type` +
  `default_threshold` + `default_model` + comparator/extra requirements)
  registered once. The three hand-rolled dispatch switches
  (`presets.build_judge`, `resolver._build_module_for_judge`,
  `methods._make_module_builder`) now all resolve through it — closing the
  #55 wiring debt; `"auto"`'s fail-fast, the spend cap, and per-judge
  thresholds are unchanged. Public seam: `register_method` / `get_method` /
  `list_methods` (exported from `langres.core`). **Id grammar reserved:**
  bare method names are built-ins; `/` is reserved for future
  `author/method` namespacing (model ids keep their slashes in the separate
  `model=` kwarg).

### Fixed

- **`langres.__version__` no longer drifts from the released version.** It was a
  hardcoded string (still `"0.2.0"` in the published 0.3.0 wheel, while pip
  metadata correctly said 0.3.0); it now resolves from the installed package
  metadata (`importlib.metadata`), so pyproject.toml is the single source of
  truth.

## [0.3.0] - 2026-07-13

Everything since 0.2.0: the judgement contract (`decision` / abstain / optional
`confidence`), eval honesty (spend caps by default, honest numbers), the $0
`EvalReport` tearsheet, the Peeters LLM-EM paper-replication seams, the
evaluation instrument (benchmark registry + bring-your-own-data `evaluate()`),
experiment tracking, a paved road for the flywheel loop, a deterministic
keyless contract for `judge="auto"`, and a slimmer PyPI package.

### The flywheel paved road & the auto-judge keyless contract (#112, #113)

#### ⚠️ Behavior changes

- **`DedupeResult.threshold` and `LinkVerdict.threshold` are new required
  fields** carrying the *resolved* decision cut — the same value the
  `JudgementLog` stamps on logged verdicts, so result, log, and
  `select_for_review` can no longer drift apart. `None` only on `dedupe`'s
  fewer-than-2-records short-circuit (no judge is resolved). External code
  constructing these models directly must now pass the field (pre-1.0).

#### Fixed

- **`judge="auto"`'s keyless fail-fast contract was unforceable in-repo.**
  Popping `OPENROUTER_API_KEY`/`OPENAI_API_KEY` from the environment did NOT
  produce a keyless run: `Settings` reads the `.env` in the CWD directly, so
  auto-discovery still found a key and made a real paid call where the
  documented `NoJudgeAvailableError` was expected. Two deterministic switches
  now exist — **`LANGRES_OFFLINE=1`** (`Settings.langres_offline`) makes
  `judge="auto"` treat every key as absent (scoped to auto-discovery; an
  explicit `judge=` in code bypasses it), and an env var set to the **empty
  string** wins over `.env` and counts as absent (now documented +
  regression-locked). The full discovery order (kwargs > process env > CWD
  `.env`, no walk-up; decided before litellm's own walk-up `load_dotenv` can
  run) is documented on `choose_auto_judge` and `Settings`.

#### Added

- **The flywheel loop's back half is now root-importable.** `Correction`,
  `CorrectionLog`, `harvest_labeled_pairs`, `derive_threshold_from_pairs`
  export eagerly; `EvalReport`, `gold_pairs_from_clusters`, and
  `derive_threshold` resolve lazily (PEP 562) so a bare `import langres` stays
  light — the whole loop now reads from a single `from langres import (...)`.
- **`examples/flywheel_min.py`** — the full zero-label loop at $0, offline:
  `dedupe` + `JudgementLog` → `select_for_review` → the real
  `langres export-csv` / `import-csv` round-trip → `harvest_labeled_pairs` →
  `derive_threshold_from_pairs` → re-run → `EvalReport` tearsheet. The toy
  dataset is crafted so tuning visibly changes the outcome (precision
  0.600 → 1.000, F1 0.750 → 1.000); it runs in the core-install CI job on
  every PR, proving the bare-`uv sync` claim.

### Packaging

- **The large third-party benchmark corpora are no longer bundled in the PyPI
  package** — the wheel drops 5.7 MB → 0.57 MB compressed (15.6 MB → 1.7 MB
  uncompressed). The DeepMatcher/Magellan corpora (Abt-Buy,
  Amazon-Google, DBLP-ACM, DBLP-Scholar, Walmart-Amazon, WDC Computers) ship in
  the **git repository only**; loading one from a pip/uv install now raises
  `BenchmarkDataNotFoundError` naming the fix (use a git checkout:
  `git clone https://github.com/fxd24/langres && pip install -e ./langres`).
  Still bundled: the synthetic `tiny_fixture` (the registry/CI fixture), the
  synthetic BSD-3 FEBRL4 person subset, Fodors-Zagat, and the Peeters
  `peeters_sampled_test.csv` pair sets (id/label triples only, no third-party
  record text).

### The judgement contract: decisions, abstentions, optional confidence

`PairwiseJudgement` now separates *deciding* from *ranking*, makes an abstention a
first-class "I don't know" (never a fabricated verdict), and carries an optional,
earned `confidence`. Builds directly on the eval-honesty groundwork below.

#### ⚠️ Behavior changes

- **`PairwiseJudgement.score` widened `float` → `float | None`**, and the model
  gained `decision: bool | None`, `confidence: float | None`, and
  `confidence_source: Literal["none","unrequested","logprob","calibrated","heuristic"]`.
  A judge is now a *ranker* (emits `score`) **or** a *decider* (emits a boolean
  `decision` — a binary Yes/No LLM has no meaningful score, so a fabricated
  `0.0`/`1.0` would lie); a logprob judge may set both. `score_type` stays
  **required**: it doubles as the judge-family tag even when `score` is `None`.
  `LinkVerdict.score` widened the same way.
- **Ask `predicted_match(judgement, threshold) -> bool | None`** — a new module
  function in `langres.core.models` (exported from `langres.core`), never a raw
  `score >= threshold`. `decision` wins over `score`; neither set → abstention →
  `None`. `classify_pairs`, the base `Clusterer`, and `CorrelationClusterer` all
  route through it, so an **abstention is excluded from the predicted set** — no
  longer graded a confident "no". `PairwiseJudgement.is_abstain` is the property
  for that neither-set case.
- **An abstention now emits `decision=None, score=None`** (was `score=0.0`) with
  `provenance["parse_error"] = True`. `LLMJudge` (default
  `on_parse_error="abstain"`, unparseable response) and `DSPyJudge` (parse /
  validation error) now abstain **identically**. `link()` raises the new
  **`JudgeAbstainedError`** (root-exported, subclasses `RuntimeError`) instead of
  a `match=None` verdict a caller's `if verdict.match:` would silently read as
  "no".
- **`DSPyJudge` no longer abstains to the opposite verdict from `LLMJudge`.** It
  previously emitted `score=0.5` — predicted a **MATCH** at any threshold ≤ 0.5,
  and invisible to the abstention count. It now abstains as the null verdict
  above, excluded from the predicted set. (The interim `score=0.0` fix from the
  eval-honesty groundwork below is superseded by this null-verdict shape.)

#### Fixed

- **The review flywheel ran as a silent no-op on a binary judge.**
  `select_for_review(strategy="uncertainty")` ranked by distance-to-threshold on
  a `score`; a decision-only log has no score to rank, so it returned `[]` — an
  empty queue that looked like "nothing to review". It now ranks by the logged
  **`confidence`** when present (`|confidence − 0.5|`), and **raises** `ValueError`
  (naming `strategy="disagreement"` or `LLMJudge(confidence="logprob")` as the
  fix) when there is no rankable signal, instead of silently returning nothing.
  `ReviewItem` gained `reasoning` / `confidence` / `confidence_source`.
- **`JudgementLog` persisted `$0` for every cascade row.** `append` / `read` only
  read `provenance["cost_usd"]`, but `CascadeModule` writes `llm_cost_usd`; the
  logged cost is now the first of `("cost_usd", "llm_cost_usd")` present.
- **`harvest_labeled_pairs` coerced an abstention into a `False` silver label.**
  A v3 abstention row (`verdict=None`) has no verdict to harvest; it is now
  **skipped** (unless a human correction supplies a label) rather than seeding
  training data with a non-match the judge never gave — the label-side twin of
  never coercing a null score to `0.0`.
- **`select_for_review("uncertainty")` dropped score-only rows in a mixed log.**
  Once any row carried a `confidence`, the selector returned only the
  confidence-bearing rows, silently discarding uncertain score-only rows (a
  `CascadeJudge` log mixes both). It now folds the score band back in, so no
  uncertain pair vanishes.

#### Added

- **`EvalReport` — a $0 evaluation tearsheet** (`langres.core.eval_report`, also
  re-exported from `langres.eval`). `EvalReport.from_log(rows, gold_pairs)` (from
  persisted `JudgementLog.read()` output) or `EvalReport.from_judgements(...)`
  (in-process) computes pair precision/recall/F1, the PR and ROC curves +
  ROC-AUC/AP, a gold-vs-non-gold score histogram, confidence calibration
  (reliability diagram + Brier + ECE), and the most-confident errors — all at
  **zero** API cost from already-logged judgements. `to_html()` renders a single
  self-contained document with inline SVG (`langres.core._svg`): no matplotlib, no
  external assets, theme-aware. A leaf module — nothing in `reports.py`/`module.py`
  imports it, and an import-budget test locks that it never pulls a heavy
  dependency. See `examples/quickstart_eval.py` (fully offline, in CI's
  core-only job). This is the supported, dependency-free replacement for the
  dead `plot_*`/`langres[viz]` matplotlib path.
- **`langres.testing.ScriptedJudge` gained an optional `confidence` provider**
  (dict or callable) + `confidence_source`, so the test double can model a
  logprob judge offline (e.g. to populate an `EvalReport` calibration panel with
  no API calls).

- **`JudgementLog` schema v3.** Rows now carry `decision` / `confidence` /
  `confidence_source` natively; `read()` backfills `decision` from the logged
  `verdict` for older v1/v2 rows (`bool(verdict)` for a real bool, else an honest
  `None` — never a coerced `False`). The logged `verdict` is the caller's
  `predicted_match`.
- **`LLMJudge(confidence="logprob")` now promotes its credence onto the
  judgement** (the eval-honesty groundwork left it in `provenance` only). With a
  usable first-token yes/no mass it sets `score = p_yes` (an honest continuous
  ranking signal), `confidence = max(p_yes, 1 − p_yes)`, and
  `confidence_source = "logprob"`, and it is now serialized in `config` so a saved
  logprob judge reloads as one. `confidence="none"` (default) tags a decision
  judge `confidence_source="unrequested"` (it *could* expose logprobs; you did not
  ask).

#### Why `confidence` is a permanent field

The field earned its place on evidence, not anticipation. On all 1206 Abt-Buy
pairs (`gpt-4o-mini`, `temperature=0`, provider-billed), the model's first-token
credence in its **own** answer scored
`roc_auc(answer_was_correct, credence) = 0.95` (Brier 0.024) — it predicts its own
errors, exactly what the flywheel needs to route the uncertain margin to review.
Had that come back ≈ 0.5, the contract would have shipped `decision` + abstain
**without** `confidence`. See `docs/research/20260710_logprob_credence_probe.md`.

**Scope — honest limits.** `confidence` is **logprob-only** and an
**OpenAI-family feature** today (`gpt-4o-mini`, one dataset, one prompt design);
it is **unverified for GLM / DeepSeek / Qwen** — the models our own paid runs use
— which is exactly why `confidence_source` separates `"none"` (structurally can't)
from `"unrequested"` (could, wasn't asked). It is free **only in output tokens, at
`explain=False`, on a logprob-returning model** — *not* free in general (generated
reasoning costs ~3.75× on the same data). Never write "confidence is free"
unqualified.

### Eval honesty: spend cap by default, argmax warning, ROC-AUC, public seams

Groundwork for the judgement-contract change (`decision` / abstain / optional
`confidence`). Nothing here touches `PairwiseJudgement`'s schema; this lands the
pieces that can regress money or silently report a wrong number.

#### ⚠️ Behavior changes

- **`evaluate()` now caps spend by default.** It builds a `BudgetedModuleRunner`
  internally; `budget_usd=` overrides it and omitting it resolves to
  `DEFAULT_BUDGET_USD` (`$1.00`). Previously `evaluate()` had *no* cap at all —
  a paid judge over a large candidate set billed until it finished. Free judges
  never reach the cap. `evaluate_judge_on_candidates()` keeps its lower-level
  `runner=` / `price_per_token_or_pair=` / `cost_track_fn=` knobs unchanged.
  **The cap is enforced *between* calls**, so a single in-flight call can push
  total spend past it by that call's own cost. When that happens the run stops
  before starting another call and reports `JudgePairEval.budget_exceeded`,
  and `evaluate()` warns naming the measured spend and the cap. It does not
  raise: the run completed, its metrics are valid, and the money is already
  spent — raising would only discard work the user paid for.
- **`evaluate()` raises `ValueError` on an empty candidate list.** It used to
  report `precision = recall = f1 = 0.0`, which is indistinguishable from a
  judge that ran fine and matched nothing.
- **`evaluate()` warns that `best_threshold` is fitted to the gold it reports
  on.** The default still sweeps `DEFAULT_PAIR_GRID` and the returned number is
  unchanged — but it is an argmax over the same gold used to score, i.e.
  optimistically biased, not a held-out estimate. It now says so once, via
  `UserWarning`. Pass `threshold=<float>` to grade honestly at a fixed cut:
  `graded_threshold` is set and `best_threshold` becomes `None`.
  The default was deliberately *not* flipped to a fixed `0.5` — a global cut
  collapses an embedding judge from F1 1.000 to 0.667, because cosine
  non-matches sit at 0.70–0.80. No single constant serves `sim_cos`,
  `heuristic`, and a binary LLM judge alike.
- **`evaluate(on_truncation=...)`** (`"raise"` default) raises
  `EvaluationTruncatedError` **only when the spend cap caused the truncation**,
  carrying the partial judgements on the exception. A judge that skips a pair
  only warns: one bad call must not blow up a run and discard results already
  paid for. `JudgePairEval.truncation_reason` records which happened.
- **`CostTrack.cost_is_real` is now a derived property, not a stored bool.**
  A single run can mix provider-billed cost, litellm-estimated cost, free local
  judges, and untracked DSPy parse failures — a bool cannot say "mixed". The
  stored field is `cost_basis: Literal["real","estimated","mixed","untracked","none"]`,
  and `CostTrack.usage` now carries the summed `LLMUsage` token vector.
  *Tokens are the fact; dollars are derived.*

#### Fixed

- **The spend cap could not detect being breached.** `evaluate()` checked the
  budget only *before* the next call, against a placeholder worst-case price, and
  never compared the real post-call cost against the cap. A single `$10.00` call
  under a `budget_usd=1.00` cap returned `truncated=False`,
  `truncation_reason="none"` and no warning at all; a breach on the final pair
  left the run looking complete and clean. The runner now compares measured spend
  against the cap after every call. (Found by adversarial review, not by tests —
  the branch was fully green when this shipped.)
- **`cost_basis` disagreed with `usd_total` about whether money was spent.**
  `_judgement_cost()` sums both `provenance["cost_usd"]` and
  `provenance["llm_cost_usd"]` (the key `CascadeModule` writes), but the basis
  classifier only recognized the first — so a real cascade run reported
  `usd_total > 0` alongside `cost_basis="none"`, `cost_is_real=False`. Both now
  read one shared key set.
- **`make_token_cost_track` (`langres.clients.openrouter`) never set `cost_basis`
  or `usage`.** The second `CostTrack` producer priced judgements from a token
  table and returned a real dollar figure labelled `cost_basis="none"` with an
  all-zero token vector. It now reports `"estimated"` (a price table is not a
  provider-billed amount, so never `"real"`) and sums the token vectors.
- **`roc_auc_score` / `average_precision_score` accepted non-finite scores.**
  A `NaN` score returned `0.75` or `0.5` for the same multiset depending on input
  order, because `NaN` breaks both `sorted()` and the equality-based tie grouping.
  A ranking containing `NaN` is undefined; it now raises `ValueError` naming the
  offending index.
- **`DSPyJudge` abstained to the opposite verdict from `LLMJudge`.** On a parse
  or validation error it emitted `score=0.5` with **no** `provenance["parse_error"]`
  key. At any threshold ≤ 0.5 that abstention was predicted a **match** — while
  `LLMJudge`'s abstention (`score=0.0`) was predicted a non-match — and
  `n_parse_errors` could not see it, so DSPy abstentions were invisible in every
  eval report. Both judges now abstain at `score=0.0` with `parse_error=True`.

- **`evaluate()` accepted a degenerate match cut.** `classify_pairs` predicts a
  match iff `score >= cut`, and both `LLMJudge` and `DSPyJudge` abstain at
  `score=0.0` — so `evaluate(threshold=0.0)` graded **every abstention as a
  confident YES**. A cut above `1.0` is unreachable for a `[0, 1]` score, making
  F1 a structural `0.0` rather than a measurement. A fixed `threshold` must now
  lie in `(0.0, 1.0]`.
  A **swept `grid`** is held to the looser `[0.0, 1.0]`: `0.0` is a PR curve's
  legitimate predict-all anchor (recall `1.0`, precision = prevalence), and
  banning it would outlaw an honest ranking-judge sweep to defend against an
  abstaining judge's convention. Instead, `evaluate()` **warns when the argmax
  lands on `0.0`** — that judge does not beat predicting every pair a match, and
  `best_threshold=0.0` must never reach production.
- **The same invariant now holds on `evaluate_judge_on_candidates()`**, the
  lower-level public path documented for paid and compiled judges. It validates
  (and materialises) `grid` **before** the judge runs, so a bad grid never costs
  an API call. `run_method()` holds a dataset-supplied `threshold_grid` to the
  same rule. An empty grid is its own `ValueError` instead of an opaque
  `max() iterable argument is empty` from inside the sweep.
- **`langres.eval.candidates_for()` silently graded the wrong split.** Any
  `split` value other than exactly `"test"` fell through to the **train** split,
  so a typo (`"valid"`, `"Test"`) produced a report that looked valid while
  scoring the wrong partition. `Literal` only protects type-checked callers; a
  CLI flag or a dict lookup reaches it untyped. Unknown splits now raise.
- **`judge="auto"` told users their spend was "hard-capped".** Both user-facing
  messages in `core/presets.py` (the `NoJudgeAvailableError` guidance and the
  paid-judge notice) promised a hard cap the `BudgetedModuleRunner` does not
  provide: it stops *between* calls, so one in-flight call can overrun the cap
  by its own cost. Same overstatement corrected in `benchmark.py`; the verbs
  path now says what it actually does.

#### Added

- **`langres.core.metrics.roc_auc_score` / `average_precision_score`** — pure
  Python: `math` only, adding no numpy or sklearn dependency (`metrics.py` stays
  import-light; sklearn remains confined to the `[trained]` extra). Tie-aware:
  ROC-AUC uses the Mann-Whitney-U form over midranks, so an all-equal score
  vector yields exactly `0.5` and a tie straddling the pos/neg boundary gets
  half credit — the exact point a naive rank-AUC silently diverges from sklearn.
  Single-class input **returns** `nan` rather than raising, so one degenerate
  slice blanks a cell instead of killing a whole report. A non-finite *score*
  (`NaN`/`±inf`), by contrast, **raises** — a ranking containing `NaN` is
  undefined, and returning a confident, order-dependent number for it is worse
  than failing.
- **`Resolver.candidates(records) -> list[ERCandidate]`** — the public seam
  replacing reaches into `Resolver._candidates`. It returns a **materialised
  list**, because `evaluate_judge_on_candidates` calls `len()` and iterates
  twice; handing it a generator would make the second pass yield nothing and
  produce a plausible-but-wrong F1 off an empty gold set. Comparison vectors
  are attached (a raw `blocker.stream()` does not attach them).
- **`langres.eval.candidates_for(bench, *, split, seed)`** — returns
  `(candidates, gold_pairs)` together, so scoring a benchmark never requires a
  private API. Facade also now exports `roc_auc_score`, `average_precision_score`,
  and `gold_pairs_from_clusters`.
- **`JudgePairEval.n_abstained` / `.abstention_rate` / `.graded_threshold`** —
  `graded_threshold` is always populated and always states which cut `pair` was
  graded at.
- **`langres.testing.ScriptedJudge`** — a public `Module` test double. It lets
  tests and examples exercise judge-shaped code (`CascadeJudge`, `evaluate()`,
  the review/harvest flywheel) with no network, no API key, and no spend —
  which matters because a real `LLMJudge` picks up `OPENROUTER_API_KEY` from the
  repo `.env` via litellm's import-time `load_dotenv()` and makes a real, billed
  call. It replaces the hand-rolled `ScriptedJudge` in
  `tests/core/modules/test_cascade_judge.py`. The four `DummyModule` copies in
  `tests/core/test_module.py` stay put on purpose: those tests exercise the
  `Module` ABC itself, and testing the ABC through a library-provided subclass
  of it would be circular. Deliberately **not** `@register`-ed (a test double
  must never enter `Resolver.load` dispatch) and **not** imported by
  `langres/__init__.py`; an import-budget test asserts `import langres` leaves
  `langres.testing` out of `sys.modules`.
- **`LLMJudge(confidence="logprob")`** — an opt-in first-token credence probe.
  It requests `logprobs` + `top_logprobs=20` (merged at **both** the sync and
  async completion call sites as standard top-level chat params — deliberately
  **not** inside `_completion_kwargs`, which early-returns `{}` off `openrouter/`
  and would silently drop logprobs on plain OpenAI) and records, **in provenance
  only**, a `p_yes` renormalised over the yes/no two-way subspace, a
  `confidence_leaked_mass` that is never normalised away, and a `p_yes_is_bound`
  flag when one side's mass is entirely below the top-k cutoff. Below a tiny
  combined-mass floor `p_yes` is `None` (credence is refused, not manufactured
  from noise). `confidence="none"` (the default) is a byte-identical no-op.
  **Nothing is added to `PairwiseJudgement`** — the probe gathers evidence
  *before* any permanent judgement-schema change. Not serialized in `config`.
- **`examples/research/peeters_llm_em_replication.py --logprobs`** — runs the
  Peeters live judge with the credence probe on via the single `_build_live_judge`
  site (byte-identical to the replication judge apart from the logprob request).
  Probe rows are **v2** (`_RESULT_SCHEMA_VERSION` 1→2: adds `correct` always, plus
  `p_yes`/`leaked_mass`/`p_yes_is_bound`) and land in a distinct
  `…__logprobs.jsonl` — a contamination firewall that cannot overwrite the
  committed replication rows — with `--results-dir` defaulting to the committed
  `examples/research/results/peeters`. `--report-only` still reads the old **v1**
  rows unchanged (the `$0` `--compare-archived` replay still reproduces F1 92.09 /
  90.71 at 99.25% per-pair archive agreement).

#### Docs

- Deleted a **false** README claim that `import langres` is heavy and "eagerly
  pulls in `torch`/`litellm`". Measured: **207 ms, zero heavy modules** in
  `sys.modules`; `tests/test_import_budget.py` enforces it.
- `docs/TECHNICAL_OVERVIEW.md` documented `langres.tasks`, `langres.flows`,
  `langres.ui`, `core.Optimizer`, `core.Evaluator`, `blockers.EmbedBlocker`,
  `EmbedSim`, and `data.SyntheticGenerator` — **none of which exist**. It also
  claimed metrics come from `sklearn.metrics` (`metrics.py` imports only `math`)
  and that `pytrec_eval` is used (it appears nowhere; ranx backs the ranking
  metrics, lazily, behind the `[eval]` extra). All rewritten against the real
  verbs → `Resolver` → `core` layering, and §8's claim that the trained judges
  had not shipped corrected — both `FellegiSunterJudge` and `RandomForestJudge`
  exist and implement the W1.0 fit hooks.
- Flagged that `reports.py`'s `plot_*` methods tell users to
  `pip install 'langres[viz]'` — **an extra that does not exist**. matplotlib is
  undeclared and arrives only transitively via `mlflow` or `seaborn ← ranx`.
  Left in place; declaring or deleting it is a separate decision.

### Paper replication: usage vector, LLM-judge seams, Peeters LLM-EM

#### ⚠️ Behavior changes

- **`LLMJudge` no longer silently returns `0.5` when it cannot parse a score.**
  The default `response_parser` now *abstains* on an unparseable response: the
  judgement carries `provenance["parse_error"] = True` with `score=0.0` (a
  flagged abstention, distinguishable downstream) instead of a plausible-looking
  mid-confidence `0.5`. `on_parse_error="raise"` turns the same case into an
  immediate `LLMParseError`. The default is `"abstain"` because aborting a long
  paid run on one flaky response is worse than a surfaced, counted abstention —
  and `evaluate()` / `evaluate_judge_on_candidates()` now expose the count as
  `JudgePairEval.n_parse_errors` and warn loudly when it is non-zero.
- **`LLMJudge` default `temperature` changed `1.0` → `0.0`** (deterministic;
  the ER-paper convention, and already the `DSPyJudge` default). Pass
  `temperature=1.0` to restore the old behavior.
- **`LLMJudge.prompt_template` now requires literal `{left}` and `{right}`
  placeholders** (validated at construction) and substitutes them by literal
  replacement rather than `str.format`, so a template containing other braces
  (e.g. a paper's JSON output schema `{"match": true}`) works instead of raising
  `KeyError`.
- **`JudgementLog` schema `"v"` bumped `1` → `2`:** the default (privacy-safe,
  `features=False`) row gained a non-PII `usage` token vector (`null` for
  non-LLM judges). Old `v: 1` rows still read back unchanged.

#### Added

- **`langres.core.usage.LLMUsage`** — a frozen Pydantic token-usage vector in the
  OpenTelemetry GenAI vocabulary (snake_case, SUBSET semantics): `input_tokens`
  and `output_tokens` (inclusive totals) with `cache_read_input_tokens`,
  `cache_creation_input_tokens`, `reasoning_tokens` as subsets, plus the serving
  `provider` and `model`. Import-light (pydantic only) so a future pricing layer
  can consume it without core's heavy deps. `LLMJudge` / `DSPyJudge` / `SelectJudge`
  now record it under `provenance["usage"]` (additive — legacy
  `prompt_tokens`/`completion_tokens` unchanged). Pinned against LiteLLM's
  Anthropic normalization (`usage.prompt_tokens` is already the inclusive input
  total, so the cache subsets are never double-counted).
- **`LLMJudge` paper-replication seams** (first-class constructor params, no
  subclass fork): `response_parser` (default `parse_score_response`; shipped
  reusable `parse_binary_yes_no` for the Yes/No prompt family), `record_serializer`
  (default `default_record_serializer`), `system_prompt`, and `on_parse_error`.
  All exported from `langres.core.modules.llm_judge` (`ParsedVerdict`,
  `LLMParseError`, the two parsers, `default_record_serializer`).

#### Added — Peeters et al. (EDBT 2025) LLM-EM replication (offline, $0)

- **`langres.data.peeters`** — a replication seam for *Entity Matching using
  Large Language Models* (Peeters, Steiner & Bizer, arXiv 2310.11244 v4). A small
  manifest + loader-factory (`list_peeters_replications` / `get_peeters_replication`,
  mirroring `data.registry`) over the pieces needed to reproduce their published
  F1 by **replaying their archived model answers** — no API key, no LLM call, $0:
  - `serialize_record` (their per-field whitespace-token truncation recipe),
    `render_prompt` (the `domain-complex-force` template), `parse_binary_answer`
    (their exact strip/de-punctuate/lowercase/`"yes" in text` parser).
  - `regenerate_sample_rows` — deterministically regenerates their sampled
    evaluation subset from our **already-vendored** DeepMatcher `test.csv`
    (numpy-only reproduction of `pandas.sample(random_state=42)`), plus
    `load_peeters_sample` / `load_peeters_records` / `render_sample_prompts` and
    the `judgements_from_answers` bridge to `core.metrics.classify_pairs`.
  - Registered slices: `abt-buy` (1206 pairs) and `amazon-google` (1234). Both are
    **slices** of the existing `abt_buy` / `amazon_google` benchmarks (a subset of
    the `test` split), so they stay out of `data.registry` (the clustering-benchmark
    manifest); their binary pair-classification protocol has no blocking/clustering/
    threshold sweep.
- **Committed pair-set artifacts** `datasets/{abt_buy,amazon_google}/peeters_sampled_test.csv`
  — regenerated from our own CSVs and verified **exactly equal** to the authors'
  published `sampled_gs` (1206/1206, 1234/1234, 0 label mismatches). No MatchGPT
  data is vendored (it ships no LICENSE; langres is Apache-2.0).
- **`examples/research/peeters_llm_em_replication.py`** — the offline replay
  harness. Reproduces arXiv v4 Table 2 `abt-buy` / `gpt-4-0613` /
  `domain-complex-force` → **F1 95.15** (prompt round-trip 100.00% byte-exact).
  amazon-google round-trips 99.51% — the 6 residual diffs are float-repr artifacts
  in *their* gold standard's `price` column (e.g. `6.5600000000000005` vs our
  vendored `6.56`), not a serializer bug.

#### Added — Peeters LLM-EM live (paid) path

- **Live-run seams in `langres.data.peeters`** so an `LLMJudge` can run the exact
  Peeters prompt over a slice: `build_llm_prompt_template(spec)` (the
  `domain-complex-force` template with `{left}`/`{right}`),
  `make_record_serializer(spec)` (the per-dataset serializer), `build_candidates(spec)`
  (the sampled pairs as `ERCandidate`s), and the `PeetersRecord` entity. A test
  pins that the live rendering (`template.replace(...)` + serializer) reproduces
  `render_sample_prompts`' archived-validated prompt **byte-for-byte** — so the
  paid run pays for precisely the prompt the $0 replay validated at F1 95.15.
- **`peeters_llm_em_replication.py` gains `--mode dry-run` and `--mode live`.**
  `dry-run` ($0, no key) renders all 1206 pairs through the live path and reports
  token counts (100,256 input over abt-buy, matching a direct o200k_base count)
  + a cost estimate. `live` (**paid, off by default**) runs `LLMJudge`
  (`domain-complex-force` template, Peeters serializer, `parse_binary_yes_no`,
  `temperature=0.0`) over the pairs under a hard `SpendMonitor` cap (default
  **$1.00**), guarded by an explicit `--yes-spend-money` flag + a priced-model
  assertion, and reports F1 + the aggregated `LLMUsage` vector + the real
  OpenRouter-billed cost (`cost_is_real`) vs the paper's published F1. Races two
  dated snapshots: `gpt-4o-mini-2024-07-18` (paper F1 90.95, est ~$0.017) and
  `gpt-4o-2024-08-06` (paper F1 90.47, est ~$0.27); measured total ≈ $0.29.
- **`PRICES_PER_1M` gains the two dated snapshots** the paid run pins:
  `openrouter/openai/gpt-4o-mini-2024-07-18` ($0.15/$0.60) and
  `openrouter/openai/gpt-4o-2024-08-06` ($2.50/$10.00) — OpenRouter list prices
  (checked 2026-07-09); the script refuses to start if a model is unpriced.
- **The live judge pins the OpenRouter → OpenAI provider route.** Our sole
  deviation from the paper's setup is the OpenRouter hop; the live `LLMJudge` now
  sets `provider={"order": ["OpenAI"], "allow_fallbacks": False}` (`LIVE_PROVIDER`,
  sent as `extra_body["provider"]`) so OpenRouter must serve the request from
  OpenAI's own backend and cannot silently substitute a different
  provider/quantization of the snapshot.
- **`--limit N` + `--seed` run a stratified subset.** `--limit N` keeps
  `round(N · pos_ratio)` positives and the rest negatives — preserving the ~17.1%
  Abt-Buy positive ratio, deterministic under `--seed` (default 0) — instead of
  all 1206 (the pair set is a positive block then a negative block, so a naive
  first-`N` would be all matches). A 150-pair gpt-4o-mini live trial costs
  **~$0.002**. Applies to `dry-run`/`live`/`replay`.
- **`--compare-archived` (`--mode live`) checks per-pair agreement against the
  authors' archived answers.** For the exact model we run, it loads the authors'
  archived per-pair answer (reusing the replay harness's cached download) and
  reports the per-pair **agreement rate**, a **2×2 confusion** of ours-vs-theirs,
  up to **10 concrete disagreeing pairs** (record text, gold label, their raw
  answer, our raw answer), and **our** F1/P/R on the judged subset next to
  **their** F1/P/R recomputed on that *same* subset (plus the published full-set
  number) — both verdicts parsed through the one canonical `parse_binary_yes_no`.
  It asserts the archived row count equals the pair-set count and that each
  rendered prompt matches the archived one, **failing loudly** on a mismatch (the
  alignment being off would make every comparison meaningless).

#### Added — Peeters LLM-EM paid run: crash-safe & resumable (no billed call is ever lost)

- **The paid run now durably persists every judged pair, so a kill loses nothing.**
  A first live run was killed partway and lost ~$0.187 of already-billed calls
  because results were only written at the very end. `peeters_llm_em_replication.py`
  now streams one JSON line per judged pair — `flush` + `os.fsync` **before** the
  next paid call — into a per-`(model, dataset, prompt-design)` JSONL under
  `--results-dir` (default the gitignored `tmp/peeters/`), mirroring the
  `m3_race.py` durability pattern at per-pair granularity (new `PeetersResultStore`
  + `results_path_for`). Each row carries `left_id`/`right_id`, `gold`, our raw
  `response_text` + parsed `verdict`, the `LLMUsage` vector, and
  `cost_usd`/`cost_is_real`/`provider`/`model`. (Justified NOT reusing
  `core.judgement_log.JudgementLog`: it has no `gold` column and buries
  `cost_is_real`/`provider` behind `features=True`; a tiny report-shaped sink with
  `fsync` and truncation-tolerant reads is simpler and keeps the operator tool
  decoupled from that core class.)
- **Resume: re-running skips already-judged pairs.** A completed model re-runs at
  **$0 with zero API calls**; a partial run picks up exactly where it stopped. The
  hard spend cap is seeded with spend already recorded (`PeetersResultStore.spent()`
  seeds the `SpendMonitor`), so the aggregate cap **holds across resumes** — a
  resumed run cannot exceed it, and one already at the cap makes no calls. A
  truncated JSONL (a kill mid-write) is recovered from: the partial trailing line is
  skipped and its pair re-judged, and `append` repairs a missing final newline so no
  intact row is ever lost.
- **The final report is computed from the JSONL**, so the numbers are identical
  whether the run finished in one pass or several. New **`--report-only`** mode
  (`report_live_from_store` / `report_compare_from_store`) reprints the full report
  — including the `--compare-archived` agreement/confusion/disagreement table and F1
  — from existing results with **zero API calls**. Progress prints every
  `--progress-every` pairs (running spend + running archive-agreement); stdout is
  line-buffered (also pass `python -u`) so a kill can't swallow it.

#### Fixed

- **Corrected the published Abt-Buy F1 for `gpt-4o-2024-08-06` from a wrong
  `89.33` to `90.47`** (P 83.27 / R 99.03) — arXiv 2310.11244 v4 Table 2 and the
  authors' `results.xlsx` agree. Fixed in the harness (`PAID_MODELS` + docstring),
  `PRICES_PER_1M`'s comment, and `docs/BENCHMARKS.md`. (`gpt-4o-mini-2024-07-18`
  stays **90.95**, P 89.25 / R 92.72.)
- **`LLMJudge` no longer corrupts a prompt when a record contains `{left}`/`{right}`.**
  `_render_prompt` chained two `str.replace` calls, so the second rescanned the
  already-inserted left record: a record whose text held the literal `{right}` had
  that token overwritten with the right record. Now a single `re.sub` pass
  substitutes template placeholders only, never data — a silent, data-dependent
  regression versus the old `str.format` behaviour.
- **Peeters results are partitioned by pair subset.** `results_path_for` now takes
  `limit`/`seed`, because those select a *different pair set*. A `--limit 150` trial
  and the full 1206-pair run previously shared one JSONL, while resume and
  `--report-only` consume every row in it — so a trial's rows would leak into the
  full report (wrong `n_judged`/cost/F1) and its prior spend would eat the budget
  cap. A full run (`limit=None`) keeps the plain three-field name.

#### Results — the replication reproduces the paper

Abt-Buy, `domain-complex-force`, all 1206 pairs, `temperature=0`, OpenAI provider
pinned. Rows committed under `examples/research/results/peeters/`; replay the table
with `--report-only` at **$0**.

| model | ours F1 | published F1 | per-pair agreement | real cost | $/1k pairs |
|---|---|---|---|---|---|
| `gpt-4o-mini-2024-07-18` | 92.09 | 90.95 | 99.25% | $0.0158 | $0.0131 |
| `gpt-4o-2024-08-06` | 90.71 | 90.47 | 99.25% | $0.2627 | $0.2178 |

Scoring the authors' **archived** per-pair answers through `langres.core.metrics`
reproduces their published F1 **exactly** — the scoring path is validated
independently of the model. Our small F1 excess is **serving nondeterminism**, not a
better method (same prompt, same pairs, `temperature=0`, but routed via OpenRouter).
Recorded per-call `cost_usd` tracked OpenRouter's billed delta to within **1.2%**.

- **Unified the two yes/no answer parsers into one canonical implementation.**
  `llm_judge.parse_binary_yes_no` and `data.peeters.parse_binary_answer` had
  shipped independent implementations of the same contract that **diverged on
  intra-word punctuation**: the judge parser did `re.sub(r"[^\w\s]", " ", …)`
  (replace punctuation with a space, and keep `_`), while the paper adapter did
  `str.translate(…, string.punctuation)` (delete punctuation, incl. `_`). They
  disagreed on e.g. `"ye-s"`, `"y-e-s"`, `"Ye's"`, `"ye_s"`, `"Y.E.S."` (MATCH
  for the paper, NON-MATCH for the judge). `parse_binary_yes_no` is now the
  single source of truth and mirrors the reference `check_for_prediction`
  exactly (strip → **delete** `string.punctuation` → lowercase → `"yes" in
  text`); `parse_binary_answer` is a thin `int` adapter over it. This matters
  because the `$0` offline replay validates `parse_binary_answer`, but the paid
  run goes through `LLMJudge(response_parser=parse_binary_yes_no)` — unification
  makes the replay validate the exact path the paid run pays for.

### The evaluation instrument: benchmark registry, sliced eval, `evaluate()` (#98)

- **`langres.eval`** — one curated, import-light facade: `evaluate` (bring-your-
  own-data pair scoring), `candidates_for` (block a registered benchmark's split
  into the `(candidates, gold_pairs)` that `evaluate` needs), benchmark
  discovery (`list_benchmarks` / `get_benchmark`), and the ER metrics —
  re-exported lazily from where they already live.
- **Benchmark registry + portfolio** (`langres.data.registry`) — name→manifest
  discovery over the vendored portfolio: Fodors-Zagat, Amazon-Google, Abt-Buy,
  FEBRL4 person, `tiny_fixture`, plus four DeepMatcher loaders (DBLP-ACM,
  DBLP-Scholar, Walmart-Amazon, WDC Computers) built on one
  `make_deepmatcher_benchmark` factory; OpenSanctions is registered
  external-only (CC-BY-NC, never vendored).
  `examples/research/portfolio_race.py` races the free methods across the
  portfolio; `docs/BENCHMARKS.md` documents each dataset and its caveats.
- **Honest blocking/pair metrics** — `reduction_ratio` (RR) and
  `generalized_merge_distance` (GMD) join BCubed/pairwise in `core.metrics`;
  ranking metrics (MRR/NDCG/MAP via ranx) moved behind the `[eval]` extra,
  imported lazily. `evaluate()` grades optional per-slice pair tracks at the
  same fixed threshold, so a degradation cannot hide behind a corpus average.

### Experiment tracking & observability — run store, `ExperimentTracker`, LLM trace correlation (#99)

The missing spine under langres's otherwise-**ephemeral** benchmark runs (rich results
were printed, then lost — no run id, no config/data snapshot, no cross-run compare):
content-addressed run identity, JSONL persistence, a pluggable tracker seam, and
end-to-end trace correlation — **dependency-free** on the core path (`import langres`
still pulls no `mlflow`/`wandb`).

- **Run store + identity** (`langres.core.runs`, root-exported) — `RunContext` (the
  recipe) + `RunRecord` (recipe + outcomes) with a content-addressed `recipe_id`
  (`sha256` over config/data/seeds, *excluding* code/env provenance so a dirty tree or
  `uv.lock` bump keeps the id) and an `attempt_id` PK; an `fcntl.flock`-guarded,
  append-only JSONL `RunStore` (`read()` collapses `running`+terminal lines
  last-wins-by-`attempt_id`); and `capture_run(context, *, store=None,
  tracker=NoOpTracker())` — writes a `running` line at start and a terminal line on
  exit, and sets the `current_run` contextvar. **`store=None` writes nothing.**
- **`ExperimentTracker` Protocol + adapters** (`langres.core.trackers`) — an
  Accelerate-style seam (`NoOpTracker` null default, `MultiTracker` fan-out to run
  MLflow *and* W&B at once, `resolve_tracker` dispatch) with lazy **MLflow**, **W&B**,
  and **Trackio** adapters behind the `[mlflow]` / `[wandb]` / `[trackio]` extras (a
  missing extra raises a helpful `pip install 'langres[<backend>]'` `ImportError`).
  MLflow defaults to a local file store out of the box; W&B supports keyless
  `offline`/`disabled` runs for CI/no-key use; **Trackio is local-first** (a local
  SQLite store, zero credentials) with an opt-in HF Space/Dataset sync
  (`space_id`/`dataset_id`) gated behind an actionable `ValueError` when no
  `HF_TOKEN` is available — verified against the installed trackio 0.20.2 API.
- **LLM trace correlation** — `capture_run` sets `current_run`; `JudgementLog` records
  the active `run_id`, and `LLMJudge` injects litellm `metadata` (`langres_attempt_id`
  + pair ids + decision step) on **both** the sync (`forward`) and async
  (`forward_async`) paths, so a Langfuse/OTel trace joins the `RunRecord` and
  `JudgementLog`. Off a run (or a non-litellm client) the calls stay byte-identical
  (no `metadata`).
- **DSPy compile lineage** — `DSPyJudge.compile(...)` records the compilation as a
  first-class optimization run via `capture_run` and stamps `_compile_run_id`, so a
  later eval run threads it into `parent_run_id` (compile → eval lineage).
- **`Settings`** — `RUN_STORE_PATH`, `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT` (the
  MLflow ones consumed by `MlflowTracker`; a zero-config default `store` from
  `RUN_STORE_PATH` is deferred to the benchmark wrap — pass `store=` explicitly today),
  and `HF_TOKEN` / `TRACKIO_SPACE_ID` / `TRACKIO_DATASET_ID` (consumed by
  `TrackioTracker` / `create_trackio_tracker`; all optional -- local runs need none).
  Docs: `docs/EXPERIMENTS.md`; runnable zero-spend
  `examples/research/experiment_tracking_demo.py`.

## [0.2.0] - 2026-07-06 — the closed flywheel loop

### ⚠️ BREAKING

- **`judge="auto"` (the default for `link`/`dedupe`) now RAISES `NoJudgeAvailableError`
  when no LLM API key is set, instead of silently falling back to fuzzy string
  matching.** Unsupervised string matching over-merges on unlabeled data (in the
  motivating demo it collapsed five distinct entities into one cluster with no
  error), so the library refuses rather than hand back a confidently-wrong answer.
  The unpinned-model-price branch raises the same error.
- **`fallback_reason` removed** from `DedupeResult` / `LinkVerdict` / `ResolvedModule`
  / `ResolvedJudge` — no path could set it after fail-fast, and an always-`None`
  field is anti-self-describing.

  **Migration:**
  - Keyless callers: pass `judge="string"` explicitly to opt into offline fuzzy
    matching (lower quality; pair it with `derive_threshold` on labeled data).
  - Keyed default path: install the `[llm]` extra (`uv sync --extra llm` /
    `pip install 'langres[llm]'`) **and** export `OPENROUTER_API_KEY`; the run is
    spend-capped at `$1` by default (`budget_usd=`).
  - Catch `NoJudgeAvailableError` (now root-exported from `langres`, alongside
    `BudgetExceeded`) on the front door.
  - Replace any `result.fallback_reason` reads with `result.judge_used` /
    `result.score_type` plus the auto-path selection notice.

### Added — the flywheel closed loop (bootstrap → log → review → harvest → train → cascade)

- **`select_for_review` + `ReviewQueue`** (`langres.core.review`, root-exported) —
  pick the judged pairs most worth a human's attention: `uncertainty` (near the
  threshold), `disagreement` (two logs differ), and first-class `audit` (a seeded
  governance sample that catches confident false merges). Snapshot-semantics queue.
- **`langres` CLI** (`langres.cli`) — `review` (terminal y/n/s/q labeler, resumable),
  `export-csv` / `import-csv` (spreadsheet round-trip, the primary review path),
  `--version`. Formula-injection + terminal-control-char hardened; fully stream-injectable.
- **`CascadeJudge`** (`cascade_judge`) — a cheap student everywhere, escalation only
  inside a `(low, high)` band; escalated provenance preserves `cost_usd`/`model`;
  serializes a fitted student through `Resolver.save`/`load`. (Old `CascadeModule`
  deprecated.)
- **Silver-only calibration guard** — `derive_threshold_from_pairs` warns when every
  pair is a judge verdict (circular); overlay human corrections first.
- **`examples/flywheel_closed_loop.py`** — the whole loop end to end at **$0** on a
  committed Fodors-Zagat fixture, with a data-derived escalation band and an honest
  "plumbing not economics" report.

### Flywheel closed loop — `docs/GETTING_STARTED.md` + doc-ladder rewire

The one entry doc for the closed flywheel (fail-fast `auto`, `select_for_review`
/ `ReviewQueue`, the `langres` review CLI, `CascadeJudge`, and
`examples/flywheel_closed_loop.py`), telling the lifecycle end to end.

- **`docs/GETTING_STARTED.md`** (new, "start here") — the flywheel at altitude:
  LLM bootstrap under a cap (fail-fast `"auto"`: bring an LLM or explicitly opt
  into `judge="string"`) → log from day 1 → review at the margin
  (`select_for_review` + `langres review` / CSV round-trip) → harvest silver +
  gold (with the circularity caveat) → train a cheap trainable student
  (RandomForestJudge, Magellan-style — **not** LLM distillation) + `derive_threshold` →
  `CascadeJudge` → `Resolver.save`/`load`. Every step carries a runnable snippet
  inline; two explicit lanes (keyless `judge="string"` / keyed default `auto`);
  a competitive-positioning section (vs dedupe/Zingg/Splink, "use Splink when… /
  use langres when…"); the audit slice as the governance/trust mechanism; the
  ids-only review mode as the PII privacy posture; and two flywheel operating
  notes (stable ids; one log per run, or dedupe rows before harvest).
- **Snippet-rot guard** (`tests/docs/test_getting_started_snippets.py`) — runs
  the guide's first keyless snippet verbatim at **$0** (hard-refuses any
  non-`judge="string"` block, so a paid snippet can never slip in).
- **Doc-ladder rewire** — `README.md` links GETTING_STARTED first as "start
  here" (+ a quickstart pointer); `docs/TUTORIAL_YOUR_OWN_CSV.md` gains it as
  the big-picture rung and in the calibration tease; `examples/README.md`
  Start-here tier gains `flywheel_closed_loop.py`.


---

Development history before 0.2.0 — the POC milestones (M1–M5, W1–W3: bootstrap
teacher, walking skeleton, the M3/M4 benchmark races, DSPy signature finding,
canonicalizer/harvest/assign) — is preserved in git history (this file prior to
0.3.0) and in `docs/research/`.
