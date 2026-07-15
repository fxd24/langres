# Experiment Tracking & LLM Observability — build plan (review-hardened)

Date: 2026-07-08
Integration branch: `feat/tracking-observability` (off `feat/eval-readiness`)
Status: approved (David) + hardened by a 4-voice gstack autoplan review
(Codex + rev-ceo + rev-eng + rev-dx, 2026-07-08).

Companion research (committed, unmerged): `docs/research/20260708_experiment_tracking_observability_analysis.md`.

## Context — why

langres benchmark runs are **ephemeral**: `run_method`/`run_methods` produce rich
Pydantic results (`MethodResult`, `CostTrack`, the `reports.py` family) that are
`.model_dump()`'d or rendered to markdown, then lost — **no run id, no config
snapshot, no dataset/split identity, no persistence, no cross-run comparison, no
lineage.** Impossible to reproduce months later or compare across sessions —
exactly what the team now needs, reproducing published ER research (Ditto/
Jellyfish/AnyMatch, epic #85) as *clients of langres*, increasingly via agents.

Five-workstream research + first-hand code verification concluded: **don't build a
tracker — add a thin pluggable adapter that wraps langres's existing seams and
reuses its existing Pydantic result models.** langres already has the hard parts
(auto per-judge capture via `LoggingMatcher`, honest cost via `SpendMonitor` +
pinned prices, rich serializable reports, a `"v":1` JSONL idiom). Missing: run
identity + config/dataset/split capture + backend persistence + lineage.

## Decisions (David, at the autoplan gate)

- **D1 — scope: BOTH backends in cut 1.** Ship MLflow **and** W&B adapters (Streams
  S3 + S4), not MLflow-only. The pluggable `ExperimentTracker` Protocol treats them
  symmetrically (no privileged default), honoring "both backends, no single default."
- **D2 — base off `feat/eval-readiness`, defer the `benchmark.py` wrap (Stream C).**
  Verified ground truth (2026-07-08): `feat/eval-readiness` = `main` + 2 planning-doc
  commits only; RR/GMD (#89), the registry (#90), and `slice_fn` are **not merged
  into it** yet (they live on separate unmerged branches; Wave C loaders are the only
  thing live, and they don't touch `benchmark.py`). So `benchmark.py` here is
  byte-identical to main. The disjoint streams build cleanly now; the `benchmark.py`
  wrap is **held until eval-readiness Wave D/E lands** (Wave D adds `slice_fn` to
  `evaluate_judge_on_candidates` — the exact function we'd wrap), so we wrap the final
  shape once, with no double-write.

## What to track — the schema (centerpiece, review-corrected)

Two frozen Pydantic models in a **dep-free** `core/runs.py`. LLM runs are
nondeterministic, so "idempotent replay" = *same recipe → same `recipe_id`*
(detect/skip/compare an already-paid config), **not** same metrics.

### Identity split (rev-ceo/Codex + rev-eng HIGH-1)
- **`recipe_id`** = `sha256(canonical_json(<recipe fields only>))[:16]`, where
  `canonical_json` = `json.dumps(obj, sort_keys=True, separators=(",", ":"))`.
  **Recipe fields (the hash domain) = inputs that determine the run:** `experiment`,
  `resolver_config`, `llm_model`, `cascade_band`, `blocking_k`, `budget_usd`,
  `method`, `dataset_name`, `dataset_fingerprint`, `split_id`, `split_seed`, `seeds`.
  **Explicitly NOT hashed (provenance-only, recorded but excluded):** `git_sha`,
  `git_dirty`, `lockfile_hash`, `langres_version`, `tracking_schema_version`, and
  every timing field. Rationale: recipe_id is a *dedup key over the logical
  experiment* (config+data+seeds), stable across code/dep churn, so a dirty tree or
  a `uv.lock` bump does not mint a new id (fixes HIGH-1; resolves the lockfile debate
  — record it, don't hash it). Code/env identity stays queryable for explaining a
  metrics move, just not part of dedup identity.
- **`attempt_id`** = record PK = `f"{recipe_id}-{started_at}"`. Each attempt is one
  record; the `running` line and the terminal line share `attempt_id`, so the reader
  does **last-wins-by-`attempt_id`**. Idempotency query = "is there a record with
  `recipe_id == X` and `status == completed`?" → skip/compare.

### `RunContext` — the recipe
- *Identity:* `experiment`, `group?`, `parent_run_id?` (lineage: a DSPy-compile run
  parents the eval runs using its program; a sweep parents its per-seed children),
  `tags: dict[str,str]`.
- *Code/env (provenance, NOT hashed):* `git_sha?`, `git_dirty`, `lockfile_hash?`
  (sha256 of `uv.lock`), `langres_version`, `tracking_schema_version=1`,
  `python_version?`, `platform?`, `reproduction_adapter_version?`.
- *Config (hashed):* `resolver_config?` (full pipeline snapshot via the new
  `Resolver.config_dict()` — **best-effort**, see S2/HIGH-2), `llm_model?`,
  `cascade_band?`, `blocking_k?`, `budget_usd?`, `method?`.
- *Data (hashed):* `dataset_name`, `dataset_fingerprint?` (sha256 over corpus+gold),
  `split_id?`, `split_seed`.
- *Seeds (hashed):* `seeds: dict[str,int]` — named union of every source (`"split"`
  + each stochastic component's `random_state`/`seed` from its `ComponentSpec.config`).
  (Store `split_seed` **once**, in `seeds["split"]`; drop the duplicate scalar field —
  rev-eng HIGH-1 minor.)

### `RunRecord` — `context` + outcomes, one JSONL line
- `attempt_id` (PK), `recipe_id`, `context`, `v=1`.
- *Timing (never hashed):* `started_at`, `finished_at?`, `duration_seconds?`.
- *Metrics (reuse existing models as an opaque dict — HIGH-4):* `metrics?` (a
  `MethodResult`/`JudgePairEval`/`HonestPairEval`/`BenchmarkTable` `.model_dump()`),
  `metric_definition?` (`"pair_f1@best_threshold"` vs `"bcubed_f1"` vs
  `"honest_fixed_split"` — self-labels so comparison can't silently mix definitions),
  `per_seed_metrics?`, `headline_metric?`.
- *Cost (langres-native):* `spend_usd` (= `SpendMonitor.spent` = Σ judgement
  `cost_usd`), `budget_exceeded`.
- *Artifacts:* `judgement_log_path?`, `trace_id?` (= `attempt_id`, threaded to
  Langfuse), `artifacts: dict[str,str]` (resolver.save dir, report md,
  `mlflow_run_url`, `wandb_run_url`, `langfuse_trace_url`).
- *Status/failure:* `status: Literal["running","completed","failed","budget_exceeded"]`
  (**`"running"` written at start** → a crashed/torn-down run leaves a visible gap),
  `error_type?`, `error_message?` (truncated, no full traceback/PII).

### Gap-closing helpers (all reuse existing internals; verified feasible)
- `Resolver.config_dict() -> dict` (S2) — factor the `save()` loop
  (`resolver.py:655-676`: `_component_spec` over `_slots()` → `ArtifactManifest`) into
  a public method returning `.model_dump()` **without** writing to disk. `save()` then
  = `config_dict()` + write + sidecars.
- `compute_recipe_id(context) -> str` (public, in `__all__`) — the hash above.
- `git_sha() -> tuple[str|None, bool]` — `git rev-parse HEAD` + `--porcelain` dirty;
  short `timeout`, `check=False`, `cwd`=repo root, swallow `FileNotFoundError` →
  `(None, False)`; one-time `logger.warning` when sha is None (rev-dx Mi2).
- `dataset_fingerprint(corpus, gold) -> str` — sha256 over **already-loaded**
  `(corpus, gold)` (thread them in; do NOT re-`load()` — rev-eng M-2); canonical order.
- `_collect_seeds(split_seed, resolver_config) -> dict[str,int]` — split seed + scan
  configs for `random_state`/`seed` keys; degrades to just the split seed if config is None.

## Design — the layer

```
core/runs.py            NEW dep-free: RunContext, RunRecord, RunStore (JSONL
                        append/read, last-wins-by-attempt_id, fcntl.flock per append),
                        capture_run(), current_run contextvar, compute_recipe_id(),
                        git_sha(), dataset_fingerprint(), _collect_seeds(), env helpers,
                        resolve_store() (str|Path|RunStore -> RunStore)
core/trackers/__init__.py  NEW dep-free: ExperimentTracker(Protocol), NoOpTracker,
                        MultiTracker (exposes .trackers), resolve_tracker(); lazy
                        __getattr__ -> Mlflow/Wandb adapters
core/trackers/mlflow_tracker.py  NEW (S3): MlflowTracker — lazy `import mlflow`
core/trackers/wandb_tracker.py   NEW (S4): WandbTracker — reuses clients/tracking.py
core/resolver.py        (S2) + Resolver.config_dict()
core/modules/llm_judge.py  (S5) + litellm metadata {langres_attempt_id,left_id,
                        right_id,decision_step} — ONLY when current_run set AND on the
                        litellm path (gate so a user-supplied client can't 400)
core/modules/dspy_judge.py (S6) + named tracker/store/parent_run_id on compile()
                        (bound before **kwargs); stamp judge._compile_run_id for lineage
core/judgement_log.py   (S5) + nullable run_id (= current attempt_id) on the row
core/__init__.py        (S1) + eager RunContext/RunRecord/RunStore/ExperimentTracker/
                        MultiTracker/NoOpTracker/capture_run/compute_recipe_id/
                        resolve_tracker/resolve_store; lazy MlflowTracker/WandbTracker
clients/settings.py     (S1) + run_store_path / mlflow_tracking_uri / mlflow_experiment
pyproject.toml          (S3) + [project.optional-dependencies] mlflow=[...], wandb=[...]
                        (real extras — rev-dx H1); mlflow added to dev group (wandb
                        already there). Attempt `uv lock`; if the index is unreachable
                        offline, flag a lock-refresh follow-up — do NOT block adapter code
tests/test_import_budget.py (S1) + "mlflow","wandb" in _HEAVY_MODULES; + subprocess
                        assertions that ranx/mlflow/wandb NOT in sys.modules after
                        `import langres` (HIGH-4 + LOW)
examples/research/experiment_tracking_demo.py (S7) NEW — mirrors judgement_log_demo.py
docs/EXPERIMENTS.md     (S7) + "Persisting & comparing runs" section
```

### `ExperimentTracker` Protocol (Accelerate `GeneralTracker` shape)
`name`; `start_run(context, *, run_name=None)`; `log_params`; `log_metrics(..., step=None)`;
`log_artifact`; `set_tags`; `finish(*, status)`; `run_url` property (deep link →
`RunRecord.artifacts`); `native` property (escape hatch — renamed from `.tracker`,
rev-dx Mi1). Backends flatten `context`→params themselves.
- **`MultiTracker`** — fan-out to N children (compose, not merge); exposes `.trackers:
  list` so a specific backend is reachable when running MLflow+W&B together (Mi1).
- **`NoOpTracker`** — null default; zero overhead when unconfigured.
- **`resolve_tracker(spec)`** — `None`→NoOp; `"mlflow"`/`"wandb"`→lazy adapter
  (availability-gated `ImportError` naming the real extra: `pip install 'langres[mlflow]'`);
  an instance→as-is (DI); a sequence→`MultiTracker`. Mirrors `resolve_judge`.
- **`resolve_store(spec)`** — `None`→None (no persistence); `str|Path`→`RunStore(path)`;
  a `RunStore`→as-is. Symmetric with the `log:` precedent on `link`/`dedupe` (rev-dx M1).

### `capture_run(context, *, store=None, tracker=NoOpTracker())` — the one primitive
Context manager: compute `recipe_id`+`attempt_id`; if `store` set, mkdir-parents and
append `status="running"` (wrap write failures as `RunStoreError` with an actionable
message — rev-dx M4); set `current_run` contextvar (use set/reset tokens so nested
capture restores the parent on exit — rev-eng); `tracker.start_run`; yield a handle
(`log_metrics`/`log_artifact`/`set_status`); on exit finalize the `RunRecord`
(status/metrics/cost/timing/artifacts + `run_url`), append terminal line, `tracker.finish`.

### Default persistence (rev-dx C1 — the headline fix)
The plan's whole point is killing ephemerality, so **persistence is free**: the
harness wrap (deferred Stream C) will default `store` to
`Settings.run_store_path` (a conventional gitignored path, e.g. `runs/langres_runs.jsonl`;
`JudgementLog` sets the mkdir-parents precedent at `judgement_log.py:84`). For cut 1
(no harness wrap yet), `capture_run(store=None)` writes nothing — the invariant is
"**`store=None` → no files**" (reconciles the old contradiction between "unconditional"
and verification #9). Agents/users pass a store or path explicitly; the zero-config
default lands with Stream C.

## Execution — parallel worktree streams → `feat/tracking-observability`

Single-owner-per-file (rule #5). Each stream = an isolated worktree + TDD agent,
committing early (worktree-teardown rule), merged into the integration branch after a
gstack `/review` + internal `langres-code-reviewer` pass.

| Stream | Files owned | Depends on | Wave |
|---|---|---|---|
| **S1 Foundation** | `core/runs.py`, `core/trackers/__init__.py`, `core/__init__.py`, `clients/settings.py`, `tests/test_import_budget.py`, `tests/test_runs.py`, `tests/test_trackers.py` | — | 1 |
| **S2 config_dict** | `core/resolver.py`, `tests/test_resolver_config_dict.py` | — | 1 |
| **S3 MLflow** | `core/trackers/mlflow_tracker.py`, `pyproject.toml`, `tests/test_mlflow_tracker.py` | S1 | 2 |
| **S4 W&B** | `core/trackers/wandb_tracker.py`, `tests/test_wandb_tracker.py` | S1 | 2 |
| **S5 LLM correlation** | `core/modules/llm_judge.py`, `core/judgement_log.py`, tests | S1 | 2 |
| **S6 DSPy compile** | `core/modules/dspy_judge.py`, tests | S1 | 2 |
| **S7 Examples+docs** | `examples/research/experiment_tracking_demo.py`, `docs/EXPERIMENTS.md` | S1 | 2 |
| **Stream C (DEFERRED)** | `core/benchmark.py` wrap + default store + run_methods example | eval Wave D/E merged | fast-follow |

Wave 1 = S1, S2 (parallel roots). **S1 merges into the integration branch first**
(it owns the exports every Wave-2 stream imports); then Wave 2 = S3, S4, S5, S6, S7
branch off the updated integration branch, fully parallel. Wave 3 = integration
verification.

## Acceptance criteria (per stream — enforce by reviewer discipline; per-PR CI does NOT gate coverage)
- **S1 invariants (the subtle-bug locus):** pure logic (RunContext/RunRecord/RunStore/
  resolve_tracker/resolve_store/NoOp/Multi/compute_recipe_id) at **95–100%** cov.
  `recipe_id` excludes `git_dirty`/`lockfile_hash`/code/env/timing. `import langres`
  pulls **no** ranx/mlflow/wandb/litellm (subprocess assertion) and stays <2.0s.
  `runs.py` result-model refs are `TYPE_CHECKING`-only (HIGH-4). RunStore append uses
  `fcntl.flock` (cross-process safety, rev-eng M-1). `store=None` → no files.
- **S2:** `config_dict()` == the dict `save()` would manifest, no disk write; round-trips
  through the registry for the built-in judges. (Compiled-DSPy state is a known
  limitation — document it; S5/S6 don't rely on it.)
- **S3/S4:** adapter bodies are behavior/smoke with `# pragma: no cover` on un-mockable
  external calls (rev-eng M-5); lazy `import mlflow`/`wandb` only inside the adapter;
  `resolve_tracker("mlflow")` raises a helpful `ImportError` when the extra is absent.
  S3 owns `pyproject.toml` (adds both `[mlflow]` and `[wandb]` extras + `mlflow` dev dep);
  S4 adds only `wandb_tracker.py` (wandb is already a dependency).
- **S5:** `completion()` gets `metadata=` **only when `current_run` is set** AND on the
  litellm path (test the omitted-tracker path asserts NO `metadata` kwarg — locks the
  byte-identical invariant); `JudgementLog` row carries the `attempt_id` for the exact
  three-way join (record ↔ judgement ↔ trace).
- **S6:** `compile()` binds `tracker`/`store`/`parent_run_id` before `**kwargs`; stamps
  `judge._compile_run_id` so a later `capture_run` can read it into `parent_run_id`.
- **S7:** demo mirrors `examples/judgement_log_demo.py`; shows `capture_run(..., store=)`
  then `RunStore(path).read()` + a filter-by-experiment + 2-run metric diff (rev-dx M5),
  plus a two-line agent idempotency+budget snippet (rev-dx M3); EXPERIMENTS.md section
  wired into its "See also".

## Verification — beyond testing (Wave 3)
1. **Import budget** — `pytest tests/test_import_budget.py`; `mlflow`+`wandb` in
   `_HEAVY_MODULES`; bare `import langres` pulls no mlflow/wandb/**ranx**, <2.0s, no env leak.
2. **Inspect a RunRecord** — `capture_run` a zero-spend flow with `store=` + a
   `MultiTracker(["mlflow","wandb"])`; read the JSONL; assert `attempt_id`, `recipe_id`,
   `git_sha`/dirty, `resolver_config`, `seeds` union, `dataset_fingerprint`, `metrics`,
   `metric_definition`, `spend_usd`, `status`, `"v":1`, and a `"running"` line preceding
   the terminal line (crash-visibility).
3. **Idempotent replay** — same recipe → `compute_recipe_id` equal; dirty tree flips
   `git_dirty=True` while `recipe_id` stays equal (HIGH-1 regression guard); distinct
   seeds → distinct `recipe_id` (no collision).
4. **Dataset-mutation detection** — mutate one corpus row → `dataset_fingerprint` and
   `recipe_id` change.
5. **Best-effort snapshot** — an unregistered/bespoke judge (no `type_name`) → capture
   yields `resolver_config=None`, **no crash** (HIGH-2).
6. **MLflow + W&B UIs** — run appears with params (`git_sha`, `llm_model`, `blocking_k`,
   `seeds.*`) and metrics; `run_url == RunRecord.artifacts[...]`.
7. **Cost cross-check** — `spend_usd == SpendMonitor.spent == Σ JudgementLog cost_usd`.
8. **Crash visibility (fault injection)** — kill mid-`capture_run` → a lone `running`
   line survives and the reader reports it.
9. **mypy strict + coverage** — `uv run mypy src/` clean; S1 pure logic 95–100%.

## Risks
- **litellm `langfuse_otel` metadata passthrough** — the one unverified seam, but
  **off the critical path**: Phase-3 correlation's *primary* join is `attempt_id`
  written into `JudgementLog` + `RunRecord` (JSONL, works regardless of Langfuse).
  Verify the OTel span-attribute passthrough in S5; fallback = set span attrs via the
  Langfuse SDK inside `forward`.
- **Cross-process RunStore append** — `fcntl.flock` per append (S1) handles the
  multi-agent case (rev-eng M-1).
- **Building on unmerged `feat/eval-readiness`** — cut 1 is disjoint from eval code, so
  it rebases cleanly; the PR-target / rebase-onto-main decision is taken at integration
  time (surfaced to David then), once eval-readiness's own timeline is clear.

## Deferred to fast-follow (after eval-readiness Wave D/E merges)
- **Stream C** — wrap `run_method`/`run_methods`/`evaluate_judge_on_candidates` (final
  post-`slice_fn` shape) in `capture_run`; default `store` on; `run_methods` opens a
  **parent** sweep run and threads `parent_run_id`+`llm_model` to children; update the
  demo to the `run_methods(store=)` one-liner.
- Aim adapter; the agent-ready sugar (`has_run`/remaining-budget/failure API) — the
  Phase-0 primitives already make these two-liners (rev-dx M3), so this is sugar only.
