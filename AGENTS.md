# Agent Guidelines for langres

> **Lean router.** Detail lives in modular rules under `.claude/rules/` and in
> `docs/`. Claude Code may auto-load those rules by scope; Hermes and other
> agents must read the relevant rule before writing code in its domain.
>
> **Keep docs in sync with code.** When a change touches behavior, paths,
> commands, conventions, or data contracts that this file, a rule, or anything
> under `docs/` describes, update the relevant doc/rule in the **same** change —
> not a follow-up. Stale docs mislead silently.

## Project Overview

**langres** is a Python entity resolution framework in early development. It aims to provide a composable, optimizable approach to entity resolution with a layered API: named **architectures** (`langres.architectures.FuzzyString` / `VectorLLMCascade` — whole ER pipelines you construct, then call `.dedupe()`/`.compare()` on) over a declarative **`ERModel`** (aliased as `Resolver`) over low-level **`langres.core`** primitives. There is no `matcher="auto"` key-sniffing front door — W4 deleted the two module-level verbs (`langres.link`/`langres.dedupe`) and `core.presets` outright; naming a model is the user's job, not a heuristic's. (Note: there is also no `langres.tasks`/`flows` layer — that was earlier doc fiction; see `docs/USE_CASES.md` and `.claude/rules/component-design.md`.)

**Current Stage**: The initial POC — validating classical rapidfuzz, semantic
vectors, and hybrid blocking + LLM matching — is **complete**;
`docs/POC.md` is an archived historical record. langres is now a shipped 0.x
beta on PyPI under Apache-2.0.

**📋 See `docs/ROADMAP.md` for current direction and milestones** and the root
`CHANGELOG.md` for what shipped.

**Current focus**: Building production-quality `langres.core` primitives under a **tiered coverage policy** (95–100% on the `core` contract, behavior/smoke on harness code — see `.claude/rules/testing.md`). This is NOT throwaway prototype code—these components will become the foundation of the full library.

## How I Work — Rules (`.claude/rules/`)

Treat the following as the rule router. Claude Code may auto-load these files;
Hermes and other agents must load the always-on rules and read each relevant
path-scoped rule before editing files in its scope.

**Always-on:**
- `expert-knowledge.md` — verify-before-asserting, hypotheses ≠ facts, own the failure, stay in scope, **commit before the worktree disappears**, timeouts. The baseline for how to reason and act.
- `data-safety.md` — irreversible-actions guardrail; uncommitted changes are sacred.
- `context-management.md` — delegate output-heavy ops to subagents; parallelize independent work.

**Path-scoped:**
- `python-style.md` *(`**/*.py`, `pyproject.toml`)* — type hints, Pydantic-first, `uv`, no `print()`, naming.
- `component-design.md` *(`src/**`)* — the layered API (architectures → ERModel → core), design principles, lightweight & composable / SRP, common patterns, adding components (incl. the three judge-dispatch sites).
- `testing.md` *(`tests/**`)* — tiered coverage (high on `core`, behavior-focused on harness), markers, human-like dev-iteration loop.
- `token-efficiency.md` *(`.claude/agents|skills|commands/**`)* — agent cost discipline (Edit-over-Write, Grep-before-Read, JSON-between-agents, reasoning-tier).

## Skills

- `prompting-Codex-4` — expert guidance for prompting Codex 4.x models (XML patterns, behavioral fixes, extended thinking). Use when writing system prompts for the LLM judge / matching modules, or any agent definition.

## Project Structure

```
langres/
├── src/langres/
│   ├── architectures/  # Named ER pipelines: FuzzyString ($0/offline), VectorLLMCascade (paid) — construct one, call .dedupe()/.compare()
│   ├── cli.py          # langres CLI: review / export-csv / import-csv (labeling loop)
│   ├── core/           # Low-level primitives + the Resolver
│   │   ├── resolver.py     # ERModel (aliased Resolver): the class + from_schema / fit / the anchor surface; no matcher="auto"
│   │   ├── _model_state.py, _model_run.py, _model_persist.py  # ERModel split by responsibility: what it IS / how it runs / how it persists
│   │   ├── _artifacts.py           # save/load leaf (paths, manifest plumbing) under _model_persist
│   │   ├── inputs.py       # normalize_records: raw dicts -> (schema, normalized records); schema inference for a schema-less dedupe()/compare()
│   │   ├── results.py      # LinkVerdict / DedupeResult — architecture + backbone + score_type + threshold
│   │   ├── registry.py     # component config-registry (type_name -> class) for save/load
│   │   ├── blocker.py, blockers/   # AllPairsBlocker, VectorBlocker
│   │   ├── comparator.py           # StringComparator, ComparisonVector
│   │   ├── module.py, modules/, judges/  # Module (judge) ABC + LLMJudge, CascadeJudge, etc.
│   │   ├── clusterer.py            # Clusterer (transitive closure)
│   │   ├── runs.py, judgement_log.py, trackers/  # → back-compat SHIMS; observability moved to langres.tracking (below). `# TEMPORARY: deleted by the W2 sweep`
│   │   ├── calibration.py          # derive_threshold
│   │   ├── reports.py              # inspection/evaluation report models (ScoreInspectionReport, BlockerEvaluationReport, ...)
│   │   └── usage.py                # LLMUsage + CostTrack — the token/cost leaf (imports nothing from langres)
│   ├── optimize.py     # langres.optimize / score_blocking: the import-light facade (stdlib-only module top; every engine import is lazy). A MODULE, not a package — `langres.optimize` is a CALLABLE, so a submodule under that name is unreachable by attribute traversal (`import langres.optimize.loop as l` → ImportError). The engine lives next door:
│   ├── autoresearch/   # the autoresearch ENGINE — blocking search, NOT ER modelling, so it sits outside core (depends on core one-way; core imports nothing from here)
│   │   ├── __init__.py     # docstring only — exports NOTHING, which is what keeps it import-light (factory/blocker_optimizer are heavy)
│   │   ├── objective.py / search_space.py / factory.py / loop.py  # the keep-if-better scorer, the config grid, config→blocker (HEAVY), the propose→run→evaluate→keep driver
│   │   └── blocker_optimizer.py  # BlockerOptimizer (Optuna study; optuna is dev-only — lazy-import only)
│   ├── tracking/       # observability, NOT ER modelling — beside core (one-way dep; the langres.core facade re-exports these for back-compat): runs.py (RunContext/RunStore + capture_run), judgement_log.py (JudgementLog/LoggingMatcher), factories.py (create_*_tracker), trackers/ (ExperimentTracker + lazy Mlflow/Wandb/Trackio adapters)
│   ├── report/         # the shared $0 rendering seam (presentation, NOT modelling — so it sits beside core, not in it)
│   ├── curation/       # human-in-the-loop labelling + gold-set cold-start (the dissolved langres.bootstrap). core/{review,harvest,anchor_store,canonicalizer}.py are TEMPORARY W2-sweep back-compat shims re-exporting from here
│   │   ├── review.py       # select_for_review + ReviewQueue (pick the uncertain margin)
│   │   ├── harvest.py      # Correction/CorrectionLog, harvest_labeled_pairs, derive_threshold_from_pairs, align_pairs
│   │   ├── anchor_store.py         # AnchorStore / ClusterDelta (hold the anchors; assign incoming records)
│   │   ├── canonicalizer.py        # Canonicalizer (survivorship: fold a cluster into one golden record)
│   │   └── base.py, miners.py, models.py, labelers.py, bootstrapper.py, report.py, _pairs.py  # gold-set cold-start: Miner/Labeler, HardNegativeMiner, GoldPair/GoldSet, Bootstrapper, BootstrapReport
│   ├── methods.py      # method registry / _make_module_builder (benchmark path)
│   ├── clients/        # OpenRouter client, SpendMonitor, pricing
│   ├── metrics/        # ER metrics + diagnostics (metrics/analysis/debugging/diagnostics) — they SCORE a resolution, not the modelling contract, so beside core; public via langres.eval, back-compat shims at core.metrics/.analysis/.debugging/.diagnostics
│   └── data/           # benchmark dataset loaders (FZ, Amazon-Google, ...)
├── tests/              # Test suite
├── examples/           # Usage examples (quickstart_models.py is the offline quickstart)
└── docs/               # Documentation
```

**Not built yet** (roadmap — do not reference as existing): `tasks`/`flows`
modules, a general `Optimizer`, a synthetic data generator.

## Dependencies

**Core** (always installed, `uv sync`): Pydantic + pydantic-settings (validation), rapidfuzz (string similarity), networkx (graph clustering), numpy. The string-judge/`AllPairsBlocker` path works with only these.

**Extras** (opt-in, `uv sync --all-extras` or `pip install langres[semantic,llm,trained]`):
- `[semantic]` — sentence-transformers, torch, faiss-cpu, onnxruntime/optimum, qdrant-client (`VectorBlocker`, embeddings, vector indexes).
- `[llm]` — litellm, dspy-ai, openai (`LLMJudge`, DSPy-compiled judges).
- `[trained]` — scikit-learn (`RandomForestJudge`, the W1.2 trained-family judge, and `core.calibration.derive_threshold`).

These heavy/optional symbols resolve lazily (PEP 562 `__getattr__` in `langres/core/__init__.py` and `langres/clients/__init__.py`) so a bare `import langres` never pulls torch/litellm/faiss/scikit-learn into `sys.modules` — see `tests/test_import_budget.py`. Optuna/wandb/langfuse/ranx are dev-only (`[dependency-groups] dev`), for eval tooling, not the production `dedupe()`/`compare()` path (scikit-learn is duplicated in the dev group too, so the repo's own test suite doesn't need `--all-extras` for a bare `uv sync`).

**Dev tools**: ruff (format + lint), pytest + pytest-cov (tests), mypy (strict-mode type checking).

## Important Notes

- **Always verify claims before you assert them.** Never present an unverified hypothesis — about code, tooling, model/library capabilities, or data — as fact. Check the source, run the code, read the data first; if you can't, label it explicitly as unverified. (Detail in `.claude/rules/expert-knowledge.md`.)
- This is an **early-stage project** - expect significant changes
- Prioritize clean, testable code over premature optimization
- Document design decisions in code comments
- Focus on the core use cases: Deduplication and Entity Linking (V1 scope)

## Agent Analysis & Expert Feedback (`.agent/`)

The `.agent/` folder contains external expert analyses of the langres project:

- **`.agent/genalysis/20251029_er_use_cases_expert_analysis.md`**: Taxonomy of 18+ entity resolution use cases, mapping each to langres components, identifying gaps (incremental resolution, temporal support, streaming), and comparing langres to state-of-the-art ER systems (Dedupe.io, Splink, Zingg). Essential for understanding production requirements and missing features.
- **`.agent/genalysis/20251029_comprehensive_documentation_evaluation.md`**: Expert evaluation (7.5/10) of architecture, feasibility, critical problems (blocking scalability, DSPy cost, clustering guarantees), and production-readiness gaps.

**When to consult**: before planning new features (check if already identified as a gap); when considering production requirements; when prioritizing work (these docs separate critical from nice-to-have).

**Note on documentation structure**: Keep `AGENTS.md` concise and actionable. Substantial new guidance (>50 lines) belongs in a focused `.claude/rules/*.md` or `.agent/` doc linked from here, not inline — this keeps the always-on instructions scannable.

## Reference Documentation (`docs/`)

- **`docs/ROADMAP.md`** ⭐ **START HERE / DIRECTION** — the post-POC vision: langres as the composable ER seam; the feature-bag architecture; the use-case compass; verifiable milestones M0–M6. Read before planning new work.
- **`docs/POC.md`** — archived original POC validation plan; historical record only.
- **`docs/TECHNICAL_OVERVIEW.md`** — API reference and data contracts (`PairwiseJudgement`, `Candidate`, method signatures, expected inputs/outputs).
- **`docs/USE_CASES.md`** — use-case taxonomy and roadmap (V1 / V1.1 / out-of-scope; streaming, temporal, collective resolution).
- **`docs/DX_RESOLVER.md`** — before/after of the M0 `Resolver`: the manual lambda pipeline vs. the declarative `from_schema` + `save`/`load` path.
- **`docs/EXPERIMENTS.md`** — experimentation DX getting-started: the `run_methods` full-pipeline race vs. `evaluate_judge_on_candidates` (judged-once) for compiled/paid judges; `derive_threshold` to kill magic constants; the `SpendMonitor` budget seam.
- **`CHANGELOG.md`** (repo root) — release history (0.3.0 / 0.2.0); pre-0.2.0 POC milestone history is preserved in git history and `docs/research/`.
