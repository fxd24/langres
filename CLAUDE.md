# Claude Code Guidelines for langres

> **Lean router.** Detail lives in modular rules under `.claude/rules/` and in
> `docs/`. Some rules are **always-on**; others are **path-scoped** (they load
> only when you touch a file in their scope). Read the relevant rule before
> writing code in its domain.
>
> **Keep docs in sync with code.** When a change touches behavior, paths,
> commands, conventions, or data contracts that this file, a rule, or anything
> under `docs/` describes, update the relevant doc/rule in the **same** change —
> not a follow-up. Stale docs mislead silently.

## Project Overview

**langres** is a Python entity resolution framework in early development. It aims to provide a composable, optimizable approach to entity resolution with a layered API: named **architectures** (`langres.architectures.FuzzyString` / `VectorLLMCascade` — whole ER pipelines you construct, then call `.dedupe()`/`.compare()` on) over a declarative **`ERModel`** (aliased as `Resolver`) over low-level **`langres.core`** primitives. There is no `matcher="auto"` key-sniffing front door — W4 deleted the two module-level verbs (`langres.link`/`langres.dedupe`) and `core.presets` outright; naming a model is the user's job, not a heuristic's. (Note: there is also no `langres.tasks`/`flows` layer — that was earlier doc fiction; see `docs/USE_CASES.md` and `.claude/rules/component-design.md`.)

**Current Stage**: The initial POC — validating the architecture through three progressively sophisticated approaches (classical rapidfuzz, semantic vectors, hybrid blocking + LLM judge) — is **complete**; `docs/POC.md` is kept as an archived record. langres is now a shipped 0.x beta (on PyPI, Apache-2.0).

**📋 See `docs/ROADMAP.md` for direction and milestones** and the root `CHANGELOG.md` for what shipped.

**Current focus**: Building production-quality `langres.core` primitives under a **tiered coverage policy** (95–100% on the `core` contract, behavior/smoke on harness code — see `.claude/rules/testing.md`). This is NOT throwaway prototype code—these components will become the foundation of the full library.

## How I Work — Rules (`.claude/rules/`)

These auto-load. **Always-on** rules apply every session; **path-scoped** rules
load only when you read/edit a file matching their `paths:`.

**Always-on:**
- `expert-knowledge.md` — verify-before-asserting, hypotheses ≠ facts, own the failure, stay in scope, **commit before the worktree disappears**, timeouts. The baseline for how to reason and act.
- `data-safety.md` — irreversible-actions guardrail; uncommitted changes are sacred.
- `context-management.md` — delegate output-heavy ops to subagents; parallelize independent work.

**Path-scoped:**
- `python-style.md` *(`**/*.py`, `pyproject.toml`)* — type hints, Pydantic-first, `uv`, no `print()`, naming.
- `component-design.md` *(`src/**`)* — the layered API (architectures → ERModel → core), design principles, lightweight & composable / SRP, common patterns, adding components (incl. the single judge/method registry, `core/method_registry.py`).
- `testing.md` *(`tests/**`)* — tiered coverage (high on `core`, behavior-focused on harness), markers, human-like dev-iteration loop.
- `token-efficiency.md` *(`.claude/agents|skills|commands/**`)* — agent cost discipline (Edit-over-Write, Grep-before-Read, JSON-between-agents, reasoning-tier).

## Skills

- `prompting-claude-4` — expert guidance for prompting Claude 4.x models (XML patterns, behavioral fixes, extended thinking). Use when writing system prompts for the LLM judge / matching modules, or any agent definition.

## Project Structure

```
langres/
├── src/langres/
│   ├── architectures/  # Named ER pipelines: FuzzyString ($0/offline), VectorLLMCascade (paid) — construct one, call .dedupe()/.compare()
│   ├── optimize.py     # langres.optimize / score_blocking: the import-light facade (stdlib-only module top; every engine import is lazy). A MODULE, not a package — `langres.optimize` is a CALLABLE, so a submodule under that name is unreachable by attribute traversal (`import langres.optimize.loop as l` → ImportError). The engine lives next door:
│   ├── autoresearch/   # the autoresearch ENGINE — blocking search, NOT ER modelling, so it sits outside core (depends on core one-way; core imports nothing from here)
│   │   ├── __init__.py     # docstring only — exports NOTHING, which is what keeps it import-light (factory/blocker_optimizer are heavy)
│   │   ├── objective.py    # the immutable keep-if-better scorer (Pareto + log_loss)
│   │   ├── search_space.py # the declarative config grid the loop enumerates
│   │   ├── factory.py      # config -> runnable blocker. HEAVY ([semantic] at module top) — lazy-import only
│   │   ├── loop.py         # propose → run → evaluate → keep, over core.runs persistence
│   │   └── blocker_optimizer.py  # BlockerOptimizer (Optuna study; optuna is dev-only — lazy-import only)
│   ├── eval.py         # Curated evaluation facade (lazy): evaluate, list_benchmarks/get_benchmark, ER metrics
│   ├── cli.py          # langres CLI: review / export-csv / import-csv (labeling loop)
│   ├── _exports/       # per-domain fragments composing the ROOT __all__ + lazy maps (add a root export HERE, not in __init__.py)
│   ├── core/           # Low-level primitives + the Resolver
│   │   ├── _exports/       # same, for langres.core (add a core export HERE, not in core/__init__.py)
│   │   ├── resolver.py     # ERModel (aliased Resolver): the class + from_schema / fit / the anchor surface; no matcher="auto"
│   │   ├── _model_state.py     # ERModel layer: what a model IS — slots, identity, the 3 construction doors, schema binding
│   │   ├── _model_run.py       # ERModel layer: how it RUNS — block → (compare) → score → cluster; dedupe / compare
│   │   ├── _model_persist.py   # ERModel layer: how it PERSISTS — the resolver.json manifest + per-slot sidecars (no pickle)
│   │   ├── _artifacts.py       # component ⇄ ComponentSpec adapters (the leaf _model_persist serializes each slot with)
│   │   ├── inputs.py       # normalize_records: raw dicts -> (schema, normalized records); schema inference for a schema-less dedupe()/compare()
│   │   ├── results.py      # LinkVerdict / DedupeResult — architecture + backbone + score_type + threshold
│   │   ├── spend.py, spend_cap.py  # SpendMonitor/BudgetExceeded ledger + SpendCappedMatcher (the ONE enforcer) + DEFAULT_BUDGET_USD; core leaf, so ERModel/every architecture can cap
│   │   ├── method_registry.py  # ONE MethodSpec registry: judge/method name -> builder + identity (all three dispatch paths resolve here)
│   │   ├── registry.py     # component config-registry (type_name -> class) for save/load
│   │   ├── blocker.py, blockers/   # AllPairsBlocker, VectorBlocker
│   │   ├── comparator.py, comparators/  # Comparator ABC (contract) + StringComparator (impl)
│   │   ├── module.py, modules/, judges/  # Module (judge) ABC + LLMJudge, CascadeJudge, etc.
│   │   ├── clusterer.py            # Clusterer (transitive closure)
│   │   ├── judgement_log.py        # JudgementLog + LoggingModule (logs every judge call: ids, score, verdict, model, cost)
│   │   ├── review.py       # select_for_review + ReviewQueue (pick the uncertain margin)
│   │   ├── harvest.py      # Correction/CorrectionLog, harvest_labeled_pairs, derive_threshold_from_pairs
│   │   ├── calibration.py          # derive_threshold
│   │   └── reports.py              # inspection/evaluation report models (ScoreInspectionReport, BlockerEvaluationReport, ...)
│   ├── methods.py      # method registry / _make_module_builder (benchmark path)
│   ├── clients/        # OpenRouter client, SpendMonitor, pricing
│   ├── report/         # the shared $0 rendering seam (presentation, NOT modelling — so it sits beside core, not in it)
│   │   ├── _svg.py         # pure-stdlib inline-SVG chart primitives (line_chart/bar_chart); imports nothing from langres
│   │   ├── _report_html.py # shared HTML scaffold: document()/section()/_num/_histogram/safe_auc
│   │   └── eval_report.py  # EvalReport, the $0 tearsheet (public home: langres.eval / root langres, both lazy)
│   └── data/           # benchmark dataset loaders (FZ, Amazon-Google, ...)
│       └── registry.py # name→benchmark manifest: list_benchmarks() / get_benchmark()
├── tests/              # Test suite
├── examples/           # Usage examples (quickstart_models.py is the offline quickstart)
└── docs/               # Documentation
```

**Not built yet** (roadmap — do not reference as existing): `tasks`/`flows`
modules, a general `Optimizer`, a synthetic data generator.

## Dependencies

**Core** (always installed, `uv sync`): Pydantic + pydantic-settings (validation), rapidfuzz (string similarity), networkx (graph clustering), numpy. The string-judge/`AllPairsBlocker` path works with only these.

**Extras** (opt-in, `uv sync --all-extras` or `pip install langres[semantic,llm,trained,eval]`):
- `[semantic]` — sentence-transformers, torch, faiss-cpu, onnxruntime/optimum, qdrant-client (`VectorBlocker`, embeddings, vector indexes).
- `[llm]` — litellm, dspy-ai, openai (`LLMJudge`, DSPy-compiled judges).
- `[trained]` — scikit-learn (`RandomForestJudge`, the W1.2 trained-family judge, and `core.calibration.derive_threshold`).
- `[eval]` — ranx (ranking metrics MRR/NDCG/MAP in `core.metrics.evaluate_blocking_with_ranking`). Imported lazily, so the rest of `core.metrics`/`core.benchmark` (BCubed/pairwise metrics, `evaluate()`) stays importable without it.

These heavy/optional symbols resolve lazily (PEP 562 `__getattr__` in `langres/core/__init__.py`, the implementation packages such as `langres/core/matchers/__init__.py`, and `langres/clients/__init__.py`) so a bare `import langres` never pulls torch/litellm/faiss/scikit-learn/ranx into `sys.modules` — see `tests/test_import_budget.py`.

**`langres.core` re-exports contracts, not implementations.** It carries the data models, the `Blocker`/`Comparator`/`Matcher`/`Clusterer` base types, the opt-in capability Protocols (`Inspectable` for `inspect_scores`, the `fit` mixins), the `Resolver` + registry, the method registry and the training/tracking primitives — the things a pipeline is *written against*. A concrete blocker/matcher/clusterer/embedder/index is imported from the package that owns it (`from langres.core.blockers import AllPairsBlocker`, `from langres.core.matchers import LLMMatcher`, `import langres.core.metrics`, …). Re-exporting an implementation puts `langres.core` *above* the components it sits beneath and re-knots the import graph; `tests/test_import_tangle.py` is the ratchet that measures the cost, and `test_import_budget.py::TestCoreLazyGetattr::test_implementations_are_not_re_exported` fails if one comes back.

**Adding a public symbol?** The two package `__init__.py` files are thin aggregators holding no per-name content: add the export (eager import, or the lazy `name -> module` + `[extra]` entry) to the per-domain fragment that owns its domain under `langres/_exports/` or `langres/core/_exports/` — never to the sorted `__all__` itself — and, for `core`, only if it is a *contract*. A heavy dep must go in `LAZY_SYMBOLS`, never a fragment's module scope: fragments are eagerly imported, so an import there lands in every bare `import langres`. Optuna/wandb/langfuse are dev-only (`[dependency-groups] dev`), for eval tooling, not the production `dedupe()`/`compare()` path. ranx backs the `[eval]` extra but is duplicated in the dev group too (like scikit-learn / `[trained]`), so the repo's own test suite doesn't need `--all-extras` for a bare `uv sync`.

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

**Note on documentation structure**: Keep `CLAUDE.md` concise and actionable. Substantial new guidance (>50 lines) belongs in a focused `.claude/rules/*.md` or `.agent/` doc linked from here, not inline — this keeps the always-on instructions scannable.

## Reference Documentation (`docs/`)

- **`docs/ROADMAP.md`** ⭐ **START HERE / DIRECTION** — the vision: langres as the composable ER seam; the feature-bag architecture; the use-case compass; verifiable milestones. Read before planning new work.
- **`docs/POC.md`** — **archived** original POC validation plan (historical record; outcomes in the root `CHANGELOG.md` and git history).
- **`docs/TECHNICAL_OVERVIEW.md`** — API reference and data contracts (`PairwiseJudgement`, `Candidate`, method signatures, expected inputs/outputs).
- **`docs/USE_CASES.md`** — use-case taxonomy and roadmap (V1 / V1.1 / out-of-scope; streaming, temporal, collective resolution).
- **`docs/DX_RESOLVER.md`** — before/after of the M0 `Resolver`: the manual lambda pipeline vs. the declarative `from_schema` + `save`/`load` path.
- **`docs/EXPERIMENTS.md`** — experimentation DX getting-started: the `run_methods` full-pipeline race vs. `evaluate_judge_on_candidates` (judged-once) for compiled/paid judges; `derive_threshold` to kill magic constants; the `SpendMonitor` budget seam.
- **`docs/BENCHMARKS.md`** — the benchmark portfolio (each dataset + why it's a target + caveats), the `data/registry` discoverability seam (`list_benchmarks` / `get_benchmark`, the `portfolio_race` example), and the `evaluate()` bring-your-own-data pair-scoring walkthrough.
- **`CHANGELOG.md`** (repo root) — release history (0.3.0 / 0.2.0); pre-0.2.0 POC milestone history is preserved in git history and `docs/research/`.
