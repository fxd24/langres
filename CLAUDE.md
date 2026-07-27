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
- `source-map.md` *(`src/**`)* — the annotated per-module map of `src/langres/`: what each file is for, which are import-heavy, which are back-compat shims. Read before changing anything in the package.
- `python-style.md` *(`**/*.py`, `pyproject.toml`)* — type hints, Pydantic-first, `uv`, no `print()`, naming, import weight (the PEP-562 lazy contract).
- `component-design.md` *(`src/**`)* — the layered API (architectures → ERModel → core), design principles, lightweight & composable / SRP, common patterns, adding components (incl. the single judge/method registry, `core/method_registry.py`), and the public-surface rules (`langres.core` re-exports contracts only; where to add an export).
- `testing.md` *(`tests/**`)* — tiered coverage (high on `core`, behavior-focused on harness), markers, human-like dev-iteration loop.
- `token-efficiency.md` *(`.claude/agents|skills|commands/**`)* — agent cost discipline (Edit-over-Write, Grep-before-Read, JSON-between-agents, reasoning-tier).

## Skills

- `prompting-claude-4` — expert guidance for prompting Claude 4.x models (XML patterns, behavioral fixes, extended thinking). Use when writing system prompts for the LLM judge / matching modules, or any agent definition.

## Project Structure

Top-level map. **The annotated per-module detail — what each file is for, and the
traps the paths don't show — lives in `.claude/rules/source-map.md`, which loads
automatically when you touch a file under `src/`.** Read it before changing
anything in the package.

```
src/langres/
├── architectures/  # named ER pipelines (FuzzyString $0/offline, VectorLLMCascade paid)
├── core/           # the primitives + ERModel/Resolver: blocker, comparator, module/judges, clusterer, spend cap, registries
├── resources/      # reusable model resources (Embedder/Reranker/LLM) + their additive Op adapters
├── experiments/    # reproducible experiment matrices: Experiment, EvaluationProtocol, ExperimentReport
├── curation/       # human-in-the-loop labelling + gold-set cold-start
├── training/       # what PRODUCES a tuned matcher (finetune, calibration, prompt-optimize methods)
├── tracking/  metrics/  report/  plotting/  # observability, scoring, $0 rendering — beside core, not in it (one-way deps)
├── autoresearch/ + optimize.py    # the blocking-search engine + its import-light facade
├── benchmarks/  eval.py  data/    # the benchmark harness, the eval facade, the dataset registry
└── clients/  cli.py  methods.py  _exports/
```

**Not built yet** (roadmap — do not reference as existing): `tasks`/`flows`
modules, a general `Optimizer`, a synthetic data generator.

## Dependencies

Exact package lists live in `pyproject.toml` (`dependencies` and
`[project.optional-dependencies]`) — read them there, don't duplicate them here.
What matters is what each tier *buys* you:

- **Core** (`uv sync`) — the string-judge/`AllPairsBlocker` path works with only these.
- **Extras** (`uv sync --all-extras` or `pip install langres[semantic,llm,trained,eval]`):
  `[semantic]` the embedding/vector stack (`VectorBlocker`, embeddings, vector indexes) ·
  `[llm]` the judge stack (`LLMJudge`, DSPy-compiled judges) ·
  `[trained]` the sklearn judge + `langres.training.calibration.derive_threshold` ·
  `[eval]` ranking metrics (MRR/NDCG/MAP). ranx is imported lazily, so the rest of
  `langres.metrics.metrics` / the `langres.benchmarks` harness (BCubed/pairwise
  metrics, `evaluate()`) stays importable without it. (The old
  `core.metrics`/`core.benchmark` paths still work via back-compat shims.)

## Important Notes

- **Always verify claims before you assert them.** Never present an unverified hypothesis — about code, tooling, model/library capabilities, or data — as fact. Check the source, run the code, read the data first; if you can't, label it explicitly as unverified. (Detail in `.claude/rules/expert-knowledge.md`.)
- Focus on the core use cases: Deduplication and Entity Linking (V1 scope)

## Agent Analysis & Expert Feedback (`.agent/`)

The `.agent/` folder contains external expert analyses of the langres project:

- **`.agent/genalysis/20251029_er_use_cases_expert_analysis.md`**: Taxonomy of 18+ entity resolution use cases, mapping each to langres components, identifying gaps (incremental resolution, temporal support, streaming), and comparing langres to state-of-the-art ER systems (Dedupe.io, Splink, Zingg). Essential for understanding production requirements and missing features.
- **`.agent/genalysis/20251029_comprehensive_documentation_evaluation.md`**: Expert evaluation (7.5/10) of architecture, feasibility, critical problems (blocking scalability, DSPy cost, clustering guarantees), and production-readiness gaps.

**When to consult**: before planning new features (check if already identified as a gap); when considering production requirements; when prioritizing work (these docs separate critical from nice-to-have).

**Note on documentation structure**: Keep `CLAUDE.md` concise and actionable. Substantial new guidance (>50 lines) belongs in a focused `.claude/rules/*.md` or `.agent/` doc linked from here, not inline — this keeps the always-on instructions scannable.

## Reference Documentation (`docs/`)

- **`docs/ROADMAP.md`** ⭐ **START HERE / DIRECTION** — the vision: langres as the composable ER seam; the feature-bag architecture; the use-case compass; verifiable milestones. Read before planning new work.
- **`docs/THEORY.md`** — the mathematical foundation, **cited**: blocking/matching/top-k/assignment/clustering are **one operation** (a constrained selection `π` over a scored relation) at different feasible classes `𝓕`; `decomposable` buys precomputation (**not** sublinearity); the staging theorem (blocking recall is a *ceiling*, **not** a thing to maximize — "≥95% recall" is the φ=0 limit); transitive closure = correlation clustering with hard positives and unpriced negatives; Swoosh's ICAR is the theory of merge. **Descriptive of where core is heading, not of what ships** — §11 lists what it contradicts in current code (incl. `blocker.py:38`), §12 is an errata list, and §0 records what is prior art (AJAX / meta-blocking / BFKPT / Swoosh / Dedupalog) vs. ours. Read before changing a core contract.
- **`docs/POC.md`** — **archived** original POC validation plan (historical record; outcomes in the root `CHANGELOG.md` and git history).
- **`docs/TECHNICAL_OVERVIEW.md`** — API reference and data contracts (`PairwiseJudgement`, `Candidate`, method signatures, expected inputs/outputs).
- **`docs/USE_CASES.md`** — use-case taxonomy and roadmap (V1 / V1.1 / out-of-scope; streaming, temporal, collective resolution).
- **`docs/DX_RESOLVER.md`** — before/after of the M0 `Resolver`: the manual lambda pipeline vs. the declarative `from_schema` + `save`/`load` path.
- **`docs/EXPERIMENTS.md`** — experimentation DX getting-started: the `run_methods` full-pipeline race vs. `evaluate_judge_on_candidates` (judged-once) for compiled/paid judges; `derive_threshold` to kill magic constants; the `SpendMonitor` budget seam.
- **`docs/BENCHMARKS.md`** — the benchmark portfolio (each dataset + why it's a target + caveats), the `data/registry` discoverability seam (`list_benchmarks` / `get_benchmark`, the `portfolio_race` example), and the `evaluate()` bring-your-own-data pair-scoring walkthrough.
- **`CHANGELOG.md`** (repo root) — release history (0.3.0 / 0.2.0); pre-0.2.0 POC milestone history is preserved in git history and `docs/research/`.
