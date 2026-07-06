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

**langres** is a Python entity resolution framework in early development. It aims to provide a composable, optimizable approach to entity resolution with a layered API: user-facing **verbs** (`langres.link` / `langres.dedupe`) over a declarative **`Resolver`** over low-level **`langres.core`** primitives. (Note: there is no `langres.tasks`/`flows` layer — that was earlier doc fiction; see `docs/USE_CASES.md` and `.claude/rules/component-design.md`.)

**Current Stage**: We are at the **initial POC (Proof of Concept) stage**. Before building the full framework, we are validating the core architecture through three progressively sophisticated approaches:
1. Classical string matching (rapidfuzz baseline)
2. Semantic vector search (embedding-based)
3. Hybrid blocking + LLM judge (target architecture)

**📋 See `docs/POC.md` for the complete POC plan** (hypothesis, success criteria — BCubed F1 ≥ 0.85 for Approach 3 — core components, TDD approach, Go/No-Go criteria).

**Current focus**: Building production-quality `langres.core` primitives with 100% test coverage. This is NOT throwaway prototype code—these components will become the foundation of the full library.

## How I Work — Rules (`.claude/rules/`)

These auto-load. **Always-on** rules apply every session; **path-scoped** rules
load only when you read/edit a file matching their `paths:`.

**Always-on:**
- `expert-knowledge.md` — verify-before-asserting, hypotheses ≠ facts, own the failure, stay in scope, **commit before the worktree disappears**, timeouts. The baseline for how to reason and act.
- `data-safety.md` — irreversible-actions guardrail; uncommitted changes are sacred.
- `context-management.md` — delegate output-heavy ops to subagents; parallelize independent work.

**Path-scoped:**
- `python-style.md` *(`**/*.py`, `pyproject.toml`)* — type hints, Pydantic-first, `uv`, no `print()`, naming.
- `component-design.md` *(`src/**`)* — the layered API (verbs → Resolver → core), design principles, lightweight & composable / SRP, common patterns, adding components (incl. the three judge-dispatch sites).
- `testing.md` *(`tests/**`)* — 100% coverage, markers, human-like dev-iteration loop.
- `token-efficiency.md` *(`.claude/agents|skills|commands/**`)* — agent cost discipline (Edit-over-Write, Grep-before-Read, JSON-between-agents, reasoning-tier).

## Skills

- `prompting-claude-4` — expert guidance for prompting Claude 4.x models (XML patterns, behavioral fixes, extended thinking). Use when writing system prompts for the LLM judge / matching modules, or any agent definition.

## Project Structure

```
langres/
├── src/langres/
│   ├── verbs.py        # User-facing verbs: link(), dedupe(), LinkVerdict
│   ├── core/           # Low-level primitives + the Resolver
│   │   ├── resolver.py     # Resolver.from_schema / resolve / save / load
│   │   ├── presets.py      # judge presets ("auto"/string/embedding/zero_shot_llm), spend cap
│   │   ├── blocker.py, blockers/   # AllPairsBlocker, VectorBlocker
│   │   ├── comparator.py           # StringComparator, ComparisonVector
│   │   ├── module.py, modules/, judges/  # Module (judge) ABC + LLMJudge, etc.
│   │   ├── clusterer.py            # Clusterer (transitive closure)
│   │   ├── calibration.py          # derive_threshold
│   │   └── optimizers/             # BlockerOptimizer (Optuna)
│   ├── methods.py      # method registry / _make_module_builder (benchmark path)
│   ├── clients/        # OpenRouter client, SpendMonitor, pricing
│   └── data/           # benchmark dataset loaders (FZ, Amazon-Google, ...)
├── tests/              # Test suite
├── examples/           # Usage examples (quickstart_verbs.py is the offline quickstart)
└── docs/               # Documentation
```

**Not built yet** (roadmap — do not reference as existing): `tasks`/`flows`
modules, a general `Optimizer`, a synthetic data generator.

## Dependencies

**Core** (always installed, `uv sync`): Pydantic + pydantic-settings (validation), rapidfuzz (string similarity), networkx (graph clustering), numpy. The string-judge/`AllPairsBlocker` path works with only these.

**Extras** (opt-in, `uv sync --all-extras` or `pip install langres[semantic,llm,trained]`):
- `[semantic]` — sentence-transformers, torch, faiss-cpu, onnxruntime/optimum, qdrant-client (`VectorBlocker`, embeddings, vector indexes).
- `[llm]` — litellm, dspy-ai, openai (`LLMJudge`, DSPy-compiled judges).
- `[trained]` — scikit-learn (`RFJudge`, the W1.2 trained-family judge, and `core.calibration.derive_threshold`).

These heavy/optional symbols resolve lazily (PEP 562 `__getattr__` in `langres/core/__init__.py` and `langres/clients/__init__.py`) so a bare `import langres` never pulls torch/litellm/faiss/scikit-learn into `sys.modules` — see `tests/test_import_budget.py`. Optuna/wandb/langfuse/ranx are dev-only (`[dependency-groups] dev`), for eval tooling, not the production `link()`/`dedupe()` path (scikit-learn is duplicated in the dev group too, so the repo's own test suite doesn't need `--all-extras` for a bare `uv sync`).

**Dev tools**: ruff (format + lint), pytest + pytest-cov (tests), mypy (strict-mode type checking).

## Important Notes

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

- **`docs/ROADMAP.md`** ⭐ **DIRECTION** — the post-POC vision: langres as the composable ER seam; the feature-bag architecture; the use-case compass; verifiable milestones M0–M6. Read alongside `POC.md` before planning new work.
- **`docs/POC.md`** ⭐ **START HERE** — current development stage and priorities; the three approaches; success criteria; what's in scope NOW vs. later. Read before any implementation work.
- **`docs/TECHNICAL_OVERVIEW.md`** — API reference and data contracts (`PairwiseJudgement`, `Candidate`, method signatures, expected inputs/outputs).
- **`docs/USE_CASES.md`** — use-case taxonomy and roadmap (V1 / V1.1 / out-of-scope; streaming, temporal, collective resolution).
- **`docs/DX_RESOLVER.md`** — before/after of the M0 `Resolver`: the manual lambda pipeline vs. the declarative `from_schema` + `save`/`load` path.
- **`docs/EXPERIMENTS.md`** — experimentation DX getting-started: the `run_methods` full-pipeline race vs. `evaluate_judge_on_candidates` (judged-once) for compiled/paid judges; `derive_threshold` to kill magic constants; the `SpendMonitor` budget seam.
- **`docs/CHANGELOG.md`** — project progress; completed POC milestones.
