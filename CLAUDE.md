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

**langres** is a Python entity resolution framework in early development. It aims to provide a composable, optimizable approach to entity resolution with a two-layer API (high-level tasks and low-level core components).

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
- `component-design.md` *(`src/**`)* — two-layer API, design principles, lightweight & composable / SRP, common patterns, adding components.
- `testing.md` *(`tests/**`)* — 100% coverage, markers, human-like dev-iteration loop.
- `token-efficiency.md` *(`.claude/agents|skills|commands/**`)* — agent cost discipline (Edit-over-Write, Grep-before-Read, JSON-between-agents, reasoning-tier).

## Skills

- `prompting-claude-4` — expert guidance for prompting Claude 4.x models (XML patterns, behavioral fixes, extended thinking). Use when writing system prompts for the LLM judge / flows, or any agent definition.

## Project Structure

```
langres/
├── src/langres/
│   ├── core/           # Low-level API (Module, Blocker, Optimizer, Clusterer, Canonicalizer)
│   ├── tasks/          # High-level API (DeduplicationTask, EntityLinkingTask)
│   ├── flows/          # Pre-built matching logic (CompanyFlow, ProductFlow)
│   ├── blockers/       # Candidate generation (DedupeBlocker, LinkingBlocker)
│   └── data/           # Synthetic data generation
├── tests/              # Test suite
├── examples/           # Usage examples
└── docs/               # Documentation
```

## Dependencies

**Core stack**: Pydantic (validation), Optuna (hyperparameter optimization), DSPy (prompt optimization), sentence-transformers (embeddings), rapidfuzz (string similarity), networkx (graph clustering), PyTorch (learnable components).

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
- **`docs/PROJECT_OVERVIEW.md`** — architecture and philosophy; the "why" behind design decisions; component relationships; the two-layer API.
- **`docs/TECHNICAL_OVERVIEW.md`** — API reference and data contracts (`PairwiseJudgement`, `Candidate`, method signatures, expected inputs/outputs).
- **`docs/USE_CASES.md`** — use-case taxonomy and roadmap (V1 / V1.1 / out-of-scope; streaming, temporal, collective resolution).
- **`docs/DX_RESOLVER.md`** — before/after of the M0 `Resolver`: the manual lambda pipeline vs. the declarative `from_schema` + `save`/`load` path.
- **`docs/EXPERIMENTS.md`** — experimentation DX getting-started: the `run_methods` full-pipeline race vs. `evaluate_judge_on_candidates` (judged-once) for compiled/paid judges; `derive_threshold` to kill magic constants; the `SpendMonitor` budget seam.
- **`docs/CHANGELOG.md`** — project progress; completed POC milestones.
