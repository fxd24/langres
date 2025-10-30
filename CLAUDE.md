# Claude Code Guidelines for langres

## Project Overview

**langres** is a Python entity resolution framework in early development. It aims to provide a composable, optimizable approach to entity resolution with a two-layer API (high-level tasks and low-level core components).

**Current Stage**: We are at the **initial POC (Proof of Concept) stage**. Before building the full framework, we are validating the core architecture through three progressively sophisticated approaches:
1. Classical string matching (rapidfuzz baseline)
2. Semantic vector search (embedding-based)
3. Hybrid blocking + LLM judge (target architecture)

**📋 See `docs/POC.md` for the complete POC plan**, including:
- The hypothesis we're validating
- Success criteria (BCubed F1 ≥ 0.85 for Approach 3)
- Core components to build (`Module`, `Blocker`, `Clusterer`)
- TDD development approach
- Go/No-Go decision criteria

**Current focus**: Building production-quality `langres.core` primitives with 100% test coverage. This is NOT throwaway prototype code—these components will become the foundation of the full library.

## Code Style & Standards

### Python Guidelines

- **Python Version**: Requires Python >=3.12
- **Code Formatting**: Use `ruff` for code formatting and linting
- **Type Hints**: Use comprehensive type hints throughout. Use built-in types (`list`, `dict`, `str`, etc.) instead of `typing.List`, `typing.Dict`, etc. (Python 3.12+ feature)
- **Type Checking**: Use `mypy` in strict mode - all code must pass type checking
- **Validation**: Pydantic-first approach - all data models should use Pydantic
- **Logging**: ALWAYS use the `logging` module instead of `print()` statements in source code and tests. Print statements are ONLY acceptable in `examples/` directory for demonstration purposes. Ruff's T201 rule enforces this.
- **Package Manager**: Use `uv add` for dependencies (runtime), `uv add --dev` for dev dependencies. Never manually edit `pyproject.toml`. See [uv docs](https://docs.astral.sh/uv/) for details.
- **Test Coverage**: 100% coverage required (POC requirement). See `[tool.coverage.*]` in pyproject.toml for configuration.

### Python Execution & File Management

- **Python Execution**: ALWAYS use `uv run python` (not system `python` or `python3`) to ensure code runs in the project's virtual environment with correct dependencies
- **Temporary Scripts**: When creating temporary test scripts or scratch files, place them in the repo's `tmp/` directory (which is gitignored), NOT in the system `/tmp` directory. This keeps temporary work organized and prevents polluting the system temp folder.
  - Example: Create scripts in `/Users/davidgraf/work/langres/tmp/test_script.py` instead of `/tmp/test_script.py`

### Naming Conventions

- **Classes**: PascalCase (e.g., `DeduplicationTask`, `CompanyFlow`)
- **Functions/Methods**: snake_case (e.g., `generate_candidates`, `compile`)
- **Private Methods**: Prefix with underscore (e.g., `_internal_method`)
- **Constants**: UPPER_SNAKE_CASE

### Project Structure

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

## Architecture Principles

### The Two-Layer API

1. **High-Level (`langres.tasks`)**: Pre-built task runners for common use cases
   - Target: 80% of users
   - Examples: `DeduplicationTask`, `EntityLinkingTask`
   - Philosophy: Like scikit-learn's Pipeline

2. **Low-Level (`langres.core`)**: Composable primitives for custom pipelines
   - Target: 20% of users (advanced use cases)
   - Components: `Module`, `Blocker`, `Optimizer`, `Clusterer`, `Canonicalizer`
   - Philosophy: Like PyTorch's primitives

### Key Design Principles

- **Pydantic-First**: All data models use Pydantic for validation
- **Full Observability**: Every `PairwiseJudgement` carries provenance and reasoning
- **Composable**: Components should be reusable across different tasks
- **Optimizable**: Support both hyperparameter tuning (Optuna) and prompt optimization (DSPy)
- **Cost-Aware**: Consider API costs, computation costs, and optimization budgets

## Dependencies

### Core Stack

- **Pydantic**: Data validation and schema management
- **Optuna**: Hyperparameter optimization
- **DSPy**: Prompt optimization for LLM-based matchers
- **sentence-transformers**: Semantic embeddings
- **rapidfuzz**: String similarity metrics
- **networkx**: Graph clustering algorithms
- **PyTorch**: Deep learning and learnable components

### Development Tools

- **ruff**: Code formatting and linting
- **pytest**: Testing framework with pytest-cov
- **mypy**: Static type checking (strict mode)

## Implementation Guidelines

### When Adding New Components

1. **Blockers**: Must implement candidate generation and schema normalization
2. **Flows (Modules)**: Must yield `PairwiseJudgement` objects
3. **Tasks**: Should compose Blocker + Flow + optional Optimizer
4. **All Components**: Should support both `.run()` and `.compile()` methods where appropriate

### Testing

- **100% test coverage required** - all code must be tested (POC requirement)
- Write tests for all new components in `tests/`
- Use descriptive test names: `test_deduplication_task_with_company_flow`
- Mark slow tests with `@pytest.mark.slow`, integration tests with `@pytest.mark.integration`
- Run tests: `uv run pytest` (pre-push hook runs non-slow, non-integration tests automatically)

### Development Workflow (Human-Like Iteration)

**Work iteratively like a human developer would:**

1. **Verify as you go**: After writing a function, immediately run it to check it works
2. **Test-first when appropriate**: If starting with tests (TDD), run them to see failures, then implement
3. **Validate data contracts**: Print/inspect input and output data to ensure correct structure
4. **Run type checking**: Use `uv run mypy src/` to catch type errors early
5. **Check coverage**: Run `uv run pytest --cov` to verify 100% coverage is maintained
6. **Incremental verification**: Don't write large blocks without testing - validate each step
7. **Use the REPL/debugger**: When uncertain about behavior, test in isolation first
8. **Read error messages carefully**: They often contain the exact fix needed

**Example workflow**:
- Write function → Run it with sample data → Fix errors → Add tests → Run tests → Check types → Check coverage → Commit

This iterative approach catches issues early and ensures code works as expected before moving forward.

### Documentation

- Update relevant docs in `docs/` when changing architecture
- Add docstrings to all public methods
- Include usage examples for new components in `examples/`

## Common Patterns

### Task Implementation

```python
class SomeTask:
    def __init__(self, flow: Module, blocker: Blocker):
        self.flow = flow
        self.blocker = blocker

    def compile(self, gold_data, metric: str):
        """Optimize hyperparameters on gold data"""
        pass

    def run(self, data):
        """Execute the task on input data"""
        pass
```

### Flow (Module) Implementation

```python
class SomeFlow(Module):
    def forward(self, candidates):
        """Yield PairwiseJudgement for each candidate pair"""
        for pair in candidates:
            score = self._compute_similarity(pair)
            yield PairwiseJudgement(
                left_id=pair.left.id,
                right_id=pair.right.id,
                score=score,
                score_type="calibrated_prob"
            )
```

## Important Notes

- This is an **early-stage project** - expect significant changes
- Prioritize clean, testable code over premature optimization
- Document design decisions in code comments
- Focus on the core use cases: Deduplication and Entity Linking (V1 scope)

## Agent Analysis & Expert Feedback

The `.agent/` folder contains external expert analyses of the langres project:

- **`.agent/genalysis/20251029_er_use_cases_expert_analysis.md`**: Comprehensive taxonomy of 18+ entity resolution use cases, mapping each to langres components, identifying gaps (incremental resolution, temporal support, streaming), and comparing langres to state-of-the-art ER systems (Dedupe.io, Splink, Zingg). Essential reading for understanding production requirements and missing features.

- **`.agent/genalysis/20251029_comprehensive_documentation_evaluation.md`**: Expert evaluation (7.5/10) of the project architecture, feasibility analysis, critical problems (blocking scalability, DSPy cost implications, clustering guarantees), and production readiness gaps. Includes practical recommendations for performance benchmarks, cost models, and quality assurance.

**When to consult**:
- Before planning new features to check if they're already identified as gaps
- When considering production deployment requirements
- To understand real-world challenges and best practices in entity resolution
- When prioritizing development work (these docs identify critical vs. nice-to-have features)

**Note on documentation structure**: Keep `CLAUDE.md` concise and actionable. When adding substantial new guidance (>50 lines), consider creating a focused document in `.agent/` instead and linking to it here. This keeps the main instructions scannable while preserving detailed context.

## Reference Documentation

When working on langres, consult these comprehensive docs for detailed context:

### `docs/CHANGELOG.md` - Project Progress
**When to use**: Understanding what's been built so far
- To see the progression from architecture to implementation
- Quick overview of completed POC milestones

### `docs/POC.md` - Proof of Concept Plan ⭐ **START HERE**
**When to use**: Understanding the current development stage and priorities
- **Before starting any implementation work** - this defines our current focus
- When deciding what to build next (we're doing core primitives, not tasks layer)
- To understand the three approaches we're validating (classical, semantic, LLM hybrid)
- When clarifying success criteria (BCubed F1 targets, TDD requirements)
- To see what's in scope NOW vs. what's planned for later (no Optimizer, no tasks yet)
- When evaluating architectural decisions (does this fit the POC validation goals?)

### `docs/PROJECT_OVERVIEW.md` - Architecture and Philosophy
**When to use**: Understanding the "why" behind design decisions
- Before proposing major architectural changes
- When clarifying the relationship between components (Blocker, Flow, Optimizer, etc.)
- To understand the philosophy behind the two-layer API
- When explaining langres to users or in documentation

### `docs/TECHNICAL_OVERVIEW.md` - API Reference and Data Contracts
**When to use**: Implementation details and type signatures
- When implementing new components (Blocker, Flow, Task, etc.)
- To understand data contracts (what `PairwiseJudgement`, `Candidate`, etc. should look like)
- When defining method signatures for consistency
- To see complete API examples and expected inputs/outputs

### `docs/USE_CASES.md` - Use Case Taxonomy and Roadmap
**When to use**: Understanding scope and future direction
- When evaluating whether a feature request is in scope
- To understand which use cases are V1 vs. V1.1 vs. out-of-scope
- When someone asks "can langres do X?" (streaming, temporal, collective resolution, etc.)
- To see the formal taxonomy of supported entity resolution patterns
