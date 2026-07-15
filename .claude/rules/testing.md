---
paths:
  - "src/**"
  - "tests/**"
  - "pyproject.toml"
  - "conftest.py"
---

# Testing & Development Workflow

**Tiered coverage: high on the library contract, behavior-focused on harness
code. Verify as you go.** Read before writing tests or running the suite.

## Testing

- **Tiered coverage** — not a blanket 100% (past-POC, a flat 100% floor just
  manufactures low-value tests):
  - **`src/langres/core/**` and the data-prep contract
    (`src/langres/data/data_profile/**`, and `src/langres/data/mining.py`
    coming next) → 95–100%.** This is the library contract users
    serialize against and depend on (`Resolver.save`/`load`, the judge/blocker
    ABCs, the registry, the data-profile/mining diagnostics). Cover behavior
    *and* edge cases: empty inputs, `None`/MISSING, boundaries, error paths.
  - **Benchmark / experiment / harness code → behavior + smoke tests.** e.g.
    `methods.py`, the `core/benchmark.py` evaluation harness, research
    `examples/` — assert they *work* (happy path + the key edges), not that
    every line is executed.
  - `# pragma: no cover` is fine for genuinely trivial or unreachable lines.
  - The goal is covering behavior and edge cases, not hitting every line for
    its own sake.
- Write tests for all new components in `tests/`
- Use descriptive test names: `test_deduplication_task_with_company_flow`
- Mark slow tests with `@pytest.mark.slow`, integration tests with `@pytest.mark.integration`
  - **Mark any heavy test `@pytest.mark.slow`** (loads embedding/ML models, runs
    torch inference, etc.). CI runs the **fast** subset (`not slow`) on every
    PR; the **slow** tests + the coverage gate run on **every merge to main
    (push)** + **on demand** (`workflow_dispatch`) via the `test-full` job. So
    per-PR CI does not gate coverage or exercise slow ML paths — mislabeling a
    heavy test as fast slows every PR, and the coverage floor is verified on
    each merge to main, not per-PR. Run the full suite locally
    (`uv run pytest`) before merging a change to ML/embedding code.
- Run tests: `uv run pytest` (pre-push hook runs non-slow, non-integration tests automatically)
- Check coverage: `uv run pytest --cov`; keep `core/**` in the 95–100% tier
  (the repo-wide gate is a relaxed 90% floor — see `pyproject.toml`)
- Type-check as you go: `uv run mypy src/`

## Development Workflow (Human-Like Iteration)

**Work iteratively like a human developer would:**

1. **Verify as you go**: After writing a function, immediately run it to check it works
2. **Test-first when appropriate**: If starting with tests (TDD), run them to see failures, then implement
3. **Validate data contracts**: Print/inspect input and output data to ensure correct structure
4. **Run type checking**: Use `uv run mypy src/` to catch type errors early
5. **Check coverage**: Run `uv run pytest --cov` — keep the `core/**` tier at 95–100%
6. **Incremental verification**: Don't write large blocks without testing - validate each step
7. **Use the REPL/debugger**: When uncertain about behavior, test in isolation first
8. **Read error messages carefully**: They often contain the exact fix needed

**Example workflow**:
- Write function → Run it with sample data → Fix errors → Add tests → Run tests → Check types → Check coverage → Commit

This iterative approach catches issues early and ensures code works as expected before moving forward.
