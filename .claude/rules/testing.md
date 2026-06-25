---
paths:
  - "src/**"
  - "tests/**"
  - "pyproject.toml"
  - "conftest.py"
---

# Testing & Development Workflow

**100% coverage is a POC requirement. Verify as you go.** Read before writing
tests or running the suite.

## Testing

- **100% test coverage required** - all code must be tested (POC requirement)
- Write tests for all new components in `tests/`
- Use descriptive test names: `test_deduplication_task_with_company_flow`
- Mark slow tests with `@pytest.mark.slow`, integration tests with `@pytest.mark.integration`
- Run tests: `uv run pytest` (pre-push hook runs non-slow, non-integration tests automatically)
- Check coverage: `uv run pytest --cov` to verify 100% is maintained
- Type-check as you go: `uv run mypy src/`

## Development Workflow (Human-Like Iteration)

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
