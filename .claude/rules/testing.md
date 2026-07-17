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
    coming next) → 95–100%.** These are the library contracts users
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
    PR; the **slow** tests + the coverage gates run on **every merge to main
    (push)** + **on demand** (`workflow_dispatch`) via the `test-full` job. So
    per-PR CI does not gate coverage or exercise slow ML paths — mislabeling a
    heavy test as fast slows every PR, and the coverage floors are verified on
    each merge to main, not per-PR. Run the full suite locally
    (`uv run pytest`) before merging a change to ML/embedding code.
- Run tests: `uv run pytest` (pre-push hook runs non-slow, non-integration tests automatically)
- Check coverage: `uv run pytest --cov`; keep `core/**` in the 95–100% tier
  (the repo-wide gate is a relaxed 90% floor — see `pyproject.toml`)
- **Two gates run on `test-full`, and they are not the same number as this
  policy.** The repo-wide floor is 90% (`--cov-fail-under` in `pyproject.toml`).
  The contract additionally has its own path-scoped gate
  (`coverage report --include="src/langres/core/*,src/langres/report/*,src/langres/autoresearch/*,src/langres/optimize.py"
  --precision=2 --fail-under=98` in `.github/workflows/test.yml`). That 98 is a
  **regression ratchet** pinned just under the measured value (98.84% at
  `ba4b1b7`), not the policy — the *target* remains 95–100%. It exists because
  the repo-wide 90% floor sits ~8 points below actual coverage, so the contract
  could be quietly declassified with CI green throughout. Raise it as the real
  number climbs; if it blocks legitimate work, lower it deliberately rather than
  deleting it.
  **When the contract moves to a new package, extend that `--include` glob** or
  the gate silently stops covering it. (`report/`, `autoresearch/` and the
  `optimize.py` facade are in the glob for exactly that reason — they carry
  public surface (`EvalReport`, `optimize()`/`score_blocking`), so letting them
  fall out would un-gate contract code while making the remaining number look
  *better*.)
  **`src/langres/optimize.py` is listed as a file, not folded into a `*` glob**:
  the facade is a module and the engine beside it is the `autoresearch/` package,
  and coverage compiles a trailing `/*` to `optimize[/\\].*` — which matches a
  directory's contents and can never match `optimize.py`. An `optimize/*` entry
  here matches **nothing** and drops the facade silently. (Trailing `/*` *is*
  recursive, so `autoresearch/*` covers the whole engine and `core/*` reaches
  `core/blockers/vector.py`.)
- **Headroom: the glob TOTAL is 98.39% against the 98 floor** (measured
  2026-07-17 on `test-full`, at `0a38e46`, before the optimize/autoresearch
  rename — which moves files without changing a line, so the number carries).
  Note the floor gates the **whole include list**, not `core/*` alone: `core/*`
  by itself measures 98.43%, which is close enough to mislead but is not the
  number the gate compares. The floor was set from a measured 98.84%, so the
  glob has drifted ~0.45pp down and the ratchet has ~0.39pp of slack left. If
  this gate fires on you after a small change, you probably did not break it —
  it has been tightening for a while. Diagnose the drift before lowering the
  floor.
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
