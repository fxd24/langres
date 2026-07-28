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
    `methods.py`, the `langres/benchmarks/` evaluation harness
    (`runner.py` / `judge_eval.py`), research
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
    PR; the **slow** tests run on **every merge to main (push)** + **on demand**
    (`workflow_dispatch`) via the `test-full` job. So per-PR CI does not exercise
    slow ML paths — mislabeling a heavy test as fast slows every PR. Run the full
    suite locally (`uv run pytest`) before merging a change to ML/embedding code.
  - **Coverage is gated in both places, but not on the same numbers** — see the
    two-gate table below. The *contract* floor runs on **every PR** (early
    warning, 97.5) and authoritatively on **each merge** (98). The *repo-wide*
    90% floor is enforced only on `test-full`, because the fast suite omits the
    slow tests that are the only cover for some ML paths.
- Run tests: `uv run pytest` (pre-push hook runs non-slow, non-integration tests automatically)
- Check coverage: `uv run pytest --cov`; keep `core/**` in the 95–100% tier
  (the repo-wide gate is a relaxed 90% floor — see `pyproject.toml`)
- **The contract coverage gate runs in TWO places, at two floors, over ONE
  include list.** The repo-wide floor is 90% (`--cov-fail-under` in
  `pyproject.toml`). The contract additionally has its own path-scoped gate:

  | where | suite | floor | role |
  |---|---|---|---|
  | `test` job (**every PR**) | fast (`not slow`) | **97.5** | early warning, ~12s |
  | `test-full` job (**merge to main**) | full incl. slow | **98** | authoritative |

  **Do not copy the `--include` list into this file or anywhere else.** It is
  defined once, as `env.CONTRACT_COVERAGE_INCLUDE` at the top of
  `.github/workflows/test.yml`, and both gates read it. A list duplicated in two
  places drifts, and the copy that drifts silently stops gating what it names —
  the paragraph that used to live here had already gone stale, omitting
  `metrics/*`, `tracking/*` and the four `curation/` contract files. Read the
  `env` block for the maintenance rules (why `optimize.py` is a FILE and not an
  `optimize/*` glob; why the curation modules are listed individually; why the
  glob follows the contract, not the average).

  That 98 is a **regression ratchet** pinned just under a measured value, not the
  policy — the *target* remains 95–100%. It exists because the repo-wide 90% floor
  sits ~8 points below actual coverage, so the contract could be quietly
  declassified with CI green throughout. Raise it as the real number climbs; if it
  blocks legitimate work, lower it deliberately rather than deleting it.
- **Why two floors is not the "two numbers drift" disease.** The thing that rots
  is the include *list*, and there is only one of it. The floors differ because the
  two gates measure genuinely different data: the per-PR job excludes the slow ML
  tests, so paths covered only by them read as uncovered there.
  The lower floor is **structural, not tuned**: the per-PR selection
  (`not integration and not slow and not finetune`) differs from `test-full`'s
  (`not integration and not finetune`) *only* by `not slow`, so its tests are a
  strict **subset**. Coverage is monotone in tests executed and both gates score
  the same include list (same denominator), so **full-suite coverage ≥ fast-suite
  coverage, always** — the per-PR gate can never demand what the authoritative one
  does not already meet. The observed gap is small, which is why 97.5 is close
  rather than slack: measured at `1b24a2a`, 2026-07-28, full suite **98.26%** vs
  fast suite **97.93%** — **0.33pp**.
- **Timings, measured 2026-07-28** (the previous "~10min" claim was stale by 4x and
  was the justification for keeping the contract gate off PRs entirely):
  `test-full` **~38–41 min**, the per-PR `test` job **~6 min**, the gate step
  itself **~12 s**. The cost is the suite, never the gate.
- **If the contract gate fires on you, diagnose the drift before touching the
  floor.** It has real precedent: `main` was red for six days across seven merges
  because #229 took the TOTAL 98.28% → 97.27% in a single commit and `test-full`
  does not run on PRs, so nothing observed it until after the merge.
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
