# Contributing to langres

Thanks for your interest! langres is in early, fast-moving development
(pre-1.0), so the best first step for anything non-trivial is to **open an
issue** and align on the approach before writing code.

## Dev setup

Requirements: Python >= 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/fxd24/langres
cd langres

# Core-only environment (string-judge path; pytest et al. come with the dev group):
uv sync

# Everything, including the [semantic]/[llm]/[trained]/[eval] extras
# (needed to run the full test suite and mypy the way CI does):
uv sync --all-extras

# Install the git hooks (ruff + hygiene on commit, fast tests on push).
# prek is a Rust single-binary drop-in for pre-commit; it reads the same
# .pre-commit-config.yaml:
uvx prek install && uvx prek install --hook-type pre-push
```

Notes:

- The **benchmark corpora** under `src/langres/data/datasets/` ship in the git
  repository but are excluded from the PyPI wheel — a checkout like the above
  is the supported way to work with them.
- Some integration tests talk to paid LLM APIs. They are opt-in (`-m
  integration`) and **skipped by default** — never required for a PR. Never
  commit API keys; be aware that a loaded `OPENROUTER_API_KEY` can make
  integration tests spend real money.

## Running tests

The suite is tiered (see `.claude/rules/testing.md` for the full policy):

```bash
# Fast subset — what per-PR CI runs and the pre-push hook enforces:
uv run pytest -m "not slow and not integration" --no-cov

# Full suite incl. slow ML tests + the coverage gate (weekly CI job):
uv run pytest

# Lint, format, types (all must pass; mypy runs in strict mode):
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
```

Coverage is tiered: **95–100 % on `src/langres/core/**`** (the library
contract), behavior/smoke tests on benchmark/experiment harness code, with a
repo-wide 90 % floor gated by the weekly `test-full` job — not per PR. Mark
heavy tests (embedding models, torch inference) `@pytest.mark.slow` and tests
needing external credentials `@pytest.mark.integration`.

## Pull request expectations

- **Small, surgical diffs.** Every changed line should trace to the task;
  don't reformat or "improve" adjacent code.
- **Tests with the change.** New components need tests in `tests/`; bug fixes
  need a test that reproduces the bug.
- **Docs in the same PR.** If a change touches behavior, paths, commands, or
  data contracts described in `README.md`, `docs/`, or `CHANGELOG.md`, update
  them in the same change.
- **Type hints + Pydantic-first** (`.claude/rules/python-style.md`): built-in
  generics (`list`, `dict`), `mypy --strict` clean, no `print()` outside
  `examples/`.
- **Commit style:** conventional-commit-ish subjects (`feat(data): …`,
  `fix(eval): …`), as in `git log`.
- CI runs the fast suite, lint, and mypy on every PR; an automated code review
  bot also comments. Address warranted findings or say why not.

## Where to start

- `README.md`, then `docs/GETTING_STARTED.md` — what langres is and the
  intended flywheel loop.
- `docs/POC.md` and `docs/ROADMAP.md` — current stage and direction.
- `CLAUDE.md` + `.claude/rules/` — the working conventions this repo is
  actually built with (they apply to humans too).
