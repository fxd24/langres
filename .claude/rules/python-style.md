---
paths:
  - "**/*.py"
  - "pyproject.toml"
---

# Python Style & Conventions

**Comprehensive type hints, Pydantic-first, `uv`-managed, no `print()` in source.**
Read before writing or editing Python in this repo.

## Python Guidelines

- **Python Version**: Requires Python >=3.12
- **Code Formatting**: Use `ruff` for code formatting and linting
- **Type Hints**: Use comprehensive type hints throughout. Use built-in types (`list`, `dict`, `str`, etc.) instead of `typing.List`, `typing.Dict`, etc. (Python 3.12+ feature)
- **Type Checking**: Use `mypy` in strict mode - all code must pass type checking
- **Validation**: Pydantic-first approach - all data models should use Pydantic
- **Logging**: ALWAYS use the `logging` module instead of `print()` statements in source code and tests. Print statements are ONLY acceptable in `examples/` directory for demonstration purposes. Ruff's T201 rule enforces this.
- **Package Manager**: Use `uv add` for dependencies (runtime), `uv add --dev` for dev dependencies. Never manually edit `pyproject.toml` for dependencies. See [uv docs](https://docs.astral.sh/uv/) for details. **Exception:** tables `uv` does not manage — notably `[project.scripts]` (console entry points) — are hand-edited by necessity; mark them with a comment (see the `langres` entry point in `pyproject.toml`).
- **Test Coverage**: 100% coverage required (POC requirement). See `[tool.coverage.*]` in pyproject.toml for configuration.

## Python Execution & File Management

- **Python Execution**: ALWAYS use `uv run python` (not system `python` or `python3`) to ensure code runs in the project's virtual environment with correct dependencies
- **Temporary Scripts**: When creating temporary test scripts or scratch files, place them in the repo's `tmp/` directory (which is gitignored), NOT in the system `/tmp` directory. This keeps temporary work organized and prevents polluting the system temp folder.
  - Example: Create scripts in `<repo-root>/tmp/test_script.py` instead of `/tmp/test_script.py`
- **Environment Variables**: The `.env` file contains environment configuration (including OpenMP settings for macOS). Use `uv run --env-file .env` for commands that need these settings. See `docs/FRICTION_LOG.md` for known issues and remedies.

## Naming Conventions

- **Classes**: PascalCase (e.g., `DeduplicationTask`, `CompanyFlow`)
- **Functions/Methods**: snake_case (e.g., `generate_candidates`, `compile`)
- **Private Methods**: Prefix with underscore (e.g., `_internal_method`)
- **Constants**: UPPER_SNAKE_CASE
