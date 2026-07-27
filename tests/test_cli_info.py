"""`langres info` -- the install diagnostic, and the two properties it must hold.

`info` answers the first question a new user hits: *which extras do I actually
have, and which benchmark datasets can I load?* A `pip install langres` gets
neither every extra nor most of the benchmark corpora (they are excluded from
the wheel), and before this command there was no way to find out but to trigger
the failure.

Two properties are load-bearing and both are tested here:

1. **It never prints a secret.** Key presence is reported as a boolean; the value
   is tested for truthiness and discarded. `info` output is exactly the sort of
   thing that ends up pasted into a bug report or a CI log.
2. **It stays import-light.** It must run on a bare core install and must decide
   whether torch/faiss/litellm exist with `importlib.util.find_spec`, never by
   importing them. `tests/test_import_budget.py` guards `import langres`; this
   guards the one command whose whole job is to report on heavy dependencies and
   which would therefore be the natural place to import them all.
"""

from __future__ import annotations

import io
import subprocess
import sys

import pytest

from langres import __version__
from langres.cli import _EXTRA_PROOF_IMPORTS, _PAID_PATH_KEYS, main

#: Every module `langres info` probes for. If `info` ever switches from
#: `find_spec` to `try: import`, these land in `sys.modules` and the
#: import-weight test below fails.
_PROBED_MODULES = sorted({module for _, proofs in _EXTRA_PROOF_IMPORTS for module in proofs})


def _run_info() -> str:
    buffer = io.StringIO()
    assert main(["info"], output_stream=buffer) == 0
    return buffer.getvalue()


def test_info_reports_the_installed_version() -> None:
    assert f"langres {__version__}" in _run_info()


def test_info_reports_every_declared_extra() -> None:
    """Each extra appears with the import(s) that prove it, present or missing."""
    output = _run_info()
    for extra, proofs in _EXTRA_PROOF_IMPORTS:
        assert f"[{extra}]" in output
        # The proving imports are named either as evidence (present) or as what
        # is missing -- a bare yes/no would not tell the user what to check.
        assert any(module in output for module in proofs)


def test_info_names_the_pip_command_for_a_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 'no' must be actionable, not just a verdict."""
    monkeypatch.setattr("langres.cli._module_installed", lambda name: False)
    output = _run_info()
    for extra, _ in _EXTRA_PROOF_IMPORTS:
        assert f"pip install 'langres[{extra}]'" in output


def test_info_reports_benchmark_dataset_loadability() -> None:
    """Datasets are probed by loading them, not read off a static manifest.

    In this repo every bundled corpus is present, so `tiny_fixture` loads; in a
    wheel install most raise `BenchmarkDataNotFoundError`. Either way the answer
    comes from trying, so it cannot drift from the install.
    """
    output = _run_info()
    assert "Benchmark datasets" in output
    assert "tiny_fixture" in output
    # opensanctions is registered `loadable=False` (CC-BY-NC, never vendored).
    assert "ExternalBenchmarkError" in output


def test_info_reports_key_presence_but_never_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The security property: presence is a boolean, the value never surfaces."""
    secret = "sk-canary-must-never-be-printed"
    for key in _PAID_PATH_KEYS:
        monkeypatch.setenv(key, secret)

    output = _run_info()

    assert secret not in output
    for key in _PAID_PATH_KEYS:
        assert key in output
    assert "not set" not in output.split("Paid path")[1]


def test_info_reports_an_absent_key_as_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _PAID_PATH_KEYS:
        monkeypatch.setenv(key, "")
    output = _run_info().split("Paid path")[1]
    for key in _PAID_PATH_KEYS:
        assert f"{key:<20} not set" in output


def test_info_is_listed_in_the_top_level_help() -> None:
    """Discoverability: an undiscoverable diagnostic answers nobody's question."""
    buffer = io.StringIO()
    assert main([], output_stream=buffer) == 0
    assert "info" in buffer.getvalue()


def test_info_imports_no_heavy_dependency_to_decide_it_exists() -> None:
    """`find_spec`, not `try: import` -- proven in a fresh interpreter.

    This test session has almost certainly imported torch already, so the check
    has to happen in a subprocess that has not. Run with `--all-extras` (as CI
    does) every probed module is installed and importable, which is precisely
    when a `try: import` implementation would look correct and cost seconds.
    """
    script = (
        "import io, sys; "
        "from langres.cli import main; "
        "main(['info'], output_stream=io.StringIO()); "
        f"leaked = [m for m in {_PROBED_MODULES!r} if m in sys.modules]; "
        "assert not leaked, f'`langres info` imported {leaked} instead of probing "
        "with importlib.util.find_spec'; "
        "print('OK')"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"`langres info` import-weight check failed.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
