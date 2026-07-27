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
import tomllib
from pathlib import Path

import pytest

from langres import __version__
from langres.cli import (
    _EXTRA_PROOF_IMPORTS,
    _EXTRAS_NOT_REPORTED,
    _KEYS_NOT_PAID_PATH,
    _PAID_PATH_KEYS,
    main,
)
from langres.clients.settings import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]

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
    for key, _ in _PAID_PATH_KEYS:
        monkeypatch.setenv(key, secret)

    output = _run_info()

    assert secret not in output
    # Assert the per-key LINES, not the section: the section's footer prose
    # mentions "not set" and would make a substring check pass vacuously.
    for key, _ in _PAID_PATH_KEYS:
        assert f"{key:<20} set" in output


def test_info_reports_an_absent_key_as_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, _ in _PAID_PATH_KEYS:
        monkeypatch.setenv(key, "")
    output = _run_info().split("Paid path")[1]
    for key, _ in _PAID_PATH_KEYS:
        assert f"{key:<20} not set" in output


def test_info_does_not_claim_the_paid_path_sees_exactly_what_it_sees() -> None:
    """`Settings` reads ./.env; litellm's load_dotenv walks UP to a parent.

    An earlier draft claimed the two discovery orders were the same. They are
    not, and the divergence runs in the dangerous direction: "not set" here for
    a key a real call would find and spend with.
    """
    output = _run_info().split("Paid path")[1]
    assert "PARENT directory" in output


def test_info_survives_an_unreadable_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken `.env` is a reason to RUN a diagnostic, not a reason for it to die.

    `Settings()` reads `./.env`; a binary one raises UnicodeDecodeError. Before
    the guard this printed most of the report and then a traceback.
    """
    (tmp_path / ".env").write_bytes(bytes(range(256)) * 4)
    monkeypatch.chdir(tmp_path)

    buffer = io.StringIO()
    assert main(["info"], output_stream=buffer) == 0
    output = buffer.getvalue()

    assert "could not read settings" in output
    # The earlier sections still printed -- a partial answer beats a crash.
    assert f"langres {__version__}" in output


def test_every_declared_extra_is_either_probed_or_deliberately_exempt() -> None:
    """A new extra must be classified, not silently absent from the report.

    The rest of this file iterates `_EXTRA_PROOF_IMPORTS`, so it can only ever
    check extras someone already added -- it regenerates its expectation from
    the thing that would break. This reads pyproject instead.
    """
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set(config["project"]["optional-dependencies"])
    reported = {extra for extra, _ in _EXTRA_PROOF_IMPORTS}

    unclassified = declared - reported - set(_EXTRAS_NOT_REPORTED)
    assert not unclassified, (
        f"pyproject declares extra(s) {sorted(unclassified)} that `langres info` neither "
        "probes nor exempts. Add proof imports to _EXTRA_PROOF_IMPORTS, or an entry with a "
        "reason to _EXTRAS_NOT_REPORTED."
    )
    assert not reported - declared, (
        f"`langres info` reports extra(s) {sorted(reported - declared)} that pyproject does "
        "not declare -- they can never resolve."
    )


def test_every_settings_credential_is_either_reported_or_deliberately_exempt() -> None:
    """A new key on `Settings` must be classified, not silently unreported.

    This is the same shape as the extras test above, for the same reason:
    `_PAID_PATH_KEYS` is a hand-written list, and a hand-written list of
    credentials silently omitted Azure -- so an Azure user with a working key
    read "not set" on every line. Deriving the expectation from `Settings`
    instead of from the list means the omission fails a test rather than
    shipping a misleading diagnostic.
    """
    declared = {name for name in Settings.model_fields if name.endswith("_key")}
    reported = {field for _, field in _PAID_PATH_KEYS}

    unclassified = declared - reported - set(_KEYS_NOT_PAID_PATH)
    assert not unclassified, (
        f"`Settings` declares credential(s) {sorted(unclassified)} that `langres info` neither "
        "reports nor exempts. Add them to _PAID_PATH_KEYS if they can make an inference call "
        "cost money, or to _KEYS_NOT_PAID_PATH with a reason if they cannot."
    )
    assert not reported - declared, (
        f"`langres info` reports credential(s) {sorted(reported - declared)} that `Settings` does "
        "not declare -- they would always read 'not set'."
    )


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
