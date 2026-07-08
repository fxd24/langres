"""Tests for the import-light benchmark registry (Wave B)."""

import subprocess
import sys

import pytest

from langres.core.benchmark import Benchmark
from langres.data import registry
from langres.data.registry import (
    BenchmarkEntry,
    ExternalBenchmarkError,
    UnknownBenchmark,
    get_benchmark,
    list_benchmarks,
    list_methods,
    register,
)

#: The four vendored benchmarks + the tiny fixture that this wave registers.
_EXPECTED_NAMES = {"abt_buy", "amazon_google", "febrl_person", "fodors_zagat", "tiny_fixture"}


# --- list_benchmarks: metadata only ---------------------------------------------


def test_list_benchmarks_returns_every_registered_entry() -> None:
    entries = list_benchmarks()
    assert {e.name for e in entries} == _EXPECTED_NAMES
    # All five ship in-repo this wave.
    assert all(e.loadable for e in entries)
    # Name-sorted for determinism.
    assert [e.name for e in entries] == sorted(e.name for e in entries)


def test_benchmark_entries_carry_correct_metadata() -> None:
    by_name = {e.name: e for e in list_benchmarks()}
    assert by_name["amazon_google"].task == "linkage"
    assert by_name["amazon_google"].domain == "product"
    assert by_name["amazon_google"].module_path == "langres.data.amazon_google"
    assert by_name["amazon_google"].loader_symbol == "AmazonGoogleBenchmark"
    assert by_name["febrl_person"].domain == "person"
    assert by_name["fodors_zagat"].domain == "restaurant"
    # No loadable entry carries a fetch hint (that's the external-only seam).
    assert all(e.fetch_hint is None for e in list_benchmarks())


# --- list_methods ---------------------------------------------------------------


def test_list_methods_surfaces_all_methods() -> None:
    from langres.methods import ALL_METHODS

    assert list_methods() == list(ALL_METHODS)


# --- get_benchmark --------------------------------------------------------------


def test_get_benchmark_returns_a_loadable_benchmark_instance() -> None:
    benchmark = get_benchmark("tiny_fixture")
    assert isinstance(benchmark, Benchmark)
    assert benchmark.name == "tiny_fixture"
    corpus, gold_clusters, gold_pairs = benchmark.load()
    assert len(corpus) == 12
    assert len(gold_pairs) == 3


def test_get_benchmark_instantiates_existing_class_entries() -> None:
    # The four vendored benchmarks expose a Benchmark *class*; get_benchmark must
    # instantiate it with ().
    benchmark = get_benchmark("fodors_zagat")
    assert isinstance(benchmark, Benchmark)
    assert benchmark.name == "fodors_zagat"


def test_unknown_name_raises_with_did_you_mean() -> None:
    with pytest.raises(UnknownBenchmark) as exc:
        get_benchmark("amazon-google")  # hyphen typo
    message = str(exc.value)
    assert "Did you mean" in message
    assert "amazon_google" in message


# --- loadable=False external-only seam ------------------------------------------


def test_external_only_entry_raises_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``loadable=False`` entry raises a clear 'fetch manually' error.

    No external-only dataset is registered this wave (Wave C adds OpenSanctions),
    so this exercises the *seam* against a temporarily-registered fake entry.
    ``monkeypatch.setitem`` reverts the manifest after the test.
    """
    fake = BenchmarkEntry(
        name="fake_external",
        task="linkage",
        domain="person",
        loadable=False,
        module_path="langres.data.fake_external",
        loader_symbol="FakeExternalBenchmark",
        fetch_hint="download from https://example.org/data and place in data/external/",
    )
    monkeypatch.setitem(registry._BENCHMARKS, "fake_external", fake)

    with pytest.raises(ExternalBenchmarkError) as exc:
        get_benchmark("fake_external")
    message = str(exc.value)
    assert "external-only" in message
    assert "https://example.org/data" in message  # the fetch hint is surfaced


def test_register_rejects_duplicate_names(monkeypatch: pytest.MonkeyPatch) -> None:
    dup = BenchmarkEntry(
        name="abt_buy",  # already registered
        task="linkage",
        domain="product",
        loadable=True,
        module_path="langres.data.abt_buy",
        loader_symbol="AbtBuyBenchmark",
    )
    with pytest.raises(ValueError, match="already registered"):
        register(dup)


# --- import budget: list_benchmarks must not pull the [semantic] stack ----------


def test_registry_list_benchmarks_stays_import_light() -> None:
    """A subprocess proves importing the registry + list_benchmarks() pulls no faiss.

    Mirrors ``tests/test_import_budget.py``: a fresh interpreter so the check is
    about the registry's own import graph, not this (already-polluted) process.
    Guards the core review-gate requirement — auto-importing loaders would break
    the core-only install and kill every benchmark if one loader failed to import.
    """
    script = (
        "import sys; import langres.data.registry as r; r.list_benchmarks(); "
        "leaked = [m for m in ['faiss', 'sentence_transformers'] if m in sys.modules]; "
        "assert not leaked, f'registry leaked heavy modules: {leaked}'; "
        "print('OK')"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"registry import-budget check failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
