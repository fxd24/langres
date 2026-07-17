"""Tests for the import-light benchmark registry (Wave B)."""

import subprocess
import sys

import pytest

from langres.data.benchmark import Benchmark
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

#: Every benchmark that ships loadable in-repo data: the five originals (Wave B)
#: plus the four DeepMatcher loaders wired in by Wave C/D.
_LOADABLE_NAMES = {
    "abt_buy",
    "amazon_google",
    "dblp_acm",
    "dblp_scholar",
    "febrl_person",
    "fodors_zagat",
    "tiny_fixture",
    "walmart_amazon",
    "wdc_computers",
}
#: External-only entries (loadable=False; fetched manually, never vendored).
_EXTERNAL_NAMES = {"opensanctions"}
#: The full manifest = loadable + external-only.
_EXPECTED_NAMES = _LOADABLE_NAMES | _EXTERNAL_NAMES


# --- list_benchmarks: metadata only ---------------------------------------------


def test_list_benchmarks_returns_every_registered_entry() -> None:
    entries = list_benchmarks()
    assert {e.name for e in entries} == _EXPECTED_NAMES
    # Loadable flags: every in-repo dataset is loadable; the external-only entry
    # (OpenSanctions) is not.
    by_name = {e.name: e for e in entries}
    assert {n for n, e in by_name.items() if e.loadable} == _LOADABLE_NAMES
    assert {n for n, e in by_name.items() if not e.loadable} == _EXTERNAL_NAMES
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
    # The Wave C/D loaders carry their own task/domain metadata.
    assert by_name["dblp_acm"].domain == "bibliographic"
    assert by_name["dblp_scholar"].loader_symbol == "DblpScholarBenchmark"
    assert by_name["walmart_amazon"].module_path == "langres.data.walmart_amazon"
    assert by_name["wdc_computers"].domain == "product"
    # No loadable entry carries a fetch hint; only the external-only seam does.
    assert all(e.fetch_hint is None for e in list_benchmarks() if e.loadable)
    assert by_name["opensanctions"].fetch_hint is not None


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


@pytest.mark.parametrize("name", ["dblp_acm", "dblp_scholar", "walmart_amazon", "wdc_computers"])
def test_get_benchmark_resolves_the_new_wave_c_loaders(name: str) -> None:
    """Each new loadable entry imports + instantiates to a ``Benchmark``.

    Only imports the loader module and builds the (cheap) benchmark instance —
    it deliberately does NOT call ``.load()`` (DBLP-Scholar is a 66879-record,
    ~8MB corpus). The per-dataset contract tests already exercise ``.load()``.
    """
    benchmark = get_benchmark(name)
    assert isinstance(benchmark, Benchmark)
    assert benchmark.name == name
    assert callable(benchmark.load)  # present, but intentionally not invoked here.


def test_get_benchmark_opensanctions_is_external_only() -> None:
    """The registered ``loadable=False`` OpenSanctions entry raises with its hint."""
    with pytest.raises(ExternalBenchmarkError) as exc:
        get_benchmark("opensanctions")
    message = str(exc.value)
    assert "external-only" in message
    assert "opensanctions.org" in message  # the fetch hint is surfaced


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


def test_missing_semantic_extra_raises_pip_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """A loader that fails to import faiss surfaces a 'pip install langres[semantic]' hint."""

    def _fail(_module_path: str) -> object:
        raise ModuleNotFoundError("No module named 'faiss'", name="faiss")

    monkeypatch.setattr(registry.importlib, "import_module", _fail)
    with pytest.raises(ImportError, match=r"pip install langres\[semantic\]"):
        get_benchmark("abt_buy")


def test_unrelated_import_error_is_reraised(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ModuleNotFoundError unrelated to an extra propagates unchanged (no false hint)."""

    def _fail(_module_path: str) -> object:
        raise ModuleNotFoundError("No module named 'nope'", name="nope")

    monkeypatch.setattr(registry.importlib, "import_module", _fail)
    with pytest.raises(ModuleNotFoundError, match="nope"):
        get_benchmark("abt_buy")


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
