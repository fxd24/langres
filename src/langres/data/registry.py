"""Import-light benchmark registry (a static manifest) + method listing (Wave B).

A discoverability + serialization seam over the dataset loaders: a static
``name -> BenchmarkEntry`` manifest of lightweight *metadata* (task, domain,
loadable, and where the loader lives) with **no loader import at module scope**.
This is deliberate — every dataset loader eagerly pulls the ``[semantic]`` stack
(``VectorBlocker`` / ``SentenceTransformer`` / ``FAISS``), so auto-importing all
loaders would break a core-only install and ``tests/test_import_budget.py``, and
one failing loader would take down the whole registry.

Therefore:

- :func:`list_benchmarks` returns metadata **without importing any loader** (no
  faiss / sentence-transformers pulled into ``sys.modules``).
- :func:`get_benchmark` imports **only the selected** module lazily and returns a
  ready benchmark instance, raising an actionable ``pip install langres[semantic]``
  error if the extra is missing (mirrors ``langres.core.registry``) and a
  ``difflib`` "did you mean" on an unknown name. A ``loadable=False`` entry (an
  external-only dataset that must be fetched manually) raises a clear
  external-only error instead.
- :func:`list_methods` surfaces ``ALL_METHODS`` by reading the import-light
  ``langres._method_names`` leaf (no heavy deps), **not** ``langres.methods``
  (which pulls the heavy stack at module scope) — so listing method names is
  safe in a core-only / partial-extras install.

This module lives in ``langres.data``, is kept **off** ``langres/__init__``'s
eager import path, and is **not** wired into ``langres.core.benchmark`` — that
would close the ``core -> data -> core`` import cycle the harness was built to
avoid. It imports nothing heavier than the stdlib at module scope.
"""

from __future__ import annotations

import difflib
import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from langres.core.benchmark import Benchmark

#: The evaluation task shape. All current entries are cross-source *linkage*
#: benchmarks; ``dedup`` is reserved for single-source dedup datasets.
BenchmarkTask = Literal["linkage", "dedup"]

#: Optional-dependency package -> the extra that ships it, for actionable errors.
_EXTRA_FOR_MODULE: dict[str, str] = {
    "faiss": "semantic",
    "sentence_transformers": "semantic",
    "torch": "semantic",
    "litellm": "llm",
    "dspy": "llm",
    "sklearn": "trained",
    "scikit_learn": "trained",
}


class UnknownBenchmark(KeyError):
    """Raised when a benchmark ``name`` is not in the manifest.

    Carries the available names and a ``difflib`` did-you-mean suggestion so a
    typo produces an actionable error (mirrors ``core.registry.UnknownComponentType``).
    """


class ExternalBenchmarkError(RuntimeError):
    """Raised when :func:`get_benchmark` is asked for a ``loadable=False`` entry.

    An external-only dataset (e.g. a license-restricted corpus that must never be
    vendored) has no importable loader; the message points at where to fetch it.
    """


@dataclass(frozen=True)
class BenchmarkEntry:
    """Lightweight, import-free metadata describing one registered benchmark.

    Attributes:
        name: Stable registry key (e.g. ``"amazon_google"``).
        task: The evaluation task shape (``"linkage"`` / ``"dedup"``).
        domain: Free-text entity domain (e.g. ``"product"``, ``"person"``).
        loadable: Whether the dataset ships in-repo and can be auto-loaded. A
            ``False`` entry is external-only (fetch manually); :func:`get_benchmark`
            raises :class:`ExternalBenchmarkError` for it.
        module_path: Importable module holding the loader (e.g.
            ``"langres.data.amazon_google"``).
        loader_symbol: Attribute in ``module_path`` that is (or builds) the
            benchmark — a ``Benchmark`` **class** (instantiated with ``()``) or an
            already-built instance.
        fetch_hint: For ``loadable=False`` entries, where/how to obtain the data;
            surfaced in the external-only error. ``None`` for loadable entries.
    """

    name: str
    task: BenchmarkTask
    domain: str
    loadable: bool
    module_path: str
    loader_symbol: str
    fetch_hint: str | None = None


#: The static manifest. Populated by :func:`register` at module import; the
#: orchestrator owns these entries centrally (one ``register(...)`` line each) so
#: parallel dataset agents never collide editing a shared dict literal.
_BENCHMARKS: dict[str, BenchmarkEntry] = {}


def register(entry: BenchmarkEntry) -> None:
    """Add ``entry`` to the manifest.

    Args:
        entry: The benchmark metadata to register.

    Raises:
        ValueError: If a benchmark with the same ``name`` is already registered.
    """
    if entry.name in _BENCHMARKS:
        raise ValueError(f"Benchmark {entry.name!r} is already registered")
    _BENCHMARKS[entry.name] = entry


def list_benchmarks() -> list[BenchmarkEntry]:
    """Return every registered benchmark's metadata (import-free), name-sorted.

    Imports **no** loader module: safe in a core-only environment and guaranteed
    not to pull faiss / sentence-transformers into ``sys.modules``.
    """
    return sorted(_BENCHMARKS.values(), key=lambda e: e.name)


def list_methods() -> list[str]:
    """Return the resolution method names the registry can race (``ALL_METHODS``).

    Reads the import-light ``langres._method_names`` leaf (zero heavy deps) — the
    single source of truth these names share with ``langres.methods`` dispatch —
    **not** ``langres.methods`` itself (which pulls faiss / sentence-transformers
    / scikit-learn at module scope). So listing method names is safe in a
    core-only / partial-extras install.
    """
    from langres._method_names import ALL_METHODS

    return list(ALL_METHODS)


def _get_entry(name: str) -> BenchmarkEntry:
    """Look up a manifest entry by name, or raise an actionable error."""
    try:
        return _BENCHMARKS[name]
    except KeyError:
        available = sorted(_BENCHMARKS)
        suggestions = difflib.get_close_matches(name, available, n=3)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise UnknownBenchmark(
            f"Unknown benchmark {name!r}.{hint} "
            f"Available: {', '.join(available) or '(none registered)'}"
        ) from None


def get_benchmark(name: str) -> Benchmark[Any]:
    """Load and return the registered benchmark named ``name``.

    Imports only the selected loader module (lazily), then returns its
    ``loader_symbol`` — instantiating it if it is a class, or returning it as-is if
    it is already a benchmark instance.

    Args:
        name: A registered benchmark name (see :func:`list_benchmarks`).

    Returns:
        A ready benchmark conforming to :class:`~langres.core.benchmark.Benchmark`.

    Raises:
        UnknownBenchmark: If ``name`` is not registered (with a did-you-mean hint).
        ExternalBenchmarkError: If the entry is ``loadable=False`` (external-only).
        ImportError: If the loader needs an optional extra that is not installed
            (with a ``pip install langres[<extra>]`` hint).
    """
    entry = _get_entry(name)
    if not entry.loadable:
        raise ExternalBenchmarkError(
            f"Benchmark {name!r} is external-only and cannot be auto-loaded; "
            f"fetch it manually: {entry.fetch_hint or entry.module_path}"
        )
    try:
        module = importlib.import_module(entry.module_path)
    except ModuleNotFoundError as exc:
        extra = _EXTRA_FOR_MODULE.get(exc.name or "")
        if extra is not None:
            raise ImportError(
                f"Benchmark {name!r} needs the [{extra}] extra "
                f"(missing package {exc.name!r}). Install it: pip install langres[{extra}]"
            ) from exc
        raise
    obj = getattr(module, entry.loader_symbol)
    # ``obj`` is a Benchmark class (instantiate) or an already-built instance.
    return cast("Benchmark[Any]", obj() if isinstance(obj, type) else obj)


# ---------------------------------------------------------------------------
# The static manifest: the existing four vendored benchmarks + the tiny fixture.
# All ship in-repo (loadable=True). External-only entries (e.g. OpenSanctions,
# loadable=False) are added by Wave C using the seam above.
# ---------------------------------------------------------------------------

register(
    BenchmarkEntry(
        name="fodors_zagat",
        task="linkage",
        domain="restaurant",
        loadable=True,
        module_path="langres.data.er_benchmarks",
        loader_symbol="FodorsZagatBenchmark",
    )
)
register(
    BenchmarkEntry(
        name="amazon_google",
        task="linkage",
        domain="product",
        loadable=True,
        module_path="langres.data.amazon_google",
        loader_symbol="AmazonGoogleBenchmark",
    )
)
register(
    BenchmarkEntry(
        name="abt_buy",
        task="linkage",
        domain="product",
        loadable=True,
        module_path="langres.data.abt_buy",
        loader_symbol="AbtBuyBenchmark",
    )
)
register(
    BenchmarkEntry(
        name="febrl_person",
        task="linkage",
        domain="person",
        loadable=True,
        module_path="langres.data.febrl_person",
        loader_symbol="FebrlPersonBenchmark",
    )
)
register(
    BenchmarkEntry(
        name="tiny_fixture",
        task="linkage",
        domain="product",
        loadable=True,
        module_path="langres.data.tiny_fixture",
        loader_symbol="TinyFixtureBenchmark",
    )
)

# ---------------------------------------------------------------------------
# Wave C/D: the four DeepMatcher-style loaders (bibliographic + product) plus
# the external-only OpenSanctions entry (loadable=False; never vendored because
# CC-BY-NC is incompatible with langres's Apache-2.0 license). Its fetch_hint
# points at the source data + the published matcher baselines.
# ---------------------------------------------------------------------------

register(
    BenchmarkEntry(
        name="dblp_acm",
        task="linkage",
        domain="bibliographic",
        loadable=True,
        module_path="langres.data.dblp_acm",
        loader_symbol="DblpAcmBenchmark",
    )
)
register(
    BenchmarkEntry(
        name="dblp_scholar",
        task="linkage",
        domain="bibliographic",
        loadable=True,
        module_path="langres.data.dblp_scholar",
        loader_symbol="DblpScholarBenchmark",
    )
)
register(
    BenchmarkEntry(
        name="walmart_amazon",
        task="linkage",
        domain="product",
        loadable=True,
        module_path="langres.data.walmart_amazon",
        loader_symbol="WalmartAmazonBenchmark",
    )
)
register(
    BenchmarkEntry(
        name="wdc_computers",
        task="linkage",
        domain="product",
        loadable=True,
        module_path="langres.data.wdc_computers",
        loader_symbol="WdcComputersBenchmark",
    )
)
register(
    BenchmarkEntry(
        name="opensanctions",
        task="linkage",
        domain="person/org",
        loadable=False,
        module_path="",
        loader_symbol="",
        fetch_hint=(
            "OpenSanctions is CC-BY-NC 4.0 — incompatible with langres's Apache-2.0 "
            "license, so it is never vendored. Fetch the entity-matching data from "
            "https://www.opensanctions.org/docs/ and compare against the published "
            "matcher baselines (rule-based F1 91.33, GPT-4o 98.95, "
            "DeepSeek-R1-Distill-14B 98.23, Llama-3.1-8B 95.94)."
        ),
    )
)
