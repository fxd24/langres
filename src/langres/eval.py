"""
langres.eval: Curated evaluation surface.

Score a judge (:func:`evaluate`), discover benchmarks
(:func:`list_benchmarks` / :func:`get_benchmark`), and reach the entity-resolution
metrics -- all from one import. A thin facade over ``core.benchmark`` /
``core.metrics`` / ``data.registry``: every name below is re-exported unchanged
from where it already lives, so ``langres.eval.evaluate is
langres.core.benchmark.evaluate``. No implementation lives here.

Names resolve lazily (PEP 562 ``__getattr__``, the same idiom as
``langres.core`` / ``langres.clients``): ``from langres.eval import evaluate``
imports only ``core.benchmark`` on first access, and none of these names pulls
the ranking-metric dependency ``ranx`` -- that ``[eval]`` extra is imported only
when :func:`~langres.core.metrics.evaluate_blocking_with_ranking` (MRR/NDCG/MAP)
is actually called. So ``import langres.eval`` stays import-light.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Only reached by mypy -- keeps every lazy name visible to `mypy --strict`
    # without importing the owning modules at runtime.
    from langres.core.benchmark import DEFAULT_PAIR_GRID, evaluate
    from langres.core.metrics import (
        calculate_bcubed_metrics,
        calculate_pairwise_metrics,
        classify_pairs,
        generalized_merge_distance,
        pair_pr_curve,
        reduction_ratio,
    )
    from langres.data.registry import (
        ExternalBenchmarkError,
        get_benchmark,
        list_benchmarks,
    )

__all__ = [
    "calculate_bcubed_metrics",
    "calculate_pairwise_metrics",
    "classify_pairs",
    "DEFAULT_PAIR_GRID",
    "evaluate",
    "ExternalBenchmarkError",
    "generalized_merge_distance",
    "get_benchmark",
    "list_benchmarks",
    "pair_pr_curve",
    "reduction_ratio",
]

#: ``name -> owning module`` for every re-exported symbol, resolved on first
#: access. The value ``__getattr__`` binds is the attribute of the owning module
#: (the same object), never a copy.
_LAZY: dict[str, str] = {
    "evaluate": "langres.core.benchmark",
    "DEFAULT_PAIR_GRID": "langres.core.benchmark",
    "list_benchmarks": "langres.data.registry",
    "get_benchmark": "langres.data.registry",
    "ExternalBenchmarkError": "langres.data.registry",
    "reduction_ratio": "langres.core.metrics",
    "generalized_merge_distance": "langres.core.metrics",
    "classify_pairs": "langres.core.metrics",
    "pair_pr_curve": "langres.core.metrics",
    "calculate_bcubed_metrics": "langres.core.metrics",
    "calculate_pairwise_metrics": "langres.core.metrics",
}


def __getattr__(name: str) -> Any:
    """PEP 562: resolve a re-exported name the first time it's accessed.

    Raises:
        AttributeError: ``name`` isn't part of the curated eval surface.
    """
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value


def __dir__() -> list[str]:
    """List the curated surface so tab-completion / ``dir()`` are discoverable."""
    return sorted(__all__)
