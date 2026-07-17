"""
langres.eval: Curated evaluation surface.

Score a judge (:func:`evaluate`), discover benchmarks
(:func:`list_benchmarks` / :func:`get_benchmark`), and reach the entity-resolution
metrics -- all from one import. Most names below are re-exported unchanged from
where they already live (``core.benchmark`` / ``core.metrics`` / ``data.registry``),
so e.g. ``langres.eval.evaluate is langres.core.benchmark.evaluate``. The one
exception is :func:`candidates_for`, implemented here: it blocks one registered
benchmark's split into the ``(candidates, gold_pairs)`` pair :func:`evaluate`
needs, so scoring a judge on a benchmark never requires reaching into
``Resolver``'s private ``_candidates`` generator.

Names resolve lazily (PEP 562 ``__getattr__``, the same idiom as
``langres.core`` / ``langres.clients``): ``from langres.eval import evaluate``
imports only ``core.benchmark`` on first access, and none of these names pulls
the ranking-metric dependency ``ranx`` -- that ``[eval]`` extra is imported only
when :func:`~langres.core.metrics.evaluate_blocking_with_ranking` (MRR/NDCG/MAP)
is actually called. :func:`candidates_for` keeps the same discipline: its
imports (``Resolver``, ``gold_pairs_from_clusters``) are local to the function
body, so ``import langres.eval`` stays import-light.
"""

import importlib
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    # Only reached by mypy -- keeps every lazy name visible to `mypy --strict`
    # without importing the owning modules at runtime.
    from langres.benchmarks.judge_eval import evaluate
    from langres.data.benchmark import (
        DEFAULT_PAIR_GRID,
        Benchmark,
        gold_pairs_from_clusters,
    )
    from langres.data.data_profile import DataProfileReport, from_embedder
    from langres.report.eval_report import EvalReport
    from langres.core.metrics import (
        average_precision_score,
        calculate_bcubed_metrics,
        calculate_pairwise_metrics,
        classify_pairs,
        generalized_merge_distance,
        pair_pr_curve,
        reduction_ratio,
        roc_auc_score,
    )
    from langres.core.models import ERCandidate
    from langres.data.registry import (
        ExternalBenchmarkError,
        get_benchmark,
        list_benchmarks,
    )

__all__ = [
    "average_precision_score",
    "calculate_bcubed_metrics",
    "calculate_pairwise_metrics",
    "candidates_for",
    "classify_pairs",
    "DataProfileReport",
    "DEFAULT_PAIR_GRID",
    "evaluate",
    "EvalReport",
    "ExternalBenchmarkError",
    "from_embedder",
    "generalized_merge_distance",
    "get_benchmark",
    "gold_pairs_from_clusters",
    "list_benchmarks",
    "pair_pr_curve",
    "reduction_ratio",
    "roc_auc_score",
]

#: ``name -> owning module`` for every re-exported symbol, resolved on first
#: access. The value ``__getattr__`` binds is the attribute of the owning module
#: (the same object), never a copy. ``candidates_for`` is NOT here -- it is a
#: real function defined below, not a re-export.
_LAZY: dict[str, str] = {
    "evaluate": "langres.benchmarks.judge_eval",
    "EvalReport": "langres.report.eval_report",
    "DataProfileReport": "langres.data.data_profile",
    "from_embedder": "langres.data.data_profile",
    "DEFAULT_PAIR_GRID": "langres.data.benchmark",
    "gold_pairs_from_clusters": "langres.data.benchmark",
    "list_benchmarks": "langres.data.registry",
    "get_benchmark": "langres.data.registry",
    "ExternalBenchmarkError": "langres.data.registry",
    "reduction_ratio": "langres.core.metrics",
    "generalized_merge_distance": "langres.core.metrics",
    "classify_pairs": "langres.core.metrics",
    "pair_pr_curve": "langres.core.metrics",
    "calculate_bcubed_metrics": "langres.core.metrics",
    "calculate_pairwise_metrics": "langres.core.metrics",
    "roc_auc_score": "langres.core.metrics",
    "average_precision_score": "langres.core.metrics",
}


# ---------------------------------------------------------------------------
# candidates_for: the one function implemented directly in this facade (every
# other name above is a re-export) -- see the module docstring for why.
# ---------------------------------------------------------------------------


def candidates_for(
    bench: "Benchmark[Any]",
    *,
    split: Literal["train", "test"] = "test",
    seed: int = 0,
) -> "tuple[list[ERCandidate[Any]], set[frozenset[str]]]":
    """Block one benchmark split into judge-ready candidates, plus its gold pairs.

    The seam :func:`evaluate` needs: ``candidates, gold = candidates_for(bench,
    split="test")`` then ``evaluate(my_judge, candidates, gold)`` -- without it,
    scoring a judge on a registered benchmark forces a caller to reach into
    ``bench.load()``/``bench.split()`` themselves and reconstruct a
    :class:`~langres.core.resolver.Resolver` just to block the split.

    Loads ``bench``, splits it (leakage-free, via ``bench.split``), and blocks
    the chosen split through the dataset's OWN pinned blocker --
    ``bench.build_blocker(bench.blocking_k)``, the same blocking config a
    method-registry race uses, never a naive all-pairs scan -- via
    :meth:`Resolver.candidates() <langres.core.resolver.Resolver.candidates>`,
    so comparison vectors are attached exactly like that method does. Gold
    pairs are derived from the chosen split's OWN gold clusters via
    :func:`~langres.core.benchmark.gold_pairs_from_clusters` (leakage-free:
    only that split's clusters contribute, never the full dataset's).

    ``bench`` must additionally satisfy ``langres.methods.BlockingBenchmark``
    (``schema`` + ``blocking_k`` + ``build_blocker``) on top of the core
    :class:`~langres.core.benchmark.Benchmark` contract this signature
    declares -- every dataset returned by :func:`get_benchmark` does. Building
    the blocker may require whatever extra ``bench.build_blocker`` itself
    needs (typically ``[semantic]`` for the vendored datasets, which all block
    with a ``VectorBlocker``).

    Args:
        bench: A benchmark from :func:`get_benchmark` (or any conformer).
        split: Which split's records to block -- ``"train"`` or ``"test"``
            (default ``"test"``, the common evaluation split).
        seed: Split seed forwarded to ``bench.split`` (default ``0``).

    Returns:
        ``(candidates, gold_pairs)`` -- the blocked, comparison-attached
        candidates for ``split``, and its order-independent gold match pairs.

    Raises:
        ValueError: If ``split`` is neither ``"train"`` nor ``"test"``.
    """
    from langres.core.resolver import Resolver
    from langres.data.benchmark import gold_pairs_from_clusters

    if split not in ("train", "test"):
        # Without this, any typo ("valid", "validation", "Test") falls through the
        # `if split == "test"` below and silently grades the TRAIN split -- a report
        # that looks valid while scoring the wrong partition. `Literal` only protects
        # type-checked callers; a CLI flag or a dict lookup reaches here untyped.
        raise ValueError(f"candidates_for(): split must be 'train' or 'test', got {split!r}")

    corpus, gold_clusters, _ = bench.load()
    train_records, test_records, train_clusters, test_clusters = bench.split(
        corpus, gold_clusters, seed=seed
    )
    records, split_clusters = (
        (test_records, test_clusters) if split == "test" else (train_records, train_clusters)
    )

    resolver = Resolver.from_schema(bench.schema)  # type: ignore[attr-defined]
    resolver.blocker = bench.build_blocker(bench.blocking_k)  # type: ignore[attr-defined]

    candidates = resolver.candidates([record.model_dump() for record in records])
    gold_pairs = gold_pairs_from_clusters(split_clusters)
    return candidates, gold_pairs


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
