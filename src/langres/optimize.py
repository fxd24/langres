"""``langres.optimize``: the public autoresearch facade (epic #145, M1).

Two entry points compose P-A (:class:`~langres.core.autoresearch.objective.Objective`),
P-B (:class:`~langres.core.autoresearch.search_space.SearchSpace` + the config→blocker
``factory``), and P-C (the ``propose → run → evaluate → keep`` loop + ``core.runs``
persistence) into a one-call blocking search:

- :func:`score_blocking` — the concrete blocking scorer for ONE config: load a
  benchmark, build the index + blocker the config describes, stream candidates,
  and return blocking metrics (``candidate_recall`` / ``reduction_ratio`` / …).
- :func:`optimize` — load a benchmark once, fingerprint it once, wrap an
  **index-caching** blocking scorer, and drive
  :func:`~langres.core.autoresearch.loop.run_loop` over ``space.configs()``,
  keeping the incumbent the ``objective`` prefers and persisting every trial.

**Import-lightness (hard requirement).** This module sits on the eager
``import langres`` path (the two symbols are root-exported), so its module top is
stdlib/typing only — every langres import (``factory``, ``langres.data``,
``core.metrics``, ``core.runs``, ``core.autoresearch.loop``, ``core.trackers``)
is **lazy, inside a function body**. A bare ``import langres`` therefore never
pulls faiss / sentence-transformers / torch through here (see
``tests/test_import_budget.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from langres.core.autoresearch.loop import LoopResult
    from langres.core.autoresearch.objective import Objective
    from langres.core.autoresearch.search_space import SearchSpace
    from langres.core.benchmark import Benchmark
    from langres.core.embeddings import EmbeddingProvider
    from langres.core.indexes.vector_index import VectorIndex
    from langres.core.runs import RunStore
    from langres.core.trackers import ExperimentTracker


def _resolve_benchmark(benchmark: str | Benchmark[Any]) -> Benchmark[Any]:
    """Accept a registered benchmark **name** or an already-built benchmark object.

    The public DX is a name string (``optimize(space, obj, "amazon_google")``);
    tests (and advanced callers) may pass a hand-built in-memory benchmark object
    directly to stay offline.
    """
    if isinstance(benchmark, str):
        from langres.data.registry import get_benchmark

        return get_benchmark(benchmark)
    return benchmark


def _source_sizes(corpus: list[Any]) -> tuple[int, int] | None:
    """``(n_left, n_right)`` for a two-source linkage corpus, else ``None`` (dedup).

    Groups records by their ``source`` attribute. Exactly two distinct sources is
    the cross-source (linkage) case — the sizes feed ``evaluate_blocking``'s
    ``n_left``/``n_right`` so the reduction ratio uses ``|A| * |B|`` — and every
    other shape (no ``source``, one source, three+) falls back to the dedup
    ``num_records`` reduction ratio.
    """
    from collections import Counter

    counts: Counter[Any] = Counter(getattr(record, "source", None) for record in corpus)
    counts.pop(None, None)
    if len(counts) != 2:
        return None
    left, right = (counts[key] for key in sorted(counts, key=str))
    return (left, right)


def _index_for_config(
    config: Mapping[str, Any],
    corpus: list[Any],
    embedder: EmbeddingProvider | None,
) -> VectorIndex:
    """Build the FAISS index a vector config describes over the corpus texts.

    Texts are extracted in corpus order (``config["text_field"]``) so the index
    positions align with the records the blocker streams — mirroring
    ``data/_benchmark_utils.sweep_blocking_k``.
    """
    from langres.core.autoresearch.factory import build_index

    texts = [getattr(record, config["text_field"]) for record in corpus]
    return build_index(config["embedding_model"], config["metric"], texts, embedder=embedder)


def _score_loaded(
    config: Mapping[str, Any],
    corpus: list[Any],
    gold_clusters: list[set[str]],
    schema: type[Any],
    source_sizes: tuple[int, int] | None,
    *,
    index: VectorIndex | None = None,
) -> dict[str, float]:
    """Score ONE config against already-loaded data — the shared scoring core.

    Builds the blocker the config describes (a prebuilt ``index`` is required for
    ``blocker == "vector"``; ignored for ``"all_pairs"``), streams the corpus to
    candidates, and evaluates blocking. For a two-source corpus the candidates are
    filtered to cross-source pairs (all gold matches are cross-source) and RR is
    computed with ``n_left``/``n_right``; otherwise RR uses ``num_records``.
    """
    from langres.core.autoresearch.factory import build_blocker_from_config
    from langres.core.metrics import evaluate_blocking

    blocker = build_blocker_from_config(config, schema=schema, index=index)
    candidates = list(blocker.stream([record.model_dump() for record in corpus]))

    if source_sizes is not None:
        # Cross-source linkage: keep only inter-source pairs (all gold matches
        # are cross-source, so recall is unchanged) and use |A|*|B| for RR.
        candidates = [c for c in candidates if c.left.source != c.right.source]
        n_left, n_right = source_sizes
        stats = evaluate_blocking(candidates, gold_clusters, n_left=n_left, n_right=n_right)
    else:
        stats = evaluate_blocking(candidates, gold_clusters, num_records=len(corpus))

    return {
        "candidate_recall": stats.candidate_recall,
        "candidate_precision": stats.candidate_precision,
        "reduction_ratio": stats.reduction_ratio,
        "total_candidates": float(stats.total_candidates),
    }


def score_blocking(
    config: Mapping[str, Any],
    benchmark: str | Benchmark[Any],
    *,
    embedder: EmbeddingProvider | None = None,
    index: VectorIndex | None = None,
) -> dict[str, float]:
    """Blocking metrics for ONE config on ``benchmark`` (the concrete scorer).

    Loads the benchmark, builds the index + blocker the config describes (mirroring
    ``sweep_blocking_k``), streams the full corpus to candidates, and returns a
    plain metrics dict — ``candidate_recall``, ``candidate_precision``,
    ``reduction_ratio``, ``total_candidates`` — ready for an
    :class:`~langres.core.autoresearch.objective.Objective`.

    Args:
        config: A config dict as yielded by ``SearchSpace.configs()`` (keys
            ``blocker``, and for ``"vector"`` also ``embedding_model`` / ``metric``
            / ``text_field`` / ``k_neighbors``).
        benchmark: A registered benchmark **name** (loaded via the data registry)
            or an already-built benchmark object (offline / test path).
        embedder: Optional pre-built embedder (a ``FakeEmbedder`` in tests) passed
            to ``build_index``; production leaves it ``None`` to load the real
            SentenceTransformer. Ignored when ``index`` is supplied or the blocker
            is ``"all_pairs"``.
        index: Optional prebuilt vector index to reuse instead of building one
            (the ``optimize`` closure threads a cached index in through here).

    Returns:
        The blocking metrics mapping (all values ``float``).
    """
    bench = _resolve_benchmark(benchmark)
    corpus, gold_clusters, _gold_pairs = bench.load()
    if not corpus:
        raise ValueError(f"benchmark {getattr(bench, 'name', bench)!r} loaded an empty corpus")
    schema = type(corpus[0])
    if index is None and config.get("blocker") == "vector":
        index = _index_for_config(config, corpus, embedder)
    return _score_loaded(config, corpus, gold_clusters, schema, _source_sizes(corpus), index=index)


def _canonical_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Collapse a config to its recipe-relevant shape (for correct dedup).

    ``all_pairs`` ignores every vector axis (``embedding_model`` / ``metric`` /
    ``text_field`` / ``k_neighbors``), so P-B's grid can yield several
    *semantically identical* ``all_pairs`` configs differing only in those unused
    keys. Reducing each to ``{"blocker": "all_pairs"}`` makes them hash to one
    ``recipe_id`` so the loop's in-run dedup skips the redundant repeats. A
    ``vector`` config is recipe-relevant in full and passes through unchanged.
    """
    if config.get("blocker") == "all_pairs":
        return {"blocker": "all_pairs"}
    return dict(config)


def optimize(
    space: SearchSpace,
    objective: Objective,
    benchmark: str | Benchmark[Any],
    *,
    seed: int | None = None,
    store: str | Path | RunStore | None = None,
    dedup: bool = True,
    split: str = "full",
    embedder: EmbeddingProvider | None = None,
    tracker: ExperimentTracker | None = None,
) -> LoopResult:
    """Search ``space`` for the blocking config ``objective`` prefers on ``benchmark``.

    Loads the benchmark **once**, fingerprints it **once**, and wraps an
    index-caching blocking scorer: because ``SearchSpace.configs()`` varies
    ``k_neighbors`` innermost, one vector index is built per
    ``(embedding_model, metric, text_field)`` group and reused across every ``k``
    (``k`` lives on the blocker, not the index). It then drives
    :func:`~langres.core.autoresearch.loop.run_loop`, which keeps the incumbent
    ``objective.is_better`` selects and persists **every** trial (accepted and
    rejected) to ``store`` — ``store=None`` persists nothing.

    Args:
        space: The declarative config grid to enumerate.
        objective: The immutable keep-if-better decision.
        benchmark: A registered benchmark **name** or an already-built benchmark
            object (offline / test path).
        seed: Optional seed recorded on every run for provenance/identity (blocking
            itself is deterministic, so this only labels the run). Recorded under
            ``seeds["optimize"]`` when given.
        store: Where to persist run records (path / ``RunStore`` / ``None``);
            ``None`` writes nothing.
        dedup: Skip a config whose ``recipe_id`` was already scored this run
            (default ``True``); the degenerate ``all_pairs`` repeats are collapsed
            first via :func:`_canonical_config`.
        split: Split label recorded on every run (``RunContext.split_id``). Default
            ``"full"`` — M1 measures blocking over the whole loaded corpus.
        embedder: Optional pre-built embedder for the vector index (a
            ``FakeEmbedder`` keeps tests offline); production leaves it ``None``.
        tracker: Optional experiment tracker; defaults to a no-op.

    Returns:
        The :class:`~langres.core.autoresearch.loop.LoopResult` (best incumbent +
        full trial trail).
    """
    from langres.core.autoresearch.loop import run_loop
    from langres.core.runs import dataset_fingerprint
    from langres.core.trackers import NoOpTracker

    bench = _resolve_benchmark(benchmark)
    dataset_name = benchmark if isinstance(benchmark, str) else bench.name
    corpus, gold_clusters, _gold_pairs = bench.load()
    if not corpus:
        raise ValueError(f"benchmark {dataset_name!r} loaded an empty corpus")
    schema = type(corpus[0])
    fingerprint = dataset_fingerprint(corpus, gold_clusters)
    source_sizes = _source_sizes(corpus)

    # One index per (embedding_model, metric, text_field); reused across all k
    # (SearchSpace yields k innermost, so the group is contiguous). all_pairs
    # configs never touch the cache (no index).
    index_cache: dict[tuple[str, str, str], VectorIndex] = {}

    def scorer(config: Mapping[str, Any]) -> dict[str, float]:
        index: VectorIndex | None = None
        if config.get("blocker") == "vector":
            key = (config["embedding_model"], config["metric"], config["text_field"])
            index = index_cache.get(key)
            if index is None:
                index = _index_for_config(config, corpus, embedder)
                index_cache[key] = index
        return _score_loaded(config, corpus, gold_clusters, schema, source_sizes, index=index)

    return run_loop(
        (_canonical_config(config) for config in space.configs()),
        scorer,
        objective,
        experiment=f"optimize_blocking:{dataset_name}",
        dataset_name=dataset_name,
        dataset_fingerprint=fingerprint,
        split_id=split,
        seeds={"optimize": seed} if seed is not None else None,
        store=store,
        tracker=tracker or NoOpTracker(),
        dedup=dedup,
    )
