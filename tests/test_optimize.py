"""Tests for the public ``langres.optimize`` facade (score_blocking + optimize).

The fast tests use a TINY hand-built in-memory benchmark (a few records + gold),
passed to the facade as a benchmark *object* (the offline seam alongside the
public name-string DX). The ``all_pairs`` path needs no embeddings at all; the
vector path injects a ``FakeEmbedder`` (and is guarded by ``importorskip`` so it
skips without the [semantic] extra). One ``@pytest.mark.slow`` smoke exercises a
registered benchmark with the real SentenceTransformer + FAISS stack.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest
from pydantic import BaseModel

from langres import optimize, score_blocking
from langres.autoresearch.loop import LoopResult
from langres.autoresearch.objective import Objective
from langres.autoresearch.search_space import SearchSpace
from langres.core.runs import RunStore


class _ProductRec(BaseModel):
    """A tiny two-source product record (offline test schema)."""

    id: str
    name: str
    source: Literal["a", "b"]


class _MemoryBenchmark:
    """A minimal in-memory benchmark: ``name`` + ``load()`` is all the facade needs."""

    name = "memory_products"

    def load(self) -> tuple[list[_ProductRec], list[set[str]], set[frozenset[str]]]:
        corpus = [
            _ProductRec(id="a1", name="apple iphone 12", source="a"),
            _ProductRec(id="a2", name="samsung galaxy s21", source="a"),
            _ProductRec(id="b1", name="apple iphone 12", source="b"),
            _ProductRec(id="b2", name="samsung galaxy s21", source="b"),
        ]
        gold_clusters = [{"a1", "b1"}, {"a2", "b2"}]
        gold_pairs = {frozenset({"a1", "b1"}), frozenset({"a2", "b2"})}
        return corpus, gold_clusters, gold_pairs


_VECTOR_CFG = {
    "blocker": "vector",
    "embedding_model": "unused-with-fake",
    "metric": "cosine",
    "text_field": "name",
    "k_neighbors": 3,
}


def _fake_embedder() -> Any:
    pytest.importorskip("faiss", reason="requires the [semantic] extra")
    from langres.core.embeddings import FakeEmbedder

    return FakeEmbedder(embedding_dim=16)


# ---------------------------------------------------------------------------
# score_blocking
# ---------------------------------------------------------------------------


def test_score_blocking_all_pairs_returns_metrics_offline() -> None:
    metrics = score_blocking({"blocker": "all_pairs"}, _MemoryBenchmark())
    assert set(metrics) == {
        "candidate_recall",
        "candidate_precision",
        "reduction_ratio",
        "total_candidates",
    }
    # AllPairs surfaces every cross-source pair -> perfect blocking recall.
    assert metrics["candidate_recall"] == 1.0
    # Cross-source filter drops the two intra-source pairs (a1-a2, b1-b2).
    assert metrics["total_candidates"] == 4.0


def test_score_blocking_vector_with_fake_embedder() -> None:
    embedder = _fake_embedder()
    metrics = score_blocking(_VECTOR_CFG, _MemoryBenchmark(), embedder=embedder)
    assert "candidate_recall" in metrics
    assert 0.0 <= metrics["candidate_recall"] <= 1.0
    assert isinstance(metrics["total_candidates"], float)


def test_score_blocking_accepts_a_prebuilt_index() -> None:
    embedder = _fake_embedder()
    from langres.autoresearch.factory import build_index

    corpus, _gc, _gp = _MemoryBenchmark().load()
    index = build_index("x", "cosine", [r.name for r in corpus], embedder=embedder)
    metrics = score_blocking(_VECTOR_CFG, _MemoryBenchmark(), index=index)
    assert "candidate_recall" in metrics


def test_score_blocking_by_registered_name_offline() -> None:
    # The public name-string DX: get_benchmark("tiny_fixture") loads offline, and
    # all_pairs blocking needs no embeddings -> a fully offline name-path check.
    metrics = score_blocking({"blocker": "all_pairs"}, "tiny_fixture")
    assert metrics["candidate_recall"] == 1.0


def test_score_blocking_empty_corpus_raises() -> None:
    class _Empty:
        name = "empty"

        def load(self) -> tuple[list[_ProductRec], list[set[str]], set[frozenset[str]]]:
            return [], [], set()

    with pytest.raises(ValueError, match="empty corpus"):
        score_blocking({"blocker": "all_pairs"}, _Empty())


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------


def test_optimize_all_pairs_returns_loop_result_and_logs_trials(tmp_path: Any) -> None:
    store_path = tmp_path / "runs.jsonl"
    space = SearchSpace(blocker=("all_pairs",), text_field=("name",), k_neighbors=(5,))
    result = optimize(
        space, Objective.maximize("candidate_recall"), _MemoryBenchmark(), store=store_path
    )

    assert isinstance(result, LoopResult)
    assert result.best_metrics is not None
    assert result.best_metrics["candidate_recall"] == 1.0
    assert result.best_config == {"blocker": "all_pairs"}  # canonicalized

    records = RunStore(store_path).read()
    assert len(records) == 1
    assert records[0].metrics["accepted"] == 1.0
    assert records[0].context.dataset_name == "memory_products"
    assert records[0].context.split_id == "full"
    assert records[0].context.dataset_fingerprint is not None


def test_optimize_dedup_collapses_degenerate_all_pairs(tmp_path: Any) -> None:
    store_path = tmp_path / "runs.jsonl"
    # Several all_pairs configs differing only in (ignored) vector axes -> after
    # canonicalization they share one recipe_id and the loop scores just one.
    space = SearchSpace(
        blocker=("all_pairs",),
        embedding_model=("m1", "m2"),
        metric=("cosine", "L2"),
        text_field=("name",),
        k_neighbors=(5, 10),
    )
    assert len(space) == 8  # 2 * 2 * 1 * 2 degenerate configs
    result = optimize(
        space, Objective.maximize("candidate_recall"), _MemoryBenchmark(), store=store_path
    )
    assert len(result.trials) == 1  # all collapsed to one
    assert len(RunStore(store_path).read()) == 1


def test_optimize_vector_reuses_index_across_k() -> None:
    pytest.importorskip("faiss", reason="requires the [semantic] extra")
    from langres.core.embeddings import FakeEmbedder

    class _CountingEmbedder(FakeEmbedder):
        encode_calls = 0

        def encode(self, texts: Any) -> Any:
            type(self).encode_calls += 1
            return super().encode(texts)

    space = SearchSpace(
        blocker=("vector",),
        embedding_model=("x",),
        metric=("cosine",),
        text_field=("name",),
        k_neighbors=(1, 2, 3),
    )
    embedder = _CountingEmbedder(embedding_dim=16)
    result = optimize(
        space, Objective.maximize("candidate_recall"), _MemoryBenchmark(), embedder=embedder
    )

    assert len(result.trials) == 3
    # One index built (one encode call) for the whole k-sweep, since k varies
    # innermost within the shared (model, metric, text_field) group.
    assert _CountingEmbedder.encode_calls == 1


def test_optimize_store_none_writes_nothing(tmp_path: Any) -> None:
    space = SearchSpace(blocker=("all_pairs",), text_field=("name",), k_neighbors=(5,))
    result = optimize(space, Objective.maximize("candidate_recall"), _MemoryBenchmark(), store=None)
    assert isinstance(result, LoopResult)
    assert list(tmp_path.iterdir()) == []


def test_optimize_records_seed_when_given(tmp_path: Any) -> None:
    store_path = tmp_path / "runs.jsonl"
    space = SearchSpace(blocker=("all_pairs",), text_field=("name",), k_neighbors=(5,))
    optimize(
        space,
        Objective.maximize("candidate_recall"),
        _MemoryBenchmark(),
        store=store_path,
        seed=7,
    )
    records = RunStore(store_path).read()
    assert records[0].context.seeds == {"optimize": 7}


# ---------------------------------------------------------------------------
# Tracker spec DX -- ``optimize(..., tracker="trackio")`` instead of
# ``tracker=resolve_tracker("trackio")``. ``optimize`` just forwards the spec
# into ``run_loop``, which is the one place it's resolved (see
# ``tests/core/test_autoresearch_loop.py`` for the resolution-branch coverage).
# ---------------------------------------------------------------------------


def test_optimize_tracker_none_default_never_raises() -> None:
    space = SearchSpace(blocker=("all_pairs",), text_field=("name",), k_neighbors=(5,))
    result = optimize(
        space, Objective.maximize("candidate_recall"), _MemoryBenchmark(), tracker=None
    )
    assert isinstance(result, LoopResult)


def test_optimize_tracker_string_resolves_to_the_named_backend_and_is_driven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tracker="trackio"`` end-to-end through ``optimize`` -- mocked like
    ``tests/test_trackio_tracker.py``'s ``fake_trackio`` fixture (no network)."""
    import langres.core.trackers.trackio_tracker as trackio_mod

    class _FakeRun:
        def __init__(self) -> None:
            self.log_calls: list[Any] = []

        def log(self, data: Any, step: int | None = None) -> None:
            self.log_calls.append((data, step))

        def finish(self) -> None:
            pass

    class _FakeTrackio:
        def __init__(self) -> None:
            self.init_kwargs: dict[str, Any] | None = None
            self.run = _FakeRun()

        def init(self, **kwargs: Any) -> Any:
            self.init_kwargs = kwargs
            return self.run

    fake = _FakeTrackio()
    monkeypatch.setattr(trackio_mod, "trackio", fake)

    space = SearchSpace(blocker=("all_pairs",), text_field=("name",), k_neighbors=(5,))
    optimize(space, Objective.maximize("candidate_recall"), _MemoryBenchmark(), tracker="trackio")
    assert fake.init_kwargs is not None  # TrackioTracker.start_run really fired


# ---------------------------------------------------------------------------
# Slow: real embedder + FAISS on a registered benchmark
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_optimize_vector_real_embedder_on_registered_benchmark() -> None:
    """End-to-end with the real SentenceTransformer + FAISS stack (downloads a model)."""
    pytest.importorskip("faiss", reason="requires the [semantic] extra")
    pytest.importorskip("sentence_transformers", reason="requires the [semantic] extra")

    space = SearchSpace(
        blocker=("vector",),
        embedding_model=("all-MiniLM-L6-v2",),
        metric=("cosine",),
        text_field=("embed_text",),
        k_neighbors=(3, 5),
    )
    result = optimize(space, Objective.maximize("candidate_recall"), "tiny_fixture")
    assert isinstance(result, LoopResult)
    assert result.best_metrics is not None
    assert 0.0 <= result.best_metrics["candidate_recall"] <= 1.0
    assert len(result.trials) == 2
