"""Smoke test: the autoresearch E1 proof example runs and its loop closes.

Drives the importable core of ``examples/research/blocking_recall_autoresearch.py``
(``run_search``) over a tiny two-config grid on the real amazon_google benchmark
with real MiniLM embeddings, asserting the loop produces a feasible incumbent with
non-zero blocking recall and persists every trial (accepted + rejected) to the
owned ``RunStore`` off-git.

Marked ``slow`` (loads an embedding model + builds a FAISS index) but network-free
and $0. Uses a 2-config space (one index build) to stay lean — the full 20-config
grid is exercised by running the example directly, not by fast CI.
"""

from __future__ import annotations

import pytest

from examples.research.blocking_recall_autoresearch import (
    RR_BUDGET,
    build_space,
    run_search,
)
from langres.autoresearch.loop import LoopResult
from langres.autoresearch.search_space import SearchSpace
from langres.core.runs import RunStore

pytestmark = pytest.mark.slow


def test_run_search_closes_the_loop_and_logs_every_trial(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A tiny k-sweep yields a feasible incumbent with recall>0; all trials logged."""
    store = str(tmp_path / "runs.jsonl")
    # Two ascending-k configs, one (metric, text_field) group => a single index build.
    tiny = SearchSpace(
        blocker=("vector",),
        embedding_model=("all-MiniLM-L6-v2",),
        metric=("cosine",),
        text_field=("embed_text",),
        k_neighbors=(5, 10),
    )

    result = run_search(tiny, RR_BUDGET, store, seed=0)

    # 1. The loop returned a real result with a feasible incumbent that has signal.
    assert isinstance(result, LoopResult)
    assert result.best_metrics is not None
    assert result.best_metrics["candidate_recall"] > 0.0
    assert result.best_metrics["reduction_ratio"] >= RR_BUDGET

    # 2. Both configs were evaluated (recall climbs with k => the second is kept).
    assert len(result.trials) == 2
    assert all(t.status == "completed" for t in result.trials)
    assert result.best_config["k_neighbors"] == 10  # more neighbours => more recall

    # 3. Every trial — accepted and rejected — is durable off-git in the RunStore.
    records = RunStore(store).read()
    assert len(records) == 2
    assert all(r.metrics is not None and "accepted" in r.metrics for r in records)
    assert sum(1 for r in records if r.metrics["accepted"] == 1.0) >= 1


def test_full_search_space_is_the_documented_modest_grid() -> None:
    """The example's grid is the modest 20-config sweep (cheap; no run)."""
    assert len(build_space()) == 20
