"""Tests for ``langres.eval`` -- the curated evaluation facade.

Behavior tier: ``langres.eval`` re-exports (never reimplements) the user-facing
evaluation surface, so every symbol must be *the same object* as its source in
``core.benchmark`` / ``core.metrics`` / ``data.registry``. Plus a tiny
bring-your-own-data ``evaluate()`` end-to-end, and a subprocess proof that the
facade path never pulls the ``[eval]``-only ``ranx`` into ``sys.modules``.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from typing import Any, cast

import pytest

import langres.core.benchmark as benchmark
import langres.core.metrics as metrics
import langres.data.registry as registry
import langres.eval as ev
from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.module import Module
from langres.core.reports import ScoreInspectionReport


def test_facade_reexports_are_the_source_objects() -> None:
    """Each name resolves to the exact object it re-exports (a facade, not a copy)."""
    assert ev.evaluate is benchmark.evaluate
    assert ev.DEFAULT_PAIR_GRID is benchmark.DEFAULT_PAIR_GRID
    assert ev.list_benchmarks is registry.list_benchmarks
    assert ev.get_benchmark is registry.get_benchmark
    assert ev.ExternalBenchmarkError is registry.ExternalBenchmarkError
    assert ev.reduction_ratio is metrics.reduction_ratio
    assert ev.generalized_merge_distance is metrics.generalized_merge_distance
    assert ev.classify_pairs is metrics.classify_pairs
    assert ev.pair_pr_curve is metrics.pair_pr_curve
    assert ev.calculate_bcubed_metrics is metrics.calculate_bcubed_metrics
    assert ev.calculate_pairwise_metrics is metrics.calculate_pairwise_metrics
    assert ev.roc_auc_score is metrics.roc_auc_score
    assert ev.average_precision_score is metrics.average_precision_score
    assert ev.gold_pairs_from_clusters is benchmark.gold_pairs_from_clusters


def test_dir_and_all_list_the_curated_surface() -> None:
    """``dir()`` and ``__all__`` advertise exactly the curated surface."""
    surface = {
        "evaluate",
        "EvalReport",
        "DEFAULT_PAIR_GRID",
        "list_benchmarks",
        "get_benchmark",
        "ExternalBenchmarkError",
        "reduction_ratio",
        "generalized_merge_distance",
        "classify_pairs",
        "pair_pr_curve",
        "calculate_bcubed_metrics",
        "calculate_pairwise_metrics",
        "roc_auc_score",
        "average_precision_score",
        "gold_pairs_from_clusters",
        "candidates_for",
    }
    assert set(ev.__all__) == surface
    assert set(dir(ev)) == surface
    # Every re-exported name is caching-cache-friendly and resolvable.
    for name in ev.__all__:
        assert getattr(ev, name) is not None


def test_new_facade_names_import_and_reexport_correctly() -> None:
    """Task 3: ``from langres.eval import ...`` works for every new name, and
    each re-export (all but ``candidates_for``, which lives here) is the exact
    source object."""
    from langres.eval import (
        average_precision_score,
        candidates_for,
        gold_pairs_from_clusters,
        roc_auc_score,
    )

    assert average_precision_score is metrics.average_precision_score
    assert roc_auc_score is metrics.roc_auc_score
    assert gold_pairs_from_clusters is benchmark.gold_pairs_from_clusters
    assert candidates_for is ev.candidates_for


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="not_a_real_symbol"):
        ev.not_a_real_symbol  # noqa: B018


class _ExactNameJudge(Module[CompanySchema]):
    """Trivial judge: score 1.0 when the two names match exactly, else 0.0."""

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        for cand in candidates:
            yield PairwiseJudgement(
                left_id=cand.left.id,
                right_id=cand.right.id,
                score=1.0 if cand.left.name == cand.right.name else 0.0,
                score_type="prob_llm",
                decision_step="exact_name",
                provenance={},
            )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        raise NotImplementedError  # pragma: no cover - unused here


def test_evaluate_one_liner_scores_byo_pairs() -> None:
    """The BYO-data one-liner: two matches, one non-match -> perfect at the cut."""
    candidates = [
        ERCandidate(
            left=CompanySchema(id="a1", name="Acme Corp"),
            right=CompanySchema(id="a2", name="Acme Corp"),
            blocker_name="test",
        ),
        ERCandidate(
            left=CompanySchema(id="b1", name="Globex"),
            right=CompanySchema(id="b2", name="Globex"),
            blocker_name="test",
        ),
        ERCandidate(
            left=CompanySchema(id="c1", name="Initech"),
            right=CompanySchema(id="c2", name="Umbrella"),
            blocker_name="test",
        ),
    ]
    gold_pairs = {frozenset({"a1", "a2"}), frozenset({"b1", "b2"})}

    result = ev.evaluate(_ExactNameJudge(), candidates, gold_pairs, grid=(0.5,))

    assert isinstance(result, benchmark.JudgePairEval)
    assert result.n_candidates == 3
    assert result.pair.precision == pytest.approx(1.0)
    assert result.pair.recall == pytest.approx(1.0)
    assert result.pair.f1 == pytest.approx(1.0)


# The DX path must be importable and runnable without the [eval] extra: no
# facade symbol (nor calling evaluate()) may pull ranx into sys.modules.
# Subprocess-based for a fresh import state (this pytest process loads ranx via
# the ranking-metric tests, so an in-process assertion would be order-dependent).
_RANX_FREE_SCRIPT = """
import sys
from langres.eval import evaluate, get_benchmark, list_benchmarks, reduction_ratio
from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.module import Module


class J(Module):
    def forward(self, candidates):
        for c in candidates:
            yield PairwiseJudgement(
                left_id=c.left.id, right_id=c.right.id,
                score=1.0 if c.left.name == c.right.name else 0.0,
                score_type="prob_llm", decision_step="t", provenance={},
            )

    def inspect_scores(self, judgements, sample_size=10):
        raise NotImplementedError


cands = [
    ERCandidate(left=CompanySchema(id="a1", name="X"),
                right=CompanySchema(id="a2", name="X"), blocker_name="t"),
    ERCandidate(left=CompanySchema(id="b1", name="Y"),
                right=CompanySchema(id="b2", name="Z"), blocker_name="t"),
]
result = evaluate(J(), cands, {frozenset({"a1", "a2"})}, grid=(0.5,))
assert type(result).__name__ == "JudgePairEval", result
assert "ranx" not in sys.modules, "the langres.eval DX path pulled in ranx"
print("OK")
"""


def test_eval_facade_path_is_ranx_free() -> None:
    """`from langres.eval import ...` + `evaluate()` never imports ranx."""
    result = subprocess.run(
        [sys.executable, "-c", _RANX_FREE_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"eval facade ranx-free check failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# candidates_for (Task 2): a hand-built Benchmark + BlockingBenchmark double,
# built entirely from core-only components (AllPairsBlocker -- no
# faiss/sentence-transformers), so these tests stay fast and offline. Mirrors
# tests/test_methods.py::_FakeBlockingBenchmark's shape.
# ---------------------------------------------------------------------------


class _FakeSplitBenchmark(benchmark.Benchmark[CompanySchema]):
    """Tiny in-test double satisfying Benchmark + ``langres.methods.BlockingBenchmark``."""

    name = "fake_split"
    threshold_grid = (0.5,)
    schema = CompanySchema
    blocking_k = 2  # unused by AllPairsBlocker; BlockingBenchmark still declares it

    _CORPUS = [
        CompanySchema(id="c1", name="Acme Corporation", address="1 Main St"),
        CompanySchema(id="c1b", name="Acme Corporation", address="1 Main St"),
        CompanySchema(id="c2", name="Zeta Holdings", address="9 Pine Rd"),
        CompanySchema(id="c3", name="Beta Incorporated", address="2 Oak Ave"),
        CompanySchema(id="c3b", name="Beta Incorporated", address="2 Oak Ave"),
        CompanySchema(id="c4", name="Omega Limited", address="7 Elm Blvd"),
    ]
    _GOLD = [{"c1", "c1b"}, {"c2"}, {"c3", "c3b"}, {"c4"}]

    def build_blocker(self, k_neighbors: int) -> AllPairsBlocker[CompanySchema]:
        return AllPairsBlocker(schema=CompanySchema)

    def load(self) -> tuple[list[CompanySchema], list[set[str]], set[frozenset[str]]]:
        gold = [set(c) for c in self._GOLD]
        return list(self._CORPUS), gold, benchmark.gold_pairs_from_clusters(gold)

    def split(
        self,
        corpus: list[CompanySchema],
        gold_clusters: list[set[str]],
        *,
        seed: int,
    ) -> tuple[list[CompanySchema], list[CompanySchema], list[set[str]], list[set[str]]]:
        by_id = {r.id: r for r in corpus}
        clusters_a = [{"c1", "c1b"}, {"c2"}]
        clusters_b = [{"c3", "c3b"}, {"c4"}]
        train_clusters, test_clusters = (
            (clusters_a, clusters_b) if seed == 0 else (clusters_b, clusters_a)
        )
        train = [by_id[i] for c in train_clusters for i in sorted(c)]
        test = [by_id[i] for c in test_clusters for i in sorted(c)]
        return train, test, train_clusters, test_clusters


def test_candidates_for_returns_list_and_set_of_frozensets() -> None:
    candidates, gold_pairs = ev.candidates_for(_FakeSplitBenchmark(), split="test")

    assert isinstance(candidates, list)
    assert isinstance(gold_pairs, set)
    assert all(isinstance(pair, frozenset) for pair in gold_pairs)


def test_candidates_for_rejects_an_unknown_split() -> None:
    # Before the guard, anything but "test" fell through to the TRAIN split, so a
    # typo ("valid", "Test") silently produced a report graded on the wrong
    # partition. `Literal` only protects type-checked callers -- a CLI flag or a
    # dict lookup reaches here untyped.
    for bad in ("valid", "validation", "Test", ""):
        with pytest.raises(ValueError, match="split must be 'train' or 'test'"):
            ev.candidates_for(_FakeSplitBenchmark(), split=cast(Any, bad))


def test_candidates_for_train_and_test_golds_actually_differ() -> None:
    # Gives the guard above its teeth: if a silent train-fallback ever returned,
    # this asserts the two splits are distinguishable, so the wrong one is a
    # detectable wrong answer rather than a coincidence.
    _, train_gold = ev.candidates_for(_FakeSplitBenchmark(), split="train")
    _, test_gold = ev.candidates_for(_FakeSplitBenchmark(), split="test")

    assert train_gold != test_gold


def test_candidates_for_gold_is_nonempty_and_ids_are_real() -> None:
    _, gold_pairs = ev.candidates_for(_FakeSplitBenchmark(), split="test")

    assert gold_pairs
    known_ids = {c.id for c in _FakeSplitBenchmark._CORPUS}
    assert all(i in known_ids for pair in gold_pairs for i in pair)


def test_candidates_for_attaches_comparison_vectors() -> None:
    """candidates_for blocks via Resolver.candidates(), so comparison vectors
    are attached -- exactly like a real Resolver.from_schema pipeline."""
    candidates, _ = ev.candidates_for(_FakeSplitBenchmark(), split="test")

    assert candidates
    assert all(c.comparison is not None for c in candidates)


def test_candidates_for_split_selects_disjoint_records() -> None:
    """``train`` vs ``test`` blocks a DIFFERENT slice of the corpus, so their
    gold pairs (and candidate ids) must differ."""
    train_candidates, train_gold = ev.candidates_for(_FakeSplitBenchmark(), split="train", seed=0)
    test_candidates, test_gold = ev.candidates_for(_FakeSplitBenchmark(), split="test", seed=0)

    assert train_gold and test_gold
    assert train_gold != test_gold
    train_ids = {c.left.id for c in train_candidates} | {c.right.id for c in train_candidates}
    test_ids = {c.left.id for c in test_candidates} | {c.right.id for c in test_candidates}
    assert train_ids.isdisjoint(test_ids)


def test_candidates_for_seed_changes_the_split() -> None:
    """Two different seeds route different records into ``split="test"``."""
    _, gold_seed_0 = ev.candidates_for(_FakeSplitBenchmark(), split="test", seed=0)
    _, gold_seed_1 = ev.candidates_for(_FakeSplitBenchmark(), split="test", seed=1)

    assert gold_seed_0 and gold_seed_1
    assert gold_seed_0 != gold_seed_1


@pytest.mark.slow
def test_candidates_for_against_the_real_registry_tiny_fixture_benchmark() -> None:
    """Integration smoke: candidates_for against a REAL registered benchmark
    (``tiny_fixture``), not just the hand-built fake above. ``tiny_fixture``'s
    ``build_blocker`` returns a real ``VectorBlocker`` (needs the [semantic]
    extra + a real, if tiny, embedding pass) -- skipped without it, and marked
    slow since it is not a cheap/offline test like the ones above.
    """
    pytest.importorskip("faiss", reason="requires the [semantic] extra")
    pytest.importorskip("sentence_transformers", reason="requires the [semantic] extra")

    bench = ev.get_benchmark("tiny_fixture")
    candidates, gold_pairs = ev.candidates_for(bench, split="test")

    assert isinstance(candidates, list)
    assert candidates
    assert gold_pairs
