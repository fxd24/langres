"""M2 Wave 2 — held-out BCubed eval harness (fast, embedding-free unit tests).

Covers the three pure helpers added for the M2 skeleton eval:

- ``complete_partition`` — singleton-completes a predicted clustering so BCubed
  averages over EVERY item (the Clusterer drops singletons).
- ``evaluate_resolver_bcubed`` — resolve -> complete -> BCubed + Pair-Completeness
  + sanity floor, as a :class:`BCubedEvalResult`.
- ``tune_threshold_on_train`` — picks the best Clusterer threshold by TRAIN F1.

These use an ``AllPairsBlocker`` resolver (``Resolver.from_schema``) or stub the
eval, so no embeddings run — they stay in the fast suite. The real-embedding
end-to-end assertion lives in ``tests/data/test_m2_skeleton_slow.py``.
"""

import pytest

from langres.data import er_benchmarks
from langres.data.er_benchmarks import (
    DEFAULT_THRESHOLD_GRID,
    BCubedEvalResult,
    RestaurantSchema,
    complete_partition,
    evaluate_resolver_bcubed,
    tune_threshold_on_train,
)
from langres.core.resolver import Resolver

# --- complete_partition ----------------------------------------------------------


def test_complete_partition_adds_singletons_for_uncovered_ids() -> None:
    completed = complete_partition([{"a", "b"}], ["a", "b", "c", "d"])
    assert completed == [{"a", "b"}, {"c"}, {"d"}]


def test_complete_partition_empty_prediction_is_all_singletons() -> None:
    # The Clusterer merged nothing -> every id becomes its own singleton.
    assert complete_partition([], ["a", "b"]) == [{"a"}, {"b"}]


def test_complete_partition_noop_when_already_complete() -> None:
    predicted = [{"a", "b"}, {"c"}]
    assert complete_partition(predicted, ["a", "b", "c"]) == predicted


def test_complete_partition_is_deterministic_in_all_ids_order() -> None:
    # Singletons are appended in all_ids order, not set-iteration order.
    completed = complete_partition([{"a"}], ["a", "z", "m"])
    assert completed == [{"a"}, {"z"}, {"m"}]


# --- evaluate_resolver_bcubed (AllPairs resolver, no embeddings) ------------------


def _tiny_corpus() -> tuple[list[RestaurantSchema], list[set[str]]]:
    """Two cross-source identical-field match pairs + two singletons.

    Identical fields make the equal-weight ``WeightedAverageMatcher`` score the
    matched pairs at ~1.0 (plenty of evidence: all five fields present), so they
    merge at a low threshold; the distinct singletons never merge.
    """
    records = [
        RestaurantSchema(
            id="f0",
            name="alpha pizza",
            addr="1 main",
            city="rome",
            phone="111",
            type="italian",
            source="fodors",
        ),
        RestaurantSchema(
            id="z0",
            name="alpha pizza",
            addr="1 main",
            city="rome",
            phone="111",
            type="italian",
            source="zagat",
        ),
        RestaurantSchema(
            id="f1",
            name="beta sushi",
            addr="2 oak",
            city="tokyo",
            phone="222",
            type="japanese",
            source="fodors",
        ),
        RestaurantSchema(
            id="z1",
            name="beta sushi",
            addr="2 oak",
            city="tokyo",
            phone="222",
            type="japanese",
            source="zagat",
        ),
        RestaurantSchema(id="f100", name="gamma diner", source="fodors"),
        RestaurantSchema(id="z200", name="delta cafe", source="zagat"),
    ]
    truth = [{"f0", "z0"}, {"f1", "z1"}, {"f100"}, {"z200"}]
    return records, truth


def test_evaluate_resolver_bcubed_returns_expected_fields() -> None:
    records, truth = _tiny_corpus()
    # AllPairsBlocker -> no embeddings; threshold low enough to merge identical pairs.
    resolver = Resolver.from_schema(RestaurantSchema, threshold=0.3)

    result = evaluate_resolver_bcubed(resolver, records, truth)

    assert isinstance(result, BCubedEvalResult)
    for value in (result.precision, result.recall, result.f1, result.pair_completeness):
        assert 0.0 <= value <= 1.0
    # AllPairs surfaces every cross-source pair -> blocking captures all matches.
    assert result.pair_completeness == 1.0
    # The matched pairs are recovered, so the run beats "merge nothing".
    assert result.f1 > result.sanity_floor_f1
    # Identical-field matches + distinct singletons -> a perfect partition here.
    assert result.f1 == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0


def test_evaluate_resolver_bcubed_sanity_floor_is_all_singletons_score() -> None:
    records, truth = _tiny_corpus()
    resolver = Resolver.from_schema(RestaurantSchema, threshold=0.3)

    result = evaluate_resolver_bcubed(resolver, records, truth)

    # The floor is below 1.0 because all-singletons misses both true match pairs.
    assert 0.0 < result.sanity_floor_f1 < 1.0


# --- tune_threshold_on_train (stubbed eval, no embeddings) ------------------------


def test_tune_threshold_picks_argmax_train_f1(monkeypatch: pytest.MonkeyPatch) -> None:
    # Map each candidate threshold to a known train F1; tune must return the max.
    f1_by_threshold = {0.3: 0.10, 0.4: 0.50, 0.5: 0.90, 0.6: 0.40, 0.7: 0.20, 0.8: 0.05}

    def fake_eval(
        resolver: Resolver,
        records: list[RestaurantSchema],
        clusters: list[set[str]],
    ) -> BCubedEvalResult:
        threshold = resolver.clusterer.threshold
        return BCubedEvalResult(
            precision=0.0,
            recall=0.0,
            f1=f1_by_threshold[threshold],
            pair_completeness=1.0,
            sanity_floor_f1=0.0,
        )

    monkeypatch.setattr(er_benchmarks, "evaluate_resolver_bcubed", fake_eval)

    best = tune_threshold_on_train([], [], thresholds=tuple(f1_by_threshold))
    assert best == 0.5


def test_tune_threshold_breaks_ties_to_first(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_eval(
        resolver: Resolver,
        records: list[RestaurantSchema],
        clusters: list[set[str]],
    ) -> BCubedEvalResult:
        return BCubedEvalResult(
            precision=0.0, recall=0.0, f1=0.5, pair_completeness=1.0, sanity_floor_f1=0.0
        )

    monkeypatch.setattr(er_benchmarks, "evaluate_resolver_bcubed", fake_eval)

    # All equal F1 -> the first (lowest) threshold in order wins.
    best = tune_threshold_on_train([], [], thresholds=(0.4, 0.5, 0.6))
    assert best == 0.4


def test_tune_threshold_rejects_empty_grid() -> None:
    with pytest.raises(ValueError, match="thresholds is empty"):
        tune_threshold_on_train([], [], thresholds=())


def test_default_threshold_grid_is_ascending_in_unit_range() -> None:
    assert DEFAULT_THRESHOLD_GRID == tuple(sorted(DEFAULT_THRESHOLD_GRID))
    assert all(0.0 < t < 1.0 for t in DEFAULT_THRESHOLD_GRID)
