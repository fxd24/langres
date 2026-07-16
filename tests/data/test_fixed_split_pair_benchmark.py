"""Unit + slow tests for the fixed-split pair-benchmark adapter and honest eval.

The fast tests use a tiny toy schema + a deterministic fake judge, so they never
import the torch-backed dataset loaders. Two ``@pytest.mark.slow`` tests exercise
the real Amazon-Google and Abt-Buy full standard splits end-to-end with the
``RandomForestMatcher`` floor.
"""

from collections.abc import Iterator

import pytest
from pydantic import BaseModel

from langres.core.calibration import derive_threshold
from langres.core.comparators import StringComparator
from langres.core.feature import ComparisonLevel
from langres.core.metrics import classify_pairs
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.matcher import Matcher
from langres.core.reports import ScoreInspectionReport, _inspect_scores_impl
from langres.data.fixed_split_pair_benchmark import (
    FixedSplitPairBenchmark,
    SplitPairData,
    evaluate_fixed_split_honest,
)


class _ToySchema(BaseModel):
    """Minimal record schema: id + two comparable string fields."""

    id: str
    name: str
    city: str | None = None


def _toy_corpus() -> list[_ToySchema]:
    return [
        _ToySchema(id="a1", name="alpha corp", city="NY"),
        _ToySchema(id="a2", name="beta llc", city="LA"),
        _ToySchema(id="b1", name="alpha corporation", city="NY"),
        _ToySchema(id="b2", name="gamma inc", city="SF"),
        _ToySchema(id="b3", name="beta limited", city="LA"),
    ]


def _toy_splits() -> dict[str, list[tuple[str, str, int]]]:
    return {
        "train": [("a1", "b1", 1), ("a2", "b3", 1), ("a1", "b2", 0)],
        "valid": [("a2", "b2", 0)],
        "test": [("a1", "b1", 1), ("a2", "b2", 0), ("a1", "b3", 0)],
    }


def _toy_benchmark() -> FixedSplitPairBenchmark[_ToySchema]:
    return FixedSplitPairBenchmark(
        name="toy",
        corpus=_toy_corpus(),
        splits=_toy_splits(),
        comparator=StringComparator.from_schema(_ToySchema),
    )


# ---------------------------------------------------------------------------
# FixedSplitPairBenchmark
# ---------------------------------------------------------------------------


def test_build_attaches_comparison_vector_to_every_candidate() -> None:
    bench = _toy_benchmark()
    train = bench.build("train")

    assert isinstance(train, SplitPairData)
    for cand in train.candidates:
        assert cand.comparison is not None
        # both name + city are present on the first pair (a1/b1, both NY).
    a1_b1 = train.candidates[0]
    assert a1_b1.comparison is not None
    assert a1_b1.comparison.levels["name"] == ComparisonLevel.PRESENT
    assert "name" in a1_b1.comparison.similarities


def test_labels_are_positionally_aligned_with_candidates() -> None:
    bench = _toy_benchmark()
    train = bench.build("train")

    assert train.labels == [True, True, False]
    assert len(train.labels) == len(train.candidates)


def test_gold_is_exactly_the_label_one_pairs() -> None:
    bench = _toy_benchmark()
    train = bench.build("train")

    assert train.gold == {frozenset({"a1", "b1"}), frozenset({"a2", "b3"})}


def test_candidates_reference_the_corpus_records_by_id() -> None:
    bench = _toy_benchmark()
    train = bench.build("train")

    first = train.candidates[0]
    assert first.left.id == "a1"
    assert first.right.id == "b1"
    assert first.left.name == "alpha corp"
    assert first.right.name == "alpha corporation"
    assert first.blocker_name == "toy_fixed_pairs"


def test_feature_specs_exposes_comparator_features() -> None:
    bench = _toy_benchmark()
    names = {spec.name for spec in bench.feature_specs}
    # id is excluded; name + city are the comparable string fields.
    assert names == {"name", "city"}


def test_split_names_lists_all_provided_splits() -> None:
    bench = _toy_benchmark()
    assert bench.split_names == ["train", "valid", "test"]


def test_build_caches_per_split() -> None:
    bench = _toy_benchmark()
    first = bench.build("test")
    second = bench.build("test")
    assert first is second


def test_build_unknown_split_raises_keyerror() -> None:
    bench = _toy_benchmark()
    with pytest.raises(KeyError, match="Unknown split"):
        bench.build("holdout")


def test_build_missing_corpus_id_raises_valueerror() -> None:
    bench = FixedSplitPairBenchmark(
        name="toy",
        corpus=_toy_corpus(),
        splits={"test": [("a1", "ZZZ", 1)]},
        comparator=StringComparator.from_schema(_ToySchema),
    )
    with pytest.raises(ValueError, match="ZZZ.*not.*corpus"):
        bench.build("test")


def test_from_loaders_builds_from_injected_loaders() -> None:
    def corpus_loader() -> tuple[list[_ToySchema], object, object]:
        return _toy_corpus(), None, None

    bench = FixedSplitPairBenchmark.from_loaders(
        name="toy",
        schema=_ToySchema,
        corpus_loader=corpus_loader,
        pair_split_loader=_toy_splits,
    )
    assert bench.split_names == ["train", "valid", "test"]
    test = bench.build("test")
    assert test.gold == {frozenset({"a1", "b1"})}
    assert {spec.name for spec in bench.feature_specs} == {"name", "city"}


def test_from_loaders_accepts_an_explicit_comparator() -> None:
    def corpus_loader() -> tuple[list[_ToySchema], object, object]:
        return _toy_corpus(), None, None

    comparator = StringComparator.from_schema(_ToySchema, exclude={"id", "city"})
    bench = FixedSplitPairBenchmark.from_loaders(
        name="toy",
        schema=_ToySchema,
        corpus_loader=corpus_loader,
        pair_split_loader=_toy_splits,
        comparator=comparator,
    )
    # The explicit comparator (name only) overrides the schema-derived default.
    assert {spec.name for spec in bench.feature_specs} == {"name"}


# ---------------------------------------------------------------------------
# evaluate_fixed_split_honest (deterministic fake judge)
# ---------------------------------------------------------------------------


class _MapJudge(Matcher[_ToySchema]):
    """A deterministic judge: score is a pure lookup on the pair's id-frozenset.

    Keying on ``frozenset({left.id, right.id})`` makes the judge a pure function
    of the candidate (a real judge scores a given pair identically in any split),
    so train/test pairs must be disjoint to control the two score distributions.
    """

    def __init__(self, scores: dict[frozenset[str], float]) -> None:
        self._scores = scores

    def forward(self, candidates: Iterator[ERCandidate[_ToySchema]]) -> Iterator[PairwiseJudgement]:
        for cand in candidates:
            key = frozenset({cand.left.id, cand.right.id})
            yield PairwiseJudgement(
                left_id=cand.left.id,
                right_id=cand.right.id,
                score=self._scores[key],
                score_type="heuristic",
                decision_step="map_judge",
                provenance={},
            )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        return _inspect_scores_impl(judgements, sample_size)


def _honest_corpus() -> list[_ToySchema]:
    return [_ToySchema(id=i, name=i) for i in ("a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4")]


def _honest_splits() -> dict[str, list[tuple[str, str, int]]]:
    # Train / valid / test use DISJOINT pairs so the map judge stays a pure fn.
    return {
        "train": [("a1", "b1", 1), ("a2", "b2", 1), ("a3", "b3", 0), ("a4", "b4", 0)],
        "valid": [("a1", "b3", 1), ("a4", "b1", 0)],
        "test": [("a1", "b2", 1), ("a3", "b1", 0), ("a2", "b4", 0)],
    }


_HONEST_SCORES: dict[frozenset[str], float] = {
    frozenset({"a1", "b1"}): 0.90,  # train +
    frozenset({"a2", "b2"}): 0.85,  # train +
    frozenset({"a3", "b3"}): 0.20,  # train -
    frozenset({"a4", "b4"}): 0.10,  # train -
    frozenset({"a1", "b3"}): 0.70,  # valid +
    frozenset({"a4", "b1"}): 0.30,  # valid -
    frozenset({"a1", "b2"}): 0.80,  # test +
    frozenset({"a3", "b1"}): 0.30,  # test -
    frozenset({"a2", "b4"}): 0.25,  # test -
}


def _honest_benchmark() -> FixedSplitPairBenchmark[_ToySchema]:
    return FixedSplitPairBenchmark(
        name="honest_toy",
        corpus=_honest_corpus(),
        splits=_honest_splits(),
        comparator=StringComparator.from_schema(_ToySchema),
    )


def test_honest_threshold_is_derived_on_train_not_test() -> None:
    bench = _honest_benchmark()
    judge = _MapJudge(_HONEST_SCORES)

    result = evaluate_fixed_split_honest(judge, bench, derive_on="train")

    # The derived threshold must equal derive_threshold on the TRAIN scores/labels
    # only — recomputed independently here.
    expected_threshold = derive_threshold([0.90, 0.85, 0.20, 0.10], [True, True, False, False])
    assert result.derived_threshold == expected_threshold
    assert result.derive_on == "train"
    # Youden separates the two train positives from the negatives at 0.85.
    assert result.derived_threshold == pytest.approx(0.85)


def test_honest_metrics_apply_the_fixed_train_threshold_to_full_test() -> None:
    bench = _honest_benchmark()
    judge = _MapJudge(_HONEST_SCORES)

    result = evaluate_fixed_split_honest(judge, bench, derive_on="train")

    # Recompute what classify_pairs must produce on the full test split at the
    # train-derived threshold, straight from the judge's test judgements.
    test = bench.build("test")
    test_judgements = list(judge.forward(iter(test.candidates)))
    expected = classify_pairs(test_judgements, test.gold, result.derived_threshold)

    assert result.honest.precision == expected.precision
    assert result.honest.recall == expected.recall
    assert result.honest.f1 == expected.f1
    assert result.honest.tp == expected.tp
    assert result.honest.fp == expected.fp
    assert result.honest.fn == expected.fn
    # test positive (a1,b2)=0.80 sits below the 0.85 train cut -> a missed match.
    assert result.honest.f1 == 0.0
    assert result.honest.fn == 1


def test_honesty_delta_captures_argmax_on_test_inflation() -> None:
    bench = _honest_benchmark()
    judge = _MapJudge(_HONEST_SCORES)

    result = evaluate_fixed_split_honest(judge, bench, derive_on="train")

    # A test-tuned threshold recovers the single positive perfectly (F1 1.0),
    # while the honest cut misses it -> the delta is the full leakage gap.
    assert result.argmax_on_test.f1 == pytest.approx(1.0)
    assert result.argmax_on_test.f1 >= result.honest.f1
    assert result.honesty_delta_f1 == pytest.approx(result.argmax_on_test.f1 - result.honest.f1)
    assert result.honesty_delta_f1 == pytest.approx(1.0)


def test_derive_on_valid_uses_the_valid_split() -> None:
    bench = _honest_benchmark()
    judge = _MapJudge(_HONEST_SCORES)

    result = evaluate_fixed_split_honest(judge, bench, derive_on="valid")

    # Valid youden cut is 0.70; test positive (a1,b2)=0.80 clears it, negatives
    # (0.30/0.25) do not -> honest F1 is perfect here.
    assert result.derive_on == "valid"
    assert result.derived_threshold == pytest.approx(0.70)
    assert result.honest.f1 == pytest.approx(1.0)
    assert result.honest.tp == 1


# ---------------------------------------------------------------------------
# Slow end-to-end: real datasets with the RandomForestMatcher floor
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "n_test", "n_test_pos"),
    [("amazon_google", 2293, 234), ("abt_buy", 1916, 206)],
)
def test_real_dataset_full_test_split_shapes(name: str, n_test: int, n_test_pos: int) -> None:
    """The adapter reproduces the standard full-split shapes for both datasets."""
    if name == "amazon_google":
        from langres.data.amazon_google import (
            ProductSchema,
            load_amazon_google,
            load_amazon_google_pair_splits,
        )

        bench = FixedSplitPairBenchmark.from_loaders(
            name=name,
            schema=ProductSchema,
            corpus_loader=load_amazon_google,
            pair_split_loader=load_amazon_google_pair_splits,
        )
    else:
        from langres.data.abt_buy import (
            AbtBuySchema,
            load_abt_buy,
            load_abt_buy_pair_splits,
        )

        bench = FixedSplitPairBenchmark.from_loaders(
            name=name,
            schema=AbtBuySchema,
            corpus_loader=load_abt_buy,
            pair_split_loader=load_abt_buy_pair_splits,
        )

    test = bench.build("test")
    assert len(test.candidates) == n_test
    assert len(test.labels) == n_test
    assert sum(test.labels) == n_test_pos
    assert len(test.gold) == n_test_pos
    # Every candidate carries a comparison vector (the whole point of the adapter).
    assert all(c.comparison is not None for c in test.candidates)


@pytest.mark.slow
def test_random_forest_floor_runs_honestly_on_amazon_google() -> None:
    """RandomForestMatcher fits on train and grades honestly on the full AG test."""
    from langres.core.matchers.random_forest_judge import RandomForestMatcher
    from langres.data.amazon_google import (
        ProductSchema,
        load_amazon_google,
        load_amazon_google_pair_splits,
    )

    bench = FixedSplitPairBenchmark.from_loaders(
        name="amazon_google",
        schema=ProductSchema,
        corpus_loader=load_amazon_google,
        pair_split_loader=load_amazon_google_pair_splits,
    )
    train = bench.build("train")
    judge: RandomForestMatcher[ProductSchema] = RandomForestMatcher(
        feature_specs=bench.feature_specs
    )
    judge.fit(iter(train.candidates), train.labels)

    result = evaluate_fixed_split_honest(judge, bench, derive_on="train")

    # Honest numbers must be real (a fit RF on this data is well above zero) and
    # never beat the leaky argmax-on-test ceiling.
    assert 0.0 < result.honest.f1 <= 1.0
    assert result.argmax_on_test.f1 >= result.honest.f1
    assert result.honesty_delta_f1 >= 0.0


# ---------------------------------------------------------------------------
# A decision-only (binary) judge has no scores to derive a threshold from
# ---------------------------------------------------------------------------


class _DecisionOnlyJudge(Matcher[_ToySchema]):
    """A binary judge: it decides directly and emits NO score."""

    def __init__(self, decisions: dict[frozenset[str], bool]) -> None:
        self._decisions = decisions

    def forward(self, candidates: Iterator[ERCandidate[_ToySchema]]) -> Iterator[PairwiseJudgement]:
        for cand in candidates:
            key = frozenset({cand.left.id, cand.right.id})
            yield PairwiseJudgement(
                left_id=cand.left.id,
                right_id=cand.right.id,
                decision=self._decisions.get(key, False),
                score_type="prob_llm",  # no score
                decision_step="decision_only",
                provenance={},
            )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        return _inspect_scores_impl(judgements, sample_size)


def test_decision_only_judge_cannot_derive_a_threshold() -> None:
    """Deriving a threshold needs scores; a decision-only judge has none.

    The eval must fail loudly and name the judge rather than silently dropping
    its score-less judgements.
    """
    bench = _honest_benchmark()
    judge = _DecisionOnlyJudge({})

    with pytest.raises(ValueError, match="_DecisionOnlyJudge"):
        evaluate_fixed_split_honest(judge, bench, derive_on="train")
