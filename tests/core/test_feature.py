"""Unit tests for the feature/comparison contracts (M0 Wave 1).

Covers:
- ComparisonLevel enum (PRESENT, MISSING, MISMATCH-reserved)
- FeatureSpec serialization contract
- ComparisonVector serialization + observability
- combine_present() evidence-floor helper (the over-merge guard)
"""

import pytest
from pydantic import ValidationError

from langres.core.feature import (
    ComparisonLevel,
    ComparisonVector,
    FeatureSpec,
    combine_present,
)


class TestComparisonLevel:
    def test_has_present_missing_mismatch(self) -> None:
        assert ComparisonLevel.PRESENT == "PRESENT"
        assert ComparisonLevel.MISSING == "MISSING"
        assert ComparisonLevel.MISMATCH == "MISMATCH"

    def test_is_str_enum_serializable(self) -> None:
        # str enum -> JSON serializes to the bare string value
        assert ComparisonLevel.PRESENT.value == "PRESENT"
        assert isinstance(ComparisonLevel.PRESENT, str)


class TestFeatureSpec:
    def test_defaults(self) -> None:
        spec = FeatureSpec(name="name")
        assert spec.name == "name"
        assert spec.kind == "string"
        assert spec.weight == 1.0
        assert spec.is_anchor is False

    def test_weight_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            FeatureSpec(name="name", weight=-0.1)

    def test_kind_only_string_supported(self) -> None:
        with pytest.raises(ValidationError):
            FeatureSpec(name="name", kind="numeric")  # type: ignore[arg-type]

    def test_round_trip_serialization(self) -> None:
        spec = FeatureSpec(name="address", weight=0.3, is_anchor=True)
        dumped = spec.model_dump()
        restored = FeatureSpec(**dumped)
        assert restored == spec
        # JSON round-trip too (must be pure data, no callables)
        assert FeatureSpec.model_validate_json(spec.model_dump_json()) == spec


class TestComparisonVector:
    def test_construct_and_access(self) -> None:
        vec = ComparisonVector(
            levels={"name": ComparisonLevel.PRESENT, "phone": ComparisonLevel.MISSING},
            similarities={"name": 0.92},
        )
        assert vec.levels["name"] == ComparisonLevel.PRESENT
        assert vec.levels["phone"] == ComparisonLevel.MISSING
        assert vec.similarities["name"] == 0.92

    def test_round_trip_serialization(self) -> None:
        vec = ComparisonVector(
            levels={"name": ComparisonLevel.PRESENT},
            similarities={"name": 0.5},
        )
        restored = ComparisonVector.model_validate_json(vec.model_dump_json())
        assert restored == vec

    def test_present_features_helper(self) -> None:
        vec = ComparisonVector(
            levels={
                "name": ComparisonLevel.PRESENT,
                "phone": ComparisonLevel.MISSING,
                "website": ComparisonLevel.PRESENT,
            },
            similarities={"name": 0.9, "website": 0.4},
        )
        assert vec.present_features() == {"name", "website"}

    def test_empty_vector(self) -> None:
        vec = ComparisonVector(levels={}, similarities={})
        assert vec.present_features() == set()


class TestCombinePresent:
    """The evidence-floor helper. This is the over-merge guard."""

    def test_all_missing_returns_zero(self) -> None:
        # No present features -> never 0/0, always 0.0
        assert combine_present(similarities={}, weights={"name": 1.0}) == 0.0

    def test_single_weak_present_floored_to_zero(self) -> None:
        # 1 present feature, present-weight 0.3 < 0.5 -> floored to 0.0
        score = combine_present(
            similarities={"name": 0.95},
            weights={"name": 0.3, "address": 0.7},
        )
        assert score == 0.0

    def test_single_present_weight_exactly_half_passes(self) -> None:
        # present-weight == 0.5 -> meets the >= 0.5 floor -> scores
        score = combine_present(
            similarities={"name": 0.8},
            weights={"name": 0.5, "address": 0.5},
        )
        # only present feature renormalizes to weight 1.0 -> score == sim
        assert score == pytest.approx(0.8)

    def test_two_present_features_pass_floor_regardless_of_weight(self) -> None:
        # 2+ present features satisfy the floor even with low total weight
        score = combine_present(
            similarities={"a": 1.0, "b": 0.0},
            weights={"a": 0.1, "b": 0.1, "c": 0.8},
        )
        # renormalize over present {a:0.1, b:0.1} -> {a:0.5, b:0.5}
        assert score == pytest.approx(0.5)

    def test_renormalizes_weights_over_present(self) -> None:
        score = combine_present(
            similarities={"name": 1.0, "address": 0.0},
            weights={"name": 0.6, "address": 0.2},
        )
        # renormalize {name:0.6, address:0.2} -> {name:0.75, address:0.25}
        assert score == pytest.approx(0.75)

    def test_zero_present_weight_no_zero_division(self) -> None:
        # 2 present features but both have weight 0 -> can't renormalize.
        # Must not raise ZeroDivisionError; returns 0.0.
        score = combine_present(
            similarities={"a": 1.0, "b": 1.0},
            weights={"a": 0.0, "b": 0.0},
        )
        assert score == 0.0

    def test_similarity_for_feature_without_weight_treated_as_zero_weight(self) -> None:
        # Defensive: a present similarity whose feature is missing from weights
        # contributes zero weight (and counts toward present-count).
        score = combine_present(
            similarities={"name": 0.9, "ghost": 0.9},
            weights={"name": 1.0},
        )
        # 2 present -> floor passes; renormalize over present-weights
        # {name:1.0, ghost:0.0} -> name:1.0 -> score == 0.9
        assert score == pytest.approx(0.9)
