"""Unit tests for the concrete StringComparator (M0 Wave 2a).

Covers the missing-aware comparison contract, the ``from_schema`` factory
(id-exclusion, non-string skipping), construction-time validation with
did-you-mean suggestions, and config round-trip.
"""

import datetime

import pytest
from pydantic import BaseModel

from langres.core.comparator import Comparator, NoComparableFeatures
from langres.core.comparators import StringComparator
from langres.core.feature import ComparisonLevel, FeatureSpec
from langres.core.models import CompanySchema


class _MixedSchema(BaseModel):
    """Schema with a mix of string and non-string fields for from_schema tests."""

    id: str
    name: str
    nickname: str | None = None
    employee_count: int = 0
    founded: datetime.date | None = None


class _NoStringSchema(BaseModel):
    """Schema whose only non-id field is non-string."""

    id: str
    employee_count: int = 0


def _company(**kwargs: object) -> CompanySchema:
    base: dict[str, object] = {"id": "x", "name": "Acme"}
    base.update(kwargs)
    return CompanySchema(**base)  # type: ignore[arg-type]


class TestCompare:
    def test_present_features_get_similarity(self) -> None:
        comp = StringComparator([FeatureSpec(name="name")])
        vec = comp.compare(
            _company(id="a", name="Acme Corporation"),
            _company(id="b", name="Acme Corporation"),
        )
        assert vec.levels["name"] == ComparisonLevel.PRESENT
        assert vec.similarities["name"] == pytest.approx(1.0)

    def test_similarity_in_unit_interval(self) -> None:
        comp = StringComparator([FeatureSpec(name="name")])
        vec = comp.compare(
            _company(id="a", name="TechStart Industries"),
            _company(id="b", name="TechStrat Industries"),
        )
        assert 0.0 < vec.similarities["name"] < 1.0

    def test_missing_on_left_side_is_missing(self) -> None:
        comp = StringComparator([FeatureSpec(name="website")])
        vec = comp.compare(
            _company(id="a", website=None),
            _company(id="b", website="https://acme.com"),
        )
        assert vec.levels["website"] == ComparisonLevel.MISSING
        assert "website" not in vec.similarities

    def test_missing_on_right_side_is_missing(self) -> None:
        comp = StringComparator([FeatureSpec(name="website")])
        vec = comp.compare(
            _company(id="a", website="https://acme.com"),
            _company(id="b", website=None),
        )
        assert vec.levels["website"] == ComparisonLevel.MISSING
        assert "website" not in vec.similarities

    def test_empty_string_counts_as_missing(self) -> None:
        comp = StringComparator([FeatureSpec(name="address")])
        vec = comp.compare(
            _company(id="a", address=""),
            _company(id="b", address="123 Main St"),
        )
        assert vec.levels["address"] == ComparisonLevel.MISSING

    def test_missing_vs_missing_never_compared(self) -> None:
        comp = StringComparator([FeatureSpec(name="website")])
        vec = comp.compare(
            _company(id="a", website=None),
            _company(id="b", website=None),
        )
        assert vec.levels["website"] == ComparisonLevel.MISSING
        assert "website" not in vec.similarities

    def test_never_emits_mismatch(self) -> None:
        comp = StringComparator([FeatureSpec(name="name"), FeatureSpec(name="website")])
        vec = comp.compare(
            _company(id="a", name="Acme", website=None),
            _company(id="b", name="Other", website="https://acme.com"),
        )
        assert ComparisonLevel.MISMATCH not in vec.levels.values()

    def test_every_feature_appears_in_levels(self) -> None:
        comp = StringComparator(
            [FeatureSpec(name="name"), FeatureSpec(name="address"), FeatureSpec(name="website")]
        )
        vec = comp.compare(
            _company(id="a", name="Acme", address="123 Main", website=None),
            _company(id="b", name="Acme", address="123 Main", website="x"),
        )
        assert set(vec.levels) == {"name", "address", "website"}


class TestFromSchema:
    def test_excludes_id(self) -> None:
        comp = StringComparator.from_schema(CompanySchema)
        names = {spec.name for spec in comp.feature_specs}
        assert "id" not in names

    def test_includes_all_string_fields(self) -> None:
        comp = StringComparator.from_schema(CompanySchema)
        names = {spec.name for spec in comp.feature_specs}
        assert names == {"name", "address", "phone", "website"}

    def test_skips_non_string_fields(self) -> None:
        comp = StringComparator.from_schema(_MixedSchema)
        names = {spec.name for spec in comp.feature_specs}
        # employee_count (int) and founded (date) skipped; only str | None kept.
        assert names == {"name", "nickname"}

    def test_default_weights_equal(self) -> None:
        comp = StringComparator.from_schema(CompanySchema)
        weights = {spec.weight for spec in comp.feature_specs}
        assert weights == {1.0}

    def test_custom_weights_applied(self) -> None:
        comp = StringComparator.from_schema(CompanySchema, weights={"name": 0.6})
        by_name = {spec.name: spec.weight for spec in comp.feature_specs}
        assert by_name["name"] == pytest.approx(0.6)
        # Unspecified fields keep the default weight of 1.0.
        assert by_name["address"] == pytest.approx(1.0)

    def test_custom_exclude_set(self) -> None:
        comp = StringComparator.from_schema(CompanySchema, exclude={"id", "phone"})
        names = {spec.name for spec in comp.feature_specs}
        assert names == {"name", "address", "website"}

    def test_no_comparable_features_raises(self) -> None:
        with pytest.raises(NoComparableFeatures):
            StringComparator.from_schema(_NoStringSchema)


class TestValidation:
    def test_unknown_feature_name_raises_with_suggestion(self) -> None:
        with pytest.raises(ValueError, match="did you mean 'name'"):
            StringComparator([FeatureSpec(name="naem")], schema=CompanySchema)

    def test_unknown_feature_without_close_match(self) -> None:
        with pytest.raises(ValueError, match="zzzzz"):
            StringComparator([FeatureSpec(name="zzzzz")], schema=CompanySchema)

    def test_empty_feature_specs_raises(self) -> None:
        with pytest.raises(NoComparableFeatures):
            StringComparator([])

    def test_no_schema_skips_field_validation(self) -> None:
        # Without a schema, arbitrary feature names are allowed (validation is
        # opt-in via from_schema or an explicit schema=).
        comp = StringComparator([FeatureSpec(name="anything")])
        assert comp.feature_specs[0].name == "anything"


class TestConfigRoundTrip:
    def test_config_is_serializable(self) -> None:
        comp = StringComparator.from_schema(CompanySchema, weights={"name": 0.6})
        cfg = comp.config
        # Pydantic-serializable: round-trips through JSON.
        import json

        json.dumps(cfg)

    def test_from_config_reconstructs_equivalent_comparator(self) -> None:
        comp = StringComparator.from_schema(CompanySchema, weights={"name": 0.6})
        rebuilt = StringComparator.from_config(comp.config)

        left = _company(id="a", name="DataFlow Solutions", address="123 Main")
        right = _company(id="b", name="DataFlow Solutions", address="123 Main")
        assert comp.compare(left, right).model_dump() == rebuilt.compare(left, right).model_dump()
        assert comp.algorithm == rebuilt.algorithm
        assert [s.model_dump() for s in comp.feature_specs] == [
            s.model_dump() for s in rebuilt.feature_specs
        ]

    def test_algorithm_preserved_in_config(self) -> None:
        comp = StringComparator([FeatureSpec(name="name")], algorithm="ratio")
        assert comp.config["algorithm"] == "ratio"
        assert StringComparator.from_config(comp.config).algorithm == "ratio"

    def test_registered_under_comparator(self) -> None:
        from langres.core.registry import get_component

        assert get_component("comparator") is StringComparator
