"""Unit tests for the Comparator ABC contract (M0 Wave 1).

Only the ABC + typed errors are built in Wave 1; the concrete impl is Wave 2a.
These tests pin the contract: Comparator cannot be instantiated, exposes an
abstract compare() returning a ComparisonVector, and the field/comparison
errors are importable for downstream waves.
"""

import pytest

from langres.core.comparator import (
    Comparator,
    MissingRequiredField,
    NoComparableFeatures,
)
from langres.core.feature import ComparisonLevel, ComparisonVector
from langres.core.models import CompanySchema


class TestComparatorABC:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            Comparator()  # type: ignore[abstract]

    def test_subclass_must_implement_compare(self) -> None:
        class _Incomplete(Comparator[CompanySchema]):
            pass

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_works(self) -> None:
        class _Stub(Comparator[CompanySchema]):
            def compare(self, left: CompanySchema, right: CompanySchema) -> ComparisonVector:
                return ComparisonVector(
                    levels={"name": ComparisonLevel.PRESENT},
                    similarities={"name": 1.0},
                )

        stub = _Stub()
        vec = stub.compare(
            CompanySchema(id="a", name="X"),
            CompanySchema(id="b", name="X"),
        )
        assert vec.present_features() == {"name"}


class TestComparatorErrors:
    def test_missing_required_field_is_exception(self) -> None:
        with pytest.raises(MissingRequiredField):
            raise MissingRequiredField("name")

    def test_no_comparable_features_is_exception(self) -> None:
        with pytest.raises(NoComparableFeatures):
            raise NoComparableFeatures("no features")
