"""Comparator contract: turn a pair of entities into a ComparisonVector.

The Comparator is the missing-aware bridge between raw entities and the scorer
Matcher. The Blocker emits candidate pairs; the Comparator compares each pair
feature-by-feature into a :class:`~langres.core.feature.ComparisonVector`; the
scorer Matcher combines that vector into a score.

**Contract only.** The concrete rapidfuzz-backed :class:`StringComparator` lives
in :mod:`langres.core.comparators` -- import it from there::

    from langres.core.comparators import StringComparator

It was split out of this module in W1: a contract that imports its own
implementation is a contract that sits *above* the components depending on it.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from langres.core.feature import ComparisonVector, FeatureSpec

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class MissingRequiredField(ValueError):
    """Raised when an entity lacks a field a feature requires to be comparable.

    Defined on the contract so implementations and callers share one error type.
    """


class NoComparableFeatures(ValueError):
    """Raised when a schema yields no comparable features at all.

    Defined on the contract so implementations and callers share one error type;
    raised by :class:`~langres.core.comparators.StringComparator`.
    """


class Comparator(ABC, Generic[SchemaT]):
    """Abstract base for missing-aware pairwise feature comparison.

    A Comparator turns two entities into a
    :class:`~langres.core.feature.ComparisonVector`: one
    :class:`~langres.core.feature.ComparisonLevel` per declared feature plus the
    raw similarity for each PRESENT feature.

    Missing-aware contract (implemented by
    :class:`~langres.core.comparators.StringComparator`; pinned here so callers
    can rely on it):

    - If **either** side's value for a feature is ``None`` or empty, the feature
      is ``MISSING`` and is dropped — it contributes no similarity, and is never
      compared as a string.
    - **Missing-vs-missing** is also dropped (never a "two empty strings match"
      false positive).
    - Only features that are PRESENT on **both** sides get a computed similarity.

    To build the default comparator for a schema, use the implementation's own
    factory -- :meth:`langres.core.comparators.StringComparator.from_schema`.
    """

    @property
    def feature_specs(self) -> list[FeatureSpec]:
        """FeatureSpecs this comparator scores on; empty for spec-less comparators.

        Part of the Comparator contract: the WeightedAverageMatcher reads
        ``comparator.feature_specs`` to weight the score. The ABC defaults to an
        empty list so a comparator that scores without declared features (e.g. a
        self-contained or learned scorer) satisfies the contract without
        raising. :class:`~langres.core.comparators.StringComparator` overrides
        this to return its declared features.
        """
        return []

    @abstractmethod
    def compare(self, left: SchemaT, right: SchemaT) -> ComparisonVector:
        """Compare two entities into a :class:`ComparisonVector`.

        Args:
            left: The left entity of the pair.
            right: The right entity of the pair.

        Returns:
            A ComparisonVector with a level for every declared feature and a
            similarity for every PRESENT feature.
        """
        ...  # pragma: no cover
