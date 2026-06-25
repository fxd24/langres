"""Comparator contract: turn a pair of entities into a ComparisonVector.

The Comparator is the missing-aware bridge between raw entities and the scorer
Module. The Blocker emits candidate pairs; the Comparator compares each pair
feature-by-feature into a :class:`~langres.core.feature.ComparisonVector`; the
scorer Module combines that vector into a score.

**M0 Wave 1 builds only the ABC and the typed errors.** The concrete
implementation and ``from_schema`` factory land in Wave 2a — they are
deliberately absent here so Wave 2/3 code against a frozen contract.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from langres.core.feature import ComparisonVector

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class MissingRequiredField(ValueError):
    """Raised when an entity lacks a field a feature requires to be comparable.

    Defined in Wave 1 so Wave 2 imports a stable error type; raised by the
    concrete Comparator (Wave 2a).
    """


class NoComparableFeatures(ValueError):
    """Raised when a schema yields no comparable features at all.

    Defined in Wave 1 so Wave 2 imports a stable error type; raised by the
    concrete Comparator / ``from_schema`` (Wave 2a).
    """


class Comparator(ABC, Generic[SchemaT]):
    """Abstract base for missing-aware pairwise feature comparison.

    A Comparator turns two entities into a
    :class:`~langres.core.feature.ComparisonVector`: one
    :class:`~langres.core.feature.ComparisonLevel` per declared feature plus the
    raw similarity for each PRESENT feature.

    Missing-aware contract (implemented by the concrete Comparator in Wave 2a;
    pinned here so downstream waves can rely on it):

    - If **either** side's value for a feature is ``None`` or empty, the feature
      is ``MISSING`` and is dropped — it contributes no similarity, and is never
      compared as a string.
    - **Missing-vs-missing** is also dropped (never a "two empty strings match"
      false positive).
    - Only features that are PRESENT on **both** sides get a computed similarity.

    The body of :meth:`compare` and the ``from_schema`` factory are NOT defined
    in Wave 1.
    """

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
