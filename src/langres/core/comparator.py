"""Comparator contract: turn a pair of entities into a ComparisonVector.

The Comparator is the missing-aware bridge between raw entities and the scorer
Module. The Blocker emits candidate pairs; the Comparator compares each pair
feature-by-feature into a :class:`~langres.core.feature.ComparisonVector`; the
scorer Module combines that vector into a score.

**M0 Wave 1 builds only the ABC and the typed errors.** The concrete
implementation and ``from_schema`` factory land in Wave 2a — they are
deliberately absent here so Wave 2/3 code against a frozen contract.
"""

import difflib
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Generic, Literal, TypeVar, cast

from pydantic import BaseModel
from rapidfuzz import fuzz

from langres.core.feature import ComparisonLevel, ComparisonVector, FeatureSpec
from langres.core.registry import register

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Supported rapidfuzz algorithms (mirrors RapidfuzzModule.Algorithm).
Algorithm = Literal["ratio", "token_sort_ratio", "token_set_ratio"]

_FUZZ_FUNCS: dict[str, Callable[[str, str], float]] = {
    "ratio": fuzz.ratio,
    "token_sort_ratio": fuzz.token_sort_ratio,
    "token_set_ratio": fuzz.token_set_ratio,
}


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

    @classmethod
    def from_schema(
        cls,
        schema: type[BaseModel],
        *,
        exclude: set[str] | None = None,
        weights: dict[str, float] | None = None,
        algorithm: Algorithm = "token_sort_ratio",
    ) -> "StringComparator[SchemaT]":
        """Auto-derive a :class:`StringComparator` from a Pydantic schema.

        One :class:`FeatureSpec` is created per ``str | None`` field, excluding
        ``id`` (and anything in ``exclude``). Non-string fields are skipped with
        a log note rather than silently ``str()``-coerced.

        Args:
            schema: The Pydantic entity schema to derive features from.
            exclude: Field names to skip. Defaults to ``{"id"}``.
            weights: Optional per-feature weight overrides. Fields absent here
                keep the default weight of ``1.0``.
            algorithm: rapidfuzz algorithm used for every feature.

        Returns:
            A configured :class:`StringComparator`.

        Raises:
            NoComparableFeatures: If the schema yields zero usable string
                features after exclusion.
        """
        return StringComparator.from_schema(
            schema, exclude=exclude, weights=weights, algorithm=algorithm
        )


@register("comparator")
class StringComparator(Comparator[SchemaT]):
    """Concrete missing-aware string Comparator backed by rapidfuzz.

    Compares two entities feature-by-feature into a
    :class:`~langres.core.feature.ComparisonVector`. For each declared
    :class:`FeatureSpec`, the value is read with ``getattr``; if either side is
    ``None`` or an empty string the feature is ``MISSING`` (dropped, never
    compared); otherwise it is ``PRESENT`` with a rapidfuzz similarity in
    ``[0, 1]``. ``MISMATCH`` is never emitted in M0.

    Construction validates feature names against ``schema.model_fields`` when a
    ``schema`` is supplied (or implicitly via :meth:`from_schema`).
    """

    def __init__(
        self,
        feature_specs: list[FeatureSpec],
        algorithm: Algorithm = "token_sort_ratio",
        *,
        schema: type[BaseModel] | None = None,
    ) -> None:
        """Initialize a StringComparator.

        Args:
            feature_specs: The features to compare, in order.
            algorithm: rapidfuzz algorithm used for every feature.
            schema: Optional schema to validate ``feature_specs`` names against.
                When given, an unknown name raises ``ValueError`` with a
                ``difflib`` did-you-mean suggestion.

        Raises:
            NoComparableFeatures: If ``feature_specs`` is empty.
            ValueError: If a feature name is not a field of ``schema``.
        """
        if not feature_specs:
            raise NoComparableFeatures(
                "StringComparator requires at least one FeatureSpec; got none."
            )
        if schema is not None:
            self._validate_against_schema(feature_specs, schema)
        self.feature_specs = feature_specs
        self.algorithm = algorithm

    @staticmethod
    def _validate_against_schema(feature_specs: list[FeatureSpec], schema: type[BaseModel]) -> None:
        valid = set(schema.model_fields)
        for spec in feature_specs:
            if spec.name not in valid:
                suggestions = difflib.get_close_matches(spec.name, sorted(valid), n=1)
                hint = f" — did you mean {suggestions[0]!r}?" if suggestions else ""
                raise ValueError(
                    f"Feature {spec.name!r} is not a field of {schema.__name__}{hint} "
                    f"Available fields: {', '.join(sorted(valid))}"
                )

    @classmethod
    def from_schema(
        cls,
        schema: type[BaseModel],
        *,
        exclude: set[str] | None = None,
        weights: dict[str, float] | None = None,
        algorithm: Algorithm = "token_sort_ratio",
    ) -> "StringComparator[SchemaT]":
        """Derive a StringComparator from a schema's ``str | None`` fields."""
        exclude = {"id"} if exclude is None else exclude
        weights = weights or {}

        specs: list[FeatureSpec] = []
        for name, info in schema.model_fields.items():
            if name in exclude:
                continue
            if not _is_string_field(info.annotation):
                logger.info(
                    "from_schema: skipping non-string field %r on %s "
                    "(only str | None fields are comparable in M0)",
                    name,
                    schema.__name__,
                )
                continue
            specs.append(FeatureSpec(name=name, weight=weights.get(name, 1.0)))

        if not specs:
            raise NoComparableFeatures(
                f"Schema {schema.__name__} yields no comparable string features "
                f"(after excluding {sorted(exclude)})."
            )
        return cls(specs, algorithm=algorithm, schema=schema)

    def compare(self, left: SchemaT, right: SchemaT) -> ComparisonVector:
        """Compare two entities into a ComparisonVector (see class docstring)."""
        fuzz_func = _FUZZ_FUNCS[self.algorithm]
        levels: dict[str, ComparisonLevel] = {}
        similarities: dict[str, float] = {}

        for spec in self.feature_specs:
            left_val = getattr(left, spec.name, None)
            right_val = getattr(right, spec.name, None)
            if not left_val or not right_val:
                # None or empty string on either side -> MISSING (incl. both).
                levels[spec.name] = ComparisonLevel.MISSING
                continue
            levels[spec.name] = ComparisonLevel.PRESENT
            similarities[spec.name] = fuzz_func(left_val, right_val) / 100.0

        return ComparisonVector(levels=levels, similarities=similarities)

    @property
    def config(self) -> dict[str, object]:
        """Serializable construction config (feature_specs + algorithm)."""
        return {
            "feature_specs": [spec.model_dump() for spec in self.feature_specs],
            "algorithm": self.algorithm,
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "StringComparator[SchemaT]":
        """Reconstruct a StringComparator from its :attr:`config`."""
        specs = [
            FeatureSpec.model_validate(s) for s in cast("list[object]", config["feature_specs"])
        ]
        return cls(specs, algorithm=cast("Algorithm", config["algorithm"]))


def _is_string_field(annotation: object) -> bool:
    """True if a Pydantic field annotation is ``str`` or ``str | None``."""
    if annotation is str:
        return True
    # str | None (and Optional[str]) -> Union with str + NoneType.
    args = getattr(annotation, "__args__", None)
    if args is not None:
        non_none = [a for a in args if a is not type(None)]
        return non_none == [str]
    return False
