"""KeyBlocker implementation for exact/normalized-key candidate generation.

This blocker buckets records by a configurable key (a declared schema field or
a full callable) and emits all pairs within each bucket. It is schema-agnostic
via the same ``schema``/``schema_factory`` mutual-exclusion pattern used by
``AllPairsBlocker``/``VectorBlocker``.
"""

from collections.abc import Callable, Iterator
from typing import Any, ClassVar

from langres.core.blocker import Blocker, SchemaT
from langres.core.blockers.all_pairs import register_schema_idempotent, schema_to_factory
from langres.core.models import ERCandidate
from langres.core.registry import get_schema, register
from langres.core.reports import CandidateInspectionReport


def _field_key_fn(field: str) -> Callable[[Any], str | None]:
    """Build a key-extraction function that reads ``field`` off an entity."""

    def _fn(entity: Any) -> str | None:
        value = getattr(entity, field)
        return None if value is None else str(value)

    return _fn


@register("key_blocker")
class KeyBlocker(Blocker[SchemaT]):
    """Schema-agnostic blocker that pairs records sharing an (normalized) key.

    Records are bucketed by a key extracted from each entity; every pair
    within a bucket becomes a candidate (a bucket of size B contributes
    B*(B-1)/2 pairs). Records whose key extraction is ``None`` (e.g. a missing
    field) are excluded entirely -- they get no candidates from this blocker.

    The key can come from either:

    - ``key_field=`` (declarative): ``getattr(entity, key_field)``, serializable.
    - ``key_fn=`` (callable): full control over key extraction, not serializable.

    ``normalize=True`` (default) applies ``.strip().lower()`` to non-``None``
    keys before bucketing, regardless of which path produced the key -- this is
    what makes "Zurich" and "  ZURICH  " land in the same bucket.

    Example:
        # Block companies by normalized city
        blocker = KeyBlocker(schema=CompanySchema, key_field="city")
        candidates = list(blocker.stream(company_records))

        # Custom key: first letter of the name
        blocker = KeyBlocker(
            schema=CompanySchema,
            key_fn=lambda c: c.name[0] if c.name else None,
        )

    Note:
        A single exact-key blocker trades recall for precision/speed: two
        matching records with slightly different key values (typos, different
        formatting) land in different buckets and are missed. Compose with a
        recall-oriented blocker (e.g. ``VectorBlocker``) via
        :class:`~langres.core.blockers.composite.CompositeBlocker` (union) to
        recover that recall.
    """

    # Registry key, mirrored as a class attribute so the Resolver's uniform
    # serialization helper can discover the type name (see resolver.py).
    type_name: ClassVar[str] = "key_blocker"

    def __init__(
        self,
        schema_factory: Callable[[dict[str, Any]], SchemaT] | None = None,
        schema: type[SchemaT] | None = None,
        *,
        key_field: str | None = None,
        key_fn: Callable[[SchemaT], str | None] | None = None,
        normalize: bool = True,
    ):
        """Initialize KeyBlocker.

        Provide exactly one of ``schema``/``schema_factory`` (entity
        normalization) and exactly one of ``key_field``/``key_fn`` (key
        extraction).

        Args:
            schema_factory: Callable that transforms a raw dict into a Pydantic
                schema object (SchemaT). Mutually exclusive with ``schema``.
            schema: Pydantic schema class for declarative reconstruction.
                Mutually exclusive with ``schema_factory``.
            key_field: Name of a schema field to use as the blocking key
                (declarative, serializable). Mutually exclusive with ``key_fn``.
            key_fn: Callable extracting a key (or ``None``) from an entity
                (full control, not serializable). Mutually exclusive with
                ``key_field``.
            normalize: If ``True`` (default), lowercase + strip non-``None``
                keys before bucketing.

        Raises:
            ValueError: If both or neither of ``schema``/``schema_factory`` are
                given, or if both or neither of ``key_field``/``key_fn`` are
                given.
        """
        if (schema is None) == (schema_factory is None):
            raise ValueError(
                "KeyBlocker requires exactly one of 'schema' or "
                "'schema_factory' (got both or neither)."
            )
        if (key_field is None) == (key_fn is None):
            raise ValueError(
                "KeyBlocker requires exactly one of 'key_field' or 'key_fn' (got both or neither)."
            )

        self._schema_type_name: str | None = None
        if schema is not None:
            self._schema_type_name = register_schema_idempotent(schema)
            self.schema_factory = schema_to_factory(schema)
        else:
            assert schema_factory is not None  # narrowed by the guard above
            self.schema_factory = schema_factory

        self._key_field_name: str | None = key_field
        if key_field is not None:
            self.key_fn: Callable[[SchemaT], str | None] = _field_key_fn(key_field)
        else:
            assert key_fn is not None  # narrowed by the guard above
            self.key_fn = key_fn

        self.normalize = normalize

    @property
    def config(self) -> dict[str, object]:
        """Serializable construction config for the registry.

        Returns:
            ``{"schema_type_name": ..., "key_field": ..., "normalize": ...}``.

        Raises:
            ValueError: If this blocker was constructed with ``schema_factory``
                or ``key_fn`` (opaque callables that cannot be serialized).
        """
        if self._schema_type_name is None:
            raise ValueError(
                "KeyBlocker built with 'schema_factory' is not serializable "
                "(a callable cannot round-trip through config); construct with "
                "schema= to persist."
            )
        if self._key_field_name is None:
            raise ValueError(
                "KeyBlocker built with 'key_fn' is not serializable (a callable "
                "cannot round-trip through config); construct with key_field= "
                "to persist."
            )
        return {
            "schema_type_name": self._schema_type_name,
            "key_field": self._key_field_name,
            "normalize": self.normalize,
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "KeyBlocker[SchemaT]":
        """Rebuild a KeyBlocker from its serialized config.

        Args:
            config: A mapping with ``"schema_type_name"``, ``"key_field"``, and
                ``"normalize"`` (see :attr:`config`).

        Returns:
            A blocker equivalent to the one that produced ``config``.
        """
        schema = get_schema(str(config["schema_type_name"]))
        return cls(
            schema=schema,  # type: ignore[arg-type]
            key_field=str(config["key_field"]),
            normalize=bool(config["normalize"]),
        )

    def _extract_key(self, entity: SchemaT) -> str | None:
        """Extract and (optionally) normalize this entity's blocking key."""
        raw = self.key_fn(entity)
        if raw is None:
            return None
        return raw.strip().lower() if self.normalize else raw

    def stream(self, data: list[Any]) -> Iterator[ERCandidate[SchemaT]]:
        """Generate candidate pairs for records sharing a (normalized) key.

        Args:
            data: List of raw data items (typically dicts). The schema_factory
                transforms these into SchemaT objects.

        Yields:
            ERCandidate[SchemaT] objects for every pair of records within the
            same key bucket, in first-seen bucket order (blocker_name:
            "key_blocker").
        """
        entities = [self.schema_factory(record) for record in data]

        buckets: dict[str, list[SchemaT]] = {}
        order: list[str] = []
        for entity in entities:
            key = self._extract_key(entity)
            if key is None:
                continue
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            buckets[key].append(entity)

        for key in order:
            bucket = buckets[key]
            for i, left in enumerate(bucket):
                for right in bucket[i + 1 :]:
                    yield ERCandidate(left=left, right=right, blocker_name="key_blocker")

    def inspect_candidates(
        self,
        candidates: list[ERCandidate[SchemaT]],
        entities: list[SchemaT],
        sample_size: int = 10,
    ) -> CandidateInspectionReport:
        """Explore KeyBlocker candidates without ground truth.

        Unlike AllPairsBlocker, KeyBlocker's candidate distribution is NOT
        uniform (it depends on bucket sizes), so this computes the real
        per-entity candidate count from ``candidates`` rather than assuming
        n-1.

        Args:
            candidates: List of generated candidate pairs.
            entities: Original list of entities.
            sample_size: Number of example pairs to include in report.

        Returns:
            CandidateInspectionReport with totals, distribution, samples, and
            key-selectivity recommendations.
        """
        n = len(entities)
        total_candidates = len(candidates)
        avg_candidates_per_entity = (2 * total_candidates / n) if n > 0 else 0.0

        counts: dict[str, int] = {str(getattr(e, "id", id(e))): 0 for e in entities}
        for cand in candidates:
            left_id = str(getattr(cand.left, "id", id(cand.left)))
            right_id = str(getattr(cand.right, "id", id(cand.right)))
            counts[left_id] = counts.get(left_id, 0) + 1
            counts[right_id] = counts.get(right_id, 0) + 1

        distribution: dict[str, int] = {}
        for count in counts.values():
            key = str(count)
            distribution[key] = distribution.get(key, 0) + 1

        examples: list[dict[str, str]] = []
        for cand in candidates[:sample_size]:
            examples.append(
                {
                    "left_id": str(getattr(cand.left, "id", id(cand.left))),
                    "right_id": str(getattr(cand.right, "id", id(cand.right))),
                    "left_text": self._extract_text(cand.left),
                    "right_text": self._extract_text(cand.right),
                }
            )

        recommendations: list[str] = []
        if total_candidates == 0:
            recommendations.append(
                "⚠️ No candidates generated -- no two records share a (normalized) "
                "key. Check the key field/function, or compose with a "
                "recall-oriented blocker (e.g. VectorBlocker) via CompositeBlocker."
            )
        else:
            recommendations.append(
                f"✅ KeyBlocker generated {total_candidates:,} candidates "
                f"(avg {avg_candidates_per_entity:.1f} per entity, n={n}). "
                f"A coarse key (few, huge buckets) approaches AllPairs cost; a "
                f"fine key (many singleton buckets) risks missing near-matches "
                f"whose key values differ slightly."
            )

        return CandidateInspectionReport(
            total_candidates=total_candidates,
            avg_candidates_per_entity=avg_candidates_per_entity,
            candidate_distribution=distribution,
            examples=examples,
            recommendations=recommendations,
        )

    def _extract_text(self, entity: SchemaT) -> str:
        """Extract human-readable text from entity."""
        if hasattr(entity, "name"):
            return str(entity.name)
        return str(entity)
