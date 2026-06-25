"""AllPairsBlocker implementation for naive all-pairs candidate generation.

This blocker generates all N*(N-1)/2 possible pairs from a dataset. It is
schema-agnostic, accepting a schema_factory callable to transform raw dicts
into any Pydantic schema type.
"""

from collections.abc import Callable, Iterator
from typing import Any

from pydantic import BaseModel

from langres.core.blocker import Blocker, SchemaT
from langres.core.models import ERCandidate
from langres.core.registry import get_schema, register, register_schema
from langres.core.reports import CandidateInspectionReport


def register_schema_idempotent(schema: type[BaseModel]) -> str:
    """Register ``schema`` under its class name, tolerating repeat calls.

    A declarative ``schema=`` blocker must register its schema so the name can
    round-trip through a config. Multiple blockers may share one schema, and the
    plain :func:`register_schema` decorator raises on duplicates, so this helper
    registers only once and returns the registry key on subsequent calls.

    Args:
        schema: The Pydantic schema class to register.

    Returns:
        The registry name (the schema's ``__name__``).

    Raises:
        ValueError: If a *different* class is already registered under this name
            (a real collision, not a benign repeat).
    """
    name = schema.__name__
    try:
        existing = get_schema(name)
    except KeyError:
        register_schema(name)(schema)
        return name
    if existing is not schema:
        raise ValueError(
            f"Schema name '{name}' is already registered to a different class "
            f"({existing!r}); cannot register {schema!r} under the same name."
        )
    return name


def schema_to_factory(schema: type[SchemaT]) -> Callable[[dict[str, Any]], SchemaT]:
    """Build a record->entity factory from a declarative schema.

    Reconstructs entities by selecting only the fields the schema declares:
    ``schema(**{f: record.get(f) for f in schema.model_fields})``. Record keys
    not on the schema are ignored.

    Args:
        schema: The Pydantic schema class to instantiate.

    Returns:
        A factory callable suitable for the ``schema_factory`` slot.
    """

    def factory(record: dict[str, Any]) -> SchemaT:
        return schema(**{field: record.get(field) for field in schema.model_fields})

    return factory


@register("all_pairs_blocker")
class AllPairsBlocker(Blocker[SchemaT]):
    """Schema-agnostic blocker that generates all N*(N-1)/2 candidate pairs.

    This is a naive blocker that generates all possible pairs without any
    blocking strategy. It's useful for small datasets or as a baseline for
    benchmarking more sophisticated blocking techniques.

    The blocker is schema-agnostic: it works with ANY Pydantic schema by
    accepting a schema_factory callable that transforms raw dicts into
    the target schema type.

    Example:
        # For companies
        def company_factory(record: dict) -> CompanySchema:
            return CompanySchema(
                id=record["id"],
                name=record["name"],
                address=record.get("address")
            )

        blocker = AllPairsBlocker(schema_factory=company_factory)
        candidates = blocker.stream(company_records)

        # For products (different schema, same blocker!)
        def product_factory(record: dict) -> ProductSchema:
            return ProductSchema(
                id=record["product_id"],
                title=record["product_name"]
            )

        blocker = AllPairsBlocker(schema_factory=product_factory)
        candidates = blocker.stream(product_records)

    Note:
        This blocker has O(N²) complexity and doesn't scale well to large
        datasets. For production use cases with >10k records, use blocking
        techniques like:
        - Blocking keys (group by attributes)
        - ANN search (embedding similarity)
        - Sorted neighborhood
        - LSH (Locality-Sensitive Hashing)
    """

    def __init__(
        self,
        schema_factory: Callable[[dict[str, Any]], SchemaT] | None = None,
        schema: type[SchemaT] | None = None,
    ):
        """Initialize AllPairsBlocker.

        Provide exactly one of ``schema`` or ``schema_factory``:

        - ``schema=`` (declarative): the blocker reconstructs entities as
          ``schema(**{f: record.get(f) for f in schema.model_fields})`` and is
          **config-serializable** (its schema is referenced by name).
        - ``schema_factory=`` (callable): full control over normalization, but
          **not serializable** (a callable cannot round-trip through JSON).

        Args:
            schema_factory: Callable that transforms a raw dict into a Pydantic
                schema object (SchemaT). Mutually exclusive with ``schema``.
            schema: Pydantic schema class for declarative reconstruction.
                Mutually exclusive with ``schema_factory``.

        Raises:
            ValueError: If both or neither of ``schema``/``schema_factory`` are
                given.
        """
        if (schema is None) == (schema_factory is None):
            raise ValueError(
                "AllPairsBlocker requires exactly one of 'schema' or "
                "'schema_factory' (got both or neither)."
            )

        # Track whether this blocker was built declaratively (serializable) or
        # from an opaque callable (not serializable).
        self._schema_type_name: str | None = None
        if schema is not None:
            self._schema_type_name = register_schema_idempotent(schema)
            self.schema_factory = schema_to_factory(schema)
        else:
            assert schema_factory is not None  # narrowed by the guard above
            self.schema_factory = schema_factory

    @property
    def config(self) -> dict[str, object]:
        """Serializable construction config for the registry.

        Returns:
            ``{"schema_type_name": <name>}`` — the registry key of the schema,
            sufficient to rebuild the blocker via :meth:`from_config`.

        Raises:
            ValueError: If this blocker was constructed with ``schema_factory``
                (an opaque callable that cannot be serialized). Construct with
                ``schema=`` to persist.
        """
        if self._schema_type_name is None:
            raise ValueError(
                "AllPairsBlocker built with 'schema_factory' is not "
                "serializable (a callable cannot round-trip through config); "
                "construct with schema= to persist."
            )
        return {"schema_type_name": self._schema_type_name}

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "AllPairsBlocker[SchemaT]":
        """Rebuild an AllPairsBlocker from its serialized config.

        Args:
            config: A mapping with ``"schema_type_name"`` naming a registered
                schema (see :func:`langres.core.registry.register_schema`).

        Returns:
            A blocker equivalent to the one that produced ``config``.
        """
        schema = get_schema(str(config["schema_type_name"]))
        return cls(schema=schema)  # type: ignore[arg-type]

    def stream(self, data: list[Any]) -> Iterator[ERCandidate[SchemaT]]:
        """Generate all N*(N-1)/2 candidate pairs from input data.

        Args:
            data: List of raw data items (typically dicts). The schema_factory
                transforms these into SchemaT objects.

        Yields:
            ERCandidate[SchemaT] objects containing:
            - left: Normalized entity (SchemaT)
            - right: Normalized entity (SchemaT)
            - blocker_name: "all_pairs_blocker"

        Note:
            This implementation maintains consistent ordering: for each pair
            (i, j) where i < j in the original data, left is data[i] and
            right is data[j]. This ensures no duplicate pairs (both (a,b)
            and (b,a)) are generated.
        """
        # 1. Normalize schema: transform raw dicts to SchemaT
        entities = [self.schema_factory(record) for record in data]

        # 2. Generate all pairs with i < j (no duplicates)
        for i, left in enumerate(entities):
            for right in entities[i + 1 :]:
                yield ERCandidate(left=left, right=right, blocker_name="all_pairs_blocker")

    def inspect_candidates(
        self,
        candidates: list[ERCandidate[SchemaT]],
        entities: list[SchemaT],
        sample_size: int = 10,
    ) -> CandidateInspectionReport:
        """Explore AllPairs candidates without ground truth.

        AllPairsBlocker generates all possible pairs (n*(n-1)/2 candidates).
        This method helps users understand scalability implications.

        Args:
            candidates: List of generated candidate pairs
            entities: Original list of entities
            sample_size: Number of example pairs to include in report

        Returns:
            CandidateInspectionReport with:
            - Total candidates and avg per entity
            - Uniform distribution (all entities have n-1 candidates)
            - Sample pairs with readable text
            - Scalability recommendations based on dataset size
        """
        n = len(entities)

        # Statistics
        total_candidates = len(candidates)
        avg_candidates_per_entity = float(n - 1) if n > 0 else 0.0

        # Distribution (all-pairs is uniform: every entity has exactly n-1 candidates)
        distribution: dict[str, int] = {}
        if n > 0:
            key = str(n - 1)  # All entities have same count
            distribution[key] = n

        # Sample candidates with readable text
        examples: list[dict[str, str]] = []
        for cand in candidates[:sample_size]:
            # Extract IDs and text from candidate entities
            left_id = str(getattr(cand.left, "id", id(cand.left)))
            right_id = str(getattr(cand.right, "id", id(cand.right)))

            left_text = self._extract_text(cand.left)
            right_text = self._extract_text(cand.right)

            examples.append(
                {
                    "left_id": left_id,
                    "right_id": right_id,
                    "left_text": left_text,
                    "right_text": right_text,
                }
            )

        # Recommendations based on dataset size
        recommendations: list[str] = []
        if n > 100:
            recommendations.append(
                f"⚠️ AllPairsBlocker not scalable for large datasets (n={n}, {total_candidates:,} pairs). "
                f"Use VectorBlocker for O(n*k) complexity instead of O(n²)."
            )
        elif n >= 50:
            recommendations.append(
                f"AllPairsBlocker feasible but consider VectorBlocker (n={n}, {total_candidates:,} pairs). "
                f"VectorBlocker reduces candidates while maintaining recall."
            )
        else:
            recommendations.append(
                f"✅ AllPairsBlocker appropriate for small dataset (n={n}, {total_candidates} pairs). "
                f"Guarantees 100% recall with exhaustive pairwise comparison."
            )

        return CandidateInspectionReport(
            total_candidates=total_candidates,
            avg_candidates_per_entity=avg_candidates_per_entity,
            candidate_distribution=distribution,
            examples=examples,
            recommendations=recommendations,
        )

    def _extract_text(self, entity: SchemaT) -> str:
        """Extract human-readable text from entity.

        Args:
            entity: Entity to extract text from

        Returns:
            Readable text representation of entity
        """
        if hasattr(entity, "name"):
            return str(entity.name)
        else:
            # Fall back to string representation
            return str(entity)
