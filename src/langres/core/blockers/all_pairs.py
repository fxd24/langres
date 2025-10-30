"""AllPairsBlocker implementation for naive all-pairs candidate generation.

This blocker generates all N*(N-1)/2 possible pairs from a dataset. It is
schema-agnostic, accepting a schema_factory callable to transform raw dicts
into any Pydantic schema type.
"""

from collections.abc import Callable, Iterator
from typing import Any


from langres.core.blocker import Blocker, SchemaT
from langres.core.models import ERCandidate


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

    def __init__(self, schema_factory: Callable[[dict[str, Any]], SchemaT]):
        """Initialize AllPairsBlocker.

        Args:
            schema_factory: Callable that transforms a raw dict into a
                Pydantic schema object (SchemaT). This allows the blocker
                to work with any schema type.
        """
        self.schema_factory = schema_factory

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
