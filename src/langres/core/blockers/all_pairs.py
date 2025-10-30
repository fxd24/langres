"""
AllPairsBlocker: Naive blocker that generates all possible pairs.

This is the simplest blocking strategy (Approach 1: Classical String Matching).
It generates all O(N²) pairs without any optimization, which is suitable for
small datasets or as a baseline for comparison.

For larger datasets (>1000 records), more sophisticated blocking strategies
should be used (e.g., blocking keys, ANN search, sorted neighborhood).
"""

from collections.abc import Iterator
from typing import Any

from langres.core.blocker import Blocker
from langres.core.models import CompanySchema, ERCandidate


class AllPairsBlocker(Blocker[CompanySchema]):
    """Naive blocker that generates all possible pairs (no blocking).

    This blocker implements the simplest possible blocking strategy:
    enumerate all N*(N-1)/2 pairs from the input data.

    Responsibilities:
    1. Schema Normalization: Convert raw dicts to CompanySchema
    2. Pair Generation: Enumerate all pairs (i < j)

    Performance:
    - Time Complexity: O(N²)
    - Space Complexity: O(1) (streaming/lazy evaluation)
    - Suitable for: Small datasets (<1000 records)

    Example:
        blocker = AllPairsBlocker()
        data = [
            {"id": "c1", "name": "Acme Corp", "address": "123 Main St"},
            {"id": "c2", "name": "TechStart Inc", "phone": "+1-555-0100"}
        ]

        for candidate in blocker.stream(data):
            # candidate is ERCandidate[CompanySchema]
            print(f"Pair: {candidate.left.name} <-> {candidate.right.name}")
    """

    def stream(self, data: list[Any]) -> Iterator[ERCandidate[CompanySchema]]:
        """Generate all possible pairs from input data.

        Args:
            data: List of raw company records (dicts with keys: id, name, address?, phone?, website?)

        Yields:
            ERCandidate[CompanySchema] for each pair (i < j)

        Note:
            Missing optional fields (address, phone, website) are set to None.
            Missing required fields (id, name) will raise Pydantic validation errors.
        """
        # Step 1: Normalize schema (raw dicts → CompanySchema)
        companies = [
            CompanySchema(
                id=record["id"],
                name=record["name"],
                address=record.get("address"),
                phone=record.get("phone"),
                website=record.get("website"),
            )
            for record in data
        ]

        # Step 2: Generate all pairs (i < j to avoid duplicates and self-pairs)
        for i, left in enumerate(companies):
            for right in companies[i + 1 :]:
                yield ERCandidate(
                    left=left,
                    right=right,
                    blocker_name="all_pairs",
                )
