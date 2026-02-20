"""Tests for AllPairsBlocker (Approach 1: Classical String Matching).

The AllPairsBlocker is a schema-agnostic blocker that generates all N*(N-1)/2
candidate pairs from a dataset. It accepts a schema_factory callable to transform
raw dicts into any Pydantic schema type.
"""

from pydantic import BaseModel

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.models import CompanySchema, ERCandidate


# Test schema: Product (demonstrates schema-agnostic design)
class ProductSchema(BaseModel):
    """Product schema for testing schema-agnostic blocker."""

    id: str
    title: str
    price: float | None = None


def test_all_pairs_blocker_with_company_schema():
    """Test AllPairsBlocker generates all pairs with CompanySchema."""

    # Schema factory for companies
    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(
            id=record["id"],
            name=record["name"],
            address=record.get("address"),
            phone=record.get("phone"),
            website=record.get("website"),
        )

    blocker = AllPairsBlocker(schema_factory=company_factory)

    # Test data: 3 companies -> 3 pairs
    data = [
        {"id": "c1", "name": "Acme Corp", "address": "123 Main St"},
        {"id": "c2", "name": "TechStart", "address": "456 Oak Ave"},
        {"id": "c3", "name": "DataFlow", "address": "789 Pine Rd"},
    ]

    candidates = list(blocker.stream(data))

    # Should generate C(3,2) = 3 pairs
    assert len(candidates) == 3

    # Check all candidates are ERCandidate[CompanySchema]
    for candidate in candidates:
        assert isinstance(candidate, ERCandidate)
        assert isinstance(candidate.left, CompanySchema)
        assert isinstance(candidate.right, CompanySchema)
        assert candidate.blocker_name == "all_pairs_blocker"

    # Check expected pairs (order doesn't matter)
    pair_ids = {(c.left.id, c.right.id) for c in candidates}
    expected_pairs = {("c1", "c2"), ("c1", "c3"), ("c2", "c3")}
    assert pair_ids == expected_pairs


def test_all_pairs_blocker_with_product_schema():
    """Test AllPairsBlocker works with different schema (ProductSchema)."""

    # Schema factory for products
    def product_factory(record: dict) -> ProductSchema:
        return ProductSchema(
            id=record["product_id"],
            title=record["product_name"],
            price=record.get("price"),
        )

    blocker = AllPairsBlocker(schema_factory=product_factory)

    # Test data: 3 products -> 3 pairs
    data = [
        {"product_id": "p1", "product_name": "iPhone 15", "price": 999.99},
        {"product_id": "p2", "product_name": "Samsung Galaxy S24", "price": 899.99},
        {"product_id": "p3", "product_name": "Google Pixel 8", "price": 699.99},
    ]

    candidates = list(blocker.stream(data))

    # Should generate C(3,2) = 3 pairs
    assert len(candidates) == 3

    # Check all candidates are ERCandidate[ProductSchema]
    for candidate in candidates:
        assert isinstance(candidate, ERCandidate)
        assert isinstance(candidate.left, ProductSchema)
        assert isinstance(candidate.right, ProductSchema)
        assert candidate.blocker_name == "all_pairs_blocker"

    # Check expected pairs
    pair_ids = {(c.left.id, c.right.id) for c in candidates}
    expected_pairs = {("p1", "p2"), ("p1", "p3"), ("p2", "p3")}
    assert pair_ids == expected_pairs


def test_all_pairs_blocker_empty_input():
    """Test AllPairsBlocker handles empty input gracefully."""

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(id=record["id"], name=record["name"])

    blocker = AllPairsBlocker(schema_factory=company_factory)

    candidates = list(blocker.stream([]))
    assert len(candidates) == 0


def test_all_pairs_blocker_single_record():
    """Test AllPairsBlocker handles single record (no pairs)."""

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(id=record["id"], name=record["name"])

    blocker = AllPairsBlocker(schema_factory=company_factory)

    data = [{"id": "c1", "name": "Acme Corp"}]
    candidates = list(blocker.stream(data))
    assert len(candidates) == 0


def test_all_pairs_blocker_two_records():
    """Test AllPairsBlocker generates exactly one pair for two records."""

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(id=record["id"], name=record["name"])

    blocker = AllPairsBlocker(schema_factory=company_factory)

    data = [
        {"id": "c1", "name": "Acme Corp"},
        {"id": "c2", "name": "TechStart"},
    ]

    candidates = list(blocker.stream(data))
    assert len(candidates) == 1

    # Check the single pair
    candidate = candidates[0]
    assert candidate.left.id == "c1"
    assert candidate.right.id == "c2"
    assert candidate.blocker_name == "all_pairs_blocker"


def test_all_pairs_blocker_large_dataset():
    """Test AllPairsBlocker generates correct number of pairs for larger dataset."""

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(id=record["id"], name=record["name"])

    blocker = AllPairsBlocker(schema_factory=company_factory)

    # 10 records -> C(10,2) = 45 pairs
    data = [{"id": f"c{i}", "name": f"Company {i}"} for i in range(10)]

    candidates = list(blocker.stream(data))
    assert len(candidates) == 45


def test_all_pairs_blocker_streaming_behavior():
    """Test AllPairsBlocker returns a generator (streaming)."""

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(id=record["id"], name=record["name"])

    blocker = AllPairsBlocker(schema_factory=company_factory)

    data = [
        {"id": "c1", "name": "Acme Corp"},
        {"id": "c2", "name": "TechStart"},
    ]

    # stream() should return a generator, not a list
    result = blocker.stream(data)
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")


def test_all_pairs_blocker_preserves_schema_fields():
    """Test AllPairsBlocker preserves all schema fields correctly."""

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(
            id=record["id"],
            name=record["name"],
            address=record.get("address"),
            phone=record.get("phone"),
            website=record.get("website"),
        )

    blocker = AllPairsBlocker(schema_factory=company_factory)

    data = [
        {
            "id": "c1",
            "name": "Acme Corp",
            "address": "123 Main St",
            "phone": "+1-555-0100",
            "website": "https://acme.com",
        },
        {
            "id": "c2",
            "name": "TechStart",
            "address": "456 Oak Ave",
            "phone": "+1-555-0200",
            # website is None
        },
    ]

    candidates = list(blocker.stream(data))
    assert len(candidates) == 1

    # Verify all fields are preserved
    candidate = candidates[0]
    assert candidate.left.name == "Acme Corp"
    assert candidate.left.address == "123 Main St"
    assert candidate.left.phone == "+1-555-0100"
    assert candidate.left.website == "https://acme.com"

    assert candidate.right.name == "TechStart"
    assert candidate.right.address == "456 Oak Ave"
    assert candidate.right.phone == "+1-555-0200"
    assert candidate.right.website is None


def test_all_pairs_blocker_no_duplicate_pairs():
    """Test AllPairsBlocker generates each unique pair exactly once."""

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(id=record["id"], name=record["name"])

    blocker = AllPairsBlocker(schema_factory=company_factory)

    data = [{"id": f"c{i}", "name": f"Company {i}"} for i in range(5)]

    candidates = list(blocker.stream(data))

    # Check for duplicates (both (a,b) and (b,a) would be duplicates)
    pair_ids = [(c.left.id, c.right.id) for c in candidates]
    reversed_pairs = [(c.right.id, c.left.id) for c in candidates]

    # No reversed pairs should exist in the original list
    assert len(set(pair_ids)) == len(pair_ids)
    assert all(rp not in pair_ids for rp in reversed_pairs)


def test_all_pairs_blocker_consistent_ordering():
    """Test AllPairsBlocker maintains consistent left < right ordering."""

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(id=record["id"], name=record["name"])

    blocker = AllPairsBlocker(schema_factory=company_factory)

    data = [{"id": f"c{i}", "name": f"Company {i}"} for i in range(5)]

    candidates = list(blocker.stream(data))

    # All pairs should have left appearing before right in original data
    # (based on index, not ID string comparison)
    data_ids = [d["id"] for d in data]
    for candidate in candidates:
        left_idx = data_ids.index(candidate.left.id)
        right_idx = data_ids.index(candidate.right.id)
        assert left_idx < right_idx
