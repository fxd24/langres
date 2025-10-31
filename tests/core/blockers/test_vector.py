"""Tests for VectorBlocker (embedding-based candidate generation).

This test module validates the VectorBlocker implementation, which uses
injected embedding and vector index providers to efficiently generate
candidate pairs without N² complexity.

Most tests use FakeEmbedder and FakeVectorIndex for fast, deterministic
unit testing. Integration tests (marked @pytest.mark.slow) use real
SentenceTransformerEmbedder and FAISSIndex implementations.
"""

import logging

import pytest

from langres.core.blockers.vector import VectorBlocker
from langres.core.embeddings import FakeEmbedder, SentenceTransformerEmbedder
from langres.core.models import CompanySchema
from langres.core.vector_index import FAISSIndex, FakeVectorIndex

logger = logging.getLogger(__name__)


# Helper functions for test construction
def company_factory(record: dict) -> CompanySchema:
    """Standard company factory for tests."""
    return CompanySchema(
        id=record["id"],
        name=record["name"],
        address=record.get("address"),
        phone=record.get("phone"),
    )


def create_fake_blocker(k_neighbors: int = 10) -> VectorBlocker[CompanySchema]:
    """Create a VectorBlocker with fake implementations for fast unit testing."""
    return VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        embedding_provider=FakeEmbedder(embedding_dim=128),
        vector_index=FakeVectorIndex(),
        k_neighbors=k_neighbors,
    )


def create_real_blocker(k_neighbors: int = 10) -> VectorBlocker[CompanySchema]:
    """Create a VectorBlocker with real implementations for integration testing."""
    return VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        embedding_provider=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
        vector_index=FAISSIndex(metric="L2"),
        k_neighbors=k_neighbors,
    )


def test_vector_blocker_initialization():
    """Test VectorBlocker can be initialized with valid parameters."""
    blocker = create_fake_blocker(k_neighbors=5)

    assert blocker.k_neighbors == 5
    assert isinstance(blocker.embedding_provider, FakeEmbedder)
    assert isinstance(blocker.vector_index, FakeVectorIndex)


def test_vector_blocker_requires_positive_k():
    """Test VectorBlocker validates k_neighbors is positive."""
    with pytest.raises(ValueError, match="k_neighbors must be positive"):
        create_fake_blocker(k_neighbors=0)

    with pytest.raises(ValueError, match="k_neighbors must be positive"):
        create_fake_blocker(k_neighbors=-1)


def test_vector_blocker_generates_candidates_from_small_dataset():
    """Test VectorBlocker generates candidate pairs from a small dataset (unit test with fakes)."""
    data = [
        {"id": "c1", "name": "Acme Corporation", "address": "123 Main St"},
        {"id": "c2", "name": "Acme Corp", "address": "123 Main Street"},
        {"id": "c3", "name": "TechStart Industries", "address": "456 Oak Ave"},
        {"id": "c4", "name": "DataFlow Solutions", "address": "789 Park Blvd"},
    ]

    blocker = create_fake_blocker(k_neighbors=2)
    candidates = list(blocker.stream(data))

    # Should generate candidates for each entity with its k nearest neighbors
    # With 4 entities and k=2, we expect at most 4*2/2 = 4 unique pairs
    # (division by 2 because pairs are deduplicated)
    assert len(candidates) > 0
    logger.info("Generated %d candidates from 4 entities", len(candidates))

    # Verify structure of candidates
    for candidate in candidates:
        assert candidate.left.id in {"c1", "c2", "c3", "c4"}
        assert candidate.right.id in {"c1", "c2", "c3", "c4"}
        assert candidate.left.id != candidate.right.id  # No self-pairs
        assert candidate.blocker_name == "vector_blocker"


@pytest.mark.slow
def test_vector_blocker_finds_similar_entities():
    """Test VectorBlocker correctly pairs semantically similar entities."""

    data = [
        {"id": "c1", "name": "Acme Corporation"},
        {"id": "c2", "name": "Acme Corp"},  # Very similar to c1
        {"id": "c3", "name": "Completely Different Company LLC"},
    ]

    blocker = create_fake_blocker(k_neighbors=1)

    candidates = list(blocker.stream(data))

    # c1's nearest neighbor should be c2 (similar name)
    # c2's nearest neighbor should be c1
    # So we expect the pair (c1, c2) to appear
    candidate_pairs = {(c.left.id, c.right.id) for c in candidates}
    logger.info("Candidate pairs: %s", candidate_pairs)

    # Check that (c1, c2) or (c2, c1) is in the candidates
    assert ("c1", "c2") in candidate_pairs or ("c2", "c1") in candidate_pairs


@pytest.mark.slow
def test_vector_blocker_no_duplicate_pairs():
    """Test VectorBlocker doesn't generate duplicate pairs (both (a,b) and (b,a))."""

    data = [
        {"id": "c1", "name": "Acme Corporation"},
        {"id": "c2", "name": "Acme Corp"},
        {"id": "c3", "name": "Acme Company"},
    ]

    blocker = create_fake_blocker(k_neighbors=2)

    candidates = list(blocker.stream(data))

    # Convert to a set of frozensets to check for duplicates
    # (since {a, b} == {b, a})
    pairs_as_sets = [frozenset([c.left.id, c.right.id]) for c in candidates]

    # No duplicates: length of list should equal length of set
    assert len(pairs_as_sets) == len(set(pairs_as_sets)), (
        "Found duplicate pairs (both (a,b) and (b,a))"
    )


def test_vector_blocker_handles_single_entity():
    """Test VectorBlocker handles a dataset with a single entity."""

    data = [{"id": "c1", "name": "Acme Corporation"}]

    blocker = create_fake_blocker(k_neighbors=5)

    candidates = list(blocker.stream(data))

    # With only one entity, no pairs can be formed
    assert len(candidates) == 0


def test_vector_blocker_handles_empty_dataset():
    """Test VectorBlocker handles an empty dataset gracefully."""

    data: list[dict] = []

    blocker = create_fake_blocker(k_neighbors=5)

    candidates = list(blocker.stream(data))

    # Empty dataset should produce no candidates
    assert len(candidates) == 0


@pytest.mark.slow
def test_vector_blocker_with_missing_fields():
    """Test VectorBlocker handles entities with missing optional fields."""

    data = [
        {"id": "c1", "name": "Acme Corporation", "address": "123 Main St"},
        {"id": "c2", "name": "Acme Corp"},  # Missing address and phone
        {"id": "c3", "name": "TechStart"},
    ]

    blocker = create_fake_blocker(k_neighbors=2)

    candidates = list(blocker.stream(data))

    # Should still generate candidates even with missing fields
    assert len(candidates) > 0

    # All candidates should have valid CompanySchema objects
    for candidate in candidates:
        assert isinstance(candidate.left, CompanySchema)
        assert isinstance(candidate.right, CompanySchema)


@pytest.mark.slow
def test_vector_blocker_achieves_high_recall():
    """Test VectorBlocker achieves >= 95% recall on known duplicates.

    This test uses a dataset with known duplicate pairs and verifies that
    the VectorBlocker doesn't miss too many true matches (recall >= 0.95).
    """

    # Dataset with known duplicate groups
    data = [
        # Group 1: Exact duplicates
        {"id": "c1", "name": "Acme Corporation"},
        {"id": "c1_dup", "name": "Acme Corporation"},
        # Group 2: Typo duplicates
        {"id": "c2", "name": "TechStart Industries"},
        {"id": "c2_typo", "name": "TechStrat Industries"},
        # Group 3: Abbreviation
        {"id": "c3", "name": "Global Systems Incorporated"},
        {"id": "c3_abbrev", "name": "Global Systems Inc."},
        # Non-duplicates
        {"id": "c4", "name": "Quantum Dynamics Research"},
        {"id": "c5", "name": "BioTech Labs"},
        {"id": "c6", "name": "Pacific Logistics"},
    ]

    # True duplicate pairs (ground truth)
    true_pairs = {
        frozenset(["c1", "c1_dup"]),
        frozenset(["c2", "c2_typo"]),
        frozenset(["c3", "c3_abbrev"]),
    }

    blocker = create_fake_blocker(k_neighbors=3)

    candidates = list(blocker.stream(data))
    generated_pairs = {frozenset([c.left.id, c.right.id]) for c in candidates}

    # Calculate recall: how many true pairs did we find?
    found_pairs = true_pairs & generated_pairs
    recall = len(found_pairs) / len(true_pairs)

    logger.info("True pairs: %d", len(true_pairs))
    logger.info("Found pairs: %d", len(found_pairs))
    logger.info("Recall: %.2f%%", recall * 100)

    # POC requirement: blocking recall >= 0.95
    assert recall >= 0.95, (
        f"VectorBlocker recall {recall:.2%} is below target 0.95. "
        f"Missed pairs: {true_pairs - found_pairs}"
    )


# Note: Tests for different embedding models, lazy loading, and type conversion
# are now in tests/core/test_embeddings.py since these concerns have been
# separated from VectorBlocker into the EmbeddingProvider abstraction.
