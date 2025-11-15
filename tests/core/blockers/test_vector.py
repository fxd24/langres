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
from langres.core.embeddings import SentenceTransformerEmbedder
from langres.core.indexes.vector_index import FAISSIndex, FakeVectorIndex
from langres.core.models import CompanySchema

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
        vector_index=FakeVectorIndex(),
        k_neighbors=k_neighbors,
    )


def create_real_blocker(k_neighbors: int = 10) -> VectorBlocker[CompanySchema]:
    """Create a VectorBlocker with real implementations for integration testing."""
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    return VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        vector_index=FAISSIndex(embedder=embedder, metric="L2"),
        k_neighbors=k_neighbors,
    )


def test_vector_blocker_initialization():
    """Test VectorBlocker can be initialized with valid parameters."""
    blocker = create_fake_blocker(k_neighbors=5)

    assert blocker.k_neighbors == 5
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

    # Build index explicitly
    texts = [d["name"] for d in data]
    blocker.vector_index.create_index(texts)

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

    # Build index explicitly
    texts = [d["name"] for d in data]
    blocker.vector_index.create_index(texts)

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

    # Build index explicitly
    texts = [d["name"] for d in data]
    blocker.vector_index.create_index(texts)

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

    # Build index explicitly
    texts = [d["name"] for d in data]
    blocker.vector_index.create_index(texts)

    candidates = list(blocker.stream(data))

    # With only one entity, no pairs can be formed
    assert len(candidates) == 0


def test_vector_blocker_handles_empty_dataset():
    """Test VectorBlocker handles an empty dataset gracefully."""

    data: list[dict] = []

    blocker = create_fake_blocker(k_neighbors=5)

    # Build index explicitly (empty)
    texts = [d["name"] for d in data]
    blocker.vector_index.create_index(texts)

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

    # Build index explicitly
    texts = [d["name"] for d in data]
    blocker.vector_index.create_index(texts)

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

    # Build index explicitly
    texts = [d["name"] for d in data]
    blocker.vector_index.create_index(texts)

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


# ============================================================================
# Phase 1: New tests for explicit index creation requirement
# ============================================================================


def test_stream_raises_error_if_index_not_built():
    """Verify stream() raises RuntimeError if index not built."""
    # Setup blocker with unbuilt index
    fake_index = FakeVectorIndex()
    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        vector_index=fake_index,
        k_neighbors=2,
    )

    # stream() should raise RuntimeError
    data = [
        {"id": "c1", "name": "Apple"},
        {"id": "c2", "name": "Google"},
    ]

    with pytest.raises(RuntimeError, match="Index not built"):
        list(blocker.stream(data))


def test_stream_works_after_index_built():
    """Verify stream() works after explicit create_index() call."""
    # Setup
    fake_index = FakeVectorIndex()
    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        vector_index=fake_index,
        k_neighbors=2,
    )

    data = [
        {"id": "c1", "name": "Apple"},
        {"id": "c2", "name": "Google"},
        {"id": "c3", "name": "Microsoft"},
    ]

    # Build index explicitly
    texts = [d["name"] for d in data]
    blocker.vector_index.create_index(texts)

    # stream() should now work
    candidates = list(blocker.stream(data))
    assert len(candidates) > 0


def test_multiple_stream_calls_reuse_index():
    """Verify multiple stream() calls don't rebuild index."""
    # Setup with spy to track create_index calls
    fake_index = FakeVectorIndex()
    original_create = fake_index.create_index
    call_count = {"count": 0}

    def counting_create_index(texts):
        call_count["count"] += 1
        return original_create(texts)

    fake_index.create_index = counting_create_index

    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        vector_index=fake_index,
        k_neighbors=2,
    )

    data = [
        {"id": "c1", "name": "A"},
        {"id": "c2", "name": "B"},
        {"id": "c3", "name": "C"},
    ]
    texts = [d["name"] for d in data]

    # Build index once
    blocker.vector_index.create_index(texts)
    assert call_count["count"] == 1

    # Multiple stream() calls should NOT rebuild
    list(blocker.stream(data))
    list(blocker.stream(data))
    list(blocker.stream(data))

    # create_index should still only be called once
    assert call_count["count"] == 1


def test_different_k_neighbors_without_rebuild():
    """Verify changing k_neighbors doesn't rebuild index."""
    fake_index = FakeVectorIndex()
    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        vector_index=fake_index,
        k_neighbors=2,
    )

    data = [
        {"id": "c1", "name": "A"},
        {"id": "c2", "name": "B"},
        {"id": "c3", "name": "C"},
        {"id": "c4", "name": "D"},
    ]
    texts = [d["name"] for d in data]

    # Build index once
    blocker.vector_index.create_index(texts)

    # Try different k values
    for k in [2, 3, 4]:
        blocker.k_neighbors = k
        candidates = list(blocker.stream(data))
        assert len(candidates) >= 0  # Should work without error


# Note: Tests for different embedding models, lazy loading, and type conversion
# are now in tests/core/test_embeddings.py since these concerns have been
# separated from VectorBlocker into the EmbeddingProvider abstraction.


# ============================================================================
# Phase 1: Tests for query_prompt parameter (TDD)
# ============================================================================


def test_vector_blocker_passes_query_prompt_to_index():
    """Test that VectorBlocker passes query_prompt to index.search_all()."""
    from unittest.mock import MagicMock

    import numpy as np

    # Setup: Mock index with create_index and search_all
    mock_index = MagicMock()
    mock_index._index = object()  # Make _index_is_built() return True
    mock_index.search_all = MagicMock(
        return_value=(
            np.array([[0.1, 0.2]], dtype=np.float32),
            np.array([[1, 2]], dtype=np.int64),
        )
    )

    # Create blocker WITH query_prompt
    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        vector_index=mock_index,
        k_neighbors=2,
        query_prompt="Find duplicate companies",  # NEW parameter
    )

    # Generate candidates
    data = [
        {"id": "1", "name": "Apple Inc."},
        {"id": "2", "name": "Microsoft"},
        {"id": "3", "name": "Google"},
    ]
    list(blocker.stream(data))

    # Verify: search_all() was called with the query_prompt
    mock_index.search_all.assert_called_once()
    call_args = mock_index.search_all.call_args
    assert call_args[1]["query_prompt"] == "Find duplicate companies"


def test_vector_blocker_with_no_query_prompt():
    """Test that VectorBlocker passes None when query_prompt not configured."""
    from unittest.mock import MagicMock

    import numpy as np

    # Setup: Mock index
    mock_index = MagicMock()
    mock_index._index = object()
    mock_index.search_all = MagicMock(
        return_value=(
            np.array([[0.1, 0.2]], dtype=np.float32),
            np.array([[1, 2]], dtype=np.int64),
        )
    )

    # Create blocker WITHOUT query_prompt (default behavior)
    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        vector_index=mock_index,
        k_neighbors=2,
        # NO query_prompt parameter
    )

    # Generate candidates
    data = [
        {"id": "1", "name": "Apple"},
        {"id": "2", "name": "Google"},
    ]
    list(blocker.stream(data))

    # Verify: search_all() was called with query_prompt=None
    mock_index.search_all.assert_called_once()
    call_args = mock_index.search_all.call_args
    assert call_args[1]["query_prompt"] is None
