"""Tests for VectorBlocker (embedding-based candidate generation).

This test module validates the VectorBlocker implementation, which uses
sentence-transformers for embeddings and FAISS for ANN search to efficiently
generate candidate pairs without N² complexity.
"""

import logging

import pytest

from langres.core.blockers.vector import VectorBlocker
from langres.core.models import CompanySchema

logger = logging.getLogger(__name__)


def test_vector_blocker_initialization():
    """Test VectorBlocker can be initialized with valid parameters."""

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(
            id=record["id"],
            name=record["name"],
            address=record.get("address"),
        )

    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        k_neighbors=5,
        model_name="all-MiniLM-L6-v2",
    )

    assert blocker.k_neighbors == 5
    assert blocker.model_name == "all-MiniLM-L6-v2"


def test_vector_blocker_requires_positive_k():
    """Test VectorBlocker validates k_neighbors is positive."""

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(id=record["id"], name=record["name"])

    with pytest.raises(ValueError, match="k_neighbors must be positive"):
        VectorBlocker(
            schema_factory=company_factory,
            text_field_extractor=lambda x: x.name,
            k_neighbors=0,
        )

    with pytest.raises(ValueError, match="k_neighbors must be positive"):
        VectorBlocker(
            schema_factory=company_factory,
            text_field_extractor=lambda x: x.name,
            k_neighbors=-1,
        )


@pytest.mark.slow
def test_vector_blocker_generates_candidates_from_small_dataset():
    """Test VectorBlocker generates candidate pairs from a small dataset."""

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(
            id=record["id"],
            name=record["name"],
            address=record.get("address"),
        )

    data = [
        {"id": "c1", "name": "Acme Corporation", "address": "123 Main St"},
        {"id": "c2", "name": "Acme Corp", "address": "123 Main Street"},
        {"id": "c3", "name": "TechStart Industries", "address": "456 Oak Ave"},
        {"id": "c4", "name": "DataFlow Solutions", "address": "789 Park Blvd"},
    ]

    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        k_neighbors=2,  # Only get 2 nearest neighbors
    )

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

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(id=record["id"], name=record["name"])

    data = [
        {"id": "c1", "name": "Acme Corporation"},
        {"id": "c2", "name": "Acme Corp"},  # Very similar to c1
        {"id": "c3", "name": "Completely Different Company LLC"},
    ]

    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        k_neighbors=1,  # Only find the single nearest neighbor
    )

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

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(id=record["id"], name=record["name"])

    data = [
        {"id": "c1", "name": "Acme Corporation"},
        {"id": "c2", "name": "Acme Corp"},
        {"id": "c3", "name": "Acme Company"},
    ]

    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        k_neighbors=2,
    )

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

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(id=record["id"], name=record["name"])

    data = [{"id": "c1", "name": "Acme Corporation"}]

    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        k_neighbors=5,
    )

    candidates = list(blocker.stream(data))

    # With only one entity, no pairs can be formed
    assert len(candidates) == 0


def test_vector_blocker_handles_empty_dataset():
    """Test VectorBlocker handles an empty dataset gracefully."""

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(id=record["id"], name=record["name"])

    data: list[dict] = []

    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        k_neighbors=5,
    )

    candidates = list(blocker.stream(data))

    # Empty dataset should produce no candidates
    assert len(candidates) == 0


@pytest.mark.slow
def test_vector_blocker_with_missing_fields():
    """Test VectorBlocker handles entities with missing optional fields."""

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(
            id=record["id"],
            name=record["name"],
            address=record.get("address"),
            phone=record.get("phone"),
        )

    data = [
        {"id": "c1", "name": "Acme Corporation", "address": "123 Main St"},
        {"id": "c2", "name": "Acme Corp"},  # Missing address and phone
        {"id": "c3", "name": "TechStart"},
    ]

    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        k_neighbors=2,
    )

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

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(id=record["id"], name=record["name"])

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

    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        k_neighbors=3,  # Should be enough to find similar pairs
    )

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


@pytest.mark.slow
def test_vector_blocker_different_model():
    """Test VectorBlocker can be initialized with a different embedding model."""

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(id=record["id"], name=record["name"])

    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        k_neighbors=5,
        model_name="paraphrase-MiniLM-L3-v2",  # Different model
    )

    assert blocker.model_name == "paraphrase-MiniLM-L3-v2"

    # Should still work with a different model
    data = [
        {"id": "c1", "name": "Acme Corporation"},
        {"id": "c2", "name": "TechStart Industries"},
    ]

    candidates = list(blocker.stream(data))
    assert len(candidates) >= 0  # Should not crash


@pytest.mark.slow
def test_vector_blocker_model_lazy_loading():
    """Test that the embedding model is lazy-loaded on first use."""

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(id=record["id"], name=record["name"])

    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        k_neighbors=2,
    )

    # Model should not be loaded yet
    assert blocker._model is None

    # First call should trigger model loading
    data = [
        {"id": "c1", "name": "Test Company 1"},
        {"id": "c2", "name": "Test Company 2"},
    ]

    list(blocker.stream(data))

    # Model should now be loaded
    assert blocker._model is not None

    # Second call should reuse loaded model (tests branch 129->132)
    model_ref = blocker._model
    list(blocker.stream(data))
    assert blocker._model is model_ref  # Same model instance


@pytest.mark.slow
def test_vector_blocker_handles_non_numpy_embeddings(mocker):
    """Test that VectorBlocker handles embeddings that aren't numpy arrays.

    This tests the defensive conversion at line 180 where embeddings
    are converted to numpy array if model.encode() returns a list.
    """

    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(id=record["id"], name=record["name"])

    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        k_neighbors=2,
    )

    data = [
        {"id": "c1", "name": "Test Company 1"},
        {"id": "c2", "name": "Test Company 2"},
        {"id": "c3", "name": "Test Company 3"},
    ]

    # Mock the model's encode method to return a list instead of numpy array
    # This triggers the conversion at line 180
    mock_encode = mocker.patch.object(
        blocker._get_model(),
        "encode",
        return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],  # List, not np.ndarray
    )

    # Should handle the list and convert to numpy array
    candidates = list(blocker.stream(data))

    # Should generate candidates despite receiving a list
    assert len(candidates) > 0
    mock_encode.assert_called_once()
