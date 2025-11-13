"""Tests for VectorBlocker similarity score population.

This test suite validates that VectorBlocker.stream() populates the
similarity_score field in ERCandidate objects, enabling ranking evaluation.
"""

import logging

from langres.core.blockers.vector import VectorBlocker
from langres.core.models import CompanySchema
from langres.core.vector_index import FakeVectorIndex

logger = logging.getLogger(__name__)


def test_vector_blocker_populates_similarity_scores() -> None:
    """Test that VectorBlocker.stream() populates similarity scores in candidates.

    This is critical for ranking evaluation - we need to know HOW SIMILAR
    each candidate pair is according to the vector index, not just whether
    they're candidates.
    """
    # Setup test data
    entities = [
        {"id": "c1", "name": "Apple Inc"},
        {"id": "c2", "name": "Apple Incorporated"},
        {"id": "c3", "name": "Microsoft Corp"},
        {"id": "c4", "name": "Microsoft Corporation"},
        {"id": "c5", "name": "Google LLC"},
    ]

    # Setup VectorBlocker with FakeVectorIndex
    fake_index = FakeVectorIndex()
    blocker = VectorBlocker(
        schema_factory=lambda x: CompanySchema(**x),
        text_field_extractor=lambda x: x.name,
        vector_index=fake_index,
        k_neighbors=3,
    )

    # Build index
    texts = [e["name"] for e in entities]
    fake_index.create_index(texts)

    # Generate candidates
    candidates = list(blocker.stream(entities))

    # Verify all candidates have similarity scores populated
    assert len(candidates) > 0, "Should generate at least one candidate"

    for candidate in candidates:
        assert candidate.similarity_score is not None, (
            f"Candidate ({candidate.left.id}, {candidate.right.id}) missing similarity_score"
        )
        logger.info(
            f"Candidate ({candidate.left.id}, {candidate.right.id}): "
            f"similarity_score={candidate.similarity_score:.4f}"
        )


def test_vector_blocker_scores_in_valid_range() -> None:
    """Test that similarity scores are in valid range [0, 1].

    The similarity score should be normalized to [0, 1], where 1.0 means
    perfect match and 0.0 means no similarity.
    """
    entities = [
        {"id": "c1", "name": "Apple Inc"},
        {"id": "c2", "name": "Apple Incorporated"},
        {"id": "c3", "name": "Totally Different Company"},
    ]

    fake_index = FakeVectorIndex()
    blocker = VectorBlocker(
        schema_factory=lambda x: CompanySchema(**x),
        text_field_extractor=lambda x: x.name,
        vector_index=fake_index,
        k_neighbors=2,
    )

    texts = [e["name"] for e in entities]
    fake_index.create_index(texts)

    candidates = list(blocker.stream(entities))

    for candidate in candidates:
        assert candidate.similarity_score is not None
        assert 0.0 <= candidate.similarity_score <= 1.0, (
            f"similarity_score {candidate.similarity_score} out of range [0, 1]"
        )
        logger.info(
            f"Candidate ({candidate.left.id}, {candidate.right.id}): "
            f"similarity_score={candidate.similarity_score:.4f} (valid)"
        )


def test_vector_blocker_scores_ranked_descending() -> None:
    """Test that candidates are yielded in descending order of similarity.

    For ranking evaluation to work well, the blocker should yield better
    matches first (higher similarity scores). This enables downstream
    systems to process the most promising candidates first.
    """
    entities = [
        {"id": "c1", "name": "Apple Inc"},
        {"id": "c2", "name": "Apple Incorporated"},
        {"id": "c3", "name": "Microsoft Corp"},
        {"id": "c4", "name": "Google LLC"},
    ]

    fake_index = FakeVectorIndex()
    blocker = VectorBlocker(
        schema_factory=lambda x: CompanySchema(**x),
        text_field_extractor=lambda x: x.name,
        vector_index=fake_index,
        k_neighbors=3,
    )

    texts = [e["name"] for e in entities]
    fake_index.create_index(texts)

    candidates = list(blocker.stream(entities))

    # For each entity, verify its candidates are in descending order
    # Group candidates by left entity
    entity_candidates: dict[str, list[tuple[str, float]]] = {}
    for candidate in candidates:
        left_id = candidate.left.id
        right_id = candidate.right.id
        score = candidate.similarity_score

        assert score is not None

        if left_id not in entity_candidates:
            entity_candidates[left_id] = []
        entity_candidates[left_id].append((right_id, score))

    # Check each entity's candidates are sorted descending
    for entity_id, cand_list in entity_candidates.items():
        scores = [score for _, score in cand_list]
        assert scores == sorted(scores, reverse=True), (
            f"Entity {entity_id} candidates not sorted by similarity (descending). "
            f"Got scores: {scores}"
        )
        logger.info(f"Entity {entity_id} candidates properly ranked: {scores}")


def test_vector_blocker_scores_empty_dataset() -> None:
    """Test that empty datasets handle similarity scores gracefully.

    Edge case: no entities means no candidates and no scores to populate.
    """
    entities: list[dict[str, str]] = []

    fake_index = FakeVectorIndex()
    blocker = VectorBlocker(
        schema_factory=lambda x: CompanySchema(**x),
        text_field_extractor=lambda x: x.name,
        vector_index=fake_index,
        k_neighbors=3,
    )

    texts: list[str] = []
    fake_index.create_index(texts)

    candidates = list(blocker.stream(entities))

    assert len(candidates) == 0
    logger.info("Empty dataset: no candidates, no scores (expected)")


def test_vector_blocker_scores_single_entity() -> None:
    """Test that single entity datasets handle similarity scores gracefully.

    Edge case: one entity means no pairs possible, no scores to populate.
    """
    entities = [{"id": "c1", "name": "Apple Inc"}]

    fake_index = FakeVectorIndex()
    blocker = VectorBlocker(
        schema_factory=lambda x: CompanySchema(**x),
        text_field_extractor=lambda x: x.name,
        vector_index=fake_index,
        k_neighbors=3,
    )

    texts = [e["name"] for e in entities]
    fake_index.create_index(texts)

    candidates = list(blocker.stream(entities))

    assert len(candidates) == 0
    logger.info("Single entity: no pairs, no scores (expected)")
