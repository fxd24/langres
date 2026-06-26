"""Tests for VectorBlocker similarity score population.

This test suite validates that VectorBlocker.stream() populates the
similarity_score field in ERCandidate objects, enabling ranking evaluation.
"""

import logging

import numpy as np
import pytest

from langres.core.blockers.vector import VectorBlocker
from langres.core.indexes import FAISSIndex, FakeVectorIndex
from langres.core.models import CompanySchema

logger = logging.getLogger(__name__)


class _StubEmbedder:
    """Deterministic embedder mapping known texts to fixed vectors.

    Lets a golden test build a *real* FAISSIndex with controlled geometry (no
    model download, no ``@pytest.mark.slow``): a near-duplicate pair is placed
    close together and a non-pair far apart, so distance/score ordering is known.
    """

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping
        self.embedding_dim = len(next(iter(mapping.values())))

    def encode(self, texts: list[str], prompt: str | None = None) -> np.ndarray:
        return np.array([self._mapping[t] for t in texts], dtype=np.float32)


def _score_for_pair(candidates: list, id_a: str, id_b: str) -> float:
    """Return the similarity_score of the candidate for the unordered pair {a, b}."""
    for candidate in candidates:
        if {candidate.left.id, candidate.right.id} == {id_a, id_b}:
            assert candidate.similarity_score is not None
            return candidate.similarity_score
    raise AssertionError(f"No candidate found for pair ({id_a}, {id_b})")


@pytest.mark.parametrize("metric", ["cosine", "L2"])
def test_vector_blocker_near_duplicate_outscores_non_pair(metric: str) -> None:
    """GOLDEN regression guard for the distance->similarity inversion.

    A known near-duplicate pair MUST receive a strictly higher similarity_score
    than a known non-pair, for BOTH metrics. Geometry is fixed via _StubEmbedder:
    c1 and c2 sit almost on top of each other; c3 is far away.

    This FAILS on the old heuristic for metric="L2": FAISS L2 returns *squared*
    distances (here ~26 for the non-pair, > 2.0), which the old code clipped to
    1.0 — inverting it into the MOST-similar score. The index-owned
    ``to_similarities`` converts L2 with ``1/(1+sqrt(d))`` and keeps order.
    """
    entities = [
        {"id": "c1", "name": "apple inc"},
        {"id": "c2", "name": "apple incorporated"},
        {"id": "c3", "name": "microsoft corp"},
    ]
    # Near-duplicate c1/c2 are nearly identical; c3 is far along another axis.
    embedder = _StubEmbedder(
        {
            "apple inc": [1.0, 0.0, 0.0],
            "apple incorporated": [1.0, 0.05, 0.0],
            "microsoft corp": [0.0, 5.0, 0.0],
        }
    )
    index = FAISSIndex(embedder=embedder, metric=metric)  # type: ignore[arg-type]

    blocker = VectorBlocker(
        schema_factory=lambda x: CompanySchema(**x),
        text_field_extractor=lambda x: x.name,
        vector_index=index,
        k_neighbors=2,  # k+1 == N: every pair is generated
    )

    texts = [e["name"] for e in entities]
    index.create_index(texts)

    candidates = list(blocker.stream(entities))

    near_dup = _score_for_pair(candidates, "c1", "c2")
    non_pair = _score_for_pair(candidates, "c1", "c3")

    logger.info(
        "metric=%s near_dup(c1,c2)=%.4f non_pair(c1,c3)=%.4f",
        metric,
        near_dup,
        non_pair,
    )
    assert near_dup > non_pair, (
        f"metric={metric}: near-duplicate score {near_dup} must exceed "
        f"non-pair score {non_pair} (distance->similarity inversion?)"
    )
    # Sanity: both are valid similarities.
    assert 0.0 <= non_pair <= near_dup <= 1.0


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


def test_vector_blocker_handles_nan_distances() -> None:
    """Test that VectorBlocker handles NaN distances gracefully.

    Some vector index implementations (e.g., Qdrant hybrid) can return NaN
    distance values in certain scenarios. The blocker should convert these
    to 0.0 (lowest similarity) rather than propagating NaN values.
    """
    entities = [
        {"id": "c1", "name": "Apple Inc"},
        {"id": "c2", "name": "Microsoft Corp"},
        {"id": "c3", "name": "Google LLC"},
    ]

    # Create a custom vector index that returns NaN values
    class NaNVectorIndex(FakeVectorIndex):
        def search_all(
            self, k: int, query_prompt: str | None = None
        ) -> tuple[np.ndarray, np.ndarray]:
            """Override to return distances with NaN values."""
            distances, indices = super().search_all(k, query_prompt=query_prompt)
            # Inject NaN into some distance values
            distances[0][1] = np.nan  # Second neighbor of first entity
            return distances, indices

    nan_index = NaNVectorIndex()
    blocker = VectorBlocker(
        schema_factory=lambda x: CompanySchema(**x),
        text_field_extractor=lambda x: x.name,
        vector_index=nan_index,
        k_neighbors=2,
    )

    texts = [e["name"] for e in entities]
    nan_index.create_index(texts)

    candidates = list(blocker.stream(entities))

    # Verify all candidates have valid (non-NaN) similarity scores
    for candidate in candidates:
        assert candidate.similarity_score is not None
        assert not np.isnan(candidate.similarity_score), (
            f"Candidate ({candidate.left.id}, {candidate.right.id}) has NaN similarity_score"
        )
        assert 0.0 <= candidate.similarity_score <= 1.0
        logger.info(
            f"Candidate ({candidate.left.id}, {candidate.right.id}): "
            f"similarity_score={candidate.similarity_score:.4f} (NaN handled correctly)"
        )
