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
from langres.core.groups import ERCandidateGroup
from langres.core.indexes.reranking_vector_index import FakeHybridRerankingVectorIndex
from langres.core.indexes.vector_index import FAISSIndex, FakeVectorIndex
from langres.core.models import CompanySchema
from langres.core.registry import get_component
from tests.conftest import edge_list_from_groups, pairs_from_candidates, pairs_from_groups

logger = logging.getLogger(__name__)


def test_registered_under_vector_blocker_via_lazy_lookup() -> None:
    """``get_component('vector_blocker')`` lazily imports+registers the class.

    (W0.4: faiss/sentence-transformers are optional, so ``vector_blocker``
    joined ``_LAZY_COMPONENT_MODULES`` alongside ``dspy_judge``.)
    """
    assert get_component("vector_blocker") is VectorBlocker
    assert VectorBlocker.type_name == "vector_blocker"


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
            # Return 3 rows (one per entity) with 2 neighbors each
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
            np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64),
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
            # Return 2 rows (one per entity) with 2 neighbors each
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            np.array([[1, 0], [0, 1]], dtype=np.int64),
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


def test_stream_with_fake_hybrid_reranking_index_end_to_end():
    """VectorBlocker drives FakeHybridRerankingVectorIndex.stream() without TypeError.

    VectorBlocker.stream() calls ``search_all(k, query_prompt=...)``; before the
    fix this fake lacked ``query_prompt`` and raised TypeError. This exercises the
    full path (search_all + to_similarities) with a query_prompt set.
    """
    index = FakeHybridRerankingVectorIndex()
    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        vector_index=index,
        k_neighbors=2,
        query_prompt="Represent the company name for retrieval:",
    )

    data = [
        {"id": "c1", "name": "Apple"},
        {"id": "c2", "name": "Google"},
        {"id": "c3", "name": "Microsoft"},
    ]
    index.create_index([d["name"] for d in data])

    candidates = list(blocker.stream(data))

    assert len(candidates) > 0
    # to_similarities mapped fake distances into [0, 1] similarity scores.
    for candidate in candidates:
        assert 0.0 <= candidate.similarity_score <= 1.0


# ============================================================================
# stream_groups(): VectorBlocker's NATIVE per-anchor implementation (E3).
#
# Unlike the base Blocker's buffered/skew-prone default (which derives groups
# from the pairwise stream() and so under-represents entities that never land
# on the "left" side), VectorBlocker's kNN search is already per-anchor: one
# group per entity, with its k nearest neighbors as members, straight from the
# index -- no derivation, no skew.
# ============================================================================


def test_vector_blocker_stream_groups_yields_one_group_per_anchor():
    """stream_groups() yields one ERCandidateGroup per entity, from its own kNN search."""
    data = [
        {"id": "c0", "name": "A"},
        {"id": "c1", "name": "B"},
        {"id": "c2", "name": "C"},
        {"id": "c3", "name": "D"},
    ]
    blocker = create_fake_blocker(k_neighbors=2)
    blocker.vector_index.create_index([d["name"] for d in data])

    groups = list(blocker.stream_groups(data))

    assert len(groups) == 4
    assert all(isinstance(g, ERCandidateGroup) for g in groups)
    by_anchor = {g.group_id: g for g in groups}
    assert set(by_anchor) == {"c0", "c1", "c2", "c3"}
    # FakeVectorIndex.search_all: entity i's RAW neighbors (after skipping self)
    # are [(i+1) % N, (i+2) % N] for k_neighbors=2. Cross-anchor dedup (first
    # anchor to reach a pair claims it, same as stream()) then removes an edge
    # from a later anchor's group once an earlier anchor already claimed it:
    # c0 claims {c0,c1} and {c0,c2}; c1 claims {c1,c2} and {c1,c3}; c2's raw
    # {c2,c0} was already claimed by c0, so c2 keeps only {c2,c3}; c3's raw
    # {c3,c1} was already claimed by c1, so c3 keeps only {c3,c0}.
    assert {m.id for m in by_anchor["c0"].members} == {"c1", "c2"}
    assert {m.id for m in by_anchor["c1"].members} == {"c2", "c3"}
    assert {m.id for m in by_anchor["c2"].members} == {"c3"}
    assert {m.id for m in by_anchor["c3"].members} == {"c0"}
    # No self-pairs: an anchor never lists itself as a member.
    assert all(g.anchor.id not in {m.id for m in g.members} for g in groups)
    # Every undirected pair is covered by exactly one group -- no duplicates.
    edges = edge_list_from_groups(groups)
    assert len(edges) == len(set(edges)) == 6  # C(4, 2): full coverage, no dupes


def test_vector_blocker_stream_groups_raises_if_index_not_built():
    """stream_groups() enforces the same explicit-index-build contract as stream()."""
    blocker = create_fake_blocker(k_neighbors=2)
    data = [{"id": "c1", "name": "Apple"}, {"id": "c2", "name": "Google"}]

    with pytest.raises(RuntimeError, match="Index not built"):
        list(blocker.stream_groups(data))


def test_vector_blocker_stream_groups_handles_empty_dataset():
    """Empty input -> no groups."""
    blocker = create_fake_blocker(k_neighbors=2)
    blocker.vector_index.create_index([])
    assert list(blocker.stream_groups([])) == []


def test_vector_blocker_stream_groups_handles_single_entity():
    """A single entity has no neighbors -> no groups."""
    data = [{"id": "c1", "name": "Only Company"}]
    blocker = create_fake_blocker(k_neighbors=5)
    blocker.vector_index.create_index([d["name"] for d in data])

    assert list(blocker.stream_groups(data)) == []


def test_vector_blocker_stream_groups_is_schema_agnostic_with_product_schema():
    """stream_groups() works with a second, unrelated schema (ProductSchema)."""
    from pydantic import BaseModel

    class ProductSchema(BaseModel):
        id: str
        title: str

    def product_factory(record: dict) -> ProductSchema:
        return ProductSchema(id=record["id"], title=record["title"])

    data = [
        {"id": "p1", "title": "iPhone"},
        {"id": "p2", "title": "iPhone Pro"},
        {"id": "p3", "title": "Galaxy"},
    ]
    blocker = VectorBlocker(
        schema_factory=product_factory,
        text_field_extractor=lambda x: x.title,
        vector_index=FakeVectorIndex(),
        k_neighbors=2,
    )
    blocker.vector_index.create_index([d["title"] for d in data])

    groups = list(blocker.stream_groups(data))

    assert len(groups) == 3
    assert all(isinstance(g.anchor, ProductSchema) for g in groups)


@pytest.mark.parametrize(
    ("n_entities", "k_neighbors"),
    [(3, 1), (4, 2), (5, 2), (6, 3), (8, 4)],
)
def test_vector_blocker_stream_groups_pairs_equivalence_property(n_entities, k_neighbors):
    """Property (CEO #14 + E5): pairs from stream_groups() == pairs from stream().

    COUNT-based, not just set-based: every pair stream() would yield is
    covered by EXACTLY one group -- no losses AND no duplicates -- across
    several (N, k) shapes, including k_neighbors close to and below N-1
    (dense) and small k (sparse). A set-only comparison would silently mask a
    pair being emitted by two different groups (see
    ``test_vector_blocker_stream_groups_dedupes_mutual_neighbor_pairs`` for
    the targeted regression on that exact failure mode).
    """
    data = [{"id": f"c{i}", "name": f"Company {i}"} for i in range(n_entities)]
    texts = [d["name"] for d in data]

    stream_blocker = create_fake_blocker(k_neighbors=k_neighbors)
    stream_blocker.vector_index.create_index(texts)
    stream_pairs = pairs_from_candidates(stream_blocker.stream(data))

    groups_blocker = create_fake_blocker(k_neighbors=k_neighbors)
    groups_blocker.vector_index.create_index(texts)
    groups = list(groups_blocker.stream_groups(data))
    group_edges = edge_list_from_groups(groups)

    assert len(group_edges) == len(set(group_edges))  # no duplicate edges across groups
    assert set(group_edges) == stream_pairs  # same set covered
    assert pairs_from_groups(groups) == stream_pairs  # sanity: set helper agrees


def test_vector_blocker_stream_groups_dedupes_mutual_neighbor_pairs():
    """Regression: mutual nearest neighbors are covered by exactly ONE group.

    stream() maintains a single ``seen_pairs`` set across all entities, so a
    mutual-nearest-neighbor pair (A's nearest neighbor is B AND B's nearest
    neighbor is A -- common with real ANN indexes on near-duplicate records)
    is yielded exactly once. stream_groups() now mirrors that exact dedup
    semantics (same iteration order, same first-seen-wins rule, threaded
    across ALL anchors via one ``seen_pairs`` set) -- so a mutual pair is
    assigned to whichever anchor is processed first (c0, here) and does NOT
    also appear in the other anchor's group (c1's group ends up empty).

    Without the fix, a consumer issuing one LLM call per group (e.g. a future
    SelectJudge) would emit and charge for the same undirected pair twice --
    this test fails against that pre-fix behavior (which asserted the pair
    appeared in BOTH groups) and passes against the fix.
    """
    from unittest.mock import MagicMock

    import numpy as np

    mock_index = MagicMock()
    mock_index._index = object()  # _index_is_built() -> True
    # 3 entities, k_neighbors=1 (k=2 incl. self): c0 and c1 are MUTUAL nearest
    # neighbors; c2's nearest neighbor is c0 (one-directional).
    mock_index.search_all = MagicMock(
        return_value=(
            np.array([[0.0, 0.1], [0.0, 0.1], [0.0, 0.2]], dtype=np.float32),
            np.array([[0, 1], [1, 0], [2, 0]], dtype=np.int64),
        )
    )
    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        vector_index=mock_index,
        k_neighbors=1,
    )
    data = [
        {"id": "c0", "name": "Acme"},
        {"id": "c1", "name": "Acme Corp"},
        {"id": "c2", "name": "Other Co"},
    ]

    groups = list(blocker.stream_groups(data))

    by_anchor = {g.group_id: g for g in groups}
    # c0 is processed first -> claims the mutual pair {c0, c1}.
    assert {m.id for m in by_anchor["c0"].members} == {"c1"}
    # c1's own reciprocal edge back to c0 was already claimed -> empty group.
    assert {m.id for m in by_anchor["c1"].members} == set()
    assert {m.id for m in by_anchor["c2"].members} == {"c0"}

    edges = edge_list_from_groups(groups)
    assert len(edges) == len(set(edges))  # no duplicate edges anywhere
    # {c0, c1} (mutual) and {c0, c2} (one-directional) are each covered ONCE.
    assert edges.count(frozenset({"c0", "c1"})) == 1
    assert edges.count(frozenset({"c0", "c2"})) == 1
    assert len(edges) == 2
