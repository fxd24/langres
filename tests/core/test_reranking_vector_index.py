"""Tests for reranking vector index implementations."""

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    ScoredPoint,
    SparseVector,
    VectorParams,
)

from langres.core.embeddings import (
    FakeEmbedder,
    FakeLateInteractionEmbedder,
    FakeSparseEmbedder,
)
from langres.core.reranking_vector_index import (
    FakeHybridRerankingVectorIndex,
    QdrantHybridRerankingIndex,
)

logger = logging.getLogger(__name__)


class TestQdrantHybridRerankingIndex:
    """Tests for QdrantHybridRerankingIndex with mocked Qdrant client."""

    def test_create_index_creates_collection_with_three_named_vectors(self):
        """Test that create_index() creates collection with dense + sparse + reranking vectors."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = FakeEmbedder(embedding_dim=128)
        sparse_embedder = FakeSparseEmbedder()
        reranking_embedder = FakeLateInteractionEmbedder(embedding_dim=128)

        index = QdrantHybridRerankingIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,
        )

        # Execute
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        # Verify recreate_collection was called with correct config
        mock_client.recreate_collection.assert_called_once()
        call_args = mock_client.recreate_collection.call_args

        assert call_args[1]["collection_name"] == "test_collection"

        # Check vectors_config for dense and reranking vectors
        vectors_config = call_args[1]["vectors_config"]
        assert "dense" in vectors_config
        assert vectors_config["dense"].size == 128
        assert vectors_config["dense"].distance == Distance.COSINE

        assert "reranking" in vectors_config
        assert vectors_config["reranking"].size == 128
        assert vectors_config["reranking"].distance == Distance.COSINE

        # Check sparse_vectors_config for sparse vector
        sparse_config = call_args[1]["sparse_vectors_config"]
        assert "sparse" in sparse_config

    def test_create_index_includes_multivector_config(self):
        """Test that reranking vector uses MultiVectorConfig with MaxSim."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = FakeEmbedder(embedding_dim=128)
        sparse_embedder = FakeSparseEmbedder()
        reranking_embedder = FakeLateInteractionEmbedder(embedding_dim=128)

        index = QdrantHybridRerankingIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,
        )

        # Execute
        texts = ["Apple Inc.", "Microsoft Corp."]
        index.create_index(texts)

        # Verify MultiVectorConfig on reranking vector
        call_args = mock_client.recreate_collection.call_args
        vectors_config = call_args[1]["vectors_config"]
        reranking_config = vectors_config["reranking"]

        assert hasattr(reranking_config, "multivector_config")
        assert isinstance(reranking_config.multivector_config, MultiVectorConfig)
        assert reranking_config.multivector_config.comparator == MultiVectorComparator.MAX_SIM

    def test_create_index_batch_upserts_points_with_three_vectors(self):
        """Test that create_index() batch upserts points with all three vector types."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = FakeEmbedder(embedding_dim=128)
        sparse_embedder = FakeSparseEmbedder()
        reranking_embedder = FakeLateInteractionEmbedder(embedding_dim=128)

        index = QdrantHybridRerankingIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,
        )

        # Execute
        texts = ["Apple Inc.", "Microsoft Corp."]
        index.create_index(texts)

        # Verify upsert was called once with batch of points
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args

        assert call_args[1]["collection_name"] == "test_collection"

        points = call_args[1]["points"]
        assert len(points) == 2

        # Check first point structure
        point0 = points[0]
        assert isinstance(point0, PointStruct)
        assert point0.id == 0
        assert "dense" in point0.vector
        assert "sparse" in point0.vector
        assert "reranking" in point0.vector

        # Dense vector should be list of floats (from numpy array)
        assert isinstance(point0.vector["dense"], list)
        assert len(point0.vector["dense"]) == 128

        # Sparse vector should be SparseVector with indices + values
        assert isinstance(point0.vector["sparse"], SparseVector)
        assert len(point0.vector["sparse"].indices) == len(point0.vector["sparse"].values)

        # Reranking vector should be multi-vector (list[list[float]])
        assert isinstance(point0.vector["reranking"], list)
        assert len(point0.vector["reranking"]) > 0  # At least one token
        assert isinstance(point0.vector["reranking"][0], list)  # Token embedding
        assert len(point0.vector["reranking"][0]) == 128  # embedding_dim

        # Check payload
        assert point0.payload["text"] == "Apple Inc."
        assert point0.payload["id"] == "0"

    def test_search_single_text_uses_reranking_query(self):
        """Test that search() with single text uses reranking query (not FusionQuery)."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = FakeEmbedder(embedding_dim=128)
        sparse_embedder = FakeSparseEmbedder()
        reranking_embedder = FakeLateInteractionEmbedder(embedding_dim=128)

        # Mock query_points response
        mock_scored_points = [
            ScoredPoint(id=0, version=0, score=0.9, payload={}, vector={}),
            ScoredPoint(id=1, version=0, score=0.8, payload={}, vector={}),
        ]
        mock_client.query_points.return_value = mock_scored_points

        index = QdrantHybridRerankingIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,
            prefetch_limit=20,
        )

        # Create index first
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        # Execute
        distances, indices = index.search("Apple", k=2)

        # Verify query_points was called with prefetch structure
        mock_client.query_points.assert_called_once()
        call_args = mock_client.query_points.call_args

        assert call_args[1]["collection_name"] == "test_collection"
        assert call_args[1]["limit"] == 2

        # Check prefetch structure (dense + sparse)
        prefetch = call_args[1]["prefetch"]
        assert len(prefetch) == 2  # Dense + sparse

        # Verify dense prefetch
        dense_prefetch = next(p for p in prefetch if p.using == "dense")
        assert isinstance(dense_prefetch.query, list)
        assert len(dense_prefetch.query) == 128
        assert dense_prefetch.limit == 20

        # Verify sparse prefetch
        sparse_prefetch = next(p for p in prefetch if p.using == "sparse")
        assert isinstance(sparse_prefetch.query, SparseVector)
        assert sparse_prefetch.limit == 20

        # Verify query uses reranking multi-vectors (NOT FusionQuery)
        query = call_args[1]["query"]
        assert not hasattr(query, "fusion")  # Not a FusionQuery
        assert isinstance(query, list)  # Multi-vector list
        assert len(query) > 0  # At least one token
        assert isinstance(query[0], list)  # Token embedding

        # Verify results shape (single query)
        assert distances.shape == (2,)
        assert indices.shape == (2,)
        assert indices[0] == 0
        assert indices[1] == 1

    def test_search_batch_texts_native_batching(self):
        """Test that search() with batch texts makes multiple query_points calls."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = FakeEmbedder(embedding_dim=128)
        sparse_embedder = FakeSparseEmbedder()
        reranking_embedder = FakeLateInteractionEmbedder(embedding_dim=128)

        # Mock query_points responses (one per query)
        mock_client.query_points.side_effect = [
            [
                ScoredPoint(id=0, version=0, score=0.9, payload={}, vector={}),
                ScoredPoint(id=1, version=0, score=0.8, payload={}, vector={}),
            ],
            [
                ScoredPoint(id=2, version=0, score=0.95, payload={}, vector={}),
                ScoredPoint(id=0, version=0, score=0.7, payload={}, vector={}),
            ],
        ]

        index = QdrantHybridRerankingIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,
        )

        # Create index first
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        # Execute batch search
        query_texts = ["Apple", "Google"]
        distances, indices = index.search(query_texts, k=2)

        # Verify query_points was called twice (once per query)
        assert mock_client.query_points.call_count == 2

        # Verify results shape (batch)
        assert distances.shape == (2, 2)  # 2 queries, 2 neighbors each
        assert indices.shape == (2, 2)

        # Verify results content
        assert indices[0, 0] == 0  # First query, first result
        assert indices[0, 1] == 1  # First query, second result
        assert indices[1, 0] == 2  # Second query, first result
        assert indices[1, 1] == 0  # Second query, second result

    def test_search_all_deduplication_pattern(self):
        """Test search_all() for efficient deduplication."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = FakeEmbedder(embedding_dim=128)
        sparse_embedder = FakeSparseEmbedder()
        reranking_embedder = FakeLateInteractionEmbedder(embedding_dim=128)

        # Mock query_points responses (one per corpus item)
        mock_client.query_points.side_effect = [
            [
                ScoredPoint(id=0, version=0, score=1.0, payload={}, vector={}),
                ScoredPoint(id=1, version=0, score=0.8, payload={}, vector={}),
                ScoredPoint(id=2, version=0, score=0.6, payload={}, vector={}),
            ],
            [
                ScoredPoint(id=1, version=0, score=1.0, payload={}, vector={}),
                ScoredPoint(id=0, version=0, score=0.7, payload={}, vector={}),
                ScoredPoint(id=2, version=0, score=0.5, payload={}, vector={}),
            ],
            [
                ScoredPoint(id=2, version=0, score=1.0, payload={}, vector={}),
                ScoredPoint(id=1, version=0, score=0.6, payload={}, vector={}),
                ScoredPoint(id=0, version=0, score=0.4, payload={}, vector={}),
            ],
        ]

        index = QdrantHybridRerankingIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,
        )

        # Create index first
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        # Execute
        distances, indices = index.search_all(k=3)

        # Verify query_points was called 3 times (once per corpus item)
        assert mock_client.query_points.call_count == 3

        # Verify results shape
        assert distances.shape == (3, 3)  # 3 items, 3 neighbors each
        assert indices.shape == (3, 3)

        # First neighbor should be itself (highest similarity)
        assert indices[0, 0] == 0
        assert indices[1, 0] == 1
        assert indices[2, 0] == 2

    def test_search_before_create_index_raises_error(self):
        """Test that search() before create_index() raises RuntimeError."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = FakeEmbedder(embedding_dim=128)
        sparse_embedder = FakeSparseEmbedder()
        reranking_embedder = FakeLateInteractionEmbedder(embedding_dim=128)

        index = QdrantHybridRerankingIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,
        )

        # Execute & Verify
        with pytest.raises(RuntimeError, match="Must call create_index"):
            index.search("Apple", k=3)

    def test_search_all_before_create_index_raises_error(self):
        """Test that search_all() before create_index() raises RuntimeError."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = FakeEmbedder(embedding_dim=128)
        sparse_embedder = FakeSparseEmbedder()
        reranking_embedder = FakeLateInteractionEmbedder(embedding_dim=128)

        index = QdrantHybridRerankingIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,
        )

        # Execute & Verify
        with pytest.raises(RuntimeError, match="Must call create_index"):
            index.search_all(k=3)

    def test_prefetch_limit_configurable(self):
        """Test that prefetch limit is configurable."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = FakeEmbedder(embedding_dim=128)
        sparse_embedder = FakeSparseEmbedder()
        reranking_embedder = FakeLateInteractionEmbedder(embedding_dim=128)

        mock_client.query_points.return_value = [
            ScoredPoint(id=0, version=0, score=0.9, payload={}, vector={})
        ]

        # Test with custom prefetch limit
        index = QdrantHybridRerankingIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,
            prefetch_limit=50,  # Custom limit
        )

        # Create index and search
        texts = ["Apple Inc."]
        index.create_index(texts)
        index.search("Apple", k=1)

        # Verify prefetch limit was used
        call_args = mock_client.query_points.call_args
        prefetch = call_args[1]["prefetch"]

        for p in prefetch:
            assert p.limit == 50


class TestFakeHybridRerankingVectorIndex:
    """Tests for FakeHybridRerankingVectorIndex test double."""

    def test_fake_create_index(self):
        """Test FakeHybridRerankingVectorIndex.create_index with texts."""
        index = FakeHybridRerankingVectorIndex()
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]

        # Should not raise
        index.create_index(texts)

    def test_fake_search_single_text(self):
        """Test FakeHybridRerankingVectorIndex.search with single text."""
        index = FakeHybridRerankingVectorIndex()
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        distances, indices = index.search("Apple", k=2)

        # Should return 1D arrays
        assert distances.shape == (2,)
        assert indices.shape == (2,)
        assert isinstance(distances, np.ndarray)
        assert isinstance(indices, np.ndarray)

    def test_fake_search_batch_texts(self):
        """Test FakeHybridRerankingVectorIndex.search with batch of texts."""
        index = FakeHybridRerankingVectorIndex()
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        query_texts = ["Apple", "Google"]
        distances, indices = index.search(query_texts, k=2)

        # Should return 2D arrays
        assert distances.shape == (2, 2)
        assert indices.shape == (2, 2)

    def test_fake_search_all(self):
        """Test FakeHybridRerankingVectorIndex.search_all."""
        index = FakeHybridRerankingVectorIndex()
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC", "Amazon"]
        index.create_index(texts)

        distances, indices = index.search_all(k=3)

        # Should return shape (N, k)
        assert distances.shape == (4, 3)
        assert indices.shape == (4, 3)

        # First neighbor should be itself (deterministic pattern)
        assert np.array_equal(indices[:, 0], [0, 1, 2, 3])

    def test_fake_search_before_create_index_raises_error(self):
        """Test that FakeHybridRerankingVectorIndex.search before create_index raises error."""
        index = FakeHybridRerankingVectorIndex()

        with pytest.raises(RuntimeError, match="Index not built"):
            index.search("Apple", k=2)

    def test_fake_search_all_before_create_index_raises_error(self):
        """Test that FakeHybridRerankingVectorIndex.search_all before create_index raises error."""
        index = FakeHybridRerankingVectorIndex()

        with pytest.raises(RuntimeError, match="Index not built"):
            index.search_all(k=2)


class TestRerankingVectorIndexProtocol:
    """Tests for VectorIndex protocol compliance."""

    def test_qdrant_index_implements_protocol(self):
        """Test that QdrantHybridRerankingIndex implements VectorIndex protocol."""
        mock_client = MagicMock()
        dense_embedder = FakeEmbedder(embedding_dim=128)
        sparse_embedder = FakeSparseEmbedder()
        reranking_embedder = FakeLateInteractionEmbedder(embedding_dim=128)

        index = QdrantHybridRerankingIndex(
            client=mock_client,
            collection_name="test",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,
        )

        assert hasattr(index, "create_index")
        assert hasattr(index, "search")
        assert hasattr(index, "search_all")
        assert callable(index.create_index)
        assert callable(index.search)
        assert callable(index.search_all)

    def test_fake_index_implements_protocol(self):
        """Test that FakeHybridRerankingVectorIndex implements VectorIndex protocol."""
        index = FakeHybridRerankingVectorIndex()

        assert hasattr(index, "create_index")
        assert hasattr(index, "search")
        assert hasattr(index, "search_all")
        assert callable(index.create_index)
        assert callable(index.search)
        assert callable(index.search_all)
