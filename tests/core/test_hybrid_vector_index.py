"""Tests for hybrid vector index implementations."""

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    PointStruct,
    ScoredPoint,
    SparseVector,
)

from langres.core.embeddings import FakeEmbedder, FakeSparseEmbedder
from langres.core.hybrid_vector_index import (
    FakeHybridVectorIndex,
    QdrantHybridIndex,
)

logger = logging.getLogger(__name__)


class TestQdrantHybridIndex:
    """Tests for QdrantHybridIndex with mocked Qdrant client."""

    def test_create_index_creates_collection_with_named_vectors(self):
        """Test that create_index() creates collection with dense + sparse vectors."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = FakeEmbedder(embedding_dim=128)
        sparse_embedder = FakeSparseEmbedder()

        index = QdrantHybridIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
        )

        # Execute
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        # Verify recreate_collection was called with correct config
        mock_client.recreate_collection.assert_called_once()
        call_args = mock_client.recreate_collection.call_args

        assert call_args[1]["collection_name"] == "test_collection"

        # Check vectors_config for dense vector
        vectors_config = call_args[1]["vectors_config"]
        assert "dense" in vectors_config
        assert vectors_config["dense"].size == 128
        assert vectors_config["dense"].distance == Distance.COSINE

        # Check sparse_vectors_config for sparse vector
        sparse_config = call_args[1]["sparse_vectors_config"]
        assert "sparse" in sparse_config

    def test_create_index_batch_upserts_points(self):
        """Test that create_index() batch upserts points with correct structure."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = FakeEmbedder(embedding_dim=128)
        sparse_embedder = FakeSparseEmbedder()

        index = QdrantHybridIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
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

        # Dense vector should be list of floats (from numpy array)
        assert isinstance(point0.vector["dense"], list)
        assert len(point0.vector["dense"]) == 128

        # Sparse vector should be SparseVector with indices + values
        assert isinstance(point0.vector["sparse"], SparseVector)
        assert len(point0.vector["sparse"].indices) == len(point0.vector["sparse"].values)

        # Check payload
        assert point0.payload["text"] == "Apple Inc."
        assert point0.payload["id"] == "0"

    def test_search_single_text_uses_hybrid_query(self):
        """Test that search() with single text uses hybrid query with prefetch."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = FakeEmbedder(embedding_dim=128)
        sparse_embedder = FakeSparseEmbedder()

        # Mock query_points response
        mock_scored_points = [
            ScoredPoint(id=0, version=0, score=0.9, payload={}, vector={}),
            ScoredPoint(id=1, version=0, score=0.8, payload={}, vector={}),
        ]
        mock_client.query_points.return_value = mock_scored_points

        index = QdrantHybridIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
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

        # Check prefetch structure
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

        # Verify fusion query (RRF by default)
        query = call_args[1]["query"]
        assert isinstance(query, FusionQuery)
        assert query.fusion == Fusion.RRF

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

        index = QdrantHybridIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
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

        index = QdrantHybridIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
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

        index = QdrantHybridIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
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

        index = QdrantHybridIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
        )

        # Execute & Verify
        with pytest.raises(RuntimeError, match="Must call create_index"):
            index.search_all(k=3)

    def test_fusion_strategy_configurable(self):
        """Test that fusion strategy can be configured (RRF vs DBSF)."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = FakeEmbedder(embedding_dim=128)
        sparse_embedder = FakeSparseEmbedder()

        mock_client.query_points.return_value = [
            ScoredPoint(id=0, version=0, score=0.9, payload={}, vector={})
        ]

        # Test with DBSF fusion
        index = QdrantHybridIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            fusion="DBSF",  # Use DBSF instead of RRF
        )

        # Create index and search
        texts = ["Apple Inc."]
        index.create_index(texts)
        index.search("Apple", k=1)

        # Verify DBSF fusion was used
        call_args = mock_client.query_points.call_args
        query = call_args[1]["query"]
        assert isinstance(query, FusionQuery)
        assert query.fusion == Fusion.DBSF

    def test_prefetch_limit_configurable(self):
        """Test that prefetch limit is configurable."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = FakeEmbedder(embedding_dim=128)
        sparse_embedder = FakeSparseEmbedder()

        mock_client.query_points.return_value = [
            ScoredPoint(id=0, version=0, score=0.9, payload={}, vector={})
        ]

        # Test with custom prefetch limit
        index = QdrantHybridIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
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


class TestFakeHybridVectorIndex:
    """Tests for FakeHybridVectorIndex test double."""

    def test_fake_create_index(self):
        """Test FakeHybridVectorIndex.create_index with texts."""
        index = FakeHybridVectorIndex()
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]

        # Should not raise
        index.create_index(texts)

    def test_fake_search_single_text(self):
        """Test FakeHybridVectorIndex.search with single text."""
        index = FakeHybridVectorIndex()
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        distances, indices = index.search("Apple", k=2)

        # Should return 1D arrays
        assert distances.shape == (2,)
        assert indices.shape == (2,)
        assert isinstance(distances, np.ndarray)
        assert isinstance(indices, np.ndarray)

    def test_fake_search_batch_texts(self):
        """Test FakeHybridVectorIndex.search with batch of texts."""
        index = FakeHybridVectorIndex()
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        query_texts = ["Apple", "Google"]
        distances, indices = index.search(query_texts, k=2)

        # Should return 2D arrays
        assert distances.shape == (2, 2)
        assert indices.shape == (2, 2)

    def test_fake_search_all(self):
        """Test FakeHybridVectorIndex.search_all."""
        index = FakeHybridVectorIndex()
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC", "Amazon"]
        index.create_index(texts)

        distances, indices = index.search_all(k=3)

        # Should return shape (N, k)
        assert distances.shape == (4, 3)
        assert indices.shape == (4, 3)

        # First neighbor should be itself (deterministic pattern)
        assert np.array_equal(indices[:, 0], [0, 1, 2, 3])

    def test_fake_search_before_create_index_raises_error(self):
        """Test that FakeHybridVectorIndex.search before create_index raises error."""
        index = FakeHybridVectorIndex()

        with pytest.raises(RuntimeError, match="Index not built"):
            index.search("Apple", k=2)

    def test_fake_search_all_before_create_index_raises_error(self):
        """Test that FakeHybridVectorIndex.search_all before create_index raises error."""
        index = FakeHybridVectorIndex()

        with pytest.raises(RuntimeError, match="Index not built"):
            index.search_all(k=2)


class TestQdrantHybridIndexInstructionPrompts:
    """Tests for QdrantHybridIndex instruction prompt support (asymmetric encoding)."""

    def test_qdrant_hybrid_documents_no_prompts(self):
        """Test that create_index encodes documents without prompts."""
        from unittest.mock import Mock

        # Create tracking embedders
        dense_call_log = []
        sparse_call_log = []

        class TrackingDenseEmbedder:
            embedding_dim = 128

            def encode(self, texts, prompt=None):
                dense_call_log.append({"texts": texts, "prompt": prompt})
                return np.random.rand(len(texts), 128).astype(np.float32)

        class TrackingSparseEmbedder:
            def encode(self, texts, prompt=None):
                sparse_call_log.append({"texts": texts, "prompt": prompt})
                # Return sparse embeddings format
                return [{"indices": [i, i + 1], "values": [0.5, 0.3]} for i in range(len(texts))]

        mock_client = MagicMock()
        dense_embedder = TrackingDenseEmbedder()
        sparse_embedder = TrackingSparseEmbedder()

        index = QdrantHybridIndex(
            client=mock_client,
            collection_name="test",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            query_prompt="Find duplicates",
        )

        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        # Verify both embedders called with prompt=None (documents shouldn't have prompts)
        assert len(dense_call_log) == 1
        assert dense_call_log[0]["prompt"] is None

        assert len(sparse_call_log) == 1
        assert sparse_call_log[0]["prompt"] is None

    def test_qdrant_hybrid_queries_dense_with_prompt_sparse_without(self):
        """Test that search encodes queries with prompt for dense, without for sparse."""
        from unittest.mock import Mock

        # Create tracking embedders
        dense_call_log = []
        sparse_call_log = []

        class TrackingDenseEmbedder:
            embedding_dim = 128

            def encode(self, texts, prompt=None):
                dense_call_log.append({"texts": texts, "prompt": prompt})
                return np.random.rand(len(texts), 128).astype(np.float32)

        class TrackingSparseEmbedder:
            def encode(self, texts, prompt=None):
                sparse_call_log.append({"texts": texts, "prompt": prompt})
                return [{"indices": [i, i + 1], "values": [0.5, 0.3]} for i in range(len(texts))]

        mock_client = MagicMock()
        mock_client.query_points.return_value = [
            ScoredPoint(id=0, version=0, score=0.9, payload={}, vector={})
        ]

        dense_embedder = TrackingDenseEmbedder()
        sparse_embedder = TrackingSparseEmbedder()

        query_prompt = "Find duplicate organization names"
        index = QdrantHybridIndex(
            client=mock_client,
            collection_name="test",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            query_prompt=query_prompt,
        )

        # Create index (first calls)
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        # Clear logs
        dense_call_log.clear()
        sparse_call_log.clear()

        # Search (second calls - should use prompt for dense only)
        index.search("Apple Company", k=2)

        # Verify dense embedder got the prompt
        assert len(dense_call_log) == 1
        assert dense_call_log[0]["prompt"] == query_prompt

        # Verify sparse embedder did NOT get prompt (BM25 doesn't use instructions)
        assert len(sparse_call_log) == 1
        assert sparse_call_log[0]["prompt"] is None

    def test_qdrant_hybrid_search_all_delegates_with_prompt(self):
        """Test that search_all delegates to search which applies prompt."""
        from unittest.mock import Mock

        # Create tracking dense embedder
        dense_call_log = []

        class TrackingDenseEmbedder:
            embedding_dim = 128

            def encode(self, texts, prompt=None):
                dense_call_log.append({"texts": texts, "prompt": prompt})
                return np.random.rand(len(texts), 128).astype(np.float32)

        class TrackingSparseEmbedder:
            def encode(self, texts, prompt=None):
                return [{"indices": [i, i + 1], "values": [0.5, 0.3]} for i in range(len(texts))]

        mock_client = MagicMock()
        mock_client.query_points.return_value = [
            ScoredPoint(id=0, version=0, score=1.0, payload={}, vector={}),
            ScoredPoint(id=1, version=0, score=0.8, payload={}, vector={}),
        ]

        dense_embedder = TrackingDenseEmbedder()
        sparse_embedder = TrackingSparseEmbedder()

        query_prompt = "test instruction"
        index = QdrantHybridIndex(
            client=mock_client,
            collection_name="test",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            query_prompt=query_prompt,
        )

        # Create index
        texts = ["Apple Inc.", "Microsoft Corp."]
        index.create_index(texts)

        # Clear logs
        dense_call_log.clear()

        # search_all should call search() which applies prompt
        index.search_all(k=2)

        # Verify dense embedder was called once with batch of corpus texts and prompt
        assert len(dense_call_log) == 1  # Single batch call
        assert dense_call_log[0]["texts"] == texts  # All corpus texts
        assert dense_call_log[0]["prompt"] == query_prompt  # With prompt


class TestHybridVectorIndexProtocol:
    """Tests for VectorIndex protocol compliance."""

    def test_qdrant_index_implements_protocol(self):
        """Test that QdrantHybridIndex implements VectorIndex protocol."""
        mock_client = MagicMock()
        dense_embedder = FakeEmbedder(embedding_dim=128)
        sparse_embedder = FakeSparseEmbedder()

        index = QdrantHybridIndex(
            client=mock_client,
            collection_name="test",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
        )

        assert hasattr(index, "create_index")
        assert hasattr(index, "search")
        assert hasattr(index, "search_all")
        assert callable(index.create_index)
        assert callable(index.search)
        assert callable(index.search_all)

    def test_fake_hybrid_index_implements_protocol(self):
        """Test that FakeHybridVectorIndex implements VectorIndex protocol."""
        index = FakeHybridVectorIndex()

        assert hasattr(index, "create_index")
        assert hasattr(index, "search")
        assert hasattr(index, "search_all")
        assert callable(index.create_index)
        assert callable(index.search)
        assert callable(index.search_all)
