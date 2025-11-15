"""Tests for reranking vector index implementations."""

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    Prefetch,
    ScoredPoint,
    SparseVector,
    VectorParams,
)

from langres.core.embeddings import (
    FakeEmbedder,
    FakeLateInteractionEmbedder,
    FakeSparseEmbedder,
)
from langres.core.indexes.reranking_vector_index import (
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
        """Test that search() with single text uses reranking query with nested prefetch."""
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

        # Check prefetch structure - should be nested Prefetch with fusion
        prefetch = call_args[1]["prefetch"]
        assert isinstance(prefetch, Prefetch)

        # Verify inner prefetches (dense + sparse)
        assert hasattr(prefetch, "prefetch")
        inner_prefetches = prefetch.prefetch
        assert isinstance(inner_prefetches, list)
        assert len(inner_prefetches) == 2

        # Verify dense prefetch
        dense_prefetch = next(p for p in inner_prefetches if p.using == "dense")
        assert isinstance(dense_prefetch.query, list)
        assert len(dense_prefetch.query) == 128
        assert dense_prefetch.limit == 20

        # Verify sparse prefetch
        sparse_prefetch = next(p for p in inner_prefetches if p.using == "sparse")
        assert isinstance(sparse_prefetch.query, SparseVector)
        assert sparse_prefetch.limit == 20

        # Verify fusion at outer Prefetch level
        assert isinstance(prefetch.query, FusionQuery)
        assert prefetch.query.fusion == Fusion.RRF  # Default fusion

        # Verify query uses reranking multi-vectors (at top level)
        query = call_args[1]["query"]
        assert isinstance(query, list)  # Multi-vector list
        assert len(query) > 0  # At least one token
        assert isinstance(query[0], list)  # Token embedding

        # Verify using parameter specifies reranking vector
        assert call_args[1]["using"] == "reranking"

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
        """Test that prefetch limit is configurable at all prefetch levels."""
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

        # Verify prefetch limit was used at all levels
        call_args = mock_client.query_points.call_args
        prefetch = call_args[1]["prefetch"]

        # Check outer Prefetch limit (fusion level)
        assert prefetch.limit == 50

        # Check inner prefetches limits (dense + sparse)
        for p in prefetch.prefetch:
            assert p.limit == 50

    def test_default_rrf_fusion(self):
        """Test that QdrantHybridRerankingIndex uses RRF fusion by default."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = FakeEmbedder(embedding_dim=128)
        sparse_embedder = FakeSparseEmbedder()
        reranking_embedder = FakeLateInteractionEmbedder(embedding_dim=128)

        # Mock query_points response
        mock_scored_points = [
            ScoredPoint(id=0, version=0, score=0.9, payload={}, vector={}),
        ]
        mock_client.query_points.return_value = mock_scored_points

        # Create index without explicit fusion parameter (should default to RRF)
        index = QdrantHybridRerankingIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,
        )

        # Create index and search
        texts = ["Apple Inc."]
        index.create_index(texts)
        index.search("Apple", k=1)

        # Verify query_points was called with nested prefetch + RRF fusion
        mock_client.query_points.assert_called_once()
        call_args = mock_client.query_points.call_args

        # Check prefetch structure - should be a single Prefetch object, not list
        prefetch = call_args[1]["prefetch"]
        assert isinstance(prefetch, Prefetch)
        assert not isinstance(prefetch, list)

        # Verify the nested prefetch contains [dense, sparse]
        assert hasattr(prefetch, "prefetch")
        assert isinstance(prefetch.prefetch, list)
        assert len(prefetch.prefetch) == 2

        # Verify the outer Prefetch uses FusionQuery with RRF
        assert hasattr(prefetch, "query")
        from qdrant_client.models import FusionQuery, Fusion

        assert isinstance(prefetch.query, FusionQuery)
        assert prefetch.query.fusion == Fusion.RRF

    def test_explicit_dbsf_fusion(self):
        """Test that QdrantHybridRerankingIndex uses DBSF fusion when specified."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = FakeEmbedder(embedding_dim=128)
        sparse_embedder = FakeSparseEmbedder()
        reranking_embedder = FakeLateInteractionEmbedder(embedding_dim=128)

        # Mock query_points response
        mock_scored_points = [
            ScoredPoint(id=0, version=0, score=0.9, payload={}, vector={}),
        ]
        mock_client.query_points.return_value = mock_scored_points

        # Create index with explicit DBSF fusion parameter
        index = QdrantHybridRerankingIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,
            fusion="DBSF",
        )

        # Create index and search
        texts = ["Apple Inc."]
        index.create_index(texts)
        index.search("Apple", k=1)

        # Verify query_points was called with nested prefetch + DBSF fusion
        mock_client.query_points.assert_called_once()
        call_args = mock_client.query_points.call_args

        # Check prefetch structure
        prefetch = call_args[1]["prefetch"]
        assert isinstance(prefetch, Prefetch)

        # Verify the outer Prefetch uses FusionQuery with DBSF
        from qdrant_client.models import FusionQuery, Fusion

        assert isinstance(prefetch.query, FusionQuery)
        assert prefetch.query.fusion == Fusion.DBSF

    def test_nested_prefetch_structure(self):
        """Test that prefetch has correct 3-stage structure: [dense, sparse] → fusion → reranking."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = FakeEmbedder(embedding_dim=128)
        sparse_embedder = FakeSparseEmbedder()
        reranking_embedder = FakeLateInteractionEmbedder(embedding_dim=128)

        # Mock query_points response
        mock_scored_points = [
            ScoredPoint(id=0, version=0, score=0.9, payload={}, vector={}),
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

        # Create index and search
        texts = ["Apple Inc."]
        index.create_index(texts)
        index.search("Apple", k=1)

        # Verify complete 3-stage structure
        call_args = mock_client.query_points.call_args

        # Stage 3 (outermost): Reranking query at top level
        assert call_args[1]["using"] == "reranking"
        query = call_args[1]["query"]
        assert isinstance(query, list)  # Multi-vector for reranking
        assert len(query) > 0
        assert isinstance(query[0], list)  # Token embeddings

        # Stage 2: Fusion at outer Prefetch level
        prefetch = call_args[1]["prefetch"]
        assert isinstance(prefetch, Prefetch)
        from qdrant_client.models import FusionQuery, Fusion

        assert isinstance(prefetch.query, FusionQuery)
        assert prefetch.query.fusion == Fusion.RRF
        assert prefetch.limit == 20

        # Stage 1: Dense + sparse prefetches at inner level
        assert hasattr(prefetch, "prefetch")
        inner_prefetches = prefetch.prefetch
        assert isinstance(inner_prefetches, list)
        assert len(inner_prefetches) == 2

        # Verify dense prefetch
        dense_prefetch = next(p for p in inner_prefetches if p.using == "dense")
        assert isinstance(dense_prefetch, Prefetch)
        assert isinstance(dense_prefetch.query, list)
        assert len(dense_prefetch.query) == 128
        assert dense_prefetch.limit == 20

        # Verify sparse prefetch
        sparse_prefetch = next(p for p in inner_prefetches if p.using == "sparse")
        assert isinstance(sparse_prefetch, Prefetch)
        assert isinstance(sparse_prefetch.query, SparseVector)
        assert sparse_prefetch.limit == 20

    def test_reranking_index_documents_encoded_without_prompt(self):
        """Test that all 3 embedders encode documents with prompt=None in create_index()."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = MagicMock(spec=["encode", "embedding_dim"])
        dense_embedder.embedding_dim = 128
        dense_embedder.encode.return_value = np.zeros((2, 128))

        sparse_embedder = MagicMock(spec=["encode"])
        sparse_embedder.encode.return_value = [
            {"indices": [0, 1], "values": [0.5, 0.3]},
            {"indices": [2, 3], "values": [0.7, 0.2]},
        ]

        reranking_embedder = MagicMock(spec=["encode", "embedding_dim"])
        reranking_embedder.embedding_dim = 128
        reranking_embedder.encode.return_value = [
            [[0.1] * 128, [0.2] * 128],  # Multi-vector for doc 0
            [[0.3] * 128, [0.4] * 128],  # Multi-vector for doc 1
        ]

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

        # Verify all 3 embedders called with prompt=None during create_index
        dense_embedder.encode.assert_called_once_with(texts, prompt=None)
        sparse_embedder.encode.assert_called_once_with(texts, prompt=None)
        reranking_embedder.encode.assert_called_once_with(texts, prompt=None)

    def test_reranking_index_queries_dense_and_reranking_with_prompt(self):
        """Test that dense + reranking embedders use query_prompt, sparse does not."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = MagicMock(spec=["encode", "embedding_dim"])
        dense_embedder.embedding_dim = 128
        dense_embedder.encode.side_effect = [
            # First call: documents (create_index)
            np.zeros((2, 128)),
            # Second call: queries (search)
            np.zeros((1, 128)),
        ]

        sparse_embedder = MagicMock(spec=["encode"])
        sparse_embedder.encode.side_effect = [
            # First call: documents
            [
                {"indices": [0, 1], "values": [0.5, 0.3]},
                {"indices": [2, 3], "values": [0.7, 0.2]},
            ],
            # Second call: queries
            [{"indices": [4, 5], "values": [0.6, 0.4]}],
        ]

        reranking_embedder = MagicMock(spec=["encode", "embedding_dim"])
        reranking_embedder.embedding_dim = 128
        reranking_embedder.encode.side_effect = [
            # First call: documents
            [
                [[0.1] * 128, [0.2] * 128],
                [[0.3] * 128, [0.4] * 128],
            ],
            # Second call: queries
            [[[0.5] * 128, [0.6] * 128]],
        ]

        # Mock query_points response
        mock_scored_points = [
            ScoredPoint(id=0, version=0, score=0.9, payload={}, vector={}),
        ]
        mock_client.query_points.return_value = mock_scored_points

        index = QdrantHybridRerankingIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,
        )

        # Create index first
        texts = ["Apple Inc.", "Microsoft Corp."]
        index.create_index(texts)

        # Execute search with query_prompt
        query_prompt = "Find duplicate companies accounting for abbreviations"
        index.search(["Apple"], k=1, query_prompt=query_prompt)

        # Verify dense embedder: documents with prompt=None, queries with prompt
        assert dense_embedder.encode.call_count == 2
        assert dense_embedder.encode.call_args_list[0] == ((texts,), {"prompt": None})
        assert dense_embedder.encode.call_args_list[1] == ((["Apple"],), {"prompt": query_prompt})

        # Verify sparse embedder: ALWAYS prompt=None (BM25 doesn't use prompts)
        assert sparse_embedder.encode.call_count == 2
        assert sparse_embedder.encode.call_args_list[0] == ((texts,), {"prompt": None})
        assert sparse_embedder.encode.call_args_list[1] == ((["Apple"],), {"prompt": None})

        # Verify reranking embedder: documents with prompt=None, queries with prompt
        assert reranking_embedder.encode.call_count == 2
        assert reranking_embedder.encode.call_args_list[0] == ((texts,), {"prompt": None})
        assert reranking_embedder.encode.call_args_list[1] == (
            (["Apple"],),
            {"prompt": query_prompt},
        )

    def test_reranking_index_no_query_prompt_backward_compatible(self):
        """Test that without query_prompt parameter, all embedders use prompt=None."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = MagicMock(spec=["encode", "embedding_dim"])
        dense_embedder.embedding_dim = 128
        dense_embedder.encode.side_effect = [
            np.zeros((2, 128)),  # documents
            np.zeros((1, 128)),  # queries
        ]

        sparse_embedder = MagicMock(spec=["encode"])
        sparse_embedder.encode.side_effect = [
            [
                {"indices": [0, 1], "values": [0.5, 0.3]},
                {"indices": [2, 3], "values": [0.7, 0.2]},
            ],
            [{"indices": [4, 5], "values": [0.6, 0.4]}],
        ]

        reranking_embedder = MagicMock(spec=["encode", "embedding_dim"])
        reranking_embedder.embedding_dim = 128
        reranking_embedder.encode.side_effect = [
            [
                [[0.1] * 128, [0.2] * 128],
                [[0.3] * 128, [0.4] * 128],
            ],
            [[[0.5] * 128, [0.6] * 128]],
        ]

        mock_client.query_points.return_value = [
            ScoredPoint(id=0, version=0, score=0.9, payload={}, vector={}),
        ]

        # Create index WITHOUT query_prompt parameter (backward compatibility)
        index = QdrantHybridRerankingIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,
            # No query_prompt parameter
        )

        # Create index and search
        texts = ["Apple Inc.", "Microsoft Corp."]
        index.create_index(texts)
        index.search(["Apple"], k=1)

        # Verify ALL embedders use prompt=None for both documents AND queries
        assert dense_embedder.encode.call_count == 2
        assert dense_embedder.encode.call_args_list[0] == ((texts,), {"prompt": None})
        assert dense_embedder.encode.call_args_list[1] == ((["Apple"],), {"prompt": None})

        assert sparse_embedder.encode.call_count == 2
        assert sparse_embedder.encode.call_args_list[0] == ((texts,), {"prompt": None})
        assert sparse_embedder.encode.call_args_list[1] == ((["Apple"],), {"prompt": None})

        assert reranking_embedder.encode.call_count == 2
        assert reranking_embedder.encode.call_args_list[0] == ((texts,), {"prompt": None})
        assert reranking_embedder.encode.call_args_list[1] == ((["Apple"],), {"prompt": None})

    def test_reranking_index_search_all_uses_cached_embeddings(self):
        """Test that search_all() uses cached dense embeddings from create_index.

        Performance optimization: search_all should reuse dense embeddings,
        not re-encode. Sparse and reranking embeddings still need to be encoded.
        """
        # Setup
        mock_client = MagicMock()
        dense_embedder = MagicMock(spec=["encode", "embedding_dim"])
        dense_embedder.embedding_dim = 128
        dense_embedder.encode.return_value = np.zeros((2, 128))  # documents only

        sparse_embedder = MagicMock(spec=["encode"])
        sparse_embedder.encode.side_effect = [
            [
                {"indices": [0, 1], "values": [0.5, 0.3]},
                {"indices": [2, 3], "values": [0.7, 0.2]},
            ],
            [
                {"indices": [4, 5], "values": [0.6, 0.4]},
                {"indices": [6, 7], "values": [0.8, 0.1]},
            ],
        ]

        reranking_embedder = MagicMock(spec=["encode", "embedding_dim"])
        reranking_embedder.embedding_dim = 128
        reranking_embedder.encode.side_effect = [
            [
                [[0.1] * 128, [0.2] * 128],
                [[0.3] * 128, [0.4] * 128],
            ],
            [
                [[0.5] * 128, [0.6] * 128],
                [[0.7] * 128, [0.8] * 128],
            ],
        ]

        mock_client.query_points.side_effect = [
            [ScoredPoint(id=0, version=0, score=1.0, payload={}, vector={})],
            [ScoredPoint(id=1, version=0, score=1.0, payload={}, vector={})],
        ]

        index = QdrantHybridRerankingIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,
        )

        # Create index
        texts = ["Apple Inc.", "Microsoft Corp."]
        index.create_index(texts)

        # Verify create_index encoded WITHOUT prompt (documents don't use instructions)
        assert dense_embedder.encode.call_count == 1
        assert dense_embedder.encode.call_args == ((texts,), {"prompt": None})

        # Execute search_all with query_prompt - should NOT call dense embedder (uses cache)
        query_prompt = "Find duplicates"
        index.search_all(k=1, query_prompt=query_prompt)

        # Verify dense embedder NOT called again (cached embeddings)
        assert dense_embedder.encode.call_count == 1, "Dense embedder should not be called (cache)"

        # Verify sparse embedder STILL called (no caching for sparse)
        assert sparse_embedder.encode.call_count == 2

        # Verify reranking embedder STILL called (no caching for late-interaction)
        assert reranking_embedder.encode.call_count == 2

        # Sparse should still use prompt=None
        assert sparse_embedder.encode.call_args_list[1] == ((texts,), {"prompt": None})

        # Reranking should use query_prompt
        assert reranking_embedder.encode.call_args_list[1] == ((texts,), {"prompt": query_prompt})

    def test_reranking_index_uses_search_time_query_prompt(self):
        """Test that query_prompt parameter in search() controls prompt usage."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = MagicMock(spec=["encode", "embedding_dim"])
        dense_embedder.embedding_dim = 128
        dense_embedder.encode.side_effect = [
            np.zeros((2, 128)),  # documents
            np.zeros((1, 128)),  # queries
        ]

        sparse_embedder = MagicMock(spec=["encode"])
        sparse_embedder.encode.side_effect = [
            [
                {"indices": [0, 1], "values": [0.5, 0.3]},
                {"indices": [2, 3], "values": [0.7, 0.2]},
            ],
            [{"indices": [4, 5], "values": [0.6, 0.4]}],
        ]

        reranking_embedder = MagicMock(spec=["encode", "embedding_dim"])
        reranking_embedder.embedding_dim = 128
        reranking_embedder.encode.side_effect = [
            # Documents
            [
                [[0.1] * 128, [0.2] * 128],
                [[0.3] * 128, [0.4] * 128],
            ],
            # Queries
            [[[0.5] * 128, [0.6] * 128]],
        ]

        # Mock query_points response
        mock_scored_points = [
            ScoredPoint(id=0, version=0, score=0.9, payload={}, vector={}),
        ]
        mock_client.query_points.return_value = mock_scored_points

        # Create index WITHOUT query_prompt in constructor
        index = QdrantHybridRerankingIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,
        )

        # Create index
        texts = ["Apple Inc.", "Microsoft Corp."]
        index.create_index(texts)

        # Execute search WITH query_prompt parameter
        query_prompt = "Find duplicate companies accounting for abbreviations"
        index.search(["Apple"], k=1, query_prompt=query_prompt)

        # Verify dense embedder: documents with prompt=None, queries with prompt
        assert dense_embedder.encode.call_count == 2
        assert dense_embedder.encode.call_args_list[0] == ((texts,), {"prompt": None})
        assert dense_embedder.encode.call_args_list[1] == ((["Apple"],), {"prompt": query_prompt})

        # Verify sparse embedder: ALWAYS prompt=None
        assert sparse_embedder.encode.call_count == 2
        assert sparse_embedder.encode.call_args_list[0] == ((texts,), {"prompt": None})
        assert sparse_embedder.encode.call_args_list[1] == ((["Apple"],), {"prompt": None})

        # Verify reranking embedder: documents with prompt=None, queries with prompt
        assert reranking_embedder.encode.call_count == 2
        assert reranking_embedder.encode.call_args_list[0] == ((texts,), {"prompt": None})
        assert reranking_embedder.encode.call_args_list[1] == (
            (["Apple"],),
            {"prompt": query_prompt},
        )

    def test_reranking_index_search_all_passes_query_prompt(self):
        """Test that search_all() accepts and passes query_prompt to underlying search()."""
        # Setup
        mock_client = MagicMock()
        dense_embedder = MagicMock(spec=["encode", "embedding_dim"])
        dense_embedder.embedding_dim = 128
        dense_embedder.encode.return_value = np.zeros((2, 128))  # documents only

        sparse_embedder = MagicMock(spec=["encode"])
        sparse_embedder.encode.side_effect = [
            [
                {"indices": [0, 1], "values": [0.5, 0.3]},
                {"indices": [2, 3], "values": [0.7, 0.2]},
            ],
            [
                {"indices": [4, 5], "values": [0.6, 0.4]},
                {"indices": [6, 7], "values": [0.8, 0.1]},
            ],
        ]

        reranking_embedder = MagicMock(spec=["encode", "embedding_dim"])
        reranking_embedder.embedding_dim = 128
        reranking_embedder.encode.side_effect = [
            [
                [[0.1] * 128, [0.2] * 128],
                [[0.3] * 128, [0.4] * 128],
            ],
            [
                [[0.5] * 128, [0.6] * 128],
                [[0.7] * 128, [0.8] * 128],
            ],
        ]

        # Mock query_points responses
        mock_client.query_points.side_effect = [
            [ScoredPoint(id=0, version=0, score=0.9, payload={}, vector={})],
            [ScoredPoint(id=1, version=0, score=0.8, payload={}, vector={})],
        ]

        index = QdrantHybridRerankingIndex(
            client=mock_client,
            collection_name="test_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,
        )

        # Create index
        texts = ["Apple Inc.", "Microsoft Corp."]
        index.create_index(texts)

        # Execute search_all with query_prompt
        query_prompt = "Find duplicate entries"
        index.search_all(k=1, query_prompt=query_prompt)

        # Verify reranking embedder received prompt on search call
        assert reranking_embedder.encode.call_count == 2
        assert reranking_embedder.encode.call_args_list[1] == ((texts,), {"prompt": query_prompt})

    def test_reranking_index_configurable_upsert_batch_size(self):
        """Test that upsert_batch_size parameter controls batching during create_index."""
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
            upsert_batch_size=50,  # Custom batch size
        )

        # Create index with 150 texts (should create 3 batches of 50)
        texts = [f"Company {i}" for i in range(150)]
        index.create_index(texts)

        # Verify upsert was called 3 times with batches of 50
        assert mock_client.upsert.call_count == 3

        # Check each batch size
        for i, call in enumerate(mock_client.upsert.call_args_list):
            points = call[1]["points"]
            assert len(points) == 50, f"Batch {i} should have 50 points"

    def test_reranking_index_default_upsert_batch_size(self):
        """Test that default upsert_batch_size is 100."""
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
            # No upsert_batch_size specified - should default to 100
        )

        # Create index with 250 texts (should create 3 batches: 100, 100, 50)
        texts = [f"Company {i}" for i in range(250)]
        index.create_index(texts)

        # Verify upsert was called 3 times
        assert mock_client.upsert.call_count == 3

        # Check batch sizes
        batch_sizes = [len(call[1]["points"]) for call in mock_client.upsert.call_args_list]
        assert batch_sizes == [100, 100, 50]


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


@pytest.mark.integration
@pytest.mark.slow
def test_reranking_index_with_real_qdrant_integration():
    """Integration test with real Qdrant and FastEmbed embedders.

    This test validates the complete pipeline with actual model loading
    and vector operations. It's marked as slow due to model downloads.
    """
    # Setup - Import real implementations
    from qdrant_client import QdrantClient

    from langres.core.embeddings import (
        FastEmbedLateInteractionEmbedder,
        FastEmbedSparseEmbedder,
        SentenceTransformerEmbedder,
    )

    # In-memory Qdrant (no external service needed)
    client = QdrantClient(":memory:")

    # Lightweight models for fast testing
    dense = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")  # 384-dim
    sparse = FastEmbedSparseEmbedder(model_name="Qdrant/bm25")  # BM25
    reranking = FastEmbedLateInteractionEmbedder(
        model_name="colbert-ir/colbertv2.0"
    )  # 128-dim tokens

    index = QdrantHybridRerankingIndex(
        client=client,
        collection_name="test_integration",
        dense_embedder=dense,
        sparse_embedder=sparse,
        reranking_embedder=reranking,
        prefetch_limit=5,
    )

    # Test corpus (small to keep test fast)
    corpus = [
        "Apple Inc. is a technology company",
        "Microsoft Corporation develops software",
        "Google LLC provides search services",
        "Amazon.com is an e-commerce platform",
        "Meta Platforms runs social media",
    ]

    # Create index
    logger.info("Creating index with real embedders...")
    index.create_index(corpus)

    # Verify collection exists
    collections = client.get_collections()
    collection_names = [c.name for c in collections.collections]
    assert "test_integration" in collection_names
    logger.info("Collection created successfully")

    # Verify collection has correct vector configuration
    collection_info = client.get_collection("test_integration")
    assert "dense" in collection_info.config.params.vectors
    assert "reranking" in collection_info.config.params.vectors
    assert collection_info.config.params.sparse_vectors is not None
    assert "sparse" in collection_info.config.params.sparse_vectors
    logger.info("Vector configuration verified")

    # Test single query search
    logger.info("Testing single query search...")
    distances, indices = index.search("Apple Company", k=3)
    assert distances.shape == (3,)
    assert indices.shape == (3,)
    assert all(0 <= idx < len(corpus) for idx in indices)
    # MaxSim scores are not normalized to 0-1 (can be > 1)
    assert all(score >= 0 for score in distances)  # Scores should be non-negative
    logger.info(f"Single query results: indices={indices}, distances={distances}")

    # Test batch query search
    logger.info("Testing batch query search...")
    queries = ["Apple", "Microsoft"]
    distances, indices = index.search(queries, k=2)
    assert distances.shape == (2, 2)
    assert indices.shape == (2, 2)
    assert all(0 <= idx < len(corpus) for row in indices for idx in row)
    logger.info(f"Batch query results: indices=\n{indices}\ndistances=\n{distances}")

    # Test search_all (deduplication pattern)
    logger.info("Testing search_all (deduplication)...")
    distances, indices = index.search_all(k=3)
    assert distances.shape == (5, 3)
    assert indices.shape == (5, 3)
    logger.info(f"Search all results shape: {indices.shape}")

    # Sanity check: first neighbor should be itself (highest similarity)
    for i in range(len(corpus)):
        assert indices[i, 0] == i
        # Self-similarity should be highest score (MaxSim is not normalized to 1.0)
        # Just verify it's the maximum in that row
        assert distances[i, 0] == max(distances[i, :])
        logger.info(f"Item {i} self-similarity: {distances[i, 0]:.4f} (highest in row)")

    # Sanity check: results should not be random
    # For "Apple Company" query, first result should be "Apple Inc..." (index 0)
    apple_query_distances, apple_query_indices = index.search("Apple Company", k=3)
    assert apple_query_indices[0] == 0, (
        f"Expected Apple Inc. (0) as first result, got {apple_query_indices[0]}"
    )
    logger.info("Sanity check passed: Apple query correctly ranks Apple Inc. first")

    logger.info("All integration tests passed!")


class TestQdrantHybridRerankingIndexPrecomputedEmbeddings:
    """Tests for QdrantHybridRerankingIndex pre-computed embedding support (performance fix)."""

    def test_qdrant_reranking_index_internal_precomputed_dense(self):
        """Test that search() with _dense_embeddings parameter skips dense encoding.

        This is an internal API used by search_all() for performance optimization.
        Sparse and late-interaction embedders are still called because Qdrant
        requires sparse vectors and late-interaction multi-vectors for hybrid reranking.
        """
        dense_call_log = []
        sparse_call_log = []
        reranking_call_log = []

        class TrackingDenseEmbedder:
            embedding_dim = 128

            def encode(self, texts, prompt=None):
                dense_call_log.append({"texts": texts, "prompt": prompt})
                return np.random.rand(len(texts), 128).astype(np.float32)

        class TrackingSparseEmbedder:
            def encode(self, texts, prompt=None):
                sparse_call_log.append({"texts": texts, "prompt": prompt})
                return [{"indices": [i, i + 1], "values": [0.5, 0.3]} for i in range(len(texts))]

        class TrackingRerankingEmbedder:
            embedding_dim = 128

            def encode(self, texts, prompt=None):
                reranking_call_log.append({"texts": texts, "prompt": prompt})
                # Return multi-vectors (list of vectors per text)
                return [
                    [np.random.rand(128).astype(np.float32) for _ in range(3)]  # 3 token vectors
                    for _ in range(len(texts))
                ]

        mock_client = MagicMock()
        mock_client.query_points.return_value = [
            ScoredPoint(id=0, version=0, score=0.9, payload={}, vector={}),
            ScoredPoint(id=1, version=0, score=0.8, payload={}, vector={}),
        ]

        dense_embedder = TrackingDenseEmbedder()
        sparse_embedder = TrackingSparseEmbedder()
        reranking_embedder = TrackingRerankingEmbedder()

        index = QdrantHybridRerankingIndex(
            client=mock_client,
            collection_name="test",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,
        )

        # Create index
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(texts)

        # Clear tracking
        dense_call_log.clear()
        sparse_call_log.clear()
        reranking_call_log.clear()

        # Call search() with internal _dense_embeddings parameter
        precomputed_dense = np.random.rand(1, 128).astype(np.float32)
        distances, indices = index.search("Apple", k=2, _dense_embeddings=precomputed_dense)

        # Verify dense embedder NOT called (pre-computed embeddings)
        assert len(dense_call_log) == 0, (
            "Dense embedder should not be called with _dense_embeddings"
        )

        # Verify sparse embedder IS called (Qdrant limitation - must encode from text)
        assert len(sparse_call_log) == 1, "Sparse embedder must be called for hybrid search"

        # Verify reranking embedder IS called (late-interaction reranking requires text)
        assert len(reranking_call_log) == 1, (
            "Reranking embedder must be called for late-interaction"
        )

        # Verify results are valid
        assert distances.shape == (2,)
        assert indices.shape == (2,)

    def test_qdrant_reranking_index_search_all_no_reencoding_dense(self):
        """Test that search_all() reuses cached dense embeddings.

        Dense embeddings are cached and reused.
        Sparse and late-interaction embeddings still need to be recomputed (Qdrant limitation).
        """
        dense_call_log = []

        class TrackingDenseEmbedder:
            embedding_dim = 128

            def encode(self, texts, prompt=None):
                dense_call_log.append({"texts": texts, "prompt": prompt})
                return np.random.rand(len(texts), 128).astype(np.float32)

        class TrackingSparseEmbedder:
            def encode(self, texts, prompt=None):
                return [{"indices": [i, i + 1], "values": [0.5, 0.3]} for i in range(len(texts))]

        class TrackingRerankingEmbedder:
            embedding_dim = 128

            def encode(self, texts, prompt=None):
                # Return multi-vectors
                return [
                    [np.random.rand(128).astype(np.float32) for _ in range(3)]
                    for _ in range(len(texts))
                ]

        mock_client = MagicMock()
        mock_client.query_points.return_value = [
            ScoredPoint(id=0, version=0, score=1.0, payload={}, vector={}),
            ScoredPoint(id=1, version=0, score=0.8, payload={}, vector={}),
        ]

        dense_embedder = TrackingDenseEmbedder()
        sparse_embedder = TrackingSparseEmbedder()
        reranking_embedder = TrackingRerankingEmbedder()

        index = QdrantHybridRerankingIndex(
            client=mock_client,
            collection_name="test",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranking_embedder=reranking_embedder,
        )

        # Create index
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC", "Amazon"]
        index.create_index(texts)

        # Verify single dense encode call (create_index)
        assert len(dense_call_log) == 1

        # Call search_all - should NOT re-encode dense embeddings
        distances, indices = index.search_all(k=3)

        # Still only one dense encode call (no re-encoding)
        assert len(dense_call_log) == 1, "search_all() should NOT re-encode dense embeddings"

        # Verify results are valid
        assert distances.shape == (4, 3)
        assert indices.shape == (4, 3)
