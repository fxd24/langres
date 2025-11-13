"""Tests for sparse embedding providers."""

import logging

import pytest

from langres.core.embeddings import (
    FakeSparseEmbedder,
    FastEmbedSparseEmbedder,
    SparseEmbeddingProvider,
)

logger = logging.getLogger(__name__)


class TestFakeSparseEmbedder:
    """Tests for FakeSparseEmbedder test double."""

    def test_encode_returns_correct_structure(self):
        """Test that encode returns sparse vectors in Qdrant format."""
        embedder = FakeSparseEmbedder()
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]

        sparse_vectors = embedder.encode(texts)

        assert len(sparse_vectors) == 3
        for sv in sparse_vectors:
            assert "indices" in sv
            assert "values" in sv
            assert isinstance(sv["indices"], list)
            assert isinstance(sv["values"], list)
            assert len(sv["indices"]) == len(sv["values"])

    def test_encode_deterministic(self):
        """Test that FakeSparseEmbedder produces deterministic results."""
        embedder1 = FakeSparseEmbedder()
        embedder2 = FakeSparseEmbedder()

        texts = ["Apple Inc.", "Microsoft Corp."]

        result1 = embedder1.encode(texts)
        result2 = embedder2.encode(texts)

        assert result1 == result2

    def test_encode_single_text(self):
        """Test encoding a single text."""
        embedder = FakeSparseEmbedder()
        result = embedder.encode(["Apple Inc."])

        assert len(result) == 1
        assert "indices" in result[0]
        assert "values" in result[0]

    def test_protocol_compliance(self):
        """Test that FakeSparseEmbedder implements protocol."""
        embedder = FakeSparseEmbedder()

        assert hasattr(embedder, "encode")
        assert callable(embedder.encode)


class TestFastEmbedSparseEmbedder:
    """Tests for FastEmbedSparseEmbedder."""

    def test_encode_returns_correct_structure(self):
        """Test that encode returns sparse vectors in Qdrant format."""
        embedder = FastEmbedSparseEmbedder("Qdrant/bm25")
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]

        sparse_vectors = embedder.encode(texts)

        assert len(sparse_vectors) == 3
        for sv in sparse_vectors:
            assert "indices" in sv
            assert "values" in sv
            assert isinstance(sv["indices"], list)
            assert isinstance(sv["values"], list)
            assert len(sv["indices"]) == len(sv["values"])
            # BM25 should produce non-empty sparse vectors
            assert len(sv["indices"]) > 0

    def test_encode_single_text(self):
        """Test encoding a single text."""
        embedder = FastEmbedSparseEmbedder("Qdrant/bm25")
        result = embedder.encode(["Apple Inc."])

        assert len(result) == 1
        assert "indices" in result[0]
        assert "values" in result[0]
        assert len(result[0]["indices"]) > 0

    def test_encode_batch_texts(self):
        """Test encoding batch of texts."""
        embedder = FastEmbedSparseEmbedder("Qdrant/bm25")
        texts = [
            "Apple Inc. is a technology company",
            "Microsoft Corporation develops software",
            "Google LLC provides internet services",
        ]

        sparse_vectors = embedder.encode(texts)

        assert len(sparse_vectors) == 3
        # Each text should have different sparse vectors
        assert sparse_vectors[0] != sparse_vectors[1]
        assert sparse_vectors[1] != sparse_vectors[2]

    def test_lazy_model_loading(self):
        """Test that model is loaded lazily on first encode."""
        embedder = FastEmbedSparseEmbedder("Qdrant/bm25")

        # Model should not be loaded yet
        assert embedder._model is None

        # Encode should trigger loading
        embedder.encode(["test"])

        # Model should now be loaded
        assert embedder._model is not None

    def test_default_model_is_bm25(self):
        """Test that default model is Qdrant/bm25."""
        embedder = FastEmbedSparseEmbedder()

        assert embedder.model_name == "Qdrant/bm25"

    def test_protocol_compliance(self):
        """Test that FastEmbedSparseEmbedder implements protocol."""
        embedder = FastEmbedSparseEmbedder("Qdrant/bm25")

        assert hasattr(embedder, "encode")
        assert callable(embedder.encode)


class TestSparseEmbeddingProviderProtocol:
    """Tests for SparseEmbeddingProvider protocol compliance."""

    @pytest.mark.parametrize(
        "embedder_class,kwargs",
        [
            (FakeSparseEmbedder, {}),
            (FastEmbedSparseEmbedder, {"model_name": "Qdrant/bm25"}),
        ],
    )
    def test_protocol_methods_exist(self, embedder_class, kwargs):
        """Test that implementations have required protocol methods."""
        embedder = embedder_class(**kwargs)

        assert hasattr(embedder, "encode")
        assert callable(embedder.encode)

    @pytest.mark.parametrize(
        "embedder_class,kwargs",
        [
            (FakeSparseEmbedder, {}),
            (FastEmbedSparseEmbedder, {"model_name": "Qdrant/bm25"}),
        ],
    )
    def test_encode_workflow(self, embedder_class, kwargs):
        """Test that all implementations support encode workflow."""
        embedder = embedder_class(**kwargs)
        texts = ["Apple Inc.", "Microsoft Corp."]

        sparse_vectors = embedder.encode(texts)

        assert isinstance(sparse_vectors, list)
        assert len(sparse_vectors) == 2
        for sv in sparse_vectors:
            assert "indices" in sv
            assert "values" in sv
