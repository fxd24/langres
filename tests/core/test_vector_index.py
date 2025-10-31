"""Tests for vector index implementations."""

import logging

import numpy as np
import pytest

from langres.core.vector_index import FAISSIndex, FakeVectorIndex, VectorIndex

logger = logging.getLogger(__name__)


class TestFAISSIndex:
    """Tests for FAISSIndex implementation."""

    def test_build_and_search_l2_metric(self):
        """Test building index and searching with L2 metric."""
        index = FAISSIndex(metric="L2")

        # Create simple embeddings
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.9, 0.1, 0.0],  # Close to first embedding
            ],
            dtype=np.float32,
        )

        index.build(embeddings)

        # Search for nearest neighbors (k=2)
        distances, indices = index.search(embeddings, k=2)

        # Verify shape
        assert distances.shape == (4, 2)
        assert indices.shape == (4, 2)

        # First neighbor should always be itself (distance ≈ 0)
        assert np.allclose(distances[:, 0], 0.0, atol=1e-5)
        assert np.array_equal(indices[:, 0], [0, 1, 2, 3])

        # Fourth embedding should have first embedding as second neighbor
        assert indices[3, 1] == 0  # Second nearest to [0.9, 0.1, 0] is [1, 0, 0]

    def test_build_and_search_cosine_metric(self):
        """Test building index and searching with cosine similarity."""
        index = FAISSIndex(metric="cosine")

        # Create embeddings (will be normalized for cosine)
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 0.0],  # 45° between first two
            ],
            dtype=np.float32,
        )

        index.build(embeddings)

        # Search
        distances, indices = index.search(embeddings, k=2)

        assert distances.shape == (3, 2)
        assert indices.shape == (3, 2)

    def test_build_accepts_numpy_array(self):
        """Test that build accepts numpy arrays."""
        index = FAISSIndex(metric="L2")
        embeddings = np.random.rand(10, 128).astype(np.float32)

        # Should not raise
        index.build(embeddings)

    def test_search_before_build_raises_error(self):
        """Test that searching before building raises an error."""
        index = FAISSIndex(metric="L2")
        embeddings = np.random.rand(5, 128).astype(np.float32)

        with pytest.raises(RuntimeError, match="Index not built"):
            index.search(embeddings, k=3)

    def test_search_returns_correct_shape(self):
        """Test that search returns arrays with correct shape."""
        index = FAISSIndex(metric="L2")
        embeddings = np.random.rand(100, 64).astype(np.float32)

        index.build(embeddings)
        distances, indices = index.search(embeddings, k=10)

        assert distances.shape == (100, 10)
        assert indices.shape == (100, 10)
        assert distances.dtype == np.float32
        assert indices.dtype == np.int64

    def test_search_with_k_larger_than_dataset(self):
        """Test searching with k larger than dataset size."""
        index = FAISSIndex(metric="L2")
        embeddings = np.random.rand(5, 32).astype(np.float32)

        index.build(embeddings)

        # k=10 but only 5 embeddings - FAISS should handle this
        distances, indices = index.search(embeddings, k=10)

        # Should return k=5 (all available)
        assert distances.shape == (5, 10)
        assert indices.shape == (5, 10)

    def test_multiple_builds_replace_index(self):
        """Test that building multiple times replaces the index."""
        index = FAISSIndex(metric="L2")

        # First build
        embeddings1 = np.random.rand(10, 32).astype(np.float32)
        index.build(embeddings1)

        # Second build with different data
        embeddings2 = np.random.rand(20, 32).astype(np.float32)
        index.build(embeddings2)

        # Search should work with second dataset size
        distances, indices = index.search(embeddings2, k=5)
        assert distances.shape == (20, 5)


class TestFakeVectorIndex:
    """Tests for FakeVectorIndex test double."""

    def test_build_and_search_returns_correct_shape(self):
        """Test that FakeVectorIndex returns correct shapes."""
        index = FakeVectorIndex()
        embeddings = np.random.rand(50, 128).astype(np.float32)

        index.build(embeddings)
        distances, indices = index.search(embeddings, k=10)

        assert distances.shape == (50, 10)
        assert indices.shape == (50, 10)

    def test_search_returns_valid_indices(self):
        """Test that FakeVectorIndex returns valid indices."""
        index = FakeVectorIndex()
        n_samples = 30
        embeddings = np.random.rand(n_samples, 64).astype(np.float32)

        index.build(embeddings)
        _, indices = index.search(embeddings, k=5)

        # All indices should be within valid range [0, n_samples)
        assert np.all(indices >= 0)
        assert np.all(indices < n_samples)

    def test_search_includes_self_as_nearest(self):
        """Test that each query's nearest neighbor is itself."""
        index = FakeVectorIndex()
        embeddings = np.random.rand(20, 32).astype(np.float32)

        index.build(embeddings)
        _, indices = index.search(embeddings, k=5)

        # First neighbor should be itself
        expected_self_indices = np.arange(20)
        assert np.array_equal(indices[:, 0], expected_self_indices)

    def test_deterministic_results(self):
        """Test that FakeVectorIndex produces deterministic results."""
        index1 = FakeVectorIndex()
        index2 = FakeVectorIndex()

        embeddings = np.random.rand(10, 16).astype(np.float32)

        index1.build(embeddings)
        index2.build(embeddings)

        distances1, indices1 = index1.search(embeddings, k=3)
        distances2, indices2 = index2.search(embeddings, k=3)

        np.testing.assert_array_equal(distances1, distances2)
        np.testing.assert_array_equal(indices1, indices2)

    def test_search_before_build_raises_error(self):
        """Test that searching before building raises an error."""
        index = FakeVectorIndex()
        embeddings = np.random.rand(5, 32).astype(np.float32)

        with pytest.raises(RuntimeError, match="Index not built"):
            index.search(embeddings, k=3)

    def test_protocol_compliance(self):
        """Test that FakeVectorIndex implements VectorIndex protocol."""
        index = FakeVectorIndex()

        assert hasattr(index, "build")
        assert hasattr(index, "search")
        assert callable(index.build)
        assert callable(index.search)


class TestVectorIndexProtocol:
    """Tests for VectorIndex protocol compliance."""

    @pytest.mark.parametrize(
        "index_class,kwargs",
        [
            (FAISSIndex, {"metric": "L2"}),
            (FakeVectorIndex, {}),
        ],
    )
    def test_protocol_methods_exist(self, index_class, kwargs):
        """Test that implementations have required protocol methods."""
        index = index_class(**kwargs)

        assert hasattr(index, "build")
        assert hasattr(index, "search")
        assert callable(index.build)
        assert callable(index.search)

    @pytest.mark.parametrize(
        "index_class,kwargs",
        [
            (FAISSIndex, {"metric": "L2"}),
            (FakeVectorIndex, {}),
        ],
    )
    def test_build_search_workflow(self, index_class, kwargs):
        """Test that all implementations support build → search workflow."""
        index = index_class(**kwargs)
        embeddings = np.random.rand(10, 32).astype(np.float32)

        # Build should accept embeddings
        index.build(embeddings)

        # Search should return distances and indices
        distances, indices = index.search(embeddings, k=3)

        assert isinstance(distances, np.ndarray)
        assert isinstance(indices, np.ndarray)
        assert distances.shape == (10, 3)
        assert indices.shape == (10, 3)
