"""Vector index implementations for approximate nearest neighbor search.

This module provides abstractions for vector indexing and similarity search,
separating index management from embedding computation to enable:
- Swapping index backends (FAISS, Annoy, cloud services)
- Using pre-computed embeddings (cached or from databases)
- Rebuilding indices without re-computing embeddings

The core abstraction is the VectorIndex Protocol, which defines a standard
interface for building indices and searching for nearest neighbors.
"""

import logging
from typing import Literal, Protocol

import faiss  # type: ignore[import-untyped]
import numpy as np

logger = logging.getLogger(__name__)


class VectorIndex(Protocol):
    """Protocol for vector indexing and approximate nearest neighbor search.

    This abstraction handles index building and ANN search, separate from
    embedding computation. This separation enables several key workflows:

    1. **Using pre-computed embeddings**:
       Load embeddings from disk/DB that were computed earlier,
       build index, search immediately.

    2. **Rebuilding indices without re-embedding**:
       After optimization, rebuild index with different parameters
       without re-computing expensive embeddings.

    3. **Swapping index backends**:
       Try FAISS, Annoy, ScaNN, or cloud services (Pinecone, Qdrant)
       with the same embeddings.

    Example (using pre-computed embeddings):
        # Load cached embeddings from optimization phase
        embeddings = np.load("production_embeddings.npy")

        # Build index and search
        index = FAISSIndex(metric="L2")
        index.build(embeddings)
        distances, indices = index.search(embeddings, k=10)

    Example (rebuilding index with different parameters):
        # Already have embeddings
        embeddings = embedder.encode(texts)

        # Try different index configurations
        for metric in ["L2", "cosine"]:
            index = FAISSIndex(metric=metric)
            index.build(embeddings)
            recall = evaluate_recall(index.search(embeddings, k=10))
            print(f"{metric}: recall={recall}")
    """

    def build(self, embeddings: np.ndarray) -> None:
        """Build index from pre-computed embeddings.

        Args:
            embeddings: Pre-computed embedding vectors, shape (N, dim).
                Must be a 2D numpy array of float32 values.

        Note:
            This method should be idempotent - calling build() multiple
            times should replace the existing index each time.

        Note:
            Implementations may modify embeddings in-place (e.g., L2
            normalization for cosine similarity). Callers should not
            rely on embeddings being unchanged after build().
        """
        ...

    def search(self, query_embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors for each query embedding.

        Args:
            query_embeddings: Query vectors, shape (N, dim).
            k: Number of nearest neighbors to return per query.

        Returns:
            Tuple of (distances, indices):
            - distances: shape (N, k), similarity/distance scores
            - indices: shape (N, k), indices of neighbors in the
              original embeddings passed to build()

        Raises:
            RuntimeError: If search() is called before build().

        Note:
            Indices refer to positions in the embeddings array passed
            to build(). For example, indices[0, 0] = 5 means the
            nearest neighbor of query 0 is embeddings[5].

        Note:
            Distance semantics depend on the metric:
            - L2: smaller = more similar (0 = identical)
            - Cosine: larger = more similar (inner product after normalization)
        """
        ...


class FAISSIndex:
    """FAISS-based vector index for approximate nearest neighbor search.

    This implementation wraps Facebook AI's FAISS library for efficient
    similarity search. It supports both L2 (Euclidean) and cosine similarity
    metrics.

    FAISS is well-suited for:
    - In-memory search on single machine (100k-10M vectors)
    - CPU or GPU acceleration
    - Exact search (IndexFlat) or approximate search (IndexIVF, IndexHNSW)

    For larger datasets or distributed search, consider cloud alternatives
    like Pinecone, Qdrant, or Weaviate.

    Example:
        index = FAISSIndex(metric="L2")
        embeddings = np.random.rand(1000, 384).astype(np.float32)
        index.build(embeddings)
        distances, indices = index.search(embeddings, k=10)

    Note:
        This implementation uses IndexFlat (exact search) for simplicity
        and correctness. For large datasets (>1M vectors), consider
        using approximate indices (IndexIVFFlat, IndexHNSW) for speed.

    Note:
        FAISS expects float32 arrays. This implementation converts
        inputs to float32 automatically.
    """

    def __init__(self, metric: Literal["L2", "cosine"] = "L2"):
        """Initialize FAISSIndex.

        Args:
            metric: Distance metric to use.
                - "L2": Euclidean distance (smaller = more similar)
                - "cosine": Cosine similarity via inner product
                  (larger = more similar)

        Note:
            For cosine similarity, embeddings will be L2-normalized
            during build() so that inner product equals cosine similarity.
        """
        self.metric = metric
        self._index: faiss.Index | None = None

    def build(self, embeddings: np.ndarray) -> None:
        """Build FAISS index from pre-computed embeddings.

        Args:
            embeddings: Pre-computed embeddings, shape (N, dim).

        Note:
            For cosine metric, embeddings are L2-normalized in-place.
            For L2 metric, embeddings are used as-is.

        Note:
            Calling build() multiple times replaces the existing index.
        """
        # Convert to float32 (FAISS requirement)
        embeddings = embeddings.astype(np.float32)

        # Get embedding dimension
        dim = embeddings.shape[1]

        # Create appropriate index based on metric
        if self.metric == "L2":
            self._index = faiss.IndexFlatL2(dim)
        elif self.metric == "cosine":
            # For cosine similarity, normalize embeddings and use inner product
            faiss.normalize_L2(embeddings)  # Modifies in-place
            self._index = faiss.IndexFlatIP(dim)  # Inner Product
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # Add embeddings to index
        self._index.add(embeddings)

        logger.info(
            "Built FAISS index with %d vectors, dim=%d, metric=%s",
            embeddings.shape[0],
            dim,
            self.metric,
        )

    def search(self, query_embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors using FAISS.

        Args:
            query_embeddings: Query vectors, shape (N, dim).
            k: Number of neighbors per query.

        Returns:
            Tuple of (distances, indices), each shape (N, k).

        Raises:
            RuntimeError: If index hasn't been built yet.

        Note:
            For cosine metric, query embeddings are L2-normalized
            before search (in-place modification).
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build() first.")

        # Convert to float32
        query_embeddings = query_embeddings.astype(np.float32)

        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query_embeddings)  # Modifies in-place

        # Search
        distances, indices = self._index.search(query_embeddings, k)

        return distances, indices


class FakeVectorIndex:
    """Test double for VectorIndex that produces deterministic neighbors.

    This implementation creates fake search results that are:
    1. Deterministic: same embeddings always produce same neighbors
    2. Valid: all indices are within bounds [0, N)
    3. Include self: each query's nearest neighbor is itself
    4. Fast: no actual similarity computation

    This is crucial for testing VectorBlocker logic without expensive
    FAISS operations or needing valid embeddings.

    Example:
        index = FakeVectorIndex()
        embeddings = np.random.rand(100, 128).astype(np.float32)
        index.build(embeddings)
        distances, indices = index.search(embeddings, k=10)
        # Returns deterministic (100, 10) arrays instantly

    Note:
        The fake neighbors are generated using a simple deterministic
        pattern: for query i, neighbors are [i, (i+1)%N, (i+2)%N, ...].
        Distances are synthetic (just scaled indices).
    """

    def __init__(self) -> None:
        """Initialize FakeVectorIndex."""
        self._n_samples: int | None = None

    def build(self, embeddings: np.ndarray) -> None:
        """Record the number of samples for generating valid indices.

        Args:
            embeddings: Embeddings to "index" (only shape is used).

        Note:
            This doesn't actually build an index - it just records
            the dataset size for generating valid neighbor indices.
        """
        self._n_samples = embeddings.shape[0]
        logger.debug("FakeVectorIndex: recorded %d samples", self._n_samples)

    def search(self, query_embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate fake nearest neighbor results.

        Args:
            query_embeddings: Query vectors, shape (N, dim).
            k: Number of neighbors per query.

        Returns:
            Tuple of (distances, indices), each shape (N, k).
            - indices[i, 0] = i (self is nearest neighbor)
            - indices[i, j] = (i + j) % n_samples (deterministic pattern)
            - distances[i, j] = j * 0.1 (synthetic distances)

        Raises:
            RuntimeError: If build() hasn't been called yet.

        Note:
            This is a deterministic fake - same inputs always produce
            same outputs, which is perfect for testing.
        """
        if self._n_samples is None:
            raise RuntimeError("Index not built. Call build() first.")

        n_queries = query_embeddings.shape[0]

        # Generate deterministic neighbor indices
        # Pattern: for query i, neighbors are [i, (i+1)%N, (i+2)%N, ...]
        indices = np.zeros((n_queries, k), dtype=np.int64)
        distances = np.zeros((n_queries, k), dtype=np.float32)

        for i in range(n_queries):
            for j in range(k):
                indices[i, j] = (i + j) % self._n_samples
                # Synthetic distances: 0 for self, then increasing
                distances[i, j] = j * 0.1

        return distances, indices
