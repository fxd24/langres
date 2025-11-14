"""Vector index implementations for approximate nearest neighbor search.

This module provides abstractions for vector indexing and similarity search.
The index owns the embedding logic, providing a clean separation between
domain logic (handled by blockers) and technical concerns (handled by indexes).

Key design principles:
- Index owns embedder: No blocker-level embedding dependencies
- Lifecycle separation: create_index() (preprocessing) vs search() (runtime)
- Native batching: Leverage FAISS/Qdrant batch APIs for efficiency
- Text-focused: Optimized for text inputs (extensible to multi-modal later)
"""

import logging
from typing import Literal, Protocol

import faiss  # type: ignore[import-untyped]
import numpy as np

from langres.core.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


class VectorIndex(Protocol):
    """Protocol for vector indexing with lifecycle separation and native batching.

    The index owns embedding logic and provides three key operations:
    1. create_index(texts): Preprocessing - embed and build searchable index
    2. search(query_texts, k): Runtime - search with text queries (native batching)
    3. search_all(k): Runtime - efficient deduplication pattern (all vs all)

    This design enables:
    - Clean separation: Index handles technical concerns (embedding, indexing)
    - Blocker simplicity: Blocker only handles domain logic
    - Efficient batching: Leverage native vector DB batch APIs
    - Swappable backends: FAISS, Qdrant, or custom implementations

    Example (FAISS backend):
        from langres.core.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        index = FAISSIndex(embedder=embedder, metric="cosine")

        # Preprocessing: build index from texts
        corpus = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(corpus)

        # Runtime: search with single query
        distances, indices = index.search("Apple Company", k=2)
        # Returns 1D arrays: distances=(2,), indices=(2,)

        # Runtime: search with batch queries (native batching!)
        queries = ["Apple", "Google"]
        distances, indices = index.search(queries, k=2)
        # Returns 2D arrays: distances=(2,2), indices=(2,2)

        # Runtime: deduplication pattern (all vs all)
        distances, indices = index.search_all(k=10)
        # Returns 2D arrays: distances=(3,10), indices=(3,10)
    """

    def create_index(self, texts: list[str]) -> None:
        """Preprocessing: Build searchable index from text corpus.

        The index handles embedding internally using its configured embedder.
        This is a one-time operation - build once, query many times.

        Args:
            texts: Corpus texts to embed and index.

        Note:
            Calling create_index() multiple times replaces the existing index.
            This enables rebuilding with new data without recreating the index object.
        """
        ...  # pragma: no cover

    def search(
        self,
        query_texts: str | list[str] | np.ndarray,
        k: int,
        query_prompt: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Runtime: Search for k nearest neighbors using text queries or pre-computed embeddings.

        Supports both single query and batch queries. For batch queries,
        implementations MUST use native batching (e.g., FAISS batch search,
        Qdrant query_batch_points) for efficiency.

        Args:
            query_texts: Single text, list of texts, or pre-computed embeddings (np.ndarray).
                - str: Single text query
                - list[str]: Batch of text queries
                - np.ndarray: Pre-computed embeddings (shape: (dim,) or (N, dim))
            k: Number of nearest neighbors to return per query.
            query_prompt: Optional instruction prompt for query encoding (asymmetric search).
                Applied only to text queries. Ignored for pre-computed embeddings.
                Default: None.

        Returns:
            Tuple of (distances, indices):
            - If single query: distances=(k,), indices=(k,)
            - If batch: distances=(N,k), indices=(N,k)

        Raises:
            RuntimeError: If search() is called before create_index().

        Note:
            When using text queries, the index embeds texts on-the-fly using its embedder.
            When using pre-computed embeddings, no encoding is performed (performance optimization).
        """
        ...  # pragma: no cover

    def search_all(self, k: int, query_prompt: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Runtime: Search all corpus items against each other (deduplication).

        Efficient batch operation that uses cached corpus embeddings.
        No re-embedding needed - reuses embeddings from create_index().

        Args:
            k: Number of nearest neighbors to return per corpus item.
            query_prompt: Optional instruction prompt for query encoding.
                Typically None for deduplication (symmetric encoding).
                Default: None.

        Returns:
            Tuple of (distances, indices), both shape (N, k) where N = corpus size.

        Raises:
            RuntimeError: If search_all() is called before create_index().

        Note:
            For deduplication pattern where you want neighbors for all items.
            More efficient than calling search(all_texts, k) because it reuses
            cached embeddings without re-encoding.
        """
        ...  # pragma: no cover


class FAISSIndex:
    """FAISS-backed index with native batch search and lifecycle separation.

    The index owns the embedder and provides three operations:
    1. create_index(texts) - Preprocessing: embed texts and build FAISS index
    2. search(query_texts, k) - Runtime: search with text queries (native batching)
    3. search_all(k) - Runtime: efficient deduplication (all vs all)

    Supports L2 (Euclidean) and cosine similarity metrics.

    Example:
        from langres.core.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        index = FAISSIndex(embedder=embedder, metric="cosine")

        # Preprocessing
        corpus = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(corpus)

        # Runtime: single query
        distances, indices = index.search("Apple Company", k=2)
        # Returns: distances=(2,), indices=(2,)

        # Runtime: batch queries (native batching!)
        distances, indices = index.search(["Apple", "Google"], k=2)
        # Returns: distances=(2,2), indices=(2,2)

        # Runtime: deduplication
        distances, indices = index.search_all(k=10)
        # Returns: distances=(3,10), indices=(3,10)
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        metric: Literal["L2", "cosine"] = "L2",
    ):
        """Initialize FAISSIndex.

        Args:
            embedder: Provider for generating embeddings from texts.
            metric: Distance metric ("L2" or "cosine").
        """
        self.embedder = embedder
        self.metric = metric

        # State (populated by create_index)
        self._corpus_embeddings: np.ndarray | None = None
        self._corpus_texts: list[str] | None = None
        self._index: faiss.Index | None = None

    def create_index(self, texts: list[str]) -> None:
        """Build FAISS index from text corpus.

        Embeds texts using the configured embedder and builds searchable index.

        Args:
            texts: Corpus texts to embed and index.
        """
        # 1. Embed corpus (index handles this!)
        # Documents are always encoded without prompts
        self._corpus_embeddings = self.embedder.encode(texts).astype(np.float32)

        # 2. Create FAISS index based on metric
        dim = self._corpus_embeddings.shape[1]

        if self.metric == "L2":
            self._index = faiss.IndexFlatL2(dim)
        elif self.metric == "cosine":
            # Normalize for cosine similarity (in-place)
            faiss.normalize_L2(self._corpus_embeddings)
            self._index = faiss.IndexFlatIP(dim)  # Inner product
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # 3. Add embeddings to index
        self._index.add(self._corpus_embeddings)

        # 4. Cache corpus texts for search_all() (needed for query_prompt support)
        self._corpus_texts = texts

        logger.info(
            "Built FAISS index with %d vectors, dim=%d, metric=%s",
            self._corpus_embeddings.shape[0],
            dim,
            self.metric,
        )

    def search(
        self,
        query_texts: str | list[str] | np.ndarray,
        k: int,
        query_prompt: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors using text queries or pre-computed embeddings.

        Supports both single query and batch queries with native FAISS batching.

        Args:
            query_texts: Single text, list of texts, or pre-computed embeddings.
            k: Number of neighbors per query.
            query_prompt: Optional instruction prompt for query encoding (asymmetric search).
                Applied only to text queries. Ignored for pre-computed embeddings.
                Default: None.

        Returns:
            - If single query: distances=(k,), indices=(k,)
            - If batch: distances=(N,k), indices=(N,k)
        """
        if self._index is None:
            raise RuntimeError("Index not built. Must call create_index() first.")

        # Separate code paths for testability and clarity
        if isinstance(query_texts, np.ndarray):
            # Path 1: Pre-computed embeddings (no encoding)
            query_embeddings = query_texts.astype(np.float32)
            is_single = query_embeddings.ndim == 1
            if is_single:
                query_embeddings = query_embeddings.reshape(1, -1)
        else:
            # Path 2: Text queries (encode with optional prompt)
            is_single = isinstance(query_texts, str)
            texts: list[str] = [query_texts] if is_single else query_texts  # type: ignore[assignment,list-item]
            query_embeddings = self.embedder.encode(texts, prompt=query_prompt).astype(np.float32)

        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query_embeddings)

        # NATIVE BATCH SEARCH (single FAISS call!)
        distances, indices = self._index.search(query_embeddings, k)

        # Return shape depends on input
        if is_single:
            return distances[0], indices[0]
        else:
            return distances, indices

    def search_all(self, k: int, query_prompt: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Search all corpus items against each other (deduplication pattern).

        Reuses cached corpus embeddings for efficiency. For deduplication,
        symmetric encoding (no prompt) is typical since both query and document
        sides come from the same corpus.

        Args:
            k: Number of neighbors per corpus item.
            query_prompt: Optional instruction prompt for query encoding.
                Typically None for deduplication (symmetric encoding).
                Default: None.

        Returns:
            distances: shape (N, k) where N = corpus size
            indices: shape (N, k)
        """
        if self._corpus_embeddings is None or self._index is None:
            raise RuntimeError("Index not built. Must call create_index() first.")

        # Pass pre-computed embeddings to search() - no re-encoding!
        # query_prompt parameter is passed through but ignored for pre-computed embeddings
        return self.search(self._corpus_embeddings, k, query_prompt=query_prompt)

    # ============ OLD API (for backward compatibility during transition) ============
    def build(self, embeddings: np.ndarray) -> None:
        """DEPRECATED: Use create_index() instead.

        Build FAISS index from pre-computed embeddings.
        """
        # Convert to float32 (FAISS requirement)
        embeddings = embeddings.astype(np.float32)
        dim = embeddings.shape[1]

        if self.metric == "L2":
            self._index = faiss.IndexFlatL2(dim)
        elif self.metric == "cosine":
            faiss.normalize_L2(embeddings)
            self._index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        self._index.add(embeddings)
        self._corpus_embeddings = embeddings

        logger.info(
            "Built FAISS index with %d vectors, dim=%d, metric=%s",
            embeddings.shape[0],
            dim,
            self.metric,
        )


class FakeVectorIndex:
    """Test double for VectorIndex with deterministic results.

    Produces fake search results that are:
    - Deterministic: Same inputs always produce same outputs
    - Valid: All indices are within bounds
    - Fast: No actual embedding or similarity computation

    Perfect for testing blocker logic without expensive operations.

    Example:
        index = FakeVectorIndex()
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]

        index.create_index(texts)

        # Single query
        distances, indices = index.search("Apple", k=2)
        # Returns: distances=(2,), indices=(2,)

        # Batch queries
        distances, indices = index.search(["Apple", "Google"], k=2)
        # Returns: distances=(2,2), indices=(2,2)

        # Deduplication
        distances, indices = index.search_all(k=2)
        # Returns: distances=(3,2), indices=(3,2)
    """

    def __init__(self) -> None:
        """Initialize FakeVectorIndex."""
        self._n_samples: int | None = None

    def create_index(self, texts: list[str]) -> None:
        """Record corpus size for generating valid indices.

        Args:
            texts: Corpus texts (only length is used).
        """
        self._n_samples = len(texts)
        logger.debug("FakeVectorIndex: recorded %d samples", self._n_samples)

    def search(
        self,
        query_texts: str | list[str] | np.ndarray,
        k: int,
        query_prompt: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate fake search results (deterministic).

        Args:
            query_texts: Single text, list of texts, or pre-computed embeddings.
            k: Number of neighbors per query.
            query_prompt: Optional instruction prompt (ignored by fake implementation).

        Returns:
            - If single query: distances=(k,), indices=(k,)
            - If batch: distances=(N,k), indices=(N,k)
        """
        if self._n_samples is None:
            raise RuntimeError("Index not built. Call create_index() first.")

        # Handle single vs batch (same logic for text and embeddings)
        if isinstance(query_texts, np.ndarray):
            # Pre-computed embeddings
            is_single = query_texts.ndim == 1
            n_queries = 1 if is_single else query_texts.shape[0]
        else:
            # Text queries
            is_single = isinstance(query_texts, str)
            n_queries = 1 if is_single else len(query_texts)

        # Generate deterministic indices
        indices = np.zeros((n_queries, k), dtype=np.int64)
        distances = np.zeros((n_queries, k), dtype=np.float32)

        for i in range(n_queries):
            for j in range(k):
                indices[i, j] = j % self._n_samples
                distances[i, j] = j * 0.1

        # Return shape depends on input
        if is_single:
            return distances[0], indices[0]
        else:
            return distances, indices

    def search_all(self, k: int, query_prompt: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate fake deduplication results (deterministic).

        Args:
            k: Number of neighbors per corpus item.
            query_prompt: Optional instruction prompt (ignored by fake implementation).

        Returns:
            distances: shape (N, k) where N = corpus size
            indices: shape (N, k)
        """
        if self._n_samples is None:
            raise RuntimeError("Index not built. Call create_index() first.")

        # Generate deterministic pattern: for item i, neighbors are [i, (i+1)%N, ...]
        indices = np.zeros(
            (self._n_samples, k), dtype=np.int64
        )  # TODO mimic behavior of FAISS, where the search is passed on to search function. do not reimplement twice.
        distances = np.zeros((self._n_samples, k), dtype=np.float32)

        for i in range(self._n_samples):
            for j in range(k):
                indices[i, j] = (i + j) % self._n_samples
                distances[i, j] = j * 0.1

        return distances, indices

    # ============ OLD API (for backward compatibility) ============
    def build(self, embeddings: np.ndarray) -> None:
        """DEPRECATED: Use create_index() instead."""
        self._n_samples = embeddings.shape[0]
        logger.debug("FakeVectorIndex: recorded %d samples", self._n_samples)
