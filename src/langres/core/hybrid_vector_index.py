"""Hybrid vector index implementation using Qdrant.

This module provides hybrid search combining dense (semantic) and sparse (keyword)
vectors using Qdrant's built-in fusion capabilities.

Key features:
- Dense + sparse vector storage in single collection
- RRF or DBSF fusion for result combination
- Native batch operations via Qdrant client
- Implements VectorIndex protocol for consistency
"""

import logging
from typing import Any, Literal

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    PointStruct,
    Prefetch,
    ScoredPoint,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from langres.core.embeddings import EmbeddingProvider, SparseEmbeddingProvider

logger = logging.getLogger(__name__)


class QdrantHybridIndex:
    """Qdrant-backed hybrid index with dense + sparse vectors.

    Combines semantic (dense) and keyword (sparse) search using Qdrant's
    fusion capabilities. Implements VectorIndex protocol for compatibility
    with existing blockers.

    The index owns both embedders and manages the complete lifecycle:
    1. create_index(texts) - Preprocessing: embed and upload to Qdrant
    2. search(query_texts, k) - Runtime: hybrid search with fusion
    3. search_all(k) - Runtime: deduplication pattern (all-vs-all)

    Example:
        from qdrant_client import QdrantClient
        from langres.core.embeddings import SentenceTransformerEmbedder, FastEmbedSparseEmbedder

        client = QdrantClient(url="http://localhost:6333")
        dense_embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        sparse_embedder = FastEmbedSparseEmbedder("Qdrant/bm25")

        index = QdrantHybridIndex(
            client=client,
            collection_name="companies",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
        )

        # Preprocessing
        corpus = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        index.create_index(corpus)

        # Runtime: Single query
        distances, indices = index.search("Apple Company", k=2)
        # Returns: distances=(2,), indices=(2,)

        # Runtime: Batch queries
        distances, indices = index.search(["Apple", "Google"], k=2)
        # Returns: distances=(2,2), indices=(2,2)

        # Runtime: Deduplication
        distances, indices = index.search_all(k=10)
        # Returns: distances=(3,10), indices=(3,10)
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_embedder: EmbeddingProvider,
        sparse_embedder: SparseEmbeddingProvider,
        fusion: Literal["RRF", "DBSF"] = "RRF",
        prefetch_limit: int = 20,
    ):
        """Initialize QdrantHybridIndex.

        Args:
            client: Qdrant client instance (injected for testing).
            collection_name: Name of the Qdrant collection to create/use.
            dense_embedder: Provider for dense vector embeddings.
            sparse_embedder: Provider for sparse vector embeddings.
            fusion: Fusion method for combining results ("RRF" or "DBSF").
                Default: "RRF" (Reciprocal Rank Fusion).
            prefetch_limit: Number of results to fetch per vector type before fusion.
                Default: 20 (20 from dense + 20 from sparse → fused to top-k).

        Note:
            The Qdrant client must be configured externally (URL, API key, etc.).
            This allows flexibility in deployment (local, cloud, custom config).
        """
        self.client = client
        self.collection_name = collection_name
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.fusion = fusion
        self.prefetch_limit = prefetch_limit

        # State (populated by create_index)
        self._corpus_texts: list[str] | None = None
        self._n_samples: int | None = None
        self._cached_dense_embeddings: np.ndarray | None = None

    def create_index(self, texts: list[str]) -> None:
        """Preprocessing: Build hybrid index from text corpus.

        Creates Qdrant collection with named vectors, embeds texts with both
        dense and sparse embedders, and batch uploads points.

        Args:
            texts: Corpus texts to embed and index.

        Note:
            Calling create_index() multiple times recreates the collection.
            This is idempotent but destroys previous data.
        """
        # 1. Create collection with named vectors (dense + sparse)
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.dense_embedder.embedding_dim,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )

        logger.info(
            "Created Qdrant collection '%s' with dense (dim=%d) + sparse vectors",
            self.collection_name,
            self.dense_embedder.embedding_dim,
        )

        # 2. Batch encode texts with both embedders
        dense_embeddings = self.dense_embedder.encode(texts)
        sparse_embeddings = self.sparse_embedder.encode(texts)

        # Cache dense embeddings for search_all() optimization
        self._cached_dense_embeddings = dense_embeddings

        # 3. Build PointStruct list with both vectors
        points = []
        for i, text in enumerate(texts):
            point = PointStruct(
                id=i,
                vector={
                    "dense": dense_embeddings[i].tolist(),  # Convert numpy to list
                    "sparse": SparseVector(
                        indices=sparse_embeddings[i]["indices"],
                        values=sparse_embeddings[i]["values"],
                    ),
                },
                payload={"text": text, "id": str(i)},
            )
            points.append(point)

        # 4. Batch upsert points (chunk to avoid payload size limits)
        # Qdrant cloud has 32MB payload limit - batch in chunks of 100 points
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            logger.debug(
                "Upserted batch %d/%d",
                i // batch_size + 1,
                (len(points) + batch_size - 1) // batch_size,
            )

        logger.info("Upserted %d points to collection '%s'", len(points), self.collection_name)

        # 5. Cache corpus for search_all()
        self._corpus_texts = texts
        self._n_samples = len(texts)

    def search(
        self,
        query_texts: str | list[str],
        k: int,
        query_prompt: str | None = None,
        _dense_embeddings: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Runtime: Hybrid search for k nearest neighbors.

        Supports both single query and batch queries. Uses Qdrant's prefetch
        + fusion for combining dense and sparse results.

        Args:
            query_texts: Single text query or list of text queries.
            k: Number of nearest neighbors to return per query.
            query_prompt: Optional instruction prompt for query encoding (asymmetric search).
                Applied only to dense embedder. Sparse embedder never uses prompts.
                Default: None.
            _dense_embeddings: INTERNAL - Pre-computed dense embeddings (used by search_all).
                When provided, dense embedder is NOT called. Sparse embedder still processes texts.

        Returns:
            Tuple of (distances, indices):
            - If single query: distances=(k,), indices=(k,)
            - If batch: distances=(N,k), indices=(N,k)

        Raises:
            RuntimeError: If search() is called before create_index().

        Note:
            For batch queries, this makes one query_points() call per query.
            Qdrant doesn't have explicit batch API, but client handles efficiently.

            Hybrid search REQUIRES text because sparse vectors need text encoding.
            Unlike FAISSIndex (dense-only), this cannot accept np.ndarray for query_texts.
            The _dense_embeddings parameter is INTERNAL for search_all() optimization only.
        """
        if self._corpus_texts is None:
            raise RuntimeError("Index not built. Must call create_index() before search().")

        # Handle single vs batch
        is_single = isinstance(query_texts, str)
        if is_single:
            texts: list[str] = [query_texts]  # type: ignore[list-item]
        else:
            texts = query_texts  # type: ignore[assignment]

        # Encode queries with both embedders
        # Dense: use cached if provided (search_all optimization), otherwise encode with prompt
        if _dense_embeddings is not None:
            dense_query_embeddings = _dense_embeddings
        else:
            # Dense: apply query_prompt for asymmetric encoding
            dense_query_embeddings = self.dense_embedder.encode(texts, prompt=query_prompt)

        # Sparse: ALWAYS encode from text (no caching possible for Qdrant hybrid)
        # Sparse never uses prompts (BM25 doesn't support instructions)
        sparse_query_embeddings = self.sparse_embedder.encode(texts, prompt=None)

        # Batch search (one query_points call per query)
        all_distances = []
        all_indices = []

        for i in range(len(texts)):
            # Build prefetch for hybrid search
            prefetch = [
                # Dense vector search
                Prefetch(
                    query=dense_query_embeddings[i].tolist(),
                    using="dense",
                    limit=self.prefetch_limit,
                ),
                # Sparse vector search
                Prefetch(
                    query=SparseVector(
                        indices=sparse_query_embeddings[i]["indices"],
                        values=sparse_query_embeddings[i]["values"],
                    ),
                    using="sparse",
                    limit=self.prefetch_limit,
                ),
            ]

            # Execute hybrid query with fusion
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch,
                query=FusionQuery(fusion=Fusion.RRF if self.fusion == "RRF" else Fusion.DBSF),
                limit=k,
            )

            # Extract distances and indices from query results
            # query_points returns QueryResponse with .points attribute
            points = results.points if hasattr(results, "points") else results

            # Pad results to ensure consistent shape (some queries may return fewer than k points)
            distances = np.full(k, np.nan, dtype=np.float32)
            indices = np.full(k, -1, dtype=np.int64)

            for j, point in enumerate(points):
                distances[j] = point.score  # type: ignore[attr-defined]
                indices[j] = point.id  # type: ignore[attr-defined]

            all_distances.append(distances)
            all_indices.append(indices)

        # Convert to numpy arrays
        distances_array = np.array(all_distances, dtype=np.float32)
        indices_array = np.array(all_indices, dtype=np.int64)

        # Return shape depends on input
        if is_single:
            return distances_array[0], indices_array[0]
        else:
            return distances_array, indices_array

    def search_all(self, k: int, query_prompt: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Runtime: Search all corpus items against each other (deduplication).

        Uses cached corpus texts and dense embeddings for efficient deduplication.

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
            Performance optimization: Reuses cached dense embeddings from create_index(),
            avoiding re-encoding the corpus. Sparse embeddings still need re-encoding
            (Qdrant limitation - query_points API requires fresh sparse vectors).
        """
        if self._corpus_texts is None:
            raise RuntimeError("Index not built. Must call create_index() before search_all().")

        # Reuse search() with cached dense embeddings (performance optimization)
        # query_prompt is passed through but not used (we use cached embeddings)
        return self.search(
            self._corpus_texts,
            k,
            query_prompt=query_prompt,
            _dense_embeddings=self._cached_dense_embeddings,
        )


class FakeHybridVectorIndex:
    """Test double for hybrid vector index.

    Produces deterministic fake results without Qdrant client or embedders.
    Perfect for fast unit testing of blocker logic.

    Example:
        index = FakeHybridVectorIndex()
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
        """Initialize FakeHybridVectorIndex."""
        self._n_samples: int | None = None
        self._texts: list[str] | None = None

    def create_index(self, texts: list[str]) -> None:
        """Record corpus size for generating valid indices.

        Args:
            texts: Corpus texts (only length is used).
        """
        self._n_samples = len(texts)
        self._texts = texts
        logger.debug("FakeHybridVectorIndex: recorded %d samples", self._n_samples)

    def search(
        self, query_texts: str | list[str], k: int, query_prompt: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate fake search results (deterministic).

        Args:
            query_texts: Single text or list of texts.
            k: Number of neighbors per query.
            query_prompt: Optional instruction prompt (ignored by fake implementation).

        Returns:
            - If single query: distances=(k,), indices=(k,)
            - If batch: distances=(N,k), indices=(N,k)
        """
        if self._n_samples is None:
            raise RuntimeError("Index not built. Call create_index() first.")

        # Handle single vs batch
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
        indices = np.zeros((self._n_samples, k), dtype=np.int64)
        distances = np.zeros((self._n_samples, k), dtype=np.float32)

        for i in range(self._n_samples):
            for j in range(k):
                indices[i, j] = (i + j) % self._n_samples
                distances[i, j] = j * 0.1

        return distances, indices
