"""Hybrid reranking vector index implementation using Qdrant.

This module provides hybrid search with 3-stage pipeline:
1. Dense prefetch (semantic similarity)
2. Sparse prefetch (keyword matching)
3. Fusion (RRF or DBSF for combining dense + sparse)
4. Late-interaction reranking (ColBERT/ColPali MaxSim)

The late-interaction reranking enables more nuanced similarity computation
by encoding each token separately and computing maximum similarity between
all token pairs (MaxSim strategy).
"""

import logging
from typing import Literal

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from langres.core.embeddings import (
    EmbeddingProvider,
    LateInteractionEmbeddingProvider,
    SparseEmbeddingProvider,
)

logger = logging.getLogger(__name__)


class QdrantHybridRerankingIndex:
    """Qdrant-backed hybrid index with dense + sparse + late-interaction reranking.

    Four-stage hybrid search pipeline:
    1. Prefetch: Dense semantic search (top-N)
    2. Prefetch: Sparse keyword search (top-N)
    3. Fusion: Combine dense + sparse results (RRF or DBSF)
    4. Rerank: Late-interaction (ColBERT/ColPali) on fused results

    Implements VectorIndex protocol for compatibility with existing blockers.

    The index owns all three embedders and manages the complete lifecycle:
    1. create_index(texts) - Preprocessing: embed and upload to Qdrant
    2. search(query_texts, k) - Runtime: hybrid search with fusion + reranking
    3. search_all(k) - Runtime: deduplication pattern (all-vs-all)

    Example:
        from qdrant_client import QdrantClient
        from langres.core.embeddings import (
            SentenceTransformerEmbedder,
            FastEmbedSparseEmbedder,
            FastEmbedLateInteractionEmbedder,
        )

        client = QdrantClient(url="http://localhost:6333")
        dense = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        sparse = FastEmbedSparseEmbedder(model_name="Qdrant/bm25")
        reranker = FastEmbedLateInteractionEmbedder(model_name="colbert-ir/colbertv2.0")

        index = QdrantHybridRerankingIndex(
            client=client,
            collection_name="companies",
            dense_embedder=dense,
            sparse_embedder=sparse,
            reranking_embedder=reranker,
            fusion="RRF",  # Reciprocal Rank Fusion (default)
            prefetch_limit=20,
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
        reranking_embedder: LateInteractionEmbeddingProvider,
        fusion: Literal["RRF", "DBSF"] = "RRF",
        prefetch_limit: int = 20,
    ):
        """Initialize QdrantHybridRerankingIndex.

        Args:
            client: Qdrant client instance (injected for testing).
            collection_name: Name of the Qdrant collection to create/use.
            dense_embedder: Provider for dense vector embeddings.
            sparse_embedder: Provider for sparse vector embeddings.
            reranking_embedder: Provider for late-interaction multi-vectors.
            fusion: Fusion method for combining dense + sparse results.
                "RRF" (Reciprocal Rank Fusion) or "DBSF" (Distribution-Based Score Fusion).
                Default: "RRF".
            prefetch_limit: Number of results to fetch per vector type before fusion.
                Default: 20 (20 from dense + 20 from sparse → fused → reranked to top-k).

        Note:
            The Qdrant client must be configured externally (URL, API key, etc.).
            This allows flexibility in deployment (local, cloud, custom config).
        """
        self.client = client
        self.collection_name = collection_name
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.reranking_embedder = reranking_embedder
        self.fusion = fusion
        self.prefetch_limit = prefetch_limit

        # State (populated by create_index)
        self._corpus_texts: list[str] | None = None
        self._n_samples: int | None = None
        self._cached_dense_embeddings: np.ndarray | None = None

        # TODO: Memory optimization (post-POC)
        # Same as QdrantHybridIndex - caching dense embeddings for search_all().
        # Acceptable for POC, may need optimization for production scale.

    def create_index(self, texts: list[str]) -> None:
        """Preprocessing: Build hybrid index from text corpus.

        Creates Qdrant collection with three named vectors (dense, sparse, reranking),
        embeds texts with all three embedders, and batch uploads points.

        Args:
            texts: Corpus texts to embed and index.

        Note:
            Calling create_index() multiple times recreates the collection.
            This is idempotent but destroys previous data.
        """
        # 1. Create collection with named vectors (dense + sparse + reranking)
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.dense_embedder.embedding_dim,
                    distance=Distance.COSINE,
                ),
                "reranking": VectorParams(
                    size=self.reranking_embedder.embedding_dim,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )

        logger.info(
            "Created Qdrant collection '%s' with dense (dim=%d) + sparse + reranking (dim=%d) vectors",
            self.collection_name,
            self.dense_embedder.embedding_dim,
            self.reranking_embedder.embedding_dim,
        )

        # 2. Batch encode texts with all three embedders (documents use prompt=None)
        dense_embeddings = self.dense_embedder.encode(texts, prompt=None)
        sparse_embeddings = self.sparse_embedder.encode(texts, prompt=None)
        reranking_embeddings = self.reranking_embedder.encode(texts, prompt=None)

        # Cache dense embeddings for search_all() optimization
        self._cached_dense_embeddings = dense_embeddings

        # 3. Build PointStruct list with all three vectors
        points: list[PointStruct] = []
        for i, text in enumerate(texts):
            point = PointStruct(
                id=i,
                vector={
                    "dense": dense_embeddings[i].tolist(),  # Convert numpy to list
                    "sparse": SparseVector(
                        indices=sparse_embeddings[i]["indices"],
                        values=sparse_embeddings[i]["values"],
                    ),
                    "reranking": reranking_embeddings[i],  # Already list[list[float]]
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
        """Runtime: Hybrid search with reranking for k nearest neighbors.

        Supports both single query and batch queries. Uses Qdrant's prefetch
        (dense + sparse) followed by late-interaction reranking.

        Args:
            query_texts: Single text query or list of text queries.
            k: Number of nearest neighbors to return per query.
            query_prompt: Optional instruction prompt for query encoding (asymmetric search).
                If provided, dense and reranking embedders will use this prompt for queries.
                Sparse embedder never uses prompts (keyword matching).
                Documents are always encoded with prompt=None.
                Default: None.
            _dense_embeddings: INTERNAL - Pre-computed dense embeddings (used by search_all).
                When provided, dense embedder is NOT called. Sparse and reranking embedders
                still process texts.

        Returns:
            Tuple of (distances, indices):
            - If single query: distances=(k,), indices=(k,)
            - If batch: distances=(N,k), indices=(N,k)

        Raises:
            RuntimeError: If search() is called before create_index().

        Note:
            For batch queries, this makes one query_points() call per query.
            Qdrant doesn't have explicit batch API, but client handles efficiently.

            Hybrid reranking REQUIRES text because sparse vectors and late-interaction
            multi-vectors need text encoding. The _dense_embeddings parameter is
            INTERNAL for search_all() optimization only.
        """
        if self._corpus_texts is None:
            raise RuntimeError("Index not built. Must call create_index() before search().")

        # Handle single vs batch
        is_single = isinstance(query_texts, str)
        if is_single:
            texts: list[str] = [query_texts]  # type: ignore[list-item]
        else:
            texts = query_texts  # type: ignore[assignment]

        # Encode queries with all three embedders
        # Dense: use cached if provided (search_all optimization), otherwise encode
        if _dense_embeddings is not None:
            dense_query_embeddings = _dense_embeddings
        else:
            # Dense and reranking use query_prompt (if provided), sparse always uses None
            dense_query_embeddings = self.dense_embedder.encode(texts, prompt=query_prompt)

        # Sparse and reranking: ALWAYS encode from text (no caching possible for Qdrant hybrid)
        sparse_query_embeddings = self.sparse_embedder.encode(texts, prompt=None)
        reranking_query_embeddings = self.reranking_embedder.encode(texts, prompt=query_prompt)

        # Batch search (one query_points call per query)
        all_distances: list[np.ndarray] = []
        all_indices: list[np.ndarray] = []

        for i in range(len(texts)):
            # Build nested prefetch: [dense, sparse] → fusion → reranking
            # Stage 1 & 2: Dense and sparse prefetches (inner level)
            inner_prefetches = [
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

            # Stage 3: Fusion (outer Prefetch level)
            prefetch = Prefetch(
                prefetch=inner_prefetches,
                query=FusionQuery(fusion=Fusion.RRF if self.fusion == "RRF" else Fusion.DBSF),
                limit=self.prefetch_limit,
            )

            # Stage 4: Late-interaction reranking (top query_points level)
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch,
                query=reranking_query_embeddings[i],  # Multi-vector for MaxSim reranking
                using="reranking",  # Specify which named vector to use for multi-vector query
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
            query_prompt: Optional instruction prompt for query encoding (asymmetric search).
                If provided, reranking embedder will use this prompt for queries.
                Dense embeddings are cached, so this only affects reranking.
                Default: None.

        Returns:
            Tuple of (distances, indices), both shape (N, k) where N = corpus size.

        Raises:
            RuntimeError: If search_all() is called before create_index().

        Note:
            Performance optimization: Reuses cached dense embeddings from create_index(),
            avoiding re-encoding the corpus. Sparse and reranking embeddings still need
            re-encoding (Qdrant limitation - query_points API requires fresh vectors).
        """
        if self._corpus_texts is None:
            raise RuntimeError("Index not built. Must call create_index() before search_all().")

        # Reuse search() with cached dense embeddings (performance optimization)
        return self.search(
            self._corpus_texts,
            k,
            query_prompt=query_prompt,
            _dense_embeddings=self._cached_dense_embeddings,
        )


class FakeHybridRerankingVectorIndex:
    """Test double for hybrid reranking vector index.

    Produces deterministic fake results without Qdrant client or embedders.
    Perfect for fast unit testing of blocker logic.

    Example:
        index = FakeHybridRerankingVectorIndex()
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
        """Initialize FakeHybridRerankingVectorIndex."""
        self._n_samples: int | None = None
        self._texts: list[str] | None = None

    def create_index(self, texts: list[str]) -> None:
        """Record corpus size for generating valid indices.

        Args:
            texts: Corpus texts (only length is used).
        """
        self._n_samples = len(texts)
        self._texts = texts
        logger.debug("FakeHybridRerankingVectorIndex: recorded %d samples", self._n_samples)

    def search(self, query_texts: str | list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate fake search results (deterministic).

        Args:
            query_texts: Single text or list of texts.
            k: Number of neighbors per query.

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

    def search_all(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate fake deduplication results (deterministic).

        Args:
            k: Number of neighbors per corpus item.

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
