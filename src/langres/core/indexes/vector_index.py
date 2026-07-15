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

import json
import logging
import os
from pathlib import Path
from typing import ClassVar, Literal, Protocol, cast

# faiss and torch each vendor their own libomp; on macOS loading both in one
# process can abort with a duplicate-OpenMP-runtime error (exit 139). Opting
# into the documented workaround before faiss loads keeps the dev/test flow
# (parallel pytest) stable. No effect once an OpenMP runtime is already loaded.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
from pydantic import BaseModel

from langres.core.embeddings import EmbeddingProvider
from langres.core.registry import get_component, register
from langres.core.serialization import ComponentSpec

logger = logging.getLogger(__name__)


def inverse_distances_to_similarities(distances: np.ndarray) -> np.ndarray:
    """Map non-negative distances (lower = closer) to similarities in ``(0, 1]``.

    Uses ``1 / (1 + d)``, monotonically decreasing in the distance: a distance
    of 0 maps to 1.0 (most similar) and decays toward 0.0 as the distance grows.
    ``NaN`` distances map to 0.0 (least similar).

    Shared by the in-memory fake indexes, whose synthetic distances are
    rank-ordered (lower = closer), so their ``to_similarities`` stays meaningful.

    Args:
        distances: Distance matrix (any shape), values ``>= 0`` (negatives are
            clamped to 0 before conversion).

    Returns:
        Similarity array of the same shape, values in ``[0, 1]``.
    """
    d = np.asarray(distances, dtype=np.float64)
    sim = 1.0 / (1.0 + np.maximum(d, 0.0))
    return np.nan_to_num(sim, nan=0.0)


def clip_scores_to_similarities(scores: np.ndarray) -> np.ndarray:
    """Clip "higher = more similar" scores into ``[0, 1]`` — bounded, but lossy.

    Use when an index returns scores that are monotonic in true similarity
    (higher = better) but **not** already on a ``[0, 1]`` scale — e.g. Qdrant
    fusion (RRF/DBSF) or late-interaction (ColBERT/ColPali) MaxSim scores.
    Despite their name these are NOT probabilities and do not sit in ``[0, 1]``:
    RRF/DBSF fusion scores are typically tiny (~0.01–0.03) and MaxSim scores
    routinely exceed 1.0. Clipping to ``[0, 1]`` is therefore a **lossy** mapping
    — most fusion scores collapse toward 0.0 and MaxSim scores saturate at 1.0 —
    so the resulting per-pair ``similarity_score`` is degenerate and should be
    treated as **observability only**, never as a calibrated similarity. It does
    not affect candidate membership: the blocker's candidate set comes from the
    index's neighbour ranking, which this clip preserves (it is monotonic). For a
    real score, run a downstream scorer (e.g. a Comparator + judge). ``NaN``
    scores (such as the padding emitted when a query returns fewer than ``k``
    results) map to 0.0.

    Args:
        scores: Score matrix (any shape), higher = more similar.

    Returns:
        Similarity array of the same shape, values clipped into ``[0, 1]``.
    """
    s = np.asarray(scores, dtype=np.float64)
    return np.nan_to_num(np.clip(s, 0.0, 1.0), nan=0.0)


class SerializableEmbedder(EmbeddingProvider, Protocol):
    """An :class:`EmbeddingProvider` that can round-trip through a ``ComponentSpec``.

    The FAISS index needs its embedder to declare a registry ``type_name`` and
    expose ``config`` / ``from_config`` so the embedder can be persisted in the
    index config and rebuilt via the component registry. Plain
    ``EmbeddingProvider`` does not require this — only embedders embedded in a
    serializable index do.
    """

    type_name: str
    config_model: type[BaseModel]

    def config(self) -> BaseModel:
        """Return the light, JSON-serializable construction config."""
        ...  # pragma: no cover

    @classmethod
    def from_config(cls, config: BaseModel) -> "SerializableEmbedder":
        """Reconstruct the embedder from its config."""
        ...  # pragma: no cover


class FAISSIndexConfig(BaseModel):
    """Light, JSON-serializable construction config for a FAISSIndex.

    Holds only what is needed to rebuild the index *shell*: the distance metric
    and a nested :class:`ComponentSpec` describing the embedder. The heavy
    corpus/index state (vectors, FAISS index, texts) is NOT stored here — it
    round-trips through :meth:`FAISSIndex.save_state` / ``load_state``.
    """

    metric: Literal["L2", "cosine"] = "L2"
    embedder: ComponentSpec


def _embedder_from_spec(spec: ComponentSpec) -> EmbeddingProvider:
    """Rebuild an embedder from its ``ComponentSpec`` via the component registry.

    Looks up the registered class by ``type_name`` and reconstructs it from the
    serialized config using the class's ``config_model`` and ``from_config``.
    """
    cls: type[SerializableEmbedder] = get_component(spec.type_name)
    config = cls.config_model.model_validate(spec.config)
    return cls.from_config(config)


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

    def to_similarities(self, distances: np.ndarray) -> np.ndarray:
        """Convert this index's raw distance/score matrix to similarities in [0, 1].

        Each concrete index knows its own metric and converts so that 1.0 = most
        similar and the result is monotonic in true similarity. ``NaN`` entries
        map to 0.0 (least similar). This keeps the distance→similarity conversion
        with the index that produced the distances, rather than having callers
        guess the metric.

        Args:
            distances: The distance/score matrix returned by :meth:`search_all`
                (or :meth:`search`), typically shape ``(N, k)``.

        Returns:
            Similarity array of the same shape, values in ``[0, 1]`` where 1.0 =
            most similar.
        """
        ...  # pragma: no cover


@register("faiss_index")
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

    type_name: ClassVar[str] = "faiss_index"
    config_model: ClassVar[type[FAISSIndexConfig]] = FAISSIndexConfig

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

        # TODO: Memory optimization (post-POC)
        # Current implementation stores embeddings twice (2× memory):
        # - _corpus_embeddings for search_all() optimization
        # - FAISS internal storage for index search
        # For large datasets (>1M vectors), this causes OOM.
        # Possible solutions:
        # 1. Remove _corpus_embeddings, rely on DiskCachedEmbedder (slower but saves RAM)
        # 2. Add automatic threshold: RAM for small datasets, disk for large
        # 3. Use FAISS quantization (IVF-PQ) for 4-32x compression
        # 4. Recommend QdrantHybridIndex for production (server-side memory management)

    def config(self) -> FAISSIndexConfig:
        """Return the light, JSON-serializable construction config.

        Captures the metric and a nested :class:`ComponentSpec` for the embedder.
        The heavy corpus/index state is excluded — it round-trips through
        :meth:`save_state` / :meth:`load_state`.
        """
        embedder = cast(SerializableEmbedder, self.embedder)
        embedder_spec = ComponentSpec(
            type_name=embedder.type_name,
            config=embedder.config().model_dump(),
        )
        return FAISSIndexConfig(metric=self.metric, embedder=embedder_spec)

    @classmethod
    def from_config(cls, config: FAISSIndexConfig) -> "FAISSIndex":
        """Rebuild the index *shell* from config, reconstructing the embedder.

        The returned index has no corpus/index state — call :meth:`load_state`
        (or :meth:`create_index`) to populate it.
        """
        embedder = _embedder_from_spec(config.embedder)
        return cls(embedder=embedder, metric=config.metric)

    def save_state(self, state_dir: Path) -> None:
        """Persist the built FAISS index, corpus vectors, and texts to ``state_dir``.

        Writes three sidecar files:
        - ``index.faiss`` — the FAISS index (via ``faiss.write_index``).
        - ``corpus_embeddings.npy`` — cached corpus vectors (via ``np.save``).
        - ``corpus.json`` — corpus texts and metric.

        If the index has not been built (``create_index`` never called), there
        is no out-of-band state to persist and this is a no-op.
        """
        if self._index is None or self._corpus_embeddings is None:
            logger.debug("FAISSIndex.save_state: index not built, nothing to persist")
            return

        faiss.write_index(self._index, str(state_dir / "index.faiss"))
        np.save(state_dir / "corpus_embeddings.npy", self._corpus_embeddings)
        (state_dir / "corpus.json").write_text(
            json.dumps({"corpus_texts": self._corpus_texts, "metric": self.metric})
        )
        logger.info("Persisted FAISS index state to %s", state_dir)

    def load_state(self, state_dir: Path) -> None:
        """Restore index, corpus vectors, and texts previously written to ``state_dir``.

        Reuses the stored vectors verbatim — the embedder is never invoked, so a
        loaded index never re-embeds its corpus.
        """
        self._index = faiss.read_index(str(state_dir / "index.faiss"))
        self._corpus_embeddings = np.load(state_dir / "corpus_embeddings.npy")

        payload = json.loads((state_dir / "corpus.json").read_text())
        self._corpus_texts = payload["corpus_texts"]
        self.metric = payload["metric"]
        logger.info("Loaded FAISS index state from %s", state_dir)

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

    def to_similarities(self, distances: np.ndarray) -> np.ndarray:
        """Convert FAISS distances to similarities in [0, 1] using the index metric.

        - ``metric="L2"`` (``IndexFlatL2``): ``distances`` are *squared* L2
          distances (``d >= 0``, lower = closer, routinely ``> 2.0``). Converted
          with ``1 / (1 + sqrt(d))``, monotonically decreasing in distance and in
          ``(0, 1]``.
        - ``metric="cosine"`` (``IndexFlatIP`` over normalized vectors):
          ``distances`` are inner products in ``[-1, 1]`` (higher = more similar).
          Mapped to ``[0, 1]`` with ``(ip + 1) / 2``.

        ``NaN`` entries map to 0.0 (least similar).

        Args:
            distances: Distance/score matrix from :meth:`search_all` / :meth:`search`.

        Returns:
            Similarity array of the same shape, values in ``[0, 1]``.
        """
        d = np.asarray(distances, dtype=np.float64)
        if self.metric == "L2":
            sim = 1.0 / (1.0 + np.sqrt(np.maximum(d, 0.0)))
        else:  # cosine: inner products of normalized vectors, in [-1, 1]
            sim = np.clip((d + 1.0) / 2.0, 0.0, 1.0)
        result: np.ndarray = np.nan_to_num(sim, nan=0.0)
        return result

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

    def to_similarities(self, distances: np.ndarray) -> np.ndarray:
        """Convert the fake's synthetic distances (lower = closer) to ``[0, 1]``.

        Uses :func:`inverse_distances_to_similarities` (``1 / (1 + d)``), matching
        the rank-ordered ``j * 0.1`` distances this double emits so its similarity
        ordering stays meaningful (nearest neighbor scores highest). ``NaN`` maps
        to 0.0.
        """
        return inverse_distances_to_similarities(distances)

    # ============ OLD API (for backward compatibility) ============
    def build(self, embeddings: np.ndarray) -> None:
        """DEPRECATED: Use create_index() instead."""
        self._n_samples = embeddings.shape[0]
        logger.debug("FakeVectorIndex: recorded %d samples", self._n_samples)
