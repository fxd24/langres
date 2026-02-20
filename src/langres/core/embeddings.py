"""Embedding providers for text-to-vector encoding.

This module provides abstractions for converting text into embeddings,
supporting both dense (semantic) and sparse (keyword) vectors for hybrid search.

Key providers:
- EmbeddingProvider: Dense embeddings (np.ndarray) for semantic similarity
- SparseEmbeddingProvider: Sparse embeddings (BM25, SPLADE) for keyword matching
- LateInteractionEmbeddingProvider: Multi-vector embeddings (ColBERT, ColPali)

Implementations:
- SentenceTransformerEmbedder: Production dense embeddings (sentence-transformers)
- DiskCachedEmbedder: Persistent caching wrapper with two-tier cache (hot memory + cold disk)
- FastEmbedSparseEmbedder: Production sparse embeddings (BM25, SPLADE)
- FastEmbedLateInteractionEmbedder: Production late-interaction embeddings (ColBERT)
- FakeEmbedder: Test double for fast deterministic testing

Caching:
The DiskCachedEmbedder wrapper provides persistent, disk-backed caching for any
EmbeddingProvider. It uses a two-tier architecture:
- Hot cache (in-memory LRU): Fast access to recently used embeddings
- Cold storage (SQLite): Unlimited disk-backed storage for all embeddings

This prevents OOM on large datasets while avoiding re-computation across runs.

Example (with caching):
    from pathlib import Path

    base_embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    cached_embedder = DiskCachedEmbedder(
        embedder=base_embedder,
        cache_dir=Path("./cache"),
        namespace="my_model",
        memory_cache_size=10_000  # Keep 10K hot in memory
    )

    # First run: computes and caches
    embeddings = cached_embedder.encode(texts)

    # Second run: loads from cache (fast!)
    embeddings = cached_embedder.encode(texts)
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for encoding text into vector embeddings.

    This abstraction separates the expensive embedding computation from
    vector search, enabling several key workflows:

    1. **Swapping models during optimization**:
       Try different embedding models (all-MiniLM, mpnet, OpenAI) to
       find the best recall/quality trade-off.

    2. **Caching embeddings for inference**:
       Compute embeddings once during training, save to disk/DB, then
       reuse during inference without re-encoding.

    3. **Using pre-computed embeddings**:
       Load embeddings from external sources (databases, vector stores)
       without coupling to a specific encoding implementation.

    Example (training phase - trying different models):
        for model in ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]:
            embedder = SentenceTransformerEmbedder(model)
            embeddings = embedder.encode(texts)
            recall = evaluate_blocking_recall(embeddings)
            print(f"{model}: recall={recall}")

    Example (inference phase - using cached embeddings):
        # Training: compute once and save
        embedder = SentenceTransformerEmbedder("best-model")
        embeddings = embedder.encode(production_texts)
        np.save("embeddings.npy", embeddings)

        # Inference: load and reuse
        embeddings = np.load("embeddings.npy")
        # Use with VectorIndex directly, no re-encoding needed
    """

    def encode(self, texts: list[str] | np.ndarray, prompt: str | None = None) -> np.ndarray:
        """Encode texts into vector embeddings.

        Args:
            texts: List of texts to encode OR pre-computed embeddings (np.ndarray).
                For text input: List of strings to encode. Can be empty.
                For pre-computed: np.ndarray of shape (N, embedding_dim).
                    Pre-computed embeddings bypass encoding and are passed through.
            prompt: Optional instruction prompt for asymmetric encoding.
                Documents (corpus) should use prompt=None for generic embeddings.
                Queries should use task-specific prompts for better retrieval.
                Ignored if texts is pre-computed embeddings.
                Default: None.

        Returns:
            Numpy array of shape (len(texts), embedding_dim).
            Returns shape (0, embedding_dim) for empty input.

        Note:
            Implementations should return consistent dtypes (typically float32)
            for compatibility with vector index backends like FAISS.

        Note:
            Pre-computed embeddings enable workflows where embeddings are cached
            externally (databases, vector stores) and reused without re-encoding.

        Note:
            Instruction prompts enable asymmetric encoding where queries are
            encoded with task context (e.g., "Find duplicate organization names")
            while documents remain generic. Models like Qwen3-Embedding, BGE,
            and E5 show 1-5% improvement with query instructions.
        """
        ...  # pragma: no cover

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors.

        Returns:
            The number of dimensions in each embedding vector.

        Note:
            This property allows VectorIndex implementations to validate
            embedding dimensions and configure index parameters correctly.
        """
        ...  # pragma: no cover


class SparseEmbeddingProvider(Protocol):
    """Protocol for sparse vector generation (BM25, SPLADE, etc.).

    Sparse embeddings complement dense embeddings in hybrid search by providing
    keyword-based matching. They return vectors in Qdrant-compatible format
    with vocabulary indices and corresponding weights.

    Example:
        sparse_embedder = FastEmbedSparseEmbedder("Qdrant/bm25")
        texts = ["Apple Inc.", "Microsoft Corp."]
        sparse_vectors = sparse_embedder.encode(texts)

        # Result format (Qdrant compatible):
        # [
        #     {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.3]},
        #     {"indices": [2, 5, 15], "values": [0.9, 0.5, 0.4]}
        # ]
    """

    def encode(self, texts: list[str], prompt: str | None = None) -> list[dict[str, Any]]:
        """Generate sparse vectors from texts.

        Args:
            texts: List of texts to encode.
            prompt: Optional instruction prompt (ignored for sparse embeddings).
                Sparse embedders (BM25, SPLADE) don't use prompts for keyword matching.
                This parameter exists for protocol compatibility.

        Returns:
            List of sparse vectors in Qdrant format.
            Each sparse vector is a dict with:
            - "indices": List of vocabulary indices (int)
            - "values": List of corresponding weights (float)

        Example:
            sparse_vectors = embedder.encode(["Hello world", "Python code"])
            # Returns:
            # [
            #     {"indices": [10, 25, 100], "values": [0.8, 0.6, 0.3]},
            #     {"indices": [15, 50, 120], "values": [0.9, 0.5, 0.4]}
            # ]

        Note:
            The prompt parameter is intentionally ignored, as sparse embedders
            perform keyword matching and don't benefit from instruction prompts.
        """
        ...  # pragma: no cover


class LateInteractionEmbeddingProvider(Protocol):
    """Protocol for late-interaction embeddings (ColBERT, ColPali, etc.).

    Generates multi-vectors (one per token) for contextualized matching.
    Compatible with Qdrant's MultiVectorConfig + MaxSim comparator.

    Late-interaction models encode each token separately, enabling more nuanced
    similarity computation compared to single-vector dense embeddings. The final
    similarity score is computed by taking the maximum similarity between all
    token pairs (MaxSim strategy).

    Example:
        embedder = FastEmbedLateInteractionEmbedder(model_name="colbert-ir/colbertv2.0")
        texts = ["Apple Inc.", "Microsoft Corp."]
        multi_vectors = embedder.encode(texts)
        # Returns: List of [num_tokens, embedding_dim] arrays, one per text

    Note:
        ColBERT models typically produce 128-dimensional token embeddings.
        The number of tokens varies per text based on tokenization.
    """

    def encode(self, texts: list[str], prompt: str | None = None) -> list[list[list[float]]]:
        """Encode texts into multi-vectors.

        Args:
            texts: List of texts to encode. Can be empty.
            prompt: Optional instruction prompt for asymmetric encoding.
                If the underlying model supports instruction prompts, this will
                be used. Otherwise, it's silently ignored (model-dependent).
                Default: None.

        Returns:
            List of multi-vectors, one per text.
            Shape: (num_texts,) with variable-length token dimension.
            Each text becomes [num_tokens, embedding_dim] array.
            Returns empty list for empty input.

        Example:
            multi_vectors = embedder.encode(["Hello", "World"])
            # Returns:
            # [
            #   [[0.1, 0.2, ...], [0.3, 0.4, ...]],  # "Hello": 2 tokens x 128 dim
            #   [[0.5, 0.6, ...], [0.7, 0.8, ...]]   # "World": 2 tokens x 128 dim
            # ]

        Note:
            Prompt support depends on the specific late-interaction model.
            Traditional ColBERT models don't use prompts, but newer variants
            may support instruction-based encoding.
        """
        ...  # pragma: no cover

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of each token embedding (e.g., 128 for ColBERTv2).

        Returns:
            The number of dimensions in each token embedding vector.

        Note:
            This is the dimension of individual token embeddings, not the
            total number of tokens (which varies per text).
        """
        ...  # pragma: no cover


class SentenceTransformerEmbedder:
    """Embedding provider using sentence-transformers library.

    This implementation wraps the popular sentence-transformers library
    for encoding text into dense vector embeddings. It supports hundreds
    of pre-trained models from the Hugging Face Hub.

    The model is lazy-loaded (only when first encode() is called) to avoid
    expensive loading during object construction. This is important for:
    - Fast testing (can construct without loading 200MB+ models)
    - Serialization (can pickle/save embedder configuration)
    - Delayed initialization in distributed settings

    Example:
        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        texts = ["Hello world", "Entity resolution"]
        embeddings = embedder.encode(texts)  # Shape: (2, 384)

    Note:
        Common models and their dimensions:
        - "all-MiniLM-L6-v2": 384 dim (fast, good quality)
        - "all-mpnet-base-v2": 768 dim (slower, better quality)
        - "all-MiniLM-L12-v2": 384 dim (medium speed/quality)

    Note:
        See https://www.sbert.net/docs/pretrained_models.html for full list
        of available models and their performance benchmarks.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ):
        """Initialize SentenceTransformerEmbedder.

        Args:
            model_name: Name of the sentence-transformers model to use.
                Default: "all-MiniLM-L6-v2" (fast, good quality baseline).
            batch_size: Number of texts to encode per batch.
                Default: 32. Use 128+ for GPU with sufficient memory.
            show_progress_bar: Display encoding progress for large datasets.
                Default: False.
            normalize_embeddings: L2-normalize embeddings to unit vectors.
                Default: True. Required for Qdrant Distance.COSINE and
                cosine similarity metrics.

        Note:
            The model is NOT loaded during __init__. It will be loaded
            lazily on the first call to encode() or embedding_dim.
            Device selection (CPU/CUDA/MPS) is automatic. sentence-transformers
            will use GPU if available.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.normalize_embeddings = normalize_embeddings
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        """Get or load the sentence-transformers model.

        Returns:
            Loaded SentenceTransformer model instance.

        Note:
            This method implements lazy loading. The model is cached
            after the first call for reuse.
        """
        if self._model is None:
            logger.info("Loading sentence-transformers model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str] | np.ndarray, prompt: str | None = None) -> np.ndarray:
        """Encode texts into embeddings using sentence-transformers.

        Args:
            texts: List of texts to encode OR pre-computed embeddings (np.ndarray).
                For text input: List of strings to encode. Can be empty.
                For pre-computed: np.ndarray of shape (N, embedding_dim).
                    Pre-computed embeddings are passed through directly.
            prompt: Optional instruction prompt for asymmetric encoding.
                If provided, the prompt guides the embedding model to focus
                on task-specific semantics. Ignored if texts is pre-computed.
                Default: None.

        Returns:
            Numpy array of shape (len(texts), embedding_dim).
            Returns shape (0, embedding_dim) for empty input.

        Note:
            Pre-computed embeddings bypass model encoding and are returned
            directly with dtype conversion to float32 if needed.

        Note:
            This method triggers model loading on first call if not
            already loaded.

        Note:
            The prompt parameter is passed to sentence-transformers'
            encode() method. Models like Qwen3-Embedding, BGE, and E5
            use this for instruction-based encoding. Models without
            prompt support will ignore this parameter.
        """
        # Pre-computed embeddings bypass model
        if isinstance(texts, np.ndarray):
            logger.debug("Pre-computed embeddings detected, bypassing model")
            return texts.astype(np.float32)

        if len(texts) == 0:
            # Return empty array with correct shape
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            prompt=prompt,
        )

        # Ensure numpy array (sentence-transformers should return this, but be explicit)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)

        return embeddings

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension of the model.

        Returns:
            The number of dimensions in embeddings produced by this model.

        Note:
            This property triggers model loading if not already loaded,
            as we need the model to determine its dimension.
        """
        model = self._get_model()
        dim = model.get_sentence_embedding_dimension()
        assert dim is not None, "Model returned None for embedding dimension"
        return dim


class FakeEmbedder:
    """Test double for EmbeddingProvider that produces deterministic embeddings.

    This implementation creates fake embeddings that are:
    1. Deterministic: same text always produces same embedding
    2. Different: different texts produce different embeddings
    3. Optionally normalized: L2-normalized to unit vectors (default: True)
    4. Fast: no model loading, instant computation

    This is crucial for testing VectorBlocker logic without expensive
    model loading (200MB+ downloads, 10+ seconds startup time).

    Example:
        embedder = FakeEmbedder(embedding_dim=128)
        embeddings = embedder.encode(["text1", "text2"])
        # Returns deterministic (2, 128) normalized array instantly

    Note:
        The fake embeddings are based on a hash of the text, ensuring
        deterministic but pseudo-random vectors.
    """

    def __init__(self, embedding_dim: int = 384, normalize_embeddings: bool = True):
        """Initialize FakeEmbedder.

        Args:
            embedding_dim: Dimensionality of fake embeddings to produce.
                Default: 384 (matches all-MiniLM-L6-v2).
            normalize_embeddings: L2-normalize embeddings to unit vectors.
                Default: True (matches SentenceTransformerEmbedder behavior).
        """
        self._embedding_dim = embedding_dim
        self.normalize_embeddings = normalize_embeddings

    def encode(self, texts: list[str] | np.ndarray, prompt: str | None = None) -> np.ndarray:
        """Generate fake deterministic embeddings from texts.

        Args:
            texts: List of texts to encode OR pre-computed embeddings (np.ndarray).
                For text input: List of strings to encode.
                For pre-computed: np.ndarray of shape (N, embedding_dim).
                    Pre-computed embeddings are passed through directly.
            prompt: Optional instruction prompt for asymmetric encoding.
                When provided, produces different embeddings to simulate
                real prompt-aware embedding models. Ignored if texts is pre-computed.
                Default: None.

        Returns:
            Numpy array of shape (len(texts), embedding_dim).
            Each embedding is an L2-normalized vector derived from
            a hash of the text (and prompt if provided).

        Note:
            Pre-computed embeddings bypass fake generation and are returned
            directly with dtype conversion to float32 if needed.

        Note:
            Same text + same prompt always produces the same embedding.
            Different prompts produce different embeddings (simulating
            real models like Qwen3-Embedding, BGE, E5).
        """
        # Pre-computed embeddings bypass fake generation
        if isinstance(texts, np.ndarray):
            logger.debug("Pre-computed embeddings detected, bypassing fake generation")
            return texts.astype(np.float32)

        if len(texts) == 0:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)

        embeddings = []
        for text in texts:
            # Create deterministic embedding from text + prompt hash
            # Include prompt in hash to simulate prompt-aware behavior
            if prompt is not None:
                # Combine text and prompt for hashing
                combined = f"{text}||PROMPT:{prompt}"
            else:
                combined = text

            # Use SHA256 to get stable, uniformly distributed values
            text_hash = hashlib.sha256(combined.encode("utf-8")).digest()

            # Convert hash bytes to integers, then to floats
            # Use numpy random with hash as seed for deterministic generation
            hash_int = int.from_bytes(text_hash[:8], byteorder="big")
            rng = np.random.default_rng(seed=hash_int)

            # Generate random values in [-1, 1]
            embedding = rng.uniform(-1.0, 1.0, size=self._embedding_dim).astype(np.float32)

            # Normalize to unit vector (L2 norm = 1.0) if requested
            if self.normalize_embeddings:
                norm = np.linalg.norm(embedding)
                embedding = embedding / norm

            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        """Get the configured embedding dimension.

        Returns:
            The dimensionality of embeddings produced by this faker.
        """
        return self._embedding_dim


# ============ SPARSE EMBEDDING IMPLEMENTATIONS ============


class FastEmbedSparseEmbedder:
    """Sparse embedding provider using FastEmbed library.

    Wraps FastEmbed's SparseTextEmbedding for efficient sparse vector generation.
    Supports BM25 and neural sparse models (SPLADE).

    Recommended models:
    - "Qdrant/bm25": Classic BM25 keyword matching (lightweight, fast)
    - "prithivida/Splade_PP_en_v1": Neural sparse embeddings (better quality)

    Example:
        # BM25 (lightweight)
        sparse_embedder = FastEmbedSparseEmbedder("Qdrant/bm25")

        # Neural sparse (higher quality)
        sparse_embedder = FastEmbedSparseEmbedder("prithivida/Splade_PP_en_v1")

        # Encode texts
        texts = ["Apple Inc.", "Microsoft Corp.", "Google LLC"]
        sparse_vectors = sparse_embedder.encode(texts)

    Note:
        FastEmbed uses ONNX runtime for efficient CPU inference.
        Model is loaded lazily on first encode() call.
    """

    def __init__(self, model_name: str = "Qdrant/bm25"):
        """Initialize FastEmbed sparse embedder.

        Args:
            model_name: FastEmbed sparse model name.
                Default: "Qdrant/bm25" (classic BM25)
                Alternative: "prithivida/Splade_PP_en_v1" (neural sparse)
        """
        self.model_name = model_name
        self._model: Any = None  # Lazy-loaded on first encode

    def encode(self, texts: list[str], prompt: str | None = None) -> list[dict[str, Any]]:
        """Generate sparse vectors using FastEmbed.

        Args:
            texts: List of texts to encode.
            prompt: Optional instruction prompt (ignored for sparse embeddings).
                Sparse embedders (BM25, SPLADE) don't use prompts for keyword matching.
                This parameter exists for protocol compatibility.

        Returns:
            List of sparse vectors in Qdrant format.
            Each vector is {"indices": [...], "values": [...]}.

        Note:
            The prompt parameter is intentionally ignored, as sparse embedders
            perform keyword matching and don't benefit from instruction prompts.
        """
        # Lazy-load model on first use
        if self._model is None:
            from fastembed import SparseTextEmbedding

            logger.info("Loading FastEmbed sparse model: %s", self.model_name)
            self._model = SparseTextEmbedding(self.model_name)

        # Generate sparse embeddings
        # FastEmbed returns SparseEmbedding objects with .indices and .values
        # Note: prompt is intentionally ignored (sparse = keyword matching)
        sparse_embeddings = list(self._model.embed(texts))

        # Convert to Qdrant format
        result = []
        for emb in sparse_embeddings:
            result.append({"indices": emb.indices.tolist(), "values": emb.values.tolist()})

        logger.debug("Generated %d sparse vectors", len(result))
        return result


class FakeSparseEmbedder:
    """Test double for SparseEmbeddingProvider.

    Produces deterministic fake sparse vectors for testing.
    No actual embedding computation - instant and deterministic.

    Example:
        sparse_embedder = FakeSparseEmbedder()
        sparse_vectors = sparse_embedder.encode(["Apple", "Google"])

        # Returns deterministic fake sparse vectors based on text hashes
        # Each text gets a single index based on hash(text) % 1000
    """

    def encode(self, texts: list[str], prompt: str | None = None) -> list[dict[str, Any]]:
        """Generate deterministic fake sparse vectors.

        Args:
            texts: List of texts to encode.
            prompt: Optional instruction prompt (ignored for sparse embeddings).
                Sparse embedders (BM25, SPLADE) don't use prompts for keyword matching.
                This parameter exists for protocol compatibility.

        Returns:
            List of fake sparse vectors in Qdrant format.
            Each vector has a single index based on text hash.

        Note:
            The prompt parameter is intentionally ignored, as sparse embedders
            perform keyword matching and don't benefit from instruction prompts.
        """
        result = []
        for text in texts:
            # Generate deterministic index from text hash
            # Note: prompt is intentionally ignored (sparse = keyword matching)
            idx = (
                abs(hash(text)) % 1000
            )  # TODO what happens when there are more then 1000 embeddings?
            result.append({"indices": [idx], "values": [1.0]})

        logger.debug("Generated %d fake sparse vectors", len(result))
        return result


# ============ LATE INTERACTION EMBEDDING IMPLEMENTATIONS ============


class FastEmbedLateInteractionEmbedder:
    """Late-interaction embedding provider using FastEmbed library.

    Wraps FastEmbed's LateInteractionTextEmbedding for ColBERT-style multi-vector
    encoding. Each text is encoded into multiple token-level embeddings, enabling
    more nuanced similarity matching via MaxSim strategy.

    The model is lazy-loaded (only when first encode() is called) to avoid
    expensive loading during object construction.

    Recommended models:
    - "colbert-ir/colbertv2.0": Standard ColBERTv2 (128-dim token embeddings)

    Example:
        embedder = FastEmbedLateInteractionEmbedder("colbert-ir/colbertv2.0")
        texts = ["Apple Inc.", "Microsoft Corp."]
        multi_vectors = embedder.encode(texts)
        # Returns: List of [num_tokens, 128] arrays, one per text

    Note:
        FastEmbed uses ONNX runtime for efficient CPU inference.
        Token count varies per text based on tokenization.
    """

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        """Initialize FastEmbed late-interaction embedder.

        Args:
            model_name: FastEmbed late-interaction model name.
                Default: "colbert-ir/colbertv2.0" (ColBERTv2 standard)
                The model_name parameter enables flexibility to use any
                FastEmbed-compatible late-interaction model.

        Note:
            The model is NOT loaded during __init__. It will be loaded
            lazily on the first call to encode() or embedding_dim.
        """
        self.model_name = model_name
        self._model: Any = None  # Lazy-loaded on first encode

    def _get_model(self) -> Any:
        """Get or load the FastEmbed late-interaction model.

        Returns:
            Loaded LateInteractionTextEmbedding model instance.

        Note:
            This method implements lazy loading. The model is cached
            after the first call for reuse.
        """
        if self._model is None:
            from fastembed import LateInteractionTextEmbedding

            logger.info("Loading FastEmbed late-interaction model: %s", self.model_name)
            self._model = LateInteractionTextEmbedding(self.model_name)
        return self._model

    def encode(self, texts: list[str], prompt: str | None = None) -> list[list[list[float]]]:
        """Encode texts into multi-vectors using FastEmbed.

        Args:
            texts: List of texts to encode. Can be empty.
            prompt: Optional instruction prompt for asymmetric encoding.
                If the underlying model supports instruction prompts, this will
                be used. Otherwise, it's silently ignored (model-dependent).
                Default: None.

        Returns:
            List of multi-vectors, one per text.
            Each text becomes [num_tokens, embedding_dim] array.
            Returns empty list for empty input.

        Note:
            This method triggers model loading on first call if not
            already loaded. Prompt support depends on the specific FastEmbed
            late-interaction model being used (e.g., ColBERT may not support
            prompts, but newer models might).
        """
        if len(texts) == 0:
            return []

        model = self._get_model()

        # FastEmbed returns generator of numpy arrays
        # Shape: (num_tokens, embedding_dim) per text
        # Note: Current FastEmbed late-interaction models (ColBERT) don't support prompts
        # We include the parameter for future compatibility, but it's currently ignored
        embeddings_generator = model.embed(texts)

        # Convert generator to list and numpy arrays to nested lists
        result = []
        for embedding in embeddings_generator:
            # embedding is numpy array of shape (num_tokens, embedding_dim)
            # Convert to list[list[float]]
            result.append(embedding.tolist())

        logger.debug(
            "Generated %d multi-vectors with dimensions %s",
            len(result),
            [len(mv) for mv in result],
        )
        return result

    @property
    def embedding_dim(self) -> int:
        """Get the token embedding dimension of the model.

        Returns:
            The number of dimensions in each token embedding.
            ColBERTv2 returns 128.

        Note:
            This property triggers model loading if not already loaded,
            as we need the model to determine its dimension.
        """
        model = self._get_model()
        # FastEmbed's LateInteractionTextEmbedding exposes embedding_size property
        dim: int = model.embedding_size
        return dim


class FakeLateInteractionEmbedder:
    """Test double for LateInteractionEmbeddingProvider.

    Produces deterministic fake multi-vectors for testing without expensive
    model loading. Each text generates a fixed number of token embeddings,
    all deterministic based on text hash + token index.

    This is crucial for testing VectorBlocker logic with late-interaction
    embeddings without 100MB+ model downloads and multi-second startup times.

    Example:
        embedder = FakeLateInteractionEmbedder(embedding_dim=128, num_tokens=5)
        multi_vectors = embedder.encode(["text1", "text2"])
        # Returns deterministic [5 tokens x 128 dim] per text, instantly

    Note:
        The fake embeddings are based on a hash of (text + token_index),
        ensuring deterministic but pseudo-random vectors.
    """

    def __init__(self, embedding_dim: int = 128, num_tokens: int = 5):
        """Initialize FakeLateInteractionEmbedder.

        Args:
            embedding_dim: Dimensionality of each token embedding.
                Default: 128 (matches ColBERTv2).
            num_tokens: Number of token embeddings per text.
                Default: 5 (simple for testing).

        Note:
            Unlike real late-interaction models where token count varies
            per text, this fake uses fixed num_tokens for simplicity and
            determinism in tests.
        """
        self._embedding_dim = embedding_dim
        self.num_tokens = num_tokens

    def encode(self, texts: list[str], prompt: str | None = None) -> list[list[list[float]]]:
        """Generate fake deterministic multi-vectors from texts.

        Args:
            texts: List of texts to encode.
            prompt: Optional instruction prompt for asymmetric encoding.
                When provided, produces different multi-vectors to simulate
                real prompt-aware late-interaction models (e.g., ColBERT-v2+).
                Default: None.

        Returns:
            List of multi-vectors, one per text.
            Each text produces exactly num_tokens token embeddings,
            each with embedding_dim dimensions.

        Note:
            Same text + same prompt always produces the same multi-vectors.
            Different prompts produce different multi-vectors (simulating
            real late-interaction models with instruction-following capability).
            Different token indices within same text produce different embeddings.
        """
        if len(texts) == 0:
            return []

        result: list[list[list[float]]] = []
        for text in texts:
            # Generate num_tokens token embeddings for this text
            text_tokens: list[list[float]] = []
            for token_idx in range(self.num_tokens):
                # Create deterministic embedding from text + prompt + token_idx hash
                # Include prompt in hash to simulate prompt-aware behavior
                if prompt is not None:
                    token_key = f"{text}||PROMPT:{prompt}__token{token_idx}"
                else:
                    token_key = f"{text}__token{token_idx}"

                # Use SHA256 to get stable, uniformly distributed values
                token_hash = hashlib.sha256(token_key.encode("utf-8")).digest()

                # Convert hash bytes to integers, then to floats
                hash_int = int.from_bytes(token_hash[:8], byteorder="big")
                rng = np.random.default_rng(seed=hash_int)

                # Generate random values in [-1, 1]
                token_embedding = rng.uniform(-1.0, 1.0, size=self._embedding_dim).astype(
                    np.float32
                )

                # Convert numpy array to list
                text_tokens.append(token_embedding.tolist())

            result.append(text_tokens)

        logger.debug(
            "Generated %d fake multi-vectors (%d tokens x %d dim each)",
            len(result),
            self.num_tokens,
            self._embedding_dim,
        )
        return result

    @property
    def embedding_dim(self) -> int:
        """Get the configured token embedding dimension.

        Returns:
            The dimensionality of each token embedding produced by this faker.
        """
        return self._embedding_dim


# ============ DISK-CACHED EMBEDDER ============


class DiskCachedEmbedder:
    """Disk-backed embedding cache with two-tier caching (hot memory + cold disk).

    This implementation prevents OOM by keeping only hot data in memory while storing
    unlimited embeddings on disk using SQLite.

    Architecture:
    - Hot cache: In-memory OrderedDict with LRU eviction (configurable size)
    - Cold storage: SQLite database for unlimited disk-backed storage

    Lookup flow:
        Text → Hash → Hot cache?
                      ↓ No
                      Cold storage (SQLite)?
                      ↓ No
                      Compute with embedder → Store in both hot + cold

    Example:
        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        cached = DiskCachedEmbedder(
            embedder=embedder,
            cache_dir=Path("./cache"),
            namespace="my_model",
            memory_cache_size=10_000,
        )
        embeddings = cached.encode(texts)  # Uses cache when available

    Note:
        Future versions will support prompt-based caching for instruction embeddings.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        cache_dir: Path,
        namespace: str = "default",
        memory_cache_size: int = 100_000,
        hash_algorithm: str = "blake2b",
    ):
        """Initialize DiskCachedEmbedder.

        Args:
            embedder: Underlying embedding provider to use for cache misses.
            cache_dir: Directory to store SQLite database(s).
            namespace: Namespace for this cache (each namespace gets separate DB).
            memory_cache_size: Maximum number of embeddings in hot cache.
            hash_algorithm: Hash algorithm for cache keys (default: blake2b).
        """
        from collections import OrderedDict

        self.embedder = embedder
        self.cache_dir = Path(cache_dir)
        self.namespace = namespace
        self.memory_cache_size = memory_cache_size
        self.hash_algorithm = hash_algorithm

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite database
        self.db_path = self.cache_dir / f"{namespace}.db"
        self._init_db()

        # Initialize hot cache (LRU using OrderedDict)
        self._hot_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        # Statistics
        self._stats = {
            "hits_hot": 0,
            "hits_cold": 0,
            "misses": 0,
        }

    def _init_db(self) -> None:
        """Initialize SQLite database with schema.

        Creates the embeddings table with:
        - text_hash (PRIMARY KEY): Hash of text (and future prompt)
        - embedding (BLOB): Serialized numpy array
        - text (TEXT): Original text (for debugging/inspection)
        - prompt (TEXT): Future prompt support
        - created_at (TIMESTAMP): Creation timestamp
        """
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create embeddings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                text TEXT,
                prompt TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index on text_hash (already primary key, but explicit for clarity)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_text_hash ON embeddings(text_hash)
        """)

        conn.commit()
        conn.close()

        logger.debug("Initialized SQLite database at %s", self.db_path)

    def _hash_text(self, text: str, prompt: str | None = None) -> str:
        """Generate cache key from text (and future prompt).

        Args:
            text: Text to hash.
            prompt: Optional prompt for instruction-based embeddings.

        Returns:
            Hexadecimal hash string.

        Note:
            Cache key design:
            - Without prompt: hash(text)
            - With prompt: hash(text + "|||PROMPT|||" + prompt)

            This ensures same text with different prompts gets different cache entries.
        """
        if prompt is None:
            cache_input = text
        else:
            # Use separator to distinguish text from prompt
            cache_input = f"{text}|||PROMPT|||{prompt}"

        # Use blake2b for fast, collision-resistant hashing
        hasher = hashlib.blake2b()
        hasher.update(cache_input.encode("utf-8"))
        return hasher.hexdigest()

    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Convert numpy array to bytes for SQLite BLOB.

        Args:
            embedding: Numpy array to serialize.

        Returns:
            Bytes representation of the array.

        Note:
            Uses numpy's tobytes() for efficient serialization.
            Stores dtype and shape information implicitly (assumes float32).
        """
        return embedding.astype(np.float32).tobytes()

    def _deserialize_embedding(self, blob: bytes) -> np.ndarray:
        """Convert SQLite BLOB back to numpy array.

        Args:
            blob: Bytes from SQLite BLOB column.

        Returns:
            Numpy array (1D, will be reshaped by caller if needed).

        Note:
            Assumes float32 dtype. Caller must know embedding dimension for reshaping.
        """
        return np.frombuffer(blob, dtype=np.float32)

    def _get_from_db(self, text_hash: str) -> np.ndarray | None:
        """Lookup embedding in SQLite.

        Args:
            text_hash: Cache key (hash of text/prompt).

        Returns:
            Embedding array if found, None otherwise.
        """
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT embedding FROM embeddings WHERE text_hash = ?", (text_hash,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        # Deserialize and reshape
        embedding = self._deserialize_embedding(row[0])
        return embedding

    def _put_to_db(
        self, text_hash: str, embedding: np.ndarray, text: str, prompt: str | None = None
    ) -> None:
        """Store embedding in SQLite with metadata.

        Args:
            text_hash: Cache key (hash of text/prompt).
            embedding: Embedding array to store.
            text: Original text (for debugging).
            prompt: Optional prompt (for future support).
        """
        import sqlite3

        serialized = self._serialize_embedding(embedding)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO embeddings (text_hash, embedding, text, prompt)
            VALUES (?, ?, ?, ?)
        """,
            (text_hash, serialized, text, prompt),
        )

        conn.commit()
        conn.close()

    def _promote_to_hot(self, text_hash: str, embedding: np.ndarray) -> None:
        """Add to hot cache with LRU eviction.

        Args:
            text_hash: Cache key.
            embedding: Embedding to add to hot cache.

        Note:
            Uses OrderedDict for LRU:
            - move_to_end() marks as most recent
            - popitem(last=False) removes least recent
        """
        # If already in hot cache, move to end (mark as most recent)
        if text_hash in self._hot_cache:
            self._hot_cache.move_to_end(text_hash)
            return

        # Add to hot cache
        self._hot_cache[text_hash] = embedding

        # Evict oldest if exceeds max size
        if len(self._hot_cache) > self.memory_cache_size:
            # Remove least recently used (first item)
            self._hot_cache.popitem(last=False)

    def encode(self, texts: list[str] | np.ndarray, prompt: str | None = None) -> np.ndarray:
        """Encode with two-tier caching (hot memory + cold disk).

        Args:
            texts: List of texts to encode OR pre-computed embeddings (np.ndarray).
                For text input: List of strings to encode. Can be empty.
                For pre-computed: np.ndarray of shape (N, embedding_dim).
                    Pre-computed embeddings bypass caching entirely.
            prompt: Optional instruction prompt for task-specific encoding.
                Different prompts create separate cache entries for same text.
                Ignored if texts is pre-computed embeddings.

        Returns:
            Numpy array of shape (len(texts), embedding_dim).

        Note:
            Pre-computed embeddings bypass both hot and cold caches and are
            returned directly with dtype conversion to float32 if needed.

        Note:
            Batch optimization: Only cache misses are computed with embedder.
            Results maintain input order.
            Cache key includes both text AND prompt for correct cache isolation.
        """
        # Pre-computed embeddings bypass cache entirely
        if isinstance(texts, np.ndarray):
            logger.debug("Pre-computed embeddings detected, bypassing cache")
            return texts.astype(np.float32)

        if len(texts) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        # Build lookup: text → hash → embedding
        results: dict[str, np.ndarray] = {}
        texts_to_compute: list[str] = []

        for text in texts:
            text_hash = self._hash_text(text, prompt=prompt)

            # Check hot cache
            if text_hash in self._hot_cache:
                results[text] = self._hot_cache[text_hash]
                self._stats["hits_hot"] += 1
                # Move to end (mark as most recent)
                self._hot_cache.move_to_end(text_hash)
                continue

            # Check cold storage (SQLite)
            cold_embedding = self._get_from_db(text_hash)
            if cold_embedding is not None:
                results[text] = cold_embedding
                self._stats["hits_cold"] += 1
                # Promote to hot cache
                self._promote_to_hot(text_hash, cold_embedding)
                continue

            # Cache miss - need to compute
            if text not in results:  # Avoid duplicates in same batch
                texts_to_compute.append(text)
            self._stats["misses"] += 1

        # Compute missing embeddings (batch)
        if texts_to_compute:
            computed = self.embedder.encode(texts_to_compute, prompt=prompt)

            # Store computed embeddings
            for text, embedding in zip(texts_to_compute, computed):
                text_hash = self._hash_text(text, prompt=prompt)

                # Store in cold storage (SQLite)
                self._put_to_db(text_hash, embedding, text, prompt=prompt)

                # Store in hot cache
                self._promote_to_hot(text_hash, embedding)

                # Store in results
                results[text] = embedding

        # Build output maintaining input order
        output = np.array([results[text] for text in texts], dtype=np.float32)
        return output

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension from underlying embedder.

        Returns:
            Embedding dimension.
        """
        return self.embedder.embedding_dim

    def cache_info(self) -> dict[str, int | float]:
        """Return cache statistics: hits_hot, hits_cold, misses, hot_size, cold_size, hit_rate.

        Returns:
            Dictionary with cache statistics:
            - hits_hot: Number of hot cache hits
            - hits_cold: Number of cold storage hits
            - misses: Number of cache misses (computed)
            - hot_size: Current hot cache size
            - cold_size: Current cold storage size
            - hit_rate: Overall hit rate (0.0 to 1.0)
        """
        import sqlite3

        # Get cold storage size
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        cold_size = cursor.fetchone()[0]
        conn.close()

        # Calculate hit rate
        total_requests = self._stats["hits_hot"] + self._stats["hits_cold"] + self._stats["misses"]
        if total_requests == 0:
            hit_rate = 0.0
        else:
            total_hits = self._stats["hits_hot"] + self._stats["hits_cold"]
            hit_rate = total_hits / total_requests

        return {
            "hits_hot": self._stats["hits_hot"],
            "hits_cold": self._stats["hits_cold"],
            "misses": self._stats["misses"],
            "hot_size": len(self._hot_cache),
            "cold_size": cold_size,
            "hit_rate": hit_rate,
        }

    def cache_clear(self) -> None:
        """Clear both hot and cold caches.

        This removes all cached embeddings and resets statistics.
        """
        import sqlite3

        # Clear hot cache
        self._hot_cache.clear()

        # Clear cold storage
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM embeddings")
        conn.commit()
        conn.close()

        # Reset statistics
        self._stats = {
            "hits_hot": 0,
            "hits_cold": 0,
            "misses": 0,
        }

        logger.debug("Cleared all caches for namespace %s", self.namespace)
