"""Embedding providers for text-to-vector encoding.

This module provides abstractions for converting text into vector embeddings,
separating embedding computation from vector search to enable:
- Swapping embedding models (sentence-transformers, OpenAI, etc.)
- Caching embeddings between train/optimize and inference phases
- Independent testing of embedding logic

The core abstraction is the EmbeddingProvider Protocol, which defines a
standard interface for encoding text into numerical vectors.
"""

import hashlib
import logging
from typing import Protocol

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

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into vector embeddings.

        Args:
            texts: List of texts to encode. Can be empty.

        Returns:
            Numpy array of shape (len(texts), embedding_dim).
            Returns shape (0, embedding_dim) for empty input.

        Note:
            Implementations should return consistent dtypes (typically float32)
            for compatibility with vector index backends like FAISS.
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

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into embeddings using sentence-transformers.

        Args:
            texts: List of texts to encode. Can be empty.

        Returns:
            Numpy array of shape (len(texts), embedding_dim).
            Returns shape (0, embedding_dim) for empty input.

        Note:
            This method triggers model loading on first call if not
            already loaded.
        """
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

    def encode(self, texts: list[str]) -> np.ndarray:
        """Generate fake deterministic embeddings from texts.

        Args:
            texts: List of texts to encode.

        Returns:
            Numpy array of shape (len(texts), embedding_dim).
            Each embedding is an L2-normalized vector derived from
            a hash of the text.

        Note:
            Same text always produces the same embedding vector.
            Different texts produce different vectors.
        """
        if len(texts) == 0:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)

        embeddings = []
        for text in texts:
            # Create deterministic embedding from text hash
            # Use SHA256 to get stable, uniformly distributed values
            text_hash = hashlib.sha256(text.encode("utf-8")).digest()

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
