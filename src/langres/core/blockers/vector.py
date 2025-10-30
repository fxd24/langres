"""VectorBlocker implementation for embedding-based candidate generation.

This blocker uses sentence-transformers for embeddings and FAISS for ANN search
to efficiently generate candidate pairs without N² complexity. It is
schema-agnostic, accepting a schema_factory and text_field_extractor to work
with any Pydantic schema type.
"""

import logging
from collections.abc import Callable, Iterator
from typing import Any

import faiss  # type: ignore[import-untyped]
import numpy as np
from sentence_transformers import SentenceTransformer

from langres.core.blocker import Blocker, SchemaT
from langres.core.models import ERCandidate

logger = logging.getLogger(__name__)


class VectorBlocker(Blocker[SchemaT]):
    """Schema-agnostic blocker using embeddings and ANN search.

    This blocker uses sentence-transformers to encode entities into vector
    embeddings, then uses FAISS (Facebook AI Similarity Search) to perform
    approximate nearest neighbor (ANN) search to efficiently find similar
    entities without O(N²) complexity.

    The blocker is schema-agnostic: it works with ANY Pydantic schema by
    accepting a schema_factory (to transform raw dicts) and a
    text_field_extractor (to extract text for embedding).

    Example:
        # For companies
        def company_factory(record: dict) -> CompanySchema:
            return CompanySchema(
                id=record["id"],
                name=record["name"],
                address=record.get("address")
            )

        blocker = VectorBlocker(
            schema_factory=company_factory,
            text_field_extractor=lambda x: x.name,
            k_neighbors=10,
            model_name="all-MiniLM-L6-v2"
        )

        candidates = blocker.stream(company_records)

        # For products (different schema, same blocker!)
        def product_factory(record: dict) -> ProductSchema:
            return ProductSchema(
                id=record["product_id"],
                title=record["product_name"]
            )

        blocker = VectorBlocker(
            schema_factory=product_factory,
            text_field_extractor=lambda x: x.title,
            k_neighbors=10
        )

        candidates = blocker.stream(product_records)

    Note:
        This blocker has O(N log N) complexity for building the index and
        O(k log N) for searching, where k is the number of neighbors. This
        is much more scalable than all-pairs O(N²) blocking.

    Note:
        The default model "all-MiniLM-L6-v2" is a good balance of speed and
        quality. It produces 384-dimensional embeddings and is fast to encode.
        For better quality (but slower), consider "all-mpnet-base-v2".

    Note:
        Blocking recall is critical - we must not miss true matches. The
        k_neighbors parameter should be tuned to achieve >= 95% recall.
        Higher k = better recall but more candidates (and cost).
    """

    def __init__(
        self,
        schema_factory: Callable[[dict[str, Any]], SchemaT],
        text_field_extractor: Callable[[SchemaT], str],
        k_neighbors: int = 10,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """Initialize VectorBlocker.

        Args:
            schema_factory: Callable that transforms a raw dict into a
                Pydantic schema object (SchemaT). This allows the blocker
                to work with any schema type.
            text_field_extractor: Callable that extracts the text to embed
                from a SchemaT object. For example, lambda x: x.name.
            k_neighbors: Number of nearest neighbors to retrieve for each
                entity. Higher values improve recall but generate more
                candidates. Default: 10.
            model_name: Name of the sentence-transformers model to use for
                embeddings. Default: "all-MiniLM-L6-v2" (fast, good quality).

        Raises:
            ValueError: If k_neighbors is not positive.
        """
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be positive")

        self.schema_factory = schema_factory
        self.text_field_extractor = text_field_extractor
        self.k_neighbors = k_neighbors
        self.model_name = model_name

        # Lazy-load the model (only when stream() is called)
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        """Get or load the sentence-transformers model.

        Returns:
            Loaded SentenceTransformer model.

        Note:
            This lazy-loads the model to avoid loading at initialization time.
            The model is cached after the first call.
        """
        if self._model is None:
            logger.info("Loading sentence-transformers model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def stream(self, data: list[Any]) -> Iterator[ERCandidate[SchemaT]]:
        """Generate candidate pairs using embedding similarity and ANN search.

        Args:
            data: List of raw data items (typically dicts). The schema_factory
                transforms these into SchemaT objects.

        Yields:
            ERCandidate[SchemaT] objects containing:
            - left: Normalized entity (SchemaT)
            - right: Normalized entity (SchemaT)
            - blocker_name: "vector_blocker"

        Note:
            This implementation:
            1. Normalizes raw data to SchemaT using schema_factory
            2. Extracts text for each entity using text_field_extractor
            3. Encodes text into embeddings using sentence-transformers
            4. Builds a FAISS index for efficient ANN search
            5. For each entity, finds k nearest neighbors
            6. Yields deduplicated pairs (no both (a,b) and (b,a))

        Note:
            Empty datasets or single-entity datasets produce no candidates.
        """
        # Handle empty dataset
        if len(data) == 0:
            return

        # 1. Normalize schema: transform raw dicts to SchemaT
        entities = [self.schema_factory(record) for record in data]

        # Handle single entity (no pairs possible)
        if len(entities) <= 1:
            return

        # 2. Extract text for embedding
        texts = [self.text_field_extractor(entity) for entity in entities]

        # 3. Encode texts into embeddings
        model = self._get_model()
        logger.info("Encoding %d entities into embeddings", len(texts))
        embeddings = model.encode(texts, convert_to_numpy=True)

        # Ensure embeddings is a numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        # 4. Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
        index.add(embeddings.astype(np.float32))

        # 5. Search for k nearest neighbors for each entity
        # k+1 because the nearest neighbor will be the entity itself
        k = min(self.k_neighbors + 1, len(entities))
        distances, indices = index.search(embeddings.astype(np.float32), k)

        # 6. Generate pairs from nearest neighbors
        # Use a set to track seen pairs and avoid duplicates
        seen_pairs: set[frozenset[str]] = set()

        for i in range(len(entities)):
            # Get neighbor indices for entity i (skip first, which is itself)
            neighbor_indices = indices[i][1:]  # Skip index 0 (self)

            for j in neighbor_indices:
                j = int(j)  # Convert numpy.int64 to int

                # Create a canonical pair representation (order-independent)
                pair_key = frozenset([entities[i].id, entities[j].id])  # type: ignore[attr-defined]

                # Skip if we've already seen this pair
                if pair_key in seen_pairs:
                    continue

                seen_pairs.add(pair_key)

                # Yield the candidate pair with consistent ordering (i < j)
                if i < j:
                    yield ERCandidate(
                        left=entities[i],
                        right=entities[j],
                        blocker_name="vector_blocker",
                    )
                else:
                    yield ERCandidate(
                        left=entities[j],
                        right=entities[i],
                        blocker_name="vector_blocker",
                    )
