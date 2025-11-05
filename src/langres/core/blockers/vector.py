"""VectorBlocker implementation for embedding-based candidate generation.

This blocker uses injected embedding and vector index providers for
efficient candidate pair generation without N² complexity. It is
schema-agnostic, accepting a schema_factory and text_field_extractor to work
with any Pydantic schema type.

The separation of embedding and indexing concerns enables:
- Swapping embedding models during optimization
- Caching embeddings between train and inference
- Testing with fake implementations (no model loading)
"""

import logging
from collections.abc import Callable, Iterator
from typing import Any

from langres.core.blocker import Blocker, SchemaT
from langres.core.embeddings import EmbeddingProvider
from langres.core.models import ERCandidate
from langres.core.reports import CandidateInspectionReport
from langres.core.vector_index import VectorIndex

logger = logging.getLogger(__name__)


class VectorBlocker(Blocker[SchemaT]):
    """Schema-agnostic blocker using embeddings and ANN search with dependency injection.

    This blocker separates embedding computation from vector indexing by
    accepting injected EmbeddingProvider and VectorIndex implementations.
    This enables:
    - Swapping embedding models during optimization
    - Caching embeddings between train and inference phases
    - Testing with fake implementations (no expensive model loading)
    - Using different vector backends (FAISS, Annoy, cloud services)

    The blocker is schema-agnostic: it works with ANY Pydantic schema by
    accepting a schema_factory (to transform raw dicts) and a
    text_field_extractor (to extract text for embedding).

    Example (production with real implementations):
        from langres.core.embeddings import SentenceTransformerEmbedder
        from langres.core.vector_index import FAISSIndex

        def company_factory(record: dict) -> CompanySchema:
            return CompanySchema(
                id=record["id"],
                name=record["name"],
                address=record.get("address")
            )

        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        index = FAISSIndex(metric="L2")

        blocker = VectorBlocker(
            schema_factory=company_factory,
            text_field_extractor=lambda x: x.name,
            embedding_provider=embedder,
            vector_index=index,
            k_neighbors=10
        )

        candidates = blocker.stream(company_records)

    Example (testing with fakes):
        from langres.core.embeddings import FakeEmbedder
        from langres.core.vector_index import FakeVectorIndex

        embedder = FakeEmbedder(embedding_dim=128)
        index = FakeVectorIndex()

        blocker = VectorBlocker(
            schema_factory=company_factory,
            text_field_extractor=lambda x: x.name,
            embedding_provider=embedder,
            vector_index=index,
            k_neighbors=10
        )

        # Instant, deterministic testing!
        candidates = blocker.stream(test_data)

    Note:
        This blocker has O(N log N) complexity for building the index and
        O(k log N) for searching, where k is the number of neighbors. This
        is much more scalable than all-pairs O(N²) blocking.

    Note:
        Blocking recall is critical - we must not miss true matches. The
        k_neighbors parameter should be tuned to achieve >= 95% recall.
        Higher k = better recall but more candidates (and cost).
    """

    def __init__(
        self,
        schema_factory: Callable[
            [dict[str, Any]], SchemaT
        ],  # TODO this seems a tight coupling to the schema.
        text_field_extractor: Callable[[SchemaT], str],
        embedding_provider: EmbeddingProvider,
        vector_index: VectorIndex,
        k_neighbors: int = 10,
    ):
        """Initialize VectorBlocker with injected dependencies.

        Args:
            schema_factory: Callable that transforms a raw dict into a
                Pydantic schema object (SchemaT). This allows the blocker
                to work with any schema type.
            text_field_extractor: Callable that extracts the text to embed
                from a SchemaT object. For example, lambda x: x.name.
            embedding_provider: Provider for encoding text into embeddings.
                Use SentenceTransformerEmbedder for production,
                FakeEmbedder for testing.
            vector_index: Index for ANN search on embeddings.
                Use FAISSIndex for production, FakeVectorIndex for testing.
            k_neighbors: Number of nearest neighbors to retrieve for each
                entity. Higher values improve recall but generate more
                candidates. Default: 10.

        Raises:
            ValueError: If k_neighbors is not positive.
        """
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be positive")

        self.schema_factory = schema_factory
        self.text_field_extractor = text_field_extractor
        self.embedding_provider = embedding_provider
        self.vector_index = vector_index
        self.k_neighbors = k_neighbors

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
            3. Encodes text into embeddings using embedding_provider
            4. Builds a vector index for efficient ANN search
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

        # 3. Encode texts into embeddings using injected provider
        logger.info("Encoding %d entities into embeddings", len(texts))
        embeddings = self.embedding_provider.encode(texts)

        # 4. Build vector index using injected index
        self.vector_index.build(embeddings)

        # 5. Search for k nearest neighbors for each entity
        # k+1 because the nearest neighbor will be the entity itself
        k = min(self.k_neighbors + 1, len(entities))
        _, indices = self.vector_index.search(embeddings, k)

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

    def inspect_candidates(
        self,
        candidates: list[ERCandidate[SchemaT]],
        entities: list[SchemaT],
        sample_size: int = 10,
    ) -> CandidateInspectionReport:
        """Explore candidates without ground truth labels.

        Use this method to understand VectorBlocker output before labeling.
        Provides statistics, distribution, examples, and k_neighbors tuning
        recommendations.

        Args:
            candidates: List of candidate pairs generated by blocker
            entities: List of normalized entities (for readable text extraction)
            sample_size: Number of examples to include in report (default: 10)

        Returns:
            CandidateInspectionReport with VectorBlocker-specific recommendations.
        """
        # Handle empty cases
        if len(candidates) == 0:
            return CandidateInspectionReport(
                total_candidates=0,
                avg_candidates_per_entity=0.0,
                candidate_distribution={},
                examples=[],
                recommendations=[
                    "No candidates generated - check data and k_neighbors parameter",
                    "Consider increasing k_neighbors if you have enough entities",
                ],
            )

        # Compute total candidates and average per entity
        total_candidates = len(candidates)
        num_entities = len(entities)
        avg_candidates_per_entity = (
            total_candidates * 2 / num_entities if num_entities > 0 else 0.0
        )  # *2 because each candidate involves 2 entities

        # Build distribution histogram
        # Count how many candidates each entity appears in
        entity_candidate_count: dict[str, int] = {}
        for candidate in candidates:
            left_id = candidate.left.id  # type: ignore[attr-defined]
            right_id = candidate.right.id  # type: ignore[attr-defined]
            entity_candidate_count[left_id] = entity_candidate_count.get(left_id, 0) + 1
            entity_candidate_count[right_id] = entity_candidate_count.get(right_id, 0) + 1

        # Create histogram buckets
        distribution: dict[str, int] = {
            "1-3": 0,
            "4-6": 0,
            "7-9": 0,
            "10+": 0,
        }

        for count in entity_candidate_count.values():
            if count <= 3:
                distribution["1-3"] += 1
            elif count <= 6:
                distribution["4-6"] += 1
            elif count <= 9:
                distribution["7-9"] += 1
            else:
                distribution["10+"] += 1

        # Sample examples with readable text
        examples = []
        for candidate in candidates[:sample_size]:
            left_text = self.text_field_extractor(candidate.left)
            right_text = self.text_field_extractor(candidate.right)
            examples.append(
                {
                    "left_id": candidate.left.id,  # type: ignore[attr-defined]
                    "right_id": candidate.right.id,  # type: ignore[attr-defined]
                    "left_text": left_text,
                    "right_text": right_text,
                }
            )

        # Generate recommendations based on statistics
        recommendations = []
        if avg_candidates_per_entity < 3:
            recommendations.append(
                f"Low candidate count (avg {avg_candidates_per_entity:.1f} per entity) - "
                f"increase k_neighbors (current: {self.k_neighbors}) for better recall"
            )
        elif avg_candidates_per_entity > 8:
            recommendations.append(
                f"High candidate count (avg {avg_candidates_per_entity:.1f} per entity) - "
                f"decrease k_neighbors (current: {self.k_neighbors}) to reduce false positives"
            )
        else:
            recommendations.append(
                f"Candidate count looks reasonable (avg {avg_candidates_per_entity:.1f} per entity)"
            )

        # Add threshold suggestion based on distribution
        if avg_candidates_per_entity > 0:
            suggested_k = int(avg_candidates_per_entity / 2)
            if suggested_k != self.k_neighbors:
                recommendations.append(
                    f"Consider trying k_neighbors={suggested_k} based on current distribution"
                )

        return CandidateInspectionReport(
            total_candidates=total_candidates,
            avg_candidates_per_entity=avg_candidates_per_entity,
            candidate_distribution=distribution,
            examples=examples,
            recommendations=recommendations,
        )
