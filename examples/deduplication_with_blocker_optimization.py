"""Entity Resolution Example: Company Deduplication with Blocker Optimization.

This example demonstrates the CORRECT usage patterns for langres.core components:

1. **VectorBlocker with Dependency Injection**:
   - Uses `schema_factory` to transform raw data to Pydantic models
   - Uses `text_field_extractor` to combine multiple fields for embedding
   - Injects `SentenceTransformerEmbedder` and `FAISSIndex` for composability

2. **LLMJudgeModule with Azure OpenAI**:
   - Accepts iterator of candidates (use `iter()` to convert list)
   - Uses LiteLLM for enhanced observability (Langfuse tracing)

3. **Clusterer for Entity Formation**:
   - Returns `list[set[str]]` (clusters as sets)
   - Threshold-based clustering using graph connected components

4. **BlockerOptimizer with Optuna + wandb**:
   - Optimizes embedding model and k_neighbors hyperparameters
   - Uses BCubed F1 metric for evaluation
   - Integrates with wandb for experiment tracking

**KEY PATTERNS DEMONSTRATED**:
- Separation of concerns: Blocker normalizes schema, Module compares
- Dependency injection: Embedding and vector index are injectable
- Type safety: All components use Pydantic for validation
- Observability: Full provenance tracking via PairwiseJudgement

Environment variables required by this example:
    AZURE_API_ENDPOINT: Azure OpenAI endpoint URL (required for Azure OpenAI)
    AZURE_API_KEY: Azure OpenAI API key (required for Azure OpenAI)
    AZURE_API_VERSION: Azure OpenAI API version (optional, defaults to 2024-02-15-preview)
    WANDB_API_KEY: Weights & Biases API key (required for experiment tracking)
    LANGFUSE_PUBLIC_KEY: Langfuse public API key (required for LLM tracing)
    LANGFUSE_SECRET_KEY: Langfuse secret API key (required for LLM tracing)
    LANGFUSE_HOST: Langfuse host URL (optional, defaults to https://cloud.langfuse.com)
    LANGFUSE_PROJECT: Langfuse project name (optional, defaults to langres)

Note:
    Environment variables are only validated when the corresponding service is used.
    For example, LANGFUSE_* vars are only required when enable_langfuse=True.
"""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from langres.clients import create_llm_client, create_wandb_tracker
from langres.clients.settings import Settings
from langres.core.blockers.vector import VectorBlocker
from langres.core.clusterer import Clusterer
from langres.core.embeddings import SentenceTransformerEmbedder
from langres.core.metrics import calculate_bcubed_metrics, calculate_pairwise_metrics
from langres.core.modules.llm_judge import LLMJudgeModule
from langres.core.optimizers.blocker_optimizer import BlockerOptimizer
from langres.core.vector_index import FAISSIndex

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CompanySchema(BaseModel):
    """Company entity schema."""

    id: str = Field(description="Unique company identifier")
    name: str = Field(description="Company name")
    address: str | None = Field(default=None, description="Company address")
    website: str | None = Field(default=None, description="Company website")


def load_companies(data_path: Path) -> list[CompanySchema]:
    """Load company records from JSON file."""
    with open(data_path) as f:
        data = json.load(f)
    return [CompanySchema(**record) for record in data]


def load_labels(labels_path: Path) -> dict[str, Any]:
    """Load ground truth labels from JSON file."""
    with open(labels_path) as f:
        data: dict[str, Any] = json.load(f)
        return data


def train_test_split(  # TODO: are we going to split in a smart way or will all the train dataset with groupings larger the oneself's entity be in train and the single/unique entities all in test?
    entities: list[CompanySchema], labels: dict[str, Any], test_size: float = 0.2
) -> tuple[list[CompanySchema], list[CompanySchema], list[list[str]], list[list[str]]]:
    """Split entities and labels into train and test sets.

    Args:
        entities: List of all entities
        labels: Ground truth labels dict with 'clusters' key
        test_size: Fraction of data for test set (default: 0.2)

    Returns:
        train_entities, test_entities, train_clusters, test_clusters
    """
    clusters = labels["clusters"]
    n_test_clusters = max(1, int(len(clusters) * test_size))

    # Simple split: last 20% of clusters go to test set
    test_clusters = clusters[-n_test_clusters:]
    train_clusters = clusters[:-n_test_clusters]

    # Extract entity IDs
    test_ids = {entity_id for cluster in test_clusters for entity_id in cluster}
    train_ids = {entity_id for cluster in train_clusters for entity_id in cluster}

    # Split entities
    train_entities = [e for e in entities if e.id in train_ids]
    test_entities = [e for e in entities if e.id in test_ids]

    logger.info(
        "Train: %d entities, %d clusters | Test: %d entities, %d clusters",
        len(train_entities),
        len(train_clusters),
        len(test_entities),
        len(test_clusters),
    )

    return train_entities, test_entities, train_clusters, test_clusters


def run_deduplication_pipeline(
    entities: list[CompanySchema],
    embedding_model: str,
    k_neighbors: int,
    llm_client: Any,
    azure_model: str,
    cluster_threshold: float = 0.5,
) -> tuple[list[set[str]], list[Any]]:
    """Run the full deduplication pipeline.

    Args:
        entities: List of entities to deduplicate
        embedding_model: Name of sentence-transformer model
        k_neighbors: Number of nearest neighbors
        llm_client: LiteLLM client for LLM judge
        azure_model: Azure OpenAI model name (e.g., "azure/gpt-5-mini")
        cluster_threshold: Threshold for clustering (default: 0.5)

    Returns:
        Tuple of (predicted_clusters, judgements) where:
        - predicted_clusters: list of sets of entity IDs
        - judgements: list of PairwiseJudgement objects (for cost tracking)
    """
    # ==================================================================================
    # STEP 1: Generate Candidates using VectorBlocker
    # ==================================================================================
    # This demonstrates the CORRECT VectorBlocker API with dependency injection.
    #
    # KEY PATTERN: Dependency Injection
    # Instead of VectorBlocker internally creating SentenceTransformer and FAISS,
    # we inject them as interfaces (EmbeddingProvider, VectorIndex). This enables:
    # - Testing with FakeEmbedder/FakeVectorIndex (no model loading)
    # - Swapping embedding models during optimization
    # - Using different vector backends (FAISS, Annoy, Qdrant, etc.)
    logger.info("Generating candidates with VectorBlocker...")

    # Initialize embedding provider (injected dependency)
    # This is an EmbeddingProvider implementation using sentence-transformers
    embedding_provider = SentenceTransformerEmbedder(model_name=embedding_model)

    # Initialize vector index (injected dependency)
    # FAISSIndex doesn't need dimension - it's inferred during build()
    vector_index = FAISSIndex(metric="L2")

    # KEY PATTERN: Schema Factory
    # VectorBlocker is schema-agnostic. The schema_factory transforms raw data
    # (typically dicts from JSON/DB) into the Pydantic schema (CompanySchema).
    # This separates data loading from candidate generation logic.
    def company_factory(record: dict[str, Any]) -> CompanySchema:
        """Transform raw dict to CompanySchema (already normalized in this example)."""
        if isinstance(record, CompanySchema):
            # Handle case where data is already normalized
            return record
        return CompanySchema(**record)

    # KEY PATTERN: Text Field Extractor
    # Instead of accepting text_fields=["name", "address"], VectorBlocker accepts
    # a function that extracts text from the schema. This is more flexible:
    # - Can combine fields with custom logic (e.g., " | " separator)
    # - Can apply transformations (e.g., lowercase, strip)
    # - Works with any schema type (not just dict-like)
    def company_text_extractor(company: CompanySchema) -> str:
        """Extract and combine text fields for embedding."""
        parts = [company.name]
        if company.address:
            parts.append(company.address)
        if company.website:
            parts.append(company.website)
        return " | ".join(parts)

    # Create VectorBlocker with dependency injection
    # All dependencies are passed as constructor arguments
    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=company_text_extractor,
        embedding_provider=embedding_provider,
        vector_index=vector_index,
        k_neighbors=k_neighbors,
    )

    # Convert CompanySchema objects to dicts for blocker.stream()
    # Note: VectorBlocker.stream() expects list[Any] (typically raw dicts)
    # The schema_factory will transform them back to CompanySchema internally
    entity_dicts = [entity.model_dump() for entity in entities]

    # Call blocker.stream() (NOT generate_candidates())
    # This returns an iterator of ERCandidate[CompanySchema]
    candidates = list(blocker.stream(entity_dicts))
    logger.info("Generated %d candidate pairs", len(candidates))

    # ==================================================================================
    # STEP 2: Score Pairs using LLMJudgeModule
    # ==================================================================================
    # KEY PATTERN: Module.forward() accepts Iterator
    # All Module implementations accept Iterator[ERCandidate[SchemaT]], not lists.
    # This enables streaming/lazy evaluation for large datasets.
    logger.info("Scoring pairs with LLMJudgeModule...")

    # Type annotation helps mypy verify schema consistency
    llm_judge: LLMJudgeModule[CompanySchema] = LLMJudgeModule(
        client=llm_client,  # Pre-configured LiteLLM client with Langfuse tracing
        model=azure_model,
        temperature=1.0,  # GPT-5 models only support temperature=1.0
    )

    # IMPORTANT: Module.forward() expects Iterator, not list
    # Convert list to iterator using iter()
    # forward() yields PairwiseJudgement with full provenance
    judgements = list(llm_judge.forward(iter(candidates)))
    logger.info("Scored %d pairs", len(judgements))

    # ==================================================================================
    # STEP 3: Form Clusters using Clusterer
    # ==================================================================================
    # KEY PATTERN: Clusterer returns list[set[str]]
    # The Clusterer uses graph connected components to form entity clusters.
    # It returns list[set[str]], NOT list[list[str]].
    logger.info("Forming clusters...")
    clusterer = Clusterer(threshold=cluster_threshold)

    # cluster() accepts Iterator[PairwiseJudgement] or list[PairwiseJudgement]
    # Returns list[set[str]] - each set is a cluster of entity IDs
    predicted_clusters = clusterer.cluster(judgements)
    logger.info("Formed %d clusters", len(predicted_clusters))

    return predicted_clusters, judgements


def main() -> None:
    """Main execution function."""
    # Load configuration from environment
    logger.info("Loading configuration...")
    settings = Settings()

    # Initialize LiteLLM client with Langfuse tracing
    logger.info("Initializing LiteLLM client with Langfuse tracing...")
    llm_client = create_llm_client(settings)

    # Azure OpenAI model (deployment name)
    azure_model = "azure/gpt-5-mini"

    # Load data
    data_dir = Path(__file__).parent / "data"
    logger.info("Loading data from %s", data_dir)

    entities = load_companies(data_dir / "companies.json")
    labels = load_labels(data_dir / "companies_labels.json")

    logger.info("Loaded %d entities with %d clusters", len(entities), len(labels["clusters"]))

    # Split into train/test
    train_entities, test_entities, train_clusters, test_clusters = train_test_split(
        entities, labels, test_size=0.2
    )

    # ==================================================================================
    # KEY PATTERN: Metrics Expect list[set[str]], but JSON Stores list[list[str]]
    # ==================================================================================
    # Ground truth clusters loaded from JSON are list[list[str]] (JSON doesn't have sets).
    # The Clusterer returns list[set[str]], and metrics functions expect list[set[str]].
    # Therefore, we convert ground truth: list[list[str]] → list[set[str]]
    train_clusters_sets = [set(cluster) for cluster in train_clusters]
    test_clusters_sets = [set(cluster) for cluster in test_clusters]

    # Define objective function for optimization
    def objective_fn(params: dict[str, Any]) -> dict[str, float]:
        """Objective function for blocker optimization.

        Args:
            params: Dictionary with 'embedding_model', 'k_neighbors', and 'cluster_threshold'

        Returns:
            Dict of all computed metrics. BlockerOptimizer will optimize the
            primary_metric specified in its constructor, and log all others
            to wandb as user attributes.
        """
        embedding_model = params["embedding_model"]
        k_neighbors = params["k_neighbors"]
        cluster_threshold = params["cluster_threshold"]

        logger.info(
            "Evaluating params: embedding_model=%s, k_neighbors=%d, cluster_threshold=%.2f",
            embedding_model,
            k_neighbors,
            cluster_threshold,
        )

        try:
            # Run pipeline on training data
            predicted_clusters, judgements = run_deduplication_pipeline(
                train_entities,
                embedding_model,
                k_neighbors,
                llm_client,
                azure_model,
                cluster_threshold,
            )

            # Calculate cost from judgements
            total_cost = sum(j.provenance.get("cost_usd", 0.0) for j in judgements)
            avg_cost_per_pair = total_cost / len(judgements) if len(judgements) > 0 else 0.0

            # Evaluate against ground truth (both are now list[set[str]])
            bcubed_metrics = calculate_bcubed_metrics(predicted_clusters, train_clusters_sets)
            pairwise_metrics = calculate_pairwise_metrics(predicted_clusters, train_clusters_sets)

            logger.info(
                "BCubed F1: %.4f | Pairwise F1: %.4f | Clusters: %d | Cost: $%.4f (avg $%.6f/pair)",
                bcubed_metrics["f1"],
                pairwise_metrics["f1"],
                len(predicted_clusters),
                total_cost,
                avg_cost_per_pair,
            )

            # Return all metrics as dict
            return {
                "bcubed_f1": bcubed_metrics["f1"],
                "bcubed_precision": bcubed_metrics["precision"],
                "bcubed_recall": bcubed_metrics["recall"],
                "pairwise_f1": pairwise_metrics["f1"],
                "pairwise_precision": pairwise_metrics["precision"],
                "pairwise_recall": pairwise_metrics["recall"],
                "pairwise_tp": float(pairwise_metrics["tp"]),
                "pairwise_fp": float(pairwise_metrics["fp"]),
                "pairwise_fn": float(pairwise_metrics["fn"]),
                "cost_usd": total_cost,
                "avg_cost_per_pair": avg_cost_per_pair,
                "num_clusters": float(len(predicted_clusters)),
                "num_pairs": float(len(judgements)),
            }

        except Exception as e:
            logger.error("Error in objective function: %s", e)
            return {
                "bcubed_f1": 0.0,
                "bcubed_precision": 0.0,
                "bcubed_recall": 0.0,
                "pairwise_f1": 0.0,
                "pairwise_precision": 0.0,
                "pairwise_recall": 0.0,
                "pairwise_tp": 0.0,
                "pairwise_fp": 0.0,
                "pairwise_fn": 0.0,
                "cost_usd": 0.0,
                "avg_cost_per_pair": 0.0,
                "num_clusters": 0.0,
                "num_pairs": 0.0,
            }

    # Define search space
    # Original search space (commented out for single-run verification)
    # search_space = {
    #     "embedding_model": [
    #         "all-MiniLM-L6-v2",  # Fast, 384 dim
    #         "all-mpnet-base-v2",  # Better quality, 768 dim
    #         "paraphrase-MiniLM-L3-v2",  # Fastest, 384 dim
    #     ],
    #     "k_neighbors": (1, 10),  # Range of k values
    #     "cluster_threshold": (0.3, 0.9),  # Clustering threshold for precision/recall tradeoff
    # }

    # Simple search space for single-run verification
    search_space = {
        "embedding_model": ["all-MiniLM-L6-v2"],  # Single model only
        "k_neighbors": (5, 5),  # Fixed value (same min/max)
        "cluster_threshold": (0.5, 0.5),  # Fixed value (same min/max)
    }

    # Initialize wandb for experiment tracking
    logger.info("Initializing wandb tracking...")
    wandb_kwargs = {
        "metric_name": "bcubed_f1",  # Name for primary metric column in wandb
        "wandb_kwargs": {  # This gets passed to wandb.init()
            "project": settings.wandb_project,
            "entity": settings.wandb_entity,
            "name": "company-dedup-blocker-opt",
            "tags": ["blocker-optimization", "company-dedup", "azure-openai"],
        },
        "as_multirun": False,  # Creates new runs for each trial
    }

    # Run optimization
    logger.info("Starting optimization with %d trials...", 1)
    optimizer = BlockerOptimizer(
        objective_fn=objective_fn,
        search_space=search_space,
        primary_metric="bcubed_f1",  # Which metric to optimize
        direction="maximize",
        n_trials=1,  # Single trial for verification
        wandb_kwargs=wandb_kwargs,
    )

    best_params = optimizer.optimize()
    logger.info("Optimization complete!")
    logger.info("Best parameters: %s", best_params)

    # Evaluate on test set with best parameters
    logger.info("Evaluating on test set with best parameters...")
    predicted_test_clusters, test_judgements = run_deduplication_pipeline(
        test_entities,
        best_params["embedding_model"],
        best_params["k_neighbors"],
        llm_client,
        azure_model,
        best_params["cluster_threshold"],
    )

    # Calculate final metrics
    test_bcubed = calculate_bcubed_metrics(predicted_test_clusters, test_clusters_sets)
    test_pairwise = calculate_pairwise_metrics(predicted_test_clusters, test_clusters_sets)
    test_total_cost = sum(j.provenance.get("cost_usd", 0.0) for j in test_judgements)
    test_avg_cost = test_total_cost / len(test_judgements) if len(test_judgements) > 0 else 0.0

    logger.info("=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    logger.info("Best embedding model: %s", best_params["embedding_model"])
    logger.info("Best k_neighbors: %d", best_params["k_neighbors"])
    logger.info("Best cluster_threshold: %.2f", best_params["cluster_threshold"])
    logger.info("-" * 80)
    logger.info("BCubed Metrics:")
    logger.info("  Precision: %.4f", test_bcubed["precision"])
    logger.info("  Recall: %.4f", test_bcubed["recall"])
    logger.info("  F1: %.4f", test_bcubed["f1"])
    logger.info("-" * 80)
    logger.info("Pairwise Metrics:")
    logger.info("  Precision: %.4f", test_pairwise["precision"])
    logger.info("  Recall: %.4f", test_pairwise["recall"])
    logger.info("  F1: %.4f", test_pairwise["f1"])
    logger.info(
        "  TP: %d, FP: %d, FN: %d", test_pairwise["tp"], test_pairwise["fp"], test_pairwise["fn"]
    )
    logger.info("-" * 80)
    logger.info("Operational Metrics:")
    logger.info(
        "  Predicted clusters: %d (gold: %d)", len(predicted_test_clusters), len(test_clusters_sets)
    )
    logger.info("  Total LLM calls: %d", len(test_judgements))
    logger.info("  Total cost: $%.4f", test_total_cost)
    logger.info("  Avg cost per pair: $%.6f", test_avg_cost)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
