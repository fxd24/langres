"""Entity Resolution Example: Company Deduplication with Blocker Optimization.

This example demonstrates:
1. VectorBlocker for candidate generation using sentence transformers
2. LLMJudgeModule with Azure OpenAI for pairwise matching
3. Clusterer for forming entity clusters
4. BlockerOptimizer with Optuna + wandb for hyperparameter tuning

The optimization process finds the best:
- Embedding model (from sentence-transformers library)
- k_neighbors (number of nearest neighbors to consider)

All components use langres.clients for centralized configuration:
- Settings loads environment variables (all optional)
- create_llm_client configures LiteLLM with optional Langfuse tracing
- create_wandb_tracker sets up experiment tracking

Environment variables required by this example:
    AZURE_API_ENDPOINT: Azure OpenAI endpoint URL (required for Azure OpenAI)
    AZURE_API_KEY: Azure OpenAI API key (required for Azure OpenAI)
    AZURE_API_VERSION: Azure OpenAI API version (optional, defaults to 2024-02-15-preview)
    WANDB_API_KEY: Weights & Biases API key (required for experiment tracking)
    LANGFUSE_PUBLIC_KEY: Langfuse public API key (required for LLM tracing)
    LANGFUSE_SECRET_KEY: Langfuse secret API key (required for LLM tracing)
    LANGFUSE_HOST: Langfuse host URL (optional, defaults to https://cloud.langfuse.com)

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
from langres.core.embeddings import SentenceTransformerEmbedding
from langres.core.metrics import bcubed_f1
from langres.core.modules.llm_judge import LLMJudgeModule
from langres.core.optimizers.blocker_optimizer import BlockerOptimizer
from langres.core.vector_index import FAISSVectorIndex

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
        return json.load(f)


def train_test_split(
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
) -> list[list[str]]:
    """Run the full deduplication pipeline.

    Args:
        entities: List of entities to deduplicate
        embedding_model: Name of sentence-transformer model
        k_neighbors: Number of nearest neighbors
        llm_client: LiteLLM client for LLM judge
        azure_model: Azure OpenAI model name (e.g., "azure/gpt-5-mini")

    Returns:
        Predicted clusters (list of lists of entity IDs)
    """
    # Step 1: Generate candidates using VectorBlocker
    logger.info("Generating candidates with VectorBlocker...")
    embedding_provider = SentenceTransformerEmbedding(model_name=embedding_model)
    vector_index = FAISSVectorIndex(dimension=embedding_provider.dimension)

    blocker = VectorBlocker(
        schema=CompanySchema,
        embedding_provider=embedding_provider,
        vector_index=vector_index,
        k_neighbors=k_neighbors,
        text_fields=["name", "address", "website"],
    )

    candidates = list(blocker.generate_candidates(entities))
    logger.info("Generated %d candidate pairs", len(candidates))

    # Step 2: Score pairs using LLMJudgeModule
    logger.info("Scoring pairs with LLMJudgeModule...")
    llm_judge = LLMJudgeModule(
        model=azure_model,
        api_key="dummy",  # API key read from environment by LiteLLM
        temperature=0.0,
        use_litellm=True,
        litellm_client=llm_client,
    )

    judgements = list(llm_judge.forward(candidates))
    logger.info("Scored %d pairs", len(judgements))

    # Step 3: Form clusters using Clusterer
    logger.info("Forming clusters...")
    clusterer = Clusterer(threshold=0.5)
    predicted_clusters = clusterer.cluster(judgements)
    logger.info("Formed %d clusters", len(predicted_clusters))

    return predicted_clusters


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

    # Define objective function for optimization
    def objective_fn(params: dict[str, Any]) -> float:
        """Objective function for blocker optimization.

        Args:
            params: Dictionary with 'embedding_model' and 'k_neighbors'

        Returns:
            BCubed F1 score on training set
        """
        embedding_model = params["embedding_model"]
        k_neighbors = params["k_neighbors"]

        logger.info(
            "Evaluating params: embedding_model=%s, k_neighbors=%d",
            embedding_model,
            k_neighbors,
        )

        try:
            # Run pipeline on training data
            predicted_clusters = run_deduplication_pipeline(
                train_entities, embedding_model, k_neighbors, llm_client, azure_model
            )

            # Evaluate against ground truth
            f1_score = bcubed_f1(predicted_clusters, train_clusters)

            logger.info("BCubed F1: %.4f", f1_score)
            return f1_score

        except Exception as e:
            logger.error("Error in objective function: %s", e)
            return 0.0

    # Define search space
    search_space = {
        "embedding_model": [
            "all-MiniLM-L6-v2",  # Fast, 384 dim
            "all-mpnet-base-v2",  # Better quality, 768 dim
            "paraphrase-MiniLM-L3-v2",  # Fastest, 384 dim
        ],
        "k_neighbors": (1, 10),  # Range of k values
    }

    # Initialize wandb for experiment tracking
    logger.info("Initializing wandb tracking...")
    wandb_kwargs = {
        "wandb_init_params": {
            "project": settings.wandb_project,
            "entity": settings.wandb_entity,
            "name": "company-dedup-blocker-opt",
            "tags": ["blocker-optimization", "company-dedup", "azure-openai"],
        },
        "as_multirun": False,
    }

    # Run optimization
    logger.info("Starting optimization with %d trials...", 10)
    optimizer = BlockerOptimizer(
        objective_fn=objective_fn,
        search_space=search_space,
        direction="maximize",
        n_trials=10,
        wandb_kwargs=wandb_kwargs,
    )

    best_params = optimizer.optimize()
    logger.info("Optimization complete!")
    logger.info("Best parameters: %s", best_params)

    # Evaluate on test set with best parameters
    logger.info("Evaluating on test set with best parameters...")
    predicted_test_clusters = run_deduplication_pipeline(
        test_entities,
        best_params["embedding_model"],
        best_params["k_neighbors"],
        llm_client,
        azure_model,
    )

    test_f1 = bcubed_f1(predicted_test_clusters, test_clusters)
    logger.info("=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    logger.info("Best embedding model: %s", best_params["embedding_model"])
    logger.info("Best k_neighbors: %d", best_params["k_neighbors"])
    logger.info("Test set BCubed F1: %.4f", test_f1)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
