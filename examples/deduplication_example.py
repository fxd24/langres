"""Entity Deduplication Example with Full Pipeline.

This example demonstrates end-to-end entity resolution using the langres library
on a real-world dataset of Swiss funder organization names.

Dataset: 1,741 organization names with 1,305 unique entities
Source: Manually curated and verified against Zefix business registry

Key features:
1. **Realistic labeled data**: Real-world ground truth with quality verification
2. **Stratified train/test split**: Preserves cluster size distribution (singletons, pairs, etc.)
3. **Full pipeline**: VectorBlocker → LLMJudge → Clusterer with optimization
4. **Comprehensive evaluation**: Blocking recall, LLM judge F1, BCubed F1

This example shows how to:
- Load labeled deduplication data with custom file names
- Create stratified train/test splits
- Run blocking, scoring, and clustering
- Evaluate performance with multiple metrics

Environment variables required:
    AZURE_API_ENDPOINT: Azure OpenAI endpoint URL
    AZURE_API_KEY: Azure OpenAI API key
    AZURE_API_VERSION: Azure OpenAI API version (optional)
    WANDB_API_KEY: Weights & Biases API key
    LANGFUSE_PUBLIC_KEY: Langfuse public API key (optional, for LLM tracing)
    LANGFUSE_SECRET_KEY: Langfuse secret API key (optional)
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
from pydantic import BaseModel, Field

from langres.clients import create_llm_client
from langres.clients.settings import Settings
from langres.core import PipelineDebugger
from langres.core.blockers.vector import VectorBlocker
from langres.core.clusterer import Clusterer
from langres.core.embeddings import SentenceTransformerEmbedder
from langres.core.metrics import (
    calculate_bcubed_metrics,
    calculate_pairwise_metrics,
    evaluate_blocking,
)
from langres.core.modules.llm_judge import LLMJudgeModule
from langres.core.optimizers.blocker_optimizer import BlockerOptimizer
from langres.core.vector_index import FAISSIndex
from langres.data import load_labeled_dedup_data, stratified_dedup_split

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OrganizationSchema(BaseModel):
    """Simple organization entity schema."""

    id: str = Field(description="Unique entity identifier")
    name: str = Field(description="Organization name")


def run_deduplication_pipeline(
    entities: list[dict[str, str]],
    embedding_model: str,
    k_neighbors: int,
    llm_client: Any,
    azure_model: str,
    cluster_threshold: float = 0.5,
    enable_llm: bool = True,
) -> tuple[list[set[str]], list[Any], list[Any]]:
    """Run the full deduplication pipeline.

    Args:
        entities: List of entity dicts with "id" and "name" keys
        embedding_model: Name of sentence-transformer model
        k_neighbors: Number of nearest neighbors for blocking
        llm_client: LiteLLM client for LLM judge
        azure_model: Azure OpenAI model name
        cluster_threshold: Threshold for clustering
        enable_llm: Whether to use LLM judge (False = blocking assessment only)

    Returns:
        Tuple of (predicted_clusters, judgements, candidates)
    """
    # ==================================================================================
    # STEP 1: Generate Candidates using VectorBlocker
    # ==================================================================================
    logger.info(
        "Generating candidates with VectorBlocker (model=%s, k=%d)...", embedding_model, k_neighbors
    )

    # Initialize embedding provider
    embedding_provider = SentenceTransformerEmbedder(model_name=embedding_model)

    # Initialize vector index
    vector_index = FAISSIndex(metric="L2")

    # Schema factory: dict → OrganizationSchema
    def org_factory(record: dict[str, Any]) -> OrganizationSchema:
        """Transform dict to OrganizationSchema."""
        if isinstance(record, OrganizationSchema):
            return record
        return OrganizationSchema(**record)

    # Text extractor: just use the name field
    def org_text_extractor(org: OrganizationSchema) -> str:
        """Extract text for embedding."""
        return org.name

    # Create VectorBlocker
    blocker = VectorBlocker(
        schema_factory=org_factory,
        text_field_extractor=org_text_extractor,
        embedding_provider=embedding_provider,
        vector_index=vector_index,
        k_neighbors=k_neighbors,
    )

    # Generate candidates
    candidates = list(blocker.stream(entities))
    logger.info("Generated %d candidate pairs", len(candidates))

    # If not using LLM, return early (for blocking assessment)
    if not enable_llm:
        return [], [], candidates

    # ==================================================================================
    # STEP 2: Score Pairs using LLMJudgeModule
    # ==================================================================================
    logger.info("Scoring pairs with LLMJudgeModule...")

    llm_judge: LLMJudgeModule[OrganizationSchema] = LLMJudgeModule(
        client=llm_client,
        model=azure_model,
        temperature=1.0,  # GPT-5 models only support temperature=1.0
    )

    # Score candidates
    judgements = list(llm_judge.forward(iter(candidates)))
    logger.info("Scored %d pairs", len(judgements))

    # ==================================================================================
    # STEP 3: Form Clusters using Clusterer
    # ==================================================================================
    logger.info("Forming clusters (threshold=%.2f)...", cluster_threshold)
    clusterer = Clusterer(threshold=cluster_threshold)

    predicted_clusters = clusterer.cluster(judgements)
    logger.info("Formed %d clusters", len(predicted_clusters))

    return predicted_clusters, judgements, candidates


def log_blocking_evaluation(stats: Any) -> None:
    """Log blocking evaluation results in formatted output.

    Args:
        stats: CandidateStats from evaluate_blocking()
    """
    logger.info("=" * 80)
    logger.info("BLOCKING EVALUATION")
    logger.info("=" * 80)
    total_gold = (
        stats.total_candidates - stats.false_positive_candidates_count + stats.missed_matches_count
    )
    logger.info("Ground truth pairs: %d", total_gold)
    logger.info("Candidate pairs generated: %d", stats.total_candidates)
    logger.info(
        "True positives: %d", stats.total_candidates - stats.false_positive_candidates_count
    )
    logger.info("False negatives (missed): %d", stats.missed_matches_count)
    logger.info("False positives (extra): %d", stats.false_positive_candidates_count)
    logger.info(
        "Recall: %.2f%% (captured %.0f%% of true duplicates)",
        stats.candidate_recall * 100,
        stats.candidate_recall * 100,
    )
    logger.info("Precision: %.2f%%", stats.candidate_precision * 100)
    f1 = (
        2
        * (stats.candidate_precision * stats.candidate_recall)
        / (stats.candidate_precision + stats.candidate_recall)
        if (stats.candidate_precision + stats.candidate_recall) > 0
        else 0.0
    )
    logger.info("F1: %.2f%%", f1 * 100)
    logger.info("=" * 80)


def main() -> None:
    """Main execution function."""
    # Load configuration
    logger.info("Loading configuration...")
    settings = Settings()

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting deduplication run: {timestamp}")

    # Initialize LiteLLM client (only if we'll use LLM judge)
    logger.info("Initializing LiteLLM client...")
    llm_client = create_llm_client(settings)
    azure_model = "azure/gpt-5-mini"

    # ==================================================================================
    # Load labeled deduplication data
    # ==================================================================================
    # This example uses a dataset with custom file names stored in tmp/dedup_iteration_05/
    # For your own data, prepare files following the langres schema:
    #   entity_names.json: {"names": {"1": "Name 1", "2": "Name 2", ...}}
    #   labeled_groups.json: {"groups": [{"canonical_name": "...", "entity_ids": [...], ...}]}
    logger.info("Loading labeled deduplication data...")
    dataset = load_labeled_dedup_data(
        data_dir="tmp/dedup_iteration_05",
        entity_names_file="all_names_with_ids.json",  # Custom file name for this dataset
        labeled_groups_file="deduplicated_groups.json",  # Custom file name for this dataset
    )

    logger.info(
        "Loaded %d entity names, %d unique entities",
        len(dataset.entity_names),
        dataset.num_unique_entities,
    )
    logger.info("  %d labeled groups", len(dataset.labeled_groups))

    # ==================================================================================
    # Stratified train/test split
    # ==================================================================================
    logger.info("Performing stratified train/test split...")
    train_records, test_records, train_clusters, test_clusters = stratified_dedup_split(
        dataset,
        test_size=0.2,
        random_state=42,
    )

    logger.info(
        "Split complete: %d train entities (%d clusters), %d test entities (%d clusters)",
        len(train_records),
        len(train_clusters),
        len(test_records),
        len(test_clusters),
    )

    # ==================================================================================
    # BLOCKING ASSESSMENT (Stage 1: Capture rate)
    # ==================================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 1: BLOCKING ASSESSMENT")
    logger.info("=" * 80)
    logger.info("Testing if VectorBlocker captures true positive pairs...")

    # Run blocking only (no LLM judge)
    _, _, train_candidates = run_deduplication_pipeline(
        entities=train_records,
        embedding_model="all-MiniLM-L6-v2",  # Fast model for initial test
        k_neighbors=5,
        llm_client=llm_client,
        azure_model=azure_model,
        cluster_threshold=0.5,
        enable_llm=False,  # Skip LLM for blocking assessment
    )

    # Assess blocking recall
    # Evaluate blocking performance
    blocking_stats = evaluate_blocking(train_candidates, train_clusters)
    log_blocking_evaluation(blocking_stats)

    # Check if blocking recall is acceptable
    if blocking_stats.candidate_recall < 0.90:
        logger.warning(
            "⚠️  Blocking recall is low (%.1f%%). Consider increasing k_neighbors or trying different embedding model.",
            blocking_stats.candidate_recall * 100,
        )
        logger.warning("Missed true matches: %d", blocking_stats.missed_matches_count)
        return  # Exit early if blocking is insufficient
    else:
        logger.info("✓ Blocking recall is good (%.1f%%)", blocking_stats.candidate_recall * 100)

    # ==================================================================================
    # FULL PIPELINE ASSESSMENT (Stage 2: End-to-end evaluation)
    # ==================================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2: FULL PIPELINE ASSESSMENT (with LLM Judge)")
    logger.info("=" * 80)

    # For POC, we'll run a single trial with fixed parameters
    # (Full optimization can be enabled by changing n_trials and search_space)
    logger.info("Running single trial for assessment...")

    # Define objective function
    def objective_fn(trial: optuna.Trial, params: dict[str, Any]) -> dict[str, float]:
        """Objective function for optimization."""
        embedding_model = params["embedding_model"]
        k_neighbors = params["k_neighbors"]
        cluster_threshold = params["cluster_threshold"]

        logger.info(
            "Evaluating params: model=%s, k=%d, threshold=%.2f",
            embedding_model,
            k_neighbors,
            cluster_threshold,
        )

        # Run full pipeline
        predicted_clusters, judgements, candidates = run_deduplication_pipeline(
            train_records,
            embedding_model,
            k_neighbors,
            llm_client,
            azure_model,
            cluster_threshold,
            enable_llm=True,
        )

        # Calculate cost
        total_cost = sum(j.provenance.get("cost_usd", 0.0) for j in judgements)
        avg_cost_per_pair = total_cost / len(judgements) if judgements else 0.0

        # Evaluate metrics
        bcubed_metrics = calculate_bcubed_metrics(predicted_clusters, train_clusters)
        pairwise_metrics = calculate_pairwise_metrics(predicted_clusters, train_clusters)

        logger.info(
            "BCubed F1: %.4f | Pairwise F1: %.4f | Clusters: %d | Cost: $%.4f",
            bcubed_metrics["f1"],
            pairwise_metrics["f1"],
            len(predicted_clusters),
            total_cost,
        )

        # Generate debug artifacts
        DEBUG_ARTIFACTS = os.getenv("DEBUG_ARTIFACTS", "true").lower() == "true"
        if DEBUG_ARTIFACTS:
            try:
                import wandb

                logger.info(f"Generating debug artifacts for trial {trial.number}...")

                # Create entities list for debugger
                train_entities = [OrganizationSchema(**r) for r in train_records]

                debugger: PipelineDebugger[OrganizationSchema] = PipelineDebugger(
                    ground_truth_clusters=train_clusters,
                    sample_size=10,
                )

                debugger.analyze_candidates(candidates, train_entities)
                debugger.analyze_scores(judgements)
                debugger.analyze_clusters(predicted_clusters)

                # Save locally
                output_dir = Path(f"tmp/debug_reports/{timestamp}")
                output_dir.mkdir(parents=True, exist_ok=True)

                base_name = f"{timestamp}_trial_{trial.number:03d}_debug_report"
                json_path = output_dir / f"{base_name}.json"
                md_path = output_dir / f"{base_name}.md"

                debugger.save_report(json_path, format="json")
                debugger.save_report(md_path, format="markdown")

                logger.info(f"Saved debug reports: {json_path}")

                # Log to wandb if available
                if wandb.run is not None:
                    artifact = wandb.Artifact(
                        name=f"trial-{trial.number:03d}-debug",
                        type="debug-report",
                        description=f"Pipeline debug analysis for trial {trial.number}",
                    )
                    artifact.add_file(str(json_path))
                    artifact.add_file(str(md_path))
                    wandb.log_artifact(artifact)

            except Exception as e:
                logger.warning(f"Failed to create debug artifacts: {e}")

        # Return metrics
        return {
            "bcubed_f1": bcubed_metrics["f1"],
            "bcubed_precision": bcubed_metrics["precision"],
            "bcubed_recall": bcubed_metrics["recall"],
            "pairwise_f1": pairwise_metrics["f1"],
            "pairwise_precision": pairwise_metrics["precision"],
            "pairwise_recall": pairwise_metrics["recall"],
            "cost_usd": total_cost,
            "avg_cost_per_pair": avg_cost_per_pair,
            "num_clusters": float(len(predicted_clusters)),
            "num_pairs": float(len(judgements)),
        }

    # Simple search space for POC (single values = no optimization)
    search_space = {
        "embedding_model": ["all-MiniLM-L6-v2"],
        "k_neighbors": (5, 5),  # Fixed value
        "cluster_threshold": (0.5, 0.5),  # Fixed value
    }

    # Initialize wandb
    wandb_kwargs = {
        "metric_name": "bcubed_f1",
        "wandb_kwargs": {
            "project": settings.wandb_project,
            "entity": settings.wandb_entity,
            "name": f"iteration05-{timestamp}",
            "tags": ["iteration-05", "swiss-funders", "poc-assessment"],
        },
        "as_multirun": True,
    }

    # Run optimization (single trial for POC)
    logger.info("Running pipeline...")
    optimizer = BlockerOptimizer(
        objective_fn=objective_fn,
        search_space=search_space,
        primary_metric="bcubed_f1",
        direction="maximize",
        n_trials=1,
        wandb_kwargs=wandb_kwargs,
    )

    best_params = optimizer.optimize()
    logger.info("Pipeline complete! Best parameters: %s", best_params)

    # ==================================================================================
    # TEST SET EVALUATION
    # ==================================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 80)

    predicted_test_clusters, test_judgements, test_candidates = run_deduplication_pipeline(
        test_records,
        best_params["embedding_model"],
        best_params["k_neighbors"],
        llm_client,
        azure_model,
        best_params["cluster_threshold"],
        enable_llm=True,
    )

    # Assess test blocking
    # Evaluate test set blocking performance
    test_blocking_stats = evaluate_blocking(test_candidates, test_clusters)
    log_blocking_evaluation(test_blocking_stats)

    # Calculate test metrics
    test_bcubed = calculate_bcubed_metrics(predicted_test_clusters, test_clusters)
    test_pairwise = calculate_pairwise_metrics(predicted_test_clusters, test_clusters)
    test_total_cost = sum(j.provenance.get("cost_usd", 0.0) for j in test_judgements)

    # Generate test debug report
    logger.info("Generating test evaluation debug report...")
    import wandb

    test_entities = [OrganizationSchema(**r) for r in test_records]
    test_debugger: PipelineDebugger[OrganizationSchema] = PipelineDebugger(
        ground_truth_clusters=test_clusters,
        sample_size=10,
    )

    test_debugger.analyze_candidates(test_candidates, test_entities)
    test_debugger.analyze_scores(test_judgements)
    test_debugger.analyze_clusters(predicted_test_clusters)

    # Save test debug reports
    output_dir = Path(f"tmp/debug_reports/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_json_path = output_dir / f"{timestamp}_test_evaluation_debug.json"
    test_md_path = output_dir / f"{timestamp}_test_evaluation_debug.md"

    test_debugger.save_report(test_json_path, format="json")
    test_debugger.save_report(test_md_path, format="markdown")

    logger.info(f"Test debug reports saved: {test_json_path}")

    # Log to wandb
    if settings.wandb_api_key:
        with wandb.init(
            project=settings.wandb_project,
            entity=settings.wandb_entity,
            name=f"{timestamp}-test-eval",
            tags=["test-evaluation", "iteration-05"],
            reinit=True,
        ) as test_run:
            test_run.log(
                {
                    "bcubed_f1": test_bcubed["f1"],
                    "bcubed_precision": test_bcubed["precision"],
                    "bcubed_recall": test_bcubed["recall"],
                    "pairwise_f1": test_pairwise["f1"],
                    "blocking_recall": test_blocking_stats.candidate_recall,
                    "cost_usd": test_total_cost,
                }
            )

            artifact = wandb.Artifact(
                name="test-evaluation-debug",
                type="debug-report",
                description="Test set evaluation analysis",
            )
            artifact.add_file(str(test_json_path))
            artifact.add_file(str(test_md_path))
            test_run.log_artifact(artifact)

    # ==================================================================================
    # FINAL SUMMARY
    # ==================================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info("Dataset: Iteration 05 (Swiss Funders)")
    logger.info("  Total entities: %d", len(dataset.entity_names))
    logger.info("  Ground truth unique: %d", dataset.num_unique_entities)
    logger.info("  Train: %d entities, %d clusters", len(train_records), len(train_clusters))
    logger.info("  Test: %d entities, %d clusters", len(test_records), len(test_clusters))
    logger.info("-" * 80)
    logger.info("Blocking Performance (Test Set):")
    tp_test = (
        test_blocking_stats.total_candidates - test_blocking_stats.false_positive_candidates_count
    )
    total_gold_test = tp_test + test_blocking_stats.missed_matches_count
    logger.info(
        "  Recall: %.2f%% (%d/%d true pairs captured)",
        test_blocking_stats.candidate_recall * 100,
        tp_test,
        total_gold_test,
    )
    logger.info("  Precision: %.2f%%", test_blocking_stats.candidate_precision * 100)
    logger.info("-" * 80)
    logger.info("End-to-End Performance (Test Set):")
    logger.info("  BCubed F1: %.4f", test_bcubed["f1"])
    logger.info("  BCubed Precision: %.4f", test_bcubed["precision"])
    logger.info("  BCubed Recall: %.4f", test_bcubed["recall"])
    logger.info("  Pairwise F1: %.4f", test_pairwise["f1"])
    logger.info("-" * 80)
    logger.info("POC Success Criterion: BCubed F1 ≥ 0.85")
    if test_bcubed["f1"] >= 0.85:
        logger.info("  ✓ PASSED (F1=%.4f)", test_bcubed["f1"])
    else:
        logger.info("  ✗ NOT MET (F1=%.4f < 0.85)", test_bcubed["f1"])
    logger.info("-" * 80)
    logger.info("Operational Metrics:")
    logger.info(
        "  Predicted clusters: %d (gold: %d)", len(predicted_test_clusters), len(test_clusters)
    )
    logger.info("  Total LLM calls: %d", len(test_judgements))
    logger.info("  Total cost: $%.4f", test_total_cost)
    logger.info("=" * 80)
    logger.info(f"All reports saved to: tmp/debug_reports/{timestamp}/")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
