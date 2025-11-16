"""Phase 2: Full Pipeline Evaluation (Blocker → LLMJudge → Clusterer).

This script evaluates the complete entity resolution pipeline and assesses POC success criteria.

POC Success Criteria:
- BCubed F1 ≥ 0.85
- Blocker Recall ≥ 0.95
- Separation ≥ 0.2

Pipeline Stages:
1. Blocking: VectorBlocker with k=50 (winner from Phase 1)
2. LLM Judgment: LLMJudgeModule with gpt-5-mini (async batch processing)
3. Clustering: Test thresholds [0.5, 0.6, 0.7, 0.8, 0.9]

Environment variables required:
    AZURE_API_ENDPOINT: Azure OpenAI endpoint URL
    AZURE_API_KEY: Azure OpenAI API key
    AZURE_API_VERSION: Azure OpenAI API version
    LANGFUSE_PUBLIC_KEY: Langfuse public API key (optional)
    LANGFUSE_SECRET_KEY: Langfuse secret API key (optional)

Performance:
    - First run: ~2-3 minutes (async LLM scoring with rate limiting)
    - Subsequent runs: ~30 seconds (cached embeddings and judgments)
    - Speedup: 12.5x faster than sequential LLM calls
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from tqdm import tqdm  # type: ignore[import-untyped]

from langres.clients import create_llm_client
from langres.clients.settings import Settings
from langres.core.blockers.vector import VectorBlocker
from langres.core.clusterer import Clusterer
from langres.core.embeddings import DiskCachedEmbedder, SentenceTransformerEmbedder
from langres.core.indexes.vector_index import FAISSIndex
from langres.core.metrics import pairs_from_clusters
from langres.core.models import PairwiseJudgement
from langres.core.modules.llm_judge import LLMJudgeModule
from langres.data import load_labeled_dedup_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FunderSchema(BaseModel):
    """Schema for funder organization entities."""

    id: str
    name: str


def evaluate_judgements(
    judgements: list[PairwiseJudgement],
    gold_clusters: list[set[str]],
    threshold: float = 0.7,
) -> dict[str, float]:
    """Evaluate LLM judgments against ground truth.

    Returns metrics: precision, recall, f1, accuracy, tp, fp, fn, tn, total_cost_usd
    """
    gold_pairs = pairs_from_clusters(gold_clusters)

    tp = fp = fn = tn = 0
    for j in judgements:
        pair = tuple(sorted([j.left_id, j.right_id]))
        is_gold = pair in gold_pairs
        is_pred = j.score >= threshold

        if is_pred and is_gold:
            tp += 1
        elif is_pred and not is_gold:
            fp += 1
        elif not is_pred and is_gold:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    total_cost = sum(j.provenance.get("cost_usd", 0) for j in judgements)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_cost_usd": total_cost,
    }


def save_judgments_cache(judgements: list[PairwiseJudgement], cache_path: Path) -> None:
    """Save LLM judgments to JSON cache file."""
    cache_data = [j.model_dump() for j in judgements]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache_data, indent=2))
    logger.info(f"Saved {len(judgements)} judgments to cache: {cache_path}")


def load_judgments_cache(cache_path: Path) -> list[PairwiseJudgement] | None:
    """Load LLM judgments from JSON cache file."""
    if not cache_path.exists():
        return None

    try:
        cache_data = json.loads(cache_path.read_text())
        judgements = [PairwiseJudgement(**j) for j in cache_data]
        logger.info(f"Loaded {len(judgements)} judgments from cache: {cache_path}")
        return judgements
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
        return None


def main() -> None:
    """Run Phase 2 full pipeline evaluation."""
    logger.info("=" * 80)
    logger.info("PHASE 2: FULL PIPELINE EVALUATION")
    logger.info("=" * 80)

    # Configuration
    WINNER_MODEL = "all-mpnet-base-v2"  # Winner from Phase 1
    K_NEIGHBORS = 50  # Optimal k from Phase 1
    CLUSTER_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
    LLM_MODEL = "azure/gpt-5-mini"  # Note: Correct Azure deployment name
    LLM_TEMPERATURE = 1.0
    CACHE_PATH = Path("tmp/llm_judgments_cache.json")
    DIAGNOSTICS_PATH = Path("tmp/diagnostics_phase2_full_pipeline.md")

    logger.info(f"Configuration:")
    logger.info(f"  Embedding model: {WINNER_MODEL}")
    logger.info(f"  k_neighbors: {K_NEIGHBORS}")
    logger.info(f"  LLM model: {LLM_MODEL}")
    logger.info(f"  LLM temperature: {LLM_TEMPERATURE}")
    logger.info(f"  Cluster thresholds: {CLUSTER_THRESHOLDS}")

    # =========================================================================
    # Load Data
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)

    data_dir = Path(__file__).parent / "data"
    dataset = load_labeled_dedup_data(
        data_dir=str(data_dir),
        entity_names_file="funder_names_with_ids.json",
        labeled_groups_file="funder_name_deduplicated_groups.json",
    )

    logger.info(f"Loaded {len(dataset.entity_names)} entities")
    logger.info(f"Ground truth: {len(dataset.labeled_groups)} entity clusters")
    logger.info(f"Unique entities: {dataset.num_unique_entities}")

    # Prepare data for blocker (as dicts)
    entities_data = [
        {"id": entity_id, "name": name} for entity_id, name in dataset.entity_names.items()
    ]

    # Convert labeled groups to sets
    gold_clusters = [set(group.entity_ids) for group in dataset.labeled_groups]

    # =========================================================================
    # STAGE 1: BLOCKING
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 1: BLOCKING")
    logger.info("=" * 80)

    # Initialize embedder with disk caching
    logger.info(f"Initializing DiskCachedEmbedder with model: {WINNER_MODEL}")
    base_embedder = SentenceTransformerEmbedder(model_name=WINNER_MODEL)
    cached_embedder = DiskCachedEmbedder(
        embedder=base_embedder,
        cache_dir=Path("tmp/embedding_cache"),
        namespace=WINNER_MODEL.replace("/", "_"),
        memory_cache_size=10_000,
    )

    # Initialize vector index
    logger.info("Building vector index...")
    vector_index = FAISSIndex(embedder=cached_embedder, metric="cosine")
    texts = [name for name in dataset.entity_names.values()]
    vector_index.create_index(texts)

    # Schema factory
    def funder_factory(record: dict[str, Any]) -> FunderSchema:
        return FunderSchema(id=record["id"], name=record["name"])

    # Text extractor
    def text_extractor(funder: FunderSchema) -> str:
        return funder.name

    # Create VectorBlocker
    logger.info(f"Creating VectorBlocker with k={K_NEIGHBORS}...")
    blocker = VectorBlocker(
        schema_factory=funder_factory,
        text_field_extractor=text_extractor,
        vector_index=vector_index,
        k_neighbors=K_NEIGHBORS,
    )

    # Generate candidates
    logger.info("Generating candidates...")
    start_time = time.time()
    candidates = list(blocker.stream(entities_data))
    blocker_runtime = time.time() - start_time
    logger.info(f"Generated {len(candidates)} candidate pairs in {blocker_runtime:.1f}s")

    # Evaluate blocker
    logger.info("Evaluating blocker...")
    report = blocker.evaluate(
        candidates=candidates,
        gold_clusters=gold_clusters,
        k_values=[1, 5, 10, 20, 50, 100],
    )

    logger.info("\nBlocker Metrics:")
    logger.info(f"  Recall:           {report.candidates.recall:.1%}")
    logger.info(f"  Precision:        {report.candidates.precision:.1%}")
    logger.info(f"  Separation:       {report.scores.separation:.3f}")
    logger.info(f"  MAP:              {report.ranking.map:.3f}")
    logger.info(f"  Total pairs:      {report.candidates.total:,}")

    # =========================================================================
    # STAGE 2: LLM JUDGMENT
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2: LLM JUDGMENT")
    logger.info("=" * 80)

    # Check for cached judgments
    judgements = load_judgments_cache(CACHE_PATH)

    if judgements is None:
        # Initialize LLM client
        logger.info("Initializing LLM client...")
        settings = Settings()
        llm_client = create_llm_client(settings, enable_langfuse=True)

        # Create LLMJudgeModule
        logger.info(f"Creating LLMJudgeModule (model={LLM_MODEL})...")
        llm_judge: LLMJudgeModule[FunderSchema] = LLMJudgeModule(
            client=llm_client,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
        )

        # Check if forward_async exists
        has_async = hasattr(llm_judge, "forward_async")
        logger.info(f"Async batch processing available: {has_async}")

        # Score candidates
        start_time = time.time()

        if has_async:
            # Use async batch processing (12.5x speedup)
            logger.info(
                "Scoring pairs with async LLM (max_concurrent=50, rate limits: 250 RPM, 250K TPM)..."
            )
            logger.info("This will take ~2-3 minutes for 1,741 entities...")

            # Run async forward_async with rate limiting
            judgements = asyncio.run(
                llm_judge.forward_async(
                    candidates=candidates,
                    max_concurrent=50,  # Process 50 requests in parallel
                    rpm_limit=250,  # Respect API rate limits
                    tpm_limit=250000,
                    max_retries=3,  # Retry on rate limit errors
                )
            )
            logger.info(
                f"✓ Async scoring complete: {len(judgements)} judgments in {time.time() - start_time:.1f}s"
            )
        else:
            # Fallback to sequential processing
            logger.info("Scoring pairs with LLM sequentially (this will take ~15-20 minutes)...")
            logger.info("Consider implementing forward_async() for 12.5x speedup")

            judgements = []
            with tqdm(total=len(candidates), desc="LLM scoring") as pbar:
                for judgement in llm_judge.forward(iter(candidates)):
                    judgements.append(judgement)
                    pbar.update(1)

        llm_runtime = time.time() - start_time
        logger.info(f"Scored {len(judgements)} pairs in {llm_runtime:.1f}s")

        # Save to cache
        save_judgments_cache(judgements, CACHE_PATH)
    else:
        logger.info("Using cached LLM judgments")
        llm_runtime = 0.0  # From cache

    # Analyze score distribution with inspect_scores
    logger.info("\nAnalyzing score distribution...")
    score_report = llm_judge.inspect_scores(judgements, sample_size=10)
    logger.info("\nScore Distribution:")
    logger.info(f"  Mean:    {score_report.score_distribution['mean']:.3f}")
    logger.info(f"  Median:  {score_report.score_distribution['median']:.3f}")
    logger.info(f"  Std:     {score_report.score_distribution['std']:.3f}")
    logger.info(f"  Min:     {score_report.score_distribution['min']:.3f}")
    logger.info(f"  Max:     {score_report.score_distribution['max']:.3f}")

    # Evaluate judgments at default threshold (0.7)
    logger.info("\nEvaluating LLM judgments at threshold=0.7...")
    llm_metrics = evaluate_judgements(judgements, gold_clusters, threshold=0.7)
    logger.info("\nLLM Judge Metrics (threshold=0.7):")
    logger.info(f"  Precision:  {llm_metrics['precision']:.1%}")
    logger.info(f"  Recall:     {llm_metrics['recall']:.1%}")
    logger.info(f"  F1:         {llm_metrics['f1']:.1%}")
    logger.info(f"  Accuracy:   {llm_metrics['accuracy']:.1%}")
    logger.info(f"  Total cost: ${llm_metrics['total_cost_usd']:.2f}")

    # =========================================================================
    # STAGE 3: CLUSTERING
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 3: CLUSTERING")
    logger.info("=" * 80)

    # Test multiple thresholds
    clustering_results = []
    best_threshold = None
    best_f1 = 0.0

    logger.info(f"Testing {len(CLUSTER_THRESHOLDS)} thresholds...")
    for threshold in CLUSTER_THRESHOLDS:
        logger.info(f"\nEvaluating threshold={threshold}...")

        # Create clusterer
        clusterer = Clusterer(threshold=threshold)

        # Generate clusters
        start_time = time.time()
        predicted_clusters = clusterer.cluster(judgements)
        cluster_runtime = time.time() - start_time

        logger.info(f"  Formed {len(predicted_clusters)} clusters in {cluster_runtime:.1f}s")

        # Evaluate clusters
        metrics = clusterer.evaluate(predicted_clusters, gold_clusters)
        bcubed = metrics["bcubed"]
        pairwise = metrics["pairwise"]

        logger.info(f"  BCubed - P: {bcubed['precision']:.1%}, R: {bcubed['recall']:.1%}, F1: {bcubed['f1']:.1%}")
        logger.info(f"  Pairwise - P: {pairwise['precision']:.1%}, R: {pairwise['recall']:.1%}, F1: {pairwise['f1']:.1%}")

        # Track best threshold
        if bcubed["f1"] > best_f1:
            best_f1 = bcubed["f1"]
            best_threshold = threshold

        clustering_results.append(
            {
                "threshold": threshold,
                "bcubed": bcubed,
                "pairwise": pairwise,
                "num_clusters": len(predicted_clusters),
                "runtime": cluster_runtime,
            }
        )

    logger.info(f"\nOptimal threshold: {best_threshold} (BCubed F1={best_f1:.1%})")

    # Inspect top clusters at optimal threshold
    logger.info("\nGenerating cluster inspection report at optimal threshold...")
    assert best_threshold is not None, "best_threshold should be set"
    optimal_clusterer = Clusterer(threshold=best_threshold)
    optimal_clusters = optimal_clusterer.cluster(judgements)
    entities_list = [FunderSchema(id=eid, name=name) for eid, name in dataset.entity_names.items()]
    cluster_report = optimal_clusterer.inspect_clusters(optimal_clusters, entities_list, sample_size=10)

    logger.info(f"\nCluster Statistics (threshold={best_threshold}):")
    logger.info(f"  Total clusters:   {cluster_report.total_clusters}")
    logger.info(f"  Singleton rate:   {cluster_report.singleton_rate:.1f}%")

    # =========================================================================
    # POC SUCCESS ASSESSMENT
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("POC SUCCESS ASSESSMENT")
    logger.info("=" * 80)

    # Get optimal metrics
    assert best_threshold is not None, "best_threshold should be set by this point"
    optimal_result = next(r for r in clustering_results if r["threshold"] == best_threshold)
    optimal_bcubed: dict[str, float] = optimal_result["bcubed"]  # type: ignore[assignment]
    optimal_runtime: float = optimal_result["runtime"]  # type: ignore[assignment]

    # Assess criteria
    bcubed_f1_pass = optimal_bcubed["f1"] >= 0.85
    blocker_recall_pass = report.candidates.recall >= 0.95
    separation_pass = report.scores.separation >= 0.2

    # Calculate cost per entity
    num_entities = len(dataset.entity_names)
    cost_per_entity = llm_metrics["total_cost_usd"] / num_entities
    cost_per_judgment = (
        llm_metrics["total_cost_usd"] / len(judgements) if len(judgements) > 0 else 0.0
    )

    logger.info("\nSuccess Criteria:")
    logger.info(f"  {'✅' if bcubed_f1_pass else '❌'} BCubed F1 ≥ 0.85: {'ACHIEVED' if bcubed_f1_pass else 'NOT MET'} ({optimal_bcubed['f1']:.3f})")
    logger.info(f"  {'✅' if blocker_recall_pass else '❌'} Blocker Recall ≥ 0.95: {'ACHIEVED' if blocker_recall_pass else 'NOT MET'} ({report.candidates.recall:.3f})")
    logger.info(f"  {'✅' if separation_pass else '❌'} Separation ≥ 0.2: {'ACHIEVED' if separation_pass else 'NOT MET'} ({report.scores.separation:.3f})")

    logger.info("\nCost Analysis:")
    logger.info(f"  Total cost:        ${llm_metrics['total_cost_usd']:.2f}")
    logger.info(f"  Cost per entity:   ${cost_per_entity:.4f}")
    logger.info(f"  Cost per judgment: ${cost_per_judgment:.5f}")

    logger.info("\nRuntime Analysis:")
    logger.info(f"  Blocker:      {blocker_runtime:.1f}s")
    logger.info(f"  LLM Judge:    {llm_runtime:.1f}s ({llm_runtime/60:.1f}m)")
    logger.info(f"  Clustering:   {optimal_runtime:.1f}s")
    logger.info(f"  Total:        {blocker_runtime + llm_runtime + optimal_runtime:.1f}s")

    all_pass = bcubed_f1_pass and blocker_recall_pass and separation_pass
    logger.info(f"\n{'✅ RECOMMENDATION' if all_pass else '❌ RECOMMENDATION'}:")
    if all_pass:
        logger.info(
            f"  Model {WINNER_MODEL} with k={K_NEIGHBORS}, threshold={best_threshold} meets POC requirements."
        )
    else:
        logger.info("  POC criteria not met. Consider:")
        if not bcubed_f1_pass:
            logger.info(f"    - Improve clustering (current F1={optimal_bcubed['f1']:.3f})")
        if not blocker_recall_pass:
            logger.info(f"    - Increase k_neighbors or try different embedding model (current recall={report.candidates.recall:.3f})")
        if not separation_pass:
            logger.info(f"    - Improve score separation (current={report.scores.separation:.3f})")

    # =========================================================================
    # SAVE COMPREHENSIVE DIAGNOSTICS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SAVING DIAGNOSTICS")
    logger.info("=" * 80)

    diagnostics = []
    diagnostics.append("# Phase 2: Full Pipeline Evaluation")
    diagnostics.append("")
    diagnostics.append("## Configuration")
    diagnostics.append(f"- **Embedding Model**: {WINNER_MODEL}")
    diagnostics.append(f"- **k_neighbors**: {K_NEIGHBORS}")
    diagnostics.append(f"- **LLM Model**: {LLM_MODEL}")
    diagnostics.append(f"- **LLM Temperature**: {LLM_TEMPERATURE}")
    diagnostics.append(f"- **Dataset**: {len(dataset.entity_names)} entities, {dataset.num_unique_entities} unique")
    diagnostics.append("")

    diagnostics.append("## Stage 1: Blocking")
    diagnostics.append("")
    diagnostics.append("| Metric          | Value   |")
    diagnostics.append("|-----------------|---------|")
    diagnostics.append(f"| Recall          | {report.candidates.recall:.1%}  |")
    diagnostics.append(f"| Precision       | {report.candidates.precision:.1%}  |")
    diagnostics.append(f"| Separation      | {report.scores.separation:.3f}   |")
    diagnostics.append(f"| MAP             | {report.ranking.map:.3f}   |")
    diagnostics.append(f"| Total pairs     | {report.candidates.total:,} |")
    diagnostics.append(f"| Runtime         | {blocker_runtime:.1f}s |")
    diagnostics.append("")

    diagnostics.append("## Stage 2: LLM Judgment")
    diagnostics.append("")
    diagnostics.append("### Score Distribution")
    diagnostics.append("")
    diagnostics.append("| Statistic | Value |")
    diagnostics.append("|-----------|-------|")
    diagnostics.append(f"| Mean      | {score_report.score_distribution['mean']:.3f} |")
    diagnostics.append(f"| Median    | {score_report.score_distribution['median']:.3f} |")
    diagnostics.append(f"| Std Dev   | {score_report.score_distribution['std']:.3f} |")
    diagnostics.append(f"| Min       | {score_report.score_distribution['min']:.3f} |")
    diagnostics.append(f"| Max       | {score_report.score_distribution['max']:.3f} |")
    diagnostics.append("")

    diagnostics.append("### Judgment Quality (threshold=0.7)")
    diagnostics.append("")
    diagnostics.append("| Metric     | Value   |")
    diagnostics.append("|------------|---------|")
    diagnostics.append(f"| Precision  | {llm_metrics['precision']:.1%}  |")
    diagnostics.append(f"| Recall     | {llm_metrics['recall']:.1%}  |")
    diagnostics.append(f"| F1         | {llm_metrics['f1']:.1%}  |")
    diagnostics.append(f"| Accuracy   | {llm_metrics['accuracy']:.1%}  |")
    diagnostics.append(f"| Total Cost | ${llm_metrics['total_cost_usd']:.2f} |")
    diagnostics.append(f"| Runtime    | {llm_runtime:.1f}s ({llm_runtime/60:.1f}m) |")
    diagnostics.append("")

    diagnostics.append("## Stage 3: Clustering")
    diagnostics.append("")
    diagnostics.append("### Threshold Comparison")
    diagnostics.append("")
    diagnostics.append("| Threshold | BCubed P | BCubed R | BCubed F1 | Pairwise P | Pairwise R | Pairwise F1 | Clusters |")
    diagnostics.append("|-----------|----------|----------|-----------|------------|------------|-------------|----------|")
    for result in clustering_results:
        result_bcubed: dict[str, float] = result["bcubed"]  # type: ignore[assignment]
        result_pairwise: dict[str, float] = result["pairwise"]  # type: ignore[assignment]
        result_threshold: float = result["threshold"]  # type: ignore[assignment]
        result_num_clusters: int = result["num_clusters"]  # type: ignore[assignment]
        marker = " ⭐" if result_threshold == best_threshold else ""
        diagnostics.append(
            f"| {result_threshold:.1f}{marker} | {result_bcubed['precision']:.1%} | {result_bcubed['recall']:.1%} | "
            f"{result_bcubed['f1']:.1%} | {result_pairwise['precision']:.1%} | {result_pairwise['recall']:.1%} | "
            f"{result_pairwise['f1']:.1%} | {result_num_clusters} |"
        )
    diagnostics.append("")

    diagnostics.append(f"**Optimal threshold**: {best_threshold} (BCubed F1={best_f1:.1%})")
    diagnostics.append("")

    diagnostics.append("### Cluster Statistics (Optimal Threshold)")
    diagnostics.append("")
    diagnostics.append(f"- **Total clusters**: {cluster_report.total_clusters}")
    diagnostics.append(f"- **Singleton rate**: {cluster_report.singleton_rate:.1f}%")
    diagnostics.append("")
    diagnostics.append("**Size distribution**:")
    for size_bucket, count in cluster_report.cluster_size_distribution.items():
        diagnostics.append(f"  - {size_bucket}: {count}")
    diagnostics.append("")

    diagnostics.append("## POC Success Assessment")
    diagnostics.append("")
    diagnostics.append("| Criterion | Target | Actual | Status |")
    diagnostics.append("|-----------|--------|--------|--------|")
    diagnostics.append(f"| BCubed F1 | ≥ 0.85 | {optimal_bcubed['f1']:.3f} | {'✅ PASS' if bcubed_f1_pass else '❌ FAIL'} |")
    diagnostics.append(f"| Blocker Recall | ≥ 0.95 | {report.candidates.recall:.3f} | {'✅ PASS' if blocker_recall_pass else '❌ FAIL'} |")
    diagnostics.append(f"| Separation | ≥ 0.2 | {report.scores.separation:.3f} | {'✅ PASS' if separation_pass else '❌ FAIL'} |")
    diagnostics.append("")

    diagnostics.append("## Cost & Performance Summary")
    diagnostics.append("")
    diagnostics.append(f"- **Total cost**: ${llm_metrics['total_cost_usd']:.2f}")
    diagnostics.append(f"- **Cost per entity**: ${cost_per_entity:.4f}")
    diagnostics.append(f"- **Cost per judgment**: ${cost_per_judgment:.5f}")
    diagnostics.append(f"- **Blocker runtime**: {blocker_runtime:.1f}s")
    diagnostics.append(f"- **LLM runtime**: {llm_runtime:.1f}s ({llm_runtime/60:.1f}m)")
    diagnostics.append(f"- **Clustering runtime**: {optimal_runtime:.1f}s")
    diagnostics.append(f"- **Total runtime**: {blocker_runtime + llm_runtime + optimal_runtime:.1f}s")
    diagnostics.append("")

    diagnostics.append("## Recommendation")
    diagnostics.append("")
    if all_pass:
        diagnostics.append(f"✅ **POC SUCCESS**: Model `{WINNER_MODEL}` with k={K_NEIGHBORS}, threshold={best_threshold} meets all POC requirements.")
        diagnostics.append("")
        diagnostics.append("The hybrid blocking + LLM judge + clustering approach is validated. Ready to proceed with:")
        diagnostics.append("- Building the full langres framework")
        diagnostics.append("- Implementing optimizer components")
        diagnostics.append("- Adding task-level APIs")
    else:
        diagnostics.append("❌ **POC NOT MET**: Some criteria failed. Recommendations:")
        diagnostics.append("")
        if not bcubed_f1_pass:
            diagnostics.append(f"- **Clustering quality low** (F1={optimal_bcubed['f1']:.3f}): Consider improving LLM prompts or using more sophisticated clustering")
        if not blocker_recall_pass:
            diagnostics.append(f"- **Blocker recall low** (recall={report.candidates.recall:.3f}): Increase k_neighbors or try different embedding model")
        if not separation_pass:
            diagnostics.append(f"- **Poor score separation** (sep={report.scores.separation:.3f}): Improve blocker quality or use different similarity metric")

    # Save diagnostics
    DIAGNOSTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_PATH.write_text("\n".join(diagnostics))
    logger.info(f"Saved comprehensive diagnostics to: {DIAGNOSTICS_PATH}")

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2 EVALUATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
