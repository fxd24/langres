"""
Comprehensive blocker evaluation example.

This example demonstrates the new blocker evaluation architecture:
1. Load labeled data (funder organizations dataset)
2. Create and run a blocker
3. Evaluate with comprehensive metrics
4. Access metrics via semantic categories
5. Create visualizations (optional - requires matplotlib)

Key features demonstrated:
- Semantic metric organization (candidates, ranking, scores, ranks, recall_curve)
- Progressive disclosure (quick access: report.candidates.recall)
- Rich analysis (distributions, rankings, trade-off curves)
- Optional visualization (plot_all(), individual plots)
"""

import logging
from pathlib import Path

from langres.core.blockers.vector import VectorBlocker
from langres.core.embeddings import SentenceTransformerEmbedder
from langres.core.indexes.vector_index import FAISSIndex
from langres.core.models import CompanySchema
from langres.data import load_labeled_dedup_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run comprehensive blocker evaluation example."""

    # =========================================================================
    # 1. Load Labeled Data
    # =========================================================================
    logger.info("Loading labeled funder organizations dataset...")

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

    # =========================================================================
    # 2. Create and Run Blocker
    # =========================================================================
    logger.info("Creating blocker...")

    # Using VectorBlocker for semantic similarity-based blocking
    # This generates similarity scores needed for comprehensive evaluation

    # Step 1: Initialize embedding provider and vector index
    embedder = SentenceTransformerEmbedder(
        model_name="all-MiniLM-L6-v2"  # Fast, lightweight model
    )
    vector_index = FAISSIndex(embedder=embedder, metric="cosine")

    # Step 2: Build the vector index (preprocessing)
    logger.info("Building vector index...")
    texts = [name for name in dataset.entity_names.values()]
    vector_index.create_index(texts)

    # Step 3: Schema factory to transform dicts to CompanySchema
    def company_factory(record: dict) -> CompanySchema:
        return CompanySchema(id=record["id"], name=record["name"])

    # Step 4: Text extractor: extract name field for embedding
    def text_extractor(company: CompanySchema) -> str:
        return company.name

    # Step 5: Create VectorBlocker with pre-built index
    blocker = VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=text_extractor,
        vector_index=vector_index,
        k_neighbors=20,  # Top-20 nearest neighbors per entity
    )

    logger.info("Generating candidates...")
    candidates = list(blocker.stream(entities_data))
    logger.info(f"Generated {len(candidates)} candidate pairs")

    # =========================================================================
    # 3. Evaluate with Comprehensive Metrics
    # =========================================================================
    logger.info("Evaluating blocker with comprehensive metrics...")

    # Convert LabeledGroup objects to sets of entity IDs
    gold_clusters = [set(group.entity_ids) for group in dataset.labeled_groups]

    # New API: Single evaluate() call returns rich report
    report = blocker.evaluate(
        candidates=candidates,
        gold_clusters=gold_clusters,
        k_values=[1, 5, 10, 20, 50, 100],  # Custom k values for recall curve
    )

    logger.info("Evaluation complete!")

    # =========================================================================
    # 4. Access Metrics via Semantic Categories
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("BLOCKER EVALUATION RESULTS")
    logger.info("=" * 70)

    # Candidate generation metrics
    logger.info("\n📊 Candidate Generation Metrics:")
    logger.info(f"  Recall:           {report.candidates.recall:.1%}")
    logger.info(f"  Precision:        {report.candidates.precision:.1%}")
    logger.info(f"  Total pairs:      {report.candidates.total:,}")
    logger.info(f"  Avg per entity:   {report.candidates.avg_per_entity:.1f}")
    logger.info(f"  Missed matches:   {report.candidates.missed_matches}")
    logger.info(f"  False positives:  {report.candidates.false_positives}")

    # Ranking quality metrics
    logger.info("\n🎯 Ranking Quality Metrics:")
    logger.info(f"  MAP (Mean Avg Precision): {report.ranking.map:.3f}")
    logger.info(f"  MRR (Mean Recip Rank):    {report.ranking.mrr:.3f}")
    logger.info(f"  NDCG@10:                  {report.ranking.ndcg_at_10:.3f}")
    logger.info(f"  NDCG@20:                  {report.ranking.ndcg_at_20:.3f}")
    logger.info(f"  Recall@5:                 {report.ranking.recall_at_5:.1%}")
    logger.info(f"  Recall@10:                {report.ranking.recall_at_10:.1%}")
    logger.info(f"  Recall@20:                {report.ranking.recall_at_20:.1%}")

    # Score distribution analysis
    logger.info("\n📈 Score Distribution:")
    logger.info(f"  True match scores:")
    logger.info(f"    Mean:    {report.scores.true_mean:.3f}")
    logger.info(f"    Median:  {report.scores.true_median:.3f}")
    logger.info(f"    Std:     {report.scores.true_std:.3f}")
    logger.info(f"  False candidate scores:")
    logger.info(f"    Mean:    {report.scores.false_mean:.3f}")
    logger.info(f"    Median:  {report.scores.false_median:.3f}")
    logger.info(f"    Std:     {report.scores.false_std:.3f}")
    logger.info(f"  Separation:        {report.scores.separation:.3f}")
    logger.info(f"  Overlap fraction:  {report.scores.overlap_fraction:.1%}")

    # Rank distribution analysis
    logger.info("\n🏆 Rank Distribution:")
    logger.info(f"  Median rank:       {report.rank_distribution.median:.1f}")
    logger.info(f"  95th percentile:   {report.rank_distribution.percentile_95:.1f}")
    logger.info(f"  % in top-5:        {report.rank_distribution.percent_in_top_5:.1f}%")
    logger.info(f"  % in top-10:       {report.rank_distribution.percent_in_top_10:.1f}%")
    logger.info(f"  % in top-20:       {report.rank_distribution.percent_in_top_20:.1f}%")

    # Recall curve analysis
    logger.info("\n📉 Recall Curve:")
    for k, recall, cost in zip(
        report.recall_curve.k_values,
        report.recall_curve.recall_values,
        report.recall_curve.avg_pairs_values,
    ):
        logger.info(f"  k={k:3d}: Recall={recall:.1%}, Avg pairs/entity={cost:.1f}")

    # Find optimal k
    optimal_k = report.recall_curve.optimal_k(target_recall=0.95)
    logger.info(f"\n✅ Optimal k for 95% recall: {optimal_k}")

    # =========================================================================
    # 5. Markdown Report
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("MARKDOWN REPORT")
    logger.info("=" * 70)

    markdown = report.to_markdown()
    logger.info(f"\n{markdown}")

    # Save to file
    report_path = Path(__file__).parent / "blocker_evaluation_report.md"
    report_path.write_text(markdown)
    logger.info(f"\n📄 Saved markdown report to: {report_path}")

    # =========================================================================
    # 6. Serialization
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("SERIALIZATION")
    logger.info("=" * 70)

    # Pydantic serialization
    report_dict = report.model_dump()
    logger.info(f"Serialized to dict: {len(report_dict)} top-level keys")
    logger.info(f"Keys: {list(report_dict.keys())}")

    # Save JSON
    import json

    json_path = Path(__file__).parent / "blocker_evaluation_report.json"
    json_path.write_text(json.dumps(report_dict, indent=2))
    logger.info(f"📄 Saved JSON report to: {json_path}")

    # =========================================================================
    # 7. Visualizations (Optional - requires matplotlib)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("VISUALIZATIONS")
    logger.info("=" * 70)

    try:
        import matplotlib

        # Use non-interactive backend for examples
        matplotlib.use("Agg")

        logger.info("Creating visualizations...")

        # Individual plots
        logger.info("  - Score distribution...")
        report.plot_score_distribution()

        logger.info("  - Rank distribution...")
        report.plot_rank_distribution()

        logger.info("  - Recall curve...")
        report.plot_recall_curve()

        # Comprehensive 4-panel summary
        logger.info("  - 4-panel summary...")
        viz_path = Path(__file__).parent / "blocker_evaluation_summary.png"
        report.plot_all(save_path=str(viz_path))

        logger.info(f"📊 Saved visualization to: {viz_path}")

    except ImportError:
        logger.warning(
            "Matplotlib not installed. Skipping visualizations.\n"
            "Install with: pip install langres[viz]"
        )

    # =========================================================================
    # 8. Interpretation Guide
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("INTERPRETATION GUIDE")
    logger.info("=" * 70)

    logger.info("\n📚 How to interpret these metrics:")

    # Candidate recall interpretation
    if report.candidates.recall >= 0.95:
        logger.info("  ✅ Excellent recall: Blocker finds >95% of true matches")
    elif report.candidates.recall >= 0.85:
        logger.info("  ⚠️  Good recall: Blocker finds >85% of true matches")
    else:
        logger.info("  ❌ Poor recall: Blocker misses too many true matches")

    # Score separation interpretation
    if report.scores.separation >= 0.3:
        logger.info("  ✅ Good separation: Easy to set threshold between true/false")
    elif report.scores.separation >= 0.1:
        logger.info("  ⚠️  Moderate separation: Threshold tuning may help")
    else:
        logger.info("  ❌ Poor separation: Consider different embedder/index")

    # Ranking interpretation
    if report.rank_distribution.median <= 5:
        logger.info("  ✅ Excellent ranking: True matches typically in top-5")
    elif report.rank_distribution.median <= 20:
        logger.info("  ⚠️  Good ranking: True matches typically in top-20")
    else:
        logger.info("  ❌ Poor ranking: Need larger k or better index")

    # Recall curve interpretation
    if optimal_k <= 10:
        logger.info(f"  ✅ Efficient: Can use small k={optimal_k} for 95% recall")
    elif optimal_k <= 50:
        logger.info(f"  ⚠️  Moderate: Need k={optimal_k} for 95% recall")
    else:
        logger.info(f"  ❌ Expensive: Need large k={optimal_k} for 95% recall")

    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE COMPLETE!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
