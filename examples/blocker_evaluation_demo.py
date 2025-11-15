"""
Blocker evaluation demo - demonstrates the new evaluation architecture.

This example shows:
1. Creating mock candidates with similarity scores
2. Running comprehensive blocker evaluation
3. Accessing metrics via semantic categories
4. Understanding the new API structure

Note: For real evaluation with VectorBlocker or other similarity-based blockers,
see examples/blocker_evaluation_comprehensive.py
"""

import logging

from langres.core.models import CompanySchema, ERCandidate
from langres.core.analysis import evaluate_blocker_detailed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run blocker evaluation demo."""

    # =========================================================================
    # 1. Create Mock Data
    # =========================================================================
    logger.info("Creating mock data for demonstration...")

    # Mock entities
    entities = [
        CompanySchema(id="1", name="Acme Corp"),
        CompanySchema(id="2", name="Acme Corporation"),  # Duplicate of 1
        CompanySchema(id="3", name="Beta Inc"),
        CompanySchema(id="4", name="Beta Incorporated"),  # Duplicate of 3
        CompanySchema(id="5", name="Gamma LLC"),
    ]

    # Mock candidates with similarity scores
    # True matches: (1,2) and (3,4)
    # False candidates: all others
    blocker_name = "demo_blocker"
    candidates = [
        # True matches (high scores)
        ERCandidate(
            left=entities[0], right=entities[1], similarity_score=0.95, blocker_name=blocker_name
        ),  # 1-2
        ERCandidate(
            left=entities[2], right=entities[3], similarity_score=0.92, blocker_name=blocker_name
        ),  # 3-4
        # False candidates (lower scores)
        ERCandidate(
            left=entities[0], right=entities[2], similarity_score=0.35, blocker_name=blocker_name
        ),  # 1-3
        ERCandidate(
            left=entities[0], right=entities[3], similarity_score=0.32, blocker_name=blocker_name
        ),  # 1-4
        ERCandidate(
            left=entities[0], right=entities[4], similarity_score=0.28, blocker_name=blocker_name
        ),  # 1-5
        ERCandidate(
            left=entities[1], right=entities[2], similarity_score=0.38, blocker_name=blocker_name
        ),  # 2-3
        ERCandidate(
            left=entities[1], right=entities[3], similarity_score=0.34, blocker_name=blocker_name
        ),  # 2-4
        ERCandidate(
            left=entities[1], right=entities[4], similarity_score=0.30, blocker_name=blocker_name
        ),  # 2-5
        ERCandidate(
            left=entities[2], right=entities[4], similarity_score=0.25, blocker_name=blocker_name
        ),  # 3-5
        ERCandidate(
            left=entities[3], right=entities[4], similarity_score=0.22, blocker_name=blocker_name
        ),  # 4-5
    ]

    # Ground truth clusters
    gold_clusters = [
        {"1", "2"},  # Acme variants
        {"3", "4"},  # Beta variants
        {"5"},  # Gamma (singleton)
    ]

    logger.info(f"Created {len(candidates)} candidates from {len(entities)} entities")
    logger.info(f"Ground truth: {len(gold_clusters)} clusters (2 duplicates, 1 singleton)")

    # =========================================================================
    # 2. Run Comprehensive Evaluation
    # =========================================================================
    logger.info("\nRunning comprehensive blocker evaluation...")

    # NEW API: Single function call returns rich report
    report = evaluate_blocker_detailed(
        candidates=candidates,
        gold_clusters=gold_clusters,
        k_values=[1, 3, 5],  # Custom k values for recall curve
    )

    logger.info("Evaluation complete!")

    # =========================================================================
    # 3. Access Metrics via Semantic Categories
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("BLOCKER EVALUATION RESULTS")
    logger.info("=" * 70)

    # Quick access: report.category.metric
    logger.info("\n📊 Candidate Generation:")
    logger.info(f"  Recall:      {report.candidates.recall:.1%}")
    logger.info(f"  Precision:   {report.candidates.precision:.1%}")
    logger.info(f"  Total pairs: {report.candidates.total}")

    logger.info("\n🎯 Ranking Quality:")
    logger.info(f"  MAP: {report.ranking.map:.3f}")
    logger.info(f"  MRR: {report.ranking.mrr:.3f}")

    logger.info("\n📈 Score Distribution:")
    logger.info(f"  True median:  {report.scores.true_median:.3f}")
    logger.info(f"  False median: {report.scores.false_median:.3f}")
    logger.info(f"  Separation:   {report.scores.separation:.3f}")

    logger.info("\n🏆 Rank Distribution:")
    logger.info(f"  Median rank:   {report.rank_distribution.median:.0f}")
    logger.info(f"  % in top-3:    {report.rank_distribution.percent_in_top_5:.0f}%")

    logger.info("\n📉 Recall Curve:")
    for k, recall in zip(report.recall_curve.k_values, report.recall_curve.recall_values):
        logger.info(f"  k={k}: {recall:.1%}")

    # Find optimal k
    optimal_k = report.recall_curve.optimal_k(target_recall=0.95)
    logger.info(f"  Optimal k for 95% recall: {optimal_k}")

    # =========================================================================
    # 4. Demonstrate Pydantic Features
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PYDANTIC FEATURES")
    logger.info("=" * 70)

    # Serialization
    report_dict = report.model_dump()
    logger.info(f"\nSerialized to dict: {len(report_dict)} categories")
    logger.info(f"Categories: {list(report_dict.keys())}")

    # Immutability (frozen model)
    try:
        report.candidates.recall = 0.5  # type: ignore
        logger.error("ERROR: Report should be immutable!")
    except Exception as e:
        logger.info(f"✅ Immutability enforced: {type(e).__name__}")

    # Validation
    logger.info("✅ All metrics validated at construction (recall ∈ [0,1], counts ≥ 0)")

    # =========================================================================
    # 5. Visualization API (Optional)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("VISUALIZATION API")
    logger.info("=" * 70)

    logger.info("\nVisualization methods available:")
    logger.info("  report.plot_score_distribution()  # Score histogram")
    logger.info("  report.plot_rank_distribution()   # Rank histogram")
    logger.info("  report.plot_recall_curve()        # Recall@k curve")
    logger.info("  report.plot_all()                 # 4-panel summary")

    logger.info("\nNote: Requires matplotlib (pip install langres[viz])")

    # =========================================================================
    # 6. API Summary
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("API SUMMARY")
    logger.info("=" * 70)

    logger.info("""
New Blocker Evaluation Architecture:

1. **Single evaluate() method**:
   report = blocker.evaluate(candidates, gold_clusters)

2. **Semantic metric organization**:
   - report.candidates.*        (recall, precision, total, etc.)
   - report.ranking.*           (MAP, MRR, NDCG, etc.)
   - report.scores.*            (separation, medians, etc.)
   - report.rank_distribution.* (median, percent_in_top_k, etc.)
   - report.recall_curve        (k_values, recalls, optimal_k())

3. **Progressive disclosure**:
   - Quick: report.candidates.recall
   - Deep: report.scores.histogram

4. **Pydantic models**:
   - Validation, serialization, immutability
   - .model_dump(), .model_dump_json()

5. **Optional visualization**:
   - report.plot_all(), report.plot_score_distribution(), etc.
   - Requires: pip install langres[viz]
""")

    logger.info("\n" + "=" * 70)
    logger.info("DEMO COMPLETE!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
