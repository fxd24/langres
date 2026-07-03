"""
Example: Debugging an entity resolution pipeline with PipelineDebugger.

This example demonstrates how to use PipelineDebugger to analyze the quality
of candidate generation, scoring, and clustering in an entity resolution pipeline.
"""

from langres.core import (
    CompanySchema,
    ERCandidate,
    PairwiseJudgement,
    PipelineDebugger,
)


def main() -> None:
    """Demonstrate pipeline debugging capabilities."""

    # Ground truth: we know e1, e2, e3 are the same company, and e4, e5 are another
    ground_truth_clusters = [
        {"e1", "e2", "e3"},  # Apple Inc variants
        {"e4", "e5"},  # Google variants
        {"e6"},  # Microsoft (singleton)
    ]

    # Create test entities
    entities = [
        CompanySchema(id="e1", name="Apple Inc", website="apple.com"),
        CompanySchema(id="e2", name="Apple Computer", website="apple.com"),
        CompanySchema(id="e3", name="Apple Corporation", website="apple.com"),
        CompanySchema(id="e4", name="Google LLC", website="google.com"),
        CompanySchema(id="e5", name="Google Inc", website="google.com"),
        CompanySchema(id="e6", name="Microsoft Corp", website="microsoft.com"),
    ]

    # Initialize debugger
    debugger = PipelineDebugger(ground_truth_clusters=ground_truth_clusters, sample_size=5)

    # Step 1: Analyze candidate generation (blocker quality)
    # Simulate a blocker that missed one pair and included one false positive
    candidates = [
        ERCandidate(left=entities[0], right=entities[1], blocker_name="vector"),  # e1-e2
        ERCandidate(left=entities[0], right=entities[2], blocker_name="vector"),  # e1-e3
        # Missing: e2-e3 (recall issue!)
        ERCandidate(left=entities[3], right=entities[4], blocker_name="vector"),  # e4-e5
        ERCandidate(
            left=entities[0], right=entities[3], blocker_name="vector"
        ),  # e1-e4 (false positive!)
    ]

    candidate_stats = debugger.analyze_candidates(candidates, entities)
    print("=" * 60)
    print("CANDIDATE GENERATION ANALYSIS")
    print("=" * 60)
    print(f"Total candidates: {candidate_stats.total_candidates}")
    print(f"Recall: {candidate_stats.candidate_recall:.2%}")
    print(f"Precision: {candidate_stats.candidate_precision:.2%}")
    print(f"Missed matches: {candidate_stats.missed_matches_count}")
    print(f"False positives: {candidate_stats.false_positive_candidates_count}")
    print()

    # Step 2: Analyze scoring quality
    # Simulate LLM scoring with some calibration issues
    judgements = [
        PairwiseJudgement(
            left_id="e1",
            right_id="e2",
            score=0.95,
            score_type="prob_llm",
            decision_step="llm",
            reasoning="Same company name and website",
            provenance={"model": "gpt-4"},
        ),
        PairwiseJudgement(
            left_id="e1",
            right_id="e3",
            score=0.25,  # Too low for a true match!
            score_type="prob_llm",
            decision_step="llm",
            reasoning="Names differ significantly",
            provenance={"model": "gpt-4"},
        ),
        PairwiseJudgement(
            left_id="e4",
            right_id="e5",
            score=0.92,
            score_type="prob_llm",
            decision_step="llm",
            reasoning="Same company, different legal forms",
            provenance={"model": "gpt-4"},
        ),
        PairwiseJudgement(
            left_id="e1",
            right_id="e4",
            score=0.85,  # Too high for a non-match!
            score_type="prob_llm",
            decision_step="llm",
            reasoning="Both tech companies",
            provenance={"model": "gpt-4"},
        ),
    ]

    score_stats = debugger.analyze_scores(judgements)
    print("=" * 60)
    print("SCORING ANALYSIS")
    print("=" * 60)
    print(f"Mean score: {score_stats.mean_score:.3f}")
    print(f"True match mean: {score_stats.true_match_mean:.3f}")
    print(f"Non-match mean: {score_stats.non_match_mean:.3f}")
    print(f"Separation: {score_stats.separation:.3f}")
    print(f"  ⚠️  Negative separation indicates poor calibration!")
    print()

    # Step 3: Analyze clustering
    # Simulate clustering that merged two companies incorrectly
    predicted_clusters = [
        {"e1", "e2", "e3", "e4"},  # Merged Apple and Google! (false merge)
        {"e5"},  # Google split from above (false split)
        {"e6"},  # Microsoft correct
    ]

    cluster_stats = debugger.analyze_clusters(predicted_clusters)
    print("=" * 60)
    print("CLUSTERING ANALYSIS")
    print("=" * 60)
    print(f"Predicted clusters: {cluster_stats.num_predicted_clusters}")
    print(f"Gold clusters: {cluster_stats.num_gold_clusters}")
    print(f"False merges: {cluster_stats.num_false_merges}")
    print(f"False splits: {cluster_stats.num_false_splits}")
    print()

    # Step 4: Generate recommendations
    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    recommendations = debugger._generate_recommendations()
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    print()

    # Step 5: Export reports
    debugger.save_report("tmp/debug_report.json", format="json")
    debugger.save_report("tmp/debug_report.md", format="markdown")
    print("Reports saved to tmp/debug_report.{json,md}")
    print()

    # Display markdown report
    print("=" * 60)
    print("MARKDOWN REPORT (excerpt)")
    print("=" * 60)
    markdown = debugger.to_markdown()
    # Print first 1000 characters
    print(markdown[:1000])
    print("\n... (truncated)")


if __name__ == "__main__":
    main()
