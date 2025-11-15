"""Report models for component exploration and evaluation.

This module provides Pydantic models for both exploratory analysis (without ground truth)
and comprehensive evaluation (with ground truth):

Inspection Reports (no ground truth required):
- CandidateInspectionReport: Blocker output exploration
- ScoreInspectionReport: Module score distribution analysis
- ClusterInspectionReport: Clusterer output exploration

Evaluation Reports (require ground truth):
- CandidateMetrics: Candidate generation quality metrics
- RankingMetrics: Ranking quality (MAP, MRR, NDCG)
- ScoreMetrics: Score distribution analysis
- RankMetrics: Rank distribution analysis
- RecallCurveStats: Recall@k curve data
- BlockerEvaluationReport: Comprehensive blocker evaluation

These reports enable progressive pipeline building and parameter tuning.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CandidateInspectionReport(BaseModel):
    """Report for candidate exploration without ground truth.

    Use this report to understand blocker output before labeling:
    - How many candidates were generated?
    - Distribution of candidates per entity
    - Sample examples with readable entity text
    - Actionable recommendations for parameter tuning

    Example:
        report = blocker.inspect_candidates(candidates, entities, sample_size=10)
        print(report.to_markdown())
        # Shows total candidates, distribution, examples, recommendations

    Attributes:
        total_candidates: Total number of candidate pairs generated
        avg_candidates_per_entity: Average candidates per entity
        candidate_distribution: Histogram of candidates per entity (e.g., {"1-3": 60, "4-6": 30})
        examples: Sample candidate pairs with readable text
        recommendations: Rule-based suggestions for parameter tuning
    """

    total_candidates: int
    avg_candidates_per_entity: float
    candidate_distribution: dict[str, int]
    examples: list[dict[str, Any]]
    recommendations: list[str]

    @property
    def stats(self) -> dict[str, Any]:
        """Return only numerical metrics (no examples or recommendations).

        Returns:
            Dictionary with total_candidates, avg_candidates_per_entity,
            and candidate_distribution.
        """
        return {
            "total_candidates": self.total_candidates,
            "avg_candidates_per_entity": self.avg_candidates_per_entity,
            "candidate_distribution": self.candidate_distribution,
        }

    def to_dict(self) -> dict[str, Any]:
        """Generate JSON-serializable dictionary.

        Returns:
            Complete report as dictionary.
        """
        return self.model_dump()

    def to_markdown(self) -> str:
        """Generate human-readable markdown report.

        Returns:
            Formatted markdown string suitable for display.
        """
        lines = ["# Candidate Inspection Report\n"]

        # Summary statistics
        lines.append("## Summary")
        lines.append(f"- **Total Candidates**: {self.total_candidates}")
        lines.append(f"- **Average Candidates per Entity**: {self.avg_candidates_per_entity:.2f}\n")

        # Distribution
        if self.candidate_distribution:
            lines.append("## Candidate Distribution")
            for range_label, count in self.candidate_distribution.items():
                lines.append(f"- {range_label} candidates: {count} entities")
            lines.append("")

        # Examples
        if self.examples:
            lines.append("## Sample Candidates")
            for i, example in enumerate(self.examples[:5], 1):
                lines.append(f"\n### Example {i}")
                for key, value in example.items():
                    lines.append(f"- **{key}**: {value}")

        # Recommendations
        if self.recommendations:
            lines.append("\n## Recommendations")
            for rec in self.recommendations:
                lines.append(f"- {rec}")

        return "\n".join(lines)


class ScoreInspectionReport(BaseModel):
    """Report for score exploration without ground truth.

    Use this report to understand module scoring before labeling:
    - Score distribution statistics (mean, median, percentiles)
    - High and low scoring examples with reasoning
    - Threshold recommendations for clustering

    Example:
        report = module.inspect_scores(judgements, sample_size=10)
        print(report.to_markdown())
        # Shows score stats, high/low examples, threshold suggestions

    Attributes:
        total_judgements: Total number of pairwise judgements
        score_distribution: Statistical summary (mean, median, std, percentiles)
        high_scoring_examples: Top scoring pairs with reasoning
        low_scoring_examples: Bottom scoring pairs with reasoning
        recommendations: Rule-based suggestions for threshold tuning
    """

    total_judgements: int
    score_distribution: dict[str, float]
    high_scoring_examples: list[dict[str, Any]]
    low_scoring_examples: list[dict[str, Any]]
    recommendations: list[str]

    @property
    def stats(self) -> dict[str, Any]:
        """Return only numerical metrics (no examples or recommendations).

        Returns:
            Dictionary with total_judgements and score_distribution.
        """
        return {
            "total_judgements": self.total_judgements,
            "score_distribution": self.score_distribution,
        }

    def to_dict(self) -> dict[str, Any]:
        """Generate JSON-serializable dictionary.

        Returns:
            Complete report as dictionary.
        """
        return self.model_dump()

    def to_markdown(self) -> str:
        """Generate human-readable markdown report.

        Returns:
            Formatted markdown string suitable for display.
        """
        lines = ["# Score Inspection Report\n"]

        # Summary statistics
        lines.append("## Summary")
        lines.append(f"- **Total Judgements**: {self.total_judgements}\n")

        # Score distribution
        if self.score_distribution:
            lines.append("## Score Distribution")
            dist = self.score_distribution
            lines.append(f"- **Mean**: {dist.get('mean', 0.0):.3f}")
            lines.append(f"- **Median**: {dist.get('median', 0.0):.3f}")
            lines.append(f"- **Std Dev**: {dist.get('std', 0.0):.3f}")
            lines.append(f"- **P25**: {dist.get('p25', 0.0):.3f}")
            lines.append(f"- **P50**: {dist.get('p50', 0.0):.3f}")
            lines.append(f"- **P75**: {dist.get('p75', 0.0):.3f}")
            lines.append(f"- **P90**: {dist.get('p90', 0.0):.3f}")
            lines.append(f"- **P95**: {dist.get('p95', 0.0):.3f}\n")

        # High scoring examples
        if self.high_scoring_examples:
            lines.append("## High Scoring Examples")
            for i, example in enumerate(self.high_scoring_examples[:5], 1):
                lines.append(f"\n### Example {i}")
                for key, value in example.items():
                    lines.append(f"- **{key}**: {value}")

        # Low scoring examples
        if self.low_scoring_examples:
            lines.append("\n## Low Scoring Examples")
            for i, example in enumerate(self.low_scoring_examples[:5], 1):
                lines.append(f"\n### Example {i}")
                for key, value in example.items():
                    lines.append(f"- **{key}**: {value}")

        # Recommendations
        if self.recommendations:
            lines.append("\n## Recommendations")
            for rec in self.recommendations:
                lines.append(f"- {rec}")

        return "\n".join(lines)


class ClusterInspectionReport(BaseModel):
    """Report for cluster exploration without ground truth.

    Use this report to understand clustering output before labeling:
    - Total cluster count and singleton rate
    - Cluster size distribution
    - Sample of largest clusters with readable text

    Example:
        report = clusterer.inspect_clusters(clusters, entities, sample_size=10)
        print(report.to_markdown())
        # Shows cluster stats, size distribution, large cluster examples

    Attributes:
        total_clusters: Total number of clusters formed
        singleton_rate: Percentage of single-entity clusters (0.0 to 1.0)
        cluster_size_distribution: Histogram of cluster sizes
        largest_clusters: Sample of largest clusters with entity text
        recommendations: Rule-based suggestions for threshold tuning
    """

    total_clusters: int
    singleton_rate: float
    cluster_size_distribution: dict[str, int]
    largest_clusters: list[dict[str, Any]]
    recommendations: list[str]

    @property
    def stats(self) -> dict[str, Any]:
        """Return only numerical metrics (no examples or recommendations).

        Returns:
            Dictionary with total_clusters, singleton_rate, and
            cluster_size_distribution.
        """
        return {
            "total_clusters": self.total_clusters,
            "singleton_rate": self.singleton_rate,
            "cluster_size_distribution": self.cluster_size_distribution,
        }

    def to_dict(self) -> dict[str, Any]:
        """Generate JSON-serializable dictionary.

        Returns:
            Complete report as dictionary.
        """
        return self.model_dump()

    def to_markdown(self) -> str:
        """Generate human-readable markdown report.

        Returns:
            Formatted markdown string suitable for display.
        """
        lines = ["# Cluster Inspection Report\n"]

        # Summary statistics
        lines.append("## Summary")
        lines.append(f"- **Total Clusters**: {self.total_clusters}")
        lines.append(f"- **Singleton Rate**: {self.singleton_rate:.1%}\n")

        # Size distribution
        if self.cluster_size_distribution:
            lines.append("## Cluster Size Distribution")
            for size_label, count in self.cluster_size_distribution.items():
                lines.append(f"- {size_label} entities: {count} clusters")
            lines.append("")

        # Largest clusters
        if self.largest_clusters:
            lines.append("## Largest Clusters")
            for i, cluster in enumerate(self.largest_clusters[:5], 1):
                lines.append(f"\n### Cluster {i}")
                for key, value in cluster.items():
                    if isinstance(value, list):
                        lines.append(f"- **{key}**: {len(value)} items")
                    else:
                        lines.append(f"- **{key}**: {value}")

        # Recommendations
        if self.recommendations:
            lines.append("\n## Recommendations")
            for rec in self.recommendations:
                lines.append(f"- {rec}")

        return "\n".join(lines)


# Evaluation Report Models (require ground truth)


class CandidateMetrics(BaseModel):
    """Candidate generation metrics.

    Measures how well the blocker captures true matches in the candidate generation stage.

    Attributes:
        recall: Fraction of true matches found (TP / (TP + FN)), range [0.0, 1.0]
        precision: Fraction of candidates that are true matches (TP / (TP + FP)), range [0.0, 1.0]
        total: Total number of candidate pairs generated
        avg_per_entity: Average number of candidates per entity
        missed_matches: Number of true matches not found by blocker (false negatives)
        false_positives: Number of false candidates generated

    Example:
        >>> metrics = CandidateMetrics(
        ...     recall=0.95,
        ...     precision=0.80,
        ...     total=1000,
        ...     avg_per_entity=10.5,
        ...     missed_matches=50,
        ...     false_positives=200,
        ... )
        >>> print(f"Blocking recall: {metrics.recall:.2%}")
    """

    recall: float = Field(ge=0.0, le=1.0, description="Fraction of true matches found")
    precision: float = Field(ge=0.0, le=1.0, description="Fraction of candidates that are true")
    total: int = Field(ge=0, description="Total candidate pairs")
    avg_per_entity: float = Field(ge=0.0, description="Average candidates per entity")
    missed_matches: int = Field(ge=0, description="True matches not found")
    false_positives: int = Field(ge=0, description="False candidates generated")

    model_config = ConfigDict(frozen=True)


class RankingMetrics(BaseModel):
    """Ranking quality metrics (MAP, MRR, NDCG).

    Measures how well true matches are ranked by the blocker. Critical for
    budget-constrained downstream processing where we want to process the most
    promising candidates first.

    Attributes:
        map: Mean Average Precision, range [0.0, 1.0]
        mrr: Mean Reciprocal Rank, range [0.0, 1.0]
        ndcg_at_10: Normalized Discounted Cumulative Gain at k=10
        ndcg_at_20: Normalized Discounted Cumulative Gain at k=20
        recall_at_5: Recall at k=5
        recall_at_10: Recall at k=10
        recall_at_20: Recall at k=20

    Example:
        >>> metrics = RankingMetrics(
        ...     map=0.85,
        ...     mrr=0.90,
        ...     ndcg_at_10=0.88,
        ...     ndcg_at_20=0.89,
        ...     recall_at_5=0.75,
        ...     recall_at_10=0.85,
        ...     recall_at_20=0.92,
        ... )
        >>> print(f"MAP: {metrics.map:.3f}, MRR: {metrics.mrr:.3f}")
    """

    map: float = Field(ge=0.0, le=1.0, description="Mean Average Precision")
    mrr: float = Field(ge=0.0, le=1.0, description="Mean Reciprocal Rank")
    ndcg_at_10: float = Field(ge=0.0, le=1.0, description="NDCG@10")
    ndcg_at_20: float = Field(ge=0.0, le=1.0, description="NDCG@20")
    recall_at_5: float = Field(ge=0.0, le=1.0, description="Recall@5")
    recall_at_10: float = Field(ge=0.0, le=1.0, description="Recall@10")
    recall_at_20: float = Field(ge=0.0, le=1.0, description="Recall@20")

    model_config = ConfigDict(frozen=True)


class ScoreMetrics(BaseModel):
    """Score distribution metrics.

    Analyzes the distribution of similarity scores for true matches vs false candidates.
    Good separation indicates well-calibrated blocker scores.

    Attributes:
        separation: True median - False median (higher is better)
        true_median: Median score for true matches
        true_mean: Mean score for true matches
        true_std: Standard deviation of true match scores
        false_median: Median score for false candidates
        false_mean: Mean score for false candidates
        false_std: Standard deviation of false candidate scores
        overlap_fraction: Fraction of scores in overlapping range, range [0.0, 1.0]
        histogram: Histograms for true and false scores

    Example:
        >>> metrics = ScoreMetrics(
        ...     separation=0.45,
        ...     true_median=0.85,
        ...     true_mean=0.82,
        ...     true_std=0.12,
        ...     false_median=0.40,
        ...     false_mean=0.38,
        ...     false_std=0.15,
        ...     overlap_fraction=0.20,
        ...     histogram={"true": {0.8: 10}, "false": {0.4: 20}},
        ... )
        >>> print(f"Score separation: {metrics.separation:.2f}")
    """

    separation: float = Field(description="True median - False median")
    true_median: float = Field(description="Median score for true matches")
    true_mean: float = Field(description="Mean score for true matches")
    true_std: float = Field(description="Std dev of true match scores")
    false_median: float = Field(description="Median score for false candidates")
    false_mean: float = Field(description="Mean score for false candidates")
    false_std: float = Field(description="Std dev of false candidate scores")
    overlap_fraction: float = Field(
        ge=0.0, le=1.0, description="Fraction of scores in overlapping range"
    )
    histogram: dict[str, dict[float, int]] = Field(
        description="Histograms: {'true': {score: count}, 'false': {score: count}}"
    )

    model_config = ConfigDict(frozen=True)


class RankMetrics(BaseModel):
    """Rank distribution metrics.

    Analyzes the rank positions where true matches appear in sorted candidate lists.
    Lower ranks (closer to 1) indicate better blocker ranking quality.

    Attributes:
        median: Median rank of true matches (>= 1.0)
        percentile_95: 95th percentile rank (>= 1.0)
        percent_in_top_5: Percentage of true matches in top-5, range [0.0, 100.0]
        percent_in_top_10: Percentage of true matches in top-10, range [0.0, 100.0]
        percent_in_top_20: Percentage of true matches in top-20, range [0.0, 100.0]
        rank_counts: Histogram of ranks {rank: count}

    Example:
        >>> metrics = RankMetrics(
        ...     median=5.0,
        ...     percentile_95=18.0,
        ...     percent_in_top_5=60.0,
        ...     percent_in_top_10=80.0,
        ...     percent_in_top_20=95.0,
        ...     rank_counts={1: 10, 2: 15, 5: 12},
        ... )
        >>> print(f"Median rank: {metrics.median}, 95th %ile: {metrics.percentile_95}")
    """

    median: float = Field(ge=1.0, description="Median rank of true matches")
    percentile_95: float = Field(ge=1.0, description="95th percentile rank")
    percent_in_top_5: float = Field(ge=0.0, le=100.0, description="% in top-5")
    percent_in_top_10: float = Field(ge=0.0, le=100.0, description="% in top-10")
    percent_in_top_20: float = Field(ge=0.0, le=100.0, description="% in top-20")
    rank_counts: dict[int, int] = Field(description="Histogram {rank: count}")

    model_config = ConfigDict(frozen=True)


class RecallCurveStats(BaseModel):
    """Recall@k curve data.

    Captures the trade-off between recall and computational cost (number of candidates).
    Used to find the optimal k that achieves target recall with minimum cost.

    Attributes:
        k_values: List of k values evaluated
        recall_values: Recall achieved at each k
        avg_pairs_values: Average candidate pairs per entity at each k (cost proxy)

    Example:
        >>> stats = RecallCurveStats(
        ...     k_values=[1, 5, 10, 20, 50],
        ...     recall_values=[0.10, 0.60, 0.85, 0.95, 0.99],
        ...     avg_pairs_values=[1.0, 5.0, 10.0, 20.0, 50.0],
        ... )
        >>> optimal_k = stats.optimal_k(target_recall=0.95)
        >>> print(f"Optimal k for 95% recall: {optimal_k}")
    """

    k_values: list[int] = Field(description="K values evaluated")
    recall_values: list[float] = Field(description="Recall at each k")
    avg_pairs_values: list[float] = Field(description="Avg pairs at each k (cost proxy)")

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_list_lengths(self) -> "RecallCurveStats":
        """Validate that all lists have the same length and are non-empty.

        Raises:
            ValueError: If lists have different lengths or are empty
        """
        lengths = {
            "k_values": len(self.k_values),
            "recall_values": len(self.recall_values),
            "avg_pairs_values": len(self.avg_pairs_values),
        }

        # Check non-empty
        if any(length == 0 for length in lengths.values()):
            raise ValueError(
                f"RecallCurveStats requires at least one data point. Got lengths: {lengths}"
            )

        # Check consistency
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(
                f"All lists in RecallCurveStats must have the same length. Got lengths: {lengths}"
            )

        return self

    def optimal_k(self, target_recall: float = 0.95) -> int:
        """Find smallest k achieving target recall.

        Args:
            target_recall: Target recall threshold (default: 0.95)

        Returns:
            Smallest k where recall >= target_recall, or largest k if target unreachable

        Example:
            >>> stats = RecallCurveStats(
            ...     k_values=[1, 5, 10, 20],
            ...     recall_values=[0.10, 0.60, 0.85, 0.95],
            ...     avg_pairs_values=[1.0, 5.0, 10.0, 20.0],
            ... )
            >>> stats.optimal_k(0.95)
            20
            >>> stats.optimal_k(0.60)
            5
        """
        for k, recall in zip(self.k_values, self.recall_values):
            if recall >= target_recall:
                return k
        # Target not reached - return largest k
        return self.k_values[-1]


class BlockerEvaluationReport(BaseModel):
    """Comprehensive blocker evaluation report.

    Provides a complete view of blocker quality across multiple dimensions:
    - Candidate generation (recall, precision)
    - Ranking quality (MAP, MRR, NDCG)
    - Score distributions (true vs false)
    - Rank distributions
    - Recall curves

    Attributes:
        candidates: Candidate generation metrics
        ranking: Ranking quality metrics
        scores: Score distribution analysis
        rank_distribution: Rank distribution analysis
        recall_curve: Recall@k curve data

    Example:
        >>> from langres.core.analysis import evaluate_blocker_detailed
        >>> report = evaluate_blocker_detailed(candidates, gold_clusters)
        >>> print(f"Candidate recall: {report.candidates.recall:.2%}")
        >>> print(f"MAP: {report.ranking.map:.3f}")
        >>> print(f"Score separation: {report.scores.separation:.2f}")
        >>> report.plot_all(save_path="blocker_eval.png")
    """

    # Metric categories
    candidates: CandidateMetrics = Field(description="Candidate generation metrics")
    ranking: RankingMetrics = Field(description="Ranking quality metrics")
    scores: ScoreMetrics = Field(description="Score distribution analysis")
    rank_distribution: RankMetrics = Field(description="Rank distribution analysis")
    recall_curve: RecallCurveStats = Field(description="Recall@k curve data")

    model_config = ConfigDict(frozen=True)

    @property
    def ranks(self) -> RankMetrics:
        """Deprecated: Use rank_distribution instead.

        .. deprecated:: 0.2.0
            Use :attr:`rank_distribution` instead. This property will be
            removed in version 0.3.0.

        Returns:
            RankMetrics: The rank distribution metrics (same as rank_distribution)
        """
        import warnings

        warnings.warn(
            "Property 'ranks' is deprecated and will be removed in v0.3.0. "
            "Use 'rank_distribution' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.rank_distribution

    @property
    def summary(self) -> dict[str, float]:
        """Get quick overview of key metrics.

        Returns dictionary with the most important metrics for blocker
        quality assessment. Useful for logging, dashboards, or quick
        comparison between different blocker configurations.

        Returns:
            Dictionary with key metrics:
            - candidate_recall: Recall of candidate generation
            - candidate_precision: Precision of candidates
            - map: Mean Average Precision
            - mrr: Mean Reciprocal Rank
            - ndcg_at_10: NDCG@10 score
            - score_separation: Separation between true/false scores
            - median_rank: Median rank of true matches
            - percent_in_top_10: % of true matches in top-10

        Example:
            >>> report = blocker.evaluate(candidates, gold_clusters)
            >>> print(report.summary)
            {'candidate_recall': 0.95, 'map': 0.82, ...}
        """
        return {
            "candidate_recall": self.candidates.recall,
            "candidate_precision": self.candidates.precision,
            "map": self.ranking.map,
            "mrr": self.ranking.mrr,
            "ndcg_at_10": self.ranking.ndcg_at_10,
            "score_separation": self.scores.separation,
            "median_rank": self.rank_distribution.median,
            "percent_in_top_10": self.rank_distribution.percent_in_top_10,
        }

    def to_markdown(self) -> str:
        """Generate human-readable markdown report.

        Returns:
            Markdown-formatted string with all evaluation metrics

        Example:
            >>> report = evaluate_blocker_detailed(candidates, gold_clusters)
            >>> markdown = report.to_markdown()
            >>> print(markdown)
        """
        lines = ["# Blocker Evaluation Report\n"]

        # Candidate metrics
        lines.append("## Candidate Generation\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Recall | {self.candidates.recall:.2%} |")
        lines.append(f"| Precision | {self.candidates.precision:.2%} |")
        lines.append(f"| Total Candidates | {self.candidates.total} |")
        lines.append(f"| Avg per Entity | {self.candidates.avg_per_entity:.2f} |")
        lines.append(f"| Missed Matches | {self.candidates.missed_matches} |")
        lines.append(f"| False Positives | {self.candidates.false_positives} |")
        lines.append("")

        # Ranking metrics
        lines.append("## Ranking Quality\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| MAP | {self.ranking.map:.3f} |")
        lines.append(f"| MRR | {self.ranking.mrr:.3f} |")
        lines.append(f"| NDCG@10 | {self.ranking.ndcg_at_10:.3f} |")
        lines.append(f"| NDCG@20 | {self.ranking.ndcg_at_20:.3f} |")
        lines.append(f"| Recall@5 | {self.ranking.recall_at_5:.2%} |")
        lines.append(f"| Recall@10 | {self.ranking.recall_at_10:.2%} |")
        lines.append(f"| Recall@20 | {self.ranking.recall_at_20:.2%} |")
        lines.append("")

        # Score distribution
        lines.append("## Score Distribution\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Separation | {self.scores.separation:.3f} |")
        lines.append(f"| True Median | {self.scores.true_median:.3f} |")
        lines.append(f"| True Mean | {self.scores.true_mean:.3f} |")
        lines.append(f"| True Std | {self.scores.true_std:.3f} |")
        lines.append(f"| False Median | {self.scores.false_median:.3f} |")
        lines.append(f"| False Mean | {self.scores.false_mean:.3f} |")
        lines.append(f"| False Std | {self.scores.false_std:.3f} |")
        lines.append(f"| Overlap Fraction | {self.scores.overlap_fraction:.2%} |")
        lines.append("")

        # Rank distribution
        lines.append("## Rank Distribution\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Median Rank | {self.rank_distribution.median:.1f} |")
        lines.append(f"| 95th Percentile Rank | {self.rank_distribution.percentile_95:.1f} |")
        lines.append(f"| % in Top-5 | {self.rank_distribution.percent_in_top_5:.1f}% |")
        lines.append(f"| % in Top-10 | {self.rank_distribution.percent_in_top_10:.1f}% |")
        lines.append(f"| % in Top-20 | {self.rank_distribution.percent_in_top_20:.1f}% |")
        lines.append("")

        # Recall curve summary
        lines.append("## Recall Curve\n")
        optimal_k = self.recall_curve.optimal_k(target_recall=0.95)
        lines.append(f"- **Optimal k for 95% recall**: {optimal_k}")
        lines.append(f"- **K values evaluated**: {self.recall_curve.k_values}")
        lines.append("")

        # Recommendations section
        lines.append("## Actionable Recommendations\n")

        recommendations = []

        # Recall recommendations
        if self.candidates.recall < 0.85:
            recommendations.append(
                "- **Low Recall**: Blocker misses many true matches. "
                "Consider: (1) Increasing k value, (2) Using different embeddings, "
                "(3) Adding more blocking methods (e.g., phonetic + semantic)"
            )
        elif self.candidates.recall >= 0.95:
            recommendations.append("- ✅ **Excellent Recall**: Blocker finds >95% of true matches")

        # Separation recommendations
        if self.scores.separation < 0.1:
            recommendations.append(
                "- **Poor Score Separation**: Hard to set threshold. "
                "Consider: (1) Different embedding model, (2) Hybrid index (dense + sparse), "
                "(3) Add reranking step"
            )
        elif self.scores.separation >= 0.3:
            recommendations.append(
                "- ✅ **Good Separation**: Easy to distinguish true/false matches"
            )

        # Ranking recommendations
        if self.rank_distribution.median > 20:
            recommendations.append(
                "- **Poor Ranking**: True matches ranked low. "
                "Consider: (1) Tuning index parameters (nlist, nprobe), "
                "(2) Different distance metric, (3) Add learned reranker"
            )
        elif self.rank_distribution.median <= 5:
            recommendations.append("- ✅ **Excellent Ranking**: True matches typically in top-5")

        # Cost/efficiency recommendations
        if optimal_k > 50:
            recommendations.append(
                f"- **High Cost**: Need k={optimal_k} for 95% recall. "
                "Consider: (1) Better index for faster search, "
                "(2) Ensemble of blockers, (3) Adaptive k per entity"
            )

        if not recommendations:
            recommendations.append("- ✅ All metrics look good! Blocker is production-ready.")

        lines.extend(recommendations)

        return "\n".join(lines)

    def optimal_k(self, target_recall: float = 0.95) -> int:
        """Find smallest k achieving target recall (convenience method).

        This method delegates to recall_curve.optimal_k() for better
        discoverability. Use when you want to find the optimal k value
        without navigating to the recall_curve property.

        Args:
            target_recall: Target recall threshold (default: 0.95)

        Returns:
            Smallest k value achieving target recall

        Raises:
            ValueError: If target recall is unreachable

        Example:
            >>> report = blocker.evaluate(candidates, gold_clusters)
            >>> k = report.optimal_k(target_recall=0.95)
            >>> print(f"Use k={k} for 95% recall")
        """
        return self.recall_curve.optimal_k(target_recall=target_recall)

    def plot_score_distribution(self, ax=None, **kwargs):  # type: ignore[no-untyped-def]
        """Plot score distribution (delegates to plotting module).

        Args:
            ax: Matplotlib axes (optional)
            **kwargs: Additional plotting arguments

        Returns:
            Matplotlib axes

        Raises:
            ImportError: If matplotlib is not installed

        Example:
            >>> report.plot_score_distribution()
        """
        try:
            from langres.plotting.blockers import plot_score_distribution
        except ImportError as e:
            raise ImportError(
                "Plotting requires matplotlib. Install with one of:\n"
                "  pip install 'langres[viz]'\n"
                "  uv add --extra viz langres\n"
                "  poetry add langres[viz]\n"
                "  conda install matplotlib seaborn"
            ) from e
        return plot_score_distribution(self, ax=ax, **kwargs)

    def plot_rank_distribution(self, ax=None, **kwargs):  # type: ignore[no-untyped-def]
        """Plot rank distribution (delegates to plotting module).

        Args:
            ax: Matplotlib axes (optional)
            **kwargs: Additional plotting arguments

        Returns:
            Matplotlib axes

        Raises:
            ImportError: If matplotlib is not installed

        Example:
            >>> report.plot_rank_distribution()
        """
        try:
            from langres.plotting.blockers import plot_rank_distribution
        except ImportError as e:
            raise ImportError(
                "Plotting requires matplotlib. Install with one of:\n"
                "  pip install 'langres[viz]'\n"
                "  uv add --extra viz langres\n"
                "  poetry add langres[viz]\n"
                "  conda install matplotlib seaborn"
            ) from e
        return plot_rank_distribution(self, ax=ax, **kwargs)

    def plot_recall_curve(self, ax=None, **kwargs):  # type: ignore[no-untyped-def]
        """Plot recall@k curve (delegates to plotting module).

        Args:
            ax: Matplotlib axes (optional)
            **kwargs: Additional plotting arguments

        Returns:
            Matplotlib axes

        Raises:
            ImportError: If matplotlib is not installed

        Example:
            >>> report.plot_recall_curve()
        """
        try:
            from langres.plotting.blockers import plot_recall_curve
        except ImportError as e:
            raise ImportError(
                "Plotting requires matplotlib. Install with one of:\n"
                "  pip install 'langres[viz]'\n"
                "  uv add --extra viz langres\n"
                "  poetry add langres[viz]\n"
                "  conda install matplotlib seaborn"
            ) from e
        return plot_recall_curve(self, ax=ax, **kwargs)

    def plot_all(self, save_path: str | None = None, figsize: tuple[int, int] = (16, 12)):  # type: ignore[no-untyped-def]
        """Create 4-panel evaluation summary (delegates to plotting module).

        Args:
            save_path: Path to save figure (optional)
            figsize: Figure size in inches (default: (16, 12))

        Returns:
            Matplotlib figure

        Raises:
            ImportError: If matplotlib is not installed

        Example:
            >>> report.plot_all(save_path="blocker_eval.png")
        """
        try:
            from langres.plotting.blockers import plot_evaluation_summary
        except ImportError as e:
            raise ImportError(
                "Plotting requires matplotlib. Install with one of:\n"
                "  pip install 'langres[viz]'\n"
                "  uv add --extra viz langres\n"
                "  poetry add langres[viz]\n"
                "  conda install matplotlib seaborn"
            ) from e
        return plot_evaluation_summary(self, save_path=save_path, figsize=figsize)
