"""Inspection report models for component exploration without ground truth.

This module provides Pydantic models for exploratory analysis of pipeline outputs:
- CandidateInspectionReport: Blocker output exploration
- ScoreInspectionReport: Module score distribution analysis
- ClusterInspectionReport: Clusterer output exploration

These reports enable progressive pipeline building and parameter tuning before
expensive ground truth labeling.
"""

from typing import Any

from pydantic import BaseModel


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
