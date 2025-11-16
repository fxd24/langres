"""
Debugging utilities for entity resolution pipelines.

.. deprecated:: 0.1.0
    This module is deprecated. Use component-level diagnostics instead:

    - For blocker diagnostics: ``BlockerEvaluationReport.diagnose()``
    - For module diagnostics: Coming in future release
    - For clusterer diagnostics: Coming in future release

    See ``docs/FUTURE.md`` for migration guide.

This module provides tools for analyzing and debugging entity resolution pipelines,
offering visibility into candidate generation, scoring quality, and clustering results.
Users can identify issues like poor blocker recall, miscalibrated scores, or incorrect
clustering thresholds.

Example:
    >>> debugger = PipelineDebugger(ground_truth_clusters=[{"e1", "e2"}, {"e3", "e4"}])
    >>> candidate_stats = debugger.analyze_candidates(candidates, entities)
    >>> score_stats = debugger.analyze_scores(judgements)
    >>> cluster_stats = debugger.analyze_clusters(predicted_clusters)
    >>> debugger.save_report("debug.json", format="json")
"""

import json
import logging
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

import numpy as np
from pydantic import BaseModel

from langres.core.models import ERCandidate, PairwiseJudgement

logger = logging.getLogger(__name__)

# Type variable for generic entity schema
SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass
class CandidateStats:
    """Statistics for candidate generation quality."""

    total_candidates: int
    avg_candidates_per_entity: float
    candidate_recall: float
    candidate_precision: float
    missed_matches_count: int
    false_positive_candidates_count: int


@dataclass
class ScoreStats:
    """Statistics for scoring quality and calibration."""

    mean_score: float
    median_score: float
    std_score: float
    p25: float
    p75: float
    p95: float
    true_match_mean: float
    non_match_mean: float
    separation: float


@dataclass
class ClusterStats:
    """Statistics for clustering quality."""

    num_predicted_clusters: int
    num_gold_clusters: int
    avg_cluster_size: float
    num_singletons: int
    largest_cluster_size: int
    num_false_merges: int
    num_false_splits: int


@dataclass
class ErrorExample:
    """Example of a pipeline error for debugging."""

    error_type: str
    entity_ids: list[str]
    entity_texts: list[str]
    explanation: str
    metadata: dict[str, Any]


class PipelineDebugger(Generic[SchemaT]):
    """
    Debugger for entity resolution pipelines.

    .. deprecated:: 0.1.0
        ``PipelineDebugger`` is deprecated. Use component-level diagnostics instead:

        For blocker diagnostics::

            report = blocker.evaluate(candidates, gold_clusters)
            diagnostics = report.diagnose(candidates, entities)
            print(diagnostics.to_markdown())

        For module/clusterer diagnostics: Coming in future release.

        See ``docs/FUTURE.md`` for migration guide and roadmap.

    This class provides comprehensive analysis of three pipeline stages:
    1. Candidate generation (blocker quality)
    2. Scoring (LLM calibration and separation)
    3. Clustering (threshold tuning)

    Attributes:
        entity_to_cluster: Mapping from entity ID to gold cluster ID
        true_matches: Set of all true match pairs (sorted tuples)
        sample_size: Maximum number of error examples to store per type
        error_examples: List of collected error examples across all analyses
    """

    def __init__(
        self,
        ground_truth_clusters: list[set[str]],
        sample_size: int = 10,
    ):
        warnings.warn(
            "PipelineDebugger is deprecated and will be removed in a future version. "
            "Use BlockerEvaluationReport.diagnose() for blocker diagnostics. "
            "See docs/FUTURE.md for details.",
            DeprecationWarning,
            stacklevel=2,
        )
        """
        Initialize debugger with ground truth labels.

        Args:
            ground_truth_clusters: List of sets, where each set contains entity IDs
                that belong to the same true cluster
            sample_size: Maximum number of error examples to store per error type
        """
        self.sample_size = sample_size
        self.error_examples: list[ErrorExample] = []

        # Build entity_to_cluster mapping
        self.entity_to_cluster: dict[str, int] = {}
        for cluster_id, cluster in enumerate(ground_truth_clusters):
            for entity_id in cluster:
                self.entity_to_cluster[entity_id] = cluster_id

        # Build true_matches set (all pairs within each cluster)
        self.true_matches: set[tuple[str, str]] = set()
        for cluster in ground_truth_clusters:
            cluster_list = list(cluster)
            for i, e1 in enumerate(cluster_list):
                for e2 in cluster_list[i + 1 :]:
                    # Store as sorted tuple for order-independent comparison
                    pair = tuple(sorted([e1, e2]))
                    assert len(pair) == 2  # Ensure it's a 2-tuple for type safety
                    self.true_matches.add((pair[0], pair[1]))

        # Store ground truth for later use
        self._ground_truth_clusters = ground_truth_clusters

        # Storage for analysis results
        self._candidate_stats: CandidateStats | None = None
        self._score_stats: ScoreStats | None = None
        self._cluster_stats: ClusterStats | None = None

        # Storage for raw data (for report generation)
        self._entities_map: dict[str, SchemaT] = {}

    def analyze_candidates(
        self,
        candidates: list[ERCandidate[SchemaT]],
        entities: list[SchemaT],
    ) -> CandidateStats:
        """
        Analyze candidate generation quality.

        Calculates recall (what % of true matches were found) and precision
        (what % of candidates are true matches), and identifies missed matches
        and false positives.

        Args:
            candidates: List of candidate pairs generated by blocker
            entities: List of all entities (must have 'id' field)

        Returns:
            CandidateStats with quality metrics
        """
        # Build entity map for later use
        self._entities_map = {str(e.id): e for e in entities}  # type: ignore[attr-defined]

        # Extract candidate pairs as sorted tuples (explicitly typed as strings)
        candidate_pairs: set[tuple[str, str]] = set()
        for c in candidates:
            left_id = str(c.left.id)  # type: ignore[attr-defined]
            right_id = str(c.right.id)  # type: ignore[attr-defined]
            pair_tuple = tuple(sorted([left_id, right_id]))
            # Ensure it's a 2-tuple for type safety
            assert len(pair_tuple) == 2
            candidate_pairs.add((pair_tuple[0], pair_tuple[1]))

        # Calculate recall: % of true matches found
        found_matches = candidate_pairs & self.true_matches
        if len(self.true_matches) > 0:
            recall = len(found_matches) / len(self.true_matches)
        else:
            recall = 1.0  # No true matches to miss

        # Calculate precision: % of candidates that are true matches
        if len(candidate_pairs) > 0:
            precision = len(found_matches) / len(candidate_pairs)
        else:
            precision = 0.0  # No candidates

        # Identify missed matches and false positives
        missed_matches = self.true_matches - candidate_pairs
        false_positives = candidate_pairs - self.true_matches

        # Calculate average candidates per entity
        # Count the number of candidate pairs each entity participates in
        entity_candidate_counts: dict[str, set[str]] = defaultdict(set)
        for left_id, right_id in candidate_pairs:
            # For each entity, track which OTHER entities it's paired with
            entity_candidate_counts[left_id].add(right_id)
            entity_candidate_counts[right_id].add(left_id)

        if len(entities) > 0:
            # Average number of partner entities per entity
            total_partners = sum(len(partners) for partners in entity_candidate_counts.values())
            avg_candidates_per_entity = total_partners / len(entities)
        else:
            avg_candidates_per_entity = 0.0

        # Generate error examples for missed matches (sample)
        for e1, e2 in list(missed_matches)[: self.sample_size]:
            entity1 = self._entities_map.get(e1)
            entity2 = self._entities_map.get(e2)

            if entity1 and entity2:
                self.error_examples.append(
                    ErrorExample(
                        error_type="missed_match",
                        entity_ids=[e1, e2],
                        entity_texts=[
                            str(getattr(entity1, "name", entity1)),
                            str(getattr(entity2, "name", entity2)),
                        ],
                        explanation=f"True match not found by blocker (cluster {self.entity_to_cluster.get(e1)})",
                        metadata={"cluster_id": self.entity_to_cluster.get(e1)},
                    )
                )

        # Store stats
        stats = CandidateStats(
            total_candidates=len(candidate_pairs),
            avg_candidates_per_entity=avg_candidates_per_entity,
            candidate_recall=recall,
            candidate_precision=precision,
            missed_matches_count=len(missed_matches),
            false_positive_candidates_count=len(false_positives),
        )
        self._candidate_stats = stats
        return stats

    def analyze_scores(self, judgements: list[PairwiseJudgement]) -> ScoreStats:
        """
        Analyze scoring quality and calibration.

        Calculates score distribution statistics and separates scores by true label
        to assess calibration quality. Identifies high-scoring non-matches and
        low-scoring matches.

        Args:
            judgements: List of pairwise similarity judgements

        Returns:
            ScoreStats with distribution and calibration metrics
        """
        # Populate _entities_map if not already done (for standalone score analysis)
        if not self._entities_map and judgements:
            # Extract entity IDs from judgements and create minimal entries
            for j in judgements:
                if j.left_id not in self._entities_map:
                    # Create a minimal entity representation for error reporting
                    # Use cast since we're creating a duck-typed entity with id/name attrs
                    entity = cast(
                        SchemaT, type("Entity", (), {"id": j.left_id, "name": j.left_id})()
                    )
                    self._entities_map[j.left_id] = entity
                if j.right_id not in self._entities_map:
                    entity = cast(
                        SchemaT, type("Entity", (), {"id": j.right_id, "name": j.right_id})()
                    )
                    self._entities_map[j.right_id] = entity

        if not judgements:
            # Return zero stats for empty input
            return ScoreStats(
                mean_score=0.0,
                median_score=0.0,
                std_score=0.0,
                p25=0.0,
                p75=0.0,
                p95=0.0,
                true_match_mean=0.0,
                non_match_mean=0.0,
                separation=0.0,
            )

        # Extract scores
        scores = np.array([j.score for j in judgements])

        # Calculate overall statistics
        mean_score = float(np.mean(scores))
        median_score = float(np.median(scores))
        std_score = float(np.std(scores))
        p25 = float(np.percentile(scores, 25))
        p75 = float(np.percentile(scores, 75))
        p95 = float(np.percentile(scores, 95))

        # Separate scores by true label
        true_match_scores: list[float] = []
        non_match_scores: list[float] = []

        for j in judgements:
            pair = tuple(sorted([j.left_id, j.right_id]))
            is_true_match = pair in self.true_matches

            if is_true_match:
                true_match_scores.append(j.score)
            else:
                non_match_scores.append(j.score)

        # Calculate means by label
        true_match_mean = float(np.mean(true_match_scores)) if true_match_scores else 0.0
        non_match_mean = float(np.mean(non_match_scores)) if non_match_scores else 0.0
        separation = true_match_mean - non_match_mean

        # Generate error examples for misaligned scores
        # 1. High-scoring non-matches (score > 0.7)
        high_nonmatch_count = 0
        for j in judgements:
            if high_nonmatch_count >= self.sample_size:
                break

            pair = tuple(sorted([j.left_id, j.right_id]))
            if pair not in self.true_matches and j.score > 0.7:
                e1 = self._entities_map.get(j.left_id)
                e2 = self._entities_map.get(j.right_id)

                if e1 and e2:
                    self.error_examples.append(
                        ErrorExample(
                            error_type="high_scoring_nonmatch",
                            entity_ids=[j.left_id, j.right_id],
                            entity_texts=[
                                str(getattr(e1, "name", e1)),
                                str(getattr(e2, "name", e2)),
                            ],
                            explanation=f"Non-match scored too high ({j.score:.2f})",
                            metadata={
                                "score": j.score,
                                "reasoning": j.reasoning,
                            },
                        )
                    )
                    high_nonmatch_count += 1

        # 2. Low-scoring matches (score < 0.3)
        low_match_count = 0
        for j in judgements:
            if low_match_count >= self.sample_size:
                break

            pair = tuple(sorted([j.left_id, j.right_id]))
            if pair in self.true_matches and j.score < 0.3:
                e1 = self._entities_map.get(j.left_id)
                e2 = self._entities_map.get(j.right_id)

                if e1 and e2:
                    self.error_examples.append(
                        ErrorExample(
                            error_type="low_scoring_match",
                            entity_ids=[j.left_id, j.right_id],
                            entity_texts=[
                                str(getattr(e1, "name", e1)),
                                str(getattr(e2, "name", e2)),
                            ],
                            explanation=f"True match scored too low ({j.score:.2f})",
                            metadata={
                                "score": j.score,
                                "reasoning": j.reasoning,
                            },
                        )
                    )
                    low_match_count += 1

        # Store stats
        stats = ScoreStats(
            mean_score=mean_score,
            median_score=median_score,
            std_score=std_score,
            p25=p25,
            p75=p75,
            p95=p95,
            true_match_mean=true_match_mean,
            non_match_mean=non_match_mean,
            separation=separation,
        )
        self._score_stats = stats
        return stats

    def analyze_clusters(self, predicted_clusters: list[set[str]]) -> ClusterStats:
        """
        Analyze clustering quality.

        Identifies false merges (predicted clusters that span multiple gold clusters)
        and false splits (gold clusters split across multiple predictions).

        Args:
            predicted_clusters: List of predicted entity clusters

        Returns:
            ClusterStats with clustering quality metrics
        """
        if not predicted_clusters:
            return ClusterStats(
                num_predicted_clusters=0,
                num_gold_clusters=len(self._ground_truth_clusters),
                avg_cluster_size=0.0,
                num_singletons=0,
                largest_cluster_size=0,
                num_false_merges=0,
                num_false_splits=0,
            )

        # Basic statistics
        num_predicted = len(predicted_clusters)
        num_gold = len(self._ground_truth_clusters)
        cluster_sizes = [len(c) for c in predicted_clusters]
        avg_size = float(np.mean(cluster_sizes)) if cluster_sizes else 0.0
        num_singletons = sum(1 for c in predicted_clusters if len(c) == 1)
        largest = max(cluster_sizes) if cluster_sizes else 0

        # Detect false merges: predicted cluster contains entities from multiple gold clusters
        num_false_merges = 0
        for pred_cluster in predicted_clusters:
            gold_cluster_ids = {
                self.entity_to_cluster.get(eid)
                for eid in pred_cluster
                if eid in self.entity_to_cluster
            }
            # Remove None if any entity IDs are not in ground truth
            gold_cluster_ids.discard(None)

            if len(gold_cluster_ids) > 1:
                num_false_merges += 1

                # Generate error example
                if (
                    len([e for e in self.error_examples if e.error_type == "false_merge"])
                    < self.sample_size
                ):
                    entity_ids = list(pred_cluster)
                    entity_texts = [
                        str(getattr(self._entities_map.get(eid), "name", eid))
                        if eid in self._entities_map
                        else eid
                        for eid in entity_ids
                    ]

                    self.error_examples.append(
                        ErrorExample(
                            error_type="false_merge",
                            entity_ids=entity_ids,
                            entity_texts=entity_texts,
                            explanation=f"Cluster contains entities from {len(gold_cluster_ids)} different gold clusters",
                            metadata={"gold_cluster_ids": list(gold_cluster_ids)},
                        )
                    )

        # Detect false splits: gold cluster split across multiple predicted clusters
        num_false_splits = 0
        for gold_cluster in self._ground_truth_clusters:
            # Find which predicted clusters contain entities from this gold cluster
            pred_clusters_with_gold: set[int] = set()
            for pred_idx, pred_cluster in enumerate(predicted_clusters):
                if any(eid in pred_cluster for eid in gold_cluster):
                    pred_clusters_with_gold.add(pred_idx)

            if len(pred_clusters_with_gold) > 1:
                num_false_splits += 1

                # Generate error example
                if (
                    len([e for e in self.error_examples if e.error_type == "false_split"])
                    < self.sample_size
                ):
                    entity_ids = list(gold_cluster)
                    entity_texts = [
                        str(getattr(self._entities_map.get(eid), "name", eid))
                        if eid in self._entities_map
                        else eid
                        for eid in entity_ids
                    ]

                    self.error_examples.append(
                        ErrorExample(
                            error_type="false_split",
                            entity_ids=entity_ids,
                            entity_texts=entity_texts,
                            explanation=f"Gold cluster split across {len(pred_clusters_with_gold)} predicted clusters",
                            metadata={"num_splits": len(pred_clusters_with_gold)},
                        )
                    )

        # Store stats
        stats = ClusterStats(
            num_predicted_clusters=num_predicted,
            num_gold_clusters=num_gold,
            avg_cluster_size=avg_size,
            num_singletons=num_singletons,
            largest_cluster_size=largest,
            num_false_merges=num_false_merges,
            num_false_splits=num_false_splits,
        )
        self._cluster_stats = stats
        return stats

    def _generate_recommendations(self) -> list[str]:
        """
        Generate actionable recommendations based on analysis results.

        Returns:
            List of recommendation strings
        """
        recommendations: list[str] = []

        # Candidate generation recommendations
        if self._candidate_stats:
            if self._candidate_stats.candidate_recall < 0.9:
                recommendations.append(
                    f"Low candidate recall ({self._candidate_stats.candidate_recall:.2%}): "
                    "Consider increasing k_neighbors in blocker to find more matches."
                )

            # Recommend improving precision if < 50% or if many false positives
            if (
                self._candidate_stats.candidate_precision < 0.5
                or self._candidate_stats.false_positive_candidates_count
                > self._candidate_stats.total_candidates / 2
            ):
                recommendations.append(
                    f"Low candidate precision ({self._candidate_stats.candidate_precision:.2%}): "
                    "Consider decreasing k_neighbors or using more selective blocking keys to reduce false positive candidates."
                )

        # Score separation recommendations
        if self._score_stats:
            if self._score_stats.separation < 0.2:
                recommendations.append(
                    f"Poor score separation ({self._score_stats.separation:.2f}): "
                    "Consider improving LLM prompt, adding examples, or recalibrating scores."
                )

        # Clustering recommendations
        if self._cluster_stats:
            if self._cluster_stats.num_false_merges > 0:
                recommendations.append(
                    f"Found {self._cluster_stats.num_false_merges} false merges: "
                    "Consider increasing cluster_threshold to be more conservative."
                )

            if self._cluster_stats.num_false_splits > 0:
                recommendations.append(
                    f"Found {self._cluster_stats.num_false_splits} false splits: "
                    "Consider decreasing cluster_threshold to merge more aggressively."
                )

            total_entities = sum(len(c) for c in self._ground_truth_clusters)
            if total_entities > 0 and self._cluster_stats.num_singletons / total_entities > 0.5:
                recommendations.append(
                    f"High singleton rate ({self._cluster_stats.num_singletons}/{total_entities}): "
                    "Check blocker recall - many entities may not be finding any candidates."
                )

        if not recommendations:
            recommendations.append("No issues detected - pipeline looks good!")

        return recommendations

    def to_dict(self) -> dict[str, Any]:
        """
        Convert all analysis to dictionary (for JSON export).

        Returns:
            Dictionary containing all stats and error examples
        """
        return {
            "candidate_stats": asdict(self._candidate_stats) if self._candidate_stats else None,
            "score_stats": asdict(self._score_stats) if self._score_stats else None,
            "cluster_stats": asdict(self._cluster_stats) if self._cluster_stats else None,
            "error_examples": [asdict(e) for e in self.error_examples],
            "recommendations": self._generate_recommendations(),
        }

    def to_markdown(self) -> str:
        """
        Generate human-readable markdown report.

        Returns:
            Markdown-formatted report string
        """
        lines: list[str] = []

        lines.append("# Pipeline Debug Report\n")

        # Section 1: Candidate Generation
        if self._candidate_stats:
            lines.append("## Candidate Generation\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Total Candidates | {self._candidate_stats.total_candidates} |")
            lines.append(
                f"| Avg Candidates per Entity | {self._candidate_stats.avg_candidates_per_entity:.2f} |"
            )
            lines.append(f"| Candidate Recall | {self._candidate_stats.candidate_recall:.2%} |")
            lines.append(
                f"| Candidate Precision | {self._candidate_stats.candidate_precision:.2%} |"
            )
            lines.append(f"| Missed Matches | {self._candidate_stats.missed_matches_count} |")
            lines.append(
                f"| False Positive Candidates | {self._candidate_stats.false_positive_candidates_count} |"
            )
            lines.append("")

        # Section 2: Score Distribution
        if self._score_stats:
            lines.append("## Score Distribution\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Mean Score | {self._score_stats.mean_score:.3f} |")
            lines.append(f"| Median Score | {self._score_stats.median_score:.3f} |")
            lines.append(f"| Std Dev | {self._score_stats.std_score:.3f} |")
            lines.append(f"| 25th Percentile | {self._score_stats.p25:.3f} |")
            lines.append(f"| 75th Percentile | {self._score_stats.p75:.3f} |")
            lines.append(f"| 95th Percentile | {self._score_stats.p95:.3f} |")
            lines.append(f"| True Match Mean | {self._score_stats.true_match_mean:.3f} |")
            lines.append(f"| Non-Match Mean | {self._score_stats.non_match_mean:.3f} |")
            lines.append(f"| Separation | {self._score_stats.separation:.3f} |")
            lines.append("")

        # Section 3: Clustering Results
        if self._cluster_stats:
            lines.append("## Clustering Results\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Predicted Clusters | {self._cluster_stats.num_predicted_clusters} |")
            lines.append(f"| Gold Clusters | {self._cluster_stats.num_gold_clusters} |")
            lines.append(f"| Avg Cluster Size | {self._cluster_stats.avg_cluster_size:.2f} |")
            lines.append(f"| Singletons | {self._cluster_stats.num_singletons} |")
            lines.append(f"| Largest Cluster | {self._cluster_stats.largest_cluster_size} |")
            lines.append(f"| False Merges | {self._cluster_stats.num_false_merges} |")
            lines.append(f"| False Splits | {self._cluster_stats.num_false_splits} |")
            lines.append("")

        # Section 4: Error Examples
        lines.append("## Error Examples\n")

        if self.error_examples:
            # Group by error type
            errors_by_type: dict[str, list[ErrorExample]] = defaultdict(list)
            for error in self.error_examples:
                errors_by_type[error.error_type].append(error)

            for error_type, errors in errors_by_type.items():
                lines.append(f"### {error_type.replace('_', ' ').title()}\n")
                for i, error in enumerate(errors[: self.sample_size], 1):
                    lines.append(f"{i}. **{error.explanation}**")
                    lines.append(f"   - Entity IDs: {', '.join(error.entity_ids)}")
                    lines.append(f"   - Entity Texts: {', '.join(error.entity_texts)}")
                    if error.metadata:
                        lines.append(f"   - Metadata: {error.metadata}")
                    lines.append("")
        else:
            lines.append("No errors detected.\n")

        # Section 5: Recommendations
        lines.append("## Recommendations\n")
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")

        return "\n".join(lines)

    def save_report(self, output_path: str | Path, format: str = "json") -> None:
        """
        Save debug report to file.

        Args:
            output_path: Path to save report
            format: Either "json" or "markdown"

        Raises:
            ValueError: If format is not "json" or "markdown"
        """
        output_path = Path(output_path)

        # Create parent directories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Saved JSON report to {output_path}")

        elif format == "markdown":
            with open(output_path, "w") as f:
                f.write(self.to_markdown())
            logger.info(f"Saved markdown report to {output_path}")

        else:
            raise ValueError(f"Invalid format: {format}. Must be 'json' or 'markdown'.")
