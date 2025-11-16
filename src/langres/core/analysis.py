"""Analysis functions for blocker evaluation.

This module provides detailed analysis of blocker performance including:
- Score distribution metrics (separation between true/false scores)
- Rank distribution metrics (where true matches appear in ranked lists)
- Recall curves (recall@k vs. computational cost)
- Comprehensive evaluation reports
"""

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from langres.core.metrics import evaluate_blocking, evaluate_blocking_with_ranking
from langres.core.models import ERCandidate
from langres.core.reports import (
    BlockerEvaluationReport,
    CandidateMetrics,
    RankingMetrics,
    RankMetrics,
    RecallCurveStats,
    ScoreMetrics,
)

if TYPE_CHECKING:
    from langres.core.diagnostics import FalsePositiveExample, MissedMatchExample

logger = logging.getLogger(__name__)


_DEFAULT_HISTOGRAM_BINS = 50
"""Default number of bins for score distribution histograms.

This value balances granularity (seeing score distribution details) with
robustness (avoiding noise from sparse bins). Suitable for typical blocker
evaluation with 100-10,000 candidate pairs.

Can be adjusted for specific use cases:
- Fewer bins (20-30): Small datasets (<1000 pairs)
- More bins (100+): Large datasets (>100,000 pairs) with high precision needs

The value 50 provides good discrimination between true and false score
distributions while being robust to outliers.
"""


def _build_ground_truth_pairs(gold_clusters: list[set[str]]) -> set[tuple[str, str]]:
    """Build set of all true matching pairs from clusters.

    Args:
        gold_clusters: Ground truth entity clusters

    Returns:
        Set of (entity_a, entity_b) tuples representing true matches.
        Tuples are sorted to ensure consistent ordering.
    """
    pairs = set()
    for cluster in gold_clusters:
        entities = sorted(cluster)  # Sort for consistent ordering
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1 :]:
                pairs.add((e1, e2))
    return pairs


def _is_true_match(left_id: str, right_id: str, true_pairs: set[tuple[str, str]]) -> bool:
    """Check if candidate is a true match.

    Args:
        left_id: Left entity ID
        right_id: Right entity ID
        true_pairs: Set of true matching pairs

    Returns:
        True if (left_id, right_id) is a true match
    """
    pair = tuple(sorted([left_id, right_id]))
    return pair in true_pairs


def _create_histogram(scores: list[float], bins: int = _DEFAULT_HISTOGRAM_BINS) -> dict[float, int]:
    """Create histogram dict from scores.

    Args:
        scores: List of similarity scores
        bins: Number of histogram bins (default: _DEFAULT_HISTOGRAM_BINS)

    Returns:
        Dict mapping bin centers to counts
    """
    if not scores:
        return {}

    counts, bin_edges = np.histogram(scores, bins=bins)
    # Use bin centers as keys
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return {float(center): int(count) for center, count in zip(bin_centers, counts, strict=False)}


def _compute_score_metrics(
    candidates: list[ERCandidate[Any]],
    gold_clusters: list[set[str]],
) -> ScoreMetrics:
    """Compute score distribution metrics.

    Separates candidates into true matches vs false candidates based on
    gold_clusters, then computes statistics and histograms for each group.

    Args:
        candidates: Candidate pairs with similarity_score populated
        gold_clusters: Ground truth entity clusters

    Returns:
        ScoreMetrics with distribution statistics

    Algorithm:
        1. Build ground truth lookup: set of (entity_a, entity_b) pairs from gold_clusters
        2. Separate candidate scores into true_scores and false_scores
        3. Compute statistics: mean, median, std for each group (use numpy)
        4. Compute histograms: 50 bins using numpy.histogram
        5. Compute separation: true_median - false_median
        6. Compute overlap_fraction: fraction of score range where distributions overlap
        7. Return ScoreMetrics
    """
    logger.debug("Computing score distribution metrics for %d candidates", len(candidates))

    # Step 1: Build ground truth lookup
    true_pairs = _build_ground_truth_pairs(gold_clusters)

    # Step 2: Separate scores
    true_scores: list[float] = []
    false_scores: list[float] = []

    for candidate in candidates:
        score = candidate.similarity_score
        if score is None:
            continue  # Skip candidates without scores

        if _is_true_match(candidate.left.id, candidate.right.id, true_pairs):
            true_scores.append(score)
        else:
            false_scores.append(score)

    logger.debug("Found %d true matches, %d false candidates", len(true_scores), len(false_scores))

    # Step 3: Compute statistics
    if true_scores:
        true_mean = float(np.mean(true_scores))
        true_median = float(np.median(true_scores))
        true_std = float(np.std(true_scores))
    else:
        true_mean = true_median = true_std = 0.0

    if false_scores:
        false_mean = float(np.mean(false_scores))
        false_median = float(np.median(false_scores))
        false_std = float(np.std(false_scores))
    else:
        false_mean = false_median = false_std = 0.0

    # Step 4: Compute histograms
    true_histogram = _create_histogram(true_scores, bins=_DEFAULT_HISTOGRAM_BINS)
    false_histogram = _create_histogram(false_scores, bins=_DEFAULT_HISTOGRAM_BINS)

    # Step 5: Compute separation
    separation = true_median - false_median

    # Step 6: Compute overlap_fraction
    if true_scores and false_scores:
        # Compute range overlap
        true_min, true_max = min(true_scores), max(true_scores)
        false_min, false_max = min(false_scores), max(false_scores)

        # Overlap is the intersection of ranges
        overlap_start = max(true_min, false_min)
        overlap_end = min(true_max, false_max)

        if overlap_start < overlap_end:
            # There is overlap
            total_range = max(true_max, false_max) - min(true_min, false_min)
            overlap_range = overlap_end - overlap_start
            overlap_fraction = overlap_range / total_range if total_range > 0 else 0.0
        else:
            # No overlap
            overlap_fraction = 0.0
    else:
        overlap_fraction = 0.0

    logger.debug(
        "Score metrics: true_mean=%.3f, false_mean=%.3f, separation=%.3f, overlap=%.3f",
        true_mean,
        false_mean,
        separation,
        overlap_fraction,
    )

    return ScoreMetrics(
        true_mean=true_mean,
        true_median=true_median,
        true_std=true_std,
        false_mean=false_mean,
        false_median=false_median,
        false_std=false_std,
        separation=separation,
        overlap_fraction=overlap_fraction,
        histogram={"true": true_histogram, "false": false_histogram},
    )


def _compute_rank_metrics(
    candidates: list[ERCandidate[Any]],
    gold_clusters: list[set[str]],
) -> RankMetrics:
    """Compute rank distribution metrics.

    For each entity, sort its candidates by similarity_score descending,
    then find the rank (position) of true matches.

    Args:
        candidates: Candidate pairs with similarity_score
        gold_clusters: Ground truth clusters

    Returns:
        RankMetrics with rank distribution statistics

    Algorithm:
        1. Build ground truth lookup
        2. Group candidates by left_id (entity being matched)
        3. For each entity's candidates:
           - Sort by similarity_score descending
           - Find positions (ranks) of true matches (1-indexed)
        4. Collect all true match ranks
        5. Compute percentiles (median, 95th)
        6. Compute percent in top-5, top-10, top-20
        7. Build rank histogram (rank → count)
        8. Return RankMetrics
    """
    logger.debug("Computing rank distribution metrics for %d candidates", len(candidates))

    # Step 1: Build ground truth lookup
    true_pairs = _build_ground_truth_pairs(gold_clusters)

    # Step 2: Group candidates by left_id
    entity_candidates: dict[str, list[ERCandidate[Any]]] = defaultdict(list)
    for candidate in candidates:
        entity_candidates[candidate.left.id].append(candidate)

    # Step 3: Find ranks of true matches
    all_true_ranks: list[int] = []

    for entity_id, entity_cands in entity_candidates.items():
        # Sort by similarity_score descending, with deterministic tie-breaking by entity ID
        sorted_cands = sorted(
            entity_cands, key=lambda c: (-(c.similarity_score or 0.0), c.right.id)
        )

        # Find positions of true matches (1-indexed)
        for rank, candidate in enumerate(sorted_cands, start=1):
            if _is_true_match(candidate.left.id, candidate.right.id, true_pairs):
                all_true_ranks.append(rank)

    logger.debug("Found %d true match ranks", len(all_true_ranks))

    if not all_true_ranks:
        # No true matches found - use 1.0 as minimum valid rank
        return RankMetrics(
            median=1.0,
            percentile_95=1.0,
            percent_in_top_5=0.0,
            percent_in_top_10=0.0,
            percent_in_top_20=0.0,
            rank_counts={},
        )

    # Step 5: Compute percentiles
    median_rank = float(np.median(all_true_ranks))
    percentile_95 = float(np.percentile(all_true_ranks, 95))

    # Step 6: Compute percent in top-k (as PERCENTAGES, not fractions)
    total_true_matches = len(all_true_ranks)
    percent_in_top_5 = 100.0 * sum(1 for r in all_true_ranks if r <= 5) / total_true_matches
    percent_in_top_10 = 100.0 * sum(1 for r in all_true_ranks if r <= 10) / total_true_matches
    percent_in_top_20 = 100.0 * sum(1 for r in all_true_ranks if r <= 20) / total_true_matches

    # Step 7: Build rank histogram
    rank_counts: dict[int, int] = {}
    for rank in all_true_ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1

    logger.debug(
        "Rank metrics: median=%.1f, p95=%.1f, top5=%.1f%%, top10=%.1f%%, top20=%.1f%%",
        median_rank,
        percentile_95,
        percent_in_top_5,
        percent_in_top_10,
        percent_in_top_20,
    )

    return RankMetrics(
        median=median_rank,
        percentile_95=percentile_95,
        percent_in_top_5=percent_in_top_5,
        percent_in_top_10=percent_in_top_10,
        percent_in_top_20=percent_in_top_20,
        rank_counts=rank_counts,
    )


def _compute_recall_curve(
    candidates: list[ERCandidate[Any]],
    gold_clusters: list[set[str]],
    k_values: list[int],
) -> RecallCurveStats:
    """Compute recall@k for different k values.

    For each k, compute what fraction of true matches appear in the
    top-k candidates for each entity.

    Args:
        candidates: Candidate pairs with similarity_score
        gold_clusters: Ground truth clusters
        k_values: List of k values to evaluate

    Returns:
        RecallCurveStats with recall and cost data

    Algorithm:
        1. Build ground truth: count total true matches
        2. Group candidates by left_id, sort by score descending
        3. For each k in k_values:
           - For each entity, take top-k candidates
           - Count how many true matches found
           - Compute recall = found / total_true_matches
           - Compute avg_pairs = total candidates considered / num entities
        4. Return RecallCurveStats
    """
    logger.debug("Computing recall curve for k_values=%s", k_values)

    # Step 1: Build ground truth
    true_pairs = _build_ground_truth_pairs(gold_clusters)
    total_true_matches = len(true_pairs)

    if total_true_matches == 0:
        logger.warning("No true matches found in gold clusters")
        return RecallCurveStats(
            k_values=k_values,
            recall_values=[0.0] * len(k_values),
            avg_pairs_values=[0.0] * len(k_values),
        )

    # Step 2: Group candidates by left_id and sort
    entity_candidates: dict[str, list[ERCandidate[Any]]] = defaultdict(list)
    for candidate in candidates:
        entity_candidates[candidate.left.id].append(candidate)

    # Sort each entity's candidates by score descending, with deterministic tie-breaking by entity ID
    for entity_id in entity_candidates:
        entity_candidates[entity_id].sort(key=lambda c: (-(c.similarity_score or 0.0), c.right.id))

    num_entities = len(entity_candidates)

    # Step 3: Compute recall@k for each k
    recall_values: list[float] = []
    avg_pairs_values: list[float] = []

    for k in k_values:
        # Use a set to track unique true match pairs found (to avoid double-counting bidirectional pairs)
        found_true_pairs: set[tuple[str, str]] = set()
        total_pairs_considered = 0

        for entity_id, sorted_cands in entity_candidates.items():
            # Take top-k candidates
            top_k = sorted_cands[:k]
            total_pairs_considered += len(top_k)

            # Count true matches in top-k (using set to deduplicate)
            for candidate in top_k:
                if _is_true_match(candidate.left.id, candidate.right.id, true_pairs):
                    # Add as sorted tuple to deduplicate (1,2) and (2,1)
                    pair = tuple(sorted([candidate.left.id, candidate.right.id]))
                    found_true_pairs.add(pair)

        # Compute metrics
        recall = len(found_true_pairs) / total_true_matches
        avg_pairs = total_pairs_considered / num_entities if num_entities > 0 else 0.0

        recall_values.append(recall)
        avg_pairs_values.append(avg_pairs)

        logger.debug("k=%d: recall=%.3f, avg_pairs=%.2f", k, recall, avg_pairs)

    return RecallCurveStats(
        k_values=k_values, recall_values=recall_values, avg_pairs_values=avg_pairs_values
    )


def evaluate_blocker_detailed(
    candidates: list[ERCandidate[Any]],
    gold_clusters: list[set[str]],
    k_values: list[int] | None = None,
) -> BlockerEvaluationReport:
    """Comprehensive blocker evaluation with all metrics.

    Computes:
    - Candidate metrics (recall, precision) via evaluate_blocking()
    - Ranking metrics (MAP, MRR, NDCG) via evaluate_blocking_with_ranking()
    - Score distribution via _compute_score_metrics()
    - Rank distribution via _compute_rank_metrics()
    - Recall curve via _compute_recall_curve()

    Args:
        candidates: Candidate pairs with similarity_score
        gold_clusters: Ground truth entity clusters
        k_values: K values for recall curve (default: [1, 5, 10, 20, 50])

    Returns:
        BlockerEvaluationReport with all metric categories
    """
    if k_values is None:
        k_values = [1, 5, 10, 20, 50]

    logger.info("Starting comprehensive blocker evaluation with %d candidates", len(candidates))

    # 1. Compute basic candidate stats
    basic_stats = evaluate_blocking(candidates, gold_clusters)
    candidate_metrics = CandidateMetrics(
        recall=basic_stats.candidate_recall,
        precision=basic_stats.candidate_precision,
        total=basic_stats.total_candidates,
        avg_per_entity=basic_stats.avg_candidates_per_entity,
        missed_matches=basic_stats.missed_matches_count,
        false_positives=basic_stats.false_positive_candidates_count,
    )

    # 2. Compute ranking metrics
    ranking_results = evaluate_blocking_with_ranking(candidates, gold_clusters, k_values)
    ranking_metrics = RankingMetrics(
        map=ranking_results["map"],
        mrr=ranking_results["mrr"],
        ndcg_at_10=ranking_results.get("ndcg@10", 0.0),
        ndcg_at_20=ranking_results.get("ndcg@20", 0.0),
        recall_at_5=ranking_results.get("recall@5", 0.0),
        recall_at_10=ranking_results.get("recall@10", 0.0),
        recall_at_20=ranking_results.get("recall@20", 0.0),
    )

    # 3. Compute distributions
    score_metrics = _compute_score_metrics(candidates, gold_clusters)
    rank_metrics = _compute_rank_metrics(candidates, gold_clusters)
    recall_curve = _compute_recall_curve(candidates, gold_clusters, k_values)

    logger.info(
        "Blocker evaluation complete: recall=%.3f, MAP=%.3f",
        candidate_metrics.recall,
        ranking_metrics.map,
    )

    report = BlockerEvaluationReport(
        candidates=candidate_metrics,
        ranking=ranking_metrics,
        scores=score_metrics,
        rank_distribution=rank_metrics,
        recall_curve=recall_curve,
    )

    # Store gold clusters for .diagnose() method
    # Use private attr to avoid breaking serialization
    object.__setattr__(report, "_gold_clusters", gold_clusters)

    return report


def extract_missed_matches(
    candidates: list[ERCandidate[Any]],
    gold_clusters: list[set[str]],
    entities: dict[str, Any],
    n: int = 20,
) -> list["MissedMatchExample"]:
    """Extract missed matches (true pairs not found by blocker).

    Similar to scikit-learn's pattern of extracting misclassified samples,
    this identifies true matches that the blocker failed to retrieve.

    Args:
        candidates: Candidate pairs generated by blocker
        gold_clusters: Ground truth entity clusters
        entities: Dict mapping entity ID to entity data (for text extraction)
        n: Maximum number of examples to return

    Returns:
        List of MissedMatchExample instances (up to n)

    Example:
        >>> missed = extract_missed_matches(candidates, gold_clusters, entities, n=10)
        >>> for ex in missed:
        ...     print(f"Missed: {ex.left_text} <-> {ex.right_text}")
    """
    from langres.core.diagnostics import MissedMatchExample

    # Build ground truth pairs
    true_pairs = _build_ground_truth_pairs(gold_clusters)

    # Build candidate pairs
    candidate_pairs = set()
    for c in candidates:
        pair = tuple(sorted([c.left.id, c.right.id]))
        candidate_pairs.add(pair)

    # Find missed matches
    missed_pairs = true_pairs - candidate_pairs

    # Build cluster lookup
    entity_to_cluster = {}
    for cluster_id, cluster in enumerate(gold_clusters):
        for entity_id in cluster:
            entity_to_cluster[entity_id] = cluster_id

    # Convert to examples
    examples = []
    for left_id, right_id in list(missed_pairs)[:n]:
        # Extract text (handle missing entities gracefully)
        left_entity = entities.get(left_id, {})
        right_entity = entities.get(right_id, {})

        left_text = (
            left_entity.get("name", left_id) if isinstance(left_entity, dict) else str(left_id)
        )
        right_text = (
            right_entity.get("name", right_id) if isinstance(right_entity, dict) else str(right_id)
        )

        examples.append(
            MissedMatchExample(
                left_id=left_id,
                left_text=left_text,
                right_id=right_id,
                right_text=right_text,
                cluster_id=entity_to_cluster.get(left_id, -1),
            )
        )

    logger.debug(
        "Extracted %d missed match examples (total missed: %d)", len(examples), len(missed_pairs)
    )
    return examples


def extract_false_positives(
    candidates: list[ERCandidate[Any]],
    gold_clusters: list[set[str]],
    entities: dict[str, Any],
    n: int = 20,
    min_score: float = 0.7,
) -> list["FalsePositiveExample"]:
    """Extract false positives (high-scoring non-matches).

    Identifies candidate pairs with high similarity scores that are NOT
    in the same gold cluster (i.e., blocker was overconfident).

    Args:
        candidates: Candidate pairs with similarity scores
        gold_clusters: Ground truth entity clusters
        entities: Dict mapping entity ID to entity data
        n: Maximum number of examples to return
        min_score: Minimum score threshold for "high-scoring" (default: 0.7)

    Returns:
        List of FalsePositiveExample instances (up to n)
    """
    from langres.core.diagnostics import FalsePositiveExample

    # Build ground truth pairs
    true_pairs = _build_ground_truth_pairs(gold_clusters)

    # Find high-scoring false positives
    false_positives = []
    for c in candidates:
        # Skip if no score
        if c.similarity_score is None:
            continue

        # Skip if score below threshold
        if c.similarity_score < min_score:
            continue

        # Check if it's a false positive
        pair = tuple(sorted([c.left.id, c.right.id]))
        if pair not in true_pairs:
            # Extract text
            left_entity = entities.get(c.left.id, {})
            right_entity = entities.get(c.right.id, {})

            left_text = (
                left_entity.get("name", c.left.id)
                if isinstance(left_entity, dict)
                else str(c.left.id)
            )
            right_text = (
                right_entity.get("name", c.right.id)
                if isinstance(right_entity, dict)
                else str(c.right.id)
            )

            false_positives.append(
                FalsePositiveExample(
                    left_id=c.left.id,
                    left_text=left_text,
                    right_id=c.right.id,
                    right_text=right_text,
                    score=c.similarity_score,
                )
            )

    # Sort by score descending, take top n
    false_positives.sort(key=lambda ex: ex.score, reverse=True)
    result = false_positives[:n]

    logger.debug(
        "Extracted %d false positive examples (total FPs: %d)", len(result), len(false_positives)
    )
    return result
