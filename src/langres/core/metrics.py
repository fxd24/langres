"""Evaluation metrics for entity resolution pipelines.

This module provides metrics for evaluating different pipeline stages:
- Blocking stage: evaluate_blocking()
- Clustering stage: evaluate_clustering(), calculate_bcubed_metrics(), calculate_pairwise_metrics()

References:
    Amigó, E., Gonzalo, J., Artiles, J., & Verdejo, F. (2009).
    A comparison of extrinsic clustering evaluation metrics based on formal constraints.
    Information Retrieval, 12(4), 461-486.
"""

from typing import Any

from langres.core.debugging import CandidateStats
from langres.core.models import ERCandidate


def calculate_bcubed_precision(
    predicted_clusters: list[set[str]], gold_clusters: list[set[str]]
) -> float:
    """Calculate BCubed Precision.

    BCubed Precision measures how many items in each predicted cluster
    belong to the same gold cluster. It is computed as the average precision
    across all items.

    Args:
        predicted_clusters: List of predicted entity clusters (sets of entity IDs)
        gold_clusters: List of gold-standard entity clusters (sets of entity IDs)

    Returns:
        BCubed Precision score in range [0.0, 1.0]

    Example:
        predicted = [{"e1", "e2"}]
        gold = [{"e1", "e2"}]
        precision = calculate_bcubed_precision(predicted, gold)  # 1.0
    """
    # Build gold cluster lookup: entity_id -> cluster_id
    gold_lookup: dict[str, int] = {}
    for cluster_id, cluster in enumerate(gold_clusters):
        for entity_id in cluster:
            gold_lookup[entity_id] = cluster_id

    # Calculate precision for each entity
    total_precision = 0.0
    entity_count = 0

    for pred_cluster in predicted_clusters:
        for entity in pred_cluster:
            # Count how many entities in this predicted cluster share
            # the same gold cluster as this entity
            same_cluster_count = sum(
                1 for other in pred_cluster if gold_lookup.get(entity) == gold_lookup.get(other)
            )

            # Precision for this entity = same_cluster / predicted_cluster_size
            precision = same_cluster_count / len(pred_cluster)
            total_precision += precision
            entity_count += 1

    return total_precision / entity_count if entity_count > 0 else 0.0


def calculate_bcubed_recall(
    predicted_clusters: list[set[str]], gold_clusters: list[set[str]]
) -> float:
    """Calculate BCubed Recall.

    BCubed Recall measures how many items from the same gold cluster
    are placed in the same predicted cluster. It is computed as the average
    recall across all items.

    Args:
        predicted_clusters: List of predicted entity clusters (sets of entity IDs)
        gold_clusters: List of gold-standard entity clusters (sets of entity IDs)

    Returns:
        BCubed Recall score in range [0.0, 1.0]

    Example:
        predicted = [{"e1"}, {"e2"}]  # All separate
        gold = [{"e1", "e2"}]  # Should be together
        recall = calculate_bcubed_recall(predicted, gold)  # 0.5
    """
    # Build predicted cluster lookup: entity_id -> cluster_id
    pred_lookup: dict[str, int] = {}
    for cluster_id, cluster in enumerate(predicted_clusters):
        for entity_id in cluster:
            pred_lookup[entity_id] = cluster_id

    # Calculate recall for each entity
    total_recall = 0.0
    entity_count = 0

    for gold_cluster in gold_clusters:
        for entity in gold_cluster:
            # Count how many entities in this gold cluster are also
            # in the same predicted cluster as this entity
            same_cluster_count = sum(
                1 for other in gold_cluster if pred_lookup.get(entity) == pred_lookup.get(other)
            )

            # Recall for this entity = same_cluster / gold_cluster_size
            recall = same_cluster_count / len(gold_cluster)
            total_recall += recall
            entity_count += 1

    return total_recall / entity_count if entity_count > 0 else 0.0


def calculate_bcubed_metrics(
    predicted_clusters: list[set[str]], gold_clusters: list[set[str]]
) -> dict[str, float]:
    """Calculate BCubed Precision, Recall, and F1.

    This is the main function for evaluating clustering quality. It computes
    all three BCubed metrics and returns them in a dictionary.

    Args:
        predicted_clusters: List of predicted entity clusters (sets of entity IDs)
        gold_clusters: List of gold-standard entity clusters (sets of entity IDs)

    Returns:
        Dictionary with keys:
        - precision: BCubed Precision score
        - recall: BCubed Recall score
        - f1: BCubed F1 score (harmonic mean of precision and recall)

    Example:
        predicted = [{"c1", "c1_dup"}, {"c2"}]
        gold = [{"c1", "c1_dup"}, {"c2"}]
        metrics = calculate_bcubed_metrics(predicted, gold)
        # {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    """
    precision = calculate_bcubed_precision(predicted_clusters, gold_clusters)
    recall = calculate_bcubed_recall(predicted_clusters, gold_clusters)

    # F1 is harmonic mean of precision and recall
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def calculate_pairwise_metrics(
    predicted_clusters: list[set[str]], gold_clusters: list[set[str]]
) -> dict[str, float]:
    """Calculate pairwise Precision, Recall, and F1.

    Pairwise metrics treat entity resolution as binary classification on pairs:
    each pair of entities is either a "match" (same cluster) or "non-match".
    This provides a complementary perspective to BCubed metrics.

    Args:
        predicted_clusters: List of predicted entity clusters (sets of entity IDs)
        gold_clusters: List of gold-standard entity clusters (sets of entity IDs)

    Returns:
        Dictionary with keys:
        - precision: Pairwise precision (TP / (TP + FP))
        - recall: Pairwise recall (TP / (TP + FN))
        - f1: Pairwise F1 score (harmonic mean of precision and recall)
        - tp: Number of true positive pairs
        - fp: Number of false positive pairs
        - fn: Number of false negative pairs

    Example:
        predicted = [{"e1", "e2"}, {"e3"}]
        gold = [{"e1", "e2"}, {"e3"}]
        metrics = calculate_pairwise_metrics(predicted, gold)
        # {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 1, 'fp': 0, 'fn': 0}
    """
    # Convert clusters to sets of pairs
    predicted_pairs = _clusters_to_pairs(predicted_clusters)
    gold_pairs = _clusters_to_pairs(gold_clusters)

    # Calculate TP, FP, FN
    tp = len(predicted_pairs & gold_pairs)  # True positives: pairs in both
    fp = len(predicted_pairs - gold_pairs)  # False positives: predicted but not gold
    fn = len(gold_pairs - predicted_pairs)  # False negatives: gold but not predicted

    # Calculate precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def pairs_from_clusters(clusters: list[set[str]]) -> set[tuple[str, str]]:
    """Convert clusters to set of entity pairs (public API).

    This function extracts all pairwise entity matches implied by a clustering.
    For each cluster, it generates all pairs of entities within that cluster.
    Pairs are returned in lexicographic order for consistency.

    Args:
        clusters: List of clusters (sets of entity IDs)

    Returns:
        Set of entity pairs (tuples with lexicographic ordering)

    Example:
        >>> clusters = [{"e1", "e2", "e3"}, {"e4", "e5"}]
        >>> pairs = pairs_from_clusters(clusters)
        >>> sorted(pairs)
        [('e1', 'e2'), ('e1', 'e3'), ('e2', 'e3'), ('e4', 'e5')]
    """
    return _clusters_to_pairs(clusters)


def _clusters_to_pairs(clusters: list[set[str]]) -> set[tuple[str, str]]:
    """Convert clusters to set of entity pairs.

    Args:
        clusters: List of clusters (sets of entity IDs)

    Returns:
        Set of entity pairs (tuples with lexicographic ordering)

    Example:
        clusters = [{"e1", "e2", "e3"}, {"e4", "e5"}]
        pairs = _clusters_to_pairs(clusters)
        # {("e1", "e2"), ("e1", "e3"), ("e2", "e3"), ("e4", "e5")}
    """
    pairs: set[tuple[str, str]] = set()
    for cluster in clusters:
        # Generate all pairs within the cluster
        cluster_list = sorted(cluster)  # Sort for consistent ordering
        for i in range(len(cluster_list)):
            for j in range(i + 1, len(cluster_list)):
                # Store pairs in lexicographic order (smaller ID first)
                pair = (cluster_list[i], cluster_list[j])
                pairs.add(pair)
    return pairs


def evaluate_blocking(
    candidates: list[ERCandidate[Any]],
    gold_clusters: list[set[str]],
) -> CandidateStats:
    """Evaluate blocking stage performance.

    Measures how well the blocker captures true duplicate pairs. This function
    provides a pure, stateless evaluation following the sklearn metrics pattern.

    Args:
        candidates: List of candidate pairs generated by blocker
        gold_clusters: List of ground truth entity clusters (sets of entity IDs)

    Returns:
        CandidateStats with blocking metrics:
        - total_candidates: Number of candidate pairs generated
        - avg_candidates_per_entity: Average candidates per entity
        - candidate_recall: % of true matches captured (TP / (TP + FN))
        - candidate_precision: % of candidates that are true matches (TP / (TP + FP))
        - missed_matches_count: Number of true matches not captured (FN)
        - false_positive_candidates_count: Number of incorrect candidates (FP)

    Example:
        >>> from langres.core.metrics import evaluate_blocking
        >>> blocker = VectorBlocker(...)
        >>> candidates = list(blocker.stream(data))
        >>> stats = evaluate_blocking(candidates, gold_clusters)
        >>> print(f"Blocking recall: {stats.candidate_recall:.2%}")
        >>> print(f"Blocking precision: {stats.candidate_precision:.2%}")

    Note:
        This is a pure function that can be called independently or via the
        convenience method blocker.evaluate(candidates, gold_clusters).

    Note:
        Blocking recall is critical for ER pipelines. If recall is too low
        (<90%), the pipeline cannot recover missed matches downstream.
    """
    # Convert gold clusters to pairs
    gold_pairs = pairs_from_clusters(gold_clusters)

    # Convert candidates to pairs
    candidate_pairs: set[tuple[str, str]] = set()
    for c in candidates:
        left_id = str(c.left.id)
        right_id = str(c.right.id)
        pair = tuple(sorted([left_id, right_id]))
        candidate_pairs.add((pair[0], pair[1]))

    # Calculate TP, FP, FN
    tp = len(gold_pairs & candidate_pairs)
    fp = len(candidate_pairs - gold_pairs)
    fn = len(gold_pairs - candidate_pairs)

    # Calculate precision and recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Calculate average candidates per entity
    # Count total unique entities in gold clusters
    total_entities = len({e_id for cluster in gold_clusters for e_id in cluster})
    avg_candidates_per_entity = (
        len(candidate_pairs) * 2 / total_entities if total_entities > 0 else 0.0
    )

    return CandidateStats(
        total_candidates=len(candidate_pairs),
        avg_candidates_per_entity=avg_candidates_per_entity,
        candidate_recall=recall,
        candidate_precision=precision,
        missed_matches_count=fn,
        false_positive_candidates_count=fp,
    )


def evaluate_clustering(
    predicted_clusters: list[set[str]],
    gold_clusters: list[set[str]],
) -> dict[str, dict[str, float]]:
    """Evaluate clustering quality with comprehensive metrics.

    Computes both BCubed and pairwise metrics for a complete view of clustering
    quality. BCubed metrics are item-based and handle singletons well, while
    pairwise metrics provide a complementary binary classification perspective.

    Args:
        predicted_clusters: List of predicted entity clusters (sets of entity IDs)
        gold_clusters: List of gold-standard entity clusters (sets of entity IDs)

    Returns:
        Dictionary with two keys:
        - 'bcubed': BCubed metrics (precision, recall, f1)
        - 'pairwise': Pairwise metrics (precision, recall, f1, tp, fp, fn)

    Example:
        >>> from langres.core.metrics import evaluate_clustering
        >>> clusterer = Clusterer(threshold=0.7)
        >>> predicted = clusterer.cluster(judgements)
        >>> metrics = evaluate_clustering(predicted, gold_clusters)
        >>> print(f"BCubed F1: {metrics['bcubed']['f1']:.2%}")
        >>> print(f"Pairwise F1: {metrics['pairwise']['f1']:.2%}")

    Note:
        This is a pure function that can be called independently or via the
        convenience method clusterer.evaluate(predicted, gold_clusters).

    Note:
        BCubed and pairwise metrics can differ significantly:
        - BCubed is more forgiving of singleton errors
        - Pairwise treats each pair as equally important
        Both perspectives are valuable for understanding clustering quality.
    """
    return {
        "bcubed": calculate_bcubed_metrics(predicted_clusters, gold_clusters),
        "pairwise": calculate_pairwise_metrics(predicted_clusters, gold_clusters),
    }
