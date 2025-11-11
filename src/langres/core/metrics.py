"""
BCubed evaluation metrics for clustering quality assessment.

This module implements BCubed Precision, Recall, and F1 metrics for evaluating
entity resolution clustering quality. BCubed metrics are item-based (rather than
cluster-based) and handle overlapping clusters gracefully.

References:
    Amigó, E., Gonzalo, J., Artiles, J., & Verdejo, F. (2009).
    A comparison of extrinsic clustering evaluation metrics based on formal constraints.
    Information Retrieval, 12(4), 461-486.
"""


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
