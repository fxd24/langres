"""Evaluation metrics for entity resolution pipelines.

This module provides metrics for evaluating different pipeline stages:
- Blocking stage: evaluate_blocking(), evaluate_blocking_with_ranking()
- Clustering stage: evaluate_clustering(), calculate_bcubed_metrics(), calculate_pairwise_metrics()

References:
    Amigó, E., Gonzalo, J., Artiles, J., & Verdejo, F. (2009).
    A comparison of extrinsic clustering evaluation metrics based on formal constraints.
    Information Retrieval, 12(4), 461-486.
"""

from typing import Any

from ranx import Qrels, Run, evaluate  # type: ignore[import-untyped]

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


def evaluate_blocking_with_ranking(
    candidates: list[ERCandidate[Any]],
    gold_clusters: list[set[str]],
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """Evaluate blocking stage with ranking metrics (MAP, MRR, NDCG@K, Recall@K, Precision@K).

    This function extends evaluate_blocking() by computing ranking metrics that measure
    HOW WELL true matches are ranked by the blocker. This is critical for budget-constrained
    downstream processing (e.g., LLM judging) where we want to process the most promising
    candidates first.

    Args:
        candidates: List of candidate pairs with similarity_score populated
        gold_clusters: List of ground truth entity clusters (sets of entity IDs)
        k_values: List of K values for Recall@K and Precision@K metrics.
            Defaults to [20] if not specified.

    Returns:
        Dictionary with ranking metrics:
        - map: Mean Average Precision (0-1, higher is better)
        - mrr: Mean Reciprocal Rank (0-1, higher is better)
        - ndcg_at_K: NDCG@K for each K in k_values (0-1, higher is better)
        - recall_at_K: Recall@K for each K (0-1, higher is better)
        - precision_at_K: Precision@K for each K (0-1, higher is better)
        - total_candidates: Number of candidate pairs
        - avg_candidates_per_entity: Average candidates per entity

    Raises:
        ValueError: If any candidate is missing similarity_score

    Example:
        >>> from langres.core.metrics import evaluate_blocking_with_ranking
        >>> blocker = VectorBlocker(...)
        >>> candidates = list(blocker.stream(data))  # with similarity_score
        >>> metrics = evaluate_blocking_with_ranking(candidates, gold_clusters)
        >>> print(f"MAP: {metrics['map']:.3f}")
        >>> print(f"MRR: {metrics['mrr']:.3f}")
        >>> print(f"NDCG@20: {metrics['ndcg_at_20']:.3f}")

    Note:
        This function requires candidates to have similarity_score populated.
        VectorBlocker.stream() automatically populates this field.

    Note:
        Ranking metrics complement precision/recall metrics:
        - Precision/Recall: "Are true matches in the candidates?"
        - Ranking: "Are true matches ranked highly?"
    """
    if k_values is None:
        k_values = [20]

    # Validate that all candidates have similarity scores
    for candidate in candidates:
        if candidate.similarity_score is None:
            raise ValueError(
                "evaluate_blocking_with_ranking requires similarity_score to be populated "
                "in all candidates. VectorBlocker.stream() automatically populates this field."
            )

    # Convert gold clusters to pairs for relevance judgments
    gold_pairs = pairs_from_clusters(gold_clusters)

    # Handle empty candidates edge case
    if len(candidates) == 0:
        result: dict[str, Any] = {
            "map": 0.0,
            "mrr": 0.0,
            "total_candidates": 0,
            "avg_candidates_per_entity": 0.0,
        }
        for k in k_values:
            result[f"ndcg_at_{k}"] = 0.0
            result[f"recall_at_{k}"] = 0.0
            result[f"precision_at_{k}"] = 0.0
        return result

    # Build per-entity candidate lists (for ranking evaluation)
    # Structure: {entity_id: [(candidate_id, similarity_score), ...]}
    entity_rankings: dict[str, list[tuple[str, float]]] = {}

    for candidate in candidates:
        left_id = str(candidate.left.id)
        right_id = str(candidate.right.id)
        score = candidate.similarity_score

        assert score is not None  # Already validated above

        # Add to left entity's ranking
        if left_id not in entity_rankings:
            entity_rankings[left_id] = []
        entity_rankings[left_id].append((right_id, score))

        # Add to right entity's ranking (bidirectional)
        if right_id not in entity_rankings:
            entity_rankings[right_id] = []
        entity_rankings[right_id].append((left_id, score))

    # Sort each entity's candidates by similarity (descending)
    for entity_id in entity_rankings:
        entity_rankings[entity_id].sort(key=lambda x: x[1], reverse=True)

    # Convert to ranx format for NDCG and MRR
    # Qrels: {query_id: {doc_id: relevance}}
    # Run: {query_id: {doc_id: score}}
    qrels_dict: dict[str, dict[str, int]] = {}
    run_dict: dict[str, dict[str, float]] = {}

    for entity_id, ranked_candidates in entity_rankings.items():
        # Build relevance judgments (qrels)
        qrels_dict[entity_id] = {}
        for candidate_id, _ in ranked_candidates:
            # Check if (entity_id, candidate_id) is a true match
            pair = tuple(sorted([entity_id, candidate_id]))
            is_relevant = pair in gold_pairs
            qrels_dict[entity_id][candidate_id] = 1 if is_relevant else 0

        # Build run (predictions with scores)
        run_dict[entity_id] = {candidate_id: score for candidate_id, score in ranked_candidates}

    # Create ranx objects
    qrels = Qrels(qrels_dict)
    run = Run(run_dict)

    # Compute ranx metrics (MRR, NDCG@K)
    ranx_metrics = evaluate(
        qrels,
        run,
        metrics=["mrr", "map"] + [f"ndcg@{k}" for k in k_values],
    )

    # Compute Recall@K and Precision@K manually
    recall_at_k = {}
    precision_at_k = {}

    for k in k_values:
        total_recall = 0.0
        total_precision = 0.0
        num_queries = 0

        for entity_id, ranked_candidates in entity_rankings.items():
            # Get top-K candidates
            top_k = ranked_candidates[:k]

            # Count true matches in top-K
            true_matches_in_top_k = 0
            for candidate_id, _ in top_k:
                pair = tuple(sorted([entity_id, candidate_id]))
                if pair in gold_pairs:
                    true_matches_in_top_k += 1

            # Count total true matches for this entity
            total_true_matches = sum(
                1
                for candidate_id, _ in ranked_candidates
                if tuple(sorted([entity_id, candidate_id])) in gold_pairs
            )

            # Recall@K = (true matches in top-K) / (total true matches)
            if total_true_matches > 0:
                recall = true_matches_in_top_k / total_true_matches
                total_recall += recall

            # Precision@K = (true matches in top-K) / K
            if len(top_k) > 0:
                precision = true_matches_in_top_k / len(top_k)
                total_precision += precision

            num_queries += 1

        # Average across all queries
        recall_at_k[k] = total_recall / num_queries if num_queries > 0 else 0.0
        precision_at_k[k] = total_precision / num_queries if num_queries > 0 else 0.0

    # Calculate average candidates per entity
    total_entities = len({e_id for cluster in gold_clusters for e_id in cluster})
    avg_candidates_per_entity = len(candidates) * 2 / total_entities if total_entities > 0 else 0.0

    # Build result dictionary
    result = {
        "map": ranx_metrics.get("map", 0.0),
        "mrr": ranx_metrics.get("mrr", 0.0),
        "total_candidates": len(candidates),
        "avg_candidates_per_entity": avg_candidates_per_entity,
    }

    # Add NDCG@K metrics
    for k in k_values:
        result[f"ndcg_at_{k}"] = ranx_metrics.get(f"ndcg@{k}", 0.0)

    # Add Recall@K and Precision@K metrics
    for k in k_values:
        result[f"recall_at_{k}"] = recall_at_k[k]
        result[f"precision_at_{k}"] = precision_at_k[k]

    return result
