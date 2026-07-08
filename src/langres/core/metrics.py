"""Evaluation metrics for entity resolution pipelines.

This module provides metrics for evaluating different pipeline stages:
- Blocking stage: evaluate_blocking(), evaluate_blocking_with_ranking()
- Clustering stage: evaluate_clustering(), calculate_bcubed_metrics(), calculate_pairwise_metrics()
- Agreement (rater-vs-rater): cohens_kappa(), matthews_corrcoef()
- Calibration (confidence-vs-outcome): brier_score(), expected_calibration_error(),
  reliability_bins()

References:
    Amigó, E., Gonzalo, J., Artiles, J., & Verdejo, F. (2009).
    A comparison of extrinsic clustering evaluation metrics based on formal constraints.
    Information Retrieval, 12(4), 461-486.
"""

import math
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel
from ranx import Qrels, Run, evaluate  # type: ignore[import-untyped]

from langres.core.debugging import CandidateStats
from langres.core.models import ERCandidate, PairwiseJudgement

BinStrategy = Literal["quantile", "uniform"]
"""Binning strategy for calibration metrics: equal-mass quantile bins or equal-width bins."""


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
    # Build gold cluster lookup: entity_id (str) -> cluster_id (int). The str/int
    # key-vs-value type split is load-bearing for the partition-safe fallback
    # below: ``get(entity, entity)`` yields the str id for gold-absent entities,
    # which can never equal an int cluster id, so two absent ids stay distinct.
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
            # the same gold cluster as this entity. Any entity absent from the
            # gold partition falls back to its own id as a unique singleton key
            # (ids are str, gold cluster keys are int, so no collision): two
            # gold-absent ids must NOT compare equal via a shared ``None``.
            same_cluster_count = sum(
                1
                for other in pred_cluster
                if gold_lookup.get(entity, entity) == gold_lookup.get(other, other)
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
    # Build predicted cluster lookup: entity_id (str) -> cluster_id (int). The
    # str/int key-vs-value type split is load-bearing for the partition-safe
    # fallback below: ``get(entity, entity)`` yields the str id for un-clustered
    # entities, which can never equal an int cluster id, so two absent ids stay
    # distinct (instead of colliding on a shared ``None``).
    pred_lookup: dict[str, int] = {}
    for cluster_id, cluster in enumerate(predicted_clusters):
        for entity_id in cluster:
            pred_lookup[entity_id] = cluster_id

    # Calculate recall for each entity
    total_recall = 0.0
    entity_count = 0

    for gold_cluster in gold_clusters:
        for entity in gold_cluster:
            # Count how many entities in this gold cluster are also in the same
            # predicted cluster as this entity. Any entity absent from all
            # predicted clusters (the Clusterer drops singletons) falls back to
            # its own id as a unique singleton key (ids are str, predicted
            # cluster keys are int, so no collision): two un-merged ids must NOT
            # compare equal via a shared ``None`` and inflate recall.
            same_cluster_count = sum(
                1
                for other in gold_cluster
                if pred_lookup.get(entity, entity) == pred_lookup.get(other, other)
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


class PairMetrics(BaseModel):
    """Pair-level (pre-clustering) classification metrics for one threshold.

    Unlike :func:`calculate_pairwise_metrics` (which scores pairs *after*
    clustering, so transitive closure can chain one false-positive edge into many
    false-positive pairs), these metrics classify each candidate
    :class:`~langres.core.models.PairwiseJudgement` directly against the gold
    match pairs at a fixed score threshold. This isolates the *scorer's* quality
    from the clusterer's amplification, giving an unbiased ranking signal across
    judges of differing recall.

    Attributes:
        threshold: Score threshold applied; a judgement is a predicted match iff
            ``score >= threshold``.
        precision: ``tp / (tp + fp)`` (0.0 when nothing is predicted).
        recall: ``tp / (tp + fn)`` (0.0 when there are no gold pairs).
        f1: Harmonic mean of precision and recall (0.0 when both are 0).
        tp: Predicted-match pairs that are gold matches.
        fp: Predicted-match pairs that are not gold matches.
        fn: Gold match pairs not predicted (missed by the blocker or rejected by
            the threshold).
    """

    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def classify_pairs(
    judgements: list[PairwiseJudgement],
    gold_pairs: set[frozenset[str]],
    threshold: float,
) -> PairMetrics:
    """Classify candidate judgements against gold match pairs at one threshold.

    Each judgement is a predicted match iff ``score >= threshold``. A judgement's
    pair is identified order-independently by ``frozenset({left_id, right_id})``,
    so candidate ordering never affects the counts. ``fn`` counts gold pairs that
    were *not* predicted — covering both pairs the blocker never surfaced as a
    judgement and pairs scored below ``threshold`` — because ``gold_pairs - {predicted}``
    is exactly the set of unrecovered true matches.

    Args:
        judgements: Candidate judgements from a scorer (pre-clustering).
        gold_pairs: True match pairs as order-independent ``frozenset`` pairs.
        threshold: Match cut-off applied to each judgement's ``score``.

    Returns:
        A :class:`PairMetrics` for this threshold.
    """
    predicted: set[frozenset[str]] = {
        frozenset({j.left_id, j.right_id}) for j in judgements if j.score >= threshold
    }
    tp = len(predicted & gold_pairs)
    fp = len(predicted - gold_pairs)
    fn = len(gold_pairs - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return PairMetrics(
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
    )


def pair_pr_curve(
    judgements: list[PairwiseJudgement],
    gold_pairs: set[frozenset[str]],
    grid: Sequence[float],
) -> list[PairMetrics]:
    """Pair-level precision/recall/F1 across a grid of thresholds.

    Calls :func:`classify_pairs` once per threshold in ``grid`` (preserving its
    order), so callers can pick the threshold maximizing pair-level F1 or trace
    the precision/recall trade-off without re-clustering.

    Args:
        judgements: Candidate judgements from a scorer (pre-clustering).
        gold_pairs: True match pairs as order-independent ``frozenset`` pairs.
        grid: Thresholds to evaluate.

    Returns:
        One :class:`PairMetrics` per threshold, in ``grid`` order.

    Raises:
        ValueError: If ``grid`` is empty.
    """
    if not grid:
        raise ValueError("grid is empty; nothing to sweep over")
    return [classify_pairs(judgements, gold_pairs, t) for t in grid]


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


def _all_possible_pairs(
    *,
    num_records: int | None,
    n_left: int | None,
    n_right: int | None,
) -> int:
    """Count of all possible pairs for a dedup or cross-source blocking setting.

    Exactly one mode must be specified:

    - **Dedup** (single corpus of ``num_records``): ``n * (n - 1) / 2``.
    - **Cross-source** (two sources of sizes ``n_left`` and ``n_right``):
      ``n_left * n_right``.

    Returns:
        The number of candidate pairs an exhaustive (all-pairs) blocker would
        emit. ``0`` for a corpus of 0 or 1 record, or when either source is empty.

    Raises:
        ValueError: If both modes (or neither) are specified, if a cross-source
            mode is missing one side, or if any count is negative.
    """
    cross = n_left is not None or n_right is not None
    dedup = num_records is not None
    if cross and dedup:
        raise ValueError(
            "specify either num_records (dedup) or n_left/n_right (cross-source), not both"
        )
    if cross:
        if n_left is None or n_right is None:
            raise ValueError("cross-source reduction ratio needs both n_left and n_right")
        if n_left < 0 or n_right < 0:
            raise ValueError(f"n_left and n_right must be non-negative, got {n_left} and {n_right}")
        return n_left * n_right
    if dedup:
        assert num_records is not None  # narrowed by ``dedup`` above (for mypy)
        if num_records < 0:
            raise ValueError(f"num_records must be non-negative, got {num_records}")
        return num_records * (num_records - 1) // 2
    raise ValueError("provide num_records (dedup) or both n_left and n_right (cross-source)")


def reduction_ratio(
    num_candidate_pairs: int,
    *,
    num_records: int | None = None,
    n_left: int | None = None,
    n_right: int | None = None,
) -> float:
    """Reduction Ratio (RR): the classic blocking-efficiency metric.

    ``RR = 1 - num_candidate_pairs / all_possible_pairs`` -- the fraction of the
    full pairwise comparison space a blocker prunes away. Higher is better
    (fewer candidates to judge), but RR must be read *alongside* pair
    completeness / :func:`evaluate_blocking`'s ``candidate_recall``: a blocker
    that emits nothing has ``RR = 1.0`` yet recovers no matches. The count of
    all possible pairs depends on the setting (specify exactly one mode):

    - **Dedup** (single corpus): pass ``num_records=n`` -> ``all = n(n-1)/2``.
    - **Cross-source linkage** (sources A, B): pass ``n_left=|A|`` and
      ``n_right=|B|`` -> ``all = |A| * |B|``.

    The true record counts must be supplied explicitly: RR cannot be inferred
    from the candidate pairs alone, because records with zero candidates never
    appear in any pair and would be silently undercounted.

    Args:
        num_candidate_pairs: Number of candidate pairs the blocker emitted.
        num_records: Corpus size ``n`` for the dedup setting.
        n_left: Size of source A for the cross-source setting.
        n_right: Size of source B for the cross-source setting.

    Returns:
        RR in ``[0.0, 1.0]`` when candidates are a subset of all possible pairs.
        Returns ``0.0`` by convention when there are 0 all-possible pairs (a
        corpus of 0 or 1 record, or an empty source) -- there is nothing to
        reduce, so no efficiency gain is claimed. (A larger candidate count than
        the possible-pair total -- a sign the record counts are wrong -- yields a
        negative value rather than being clamped, so the mistake stays visible.)

    Raises:
        ValueError: If ``num_candidate_pairs`` is negative, if both modes (or
            neither) are given, if a cross-source side is missing, or if any
            count is negative.

    Example:
        >>> reduction_ratio(10, num_records=100)  # 10 of 4950 possible
        0.9979797979797979
        >>> reduction_ratio(6, n_left=3, n_right=4)  # 6 of 12 possible
        0.5
    """
    if num_candidate_pairs < 0:
        raise ValueError(f"num_candidate_pairs must be non-negative, got {num_candidate_pairs}")
    total_possible = _all_possible_pairs(num_records=num_records, n_left=n_left, n_right=n_right)
    if total_possible == 0:
        return 0.0
    return 1.0 - num_candidate_pairs / total_possible


def evaluate_blocking(
    candidates: list[ERCandidate[Any]],
    gold_clusters: list[set[str]],
    *,
    num_records: int | None = None,
    n_left: int | None = None,
    n_right: int | None = None,
) -> CandidateStats:
    """Evaluate blocking stage performance.

    Measures how well the blocker captures true duplicate pairs. This function
    provides a pure, stateless evaluation following the sklearn metrics pattern.

    Args:
        candidates: List of candidate pairs generated by blocker
        gold_clusters: List of ground truth entity clusters (sets of entity IDs)
        num_records: Total corpus size ``n`` for the dedup Reduction Ratio
            (``all_possible = n(n-1)/2``). When omitted (and no cross-source
            sizes are given), ``n`` is derived from the gold clusters, which
            must enumerate every record -- including singletons -- for RR to be
            exact.
        n_left: Size of source A for a cross-source (linkage) Reduction Ratio
            (``all_possible = n_left * n_right``); pass together with ``n_right``.
        n_right: Size of source B for a cross-source Reduction Ratio.

    Returns:
        CandidateStats with blocking metrics:
        - total_candidates: Number of candidate pairs generated
        - avg_candidates_per_entity: Average candidates per entity
        - candidate_recall: % of true matches captured (TP / (TP + FN))
        - candidate_precision: % of candidates that are true matches (TP / (TP + FP))
        - missed_matches_count: Number of true matches not captured (FN)
        - false_positive_candidates_count: Number of incorrect candidates (FP)
        - reduction_ratio: Fraction of all possible pairs pruned by the blocker
          (see :func:`reduction_ratio`)

    Example:
        >>> from langres.core.metrics import evaluate_blocking
        >>> blocker = VectorBlocker(...)
        >>> candidates = list(blocker.stream(data))
        >>> stats = evaluate_blocking(candidates, gold_clusters)
        >>> print(f"Blocking recall: {stats.candidate_recall:.2%}")
        >>> print(f"Blocking precision: {stats.candidate_precision:.2%}")
        >>> print(f"Reduction ratio: {stats.reduction_ratio:.2%}")

    Note:
        This is a pure function that can be called independently or via the
        convenience method blocker.evaluate(candidates, gold_clusters).

    Note:
        Blocking recall is critical for ER pipelines. If recall is too low
        (<90%), the pipeline cannot recover missed matches downstream.

    Note:
        Read ``reduction_ratio`` and ``candidate_recall`` together: RR alone
        rewards a blocker for emitting fewer candidates even when it drops true
        matches. For a cross-source (linkage) blocker, pass ``n_left``/``n_right``
        so RR uses ``|A| * |B|`` rather than the dedup ``n(n-1)/2`` derived from
        the pooled gold clusters.
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

    # Reduction Ratio: prefer explicit record counts (cross-source sizes take
    # precedence, then an explicit dedup n); otherwise derive the dedup n from
    # the gold clusters, which enumerate every gold entity.
    if n_left is not None or n_right is not None:
        rr = reduction_ratio(len(candidate_pairs), n_left=n_left, n_right=n_right)
    else:
        rr = reduction_ratio(
            len(candidate_pairs),
            num_records=num_records if num_records is not None else total_entities,
        )

    return CandidateStats(
        total_candidates=len(candidate_pairs),
        avg_candidates_per_entity=avg_candidates_per_entity,
        candidate_recall=recall,
        candidate_precision=precision,
        missed_matches_count=fn,
        false_positive_candidates_count=fp,
        reduction_ratio=rr,
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


def generalized_merge_distance(
    predicted_clusters: list[set[str]],
    gold_clusters: list[set[str]],
    *,
    merge_cost: float = 1.0,
    split_cost: float = 1.0,
) -> float:
    """Generalized Merge Distance (GMD) between a predicted and gold partition.

    Menestrina, Whang & Garcia-Molina, "Evaluating Entity Resolution Results"
    (VLDB 2010). GMD is the minimum total cost of edits -- *merges* (join two
    clusters) and *splits* (break one cluster in two) -- that transform the
    predicted partition into the gold partition, with independently configurable
    ``merge_cost`` (``c_m``) and ``split_cost`` (``c_s``). Because splitting and
    merging carry separate costs it is *merge/split-asymmetric*: it can penalise
    over-merging differently from over-splitting, unlike the symmetric Pair-F1 /
    BCubed pair, and it is not biased toward either large or small clusters.

    Computed by the paper's single-pass **slice** algorithm: each predicted
    cluster is cut along the gold clusters it spans (its "slices"); separating
    ``k`` slices costs ``(k - 1) * split_cost``, and every slice after the first
    that lands in a gold cluster already receiving mass costs ``merge_cost`` to
    join it. With ``merge_cost = split_cost = 1`` (the default) GMD is the raw
    number of merge+split operations.

    Args:
        predicted_clusters: Predicted partition (clusters as sets of record ids).
        gold_clusters: Gold partition over the *same* record ids.
        merge_cost: Cost ``c_m`` charged per merge operation.
        split_cost: Cost ``c_s`` charged per split operation.

    Returns:
        The total edit cost as a float (``0.0`` for identical partitions, and for
        two empty partitions). Larger means the partitions are further apart.

    Raises:
        ValueError: If the predicted and gold partitions do not cover exactly the
            same set of record ids.

    Example:
        >>> generalized_merge_distance([{"a"}, {"b"}, {"c"}], [{"a", "b", "c"}])
        2.0
        >>> generalized_merge_distance([{"a", "b", "c"}], [{"a"}, {"b"}, {"c"}])
        2.0
    """
    # Map every record to the index of its gold cluster; also validate coverage.
    gold_id: dict[str, int] = {}
    for gid, cluster in enumerate(gold_clusters):
        for rec in cluster:
            gold_id[rec] = gid

    predicted_records = {rec for cluster in predicted_clusters for rec in cluster}
    gold_records = set(gold_id)
    if predicted_records != gold_records:
        only_pred = predicted_records - gold_records
        only_gold = gold_records - predicted_records
        raise ValueError(
            "predicted and gold partitions must cover the same record set; "
            f"predicted-only={sorted(only_pred)}, gold-only={sorted(only_gold)}"
        )

    cost = 0.0
    seen: set[int] = set()  # gold clusters that have already received some mass
    for cluster in predicted_clusters:
        slice_gids = {gold_id[rec] for rec in cluster}
        # Splits needed to separate this predicted cluster into its gold slices.
        cost += split_cost * (len(slice_gids) - 1)
        for gid in slice_gids:
            if gid in seen:
                # This slice must be merged into the gold cluster's existing mass.
                cost += merge_cost
            else:
                seen.add(gid)
    return cost


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


# ---------------------------------------------------------------------------
# Agreement metrics (rater-vs-rater)
# ---------------------------------------------------------------------------
#
# These treat two boolean label vectors (e.g. teacher labels vs. ground truth)
# as a binary classification and quantify *agreement beyond chance*. The
# positive class is ``True`` (a "match").


def _validate_binary(y_true: list[bool], y_pred: list[bool]) -> None:
    """Validate two boolean label vectors share a non-zero, equal length.

    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have equal length, got {len(y_true)} and {len(y_pred)}"
        )
    if not y_true:
        raise ValueError("y_true and y_pred must be non-empty")


def cohens_kappa(y_true: list[bool], y_pred: list[bool]) -> float:
    """Cohen's kappa: chance-corrected agreement between two boolean raters.

    ``kappa = (p_o - p_e) / (1 - p_e)`` where ``p_o`` is observed agreement and
    ``p_e`` is the agreement expected by chance given each rater's marginal
    class frequencies.

    Caveat -- the *prevalence paradox*: under the heavily skewed class balance
    of bootstrap labeling (~2% positives), kappa can be near zero even when raw
    agreement is very high, because chance agreement ``p_e`` is itself near the
    observed agreement. Report :func:`matthews_corrcoef` alongside it (W5).

    Args:
        y_true: Ground-truth boolean labels (positive class is ``True``).
        y_pred: Predicted boolean labels, aligned with ``y_true``.

    Returns:
        Kappa in ``[-1.0, 1.0]``. Returns ``0.0`` when chance agreement is
        perfect (``p_e == 1``, i.e. a rater has no class variance), where kappa
        is otherwise undefined (``0 / 0``).

    Raises:
        ValueError: If inputs differ in length or are empty.

    Example:
        >>> cohens_kappa([True, True, False, False], [True, False, False, False])
        0.5
    """
    _validate_binary(y_true, y_pred)
    n = len(y_true)

    p_observed = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p) / n
    p_true_pos = sum(y_true) / n
    p_pred_pos = sum(y_pred) / n
    p_expected = p_true_pos * p_pred_pos + (1.0 - p_true_pos) * (1.0 - p_pred_pos)

    # p_expected == 1.0 only when a rater is entirely one class, in which case
    # both marginal products are 0 or 1 exactly -- the == comparison is safe.
    denominator = 1.0 - p_expected
    if denominator == 0.0:
        # No class variance in at least one rater -> kappa undefined; convention 0.0.
        return 0.0
    return (p_observed - p_expected) / denominator


def matthews_corrcoef(y_true: list[bool], y_pred: list[bool]) -> float:
    """Matthews correlation coefficient (MCC) for two boolean label vectors.

    ``MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))``. MCC is a
    balanced measure that stays informative under the ~2% positive prevalence of
    bootstrap labeling, where Cohen's kappa suffers the prevalence paradox (W5).

    Args:
        y_true: Ground-truth boolean labels (positive class is ``True``).
        y_pred: Predicted boolean labels, aligned with ``y_true``.

    Returns:
        MCC in ``[-1.0, 1.0]``. Returns ``0.0`` when the denominator is zero
        (any of the four marginal sums is empty), where MCC is otherwise
        undefined.

    Raises:
        ValueError: If inputs differ in length or are empty.

    Example:
        >>> matthews_corrcoef([True, True, False, False], [True, False, False, False])
        0.5773502691896258
    """
    _validate_binary(y_true, y_pred)

    tp = tn = fp = fn = 0
    for t, p in zip(y_true, y_pred, strict=True):
        if t and p:
            tp += 1
        elif not t and not p:
            tn += 1
        elif not t and p:
            fp += 1
        else:
            fn += 1

    # The radicand is a product of integer counts, so math.sqrt(0) is exactly
    # 0.0 -- the == comparison is safe (no floating-point near-zero ambiguity).
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0.0:
        return 0.0
    return (tp * tn - fp * fn) / denominator


# ---------------------------------------------------------------------------
# Calibration metrics (confidence-vs-outcome)
# ---------------------------------------------------------------------------
#
# These assess whether a verbalized confidence in ``[0, 1]`` matches the
# observed outcome frequency. ``confidences[i]`` is a self-reported probability
# and ``outcomes[i]`` is whether that prediction was borne out (boolean).


class ReliabilityBin(BaseModel):
    """One bin of a reliability diagram.

    Attributes:
        mean_confidence: Mean predicted confidence of the items in this bin.
        observed_frequency: Fraction of items in this bin whose outcome was
            ``True`` (the empirical accuracy for this confidence level).
        count: Number of items in this bin.
    """

    mean_confidence: float
    observed_frequency: float
    count: int


def _validate_calibration(confidences: list[float], outcomes: list[bool]) -> None:
    """Validate confidences and outcomes are aligned, non-empty, and in range.

    Raises:
        ValueError: If lengths differ, inputs are empty, or any confidence lies
            outside ``[0, 1]``.
    """
    if len(confidences) != len(outcomes):
        raise ValueError(
            f"confidences and outcomes must have equal length, "
            f"got {len(confidences)} and {len(outcomes)}"
        )
    if not confidences:
        raise ValueError("confidences and outcomes must be non-empty")
    for p in confidences:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"confidences must lie in [0, 1], got {p}")


def brier_score(confidences: list[float], outcomes: list[bool]) -> float:
    """Brier score: ``mean((p - y)^2)`` -- the primary proper calibration score.

    The Brier score is a strictly proper scoring rule, so unlike ECE it needs no
    binning and cannot be gamed by bin-edge choices (W6). Lower is better;
    ``0.0`` is perfect, ``0.25`` is the score of a constant ``0.5`` forecaster.

    Args:
        confidences: Predicted probabilities in ``[0, 1]``.
        outcomes: Realized boolean outcomes, aligned with ``confidences``.

    Returns:
        Mean squared error between confidence and outcome in ``[0.0, 1.0]``.

    Raises:
        ValueError: If inputs differ in length, are empty, or any confidence is
            outside ``[0, 1]``.

    Example:
        >>> brier_score([0.9, 0.1], [True, False])
        0.009999999999999995
    """
    _validate_calibration(confidences, outcomes)
    return sum(
        (p - (1.0 if y else 0.0)) ** 2 for p, y in zip(confidences, outcomes, strict=True)
    ) / len(confidences)


def _bin_indices(confidences: list[float], n_bins: int, strategy: BinStrategy) -> list[list[int]]:
    """Group item indices into bins by confidence; drop empty bins.

    ``"quantile"`` produces (approximately) equal-mass bins (each holds ~``N /
    n_bins`` items), robust to the clumping of verbalized LLM confidences that
    leaves equal-width bins mostly empty. ``"uniform"`` produces equal-width
    ``[0, 1]`` bins.

    Tied confidences are never split across bins: a bin boundary is extended past
    any run of equal confidence values. This keeps the result independent of
    input order (deterministic) and keeps calibration honest -- e.g. all-``0.5``
    predictions land in one bin, so a model that is right half the time scores
    ECE ``0.0`` rather than a spurious non-zero from an order-dependent split.
    As a consequence bins may be unequal in mass and fewer than ``n_bins`` may be
    returned. Empty bins are dropped.

    Raises:
        ValueError: If ``n_bins < 1`` or ``strategy`` is unknown.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    if strategy == "quantile":
        n = len(confidences)
        order = sorted(range(n), key=lambda i: confidences[i])
        groups: list[list[int]] = []
        start = 0
        for b in range(n_bins):
            if start >= n:
                break
            # Target an even cut, but never split equal confidence values and
            # always take at least one item so each visited bin is non-empty.
            cut = max(round((b + 1) * n / n_bins), start + 1)
            cut = min(cut, n)
            while cut < n and confidences[order[cut]] == confidences[order[cut - 1]]:
                cut += 1
            groups.append(order[start:cut])
            start = cut
        return groups

    if strategy == "uniform":
        buckets: list[list[int]] = [[] for _ in range(n_bins)]
        for i, p in enumerate(confidences):
            idx = min(int(p * n_bins), n_bins - 1)
            buckets[idx].append(i)
        return [g for g in buckets if g]

    raise ValueError(f"strategy must be 'quantile' or 'uniform', got {strategy!r}")


def expected_calibration_error(
    confidences: list[float],
    outcomes: list[bool],
    *,
    n_bins: int = 8,
    strategy: BinStrategy = "quantile",
) -> float:
    """Expected Calibration Error: count-weighted mean gap between confidence and accuracy.

    ``ECE = sum_b (n_b / N) * |acc_b - conf_b|`` over non-empty bins. The default
    is **equal-mass / quantile** binning: verbalized LLM confidences clump into a
    few values, so equal-width bins leave most items in 2-3 bins and understate
    miscalibration (W6).

    This is a *verbalized-confidence* ECE -- it calibrates self-reported
    confidence scores, not softmax logits. It is a secondary, binning-dependent
    diagnostic; prefer :func:`brier_score` as the headline calibration number.

    Args:
        confidences: Predicted probabilities in ``[0, 1]``.
        outcomes: Realized boolean outcomes, aligned with ``confidences``.
        n_bins: Number of bins (``>= 1``).
        strategy: ``"quantile"`` (equal-mass, default) or ``"uniform"`` (equal-width).

    Returns:
        ECE in ``[0.0, 1.0]``; ``0.0`` is perfectly calibrated.

    Raises:
        ValueError: If inputs differ in length, are empty, any confidence is
            outside ``[0, 1]``, ``n_bins < 1``, or ``strategy`` is unknown.

    Example:
        >>> expected_calibration_error([0.8, 0.8], [True, False], n_bins=1)
        0.30000000000000004
    """
    _validate_calibration(confidences, outcomes)
    n = len(confidences)
    ece = 0.0
    for group in _bin_indices(confidences, n_bins, strategy):
        conf_mean = sum(confidences[i] for i in group) / len(group)
        accuracy = sum(1.0 for i in group if outcomes[i]) / len(group)
        ece += (len(group) / n) * abs(accuracy - conf_mean)
    return ece


def reliability_bins(
    confidences: list[float],
    outcomes: list[bool],
    *,
    n_bins: int = 8,
    strategy: BinStrategy = "quantile",
) -> list[ReliabilityBin]:
    """Per-bin (mean_confidence, observed_frequency, count) for a reliability diagram.

    Uses the same binning as :func:`expected_calibration_error`. A well-calibrated
    model has ``observed_frequency ~= mean_confidence`` in every bin (points on the
    diagonal). Only non-empty bins are returned.

    Args:
        confidences: Predicted probabilities in ``[0, 1]``.
        outcomes: Realized boolean outcomes, aligned with ``confidences``.
        n_bins: Number of bins (``>= 1``).
        strategy: ``"quantile"`` (equal-mass, default) or ``"uniform"`` (equal-width).

    Returns:
        One :class:`ReliabilityBin` per non-empty bin, ordered from lowest to
        highest confidence.

    Raises:
        ValueError: If inputs differ in length, are empty, any confidence is
            outside ``[0, 1]``, ``n_bins < 1``, or ``strategy`` is unknown.

    Example:
        >>> bins = reliability_bins([0.2, 0.8], [False, True], n_bins=2)
        >>> [(b.mean_confidence, b.observed_frequency, b.count) for b in bins]
        [(0.2, 0.0, 1), (0.8, 1.0, 1)]
    """
    _validate_calibration(confidences, outcomes)
    bins: list[ReliabilityBin] = []
    for group in _bin_indices(confidences, n_bins, strategy):
        conf_mean = sum(confidences[i] for i in group) / len(group)
        accuracy = sum(1.0 for i in group if outcomes[i]) / len(group)
        bins.append(
            ReliabilityBin(
                mean_confidence=conf_mean,
                observed_frequency=accuracy,
                count=len(group),
            )
        )
    return bins
