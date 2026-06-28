"""
Tests for BCubed evaluation metrics.

This test suite validates the BCubed Precision, Recall, and F1 implementations
which are used to evaluate clustering quality in entity resolution.
"""

import pytest


def test_bcubed_metrics_module_exists():
    """Test that the metrics module can be imported."""
    from langres.core import metrics  # noqa: F401


def test_bcubed_perfect_clustering():
    """Test BCubed metrics with perfect clustering (P=R=F1=1.0)."""
    from langres.core.metrics import calculate_bcubed_metrics

    predicted = [{"e1", "e2"}, {"e3", "e4"}]
    gold = [{"e1", "e2"}, {"e3", "e4"}]

    metrics = calculate_bcubed_metrics(predicted, gold)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_bcubed_all_separate():
    """Test BCubed metrics when all entities are in separate clusters."""
    from langres.core.metrics import calculate_bcubed_metrics

    # Predicted: all separate
    predicted = [{"e1"}, {"e2"}, {"e3"}, {"e4"}]
    # Gold: two groups
    gold = [{"e1", "e2"}, {"e3", "e4"}]

    metrics = calculate_bcubed_metrics(predicted, gold)

    # Precision should be perfect (each singleton is pure)
    assert metrics["precision"] == 1.0
    # Recall should be low (missing connections)
    assert metrics["recall"] < 1.0
    # F1 should be between precision and recall
    assert 0.0 < metrics["f1"] < 1.0


def test_bcubed_all_together():
    """Test BCubed metrics when all entities are in one cluster."""
    from langres.core.metrics import calculate_bcubed_metrics

    # Predicted: all together
    predicted = [{"e1", "e2", "e3", "e4"}]
    # Gold: two groups
    gold = [{"e1", "e2"}, {"e3", "e4"}]

    metrics = calculate_bcubed_metrics(predicted, gold)

    # Precision should be low (mixing different gold clusters)
    assert metrics["precision"] < 1.0
    # Recall should be perfect (all gold pairs are together)
    assert metrics["recall"] == 1.0
    # F1 should be between precision and recall
    assert 0.0 < metrics["f1"] < 1.0


def test_bcubed_empty_clusters():
    """Test BCubed metrics with empty cluster lists."""
    from langres.core.metrics import calculate_bcubed_metrics

    predicted = []
    gold = []

    metrics = calculate_bcubed_metrics(predicted, gold)

    # Empty clusters should return 0.0 for all metrics
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_bcubed_single_entity_clusters():
    """Test BCubed metrics with single-entity clusters."""
    from langres.core.metrics import calculate_bcubed_metrics

    predicted = [{"e1"}]
    gold = [{"e1"}]

    metrics = calculate_bcubed_metrics(predicted, gold)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_bcubed_precision_calculation():
    """Test BCubed precision calculation directly."""
    from langres.core.metrics import calculate_bcubed_precision

    predicted = [{"e1", "e2", "e3"}]  # One cluster with mixed entities
    gold = [{"e1", "e2"}, {"e3"}]  # Two separate gold clusters

    precision = calculate_bcubed_precision(predicted, gold)

    # For e1: 2 out of 3 entities in predicted cluster share gold cluster -> 2/3
    # For e2: 2 out of 3 entities in predicted cluster share gold cluster -> 2/3
    # For e3: 1 out of 3 entities in predicted cluster share gold cluster -> 1/3
    # Average: (2/3 + 2/3 + 1/3) / 3 = 5/9 ≈ 0.556
    assert abs(precision - 5 / 9) < 0.001


def test_bcubed_recall_calculation():
    """Test BCubed recall calculation directly."""
    from langres.core.metrics import calculate_bcubed_recall

    predicted = [{"e1"}, {"e2"}, {"e3"}]  # All separate
    gold = [{"e1", "e2"}, {"e3"}]  # Two gold clusters

    recall = calculate_bcubed_recall(predicted, gold)

    # For e1: 1 out of 2 entities in gold cluster are together -> 1/2
    # For e2: 1 out of 2 entities in gold cluster are together -> 1/2
    # For e3: 1 out of 1 entities in gold cluster are together -> 1/1
    # Average: (1/2 + 1/2 + 1) / 3 = 2/3 ≈ 0.667
    assert abs(recall - 2 / 3) < 0.001


def test_bcubed_f1_calculation():
    """Test that F1 is harmonic mean of precision and recall."""
    from langres.core.metrics import calculate_bcubed_metrics

    predicted = [{"e1", "e2"}]
    gold = [{"e1", "e2", "e3"}]

    metrics = calculate_bcubed_metrics(predicted, gold)

    # Verify F1 is harmonic mean
    precision = metrics["precision"]
    recall = metrics["recall"]
    expected_f1 = 2 * (precision * recall) / (precision + recall)

    assert abs(metrics["f1"] - expected_f1) < 0.001


def test_bcubed_metrics_with_company_dataset():
    """Test BCubed metrics with realistic company deduplication data."""
    from langres.core.metrics import calculate_bcubed_metrics

    # Simulated company deduplication result
    predicted = [
        {"c1", "c1_dup"},  # Correctly identified duplicate
        {"c2"},  # Singleton (no duplicates found)
        {"c3", "c4"},  # False positive (merged unrelated companies)
    ]

    gold = [
        {"c1", "c1_dup"},  # True duplicate group
        {"c2"},  # True singleton
        {"c3"},  # Separate company
        {"c4"},  # Separate company
    ]

    metrics = calculate_bcubed_metrics(predicted, gold)

    # Precision should be less than 1.0 due to false positive (c3, c4)
    assert 0.0 < metrics["precision"] < 1.0
    # Recall should be 1.0 (all gold clusters are fully captured)
    assert metrics["recall"] == 1.0
    # F1 should be between precision and recall
    assert 0.0 < metrics["f1"] < 1.0


def test_bcubed_metrics_return_type():
    """Test that calculate_bcubed_metrics returns correct structure."""
    from langres.core.metrics import calculate_bcubed_metrics

    predicted = [{"e1", "e2"}]
    gold = [{"e1", "e2"}]

    metrics = calculate_bcubed_metrics(predicted, gold)

    # Should return a dict with these three keys
    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {"precision", "recall", "f1"}
    assert all(isinstance(v, float) for v in metrics.values())
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


# --- Partition-safety regression (P1-5): unclustered ids are own singletons -------
#
# The Clusterer drops singletons, so a record with no merge is simply ABSENT from
# the predicted clusters. The metric must treat each absent id as its own
# singleton cluster, NOT as a shared implicit "None" cluster (which would make two
# un-merged ids compare equal and inflate the score). These cases are hand-verified
# below. The fix must be a no-op when the partition is already complete (every id
# present), so all the tests above stay green.


def test_bcubed_recall_unmerged_pair_not_scored_as_recalled():
    """Hand-verified: an empty prediction must NOT score the {a,b} pair recall 1.0.

    gold = [{a, b}, {c}], predicted = [] (the Clusterer merged nothing, so every
    record is absent from the predicted clusters).

    Correct per-item recall:
      a: only a itself shares a's (absent) cluster -> 1/2
      b: only b itself shares b's (absent) cluster -> 1/2
      c: c is alone in its gold cluster            -> 1/1
      mean = (0.5 + 0.5 + 1.0) / 3 = 2/3

    The latent bug mapped every absent id to a single ``None`` cluster, so a and b
    compared equal and scored 1.0 each -> overall recall 1.0 (wrong).
    """
    from langres.core.metrics import calculate_bcubed_recall

    recall = calculate_bcubed_recall([], [{"a", "b"}, {"c"}])
    assert recall == pytest.approx(2 / 3)
    assert recall < 1.0


def test_bcubed_recall_complete_partition_still_perfect():
    """The fix is a no-op when nothing is unclustered: a correct prediction is 1.0."""
    from langres.core.metrics import calculate_bcubed_recall

    recall = calculate_bcubed_recall([{"a", "b"}, {"c"}], [{"a", "b"}, {"c"}])
    assert recall == 1.0


def test_bcubed_precision_absent_from_gold_not_scored_as_pure():
    """Symmetric guard: ids absent from the gold partition are their own singletons.

    predicted = [{x, y}], gold = [{z}] (x and y are not in any gold cluster).

    Correct per-item precision:
      x: only x itself shares x's (absent) gold cluster -> 1/2
      y: only y itself shares y's (absent) gold cluster -> 1/2
      mean = 0.5

    The latent bug mapped both to a single ``None`` gold cluster -> precision 1.0.
    """
    from langres.core.metrics import calculate_bcubed_precision

    precision = calculate_bcubed_precision([{"x", "y"}], [{"z"}])
    assert precision == pytest.approx(0.5)
    assert precision < 1.0


def test_pairwise_metrics_perfect_clustering():
    """Test pairwise metrics with perfect clustering."""
    from langres.core.metrics import calculate_pairwise_metrics

    predicted = [{"e1", "e2"}, {"e3", "e4"}]
    gold = [{"e1", "e2"}, {"e3", "e4"}]

    metrics = calculate_pairwise_metrics(predicted, gold)

    # Perfect clustering should have precision, recall, F1 = 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    # Should have 2 true positives: (e1, e2) and (e3, e4)
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0


def test_pairwise_metrics_all_separate():
    """Test pairwise metrics when all entities are in separate clusters."""
    from langres.core.metrics import calculate_pairwise_metrics

    # Predicted: all separate (no pairs)
    predicted = [{"e1"}, {"e2"}, {"e3"}, {"e4"}]
    # Gold: two groups
    gold = [{"e1", "e2"}, {"e3", "e4"}]

    metrics = calculate_pairwise_metrics(predicted, gold)

    # No predicted pairs, but 2 gold pairs
    assert metrics["tp"] == 0
    assert metrics["fp"] == 0
    assert metrics["fn"] == 2
    assert metrics["precision"] == 0.0  # No predicted pairs
    assert metrics["recall"] == 0.0  # Missed all gold pairs
    assert metrics["f1"] == 0.0


def test_pairwise_metrics_all_together():
    """Test pairwise metrics when all entities are in one cluster."""
    from langres.core.metrics import calculate_pairwise_metrics

    # Predicted: all together
    predicted = [{"e1", "e2", "e3", "e4"}]
    # Gold: two groups
    gold = [{"e1", "e2"}, {"e3", "e4"}]

    metrics = calculate_pairwise_metrics(predicted, gold)

    # Predicted pairs: (e1,e2), (e1,e3), (e1,e4), (e2,e3), (e2,e4), (e3,e4) = 6 pairs
    # Gold pairs: (e1,e2), (e3,e4) = 2 pairs
    # TP: (e1,e2), (e3,e4) = 2
    # FP: (e1,e3), (e1,e4), (e2,e3), (e2,e4) = 4
    # FN: 0 (all gold pairs are in predicted)
    assert metrics["tp"] == 2
    assert metrics["fp"] == 4
    assert metrics["fn"] == 0
    assert metrics["precision"] == 2 / 6  # 2 TP / (2 TP + 4 FP)
    assert metrics["recall"] == 1.0  # 2 TP / (2 TP + 0 FN)
    assert abs(metrics["f1"] - 2 * (1 / 3 * 1.0) / (1 / 3 + 1.0)) < 0.001


def test_pairwise_metrics_partial_match():
    """Test pairwise metrics with partial matching."""
    from langres.core.metrics import calculate_pairwise_metrics

    # Predicted: correctly merged e1, e2 but incorrectly merged e3, e4
    predicted = [{"e1", "e2"}, {"e3", "e4"}]
    # Gold: e1, e2 should be together, but e3, e4 should be separate
    gold = [{"e1", "e2"}, {"e3"}, {"e4"}]

    metrics = calculate_pairwise_metrics(predicted, gold)

    # Predicted pairs: (e1,e2), (e3,e4)
    # Gold pairs: (e1,e2)
    # TP: (e1,e2) = 1
    # FP: (e3,e4) = 1
    # FN: 0
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 0
    assert metrics["precision"] == 0.5  # 1 / 2
    assert metrics["recall"] == 1.0  # 1 / 1
    assert abs(metrics["f1"] - 2 * (0.5 * 1.0) / (0.5 + 1.0)) < 0.001


def test_pairwise_metrics_empty_clusters():
    """Test pairwise metrics with empty cluster lists."""
    from langres.core.metrics import calculate_pairwise_metrics

    predicted = []
    gold = []

    metrics = calculate_pairwise_metrics(predicted, gold)

    # No pairs at all
    assert metrics["tp"] == 0
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_pairwise_metrics_single_entity_clusters():
    """Test pairwise metrics with single-entity clusters."""
    from langres.core.metrics import calculate_pairwise_metrics

    predicted = [{"e1"}]
    gold = [{"e1"}]

    metrics = calculate_pairwise_metrics(predicted, gold)

    # Single entities produce no pairs
    assert metrics["tp"] == 0
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_pairwise_metrics_return_type():
    """Test that calculate_pairwise_metrics returns correct structure."""
    from langres.core.metrics import calculate_pairwise_metrics

    predicted = [{"e1", "e2"}]
    gold = [{"e1", "e2"}]

    metrics = calculate_pairwise_metrics(predicted, gold)

    # Should return a dict with these keys
    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {"precision", "recall", "f1", "tp", "fp", "fn"}
    assert all(isinstance(v, (int, float)) for v in metrics.values())
    assert all(0.0 <= metrics[k] <= 1.0 for k in ["precision", "recall", "f1"])
    assert all(metrics[k] >= 0 for k in ["tp", "fp", "fn"])


def test_clusters_to_pairs_single_cluster():
    """Test _clusters_to_pairs with a single cluster."""
    from langres.core.metrics import _clusters_to_pairs

    clusters = [{"e1", "e2", "e3"}]
    pairs = _clusters_to_pairs(clusters)

    # Should produce 3 pairs: (e1,e2), (e1,e3), (e2,e3)
    assert len(pairs) == 3
    assert ("e1", "e2") in pairs
    assert ("e1", "e3") in pairs
    assert ("e2", "e3") in pairs


def test_clusters_to_pairs_multiple_clusters():
    """Test _clusters_to_pairs with multiple clusters."""
    from langres.core.metrics import _clusters_to_pairs

    clusters = [{"e1", "e2"}, {"e3", "e4"}]
    pairs = _clusters_to_pairs(clusters)

    # Should produce 2 pairs: (e1,e2) and (e3,e4)
    assert len(pairs) == 2
    assert ("e1", "e2") in pairs
    assert ("e3", "e4") in pairs


def test_clusters_to_pairs_empty_cluster():
    """Test _clusters_to_pairs with empty clusters."""
    from langres.core.metrics import _clusters_to_pairs

    clusters = []
    pairs = _clusters_to_pairs(clusters)

    # Should produce no pairs
    assert len(pairs) == 0


def test_clusters_to_pairs_singleton_clusters():
    """Test _clusters_to_pairs with singleton clusters."""
    from langres.core.metrics import _clusters_to_pairs

    clusters = [{"e1"}, {"e2"}, {"e3"}]
    pairs = _clusters_to_pairs(clusters)

    # Singletons produce no pairs
    assert len(pairs) == 0


def test_clusters_to_pairs_lexicographic_ordering():
    """Test that _clusters_to_pairs produces lexicographically ordered pairs."""
    from langres.core.metrics import _clusters_to_pairs

    # Test with strings that would sort differently
    clusters = [{"z1", "a1"}]
    pairs = _clusters_to_pairs(clusters)

    # Should be ordered: smaller ID first
    assert pairs == {("a1", "z1")}


def test_clusters_to_pairs_large_cluster():
    """Test _clusters_to_pairs with a larger cluster."""
    from langres.core.metrics import _clusters_to_pairs

    clusters = [{"e1", "e2", "e3", "e4"}]
    pairs = _clusters_to_pairs(clusters)

    # n=4 entities should produce n*(n-1)/2 = 6 pairs
    assert len(pairs) == 6
    expected_pairs = {
        ("e1", "e2"),
        ("e1", "e3"),
        ("e1", "e4"),
        ("e2", "e3"),
        ("e2", "e4"),
        ("e3", "e4"),
    }
    assert pairs == expected_pairs
