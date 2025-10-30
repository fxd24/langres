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
