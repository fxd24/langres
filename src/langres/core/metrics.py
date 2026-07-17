# TEMPORARY: deleted by the W2 sweep
"""Back-compat shim: the ER metrics moved to :mod:`langres.metrics.metrics`.

Metrics *score* a resolution; they are not part of the modelling contract, so
they now live in the ``langres.metrics`` package beside ``langres.core`` rather
than inside it. This shim keeps the old ``langres.core.metrics`` import path
working while the refactor's final sweep repoints callers; it re-exports the
full public surface of the new home (``ranx`` stays lazy inside
``evaluate_blocking_with_ranking``, exactly as before).
"""

from langres.metrics.metrics import (
    PairMetrics,
    ReliabilityBin,
    average_precision_score,
    brier_score,
    calculate_bcubed_metrics,
    calculate_bcubed_precision,
    calculate_bcubed_recall,
    calculate_pairwise_metrics,
    classify_pairs,
    cohens_kappa,
    evaluate_blocking,
    evaluate_blocking_with_ranking,
    evaluate_clustering,
    expected_calibration_error,
    generalized_merge_distance,
    log_loss,
    matthews_corrcoef,
    pair_pr_curve,
    pairs_from_clusters,
    reduction_ratio,
    reliability_bins,
)

__all__ = [
    "PairMetrics",
    "ReliabilityBin",
    "average_precision_score",
    "brier_score",
    "calculate_bcubed_metrics",
    "calculate_bcubed_precision",
    "calculate_bcubed_recall",
    "calculate_pairwise_metrics",
    "classify_pairs",
    "cohens_kappa",
    "evaluate_blocking",
    "evaluate_blocking_with_ranking",
    "evaluate_clustering",
    "expected_calibration_error",
    "generalized_merge_distance",
    "log_loss",
    "matthews_corrcoef",
    "pair_pr_curve",
    "pairs_from_clusters",
    "reduction_ratio",
    "reliability_bins",
]
