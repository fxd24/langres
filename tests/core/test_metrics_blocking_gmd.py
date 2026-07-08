"""Tests for Reduction Ratio (RR) and Generalized Merge Distance (GMD).

RR is the classic blocking-efficiency metric (``1 - candidates / all_possible``),
covered both as a standalone function and as it is threaded through
:func:`evaluate_blocking` / :class:`CandidateStats`. GMD is the cost-based,
merge/split-asymmetric partition distance of Menestrina, Whang & Garcia-Molina
(VLDB 2010); the oracle cases below are hand-computed and cross-checked against
the paper's slice algorithm.
"""

import pytest

from langres.core.metrics import (
    _all_possible_pairs,
    evaluate_blocking,
    generalized_merge_distance,
    reduction_ratio,
)
from langres.core.models import CompanySchema, ERCandidate

# ---------------------------------------------------------------------------
# Reduction Ratio -- standalone function
# ---------------------------------------------------------------------------


def test_reduction_ratio_dedup_basic() -> None:
    """Dedup RR: 10 candidates out of n(n-1)/2 = 4950 possible for n=100."""
    rr = reduction_ratio(10, num_records=100)
    assert rr == pytest.approx(1.0 - 10 / 4950)


def test_reduction_ratio_dedup_no_reduction() -> None:
    """An all-pairs blocker (emits every possible pair) has RR = 0.0."""
    # n=4 -> 6 possible pairs; emitting all 6 means no reduction.
    assert reduction_ratio(6, num_records=4) == 0.0


def test_reduction_ratio_dedup_perfect_reduction() -> None:
    """Emitting zero candidates prunes everything -> RR = 1.0."""
    assert reduction_ratio(0, num_records=100) == 1.0


def test_reduction_ratio_cross_source_basic() -> None:
    """Cross-source RR uses |A|*|B|: 6 of 3*4 = 12 possible -> 0.5."""
    assert reduction_ratio(6, n_left=3, n_right=4) == pytest.approx(0.5)


def test_reduction_ratio_cross_source_perfect_reduction() -> None:
    """Zero candidates against a |A|*|B| space -> RR = 1.0."""
    assert reduction_ratio(0, n_left=5, n_right=7) == 1.0


def test_reduction_ratio_zero_records_is_zero() -> None:
    """A corpus of 0 records has no possible pairs; RR defined as 0.0."""
    assert reduction_ratio(0, num_records=0) == 0.0


def test_reduction_ratio_single_record_is_zero() -> None:
    """A corpus of 1 record has no possible pairs (n(n-1)/2 = 0); RR = 0.0."""
    assert reduction_ratio(0, num_records=1) == 0.0


def test_reduction_ratio_empty_source_is_zero() -> None:
    """An empty source makes |A|*|B| = 0; RR defined as 0.0."""
    assert reduction_ratio(0, n_left=0, n_right=10) == 0.0
    assert reduction_ratio(0, n_left=10, n_right=0) == 0.0


def test_reduction_ratio_more_candidates_than_possible_is_negative_not_clamped() -> None:
    """A candidate count exceeding the possible-pair total is left negative.

    This only happens when the supplied record counts are wrong (too small), so
    surfacing a negative RR keeps the mistake visible rather than hiding it.
    """
    # n=3 -> 3 possible pairs; claiming 10 candidates is inconsistent.
    assert reduction_ratio(10, num_records=3) == pytest.approx(1.0 - 10 / 3)
    assert reduction_ratio(10, num_records=3) < 0.0


def test_reduction_ratio_negative_candidates_raises() -> None:
    with pytest.raises(ValueError, match="num_candidate_pairs must be non-negative"):
        reduction_ratio(-1, num_records=10)


def test_reduction_ratio_both_modes_raises() -> None:
    with pytest.raises(ValueError, match="not both"):
        reduction_ratio(5, num_records=10, n_left=3, n_right=4)


def test_reduction_ratio_no_mode_raises() -> None:
    with pytest.raises(ValueError, match="provide num_records"):
        reduction_ratio(5)


def test_reduction_ratio_cross_source_missing_side_raises() -> None:
    with pytest.raises(ValueError, match="both n_left and n_right"):
        reduction_ratio(5, n_left=3)
    with pytest.raises(ValueError, match="both n_left and n_right"):
        reduction_ratio(5, n_right=4)


def test_reduction_ratio_negative_num_records_raises() -> None:
    with pytest.raises(ValueError, match="num_records must be non-negative"):
        reduction_ratio(0, num_records=-5)


def test_reduction_ratio_negative_cross_source_raises() -> None:
    with pytest.raises(ValueError, match="n_left and n_right must be non-negative"):
        reduction_ratio(0, n_left=-1, n_right=4)


def test_all_possible_pairs_helper_modes() -> None:
    """The shared helper returns n(n-1)/2 for dedup and |A|*|B| for cross-source."""
    assert _all_possible_pairs(num_records=100, n_left=None, n_right=None) == 4950
    assert _all_possible_pairs(num_records=None, n_left=3, n_right=4) == 12
    assert _all_possible_pairs(num_records=1, n_left=None, n_right=None) == 0


# ---------------------------------------------------------------------------
# Reduction Ratio -- threaded through evaluate_blocking / CandidateStats
# ---------------------------------------------------------------------------


def _candidate(left_id: str, right_id: str) -> ERCandidate[CompanySchema]:
    """Build a minimal candidate pair (no similarity score needed for RR/recall)."""
    return ERCandidate(
        left=CompanySchema(id=left_id, name=left_id),
        right=CompanySchema(id=right_id, name=right_id),
        blocker_name="test",
    )


def test_candidate_stats_reduction_ratio_field_default() -> None:
    """CandidateStats stays constructible without the new field (defaults to 0.0)."""
    from langres.core.debugging import CandidateStats

    stats = CandidateStats(
        total_candidates=1,
        avg_candidates_per_entity=1.0,
        candidate_recall=1.0,
        candidate_precision=1.0,
        missed_matches_count=0,
        false_positive_candidates_count=0,
    )
    assert stats.reduction_ratio == 0.0


def test_evaluate_blocking_reduction_ratio_derived_from_gold() -> None:
    """Default path: n is derived from the gold clusters (4 records -> 6 pairs)."""
    # Gold enumerates 4 records; emit 1 candidate pair -> RR = 1 - 1/6.
    candidates = [_candidate("a", "b")]
    gold = [{"a", "b"}, {"c"}, {"d"}]
    stats = evaluate_blocking(candidates, gold)
    assert stats.total_candidates == 1
    assert stats.reduction_ratio == pytest.approx(1.0 - 1 / 6)


def test_evaluate_blocking_reduction_ratio_explicit_num_records() -> None:
    """Explicit num_records overrides the gold-derived n (records with 0 candidates)."""
    # Gold names only 3 records, but the true corpus is 100 (singletons omitted
    # from gold). Explicit num_records makes RR reflect the real space.
    candidates = [_candidate("a", "b")]
    gold = [{"a", "b"}, {"c"}]
    stats = evaluate_blocking(candidates, gold, num_records=100)
    assert stats.reduction_ratio == pytest.approx(1.0 - 1 / 4950)


def test_evaluate_blocking_reduction_ratio_cross_source() -> None:
    """Cross-source sizes make RR use |A|*|B| instead of the pooled n(n-1)/2."""
    candidates = [_candidate("a1", "b1"), _candidate("a1", "b2")]
    gold = [{"a1", "b1"}]
    stats = evaluate_blocking(candidates, gold, n_left=3, n_right=4)
    # 2 candidates of 3*4 = 12 possible.
    assert stats.reduction_ratio == pytest.approx(1.0 - 2 / 12)


def test_evaluate_blocking_backward_compatible_positional_call() -> None:
    """The pre-existing 2-arg positional signature still works and now carries RR."""
    candidates = [_candidate("a", "b")]
    gold = [{"a", "b"}]
    stats = evaluate_blocking(candidates, gold)
    # Existing fields unchanged.
    assert stats.candidate_recall == 1.0
    assert stats.candidate_precision == 1.0
    # New field present and meaningful (2 gold records -> 1 possible pair, emitted).
    assert stats.reduction_ratio == 0.0


def test_evaluate_blocking_empty_candidates_reduction_ratio() -> None:
    """No candidates against a non-trivial gold space -> RR = 1.0."""
    gold = [{"a", "b"}, {"c", "d"}]
    stats = evaluate_blocking([], gold)
    assert stats.total_candidates == 0
    assert stats.reduction_ratio == 1.0


# ---------------------------------------------------------------------------
# Generalized Merge Distance (GMD) -- oracle cases (cm = cs = 1)
# ---------------------------------------------------------------------------


def test_gmd_identical_partitions_is_zero() -> None:
    assert generalized_merge_distance([{"a", "b", "c"}], [{"a", "b", "c"}]) == 0.0


def test_gmd_three_singletons_to_one_cluster_two_merges() -> None:
    """gold [{a,b,c}], predicted [{a},{b},{c}] -> 2 (two merges)."""
    assert generalized_merge_distance([{"a"}, {"b"}, {"c"}], [{"a", "b", "c"}]) == 2.0


def test_gmd_one_cluster_to_three_singletons_two_splits() -> None:
    """gold [{a},{b},{c}], predicted [{a,b,c}] -> 2 (two splits)."""
    assert generalized_merge_distance([{"a", "b", "c"}], [{"a"}, {"b"}, {"c"}]) == 2.0


def test_gmd_two_clusters_merged_one_split() -> None:
    """gold [{a,b},{c,d}], predicted [{a,b,c,d}] -> 1 (one split)."""
    assert generalized_merge_distance([{"a", "b", "c", "d"}], [{"a", "b"}, {"c", "d"}]) == 1.0


def test_gmd_one_cluster_split_one_merge() -> None:
    """gold [{a,b,c,d}], predicted [{a,b},{c,d}] -> 1 (one merge)."""
    assert generalized_merge_distance([{"a", "b"}, {"c", "d"}], [{"a", "b", "c", "d"}]) == 1.0


def test_gmd_crossed_partition_two_splits_two_merges() -> None:
    """gold [{a,b},{c,d}], predicted [{a,c},{b,d}] -> 4 (2 splits + 2 merges)."""
    assert generalized_merge_distance([{"a", "c"}, {"b", "d"}], [{"a", "b"}, {"c", "d"}]) == 4.0


def test_gmd_empty_partitions_is_zero() -> None:
    assert generalized_merge_distance([], []) == 0.0


def test_gmd_singletons_only_identical_is_zero() -> None:
    assert generalized_merge_distance([{"a"}, {"b"}], [{"a"}, {"b"}]) == 0.0


def test_gmd_returns_float() -> None:
    result = generalized_merge_distance([{"a"}, {"b"}], [{"a", "b"}])
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# GMD -- asymmetry: merge_cost and split_cost honored independently
# ---------------------------------------------------------------------------


def test_gmd_asymmetric_merges_cost_more() -> None:
    """With cm=2, cs=1 the '3 singletons -> 1' case costs 2 merges * 2 = 4."""
    assert (
        generalized_merge_distance(
            [{"a"}, {"b"}, {"c"}], [{"a", "b", "c"}], merge_cost=2.0, split_cost=1.0
        )
        == 4.0
    )


def test_gmd_asymmetric_splits_unaffected_by_merge_cost() -> None:
    """With cm=2, cs=1 the '1 -> 3 singletons' case costs 2 splits * 1 = 2 (merges not charged)."""
    assert (
        generalized_merge_distance(
            [{"a", "b", "c"}], [{"a"}, {"b"}, {"c"}], merge_cost=2.0, split_cost=1.0
        )
        == 2.0
    )


def test_gmd_custom_split_cost() -> None:
    """split_cost scales only the split operations."""
    # gold [{a,b},{c,d}], predicted [{a,b,c,d}] -> 1 split * cs=3 = 3.
    assert (
        generalized_merge_distance([{"a", "b", "c", "d"}], [{"a", "b"}, {"c", "d"}], split_cost=3.0)
        == 3.0
    )


# ---------------------------------------------------------------------------
# GMD -- record-set coverage validation
# ---------------------------------------------------------------------------


def test_gmd_predicted_extra_record_raises() -> None:
    with pytest.raises(ValueError, match="same record set"):
        generalized_merge_distance([{"a", "b", "x"}], [{"a", "b"}])


def test_gmd_gold_extra_record_raises() -> None:
    with pytest.raises(ValueError, match="same record set"):
        generalized_merge_distance([{"a", "b"}], [{"a", "b", "z"}])


def test_gmd_disjoint_record_sets_raises() -> None:
    with pytest.raises(ValueError, match="same record set"):
        generalized_merge_distance([{"a"}], [{"b"}])
