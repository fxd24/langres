"""Tests for the ranking metrics ``roc_auc_score`` / ``average_precision_score``.

Both are pure-Python, tie-aware reimplementations of the sklearn functions of
the same name (``metrics.py`` must stay numpy-free -- see
``tests/test_import_budget.py``). Cross-checks against sklearn are guarded by
``pytest.importorskip`` and run only on genuine 2-class inputs: this repo's
dev-env sklearn (1.7.2) does *not* raise on a single-class ``y_true`` the way
older sklearn versions did -- it warns and returns ``nan``/``0.0``/``1.0`` --
so our fixed, warning-free ``nan``/``1.0`` contract for single-class inputs
is verified directly against the edge-case table, not by expecting sklearn to
raise. Coverage targets 95-100% of the two new functions and their helpers,
including every validation and edge branch.
"""

import math
import random

import pytest

from langres.core.metrics import (
    _midranks,
    _validate_binary_scores,
    average_precision_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# _midranks
# ---------------------------------------------------------------------------


def test_midranks_no_ties_is_ordinal() -> None:
    assert _midranks([30.0, 10.0, 20.0]) == [3.0, 1.0, 2.0]


def test_midranks_all_tied_is_average_rank() -> None:
    # 3 tied items share the average of ranks 1, 2, 3 -> 2.0 each.
    assert _midranks([5.0, 5.0, 5.0]) == [2.0, 2.0, 2.0]


def test_midranks_partial_tie() -> None:
    # sorted: 1.0(rank1), 2.0,2.0(ranks 2,3 -> midrank 2.5), 4.0(rank4)
    assert _midranks([2.0, 4.0, 1.0, 2.0]) == [2.5, 4.0, 1.0, 2.5]


# ---------------------------------------------------------------------------
# _validate_binary_scores
# ---------------------------------------------------------------------------


def test_validate_binary_scores_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal length"):
        _validate_binary_scores([True], [1.0, 2.0])


def test_validate_binary_scores_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _validate_binary_scores([], [])


# ---------------------------------------------------------------------------
# roc_auc_score -- edge-case contract
# ---------------------------------------------------------------------------


def test_roc_auc_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        roc_auc_score([], [])


def test_roc_auc_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal length"):
        roc_auc_score([True, False], [1.0])


def test_roc_auc_only_positives_is_nan() -> None:
    assert math.isnan(roc_auc_score([True, True, True], [1.0, 2.0, 3.0]))


def test_roc_auc_only_negatives_is_nan() -> None:
    assert math.isnan(roc_auc_score([False, False, False], [1.0, 2.0, 3.0]))


def test_roc_auc_all_scores_identical_is_half() -> None:
    assert roc_auc_score([True, False, True, False], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_roc_auc_perfect_separation_is_one() -> None:
    assert roc_auc_score([False, False, True, True], [1.0, 2.0, 3.0, 4.0]) == 1.0


def test_roc_auc_worst_separation_is_zero() -> None:
    # All positives score strictly below all negatives -> U=0 -> AUC=0.
    assert roc_auc_score([True, True, False, False], [1.0, 2.0, 3.0, 4.0]) == 0.0


def test_roc_auc_hand_computed_no_ties() -> None:
    # y=[F,T,F,T], scores=[1,2,3,4] -> sorted order is already ascending,
    # ranks 1..4. Positives at scores 2,4 -> ranks 2,4. rank_sum_pos=6.
    # n_pos=2,n_neg=2. U = 6 - 2*3/2 = 3. AUC = 3/4 = 0.75.
    assert roc_auc_score([False, True, False, True], [1.0, 2.0, 3.0, 4.0]) == 0.75


# ---------------------------------------------------------------------------
# roc_auc_score -- ties (mandatory, not optional)
# ---------------------------------------------------------------------------


def test_roc_auc_single_tie_straddling_class_boundary() -> None:
    """A single tied pair spanning the pos/neg boundary is exactly where a
    naive ordinal-rank implementation diverges from the midrank-correct one.

    y=[F,F,F,T,T], scores=[1,2,3,3,4]: the two items tied at score 3 are one
    negative and one positive. Naive ordinal ranking would score this tie as
    a full win for the positive (AUC=1.0); midrank-correct scoring gives the
    tied pair 0.5 credit instead.
    """
    y_true = [False, False, False, True, True]
    scores = [1.0, 2.0, 3.0, 3.0, 4.0]
    # midranks: 1,2,3.5,3.5,5. rank_sum_pos = 3.5+5=8.5. n_pos=2,n_neg=3.
    # U = 8.5 - 2*3/2 = 5.5. AUC = 5.5/6.
    assert roc_auc_score(y_true, scores) == pytest.approx(5.5 / 6)
    # A naive ordinal-rank AUC would instead be a full 1.0 -- confirms this
    # is the exact bug the midrank implementation must not have.
    assert roc_auc_score(y_true, scores) != 1.0


def test_roc_auc_tie_within_one_class_only() -> None:
    # Ties confined to the negative class only must not distort AUC away
    # from the naive-ranking result, since no pos/neg tie is involved.
    y_true = [False, False, True]
    scores = [1.0, 1.0, 2.0]
    # midranks: 1.5, 1.5, 3. rank_sum_pos=3. n_pos=1,n_neg=2.
    # U = 3 - 1 = 2. AUC = 2/2 = 1.0 (positive strictly above both negatives).
    assert roc_auc_score(y_true, scores) == 1.0


def test_roc_auc_seeded_random_with_duplicate_scores() -> None:
    """Seeded random vectors with a coarse score grid (forces duplicates)."""
    rng = random.Random(1234)
    for _ in range(20):
        n = rng.randint(4, 15)
        y_true = [rng.random() < 0.4 for _ in range(n)]
        if sum(y_true) == 0 or sum(y_true) == n:
            continue
        scores = [round(rng.uniform(0, 1), 1) for _ in range(n)]  # coarse -> ties
        auc = roc_auc_score(y_true, scores)
        assert 0.0 <= auc <= 1.0


# ---------------------------------------------------------------------------
# roc_auc_score -- properties
# ---------------------------------------------------------------------------


def test_roc_auc_invariant_under_strictly_increasing_transform() -> None:
    rng = random.Random(99)
    for _ in range(30):
        n = rng.randint(4, 12)
        y_true = [rng.random() < 0.5 for _ in range(n)]
        if sum(y_true) == 0 or sum(y_true) == n:
            continue
        scores = [round(rng.uniform(-5, 5), 1) for _ in range(n)]
        transformed = [s**3 + 2 * s for s in scores]  # strictly increasing on R
        assert roc_auc_score(y_true, scores) == pytest.approx(roc_auc_score(y_true, transformed))


def test_roc_auc_negation_complements_to_one() -> None:
    rng = random.Random(2024)
    for _ in range(30):
        n = rng.randint(4, 12)
        y_true = [rng.random() < 0.5 for _ in range(n)]
        if sum(y_true) == 0 or sum(y_true) == n:
            continue
        scores = [round(rng.uniform(-5, 5), 1) for _ in range(n)]
        negated = [-s for s in scores]
        assert roc_auc_score(y_true, negated) == pytest.approx(1.0 - roc_auc_score(y_true, scores))


# ---------------------------------------------------------------------------
# average_precision_score -- edge-case contract
# ---------------------------------------------------------------------------


def test_ap_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        average_precision_score([], [])


def test_ap_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal length"):
        average_precision_score([True, False], [1.0])


def test_ap_only_positives_is_one() -> None:
    assert average_precision_score([True, True, True], [1.0, 2.0, 3.0]) == 1.0


def test_ap_only_negatives_is_nan() -> None:
    assert math.isnan(average_precision_score([False, False, False], [1.0, 2.0, 3.0]))


def test_ap_all_scores_identical_is_prevalence() -> None:
    y_true = [True, False, True, False]
    prevalence = sum(y_true) / len(y_true)
    assert average_precision_score(y_true, [0.5, 0.5, 0.5, 0.5]) == pytest.approx(prevalence)


def test_ap_perfect_separation_is_one() -> None:
    assert average_precision_score([False, False, True, True], [1.0, 2.0, 3.0, 4.0]) == 1.0


def test_ap_worst_separation_is_below_prevalence() -> None:
    # All positives ranked strictly below all negatives.
    y_true = [True, True, False, False]
    scores = [1.0, 2.0, 3.0, 4.0]
    prevalence = sum(y_true) / len(y_true)
    assert average_precision_score(y_true, scores) < prevalence


def test_ap_hand_computed_no_ties() -> None:
    # descending order: score4(T), score3(F), score2(T), score1(F)
    # k=1: tp=1,cum=1 -> R=1/2=.5, P=1.0 -> contrib .5*1.0=.5
    # k=2: tp=1,cum=2 -> R=.5, P=.5 -> contrib (0)*.5=0 (no recall increase)
    # k=3: tp=2,cum=3 -> R=1.0, P=2/3 -> contrib .5*2/3=1/3
    # k=4: tp=2,cum=4 -> R=1.0, P=.5 -> contrib 0
    # AP = .5 + 1/3 = 5/6
    y_true = [False, True, False, True]
    scores = [1.0, 2.0, 3.0, 4.0]
    assert average_precision_score(y_true, scores) == pytest.approx(5 / 6)


# ---------------------------------------------------------------------------
# average_precision_score -- ties (mandatory, not optional)
# ---------------------------------------------------------------------------


def test_ap_single_tie_straddling_class_boundary() -> None:
    """Tie-grouped processing: the tied block at score 3 (one neg, one pos)
    must be cut once using the block's cumulative TP/FP, not per item.

    y=[F,F,F,T,T], scores=[1,2,3,3,4] descending order groups: [4](T),
    [3,3]({F,T} tied), [2](F), [1](F).
    k=1 (score4): tp=1,cum=1 -> R=1/2=.5,P=1 -> contrib .5
    tie block (score3, 2 items): tp becomes 2 (one of the two is positive),
      cum=3 -> R=2/2=1.0,P=2/3 -> contrib (1.0-.5)*2/3=1/3
    remaining negatives at score2,1 add no further TP -> R stays 1.0, no
      further recall delta -> no further contribution.
    AP = .5 + 1/3 = 5/6.
    """
    y_true = [False, False, False, True, True]
    scores = [1.0, 2.0, 3.0, 3.0, 4.0]
    assert average_precision_score(y_true, scores) == pytest.approx(5 / 6)


def test_ap_tie_block_not_overcounted_per_item() -> None:
    """A naive per-item walk through a tie block (instead of one cut after
    the whole block) would overstate precision for the first item processed
    inside the block. This asserts the tie-grouped value, which differs from
    that naive per-item value.

    y=[T,F], scores=[5,5] (single tie block covering everything).
    Correct (tie-grouped): one cut after both items -> tp=1,cum=2 -> R=1.0,
      P=0.5 -> AP=0.5 (== prevalence, as required for all-identical scores).
    Naive per-item (WRONG, order-dependent): if the positive were processed
      first it would score a spurious 1.0 at that step.
    """
    y_true = [True, False]
    scores = [5.0, 5.0]
    assert average_precision_score(y_true, scores) == 0.5


def test_ap_seeded_random_with_duplicate_scores() -> None:
    rng = random.Random(4321)
    for _ in range(20):
        n = rng.randint(4, 15)
        y_true = [rng.random() < 0.4 for _ in range(n)]
        if sum(y_true) == 0:
            continue
        scores = [round(rng.uniform(0, 1), 1) for _ in range(n)]  # coarse -> ties
        ap = average_precision_score(y_true, scores)
        assert 0.0 <= ap <= 1.0


# ---------------------------------------------------------------------------
# average_precision_score -- properties
# ---------------------------------------------------------------------------


def test_ap_invariant_under_strictly_increasing_transform() -> None:
    rng = random.Random(55)
    for _ in range(30):
        n = rng.randint(4, 12)
        y_true = [rng.random() < 0.5 for _ in range(n)]
        if sum(y_true) == 0:
            continue
        scores = [round(rng.uniform(-5, 5), 1) for _ in range(n)]
        transformed = [s**3 + 2 * s for s in scores]
        assert average_precision_score(y_true, scores) == pytest.approx(
            average_precision_score(y_true, transformed)
        )


def test_ap_bounded_in_unit_interval() -> None:
    rng = random.Random(77)
    for _ in range(50):
        n = rng.randint(4, 15)
        y_true = [rng.random() < 0.4 for _ in range(n)]
        if sum(y_true) == 0:
            continue
        scores = [rng.uniform(0, 1) for _ in range(n)]
        ap = average_precision_score(y_true, scores)
        assert 0.0 <= ap <= 1.0


def test_ap_at_least_prevalence_for_a_correctly_ordered_ranker() -> None:
    """AP >= prevalence whenever a (possibly partial) subset of positives is
    ranked strictly ahead of everything else.

    Construction (deterministic, holds for *any* split): take a non-empty
    subset of the positives and give them a uniquely-highest score (a "pure"
    top block); tie every remaining item (the rest of the positives plus all
    negatives) at one lower score. The trailing tied block, because it spans
    every remaining item, always resolves to exactly the dataset prevalence
    at that cut (a well-known fact: precision at 100% recall always equals
    prevalence). AP is then a weighted average of the top block's precision
    (1.0) and the trailing block's precision (prevalence), which can never
    fall below prevalence.
    """
    rng = random.Random(314)
    for _ in range(50):
        n_pos = rng.randint(1, 8)
        n_neg = rng.randint(1, 8)
        n = n_pos + n_neg
        y_true = [True] * n_pos + [False] * n_neg
        rng.shuffle(y_true)
        prevalence = n_pos / n

        top_block_size = rng.randint(1, n_pos)  # non-empty subset of positives
        pos_indices = [i for i, t in enumerate(y_true) if t]
        rng.shuffle(pos_indices)
        top_indices = set(pos_indices[:top_block_size])

        scores = [2.0 if i in top_indices else 1.0 for i in range(n)]
        ap = average_precision_score(y_true, scores)
        assert ap >= prevalence - 1e-12


# ---------------------------------------------------------------------------
# Cross-check against sklearn (2-class inputs only -- see module docstring)
# ---------------------------------------------------------------------------


def test_roc_auc_matches_sklearn_with_ties() -> None:
    pytest.importorskip("sklearn", reason="requires the [trained] extra")
    from sklearn.metrics import roc_auc_score as sk_roc_auc_score

    rng = random.Random(8675309)
    for _ in range(200):
        n = rng.randint(2, 25)
        y_true = [rng.random() < 0.4 for _ in range(n)]
        if sum(y_true) == 0 or sum(y_true) == n:
            continue
        scores = [round(rng.uniform(0, 1), 1) for _ in range(n)]  # coarse -> ties
        ours = roc_auc_score(y_true, scores)
        theirs = sk_roc_auc_score([int(t) for t in y_true], scores)
        assert ours == pytest.approx(theirs, abs=1e-9)


def test_ap_matches_sklearn_with_ties() -> None:
    pytest.importorskip("sklearn", reason="requires the [trained] extra")
    from sklearn.metrics import average_precision_score as sk_ap_score

    rng = random.Random(112358)
    for _ in range(200):
        n = rng.randint(2, 25)
        y_true = [rng.random() < 0.4 for _ in range(n)]
        if sum(y_true) == 0 or sum(y_true) == n:
            continue
        scores = [round(rng.uniform(0, 1), 1) for _ in range(n)]
        ours = average_precision_score(y_true, scores)
        theirs = sk_ap_score([int(t) for t in y_true], scores)
        assert ours == pytest.approx(theirs, abs=1e-9)


def test_roc_auc_boundary_tie_matches_sklearn() -> None:
    """The hand-built boundary-tie fixture, cross-checked against sklearn --
    the case where a naive (non-midrank) implementation would diverge."""
    pytest.importorskip("sklearn", reason="requires the [trained] extra")
    from sklearn.metrics import roc_auc_score as sk_roc_auc_score

    y_true = [False, False, False, True, True]
    scores = [1.0, 2.0, 3.0, 3.0, 4.0]
    ours = roc_auc_score(y_true, scores)
    theirs = sk_roc_auc_score([int(t) for t in y_true], scores)
    assert ours == pytest.approx(theirs, abs=1e-9)


def test_ap_boundary_tie_matches_sklearn() -> None:
    pytest.importorskip("sklearn", reason="requires the [trained] extra")
    from sklearn.metrics import average_precision_score as sk_ap_score

    y_true = [False, False, False, True, True]
    scores = [1.0, 2.0, 3.0, 3.0, 4.0]
    ours = average_precision_score(y_true, scores)
    theirs = sk_ap_score([int(t) for t in y_true], scores)
    assert ours == pytest.approx(theirs, abs=1e-9)


def test_roc_auc_all_equal_scores_matches_sklearn() -> None:
    pytest.importorskip("sklearn", reason="requires the [trained] extra")
    from sklearn.metrics import roc_auc_score as sk_roc_auc_score

    y_true = [True, False, True, False]
    scores = [0.5, 0.5, 0.5, 0.5]
    ours = roc_auc_score(y_true, scores)
    theirs = sk_roc_auc_score([int(t) for t in y_true], scores)
    assert ours == pytest.approx(theirs, abs=1e-9)


def test_ap_all_equal_scores_matches_sklearn() -> None:
    pytest.importorskip("sklearn", reason="requires the [trained] extra")
    from sklearn.metrics import average_precision_score as sk_ap_score

    y_true = [True, False, True, False]
    scores = [0.5, 0.5, 0.5, 0.5]
    ours = average_precision_score(y_true, scores)
    theirs = sk_ap_score([int(t) for t in y_true], scores)
    assert ours == pytest.approx(theirs, abs=1e-9)
