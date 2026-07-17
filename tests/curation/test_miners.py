"""Tests for HardNegativeMiner (stratified candidate sampling)."""

import pytest

from langres.curation.miners import HardNegativeMiner, _percentile
from langres.core.models import CompanySchema, ERCandidate


def _cand(left_id: str, right_id: str, score: float | None) -> ERCandidate[CompanySchema]:
    return ERCandidate[CompanySchema](
        left=CompanySchema(id=left_id, name=left_id),
        right=CompanySchema(id=right_id, name=right_id),
        blocker_name="test",
        similarity_score=score,
    )


def _spread(n: int = 100) -> list[ERCandidate[CompanySchema]]:
    """``n`` unique pairs with similarity scores evenly spread over [0, (n-1)/n]."""
    return [_cand(f"l{i}", f"r{i}", i / n) for i in range(n)]


def _ids(cands: list[ERCandidate[CompanySchema]]) -> set[str]:
    return {c.left.id for c in cands}


# --- _percentile helper -----------------------------------------------------


def test_percentile_single_value() -> None:
    assert _percentile([0.7], 85.0) == 0.7


def test_percentile_integer_rank_no_interpolation() -> None:
    # rank = 0.5 * (4) = 2.0 -> exact index, no interpolation
    assert _percentile([0.0, 0.25, 0.5, 0.75, 1.0], 50.0) == 0.5


def test_percentile_interpolates() -> None:
    # rank = 0.5 * 1 = 0.5 -> halfway between 0.0 and 1.0
    assert _percentile([0.0, 1.0], 50.0) == 0.5


def test_percentile_boundaries() -> None:
    # pct=0 / pct=100 are reachable via mid_pct=0 / high_pct=100.
    vals = [0.1, 0.4, 0.9]
    assert _percentile(vals, 0.0) == 0.1
    assert _percentile(vals, 100.0) == 0.9


# --- stratum membership -----------------------------------------------------


def test_high_stratum_only_returns_top_band() -> None:
    miner = HardNegativeMiner(high_proportion=1.0, mid_proportion=0.0, low_proportion=0.0)
    out = miner.mine(_spread(), max_pairs=10)
    assert len(out) == 10
    # high stratum is score >= 85th pct (~0.8415) -> scores 0.85..0.99
    assert all(c.similarity_score is not None and c.similarity_score >= 0.85 for c in out)


def test_low_stratum_only_returns_bottom_band() -> None:
    miner = HardNegativeMiner(high_proportion=0.0, mid_proportion=0.0, low_proportion=1.0)
    out = miner.mine(_spread(), max_pairs=10)
    assert len(out) == 10
    # low stratum is score < 40th pct (~0.396) -> scores <= 0.39
    assert all(c.similarity_score is not None and c.similarity_score < 0.396 for c in out)


def test_default_proportions_weight_mid_heaviest() -> None:
    miner = HardNegativeMiner()  # 0.25 / 0.50 / 0.25
    out = miner.mine(_spread(), max_pairs=20)
    assert len(out) == 20
    high = sum(1 for c in out if c.similarity_score is not None and c.similarity_score >= 0.85)
    low = sum(1 for c in out if c.similarity_score is not None and c.similarity_score < 0.396)
    mid = len(out) - high - low
    assert (high, mid, low) == (5, 10, 5)


# --- determinism ------------------------------------------------------------


def test_same_seed_same_selection() -> None:
    a = HardNegativeMiner(seed=7).mine(_spread(), max_pairs=20)
    b = HardNegativeMiner(seed=7).mine(_spread(), max_pairs=20)
    assert _ids(a) == _ids(b)


def test_different_seed_different_selection() -> None:
    a = HardNegativeMiner(seed=1).mine(_spread(), max_pairs=20)
    b = HardNegativeMiner(seed=2).mine(_spread(), max_pairs=20)
    assert _ids(a) != _ids(b)


def test_output_preserves_input_order() -> None:
    out = HardNegativeMiner(seed=3).mine(_spread(), max_pairs=20)
    scores = [c.similarity_score for c in out]
    assert scores == sorted(scores)  # input is built in ascending-score order


# --- caps and pass-through --------------------------------------------------


def test_max_pairs_cap_respected() -> None:
    out = HardNegativeMiner().mine(_spread(), max_pairs=13)
    assert len(out) == 13


def test_uncapped_returns_all_unique() -> None:
    out = HardNegativeMiner().mine(_spread(50), max_pairs=None)
    assert len(out) == 50


def test_max_pairs_ge_total_returns_all() -> None:
    out = HardNegativeMiner().mine(_spread(30), max_pairs=999)
    assert len(out) == 30


def test_max_pairs_zero_returns_empty() -> None:
    # Exercises the single-value percentile path and a zero allocation.
    out = HardNegativeMiner().mine([_cand("a", "b", 0.5)], max_pairs=0)
    assert out == []


def test_empty_input_returns_empty() -> None:
    assert HardNegativeMiner().mine([]) == []


# --- dedup and validation ---------------------------------------------------


def test_deduplicates_order_independent_pairs() -> None:
    cands = [_cand("a", "b", 0.9), _cand("b", "a", 0.9), _cand("c", "d", 0.1)]
    out = HardNegativeMiner().mine(cands, max_pairs=None)
    keys = {tuple(sorted((c.left.id, c.right.id))) for c in out}
    assert len(out) == 2
    assert keys == {("a", "b"), ("c", "d")}


def test_missing_similarity_score_raises() -> None:
    with pytest.raises(ValueError, match="similarity_score"):
        HardNegativeMiner().mine([_cand("a", "b", None)])


def test_negative_max_pairs_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        HardNegativeMiner().mine(_spread(10), max_pairs=-1)


# --- allocation edge cases --------------------------------------------------


def test_allocation_leftover_goes_to_largest_remainder() -> None:
    # ideal = 5.25 / 10.5 / 5.25 -> floors 5/10/5, leftover 1 -> mid (largest frac)
    out = HardNegativeMiner().mine(_spread(), max_pairs=21)
    assert len(out) == 21
    mid = sum(
        1 for c in out if c.similarity_score is not None and 0.396 <= c.similarity_score < 0.85
    )
    assert mid == 11


def test_allocation_redistributes_when_stratum_capped() -> None:
    # high stratum has only 15 pairs but the proportion asks for ~27;
    # the shortfall must spill into mid/low while total stays exact.
    miner = HardNegativeMiner(high_proportion=0.9, mid_proportion=0.05, low_proportion=0.05)
    out = miner.mine(_spread(), max_pairs=30)
    assert len(out) == 30
    high = sum(1 for c in out if c.similarity_score is not None and c.similarity_score >= 0.85)
    assert high == 15  # capped at availability


# --- constructor validation -------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mid_pct": 90.0, "high_pct": 85.0},  # mid >= high
        {"mid_pct": -1.0},  # below 0
        {"high_pct": 101.0},  # above 100
        {"high_proportion": -0.1},  # negative proportion
        {"high_proportion": 0.0, "mid_proportion": 0.0, "low_proportion": 0.0},  # all zero
    ],
)
def test_invalid_constructor_raises(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        HardNegativeMiner(**kwargs)
