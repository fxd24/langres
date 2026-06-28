"""Tests for GroundTruthLabeler and the budget-capped TeacherLabeler.

The teacher is exercised with a fake judge that yields synthetic
``PairwiseJudgement``s (no network, no API key), so every budget branch is
covered deterministically.
"""

from collections.abc import Iterator

import pytest

from langres.bootstrap.labelers import (
    BlindCostError,
    FakeLabeler,
    GroundTruthLabeler,
    TeacherLabeler,
)
from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement


def _cand(left_id: str, right_id: str) -> ERCandidate[CompanySchema]:
    return ERCandidate[CompanySchema](
        left=CompanySchema(id=left_id, name=left_id),
        right=CompanySchema(id=right_id, name=right_id),
        blocker_name="test",
    )


# --- GroundTruthLabeler -----------------------------------------------------


def test_ground_truth_labels_match_and_non_match() -> None:
    labeler = GroundTruthLabeler.from_clusters([{"a", "b"}, {"c", "d"}])
    out = labeler.label([_cand("a", "b"), _cand("b", "a"), _cand("a", "c")])
    assert [p.label for p in out] == [True, True, False]
    assert all(p.source == "ground_truth" for p in out)
    # confidence is P(match): 1.0 for known matches, 0.0 for known non-matches.
    assert [p.confidence for p in out] == [1.0, 1.0, 0.0]


def test_ground_truth_from_clusters_expands_multi_member_cluster() -> None:
    labeler = GroundTruthLabeler.from_clusters([{"x", "y", "z"}])
    out = labeler.label([_cand("x", "y"), _cand("x", "z"), _cand("y", "z"), _cand("x", "w")])
    assert [p.label for p in out] == [True, True, True, False]


def test_ground_truth_direct_constructor() -> None:
    labeler = GroundTruthLabeler({("a", "b")})
    assert labeler.label([_cand("b", "a")])[0].label is True


def test_ground_truth_label_empty_returns_empty() -> None:
    assert GroundTruthLabeler.from_clusters([{"a", "b"}]).label([]) == []


# --- FakeJudge --------------------------------------------------------------


class FakeJudge:
    """Minimal stand-in for LLMJudge.forward used by TeacherLabeler tests."""

    def __init__(
        self,
        *,
        prompt_tokens: int = 1000,
        completion_tokens: int = 500,
        cost_usd: float = 0.0,
        score: float = 0.9,
        fail_ids: frozenset[str] = frozenset(),
        empty_ids: frozenset[str] = frozenset(),
        blind_ids: frozenset[str] = frozenset(),
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cost_usd = cost_usd
        self.score = score
        self.fail_ids = fail_ids
        self.empty_ids = empty_ids
        self.blind_ids = blind_ids

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        for cand in candidates:
            cid = cand.left.id
            if cid in self.fail_ids:
                raise RuntimeError("simulated judge failure")
            if cid in self.empty_ids:
                # Yield nothing for this pair. The teacher always calls forward
                # with single-item iterators, so stopping the generator here is
                # equivalent to "no judgement for this one pair".
                return
            if cid in self.blind_ids:
                provenance: dict[str, object] = {"model": "fake", "cost_usd": 0.0}
            else:
                provenance = {
                    "model": "fake",
                    "cost_usd": self.cost_usd,
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                }
            yield PairwiseJudgement(
                left_id=cand.left.id,
                right_id=cand.right.id,
                score=self.score,
                score_type="prob_llm",
                decision_step="fake",
                reasoning="fake reasoning",
                provenance=provenance,
            )


def _teacher(judge: FakeJudge, **overrides: object) -> TeacherLabeler:
    kwargs: dict[str, object] = {
        "price_per_1m_prompt_tokens": 1.0,
        "price_per_1m_completion_tokens": 2.0,
        "worst_case_tokens_per_pair": 2000,
        "budget_usd": 20.0,
        "budget_soft_usd": 15.0,
        "batch_size": 50,
    }
    kwargs.update(overrides)
    return TeacherLabeler(judge, **kwargs)  # type: ignore[arg-type]


# --- TeacherLabeler: happy path + tally -------------------------------------


def test_tally_accumulates_from_tokens_times_price() -> None:
    teacher = _teacher(FakeJudge())  # 1000 prompt @1/M + 500 completion @2/M = 0.002/pair
    out = teacher.label([_cand("a", "b"), _cand("c", "d"), _cand("e", "f")])
    assert teacher.labeled_count == 3
    assert teacher.skipped_count == 0
    assert teacher.dropped_by_cap_count == 0
    assert teacher.total_spent_usd == pytest.approx(0.006)
    first = out[0]
    assert first.source == "teacher"
    assert first.label is True and first.confidence == pytest.approx(0.9)
    assert first.provenance["tokens"] == {"prompt": 1000, "completion": 500}
    assert first.provenance["cost_usd"] == pytest.approx(0.002)
    assert first.provenance["model"] == "fake"


def test_label_false_below_threshold() -> None:
    teacher = _teacher(FakeJudge(score=0.3), threshold=0.5)
    out = teacher.label([_cand("a", "b")])
    assert out[0].label is False


def test_cost_uses_max_of_token_and_reported_cost() -> None:
    # reported cost_usd (1.0) dominates the token-derived cost (0.002)
    teacher = _teacher(FakeJudge(cost_usd=1.0))
    teacher.label([_cand("a", "b")])
    assert teacher.total_spent_usd == pytest.approx(1.0)


# --- TeacherLabeler: pre-flight cap -----------------------------------------


def test_preflight_cap_truncates_input() -> None:
    # worst-case 2000 tok @ max price 2/M = 0.004/pair; the *soft* budget (0.01)
    # sizes the cap -> floor(0.01/0.004)=2, with headroom below the hard budget.
    teacher = _teacher(FakeJudge(), budget_soft_usd=0.01, budget_usd=0.02)
    out = teacher.label([_cand(f"l{i}", f"r{i}") for i in range(5)])
    assert teacher.dropped_by_cap_count == 3
    assert teacher.labeled_count == 2
    assert len(out) == 2


# --- TeacherLabeler: budget stop --------------------------------------------


def test_budget_stop_returns_partial() -> None:
    # Under-estimated worst case (1000 tok) lets pre-flight keep all pairs, but the
    # real per-pair spend (5M prompt tokens @1/M = $5) trips the hard-budget stop.
    judge = FakeJudge(prompt_tokens=5_000_000, completion_tokens=0)
    teacher = _teacher(
        judge,
        price_per_1m_prompt_tokens=1.0,
        price_per_1m_completion_tokens=1.0,
        worst_case_tokens_per_pair=1000,
        budget_soft_usd=15.0,
        budget_usd=15.0,
        batch_size=1,
    )
    out = teacher.label([_cand(f"l{i}", f"r{i}") for i in range(10)])
    assert teacher.labeled_count == 3
    assert len(out) == 3
    assert teacher.total_spent_usd == pytest.approx(15.0)
    assert teacher.dropped_by_cap_count == 0


def test_single_pair_can_overshoot_budget_by_one_inflight_pair() -> None:
    # Documented residual (class docstring): the gate projects with the
    # worst-case ESTIMATE, so when that estimate understates real cost, one
    # in-flight pair's actual spend can push total_spent above budget_usd. The
    # overshoot is bounded to a single pair — the next gate check then stops.
    judge = FakeJudge(prompt_tokens=5_000_000, completion_tokens=0)  # $5 real/pair
    teacher = _teacher(
        judge,
        price_per_1m_prompt_tokens=1.0,
        price_per_1m_completion_tokens=1.0,
        worst_case_tokens_per_pair=1000,  # estimate ~ $0.001/pair (far too low)
        budget_soft_usd=4.9,
        budget_usd=4.9,
        batch_size=1,
    )
    out = teacher.label([_cand(f"l{i}", f"r{i}") for i in range(10)])
    assert teacher.labeled_count == 1
    assert len(out) == 1
    assert teacher.total_spent_usd == pytest.approx(5.0)
    assert teacher.total_spent_usd > teacher.budget_usd  # the bounded overshoot


def test_max_labelable_is_the_preflight_budget_cap() -> None:
    # The cap an orchestrator queries == the pre-flight ceiling, independent of
    # the candidate-pool size: floor(budget_soft / worst_case_per_pair_cost).
    judge = FakeJudge(prompt_tokens=10, completion_tokens=10)
    teacher = _teacher(
        judge,
        price_per_1m_prompt_tokens=1.0,
        price_per_1m_completion_tokens=1.0,
        worst_case_tokens_per_pair=1000,  # -> $0.001/pair
        budget_soft_usd=15.0,
        budget_usd=20.0,
        batch_size=1,
    )
    assert teacher.max_labelable(999_999) == 15_000  # floor(15 / 0.001)
    assert teacher.max_labelable(3) == 15_000  # pool size does not change the cap


def test_zero_spend_labelers_are_uncapped() -> None:
    # The default Labeler contract: no budget cap -> None (mine the whole pool).
    assert GroundTruthLabeler(set()).max_labelable(100) is None
    assert FakeLabeler(threshold=0.5).max_labelable(100) is None


def test_preflight_cap_zero_drops_all_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    # Misconfiguration: soft budget below the worst-case per-pair cost -> the
    # pre-flight cap keeps 0 pairs. The run must surface a WARNING (not a silent
    # empty gold set) and label nothing.
    judge = FakeJudge(prompt_tokens=10, completion_tokens=10)
    teacher = _teacher(
        judge,
        price_per_1m_prompt_tokens=1.0,
        price_per_1m_completion_tokens=1.0,
        worst_case_tokens_per_pair=1000,  # ~$0.001/pair
        budget_soft_usd=0.0001,  # below per-pair cost -> max_pairs == 0
        budget_usd=0.0001,
        batch_size=1,
    )
    with caplog.at_level("WARNING"):
        out = teacher.label([_cand("a", "b"), _cand("c", "d")])
    assert out == []
    assert teacher.labeled_count == 0
    assert teacher.dropped_by_cap_count == 2
    assert "NO pairs can be labeled" in caplog.text


def test_budget_stop_holds_within_a_large_batch() -> None:
    # Same over-spend scenario but with the default batch_size=50: the whole run
    # is one batch. The per-pair gate must still stop at the cap rather than
    # dispatching all 10 pairs ($50) before re-checking (codex P1 regression).
    judge = FakeJudge(prompt_tokens=5_000_000, completion_tokens=0)
    teacher = _teacher(
        judge,
        price_per_1m_prompt_tokens=1.0,
        price_per_1m_completion_tokens=1.0,
        worst_case_tokens_per_pair=1000,
        budget_soft_usd=15.0,
        budget_usd=15.0,
        batch_size=50,
    )
    out = teacher.label([_cand(f"l{i}", f"r{i}") for i in range(10)])
    assert len(out) == 3
    assert teacher.total_spent_usd == pytest.approx(15.0)


# --- TeacherLabeler: per-call resilience ------------------------------------


def test_failed_call_is_skipped_and_loop_continues() -> None:
    teacher = _teacher(FakeJudge(fail_ids=frozenset({"l1"})))
    out = teacher.label([_cand("l0", "r0"), _cand("l1", "r1"), _cand("l2", "r2")])
    assert teacher.labeled_count == 2
    assert teacher.skipped_count == 1
    assert {p.left_id for p in out} == {"l0", "l2"}


def test_empty_judgement_is_skipped() -> None:
    teacher = _teacher(FakeJudge(empty_ids=frozenset({"l0"})))
    out = teacher.label([_cand("l0", "r0"), _cand("l1", "r1")])
    assert teacher.labeled_count == 1
    assert teacher.skipped_count == 1
    assert out[0].left_id == "l1"


# --- TeacherLabeler: blind-cap abort ----------------------------------------


def test_blind_cost_aborts_after_recording_prior_spend() -> None:
    # l0 labels normally; l1 reports neither tokens nor cost -> abort.
    judge = FakeJudge(blind_ids=frozenset({"l1"}))
    teacher = _teacher(judge)
    with pytest.raises(BlindCostError) as excinfo:
        teacher.label([_cand("l0", "r0"), _cand("l1", "r1")])
    assert teacher.labeled_count == 1
    assert teacher.total_spent_usd == pytest.approx(0.002)
    # The already-paid pair is recoverable from the exception.
    assert [p.left_id for p in excinfo.value.partial] == ["l0"]


# --- TeacherLabeler: stats reset per call + empty input ----------------------


def test_stats_reset_between_calls() -> None:
    # Attributes describe only the most recent label() call, not cumulative spend.
    teacher = _teacher(FakeJudge())
    teacher.label([_cand("a", "b")])
    assert teacher.labeled_count == 1
    assert teacher.total_spent_usd == pytest.approx(0.002)
    teacher.label([_cand("c", "d"), _cand("e", "f")])
    assert teacher.labeled_count == 2  # reset to this call, not 3
    assert teacher.total_spent_usd == pytest.approx(0.004)


def test_teacher_label_empty_returns_empty() -> None:
    teacher = _teacher(FakeJudge())
    assert teacher.label([]) == []
    assert teacher.labeled_count == 0


def test_worst_case_cost_reflects_live_attrs() -> None:
    # Computed live (not cached), so updating a price is reflected immediately.
    teacher = _teacher(FakeJudge(), worst_case_tokens_per_pair=1000)
    before = teacher._worst_case_per_pair_cost
    teacher.price_per_1m_prompt_tokens *= 10
    teacher.price_per_1m_completion_tokens *= 10
    assert teacher._worst_case_per_pair_cost == pytest.approx(before * 10)


# --- TeacherLabeler: from_env (no network, no key) --------------------------


def test_from_env_builds_teacher_without_langfuse() -> None:
    teacher = TeacherLabeler.from_env(
        price_per_1m_prompt_tokens=0.1,
        price_per_1m_completion_tokens=0.2,
        worst_case_tokens_per_pair=3000,
        model="gpt-5-mini",
        budget_usd=10.0,
        budget_soft_usd=8.0,
    )
    # The point is that construction succeeds without LANGFUSE_* / OPENAI_API_KEY
    # (enable_langfuse=False); we assert only the public config, not internals.
    assert isinstance(teacher, TeacherLabeler)
    assert teacher.budget_usd == 10.0
    assert teacher.budget_soft_usd == 8.0


# --- TeacherLabeler: constructor validation ---------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"price_per_1m_prompt_tokens": 0.0},
        {"price_per_1m_completion_tokens": -1.0},
        {"worst_case_tokens_per_pair": 0},
        {"budget_usd": 0.0},
        {"budget_soft_usd": 0.0},
        {"budget_soft_usd": 30.0, "budget_usd": 20.0},  # soft > hard
        {"batch_size": 0},
        {"threshold": 1.5},
    ],
)
def test_invalid_constructor_raises(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _teacher(FakeJudge(), **overrides)


# --- FakeLabeler ------------------------------------------------------------


def _scored(left_id: str, right_id: str, score: float) -> ERCandidate[CompanySchema]:
    return ERCandidate[CompanySchema](
        left=CompanySchema(id=left_id, name=left_id),
        right=CompanySchema(id=right_id, name=right_id),
        blocker_name="test",
        similarity_score=score,
    )


def test_fake_labeler_thresholds_on_similarity() -> None:
    out = FakeLabeler(threshold=0.5).label(
        [_scored("a", "b", 0.9), _scored("c", "d", 0.5), _scored("e", "f", 0.49)]
    )
    assert [p.label for p in out] == [True, True, False]
    assert all(p.source == "fake" for p in out)


def test_fake_labeler_confidence_is_p_match_and_overconfident() -> None:
    # confidence is P(match): high for predicted matches, low (the complement)
    # for predicted non-matches, over-confident on the near-threshold band.
    out = FakeLabeler(threshold=0.5).label(
        [_scored("a", "b", 0.5), _scored("c", "d", 1.0), _scored("e", "f", 0.0)]
    )
    # Predicted match at the threshold: floor P(match) of 0.7.
    assert out[0].confidence == pytest.approx(0.7)
    # Far-from-threshold match: higher, capped at 0.99.
    assert out[1].confidence == pytest.approx(0.99)
    # Confident predicted non-match: low P(match) = 1 - 0.99 = 0.01.
    assert out[2].label is False
    assert out[2].confidence == pytest.approx(0.01)
    assert all(p.confidence is not None and 0.0 <= p.confidence <= 1.0 for p in out)


def test_fake_labeler_is_deterministic() -> None:
    cands = [_scored("a", "b", 0.8), _scored("c", "d", 0.2)]
    first = FakeLabeler().label(cands)
    second = FakeLabeler().label(cands)
    assert [(p.label, p.confidence) for p in first] == [(p.label, p.confidence) for p in second]


def test_fake_labeler_zero_spend() -> None:
    assert FakeLabeler().total_spent_usd == 0.0


def test_fake_labeler_empty_returns_empty() -> None:
    assert FakeLabeler().label([]) == []


def test_fake_labeler_requires_similarity_score() -> None:
    cand = ERCandidate[CompanySchema](
        left=CompanySchema(id="a", name="a"),
        right=CompanySchema(id="b", name="b"),
        blocker_name="test",
    )
    with pytest.raises(ValueError, match="similarity_score"):
        FakeLabeler().label([cand])


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_fake_labeler_rejects_out_of_range_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        FakeLabeler(threshold=threshold)
