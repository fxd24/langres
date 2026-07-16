"""Tests for langres.core.spend_cap (the spend-cap enforcer).

Zero-spend throughout: every "paid" matcher here is a tiny fake that stamps a
made-up ``provenance["cost_usd"]``. No network, no LLM, no API key -- the cap
only ever reads the number a judgement reports, so a fake proves the arithmetic
exactly as well as a real model would, for $0.

The behaviour under test that the pre-B1 cap got wrong: the ledger is per
INSTANCE, not per ``forward()`` call.
"""

from collections.abc import Iterator

import pytest

from langres.core.matcher import Matcher
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.spend import BudgetExceeded, SpendMonitor
from langres.core.spend_cap import (
    DEFAULT_BUDGET_USD,
    UNCAPPED_BUDGET_USD,
    SpendCappedMatcher,
    effective_budget,
)


class _CountingCostlyMatcher(Matcher[object]):
    """Yields ``n`` judgements of fixed cost, counting every one it produced.

    ``produced`` is the "how many paid calls actually fired" probe: the cap can
    only be credited with *preventing* spend if this number stops growing.
    """

    def __init__(self, n: int, cost_each: float) -> None:
        self._n = n
        self._cost_each = cost_each
        self.produced = 0

    def forward(self, candidates: Iterator[ERCandidate[object]]) -> Iterator[PairwiseJudgement]:
        list(candidates)
        for i in range(self._n):
            self.produced += 1
            yield PairwiseJudgement(
                left_id=str(i),
                right_id=str(i + 1),
                score=0.9,
                score_type="prob_llm",
                decision_step="fake",
                provenance={"cost_usd": self._cost_each},
            )

    def inspect_scores(self, judgements: list[PairwiseJudgement], sample_size: int = 10) -> object:
        raise NotImplementedError


def _candidates(n: int = 3) -> Iterator[ERCandidate[object]]:
    return iter([])  # the fakes ignore candidates entirely


class TestEffectiveBudget:
    def test_none_resolves_to_the_default(self) -> None:
        assert effective_budget(None) == DEFAULT_BUDGET_USD

    def test_explicit_budget_is_returned_verbatim(self) -> None:
        assert effective_budget(2.5) == 2.5

    def test_zero_is_honored_not_treated_as_missing(self) -> None:
        """`0.0 or DEFAULT` would silently hand a $0 caller a $1 budget."""
        assert effective_budget(0.0) == 0.0

    def test_uncapped_passes_through(self) -> None:
        assert effective_budget(UNCAPPED_BUDGET_USD) == UNCAPPED_BUDGET_USD


class TestSpendCappedMatcherConstruction:
    def test_budget_and_monitor_together_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            SpendCappedMatcher(
                _CountingCostlyMatcher(1, 0.0),
                budget_usd=1.0,
                monitor=SpendMonitor(budget_usd=2.0),
            )

    def test_none_budget_uses_the_default(self) -> None:
        capped = SpendCappedMatcher(_CountingCostlyMatcher(1, 0.0))
        assert capped.monitor.budget_usd == DEFAULT_BUDGET_USD

    def test_injected_monitor_is_used_as_is(self) -> None:
        monitor = SpendMonitor(budget_usd=7.0)
        capped = SpendCappedMatcher(_CountingCostlyMatcher(1, 0.0), monitor=monitor)
        assert capped.monitor is monitor


class TestSpendCapIsPerInstanceNotPerCall:
    """THE B1 bug: a monitor built inside forward() reset the tally every call,
    so N forward()s each got a fresh full budget and spent N x budget."""

    def test_budget_binds_across_two_forward_calls(self) -> None:
        inner = _CountingCostlyMatcher(1, 0.5)
        capped = SpendCappedMatcher(inner, budget_usd=0.9)

        assert len(list(capped.forward(_candidates()))) == 1  # spent 0.50 <= 0.90
        with pytest.raises(BudgetExceeded):
            list(capped.forward(_candidates()))  # spent 1.00 > 0.90
        assert capped.monitor.spent == 1.0

    def test_a_tripped_cap_costs_nothing_on_the_next_call(self) -> None:
        """The pre-check: once over budget, the NEXT forward() must refuse
        before pulling anything, not pay for one more judgement first."""
        inner = _CountingCostlyMatcher(1, 2.0)
        capped = SpendCappedMatcher(inner, budget_usd=0.9)

        with pytest.raises(BudgetExceeded):
            list(capped.forward(_candidates()))
        assert inner.produced == 1

        with pytest.raises(BudgetExceeded):
            list(capped.forward(_candidates()))
        assert inner.produced == 1, "a second forward() paid for another call past the cap"

    def test_two_wrappers_sharing_one_monitor_share_one_budget(self) -> None:
        monitor = SpendMonitor(budget_usd=0.9)
        first = SpendCappedMatcher(_CountingCostlyMatcher(1, 0.5), monitor=monitor)
        second = SpendCappedMatcher(_CountingCostlyMatcher(1, 0.5), monitor=monitor)

        list(first.forward(_candidates()))
        with pytest.raises(BudgetExceeded):
            list(second.forward(_candidates()))


class TestSpendCapBudgetEdges:
    def test_exactly_at_budget_does_not_trip(self) -> None:
        """The check is `spent > budget`, strictly -- landing on the number is fine."""
        capped = SpendCappedMatcher(_CountingCostlyMatcher(2, 0.5), budget_usd=1.0)
        assert len(list(capped.forward(_candidates()))) == 2
        assert capped.monitor.spent == 1.0

    def test_one_cent_over_budget_trips(self) -> None:
        capped = SpendCappedMatcher(_CountingCostlyMatcher(3, 0.5), budget_usd=1.0)
        with pytest.raises(BudgetExceeded):
            list(capped.forward(_candidates()))

    def test_zero_budget_refuses_the_first_paid_call(self) -> None:
        inner = _CountingCostlyMatcher(5, 0.01)
        capped = SpendCappedMatcher(inner, budget_usd=0.0)
        with pytest.raises(BudgetExceeded) as excinfo:
            list(capped.forward(_candidates()))
        # Exactly one call was paid for -- the cap cannot un-spend it, but it
        # stops everything after it. This IS the "budget + 1 call" bound.
        assert inner.produced == 1
        assert len(excinfo.value.partial_judgements) == 1

    def test_zero_budget_still_allows_unlimited_free_calls(self) -> None:
        """$0 spend never exceeds a $0 budget -- a free matcher is not blocked."""
        capped = SpendCappedMatcher(_CountingCostlyMatcher(5, 0.0), budget_usd=0.0)
        assert len(list(capped.forward(_candidates()))) == 5

    def test_uncapped_budget_never_trips(self) -> None:
        capped = SpendCappedMatcher(
            _CountingCostlyMatcher(5, 1_000.0), budget_usd=UNCAPPED_BUDGET_USD
        )
        assert len(list(capped.forward(_candidates()))) == 5
        assert capped.monitor.spent == 5_000.0

    def test_missing_cost_provenance_is_treated_as_free(self) -> None:
        class _NoCostMatcher(Matcher[object]):
            def forward(
                self, candidates: Iterator[ERCandidate[object]]
            ) -> Iterator[PairwiseJudgement]:
                yield PairwiseJudgement(
                    left_id="a",
                    right_id="b",
                    score=0.9,
                    score_type="heuristic",
                    decision_step="fake",
                    provenance={},
                )

            def inspect_scores(
                self, judgements: list[PairwiseJudgement], sample_size: int = 10
            ) -> object:
                raise NotImplementedError

        capped = SpendCappedMatcher(_NoCostMatcher(), budget_usd=0.0)
        assert len(list(capped.forward(_candidates()))) == 1

    def test_none_cost_provenance_is_treated_as_free(self) -> None:
        class _NoneCostMatcher(Matcher[object]):
            def forward(
                self, candidates: Iterator[ERCandidate[object]]
            ) -> Iterator[PairwiseJudgement]:
                yield PairwiseJudgement(
                    left_id="a",
                    right_id="b",
                    score=0.9,
                    score_type="heuristic",
                    decision_step="fake",
                    provenance={"cost_usd": None},
                )

            def inspect_scores(
                self, judgements: list[PairwiseJudgement], sample_size: int = 10
            ) -> object:
                raise NotImplementedError

        capped = SpendCappedMatcher(_NoneCostMatcher(), budget_usd=0.0)
        assert len(list(capped.forward(_candidates()))) == 1
