"""Tests for the inspection contract: the ``Inspectable`` Protocol (W1, I1).

``inspect_scores`` used to be an ``@abstractmethod`` on ``Matcher``, so a
Matcher that had nothing to inspect still had to implement it -- 21 classes
carried a delegation or a ``raise NotImplementedError`` stub for a method with
two callers, both pass-throughs. It is now a runtime-checkable structural
Protocol: a Matcher opts in by implementing the method, and callers detect it
with ``isinstance()``. Same convention (and same rationale) as the fit hooks in
``langres.core.fit`` -- see ``tests/core/test_fit_mixins.py``.

These tests pin the contract from both sides: not implementing it is now
*expressible* (the I1 proof below), and the two real callers still delegate to
the matchers that do implement it.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from langres.core.inspection import Inspectable
from langres.tracking.judgement_log import JudgementLog, LoggingMatcher
from langres.core.matcher import Matcher
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.spend_cap import SpendCappedMatcher as _SpendCappedMatcher
from langres.core.reports import ScoreInspectionReport


def _judgement() -> PairwiseJudgement:
    return PairwiseJudgement(
        left_id="a",
        right_id="b",
        score=0.5,
        score_type="heuristic",
        decision_step="test",
        provenance={},
    )


class _BareMatcher(Matcher[object]):
    """A Matcher implementing ONLY ``forward`` -- no ``inspect_scores`` at all.

    Before W1 this class could not be instantiated: ``inspect_scores`` was
    abstract, so ``Matcher`` refused to construct without it. That it defines
    nothing but ``forward`` IS the assertion.
    """

    def forward(self, candidates: Iterator[ERCandidate[object]]) -> Iterator[PairwiseJudgement]:
        return iter([])


class _InspectableMatcher(Matcher[object]):
    """A Matcher that opts in by implementing ``inspect_scores``."""

    def forward(self, candidates: Iterator[ERCandidate[object]]) -> Iterator[PairwiseJudgement]:
        return iter([])

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        return ScoreInspectionReport(
            total_judgements=len(judgements),
            score_distribution={},
            high_scoring_examples=[],
            low_scoring_examples=[],
            recommendations=[],
        )


def test_a_matcher_without_inspect_scores_instantiates() -> None:
    """I1: ``forward`` is the ONLY abstract method on the Matcher contract."""
    matcher = _BareMatcher()  # would raise TypeError while inspect_scores was abstract
    assert isinstance(matcher, Matcher)
    assert list(matcher.forward(iter([]))) == []


def test_isinstance_detects_opt_in_structurally() -> None:
    """The Protocol is opt-in: implementing the method is the whole opt-in."""
    assert isinstance(_InspectableMatcher(), Inspectable)
    assert not isinstance(_BareMatcher(), Inspectable)


def test_inspectable_needs_no_subclassing() -> None:
    """Structural typing: a plain object with the method satisfies the contract."""

    class _NotAMatcher:
        def inspect_scores(
            self, judgements: list[PairwiseJudgement], sample_size: int = 10
        ) -> ScoreInspectionReport:  # pragma: no cover — presence is what's asserted
            raise NotImplementedError

    assert isinstance(_NotAMatcher(), Inspectable)


class TestWrappersDelegate:
    """The two real callers pass through to an Inspectable, and say so if they can't."""

    def test_logging_matcher_rejects_a_non_inspectable_module(self, tmp_path: Path) -> None:
        wrapped = LoggingMatcher(
            _BareMatcher(), log=JudgementLog(tmp_path / "l.jsonl"), threshold=0.5
        )
        with pytest.raises(TypeError, match="does not implement inspect_scores"):
            wrapped.inspect_scores([_judgement()])

    def test_spend_capped_matcher_rejects_a_non_inspectable_module(self) -> None:
        wrapped = _SpendCappedMatcher(_BareMatcher(), budget_usd=1.0)
        with pytest.raises(TypeError, match="does not implement inspect_scores"):
            wrapped.inspect_scores([_judgement()])

    def test_spend_capped_matcher_delegates_to_an_inspectable_module(self) -> None:
        wrapped = _SpendCappedMatcher(_InspectableMatcher(), budget_usd=1.0)
        assert wrapped.inspect_scores([_judgement()]).total_judgements == 1
