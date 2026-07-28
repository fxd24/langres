"""Tests for the self-describing result models (`langres.core.results`).

`LinkVerdict` is what `ERModel.compare` hands a user, so its two affordances --
`if verdict:` and a readable `repr` -- are public API, not conveniences. Both are
documented in the class docstring and neither had a test.
"""

from langres.core.models import PairwiseJudgement
from langres.core.results import LinkVerdict


def _judgement(score: float | None = 0.91) -> PairwiseJudgement:
    return PairwiseJudgement(
        left_id="a",
        right_id="b",
        score=score,
        score_type="heuristic",
        decision_step="test",
        provenance={},
    )


def _verdict(*, match: bool, score: float | None = 0.91) -> LinkVerdict:
    return LinkVerdict(
        match=match,
        score=score,
        architecture="FuzzyString",
        score_type="heuristic",
        threshold=0.5,
        judgement=_judgement(score),
    )


class TestLinkVerdictTruthiness:
    """`if model.compare(a, b):` must read the verdict, not the object's existence.

    Without `__bool__`, every LinkVerdict is truthy -- so the documented one-liner
    would silently report "match" for a non-match. That is a wrong answer, not a
    missing feature.
    """

    def test_a_match_is_truthy(self) -> None:
        assert _verdict(match=True)

    def test_a_non_match_is_falsy(self) -> None:
        assert not _verdict(match=False)

    def test_truthiness_follows_match_not_score(self) -> None:
        # A high score that did not clear the cut is still a non-match.
        assert not _verdict(match=False, score=0.99)
        # And a match with no score at all (a decider judge) is still truthy.
        assert _verdict(match=True, score=None)


class TestLinkVerdictRepr:
    def test_repr_names_the_verdict_score_and_architecture(self) -> None:
        text = repr(_verdict(match=True))
        assert "MATCH" in text
        assert "0.910" in text
        assert "FuzzyString" in text

    def test_repr_says_no_match_and_handles_a_scoreless_decider(self) -> None:
        text = repr(_verdict(match=False, score=None))
        assert "NO MATCH" in text
        assert "n/a" in text  # never a fabricated 0.000
