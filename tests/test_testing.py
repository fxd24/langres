"""Tests for langres.testing.ScriptedJudge.

``ScriptedJudge`` is a public :class:`~langres.core.module.Module` double for
tests and examples: no network, no API key, no spend. These tests cover the
scripted-score contract (dict + callable forms, order-independent pair keys,
``default_score`` for unmapped pairs), the ``seen`` escalation-laziness spy,
field stamping (``score_type``/``decision_step``/``reasoning``/``provenance``),
``inspect_scores`` delegating to the shared ``_inspect_scores_impl``, and an
end-to-end check that scripted ``provenance["cost_usd"]`` reaches
:func:`~langres.eval.evaluate`'s ``cost.usd_total``.
"""

import pytest

from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.registry import _COMPONENT_REGISTRY
from langres.core.reports import _inspect_scores_impl
from langres.eval import evaluate
from langres.testing import ScriptedJudge


def _pair(left_id: str, right_id: str) -> ERCandidate[CompanySchema]:
    """Build a minimal CompanySchema candidate pair for scripting."""
    return ERCandidate(
        left=CompanySchema(id=left_id, name=f"Company {left_id}"),
        right=CompanySchema(id=right_id, name=f"Company {right_id}"),
        blocker_name="test",
    )


class TestScriptedJudgeForward:
    """forward() yields one scripted judgement per candidate, in order."""

    def test_yields_one_judgement_per_candidate_in_order(self) -> None:
        candidates = [_pair("a", "b"), _pair("c", "d"), _pair("e", "f")]
        judge = ScriptedJudge(
            {
                frozenset({"a", "b"}): 0.9,
                frozenset({"c", "d"}): 0.1,
                frozenset({"e", "f"}): 0.5,
            }
        )

        results = list(judge.forward(iter(candidates)))

        assert [j.left_id for j in results] == ["a", "c", "e"]
        assert [j.right_id for j in results] == ["b", "d", "f"]
        assert [j.score for j in results] == [0.9, 0.1, 0.5]

    def test_score_map_is_order_independent(self) -> None:
        """(a, b) and (b, a) hit the same score-map entry."""
        judge = ScriptedJudge({frozenset({"a", "b"}): 0.75})

        forward_score = next(iter(judge.forward(iter([_pair("a", "b")])))).score
        reverse_score = next(iter(judge.forward(iter([_pair("b", "a")])))).score

        assert forward_score == reverse_score == 0.75

    def test_default_score_applies_to_unmapped_pairs(self) -> None:
        judge = ScriptedJudge({frozenset({"a", "b"}): 0.9}, default_score=0.42)

        results = list(judge.forward(iter([_pair("c", "d")])))

        assert results[0].score == 0.42

    def test_default_score_falls_back_to_half_when_unset(self) -> None:
        judge = ScriptedJudge({})

        results = list(judge.forward(iter([_pair("x", "y")])))

        assert results[0].score == 0.5

    def test_callable_form_scores_each_candidate(self) -> None:
        judge = ScriptedJudge(lambda candidate: len(candidate.left.id) / 10.0)

        results = list(judge.forward(iter([_pair("abc", "xyz")])))

        assert results[0].score == pytest.approx(0.3)

    def test_handles_empty_candidate_stream(self) -> None:
        judge = ScriptedJudge({})

        assert list(judge.forward(iter([]))) == []
        assert judge.seen == []


class TestScriptedJudgeSeenSpy:
    """``seen`` is the escalation-laziness spy CascadeJudge-style tests rely on."""

    def test_records_every_pair_in_order_pulled(self) -> None:
        judge = ScriptedJudge({})
        candidates = [_pair("a", "b"), _pair("c", "d")]

        result_iter = judge.forward(iter(candidates))
        assert judge.seen == []

        next(result_iter)
        assert judge.seen == [frozenset({"a", "b"})]

        list(result_iter)
        assert judge.seen == [frozenset({"a", "b"}), frozenset({"c", "d"})]

    def test_records_unordered_pair_regardless_of_left_right_order(self) -> None:
        judge = ScriptedJudge({frozenset({"a", "b"}): 1.0, frozenset({"c", "d"}): 0.0})

        list(judge.forward(iter([_pair("b", "a"), _pair("d", "c")])))

        assert judge.seen == [frozenset({"a", "b"}), frozenset({"c", "d"})]


class TestScriptedJudgeFieldStamping:
    """score_type / decision_step / reasoning / provenance are stamped as scripted."""

    def test_score_type_decision_step_and_reasoning_are_applied(self) -> None:
        judge = ScriptedJudge(
            {frozenset({"a", "b"}): 0.8},
            score_type="prob_llm",
            decision_step="scripted_stage",
            reasoning="because the test said so",
        )

        result = next(iter(judge.forward(iter([_pair("a", "b")]))))

        assert result.score_type == "prob_llm"
        assert result.decision_step == "scripted_stage"
        assert result.reasoning == "because the test said so"

    def test_defaults_are_heuristic_scripted_and_no_reasoning(self) -> None:
        judge = ScriptedJudge({frozenset({"a", "b"}): 0.5})

        result = next(iter(judge.forward(iter([_pair("a", "b")]))))

        assert result.score_type == "heuristic"
        assert result.decision_step == "scripted"
        assert result.reasoning is None

    def test_provenance_reaches_the_judgement(self) -> None:
        judge = ScriptedJudge({frozenset({"a", "b"}): 0.5}, provenance={"cost_usd": 0.001})

        result = next(iter(judge.forward(iter([_pair("a", "b")]))))

        assert result.provenance == {"cost_usd": 0.001}

    def test_provenance_defaults_to_empty_dict(self) -> None:
        judge = ScriptedJudge({frozenset({"a", "b"}): 0.5})

        result = next(iter(judge.forward(iter([_pair("a", "b")]))))

        assert result.provenance == {}

    def test_provenance_is_copied_per_judgement_not_shared(self) -> None:
        """Mutating one judgement's provenance must not leak into another's."""
        judge = ScriptedJudge(
            {frozenset({"a", "b"}): 0.5, frozenset({"c", "d"}): 0.5},
            provenance={"cost_usd": 0.001},
        )

        results = list(judge.forward(iter([_pair("a", "b"), _pair("c", "d")])))
        results[0].provenance["cost_usd"] = 999.0

        assert results[1].provenance["cost_usd"] == 0.001


class TestScriptedJudgeInspectScores:
    """inspect_scores() delegates to the shared _inspect_scores_impl."""

    def test_matches_shared_impl_output(self) -> None:
        judge = ScriptedJudge({frozenset({"a", "b"}): 0.9, frozenset({"c", "d"}): 0.2})
        judgements = list(judge.forward(iter([_pair("a", "b"), _pair("c", "d")])))

        report = judge.inspect_scores(judgements)

        assert report == _inspect_scores_impl(judgements)

    def test_handles_empty_judgements_list(self) -> None:
        judge = ScriptedJudge({})

        report = judge.inspect_scores([])

        assert report == _inspect_scores_impl([])

    def test_respects_sample_size(self) -> None:
        judge = ScriptedJudge(lambda c: 0.5)
        judgements = list(judge.forward(iter([_pair(str(i), str(i + 1)) for i in range(20)])))

        report = judge.inspect_scores(judgements, sample_size=3)

        assert report == _inspect_scores_impl(judgements, 3)


class TestScriptedJudgeIsATestDouble:
    """ScriptedJudge must stay out of the Resolver.load config-registry dispatch."""

    def test_is_not_registered_as_a_component(self) -> None:
        assert ScriptedJudge not in _COMPONENT_REGISTRY.values()


class TestScriptedJudgeIntegrationWithEvaluate:
    """A scripted provenance reaches evaluate()'s cost accounting -- an honest E2E."""

    def test_scripted_cost_usd_reaches_evaluate_cost_total(self) -> None:
        candidates = [_pair("a", "b"), _pair("c", "d")]
        judge = ScriptedJudge(
            {frozenset({"a", "b"}): 0.9, frozenset({"c", "d"}): 0.1},
            provenance={"cost_usd": 0.001},
        )

        result = evaluate(judge, candidates, gold_pairs={frozenset({"a", "b"})})

        assert result.cost.usd_total == pytest.approx(0.002)
        assert result.pair.precision == 1.0
        assert result.pair.recall == 1.0

    def test_forward_output_is_a_real_pairwise_judgement(self) -> None:
        """Sanity: results are genuine PairwiseJudgement instances, not doubles."""
        judge = ScriptedJudge({frozenset({"a", "b"}): 0.5})

        result = next(iter(judge.forward(iter([_pair("a", "b")]))))

        assert isinstance(result, PairwiseJudgement)


class TestScriptedJudgeAbstain:
    """The ``abstain`` predicate yields a score-less, decision-less judgement."""

    def test_abstain_predicate_yields_an_abstention(self) -> None:
        judge = ScriptedJudge(
            {frozenset({"a", "b"}): 0.9, frozenset({"c", "d"}): 0.9},
            abstain=lambda cand: cand.left.id == "c",
        )

        judgements = list(judge.forward(iter([_pair("a", "b"), _pair("c", "d")])))

        assert judgements[0].score == 0.9
        assert judgements[0].is_abstain is False
        assert judgements[1].is_abstain is True
        assert judgements[1].score is None
        assert judgements[1].decision is None

    def test_no_abstain_predicate_never_abstains(self) -> None:
        judge = ScriptedJudge({frozenset({"a", "b"}): 0.9})

        [judgement] = list(judge.forward(iter([_pair("a", "b")])))

        assert judgement.is_abstain is False
