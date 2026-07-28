"""The per-score-family default threshold: the map, the resolver, and the wiring.

``MethodSpec.default_threshold`` existed with **zero runtime readers** — every
front door hard-coded its own ``0.5`` instead — so the field could say one thing
while the shipped behaviour said another and nothing would notice. These tests
are what makes that impossible now: they check the map is complete, that
``resolve_threshold`` distinguishes "omitted" from "explicitly the same number",
and that the registry and the front doors both actually *read* the map.
"""

from __future__ import annotations

import warnings
from typing import Any, get_args

import pytest
from pydantic import BaseModel

from langres.architectures import (
    FuzzyString,
    Reranker,
    Retrieve,
    RetrieveLLM,
    RetrieveRerank,
    RetrieveRerankLLM,
    VectorLLMCascade,
)
from langres.core.matcher import Matcher
from langres.core.method_registry import MethodSpec, get_method, list_methods
from langres.core.score_type import DEFAULT_THRESHOLDS, ScoreType, resolve_threshold
from langres.resources import FakeEmbedder, FakeLLM, FakeReranker


# Module-unique name ON PURPOSE. Binding a schema auto-registers the class
# GLOBALLY under its bare __name__ (core.registry), so two test modules that
# both call theirs `_Record` silently share one registration -- and the loser
# gets the other's class back. That poisoned 14 tests in tests/experiments/
# when this file was written with the obvious name.
class _ThresholdDefaultsRecord(BaseModel):
    id: str
    name: str


def _build(_schema: type[BaseModel], **_kwargs: Any) -> Matcher[Any]:
    raise AssertionError("not built in these tests")  # pragma: no cover


def _built_cut(model: Any) -> float:
    """The threshold the ASSEMBLED chain will actually cut at.

    Read off ``execution_plan()`` rather than ``model.threshold`` on purpose: the
    explicit-chain recipes replace their ``__dict__`` wholesale when they bind a
    schema (``_ResearchRecipe._adopt_topology``), so the constructor attribute is
    gone by then. The plan is the number that decides matches, which is the claim
    worth testing.
    """
    cuts = [
        step.spec.params["threshold"]
        for step in model.execution_plan().steps
        if step.spec.role == "threshold_select"
    ]
    assert len(cuts) == 1, f"expected exactly one ThresholdSelect, got {cuts}"
    return float(cuts[0])


class TestDefaultThresholdsMap:
    def test_covers_every_score_family(self) -> None:
        """A missing family surfaces as a KeyError in a user's first dedupe()."""
        assert set(DEFAULT_THRESHOLDS) == set(get_args(ScoreType))

    def test_every_default_is_a_usable_cut(self) -> None:
        assert all(0.0 <= value <= 1.0 for value in DEFAULT_THRESHOLDS.values())

    def test_is_immutable(self) -> None:
        """A shared mutable default would let one caller retune every other's cut."""
        with pytest.raises(TypeError):
            DEFAULT_THRESHOLDS["heuristic"] = 0.9  # type: ignore[index]


class TestResolveThreshold:
    def test_none_takes_the_family_default(self) -> None:
        assert resolve_threshold(None, "prob_llm") == DEFAULT_THRESHOLDS["prob_llm"]
        assert resolve_threshold(None, "sim_cos") == DEFAULT_THRESHOLDS["sim_cos"]

    def test_an_explicit_value_is_returned_untouched(self) -> None:
        assert resolve_threshold(0.31, "prob_llm") == 0.31

    def test_an_explicit_value_equal_to_the_default_is_still_explicit(self) -> None:
        """0.0 and 0.5 must not be mistaken for 'unset' -- 0.0 is falsy."""
        assert resolve_threshold(0.0, "heuristic") == 0.0
        assert (
            resolve_threshold(DEFAULT_THRESHOLDS["heuristic"], "heuristic")
            == (DEFAULT_THRESHOLDS["heuristic"])
        )

    def test_an_unknown_family_raises_rather_than_falling_back(self) -> None:
        """Silently cutting an unknown scale at 0.5 is the bug this map exists to end."""
        with pytest.raises(KeyError):
            resolve_threshold(None, "not_a_family")  # type: ignore[arg-type]


class TestMethodSpecInheritsTheFamilyDefault:
    def test_omitted_threshold_comes_from_score_type(self) -> None:
        spec = MethodSpec(name="t_omitted", build=_build, score_type="prob_llm")
        assert spec.default_threshold == DEFAULT_THRESHOLDS["prob_llm"] == 0.7

    def test_an_explicit_threshold_overrides_the_family(self) -> None:
        spec = MethodSpec(
            name="t_explicit", build=_build, score_type="prob_llm", default_threshold=0.11
        )
        assert spec.default_threshold == 0.11

    def test_an_explicit_threshold_equal_to_the_field_default_survives(self) -> None:
        """mode='before' is why: after validation 'omitted' and '0.5' look identical."""
        spec = MethodSpec(
            name="t_explicit_half", build=_build, score_type="prob_llm", default_threshold=0.5
        )
        assert spec.default_threshold == 0.5

    def test_an_unknown_family_is_left_to_the_fields_own_validation(self) -> None:
        spec = MethodSpec(name="t_unknown", build=_build, score_type="not_a_family")
        assert spec.default_threshold == 0.5

    @pytest.mark.parametrize("name", sorted(list_methods()))
    def test_every_registered_method_matches_its_family(self, name: str) -> None:
        """No built-in silently deviates: the registry IS the family map, applied."""
        spec = get_method(name)
        assert spec.default_threshold == DEFAULT_THRESHOLDS[spec.score_type]  # type: ignore[index]


class TestFrontDoorsResolveTheFamilyDefault:
    """The gate that would have caught the original bug: a front door that stops
    reading the map (or goes back to a literal) fails here, not in a user's data.
    """

    def test_fuzzy_string(self) -> None:
        assert FuzzyString().threshold == DEFAULT_THRESHOLDS["heuristic"]
        assert FuzzyString(threshold=0.83).threshold == 0.83

    def test_vector_llm_cascade_takes_the_llm_family_cut_not_0_5(self) -> None:
        model = VectorLLMCascade(llm="openrouter/openai/gpt-4o-mini")
        assert model.threshold == DEFAULT_THRESHOLDS["prob_llm"] == 0.7
        assert VectorLLMCascade(llm="openrouter/openai/gpt-4o-mini", threshold=0.5).threshold == 0.5

    def test_retrieve_takes_sim_cos(self) -> None:
        model = Retrieve(embedder=FakeEmbedder(), schema=_ThresholdDefaultsRecord)
        assert _built_cut(model) == DEFAULT_THRESHOLDS["sim_cos"]

    def test_retrieve_rerank_takes_the_rerank_ops_declared_family(self) -> None:
        model = RetrieveRerank(
            embedder=FakeEmbedder(), reranker=FakeReranker(), schema=_ThresholdDefaultsRecord
        )
        assert _built_cut(model) == DEFAULT_THRESHOLDS["heuristic"]

    def test_reranker_for_schema_no_longer_requires_a_threshold(self) -> None:
        """It used to demand an explicit number for a score built by the same
        ``WeightedAverageMatcher``, over the same features, as ``FuzzyString``.
        """
        assert (
            _built_cut(Reranker.for_schema(_ThresholdDefaultsRecord, k=2))
            == DEFAULT_THRESHOLDS["heuristic"]
        )
        assert (
            _built_cut(Reranker.for_schema(_ThresholdDefaultsRecord, k=2, threshold=0.85)) == 0.85
        )

    def test_retrieve_still_range_checks_after_resolving(self) -> None:
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            Retrieve(embedder=FakeEmbedder(), schema=_ThresholdDefaultsRecord, threshold=1.5)


class TestInertThresholdIsLoudNotSilent:
    """``RetrieveLLM``/``RetrieveRerankLLM`` parse the LLM answer into a
    ``decision``, and ``predicted_match`` gives a decision precedence over any
    score — so their ``threshold`` cannot move the output. It must say so.
    """

    def _records(self) -> list[dict[str, str]]:
        return [
            {"id": "a1", "name": "Acme Corporation"},
            {"id": "a2", "name": "Acme Corp"},
            {"id": "b1", "name": "Globex Inc"},
        ]

    def _retrieve_llm(self, threshold: float | None) -> RetrieveLLM:
        return RetrieveLLM(
            embedder=FakeEmbedder(),
            llm=FakeLLM(default_response="MATCH"),
            schema=_ThresholdDefaultsRecord,
            retrieve_k=3,
            llm_k=3,
            threshold=threshold,
        )

    def test_an_explicit_threshold_warns(self) -> None:
        with pytest.warns(UserWarning, match="has no effect"):
            self._retrieve_llm(0.9)

    def test_retrieve_rerank_llm_warns_too(self) -> None:
        with pytest.warns(UserWarning, match="has no effect"):
            RetrieveRerankLLM(
                embedder=FakeEmbedder(),
                reranker=FakeReranker(),
                llm=FakeLLM(default_response="MATCH"),
                schema=_ThresholdDefaultsRecord,
                retrieve_k=3,
                llm_k=3,
                threshold=0.9,
            )

    def test_the_default_does_not_warn(self) -> None:
        """None means the caller expressed no preference; there is nothing to contradict."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._retrieve_llm(None)

    def test_the_warning_is_true__output_really_is_threshold_invariant(self) -> None:
        """Asserting the *claim* the warning makes, not just that a warning fired."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs = [
                sorted(tuple(sorted(c)) for c in self._retrieve_llm(t).dedupe(self._records()))
                for t in (0.0, 0.5, 0.99)
            ]
        assert outputs[0] == outputs[1] == outputs[2]

    def test_rapidfuzz_matcher_threshold_is_inert_too(self) -> None:
        """The third decorative knob in the PR #250 inventory, pinned as behaviour.

        ``RapidfuzzMatcher`` range-checks and stores ``threshold`` and then never
        reads it -- it is a *ranker*, so the caller's cut decides. Verified by
        construction as well as by output: ``method_registry._build_rapidfuzz``
        passes no threshold at all, and the class has no ``config``/``from_config``
        that could round-trip one.
        """
        from langres.core.matchers.rapidfuzz import RapidfuzzMatcher
        from langres.core.models import ERCandidate

        def _score_at(threshold: float) -> float:
            matcher: RapidfuzzMatcher[Any] = RapidfuzzMatcher(
                field_extractors={"name": (lambda e: str(e.name), 1.0)},
                threshold=threshold,
            )
            candidate: ERCandidate[Any] = ERCandidate(
                left=_ThresholdDefaultsRecord(id="a", name="Acme Corporation"),
                right=_ThresholdDefaultsRecord(id="b", name="Acme Corp"),
                blocker_name="test",
            )
            judgement = next(iter(matcher.forward(iter([candidate]))))
            assert judgement.score is not None
            return judgement.score

        # A threshold above AND below the pair's similarity: if the knob were
        # read at all, at least one of these would differ.
        assert _score_at(0.0) == _score_at(0.5) == _score_at(1.0)

    def test_a_scoring_recipe_is_NOT_threshold_invariant(self) -> None:
        """The control: without it, the test above would pass on a broken pipeline."""
        outputs = {
            t: sorted(
                tuple(sorted(c))
                for c in Retrieve(
                    embedder=FakeEmbedder(),
                    schema=_ThresholdDefaultsRecord,
                    retrieve_k=3,
                    threshold=t,
                ).dedupe(self._records())
            )
            for t in (0.0, 0.99)
        }
        assert outputs[0.0] != outputs[0.99]
