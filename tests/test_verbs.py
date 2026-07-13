"""Tests for langres.verbs (link/dedupe: the three-verb DX layer).

Zero-spend throughout. The only judge that could ever cost money
("zero_shot_llm") is exercised exclusively via an injected
``DSPyJudge(lm=DummyLM(...))`` -- the documented $0 test seam (never
judge="auto" against a real key, and never past construction for a bare
"zero_shot_llm" string).
"""

import inspect
import math
import subprocess
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from dspy.utils.dummies import DummyLM
from pydantic import BaseModel

from langres import NoJudgeAvailableError
from langres.clients.openrouter import BudgetExceeded
from langres.core.judgement_log import JudgementLog
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.module import Module
from langres.core.modules.dspy_judge import DSPyJudge
from langres.core.presets import ResolvedModule, _SpendCappedModule
from langres.verbs import (
    DedupeResult,
    LinkVerdict,
    _check_no_duplicate_ids,
    _coerce_log,
    _coerce_scalar,
    _field_union,
    _infer,
    _infer_schema,
    _resolve_ids,
    dedupe,
    link,
)


class VerbCompany(BaseModel):
    id: str
    name: str | None = None
    address: str | None = None


class _EmptyModule(Module[object]):
    """Yields nothing for any candidate -- exercises link()'s no-judgement guard."""

    def forward(self, candidates: Iterator[ERCandidate[object]]) -> Iterator[PairwiseJudgement]:
        list(candidates)
        return
        yield  # pragma: no cover - makes this a generator function

    def inspect_scores(self, judgements: list[PairwiseJudgement], sample_size: int = 10) -> object:
        raise NotImplementedError


class _AbstainingModule(Module[object]):
    """Yields one abstention (no decision, no score) -- exercises link()'s abstain guard."""

    def forward(self, candidates: Iterator[ERCandidate[object]]) -> Iterator[PairwiseJudgement]:
        for candidate in candidates:
            yield PairwiseJudgement(
                left_id=candidate.left.id,  # type: ignore[attr-defined]
                right_id=candidate.right.id,  # type: ignore[attr-defined]
                score=None,
                score_type="prob_llm",
                decision_step="parse_error",
                provenance={"parse_error": True},
            )

    def inspect_scores(self, judgements: list[PairwiseJudgement], sample_size: int = 10) -> object:
        raise NotImplementedError


class _CostlyModule(Module[object]):
    """Yields N judgements at a fixed cost each -- for cap-breach tests."""

    def __init__(self, n: int, cost_each: float) -> None:
        self._n = n
        self._cost_each = cost_each

    def forward(self, candidates: Iterator[ERCandidate[object]]) -> Iterator[PairwiseJudgement]:
        pairs = list(candidates)
        for i in range(min(self._n, len(pairs))):
            pair = pairs[i]
            yield PairwiseJudgement(
                left_id=pair.left.id,  # type: ignore[attr-defined]
                right_id=pair.right.id,  # type: ignore[attr-defined]
                score=0.9,
                score_type="prob_llm",
                decision_step="fake",
                provenance={"cost_usd": self._cost_each},
            )

    def inspect_scores(self, judgements: list[PairwiseJudgement], sample_size: int = 10) -> object:
        raise NotImplementedError


def _dummy_judge(match: bool = True, prob: float = 0.9) -> DSPyJudge[VerbCompany]:
    answer = {
        "reasoning": "same company" if match else "different company",
        "match": "True" if match else "False",
        "match_probability": str(prob),
    }
    return DSPyJudge(lm=DummyLM([answer] * 20), entity_noun="company")


# ---------------------------------------------------------------------------
# LinkVerdict / DedupeResult
# ---------------------------------------------------------------------------


class TestLinkVerdict:
    def test_bool_reflects_match(self) -> None:
        judgement = PairwiseJudgement(
            left_id="a",
            right_id="b",
            score=0.9,
            score_type="heuristic",
            decision_step="x",
            provenance={},
        )
        verdict = LinkVerdict(
            match=True,
            score=0.9,
            reasoning=None,
            judge_used="string",
            score_type="heuristic",
            threshold=0.5,
            judgement=judgement,
        )
        assert bool(verdict) is True
        assert verdict
        no_match = verdict.model_copy(update={"match": False})
        assert bool(no_match) is False
        assert not no_match

    def test_repr_is_friendly(self) -> None:
        judgement = PairwiseJudgement(
            left_id="a",
            right_id="b",
            score=0.42,
            score_type="heuristic",
            decision_step="x",
            provenance={},
        )
        verdict = LinkVerdict(
            match=False,
            score=0.42,
            reasoning=None,
            judge_used="string",
            score_type="heuristic",
            threshold=0.5,
            judgement=judgement,
        )
        text = repr(verdict)
        assert "NO MATCH" in text
        assert "0.420" in text
        assert "'string'" in text


class TestDedupeResult:
    def test_behaves_like_a_plain_list(self) -> None:
        result = DedupeResult(
            [{"a", "b"}], judge_used="string", score_type="heuristic", threshold=0.5
        )
        assert result == [{"a", "b"}]
        assert len(result) == 1
        assert list(result) == [{"a", "b"}]

    def test_carries_judge_metadata(self) -> None:
        result = DedupeResult([], judge_used="embedding", score_type="sim_cos", threshold=0.7)
        assert result.judge_used == "embedding"
        assert result.score_type == "sim_cos"
        assert result.threshold == 0.7

    def test_repr_includes_judge_used_and_threshold(self) -> None:
        result = DedupeResult([], judge_used="string", score_type="heuristic", threshold=0.5)
        assert "judge_used='string'" in repr(result)
        assert "threshold=0.5" in repr(result)


# ---------------------------------------------------------------------------
# Schema inference primitives
# ---------------------------------------------------------------------------


class TestCoerceScalar:
    def test_none_stays_none(self) -> None:
        assert _coerce_scalar(None) is None

    def test_nan_becomes_none_not_the_string_nan(self) -> None:
        assert _coerce_scalar(float("nan")) is None

    def test_scalars_coerce_to_str(self) -> None:
        assert _coerce_scalar(42) == "42"
        assert _coerce_scalar(3.14) == "3.14"
        assert _coerce_scalar(True) == "True"
        assert _coerce_scalar("already") == "already"

    def test_nested_list_raises_with_guidance(self) -> None:
        with pytest.raises(ValueError, match="nested list"):
            _coerce_scalar([1, 2, 3])

    def test_nested_dict_raises_with_guidance(self) -> None:
        with pytest.raises(ValueError, match="nested dict"):
            _coerce_scalar({"a": 1})


class TestFieldUnion:
    def test_union_excludes_id(self) -> None:
        records = [{"id": "1", "name": "a"}, {"id": "2", "city": "b"}]
        assert _field_union(records) == frozenset({"name", "city"})


class TestResolveIds:
    def test_all_present_uses_explicit_ids(self) -> None:
        records = [{"id": "x"}, {"id": "y"}]
        assert _resolve_ids(records) == ["x", "y"]

    def test_none_present_assigns_positional_ids(self) -> None:
        records = [{"name": "a"}, {"name": "b"}]
        assert _resolve_ids(records) == ["0", "1"]

    def test_mixed_presence_raises(self) -> None:
        records = [{"id": "x"}, {"name": "no id"}]
        with pytest.raises(ValueError, match="some records have an 'id' key"):
            _resolve_ids(records)


class TestInferSchema:
    def test_memoized_by_field_set(self) -> None:
        a = _infer_schema(frozenset({"name", "city"}))
        b = _infer_schema(frozenset({"city", "name"}))  # same set, different order
        assert a is b

    def test_deterministic_name(self) -> None:
        schema = _infer_schema(frozenset({"name"}))
        assert schema.__name__.startswith("Inferred_")
        assert len(schema.__name__) == len("Inferred_") + 8

    def test_different_field_sets_produce_different_schemas(self) -> None:
        a = _infer_schema(frozenset({"name"}))
        b = _infer_schema(frozenset({"title"}))
        assert a is not b
        assert a.__name__ != b.__name__

    def test_all_fields_are_optional_strings(self) -> None:
        schema = _infer_schema(frozenset({"name", "city"}))
        assert schema.model_fields["name"].annotation == str | None
        assert schema.model_fields["id"].is_required()


class TestInfer:
    def test_coerces_and_assigns_ids(self) -> None:
        records = [{"id": "1", "n": 42}, {"id": "2", "n": None}]
        schema, coerced = _infer(records)
        assert coerced == [{"id": "1", "n": "42"}, {"id": "2", "n": None}]
        assert schema(**coerced[0]).id == "1"

    def test_does_not_check_duplicate_ids(self) -> None:
        # link(a, a) relies on this -- _infer alone must never raise on repeats.
        records = [{"id": "same", "n": "x"}, {"id": "same", "n": "x"}]
        schema, coerced = _infer(records)
        assert len(coerced) == 2


class TestCheckNoDuplicateIds:
    def test_unique_ids_pass(self) -> None:
        _check_no_duplicate_ids(["a", "b", "c"])  # no raise

    def test_duplicate_ids_raise_with_the_offending_id(self) -> None:
        with pytest.raises(ValueError, match=r"duplicate ids in input: \['a'\]"):
            _check_no_duplicate_ids(["a", "b", "a"])


# ---------------------------------------------------------------------------
# dedupe()
# ---------------------------------------------------------------------------


class TestDedupe:
    def test_empty_input_returns_empty_result(self) -> None:
        """dedupe([]) short-circuits BEFORE judge resolution: keyless empty
        input must never raise NoJudgeAvailableError."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("pydantic_settings.sources.DotEnvSettingsSource.__call__", return_value={}),
        ):
            result = dedupe([])
        assert result == []
        assert result.judge_used == "none"

    def test_single_record_returns_empty(self) -> None:
        result = dedupe([{"id": "a", "name": "Solo"}], judge="string")
        assert result == []
        assert result.judge_used == "none"

    def test_single_record_keyless_short_circuits_before_judge_resolution(self) -> None:
        """The docstring promises a single record -> [] (no pair possible):
        one keyless record must return [] under the default judge="auto",
        never raise NoJudgeAvailableError -- zero spend is possible."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("pydantic_settings.sources.DotEnvSettingsSource.__call__", return_value={}),
        ):
            result = dedupe([{"id": "a", "name": "Solo"}])
        assert result == []
        assert result.judge_used == "none"

    def test_duplicate_id_raises(self) -> None:
        records = [{"id": "a", "name": "X"}, {"id": "a", "name": "Y"}]
        with pytest.raises(ValueError, match="duplicate ids"):
            dedupe(records, judge="string")

    def test_string_judge_inferred_schema_clusters_similar_names(self) -> None:
        records = [
            {"id": "1", "name": "Acme Corporation"},
            {"id": "2", "name": "Acme Corporation"},
            {"id": "3", "name": "Totally Unrelated Restaurant"},
        ]
        result = dedupe(records, judge="string", threshold=0.5)
        assert {"1", "2"} in result
        assert result.judge_used == "string"
        assert result.score_type == "heuristic"

    def test_result_threshold_reports_the_explicit_cut(self) -> None:
        records = [{"id": "1", "name": "Acme"}, {"id": "2", "name": "Acme"}]
        result = dedupe(records, judge="string", threshold=0.8)
        assert result.threshold == 0.8

    def test_result_threshold_reports_the_resolved_default(self) -> None:
        """threshold=None resolves to the string judge's 0.5 default -- and the
        result must report the RESOLVED value, ready for select_for_review,
        not echo the None back."""
        records = [{"id": "1", "name": "Acme"}, {"id": "2", "name": "Acme"}]
        result = dedupe(records, judge="string")
        assert result.threshold == 0.5

    def test_short_circuit_result_threshold_is_none(self) -> None:
        """The <2-records short-circuit resolves no judge, hence no cut."""
        result = dedupe([{"id": "a", "name": "Solo"}], judge="string")
        assert result.threshold is None

    def test_explicit_schema_used_verbatim(self) -> None:
        records = [
            {"id": "1", "name": "Acme Corporation", "address": "1 Main St"},
            {"id": "2", "name": "Acme Corporation", "address": "1 Main St"},
        ]
        result = dedupe(records, judge="string", schema=VerbCompany, threshold=0.5)
        assert {"1", "2"} in result

    def test_explicit_schema_no_id_key_uses_positional_ids(self) -> None:
        """BUG regression: id-less records under an explicit schema= must not
        be mis-flagged as duplicates. The old code skipped _resolve_ids() on
        the explicit-schema path and did str(record.get("id")) for every
        record -- two id-less records both read as the string "None", a
        false "duplicate ids" collision even though nothing was duplicated.
        Explicit-schema and inferred-schema must behave identically here:
        positional ids when none are present."""
        records = [
            {"name": "Acme Corporation", "address": "1 Main St"},
            {"name": "Acme Corporation", "address": "1 Main St"},
        ]
        result = dedupe(records, judge="string", schema=VerbCompany, threshold=0.5)
        assert {"0", "1"} in result

    def test_dummy_lm_e2e_at_zero_cost(self) -> None:
        """Hostile-test requirement: judge=DSPyJudge(lm=DummyLM(...)) through dedupe()."""
        records = [
            {"id": "1", "name": "Acme Corporation"},
            {"id": "2", "name": "Acme Corp"},
        ]
        result = dedupe(records, judge=_dummy_judge(match=True, prob=0.95), threshold=0.5)
        assert result.judge_used == "custom"
        assert {"1", "2"} in result

    def test_abstaining_judge_leaves_pair_unmerged_and_does_not_raise(self) -> None:
        """dedupe()'s abstain contract, the deliberate asymmetry with link().

        link() raises JudgeAbstainedError on an abstain (one caller needs a
        verdict). dedupe() judges many pairs to build clusters, so an abstained
        pair is conservatively left UNMERGED rather than aborting the whole
        batch -- one unparseable judgement must not sink a dedupe run. Here the
        only pair abstains, so the two records stay unclustered and no exception
        is raised.
        """
        records = [
            {"id": "a", "name": "Acme Corporation"},
            {"id": "b", "name": "Acme Corp"},
        ]
        result = dedupe(records, judge=_AbstainingModule(), threshold=0.5)
        # No merge, no crash: the abstained pair simply did not connect a, b.
        assert {"a", "b"} not in result
        assert all(len(cluster) == 1 for cluster in result)

    def test_cap_breach_mid_stream_raises_with_partial_judgements(self) -> None:
        records = [{"id": str(i), "name": f"n{i}"} for i in range(4)]  # C(4,2) = 6 pairs
        with pytest.raises(BudgetExceeded) as excinfo:
            dedupe(records, judge=_CostlyModule(6, 0.5), budget_usd=0.9)
        partial = excinfo.value.partial_judgements
        assert len(partial) == 2

    def test_no_key_raises_no_judge_available_error(self) -> None:
        records = [
            {"id": "1", "name": "Acme"},
            {"id": "2", "name": "Acme"},
        ]
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("pydantic_settings.sources.DotEnvSettingsSource.__call__", return_value={}),
            pytest.raises(NoJudgeAvailableError, match="no API key"),
        ):
            dedupe(records)


# ---------------------------------------------------------------------------
# link()
# ---------------------------------------------------------------------------


class TestLink:
    def test_string_judge_match(self) -> None:
        verdict = link(
            {"id": "a", "name": "Acme Corporation"},
            {"id": "b", "name": "Acme Corporation"},
            judge="string",
            threshold=0.5,
        )
        assert verdict.match is True
        assert verdict.judge_used == "string"
        assert verdict.score_type == "heuristic"

    def test_string_judge_no_match(self) -> None:
        verdict = link(
            {"id": "a", "name": "Acme Corporation"},
            {"id": "b", "name": "Totally Unrelated Restaurant"},
            judge="string",
            threshold=0.9,
        )
        assert verdict.match is False

    def test_link_a_a_is_well_defined(self) -> None:
        a = {"id": "same", "name": "Acme Corporation"}
        verdict = link(a, a, judge="string", threshold=0.5)
        assert verdict.match is True
        assert verdict.score == pytest.approx(1.0)

    def test_dummy_lm_injected_module_zero_cost(self) -> None:
        verdict = link(
            {"id": "a", "name": "Acme"},
            {"id": "b", "name": "Acme Inc"},
            judge=_dummy_judge(match=True, prob=0.87),
            threshold=0.5,
        )
        assert verdict.judge_used == "custom"
        assert verdict.score == pytest.approx(0.87)
        assert verdict.match is True

    def test_explicit_schema_used_verbatim(self) -> None:
        verdict = link(
            {"id": "a", "name": "Acme", "address": "1 Main St"},
            {"id": "b", "name": "Acme", "address": "1 Main St"},
            judge="string",
            schema=VerbCompany,
            threshold=0.5,
        )
        assert verdict.match is True

    def test_default_threshold_used_when_none(self) -> None:
        verdict = link({"id": "a", "name": "Acme"}, {"id": "b", "name": "Acme"}, judge="string")
        assert verdict.score >= 0.0  # threshold resolved without raising
        # The verdict reports the RESOLVED cut (the string judge's 0.5
        # default), not the None the caller passed.
        assert verdict.threshold == 0.5

    def test_verdict_threshold_reports_the_explicit_cut(self) -> None:
        verdict = link(
            {"id": "a", "name": "Acme"},
            {"id": "b", "name": "Acme"},
            judge="string",
            threshold=0.9,
        )
        assert verdict.threshold == 0.9

    def test_no_judgement_produced_raises_runtime_error(self) -> None:
        with pytest.raises(RuntimeError, match="produced no judgement"):
            link({"id": "a", "name": "X"}, {"id": "b", "name": "Y"}, judge=_EmptyModule())

    def test_abstaining_judge_raises_judge_abstained_error(self) -> None:
        """A judge that neither decided nor scored gives link() no verdict to return.

        link() must raise rather than fabricate a match: a ``match=None`` verdict
        would be read as "no match" by the obvious ``if verdict.match:`` and
        silently recreate the confident-no bug the contract exists to remove.
        """
        from langres import JudgeAbstainedError

        with pytest.raises(JudgeAbstainedError, match="abstained"):
            link({"id": "a", "name": "X"}, {"id": "b", "name": "Y"}, judge=_AbstainingModule())

    def test_judge_abstained_error_is_a_runtime_error(self) -> None:
        """It subclasses RuntimeError so ``except RuntimeError`` still catches it."""
        from langres import JudgeAbstainedError

        with pytest.raises(RuntimeError):
            link({"id": "a", "name": "X"}, {"id": "b", "name": "Y"}, judge=_AbstainingModule())

    def test_no_key_raises_no_judge_available_error(self) -> None:
        """link() fails fast on the keyless auto path exactly like dedupe()."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("pydantic_settings.sources.DotEnvSettingsSource.__call__", return_value={}),
            pytest.raises(NoJudgeAvailableError, match="no API key"),
        ):
            link({"id": "a", "name": "Acme"}, {"id": "b", "name": "Acme"})

    def test_cap_breach_raises_with_partial_judgements(self) -> None:
        with pytest.raises(BudgetExceeded) as excinfo:
            link(
                {"id": "a", "name": "X"},
                {"id": "b", "name": "Y"},
                judge=_CostlyModule(1, 5.0),
                budget_usd=1.0,
            )
        partial = excinfo.value.partial_judgements
        assert len(partial) == 1

    def test_zero_shot_llm_emits_pre_scoring_notice(self) -> None:
        """Covers the paid-notice branch without any real network call.

        ``resolve_judge`` is patched to report ``judge_used="zero_shot_llm"``
        (as a real key would resolve) while the actual scorer stays the $0
        DummyLM-backed module -- the notice fires, but nothing paid runs.
        """
        fake_resolved = ResolvedModule(
            _SpendCappedModule(_dummy_judge(match=True, prob=0.8), budget_usd=1.0),
            "zero_shot_llm",
            "openrouter/openai/gpt-4o-mini",
        )
        with (
            patch("langres.verbs.resolve_judge", return_value=fake_resolved),
            pytest.warns(UserWarning, match="scoring ~1 pairs"),
        ):
            verdict = link({"id": "a", "name": "X"}, {"id": "b", "name": "Y"}, judge="auto")
        assert verdict.judge_used == "zero_shot_llm"
        assert verdict.score == pytest.approx(0.8)

    def test_zero_shot_llm_unpinned_model_warns_blind_cap_via_link(self) -> None:
        """M1 regression: link()'s own notice_pre_scoring_cost call site must
        forward the effective budget too, so an explicit unpinned paid model
        warns about the blind cap instead of the reassuring $0.0000 estimate
        (same DummyLM-backed / patched resolve_judge zero-spend seam as
        test_zero_shot_llm_emits_pre_scoring_notice above)."""
        fake_resolved = ResolvedModule(
            _SpendCappedModule(_dummy_judge(match=True, prob=0.8), budget_usd=2.0),
            "zero_shot_llm",
            "unknown/model-not-in-table",
        )
        with (
            patch("langres.verbs.resolve_judge", return_value=fake_resolved),
            pytest.warns(UserWarning, match="CANNOT enforce a limit") as record,
        ):
            verdict = link(
                {"id": "a", "name": "X"}, {"id": "b", "name": "Y"}, judge="auto", budget_usd=2.0
            )
        assert "est. cost $0.0000" not in str(record[0].message)
        assert "$2.00" in str(record[0].message)
        assert verdict.judge_used == "zero_shot_llm"


@pytest.mark.slow
class TestLinkEmbeddingJudge:
    def test_embedding_judge_scores_identical_text_near_one(self) -> None:
        a = {"id": "a", "name": "Acme Corporation"}
        verdict = link(a, dict(a, id="b"), judge="embedding", threshold=0.5)
        assert verdict.score > 0.99
        assert verdict.score_type == "sim_cos"
        assert verdict.judge_used == "embedding"


# ---------------------------------------------------------------------------
# judge="auto" selection notice: which model, that money is involved, the cap.
# Must fire BEFORE any paid call, on both verbs, and ONLY on the auto path.
# ---------------------------------------------------------------------------


class _EventRecordingJudge(Module[object]):
    """$0 stand-in for the auto-picked LLM judge that records when scoring
    actually happens -- lets a test assert the selection notice fired first."""

    def __init__(self, events: list[str]) -> None:
        self._events = events

    def forward(self, candidates: Iterator[ERCandidate[object]]) -> Iterator[PairwiseJudgement]:
        for candidate in candidates:
            self._events.append("score")
            yield PairwiseJudgement(
                left_id=candidate.left.id,  # type: ignore[attr-defined]
                right_id=candidate.right.id,  # type: ignore[attr-defined]
                score=0.9,
                score_type="prob_llm",
                decision_step="fake",
                provenance={"cost_usd": 0.0},
            )

    def inspect_scores(self, judgements: list[PairwiseJudgement], sample_size: int = 10) -> object:
        raise NotImplementedError


class TestAutoSelectionNotice:
    """Zero-spend throughout: the env key is fake and ``build_judge`` is
    patched to a $0 recording judge, so nothing paid can ever run."""

    @staticmethod
    def _patch_auto_path(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-not-a-real-key")
        monkeypatch.setattr(
            "langres.core.presets._notice",
            lambda message: events.append(f"notice:{message}"),
        )
        judge_module = _EventRecordingJudge(events)
        monkeypatch.setattr(
            "langres.core.presets.build_judge",
            lambda judge, schema, *, model=None, entity_noun="entity": judge_module,
        )

    def test_dedupe_auto_notice_fires_before_any_scoring(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[str] = []
        self._patch_auto_path(monkeypatch, events)
        records = [{"id": "1", "name": "Acme"}, {"id": "2", "name": "Acme Corp"}]
        result = dedupe(records, judge="auto")
        assert result.judge_used == "zero_shot_llm"
        selection = [i for i, e in enumerate(events) if "selected the LLM judge" in e]
        assert len(selection) == 1
        assert "score" in events
        assert selection[0] < events.index("score")

    def test_link_auto_notice_fires_before_any_scoring(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[str] = []
        self._patch_auto_path(monkeypatch, events)
        verdict = link({"id": "a", "name": "Acme"}, {"id": "b", "name": "Acme Corp"}, judge="auto")
        assert verdict.judge_used == "zero_shot_llm"
        selection = [i for i, e in enumerate(events) if "selected the LLM judge" in e]
        assert len(selection) == 1
        assert "score" in events
        assert selection[0] < events.index("score")

    def test_dedupe_auto_honors_model_override_and_notice_names_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: the auto pick used to clobber a caller-supplied model=."""
        seen_models: list[str | None] = []
        judge_module = _EventRecordingJudge([])

        def fake_build_judge(
            judge: object,
            schema: object,
            *,
            model: str | None = None,
            entity_noun: str = "entity",
        ) -> Module[object]:
            seen_models.append(model)
            return judge_module

        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-not-a-real-key")
        monkeypatch.setattr("langres.core.presets.build_judge", fake_build_judge)
        records = [{"id": "1", "name": "Acme"}, {"id": "2", "name": "Acme Corp"}]
        with pytest.warns(UserWarning, match=r"selected the LLM judge 'openai/gpt-5-mini'"):
            result = dedupe(records, judge="auto", model="openai/gpt-5-mini")
        assert result.judge_used == "zero_shot_llm"
        assert seen_models == ["openai/gpt-5-mini"]

    def test_string_judge_emits_no_selection_notice(self) -> None:
        """The notice is auto-only: an explicit judge stays warning-free."""
        records = [{"id": "1", "name": "Acme"}, {"id": "2", "name": "Acme Corp"}]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = dedupe(records, judge="string", threshold=0.5)
        assert result.judge_used == "string"
        assert not caught, f"unexpected warnings: {[str(w.message) for w in caught]}"


def test_front_door_exceptions_are_root_exported() -> None:
    """After fail-fast, NoJudgeAvailableError and BudgetExceeded are exactly
    the two exceptions a front-door user must catch -- both live on `langres`
    (hiding one in langres.clients.openrouter would read as inconsistency)."""
    import langres
    from langres import BudgetExceeded as root_budget_exceeded
    from langres import NoJudgeAvailableError as root_no_judge
    from langres.core.presets import NoJudgeAvailableError as presets_no_judge

    assert root_no_judge is presets_no_judge
    assert root_budget_exceeded is BudgetExceeded  # langres.clients.openrouter's class
    assert "NoJudgeAvailableError" in langres.__all__
    assert "BudgetExceeded" in langres.__all__
    assert issubclass(root_no_judge, RuntimeError)


# ---------------------------------------------------------------------------
# log= (W0.2 JudgementLog signal seam): opt-in, kwarg-only, zero overhead when
# omitted (E10 boundary-component wrapper -- see langres.core.judgement_log).
# ---------------------------------------------------------------------------


class TestCoerceLog:
    def test_none_stays_none(self) -> None:
        assert _coerce_log(None) is None

    def test_judgementlog_instance_passes_through(self, tmp_path: Path) -> None:
        log = JudgementLog(tmp_path / "log.jsonl")
        assert _coerce_log(log) is log

    def test_path_or_str_is_wrapped_in_a_judgementlog(self, tmp_path: Path) -> None:
        coerced = _coerce_log(tmp_path / "log.jsonl")
        assert isinstance(coerced, JudgementLog)
        assert coerced.path == tmp_path / "log.jsonl"


class TestDedupeWithLog:
    def test_writes_a_jsonl_log_that_round_trips(self, tmp_path: Path) -> None:
        records = [
            {"id": "1", "name": "Acme Corporation"},
            {"id": "2", "name": "Acme Corporation"},
            {"id": "3", "name": "Totally Unrelated Restaurant"},
        ]
        log_path = tmp_path / "run.jsonl"
        result = dedupe(records, judge="string", threshold=0.5, log=log_path)

        rows = JudgementLog(log_path).read()
        assert len(rows) == 3  # C(3,2) all-pairs candidates
        assert all(row["v"] == 3 for row in rows)
        # A string judge carries no token usage — the vector is logged as null.
        assert all(row["usage"] is None for row in rows)
        assert {"1", "2"} in result

    def test_accepts_a_judgementlog_instance_directly(self, tmp_path: Path) -> None:
        records = [
            {"id": "1", "name": "Acme Corporation"},
            {"id": "2", "name": "Acme Corporation"},
        ]
        log = JudgementLog(tmp_path / "run.jsonl")
        dedupe(records, judge="string", threshold=0.5, log=log)
        assert len(log.read()) == 1

    def test_verdict_agrees_with_the_resolved_threshold(self, tmp_path: Path) -> None:
        records = [
            {"id": "1", "name": "Acme Corporation"},
            {"id": "2", "name": "Acme Corporation"},
            {"id": "3", "name": "Totally Unrelated Restaurant"},
        ]
        log = JudgementLog(tmp_path / "run.jsonl")
        dedupe(records, judge="string", threshold=0.5, log=log)
        rows = {(row["left_id"], row["right_id"]): row for row in log.read()}
        assert rows[("1", "2")]["verdict"] is True
        assert rows[("1", "3")]["verdict"] is False

    def test_features_true_includes_the_judge_provenance(self, tmp_path: Path) -> None:
        records = [
            {"id": "1", "name": "Acme Corporation"},
            {"id": "2", "name": "Acme Corporation"},
        ]
        log = JudgementLog(tmp_path / "run.jsonl", features=True)
        dedupe(records, judge="string", threshold=0.5, log=log)
        row = log.read()[0]
        assert "similarities" in row["provenance"]

    def test_empty_input_never_touches_the_log(self, tmp_path: Path) -> None:
        log_path = tmp_path / "run.jsonl"
        dedupe([], log=log_path)
        assert not log_path.exists()

    def test_log_omitted_is_identical_to_log_none(self) -> None:
        records = [
            {"id": "1", "name": "Acme Corporation"},
            {"id": "2", "name": "Acme Corporation"},
            {"id": "3", "name": "Totally Unrelated Restaurant"},
        ]
        without_kwarg = dedupe(records, judge="string", threshold=0.5)
        with_none = dedupe(records, judge="string", threshold=0.5, log=None)
        assert list(without_kwarg) == list(with_none)
        assert without_kwarg.judge_used == with_none.judge_used
        assert without_kwarg.score_type == with_none.score_type

    def test_budget_breach_still_logs_the_tripping_judgement(self, tmp_path: Path) -> None:
        """Regression (codex review, PR #62): LoggingModule wraps the
        already spend-capped module, so without this fix the paid judgement
        that trips the cap -- recorded on BudgetExceeded.partial_judgements
        but never yielded -- would be silently missing from the JSONL."""
        records = [{"id": str(i), "name": f"n{i}"} for i in range(4)]  # C(4,2) = 6 pairs
        log = JudgementLog(tmp_path / "run.jsonl")
        with pytest.raises(BudgetExceeded) as excinfo:
            dedupe(records, judge=_CostlyModule(6, 0.5), budget_usd=0.9, log=log)
        partial = excinfo.value.partial_judgements
        rows = log.read()
        assert len(rows) == len(partial) == 2
        assert [(r["left_id"], r["right_id"]) for r in rows] == [
            (j.left_id, j.right_id) for j in partial
        ]


class TestLinkWithLog:
    def test_writes_one_row_matching_the_verdict(self, tmp_path: Path) -> None:
        log_path = tmp_path / "run.jsonl"
        verdict = link(
            {"id": "a", "name": "Acme Corporation"},
            {"id": "b", "name": "Acme Corporation"},
            judge="string",
            threshold=0.5,
            log=log_path,
        )
        rows = JudgementLog(log_path).read()
        assert len(rows) == 1
        assert rows[0]["left_id"] == "a"
        assert rows[0]["right_id"] == "b"
        assert rows[0]["verdict"] == verdict.match

    def test_log_omitted_is_identical_to_log_none(self) -> None:
        a = {"id": "a", "name": "Acme Corporation"}
        b = {"id": "b", "name": "Acme Corporation"}
        without_kwarg = link(a, b, judge="string", threshold=0.5)
        with_none = link(a, b, judge="string", threshold=0.5, log=None)
        assert without_kwarg.match == with_none.match
        assert without_kwarg.score == pytest.approx(with_none.score)

    def test_budget_breach_still_logs_the_tripping_judgement(self, tmp_path: Path) -> None:
        """Regression (codex review, PR #62) -- see the matching dedupe() test."""
        log = JudgementLog(tmp_path / "run.jsonl")
        with pytest.raises(BudgetExceeded) as excinfo:
            link(
                {"id": "a", "name": "X"},
                {"id": "b", "name": "Y"},
                judge=_CostlyModule(1, 5.0),
                budget_usd=1.0,
                log=log,
            )
        partial = excinfo.value.partial_judgements
        rows = log.read()
        assert len(rows) == len(partial) == 1
        assert (rows[0]["left_id"], rows[0]["right_id"]) == (
            partial[0].left_id,
            partial[0].right_id,
        )


def test_log_param_is_keyword_only_on_both_verbs() -> None:
    for fn in (link, dedupe):
        params = inspect.signature(fn).parameters
        assert "log" in params
        assert params["log"].kind == inspect.Parameter.KEYWORD_ONLY


# ---------------------------------------------------------------------------
# Hostile: 10k-record fuzz (nested values + mixed types), per adopted plan
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_infer_10k_record_fuzz_with_mixed_types() -> None:
    records = []
    for i in range(10_000):
        records.append(
            {
                "id": str(i),
                "n": i if i % 3 == 0 else (float(i) if i % 3 == 1 else str(i)),
                "maybe_nan": float("nan") if i % 7 == 0 else "ok",
                "missing_sometimes": "present" if i % 2 == 0 else None,
            }
        )
    schema, coerced = _infer(records)
    assert len(coerced) == 10_000
    assert all(isinstance(record["n"], str) for record in coerced)
    assert all(record["maybe_nan"] is None or record["maybe_nan"] == "ok" for record in coerced)
    # every coerced record round-trips through the inferred schema
    schema(**coerced[0])
    schema(**coerced[-1])


def test_infer_fuzz_rejects_nested_value_with_guidance() -> None:
    records = [{"id": "1", "n": 1}, {"id": "2", "n": {"nested": True}}]
    with pytest.raises(ValueError, match="nested dict"):
        _infer(records)


def test_infer_fuzz_isnan_is_only_checked_for_floats() -> None:
    # A non-float value can't be nan; this just documents math.isnan's guard.
    assert math.isnan(float("nan"))
    assert _coerce_scalar(0) == "0"


# ---------------------------------------------------------------------------
# Import-safety: dspy must stay out of sys.modules for plain imports.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_import_langres_verbs_does_not_import_dspy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, langres.verbs; "
            "assert 'dspy' not in sys.modules, 'dspy leaked into import langres.verbs'; "
            "print('OK')",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"import-safety check failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


@pytest.mark.slow
def test_import_langres_top_level_does_not_import_dspy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, langres; "
            "assert 'dspy' not in sys.modules, 'dspy leaked into import langres'; "
            "print('OK')",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"import-safety check failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
