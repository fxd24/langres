"""Tests for langres.core.presets (judge="auto" resolution + spend-capped Resolver).

Zero-spend throughout: real network/LLM calls are never made. The
"zero_shot_llm" branch is only ever exercised up to *construction* (verifying
the DSPyJudge instance's model/price), never ``.forward()`` -- any pairwise
scoring test injects a DummyLM-backed ``DSPyJudge`` or a tiny fake Module
instead. "embedding" tests that need a real ``.encode()`` call load the local
MiniLM model (no API key, no paid call) and are marked ``@pytest.mark.slow``.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest
from dspy.utils.dummies import DummyLM
from pydantic import BaseModel

from langres.clients.openrouter import PRICES_PER_1M, BudgetExceeded
from langres.clients.settings import Settings
from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.blockers.vector import VectorBlocker
from langres.core.judges.embedding_score import EmbeddingScoreJudge
from langres.core.judges.weighted_average import WeightedAverageJudge
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.module import Module
from langres.core.modules.dspy_judge import DSPyJudge
from langres.core.presets import (
    DEFAULT_BUDGET_USD,
    _ALL_PAIRS_MAX_N,
    _build_vector_blocker,
    _estimate_n_pairs,
    _OPENAI_MODEL,
    _OPENROUTER_MODEL,
    _SpendCappedModule,
    _text_field_extractor,
    build_embedding_candidate,
    build_judge,
    build_resolver,
    choose_auto_judge,
    notice_pre_scoring_cost,
    resolve_judge,
)


class PresetCompany(BaseModel):
    id: str
    name: str | None = None
    address: str | None = None


class PresetProduct(BaseModel):
    id: str
    title: str | None = None
    brand: str | None = None


def _settings(*, openrouter: str | None = None, openai: str | None = None) -> Settings:
    return Settings(openrouter_api_key=openrouter, openai_api_key=openai)


class _FakeCostlyModule(Module[object]):
    """Yields N judgements each costing a fixed amount -- for cap-breach tests."""

    def __init__(self, n: int, cost_each: float) -> None:
        self._n = n
        self._cost_each = cost_each

    def forward(self, candidates: Iterator[ERCandidate[object]]) -> Iterator[PairwiseJudgement]:
        list(candidates)  # drain (content unused)
        for i in range(self._n):
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


class _FakeVectorIndex:
    """No-op stand-in for a real FAISS index -- create_index does nothing."""

    def create_index(self, texts: list[str]) -> None:
        pass


class _FakeEmptyStreamBlocker:
    """A fake VectorBlocker whose stream() yields no candidates (L2 test seam).

    Exercises build_embedding_candidate's defensive "no candidate" guard
    without needing a real, genuinely-broken VectorBlocker -- that case
    "should never happen" for a real 2-record stream, so it's only reachable
    by substituting a blocker built this way.
    """

    vector_index = _FakeVectorIndex()

    @staticmethod
    def schema_factory(record: dict[str, Any]) -> dict[str, Any]:
        return record

    @staticmethod
    def text_field_extractor(entity: dict[str, Any]) -> str:
        return str(entity)

    @staticmethod
    def stream(records: list[dict[str, Any]]) -> Iterator[ERCandidate[Any]]:
        return iter(())


# ---------------------------------------------------------------------------
# choose_auto_judge
# ---------------------------------------------------------------------------


class TestChooseAutoJudge:
    def test_no_keys_falls_back_to_string_with_notice(self) -> None:
        with pytest.warns(UserWarning, match="no OPENROUTER_API_KEY or OPENAI_API_KEY"):
            judge, model, reason = choose_auto_judge(_settings())
        assert judge == "string"
        assert model is None
        assert reason is not None and "OPENROUTER_API_KEY" in reason

    def test_openrouter_key_resolves_to_zero_shot_llm_with_nonzero_price(self) -> None:
        with warnings_none():
            judge, model, reason = choose_auto_judge(_settings(openrouter="or-key"))
        assert judge == "zero_shot_llm"
        assert model == _OPENROUTER_MODEL
        assert reason is None
        assert PRICES_PER_1M[model][0] > 0.0 and PRICES_PER_1M[model][1] > 0.0

    def test_openai_key_used_only_when_no_openrouter_key(self) -> None:
        with warnings_none():
            judge, model, reason = choose_auto_judge(_settings(openai="oai-key"))
        assert judge == "zero_shot_llm"
        assert model == _OPENAI_MODEL
        assert reason is None
        assert PRICES_PER_1M[model][0] > 0.0 and PRICES_PER_1M[model][1] > 0.0

    def test_openrouter_key_preferred_over_openai_key(self) -> None:
        judge, model, _ = choose_auto_judge(_settings(openrouter="or-key", openai="oai-key"))
        assert (judge, model) == ("zero_shot_llm", _OPENROUTER_MODEL)

    def test_refuses_paid_judge_with_unpinned_price(self) -> None:
        """E1: a candidate model with no pinned price is refused, not silently $0-capped."""
        unpriced = {k: v for k, v in PRICES_PER_1M.items() if k != _OPENROUTER_MODEL}
        with (
            patch.dict("langres.clients.openrouter.PRICES_PER_1M", unpriced, clear=True),
            pytest.warns(UserWarning, match="no pinned price"),
        ):
            judge, model, reason = choose_auto_judge(_settings(openrouter="or-key"))
        assert judge == "string"
        assert model is None
        assert reason is not None and _OPENROUTER_MODEL in reason


def warnings_none() -> "_NoWarnings":
    """Assert the wrapped block emits no warnings at all."""
    return _NoWarnings()


class _NoWarnings:
    def __enter__(self) -> "_NoWarnings":
        import warnings as _w

        self._catch = _w.catch_warnings(record=True)
        self._records = self._catch.__enter__()
        _w.simplefilter("always")
        return self

    def __exit__(self, *exc: object) -> None:
        assert not self._records, f"unexpected warnings: {[str(r.message) for r in self._records]}"
        self._catch.__exit__(*exc)


# ---------------------------------------------------------------------------
# build_judge
# ---------------------------------------------------------------------------


class TestBuildJudge:
    def test_string_returns_weighted_average_judge_with_schema_features(self) -> None:
        module = build_judge("string", PresetCompany)
        assert isinstance(module, WeightedAverageJudge)
        names = {spec.name for spec in module.feature_specs}
        assert names == {"name", "address"}

    def test_string_is_schema_agnostic_across_two_different_schemas(self) -> None:
        company_module = build_judge("string", PresetCompany)
        product_module = build_judge("string", PresetProduct)
        assert {s.name for s in company_module.feature_specs} == {"name", "address"}
        assert {s.name for s in product_module.feature_specs} == {"title", "brand"}

    def test_embedding_returns_embedding_score_judge(self) -> None:
        assert isinstance(build_judge("embedding", PresetCompany), EmbeddingScoreJudge)

    def test_zero_shot_llm_returns_dspy_judge_with_default_model_and_pinned_price(self) -> None:
        module = build_judge("zero_shot_llm", PresetCompany, entity_noun="company")
        assert isinstance(module, DSPyJudge)
        assert module.model == _OPENROUTER_MODEL
        assert module.entity_noun == "company"
        assert module.price_per_1k_tokens > 0.0

    def test_zero_shot_llm_respects_model_override(self) -> None:
        module = build_judge("zero_shot_llm", PresetCompany, model=_OPENAI_MODEL)
        assert isinstance(module, DSPyJudge)
        assert module.model == _OPENAI_MODEL
        assert module.price_per_1k_tokens > 0.0

    def test_module_instance_passed_through_verbatim(self) -> None:
        injected: DSPyJudge[PresetCompany] = DSPyJudge(lm=DummyLM([]), entity_noun="thing")
        assert build_judge(injected, PresetCompany) is injected

    def test_unknown_judge_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown judge"):
            build_judge("not-a-real-judge", PresetCompany)  # type: ignore[arg-type]

    def test_auto_is_not_resolved_here(self) -> None:
        with pytest.raises(ValueError, match="unknown judge"):
            build_judge("auto", PresetCompany)


# ---------------------------------------------------------------------------
# _SpendCappedModule / resolve_judge
# ---------------------------------------------------------------------------


class TestSpendCappedModule:
    def test_zero_cost_judgements_never_trip_the_cap(self) -> None:
        module = _SpendCappedModule(_FakeCostlyModule(5, 0.0), budget_usd=0.01)
        candidates = iter([_candidate(str(i)) for i in range(5)])
        judgements = list(module.forward(candidates))
        assert len(judgements) == 5

    def test_cap_breach_raises_budget_exceeded_with_partial_judgements(self) -> None:
        module = _SpendCappedModule(_FakeCostlyModule(5, 0.5), budget_usd=0.9)
        candidates = iter([_candidate(str(i)) for i in range(5)])
        with pytest.raises(BudgetExceeded) as excinfo:
            list(module.forward(candidates))
        partial = excinfo.value.partial_judgements
        # 0.5 -> ok, 1.0 -> breaches $0.9 cap: exactly 2 judgements were paid for.
        assert len(partial) == 2
        assert all(isinstance(j, PairwiseJudgement) for j in partial)

    def test_inspect_scores_delegates_to_wrapped_module(self) -> None:
        inner = build_judge("string", PresetCompany)
        module = _SpendCappedModule(inner, budget_usd=1.0)
        report = module.inspect_scores([])
        assert report is not None


def _candidate(suffix: str) -> ERCandidate[PresetCompany]:
    return ERCandidate(
        left=PresetCompany(id=f"l{suffix}", name="A"),
        right=PresetCompany(id=f"r{suffix}", name="B"),
        blocker_name="test",
    )


class TestResolveJudge:
    def test_string_judge_used_and_default_budget(self) -> None:
        resolved = resolve_judge("string", PresetCompany)
        assert resolved.judge_used == "string"
        assert resolved.model is None
        assert resolved.fallback_reason is None
        assert isinstance(resolved.module, _SpendCappedModule)
        assert resolved.module._budget_usd == DEFAULT_BUDGET_USD

    def test_custom_budget_usd_override(self) -> None:
        resolved = resolve_judge("string", PresetCompany, budget_usd=3.5)
        assert resolved.module._budget_usd == 3.5

    def test_zero_shot_llm_explicit_defaults_model_when_none(self) -> None:
        resolved = resolve_judge("zero_shot_llm", PresetCompany, model=None)
        assert resolved.judge_used == "zero_shot_llm"
        assert resolved.model == _OPENROUTER_MODEL
        assert resolved.module._module.model == _OPENROUTER_MODEL  # type: ignore[attr-defined]

    def test_injected_module_reports_judge_used_custom(self) -> None:
        injected: DSPyJudge[PresetCompany] = DSPyJudge(lm=DummyLM([]), entity_noun="thing")
        resolved = resolve_judge(injected, PresetCompany)
        assert resolved.judge_used == "custom"
        assert resolved.module._module is injected

    def test_auto_resolution_is_delegated_to_choose_auto_judge(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "fake-not-a-real-key"}, clear=True):
            resolved = resolve_judge("auto", PresetCompany)
        assert resolved.judge_used == "zero_shot_llm"
        assert resolved.model == _OPENROUTER_MODEL
        assert isinstance(resolved.module._module, DSPyJudge)

    def test_auto_resolution_falls_back_to_string_without_keys(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("pydantic_settings.sources.DotEnvSettingsSource.__call__", return_value={}),
            pytest.warns(UserWarning),
        ):
            resolved = resolve_judge("auto", PresetCompany)
        assert resolved.judge_used == "string"
        assert resolved.fallback_reason is not None


# ---------------------------------------------------------------------------
# notice_pre_scoring_cost / _estimate_n_pairs
# ---------------------------------------------------------------------------


class TestNoticeAndEstimate:
    def test_notice_message_format(self) -> None:
        with pytest.warns(UserWarning, match=r"scoring ~10 pairs with '.*', est\. cost \$"):
            notice_pre_scoring_cost(_OPENROUTER_MODEL, 10)

    def test_unpinned_model_warns_blind_cap_not_reassuring_zero(self) -> None:
        """M1 regression: an unpinned paid model must never print the
        reassuring (and false) "est. cost $0.0000" -- the spend cap tallies
        that same $0 and can never trip, so it's silently blind while
        OpenRouter still bills. The notice must say so honestly."""
        with pytest.warns(UserWarning, match=r"CANNOT enforce a limit") as record:
            notice_pre_scoring_cost("unknown/model-not-in-table", 10, budget_usd=2.5)
        message = str(record[0].message)
        assert "est. cost $0.0000" not in message
        assert "unknown/model-not-in-table" in message
        assert "$2.50" in message

    def test_unpinned_model_defaults_budget_in_message_when_omitted(self) -> None:
        with pytest.warns(UserWarning, match=rf"\${DEFAULT_BUDGET_USD:.2f}"):
            notice_pre_scoring_cost("unknown/model-not-in-table", 10)

    def test_estimate_all_pairs(self) -> None:
        assert _estimate_n_pairs(5, use_vector=False) == 10  # C(5,2)

    def test_estimate_vector(self) -> None:
        assert _estimate_n_pairs(5, use_vector=True) == 50  # 5 * k=10


# ---------------------------------------------------------------------------
# build_resolver
# ---------------------------------------------------------------------------


class TestBuildResolver:
    def test_string_judge_small_n_uses_all_pairs_blocker(self) -> None:
        resolved = build_resolver(
            PresetCompany,
            judge="string",
            model=None,
            entity_noun="entity",
            threshold=None,
            n_records=5,
        )
        assert resolved.judge_used == "string"
        assert resolved.score_type == "heuristic"
        assert isinstance(resolved.resolver.blocker, AllPairsBlocker)
        assert resolved.resolver.comparator is not None

    def test_string_judge_large_n_uses_vector_blocker(self) -> None:
        resolved = build_resolver(
            PresetCompany,
            judge="string",
            model=None,
            entity_noun="entity",
            threshold=None,
            n_records=_ALL_PAIRS_MAX_N + 1,
        )
        assert isinstance(resolved.resolver.blocker, VectorBlocker)

    def test_embedding_judge_always_uses_vector_blocker_even_for_small_n(self) -> None:
        resolved = build_resolver(
            PresetCompany,
            judge="embedding",
            model=None,
            entity_noun="entity",
            threshold=None,
            n_records=2,
        )
        assert isinstance(resolved.resolver.blocker, VectorBlocker)
        assert resolved.resolver.comparator is None
        assert resolved.score_type == "sim_cos"

    def test_threshold_defaults_per_judge_when_none(self) -> None:
        resolved = build_resolver(
            PresetCompany,
            judge="embedding",
            model=None,
            entity_noun="e",
            threshold=None,
            n_records=2,
        )
        assert resolved.resolver.clusterer.threshold == 0.5

    def test_explicit_threshold_overrides_default(self) -> None:
        resolved = build_resolver(
            PresetCompany, judge="string", model=None, entity_noun="e", threshold=0.9, n_records=2
        )
        assert resolved.resolver.clusterer.threshold == 0.9

    def test_zero_shot_llm_emits_pre_scoring_notice(self) -> None:
        with pytest.warns(UserWarning, match="scoring ~"):
            build_resolver(
                PresetCompany,
                judge="zero_shot_llm",
                model=_OPENROUTER_MODEL,
                entity_noun="e",
                threshold=None,
                n_records=4,
            )

    def test_zero_shot_llm_explicit_unpinned_model_warns_blind_cap(self) -> None:
        """M1 regression: an explicit judge="zero_shot_llm" with an unpinned
        model= must warn that the spend cap is blind (construction only --
        DummyLM-equivalent zero-spend, DSPyJudge is never .forward()ed here),
        not silently proceed under a reassuring but false $0.0000 estimate."""
        with pytest.warns(UserWarning, match="CANNOT enforce a limit") as record:
            build_resolver(
                PresetCompany,
                judge="zero_shot_llm",
                model="unknown/model-not-in-table",
                entity_noun="e",
                threshold=None,
                n_records=4,
                budget_usd=3.0,
            )
        message = str(record[0].message)
        assert "est. cost $0.0000" not in message
        assert "$3.00" in message

    def test_string_judge_emits_no_notice(self) -> None:
        with warnings_none():
            build_resolver(
                PresetCompany,
                judge="string",
                model=None,
                entity_noun="e",
                threshold=None,
                n_records=4,
            )

    def test_custom_module_uses_n_based_blocker_rule_and_no_notice(self) -> None:
        injected: DSPyJudge[PresetCompany] = DSPyJudge(lm=DummyLM([]), entity_noun="thing")
        with warnings_none():
            resolved = build_resolver(
                PresetCompany,
                judge=injected,
                model=None,
                entity_noun="e",
                threshold=None,
                n_records=4,
            )
        assert resolved.judge_used == "custom"
        assert isinstance(resolved.resolver.blocker, AllPairsBlocker)
        assert resolved.score_type == "unknown"


# ---------------------------------------------------------------------------
# _text_field_extractor (schema-agnostic)
# ---------------------------------------------------------------------------


class TestTextFieldExtractor:
    def test_concatenates_non_empty_string_fields(self) -> None:
        extractor = _text_field_extractor(PresetCompany)
        entity = PresetCompany(id="1", name="Acme", address=None)
        assert extractor(entity) == "Acme"

    def test_schema_agnostic_second_schema(self) -> None:
        extractor = _text_field_extractor(PresetProduct)
        entity = PresetProduct(id="1", title="Widget", brand="Acme")
        text = extractor(entity)
        assert "Widget" in text and "Acme" in text


# ---------------------------------------------------------------------------
# Vector-blocker / embedding construction (real MiniLM load -- slow, local, $0)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestVectorBlockerAndEmbedding:
    def test_build_vector_blocker_shape(self) -> None:
        blocker = _build_vector_blocker(PresetCompany)
        assert isinstance(blocker, VectorBlocker)
        assert blocker.k_neighbors == 10

    def test_build_embedding_candidate_identical_texts_score_near_one(self) -> None:
        record = {"id": "a", "name": "Acme Corporation"}
        candidate = build_embedding_candidate(PresetCompany, record, dict(record, id="b"))
        assert candidate.similarity_score is not None
        assert candidate.similarity_score > 0.99

    def test_build_embedding_candidate_different_texts_score_lower(self) -> None:
        left = {"id": "a", "name": "Acme Corporation"}
        right = {"id": "b", "name": "Totally Unrelated Restaurant Chain"}
        candidate = build_embedding_candidate(PresetCompany, left, right)
        assert candidate.similarity_score is not None
        assert candidate.similarity_score < 0.99


class TestBuildEmbeddingCandidateNoCandidateGuard:
    """L2 regression: a bare StopIteration must never leak from next()."""

    def test_empty_blocker_stream_raises_clear_runtime_error(self) -> None:
        fake_blocker = _FakeEmptyStreamBlocker()
        with patch("langres.core.presets._build_vector_blocker", return_value=fake_blocker):
            with pytest.raises(RuntimeError, match="produced no candidate"):
                build_embedding_candidate(
                    PresetCompany, {"id": "a", "name": "X"}, {"id": "b", "name": "Y"}
                )
