"""Tests for Resolver.from_schema's judge= parameter (W0.1 bullet D).

Zero-spend: the "zero_shot_llm" branch is only exercised up to construction
(model/price), never a real `.forward()` call.
"""

import pytest
from dspy.utils.dummies import DummyLM
from pydantic import BaseModel

from langres.core.judges.embedding_score import EmbeddingScoreJudge
from langres.core.judges.weighted_average import WeightedAverageJudge
from langres.core.modules.dspy_judge import DSPyJudge
from langres.core.resolver import Resolver, _build_module_for_judge


class ResolverJudgeCo(BaseModel):
    id: str
    name: str | None = None


class TestFromSchemaJudgeDefault:
    def test_default_judge_is_string_byte_identical_to_pre_existing_behavior(self) -> None:
        """No judge= kwarg -> the exact WeightedAverageJudge construction that
        existed before this parameter was added (regression guard)."""
        resolver = Resolver.from_schema(ResolverJudgeCo)
        assert isinstance(resolver.module, WeightedAverageJudge)
        assert resolver.module.feature_specs == resolver.comparator.feature_specs  # type: ignore[union-attr]

    def test_explicit_judge_string_matches_default(self) -> None:
        default_resolver = Resolver.from_schema(ResolverJudgeCo)
        explicit_resolver = Resolver.from_schema(ResolverJudgeCo, judge="string")
        assert type(default_resolver.module) is type(explicit_resolver.module)
        assert default_resolver.module.feature_specs == explicit_resolver.module.feature_specs  # type: ignore[attr-defined]


class TestFromSchemaJudgeOptions:
    def test_embedding_judge(self) -> None:
        resolver = Resolver.from_schema(ResolverJudgeCo, judge="embedding")
        assert isinstance(resolver.module, EmbeddingScoreJudge)

    def test_embedding_judge_wires_vector_blocker_not_all_pairs(self) -> None:
        """BUG regression: the default AllPairsBlocker's candidates never
        carry similarity_score, which EmbeddingScoreJudge requires -- so
        judge="embedding" must wire a VectorBlocker instead, exactly like
        core.presets.build_resolver does for the verb layer."""
        from langres.core.blockers.vector import VectorBlocker

        resolver = Resolver.from_schema(ResolverJudgeCo, judge="embedding")
        assert isinstance(resolver.blocker, VectorBlocker)

    @pytest.mark.slow
    def test_embedding_judge_resolves_without_similarity_score_error(self) -> None:
        """End-to-end: resolve() must not raise the ValueError an
        AllPairsBlocker + EmbeddingScoreJudge pairing would produce (real
        local MiniLM embed -- $0, no API key, just slow)."""
        records = [
            {"id": "1", "name": "Acme Corporation"},
            {"id": "2", "name": "Acme Corp"},
            {"id": "3", "name": "Totally Unrelated Restaurant"},
        ]
        resolver = Resolver.from_schema(ResolverJudgeCo, judge="embedding", threshold=0.5)
        clusters = resolver.resolve(records)
        assert isinstance(clusters, list)

    def test_zero_shot_llm_judge_default_model_and_pinned_price(self) -> None:
        resolver = Resolver.from_schema(
            ResolverJudgeCo, judge="zero_shot_llm", entity_noun="company"
        )
        assert isinstance(resolver.module, DSPyJudge)
        assert resolver.module.model == "openrouter/openai/gpt-4o-mini"
        assert resolver.module.entity_noun == "company"
        assert resolver.module.price_per_1k_tokens > 0.0

    def test_zero_shot_llm_judge_model_override(self) -> None:
        resolver = Resolver.from_schema(
            ResolverJudgeCo, judge="zero_shot_llm", model="openai/gpt-5-mini"
        )
        assert isinstance(resolver.module, DSPyJudge)
        assert resolver.module.model == "openai/gpt-5-mini"

    def test_injected_module_instance_used_verbatim(self) -> None:
        injected: DSPyJudge[ResolverJudgeCo] = DSPyJudge(lm=DummyLM([]), entity_noun="thing")
        resolver = Resolver.from_schema(ResolverJudgeCo, judge=injected)
        assert resolver.module is injected

    def test_auto_is_rejected_with_guidance_to_verbs(self) -> None:
        with pytest.raises(ValueError, match="verbs-layer feature"):
            Resolver.from_schema(ResolverJudgeCo, judge="auto")  # type: ignore[arg-type]

    def test_unknown_judge_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported judge"):
            Resolver.from_schema(ResolverJudgeCo, judge="not-a-real-judge")  # type: ignore[arg-type]


class TestBuildModuleForJudgeDirect:
    def test_returns_module_instance_verbatim(self) -> None:
        from langres.core.comparator import Comparator

        comparator = Comparator.from_schema(ResolverJudgeCo)
        injected: DSPyJudge[ResolverJudgeCo] = DSPyJudge(lm=DummyLM([]))
        assert (
            _build_module_for_judge(injected, comparator, model=None, entity_noun="entity")
            is injected
        )
