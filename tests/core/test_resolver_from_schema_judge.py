"""Tests for Resolver.from_schema's matcher= parameter (W0.1 bullet D).

Zero-spend: the "zero_shot_llm" branch is only exercised up to construction
(model/price), never a real `.forward()` call.
"""

import pytest
from dspy.utils.dummies import DummyLM
from pydantic import BaseModel

from langres.core.matchers.embedding_score import EmbeddingScoreMatcher
from langres.core.matchers.weighted_average import WeightedAverageMatcher
from langres.core.matchers.dspy_judge import DSPyMatcher
from langres.core.resolver import Resolver, _build_module_for_judge


class ResolverJudgeCo(BaseModel):
    id: str
    name: str | None = None


class TestFromSchemaJudgeDefault:
    def test_default_judge_is_string_byte_identical_to_pre_existing_behavior(self) -> None:
        """No matcher= kwarg -> the exact WeightedAverageMatcher construction that
        existed before this parameter was added (regression guard)."""
        resolver = Resolver.from_schema(ResolverJudgeCo)
        assert isinstance(resolver.module, WeightedAverageMatcher)
        assert resolver.module.feature_specs == resolver.comparator.feature_specs  # type: ignore[union-attr]

    def test_explicit_judge_string_matches_default(self) -> None:
        default_resolver = Resolver.from_schema(ResolverJudgeCo)
        explicit_resolver = Resolver.from_schema(ResolverJudgeCo, matcher="string")
        assert type(default_resolver.module) is type(explicit_resolver.module)
        assert default_resolver.module.feature_specs == explicit_resolver.module.feature_specs  # type: ignore[attr-defined]


class TestFromSchemaJudgeOptions:
    def test_embedding_judge(self) -> None:
        resolver = Resolver.from_schema(ResolverJudgeCo, matcher="embedding")
        assert isinstance(resolver.module, EmbeddingScoreMatcher)

    def test_embedding_judge_wires_vector_blocker_not_all_pairs(self) -> None:
        """BUG regression: the default AllPairsBlocker's candidates never
        carry similarity_score, which EmbeddingScoreMatcher requires -- so
        matcher="embedding" must wire a VectorBlocker instead, exactly like
        core.presets.build_resolver does for the verb layer."""
        from langres.core.blockers.vector import VectorBlocker

        resolver = Resolver.from_schema(ResolverJudgeCo, matcher="embedding")
        assert isinstance(resolver.blocker, VectorBlocker)

    @pytest.mark.slow
    def test_embedding_judge_resolves_without_similarity_score_error(self) -> None:
        """End-to-end: resolve() must not raise the ValueError an
        AllPairsBlocker + EmbeddingScoreMatcher pairing would produce (real
        local MiniLM embed -- $0, no API key, just slow)."""
        records = [
            {"id": "1", "name": "Acme Corporation"},
            {"id": "2", "name": "Acme Corp"},
            {"id": "3", "name": "Totally Unrelated Restaurant"},
        ]
        resolver = Resolver.from_schema(ResolverJudgeCo, matcher="embedding", threshold=0.5)
        clusters = resolver.resolve(records)
        assert isinstance(clusters, list)

    def test_zero_shot_llm_judge_default_model_and_pinned_price(self) -> None:
        resolver = Resolver.from_schema(
            ResolverJudgeCo, matcher="zero_shot_llm", entity_noun="company"
        )
        assert isinstance(resolver.module, DSPyMatcher)
        assert resolver.module.model == "openrouter/openai/gpt-4o-mini"
        assert resolver.module.entity_noun == "company"
        assert resolver.module.price_per_1k_tokens > 0.0

    def test_zero_shot_llm_judge_model_override(self) -> None:
        resolver = Resolver.from_schema(
            ResolverJudgeCo, matcher="zero_shot_llm", model="openai/gpt-5-mini"
        )
        assert isinstance(resolver.module, DSPyMatcher)
        assert resolver.module.model == "openai/gpt-5-mini"

    def test_zero_shot_llm_unpinned_model_warns_blind_cap(self) -> None:
        """M1 regression: Resolver.from_schema builds an UNCAPPED pipeline
        (see the matcher= docstring caution) -- an unpinned model must not
        silently self-report $0/pair without any warning, since nothing here
        would ever stop a runaway bill. Construction only (zero-spend)."""
        with pytest.warns(UserWarning, match="UNCAPPED pipeline"):
            resolver = Resolver.from_schema(
                ResolverJudgeCo, matcher="zero_shot_llm", model="unknown/model-not-in-table"
            )
        assert isinstance(resolver.module, DSPyMatcher)
        assert resolver.module.price_per_1k_tokens == 0.0

    def test_injected_module_instance_used_verbatim(self) -> None:
        injected: DSPyMatcher[ResolverJudgeCo] = DSPyMatcher(lm=DummyLM([]), entity_noun="thing")
        resolver = Resolver.from_schema(ResolverJudgeCo, matcher=injected)
        assert resolver.module is injected

    def test_auto_is_rejected_with_guidance_to_verbs(self) -> None:
        with pytest.raises(ValueError, match="verbs-layer feature"):
            Resolver.from_schema(ResolverJudgeCo, matcher="auto")  # type: ignore[arg-type]

    def test_unknown_judge_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported judge"):
            Resolver.from_schema(ResolverJudgeCo, matcher="not-a-real-judge")  # type: ignore[arg-type]


class TestBuildModuleForJudgeDirect:
    def test_returns_module_instance_verbatim(self) -> None:
        from langres.core.comparator import Comparator

        comparator = Comparator.from_schema(ResolverJudgeCo)
        injected: DSPyMatcher[ResolverJudgeCo] = DSPyMatcher(lm=DummyLM([]))
        assert (
            _build_module_for_judge(
                injected, ResolverJudgeCo, comparator, model=None, entity_noun="entity"
            )
            is injected
        )
