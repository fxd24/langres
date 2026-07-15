"""Tests for the one MethodSpec registry (the v0.3 model-identity slice).

Covers the registry contract itself (lookup, did-you-mean, the reserved ``/``
id grammar, collision), the identity metadata each built-in spec carries
(``default_model`` / ``accepts_model`` / ``default_threshold`` /
``score_type``), and that the three former dispatch sites -- the verbs'
``presets.build_judge``, ``Resolver.from_schema``, and the benchmark
harness's ``methods._make_module_builder`` -- now build the SAME class for
the same name (docs/research/20260713_model_identity_and_hub.md, closing
issue #55's three-site wiring debt).
"""

from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from langres.clients.openrouter import DEFAULT_OPENROUTER_MODEL
from langres.core.comparator import Comparator
from langres.core.method_registry import (
    DEFAULT_EMBEDDING_MODEL,
    MethodSpec,
    UnknownMethodError,
    get_method,
    list_methods,
    register_method,
)
from langres.core.matcher import Matcher
from langres.core.presets import DEFAULT_AUTO_MODEL


class RegistryCompany(BaseModel):
    id: str
    name: str | None = None
    address: str | None = None


def _build(name: str, *, client: Any = None, **params: Any) -> Matcher[Any]:
    """Build ``name``'s module with the standard call convention."""
    return get_method(name).build(
        RegistryCompany,
        model=None,
        entity_noun="entity",
        client=client,
        comparator=None,
        **params,
    )


class TestIdGrammar:
    """Bare names are built-ins; '/' is reserved for author/method namespacing."""

    def test_slash_in_get_method_is_reserved(self) -> None:
        with pytest.raises(UnknownMethodError, match="reserved for future"):
            get_method("jdoe/ditto")

    def test_slash_message_points_model_ids_at_the_model_kwarg(self) -> None:
        """A user pasting a model id as the judge name gets steered to model=."""
        with pytest.raises(UnknownMethodError, match="model="):
            get_method("openrouter/openai/gpt-4o-mini")

    def test_slash_in_register_method_is_rejected(self) -> None:
        spec = MethodSpec(name="jdoe/ditto", build=lambda *a, **kw: Mock(), score_type="prob_llm")
        with pytest.raises(UnknownMethodError, match="reserved"):
            register_method(spec)

    def test_unknown_name_lists_available_with_did_you_mean(self) -> None:
        with pytest.raises(UnknownMethodError, match="Did you mean: string"):
            get_method("strng")

    def test_unknown_method_error_is_a_value_error(self) -> None:
        assert issubclass(UnknownMethodError, ValueError)

    def test_collision_raises_loudly(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            register_method(
                MethodSpec(name="string", build=lambda *a, **kw: Mock(), score_type="heuristic")
            )


class TestBuiltinSpecs:
    def test_every_builtin_name_is_registered(self) -> None:
        assert set(list_methods()) >= {
            "string",
            "embedding",
            "zero_shot_llm",
            "prompt_llm",
            "rapidfuzz",
            "weighted_average",
            "embedding_cosine",
            "llm_judge",
            "dspy_judge",
            "select_judge",
            "cascade",
            "fellegi_sunter",
            "random_forest",
        }

    @pytest.mark.parametrize(
        ("name", "class_name"),
        [
            ("string", "WeightedAverageMatcher"),
            ("weighted_average", "WeightedAverageMatcher"),
            ("embedding", "EmbeddingScoreMatcher"),
            ("embedding_cosine", "EmbeddingScoreMatcher"),
            ("zero_shot_llm", "DSPyMatcher"),
            ("dspy_judge", "DSPyMatcher"),
            ("prompt_llm", "LLMMatcher"),
            ("llm_judge", "LLMMatcher"),
            ("select_judge", "SelectMatcher"),
            ("cascade", "CascadeChainMatcher"),
            ("fellegi_sunter", "FellegiSunterMatcher"),
            ("random_forest", "RandomForestMatcher"),
        ],
    )
    def test_each_name_builds_exactly_one_class(self, name: str, class_name: str) -> None:
        """One name, one meaning -- the pre-registry 'llm_judge builds LLMMatcher
        but zero_shot_llm builds DSPyMatcher' split-brain cannot recur silently."""
        module = _build(name, client=Mock() if name == "cascade" else None)
        assert type(module).__name__ == class_name

    def test_llm_family_identity_metadata(self) -> None:
        for name in ("zero_shot_llm", "prompt_llm", "llm_judge", "dspy_judge"):
            spec = get_method(name)
            assert spec.default_model == DEFAULT_OPENROUTER_MODEL, name
            assert spec.accepts_model is True, name
            assert spec.default_threshold == 0.7, name
            assert spec.requires_extra == "llm", name

    def test_random_forest_declares_the_trained_extra(self) -> None:
        """The supervised forest needs scikit-learn, so its spec names the extra."""
        spec = get_method("random_forest")
        assert spec.requires_extra == "trained"
        assert spec.needs_comparator is True
        assert spec.score_type == "prob_rf"

    def test_trained_extra_error_is_actionable(self) -> None:
        """The [trained]-absent path points at ``langres[trained]`` (mirrors the
        LLM-family ``_llm_extra_error``), not a bare ModuleNotFoundError."""
        from langres.core.method_registry import _trained_extra_error

        err = _trained_extra_error("random_forest", "scikit-learn")
        assert isinstance(err, ImportError)
        message = str(err)
        assert "[trained] extra" in message
        assert "scikit-learn is not installed" in message
        assert "langres[trained]" in message

    def test_embedding_reports_the_pinned_embedder_as_its_model(self) -> None:
        spec = get_method("embedding")
        assert spec.default_model == DEFAULT_EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        # model= is ignored by the builder, so the spec must not honor it either.
        assert spec.accepts_model is False

    def test_string_has_no_model_identity(self) -> None:
        spec = get_method("string")
        assert spec.default_model is None
        assert spec.accepts_model is False
        assert spec.score_type == "heuristic"
        assert spec.default_threshold == 0.5

    def test_registry_and_auto_policy_share_one_default_model(self) -> None:
        """The pinned auto default and the registry's LLM specs cannot drift:
        all alias clients.openrouter.DEFAULT_OPENROUTER_MODEL."""
        assert DEFAULT_AUTO_MODEL == DEFAULT_OPENROUTER_MODEL
        assert get_method("zero_shot_llm").default_model == DEFAULT_AUTO_MODEL
        assert get_method("prompt_llm").default_model == DEFAULT_AUTO_MODEL

    def test_custom_comparator_weights_flow_into_string_builder(self) -> None:
        """from_schema's weights=/exclude= customization must reach the judge."""
        weighted = Comparator.from_schema(RegistryCompany, weights={"name": 0.9, "address": 0.1})
        module = get_method("string").build(
            RegistryCompany, model=None, entity_noun="entity", client=None, comparator=weighted
        )
        specs_by_name = {spec.name: spec for spec in module.feature_specs}  # type: ignore[attr-defined]
        assert specs_by_name["name"].weight == pytest.approx(0.9)
        assert specs_by_name["address"].weight == pytest.approx(0.1)

    def test_unknown_judge_param_fails_loudly(self) -> None:
        with pytest.raises(TypeError):
            _build("prompt_llm", not_a_real_knob=1)


class TestThreeSitesResolveIdentically:
    """The verbs, from_schema, and the benchmark harness share the registry."""

    def test_verbs_and_from_schema_build_the_same_string_judge(self) -> None:
        from langres.core.presets import build_judge
        from langres.core.resolver import Resolver

        via_presets = build_judge("string", RegistryCompany)
        via_from_schema = Resolver.from_schema(RegistryCompany, matcher="string").module
        assert type(via_presets) is type(via_from_schema)

    def test_benchmark_path_builds_llm_judge_via_the_registry(self) -> None:
        from langres.methods import _make_module_builder

        build, comparator = _make_module_builder(
            "llm_judge",
            RegistryCompany,
            llm_client=Mock(),
            llm_model=DEFAULT_OPENROUTER_MODEL,
            cascade_low=0.3,
            cascade_high=0.9,
        )
        assert type(build()).__name__ == "LLMMatcher"
        assert comparator is None

    def test_benchmark_path_unknown_method_is_a_value_error_with_suggestions(self) -> None:
        from langres.methods import _make_module_builder

        with pytest.raises(ValueError, match="unknown method"):
            _make_module_builder(
                "no_such_method",
                RegistryCompany,
                llm_client=None,
                llm_model="m",
                cascade_low=0.3,
                cascade_high=0.9,
            )

    def test_from_schema_prompt_llm_saves_and_loads_the_named_parser(self, tmp_path: Any) -> None:
        """#103's acceptance shape: a bring-your-own-prompt judge is expressible
        by name, serializable, and round-trips its registered parser."""
        from langres.core.matchers.llm_judge import LLMMatcher, parse_binary_yes_no
        from langres.core.registry import _SCHEMA_REGISTRY, register_schema
        from langres.core.resolver import Resolver

        if "RegistryCompany" not in _SCHEMA_REGISTRY:
            register_schema("RegistryCompany")(RegistryCompany)
        resolver = Resolver.from_schema(
            RegistryCompany,
            matcher="prompt_llm",
            prompt_template="Same {entity}? A: {left} B: {right}".replace("{entity}", "company"),
            system_prompt="You are a matcher.",
            response_parser="binary_yes_no",
        )
        module = resolver.module
        assert isinstance(module, LLMMatcher)
        assert module.config["response_parser"] == "binary_yes_no"

        resolver.save(tmp_path / "artifact")
        loaded = Resolver.load(tmp_path / "artifact")
        assert isinstance(loaded.module, LLMMatcher)
        assert loaded.module._parse is parse_binary_yes_no
        assert loaded.module.prompt_template == module.prompt_template
        assert loaded.module.system_prompt == "You are a matcher."

    def test_from_schema_rejects_prompt_seam_kwargs_for_other_judges(self) -> None:
        from langres.core.resolver import Resolver

        with pytest.raises(ValueError, match="prompt_llm"):
            Resolver.from_schema(
                RegistryCompany, matcher="string", prompt_template="x{left}{right}"
            )
