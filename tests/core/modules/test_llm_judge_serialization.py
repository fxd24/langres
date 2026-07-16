"""Serialization, lazy-client, and neutral-prompt tests for LLMMatcher.

These cover the M0.5 W-C contract: LLMMatcher is a first-class, serializable
Resolver Matcher — it registers under ``llm_judge``, exposes a pure ``config``
that never carries the client/secrets, rebuilds from env via the lazy-client
path, and a Resolver with an LLMMatcher in the ``module`` slot can ``save`` /
``load`` with no network.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from langres.core.matchers.llm_judge import (
    DEFAULT_PROMPT,
    LLMMatcher,
    LLMMatcher,
    render_default_prompt,
)
from langres.core.registry import get_component


def test_llm_judge_is_registered_with_type_name() -> None:
    """LLMMatcher is discoverable in the component registry under ``llm_judge``."""
    assert get_component("llm_judge") is LLMMatcher
    assert LLMMatcher.type_name == "llm_judge"
    # Backward-compat alias keeps old imports working.
    assert LLMMatcher is LLMMatcher


def test_config_excludes_client_and_secrets() -> None:
    """``config`` carries only pure, serializable data — never the client."""
    judge = LLMMatcher(
        model="openrouter/openai/gpt-4o-mini",
        client=object(),  # non-serializable stub client
        temperature=0.3,
        entity_noun="company",
    )

    config = judge.config

    assert set(config) == {
        "model",
        "api_base",
        "temperature",
        "prompt_template",
        "entity_noun",
        "provider",
        "system_prompt",
        "on_parse_error",
        "confidence",
        "response_parser",
        "record_serializer",
    }
    assert config["model"] == "openrouter/openai/gpt-4o-mini"
    assert config["api_base"] is None  # no served endpoint by default
    assert config["temperature"] == 0.3
    assert config["entity_noun"] == "company"
    assert config["provider"] is None  # no provider pin by default
    # The client (and any secret it holds) is never serialized.
    assert "client" not in config
    assert object() not in config.values()
    # Whole config must be JSON-serializable.
    json.dumps(config)


def test_from_config_round_trips_via_lazy_client_path() -> None:
    """``from_config`` rebuilds an equivalent judge with the client left lazy."""
    original = LLMMatcher(
        model="gpt-5-mini",
        client=object(),
        temperature=0.7,
        entity_noun="product",
    )

    rebuilt = LLMMatcher.from_config(original.config)

    assert rebuilt.config == original.config
    # Client is NOT persisted — it is reconstructed from env on first use.
    assert rebuilt.client is None
    assert rebuilt.model == "gpt-5-mini"
    assert rebuilt.temperature == 0.7
    assert rebuilt.entity_noun == "product"
    assert rebuilt.prompt_template == original.prompt_template


def test_confidence_round_trips_through_config() -> None:
    """A ``confidence="logprob"`` judge survives ``from_config`` (PR #105 review).

    Without ``confidence`` in ``config``, ``Resolver.save``/``load`` would silently
    revert a logprob judge to ``confidence="none"`` -- dropping the credence probe.
    """
    original = LLMMatcher(
        model="openrouter/openai/gpt-4o-mini",
        client=object(),
        confidence="logprob",
    )
    assert original.config["confidence"] == "logprob"

    rebuilt = LLMMatcher.from_config(original.config)
    assert rebuilt.confidence == "logprob"
    assert rebuilt.config == original.config

    # An older artifact with no ``confidence`` key falls back to "none".
    legacy = dict(original.config)
    del legacy["confidence"]
    assert LLMMatcher.from_config(legacy).confidence == "none"


def test_provider_pin_round_trips_through_config() -> None:
    """A provider pin survives ``config`` -> ``from_config`` (reproducible runs)."""
    pin = {"order": ["DeepInfra"], "allow_fallbacks": False}
    original = LLMMatcher(
        model="openrouter/z-ai/glm-5.2",
        client=object(),
        provider=pin,
    )

    config = original.config
    assert config["provider"] == pin
    json.dumps(config)  # a provider pin stays JSON-serializable

    rebuilt = LLMMatcher.from_config(config)
    assert rebuilt.provider == pin
    assert rebuilt.config == original.config


def test_from_env_builds_client_from_environment(mocker) -> None:
    """``from_env`` is the happy path: client comes from ``create_llm_client``."""
    sentinel = object()
    create = mocker.patch("langres.clients.create_llm_client", return_value=sentinel)

    judge = LLMMatcher.from_env(model="gpt-5-mini", temperature=0.0, entity_noun="person")

    assert judge.client is sentinel
    assert judge.model == "gpt-5-mini"
    assert judge.temperature == 0.0
    assert judge.entity_noun == "person"
    create.assert_called_once()


def test_client_is_lazily_built_from_env_when_omitted(mocker) -> None:
    """An omitted client is built once from env on first use, then cached."""
    built = object()
    create = mocker.patch("langres.clients.create_llm_client", return_value=built)

    judge: LLMMatcher = LLMMatcher(model="gpt-5-mini")  # no client
    assert judge.client is None  # not built at construction

    first = judge._get_client()
    second = judge._get_client()

    assert first is built
    assert second is built
    assert judge.client is built  # cached on the instance
    create.assert_called_once()  # built exactly once


def test_default_prompt_is_domain_neutral() -> None:
    """The default prompt mentions no specific domain (no 'company')."""
    rendered = render_default_prompt()
    assert "company" not in rendered.lower()
    assert "entity" in rendered.lower()
    # The default judge (no overrides) uses the neutral prompt.
    assert "company" not in LLMMatcher(client=object()).prompt_template.lower()
    # The centralized template placeholder is the single source of truth.
    assert "{entity_noun}" in DEFAULT_PROMPT


def test_entity_noun_is_woven_into_the_prompt() -> None:
    """``entity_noun`` parametrizes the default prompt for a specific domain."""
    judge = LLMMatcher(client=object(), entity_noun="company")
    assert "company" in judge.prompt_template.lower()
    # ``{left}`` / ``{right}`` survive for judgement-time formatting.
    assert "{left}" in judge.prompt_template
    assert "{right}" in judge.prompt_template


def test_custom_prompt_template_is_the_escape_hatch() -> None:
    """An explicit ``prompt_template`` wins and ignores ``entity_noun``."""
    custom = "Same? A={left} B={right}"
    judge = LLMMatcher(client=object(), prompt_template=custom, entity_noun="company")
    assert judge.prompt_template == custom


def test_resolver_with_llm_judge_module_saves_and_loads(tmp_path: Path, mocker) -> None:
    """A Resolver with an LLMMatcher in the module slot round-trips with no network.

    Save serializes only the pure config; load rebuilds the judge with a lazy
    (env-reconstructed) client. We patch ``create_llm_client`` to raise so the
    test fails loudly if load ever tries to build a client (it must not).
    """
    from langres.core import Clusterer, Resolver
    from langres.core.blockers import AllPairsBlocker
    from langres.core.models import CompanySchema

    # If load builds a client, this blows up — proving load stays offline/lazy.
    mocker.patch(
        "langres.clients.create_llm_client",
        side_effect=AssertionError("client must not be built during save/load"),
    )

    judge: LLMMatcher[CompanySchema] = LLMMatcher(
        model="openrouter/openai/gpt-4o-mini",
        client=object(),
        entity_noun="company",
    )
    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        matcher=judge,
        clusterer=Clusterer(threshold=0.7),
    )

    resolver.save(tmp_path)

    manifest = json.loads((tmp_path / "resolver.json").read_text())
    module_spec = next(c for c in manifest["components"] if c["slot"] == "module")
    assert module_spec["type_name"] == "llm_judge"
    assert module_spec["config"]["model"] == "openrouter/openai/gpt-4o-mini"
    assert module_spec["config"]["entity_noun"] == "company"
    assert "client" not in module_spec["config"]

    reloaded = Resolver.load(tmp_path)
    assert isinstance(reloaded.module, LLMMatcher)
    assert reloaded.module.client is None  # lazy — not built at load
    assert reloaded.module.config == judge.config


@pytest.mark.slow
def test_resolver_load_registers_llm_judge_in_a_fresh_process(tmp_path: Path) -> None:
    """A clean process can ``Resolver.load`` an LLMMatcher artifact via ``langres.core`` alone.

    Regression for the load-path registration bug: ``@register("llm_judge")`` only
    fires when ``langres.core.matchers.llm_judge`` is imported. ``langres.core`` must
    import it so a fresh process that *only* does ``from langres.core import
    Resolver`` finds ``llm_judge`` in the registry. Without the ``__init__`` import,
    ``Resolver.load`` raises ``UnknownComponentType`` here (this test fails); with
    it, load succeeds and stays offline (the client is rebuilt lazily, not at load).

    The subprocess deliberately does NOT import ``langres.core.matchers.llm_judge``
    — that is the whole point of the check.
    """
    from langres.core import Clusterer, Resolver
    from langres.core.blockers import AllPairsBlocker
    from langres.core.models import CompanySchema

    judge: LLMMatcher[CompanySchema] = LLMMatcher(
        model="openrouter/openai/gpt-4o-mini",
        client=object(),
        entity_noun="company",
    )
    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        matcher=judge,
        clusterer=Clusterer(threshold=0.7),
    )
    resolver.save(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"from langres.core import Resolver; Resolver.load(r'{tmp_path}')",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        "fresh-process Resolver.load failed (LLMMatcher not registered on the "
        f"import-langres.core path).\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "UnknownComponentType" not in result.stderr


# ---------------------------------------------------------------------------
# Named parsers/serializers (v0.3): the config round-trip gap, closed
# ---------------------------------------------------------------------------


def test_named_response_parser_round_trips_through_config() -> None:
    """A registered parser NAME serializes and reloads as the same callable --
    pre-v0.3, a paper-replication judge silently reverted to the default
    Score:-line parser on Resolver.load."""
    from langres.core.matchers.llm_judge import parse_binary_yes_no

    original = LLMMatcher(
        model="openrouter/openai/gpt-4o-mini",
        client=object(),
        response_parser="binary_yes_no",
    )
    assert original._parse is parse_binary_yes_no
    assert original.config["response_parser"] == "binary_yes_no"

    rebuilt = LLMMatcher.from_config(original.config)
    assert rebuilt._parse is parse_binary_yes_no
    assert rebuilt.config == original.config


def test_registered_callable_serializes_as_its_name() -> None:
    """Passing the registered callable itself (not the name) still serializes."""
    from langres.core.matchers.llm_judge import parse_binary_yes_no

    judge = LLMMatcher(client=object(), response_parser=parse_binary_yes_no)
    assert judge.config["response_parser"] == "binary_yes_no"


def test_custom_callable_parser_serializes_as_none_and_reverts_on_load() -> None:
    """An unregistered callable cannot travel (no-pickle invariant): config
    carries None and from_config falls back to the default parser."""
    from langres.core.matchers.llm_judge import ParsedVerdict, parse_score_response

    def my_parser(content: str) -> ParsedVerdict:
        return ParsedVerdict(score=1.0)

    judge = LLMMatcher(client=object(), response_parser=my_parser)
    assert judge._parse is my_parser
    assert judge.config["response_parser"] is None

    rebuilt = LLMMatcher.from_config(judge.config)
    assert rebuilt._parse is parse_score_response


def test_unknown_parser_name_raises_listing_registered_names() -> None:
    with pytest.raises(ValueError, match="binary_yes_no"):
        LLMMatcher(client=object(), response_parser="nope")


def test_default_parser_and_serializer_names_in_config() -> None:
    judge = LLMMatcher(client=object())
    assert judge.config["response_parser"] == "score"
    assert judge.config["record_serializer"] == "json"


def test_pre_v03_config_without_parser_keys_falls_back_to_defaults() -> None:
    """Artifacts saved before the named-parser keys existed keep loading."""
    from langres.core.matchers.llm_judge import default_record_serializer, parse_score_response

    judge = LLMMatcher(client=object())
    legacy = {
        k: v for k, v in judge.config.items() if k not in ("response_parser", "record_serializer")
    }
    rebuilt = LLMMatcher.from_config(legacy)
    assert rebuilt._parse is parse_score_response
    assert rebuilt._serialize is default_record_serializer


def test_named_record_serializer_resolves_and_serializes() -> None:
    from langres.core.matchers.llm_judge import default_record_serializer

    judge = LLMMatcher(client=object(), record_serializer="json")
    assert judge._serialize is default_record_serializer
    assert judge.config["record_serializer"] == "json"


def test_colval_serializer_is_registered_and_round_trips() -> None:
    """The AnyMatch/Ditto COL/VAL serializer is a name-selectable sibling of json."""
    from langres.core.matchers.llm_judge import colval_serializer

    judge = LLMMatcher(client=object(), record_serializer="colval")
    assert judge._serialize is colval_serializer
    assert judge.config["record_serializer"] == "colval"
    assert LLMMatcher.from_config(judge.config)._serialize is colval_serializer


def test_colval_serializer_renders_col_val_and_blanks_none() -> None:
    """`COL <field> VAL <value>` over model_dump; a None field renders a blank VAL."""
    from langres.core.matchers.llm_judge import colval_serializer
    from langres.core.models import CompanySchema

    text = colval_serializer(CompanySchema(id="1", name="Acme Corp", address=None))
    assert text.startswith("COL id VAL 1 COL name VAL Acme Corp")
    # A None attribute is present-but-blank, not omitted.
    assert "COL address VAL " in text
