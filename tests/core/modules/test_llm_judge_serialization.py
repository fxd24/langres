"""Serialization, lazy-client, and neutral-prompt tests for LLMJudge.

These cover the M0.5 W-C contract: LLMJudge is a first-class, serializable
Resolver Module — it registers under ``llm_judge``, exposes a pure ``config``
that never carries the client/secrets, rebuilds from env via the lazy-client
path, and a Resolver with an LLMJudge in the ``module`` slot can ``save`` /
``load`` with no network.
"""

import json
from pathlib import Path

from langres.core.modules.llm_judge import (
    DEFAULT_PROMPT,
    LLMJudge,
    LLMJudgeModule,
    render_default_prompt,
)
from langres.core.registry import get_component


def test_llm_judge_is_registered_with_type_name() -> None:
    """LLMJudge is discoverable in the component registry under ``llm_judge``."""
    assert get_component("llm_judge") is LLMJudge
    assert LLMJudge.type_name == "llm_judge"
    # Backward-compat alias keeps old imports working.
    assert LLMJudgeModule is LLMJudge


def test_config_excludes_client_and_secrets() -> None:
    """``config`` carries only pure, serializable data — never the client."""
    judge = LLMJudge(
        model="openrouter/openai/gpt-4o-mini",
        client=object(),  # non-serializable stub client
        temperature=0.3,
        entity_noun="company",
    )

    config = judge.config

    assert set(config) == {"model", "temperature", "prompt_template", "entity_noun"}
    assert config["model"] == "openrouter/openai/gpt-4o-mini"
    assert config["temperature"] == 0.3
    assert config["entity_noun"] == "company"
    # The client (and any secret it holds) is never serialized.
    assert "client" not in config
    assert object() not in config.values()
    # Whole config must be JSON-serializable.
    json.dumps(config)


def test_from_config_round_trips_via_lazy_client_path() -> None:
    """``from_config`` rebuilds an equivalent judge with the client left lazy."""
    original = LLMJudge(
        model="gpt-5-mini",
        client=object(),
        temperature=0.7,
        entity_noun="product",
    )

    rebuilt = LLMJudge.from_config(original.config)

    assert rebuilt.config == original.config
    # Client is NOT persisted — it is reconstructed from env on first use.
    assert rebuilt.client is None
    assert rebuilt.model == "gpt-5-mini"
    assert rebuilt.temperature == 0.7
    assert rebuilt.entity_noun == "product"
    assert rebuilt.prompt_template == original.prompt_template


def test_from_env_builds_client_from_environment(mocker) -> None:
    """``from_env`` is the happy path: client comes from ``create_llm_client``."""
    sentinel = object()
    create = mocker.patch("langres.clients.create_llm_client", return_value=sentinel)

    judge = LLMJudge.from_env(model="gpt-5-mini", temperature=0.0, entity_noun="person")

    assert judge.client is sentinel
    assert judge.model == "gpt-5-mini"
    assert judge.temperature == 0.0
    assert judge.entity_noun == "person"
    create.assert_called_once()


def test_client_is_lazily_built_from_env_when_omitted(mocker) -> None:
    """An omitted client is built once from env on first use, then cached."""
    built = object()
    create = mocker.patch("langres.clients.create_llm_client", return_value=built)

    judge: LLMJudge = LLMJudge(model="gpt-5-mini")  # no client
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
    assert "company" not in LLMJudge(client=object()).prompt_template.lower()
    # The centralized template placeholder is the single source of truth.
    assert "{entity_noun}" in DEFAULT_PROMPT


def test_entity_noun_is_woven_into_the_prompt() -> None:
    """``entity_noun`` parametrizes the default prompt for a specific domain."""
    judge = LLMJudge(client=object(), entity_noun="company")
    assert "company" in judge.prompt_template.lower()
    # ``{left}`` / ``{right}`` survive for judgement-time formatting.
    assert "{left}" in judge.prompt_template
    assert "{right}" in judge.prompt_template


def test_custom_prompt_template_is_the_escape_hatch() -> None:
    """An explicit ``prompt_template`` wins and ignores ``entity_noun``."""
    custom = "Same? A={left} B={right}"
    judge = LLMJudge(client=object(), prompt_template=custom, entity_noun="company")
    assert judge.prompt_template == custom


def test_resolver_with_llm_judge_module_saves_and_loads(tmp_path: Path, mocker) -> None:
    """A Resolver with an LLMJudge in the module slot round-trips with no network.

    Save serializes only the pure config; load rebuilds the judge with a lazy
    (env-reconstructed) client. We patch ``create_llm_client`` to raise so the
    test fails loudly if load ever tries to build a client (it must not).
    """
    from langres.core import AllPairsBlocker, Clusterer, Resolver
    from langres.core.models import CompanySchema

    # If load builds a client, this blows up — proving load stays offline/lazy.
    mocker.patch(
        "langres.clients.create_llm_client",
        side_effect=AssertionError("client must not be built during save/load"),
    )

    judge: LLMJudge[CompanySchema] = LLMJudge(
        model="openrouter/openai/gpt-4o-mini",
        client=object(),
        entity_noun="company",
    )
    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        module=judge,
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
    assert isinstance(reloaded.module, LLMJudge)
    assert reloaded.module.client is None  # lazy — not built at load
    assert reloaded.module.config == judge.config
