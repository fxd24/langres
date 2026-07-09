"""Tests for LLMJudgeModule (LLM-based matching).

This test module validates the LLMJudgeModule implementation, which uses
OpenAI API (or similar) for match judgments with natural language reasoning.
"""

import logging
from unittest.mock import Mock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from langres.clients.openrouter import SpendMonitor
from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.modules.llm_judge import LLMJudge, LLMJudgeModule
from langres.core.registry import get_component

logger = logging.getLogger(__name__)

# The hidden-param header LiteLLM's OpenRouter transform writes the real cost to.
_COST_HEADER = "llm_provider-x-litellm-response-cost"


def _pair() -> ERCandidate[CompanySchema]:
    """A minimal company candidate pair for judge tests."""
    return ERCandidate(
        left=CompanySchema(id="c1", name="Acme Corporation"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )


def _openrouter_response(
    content: str,
    *,
    real_cost: object,
    provider: object,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> Mock:
    """A completion response carrying OpenRouter's real billed cost + serving provider.

    Mirrors how LiteLLM's OpenRouter transform surfaces usage-accounting cost
    (``_hidden_params['additional_headers']``) and the serving provider.
    """
    resp = Mock()
    resp.choices = [Mock()]
    resp.choices[0].message.content = content
    resp.usage = Mock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp._hidden_params = {"additional_headers": {_COST_HEADER: real_cost}}
    resp.provider = provider
    return resp


def test_registered_under_llm_judge_via_lazy_lookup() -> None:
    """``get_component('llm_judge')`` lazily imports+registers the class.

    (W0.4: litellm is optional, so ``llm_judge`` joined
    ``_LAZY_COMPONENT_MODULES`` alongside ``dspy_judge``.)
    """
    assert get_component("llm_judge") is LLMJudge
    assert LLMJudge.type_name == "llm_judge"


@pytest.fixture
def mock_llm_client():
    """Mock LiteLLM client for testing without API calls."""
    return Mock()


def test_llm_judge_initialization(mock_llm_client):
    """Test LLMJudgeModule can be initialized with valid parameters."""
    module = LLMJudgeModule(
        client=mock_llm_client,
        model="gpt-4o-mini",
        temperature=0.0,
    )

    assert module.client is mock_llm_client
    assert module.model == "gpt-4o-mini"
    assert module.temperature == 0.0


def test_llm_judge_requires_valid_temperature(mock_llm_client):
    """Test LLMJudgeModule validates temperature is in range [0, 2]."""
    with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
        LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini", temperature=2.5)


def test_llm_judge_scores_single_pair(mock_llm_client):
    """Test LLMJudgeModule scores a single entity pair."""
    # Setup mock response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[
        0
    ].message.content = "MATCH\nScore: 0.95\nReasoning: These are clearly the same company with minor name variations."
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_llm_client.completion.return_value = mock_response

    # Create module
    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    # Create candidate pair
    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme Corporation"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test_blocker",
    )

    # Score the pair
    judgements = list(module.forward([candidate]))

    assert len(judgements) == 1
    j = judgements[0]
    assert j.left_id == "c1"
    assert j.right_id == "c2"
    assert 0.0 <= j.score <= 1.0
    assert j.score_type == "prob_llm"
    assert j.decision_step == "llm_judgment"
    assert j.reasoning is not None
    assert len(j.reasoning) > 0


def test_llm_judge_extracts_score_from_response(mock_llm_client):
    """Test LLMJudgeModule correctly extracts score from LLM response."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[
        0
    ].message.content = "NO_MATCH\nScore: 0.15\nReasoning: Completely different companies."
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 30
    mock_llm_client.completion.return_value = mock_response

    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme Corporation"),
        right=CompanySchema(id="c2", name="TechStart Industries"),
        blocker_name="test",
    )

    judgements = list(module.forward([candidate]))
    j = judgements[0]

    # Should extract score 0.15
    assert j.score == 0.15


def test_llm_judge_tracks_cost_in_provenance(mock_llm_client, mocker):
    """Test LLMJudgeModule tracks API cost in provenance via litellm pricing."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "MATCH\nScore: 0.90\nReasoning: Same company."
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_llm_client.completion.return_value = mock_response

    # litellm.completion_cost owns pricing; stub it so the Mock response prices.
    completion_cost = mocker.patch(
        "langres.core.modules.llm_judge.litellm.completion_cost", return_value=0.000123
    )

    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme Corporation"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )

    judgements = list(module.forward([candidate]))
    j = judgements[0]

    # Should have cost tracking in provenance, sourced from litellm.completion_cost
    assert "cost_usd" in j.provenance
    assert isinstance(j.provenance["cost_usd"], float)
    assert j.provenance["cost_usd"] == 0.000123
    completion_cost.assert_called_once_with(completion_response=mock_response)
    assert "prompt_tokens" in j.provenance
    assert j.provenance["prompt_tokens"] == 100
    assert "completion_tokens" in j.provenance
    assert j.provenance["completion_tokens"] == 50


def test_llm_judge_handles_multiple_pairs(mock_llm_client):
    """Test LLMJudgeModule processes multiple pairs in sequence."""
    # Mock responses for each pair
    mock_resp1 = Mock()
    mock_resp1.choices = [Mock()]
    mock_resp1.choices[0].message.content = "MATCH\nScore: 0.95\nReasoning: Same company."
    mock_resp1.usage = Mock()
    mock_resp1.usage.prompt_tokens = 100
    mock_resp1.usage.completion_tokens = 30

    mock_resp2 = Mock()
    mock_resp2.choices = [Mock()]
    mock_resp2.choices[0].message.content = "NO_MATCH\nScore: 0.10\nReasoning: Different companies."
    mock_resp2.usage = Mock()
    mock_resp2.usage.prompt_tokens = 100
    mock_resp2.usage.completion_tokens = 30

    mock_llm_client.completion.side_effect = [mock_resp1, mock_resp2]

    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    candidates = [
        ERCandidate(
            left=CompanySchema(id="c1", name="Acme Corporation"),
            right=CompanySchema(id="c2", name="Acme Corp"),
            blocker_name="test",
        ),
        ERCandidate(
            left=CompanySchema(id="c3", name="TechStart Industries"),
            right=CompanySchema(id="c4", name="DataFlow Solutions"),
            blocker_name="test",
        ),
    ]

    judgements = list(module.forward(candidates))

    assert len(judgements) == 2
    assert judgements[0].score == 0.95
    assert judgements[1].score == 0.10


def test_llm_judge_handles_api_error(mock_llm_client):
    """Test LLMJudgeModule handles API errors gracefully."""
    mock_llm_client.completion.side_effect = Exception("API Error")

    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme Corporation"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )

    with pytest.raises(Exception, match="API Error"):
        list(module.forward([candidate]))


def test_llm_judge_uses_custom_prompt(mock_llm_client):
    """Test LLMJudgeModule accepts custom prompt template (with the required
    ``{left}``/``{right}`` placeholders)."""
    custom_prompt = "Are these the same? {left} vs {right}"

    module = LLMJudgeModule(
        client=mock_llm_client, model="gpt-4o-mini", prompt_template=custom_prompt
    )

    assert module.prompt_template == custom_prompt


def test_llm_judge_score_extraction_abstains_with_flag():
    """An unparseable score now ABSTAINS (flagged 0.0), never a silent 0.5.

    Behavior change (defect fix): the old code returned a real-looking 0.5 that
    was indistinguishable downstream from a genuine mid-confidence verdict. The
    default ``on_parse_error='abstain'`` policy instead emits score 0.0 flagged
    with ``provenance['parse_error']`` so the abstention is visible.
    """
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[
        0
    ].message.content = "These entities might be the same, I'm not sure."  # No score
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 20
    mock_response.usage.prompt_tokens_details = None
    mock_response.usage.completion_tokens_details = None
    mock_client.completion.return_value = mock_response

    module = LLMJudgeModule(client=mock_client, model="gpt-4o-mini")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme"),
        right=CompanySchema(id="c2", name="Beta"),
        blocker_name="test",
    )

    judgements = list(module.forward([candidate]))
    assert len(judgements) == 1
    assert judgements[0].score == 0.0
    assert judgements[0].provenance["parse_error"] is True


def test_llm_judge_reasoning_extraction_fallback():
    """Test that LLMJudgeModule falls back to full content when reasoning extraction fails."""
    # Mock response without "Reasoning:" prefix
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    # No explicit "Reasoning:" - should return full content
    mock_response.choices[0].message.content = "Score: 0.8\nThese are similar companies."
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 20
    mock_client.completion.return_value = mock_response

    module = LLMJudgeModule(client=mock_client, model="gpt-4o-mini")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme"),
        right=CompanySchema(id="c2", name="Acme"),
        blocker_name="test",
    )

    judgements = list(module.forward([candidate]))
    # Should use full content as reasoning
    assert len(judgements) == 1
    assert judgements[0].reasoning is not None
    assert "similar companies" in judgements[0].reasoning.lower()


def test_llm_judge_cost_uses_litellm_completion_cost(mocker):
    """The reported cost is whatever litellm.completion_cost returns."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "MATCH\nScore: 0.9\nReasoning: Same company"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 1000
    mock_response.usage.completion_tokens = 100
    mock_client.completion.return_value = mock_response

    completion_cost = mocker.patch(
        "langres.core.modules.llm_judge.litellm.completion_cost", return_value=0.036
    )

    module = LLMJudgeModule(client=mock_client, model="gpt-4")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )

    judgements = list(module.forward([candidate]))

    assert judgements[0].provenance["cost_usd"] == 0.036
    completion_cost.assert_called_once_with(completion_response=mock_response)


def test_llm_judge_cost_falls_back_to_zero_on_exception(mocker, caplog):
    """If litellm.completion_cost raises (unknown model/usage), cost is 0.0."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Score: 0.5\nReasoning: Test"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 20
    mock_client.completion.return_value = mock_response

    mocker.patch(
        "langres.core.modules.llm_judge.litellm.completion_cost",
        side_effect=Exception("unknown model"),
    )

    module = LLMJudgeModule(client=mock_client, model="gpt-future-5")  # Unknown model

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme"),
        right=CompanySchema(id="c2", name="Beta"),
        blocker_name="test",
    )

    with caplog.at_level(logging.WARNING):
        judgements = list(module.forward([candidate]))

    assert judgements[0].provenance["cost_usd"] == 0.0
    assert "completion_cost unavailable" in caplog.text


# Client Integration Tests


def test_llm_judge_client_integration(mock_llm_client):
    """Test LLMJudgeModule uses client.completion() API."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "MATCH\nScore: 0.90\nReasoning: Same company"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 30
    mock_llm_client.completion.return_value = mock_response

    module = LLMJudgeModule(
        client=mock_llm_client,
        model="gpt-4o-mini",
        temperature=0.5,
    )

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )

    judgements = list(module.forward([candidate]))

    # Verify client.completion was called
    mock_llm_client.completion.assert_called_once()
    call_kwargs = mock_llm_client.completion.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["temperature"] == 0.5

    # Verify judgement
    assert len(judgements) == 1
    assert judgements[0].score == 0.90


def test_llm_judge_handles_missing_usage_info_in_response(mock_llm_client):
    """Test LLMJudgeModule handles response with no usage information."""
    # Setup mock response without usage info
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "MATCH\nScore: 0.85\nReasoning: Same entity"
    mock_response.usage = None  # No usage information
    mock_llm_client.completion.return_value = mock_response

    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Test Company"),
        right=CompanySchema(id="c2", name="Test Co"),
        blocker_name="test",
    )

    # Should not raise an error
    judgements = list(module.forward([candidate]))

    # Verify judgement is created successfully
    assert len(judgements) == 1
    assert judgements[0].score == 0.85
    # Cost should be 0.0 when usage is None
    assert judgements[0].provenance["cost_usd"] == 0.0
    assert judgements[0].provenance["prompt_tokens"] == 0
    assert judgements[0].provenance["completion_tokens"] == 0


# Real OpenRouter cost (usage accounting) + provider pinning


def test_forward_records_real_openrouter_cost_and_provider(mocker):
    """The real billed cost + serving provider are recorded; the estimate is skipped."""
    client = Mock()
    client.completion.return_value = _openrouter_response(
        "MATCH\nScore: 0.9\nReasoning: Same company", real_cost=0.00042, provider="DeepInfra"
    )
    # If real cost is used, the pinned-table estimator must NOT be consulted.
    completion_cost = mocker.patch("langres.core.modules.llm_judge.litellm.completion_cost")

    module = LLMJudgeModule(client=client, model="openrouter/z-ai/glm-5.2")
    j = list(module.forward([_pair()]))[0]

    assert j.provenance["cost_usd"] == pytest.approx(0.00042)
    assert j.provenance["cost_is_real"] is True
    assert j.provenance["provider"] == "DeepInfra"
    completion_cost.assert_not_called()


def test_spend_monitor_records_the_real_cost() -> None:
    """SpendMonitor accumulates the real billed cost that forward() records."""
    client = Mock()
    client.completion.return_value = _openrouter_response(
        "MATCH\nScore: 0.8\nReasoning: x", real_cost=0.0031, provider="Together"
    )
    module = LLMJudgeModule(client=client, model="openrouter/z-ai/glm-5.2")

    monitor = SpendMonitor(budget_usd=1.0)
    for j in module.forward([_pair()]):
        monitor.add(float(j.provenance["cost_usd"]))

    assert monitor.spent == pytest.approx(0.0031)


def test_forward_falls_back_to_pinned_estimate_without_real_cost(mocker):
    """With no real cost on the response, cost falls back to litellm's estimate."""
    client = Mock()
    resp = Mock()
    resp.choices = [Mock()]
    resp.choices[0].message.content = "MATCH\nScore: 0.7\nReasoning: x"
    resp.usage = Mock()
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 50
    resp._hidden_params = {}  # no additional_headers → no real cost
    resp.usage.cost = None  # and no raw usage.cost either
    resp.provider = None
    client.completion.return_value = resp

    completion_cost = mocker.patch(
        "langres.core.modules.llm_judge.litellm.completion_cost", return_value=0.00777
    )

    module = LLMJudgeModule(client=client, model="openrouter/z-ai/glm-5.2")
    j = list(module.forward([_pair()]))[0]

    assert j.provenance["cost_usd"] == pytest.approx(0.00777)
    assert j.provenance["cost_is_real"] is False
    assert j.provenance["provider"] is None
    completion_cost.assert_called_once_with(completion_response=resp)


def test_openrouter_model_requests_usage_accounting() -> None:
    """An openrouter/ model sends extra_body={"usage": {"include": True}}."""
    client = Mock()
    client.completion.return_value = _openrouter_response(
        "MATCH\nScore: 0.9\nReasoning: x", real_cost=0.0001, provider="DeepInfra"
    )
    module = LLMJudgeModule(client=client, model="openrouter/z-ai/glm-5.2")

    list(module.forward([_pair()]))

    extra_body = client.completion.call_args.kwargs["extra_body"]
    assert extra_body["usage"] == {"include": True}
    assert "provider" not in extra_body  # no pin configured


def test_provider_pin_is_threaded_into_extra_body() -> None:
    """A provider pin is passed through as extra_body["provider"] for reproducibility."""
    client = Mock()
    client.completion.return_value = _openrouter_response(
        "MATCH\nScore: 0.9\nReasoning: x", real_cost=0.0001, provider="DeepInfra"
    )
    pin = {"order": ["DeepInfra"], "allow_fallbacks": False}
    module = LLMJudgeModule(client=client, model="openrouter/z-ai/glm-5.2", provider=pin)

    list(module.forward([_pair()]))

    extra_body = client.completion.call_args.kwargs["extra_body"]
    assert extra_body["provider"] == pin
    assert extra_body["usage"] == {"include": True}


def test_non_openrouter_model_sends_no_extra_body() -> None:
    """Off OpenRouter, no usage/provider extra_body is sent (and a pin is ignored)."""
    client = Mock()
    resp = Mock()
    resp.choices = [Mock()]
    resp.choices[0].message.content = "MATCH\nScore: 0.9\nReasoning: x"
    resp.usage = Mock()
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 50
    client.completion.return_value = resp

    module = LLMJudgeModule(client=client, model="gpt-5-mini", provider={"only": ["X"]})
    list(module.forward([_pair()]))

    assert "extra_body" not in client.completion.call_args.kwargs
