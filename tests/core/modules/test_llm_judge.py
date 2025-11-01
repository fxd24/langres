"""Tests for LLMJudgeModule (LLM-based matching).

This test module validates the LLMJudgeModule implementation, which uses
OpenAI API (or similar) for match judgments with natural language reasoning.
"""

import logging
from unittest.mock import Mock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.modules.llm_judge import LLMJudgeModule

logger = logging.getLogger(__name__)


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


def test_llm_judge_tracks_cost_in_provenance(mock_llm_client):
    """Test LLMJudgeModule tracks API cost in provenance."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "MATCH\nScore: 0.90\nReasoning: Same company."
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_llm_client.completion.return_value = mock_response

    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme Corporation"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )

    judgements = list(module.forward([candidate]))
    j = judgements[0]

    # Should have cost tracking in provenance
    assert "cost_usd" in j.provenance
    assert isinstance(j.provenance["cost_usd"], float)
    assert j.provenance["cost_usd"] > 0
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
    """Test LLMJudgeModule accepts custom prompt template."""
    custom_prompt = "Are these the same? {left_name} vs {right_name}"

    module = LLMJudgeModule(
        client=mock_llm_client, model="gpt-4o-mini", prompt_template=custom_prompt
    )

    assert module.prompt_template == custom_prompt


def test_llm_judge_score_extraction_fallback():
    """Test that LLMJudgeModule falls back to 0.5 when score extraction fails."""
    # Mock client to return response without score
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[
        0
    ].message.content = "These entities might be the same, I'm not sure."  # No score
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 20
    mock_client.completion.return_value = mock_response

    module = LLMJudgeModule(client=mock_client, model="gpt-4o-mini")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme"),
        right=CompanySchema(id="c2", name="Beta"),
        blocker_name="test",
    )

    # Should log warning and use 0.5 as default
    judgements = list(module.forward([candidate]))
    assert len(judgements) == 1
    assert judgements[0].score == 0.5  # Default fallback


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


def test_llm_judge_gpt4_pricing():
    """Test that GPT-4 pricing is calculated correctly."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "MATCH\nScore: 0.9\nReasoning: Same company"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 1000
    mock_response.usage.completion_tokens = 100
    mock_client.completion.return_value = mock_response

    module = LLMJudgeModule(client=mock_client, model="gpt-4")  # Standard GPT-4

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )

    judgements = list(module.forward([candidate]))

    # GPT-4: $30/1M input, $60/1M output
    expected_cost = (1000 * 30.0 + 100 * 60.0) / 1_000_000
    assert abs(judgements[0].provenance["cost_usd"] - expected_cost) < 0.001


def test_llm_judge_unknown_model_pricing():
    """Test that unknown models default to gpt-4o-mini pricing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Score: 0.5\nReasoning: Test"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 20
    mock_client.completion.return_value = mock_response

    module = LLMJudgeModule(client=mock_client, model="gpt-future-5")  # Unknown model

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme"),
        right=CompanySchema(id="c2", name="Beta"),
        blocker_name="test",
    )

    judgements = list(module.forward([candidate]))

    # Should use default (gpt-4o-mini) pricing
    expected_cost = (100 * 0.150 + 20 * 0.600) / 1_000_000
    assert abs(judgements[0].provenance["cost_usd"] - expected_cost) < 0.001


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
    assert call_kwargs["temperature"] == 0.0

    # Verify judgement
    assert len(judgements) == 1
    assert judgements[0].score == 0.90
