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
def mock_openai_client(mocker):
    """Mock OpenAI client for testing without API calls."""
    mock_client = Mock()
    mocker.patch("langres.core.modules.llm_judge.OpenAI", return_value=mock_client)
    return mock_client


def test_llm_judge_initialization():
    """Test LLMJudgeModule can be initialized with valid parameters."""
    module = LLMJudgeModule(
        model="gpt-4o-mini",
        api_key="test-key",
        temperature=0.0,
    )

    assert module.model == "gpt-4o-mini"
    assert module.temperature == 0.0


def test_llm_judge_requires_api_key():
    """Test LLMJudgeModule validates API key is provided."""
    with pytest.raises(ValueError, match="API key is required"):
        LLMJudgeModule(model="gpt-4o-mini", api_key="")


def test_llm_judge_requires_valid_temperature():
    """Test LLMJudgeModule validates temperature is in range [0, 2]."""
    with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
        LLMJudgeModule(model="gpt-4o-mini", api_key="test-key", temperature=2.5)


def test_llm_judge_scores_single_pair(mock_openai_client):
    """Test LLMJudgeModule scores a single entity pair."""
    # Setup mock response
    mock_response = ChatCompletion(
        id="test",
        model="gpt-4o-mini",
        object="chat.completion",
        created=0,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="MATCH\nScore: 0.95\nReasoning: These are clearly the same company with minor name variations.",
                ),
                finish_reason="stop",
            )
        ],
    )
    mock_openai_client.chat.completions.create.return_value = mock_response

    # Create module
    module = LLMJudgeModule(model="gpt-4o-mini", api_key="test-key")

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


def test_llm_judge_extracts_score_from_response(mock_openai_client):
    """Test LLMJudgeModule correctly extracts score from LLM response."""
    mock_response = ChatCompletion(
        id="test",
        model="gpt-4o-mini",
        object="chat.completion",
        created=0,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="NO_MATCH\nScore: 0.15\nReasoning: Completely different companies.",
                ),
                finish_reason="stop",
            )
        ],
    )
    mock_openai_client.chat.completions.create.return_value = mock_response

    module = LLMJudgeModule(model="gpt-4o-mini", api_key="test-key")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme Corporation"),
        right=CompanySchema(id="c2", name="TechStart Industries"),
        blocker_name="test",
    )

    judgements = list(module.forward([candidate]))
    j = judgements[0]

    # Should extract score 0.15
    assert j.score == 0.15


def test_llm_judge_tracks_cost_in_provenance(mock_openai_client):
    """Test LLMJudgeModule tracks API cost in provenance."""
    mock_response = ChatCompletion(
        id="test",
        model="gpt-4o-mini",
        object="chat.completion",
        created=0,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant", content="MATCH\nScore: 0.90\nReasoning: Same company."
                ),
                finish_reason="stop",
            )
        ],
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )
    mock_openai_client.chat.completions.create.return_value = mock_response

    module = LLMJudgeModule(model="gpt-4o-mini", api_key="test-key")

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


def test_llm_judge_handles_multiple_pairs(mock_openai_client):
    """Test LLMJudgeModule processes multiple pairs in sequence."""
    # Mock responses for each pair
    responses = [
        ChatCompletion(
            id="test1",
            model="gpt-4o-mini",
            object="chat.completion",
            created=0,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="MATCH\nScore: 0.95\nReasoning: Same company.",
                    ),
                    finish_reason="stop",
                )
            ],
        ),
        ChatCompletion(
            id="test2",
            model="gpt-4o-mini",
            object="chat.completion",
            created=0,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="NO_MATCH\nScore: 0.10\nReasoning: Different companies.",
                    ),
                    finish_reason="stop",
                )
            ],
        ),
    ]
    mock_openai_client.chat.completions.create.side_effect = responses

    module = LLMJudgeModule(model="gpt-4o-mini", api_key="test-key")

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


def test_llm_judge_handles_api_error(mock_openai_client):
    """Test LLMJudgeModule handles API errors gracefully."""
    mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

    module = LLMJudgeModule(model="gpt-4o-mini", api_key="test-key")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme Corporation"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )

    with pytest.raises(Exception, match="API Error"):
        list(module.forward([candidate]))


def test_llm_judge_uses_custom_prompt():
    """Test LLMJudgeModule accepts custom prompt template."""
    custom_prompt = "Are these the same? {left_name} vs {right_name}"

    module = LLMJudgeModule(model="gpt-4o-mini", api_key="test-key", prompt_template=custom_prompt)

    assert module.prompt_template == custom_prompt
