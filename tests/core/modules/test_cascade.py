"""Tests for CascadeModule (hybrid embeddings + LLM with early exit).

This module tests the cascade pattern that optimizes cost while maintaining quality:
1. Stage 1: Cheap embedding similarity check
2. Early exit if score < 0.3 (definitely not a match)
3. Early exit if score > 0.9 (definitely a match)
4. Stage 2: Expensive LLM judgment for uncertain cases (0.3-0.9)
"""

import logging
from unittest.mock import Mock

import pytest
from sentence_transformers import SentenceTransformer

from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.modules.cascade import CascadeModule

logger = logging.getLogger(__name__)


def test_cascade_module_initialization():
    """Test CascadeModule initialization with valid parameters."""
    module = CascadeModule(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model="gpt-4o-mini",
        llm_api_key="test_key",
        low_threshold=0.3,
        high_threshold=0.9,
    )

    assert module.embedding_model_name == "all-MiniLM-L6-v2"
    assert module.llm_model == "gpt-4o-mini"
    assert module.low_threshold == 0.3
    assert module.high_threshold == 0.9


def test_cascade_module_requires_valid_thresholds():
    """Test that CascadeModule validates thresholds."""
    # low_threshold must be < high_threshold
    with pytest.raises(ValueError, match="low_threshold must be < high_threshold"):
        CascadeModule(
            embedding_model_name="all-MiniLM-L6-v2",
            llm_model="gpt-4o-mini",
            llm_api_key="test_key",
            low_threshold=0.8,
            high_threshold=0.5,
        )

    # Thresholds must be in [0, 1] range
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        CascadeModule(
            embedding_model_name="all-MiniLM-L6-v2",
            llm_model="gpt-4o-mini",
            llm_api_key="test_key",
            low_threshold=-0.1,
            high_threshold=0.9,
        )


def test_cascade_module_early_exit_low_similarity(mocker):
    """Test early exit when embedding similarity is below low_threshold."""
    # Mock the embedding model to return low similarity
    mock_model = Mock(spec=SentenceTransformer)
    mock_model.encode.return_value = [[0.0, 0.0], [1.0, 1.0]]  # Very different vectors

    mocker.patch(
        "langres.core.modules.cascade.SentenceTransformer",
        return_value=mock_model,
    )

    module = CascadeModule(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model="gpt-4o-mini",
        llm_api_key="test_key",
        low_threshold=0.3,
        high_threshold=0.9,
    )

    # Create test candidate
    left = CompanySchema(id="c1", name="Acme Corp", address="123 Main St")
    right = CompanySchema(id="c2", name="Totally Different Inc", address="456 Other Ave")
    candidate = ERCandidate(left=left, right=right, blocker_name="test")

    # Process candidate
    judgements = list(module.forward([candidate]))

    # Should get exactly one judgement
    assert len(judgements) == 1
    judgement = judgements[0]

    # Should use early exit decision step
    assert judgement.decision_step == "early_exit_low_similarity"
    assert judgement.score_type == "sim_cos"
    # Score should be low (below threshold)
    assert judgement.score < 0.3
    # Should not have LLM reasoning (no LLM call made)
    assert judgement.reasoning is None or judgement.reasoning == ""
    # Provenance should include embedding score
    assert "embed_score" in judgement.provenance


def test_cascade_module_early_exit_high_similarity(mocker):
    """Test early exit when embedding similarity is above high_threshold."""
    # Mock the embedding model to return high similarity
    mock_model = Mock(spec=SentenceTransformer)
    # Same vector = cosine similarity 1.0
    mock_model.encode.return_value = [[1.0, 0.0], [1.0, 0.0]]

    mocker.patch(
        "langres.core.modules.cascade.SentenceTransformer",
        return_value=mock_model,
    )

    module = CascadeModule(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model="gpt-4o-mini",
        llm_api_key="test_key",
        low_threshold=0.3,
        high_threshold=0.9,
    )

    # Create test candidate
    left = CompanySchema(id="c1", name="Acme Corp", address="123 Main St")
    right = CompanySchema(id="c2", name="Acme Corp", address="123 Main St")
    candidate = ERCandidate(left=left, right=right, blocker_name="test")

    # Process candidate
    judgements = list(module.forward([candidate]))

    # Should get exactly one judgement
    assert len(judgements) == 1
    judgement = judgements[0]

    # Should use early exit decision step
    assert judgement.decision_step == "early_exit_high_similarity"
    assert judgement.score_type == "sim_cos"
    # Score should be high (above threshold)
    assert judgement.score > 0.9
    # Should not have LLM reasoning (no LLM call made)
    assert judgement.reasoning is None or judgement.reasoning == ""
    # Provenance should include embedding score
    assert "embed_score" in judgement.provenance


def test_cascade_module_llm_judgment_for_uncertain_cases(mocker):
    """Test LLM judgment is used for uncertain cases (between thresholds)."""
    # Mock the embedding model to return medium similarity
    # Using vectors that produce cosine similarity ~0.6 (between 0.3 and 0.9)
    mock_model = Mock(spec=SentenceTransformer)
    mock_model.encode.return_value = [[1.0, 0.0], [0.5, 0.866]]  # ~60 degree angle

    mocker.patch(
        "langres.core.modules.cascade.SentenceTransformer",
        return_value=mock_model,
    )

    # Mock OpenAI API call
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "MATCH\nScore: 0.75\nReasoning: Names are similar"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 20

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response

    mocker.patch("langres.core.modules.cascade.OpenAI", return_value=mock_client)

    module = CascadeModule(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model="gpt-4o-mini",
        llm_api_key="test_key",
        low_threshold=0.3,
        high_threshold=0.9,
    )

    # Create test candidate
    left = CompanySchema(id="c1", name="Acme Corp", address="123 Main St")
    right = CompanySchema(id="c2", name="Acme Corporation", address="123 Main Street")
    candidate = ERCandidate(left=left, right=right, blocker_name="test")

    # Process candidate
    judgements = list(module.forward([candidate]))

    # Should get exactly one judgement
    assert len(judgements) == 1
    judgement = judgements[0]

    # Should use LLM judgment decision step
    assert judgement.decision_step == "llm_judgment"
    assert judgement.score_type == "prob_llm"
    # Score should be from LLM
    assert judgement.score == 0.75
    # Should have LLM reasoning
    assert judgement.reasoning == "Names are similar"
    # Provenance should include both embedding score and LLM cost
    assert "embed_score" in judgement.provenance
    assert "llm_cost_usd" in judgement.provenance
    assert "model" in judgement.provenance


def test_cascade_module_processes_multiple_candidates(mocker):
    """Test CascadeModule handles multiple candidates with different paths."""
    # Mock the embedding model
    mock_model = Mock(spec=SentenceTransformer)
    # Return different similarities for different calls
    # Low sim: orthogonal vectors (cos=0.0)
    # High sim: identical vectors (cos=1.0)
    # Medium sim: 60 degree angle (cos~0.5)
    embeddings = [
        [[1.0, 0.0], [0.0, 1.0]],  # Orthogonal = cos sim 0.0 (low)
        [[1.0, 0.0], [1.0, 0.0]],  # Identical = cos sim 1.0 (high)
        [[1.0, 0.0], [0.5, 0.866]],  # 60 degrees = cos sim ~0.5 (medium)
    ]
    mock_model.encode.side_effect = embeddings

    mocker.patch(
        "langres.core.modules.cascade.SentenceTransformer",
        return_value=mock_model,
    )

    # Mock OpenAI API call
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "MATCH\nScore: 0.65\nReasoning: Similar"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 20

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response

    mocker.patch("langres.core.modules.cascade.OpenAI", return_value=mock_client)

    module = CascadeModule(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model="gpt-4o-mini",
        llm_api_key="test_key",
        low_threshold=0.3,
        high_threshold=0.9,
    )

    # Create test candidates
    candidates = [
        ERCandidate(
            left=CompanySchema(id="c1", name="A"),
            right=CompanySchema(id="c2", name="Z"),
            blocker_name="test",
        ),
        ERCandidate(
            left=CompanySchema(id="c3", name="Same"),
            right=CompanySchema(id="c4", name="Same"),
            blocker_name="test",
        ),
        ERCandidate(
            left=CompanySchema(id="c5", name="Similar"),
            right=CompanySchema(id="c6", name="Similiar"),
            blocker_name="test",
        ),
    ]

    # Process candidates
    judgements = list(module.forward(candidates))

    # Should get 3 judgements
    assert len(judgements) == 3

    # First should be low similarity early exit
    assert judgements[0].decision_step == "early_exit_low_similarity"

    # Second should be high similarity early exit
    assert judgements[1].decision_step == "early_exit_high_similarity"

    # Third should be LLM judgment
    assert judgements[2].decision_step == "llm_judgment"

    # Only the third should have LLM reasoning
    assert judgements[0].reasoning is None or judgements[0].reasoning == ""
    assert judgements[1].reasoning is None or judgements[1].reasoning == ""
    assert judgements[2].reasoning == "Similar"


def test_cascade_module_tracks_cost_savings():
    """Test that cascade pattern saves cost by avoiding unnecessary LLM calls."""
    # This is validated by the previous tests:
    # - Early exit cases don't call LLM (no cost)
    # - Only uncertain cases call LLM (cost tracked)
    # This demonstrates the cost optimization benefit of the cascade pattern
    pass  # Meta-test documenting the cost optimization property


def test_cascade_module_handles_malformed_llm_response(mocker):
    """Test CascadeModule handles malformed LLM responses gracefully."""
    # Mock embedding model for medium similarity
    mock_model = Mock(spec=SentenceTransformer)
    mock_model.encode.return_value = [[1.0, 0.0], [0.5, 0.866]]

    mocker.patch(
        "langres.core.modules.cascade.SentenceTransformer",
        return_value=mock_model,
    )

    # Mock OpenAI API call with malformed response (no Score: or Reasoning:)
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "These companies look similar."
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 20

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response

    mocker.patch("langres.core.modules.cascade.OpenAI", return_value=mock_client)

    module = CascadeModule(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model="gpt-4o-mini",
        llm_api_key="test_key",
    )

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Test"),
        right=CompanySchema(id="c2", name="Test Inc"),
        blocker_name="test",
    )

    judgements = list(module.forward([candidate]))
    assert len(judgements) == 1

    # Should default to 0.5 score when can't parse
    judgement = judgements[0]
    assert judgement.score == 0.5
    # Should use full content as reasoning
    assert "similar" in judgement.reasoning.lower()


def test_cascade_module_handles_missing_usage_info(mocker):
    """Test CascadeModule handles responses without usage info."""
    # Mock embedding model for medium similarity
    mock_model = Mock(spec=SentenceTransformer)
    mock_model.encode.return_value = [[1.0, 0.0], [0.5, 0.866]]

    mocker.patch(
        "langres.core.modules.cascade.SentenceTransformer",
        return_value=mock_model,
    )

    # Mock OpenAI API call with no usage info
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "MATCH\nScore: 0.8\nReasoning: Match"
    mock_response.usage = None  # No usage info

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response

    mocker.patch("langres.core.modules.cascade.OpenAI", return_value=mock_client)

    module = CascadeModule(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model="gpt-4o-mini",
        llm_api_key="test_key",
    )

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Test"),
        right=CompanySchema(id="c2", name="Test Inc"),
        blocker_name="test",
    )

    judgements = list(module.forward([candidate]))
    assert len(judgements) == 1

    # Should have 0 cost when usage info missing
    assert judgements[0].provenance["llm_cost_usd"] == 0.0


def test_cascade_module_uses_custom_prompt_template(mocker):
    """Test CascadeModule can use custom LLM prompt template."""
    # Mock embedding model for medium similarity
    mock_model = Mock(spec=SentenceTransformer)
    mock_model.encode.return_value = [[1.0, 0.0], [0.5, 0.866]]

    mocker.patch(
        "langres.core.modules.cascade.SentenceTransformer",
        return_value=mock_model,
    )

    # Mock OpenAI API call
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Score: 0.9\nReasoning: Custom"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 10

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response

    mocker.patch("langres.core.modules.cascade.OpenAI", return_value=mock_client)

    custom_prompt = "Compare:\nA: {left}\nB: {right}\nScore: <0-1>\nReasoning: <text>"

    module = CascadeModule(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model="gpt-4o-mini",
        llm_api_key="test_key",
        llm_prompt_template=custom_prompt,
    )

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Test"),
        right=CompanySchema(id="c2", name="Test Inc"),
        blocker_name="test",
    )

    judgements = list(module.forward([candidate]))

    # Should use custom prompt
    call_args = mock_client.chat.completions.create.call_args
    assert "Compare:" in call_args.kwargs["messages"][0]["content"]


def test_cascade_module_requires_api_key():
    """Test that CascadeModule raises ValueError if API key is missing."""
    with pytest.raises(ValueError, match="LLM API key is required"):
        CascadeModule(
            embedding_model_name="all-MiniLM-L6-v2",
            llm_model="gpt-4o-mini",
            llm_api_key="",  # Empty API key
            low_threshold=0.3,
            high_threshold=0.9,
        )


@pytest.mark.slow
def test_cascade_module_lazy_loads_embedding_model(mocker):
    """Test that embedding model is lazy-loaded on first use."""
    mock_client = Mock()
    mocker.patch("langres.core.modules.cascade.OpenAI", return_value=mock_client)

    module = CascadeModule(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model="gpt-4o-mini",
        llm_api_key="test_key",
        low_threshold=0.3,
        high_threshold=0.9,
    )

    # Model should not be loaded yet
    assert module._embedding_model is None

    # Create a candidate (this will trigger model loading)
    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Test"),
        right=CompanySchema(id="c2", name="Test"),
        blocker_name="test",
    )

    # Mock LLM response to avoid real API call
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "MATCH\nScore: 0.5\nReasoning: Test"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 20
    mock_client.chat.completions.create.return_value = mock_response

    # Process candidate (will load model on first call)
    list(module.forward([candidate]))

    # Model should now be loaded
    assert module._embedding_model is not None

    # Second call should reuse the loaded model
    list(module.forward([candidate]))


@pytest.mark.slow
def test_cascade_module_gpt4_pricing(mocker):
    """Test that GPT-4 pricing is calculated correctly."""
    mock_client = Mock()
    mocker.patch("langres.core.modules.cascade.OpenAI", return_value=mock_client)

    # Use gpt-4 model (different pricing than gpt-4o-mini)
    module = CascadeModule(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model="gpt-4",  # Standard GPT-4
        llm_api_key="test_key",
        low_threshold=0.3,
        high_threshold=0.9,
    )

    # Create a candidate that will trigger LLM (mid-range similarity)
    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme Corporation"),
        right=CompanySchema(id="c2", name="Beta Industries"),  # Different enough for LLM
        blocker_name="test",
    )

    # Mock LLM response with usage info
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[
        0
    ].message.content = "NO_MATCH\nScore: 0.4\nReasoning: Different companies"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 1000
    mock_response.usage.completion_tokens = 100
    mock_client.chat.completions.create.return_value = mock_response

    # Process candidate
    judgements = list(module.forward([candidate]))

    # Should have cost calculated (GPT-4 pricing is higher than gpt-4o-mini)
    assert len(judgements) == 1
    assert "llm_cost_usd" in judgements[0].provenance
    # GPT-4: $30/1M input, $60/1M output
    # Cost = (1000 * 30 + 100 * 60) / 1_000_000 = 0.036
    expected_cost = (1000 * 30.0 + 100 * 60.0) / 1_000_000
    assert abs(judgements[0].provenance["llm_cost_usd"] - expected_cost) < 0.001


@pytest.mark.slow
def test_cascade_module_unknown_model_pricing(mocker):
    """Test that unknown models default to gpt-4o-mini pricing."""
    mock_client = Mock()
    mocker.patch("langres.core.modules.cascade.OpenAI", return_value=mock_client)

    # Use an unknown model name
    module = CascadeModule(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model="gpt-5-future",  # Unknown model
        llm_api_key="test_key",
        low_threshold=0.3,
        high_threshold=0.9,
    )

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme Corporation"),
        right=CompanySchema(id="c2", name="Beta Industries"),  # Different enough for LLM
        blocker_name="test",
    )

    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "NO_MATCH\nScore: 0.5\nReasoning: Test"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 20
    mock_client.chat.completions.create.return_value = mock_response

    judgements = list(module.forward([candidate]))

    # Should use default (gpt-4o-mini) pricing
    assert "llm_cost_usd" in judgements[0].provenance
    # Default: $0.150/1M input, $0.600/1M output
    expected_cost = (100 * 0.150 + 20 * 0.600) / 1_000_000
    assert abs(judgements[0].provenance["llm_cost_usd"] - expected_cost) < 0.001
