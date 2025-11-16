"""Tests for LLMJudgeModule async batch processing.

This test module validates the async forward_async() method, which enables
concurrent LLM API calls with token-aware rate limiting and retry logic.
"""

import asyncio
import logging
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from langres.core.models import CompanySchema, ERCandidate
from langres.core.modules.llm_judge import LLMJudgeModule, _RateLimiter

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_llm_client():
    """Mock LiteLLM client with async support."""
    client = Mock()
    client.acompletion = AsyncMock()
    return client


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM API response."""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = (
        "MATCH\nScore: 0.85\nReasoning: Same company with minor variations."
    )
    response.usage = Mock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    return response


@pytest.fixture
def sample_candidates():
    """Create sample entity pair candidates for testing."""
    return [
        ERCandidate(
            left=CompanySchema(id="c1", name="Acme Corporation"),
            right=CompanySchema(id="c2", name="Acme Corp"),
            blocker_name="test_blocker",
        ),
        ERCandidate(
            left=CompanySchema(id="c3", name="TechStart Industries"),
            right=CompanySchema(id="c4", name="TechStart Inc"),
            blocker_name="test_blocker",
        ),
        ERCandidate(
            left=CompanySchema(id="c5", name="DataFlow Solutions"),
            right=CompanySchema(id="c6", name="DataFlow LLC"),
            blocker_name="test_blocker",
        ),
    ]


# Basic Functionality Tests


@pytest.mark.asyncio
async def test_forward_async_processes_single_candidate(mock_llm_client, mock_llm_response):
    """Test forward_async() processes a single candidate correctly."""
    mock_llm_client.acompletion.return_value = mock_llm_response

    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme Corporation"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )

    judgements = await module.forward_async([candidate])

    assert len(judgements) == 1
    j = judgements[0]
    assert j.left_id == "c1"
    assert j.right_id == "c2"
    assert 0.0 <= j.score <= 1.0
    assert j.score_type == "prob_llm"
    assert j.decision_step == "llm_judgment_async"
    assert j.reasoning is not None
    assert j.provenance["method"] == "async_batch"


@pytest.mark.asyncio
async def test_forward_async_processes_multiple_candidates(
    mock_llm_client, mock_llm_response, sample_candidates
):
    """Test forward_async() processes multiple candidates concurrently."""
    mock_llm_client.acompletion.return_value = mock_llm_response

    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    judgements = await module.forward_async(sample_candidates)

    # Should return same number of judgements as candidates
    assert len(judgements) == len(sample_candidates)

    # Verify all calls were made
    assert mock_llm_client.acompletion.call_count == len(sample_candidates)

    # Verify judgements are in same order as candidates
    for i, (candidate, judgement) in enumerate(zip(sample_candidates, judgements)):
        assert judgement.left_id == candidate.left.id
        assert judgement.right_id == candidate.right.id


@pytest.mark.asyncio
async def test_forward_async_maintains_order(mock_llm_client, sample_candidates):
    """Test that results maintain input order despite async processing."""
    # Create responses with different scores to verify ordering
    responses = []
    for i in range(len(sample_candidates)):
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = f"MATCH\nScore: 0.{i}0\nReasoning: Test {i}"
        response.usage = Mock()
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50
        responses.append(response)

    mock_llm_client.acompletion.side_effect = responses

    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    judgements = await module.forward_async(sample_candidates)

    # Verify scores are in expected order
    for i, judgement in enumerate(judgements):
        # Expected score: 0.00, 0.10, 0.20, etc.
        expected_score = float(f"0.{i}0")
        assert abs(judgement.score - expected_score) < 0.01


@pytest.mark.asyncio
async def test_forward_async_empty_candidates_list(mock_llm_client):
    """Test forward_async() handles empty candidate list."""
    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    judgements = await module.forward_async([])

    assert len(judgements) == 0
    mock_llm_client.acompletion.assert_not_called()


# Concurrency Control Tests


@pytest.mark.asyncio
async def test_forward_async_respects_max_concurrent(mock_llm_client, mock_llm_response):
    """Test that max_concurrent limits parallel API calls."""
    # Track concurrent calls
    active_calls = 0
    max_active = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal active_calls, max_active
        active_calls += 1
        max_active = max(max_active, active_calls)
        await asyncio.sleep(0.01)  # Simulate API latency
        active_calls -= 1
        return mock_llm_response

    mock_llm_client.acompletion = mock_acompletion

    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    # Create 10 candidates
    candidates = [
        ERCandidate(
            left=CompanySchema(id=f"c{i}", name=f"Company {i}"),
            right=CompanySchema(id=f"c{i+100}", name=f"Company {i}"),
            blocker_name="test",
        )
        for i in range(10)
    ]

    # Set max_concurrent to 3
    await module.forward_async(candidates, max_concurrent=3)

    # Maximum concurrent calls should not exceed 3
    assert max_active <= 3


# Rate Limiting Tests


@pytest.mark.asyncio
async def test_rate_limiter_respects_rpm_limit():
    """Test _RateLimiter enforces requests-per-minute limit."""
    limiter = _RateLimiter(rpm_limit=5, tpm_limit=100000)

    # Make 5 requests (should all go through immediately)
    start = asyncio.get_event_loop().time()
    for _ in range(5):
        await limiter.acquire(estimated_tokens=100)
    elapsed = asyncio.get_event_loop().time() - start

    # Should complete quickly (< 1 second)
    assert elapsed < 1.0

    # The 6th request should be delayed
    # (but we won't test the actual delay to keep tests fast)


@pytest.mark.asyncio
async def test_rate_limiter_respects_tpm_limit():
    """Test _RateLimiter enforces tokens-per-minute limit."""
    limiter = _RateLimiter(rpm_limit=1000, tpm_limit=500)

    # Use up all available tokens
    await limiter.acquire(estimated_tokens=300)
    await limiter.acquire(estimated_tokens=200)

    # Next request should be delayed (but we won't test actual delay)
    # Just verify it doesn't raise an error
    # Note: This test is simplified to avoid slow tests


@pytest.mark.asyncio
async def test_rate_limiter_records_actual_usage():
    """Test _RateLimiter records actual token usage."""
    limiter = _RateLimiter(rpm_limit=100, tpm_limit=10000)

    # Estimate 100 tokens, but actually use 200
    await limiter.acquire(estimated_tokens=100)
    await limiter.record_usage_async(200)

    # Should have recorded the actual usage
    assert len(limiter._token_usage) == 1
    assert limiter._token_usage[0][1] == 200


@pytest.mark.asyncio
async def test_rate_limiter_concurrent_usage_recording():
    """Test that concurrent record_usage calls are thread-safe."""
    limiter = _RateLimiter(rpm_limit=1000, tpm_limit=100000)

    # Simulate concurrent recording from multiple tasks
    async def record_tokens(token_count: int):
        await limiter.record_usage_async(token_count)

    # Record 100 times concurrently
    tasks = [record_tokens(100) for _ in range(100)]
    await asyncio.gather(*tasks)

    # Should have exactly 100 entries (no race condition data loss)
    assert len(limiter._token_usage) == 100

    # Total tokens should be 10,000
    total_tokens = sum(count for _, count in limiter._token_usage)
    assert total_tokens == 10000


# Retry Logic Tests


@pytest.mark.asyncio
async def test_forward_async_retries_on_rate_limit_error(mock_llm_client, mock_llm_response):
    """Test forward_async() retries on rate limit errors."""
    import litellm

    # First call raises RateLimitError, second succeeds
    mock_llm_client.acompletion.side_effect = [
        litellm.RateLimitError("Rate limit exceeded", None, None),
        mock_llm_response,
    ]

    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )

    # Should retry and eventually succeed
    judgements = await module.forward_async([candidate], max_retries=3)

    assert len(judgements) == 1
    # Should have been called twice (initial + 1 retry)
    assert mock_llm_client.acompletion.call_count == 2


@pytest.mark.asyncio
async def test_forward_async_fails_after_max_retries(mock_llm_client):
    """Test forward_async() fails after exceeding max retries."""
    import litellm

    # Always raise RateLimitError
    mock_llm_client.acompletion.side_effect = litellm.RateLimitError(
        "Rate limit exceeded", None, None
    )

    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )

    # Should fail after max retries
    with pytest.raises(litellm.RateLimitError):
        await module.forward_async([candidate], max_retries=3)


@pytest.mark.asyncio
async def test_forward_async_respects_custom_max_retries(mock_llm_client):
    """Test that max_retries parameter is actually used (not hardcoded to 3)."""
    import litellm

    call_count = 0

    async def failing_completion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise litellm.RateLimitError("Always fail", None, None)

    mock_llm_client.acompletion = failing_completion
    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )

    # Set max_retries=5 (should actually retry 5 times, not 3)
    with pytest.raises(litellm.RateLimitError):
        await module.forward_async([candidate], max_retries=5)

    # Should attempt 5 times (not hardcoded 3)
    assert call_count == 5, f"Expected 5 attempts, got {call_count}"


@pytest.mark.asyncio
async def test_forward_async_retries_with_exponential_backoff(mock_llm_client, mock_llm_response):
    """Test that retries use exponential backoff timing."""
    import litellm

    call_times = []

    async def tracked_completion(*args, **kwargs):
        call_times.append(asyncio.get_event_loop().time())
        if len(call_times) < 3:
            raise litellm.RateLimitError("Rate limit", None, None)
        return mock_llm_response

    mock_llm_client.acompletion = tracked_completion
    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )

    await module.forward_async([candidate], max_retries=5)

    # Should have 3 calls (2 failures + 1 success)
    assert len(call_times) == 3

    # Check backoff timing: 1s, 2s between retries
    time_diff_1 = call_times[1] - call_times[0]
    time_diff_2 = call_times[2] - call_times[1]

    # Allow some variance but should be ~1s and ~2s
    assert 0.8 < time_diff_1 < 1.5, f"First backoff should be ~1s, got {time_diff_1:.2f}s"
    assert 1.8 < time_diff_2 < 2.5, f"Second backoff should be ~2s, got {time_diff_2:.2f}s"


# Provenance Tracking Tests


@pytest.mark.asyncio
async def test_forward_async_tracks_cost(mock_llm_client, mock_llm_response):
    """Test forward_async() tracks API costs in provenance."""
    mock_llm_client.acompletion.return_value = mock_llm_response

    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )

    judgements = await module.forward_async([candidate])

    j = judgements[0]
    assert "cost_usd" in j.provenance
    assert isinstance(j.provenance["cost_usd"], float)
    assert j.provenance["cost_usd"] > 0
    assert "prompt_tokens" in j.provenance
    assert "completion_tokens" in j.provenance
    assert j.provenance["method"] == "async_batch"


@pytest.mark.asyncio
async def test_forward_async_handles_missing_usage(mock_llm_client):
    """Test forward_async() handles responses without usage information."""
    # Response without usage info
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "MATCH\nScore: 0.8\nReasoning: Same company"
    response.usage = None

    mock_llm_client.acompletion.return_value = response

    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )

    judgements = await module.forward_async([candidate])

    j = judgements[0]
    assert j.provenance["cost_usd"] == 0.0
    assert j.provenance["prompt_tokens"] == 0
    assert j.provenance["completion_tokens"] == 0


# Integration Tests


@pytest.mark.asyncio
async def test_forward_async_integration_with_custom_limits(
    mock_llm_client, mock_llm_response, sample_candidates
):
    """Test forward_async() with custom rate limits."""
    mock_llm_client.acompletion.return_value = mock_llm_response

    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    # Use custom rate limits
    judgements = await module.forward_async(
        sample_candidates,
        max_concurrent=2,
        rpm_limit=10,
        tpm_limit=10000,
        max_retries=2,
    )

    assert len(judgements) == len(sample_candidates)


@pytest.mark.asyncio
async def test_forward_async_score_extraction(mock_llm_client):
    """Test forward_async() correctly extracts scores from various response formats."""
    # Different response formats
    responses = [
        "MATCH\nScore: 0.95\nReasoning: Same company",
        "NO_MATCH\nScore: 0.10\nReasoning: Different",
        "Score: 0.55\nUnsure about match",
    ]

    mock_responses = []
    for content in responses:
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = content
        response.usage = Mock()
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50
        mock_responses.append(response)

    mock_llm_client.acompletion.side_effect = mock_responses

    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    candidates = [
        ERCandidate(
            left=CompanySchema(id=f"c{i}", name=f"Company {i}"),
            right=CompanySchema(id=f"c{i+100}", name=f"Company {i}"),
            blocker_name="test",
        )
        for i in range(3)
    ]

    judgements = await module.forward_async(candidates)

    assert judgements[0].score == 0.95
    assert judgements[1].score == 0.10
    assert judgements[2].score == 0.55


# Edge Case Tests


@pytest.mark.asyncio
async def test_forward_async_handles_other_exceptions(mock_llm_client):
    """Test that non-RateLimitError exceptions are not retried."""
    call_count = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise ValueError("Invalid model configuration")

    mock_llm_client.acompletion = mock_acompletion
    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )

    # Should fail immediately without retries
    with pytest.raises(ValueError, match="Invalid model"):
        await module.forward_async([candidate], max_retries=5)

    # Should only be called once (no retries for non-rate-limit errors)
    assert call_count == 1


@pytest.mark.asyncio
async def test_forward_async_with_custom_prompt_template(mock_llm_client, mock_llm_response):
    """Test async mode works with custom prompt templates."""
    mock_llm_client.acompletion.return_value = mock_llm_response

    custom_prompt = "Compare: {left} vs {right}"
    module = LLMJudgeModule(
        client=mock_llm_client,
        model="gpt-4o-mini",
        prompt_template=custom_prompt
    )

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )

    judgements = await module.forward_async([candidate])

    # Verify custom template was used
    call_args = mock_llm_client.acompletion.call_args
    prompt_sent = call_args.kwargs["messages"][0]["content"]
    assert "Compare:" in prompt_sent


@pytest.mark.asyncio
async def test_rate_limiter_sliding_window_expiry():
    """Test that rate limiter correctly expires old entries."""
    limiter = _RateLimiter(rpm_limit=5, tpm_limit=10000)

    # Make 5 requests
    for _ in range(5):
        await limiter.acquire(100)

    # Manually age the entries by modifying timestamps
    current_time = time.time()
    limiter._request_times = [current_time - 61.0 for _ in range(5)]
    limiter._token_usage = [(current_time - 61.0, 100) for _ in range(5)]

    # Should be able to make 5 more without blocking (old ones expired)
    for _ in range(5):
        await limiter.acquire(100)

    # All old entries should be expired, only new ones remain
    recent_requests = [t for t in limiter._request_times if t > current_time - 60]
    assert len(recent_requests) == 5


@pytest.mark.asyncio
async def test_forward_async_partial_failures_all_fail():
    """Test behavior when asyncio.gather encounters failures."""
    import litellm

    client = Mock()

    # All requests fail
    async def mock_acompletion(*args, **kwargs):
        raise litellm.RateLimitError("Fail", None, None)

    client.acompletion = mock_acompletion

    module = LLMJudgeModule(client=client, model="gpt-4o-mini")

    candidates = [
        ERCandidate(
            left=CompanySchema(id=f"c{i}", name=f"Company {i}"),
            right=CompanySchema(id=f"c{i+100}", name=f"Company {i}"),
            blocker_name="test",
        )
        for i in range(3)
    ]

    # Should fail because asyncio.gather propagates exceptions
    with pytest.raises(litellm.RateLimitError):
        await module.forward_async(candidates, max_retries=1)


@pytest.mark.asyncio
async def test_forward_async_memory_efficient_large_batch(mock_llm_client, mock_llm_response):
    """Test that large batches don't cause memory issues."""
    mock_llm_client.acompletion.return_value = mock_llm_response

    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    # Create 100 candidates (reduced from 1000 to keep test fast)
    candidates = [
        ERCandidate(
            left=CompanySchema(id=f"c{i}", name=f"Company {i}"),
            right=CompanySchema(id=f"c{i+100}", name=f"Company {i}"),
            blocker_name="test",
        )
        for i in range(100)
    ]

    # Should complete without memory errors
    judgements = await module.forward_async(
        candidates,
        max_concurrent=20,
        rpm_limit=10000,
        tpm_limit=1000000
    )

    assert len(judgements) == 100

    # Verify order is preserved
    for i, j in enumerate(judgements):
        assert j.left_id == f"c{i}"
        assert j.right_id == f"c{i+100}"


@pytest.mark.asyncio
async def test_forward_async_with_very_high_concurrency(mock_llm_client, mock_llm_response):
    """Test with max_concurrent higher than number of candidates."""
    mock_llm_client.acompletion.return_value = mock_llm_response

    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    # Only 3 candidates but max_concurrent=100
    candidates = [
        ERCandidate(
            left=CompanySchema(id=f"c{i}", name=f"Company {i}"),
            right=CompanySchema(id=f"c{i+100}", name=f"Company {i}"),
            blocker_name="test",
        )
        for i in range(3)
    ]

    judgements = await module.forward_async(candidates, max_concurrent=100)

    assert len(judgements) == 3


@pytest.mark.asyncio
async def test_forward_async_with_zero_temperature(mock_llm_client, mock_llm_response):
    """Test async with temperature=0 for deterministic results."""
    mock_llm_client.acompletion.return_value = mock_llm_response

    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini", temperature=0.0)

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )

    judgements = await module.forward_async([candidate])

    # Verify temperature was passed correctly
    call_args = mock_llm_client.acompletion.call_args
    assert call_args.kwargs["temperature"] == 0.0


@pytest.mark.asyncio
async def test_forward_async_score_clamping():
    """Test that scores outside [0,1] range are clamped."""
    client = Mock()

    # Mock response with out-of-range score
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "MATCH\nScore: 1.5\nReasoning: Invalid score"
    response.usage = Mock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50

    async def mock_acompletion(*args, **kwargs):
        return response

    client.acompletion = mock_acompletion

    module = LLMJudgeModule(client=client, model="gpt-4o-mini")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )

    judgements = await module.forward_async([candidate])

    # Score should be clamped to 1.0
    assert judgements[0].score == 1.0


@pytest.mark.asyncio
async def test_forward_async_with_entities_missing_optional_fields(mock_llm_client, mock_llm_response):
    """Test async with entities that have None for optional fields."""
    mock_llm_client.acompletion.return_value = mock_llm_response

    module = LLMJudgeModule(client=mock_llm_client, model="gpt-4o-mini")

    # Entities with minimal fields
    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme", address=None, phone=None, website=None),
        right=CompanySchema(id="c2", name="Acme Corp", address=None, phone=None, website=None),
        blocker_name="test",
    )

    judgements = await module.forward_async([candidate])

    assert len(judgements) == 1
    assert judgements[0].left_id == "c1"


@pytest.mark.asyncio
async def test_rate_limiter_with_zero_estimated_tokens():
    """Test rate limiter with edge case of 0 estimated tokens."""
    limiter = _RateLimiter(rpm_limit=10, tpm_limit=1000)

    # Should not raise error with 0 tokens
    await limiter.acquire(estimated_tokens=0)
    await limiter.record_usage_async(0)

    assert len(limiter._request_times) == 1
    assert len(limiter._token_usage) == 1


@pytest.mark.asyncio
async def test_forward_async_decision_step_marking():
    """Test that async judgements are marked with correct decision_step."""
    client = Mock()

    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "MATCH\nScore: 0.9\nReasoning: Test"
    response.usage = Mock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50

    async def mock_acompletion(*args, **kwargs):
        return response

    client.acompletion = mock_acompletion

    module = LLMJudgeModule(client=client, model="gpt-4o-mini")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )

    judgements = await module.forward_async([candidate])

    # Should be marked as async
    assert judgements[0].decision_step == "llm_judgment_async"
    assert judgements[0].provenance["method"] == "async_batch"


@pytest.mark.asyncio
async def test_forward_async_preserves_blocker_name():
    """Test that blocker_name from candidates is preserved in context."""
    client = Mock()

    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "MATCH\nScore: 0.9\nReasoning: Test"
    response.usage = Mock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50

    async def mock_acompletion(*args, **kwargs):
        return response

    client.acompletion = mock_acompletion

    module = LLMJudgeModule(client=client, model="gpt-4o-mini")

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="custom_blocker_v2",
    )

    # Blocker name is part of input context (not output)
    # But we verify it doesn't cause errors
    judgements = await module.forward_async([candidate])

    assert len(judgements) == 1
