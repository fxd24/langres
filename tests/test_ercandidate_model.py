"""Tests for ERCandidate model extension with similarity_score field.

This test suite validates that ERCandidate accepts an optional similarity_score
field for ranking evaluation purposes, while maintaining backward compatibility.
"""

import logging

import pytest
from pydantic import ValidationError

from langres.core.models import CompanySchema, ERCandidate

logger = logging.getLogger(__name__)


def test_ercandidate_accepts_similarity_score() -> None:
    """Test that ERCandidate accepts optional similarity_score field.

    This validates that we can create candidates with similarity scores
    for ranking evaluation, which is essential for measuring how well
    true matches are ranked by the blocker.
    """
    left = CompanySchema(id="c1", name="Apple Inc")
    right = CompanySchema(id="c2", name="Apple Incorporated")

    # Create candidate with similarity score
    candidate = ERCandidate(
        left=left,
        right=right,
        blocker_name="test_blocker",
        similarity_score=0.95,
    )

    assert candidate.similarity_score == 0.95
    logger.info(f"Created candidate with similarity_score={candidate.similarity_score}")


def test_ercandidate_similarity_score_is_optional() -> None:
    """Test that similarity_score field is optional for backward compatibility.

    Existing code that creates ERCandidate without similarity_score should
    continue to work. The field defaults to None.
    """
    left = CompanySchema(id="c1", name="Apple Inc")
    right = CompanySchema(id="c2", name="Apple Incorporated")

    # Create candidate without similarity score (backward compatibility)
    candidate = ERCandidate(
        left=left,
        right=right,
        blocker_name="test_blocker",
    )

    assert candidate.similarity_score is None
    logger.info(f"Created candidate without similarity_score (defaults to None)")


def test_ercandidate_similarity_score_range_valid() -> None:
    """Test that similarity_score accepts valid float values in [0, 1].

    The similarity score represents normalized similarity, so it should
    accept any float in the valid range [0.0, 1.0].
    """
    left = CompanySchema(id="c1", name="Apple Inc")
    right = CompanySchema(id="c2", name="Apple Incorporated")

    # Test boundary values
    candidate_zero = ERCandidate(
        left=left,
        right=right,
        blocker_name="test_blocker",
        similarity_score=0.0,
    )
    assert candidate_zero.similarity_score == 0.0

    candidate_one = ERCandidate(
        left=left,
        right=right,
        blocker_name="test_blocker",
        similarity_score=1.0,
    )
    assert candidate_one.similarity_score == 1.0

    # Test middle value
    candidate_mid = ERCandidate(
        left=left,
        right=right,
        blocker_name="test_blocker",
        similarity_score=0.5,
    )
    assert candidate_mid.similarity_score == 0.5

    logger.info("Validated similarity_score accepts values in [0.0, 1.0]")


def test_ercandidate_similarity_score_invalid_range() -> None:
    """Test that similarity_score rejects values outside [0, 1].

    Since similarity scores represent normalized similarity, values
    outside [0, 1] should raise a validation error.
    """
    left = CompanySchema(id="c1", name="Apple Inc")
    right = CompanySchema(id="c2", name="Apple Incorporated")

    # Test value > 1
    with pytest.raises(ValidationError) as exc_info:
        ERCandidate(
            left=left,
            right=right,
            blocker_name="test_blocker",
            similarity_score=1.5,
        )
    assert "less than or equal to 1" in str(exc_info.value).lower()

    # Test value < 0
    with pytest.raises(ValidationError) as exc_info:
        ERCandidate(
            left=left,
            right=right,
            blocker_name="test_blocker",
            similarity_score=-0.1,
        )
    assert "greater than or equal to 0" in str(exc_info.value).lower()

    logger.info("Validated similarity_score rejects invalid ranges")
