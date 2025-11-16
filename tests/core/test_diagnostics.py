"""Tests for diagnostic models."""

import pytest

from langres.core.diagnostics import (
    DiagnosticExamples,
    FalsePositiveExample,
    MissedMatchExample,
)


def test_missed_match_example_creation():
    """Test MissedMatchExample model creation."""
    example = MissedMatchExample(
        left_id="e1",
        left_text="Acme Corp",
        right_id="e2",
        right_text="Acme Corporation",
        cluster_id=0,
    )

    assert example.left_id == "e1"
    assert example.left_text == "Acme Corp"
    assert example.right_id == "e2"
    assert example.right_text == "Acme Corporation"
    assert example.cluster_id == 0
    assert example.baseline_similarity is None


def test_missed_match_example_with_baseline_similarity():
    """Test MissedMatchExample with baseline similarity."""
    example = MissedMatchExample(
        left_id="e1",
        left_text="Acme Corp",
        right_id="e2",
        right_text="Acme Corporation",
        cluster_id=0,
        baseline_similarity=0.85,
    )

    assert example.baseline_similarity == 0.85


def test_false_positive_example_creation():
    """Test FalsePositiveExample model creation."""
    example = FalsePositiveExample(
        left_id="e1",
        left_text="Apple Inc",
        right_id="e3",
        right_text="Apple Fruit",
        score=0.92,
    )

    assert example.left_id == "e1"
    assert example.left_text == "Apple Inc"
    assert example.right_id == "e3"
    assert example.right_text == "Apple Fruit"
    assert example.score == 0.92


def test_diagnostic_examples_empty():
    """Test DiagnosticExamples with no examples."""
    examples = DiagnosticExamples()

    assert examples.missed_matches == []
    assert examples.false_positives == []


def test_diagnostic_examples_with_data():
    """Test DiagnosticExamples with sample data."""
    missed = [
        MissedMatchExample(
            left_id="e1",
            left_text="Acme Corp",
            right_id="e2",
            right_text="Acme Corporation",
            cluster_id=0,
        ),
    ]

    false_pos = [
        FalsePositiveExample(
            left_id="e1",
            left_text="Apple Inc",
            right_id="e3",
            right_text="Apple Fruit",
            score=0.92,
        ),
    ]

    examples = DiagnosticExamples(
        missed_matches=missed,
        false_positives=false_pos,
    )

    assert len(examples.missed_matches) == 1
    assert len(examples.false_positives) == 1


def test_diagnostic_examples_to_markdown():
    """Test DiagnosticExamples markdown generation."""
    examples = DiagnosticExamples(
        missed_matches=[
            MissedMatchExample(
                left_id="e1",
                left_text="Acme Corp",
                right_id="e2",
                right_text="Acme Corporation",
                cluster_id=0,
                baseline_similarity=0.85,
            ),
        ],
        false_positives=[
            FalsePositiveExample(
                left_id="e1",
                left_text="Apple Inc",
                right_id="e3",
                right_text="Apple Fruit",
                score=0.92,
            ),
        ],
    )

    md = examples.to_markdown()

    # Check structure
    assert "# Diagnostic Examples" in md
    assert "## Missed Matches" in md
    assert "## False Positives" in md

    # Check content
    assert "Acme Corp" in md
    assert "Acme Corporation" in md
    assert "Cluster: 0" in md
    assert "Baseline similarity: 0.85" in md
    assert "Apple Inc" in md
    assert "Apple Fruit" in md
    assert "Score: 0.92" in md


def test_diagnostic_examples_to_markdown_empty():
    """Test markdown generation with no examples."""
    examples = DiagnosticExamples()
    md = examples.to_markdown()

    assert "No missed matches" in md
    assert "No false positives" in md


def test_diagnostic_examples_serialization():
    """Test Pydantic serialization."""
    examples = DiagnosticExamples(
        missed_matches=[
            MissedMatchExample(
                left_id="e1",
                left_text="Acme Corp",
                right_id="e2",
                right_text="Acme Corporation",
                cluster_id=0,
            ),
        ],
    )

    # Test dict export
    data = examples.model_dump()
    assert "missed_matches" in data
    assert "false_positives" in data
    assert len(data["missed_matches"]) == 1

    # Test JSON export
    json_str = examples.model_dump_json()
    assert "Acme Corp" in json_str
