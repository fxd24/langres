"""Tests for AllPairsBlocker.inspect_candidates() method.

Tests the inspection capabilities of AllPairsBlocker, which generates
all possible entity pairs (n*(n-1)/2 candidates).
"""

import pytest
from pydantic import BaseModel

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.models import CompanySchema, ERCandidate
from langres.core.reports import CandidateInspectionReport


class SimpleEntity(BaseModel):
    """Simple entity for testing."""

    id: str
    name: str


def test_inspect_small_dataset():
    """Test inspection with small dataset (n=5) shows AllPairs is appropriate."""
    # Given: 5 entities
    entities = [SimpleEntity(id=str(i), name=f"Entity {i}") for i in range(5)]
    blocker = AllPairsBlocker(schema_factory=lambda e: e)

    # When: Generate candidates
    candidates = list(blocker.generate_candidates(entities))

    # Then: Inspect should show appropriate usage
    report = blocker.inspect_candidates(candidates, entities, sample_size=5)

    # Verify statistics
    assert report.total_candidates == 10  # 5*4/2 = 10
    assert report.avg_candidates_per_entity == 4.0  # n-1 = 4

    # Verify recommendation suggests AllPairs is appropriate
    assert len(report.recommendations) > 0
    assert "✅ AllPairsBlocker appropriate" in report.recommendations[0]

    # Verify distribution (all entities have exactly 4 candidates)
    assert "4" in report.candidate_distribution
    assert report.candidate_distribution["4"] == 5


def test_inspect_medium_dataset():
    """Test inspection with medium dataset (n=50) suggests considering VectorBlocker."""
    # Given: 50 entities
    entities = [SimpleEntity(id=str(i), name=f"Entity {i}") for i in range(50)]
    blocker = AllPairsBlocker(schema_factory=lambda e: e)

    # When: Generate candidates
    candidates = list(blocker.generate_candidates(entities))

    # Then: Inspect should suggest VectorBlocker
    report = blocker.inspect_candidates(candidates, entities, sample_size=10)

    # Verify statistics
    assert report.total_candidates == 1225  # 50*49/2 = 1225
    assert report.avg_candidates_per_entity == 49.0  # n-1 = 49

    # Verify recommendation mentions VectorBlocker
    assert len(report.recommendations) > 0
    assert "VectorBlocker" in report.recommendations[0]
    assert "feasible" in report.recommendations[0].lower()


def test_inspect_large_dataset():
    """Test inspection with large dataset (n=150) warns about scalability."""
    # Given: 150 entities
    entities = [SimpleEntity(id=str(i), name=f"Entity {i}") for i in range(150)]
    blocker = AllPairsBlocker(schema_factory=lambda e: e)

    # When: Generate candidates
    candidates = list(blocker.generate_candidates(entities))

    # Then: Inspect should warn about scalability
    report = blocker.inspect_candidates(candidates, entities, sample_size=10)

    # Verify statistics
    assert report.total_candidates == 11175  # 150*149/2 = 11175
    assert report.avg_candidates_per_entity == 149.0  # n-1 = 149

    # Verify recommendation warns about scalability
    assert len(report.recommendations) > 0
    assert "⚠️" in report.recommendations[0]
    assert "not scalable" in report.recommendations[0]


def test_inspect_sample_extraction():
    """Test that sample_size is respected and readable text is extracted."""
    # Given: 10 entities with names
    entities = [SimpleEntity(id=str(i), name=f"Company {chr(65 + i)}") for i in range(10)]
    blocker = AllPairsBlocker(schema_factory=lambda e: e)

    # When: Generate candidates and inspect with sample_size=5
    candidates = list(blocker.generate_candidates(entities))
    report = blocker.inspect_candidates(candidates, entities, sample_size=5)

    # Then: Should have exactly 5 examples
    assert len(report.examples) == 5

    # Verify examples contain readable text
    for example in report.examples:
        assert "left_id" in example
        assert "right_id" in example
        assert "left_text" in example
        assert "right_text" in example
        # Text should be entity name, not just ID
        assert "Company" in example["left_text"]
        assert "Company" in example["right_text"]


def test_inspect_distribution_uniformity():
    """Test that distribution shows uniformity (all entities have n-1 candidates)."""
    # Given: 8 entities
    entities = [SimpleEntity(id=str(i), name=f"Entity {i}") for i in range(8)]
    blocker = AllPairsBlocker(schema_factory=lambda e: e)

    # When: Generate candidates
    candidates = list(blocker.generate_candidates(entities))
    report = blocker.inspect_candidates(candidates, entities)

    # Then: Distribution should show all 8 entities have exactly 7 candidates
    assert len(report.candidate_distribution) == 1  # Only one bucket
    assert "7" in report.candidate_distribution
    assert report.candidate_distribution["7"] == 8


def test_inspect_empty_entity_list():
    """Test inspection with empty entity list."""
    # Given: Empty entity list
    entities = []
    blocker = AllPairsBlocker(schema_factory=lambda e: e)

    # When: Generate candidates (none)
    candidates = list(blocker.generate_candidates(entities))
    report = blocker.inspect_candidates(candidates, entities)

    # Then: Should handle gracefully
    assert report.total_candidates == 0
    assert report.avg_candidates_per_entity == 0.0
    assert len(report.examples) == 0
    assert report.candidate_distribution == {}


def test_inspect_single_entity():
    """Test inspection with single entity (0 pairs)."""
    # Given: Single entity
    entities = [SimpleEntity(id="1", name="Only Entity")]
    blocker = AllPairsBlocker(schema_factory=lambda e: e)

    # When: Generate candidates (none, can't pair with self)
    candidates = list(blocker.generate_candidates(entities))
    report = blocker.inspect_candidates(candidates, entities)

    # Then: Should show 0 pairs
    assert report.total_candidates == 0
    assert report.avg_candidates_per_entity == 0.0
    # Distribution should show 1 entity with 0 candidates
    assert "0" in report.candidate_distribution
    assert report.candidate_distribution["0"] == 1


def test_inspect_two_entities():
    """Test inspection with two entities (1 pair)."""
    # Given: Two entities
    entities = [
        SimpleEntity(id="1", name="Entity A"),
        SimpleEntity(id="2", name="Entity B"),
    ]
    blocker = AllPairsBlocker(schema_factory=lambda e: e)

    # When: Generate candidates (1 pair)
    candidates = list(blocker.generate_candidates(entities))
    report = blocker.inspect_candidates(candidates, entities)

    # Then: Should show 1 pair
    assert report.total_candidates == 1
    assert report.avg_candidates_per_entity == 1.0
    # Both entities have 1 candidate each
    assert "1" in report.candidate_distribution
    assert report.candidate_distribution["1"] == 2


def test_inspect_to_markdown():
    """Test to_markdown() returns formatted string."""
    # Given: Entities and candidates
    entities = [SimpleEntity(id=str(i), name=f"Entity {i}") for i in range(5)]
    blocker = AllPairsBlocker(schema_factory=lambda e: e)
    candidates = list(blocker.generate_candidates(entities))
    report = blocker.inspect_candidates(candidates, entities)

    # When: Convert to markdown
    markdown = report.to_markdown()

    # Then: Should contain key information
    assert isinstance(markdown, str)
    assert "Candidate Inspection Report" in markdown
    assert "Total candidates: 10" in markdown
    assert "Average candidates per entity: 4.0" in markdown
    assert "Distribution" in markdown
    assert "Recommendations" in markdown


def test_inspect_to_dict():
    """Test to_dict() returns valid JSON-serializable dict."""
    # Given: Entities and candidates
    entities = [SimpleEntity(id=str(i), name=f"Entity {i}") for i in range(5)]
    blocker = AllPairsBlocker(schema_factory=lambda e: e)
    candidates = list(blocker.generate_candidates(entities))
    report = blocker.inspect_candidates(candidates, entities)

    # When: Convert to dict
    data = report.to_dict()

    # Then: Should contain all fields
    assert isinstance(data, dict)
    assert data["total_candidates"] == 10
    assert data["avg_candidates_per_entity"] == 4.0
    assert "candidate_distribution" in data
    assert "examples" in data
    assert "recommendations" in data


def test_inspect_stats_property():
    """Test stats property contains only numerical data."""
    # Given: Entities and candidates
    entities = [SimpleEntity(id=str(i), name=f"Entity {i}") for i in range(5)]
    blocker = AllPairsBlocker(schema_factory=lambda e: e)
    candidates = list(blocker.generate_candidates(entities))
    report = blocker.inspect_candidates(candidates, entities)

    # When: Access stats
    stats = report.stats

    # Then: Should contain only numerical metrics
    assert isinstance(stats, dict)
    assert stats["total_candidates"] == 10
    assert stats["avg_candidates_per_entity"] == 4.0
    # Stats should not include examples or recommendations
    assert "examples" not in stats
    assert "recommendations" not in stats


def test_inspect_with_company_schema():
    """Test inspection with CompanySchema entities (real-world schema)."""
    # Given: CompanySchema entities
    entities = [
        CompanySchema(id=str(i), name=f"Company {chr(65 + i)}", country="US") for i in range(5)
    ]
    blocker = AllPairsBlocker(schema_factory=lambda e: e)

    # When: Generate candidates
    candidates = list(blocker.generate_candidates(entities))
    report = blocker.inspect_candidates(candidates, entities, sample_size=3)

    # Then: Should extract company names
    assert len(report.examples) == 3
    for example in report.examples:
        # Text should be company name
        assert "Company" in example["left_text"]
        assert "Company" in example["right_text"]

    # Verify statistics
    assert report.total_candidates == 10
    assert report.avg_candidates_per_entity == 4.0
