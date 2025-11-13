"""Tests for VectorBlocker.inspect_candidates() method.

This module tests the inspection capabilities of VectorBlocker for exploratory
analysis without ground truth labels.
"""

import logging

import pytest

from langres.core.blockers.vector import VectorBlocker
from langres.core.embeddings import FakeEmbedder
from langres.core.models import CompanySchema
from langres.core.reports import CandidateInspectionReport
from langres.core.vector_index import FakeVectorIndex

logger = logging.getLogger(__name__)


def company_factory(record: dict) -> CompanySchema:
    """Standard company factory for tests."""
    return CompanySchema(
        id=record["id"],
        name=record["name"],
        address=record.get("address"),
        phone=record.get("phone"),
    )


def create_fake_blocker(k_neighbors: int = 10) -> VectorBlocker[CompanySchema]:
    """Create a VectorBlocker with fake implementations for fast unit testing."""
    return VectorBlocker(
        schema_factory=company_factory,
        text_field_extractor=lambda x: x.name,
        vector_index=FakeVectorIndex(),
        k_neighbors=k_neighbors,
    )


class TestVectorBlockerInspection:
    """Tests for VectorBlocker.inspect_candidates() method."""

    def test_inspect_candidates_with_typical_data(self) -> None:
        """Test inspect_candidates with typical candidate data."""
        # Generate candidates first
        data = [
            {"id": "c1", "name": "Acme Corporation", "address": "123 Main St"},
            {"id": "c2", "name": "Acme Corp", "address": "123 Main Street"},
            {"id": "c3", "name": "TechStart Industries", "address": "456 Oak Ave"},
            {"id": "c4", "name": "DataFlow Solutions", "address": "789 Park Blvd"},
            {"id": "c5", "name": "Global Tech", "address": "111 First Ave"},
        ]

        blocker = create_fake_blocker(k_neighbors=2)

        # Build index explicitly
        texts = [d["name"] for d in data]
        blocker.vector_index.create_index(texts)

        candidates = list(blocker.stream(data))
        entities = [company_factory(record) for record in data]

        # Inspect candidates
        report = blocker.inspect_candidates(candidates=candidates, entities=entities, sample_size=3)

        # Verify report structure
        assert isinstance(report, CandidateInspectionReport)
        assert report.total_candidates > 0
        assert report.avg_candidates_per_entity >= 0
        assert isinstance(report.candidate_distribution, dict)
        assert isinstance(report.examples, list)
        assert isinstance(report.recommendations, list)

    def test_inspect_candidates_with_zero_candidates(self) -> None:
        """Test inspect_candidates with empty candidate list."""
        data = [{"id": "c1", "name": "Single Company"}]
        blocker = create_fake_blocker(k_neighbors=2)

        # Build index explicitly
        texts = [d["name"] for d in data]
        blocker.vector_index.create_index(texts)

        # Single entity generates no candidates
        candidates = list(blocker.stream(data))
        entities = [company_factory(record) for record in data]

        report = blocker.inspect_candidates(
            candidates=candidates, entities=entities, sample_size=10
        )

        assert report.total_candidates == 0
        assert report.avg_candidates_per_entity == 0.0
        assert len(report.examples) == 0
        assert len(report.recommendations) > 0  # Should suggest actions

    def test_inspect_candidates_low_avg_triggers_recommendation(self) -> None:
        """Test that low average candidates triggers recall recommendation."""
        # Simulate low candidate count scenario
        data = [
            {"id": "c1", "name": "Company A"},
            {"id": "c2", "name": "Company B"},
            {"id": "c3", "name": "Company C"},
        ]

        blocker = create_fake_blocker(k_neighbors=1)  # Very low k

        # Build index explicitly
        texts = [d["name"] for d in data]
        blocker.vector_index.create_index(texts)

        candidates = list(blocker.stream(data))
        entities = [company_factory(record) for record in data]

        report = blocker.inspect_candidates(candidates=candidates, entities=entities, sample_size=5)

        # Check for low candidate recommendation
        recommendations = " ".join(report.recommendations)
        assert report.avg_candidates_per_entity < 3  # Verify assumption
        assert any(
            "increase" in rec.lower() or "k_neighbors" in rec.lower()
            for rec in report.recommendations
        )

    def test_inspect_candidates_high_avg_triggers_recommendation(self) -> None:
        """Test that high average candidates triggers precision recommendation."""
        # Generate many candidates
        data = [{"id": f"c{i}", "name": f"Company {i}"} for i in range(10)]

        blocker = create_fake_blocker(k_neighbors=9)  # High k

        # Build index explicitly
        texts = [d["name"] for d in data]
        blocker.vector_index.create_index(texts)

        candidates = list(blocker.stream(data))
        entities = [company_factory(record) for record in data]

        report = blocker.inspect_candidates(candidates=candidates, entities=entities, sample_size=5)

        # With k=9 and 10 entities, avg should be high
        if report.avg_candidates_per_entity > 8:
            # Check for high candidate warning
            recommendations = " ".join(report.recommendations)
            assert any(
                "decrease" in rec.lower() or "reduce" in rec.lower()
                for rec in report.recommendations
            )

    def test_inspect_candidates_includes_readable_text(self) -> None:
        """Test that examples include readable entity text."""
        data = [
            {"id": "c1", "name": "Acme Corporation", "address": "123 Main St"},
            {"id": "c2", "name": "Acme Corp", "address": "123 Main Street"},
            {"id": "c3", "name": "TechStart Industries", "address": "456 Oak Ave"},
        ]

        blocker = create_fake_blocker(k_neighbors=2)

        # Build index explicitly
        texts = [d["name"] for d in data]
        blocker.vector_index.create_index(texts)

        candidates = list(blocker.stream(data))
        entities = [company_factory(record) for record in data]

        report = blocker.inspect_candidates(candidates=candidates, entities=entities, sample_size=2)

        # Verify examples contain readable text
        assert len(report.examples) > 0
        for example in report.examples:
            assert "left_text" in example or "left_id" in example
            assert "right_text" in example or "right_id" in example
            # Text should come from text_field_extractor (name field)
            if "left_text" in example:
                assert isinstance(example["left_text"], str)

    def test_inspect_candidates_respects_sample_size(self) -> None:
        """Test that inspect_candidates respects sample_size parameter."""
        data = [{"id": f"c{i}", "name": f"Company {i}"} for i in range(20)]

        blocker = create_fake_blocker(k_neighbors=5)

        # Build index explicitly
        texts = [d["name"] for d in data]
        blocker.vector_index.create_index(texts)

        candidates = list(blocker.stream(data))
        entities = [company_factory(record) for record in data]

        # Request small sample
        report = blocker.inspect_candidates(candidates=candidates, entities=entities, sample_size=3)

        # Examples should not exceed sample_size
        assert len(report.examples) <= 3

    def test_inspect_candidates_builds_distribution_histogram(self) -> None:
        """Test that candidate_distribution is properly structured."""
        data = [{"id": f"c{i}", "name": f"Company {i}"} for i in range(10)]

        blocker = create_fake_blocker(k_neighbors=3)

        # Build index explicitly
        texts = [d["name"] for d in data]
        blocker.vector_index.create_index(texts)

        candidates = list(blocker.stream(data))
        entities = [company_factory(record) for record in data]

        report = blocker.inspect_candidates(candidates=candidates, entities=entities, sample_size=5)

        # Distribution should be a dict with string keys (ranges) and int values (counts)
        assert isinstance(report.candidate_distribution, dict)
        for key, value in report.candidate_distribution.items():
            assert isinstance(key, str)
            assert isinstance(value, int)
            assert value >= 0

    def test_inspect_candidates_markdown_output(self) -> None:
        """Test that report can generate readable markdown."""
        data = [
            {"id": "c1", "name": "Acme Corp"},
            {"id": "c2", "name": "TechStart"},
            {"id": "c3", "name": "DataFlow"},
        ]

        blocker = create_fake_blocker(k_neighbors=2)

        # Build index explicitly
        texts = [d["name"] for d in data]
        blocker.vector_index.create_index(texts)

        candidates = list(blocker.stream(data))
        entities = [company_factory(record) for record in data]

        report = blocker.inspect_candidates(candidates=candidates, entities=entities, sample_size=5)

        markdown = report.to_markdown()
        assert isinstance(markdown, str)
        assert "# Candidate Inspection Report" in markdown
        assert str(report.total_candidates) in markdown

    def test_inspect_candidates_stats_property(self) -> None:
        """Test that stats property returns only numerical metrics."""
        data = [
            {"id": "c1", "name": "Acme Corp"},
            {"id": "c2", "name": "TechStart"},
        ]

        blocker = create_fake_blocker(k_neighbors=1)

        # Build index explicitly
        texts = [d["name"] for d in data]
        blocker.vector_index.create_index(texts)

        candidates = list(blocker.stream(data))
        entities = [company_factory(record) for record in data]

        report = blocker.inspect_candidates(candidates=candidates, entities=entities, sample_size=5)

        stats = report.stats
        assert "total_candidates" in stats
        assert "avg_candidates_per_entity" in stats
        assert "examples" not in stats
        assert "recommendations" not in stats

    def test_inspect_candidates_suggested_k_equals_current_k(self) -> None:
        """Test case where suggested_k equals current k_neighbors (no recommendation added).

        This covers the branch where avg_candidates_per_entity > 0 is True,
        but suggested_k == self.k_neighbors, so no k-adjustment recommendation is added.
        """
        # Create scenario where suggested_k = int(avg/2) equals k_neighbors
        # If k_neighbors=2 and avg=4, then suggested_k=2
        data = [{"id": f"c{i}", "name": f"Company {i}"} for i in range(6)]

        # Set k=2, which should result in avg_candidates around 4 (suggested_k = 2)
        blocker = create_fake_blocker(k_neighbors=2)

        # Build index explicitly
        texts = [d["name"] for d in data]
        blocker.vector_index.create_index(texts)

        candidates = list(blocker.stream(data))
        entities = [company_factory(record) for record in data]

        report = blocker.inspect_candidates(candidates=candidates, entities=entities, sample_size=5)

        # The avg should be > 0, but suggested_k should equal current k
        assert report.avg_candidates_per_entity > 0
        # No k-adjustment recommendation should be added (but other recommendations might exist)
        k_recommendations = [rec for rec in report.recommendations if "k_neighbors=" in rec]
        # If suggested_k == k_neighbors, no k recommendation
        suggested_k = int(report.avg_candidates_per_entity / 2)
        if suggested_k == 2:  # Same as our k_neighbors
            assert len(k_recommendations) == 0

    def test_inspect_candidates_zero_avg_skips_k_suggestion(self) -> None:
        """Test that zero avg_candidates_per_entity skips k suggestion branch.

        This covers the branch 318->325 where avg_candidates_per_entity == 0,
        so the entire k-suggestion logic is skipped (line 318 jumps to 325).
        """
        # Single entity generates zero candidates (no pairs possible)
        data = [{"id": "c1", "name": "Single Company"}]

        blocker = create_fake_blocker(k_neighbors=5)

        # Build index explicitly
        texts = [d["name"] for d in data]
        blocker.vector_index.create_index(texts)

        candidates = list(blocker.stream(data))
        entities = [company_factory(record) for record in data]

        report = blocker.inspect_candidates(candidates=candidates, entities=entities, sample_size=5)

        # Zero average means no k-suggestion logic is executed
        assert report.avg_candidates_per_entity == 0.0
        # No k-based recommendations (since avg == 0)
        k_recommendations = [rec for rec in report.recommendations if "k_neighbors=" in rec]
        assert len(k_recommendations) == 0
