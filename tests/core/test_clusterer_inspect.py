"""Tests for Clusterer.inspect_clusters() method.

This module tests the inspection capabilities of Clusterer for exploratory
analysis without ground truth labels.
"""

import logging

import pytest

from langres.core.clusterer import Clusterer
from langres.core.models import CompanySchema
from langres.core.reports import ClusterInspectionReport

logger = logging.getLogger(__name__)


def create_company(company_id: str, name: str) -> CompanySchema:
    """Create a CompanySchema for testing."""
    return CompanySchema(
        id=company_id,
        name=name,
        address=None,
        phone=None,
    )


class TestClustererInspection:
    """Tests for Clusterer.inspect_clusters() method."""

    def test_inspect_clusters_with_typical_distribution(self) -> None:
        """Test inspect_clusters with normal mix of singletons and clusters."""
        # Mix of cluster sizes: 1, 2, 3, 5
        clusters = [
            {"c1", "c2", "c3", "c4", "c5"},  # Size 5
            {"c6", "c7", "c8"},  # Size 3
            {"c9", "c10"},  # Size 2
            {"c11"},  # Singleton
            {"c12"},  # Singleton
        ]

        entities = [create_company(f"c{i}", f"Company {i}") for i in range(1, 13)]

        clusterer = Clusterer(threshold=0.5)
        report = clusterer.inspect_clusters(clusters=clusters, entities=entities, sample_size=10)

        # Verify report structure
        assert isinstance(report, ClusterInspectionReport)
        assert report.total_clusters == 5
        assert report.singleton_rate > 0  # 2/5 = 40%
        assert isinstance(report.cluster_size_distribution, dict)
        assert isinstance(report.largest_clusters, list)
        assert isinstance(report.recommendations, list)

    def test_inspect_clusters_high_singleton_rate_recommendation(self) -> None:
        """Test that high singleton rate (>70%) triggers threshold lowering recommendation."""
        # 8 singletons out of 10 clusters = 80% singleton rate
        clusters = [
            {"c1", "c2"},  # Size 2
            {"c3", "c4"},  # Size 2
            {"c5"},  # Singleton
            {"c6"},  # Singleton
            {"c7"},  # Singleton
            {"c8"},  # Singleton
            {"c9"},  # Singleton
            {"c10"},  # Singleton
            {"c11"},  # Singleton
            {"c12"},  # Singleton
        ]

        entities = [create_company(f"c{i}", f"Company {i}") for i in range(1, 13)]

        clusterer = Clusterer(threshold=0.7)
        report = clusterer.inspect_clusters(clusters=clusters, entities=entities, sample_size=5)

        assert report.singleton_rate > 70  # Should be 80%
        # Check for threshold lowering recommendation
        recommendations = " ".join(report.recommendations)
        assert "threshold may be too high" in recommendations.lower()
        assert "lowering threshold" in recommendations.lower() or "lower" in recommendations.lower()

    def test_inspect_clusters_low_singleton_rate_recommendation(self) -> None:
        """Test that low singleton rate (<20%) with large clusters triggers threshold raising."""
        # 1 singleton out of 6 clusters = ~16.7% singleton rate
        # Average cluster size = (10 + 8 + 7 + 6 + 5 + 1) / 6 = 6.17 > 5
        clusters = [
            {f"c{i}" for i in range(1, 11)},  # Size 10
            {f"c{i}" for i in range(11, 19)},  # Size 8
            {f"c{i}" for i in range(19, 26)},  # Size 7
            {f"c{i}" for i in range(26, 32)},  # Size 6
            {f"c{i}" for i in range(32, 37)},  # Size 5
            {"c37"},  # Singleton
        ]

        entities = [create_company(f"c{i}", f"Company {i}") for i in range(1, 38)]

        clusterer = Clusterer(threshold=0.3)
        report = clusterer.inspect_clusters(clusters=clusters, entities=entities, sample_size=10)

        assert report.singleton_rate < 20
        # Check for threshold raising recommendation
        recommendations = " ".join(report.recommendations)
        assert "threshold may be too low" in recommendations.lower()
        assert "raising threshold" in recommendations.lower() or "raise" in recommendations.lower()

    def test_inspect_clusters_reasonable_singleton_rate(self) -> None:
        """Test that reasonable singleton rate (20-70%) gets positive feedback."""
        # 3 singletons out of 7 clusters = ~42.9% singleton rate
        clusters = [
            {"c1", "c2", "c3"},  # Size 3
            {"c4", "c5"},  # Size 2
            {"c6", "c7"},  # Size 2
            {"c8", "c9"},  # Size 2
            {"c10"},  # Singleton
            {"c11"},  # Singleton
            {"c12"},  # Singleton
        ]

        entities = [create_company(f"c{i}", f"Company {i}") for i in range(1, 13)]

        clusterer = Clusterer(threshold=0.6)
        report = clusterer.inspect_clusters(clusters=clusters, entities=entities, sample_size=5)

        assert 20 <= report.singleton_rate <= 70
        # Check for reasonable rate acknowledgment
        recommendations = " ".join(report.recommendations)
        assert (
            "looks reasonable" in recommendations.lower()
            or "singleton rate" in recommendations.lower()
        )

    def test_inspect_clusters_very_large_cluster_warning(self) -> None:
        """Test that very large cluster (>20 entities) triggers over-merging warning."""
        # One cluster with 25 entities
        clusters = [
            {f"c{i}" for i in range(1, 26)},  # Size 25
            {"c26", "c27"},  # Size 2
        ]

        entities = [create_company(f"c{i}", f"Company {i}") for i in range(1, 28)]

        clusterer = Clusterer(threshold=0.4)
        report = clusterer.inspect_clusters(clusters=clusters, entities=entities, sample_size=5)

        # Check for large cluster warning
        recommendations = " ".join(report.recommendations)
        assert (
            "very large cluster" in recommendations.lower()
            or "over-merging" in recommendations.lower()
        )
        assert "25" in recommendations or "20" in recommendations  # Should mention the size

    def test_inspect_clusters_with_empty_clusters_list(self) -> None:
        """Test inspect_clusters with empty clusters list."""
        clusters: list[set[str]] = []
        entities: list[CompanySchema] = []

        clusterer = Clusterer(threshold=0.5)
        report = clusterer.inspect_clusters(clusters=clusters, entities=entities, sample_size=10)

        assert report.total_clusters == 0
        assert report.singleton_rate == 0.0
        assert len(report.largest_clusters) == 0
        assert isinstance(report.recommendations, list)

    def test_inspect_clusters_all_singletons(self) -> None:
        """Test inspect_clusters when all clusters are singletons."""
        clusters = [{"c1"}, {"c2"}, {"c3"}, {"c4"}, {"c5"}]
        entities = [create_company(f"c{i}", f"Company {i}") for i in range(1, 6)]

        clusterer = Clusterer(threshold=0.9)
        report = clusterer.inspect_clusters(clusters=clusters, entities=entities, sample_size=10)

        assert report.total_clusters == 5
        assert report.singleton_rate == 100.0  # All are singletons
        # All should be in "1" bucket
        assert report.cluster_size_distribution["1"] == 5

    def test_inspect_clusters_single_large_cluster(self) -> None:
        """Test inspect_clusters with single large cluster."""
        clusters = [{f"c{i}" for i in range(1, 16)}]  # One cluster with 15 entities
        entities = [create_company(f"c{i}", f"Company {i}") for i in range(1, 16)]

        clusterer = Clusterer(threshold=0.2)
        report = clusterer.inspect_clusters(clusters=clusters, entities=entities, sample_size=10)

        assert report.total_clusters == 1
        assert report.singleton_rate == 0.0  # No singletons
        assert len(report.largest_clusters) == 1
        assert report.largest_clusters[0]["size"] == 15

    def test_inspect_clusters_sample_size_larger_than_clusters(self) -> None:
        """Test that sample_size larger than number of clusters works correctly."""
        clusters = [{"c1", "c2"}, {"c3", "c4"}]
        entities = [create_company(f"c{i}", f"Company {i}") for i in range(1, 5)]

        clusterer = Clusterer(threshold=0.5)
        report = clusterer.inspect_clusters(
            clusters=clusters,
            entities=entities,
            sample_size=100,  # Much larger than 2 clusters
        )

        # Should only return 2 examples (all clusters)
        assert len(report.largest_clusters) <= 2

    def test_inspect_clusters_entity_text_extraction_with_name(self) -> None:
        """Test that entity text is extracted from name attribute."""
        clusters = [{"c1", "c2", "c3"}]
        entities = [
            create_company("c1", "Acme Corporation"),
            create_company("c2", "TechStart Industries"),
            create_company("c3", "Global Solutions"),
        ]

        clusterer = Clusterer(threshold=0.5)
        report = clusterer.inspect_clusters(clusters=clusters, entities=entities, sample_size=5)

        # Check that entity texts are extracted correctly
        assert len(report.largest_clusters) == 1
        cluster_example = report.largest_clusters[0]
        assert "entity_texts" in cluster_example
        entity_texts = cluster_example["entity_texts"]
        assert "Acme Corporation" in entity_texts
        assert "TechStart Industries" in entity_texts
        assert "Global Solutions" in entity_texts

    def test_inspect_clusters_entity_text_extraction_custom_str(self) -> None:
        """Test entity text extraction for entities with custom __str__."""

        class CustomEntity:
            def __init__(self, entity_id: str, label: str):
                self.id = entity_id
                self.label = label

            def __str__(self) -> str:
                return f"CustomEntity({self.label})"

        clusters = [{"e1", "e2"}]
        entities = [
            CustomEntity("e1", "First"),
            CustomEntity("e2", "Second"),
        ]

        clusterer = Clusterer(threshold=0.5)
        report = clusterer.inspect_clusters(clusters=clusters, entities=entities, sample_size=5)

        cluster_example = report.largest_clusters[0]
        entity_texts = cluster_example["entity_texts"]
        assert "CustomEntity(First)" in entity_texts
        assert "CustomEntity(Second)" in entity_texts

    def test_inspect_clusters_entity_text_extraction_fallback(self) -> None:
        """Test entity text extraction fallback when no name or __str__."""

        class MinimalEntity:
            def __init__(self, entity_id: str):
                self.id = entity_id

        clusters = [{"e1", "e2"}]
        entities = [
            MinimalEntity("e1"),
            MinimalEntity("e2"),
        ]

        clusterer = Clusterer(threshold=0.5)
        report = clusterer.inspect_clusters(clusters=clusters, entities=entities, sample_size=5)

        # Should use entity ID as fallback
        cluster_example = report.largest_clusters[0]
        assert "entity_texts" in cluster_example
        assert len(cluster_example["entity_texts"]) == 2

    def test_inspect_clusters_size_distribution_buckets(self) -> None:
        """Test cluster size distribution bucketing."""
        # Create clusters in each bucket
        clusters = [
            {"c1"},  # "1"
            {"c2", "c3"},  # "2-3"
            {"c4", "c5", "c6"},  # "2-3"
            {"c7", "c8", "c9", "c10"},  # "4-6"
            {"c11", "c12", "c13", "c14", "c15"},  # "4-6"
            {"c16", "c17", "c18", "c19", "c20", "c21"},  # "4-6"
            {f"c{i}" for i in range(22, 30)},  # "7-10" (8 entities)
            {f"c{i}" for i in range(30, 40)},  # "7-10" (10 entities)
            {f"c{i}" for i in range(40, 52)},  # "11+" (12 entities)
        ]

        entities = [create_company(f"c{i}", f"Company {i}") for i in range(1, 52)]

        clusterer = Clusterer(threshold=0.5)
        report = clusterer.inspect_clusters(clusters=clusters, entities=entities, sample_size=10)

        dist = report.cluster_size_distribution
        assert dist["1"] == 1  # 1 singleton
        assert dist["2-3"] == 2  # 2 clusters with 2-3 entities
        assert dist["4-6"] == 3  # 3 clusters with 4-6 entities
        assert dist["7-10"] == 2  # 2 clusters with 7-10 entities
        assert dist["11+"] == 1  # 1 cluster with 11+ entities

    def test_inspect_clusters_markdown_output(self) -> None:
        """Test that report can generate readable markdown."""
        clusters = [
            {"c1", "c2", "c3"},
            {"c4", "c5"},
            {"c6"},
        ]
        entities = [create_company(f"c{i}", f"Company {i}") for i in range(1, 7)]

        clusterer = Clusterer(threshold=0.5)
        report = clusterer.inspect_clusters(clusters=clusters, entities=entities, sample_size=5)

        markdown = report.to_markdown()
        assert isinstance(markdown, str)
        assert "# Cluster Inspection Report" in markdown
        assert str(report.total_clusters) in markdown
        assert "Singleton Rate" in markdown

    def test_inspect_clusters_to_dict_structure(self) -> None:
        """Test that to_dict returns complete report structure."""
        clusters = [{"c1", "c2"}, {"c3"}]
        entities = [create_company(f"c{i}", f"Company {i}") for i in range(1, 4)]

        clusterer = Clusterer(threshold=0.5)
        report = clusterer.inspect_clusters(clusters=clusters, entities=entities, sample_size=5)

        report_dict = report.to_dict()
        assert isinstance(report_dict, dict)
        assert "total_clusters" in report_dict
        assert "singleton_rate" in report_dict
        assert "cluster_size_distribution" in report_dict
        assert "largest_clusters" in report_dict
        assert "recommendations" in report_dict

    def test_inspect_clusters_stats_property_numerical_only(self) -> None:
        """Test that stats property returns only numerical metrics."""
        clusters = [{"c1", "c2"}, {"c3"}]
        entities = [create_company(f"c{i}", f"Company {i}") for i in range(1, 4)]

        clusterer = Clusterer(threshold=0.5)
        report = clusterer.inspect_clusters(clusters=clusters, entities=entities, sample_size=5)

        stats = report.stats
        assert "total_clusters" in stats
        assert "singleton_rate" in stats
        assert "cluster_size_distribution" in stats
        assert "largest_clusters" not in stats  # Examples excluded
        assert "recommendations" not in stats  # Recommendations excluded

    def test_inspect_clusters_small_total_clusters_recommendation(self) -> None:
        """Test that small number of clusters (<10) triggers manual review suggestion."""
        clusters = [
            {"c1", "c2"},
            {"c3", "c4"},
            {"c5"},
        ]
        entities = [create_company(f"c{i}", f"Company {i}") for i in range(1, 6)]

        clusterer = Clusterer(threshold=0.5)
        report = clusterer.inspect_clusters(clusters=clusters, entities=entities, sample_size=5)

        assert report.total_clusters < 10
        recommendations = " ".join(report.recommendations)
        assert "small number" in recommendations.lower() or "review" in recommendations.lower()

    def test_inspect_clusters_large_total_clusters_recommendation(self) -> None:
        """Test that large number of clusters (>1000) triggers sampling suggestion."""
        # Create 1001 singleton clusters
        clusters = [{f"c{i}"} for i in range(1, 1002)]
        entities = [create_company(f"c{i}", f"Company {i}") for i in range(1, 1002)]

        clusterer = Clusterer(threshold=0.99)
        report = clusterer.inspect_clusters(clusters=clusters, entities=entities, sample_size=5)

        assert report.total_clusters > 1000
        recommendations = " ".join(report.recommendations)
        assert "large number" in recommendations.lower() or "sampling" in recommendations.lower()

    def test_inspect_clusters_recommendations_change_based_on_patterns(self) -> None:
        """Test that recommendations vary based on different cluster patterns."""
        # High singleton rate scenario
        clusters_high_singleton = [{"c1"}, {"c2"}, {"c3"}, {"c4", "c5"}]
        entities1 = [create_company(f"c{i}", f"Company {i}") for i in range(1, 6)]

        clusterer1 = Clusterer(threshold=0.8)
        report1 = clusterer1.inspect_clusters(
            clusters=clusters_high_singleton, entities=entities1, sample_size=5
        )

        # Low singleton rate scenario
        clusters_low_singleton = [
            {f"c{i}" for i in range(1, 11)},
            {f"c{i}" for i in range(11, 21)},
            {"c21"},
        ]
        entities2 = [create_company(f"c{i}", f"Company {i}") for i in range(1, 22)]

        clusterer2 = Clusterer(threshold=0.3)
        report2 = clusterer2.inspect_clusters(
            clusters=clusters_low_singleton, entities=entities2, sample_size=5
        )

        # Recommendations should be different
        recs1 = " ".join(report1.recommendations)
        recs2 = " ".join(report2.recommendations)
        assert recs1 != recs2

    def test_inspect_clusters_largest_clusters_sorted_by_size(self) -> None:
        """Test that largest_clusters are sorted by size (descending)."""
        clusters = [
            {"c1", "c2"},  # Size 2
            {f"c{i}" for i in range(3, 13)},  # Size 10
            {"c13", "c14", "c15"},  # Size 3
            {f"c{i}" for i in range(16, 22)},  # Size 6
        ]
        entities = [create_company(f"c{i}", f"Company {i}") for i in range(1, 22)]

        clusterer = Clusterer(threshold=0.5)
        report = clusterer.inspect_clusters(clusters=clusters, entities=entities, sample_size=10)

        # Largest clusters should be sorted by size descending
        sizes = [cluster["size"] for cluster in report.largest_clusters]
        assert sizes == sorted(sizes, reverse=True)
        assert sizes[0] == 10  # Largest
        assert sizes[-1] == 2  # Smallest in the sample

    def test_inspect_clusters_includes_entity_ids(self) -> None:
        """Test that largest_clusters include entity_ids list."""
        clusters = [{"c1", "c2", "c3"}]
        entities = [create_company(f"c{i}", f"Company {i}") for i in range(1, 4)]

        clusterer = Clusterer(threshold=0.5)
        report = clusterer.inspect_clusters(clusters=clusters, entities=entities, sample_size=5)

        cluster_example = report.largest_clusters[0]
        assert "entity_ids" in cluster_example
        entity_ids = cluster_example["entity_ids"]
        assert len(entity_ids) == 3
        assert "c1" in entity_ids
        assert "c2" in entity_ids
        assert "c3" in entity_ids

    def test_inspect_clusters_includes_cluster_id(self) -> None:
        """Test that largest_clusters include cluster_id (index in original list)."""
        clusters = [
            {"c1", "c2"},  # Index 0
            {"c3", "c4", "c5"},  # Index 1
            {"c6"},  # Index 2
        ]
        entities = [create_company(f"c{i}", f"Company {i}") for i in range(1, 7)]

        clusterer = Clusterer(threshold=0.5)
        report = clusterer.inspect_clusters(clusters=clusters, entities=entities, sample_size=10)

        # Largest clusters should be sorted by size, so cluster_id might not be sequential
        for cluster_info in report.largest_clusters:
            assert "cluster_id" in cluster_info
            assert isinstance(cluster_info["cluster_id"], int)
            assert 0 <= cluster_info["cluster_id"] < 3
