"""
Graph-based clusterer for entity formation.

This module provides the Clusterer class, which converts pairwise match
judgements into entity clusters using graph algorithms (connected components).
"""

from collections.abc import Iterator
from typing import Any, Generic, TypeVar

import networkx as nx

from langres.core.models import PairwiseJudgement
from langres.core.reports import ClusterInspectionReport

SchemaT = TypeVar("SchemaT")


class Clusterer:
    """Graph-based clusterer for entity formation.

    Converts pairwise match decisions into entity clusters using graph algorithms.

    Example:
        clusterer = Clusterer(threshold=0.7)

        # judgements is an iterator of PairwiseJudgement objects
        clusters = clusterer.cluster(judgements)

        # clusters is a list of sets: [{"id1", "id2"}, {"id3", "id4", "id5"}]
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize clusterer.

        Args:
            threshold: Minimum score to consider a match (0.0 to 1.0).
                Judgements with score >= threshold are treated as matches.

        Raises:
            ValueError: If threshold is not in range [0.0, 1.0].
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.threshold = threshold

    def cluster(
        self,
        judgements: Iterator[PairwiseJudgement] | list[PairwiseJudgement],
    ) -> list[set[str]]:
        """Form entity clusters from pairwise judgements.

        Args:
            judgements: Iterator or list of PairwiseJudgement objects

        Returns:
            List of clusters, where each cluster is a set of entity IDs

        Note:
            Uses connected components algorithm from networkx.
            All IDs connected by edges >= threshold are grouped together.
        """
        # Build an undirected graph from judgements that meet the threshold
        G: Any = nx.Graph()

        for judgement in judgements:
            if judgement.score >= self.threshold:
                G.add_edge(judgement.left_id, judgement.right_id)

        # Get connected components (transitive closure)
        clusters = [set(component) for component in nx.connected_components(G)]

        return clusters

    def inspect_clusters(
        self,
        clusters: list[set[str]],
        entities: list[SchemaT],
        sample_size: int = 10,
    ) -> ClusterInspectionReport:
        """Explore clusters without ground truth labels.

        Use this method to understand clustering output before labeling:
        - Total cluster count and size distribution
        - Singleton rate (percentage of 1-entity clusters)
        - Sample of largest clusters with readable entity text
        - Threshold tuning recommendations

        For quality evaluation with ground truth labels, use
        PipelineDebugger.analyze_clusters() instead.

        Args:
            clusters: List of sets containing entity IDs (output from cluster() method)
            entities: List of entity objects for text extraction
            sample_size: Number of largest clusters to include as examples (default: 10)

        Returns:
            ClusterInspectionReport with statistics, examples, and recommendations

        Example:
            >>> clusterer = Clusterer(threshold=0.5)
            >>> clusters = clusterer.cluster(judgements)
            >>> report = clusterer.inspect_clusters(clusters, entities)
            >>> print(report.to_markdown())
        """
        # Handle empty clusters
        if not clusters:
            return ClusterInspectionReport(
                total_clusters=0,
                singleton_rate=0.0,
                cluster_size_distribution={
                    "1": 0,
                    "2-3": 0,
                    "4-6": 0,
                    "7-10": 0,
                    "11+": 0,
                },
                largest_clusters=[],
                recommendations=[],
            )

        # Compute cluster statistics
        total_clusters = len(clusters)
        cluster_sizes = [len(cluster) for cluster in clusters]
        singleton_count = sum(1 for size in cluster_sizes if size == 1)
        singleton_rate = (singleton_count / total_clusters) * 100  # As percentage 0-100
        avg_cluster_size = sum(cluster_sizes) / total_clusters
        max_cluster_size = max(cluster_sizes)

        # Compute cluster size distribution
        distribution: dict[str, int] = {
            "1": 0,
            "2-3": 0,
            "4-6": 0,
            "7-10": 0,
            "11+": 0,
        }

        for size in cluster_sizes:
            if size == 1:
                distribution["1"] += 1
            elif 2 <= size <= 3:
                distribution["2-3"] += 1
            elif 4 <= size <= 6:
                distribution["4-6"] += 1
            elif 7 <= size <= 10:
                distribution["7-10"] += 1
            else:
                distribution["11+"] += 1

        # Create ID to entity map for O(1) lookup
        id_to_entity: dict[str, Any] = {}
        for entity in entities:
            if hasattr(entity, "id"):
                id_to_entity[entity.id] = entity
            elif hasattr(entity, "pk"):
                id_to_entity[entity.pk] = entity

        # Extract largest cluster examples
        # Sort clusters by size (descending), then take top sample_size
        clusters_with_indices = [(i, cluster) for i, cluster in enumerate(clusters)]
        clusters_with_indices.sort(key=lambda x: len(x[1]), reverse=True)

        largest_clusters = []
        for cluster_id, cluster in clusters_with_indices[:sample_size]:
            entity_ids = list(cluster)
            entity_texts = [
                self._extract_entity_text(id_to_entity.get(entity_id, entity_id))
                for entity_id in entity_ids
            ]

            largest_clusters.append(
                {
                    "cluster_id": cluster_id,
                    "size": len(cluster),
                    "entity_ids": entity_ids,
                    "entity_texts": entity_texts,
                }
            )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            singleton_rate=singleton_rate,
            avg_cluster_size=avg_cluster_size,
            max_cluster_size=max_cluster_size,
            total_clusters=total_clusters,
        )

        return ClusterInspectionReport(
            total_clusters=total_clusters,
            singleton_rate=singleton_rate,
            cluster_size_distribution=distribution,
            largest_clusters=largest_clusters,
            recommendations=recommendations,
        )

    def _extract_entity_text(self, entity: Any) -> str:
        """Extract human-readable text from entity object.

        Args:
            entity: Entity object to extract text from

        Returns:
            Human-readable string representation
        """
        if hasattr(entity, "name"):
            return str(entity.name)
        else:
            return str(entity)

    def _generate_recommendations(
        self,
        singleton_rate: float,
        avg_cluster_size: float,
        max_cluster_size: int,
        total_clusters: int,
    ) -> list[str]:
        """Generate rule-based recommendations for threshold tuning.

        Args:
            singleton_rate: Percentage of singleton clusters (0-100)
            avg_cluster_size: Average cluster size
            max_cluster_size: Size of largest cluster
            total_clusters: Total number of clusters

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Singleton rate analysis
        if singleton_rate > 70:
            suggested_threshold = self.threshold * 0.8
            recommendations.append(
                f"⚠️ High singleton rate ({singleton_rate:.1f}%) - threshold may be too high. "
                f"Consider lowering threshold to {suggested_threshold:.2f}"
            )
        elif singleton_rate < 20 and avg_cluster_size > 5:
            suggested_threshold = self.threshold * 1.2
            recommendations.append(
                f"⚠️ Low singleton rate ({singleton_rate:.1f}%) with large clusters - "
                f"threshold may be too low. Consider raising threshold to {suggested_threshold:.2f}"
            )
        else:
            recommendations.append(f"✅ Singleton rate ({singleton_rate:.1f}%) looks reasonable")

        # Cluster size analysis
        if max_cluster_size > 20:
            recommendations.append(
                f"⚠️ Very large cluster detected ({max_cluster_size} entities) - "
                f"may indicate over-merging. Review largest clusters."
            )

        if avg_cluster_size < 1.5:
            recommendations.append(
                f"Low average cluster size ({avg_cluster_size:.1f}) - most entities are singletons"
            )
        elif avg_cluster_size > 8:
            recommendations.append(
                f"High average cluster size ({avg_cluster_size:.1f}) - clusters may be too broad"
            )

        # Sample size guidance
        if total_clusters < 10:
            recommendations.append("Small number of clusters - review each manually")
        elif total_clusters > 1000:
            recommendations.append("Large number of clusters - consider sampling for review")

        return recommendations
