"""CorrelationClusterer (C6): a merge-resistant Clusterer variant.

The base :class:`~langres.core.clusterer.Clusterer` builds a graph from edges
scoring >= threshold and takes connected components -- i.e. FULL transitive
closure. A chain of edges (A-B, B-C) with no direct A-C edge still merges A,
B, and C into one cluster, because connectivity alone (not direct evidence)
drives the merge. This is the documented M3 over-merge failure mode (-0.63
BCubed): one weak link in a long chain can pull unrelated records together.

``CorrelationClusterer`` implements the classic *pivot algorithm* for
correlation clustering (Ailon, Charikar & Newman, "Aggregating Inconsistent
Information: Ranking and Clustering", JACM 2008, building on Bansal, Blum &
Chawla's correlation-clustering formulation): a node only joins a cluster if
it has a DIRECT edge >= threshold to that cluster's pivot. Chains without a
direct edge to the pivot don't force a merge -- structurally resistant to the
classic "chaining" failure, while a genuinely well-connected group (e.g. a
clique where every pair was directly compared and matched) still merges fully,
same as the base Clusterer.
"""

from collections.abc import Iterator
from typing import ClassVar

from langres.core.clusterer import Clusterer
from langres.core.models import PairwiseJudgement, predicted_match
from langres.core.registry import register


@register("correlation_clusterer")
class CorrelationClusterer(Clusterer):
    """Merge-resistant Clusterer: the pivot algorithm for correlation clustering.

    Drop-in alternative to the base :class:`~langres.core.clusterer.Clusterer`
    (same ``threshold`` constructor, same ``config``/``from_config``,
    inherits ``evaluate()``/``inspect_clusters()`` unchanged -- only
    :meth:`cluster` differs). NOT the default: benchmark before switching (see
    ``examples/research/w1_blocking_algebra_output.md``).

    Algorithm, per call to :meth:`cluster`:

    1. Build an undirected weighted graph from judgements with
       ``score >= threshold`` (max score kept for a duplicate pair; self-pairs
       ignored) -- same edge set the base Clusterer would use.
    2. Process nodes in a deterministic order: highest max-incident-edge-score
       first, ties broken by node id (so results are reproducible and biased
       toward the most-confident evidence first).
    3. For each unprocessed node (in that order), form a cluster from the node
       plus every one of its DIRECT neighbours that is still unprocessed.
       Remove those nodes from further consideration and continue.

    A node with only an indirect (multi-hop) path to a cluster is never pulled
    in -- unlike connected components, which merges anything reachable by any
    chain of qualifying edges.
    """

    type_name: ClassVar[str] = "correlation_clusterer"

    def cluster(
        self,
        judgements: Iterator[PairwiseJudgement] | list[PairwiseJudgement],
    ) -> list[set[str]]:
        """Form entity clusters via the pivot algorithm for correlation clustering.

        Args:
            judgements: Iterator or list of PairwiseJudgement objects.

        Returns:
            List of clusters (sets of entity ids). Like the base Clusterer,
            entities with no qualifying edge are simply absent (no singleton
            clusters).
        """
        adjacency = self._build_adjacency(judgements)

        remaining = set(adjacency)
        clusters: list[set[str]] = []
        for node in sorted(adjacency, key=lambda n: self._pivot_priority(n, adjacency)):
            if node not in remaining:
                continue
            cluster = {node} | (set(adjacency[node]) & remaining)
            remaining -= cluster
            clusters.append(cluster)
        return clusters

    def _build_adjacency(
        self,
        judgements: Iterator[PairwiseJudgement] | list[PairwiseJudgement],
    ) -> dict[str, dict[str, float]]:
        """Build a symmetric adjacency map from judgements meeting the threshold."""
        edges: dict[frozenset[str], float] = {}
        for judgement in judgements:
            if judgement.left_id == judgement.right_id:
                continue
            if predicted_match(judgement, self.threshold) is not True:
                continue
            # The edge weight is the confidence-ordered value. A ranker's ``score``
            # is it; a decider that carries no score falls back to ``confidence``,
            # else a unit weight (a bare "yes" is still a full-strength edge, never
            # a silent zero that would drop the merge).
            weight = (
                judgement.score
                if judgement.score is not None
                else judgement.confidence
                if judgement.confidence is not None
                else 1.0
            )
            key = frozenset((judgement.left_id, judgement.right_id))
            if key not in edges or weight > edges[key]:
                edges[key] = weight

        adjacency: dict[str, dict[str, float]] = {}
        for key, score in edges.items():
            left, right = tuple(key)
            adjacency.setdefault(left, {})[right] = score
            adjacency.setdefault(right, {})[left] = score
        return adjacency

    def _pivot_priority(
        self, node: str, adjacency: dict[str, dict[str, float]]
    ) -> tuple[float, str]:
        """Sort key: highest-confidence edge first, ties broken by node id."""
        best_score = max(adjacency[node].values())
        return (-best_score, node)
