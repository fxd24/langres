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

Measured on the 9-benchmark portfolio in
``docs/research/20260727_closure_diagnostic.md`` and **recommended** as a result;
opt in with ``FuzzyString(clusterer=CorrelationClusterer())``, or the same
argument on ``VectorLLMCascade`` / the four retrieval recipes (``Reranker``
hardcodes its cluster stage and takes no ``clusterer=``). It is deliberately
still not the default. See the class docstring for the numbers and for what this
does *not* fix.
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
    (same ``threshold`` constructor, same ``config``/``from_config``, same output
    shape on distinct-id judgements -- see :meth:`cluster` for the one self-pair
    exception -- inherits ``evaluate()``/``inspect_clusters()`` unchanged, and
    only :meth:`cluster` differs).

    **It has now been benchmarked, and it is the recommended choice.** It is
    still not the default -- that is a deliberate product decision, not an
    absence of evidence. On the 9-benchmark portfolio
    (``docs/research/20260727_closure_diagnostic.md``), against the default
    transitive-closure ``Clusterer`` over the identical judgement set:

    * BCubed F1 is **strictly higher at 36 of the 45 scored grid points, tied at
      9, and lower at 0**. Nine further grid points are **unscorable** -- closure's
      giant component was too large to score, which is itself the result.
    * Judged-and-*rejected* pairs sitting inside an output cluster drop at the
      tuned operating point: **3,776 -> 676** across the portfolio (5.6x; per
      benchmark the reduction ranges 3.0x-7.4x). Closure's worst benchmark is
      ``amazon_google``, which contributes **944** of that 3,776 -- and those 944
      are **39.8% of every pair sharing one of its output clusters**, not 39.8%
      of the portfolio total. Pivot brings it to 128 (14.8%).
    * The real gap is off the tuned point: one grid step down, closure collapses
      into a giant component (on ``walmart_amazon`` at t=0.50, 6,798 of the
      7,386-record split in one cluster) while pivot degrades smoothly. Closure's
      quality is a cliff; pivot's is a slope.

    **What those numbers were measured on.** One **$0 scorer** (``rapidfuzz``)
    and **one split seed** (0). The diagnostic states outright that a matcher
    with a different error profile -- a decider LLM in particular -- "could move
    the numbers", and that the *ordering* surviving is "a hypothesis, not a
    measurement" (§4). The chaining mechanism is structural and
    scorer-independent; the magnitudes above are not.

    Opt in via the ``clusterer=`` argument of the architectures that expose one --
    :class:`~langres.architectures.fuzzy_string.FuzzyString`,
    :class:`~langres.architectures.vector_llm_cascade.VectorLLMCascade`, and the
    four retrieval recipes. (:class:`~langres.architectures.reranker.Reranker`
    does **not**: it hardcodes its cluster stage.) Or set the ``clusterer`` slot
    on a :class:`~langres.core.resolver.ERModel` directly::

        FuzzyString(clusterer=CorrelationClusterer())

    **What it does NOT do.** It *mitigates chaining*; it does **not** consume
    negative evidence. Rejected edges are discarded before pivoting
    (:meth:`_build_adjacency` keeps only rows where
    :func:`~langres.core.models.predicted_match` is ``True``), exactly as the base
    Clusterer discards them -- so a rejected pair can still land inside one
    cluster, just far less often. Pricing negatives into the objective is a
    different clusterer, not this one (``docs/THEORY.md`` §7).

    **On the citation.** The pivot order here is *deterministic* -- highest
    incident score first, ties by node id (:meth:`_pivot_priority`) -- not the
    uniformly random pivot of Ailon-Charikar-Newman. Their 3-approximation bound
    is a property of that randomization and does **not** transfer to this
    implementation. The determinism is deliberate (reproducible without a seed);
    it is simply not an approximation guarantee.

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
    4. Drop the one-node clusters. A pivot whose every neighbour was already
       claimed forms ``{node}`` at step 3; that says "this record merged with
       nothing", which is exactly what the base Clusterer says by leaving an
       unmerged record **out**. Emitting it instead would hand anyone who opts in
       a different output *shape* for the same meaning -- see :meth:`cluster`.

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

        Every returned cluster holds at least two ids. Two different situations
        collapse to that one convention, and both mean "this record merged with
        nothing":

        * a record with **no** qualifying edge never enters the adjacency map, so
          it is absent -- identical to the base Clusterer, whose graph an isolated
          record never enters either;
        * a record *with* a qualifying edge whose neighbours an earlier pivot had
          already claimed forms a one-node ``{node}`` cluster, which is dropped
          here.

        Dropping the second is what makes this class a genuine **drop-in**: the
        documented ``dedupe()`` contract returns the multi-record clusters and
        leaves singletons out, so for judgements with **distinct ids** a caller
        who opts in gets a different (better) *partition* without a different
        output *shape*. It costs no information -- an id absent from the result is
        a singleton entity by that same contract -- and it costs no measurement
        either: the portfolio harness scores ``complete_partition(clusters,
        all_ids)``, which restores every unlisted id as its own singleton before
        computing BCubed, so the diagnostic's numbers are unchanged by this.

        **The one exception, and it runs the other way.** A *self-pair*
        (``left_id == right_id``) that clears the threshold is skipped outright by
        :meth:`_build_adjacency`, so that record is absent here -- while the base
        Clusterer feeds it to ``nx.add_edge(x, x)`` and returns a ``{x}``
        singleton. On that input the BASE clusterer is the one emitting a
        singleton -- pinned by
        ``test_a_self_pair_is_the_one_input_where_the_two_shapes_still_differ``.
        Left as-is deliberately: changing the base clusterer would alter the
        default path's behaviour.

        Neither shipped blocker hands a clusterer this input. ``AllPairsBlocker``
        pairs only positions ``i < j``, and ``VectorBlocker`` drops the anchor's
        self-match **by identity** (``blockers/vector.py:_neighbor_columns``) --
        its docstring says it does so specifically to avoid "yielding a
        degenerate self-pair". Reaching this case therefore takes duplicate ids
        in the input or a custom blocker, not a default pipeline.

        Args:
            judgements: Iterator or list of PairwiseJudgement objects.

        Returns:
            List of clusters (sets of entity ids), each with >= 2 ids.
        """
        adjacency = self._build_adjacency(judgements)

        remaining = set(adjacency)
        clusters: list[set[str]] = []
        for node in sorted(adjacency, key=lambda n: self._pivot_priority(n, adjacency)):
            if node not in remaining:
                continue
            cluster = {node} | (set(adjacency[node]) & remaining)
            # Claim the nodes BEFORE the size test: a dropped singleton is still
            # processed, or the loop would revisit it forever.
            remaining -= cluster
            if len(cluster) > 1:
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
