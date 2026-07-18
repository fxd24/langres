"""Op-role adapters over the legacy components (W2, epic #193).

The :mod:`~langres.core.op` module says *position is not a type*: a blocker, a
comparator, a matcher and a clusterer are the same handful of operations
(:class:`~langres.core.op.Source`, :class:`~langres.core.op.Score`,
:class:`~langres.core.op.ClusterStage`, :class:`~langres.core.op.Finalize`) at
different settings, and each speaks the one :class:`~langres.core.pairs.Pairs`
carrier. The legacy components still speak their historical dialects
(``Blocker.stream(records) -> Iterator[ERCandidate]``,
``Comparator.compare(left, right) -> ComparisonVector``,
``Matcher.forward(Iterator[ERCandidate]) -> Iterator[PairwiseJudgement]``,
``Clusterer.cluster(judgements) -> list[set[str]]``,
``Canonicalizer.canonicalize(records) -> dict``) and 200+ call sites pin those
method names and signatures.

This module bridges the two **additively**. Each adapter *holds* a legacy
component instance and implements the Op-role ``forward`` by translating through
the carrier — it does **not** rename or re-parent the legacy classes. Nothing in
the spine uses these adapters yet: the spine still drives the legacy path
directly, and the flip that adopts these adapters is W3. Their only correctness
proof this wave is ``tests/core/test_op_adapters.py``.

**What each adapter deliberately does NOT absorb.** Index lifecycle
(``_ensure_index_built`` / ``iter_vector_blockers``) stays in the spine this
wave, so :class:`BlockerSource` is a thin ``stream`` bridge — a ``VectorBlocker``
must already have its index built before its ``BlockerSource`` runs. The lazy
spend/logging wrappers (``SpendCappedMatcher`` / ``LoggingMatcher``) are NOT
migrated either: they depend on the scorer being a lazy generator
(budget-check-before-next-pull), which fights the eager materialized ``Pairs``;
that is a W3 spend-seam problem. These adapters wrap the *raw* components.

**Import discipline.** This module imports both :mod:`~langres.core.op` and the
concrete legacy component contracts, so it is NOT a leaf and must NOT be
re-exported from the ``langres.core`` contracts surface. It is mapped to the
``architectures`` target package (the tier that already imports both ``core`` and
``curation``), beside the spine (``_model_run``) that will adopt it in W3 — see
``tools/refactor_target_packages.json``. The one ``curation`` dependency
(:class:`~langres.curation.canonicalizer.Canonicalizer`) is referenced under
``TYPE_CHECKING`` only (annotation), so this module adds no runtime edge into
``curation``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel

from langres.core.blocker import Blocker
from langres.core.clusterer import Clusterer
from langres.core.comparator import Comparator
from langres.core.groups import derive_groups_from_pairs
from langres.core.matcher import GroupwiseMatcher, Matcher
from langres.core.op import (
    ClusterStage,
    Finalize,
    OutSpace,
    Records,
    Score,
    Source,
    Spending,
)
from langres.core.pairs import PairRow, Pairs

if TYPE_CHECKING:
    from langres.core.models import PairwiseJudgement
    from langres.curation.canonicalizer import Canonicalizer

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _rescore(pairs: Pairs[SchemaT], judgements: Iterable[PairwiseJudgement]) -> Pairs[SchemaT]:
    """Map each judgement back onto its row by ``(left_id, right_id)`` identity.

    The shared back-map for every :class:`Score` that runs a legacy
    :class:`~langres.core.matcher.Matcher`: a matcher yields one
    :class:`~langres.core.models.PairwiseJudgement` per candidate keyed by
    ``(left_id, right_id)``, and a set-wise matcher re-orders its output by
    anchor group — so the rows are re-associated by **identity**, not position,
    and emitted in the *incoming* ``pairs`` order (preserving row identity and
    order). Every judge output field is copied onto the row; the row's
    ``score_type`` becomes the judgement's OWN family (never the adapter's
    declared ``out_space``), so a mixed-family matcher (a cascade) faithfully
    labels each row rather than being flattened to one family.

    A row with no matching judgement is passed through unchanged (a matcher's
    contract is one judgement per candidate, so this is a defensive no-op).

    Uniqueness assumption: the back-map keys judgements by ``(left_id, right_id)``
    and assumes each such pair appears **at most once** per ``Pairs`` — the
    invariant every shipping blocker already satisfies (``AllPairsBlocker`` emits
    strictly ``i < j``, ``VectorBlocker`` canonicalizes pair order by id, and
    ``CompositeBlocker`` dedups via frozenset pair-keys). ``Pairs`` does not
    *enforce* uniqueness, so were a future carrier to hold duplicate
    ``(left_id, right_id)`` rows, the ``by_pair`` dict would collapse them
    last-wins and every duplicate row would take the same judgement. A
    ``Pairs``-level uniqueness invariant is a possible future hardening (noted,
    not added here — no shipping blocker produces duplicates, so a runtime guard
    would defend a state that cannot occur). A judgement whose key matches no row
    is simply dropped (consistent with the one-judgement-per-received-candidate
    matcher contract).

    Args:
        pairs: The incoming scored relation (rows carry the identities to map to).
        judgements: The matcher's output, consumed once.

    Returns:
        A new ``Pairs`` over the *same* store with rescored rows in input order.
    """
    by_pair: dict[tuple[str, str], PairwiseJudgement] = {
        (judgement.left_id, judgement.right_id): judgement for judgement in judgements
    }
    rows: list[PairRow[SchemaT]] = []
    for row in pairs.rows:
        judgement = by_pair.get((row.left_id, row.right_id))
        if judgement is None:
            rows.append(row)
            continue
        rows.append(
            row.model_copy(
                update={
                    "score": judgement.score,
                    "score_type": judgement.score_type,
                    "decision": judgement.decision,
                    "confidence": judgement.confidence,
                    "confidence_source": judgement.confidence_source,
                    "decision_step": judgement.decision_step,
                    "reasoning": judgement.reasoning,
                    "provenance": judgement.provenance,
                }
            )
        )
    return Pairs(store=pairs.store, rows=rows)


class BlockerSource(Source[SchemaT], Generic[SchemaT]):
    """:class:`~langres.core.op.Source` adapter over a legacy :class:`~langres.core.blocker.Blocker`.

    Bridges ``blocker.stream(records) -> Iterator[ERCandidate]`` to a
    :class:`~langres.core.pairs.Pairs` via
    :meth:`~langres.core.pairs.Pairs.from_candidates` (the blocker's
    ``similarity_score`` lands as an *unscored* row score, ``score_type is None``
    — a blocker similarity, never a judge score).

    Index lifecycle is **not** absorbed: a ``VectorBlocker`` must already have its
    index built (the spine's ``_ensure_index_built`` still owns that this wave),
    so this adapter is a plain ``stream`` bridge.
    """

    def __init__(self, blocker: Blocker[SchemaT]) -> None:
        """Wrap ``blocker``.

        Args:
            blocker: The legacy blocker to adapt. Any :class:`Blocker` works; its
                index (if any) must already be built.
        """
        self.blocker = blocker

    def forward(self, records: Records) -> Pairs[SchemaT]:
        """Block ``records`` into a ``Pairs`` (candidates from ``blocker.stream``)."""
        return Pairs.from_candidates(list(self.blocker.stream(records)))


class ComparatorScore(Score[SchemaT], Generic[SchemaT]):
    """:class:`~langres.core.op.Score` adapter over a legacy :class:`~langres.core.comparator.Comparator`.

    A ``Score`` at ``out_space="vector"``: it attaches each row's per-feature
    :class:`~langres.core.feature.ComparisonVector` and leaves ``score`` /
    ``score_type`` **untouched**. A vector is not a scalar, so the rows stay
    "unscored" (``score_type is None``) — a downstream scalarizer :class:`Score`
    (e.g. a weighted average) turns the vector into a score.
    """

    def __init__(self, comparator: Comparator[SchemaT]) -> None:
        """Wrap ``comparator`` as a vector-space Score.

        Args:
            comparator: The legacy comparator to adapt.
        """
        super().__init__(scope="pair", out_space="vector")
        self.comparator = comparator

    def forward(self, pairs: Pairs[SchemaT]) -> Pairs[SchemaT]:
        """Attach ``comparator.compare(left, right)`` to every row; scores unchanged."""
        rows = [
            row.model_copy(update={"comparison": self.comparator.compare(row.left, row.right)})
            for row in pairs.rows
        ]
        return Pairs(store=pairs.store, rows=rows)


class MatcherScore(Score[SchemaT], Spending, Generic[SchemaT]):
    """:class:`~langres.core.op.Score` adapter over a legacy pairwise :class:`~langres.core.matcher.Matcher`.

    ``Spending`` (declares it may bill): the wrapped ``Matcher`` can be a paid LLM,
    so the explicit-chain door caps it through the model's ``SpendMonitor``.

    Bridges ``matcher.forward(Iterator[ERCandidate]) -> Iterator[PairwiseJudgement]``:
    the incoming ``Pairs`` is projected to candidates
    (:meth:`~langres.core.pairs.Pairs.to_candidates`), scored, and each judgement
    is mapped back onto its row by ``(left_id, right_id)`` identity (order
    preserved — see :func:`_rescore`). Every judge output field
    (score / score_type / decision / confidence / confidence_source /
    decision_step / reasoning / provenance) lands on the row.

    ``out_space`` is **declared, not inferred**: a legacy ``Matcher`` tags its
    ``score_type`` per-judgement (there is no class-level constant), so the caller
    names the family the matcher produces. For a registry-built matcher that is
    ``get_method(name).score_type``. For a *mixed-family* matcher (``CascadeMatcher`` /
    ``CascadeChainMatcher`` — early-exits emit ``sim_cos``, escalations
    ``prob_llm``) declare the escalated/authoritative family (``"prob_llm"``),
    matching the method registry's own ``cascade`` spec. The declared
    ``out_space`` is the Score's advertised metadata; each row still carries its
    judgement's OWN ``score_type``, so a mixed matcher never silently mislabels a
    row.

    The back-map assumes each ``(left_id, right_id)`` appears at most once per
    ``Pairs`` (the invariant every shipping blocker satisfies) — see
    :func:`_rescore` for the assumption and its future-hardening note.
    """

    def __init__(self, matcher: Matcher[SchemaT], *, out_space: OutSpace) -> None:
        """Wrap ``matcher`` as a Score declaring ``out_space``.

        Args:
            matcher: The legacy pairwise matcher to adapt.
            out_space: The score family this matcher produces (its per-judgement
                ``score_type``); validated against the known families by
                :class:`~langres.core.op.Score`.
        """
        super().__init__(scope="pair", out_space=out_space)
        self.matcher = matcher

    def forward(self, pairs: Pairs[SchemaT]) -> Pairs[SchemaT]:
        """Score every row through ``matcher.forward`` and map judgements back onto rows."""
        judgements = self.matcher.forward(iter(pairs.to_candidates()))
        return _rescore(pairs, judgements)


class GroupwiseMatcherScore(Score[SchemaT], Spending, Generic[SchemaT]):
    """:class:`~langres.core.op.Score` adapter over a :class:`~langres.core.matcher.GroupwiseMatcher`.

    ``Spending`` (declares it may bill): its ``GroupwiseMatcher`` can be a paid
    set-wise LLM. The explicit-chain door cannot auto-cap it (a
    ``SpendCappedMatcher`` meters ``forward``, not the ``forward_groups`` this
    adapter calls), so ``from_topology`` rejects it rather than run it off-ledger.

    A group-scope ``Score`` (``scope="group"``, ``out_space="prob_group_llm"``):
    it derives per-anchor groups from the ``Pairs``'s candidates
    (:func:`~langres.core.groups.derive_groups_from_pairs`, the same default the
    matcher's own ``forward`` uses), calls
    :meth:`~langres.core.matcher.GroupwiseMatcher.forward_groups` directly, and
    maps the resulting judgements back onto rows by ``(left_id, right_id)``
    identity (:func:`_rescore`). Calling ``forward_groups`` explicitly (rather
    than the inherited pairwise ``forward``) makes the group scope legible at the
    adapter boundary.
    """

    def __init__(self, matcher: GroupwiseMatcher[SchemaT]) -> None:
        """Wrap ``matcher`` as a group-scope Score.

        Args:
            matcher: The set-wise matcher to adapt.
        """
        super().__init__(scope="group", out_space="prob_group_llm")
        self.matcher = matcher

    def forward(self, pairs: Pairs[SchemaT]) -> Pairs[SchemaT]:
        """Derive groups, score via ``forward_groups``, and map judgements back onto rows."""
        groups = derive_groups_from_pairs(iter(pairs.to_candidates()))
        judgements = self.matcher.forward_groups(groups)
        return _rescore(pairs, judgements)


class ClustererStage(ClusterStage[SchemaT], Generic[SchemaT]):
    """:class:`~langres.core.op.ClusterStage` adapter over a legacy :class:`~langres.core.clusterer.Clusterer`.

    Bridges ``Pairs -> list[set[str]]``: only the SCORED rows
    (``score_type is not None``) are projected to
    :class:`~langres.core.models.PairwiseJudgement` (an unscored row has no
    judgement to cluster on, and :meth:`~langres.core.pairs.PairRow.to_judgement`
    would refuse it) and handed to ``clusterer.cluster``.

    The legacy clusterer owns the ``threshold``; the :class:`ClusterStage` owns
    the ``algorithm`` name. The adapter reconciles the two: it holds the legacy
    clusterer (which keeps thresholding on the projected judgements) and reports
    the algorithm up to :class:`ClusterStage`, mapped from the clusterer's
    ``type_name`` — ``"correlation_clusterer"`` (the pivot algorithm) →
    ``"pivot"``, the base transitive-closure ``Clusterer`` → ``"transitive_closure"``.
    """

    def __init__(self, clusterer: Clusterer) -> None:
        """Wrap ``clusterer``, deriving the ClusterStage algorithm from its type.

        Args:
            clusterer: The legacy clusterer to adapt (owns its ``threshold``).
        """
        algorithm = (
            "pivot"
            if getattr(clusterer, "type_name", None) == "correlation_clusterer"
            else "transitive_closure"
        )
        super().__init__(algorithm=algorithm)
        self.clusterer = clusterer

    def forward(self, pairs: Pairs[SchemaT]) -> list[set[str]]:
        """Cluster the SCORED rows' judgements (unscored rows carry no judgement)."""
        judgements = (row.to_judgement() for row in pairs.rows if row.score_type is not None)
        return self.clusterer.cluster(judgements)


class CanonicalizeFinalize(Finalize):
    """:class:`~langres.core.op.Finalize` adapter over a :class:`~langres.curation.canonicalizer.Canonicalizer`.

    Fuses ONE id cluster into one golden record via the canonicalizer's
    survivorship. The :class:`Finalize` contract hands ``forward`` id clusters
    (``list[set[str]]``), but survivorship needs the actual records — so this
    adapter is constructed with the ``store`` (id → entity) that resolves them,
    exactly the store a :class:`~langres.core.pairs.Pairs` already carries. A
    cluster's entities are dumped to dicts, canonicalized, and re-validated into a
    single :data:`~langres.core.op.GoldenRecord` (a Pydantic model).

    **Judgment call (flagged):** the :class:`Finalize` contract's ``forward`` takes
    a *list* of clusters but a canonicalize Finalize returns a *single*
    ``GoldenRecord`` ("fuses **a** cluster"). This adapter therefore requires
    exactly one cluster and raises otherwise, rather than silently fusing the
    first and dropping the rest.
    """

    def __init__(self, canonicalizer: Canonicalizer, *, store: dict[str, BaseModel]) -> None:
        """Wrap ``canonicalizer`` with the ``store`` that resolves cluster ids to records.

        Args:
            canonicalizer: The survivorship policy to delegate to.
            store: The id → entity map (e.g. a ``Pairs.store``) the cluster's ids
                resolve against.
        """
        self.canonicalizer = canonicalizer
        self.store = store

    def forward(self, clusters: list[set[str]]) -> BaseModel:
        """Fuse the single input cluster into one golden record.

        Args:
            clusters: Exactly one id cluster.

        Returns:
            One :data:`~langres.core.op.GoldenRecord` — the survivorship-merged
            record, re-validated into the cluster entities' own schema.

        Raises:
            ValueError: If ``clusters`` does not hold exactly one cluster, or the
                cluster is empty.
        """
        if len(clusters) != 1:
            raise ValueError(
                f"CanonicalizeFinalize fuses exactly one cluster into one golden record, but got "
                f"{len(clusters)} clusters. Cause: a canonicalize Finalize returns a single "
                f"GoldenRecord. Fix: pass the one cluster to canonicalize (e.g. [clusters[i]])."
            )
        (cluster,) = clusters
        if not cluster:
            raise ValueError("CanonicalizeFinalize cannot fuse an empty cluster.")
        # ``cluster`` is a set, so iterating it is hash-randomized across processes.
        # Record order drives the golden id (defaults to records[0]["id"]) and every
        # first-seen survivorship tiebreak, so canonicalize over ids in a DETERMINISTIC
        # (sorted) order — otherwise the fused record's id/tiebroken fields vary run-to-run.
        entities = [self.store[entity_id] for entity_id in sorted(cluster)]
        records = [entity.model_dump() for entity in entities]
        golden = self.canonicalizer.canonicalize(records)
        return type(entities[0]).model_validate(golden)


__all__ = [
    "BlockerSource",
    "CanonicalizeFinalize",
    "ClustererStage",
    "ComparatorScore",
    "GroupwiseMatcherScore",
    "MatcherScore",
]
