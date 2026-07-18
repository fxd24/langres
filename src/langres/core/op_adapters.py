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
the carrier — it does **not** rename or re-parent the legacy classes. Both the
classic four-slot spine and explicit topologies execute through these adapters.

``BlockerSource.prepare`` owns the shared vector-index bind/build lifecycle:
same corpus reuses the index and changed input rebuilds it before ``forward``.
Spend and per-call logging remain wrappers around the matcher held by
``MatcherScore``; this preserves lazy budget enforcement while keeping the
durable topology free of run-specific logging state.

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

from copy import copy
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel

from langres.core.blocker import Blocker
from langres.core.clusterer import Clusterer
from langres.core.comparator import Comparator
from langres.core.fit import CalibratorFitMixin
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
    from langres.core.spend import SpendMonitor
    from langres.curation.canonicalizer import Canonicalizer

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _unordered_pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    """Canonical identity for an undirected entity pair.

    Matchers historically may emit the same pair in either orientation. The
    carrier keeps its original orientation, but correspondence validation must
    treat ``(left, right)`` and ``(right, left)`` as the same requested pair.
    """
    return (left_id, right_id) if left_id <= right_id else (right_id, left_id)


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

    Pair identity is orientation-insensitive because legacy matchers may return
    ``(right_id, left_id)`` for a requested ``(left_id, right_id)``. Otherwise
    the mapping is a strict bijection: duplicate input rows, duplicate outputs,
    missing outputs, and unexpected outputs all raise before any result is
    returned. The sole compatibility exception is an empty matcher result over
    an entirely unscored carrier, which is a safe no-op used by non-trainable
    classic matchers. An empty result over rows carrying any upstream score still
    fails loudly, so a later Select cannot mistake that score for this matcher's
    decision.

    Args:
        pairs: The incoming scored relation (rows carry the identities to map to).
        judgements: The matcher's output, consumed once.

    Returns:
        A new ``Pairs`` over the *same* store with rescored rows in input order.
    """
    input_keys = [_unordered_pair_key(row.left_id, row.right_id) for row in pairs.rows]
    duplicate_inputs = _duplicates(input_keys)
    if duplicate_inputs:
        raise ValueError(
            "MatcherScore cannot rescore duplicate input pairs: "
            f"{_pair_preview(duplicate_inputs)}. Each pair identity must occur exactly once."
        )

    materialized = list(judgements)
    if not materialized and all(row.score_type is None for row in pairs.rows):
        return pairs

    output_keys = [
        _unordered_pair_key(judgement.left_id, judgement.right_id) for judgement in materialized
    ]
    duplicate_outputs = _duplicates(output_keys)
    if duplicate_outputs:
        raise ValueError(
            "MatcherScore received duplicate judgements: "
            f"{_pair_preview(duplicate_outputs)}. A matcher must emit exactly one judgement "
            "per input pair."
        )

    input_set = set(input_keys)
    output_set = set(output_keys)
    missing = input_set - output_set
    unexpected = output_set - input_set
    faults: list[str] = []
    if missing:
        faults.append(f"missing judgements for {_pair_preview(missing)}")
    if unexpected:
        faults.append(f"unexpected judgements for {_pair_preview(unexpected)}")
    if faults:
        raise ValueError(
            "MatcherScore requires a one-to-one pair/judgement mapping; " + "; ".join(faults) + "."
        )

    by_pair = dict(zip(output_keys, materialized, strict=True))
    rows: list[PairRow[SchemaT]] = []
    for row in pairs.rows:
        judgement = by_pair[_unordered_pair_key(row.left_id, row.right_id)]
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


def _duplicates(keys: Iterable[tuple[str, str]]) -> set[tuple[str, str]]:
    """Return identities occurring more than once."""
    seen: set[tuple[str, str]] = set()
    duplicates: set[tuple[str, str]] = set()
    for key in keys:
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return duplicates


def _pair_preview(keys: Iterable[tuple[str, str]]) -> str:
    """Render a deterministic bounded identity preview for contract errors."""
    return ", ".join(repr(key) for key in sorted(keys)[:3])


class BlockerSource(Source[SchemaT], Generic[SchemaT]):
    """:class:`~langres.core.op.Source` adapter over a legacy :class:`~langres.core.blocker.Blocker`.

    Bridges ``blocker.stream(records) -> Iterator[ERCandidate]`` to a
    :class:`~langres.core.pairs.Pairs` via
    :meth:`~langres.core.pairs.Pairs.from_candidates` (the blocker's
    ``similarity_score`` lands as an *unscored* row score, ``score_type is None``
    — a blocker similarity, never a judge score).

    ``prepare(records)`` owns the vector-index lifecycle for every nested
    ``VectorBlocker``: identical corpora reuse the current index and changed
    corpora rebuild before ``forward`` streams candidates.
    """

    def __init__(self, blocker: Blocker[SchemaT]) -> None:
        """Wrap ``blocker``.

        Args:
            blocker: The legacy blocker to adapt. Any :class:`Blocker` works;
                vector indexes are prepared automatically before execution.
        """
        self.blocker = blocker
        self._prepared_corpora: dict[int, list[str]] = {}

    @property
    def schema(self) -> type[BaseModel] | None:
        """The wrapped blocker's declarative schema, when available."""
        return self.blocker.schema

    def prepare(self, records: Records) -> None:
        """Build vector indexes in the wrapped blocker tree for ``records``.

        Identical corpora reuse their current index; changed corpora rebuild
        before streaming. Detection reads the import-light ``type_name`` rather
        than importing semantic backends.
        """

        def vector_blockers(blocker: object) -> Iterable[Any]:
            if getattr(blocker, "type_name", None) == "vector_blocker":
                yield blocker
            for child in getattr(blocker, "children", ()):
                yield from vector_blockers(child)

        for blocker in vector_blockers(self.blocker):
            entities = [blocker.schema_factory(record) for record in records]
            texts = [blocker.text_field_extractor(entity) for entity in entities]
            index = blocker.vector_index
            indexed_texts = getattr(index, "_corpus_texts", None)
            if indexed_texts is None:
                indexed_texts = self._prepared_corpora.get(id(blocker))
            if blocker._index_is_built() and indexed_texts == texts:
                continue
            index.create_index(texts)
            # Some lightweight/custom indexes record only cardinality. The
            # durable Source remembers exact input so same-sized changed corpora
            # still rebuild and identical explicit executions reuse.
            self._prepared_corpora[id(blocker)] = list(texts)

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

    @property
    def spend_monitor(self) -> SpendMonitor | None:
        """The wrapped matcher's ledger, or ``None`` before spend binding."""
        from langres.core.spend_cap import SpendCappedMatcher

        if isinstance(self.matcher, SpendCappedMatcher):
            return self.matcher.monitor
        return None

    def with_matcher(self, matcher: Matcher[SchemaT]) -> "MatcherScore[SchemaT]":
        """Clone this exact adapter class while replacing only its matcher.

        A shallow clone preserves state carried by a ``MatcherScore`` subclass,
        which is required both for spend binding and per-call logging. Rebuilding
        a subclass as the base class would silently discard that state.
        """
        rebound = copy(self)
        rebound.matcher = matcher
        return rebound

    def bind_spend_monitor(self, monitor: SpendMonitor) -> "MatcherScore[SchemaT]":
        """Return this score metered by ``monitor``, preserving exact subclass state."""
        from langres.core.spend_cap import SpendCappedMatcher

        if isinstance(self.matcher, SpendCappedMatcher):
            if self.matcher.monitor is not monitor:
                raise ValueError(
                    "MatcherScore is already bound to a different SpendMonitor. "
                    "Pass that monitor to from_topology(monitor=...), or provide the raw matcher."
                )
            return self
        return self.with_matcher(SpendCappedMatcher(self.matcher, monitor=monitor))


class CalibratorScore(Score[SchemaT], Generic[SchemaT]):
    """Apply a fitted score-to-probability calibrator as an ordinary Score."""

    def __init__(self, calibrator: CalibratorFitMixin) -> None:
        super().__init__(scope="pair", out_space="calibrated_prob")
        self.calibrator = calibrator

    def forward(self, pairs: Pairs[SchemaT]) -> Pairs[SchemaT]:
        """Calibrate scored rows; pass deciders and unscored rows through."""
        rows: list[PairRow[SchemaT]] = []
        for row in pairs.rows:
            if row.score_type is None or row.score is None:
                rows.append(row)
                continue
            calibrated = self.calibrator.transform([row.score])[0]
            rows.append(
                row.model_copy(
                    update={
                        "score": calibrated,
                        "score_type": "calibrated_prob",
                        "provenance": {
                            **row.provenance,
                            "calibration": {
                                "method": getattr(self.calibrator, "method", None),
                                "raw_score": row.score,
                            },
                        },
                    }
                )
            )
        return Pairs(store=pairs.store, rows=rows)


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
    "CalibratorScore",
    "CanonicalizeFinalize",
    "ClustererStage",
    "ComparatorScore",
    "GroupwiseMatcherScore",
    "MatcherScore",
]
