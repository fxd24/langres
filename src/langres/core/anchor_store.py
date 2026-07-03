"""Incremental single-record assignment against a persisted anchor set (M5 / W2.2).

A batch :meth:`Resolver.resolve` answers "given all these records at once, which
group together?" — but it returns only *id-sets*: positional, without stable
entity ids, and with **singletons dropped** (the Clusterer emits only connected
components with an edge; see ``clusterer.py`` / ``resolver.py``). That output
cannot answer the *incremental* question "here is one NEW record — which existing
entity does it belong to, or is it new?", because it retains no stable ids, no
records, and structurally cannot represent a record that matched nothing.

:class:`AnchorStore` fills that gap. It is built by a **dedicated pass** over a
prior batch (:meth:`AnchorStore.build`) that:

- runs the resolver once (which also builds any vector index in place), then
- enumerates **every** input record — including the ones the clusterer dropped
  as singletons — and mints each a **stable, monotonic entity id** from an
  append-only allocator (ids are minted by the store, never derived from list
  position, since ``resolve()``'s order is non-deterministic).

:meth:`AnchorStore.assign` then answers the incremental question for one new
record by reusing the resolver's *existing* seams — the vector index's
single-record kNN (or all-pairs when there is no index) for candidate anchors,
and the very same Comparator + Module judge the batch pipeline uses — and
returns a :class:`ClusterDelta` carrying a **stable** entity id: ``link`` to an
existing entity, or ``new`` with a freshly minted id.

The store round-trips through the same config-registry artifact seam as the
Resolver (no pickle): :meth:`save` delegates the pipeline to
:meth:`Resolver.save` (including a built FAISS index's sidecar state) and writes
a small ``anchor_store.json`` for the id bookkeeping; :meth:`load` reverses it in
a fresh process.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from langres.core.models import ERCandidate, PairwiseJudgement

if TYPE_CHECKING:
    from langres.core.resolver import Resolver

logger = logging.getLogger(__name__)

#: Bump when the ``anchor_store.json`` layout changes incompatibly.
ANCHOR_STORE_VERSION = "1"

_MANIFEST_FILENAME = "anchor_store.json"
_RESOLVER_SUBDIR = "resolver"

#: The kinds of change an :meth:`AnchorStore.assign` can report. ``new`` and
#: ``link`` are the only two W2.2 produces (single-record assignment); the rest
#: are **reserved** so the wider entity-maintenance surface (W2.4 flywheel / M6)
#: can add them without reshaping this contract:
#: ``merge`` (fold two existing entities into one), ``split`` (break one entity
#: apart), ``reject`` (an assignment a reviewer overturned).
ClusterDeltaType = Literal["new", "link", "merge", "split", "reject"]


class ClusterDelta(BaseModel):
    """The outcome of assigning one record against an :class:`AnchorStore`.

    Attributes:
        type: What happened — ``"new"`` (a fresh entity was minted) or
            ``"link"`` (the record attached to an existing entity). ``"merge"`` /
            ``"split"`` / ``"reject"`` are reserved (see :data:`ClusterDeltaType`).
        record_id: The assigned record's id (``entity.id`` after normalization).
        entity_id: The **stable** entity id the record now belongs to. For
            ``new`` this is freshly minted; for ``link`` it is the existing id.
            An assigned record's ``entity_id`` never changes on later assigns
            (the allocator is append-only), which is the guarantee incremental
            callers rely on.
        matched_anchor_ids: Anchor record ids that cleared the match threshold —
            the evidence behind a ``link`` (empty for ``new``). More than one
            *distinct* entity among these is a merge signal; W2.2 links to the
            lowest-ordinal entity and leaves ``merge`` to a later milestone.
        score: The best matching score observed across the judged candidates
            (observability only); ``None`` when there were no candidates.
        reasoning: Optional human-readable note about the decision.
    """

    type: ClusterDeltaType
    record_id: str
    entity_id: str
    matched_anchor_ids: list[str] = Field(default_factory=list)
    score: float | None = None
    reasoning: str | None = None


class AnchorStoreManifest(BaseModel):
    """Typed shape of ``anchor_store.json`` — the store's id bookkeeping.

    The heavy pipeline (blocker/comparator/module/clusterer, plus a built vector
    index's sidecar state) is NOT stored here; it round-trips through the nested
    :meth:`Resolver.save` artifact under the ``resolver/`` subdirectory.

    Attributes:
        store_version: Layout version (see :data:`ANCHOR_STORE_VERSION`).
        entity_prefix: Prefix for minted entity ids (e.g. ``"e"`` -> ``"e0"``).
        next_ordinal: The next unused allocator ordinal (append-only cursor).
        anchor_ids: Anchor record ids in **corpus order** — index position ``i``
            in a vector index maps back to ``anchor_ids[i]``.
        records: ``record_id -> raw record dict`` for every anchor (needed to
            reconstruct anchor entities for the judge at assign time).
        assignments: ``record_id -> entity_id`` for every known record.
    """

    store_version: str
    entity_prefix: str
    next_ordinal: int
    anchor_ids: list[str]
    records: dict[str, dict[str, Any]]
    assignments: dict[str, str]


class AnchorStore:
    """A persisted anchor set that answers incremental single-record assignment.

    Build one from a prior batch, then assign new records against it:

        resolver = Resolver.from_schema(CompanySchema, judge="string")
        store = AnchorStore.build(resolver, records)   # dedicated pass
        delta = store.assign(new_record)               # -> ClusterDelta
        store.save("artifacts/anchors")
        reloaded = AnchorStore.load("artifacts/anchors")

    The store is a thin, composable unit *around* a Resolver — it owns only the
    id bookkeeping (records, ``record_id -> entity_id``, the monotonic allocator)
    and delegates all matching to the resolver's existing seams. It does not
    reach into the Resolver's internals beyond the public blocker/comparator/
    module/clusterer slots.
    """

    def __init__(
        self,
        resolver: "Resolver",
        records: dict[str, dict[str, Any]],
        assignments: dict[str, str],
        anchor_ids: list[str],
        next_ordinal: int,
        entity_prefix: str = "e",
    ) -> None:
        """Construct a store from already-computed bookkeeping.

        Prefer :meth:`build` (from a batch) or :meth:`load` (from disk); this
        constructor is the shared low-level entry both funnel through.

        Args:
            resolver: The built pipeline whose blocker/comparator/module/
                clusterer seams :meth:`assign` reuses.
            records: ``record_id -> raw record dict`` for every anchor.
            assignments: ``record_id -> entity_id`` for every known record.
            anchor_ids: Anchor record ids in corpus (index-position) order.
            next_ordinal: The next unused allocator ordinal.
            entity_prefix: Prefix for minted entity ids.
        """
        self._resolver = resolver
        self._records = records
        self._assignments = assignments
        self._anchor_ids = anchor_ids
        self._next_ordinal = next_ordinal
        self._entity_prefix = entity_prefix

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        resolver: "Resolver",
        records: list[dict[str, Any]],
        *,
        entity_prefix: str = "e",
    ) -> "AnchorStore":
        """Build an anchor store from a batch, minting a stable id for EVERY record.

        Runs ``resolver.resolve(records)`` once (which also builds any vector
        index in place, so :meth:`assign` can search it without re-embedding),
        then walks the records in input order minting entity ids: all members of
        one returned multi-record cluster share one freshly minted id, and every
        record the clusterer dropped as a singleton gets its **own** id. This
        singleton coverage is the correctness core — the most common incremental
        case is "matches nothing -> new", which requires the store to know about
        every prior record, including the ones ``resolve()`` never surfaces. The
        pass is clusterer-agnostic: it enumerates the record-id universe itself
        rather than trusting the clusterer to report singletons (no clusterer
        does).

        Args:
            resolver: A built, serializable Resolver (e.g. from
                ``Resolver.from_schema``). Records are fed as raw dicts, exactly
                as ``resolve()`` accepts them.
            records: The batch of raw record dicts to anchor on. Ids must be
                unique (the resolver's schema ``id`` field).
            entity_prefix: Prefix for minted entity ids (default ``"e"``).

        Returns:
            An :class:`AnchorStore` with one stable entity id per input record.
        """
        clusters = resolver.resolve(records)

        factory = resolver.blocker.schema_factory
        anchor_ids = [factory(record).id for record in records]  # type: ignore[attr-defined]
        records_by_id = dict(zip(anchor_ids, records))

        id_to_cluster: dict[str, frozenset[str]] = {}
        for cluster in clusters:
            frozen = frozenset(cluster)
            for record_id in cluster:
                id_to_cluster[record_id] = frozen

        assignments: dict[str, str] = {}
        entity_of_cluster: dict[frozenset[str], str] = {}
        ordinal = 0

        for record_id in anchor_ids:
            cluster = id_to_cluster.get(record_id)
            if cluster is None:
                # A record no threshold-passing pair covered: its own singleton
                # entity (NOT dropped as the clusterer would).
                assignments[record_id] = f"{entity_prefix}{ordinal}"
                ordinal += 1
            else:
                if cluster not in entity_of_cluster:
                    entity_of_cluster[cluster] = f"{entity_prefix}{ordinal}"
                    ordinal += 1
                assignments[record_id] = entity_of_cluster[cluster]

        logger.info(
            "Built AnchorStore: %d records, %d entities (%d multi-record clusters)",
            len(anchor_ids),
            ordinal,
            len(entity_of_cluster),
        )
        return cls(
            resolver=resolver,
            records=records_by_id,
            assignments=assignments,
            anchor_ids=anchor_ids,
            next_ordinal=ordinal,
            entity_prefix=entity_prefix,
        )

    # ------------------------------------------------------------------
    # Incremental assignment
    # ------------------------------------------------------------------

    def assign(self, record: dict[str, Any]) -> ClusterDelta:
        """Assign one new record to an existing entity, or mint a new one.

        Generates candidate anchors (the vector index's single-record kNN when
        the resolver blocks on a vector index, else all anchors), runs the
        resolver's own Comparator + Module judge on the ``(record, anchor)``
        pairs, and links the record to the matched entity — or mints a new one
        when nothing clears the clusterer threshold. The record's assignment is
        recorded (append-only), so assigning a record whose id is already known
        returns its existing entity id unchanged (idempotent, never renumbered).

        Args:
            record: A raw record dict, same shape as :meth:`build` / ``resolve``.

        Returns:
            A :class:`ClusterDelta` with a stable ``entity_id`` and the delta
            ``type`` (``"link"`` or ``"new"``).
        """
        entity = self._resolver.blocker.schema_factory(record)  # type: ignore[attr-defined]
        record_id: str = entity.id

        # Idempotent on record id: a record we have already assigned keeps its
        # entity id (append-only allocator never renumbers a prior assignment).
        if record_id in self._assignments:
            return ClusterDelta(
                type="link",
                record_id=record_id,
                entity_id=self._assignments[record_id],
                reasoning="record id already assigned",
            )

        anchor_ids = self._candidate_anchor_ids(entity)
        candidates = [self._candidate(entity, anchor_id) for anchor_id in anchor_ids]
        judgements = self._judge(candidates)

        threshold = self._resolver.clusterer.threshold
        matched_anchor_ids: list[str] = []
        best_score: float | None = None
        for judgement in judgements:
            best_score = (
                judgement.score if best_score is None else max(best_score, judgement.score)
            )
            if judgement.score >= threshold:
                anchor_id = (
                    judgement.right_id if judgement.left_id == record_id else judgement.left_id
                )
                matched_anchor_ids.append(anchor_id)

        # Distinct matched entities, in first-seen order.
        matched_entity_ids: list[str] = []
        for anchor_id in matched_anchor_ids:
            entity_id = self._assignments.get(anchor_id)
            if entity_id is not None and entity_id not in matched_entity_ids:
                matched_entity_ids.append(entity_id)

        if matched_entity_ids:
            # >1 distinct entity is a merge signal; W2.2 links to the
            # lowest-ordinal (oldest) entity and reserves merge for later.
            resolved_entity_id = min(matched_entity_ids, key=self._ordinal_of)
            delta_type: ClusterDeltaType = "link"
            reasoning = (
                f"linked to existing entity via {len(matched_anchor_ids)} matched anchor(s)"
            )
        else:
            resolved_entity_id = self._mint()
            delta_type = "new"
            reasoning = "no anchor cleared the match threshold"

        # Register (append-only): makes assign idempotent on this record id.
        self._assignments[record_id] = resolved_entity_id
        self._records.setdefault(record_id, record)

        return ClusterDelta(
            type=delta_type,
            record_id=record_id,
            entity_id=resolved_entity_id,
            matched_anchor_ids=matched_anchor_ids,
            score=best_score,
            reasoning=reasoning,
        )

    def entity_id_of(self, record_id: str) -> str | None:
        """Return the stable entity id assigned to ``record_id``, or ``None``."""
        return self._assignments.get(record_id)

    @property
    def entity_ids(self) -> set[str]:
        """The set of distinct entity ids currently known to the store."""
        return set(self._assignments.values())

    def _candidate_anchor_ids(self, entity: Any) -> list[str]:
        """Anchor ids to judge ``entity`` against: kNN when indexed, else all.

        Reuses the vector index's single-record ``search`` (the thin
        new-record-against-built-corpus path) when the resolver blocks on a
        vector index; falls back to every anchor when there is no index (e.g. an
        ``AllPairsBlocker``), which is the correct all-pairs behaviour for one
        new record against the anchor set.
        """
        blocker = self._resolver.blocker
        if getattr(blocker, "type_name", None) != "vector_blocker" or not self._anchor_ids:
            return list(self._anchor_ids)

        text = blocker.text_field_extractor(entity)  # type: ignore[attr-defined]
        k = min(blocker.k_neighbors, len(self._anchor_ids))  # type: ignore[attr-defined]
        _distances, indices = blocker.vector_index.search(text, k)  # type: ignore[attr-defined]
        anchor_ids: list[str] = []
        for raw_index in indices.tolist():
            index = int(raw_index)
            # FAISS pads with -1 when fewer than k neighbours exist.
            if 0 <= index < len(self._anchor_ids):
                anchor_ids.append(self._anchor_ids[index])
        return anchor_ids

    def _candidate(self, entity: Any, anchor_id: str) -> ERCandidate[Any]:
        """Build a ``(new_record, anchor)`` candidate pair for the judge."""
        anchor_entity = self._resolver.blocker.schema_factory(self._records[anchor_id])  # type: ignore[attr-defined]
        return ERCandidate(left=entity, right=anchor_entity, blocker_name="anchor_store")

    def _judge(self, candidates: list[ERCandidate[Any]]) -> list[PairwiseJudgement]:
        """Score candidates with the resolver's SAME comparator + module judge."""
        comparator = self._resolver.comparator
        if comparator is not None:
            candidates = [
                candidate.model_copy(
                    update={"comparison": comparator.compare(candidate.left, candidate.right)}
                )
                for candidate in candidates
            ]
        return list(self._resolver.module.forward(iter(candidates)))

    def _mint(self) -> str:
        """Mint the next stable entity id from the append-only allocator."""
        entity_id = f"{self._entity_prefix}{self._next_ordinal}"
        self._next_ordinal += 1
        return entity_id

    def _ordinal_of(self, entity_id: str) -> int:
        """Parse an entity id's ordinal for oldest-wins tie-breaking."""
        return int(entity_id[len(self._entity_prefix) :])

    # ------------------------------------------------------------------
    # Persistence (config-registry seam; no pickle)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the store to ``path`` as a self-describing artifact.

        Delegates the pipeline to :meth:`Resolver.save` under a ``resolver/``
        subdirectory (which persists a built vector index's sidecar state), and
        writes ``anchor_store.json`` for the id bookkeeping. No pickle, no code
        execution on load.

        Args:
            path: Directory to write the artifact into (created if absent).
        """
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._resolver.save(out_dir / _RESOLVER_SUBDIR)
        manifest = AnchorStoreManifest(
            store_version=ANCHOR_STORE_VERSION,
            entity_prefix=self._entity_prefix,
            next_ordinal=self._next_ordinal,
            anchor_ids=self._anchor_ids,
            records=self._records,
            assignments=self._assignments,
        )
        (out_dir / _MANIFEST_FILENAME).write_text(manifest.model_dump_json(indent=2))
        logger.info("Saved AnchorStore artifact to %s", out_dir)

    @classmethod
    def load(cls, path: str | Path) -> "AnchorStore":
        """Reconstruct a store written by :meth:`save`, pipeline and all.

        Rebuilds the Resolver from its artifact (restoring a built vector
        index's state, so :meth:`assign` can search it) and rehydrates the id
        bookkeeping from ``anchor_store.json``.

        Args:
            path: Directory containing ``anchor_store.json`` and ``resolver/``.

        Returns:
            An :class:`AnchorStore` equivalent to the one that was saved.
        """
        from langres.core.resolver import Resolver

        in_dir = Path(path)
        manifest = AnchorStoreManifest.model_validate_json(
            (in_dir / _MANIFEST_FILENAME).read_text()
        )
        resolver = Resolver.load(in_dir / _RESOLVER_SUBDIR)
        return cls(
            resolver=resolver,
            records=manifest.records,
            assignments=manifest.assignments,
            anchor_ids=manifest.anchor_ids,
            next_ordinal=manifest.next_ordinal,
            entity_prefix=manifest.entity_prefix,
        )
