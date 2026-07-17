"""``Pairs`` — the ONE id-referenced carrier for a set of entity pairs.

Two datums the legacy contracts spell twice are unified here: a
:class:`~langres.core.models.ERCandidate` carries a blocker's
``similarity_score`` and a :class:`~langres.core.models.PairwiseJudgement`
carries a judge's ``score`` — both ``float | None`` in ``[0, 1]``, the same kind
of number written in two places. ``Pairs`` folds them into ONE ``score`` column
whose meaning is disambiguated by ``score_type``:

- ``score_type is None`` — **blocked, not yet scored.** ``score`` (if any) is the
  *blocker's* similarity. This is a lifecycle state, NOT a widening of the
  frozen 7-value :data:`~langres.core.score_type.ScoreType`.
- ``score_type`` set — **scored.** ``score`` is the *judge's* score of that
  family.

That single rule (``F-W1a``) is why the bridges below never let a judge score
masquerade as a blocker similarity or vice-versa.

Design:

- **Ids, not inline entities.** A :class:`PairRow` references entities by id;
  the entities live once in the owning :class:`Pairs` ``store``. ``row.left`` /
  ``row.right`` materialize the typed entity on access (so ``row.left.name``
  still works and stays typed), backed by a ``_store`` private attr that is
  **never serialized**.
- **The store binds at construction.** :class:`Pairs` injects itself into every
  row's ``_store`` in a post-init validator, so a row taken out of a ``Pairs``
  and copied (``row.model_copy(update=...)``) keeps working — the binding does
  not depend on iterating the ``Pairs`` (``F-W1b``).

Import discipline: this is a strict leaf. Its *eager* imports are stdlib +
pydantic + the two sibling leaves (``feature``, ``score_type``). The legacy
``ERCandidate`` / ``PairwiseJudgement`` types it bridges to are referenced under
``TYPE_CHECKING`` (annotations) and imported lazily inside the bridge bodies, so
``pairs`` adds no eager edge and stays off every import cycle. ``models`` /
``matcher`` / ``blocker`` may import *from* ``pairs`` later; ``pairs`` never
eagerly imports back.
"""

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from langres.core.feature import ComparisonVector
from langres.core.score_type import ConfidenceSource, ScoreType

if TYPE_CHECKING:
    from langres.core.models import ERCandidate, PairwiseJudgement

# Bound to BaseModel for Pydantic compatibility (mirrors models.SchemaT). Entities
# are addressed by their ``id`` (EntityProtocol) — read via ``# type: ignore``
# where the bound cannot express the ``id`` attribute, exactly as the Matcher /
# Blocker contracts already do.
SchemaT = TypeVar("SchemaT", bound=BaseModel)

#: A read-only id -> entity map. The shape a :class:`Pairs` ``store`` satisfies
#: and a :class:`PairRow` reads through. An alias (zero runtime dependency) so
#: downstream code can name the contract without reaching into ``Pairs``.
RecordStore: TypeAlias = Mapping[str, SchemaT]


class PairRow(BaseModel, Generic[SchemaT]):
    """One id-referenced pair: the unified row of a :class:`Pairs`.

    Carries everything both legacy contracts carried, keyed by id rather than by
    inline entity. The ONE ``score`` column subsumes both the blocker's
    ``similarity_score`` and a judge's ``score``; ``score_type`` disambiguates
    which (``None`` = blocked, not yet scored — see the module docstring).

    ``left`` / ``right`` are not stored fields: they materialize the entity from
    the owning :class:`Pairs`'s ``store`` on access, so a row stays small and
    serialization-safe while ``row.left.name`` still reads a typed entity.

    Attributes:
        left_id: Id of the left entity (resolved against the store by ``left``).
        right_id: Id of the right entity (resolved by ``right``).
        blocker_name: Name of the blocker that generated this pair.
        score: The ONE score column. ``None`` or a blocker similarity while
            ``score_type is None``; a judge score once ``score_type`` is set.
        score_type: The score family, or ``None`` for "blocked, not yet scored".
            ``None`` is a lifecycle state, never an 8th :data:`ScoreType` value.
        decision: A decider's explicit verdict, else ``None`` (see
            :func:`~langres.core.models.predicted_match`).
        confidence: Optional orthogonal "how sure am I" in ``[0, 1]``.
        confidence_source: Provenance of ``confidence`` (default ``"none"``).
        decision_step: Which logic branch last wrote this row; ``""`` for a
            freshly-blocked, unscored row.
        reasoning: Optional natural-language explanation.
        comparison: Per-feature ``ComparisonVector`` for the two-phase pipeline
            (as on ``ERCandidate``), else ``None``.
        provenance: Free-form audit trail (default empty).
    """

    left_id: str
    right_id: str
    blocker_name: str
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    score_type: ScoreType | None = None
    decision: bool | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_source: ConfidenceSource = "none"
    decision_step: str = ""
    reasoning: str | None = None
    comparison: ComparisonVector | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    # Bound by the owning Pairs at construction (see Pairs._bind_store). NEVER a
    # field, so it is never serialized and never revalidated; survives
    # ``model_copy`` because pydantic copies ``__pydantic_private__`` (F-W1b).
    _store: RecordStore[SchemaT] | None = PrivateAttr(default=None)

    def _bind(self, store: "RecordStore[SchemaT]") -> None:
        """Bind this row to its owning store (called by :class:`Pairs`)."""
        self._store = store

    def _entity(self, entity_id: str) -> SchemaT:
        if self._store is None:
            raise RuntimeError(
                "PairRow is not bound to a record store; entities are only "
                "reachable on a row obtained from a Pairs (which binds the store "
                "at construction). Build one via Pairs(...) / Pairs.from_candidates."
            )
        return self._store[entity_id]

    @property
    def left(self) -> SchemaT:
        """The left entity, materialized from the owning store (typed)."""
        return self._entity(self.left_id)

    @property
    def right(self) -> SchemaT:
        """The right entity, materialized from the owning store (typed)."""
        return self._entity(self.right_id)

    @property
    def is_abstain(self) -> bool:
        """``True`` iff this row carries no actionable judge signal.

        Delegates to :attr:`~langres.core.models.PairwiseJudgement.is_abstain`
        for a SCORED row. An UNSCORED row (``score_type is None``) has no judge
        verdict at all — its ``score`` is a blocker similarity, not a decision
        (``F-W1a``) — so it abstains iff it also carries no explicit ``decision``,
        which is exactly the models-layer formula with the judge score read as
        ``None``.
        """
        if self.score_type is None:
            return self.decision is None
        return self.to_judgement().is_abstain

    def predicted_match(self, threshold: float) -> bool | None:
        """Is this pair a predicted match? Delegates to models ``predicted_match``.

        ``F-W1a``: only a SCORED row's ``score`` is a judge score. An UNSCORED
        row's ``score`` is a blocker similarity, never a match verdict, so the
        threshold must not be applied to it — only an explicit ``decision`` can
        make an unscored row a match (``decision`` still wins, exactly as the
        models-layer rule, with the judge score read as ``None``).
        """
        from langres.core.models import predicted_match as _predicted_match

        if self.score_type is None:
            return self.decision
        return _predicted_match(self.to_judgement(), threshold)

    def to_judgement(self) -> "PairwiseJudgement":
        """Project a SCORED row to a :class:`~langres.core.models.PairwiseJudgement`.

        Ids-only (no entities). A judgement's ``score_type`` is required, so this
        refuses an unscored row: ``score_type is None`` means "blocked, not yet
        scored", which has no judgement to emit.

        Raises:
            ValueError: If ``score_type is None``.
        """
        from langres.core.models import PairwiseJudgement

        if self.score_type is None:
            raise ValueError(
                "to_judgement() requires a scored row: score_type is None means "
                "'blocked, not yet scored', which has no judgement to project. "
                "Score the row first."
            )
        return PairwiseJudgement(
            left_id=self.left_id,
            right_id=self.right_id,
            decision=self.decision,
            score=self.score,
            score_type=self.score_type,
            confidence=self.confidence,
            confidence_source=self.confidence_source,
            decision_step=self.decision_step,
            reasoning=self.reasoning,
            provenance=self.provenance,
        )


class Pairs(BaseModel, Generic[SchemaT]):
    """The ONE carrier: an owned entity ``store`` plus id-referenced ``rows``.

    Owns the entities exactly once (``store``); every :class:`PairRow` references
    them by id. Iterating a ``Pairs`` yields its rows, each already bound to the
    store (bound at construction, not on iterate — ``F-W1b``).

    Attributes:
        store: The id -> entity map the rows resolve against. Entities are held
            by reference (never deep-copied), so ``from_candidates`` does not
            duplicate a blocker's entities.
        rows: The pair rows.
    """

    store: dict[str, SchemaT]
    rows: list[PairRow[SchemaT]]

    @model_validator(mode="after")
    def _bind_store(self) -> "Pairs[SchemaT]":
        """Bind every row to this Pairs's store, at construction (F-W1b)."""
        for row in self.rows:
            row._bind(self.store)
        return self

    def __iter__(self) -> Iterator[PairRow[SchemaT]]:  # type: ignore[override]
        """Iterate the rows (each bound to the store)."""
        return iter(self.rows)

    def __len__(self) -> int:
        """Number of rows."""
        return len(self.rows)

    @classmethod
    def from_candidates(cls, candidates: "Sequence[ERCandidate[SchemaT]]") -> "Pairs[SchemaT]":
        """Fold legacy inline-entity ``ERCandidate``s into id-rows + a store.

        Each candidate's entities are placed in the store **by reference** (never
        deep-copied — ``F-W1c``), and its ``similarity_score`` becomes the row's
        ``score`` with ``score_type=None`` (blocked, not yet scored). When two
        candidates share an entity id, the first occurrence wins (they are the
        same entity by id).

        Args:
            candidates: Legacy ``ERCandidate`` pairs (typically a blocker's
                output).

        Returns:
            A ``Pairs`` referencing the same entity objects the candidates held.
        """
        store: dict[str, SchemaT] = {}
        rows: list[PairRow[SchemaT]] = []
        for candidate in candidates:
            left = candidate.left
            right = candidate.right
            left_id: str = left.id  # type: ignore[attr-defined]
            right_id: str = right.id  # type: ignore[attr-defined]
            store.setdefault(left_id, left)
            store.setdefault(right_id, right)
            rows.append(
                PairRow(
                    left_id=left_id,
                    right_id=right_id,
                    blocker_name=candidate.blocker_name,
                    score=candidate.similarity_score,
                    score_type=None,
                    comparison=candidate.comparison,
                )
            )
        return cls(store=store, rows=rows)

    def to_candidates(self) -> "list[ERCandidate[SchemaT]]":
        """Project back to legacy inline-entity ``ERCandidate`` form.

        ``F-W1a``: ``similarity_score`` is the *blocker's* input datum, so only an
        UNSCORED row's ``score`` (its blocker similarity) flows into it. A SCORED
        row's ``score`` is a *judge* score and must NOT masquerade as a blocker
        similarity, so its ``similarity_score`` is emitted as ``None``.

        Entities are referenced from the store (never deep-copied).

        Returns:
            One ``ERCandidate`` per row, entities taken from the store.
        """
        from langres.core.models import ERCandidate

        candidates: list[ERCandidate[SchemaT]] = []
        for row in self.rows:
            similarity = row.score if row.score_type is None else None
            candidates.append(
                ERCandidate(
                    left=self.store[row.left_id],
                    right=self.store[row.right_id],
                    blocker_name=row.blocker_name,
                    similarity_score=similarity,
                    comparison=row.comparison,
                )
            )
        return candidates
