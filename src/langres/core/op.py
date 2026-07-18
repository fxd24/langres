"""The one component contract: an ``Op`` over a scored relation, and the stages
that bound it (``Source`` / ``ClusterStage`` / ``Finalize``), plus ``Sequential``
— the ordered pipeline whose wiring is checked at construction.

**What this is.** ``langres``'s components are named after *positions in a
pipeline* — a blocker, a matcher, a clusterer. This module says position is not a
type: a blocker, a matcher, a reranker and a threshold are all the *same*
operation — rescore some rows, then keep some rows — at different settings. So
there is ONE component type, :class:`Op`, with two roles:

- :class:`Score` — "same rows, new scores" (rescore in place). A blocker's
  similarity, a comparator's per-feature vector, and a matcher's probability are
  all Scores; they differ only in the score family they *produce* (``out_space``).
- :class:`Select` — "same scores, fewer rows" (keep a subset). Threshold
  matching, top-k blocking, entity linking, 1-to-1 assignment and clustering are
  all Selects; they differ only in the :class:`Feasible` class they keep the
  answer inside.

The carriers differ at the two ends of the pipeline, so the source and the two
exits are separate contracts, not ``Op``\\ s: a :class:`Source` turns records
into pairs, a :class:`ClusterStage` turns pairs into clusters (the phase-1 exit),
and a :class:`Finalize` verifies or canonicalizes clusters (the phase-2 exit).

**Naming.** No mathematical symbols appear here on purpose — the algebra and its
notation live in ``docs/THEORY.md``; this is its ASCII realization. Read the
theory for *why* these are one operation (§2–§8); read this for *what* to build
against.

**Import discipline.** A strict leaf: it imports only two sibling core leaves
(:mod:`langres.core.pairs`, :mod:`langres.core.score_type`) plus stdlib /
pydantic / typing. It imports nothing from ``matcher`` / ``blocker`` /
``comparator`` / ``clusterer`` / ``resolver`` — those adopt *this* contract in a
later wave, so the edge runs one way, into this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Literal, TypeAlias, TypeVar, get_args

from pydantic import BaseModel

from langres.core.pairs import PairRow, Pairs
from langres.core.score_type import ScoreType

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# --------------------------------------------------------------------------------------
# Carriers — the types that flow between stages. Thin aliases over what the spine
# already uses; this module invents no new heavy model.
# --------------------------------------------------------------------------------------

#: The normalized record input the spine already consumes (``ERModel.resolve``
#: takes ``list[Any]`` — raw dicts or normalized entities). A :class:`Source`
#: turns this into a :class:`~langres.core.pairs.Pairs`.
Records: TypeAlias = list[Any]

#: A set of id clusters — the phase-1 exit carrier, exactly what the Clusterer
#: already returns (``list[set[str]]`` of entity ids).
Clusters: TypeAlias = list[set[str]]

#: A single canonicalized ("golden") record — the phase-2 exit carrier a
#: :class:`Finalize` may fuse a cluster down to. A plain record (Pydantic model).
GoldenRecord: TypeAlias = BaseModel

#: Where a :class:`Score` declares its output lives: ``"pair"`` / ``"group"`` /
#: ``"global"``. Kept identical to :attr:`Score.scope`.
Scope: TypeAlias = Literal["pair", "group", "global"]

#: The score family a :class:`Score` produces: one of the frozen
#: :data:`~langres.core.score_type.ScoreType` families, ``"vector"`` when it
#: emits a :class:`~langres.core.feature.ComparisonVector` (the old Comparator
#: role — a vector score, ``S = R^d``, which is not orderable), or ``"unknown"``
#: — a scalar-family placeholder for a Score that produces an *orderable* scalar
#: whose exact family is not (yet) pinned. Unlike ``"vector"``, ``"unknown"`` is
#: a scalar, so a :class:`Select` may follow it (selection is defined on any
#: scalar order); it is the honest sentinel for "some scalar score" — never a
#: real :data:`ScoreType` value borrowed as a stand-in.
OutSpace: TypeAlias = ScoreType | Literal["vector", "unknown"]

_VALID_SCOPES: frozenset[str] = frozenset(get_args(Scope))
_SCORE_FAMILIES: frozenset[str] = frozenset(get_args(ScoreType))
#: The scalar-family sentinel: an orderable score whose family is not (yet)
#: pinned. NOT a vector, so it is admissible upstream of a :class:`Select`.
_UNKNOWN_OUT_SPACE: str = "unknown"
_VALID_OUT_SPACES: frozenset[str] = _SCORE_FAMILIES | {"vector", _UNKNOWN_OUT_SPACE}

#: The heuristics a raw ``Select(CLUSTERING)`` or a :class:`ClusterStage` may
#: name. Correlation clustering with real weights is not approximable, so there
#: is no exact algorithm to default to — the caller names one.
_CLUSTERING_ALGORITHMS: frozenset[str] = frozenset({"transitive_closure", "pivot"})


def _validate_clustering_algorithm(algorithm: str) -> None:
    """Raise if ``algorithm`` is not a known clustering heuristic.

    Shared by the two stages that name a clustering heuristic — a raw
    ``Select(Feasible.CLUSTERING)`` and a :class:`ClusterStage` — so both reject
    the same unknown values with the same problem + fix message.

    Raises:
        ValueError: If ``algorithm`` is not one of :data:`_CLUSTERING_ALGORITHMS`.
    """
    if algorithm not in _CLUSTERING_ALGORITHMS:
        raise ValueError(
            f"algorithm={algorithm!r} is not a known clustering heuristic. Cause: weighted "
            f"correlation clustering is not approximable, so only named heuristics are allowed. "
            f"Fix: pass one of {sorted(_CLUSTERING_ALGORITHMS)}."
        )


# --------------------------------------------------------------------------------------
# Feasible — the one parameter that gives selection its many names.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FeasibleShape:
    """The metadata that distinguishes one feasible class from another.

    Attributes:
        shape: Plain-language description of the rule on the *whole* kept answer
            (e.g. "at most k per left record").
        exact: Whether an exact optimum is obtainable. ``False`` only for
            :attr:`Feasible.CLUSTERING`, whose weighted form is not approximable.
        algorithm_forced: Whether keeping this shape forces a *named heuristic*
            (rather than a promised argmax). ``True`` only for clustering.
        implied_scope: The scope a :class:`Select` at this feasible operates at —
            per-``"pair"``, per-anchor ``"group"``, or ``"global"`` over the whole
            relation.
    """

    shape: str
    exact: bool
    algorithm_forced: bool
    implied_scope: Scope


class Feasible(Enum):
    """The feasible class a :class:`Select` keeps its answer inside.

    One selection operation wears many names purely by changing this parameter:
    a threshold, a top-k, an entity-linking argmax, a 1-to-1 assignment, and a
    clustering are all the same "keep the highest-scoring legally-shaped subset"
    at five different feasible classes. Each member carries its
    :class:`FeasibleShape` metadata, surfaced as read-only properties so callers
    write ``Feasible.CLUSTERING.exact`` rather than reaching into ``.value``.

    Members:
        THRESHOLD: No rule on the answer's shape — keep every row that pays the
            price (threshold matching).
        TOPK: At most k rows per left record (top-k blocking / retrieval).
        LINK: At most one target per mention (entity linking, with an implicit
            NIL when nothing clears the price).
        ASSIGNMENT: One-to-one on both sides (1-to-1 assignment / Hungarian).
        CLUSTERING: An equivalence relation over the records. **Not
            approximable** with real weights (weighted correlation clustering),
            so it is inexact and forces a named heuristic — the type never
            promises an argmax here. (The theory calls this shape "equivalence";
            it is spelled ``CLUSTERING`` in code.)
    """

    THRESHOLD = FeasibleShape(
        shape="no rule on the kept answer",
        exact=True,
        algorithm_forced=False,
        implied_scope="pair",
    )
    TOPK = FeasibleShape(
        shape="at most k per left record",
        exact=True,
        algorithm_forced=False,
        implied_scope="group",
    )
    LINK = FeasibleShape(
        shape="at most one target per mention (NIL otherwise)",
        exact=True,
        algorithm_forced=False,
        implied_scope="group",
    )
    ASSIGNMENT = FeasibleShape(
        shape="one-to-one on both sides",
        exact=True,
        algorithm_forced=False,
        implied_scope="global",
    )
    CLUSTERING = FeasibleShape(
        shape="an equivalence relation over the records",
        exact=False,
        algorithm_forced=True,
        implied_scope="global",
    )

    @property
    def _meta(self) -> FeasibleShape:
        value = self.value
        # Every member's value is a FeasibleShape; the assert narrows Enum.value
        # (typed Any) so the metadata properties below stay precisely typed.
        assert isinstance(value, FeasibleShape)
        return value

    @property
    def shape(self) -> str:
        """Plain-language description of the rule on the whole kept answer."""
        return self._meta.shape

    @property
    def exact(self) -> bool:
        """Whether an exact optimum is obtainable (``False`` only for CLUSTERING)."""
        return self._meta.exact

    @property
    def algorithm_forced(self) -> bool:
        """Whether this feasible forces a named heuristic (``True`` only for CLUSTERING)."""
        return self._meta.algorithm_forced

    @property
    def implied_scope(self) -> Scope:
        """The scope a Select at this feasible operates at (its scope is implied, not declared)."""
        return self._meta.implied_scope


# --------------------------------------------------------------------------------------
# Op — the one component type — and its two roles.
# --------------------------------------------------------------------------------------


class Op(ABC, Generic[SchemaT]):
    """The one component type: a stage that maps a scored relation to a scored relation.

    Every in-pipeline component — blocker, comparator, matcher, reranker,
    threshold — IS-A ``Op``: it takes a :class:`~langres.core.pairs.Pairs` and
    returns a :class:`~langres.core.pairs.Pairs`. Because both ends are the same
    type, the composite of two ``Op``\\ s is itself an ``Op`` — the property that
    lets pipelines nest. The two roles differ only in *which half* they do:
    :class:`Score` changes numbers, :class:`Select` deletes rows.
    """

    @abstractmethod
    def forward(self, pairs: Pairs[SchemaT]) -> Pairs[SchemaT]:
        """Map a ``Pairs`` to a ``Pairs``.

        Args:
            pairs: The incoming scored relation.

        Returns:
            The transformed scored relation (rescored, or with rows selected).
        """
        ...  # pragma: no cover


class Score(Op[SchemaT]):
    """The rescore role: "same rows, new scores".

    A ``Score`` keeps every row and replaces its number. What varies between
    Scores is the score family they *produce*, declared as :attr:`out_space`, and
    the :attr:`scope` at which that number is meaningful. A blocker similarity, a
    comparator's per-feature vector (``out_space="vector"``), and a matcher's
    probability are all Scores. Still abstract — a concrete Score implements
    :meth:`~Op.forward` to return the *same rows* carrying a new
    ``score``/``score_type`` (or a ``comparison`` when ``out_space == "vector"``).

    Args:
        scope: The scope the produced score lives at — ``"pair"`` /
            ``"group"`` / ``"global"``. **Declared, not inferred.**
        out_space: The score family produced — a
            :data:`~langres.core.score_type.ScoreType`, or ``"vector"`` for a
            :class:`~langres.core.feature.ComparisonVector`.

    Raises:
        ValueError: If ``scope`` or ``out_space`` is not a recognized value.
    """

    def __init__(self, *, scope: Scope, out_space: OutSpace) -> None:
        if scope not in _VALID_SCOPES:
            raise ValueError(
                f"Score got scope={scope!r}, which is not a valid scope. "
                f"Fix: pass one of {sorted(_VALID_SCOPES)}."
            )
        if out_space not in _VALID_OUT_SPACES:
            raise ValueError(
                f"Score got out_space={out_space!r}, which is not a known score family "
                f"or 'vector'. Fix: pass one of {sorted(_VALID_OUT_SPACES)}."
            )
        self.scope: Scope = scope
        self.out_space: OutSpace = out_space


class Select(Op[SchemaT]):
    """The select role: "same scores, fewer rows".

    A ``Select`` keeps every score untouched and deletes rows, keeping the
    highest-scoring subset that stays inside a :class:`Feasible` class. Its
    *scope is implied by* that feasible (:attr:`Feasible.implied_scope`), never
    declared. Threshold, top-k, entity linking, assignment and clustering are all
    Selects at different feasibles. Still abstract — a concrete Select implements
    :meth:`~Op.forward` to return a *subset* of the incoming rows.

    ``Select(Feasible.CLUSTERING)`` is the escape hatch for clustering: because
    weighted correlation clustering is not approximable, it **refuses to
    construct without an explicit** ``algorithm=`` (``"transitive_closure"`` or
    ``"pivot"``) and stamps :attr:`is_heuristic` into its :attr:`label`. The
    common clustering path is a :class:`ClusterStage` (which defaults the
    algorithm); this raw form forces the caller to name the heuristic. A Select at
    an *exact* feasible needs no ``algorithm=``.

    Args:
        feasible: The feasible class this Select keeps the answer inside.
        algorithm: The named clustering heuristic. Required (and validated) when
            ``feasible`` forces one (CLUSTERING); ``None`` otherwise.

    Raises:
        ValueError: If ``feasible`` forces an algorithm and ``algorithm`` is
            missing or unknown.
    """

    def __init__(self, *, feasible: Feasible, algorithm: str | None = None) -> None:
        if feasible.algorithm_forced:
            if algorithm is None:
                raise ValueError(
                    f"Select at feasible={feasible.name} must be constructed with an explicit "
                    f"algorithm=. Cause: weighted correlation clustering is not approximable, so "
                    f"the type must not promise an exact argmax here — the caller has to name the "
                    f"heuristic. Fix: pass algorithm='transitive_closure' (or 'pivot'), or use a "
                    f"ClusterStage (which defaults algorithm='transitive_closure')."
                )
            _validate_clustering_algorithm(algorithm)
        self.feasible: Feasible = feasible
        self.algorithm: str | None = algorithm

    @property
    def is_heuristic(self) -> bool:
        """``True`` iff this Select runs a named heuristic (i.e. its feasible forces one)."""
        return self.feasible.algorithm_forced

    @property
    def label(self) -> dict[str, object]:
        """A provenance stamp naming the feasible, the algorithm, and heuristic-ness.

        The surface a downstream consumer reads to see that a clustering Select is
        a *named heuristic*, not an exact argmax (per THEORY §8).
        """
        return {
            "role": "select",
            "feasible": self.feasible.name,
            "algorithm": self.algorithm,
            "is_heuristic": self.is_heuristic,
        }


# --------------------------------------------------------------------------------------
# Concrete Selects — the two exact-feasible selections that need no named heuristic.
# A blocker's top-k prune and a matcher's threshold gate are the SAME select role at
# two feasibles; both keep a subset of the incoming rows and touch no score.
# --------------------------------------------------------------------------------------


class ThresholdSelect(Select[SchemaT], Generic[SchemaT]):
    """:class:`Select` at :attr:`Feasible.THRESHOLD`: keep every row that clears ``threshold``.

    The matcher's match gate, as a first-class selection ``Op`` — no shape rule on
    the kept answer, just "keep the rows that pay the price". It asks the canonical
    match rule (:meth:`~langres.core.pairs.PairRow.predicted_match` — a ``decision``
    wins over the score, an abstention is dropped) rather than testing
    ``score >= threshold`` by hand, so a decider row and an abstaining row are
    handled exactly as everywhere else in the library.

    Args:
        threshold: The price a row's score must clear to be kept.
    """

    def __init__(self, threshold: float) -> None:
        super().__init__(feasible=Feasible.THRESHOLD)
        self.threshold = threshold

    def forward(self, pairs: Pairs[SchemaT]) -> Pairs[SchemaT]:
        """Keep the rows whose ``predicted_match(threshold)`` is ``True`` (order preserved).

        An abstaining row (``predicted_match`` returns ``None``) is dropped — it is
        never graded a confident "no" — and a decider's explicit ``decision`` wins
        over its score, exactly as :func:`~langres.core.models.predicted_match`.
        """
        kept = [row for row in pairs.rows if row.predicted_match(self.threshold) is True]
        return Pairs(store=pairs.store, rows=kept)


class TopKSelect(Select[SchemaT], Generic[SchemaT]):
    """:class:`Select` at :attr:`Feasible.TOPK`: keep the ``k`` best rows per left id.

    The retrieval / blocking shape, as a first-class selection ``Op`` — each anchor
    keeps its ``k`` highest-scoring partners. Grouping by ``left_id`` is exactly
    "at most k per left record" over a blocker's ``i < j`` pair set; ties keep input
    order, and a row missing a score sorts last (``score`` read as ``-1.0``).

    Args:
        k: The maximum number of rows to keep per ``left_id``.
    """

    def __init__(self, k: int) -> None:
        super().__init__(feasible=Feasible.TOPK)
        self.k = k

    def forward(self, pairs: Pairs[SchemaT]) -> Pairs[SchemaT]:
        """Keep at most ``k`` highest-scoring rows per ``left_id`` (input order preserved)."""
        by_left: dict[str, list[PairRow[SchemaT]]] = defaultdict(list)
        for row in pairs.rows:
            by_left[row.left_id].append(row)

        keep: set[tuple[str, str]] = set()
        for group in by_left.values():
            ranked = sorted(
                group,
                key=lambda row: row.score if row.score is not None else -1.0,
                reverse=True,
            )
            for row in ranked[: self.k]:
                keep.add((row.left_id, row.right_id))

        kept = [row for row in pairs.rows if (row.left_id, row.right_id) in keep]
        return Pairs(store=pairs.store, rows=kept)


# --------------------------------------------------------------------------------------
# Boundary stages — the honest source/sink codomains (NOT Op subclasses: their
# carriers differ from Pairs -> Pairs).
# --------------------------------------------------------------------------------------


class Source(ABC, Generic[SchemaT]):
    """The pipeline entry: records in, pairs out.

    A ``Source`` turns :data:`Records` into a
    :class:`~langres.core.pairs.Pairs` — a blocker's job, plus (in a later wave)
    ownership of its own index lifecycle (building a vector index moves here). It
    is not an :class:`Op` because its input carrier is records, not pairs.
    """

    @abstractmethod
    def forward(self, records: Records) -> Pairs[SchemaT]:
        """Generate the candidate pairs for ``records``.

        Args:
            records: The normalized record input.

        Returns:
            A ``Pairs`` of candidate rows (typically unscored, ``score_type=None``).
        """
        ...  # pragma: no cover


class ClusterStage(ABC, Generic[SchemaT]):
    """The phase-1 exit: pairs in, clusters out.

    Clustering is the equivalence selection — the point where the table of scored
    pairs collapses into id clusters. Because that selection is not approximable
    (THEORY §8), a ``ClusterStage`` is inherently a *named heuristic*: it carries
    an :attr:`algorithm` (default ``"transitive_closure"``) and stamps
    :attr:`is_heuristic` = ``True`` into its :attr:`label`. It is not an
    :class:`Op` because its output carrier is clusters, not pairs.

    Args:
        algorithm: The named clustering heuristic (default ``"transitive_closure"``).

    Raises:
        ValueError: If ``algorithm`` is not a known clustering heuristic.
    """

    def __init__(self, algorithm: str = "transitive_closure") -> None:
        _validate_clustering_algorithm(algorithm)
        self.algorithm: str = algorithm

    @property
    def is_heuristic(self) -> bool:
        """Always ``True`` — a ClusterStage is a named heuristic, never an exact argmax."""
        return True

    @property
    def label(self) -> dict[str, object]:
        """A provenance stamp naming the algorithm and marking the stage heuristic (§8)."""
        return {"role": "cluster_stage", "algorithm": self.algorithm, "is_heuristic": True}

    @abstractmethod
    def forward(self, pairs: Pairs[SchemaT]) -> Clusters:
        """Collapse the scored pairs into id clusters.

        Args:
            pairs: The scored relation to cluster.

        Returns:
            The id clusters (``list[set[str]]``).
        """
        ...  # pragma: no cover


class Finalize(ABC):
    """The phase-2 exit: clusters in, refined clusters OR a golden record out.

    The verify/canonicalize exit — the future home of the ``Canonicalizer``. A
    *verify* Finalize splits what clustering over-merged and returns refined
    :data:`Clusters`; a *canonicalize* Finalize fuses a cluster into one
    :data:`GoldenRecord`. It is not an :class:`Op`, nor generic over the schema,
    because its input carrier is clusters (ids), not pairs.
    """

    @abstractmethod
    def forward(self, clusters: Clusters) -> Clusters | GoldenRecord:
        """Verify or canonicalize the clusters.

        Args:
            clusters: The id clusters from a :class:`ClusterStage`.

        Returns:
            Refined :data:`Clusters` (verify) or a single :data:`GoldenRecord`
            (canonicalize).
        """
        ...  # pragma: no cover


# --------------------------------------------------------------------------------------
# Sequential — the ordered pipeline whose wiring is checked at construction.
# --------------------------------------------------------------------------------------

#: A stage a :class:`Sequential` may hold, in pipeline order.
Stage: TypeAlias = Source[Any] | Op[Any] | ClusterStage[Any] | Finalize


class Sequential(Generic[SchemaT]):
    """An ordered pipeline: a :class:`Source`, then zero+ :class:`Op`\\ s, then a
    :class:`ClusterStage`, optionally a :class:`Finalize`.

    Its :meth:`check` runs **automatically at construction** — an opt-in wiring
    guard would be a green light decoupled from what it checks, so there is no way
    to build a mis-wired ``Sequential``. A wiring error raises immediately with a
    single *problem + cause + fix* message.

    Args:
        stages: The stages, in pipeline order.

    Raises:
        ValueError: If the wiring is invalid (see :meth:`check`).
        TypeError: If a stage is not a Source / Op / ClusterStage / Finalize.
    """

    def __init__(self, stages: Sequence[Stage]) -> None:
        self.stages: list[Stage] = list(stages)
        self.check()

    def check(self) -> None:
        """Validate the pipeline's wiring; raise a clear error on the first fault.

        Two faults are caught:

        1. **Carrier mismatch.** Adjacent stages whose output/input carriers do
           not line up — e.g. a Finalize before the ClusterStage, or an Op after
           it. The pipeline must also start with a Source (the only stage that
           consumes the initial ``records`` carrier).
        2. **Select on a vector space.** A Select positioned to consume rows a
           Score produced with ``out_space == "vector"``. A ``ComparisonVector``
           (``S = R^d``) is not orderable, so selecting on it is a type error; the
           fix is a scalarizer Score (vector -> scalar) in between. Only ``"vector"``
           is rejected — a Select after any *scalar* family, including the
           ``"unknown"`` sentinel, is legal (an orderable scalar admits selection).

        Raises:
            ValueError: On any wiring fault, with a problem + cause + fix message.
            TypeError: If a stage is of an unrecognized type.
        """
        carrier = "records"
        # The score family currently on the rows: the most recent Score's
        # out_space, or None while unscored. Drives the vector check.
        current_space: str | None = None

        # A score-space union ACROSS score families under one selection (THEORY
        # §5/§9 — one selection is one order over one score space) is a real
        # hazard, but it is structurally impossible on this carrier: a ``Pairs``
        # row holds a single ``score``/``score_type``, so a later Score overwrites
        # the column and a Select always sees one coherent family. It returns as a
        # wiring-time error once a multi-column / combine (union/intersection)
        # carrier exists (§2/§10) — there is deliberately no check for it here.

        for index, stage in enumerate(self.stages):
            expected_in, produced_out = _stage_carriers(stage, index)
            if expected_in != carrier:
                raise ValueError(_carrier_mismatch_message(index, stage, carrier, expected_in))

            if isinstance(stage, Score):
                current_space = stage.out_space
            elif isinstance(stage, Select):
                # Only a vector space is rejected: selection is undefined on a
                # non-orderable ComparisonVector. Every scalar family is fine, and
                # so is the ``"unknown"`` sentinel — an orderable scalar whose family
                # is not yet pinned is still orderable, so a Select may follow it.
                if current_space == "vector":
                    raise ValueError(_select_on_vector_message(index))

            carrier = produced_out


# --------------------------------------------------------------------------------------
# check() helpers — carrier table + one message per fault (problem + cause + fix).
# --------------------------------------------------------------------------------------


def _stage_carriers(stage: Stage, index: int) -> tuple[str, str]:
    """The (input, output) carrier names of a stage. Order matters: Source /
    ClusterStage / Finalize before the Op roles, since they are not Ops."""
    if isinstance(stage, Source):
        return ("records", "pairs")
    if isinstance(stage, ClusterStage):
        return ("pairs", "clusters")
    if isinstance(stage, Finalize):
        return ("clusters", "final")
    if isinstance(stage, Op):
        return ("pairs", "pairs")
    raise TypeError(
        f"Sequential stage {index} is a {type(stage).__name__}, which is not a Source, Op "
        f"(Score/Select), ClusterStage or Finalize. Fix: pass only these stage types."
    )


def _carrier_mismatch_message(index: int, stage: Stage, carrier: str, expected_in: str) -> str:
    name = type(stage).__name__
    if carrier == "records":
        return (
            f"Sequential wiring error: stage {index} is a {name}, but a pipeline must start "
            f"with a Source (records -> pairs). Cause: only a Source consumes the 'records' "
            f"carrier; every other stage needs a 'pairs' or 'clusters' input that no upstream "
            f"stage has produced yet. Fix: put a Source first."
        )
    return (
        f"Sequential wiring error: stage {index} ({name}) consumes a '{expected_in}' carrier, "
        f"but the previous stage produced a '{carrier}' carrier. Cause: the stages are out of "
        f"pipeline order — a pipeline runs Source (records -> pairs), then Ops (pairs -> pairs), "
        f"then a ClusterStage (pairs -> clusters), then an optional Finalize (clusters -> "
        f"record). Fix: reorder so this stage's input carrier matches the upstream output — keep "
        f"every Op before the ClusterStage, and any Finalize after it."
    )


def _select_on_vector_message(index: int) -> str:
    return (
        f"Sequential wiring error: stage {index} (Select) selects over rows scored into a vector "
        f"space (out_space='vector'). Cause: a ComparisonVector (S = R^d) is not orderable, so "
        f"'keep the top / keep above a threshold' is undefined on it — selecting on it is a type "
        f"error. Fix: insert a scalarizer Score that maps the vector to a scalar score (e.g. a "
        f"weighted average over the ComparisonVector) before this Select."
    )
