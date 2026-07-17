"""The immutable Objective the autoresearch loop steers on.

The autoresearch loop is ``propose → run → evaluate → keep-if-better`` (epic
#145). The :class:`Objective` is the *immutable scorer* that decides the last
step: given a candidate config's metrics and the incumbent's, is the candidate
**better**? ER F1 saturates near 99%, so the loop steers on loss-like / Pareto
signals (``recall@budget``, ``log_loss``, quality×cost) instead of a thresholded
F1, and the Objective supports both a single goal with feasibility constraints
*and* multi-objective Pareto dominance.

**Metric-source-agnostic by design.** The Objective operates on a plain
``Mapping[str, float]`` of already-computed metrics; it never computes a metric
itself (the run/evaluate stage does that and hands the numbers over). It imports
nothing heavy — pure stdlib + typing — so a bare ``import langres`` stays
import-light (see ``tests/test_import_budget.py``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, cast, get_args

Direction = Literal["maximize", "minimize"]
"""Optimization sense of a goal metric: higher-is-better or lower-is-better."""

ConstraintOp = Literal[">=", "<=", ">", "<"]
"""Comparison operator for a feasibility constraint (``metric <op> threshold``)."""

_DIRECTIONS: tuple[Direction, ...] = get_args(Direction)
_OPS: dict[ConstraintOp, Callable[[float, float], bool]] = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
}


def _require(metrics: Mapping[str, float], name: str) -> float:
    """Return ``metrics[name]`` or raise a clear error naming what is available.

    Missing metrics fail loud rather than defaulting to 0.0: a silently-absent
    metric would corrupt every feasibility and dominance decision downstream.
    """
    try:
        return metrics[name]
    except KeyError:
        raise ValueError(
            f"metric {name!r} is missing from the metrics mapping (available: {sorted(metrics)})"
        ) from None


@dataclass(frozen=True, slots=True)
class Goal:
    """A single optimization goal: one metric plus the direction to push it.

    Attributes:
        metric: The metric key looked up in a metrics mapping.
        direction: ``"maximize"`` (higher is better) or ``"minimize"`` (lower is
            better).
    """

    metric: str
    direction: Direction

    def __post_init__(self) -> None:
        if self.direction not in _DIRECTIONS:
            raise ValueError(
                f"direction must be one of {list(_DIRECTIONS)}, got {self.direction!r}"
            )


@dataclass(frozen=True, slots=True)
class Constraint:
    """A feasibility constraint: ``metrics[metric] <op> threshold`` must hold.

    Attributes:
        metric: The metric key looked up in a metrics mapping.
        op: One of ``">="``, ``"<="``, ``">"``, ``"<"``.
        threshold: The value the metric is compared against.
    """

    metric: str
    op: ConstraintOp
    threshold: float

    def __post_init__(self) -> None:
        if self.op not in _OPS:
            raise ValueError(f"constraint op must be one of {list(_OPS)}, got {self.op!r}")

    def satisfied(self, metrics: Mapping[str, float]) -> bool:
        """Whether ``metrics`` satisfies this constraint.

        Raises:
            ValueError: If ``metric`` is absent from ``metrics``.
        """
        return _OPS[self.op](_require(metrics, self.metric), self.threshold)


@dataclass(frozen=True, slots=True)
class Objective:
    """The immutable keep-if-better scorer for the autoresearch loop.

    An Objective bundles one or more :class:`Goal` s (the optimization targets)
    with zero or more :class:`Constraint` s (feasibility gates). It scores a
    candidate config's metrics against the incumbent's and answers the loop's
    single question via :meth:`is_better`.

    Prefer the ergonomic constructors :meth:`maximize`, :meth:`minimize`, and
    :meth:`pareto` over the raw tuple form.

    Attributes:
        goals: The optimization goals (at least one). Multiple goals are treated
            as a Pareto front — never scalarized.
        constraints: Feasibility gates; a candidate violating any is infeasible.
    """

    goals: tuple[Goal, ...]
    constraints: tuple[Constraint, ...] = ()

    def __post_init__(self) -> None:
        if not self.goals:
            raise ValueError("an Objective needs at least one goal")

    def is_feasible(self, metrics: Mapping[str, float]) -> bool:
        """True iff every constraint holds (vacuously true with no constraints).

        Raises:
            ValueError: If a constrained metric is absent from ``metrics``.
        """
        return all(c.satisfied(metrics) for c in self.constraints)

    def value(self, metrics: Mapping[str, float]) -> tuple[float, ...]:
        """The goal metrics as a tuple, each normalized to higher-is-better.

        A ``minimize`` goal is negated so a larger normalized value is always
        better; this makes Pareto dominance uniform across goals (see
        :meth:`dominates`). The tuple is ordered to match ``goals``.

        Raises:
            ValueError: If a goal metric is absent from ``metrics``.
        """
        out: list[float] = []
        for g in self.goals:
            v = _require(metrics, g.metric)
            out.append(v if g.direction == "maximize" else -v)
        return tuple(out)

    def dominates(self, a: Mapping[str, float], b: Mapping[str, float]) -> bool:
        """Pareto dominance: ``a`` dominates ``b`` over the goal metrics.

        Compares the higher-is-better normalized value tuples (see
        :meth:`value`): ``a`` dominates ``b`` iff it is ``>=`` on *every* goal
        and strictly ``>`` on *at least one*. For a single goal this collapses
        to a strict scalar improvement (``a > b``). Feasibility is not
        considered here — :meth:`is_better` gates on that first.

        Raises:
            ValueError: If a goal metric is absent from ``a`` or ``b``.
        """
        va = self.value(a)
        vb = self.value(b)
        no_worse = all(x >= y for x, y in zip(va, vb, strict=True))
        strictly_better = any(x > y for x, y in zip(va, vb, strict=True))
        return no_worse and strictly_better

    def is_better(
        self,
        candidate: Mapping[str, float],
        incumbent: Mapping[str, float] | None,
    ) -> bool:
        """The loop's keep-if-better decision for ``candidate`` vs ``incumbent``.

        Semantics, in order:

        1. **Feasibility gates first.** An infeasible candidate is never better
           (returns ``False``), whatever the incumbent. A feasible candidate
           beats a ``None`` incumbent (the first feasible config seen) or an
           infeasible incumbent.
        2. **Pareto improvement.** With both feasible, the candidate is better
           iff it *dominates* the incumbent (see :meth:`dominates`) — ``>=`` on
           every goal and ``>`` on at least one. For a single goal this is a
           strict scalar improvement.

        When neither dominates the other — a tie, or an incomparable
        multi-objective trade-off (better on one goal, worse on another) — the
        candidate is **not** better and the loop keeps the incumbent. This makes
        the decision deterministic and monotone for M1: trade-offs are never
        scalarized, so the incumbent is displaced only by a strict Pareto win.

        Raises:
            ValueError: If a referenced goal metric is absent from ``candidate``,
                or from ``incumbent`` when the comparison reaches it, or a
                constrained metric is absent from a mapping it is checked against.
        """
        if not self.is_feasible(candidate):
            return False
        if incumbent is None or not self.is_feasible(incumbent):
            return True
        return self.dominates(candidate, incumbent)

    @classmethod
    def maximize(
        cls,
        metric: str,
        *,
        subject_to: Iterable[tuple[str, str, float]] = (),
    ) -> Objective:
        """A single maximize goal, optionally subject to constraints.

        Args:
            metric: The metric to maximize (higher is better).
            subject_to: Constraints as ``(metric, op, threshold)`` triples, e.g.
                ``[("precision", ">=", 0.9)]``.
        """
        return cls(goals=(Goal(metric, "maximize"),), constraints=_to_constraints(subject_to))

    @classmethod
    def minimize(
        cls,
        metric: str,
        *,
        subject_to: Iterable[tuple[str, str, float]] = (),
    ) -> Objective:
        """A single minimize goal (e.g. ``log_loss``, cost), optionally constrained.

        Args:
            metric: The metric to minimize (lower is better).
            subject_to: Constraints as ``(metric, op, threshold)`` triples.
        """
        return cls(goals=(Goal(metric, "minimize"),), constraints=_to_constraints(subject_to))

    @classmethod
    def pareto(
        cls,
        goals: Iterable[tuple[str, str]],
        *,
        subject_to: Iterable[tuple[str, str, float]] = (),
    ) -> Objective:
        """A multi-objective Pareto goal set.

        Args:
            goals: ``(metric, direction)`` pairs where direction is
                ``"maximize"`` or ``"minimize"``, e.g.
                ``[("recall", "maximize"), ("cost", "minimize")]``.
            subject_to: Constraints as ``(metric, op, threshold)`` triples.
        """
        return cls(
            goals=tuple(Goal(m, cast(Direction, d)) for m, d in goals),
            constraints=_to_constraints(subject_to),
        )


def _to_constraints(triples: Iterable[tuple[str, str, float]]) -> tuple[Constraint, ...]:
    """Build validated :class:`Constraint` s from ``(metric, op, threshold)`` triples.

    The ``op`` is cast to :data:`ConstraintOp` to satisfy the type checker;
    :meth:`Constraint.__post_init__` is the real runtime guard that rejects an
    unknown operator with a clear ``ValueError``.
    """
    return tuple(
        Constraint(metric, cast(ConstraintOp, op), threshold) for metric, op, threshold in triples
    )
