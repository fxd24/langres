"""Inspection contract: how a Matcher opts into score inspection (I1).

A runtime-checkable, structural `Protocol` -- **not** an abstract method on
:class:`~langres.core.matcher.Matcher`. This is the same opt-in convention
``langres.core.fit`` uses for the trainable roles, and for the same reason: an
``@abstractmethod`` forces *every* Matcher to implement the method, whether or
not it has anything to inspect. ``inspect_scores`` was abstract until W1, which
made 21 classes -- including every test double -- carry a delegation (or a
``raise NotImplementedError`` stub) to satisfy a method with two callers, both
of them pass-throughs.

A Matcher opts in by implementing ``inspect_scores``; callers detect it with
``isinstance(matcher, Inspectable)``. Every matcher that implemented it before
W1 still does, so nothing observable changed for them -- what changed is that
not implementing it is now *expressible*.

The shared implementation stays where it was, in
:func:`~langres.core.reports._inspect_scores_impl`: an opt-in contract makes the
method optional, not un-shareable, and the ~12 concrete matchers remain 2-line
pass-throughs to it.
"""

from typing import Protocol, runtime_checkable

from langres.core.models import PairwiseJudgement
from langres.core.reports import ScoreInspectionReport


@runtime_checkable
class Inspectable(Protocol):
    """A Matcher that can summarize its own scores without ground-truth labels.

    Implement ``inspect_scores(judgements, sample_size)`` to opt in; no
    subclassing required (structural typing via ``isinstance()``, enabled by
    ``@runtime_checkable``).
    """

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        """Explore scores without ground truth labels.

        Args:
            judgements: List of PairwiseJudgement objects to analyze.
            sample_size: Number of examples to include (default: 10).

        Returns:
            ScoreInspectionReport with statistics, examples, and recommendations.
        """
        ...  # pragma: no cover — Protocol method body, never executed


def _ensure_inspectable(module: object) -> Inspectable:
    """Narrow ``module`` to :class:`Inspectable`, or raise naming the contract.

    For the wrapper matchers (the spend cap, the JudgementLog) that forward
    ``inspect_scores`` to whatever they wrap: since opting in is optional, the
    wrapped matcher may not implement it, and an unguarded delegation would
    surface as a bare ``AttributeError``. Lives here so the check and its
    wording stay in ONE place rather than drifting between wrappers.
    """
    if not isinstance(module, Inspectable):
        raise TypeError(
            f"{type(module).__name__} does not implement inspect_scores(); "
            "see langres.core.inspection.Inspectable"
        )
    return module
