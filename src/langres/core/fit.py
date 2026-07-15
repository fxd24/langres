"""Fit-hook contracts: how a pipeline role opts into being trainable (W1.0, E6).

Runtime-checkable, structural `Protocol`s -- **not** abstract methods on any
base class. Adding an abstract ``fit`` to ``Matcher`` would break every
existing, non-learnable matcher (``WeightedAverageMatcher``, ``LLMMatcher``,
...), which is not the goal here: most components stay heuristic or zero-shot
and never need a fit hook. Instead, a component *opts in* structurally by
implementing the method with the matching name and signature; callers (the
Resolver) detect this with ``isinstance(component, <Mixin>)``.

Three pipeline roles can be trained, each with its own fit signature:

- **Matcher** -- ``SupervisedFitMixin`` (``fit(candidates, labels)``, e.g. a
  ``RandomForestMatcher`` fitting an sklearn forest over
  ``ComparisonVector.similarities``) and ``UnsupervisedFitMixin``
  (``fit_unlabeled(candidates)``, e.g. a ``FellegiSunterMatcher`` fitting m/u
  probabilities via EM).
- **Blocker** -- ``BlockerFitMixin`` (``fit_blocker(records, pairs)``: learn a
  high-recall blocking key/index from known match pairs). Contract-only here;
  the concrete ``TrainableVectorBlocker`` is a later PR -- this makes learned
  blocking *expressible* in the role taxonomy.
- **Calibrator** -- ``CalibratorFitMixin`` (``fit_calibrator(scores, labels)``:
  learn a score->probability map). Contract-only here; the concrete
  Platt/isotonic impl is a later PR.

See ``Resolver.fit()`` for how the matcher mixins are consumed, and
docs/TECHNICAL_OVERVIEW.md ("Fit-hook contract") for the full picture.
"""

from collections.abc import Iterator, Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from langres.core.models import ERCandidate

# Generic type variable for schema types (must be a Pydantic model). Defined
# locally to avoid a circular import, matching the existing per-module SchemaT
# convention (see models.py, module.py, groups.py).
SchemaT = TypeVar("SchemaT", bound=BaseModel)


@runtime_checkable
class SupervisedFitMixin(Protocol[SchemaT]):
    """A Matcher that learns from labeled candidate pairs.

    Implement ``fit(candidates, labels)`` to opt in; no subclassing required
    (structural typing via ``isinstance()``, enabled by ``@runtime_checkable``).
    """

    def fit(self, candidates: Iterator[ERCandidate[SchemaT]], labels: Sequence[bool]) -> None:
        """Fit the module's parameters from labeled candidate pairs.

        Args:
            candidates: The blocked (and, if a comparator is configured,
                comparison-attached) candidate stream to learn from.
            labels: Gold match/non-match labels, positionally aligned with
                ``candidates`` (same convention as
                :func:`~langres.core.calibration.derive_threshold`).
        """
        ...  # pragma: no cover — Protocol method body, never executed


@runtime_checkable
class UnsupervisedFitMixin(Protocol[SchemaT]):
    """A Matcher that learns from candidate pairs without labels.

    Implement ``fit_unlabeled(candidates)`` to opt in; no subclassing required
    (structural typing via ``isinstance()``, enabled by ``@runtime_checkable``).
    """

    def fit_unlabeled(self, candidates: Iterator[ERCandidate[SchemaT]]) -> None:
        """Fit the module's parameters from unlabeled candidate pairs.

        Args:
            candidates: The blocked (and, if a comparator is configured,
                comparison-attached) candidate stream to learn from.
        """
        ...  # pragma: no cover — Protocol method body, never executed


@runtime_checkable
class BlockerFitMixin(Protocol):
    """A Blocker that learns its blocking scheme from known match pairs.

    Implement ``fit_blocker(records, pairs)`` to opt in; no subclassing required
    (structural typing via ``isinstance()``, enabled by ``@runtime_checkable``).
    Contract-only for now -- the concrete ``TrainableVectorBlocker`` is a later
    PR; this makes learned blocking *expressible* in the role taxonomy without
    shipping an impl.
    """

    def fit_blocker(self, records: Sequence[Any], pairs: Sequence[tuple[str, str]]) -> None:
        """Fit the blocker's parameters from records + known match pairs.

        Args:
            records: The raw records to index (same shape ``Blocker.stream``
                accepts).
            pairs: Known ``(left_id, right_id)`` match pairs -- the recall target
                the learned blocking key/index must keep.
        """
        ...  # pragma: no cover — Protocol method body, never executed


@runtime_checkable
class CalibratorFitMixin(Protocol):
    """A calibrator that learns a score->probability map from scores + labels.

    Implement ``fit_calibrator(scores, labels)`` to opt in; no subclassing
    required (structural typing via ``isinstance()``). Contract-only for now --
    the concrete Platt/isotonic impl is a later PR; this makes score calibration
    *expressible* as its own fit role, distinct from a matcher's
    ``fit(candidates, labels)``.
    """

    def fit_calibrator(self, scores: Sequence[float], labels: Sequence[bool]) -> None:
        """Fit the calibrator's parameters from scores + gold labels.

        Args:
            scores: The matcher scores to calibrate, positionally aligned with
                ``labels``.
            labels: Gold match/non-match labels for each score.
        """
        ...  # pragma: no cover — Protocol method body, never executed
