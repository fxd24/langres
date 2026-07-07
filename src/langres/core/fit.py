"""Fit-hook contracts: how a Module opts into being trainable (W1.0, E6).

Two runtime-checkable, structural `Protocol`s -- **not** abstract methods on
the base ``Module``. Adding an abstract ``fit``/``fit_unlabeled`` to ``Module``
would break every existing, non-learnable module (``WeightedAverageJudge``,
``LLMJudge``, ...), which is not the goal here: most judges stay heuristic or
zero-shot and never need a fit hook. Instead, a module *opts in* structurally
by implementing the method with the matching name and signature; callers (the
Resolver) detect this with ``isinstance(module, SupervisedFitMixin)`` /
``isinstance(module, UnsupervisedFitMixin)``.

- ``SupervisedFitMixin``: for judges that learn from labeled pairs (e.g. a
  future ``RandomForestJudge`` fitting an sklearn RandomForest over
  ``ComparisonVector.similarities``).
- ``UnsupervisedFitMixin``: for judges that learn without labels (e.g. a
  future ``FellegiSunterJudge`` fitting m/u probabilities via EM).

See ``Resolver.fit()`` for how these are consumed, and
docs/TECHNICAL_OVERVIEW.md ("Fit-hook contract") for the full picture.
"""

from collections.abc import Iterator, Sequence
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from langres.core.models import ERCandidate

# Generic type variable for schema types (must be a Pydantic model). Defined
# locally to avoid a circular import, matching the existing per-module SchemaT
# convention (see models.py, module.py, groups.py).
SchemaT = TypeVar("SchemaT", bound=BaseModel)


@runtime_checkable
class SupervisedFitMixin(Protocol[SchemaT]):
    """A Module that learns from labeled candidate pairs.

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
    """A Module that learns from candidate pairs without labels.

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
