"""Public test/example doubles for :class:`~langres.core.module.Module`.

``langres.testing`` is a small, dependency-light home for scripted stand-ins
for the library's abstract components -- currently :class:`ScriptedJudge`, a
``Module`` that returns pre-scripted scores instead of computing them. It is
the safe way to exercise judge-shaped code (``CascadeJudge``,
:func:`~langres.core.benchmark.evaluate`, the review/harvest flywheel, ...) in
tests and examples with no network, no API key, and no spend. A real
``LLMJudge`` picks up ``OPENROUTER_API_KEY`` from the repo's ``.env`` via
litellm's import-time ``load_dotenv()`` and would make a real, billed call;
``ScriptedJudge`` never imports litellm (or any other heavy/optional
dependency) at all.

It also replaces one genuine hand-rolled duplicate: a near-identical
``ScriptedJudge`` that used to live in
``tests/core/modules/test_cascade_judge.py`` (same name, same purpose, same
``seen`` escalation-laziness spy, same ``inspect_scores`` delegation to
``_inspect_scores_impl``) now imports this class instead. Two other
``Module``-shaped test doubles were deliberately left in place, not migrated:

- ``DummyBlocker`` in ``tests/core/test_blocker.py`` subclasses
  :class:`~langres.core.blocker.Blocker`, not ``Module`` -- a ``Module``
  double cannot structurally replace it.
- The four ``DummyModule`` classes in ``tests/core/test_module.py`` test the
  ``Module`` abstract base class's own contract (it can't be instantiated
  directly, a concrete subclass's ``forward()`` is a lazy iterator, etc.);
  using a library-provided ``Module`` subclass to test the ABC itself would
  be circular, so they stay as minimal, from-scratch stubs.

Not part of the core import graph: a bare ``import langres`` does not import
this module (use ``from langres.testing import ScriptedJudge`` explicitly),
and it is deliberately NOT registered in :mod:`langres.core.registry` (see the
comment on :class:`ScriptedJudge` below) -- both by design, not oversight.
"""

from collections.abc import Callable, Iterator
from typing import Any

from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.module import Module, SchemaT
from langres.core.reports import ScoreInspectionReport, _inspect_scores_impl

__all__ = ["ScriptedJudge"]


class ScriptedJudge(Module[SchemaT]):
    """A :class:`~langres.core.module.Module` that returns pre-scripted scores.

    For tests and examples that need a judge-shaped stand-in with no network,
    no API key, and no spend -- e.g. driving a
    :class:`~langres.core.modules.cascade_judge.CascadeJudge`, exercising
    :func:`~langres.core.benchmark.evaluate`, or demonstrating the
    review/harvest flywheel without a real ``LLMJudge``.

    Scores come from either a ``dict`` keyed by the unordered pair
    (``frozenset({left_id, right_id})`` -- order-independent, matching how the
    rest of the codebase keys pairs) or a callable applied to each candidate.
    Every pair ``forward()`` sees is recorded, in order, on :attr:`seen` -- an
    escalation-laziness spy: because ``forward()`` is a generator, partially
    consuming its output only records the pairs actually pulled, which is how
    a test proves a cascade judge skipped its expensive stage for an
    in-band-only pair.

    NOT registered in :mod:`langres.core.registry` (deliberately: it is a test
    double, not a real component -- registering it would put it in
    ``Resolver.load``'s config-driven dispatch) and NOT imported by
    ``langres/__init__.py`` (a bare ``import langres`` never sees this module).

    Example:
        >>> from langres.core.models import CompanySchema, ERCandidate
        >>> from langres.testing import ScriptedJudge
        >>> pair = ERCandidate(
        ...     left=CompanySchema(id="a", name="Acme"),
        ...     right=CompanySchema(id="b", name="Acme Corp"),
        ...     blocker_name="test",
        ... )
        >>> judge = ScriptedJudge({frozenset({"a", "b"}): 0.9})
        >>> judgement = next(judge.forward(iter([pair])))
        >>> judgement.score
        0.9

    Attributes:
        seen: Every pair key (``frozenset({left_id, right_id})``) ``forward()``
            has yielded a judgement for, in the order it was pulled from the
            input stream.
    """

    def __init__(
        self,
        scores: dict[frozenset[str], float] | Callable[[ERCandidate[SchemaT]], float],
        *,
        score_type: str = "heuristic",
        decision_step: str = "scripted",
        reasoning: str | None = None,
        provenance: dict[str, Any] | None = None,
        default_score: float = 0.5,
    ) -> None:
        """Build a scripted judge.

        Args:
            scores: Either a ``dict`` mapping ``frozenset({left_id, right_id})``
                to a score, or a callable taking the ``ERCandidate`` and
                returning a score directly.
            score_type: ``PairwiseJudgement.score_type`` to stamp on every
                judgement (one of the ``models.py`` literals -- not validated
                here, so an invalid value surfaces as a ``ValidationError`` at
                yield time).
            decision_step: ``PairwiseJudgement.decision_step`` to stamp on
                every judgement.
            reasoning: Optional ``PairwiseJudgement.reasoning`` to stamp on
                every judgement.
            provenance: Optional provenance dict copied onto every judgement
                (e.g. ``{"cost_usd": 0.001}`` so a paid-judge cost path, like
                :func:`~langres.core.benchmark.evaluate`'s ``cost.usd_total``,
                has something real to sum).
            default_score: Score returned for a pair not present in a ``dict``
                ``scores`` map. Ignored when ``scores`` is a callable.
        """
        self.scores = scores
        self.score_type = score_type
        self.decision_step = decision_step
        self.reasoning = reasoning
        self.provenance: dict[str, Any] = dict(provenance) if provenance is not None else {}
        self.default_score = default_score
        self.seen: list[frozenset[str]] = []

    def forward(self, candidates: Iterator[ERCandidate[SchemaT]]) -> Iterator[PairwiseJudgement]:
        """Yield one scripted :class:`PairwiseJudgement` per candidate, in order.

        Records every pair on :attr:`seen` as it is pulled -- see the class
        docstring's laziness note.
        """
        for candidate in candidates:
            key = frozenset(
                {candidate.left.id, candidate.right.id}  # type: ignore[attr-defined]
            )
            self.seen.append(key)
            if isinstance(self.scores, dict):
                score = self.scores.get(key, self.default_score)
            else:
                score = self.scores(candidate)
            yield PairwiseJudgement(
                left_id=candidate.left.id,  # type: ignore[attr-defined]
                right_id=candidate.right.id,  # type: ignore[attr-defined]
                score=score,
                score_type=self.score_type,  # type: ignore[arg-type]
                decision_step=self.decision_step,
                reasoning=self.reasoning,
                provenance=dict(self.provenance),
            )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        """Delegate to the shared :func:`_inspect_scores_impl` (same as every real Module)."""
        return _inspect_scores_impl(judgements, sample_size)
