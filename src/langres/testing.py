"""Public test/example doubles for :class:`~langres.core.matcher.Matcher`.

``langres.testing`` is a small, dependency-light home for scripted stand-ins
for the library's abstract components -- currently :class:`ScriptedJudge`, a
``Matcher`` that returns pre-scripted scores instead of computing them. It is
the safe way to exercise judge-shaped code (``CascadeMatcher``,
:func:`~langres.core.benchmark.evaluate`, the review/harvest flywheel, ...) in
tests and examples with no network, no API key, and no spend. A real
``LLMMatcher`` picks up ``OPENROUTER_API_KEY`` from the repo's ``.env`` via
litellm's import-time ``load_dotenv()`` and would make a real, billed call;
``ScriptedJudge`` never imports litellm (or any other heavy/optional
dependency) at all.

It also replaces one genuine hand-rolled duplicate: a near-identical
``ScriptedJudge`` that used to live in
``tests/core/modules/test_cascade_judge.py`` (same name, same purpose, same
``seen`` escalation-laziness spy, same ``inspect_scores`` delegation to
``_inspect_scores_impl``) now imports this class instead. Two other
``Matcher``-shaped test doubles were deliberately left in place, not migrated:

- ``DummyBlocker`` in ``tests/core/test_blocker.py`` subclasses
  :class:`~langres.core.blocker.Blocker`, not ``Matcher`` -- a ``Matcher``
  double cannot structurally replace it.
- The four ``DummyModule`` classes in ``tests/core/test_module.py`` test the
  ``Matcher`` abstract base class's own contract (it can't be instantiated
  directly, a concrete subclass's ``forward()`` is a lazy iterator, etc.);
  using a library-provided ``Matcher`` subclass to test the ABC itself would
  be circular, so they stay as minimal, from-scratch stubs.

Not part of the core import graph: a bare ``import langres`` does not import
this module (use ``from langres.testing import ScriptedJudge`` explicitly),
and it is deliberately NOT registered in :mod:`langres.core.registry` (see the
comment on :class:`ScriptedJudge` below) -- both by design, not oversight.
"""

from collections.abc import Callable, Iterator
from typing import Any

from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.matcher import Matcher, SchemaT
from langres.core.reports import ScoreInspectionReport, _inspect_scores_impl

__all__ = ["ScriptedJudge"]


class ScriptedJudge(Matcher[SchemaT]):
    """A :class:`~langres.core.matcher.Matcher` that returns pre-scripted scores.

    For tests and examples that need a judge-shaped stand-in with no network,
    no API key, and no spend -- e.g. driving a
    :class:`~langres.core.matchers.cascade_judge.CascadeMatcher`, exercising
    :func:`~langres.core.benchmark.evaluate`, or demonstrating the
    review/harvest flywheel without a real ``LLMMatcher``.

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
        abstain: Callable[[ERCandidate[SchemaT]], bool] | None = None,
        confidence: dict[frozenset[str], float]
        | Callable[[ERCandidate[SchemaT]], float | None]
        | None = None,
        confidence_source: str = "none",
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
            abstain: Optional predicate; when it returns ``True`` for a
                candidate the judge yields an abstention (``decision=None,
                score=None``) instead of a scored judgement, so the abstain
                paths of downstream code (``predicted_match``, ``link``'s
                ``MatcherAbstainedError``, the review flywheel) are exercisable
                with no network. Evaluated before ``scores``.
            confidence: Optional per-pair confidence (credence in [0, 1]),
                mirroring ``scores`` -- a ``dict`` keyed by the unordered pair or
                a callable returning a ``float | None``. Lets the double model a
                logprob judge offline, so the calibration panel of an
                :class:`~langres.core.eval_report.EvalReport` has a real signal to
                plot. A callable/dict returning ``None`` for a pair leaves that
                judgement's ``confidence`` unset. ``None`` (default): no
                confidence on any judgement.
            confidence_source: ``PairwiseJudgement.confidence_source`` stamped on
                a judgement that gets a confidence (e.g. ``"logprob"``). Ignored
                when ``confidence`` is ``None`` or yields ``None`` for the pair.
        """
        self.scores = scores
        self.score_type = score_type
        self.decision_step = decision_step
        self.reasoning = reasoning
        self.provenance: dict[str, Any] = dict(provenance) if provenance is not None else {}
        self.default_score = default_score
        self.abstain = abstain
        self.confidence = confidence
        self.confidence_source = confidence_source
        self.seen: list[frozenset[str]] = []

    def _confidence_for(self, candidate: ERCandidate[SchemaT], key: frozenset[str]) -> float | None:
        """Resolve the scripted confidence for one candidate, or ``None``."""
        if self.confidence is None:
            return None
        if isinstance(self.confidence, dict):
            return self.confidence.get(key)
        return self.confidence(candidate)

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
            if self.abstain is not None and self.abstain(candidate):
                yield PairwiseJudgement(
                    left_id=candidate.left.id,  # type: ignore[attr-defined]
                    right_id=candidate.right.id,  # type: ignore[attr-defined]
                    score=None,
                    score_type=self.score_type,  # type: ignore[arg-type]
                    decision_step=self.decision_step,
                    reasoning=self.reasoning,
                    provenance=dict(self.provenance),
                )
                continue
            if isinstance(self.scores, dict):
                score = self.scores.get(key, self.default_score)
            else:
                score = self.scores(candidate)
            conf = self._confidence_for(candidate, key)
            yield PairwiseJudgement(
                left_id=candidate.left.id,  # type: ignore[attr-defined]
                right_id=candidate.right.id,  # type: ignore[attr-defined]
                score=score,
                score_type=self.score_type,  # type: ignore[arg-type]
                confidence=conf,
                confidence_source=(self.confidence_source if conf is not None else "none"),  # type: ignore[arg-type]
                decision_step=self.decision_step,
                reasoning=self.reasoning,
                provenance=dict(self.provenance),
            )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        """Delegate to the shared :func:`_inspect_scores_impl` (same as every real Matcher)."""
        return _inspect_scores_impl(judgements, sample_size)
