"""Reusable explicit-chain fixtures for the epic #193 generalized spine (PR-A on).

A SYNTHETIC, ``$0``, offline explicit Op chain built via
:meth:`~langres.core._model_state.ModelState.from_topology` -- the *else* branch
of the generalized spine, the topology the four legacy slots cannot express (a
Score after a Select, a second matcher).

Two things make it honest without spending a cent:

- **Its Scores are spend-capped by the door, from RAW matchers.** These builders
  pass an *unwrapped* matcher into each ``MatcherScore`` on purpose:
  ``from_topology`` is what enforces the cap, auto-wrapping every ``MatcherScore``
  in a :class:`~langres.core.spend_cap.SpendCappedMatcher` sharing the model's
  ledger. That is the guarantee under test -- a paid explicit-chain Score can no
  longer bill off-ledger even when the caller forgot to wrap it (the
  ``.module.forward`` AST-ban only catches the literal slot call). The chain and
  the model share ONE :class:`~langres.core.spend.SpendMonitor` (passed to
  ``from_topology(monitor=)``). ``build_precapped_chain_model`` covers the other
  branch: a matcher the caller already wrapped is left alone (no double-wrap).
- **The "paid" matcher is a fake** stamping a made-up ``cost_usd`` (no network, no
  key -- exactly like ``tests/core/test_resolver_spend_cap.py``), so the ledger
  moves under a real budget without real spend.

Reused by the later persist PR (a ``from_topology`` chain must ``save``/``load``),
so the builders here stay clean and importable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from pydantic import BaseModel

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.blockers.key import KeyBlocker
from langres.core.clusterer import Clusterer
from langres.core.comparators import StringComparator
from langres.core.matcher import Matcher
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.op import Stage, ThresholdSelect, TopKSelect
from langres.core.op_adapters import (
    BlockerSource,
    ClustererStage,
    ComparatorScore,
    MatcherScore,
)
from langres.core.resolver import ERModel
from langres.core.score_type import ScoreType
from langres.core.spend import SpendMonitor
from langres.core.spend_cap import SpendCappedMatcher, effective_budget


class ChainCo(BaseModel):
    """The tiny two-field entity the fixtures resolve."""

    id: str
    name: str | None = None


#: Two Acme duplicates ({a1, a2}), one Beta and one Gamma singleton -- so a
#: name-equality matcher merges exactly {a1, a2}.
RECORDS: list[dict[str, Any]] = [
    {"id": "a1", "name": "Acme"},
    {"id": "a2", "name": "Acme"},
    {"id": "b1", "name": "Beta"},
    {"id": "c1", "name": "Gamma"},
]

#: The match cut every fixture chain thresholds at.
THRESHOLD = 0.5


class CostedNameMatcher(Matcher[Any]):
    """``$0`` fake matcher: scores ``1.0`` iff the two names match else ``0.0``.

    Tags a chosen ``score_type`` and stamps a made-up ``cost_usd`` so the spend
    ledger moves under a real budget -- the cost is fabricated, there is no
    network and no key (the same technique as ``test_resolver_spend_cap``). An
    optional ``model`` is advertised so :attr:`~langres.core._model_state.ModelState.backbone`
    reporting through the chain can be exercised.
    """

    def __init__(
        self,
        *,
        cost_each: float = 0.0,
        score_type: ScoreType = "prob_llm",
        model: str | None = None,
    ) -> None:
        self._cost_each = cost_each
        self._score_type: ScoreType = score_type
        self.produced = 0
        #: Advertised backbone id (``None`` = weightless, like a string matcher).
        self.model: str | None = model

    def forward(self, candidates: Iterator[ERCandidate[Any]]) -> Iterator[PairwiseJudgement]:
        for candidate in candidates:
            self.produced += 1
            yield PairwiseJudgement(
                left_id=str(candidate.left.id),
                right_id=str(candidate.right.id),
                score=1.0 if candidate.left.name == candidate.right.name else 0.0,
                score_type=self._score_type,
                decision_step="costed_name",
                provenance={"cost_usd": self._cost_each},
            )

    def inspect_scores(self, judgements: list[PairwiseJudgement], sample_size: int = 10) -> Any:
        raise NotImplementedError


class AbstainingMatcher(Matcher[Any]):
    """``$0`` fake that abstains on every pair: a stamped ``score_type`` but neither a
    ``score`` nor a ``decision``. Used to exercise ``compare``'s
    :class:`~langres.core.models.MatcherAbstainedError` path on an explicit chain."""

    def forward(self, candidates: Iterator[ERCandidate[Any]]) -> Iterator[PairwiseJudgement]:
        for candidate in candidates:
            yield PairwiseJudgement(
                left_id=str(candidate.left.id),
                right_id=str(candidate.right.id),
                score=None,
                score_type="prob_llm",
                decision=None,
                decision_step="abstain",
                provenance={},
            )

    def inspect_scores(self, judgements: list[PairwiseJudgement], sample_size: int = 10) -> Any:
        raise NotImplementedError


def chain_ops(
    *,
    threshold: float = THRESHOLD,
    cost_each: float = 0.0,
    model: str | None = None,
    source: Callable[[], Any] | None = None,
    with_comparator: bool = True,
) -> tuple[list[Stage], CostedNameMatcher]:
    """A canonical explicit chain (with a RAW matcher) and that matcher.

    ``BlockerSource -> [ComparatorScore] -> MatcherScore(costed) -> ThresholdSelect
    -> ClustererStage(Clusterer(0.0))`` -- the match cut is the explicit
    ThresholdSelect and the clusterer runs pure transitive closure over the
    survivors. The matcher is passed UNWRAPPED: ``from_topology`` is what caps it.
    Returns the ops and the raw ``CostedNameMatcher`` so a test can read
    ``matcher.produced``.
    """
    matcher = CostedNameMatcher(cost_each=cost_each, model=model)
    build_source = source if source is not None else (lambda: AllPairsBlocker(schema=ChainCo))
    ops: list[Stage] = [BlockerSource(build_source())]
    if with_comparator:
        ops.append(ComparatorScore(StringComparator.from_schema(ChainCo)))
    ops += [
        MatcherScore(matcher, out_space="prob_llm"),
        ThresholdSelect(threshold),
        ClustererStage(Clusterer(threshold=0.0)),
    ]
    return ops, matcher


def build_explicit_chain_model(
    *, budget_usd: float = 1.0, cost_each: float = 0.05, model: str | None = None
) -> tuple[ERModel, SpendMonitor, CostedNameMatcher]:
    """The canonical explicit-chain model + its shared ledger + its raw matcher.

    The matcher goes in raw; ``from_topology(monitor=)`` caps it, so the returned
    model's Score bills through ``monitor``.
    """
    monitor = SpendMonitor(budget_usd=effective_budget(budget_usd))
    ops, matcher = chain_ops(cost_each=cost_each, model=model)
    return ERModel.from_topology(ops=ops, monitor=monitor), monitor, matcher


def build_precapped_chain_model(
    *, budget_usd: float = 1.0
) -> tuple[ERModel, SpendMonitor, SpendCappedMatcher]:
    """A chain whose ``MatcherScore`` matcher the CALLER already wrapped in a
    :class:`~langres.core.spend_cap.SpendCappedMatcher` -- so ``from_topology`` must
    leave it alone (no double-wrap). Returns the model, its ledger, and the exact
    capped matcher so a test can assert identity is preserved."""
    monitor = SpendMonitor(budget_usd=effective_budget(budget_usd))
    capped = SpendCappedMatcher(CostedNameMatcher(), monitor=monitor)
    ops: list[Stage] = [
        BlockerSource(AllPairsBlocker(schema=ChainCo)),
        MatcherScore(capped, out_space="prob_llm"),
        ThresholdSelect(THRESHOLD),
        ClustererStage(Clusterer(threshold=0.0)),
    ]
    return ERModel.from_topology(ops=ops, monitor=monitor), monitor, capped


def build_score_after_select_model(*, budget_usd: float = 1.0, k: int = 5) -> ERModel:
    """A chain exercising Score-after-Select: a cheap ``sim_cos`` student, a
    ``TopKSelect`` prune, then an escalated ``prob_llm`` matcher, then the cut.
    Both matchers go in raw; the door caps each."""
    monitor = SpendMonitor(budget_usd=effective_budget(budget_usd))
    ops: list[Stage] = [
        BlockerSource(AllPairsBlocker(schema=ChainCo)),
        MatcherScore(CostedNameMatcher(score_type="sim_cos"), out_space="sim_cos"),
        TopKSelect(k=k),
        MatcherScore(CostedNameMatcher(score_type="prob_llm"), out_space="prob_llm"),
        ThresholdSelect(THRESHOLD),
        ClustererStage(Clusterer(threshold=0.0)),
    ]
    return ERModel.from_topology(ops=ops, monitor=monitor)


def build_no_threshold_chain_model(*, budget_usd: float = 1.0) -> ERModel:
    """A chain with NO terminal ThresholdSelect: the cut lives in the ClusterStage's
    clusterer instead. Exercises ``dedupe``'s ``threshold=None`` report and
    ``compare``'s no-cut guard."""
    monitor = SpendMonitor(budget_usd=effective_budget(budget_usd))
    ops: list[Stage] = [
        BlockerSource(AllPairsBlocker(schema=ChainCo)),
        MatcherScore(CostedNameMatcher(), out_space="prob_llm"),
        ClustererStage(Clusterer(threshold=THRESHOLD)),
    ]
    return ERModel.from_topology(ops=ops, monitor=monitor)


def build_key_source_model(*, budget_usd: float = 1.0) -> ERModel:
    """A chain whose Source is a ``KeyBlocker`` (blocks on name), so ``compare`` on
    two DIFFERENT-name records yields no candidate and must build the pair directly
    (exercises the ``_chain_pair_candidate`` fallback)."""
    monitor = SpendMonitor(budget_usd=effective_budget(budget_usd))
    ops, _matcher = chain_ops(
        source=lambda: KeyBlocker(schema=ChainCo, key_field="name"),
        with_comparator=False,
    )
    return ERModel.from_topology(ops=ops, monitor=monitor)


def build_factory_source_model(*, budget_usd: float = 1.0) -> ERModel:
    """A chain whose Source blocker carries no schema (an opaque ``schema_factory``),
    so ``_chain_source_schema`` is ``None`` and the front door infers a schema."""
    monitor = SpendMonitor(budget_usd=effective_budget(budget_usd))
    ops, _matcher = chain_ops(
        source=lambda: AllPairsBlocker(schema_factory=lambda record: ChainCo(**record)),
        with_comparator=False,
    )
    return ERModel.from_topology(ops=ops, monitor=monitor)


def build_abstaining_chain_model(*, budget_usd: float = 1.0) -> ERModel:
    """A chain whose matcher abstains on every pair -- so ``compare`` raises
    :class:`~langres.core.models.MatcherAbstainedError` rather than fabricating a verdict."""
    monitor = SpendMonitor(budget_usd=effective_budget(budget_usd))
    ops: list[Stage] = [
        BlockerSource(AllPairsBlocker(schema=ChainCo)),
        MatcherScore(AbstainingMatcher(), out_space="prob_llm"),
        ThresholdSelect(THRESHOLD),
        ClustererStage(Clusterer(threshold=0.0)),
    ]
    return ERModel.from_topology(ops=ops, monitor=monitor)
