"""The spend cap: one :class:`SpendCappedMatcher` that hard-stops a paid pipeline.

**The enforcer.** Not :func:`effective_budget` -- that only resolves ``None`` to
a default number. Nothing is enforced unless a matcher is wrapped in
:class:`SpendCappedMatcher`.

Why this is a core leaf
-----------------------
This class used to live in :mod:`langres.core.presets`, which sits strictly
*above* :class:`~langres.core.resolver.Resolver` and imports it -- so the
Resolver, the entire low-level public API, could not reach the cap and ran
**uncapped**. Both layers need it, so it lives beneath both, importing only
:mod:`langres.core.spend`, :mod:`langres.core.matcher`,
:mod:`langres.core.inspection` and :mod:`langres.core.models` -- all leaves, so
this module is in no import cycle (``tools/import_graph.py cycles``; the
all-edges SCC stays at 10). ``presets`` re-exports it as ``_SpendCappedMatcher``
for its long-standing callers::

    verbs -> presets -> Resolver -> spend_cap -> spend

What the cap actually guarantees
--------------------------------
**Spend is bounded by ``budget_usd`` plus the cost of at most one further
call.** No more than that, and any doc claiming otherwise is wrong. The cost of
an LLM call is not knowable until it has been made, so "check before" can only
ever mean *"am I already at or over budget? then refuse the NEXT call"*. That
is exactly what this does, in two places:

1. Before pulling anything from the wrapped matcher (so a cap that already
   tripped costs **$0** on every subsequent :meth:`forward` -- the case a
   long-lived ``Resolver`` makes real).
2. After each judgement's metered cost lands, refusing to pull the next one.

The overshoot is one call because the tripping judgement's own cost is only
observable after it was paid for. A tighter bound would need a reliable
*pre-call* cost estimate, which no provider offers.

The monitor is per-INSTANCE, not per-call
-----------------------------------------
The :class:`~langres.core.spend.SpendMonitor` is built once, in
:meth:`__init__`. A fresh monitor per ``forward()`` -- what this class did while
it only ever served one-shot verb calls -- means ``r.resolve(a); r.resolve(b)``
spends **2x** the budget, and N resolves spend N x. Pass ``monitor=`` to share
ONE ledger across several wrappers (what ``Resolver`` does, so its cap survives
``self.module`` being reassigned).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langres.core.inspection import _ensure_inspectable
from langres.core.matcher import Matcher
from langres.core.spend import BudgetExceeded, SpendMonitor

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langres.core.models import ERCandidate, PairwiseJudgement

#: Default spend cap in USD for any capped pipeline -- the verbs
#: (``link``/``dedupe``) and :class:`~langres.core.resolver.Resolver` alike
#: (CEO decision #8), overridable via ``budget_usd=``. Zero-spend matchers
#: (string, embedding) never approach it.
DEFAULT_BUDGET_USD = 1.0

#: The explicit "no cap" budget. ``SpendMonitor.check`` raises on
#: ``spent > budget_usd``, so an infinite budget can never trip (and never
#: warns: ``warn_frac * inf`` is ``inf``). This is the escape hatch for a
#: deliberately unbounded paid run -- ``None`` means "use the default", never
#: "uncapped", so a caller can't disable the cap by forgetting to pass it.
UNCAPPED_BUDGET_USD = float("inf")


def effective_budget(budget_usd: float | None) -> float:
    """Resolve a caller's ``budget_usd=None`` to :data:`DEFAULT_BUDGET_USD` (DRY).

    A *defaulting* helper only -- it enforces nothing. The enforcement lives in
    :class:`SpendCappedMatcher`, which is what a budget must be handed to for
    any of this to bind.
    """
    return DEFAULT_BUDGET_USD if budget_usd is None else budget_usd


class SpendCappedMatcher(Matcher[Any]):
    """Wrap a Matcher, hard-stopping the moment cumulative cost crosses a budget.

    Reuses :class:`~langres.core.spend.SpendMonitor` for the tally + threshold
    check, per pair, and re-raises its
    :class:`~langres.core.spend.BudgetExceeded` with every judgement already
    produced (and paid for) attached as ``.partial_judgements`` (E9; mirrors
    :class:`~langres.core.benchmark.BlindCostError`'s "set by the catcher, not
    at raise time" pattern). For a group-wise module (``SelectMatcher``), a
    group is never split across the cap boundary: the already-paid-for siblings
    of a tripping judgement are drained in too (see ``forward``'s
    ``provenance["group_end"]`` handling).

    The budget binds across the wrapper's whole LIFETIME, not per ``forward``
    call -- see the module docstring. See it there, too, for the exact
    guarantee: budget + at most one further call.

    Deliberately NOT :class:`~langres.core.benchmark.BudgetedModuleRunner`:
    that runner *silently truncates* past its soft cap (correct for the
    benchmark harness, wrong here -- a capped pipeline must raise, never
    silently hand back a partially-scored, partially-clustered result).
    """

    def __init__(
        self,
        module: Matcher[Any],
        *,
        budget_usd: float | None = None,
        monitor: SpendMonitor | None = None,
    ) -> None:
        """Wrap ``module`` in a spend cap.

        Args:
            module: The scorer to meter.
            budget_usd: The cap, in USD. ``None`` resolves to
                :data:`DEFAULT_BUDGET_USD`; :data:`UNCAPPED_BUDGET_USD` never
                trips. Mutually exclusive with ``monitor``.
            monitor: An existing ledger to share instead of building one. This
                is how several wrappers enforce ONE cumulative budget --
                :class:`~langres.core.resolver.Resolver` owns a monitor and
                re-wraps its (reassignable) matcher slot per scoring pass.

        Raises:
            ValueError: If both ``budget_usd`` and ``monitor`` are given -- the
                monitor already carries a budget, so honoring both is
                ambiguous.
        """
        if monitor is not None and budget_usd is not None:
            raise ValueError(
                "pass budget_usd= or monitor=, not both: a SpendMonitor already carries its "
                f"own budget (${monitor.budget_usd:.2f}), so budget_usd={budget_usd!r} would "
                "silently lose."
            )
        self._module = module
        # ONE monitor for this wrapper's lifetime: a monitor built per forward()
        # would reset the tally on every call, so N resolves cost N x budget.
        self._monitor = (
            monitor
            if monitor is not None
            else SpendMonitor(budget_usd=effective_budget(budget_usd))
        )

    @property
    def monitor(self) -> SpendMonitor:
        """The ledger enforcing this cap (``.spent`` / ``.remaining`` are live)."""
        return self._monitor

    def forward(self, candidates: Iterator[ERCandidate[Any]]) -> Iterator[PairwiseJudgement]:
        # Refuse to pull (and pay for) anything if the budget is already gone --
        # the tally survives across forward() calls, so this is what makes a
        # second resolve() on an over-budget Resolver cost $0 instead of one
        # more call. Note forward() is a generator: this runs on the first
        # next(), still before the wrapped matcher is ever pulled.
        self._monitor.check()
        produced: list[PairwiseJudgement] = []
        judgements = self._module.forward(candidates)
        for judgement in judgements:
            produced.append(judgement)
            cost = judgement.provenance.get("cost_usd", 0.0)
            self._monitor.add(float(cost) if cost is not None else 0.0)
            try:
                self._monitor.check()
            except BudgetExceeded as exc:
                # A group-wise module (SelectMatcher) stamps the full call cost
                # on the group's first judgement and $0 on its K-1 siblings,
                # all sharing provenance["group_id"] and with
                # provenance["group_end"] = True on the LAST one (E5,
                # stamp_group_cost). If the cap trips here, those
                # already-paid-for siblings must still land in
                # partial_judgements -- a group must never be split across
                # the cap boundary. Drain them from the same underlying
                # iterator up to (and including) the group_end marker.
                #
                # This must NOT peek at the next judgement's group_id to
                # detect the boundary: for a real GroupwiseMatcher the
                # generator is lazy, so pulling one item past the group's
                # last already-materialized judgement resumes forward_groups
                # and fires the NEXT group's paid LLM call before there is
                # anything to compare against -- silently discarding that
                # judgement and its cost. group_end lets the drain stop
                # exactly at the boundary without ever pulling past it.
                #
                # Because a sibling always carries $0 cost, monitor.check()
                # can only ever trip on a group's FIRST judgement (a passing
                # check means spend was <= budget; adding $0 can't newly
                # exceed it) -- so `judgement` here is always a group's first,
                # never a mid-group sibling, and "not group_end" correctly
                # means "there are siblings left to drain".
                group_id = judgement.provenance.get("group_id")
                if group_id is not None and not judgement.provenance.get("group_end"):
                    for sibling in judgements:
                        produced.append(sibling)
                        if sibling.provenance.get("group_end"):
                            break
                exc.partial_judgements = list(produced)
                raise
            yield judgement

    def inspect_scores(self, judgements: list[PairwiseJudgement], sample_size: int = 10) -> Any:
        """Delegate to the wrapped matcher, which must opt into ``Inspectable``."""
        return _ensure_inspectable(self._module).inspect_scores(judgements, sample_size)
