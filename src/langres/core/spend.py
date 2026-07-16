"""The cumulative-spend ledger: :class:`SpendMonitor` + :class:`BudgetExceeded`.

A **core leaf**, deliberately. This ledger is pure arithmetic over USD --
nothing about it is OpenRouter-specific -- and it must be reachable from the
one layer that has to *enforce* a budget: :class:`~langres.core.resolver.Resolver`
(via :mod:`langres.core.spend_cap`). It used to live in
:mod:`langres.clients.openrouter`, which made that impossible without an import
cycle::

    spend_cap -> clients.openrouter -> core.benchmark -> core.presets -> spend_cap

Measured with ``tools/import_graph.py cycles``, that edge grew the all-edges SCC
from 10 to 11. Here -- importing nothing but :mod:`langres.core.models`, and
that only under ``TYPE_CHECKING`` -- it is in no cycle at all, and a *transport*
module (``clients.openrouter``) no longer owns a *policy* primitive that core
depends on.

``clients.openrouter`` re-exports both names, so the long-standing
``from langres.clients.openrouter import BudgetExceeded, SpendMonitor`` keeps
working unchanged; ``langres.BudgetExceeded`` is unaffected.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langres.core.models import PairwiseJudgement

logger = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """Raised by :meth:`SpendMonitor.check` when cumulative spend passes the budget.

    ``partial_judgements`` carries every judgement already produced (and paid
    for) before the cap tripped (E9) -- populated by the catcher, not here
    (see :class:`~langres.core.spend_cap.SpendCappedMatcher`). Declared with a
    default empty list so any future raiser is safe even if it never sets it,
    and callers/mypy see the attribute without an ad hoc
    ``# type: ignore[attr-defined]`` at the one call site that populates it.
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.partial_judgements: list[PairwiseJudgement] = []


class SpendMonitor:
    """A KISS cumulative-cost ledger for budget-aware paid runs.

    Accumulate the honest cost of each paid call with :meth:`add`, then call
    :meth:`check` to log a warning once spend passes ``warn_frac * budget_usd``
    and raise :class:`BudgetExceeded` once it passes ``budget_usd``. This is a
    monitoring guard, not a hard cap: it never wraps or throttles the LM, it only
    observes and warns/raises. Pure — no I/O beyond ``logging``.

    **The check is ``spent > budget_usd``, strictly.** Landing *exactly* on the
    budget is not a breach, and a ``budget_usd=0`` ledger therefore still lets
    unlimited $0 (free) calls through while refusing the first cent of real
    spend. ``budget_usd=float("inf")`` never raises -- the explicit "no cap"
    value (see :func:`~langres.core.spend_cap.effective_budget`).
    """

    def __init__(self, *, budget_usd: float = 5.0, warn_frac: float = 0.8) -> None:
        """Initialize the ledger.

        Args:
            budget_usd: Total spend budget in USD. :meth:`check` raises past it.
            warn_frac: Fraction of ``budget_usd`` at which :meth:`check` warns.
        """
        self._budget_usd = budget_usd
        self._warn_frac = warn_frac
        self._spent = 0.0

    def add(self, cost_usd: float) -> None:
        """Accumulate ``cost_usd`` into the running total."""
        self._spent += cost_usd

    @property
    def budget_usd(self) -> float:
        """The configured total spend budget (USD)."""
        return self._budget_usd

    @property
    def spent(self) -> float:
        """Cumulative spend recorded so far (USD)."""
        return self._spent

    @property
    def remaining(self) -> float:
        """Budget left before the cap (USD); negative once over budget."""
        return self._budget_usd - self._spent

    def check(self) -> None:
        """Warn past the warn threshold; raise :class:`BudgetExceeded` past the budget.

        Raises:
            BudgetExceeded: If cumulative spend exceeds ``budget_usd``.
        """
        if self._spent > self._budget_usd:
            raise BudgetExceeded(f"spend ${self._spent:.4f} exceeds budget ${self._budget_usd:.2f}")
        if self._spent >= self._warn_frac * self._budget_usd:
            logger.warning(
                "spend $%.4f has passed %.0f%% of the $%.2f budget (remaining $%.4f)",
                self._spent,
                self._warn_frac * 100.0,
                self._budget_usd,
                self.remaining,
            )
