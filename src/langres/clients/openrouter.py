"""Reusable OpenRouter cost/pricing helpers + a KISS cumulative-spend monitor.

OpenRouter (via LiteLLM or the OpenAI SDK) is the paid backend for langres
experiments, but LiteLLM prices most OpenRouter models at ``$0`` (their
``response.model`` carries a *dated* id like ``z-ai/glm-5.2-20260616`` with no
provider prefix, so ``completion_cost`` cannot resolve a price). That silently
reports ``$0`` spend and hides real cost from any budget tally.

This module lifts the *reusable* plumbing that makes cost honest — pinning
published per-token prices into ``litellm.model_cost`` (including the dated
runtime id), pricing judgements from their captured token counts, and building a
no-keep-alive HTTP client that never stalls on a dead socket — out of the
experiment scripts that first grew it, plus a small :class:`SpendMonitor` for
budget-aware paid runs.

Experiment *policy* (which models to race, per-cell ledgers, model-selection
lists) deliberately stays in the calling script: :data:`PRICES_PER_1M` is a
*default* price table callers may override or extend, not frozen policy.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langres.core.benchmark import CostTrack
    from langres.core.models import PairwiseJudgement

logger = logging.getLogger(__name__)


#: Default published per-1M-token ``(input, output)`` USD prices, pinned so cost
#: is honest even when LiteLLM prices a model at ``$0``. This is *data*, not
#: policy: pass a caller-owned mapping to any helper below to override or extend
#: it (e.g. to add a new model, or use negotiated rates).
PRICES_PER_1M: dict[str, tuple[float, float]] = {
    "openrouter/z-ai/glm-5.2": (0.95, 3.00),
    "openrouter/z-ai/glm-4.6": (0.60, 2.20),
    "openrouter/openai/gpt-4o": (2.50, 10.00),
    # OpenRouter path for judge="auto" (OPENROUTER_API_KEY set) — see
    # langres.core.presets.choose_auto_judge. OpenRouter's own listing for this
    # route isn't in LiteLLM's pricing table; conservative-high over OpenAI's
    # published $0.15/$0.60 per-1M list price (litellm model_prices_and_context_
    # window.json, "gpt-4o-mini", checked 2026-07) to absorb any provider markup.
    "openrouter/openai/gpt-4o-mini": (0.20, 0.80),
    # Direct-OpenAI path for judge="auto" (OPENAI_API_KEY set, no OPENROUTER_API_KEY)
    # — see choose_auto_judge. Matches OpenAI's published gpt-5-mini list price
    # exactly (litellm model_prices_and_context_window.json, "gpt-5-mini" and
    # "openrouter/openai/gpt-5-mini" agree: $0.25/$2.00 per 1M, checked 2026-07).
    "openai/gpt-5-mini": (0.25, 2.00),
    "openrouter/anthropic/claude-3.7-sonnet": (3.00, 15.00),
    "openrouter/anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "openrouter/google/gemini-2.0-flash-001": (0.10, 0.40),
}

#: Default per-call timeout (seconds) for the no-keep-alive HTTP client. LiteLLM
#: ships a 6000s default, so a stalled keep-alive socket can hang a whole
#: sequential run; a bounded timeout makes a stalled call fail fast and recover.
DEFAULT_TIMEOUT_S = 60.0

#: The OpenRouter route ``judge="auto"``/``judge="zero_shot_llm"`` default to
#: when no model id is given -- ``core.presets.choose_auto_judge`` and
#: ``Resolver.from_schema`` both need this literal. Defined once here (the
#: dspy-free, cycle-safe layer both of those sit above) so the two call sites
#: can't drift on the string.
DEFAULT_OPENROUTER_MODEL = "openrouter/openai/gpt-4o-mini"


# ---------------------------------------------------------------------------
# Price pinning
# ---------------------------------------------------------------------------


def _price_for(model: str, prices: Mapping[str, tuple[float, float]]) -> tuple[float, float]:
    """Look up ``model``'s ``(input, output)`` per-1M price, with a descriptive error.

    A bare ``prices[model]`` raises ``KeyError(model)`` — unhelpful on a typo'd or
    unpinned id. This names the offending id and the known ones so the fix is obvious.
    """
    try:
        return prices[model]
    except KeyError:
        raise KeyError(f"unknown model {model!r}; known: {sorted(prices)}") from None


def patch_litellm_prices(
    model: str, prices: Mapping[str, tuple[float, float]] = PRICES_PER_1M
) -> None:
    """Pin ``model``'s per-token price into ``litellm.model_cost`` so cost is honest.

    LiteLLM prices many OpenRouter models at ``$0`` (unknown), which silently
    reports ``$0`` spend and (for a budget tally) hides real cost. We write the
    pinned per-token price under **both** the LiteLLM-routing key
    (``openrouter/...``) and the bare provider key (``z-ai/glm-5.2``) that an
    OpenAI/OpenRouter response carries in ``response.model``, so both the LiteLLM
    and OpenAI-client cost paths resolve.

    Args:
        model: The routing model id (e.g. ``"openrouter/z-ai/glm-5.2"``). Must be
            a key of ``prices``.
        prices: Per-1M-token ``(input, output)`` price table. Defaults to
            :data:`PRICES_PER_1M`; pass your own to override or extend it.
    """
    import litellm

    in_per_1m, out_per_1m = _price_for(model, prices)
    entry = {
        "input_cost_per_token": in_per_1m / 1_000_000.0,
        "output_cost_per_token": out_per_1m / 1_000_000.0,
        "litellm_provider": "openrouter",
        "mode": "chat",
    }
    litellm.model_cost[model] = entry
    bare = model.split("/", 1)[1] if "/" in model else model
    litellm.model_cost[bare] = entry


def register_runtime_model_price(
    model: str, prices: Mapping[str, tuple[float, float]] = PRICES_PER_1M
) -> str | None:
    """Probe ``model`` once and pin its *dated* runtime id into ``model_cost``.

    OpenRouter returns a dated model id (e.g. ``z-ai/glm-5.2-20260616``) in
    ``response.model``, and ``litellm.completion_cost(completion_response=r)``
    resolves the price against that dated id — which is absent from LiteLLM's
    table, so cost silently reports ``$0``. We make ONE cheap call, read the
    dated id, and pin the published price under it so the cost path becomes
    honest.

    Args:
        model: The routing model id to probe. Must be a key of ``prices``.
        prices: Per-1M-token ``(input, output)`` price table. Defaults to
            :data:`PRICES_PER_1M`.

    Returns:
        The dated model id, or ``None`` if ``model`` is unknown to ``prices`` or
        the probe call failed (the caller then tries a fallback model).
    """
    import litellm

    if model not in prices:
        return None
    patch_litellm_prices(model, prices)
    try:
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0,
            max_tokens=1,
        )
    except Exception as exc:  # noqa: BLE001 — probe; caller falls back on None
        logger.warning("price probe failed for %s: %s: %s", model, type(exc).__name__, exc)
        return None
    dated = str(resp.model)
    in_per_1m, out_per_1m = _price_for(model, prices)
    litellm.model_cost[dated] = {
        "input_cost_per_token": in_per_1m / 1_000_000.0,
        "output_cost_per_token": out_per_1m / 1_000_000.0,
        "litellm_provider": "openrouter",
        "mode": "chat",
    }
    logger.info(
        "registered dated id %r for %s at $%s/$%s per 1M", dated, model, in_per_1m, out_per_1m
    )
    return dated


# ---------------------------------------------------------------------------
# Cost from token counts
# ---------------------------------------------------------------------------


def per_token_worst_price(
    model: str, prices: Mapping[str, tuple[float, float]] = PRICES_PER_1M
) -> float:
    """Worst-case per-token price (the dearer of input/output), for a budget cap.

    Args:
        model: The routing model id. Must be a key of ``prices``.
        prices: Per-1M-token ``(input, output)`` price table. Defaults to
            :data:`PRICES_PER_1M`.
    """
    in_per_1m, out_per_1m = _price_for(model, prices)
    return max(in_per_1m, out_per_1m) / 1_000_000.0


def dspy_price_per_1k(
    model: str, prices: Mapping[str, tuple[float, float]] = PRICES_PER_1M
) -> float:
    """Per-1k-token price for ``model`` from ``prices`` (0.0 if unknown).

    ``DSPyJudge`` prices each pair as ``tokens/1000 * price_per_1k_tokens`` -- a
    single blended per-1k rate over ``prompt + completion`` tokens -- so this
    takes the worst-case (dearer of input/output) per-token price and scales it
    to per-1k. Worst-case is the same price basis
    :class:`~langres.core.benchmark.BudgetedModuleRunner` uses for its cap, so a
    judge's self-reported cost and a live budget-stop agree.

    Unknown models keep ``0.0`` (zero-spend/test runs stay free and never
    crash), mirroring :func:`register_runtime_model_price` returning ``None``
    for unknown ids -- rather than guessing a price.

    This function lives here (not in ``langres.methods``, where it started) so
    both ``langres.methods`` and ``langres.core.presets`` can import it without
    creating a ``core -> methods -> core`` cycle -- this module is dspy-free and
    layer-neutral, sitting below both.

    Args:
        model: The routing model id.
        prices: Per-1M-token ``(input, output)`` price table. Defaults to
            :data:`PRICES_PER_1M`.

    Returns:
        The worst-case per-1k-token price in USD, or ``0.0`` if ``model`` is
        unknown to ``prices``.
    """
    if model not in prices:
        return 0.0
    return per_token_worst_price(model, prices) * 1_000.0


def make_token_cost_track(
    model: str, prices: Mapping[str, tuple[float, float]] = PRICES_PER_1M
) -> Callable[[list[PairwiseJudgement]], CostTrack]:
    """Build a cost function that prices judgements from their captured token counts.

    ``litellm.completion_cost`` returns ``$0`` for OpenRouter responses (their
    dated ``response.model`` has no provider prefix, so LiteLLM raises "LLM
    Provider NOT provided" and the judge swallows it to ``$0``). The judge does,
    however, record ``prompt_tokens`` / ``completion_tokens`` in provenance — so
    we price the run deterministically from those counts against the pinned
    per-1M rates. This is the honest, source-of-truth cost for paid cells.

    Args:
        model: The routing model id. Must be a key of ``prices``.
        prices: Per-1M-token ``(input, output)`` price table. Defaults to
            :data:`PRICES_PER_1M`.

    Returns:
        A ``track(judgements) -> CostTrack`` closure priced against ``model``.
    """
    from langres.core.benchmark import CostTrack

    in_per_1m, out_per_1m = _price_for(model, prices)
    in_per_tok, out_per_tok = in_per_1m / 1_000_000.0, out_per_1m / 1_000_000.0

    def track(judgements: list[PairwiseJudgement]) -> CostTrack:
        usd_total = 0.0
        for j in judgements:
            prov = j.provenance
            usd_total += int(prov.get("prompt_tokens", 0) or 0) * in_per_tok
            usd_total += int(prov.get("completion_tokens", 0) or 0) * out_per_tok
        n = len(judgements)
        per_pair = usd_total / n if n > 0 else 0.0
        return CostTrack(
            usd_total=usd_total,
            usd_per_1k_pairs=per_pair * 1_000.0,
            est_usd_per_100k=per_pair * 100_000.0,
        )

    return track


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


def no_keepalive_http_client(timeout_s: float = DEFAULT_TIMEOUT_S) -> Any:
    """Build an httpx client that never reuses a connection (fresh socket per request).

    LiteLLM/httpx reusing ONE keep-alive socket across thousands of sequential
    calls can wedge a whole run: when that socket goes stale the run stalls
    (~2 KiB / 24 s) even though fresh calls work instantly. Disabling keep-alive
    (``max_keepalive_connections=0``) forces a new connection per call, so a dead
    socket can never persist and stall the run.

    Args:
        timeout_s: Per-request timeout in seconds (default :data:`DEFAULT_TIMEOUT_S`).

    Returns:
        An ``httpx.Client`` configured with no keep-alive and the given timeout.
    """
    import httpx

    return httpx.Client(
        limits=httpx.Limits(max_keepalive_connections=0, max_connections=8),
        timeout=httpx.Timeout(timeout_s),
    )


# ---------------------------------------------------------------------------
# Spend monitor
# ---------------------------------------------------------------------------


class BudgetExceeded(RuntimeError):
    """Raised by :meth:`SpendMonitor.check` when cumulative spend passes the budget.

    ``partial_judgements`` carries every judgement already produced (and paid
    for) before the cap tripped (E9) -- populated by the catcher, not here
    (see :class:`~langres.core.presets._SpendCappedModule`). Declared with a
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
