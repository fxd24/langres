"""Prompt-optimization :class:`~langres.core.methods_api.Method` s (``kind == "prompt"``).

The concrete strategies :meth:`Resolver.fit(method=...) <langres.core.resolver.Resolver.fit>`
dispatches to when the module is a compilable
:class:`~langres.core.matchers.dspy_judge.DSPyMatcher`: tune the prompt from
labeled pairs by **compiling** the DSPy program against a gold set. Each strategy
maps to one DSPy optimizer:

- :class:`Bootstrap` -> ``BootstrapFewShot`` (deterministic under ``DummyLM`` --
  the zero-spend path the CI suite exercises);
- :class:`MIPRO` -> ``MIPROv2`` (proposes+evaluates instructions via real LM
  calls -- the paid path, exercised only by the example / a ``slow`` test).

Import-light by construction (Pydantic only -- no ``dspy``/``litellm``): a
``Method`` is a *value* the caller constructs at the fit call site, so
constructing ``MIPRO(budget_usd=5.0)`` must never pull a training backend into
``sys.modules`` (locked by ``tests/test_import_budget.py``). The heavy ``dspy``
import stays lazy inside
:mod:`langres.core.matchers.dspy_judge`, reached only when
:meth:`~langres.core.resolver.Resolver.fit` actually compiles.

.. note::
   The ``metric`` DSPy compiles against is fixed to the built-in match-decision
   metric (``dspy_judge._pair_metric``); a user-selectable metric would be dead
   config today, so these methods deliberately omit a ``metric`` knob.
"""

from typing import Any, ClassVar, Literal

from langres.core.methods_api import Method

__all__ = ["Bootstrap", "MIPRO", "PromptMethod"]


class PromptMethod(Method):
    """Base for the prompt-optimization strategies (``kind == "prompt"``).

    Fixes the contract :meth:`~langres.core.resolver.Resolver._fit_prompt` reads:
    an :attr:`optimizer` naming the DSPy optimizer to run, an optional
    :attr:`budget_usd` cap, and a :meth:`compile_kwargs` hook mapping the
    method's config onto ``DSPyMatcher.compile``'s arguments. Like
    :attr:`~langres.core.methods_api.Method.kind`, :attr:`optimizer` is a
    ``ClassVar`` -- strategy-type identity, not serialized config -- so the base
    declares it and each concrete subclass sets it.

    Attributes:
        budget_usd: Spend cap for the compile, enforced through the existing
            :class:`~langres.clients.openrouter.SpendMonitor` seam. ``None`` (the
            default) means uncapped. Surfaced in :meth:`describe` and the
            ``FitReport`` so a caller sees the budget the fit ran under.
    """

    kind: ClassVar[str] = "prompt"
    #: The ``DSPyMatcher.compile`` optimizer this strategy runs ("bootstrap" /
    #: "mipro"). Set by concrete subclasses; unset on the base (never instantiated).
    optimizer: ClassVar[str]

    budget_usd: float | None = None

    def compile_kwargs(self) -> dict[str, Any]:
        """Return the method's config as ``DSPyMatcher.compile`` keyword arguments.

        The base maps nothing (``BootstrapFewShot`` takes no extra config here);
        subclasses override to thread their knobs (e.g. ``MIPROv2``'s ``auto``
        level). The ``optimizer`` itself is passed separately, not via this dict.
        """
        return {}


def _with_budget(base: str, budget_usd: float | None) -> str:
    """Append ``", budget $<n>"`` to a describe() string when a budget is set."""
    return base if budget_usd is None else f"{base}, budget ${budget_usd:g}"


class Bootstrap(PromptMethod):
    """Prompt-optimize by bootstrapping few-shot demos (``BootstrapFewShot``).

    The zero-spend optimizer: it selects demonstrations that reproduce the gold
    match decision, deterministic under a ``DummyLM`` -- so a
    ``resolver.fit(pairs=..., method=Bootstrap())`` on a ``DummyLM``-backed
    ``DSPyMatcher`` compiles at ``$0`` (the path the unit suite locks).
    """

    optimizer: ClassVar[str] = "bootstrap"

    def describe(self) -> str:
        """One-liner: ``"prompt-optimize (BootstrapFewShot[, budget $N])"``."""
        return _with_budget("prompt-optimize (BootstrapFewShot)", self.budget_usd)


class MIPRO(PromptMethod):
    """Prompt-optimize by proposing+evaluating instructions (``MIPROv2``).

    The paid optimizer: ``MIPROv2`` drafts candidate instructions and scores them
    with real LM calls, so it is non-deterministic and kept out of the zero-spend
    unit suite (exercised only by the example / a ``slow`` test). Pair it with a
    :attr:`budget_usd` cap.

    Attributes:
        auto: ``MIPROv2``'s search-budget preset (``"light"`` / ``"medium"`` /
            ``"heavy"``) -- more compute for a better prompt. Threaded onto
            ``compile`` via :meth:`compile_kwargs`.
    """

    optimizer: ClassVar[str] = "mipro"

    auto: Literal["light", "medium", "heavy"] = "light"

    def compile_kwargs(self) -> dict[str, Any]:
        """Thread the ``auto`` search-budget preset onto ``DSPyMatcher.compile``."""
        return {"auto": self.auto}

    def describe(self) -> str:
        """One-liner: ``"prompt-optimize (MIPROv2, auto=<level>[, budget $N])"``."""
        return _with_budget(f"prompt-optimize (MIPROv2, auto={self.auto})", self.budget_usd)
