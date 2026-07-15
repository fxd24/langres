"""
langres: A composable entity resolution framework.

This package provides:
- ``link`` / ``dedupe``: the two-verb DX layer (schema-optional, matcher="auto"
  by default, spend-capped) -- see ``langres.verbs``.
- ``langres.core``: Low-level primitives for custom pipelines (``Resolver``,
  ``Blocker``, ``Matcher``, ``Clusterer``, ...).
- The flywheel loop, end to end at the root: ``JudgementLog`` (the inlet --
  wire via ``log=`` on ``link``/``dedupe``), ``select_for_review`` /
  ``ReviewQueue`` (pick the uncertain margin), ``Correction`` /
  ``CorrectionLog`` (the human labels the ``langres import-csv`` CLI writes),
  ``harvest_labeled_pairs`` + ``derive_threshold_from_pairs`` (labels -> a
  tuned threshold; ``derive_threshold`` is the score/label primitive under
  it), and ``gold_pairs_from_clusters`` + ``EvalReport`` (grade a run against
  gold at $0). See ``examples/flywheel_min.py`` for the whole loop in one
  script.
- ``NoMatcherAvailableError`` / ``MatcherAbstainedError`` / ``BudgetExceeded``: the
  exceptions a front-door user must catch (fail-fast ``matcher="auto"``; a judge
  that abstained on the pair; the spend cap).

Import weight: most root exports are cheap and eager -- including the
training-surface pieces that make ``Resolver.fit`` legible: the method objects
``Method`` / ``Bootstrap`` / ``MIPRO`` / ``GEPA`` (prompt) and ``Platt`` /
``Isotonic`` (calibrate), the ``align_pairs`` pairs->candidates bridge, and the
``FitReport`` digest (all import-light config/primitives; dspy/sklearn stay lazy
inside their fit paths). ``EvalReport``, ``gold_pairs_from_clusters``,
``derive_threshold`` and ``LLMMatcher`` resolve lazily via PEP 562
``__getattr__`` (same pattern as ``langres.core``) so a bare ``import langres``
never pulls the eval-report/benchmark modules -- or scikit-learn
(``derive_threshold``, the ``[trained]`` extra) or litellm (``LLMMatcher``, the
``[llm]`` extra) -- into ``sys.modules``. See ``tests/test_import_budget.py``.
"""

import importlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version
from typing import TYPE_CHECKING, Any

from langres.clients.openrouter import BudgetExceeded
from langres.core import (
    Bootstrap,
    CompanySchema,
    Correction,
    CorrectionLog,
    ERCandidate,
    FitReport,
    GEPA,
    Isotonic,
    JudgementLog,
    MIPRO,
    Method,
    PairwiseJudgement,
    Platt,
    Resolver,
    ReviewQueue,
    align_pairs,
    derive_threshold_from_pairs,
    harvest_labeled_pairs,
    select_for_review,
)
from langres.core.models import MatcherAbstainedError
from langres.core.presets import DEFAULT_AUTO_MODEL, NoMatcherAvailableError
from langres.optimize import optimize, score_blocking
from langres.verbs import LinkVerdict, dedupe, link

if TYPE_CHECKING:
    # Only reached by mypy (never at runtime) -- keeps the lazy names visible
    # to `mypy --strict` without executing the imports below on a bare import.
    from langres.core.benchmark import gold_pairs_from_clusters
    from langres.core.calibration import derive_threshold
    from langres.data.data_profile import DataProfileReport
    from langres.core.eval_report import EvalReport
    from langres.core.matchers.llm_judge import LLMMatcher

__all__ = [
    "Bootstrap",
    "BudgetExceeded",
    "CompanySchema",
    "DEFAULT_AUTO_MODEL",
    "Correction",
    "CorrectionLog",
    "DataProfileReport",
    "ERCandidate",
    "EvalReport",
    "FitReport",
    "GEPA",
    "Isotonic",
    "MatcherAbstainedError",
    "Method",
    "MIPRO",
    "JudgementLog",
    "LinkVerdict",
    "LLMMatcher",
    "NoMatcherAvailableError",
    "PairwiseJudgement",
    "Platt",
    "QLoRA",
    "Resolver",
    "ReviewQueue",
    "align_pairs",
    "dedupe",
    "derive_threshold",
    "derive_threshold_from_pairs",
    "finetune",
    "gold_pairs_from_clusters",
    "harvest_labeled_pairs",
    "link",
    "optimize",
    "run_finetune",
    "score_blocking",
    "select_for_review",
]

# Single source of truth is pyproject.toml; resolved from installed metadata so
# a version bump can never miss this string again. Falls back for source trees
# imported without installation.
try:
    __version__ = _metadata_version("langres")
except PackageNotFoundError:  # pragma: no cover - only hit on uninstalled source trees
    __version__ = "0.0.0.dev0"

#: ``name -> owning module`` for root exports resolved on first access (PEP
#: 562, mirroring ``langres.core.__getattr__``). ``EvalReport`` and
#: ``gold_pairs_from_clusters`` are import-light but live in modules kept out
#: of the eager import graph on purpose (``core.eval_report`` pulls
#: ``core.benchmark``/``core.metrics``); ``derive_threshold`` imports
#: scikit-learn at module scope (the ``[trained]`` extra).
_LAZY_SYMBOLS: dict[str, str] = {
    "EvalReport": "langres.core.eval_report",
    "DataProfileReport": "langres.data.data_profile",
    "derive_threshold": "langres.core.calibration",
    "gold_pairs_from_clusters": "langres.core.benchmark",
    # The LLM matcher (serve a fine-tuned model_ref, a vLLM api_base, or a paid
    # judge). Importing the class pulls litellm -> the [llm] extra, so it stays
    # lazy: a bare `import langres` never touches litellm.
    "LLMMatcher": "langres.core.matchers.llm_judge",
    # The finetune surface: the module is import-light (peft/trl/torch import
    # lazily inside QLoRATrainer.train), so these symbols carry no [finetune] extra
    # here -- an ImportError from importing the symbols is a genuine bug. The
    # actionable "pip install langres[finetune]" hint is raised at train time.
    "QLoRA": "langres.core.finetune",
    "finetune": "langres.core.finetune",
    "run_finetune": "langres.core.finetune",
}

#: ``name -> extra`` for the lazy symbols where a missing dependency has a
#: ``pip install langres[<extra>]`` fix. Symbols absent here need no extra --
#: an ImportError from them is a genuine bug and propagates unchanged.
_EXTRA_BY_SYMBOL: dict[str, str] = {
    "derive_threshold": "trained",
    "LLMMatcher": "llm",
}


def __getattr__(name: str) -> Any:
    """PEP 562: resolve a lazy root export the first time it's accessed.

    Raises:
        AttributeError: ``name`` isn't a known attribute of this module.
        ImportError: The owning module's optional dependency isn't installed --
            re-raised with a ``pip install langres[<extra>]`` hint when
            :data:`_EXTRA_BY_SYMBOL` knows the extra that fixes it.
    """
    if name not in _LAZY_SYMBOLS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        value = getattr(importlib.import_module(_LAZY_SYMBOLS[name]), name)
    except ImportError as exc:
        extra = _EXTRA_BY_SYMBOL.get(name)
        if extra is None:
            raise
        raise ImportError(
            f"langres.{name} requires the {extra!r} extra: "
            f"pip install 'langres[{extra}]' (or uv add 'langres[{extra}]')"
        ) from exc
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value
