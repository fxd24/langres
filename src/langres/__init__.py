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

**This module is a thin aggregator.** The exports -- eager imports included --
live in per-domain fragments under :mod:`langres._exports`, one file per
work-stream (verbs, optimize, core, flywheel, training, data). Both the sorted
``__all__`` and the eager import block were merge-conflict hotspots: their lines
belong to different streams, so any two streams collided on this file.

**To add an export, edit the owning fragment, not this file.** Nothing below is
per-*name*; only a brand new domain touches this module. See
``langres/_exports/__init__.py`` for the fragment contract.
"""

import importlib
from typing import Any

from langres import _exports

# Re-exported (`as` marks it explicit for type checkers -- langres ships
# py.typed, and `__version__` is not in `__all__` to carry the re-export). The
# computation lives in a stdlib-only leaf so `core.resolver` can read the
# version WITHOUT importing this package: that one edge was the whole runtime
# import cycle. See `langres/_version.py` and `tests/test_import_tangle.py`.
from langres._version import __version__ as __version__

# Bind each fragment's EAGER names into this namespace. Every star-import is
# bounded by that fragment's own `__all__`, so this imports exactly the names
# the fragment declares -- and nothing lazy (a lazy name is deliberately not
# defined at runtime; see the _exports contract).
from langres._exports._core import *  # noqa: F403
from langres._exports._data import *  # noqa: F403
from langres._exports._flywheel import *  # noqa: F403
from langres._exports._optimize import *  # noqa: F403
from langres._exports._training import *  # noqa: F403
from langres._exports._verbs import *  # noqa: F403

#: The composed public surface -- every fragment's slice, deduplicated and
#: sorted (see :data:`langres._exports.NAMES`).
__all__ = list(_exports.NAMES)

#: ``name -> owning module`` for root exports resolved on first access (PEP
#: 562, mirroring ``langres.core.__getattr__``).
_LAZY_SYMBOLS: dict[str, str] = _exports.LAZY_SYMBOLS

#: ``name -> extra`` for the lazy symbols where a missing dependency has a
#: ``pip install langres[<extra>]`` fix. Symbols absent here need no extra --
#: an ImportError from them is a genuine bug and propagates unchanged.
_EXTRA_BY_SYMBOL: dict[str, str] = _exports.EXTRA_BY_SYMBOL


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
