"""
langres.core: Low-level API for entity resolution.

This module provides the foundational primitives for building custom
entity resolution pipelines.

Import weight (W0.4): most names below are cheap (pydantic/rapidfuzz/networkx,
the core dependencies) and stay eager. A handful pull an optional, heavy
dependency -- the embedding/vector stack (torch/sentence-transformers/faiss/
qdrant-client, the ``[semantic]`` extra), the LLM stack (litellm, the ``[llm]``
extra), the trained-judge stack (scikit-learn, the ``[trained]`` extra --
:class:`~langres.core.matchers.random_forest_judge.RandomForestMatcher`), or dev/eval tooling
(ranx/optuna/wandb) -- and are resolved lazily via PEP 562 ``__getattr__``
(see :data:`_LAZY_SUBMODULES` / :data:`_LAZY_SYMBOLS` below): ``from
langres.core import VectorBlocker`` still works, but the actual ``import`` of
``faiss``/``torch``/etc. only happens the first time ``VectorBlocker`` (or
another lazy name) is actually accessed -- so plain ``import langres`` stays
fast and never touches ``sys.modules`` for a dependency the caller hasn't
asked for. Accessing a ``[semantic]``/``[llm]``/``[trained]`` symbol without
that extra installed raises a clear ``ImportError`` naming the extra to
install.

**This module is a thin aggregator.** The exports themselves live in
per-domain fragments under :mod:`langres.core._exports` -- one file per domain,
each owning its own eager imports, its ``__all__`` slice, and its slice of the
three lazy maps. A single sorted ~100-name ``__all__`` was the repo's worst
merge-conflict hotspot (21 touches in 30 days): N concurrent streams each
inserting a name at its sorted position = N guaranteed conflicts. Fragments
make those streams edit disjoint files instead.

**To add an export, edit the owning fragment, not this file.** Nothing below is
per-*name*; only a brand new domain touches this module. See
``langres/core/_exports/__init__.py`` for the fragment contract.
"""

import importlib
from typing import Any

from langres.core import _exports

# Bind each fragment's EAGER names into this namespace. Every star-import is
# bounded by that fragment's own `__all__`, so this imports exactly the names
# the fragment declares -- and nothing lazy (a lazy name is deliberately not
# defined at runtime; see the _exports contract).
from langres.core._exports._blocking import *  # noqa: F403
from langres.core._exports._clustering import *  # noqa: F403
from langres.core._exports._eval import *  # noqa: F403
from langres.core._exports._flywheel import *  # noqa: F403
from langres.core._exports._matchers import *  # noqa: F403
from langres.core._exports._methods import *  # noqa: F403
from langres.core._exports._models import *  # noqa: F403
from langres.core._exports._resolver import *  # noqa: F403
from langres.core._exports._semantic import *  # noqa: F403
from langres.core._exports._tracking import *  # noqa: F403
from langres.core._exports._training import *  # noqa: F403

#: The composed public surface -- every fragment's slice, deduplicated and
#: sorted (see :data:`langres.core._exports.NAMES`).
__all__ = list(_exports.NAMES)

#: Names resolved to a *submodule of this package* on first access -- unlike
#: :data:`_LAZY_SYMBOLS`, the value ``__getattr__`` binds is the imported
#: module itself (``langres.core.benchmark``, not an attribute of it).
_LAZY_SUBMODULES: frozenset[str] = _exports.LAZY_SUBMODULES

#: ``name -> owning module`` for symbols resolved on first access. Each entry
#: needs an optional extra installed; see :data:`_EXTRA_BY_SYMBOL` for the
#: ``pip install langres[<extra>]`` hint a missing dependency should surface.
_LAZY_SYMBOLS: dict[str, str] = _exports.LAZY_SYMBOLS

#: ``name -> extra`` for the lazy symbols a ``pip install langres[<extra>]``
#: actually fixes -- everything in :data:`_LAZY_SYMBOLS` except the submodules
#: (dev/eval tooling, not distributed as a pip extra; see
#: :data:`_LAZY_SUBMODULES`). ``RandomForestMatcher``/``Calibrator`` need
#: scikit-learn (``[trained]``); ``LLMMatcher``/``SelectMatcher`` need
#: ``[llm]`` (litellm/dspy-ai); the embedding/vector names need
#: ``[semantic]``; the tracker adapters name their own backend.
_EXTRA_BY_SYMBOL: dict[str, str] = _exports.EXTRA_BY_SYMBOL


def __getattr__(name: str) -> Any:
    """PEP 562: resolve a heavy/optional name the first time it's accessed.

    Raises:
        AttributeError: ``name`` isn't a known attribute of this module.
        ImportError: The owning module's dependency isn't installed --
            re-raised with a ``pip install langres[<extra>]`` hint instead of
            the raw ``ModuleNotFoundError`` (:data:`_EXTRA_BY_SYMBOL`).
    """
    if name in _LAZY_SUBMODULES:
        value: Any = importlib.import_module(f"{__name__}.{name}")
    elif name in _LAZY_SYMBOLS:
        try:
            value = getattr(importlib.import_module(_LAZY_SYMBOLS[name]), name)
        except ImportError as exc:
            extra = _EXTRA_BY_SYMBOL[name]
            raise ImportError(
                f"langres.core.{name} requires the {extra!r} extra: "
                f"pip install 'langres[{extra}]' (or uv add 'langres[{extra}]')"
            ) from exc
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value
