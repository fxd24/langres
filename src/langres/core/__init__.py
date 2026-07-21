"""
langres.core: the **contracts** of entity resolution.

This module carries the foundational types a custom pipeline is written
against -- the data models (``ERCandidate``, ``PairwiseJudgement``, ...), the
``Blocker`` / ``Comparator`` / ``Matcher`` / ``Clusterer`` base types, the
``Resolver`` and its save/load registry, the method registry, and the
training/tracking primitives.

**It does not re-export implementations.** A concrete blocker, matcher,
clusterer, embedder or index is imported from the package that owns it::

    from langres.core.blockers import AllPairsBlocker, VectorBlocker
    from langres.core.matchers import CascadeMatcher, LLMMatcher
    from langres.core.clusterers import CorrelationClusterer
    from langres.core.embeddings import SentenceTransformerEmbedder
    from langres.core.indexes import FAISSIndex
    from langres.metrics.debugging import PipelineDebugger
    import langres.metrics.metrics  # (langres.core.benchmark for the eval harness)

That indirection was removed deliberately: re-exporting implementations put
this module *above* the components it is supposed to sit beneath, which knotted
the import graph (``tests/test_import_tangle.py`` is the ratchet that measures
it). ``langres.core`` is the floor, not the ceiling -- the contracts import
nothing that reaches back up, so they cannot form a cycle.

Import weight (W0.4): the contracts are cheap (pydantic/rapidfuzz/networkx, the
core dependencies) and eager. The few names that pull an optional, heavy
dependency -- ``Calibrator`` (scikit-learn, ``[trained]``) and the
``MlflowTracker``/``WandbTracker`` adapters -- resolve lazily via PEP 562
``__getattr__`` (:data:`_LAZY_SYMBOLS`), so a plain ``import langres`` never
touches ``sys.modules`` for a dependency the caller hasn't asked for.
Accessing one without its extra installed raises a clear ``ImportError`` naming
the extra to install. The implementation packages above keep their own lazy
seams (e.g. ``langres.core.matchers.__getattr__`` for ``LLMMatcher``), so those
imports are exactly as light as they were here.

**This module is a thin aggregator.** The exports themselves live in
per-domain fragments under :mod:`langres.core._exports` -- one file per domain,
each owning its own eager imports, its ``__all__`` slice, and its slice of the
two lazy maps. A single sorted ~100-name ``__all__`` was the repo's worst
merge-conflict hotspot (21 touches in 30 days): N concurrent streams each
inserting a name at its sorted position = N guaranteed conflicts. Fragments
make those streams edit disjoint files instead.

**To add an export, edit the owning fragment, not this file** -- and only if it
is a *contract*. Nothing below is per-*name*; only a brand new domain touches
this module. See ``langres/core/_exports/__init__.py`` for the fragment
contract.
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
from langres.core._exports._flywheel import *  # noqa: F403
from langres.core._exports._matchers import *  # noqa: F403
from langres.core._exports._methods import *  # noqa: F403
from langres.core._exports._models import *  # noqa: F403
from langres.core._exports._ops import *  # noqa: F403
from langres.core._exports._resolver import *  # noqa: F403
from langres.core._exports._tracking import *  # noqa: F403
from langres.core._exports._training import *  # noqa: F403

#: The composed public surface -- every fragment's slice, deduplicated and
#: sorted (see :data:`langres.core._exports.NAMES`).
__all__ = list(_exports.NAMES)

#: ``name -> owning module`` for symbols resolved on first access. Each entry
#: needs an optional extra installed; see :data:`_EXTRA_BY_SYMBOL` for the
#: ``pip install langres[<extra>]`` hint a missing dependency should surface.
_LAZY_SYMBOLS: dict[str, str] = _exports.LAZY_SYMBOLS

#: ``name -> extra`` for the lazy symbols a ``pip install langres[<extra>]``
#: actually fixes. ``Calibrator`` needs scikit-learn (``[trained]``); the
#: tracker adapters name their own backend.
_EXTRA_BY_SYMBOL: dict[str, str] = _exports.EXTRA_BY_SYMBOL


def __getattr__(name: str) -> Any:
    """PEP 562: resolve a heavy/optional name the first time it's accessed.

    Raises:
        AttributeError: ``name`` isn't a known attribute of this module.
        ImportError: The owning module's dependency isn't installed --
            re-raised with a ``pip install langres[<extra>]`` hint instead of
            the raw ``ModuleNotFoundError`` (:data:`_EXTRA_BY_SYMBOL`).
    """
    if name not in _LAZY_SYMBOLS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        value: Any = getattr(importlib.import_module(_LAZY_SYMBOLS[name]), name)
    except ImportError as exc:
        extra = _EXTRA_BY_SYMBOL[name]
        raise ImportError(
            f"langres.core.{name} requires the {extra!r} extra: "
            f"pip install 'langres[{extra}]' (or uv add 'langres[{extra}]')"
        ) from exc
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value
