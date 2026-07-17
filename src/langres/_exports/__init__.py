"""Per-domain export fragments composed by ``langres.__init__``.

Same motivation and contract as :mod:`langres.core._exports` (read that first),
applied to the root package -- where the **eager import block** was the sharper
edge: its lines split across four different work-streams (verbs/presets,
optimize, the flywheel, training), so any two of them touching the root
``__init__`` collided. Each fragment below owns one of those streams.

**The contract.** Every fragment declares exactly these four names::

    __all__: list[str]                 # EAGER names -- `import *` binds these
    LAZY_SYMBOLS: dict[str, str]       # name -> owning module (resolved on access)
    EXTRA_BY_SYMBOL: dict[str, str]    # name -> pip extra, for the ImportError hint
    NAMES: tuple[str, ...]             # this fragment's slice of langres.__all__

``__all__`` holds only the *eager* names because that is what
``from ._exports._x import *`` actually binds -- a lazy name is deliberately not
defined at runtime (it lives under ``TYPE_CHECKING`` for mypy only). ``NAMES``
is therefore always *derived*, never hand-maintained::

    NAMES = (*__all__, *LAZY_SYMBOLS)

Unlike ``core``'s, this contract has no ``LAZY_SUBMODULES`` (the root resolves
no submodules lazily), and ``EXTRA_BY_SYMBOL`` is a *subset* of
``LAZY_SYMBOLS``: a lazy symbol absent from it needs no extra, and an
ImportError from it is a genuine bug that propagates unchanged. (``core``'s
must be *total* -- its ``__getattr__`` indexes the map rather than ``.get()``
-ing it; ``tests/test_export_fragments.py`` locks both rules.)

**Root fragments vs. their ``core`` namesakes.** Several names appear in both
trees (``_training``, ``_flywheel``) -- this is re-export, not duplication, and
the two have distinct jobs:

* ``langres/core/_exports/*`` **owns** the export: it imports from the
  implementation module (``langres.core.harvest``, ``langres.training.fit_report``,
  ...) and decides eager-vs-lazy for ``langres.core``.
* ``langres/_exports/*`` **re-surfaces** a curated subset at the root, mostly
  importing from ``langres.core`` itself (already-eager names come for free).
  The root is a smaller, opinionated front door -- ``langres.core`` exports 103
  names, the root 36 -- so it is deliberately NOT a mirror.

A name therefore lives in a root fragment only if it is part of the paved-road
surface. Adding one to ``core`` does not imply adding it here.

**Adding an export**: edit the one fragment that owns its domain. **Adding a
domain** (rare): add the fragment here (import + the three merges below) and one
star-import line in ``langres/__init__.py``.
"""

from langres._exports import _core, _data, _flywheel, _models, _optimize, _training

#: The composed public surface: every fragment's slice, deduplicated and sorted
#: case-insensitively (the dominant convention of the hand-maintained list this
#: replaced). Order is cosmetic -- ``__all__`` is only ever read as a set.
NAMES: tuple[str, ...] = tuple(
    sorted(
        {
            *_core.NAMES,
            *_data.NAMES,
            *_flywheel.NAMES,
            *_optimize.NAMES,
            *_training.NAMES,
            *_models.NAMES,
        },
        key=str.lower,
    )
)

#: ``name -> owning module`` for root exports resolved on first access.
LAZY_SYMBOLS: dict[str, str] = {
    **_core.LAZY_SYMBOLS,
    **_data.LAZY_SYMBOLS,
    **_flywheel.LAZY_SYMBOLS,
    **_optimize.LAZY_SYMBOLS,
    **_training.LAZY_SYMBOLS,
    **_models.LAZY_SYMBOLS,
}

#: ``name -> extra`` for the lazy symbols where a missing dependency has a
#: ``pip install langres[<extra>]`` fix. Symbols absent here need no extra --
#: an ImportError from them is a genuine bug and propagates unchanged.
EXTRA_BY_SYMBOL: dict[str, str] = {
    **_core.EXTRA_BY_SYMBOL,
    **_data.EXTRA_BY_SYMBOL,
    **_flywheel.EXTRA_BY_SYMBOL,
    **_optimize.EXTRA_BY_SYMBOL,
    **_training.EXTRA_BY_SYMBOL,
    **_models.EXTRA_BY_SYMBOL,
}
