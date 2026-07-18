"""Per-domain export fragments composed by ``langres.core.__init__``.

**Why this package exists.** ``langres/core/__init__.py`` used to carry one
alphabetically sorted ~100-name ``__all__`` plus the three lazy maps. A sorted
list is the one shape git cannot auto-merge: N concurrent work-streams each
inserting a name at its sorted position produce N guaranteed conflicts, and the
file was touched 21 times in 30 days. Splitting the list into per-domain
fragments makes those streams edit **disjoint files**, so they merge cleanly.

**Contracts only.** ``langres.core`` re-exports the types a pipeline is
*written against*, not the implementations. A concrete blocker/matcher/
clusterer/embedder/index belongs to its own package
(``langres.core.blockers``, ``.matchers``, ...) and is imported from there --
see ``langres.core.__init__`` for why, and ``tests/test_import_tangle.py`` for
the ratchet that measures it. Adding an implementation to a fragment here
re-creates the indirection this layout exists to remove.

**The contract.** Every fragment module in this package declares exactly these
four module-level names, and ``langres.core.__init__`` composes them::

    __all__: list[str]                 # EAGER names -- `import *` binds these
    LAZY_SYMBOLS: dict[str, str]       # name -> owning module (resolved on access)
    EXTRA_BY_SYMBOL: dict[str, str]    # name -> pip extra, for the ImportError hint
    NAMES: tuple[str, ...]             # this fragment's slice of langres.core.__all__

``__all__`` holds only the *eager* names because that is what
``from ._exports._x import *`` actually binds -- a lazy name is deliberately not
defined at runtime (it lives under ``TYPE_CHECKING`` for mypy only), so listing
it in ``__all__`` would make the star-import raise ``AttributeError``. ``NAMES``
is therefore always *derived*, never hand-maintained::

    NAMES = (*__all__, *LAZY_SYMBOLS)

All four are declared by every fragment even when empty. That uniformity is the
whole point: it keeps ``core/__init__.py`` free of per-*name* content, so adding
an export to an existing domain touches one fragment and nothing else. (The root
``langres._exports`` fragments declare the same four names -- the two contracts
are identical.)

**Adding an export**: edit the one fragment that owns its domain -- add the
import + the ``__all__`` entry (eager), or the ``LAZY_SYMBOLS`` +
``EXTRA_BY_SYMBOL`` entry and a ``TYPE_CHECKING`` import (lazy). Nothing else
changes: this module composes whatever the fragments declare.

**Adding a domain** (rare): add the fragment here (import + the three merges
below) and one star-import line in ``langres/core/__init__.py``.

**Keep lazy names lazy**: a symbol pulling an optional/heavy dependency
(torch/litellm/faiss/scikit-learn/mlflow/wandb) must go in ``LAZY_SYMBOLS``, never
be imported at a fragment's module scope -- fragments are eagerly imported by
``langres.core``, so an import here lands in every bare ``import langres``.
``tests/test_import_budget.py`` is the gate.
"""

from langres.core._exports import (
    _blocking,
    _clustering,
    _flywheel,
    _matchers,
    _methods,
    _models,
    _ops,
    _resolver,
    _tracking,
    _training,
)

#: The composed public surface: every fragment's slice, deduplicated and sorted
#: case-insensitively (the dominant convention of the hand-maintained list this
#: replaced). Order is cosmetic -- ``__all__`` is only ever read as a set.
NAMES: tuple[str, ...] = tuple(
    sorted(
        {
            *_blocking.NAMES,
            *_clustering.NAMES,
            *_flywheel.NAMES,
            *_matchers.NAMES,
            *_methods.NAMES,
            *_models.NAMES,
            *_ops.NAMES,
            *_resolver.NAMES,
            *_tracking.NAMES,
            *_training.NAMES,
        },
        key=str.lower,
    )
)

#: ``name -> owning module`` for symbols resolved on first access.
LAZY_SYMBOLS: dict[str, str] = {
    **_blocking.LAZY_SYMBOLS,
    **_clustering.LAZY_SYMBOLS,
    **_flywheel.LAZY_SYMBOLS,
    **_matchers.LAZY_SYMBOLS,
    **_methods.LAZY_SYMBOLS,
    **_models.LAZY_SYMBOLS,
    **_ops.LAZY_SYMBOLS,
    **_resolver.LAZY_SYMBOLS,
    **_tracking.LAZY_SYMBOLS,
    **_training.LAZY_SYMBOLS,
}

#: ``name -> extra`` for the lazy symbols a ``pip install langres[<extra>]``
#: actually fixes.
EXTRA_BY_SYMBOL: dict[str, str] = {
    **_blocking.EXTRA_BY_SYMBOL,
    **_clustering.EXTRA_BY_SYMBOL,
    **_flywheel.EXTRA_BY_SYMBOL,
    **_matchers.EXTRA_BY_SYMBOL,
    **_methods.EXTRA_BY_SYMBOL,
    **_models.EXTRA_BY_SYMBOL,
    **_ops.EXTRA_BY_SYMBOL,
    **_resolver.EXTRA_BY_SYMBOL,
    **_tracking.EXTRA_BY_SYMBOL,
    **_training.EXTRA_BY_SYMBOL,
}
