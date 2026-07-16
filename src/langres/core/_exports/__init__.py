"""Per-domain export fragments composed by ``langres.core.__init__``.

**Why this package exists.** ``langres/core/__init__.py`` used to carry one
alphabetically sorted ~100-name ``__all__`` plus the three lazy maps. A sorted
list is the one shape git cannot auto-merge: N concurrent work-streams each
inserting a name at its sorted position produce N guaranteed conflicts, and the
file was touched 21 times in 30 days. Splitting the list into per-domain
fragments makes those streams edit **disjoint files**, so they merge cleanly.

**The contract.** Every fragment module in this package declares exactly these
five module-level names, and ``langres.core.__init__`` composes them::

    __all__: list[str]                 # EAGER names -- `import *` binds these
    LAZY_SUBMODULES: tuple[str, ...]   # resolved to a submodule of langres.core
    LAZY_SYMBOLS: dict[str, str]       # name -> owning module (resolved on access)
    EXTRA_BY_SYMBOL: dict[str, str]    # name -> pip extra, for the ImportError hint
    NAMES: tuple[str, ...]             # this fragment's slice of langres.core.__all__

``__all__`` holds only the *eager* names because that is what
``from ._exports._x import *`` actually binds -- a lazy name is deliberately not
defined at runtime (it lives under ``TYPE_CHECKING`` for mypy only), so listing
it in ``__all__`` would make the star-import raise ``AttributeError``. ``NAMES``
is therefore always *derived*, never hand-maintained::

    NAMES = (*__all__, *LAZY_SUBMODULES, *LAZY_SYMBOLS)

All five are declared by every fragment even when empty. That uniformity is the
whole point: it keeps ``core/__init__.py`` free of per-*name* content, so adding
an export to an existing domain touches one fragment and nothing else.

**Adding an export**: edit the one fragment that owns its domain -- add the
import + the ``__all__`` entry (eager), or the ``LAZY_SYMBOLS`` +
``EXTRA_BY_SYMBOL`` entry and a ``TYPE_CHECKING`` import (lazy). Only a brand
new *domain* requires touching ``core/__init__.py``.

**Keep lazy names lazy**: a symbol pulling an optional/heavy dependency
(torch/litellm/faiss/scikit-learn/mlflow/wandb) must go in ``LAZY_SYMBOLS``, never
be imported at a fragment's module scope -- fragments are eagerly imported by
``langres.core``, so an import here lands in every bare ``import langres``.
``tests/test_import_budget.py`` is the gate.
"""
