"""The autoresearch facade (``langres.optimize`` / ``score_blocking``).

Eager on purpose, and import-light by construction: ``langres/optimize.py``'s
module top is stdlib/typing only (every factory / loop / data / metrics / faiss
import is lazy inside a function body), so pulling it into the eager graph must
not drag torch / faiss / sentence-transformers / litellm / scikit-learn / optuna
into ``sys.modules``. ``tests/test_import_budget.py`` locks both halves.

Note this imports the *module* ``langres/optimize.py`` — the facade only. The
engine it drives lives in the ``langres.autoresearch`` package, whose heavy
members (``factory``, ``blocker_optimizer``) this never executes.

**The binding below is why the facade must stay a module.** It rebinds the
attribute ``langres.optimize`` from the module to the *function*, so any
``langres.optimize.<submodule>`` is unreachable by attribute traversal
(``import langres.optimize.loop as l`` → ``ImportError``). The engine is under
its own un-shadowed package name for exactly that reason.

See ``langres._exports`` for the fragment contract.
"""

from langres.optimize import optimize, score_blocking

__all__ = [
    "optimize",
    "score_blocking",
]

LAZY_SYMBOLS: dict[str, str] = {}
EXTRA_BY_SYMBOL: dict[str, str] = {}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SYMBOLS)
