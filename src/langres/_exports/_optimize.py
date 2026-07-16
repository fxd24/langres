"""The autoresearch facade (``langres.optimize`` / ``score_blocking``).

Eager on purpose, and import-light by construction: ``langres/optimize.py``'s
module top is stdlib/typing only (every factory / data / metrics / faiss import
is lazy inside a function body), so pulling it into the eager graph must not
drag torch / faiss / sentence-transformers / litellm / scikit-learn into
``sys.modules``. ``tests/test_import_budget.py`` locks both halves.

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
