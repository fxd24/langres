"""Candidate generation: the ``Blocker`` and ``Comparator`` **contracts**.

Only the two ABCs live here. The concrete blockers (``AllPairsBlocker``,
``CompositeBlocker``, ``KeyBlocker``, ``VectorBlocker``) and ``StringComparator``
are **implementations**, and ``langres.core`` no longer re-exports them -- import
them from the package that owns them::

    from langres.core.blockers import AllPairsBlocker, VectorBlocker
    from langres.core.comparator import StringComparator

See ``langres.core._exports`` for the fragment contract, and
``langres.core.__init__`` for why the facade carries contracts only.
"""

from langres.core.blocker import Blocker
from langres.core.comparator import Comparator

__all__ = [
    "Blocker",
    "Comparator",
]

LAZY_SYMBOLS: dict[str, str] = {}
EXTRA_BY_SYMBOL: dict[str, str] = {}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SYMBOLS)
