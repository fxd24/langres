"""Candidate generation: the ``Blocker`` family + the ``Comparator`` feeding it.

See ``langres.core._exports`` for the fragment contract.
"""

from typing import TYPE_CHECKING

from langres.core.blocker import Blocker
from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.blockers.composite import CompositeBlocker
from langres.core.blockers.key import KeyBlocker
from langres.core.comparator import Comparator, StringComparator

if TYPE_CHECKING:
    # Never executed at runtime -- keeps the lazy names visible to `mypy --strict`
    # without pulling faiss/torch into a bare `import langres`.
    from langres.core.blockers.vector import VectorBlocker

__all__ = [
    "AllPairsBlocker",
    "Blocker",
    "Comparator",
    "CompositeBlocker",
    "KeyBlocker",
    "StringComparator",
]

LAZY_SUBMODULES: tuple[str, ...] = ()

LAZY_SYMBOLS: dict[str, str] = {
    "VectorBlocker": "langres.core.blockers.vector",
}

EXTRA_BY_SYMBOL: dict[str, str] = {
    "VectorBlocker": "semantic",
}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SUBMODULES, *LAZY_SYMBOLS)
