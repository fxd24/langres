"""Blocker implementations for candidate pair generation.

``VectorBlocker`` needs faiss (the ``[semantic]`` extra, see
``langres.core``'s module docstring for the W0.4 rationale) -- resolved lazily
via PEP 562 so importing this package (a side effect of e.g.
``from langres.core.blockers.all_pairs import AllPairsBlocker``) never pulls
faiss in for a caller who only wants the light blocker.
"""

from typing import TYPE_CHECKING, Any

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.blockers.composite import CompositeBlocker
from langres.core.blockers.key import KeyBlocker

if TYPE_CHECKING:
    from langres.core.blockers.vector import VectorBlocker

__all__ = ["AllPairsBlocker", "CompositeBlocker", "KeyBlocker", "VectorBlocker"]


def __getattr__(name: str) -> Any:
    if name == "VectorBlocker":
        from langres.core.blockers.vector import VectorBlocker

        return VectorBlocker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
