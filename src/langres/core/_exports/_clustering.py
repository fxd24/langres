"""Clustering, canonicalization, and the incremental anchor store.

See ``langres.core._exports`` for the fragment contract.
"""

from langres.core.adapters.glinker import GLinkerAdapter
from langres.core.anchor_store import AnchorStore, ClusterDelta
from langres.core.canonicalizer import Canonicalizer
from langres.core.clusterer import Clusterer
from langres.core.clusterers.correlation import CorrelationClusterer

__all__ = [
    "AnchorStore",
    "Canonicalizer",
    "ClusterDelta",
    "Clusterer",
    "CorrelationClusterer",
    "GLinkerAdapter",
]

LAZY_SUBMODULES: tuple[str, ...] = ()
LAZY_SYMBOLS: dict[str, str] = {}
EXTRA_BY_SYMBOL: dict[str, str] = {}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SUBMODULES, *LAZY_SYMBOLS)
