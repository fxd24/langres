"""The ``Clusterer`` **contract**: the base type the clusterer family extends.

Only the base type lives here -- it is what ``Resolver(clusterer=...)`` is
annotated against and what ``clusterers/`` subclasses. (It is a concrete
transitive-closure implementation rather than a declared ABC, but it plays the
contract role: it is the family's base type, and ``core.clusterer`` imports
nothing that reaches back into the facade.)

The alternatives and the surrounding machinery are **implementations**, and
``langres.core`` no longer re-exports them -- import them from the package that
owns them::

    from langres.core.clusterers import CorrelationClusterer
    from langres.core.anchor_store import AnchorStore, ClusterDelta
    from langres.core.canonicalizer import Canonicalizer
    from langres.core.adapters.glinker import GLinkerAdapter

See ``langres.core._exports`` for the fragment contract, and
``langres.core.__init__`` for why the facade carries contracts only.
"""

from langres.core.clusterer import Clusterer

__all__ = [
    "Clusterer",
]

LAZY_SYMBOLS: dict[str, str] = {}
EXTRA_BY_SYMBOL: dict[str, str] = {}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SYMBOLS)
