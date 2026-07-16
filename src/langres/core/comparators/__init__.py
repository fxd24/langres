"""Comparator implementations for pairwise feature comparison.

The ``Comparator`` **contract** lives in ``langres.core.comparator``; the
concrete comparators live here, mirroring the ``blockers`` / ``matchers`` /
``clusterers`` packages::

    from langres.core.comparator  import Comparator        # the contract
    from langres.core.comparators import StringComparator  # an implementation

``StringComparator`` needs only rapidfuzz (a core dependency, not an extra), so
-- unlike ``langres.core.blockers`` -- this package needs no lazy
``__getattr__``: importing it pulls in nothing optional.
"""

from langres.core.comparators.string import Algorithm, StringComparator

__all__ = ["Algorithm", "StringComparator"]
