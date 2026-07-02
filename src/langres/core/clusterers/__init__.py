"""Pluggable Clusterer variants.

``langres.core.clusterer`` holds the base ``Clusterer`` (transitive-closure /
connected-components); this package holds alternative clustering strategies
that plug into the same ``Blocker``/``blockers/`` split convention.
"""

from langres.core.clusterers.correlation import CorrelationClusterer

__all__ = ["CorrelationClusterer"]
