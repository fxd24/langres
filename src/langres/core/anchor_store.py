# TEMPORARY: deleted by the W2 sweep
"""Back-compat shim: ``langres.core.anchor_store`` moved to ``langres.curation.anchor_store``.

The anchor store (hold the anchors; assign incoming records against them) is part
of the curation package now. This re-export keeps
``from langres.core.anchor_store import ...`` working for one wave; the W2 sweep
deletes this file. Import from ``langres.curation.anchor_store``.
"""

from langres.curation.anchor_store import (  # pragma: no cover
    AnchorStore,
    AnchorStoreManifest,
    ClusterDelta,
)

__all__ = ["AnchorStore", "AnchorStoreManifest", "ClusterDelta"]  # pragma: no cover
