# TEMPORARY: deleted by the W2 sweep
"""Back-compat shim: ``langres.core.canonicalizer`` moved to ``langres.curation.canonicalizer``.

The canonicalizer (survivorship: fold a cluster's records into one golden record)
is part of the curation package now. This re-export keeps
``from langres.core.canonicalizer import ...`` working for one wave; the W2 sweep
deletes this file. Import from ``langres.curation.canonicalizer``.
"""

from langres.curation.canonicalizer import (  # pragma: no cover
    Canonicalizer,
    CanonicalizerManifest,
    FieldContext,
)

__all__ = ["Canonicalizer", "CanonicalizerManifest", "FieldContext"]  # pragma: no cover
