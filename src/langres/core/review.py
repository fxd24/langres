# TEMPORARY: deleted by the W2 sweep
"""Back-compat shim: ``langres.core.review`` moved to ``langres.curation.review``.

The review surface (pick the uncertain margin) is part of the curation package
now. This re-export keeps ``from langres.core.review import ...`` working for one
wave; the W2 sweep deletes this file. Import from ``langres.curation.review``.
"""

from langres.curation.review import ReviewItem, ReviewQueue, select_for_review

__all__ = ["ReviewItem", "ReviewQueue", "select_for_review"]
