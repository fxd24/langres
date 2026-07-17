# TEMPORARY: deleted by the W2 sweep
"""Back-compat shim: ``langres.core.harvest`` moved to ``langres.curation.harvest``.

The harvest surface (turn answered reviews back into labeled pairs, derive a
threshold from them) is part of the curation package now. This re-export keeps
``from langres.core.harvest import ...`` working for one wave; the W2 sweep
deletes this file. Import from ``langres.curation.harvest``.
"""

from langres.curation.harvest import (  # pragma: no cover
    AlignedPairs,
    AlignedSplit,
    Correction,
    CorrectionLog,
    GoldCoverage,
    LabeledPair,
    PairLabel,
    align_pairs,
    derive_threshold_from_pairs,
    harvest_labeled_pairs,
)

__all__ = [  # pragma: no cover
    "AlignedPairs",
    "AlignedSplit",
    "Correction",
    "CorrectionLog",
    "GoldCoverage",
    "LabeledPair",
    "PairLabel",
    "align_pairs",
    "derive_threshold_from_pairs",
    "harvest_labeled_pairs",
]
