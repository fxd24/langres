"""Feature-comparison contracts for the M0 Resolver.

These are the frozen data contracts that sit between the Comparator (which
turns a pair of entities into a per-feature comparison) and the scorer Matcher
(which combines that comparison into a single match score).

Three pieces live here:

- :class:`ComparisonLevel` — the per-feature outcome of a comparison.
- :class:`FeatureSpec` — a serializable declaration of one comparable feature.
- :class:`ComparisonVector` — the per-pair comparison result the scorer consumes.

plus the :func:`combine_present` evidence-floor helper — the over-merge guard
that the Wave 2a scorer reuses to turn a ComparisonVector into a score.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# Minimum fraction of total weight that present features must cover for a pair
# to be eligible to score above zero (when there is only a single present
# feature). Two or more present features always satisfy the evidence floor.
_EVIDENCE_FLOOR_WEIGHT = 0.5
_EVIDENCE_FLOOR_MIN_PRESENT = 2


class ComparisonLevel(str, Enum):
    """Per-feature outcome of comparing two entities on one feature.

    Values:
        PRESENT: Both sides had a comparable value; a similarity was computed.
        MISSING: At least one side lacked a value (None/empty), so the feature
            was dropped and contributes no evidence.
        MISMATCH: **RESERVED for forward-compatibility.** Defined here so the
            enum is stable across versions, but **no M0 logic emits MISMATCH**.
            It is intended for a future "hard-negative" signal (e.g. conflicting
            anchor values). Do not branch on it in M0 code.
    """

    PRESENT = "PRESENT"
    MISSING = "MISSING"
    MISMATCH = "MISMATCH"


class FeatureSpec(BaseModel):
    """Serializable declaration of one comparable feature.

    Pure data (no callables) so a Resolver config can be persisted to JSON and
    reloaded in a fresh process. The Comparator (Wave 2a) reads a list of these
    to know which fields to compare and how to weight them.

    Attributes:
        name: Feature name; matches the entity field it is derived from.
        kind: Comparison kind. Only ``"string"`` is supported in M0; the field
            is reserved so future kinds (numeric, geo, date) can be added
            without breaking the artifact schema.
        weight: Relative importance of this feature, ``>= 0``. Weights are
            renormalized over present features at scoring time.
        is_anchor: **DORMANT — reserved.** No M0 logic reads this. Intended for
            a future rule where a strong anchor mismatch can veto a match.
    """

    name: str
    kind: Literal["string"] = "string"
    weight: float = Field(default=1.0, ge=0.0)
    is_anchor: bool = False


class ComparisonVector(BaseModel):
    """Per-pair comparison result — the contract between Comparator and scorer.

    The Comparator emits one ComparisonVector per candidate pair. The scorer
    Matcher consumes it (via :func:`combine_present`) to produce a single score.
    It is fully serializable and observable so a pair's evidence can be logged
    and inspected.

    Attributes:
        levels: Per-feature :class:`ComparisonLevel`. Every declared feature
            appears here, so a reader can see which features were MISSING vs
            PRESENT for this pair.
        similarities: Raw similarity in ``[0, 1]`` for each PRESENT feature.
            Missing features are absent from this map (never stored as 0.0,
            which would be indistinguishable from a real zero similarity).
    """

    levels: dict[str, ComparisonLevel]
    similarities: dict[str, float]

    def present_features(self) -> set[str]:
        """Names of features that were PRESENT for this pair."""
        return {name for name, level in self.levels.items() if level == ComparisonLevel.PRESENT}


def combine_present(similarities: dict[str, float], weights: dict[str, float]) -> float:
    """Combine present-feature similarities into one score, with an evidence floor.

    This is the over-merge guard. A pair built on too little evidence (a single
    low-weight feature) must not be allowed to score highly just because that
    one feature happened to match. The rule:

    1. **Drop missing features.** Only the features in ``similarities`` (the
       PRESENT ones) contribute. Anything else is ignored.
    2. **Renormalize weights over present features.** The combined score is the
       present-weighted average of the present similarities.
    3. **Apply the evidence floor.** A pair is eligible to score ``> 0`` only if
       it has **>= 2 present features** OR its **total present-weight >= 0.5**
       (measured against the full weight map). Otherwise the score is ``0.0``.
    4. **Never divide by zero.** All-missing (empty ``similarities``) and
       zero-present-weight both return ``0.0`` instead of raising.

    Args:
        similarities: Raw similarity in ``[0, 1]`` per PRESENT feature.
        weights: Declared weight per feature (the full map, including features
            that may be missing for this pair). Features present in
            ``similarities`` but absent here are treated as weight ``0.0``.

    Returns:
        The combined score in ``[0, 1]``, or ``0.0`` when the evidence floor is
        not met or there is no present evidence.
    """
    # (1) Drop missing: only present features (keys of `similarities`) count.
    if not similarities:
        # All-features-missing — never 0/0.
        return 0.0

    present_weights = {name: weights.get(name, 0.0) for name in similarities}
    total_present_weight = sum(present_weights.values())
    num_present = len(similarities)

    # (2) Evidence floor: a single low-weight present feature is not enough.
    meets_floor = (
        num_present >= _EVIDENCE_FLOOR_MIN_PRESENT or total_present_weight >= _EVIDENCE_FLOOR_WEIGHT
    )
    if not meets_floor:
        return 0.0

    # (3) Guard against zero present-weight (e.g. all present features weight 0).
    if total_present_weight <= 0.0:
        return 0.0

    # (4) Renormalize weights over present features and take the weighted average.
    score = sum(
        similarities[name] * (present_weights[name] / total_present_weight) for name in similarities
    )
    return score
