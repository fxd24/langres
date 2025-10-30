"""
RapidfuzzModule: Classical string matching using rapidfuzz (Approach 1).

This module implements a tunable string similarity matcher that supports:
- Multiple rapidfuzz algorithms (ratio, partial_ratio, token_set_ratio)
- Configurable threshold for match decisions
- Multi-field weighted scoring (name, address, phone, website)
- Full provenance tracking for optimization

Key design principle: All parameters are exposed for future optimization with Optuna.
"""

from collections.abc import Iterator
from typing import Literal

from rapidfuzz import fuzz

from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.module import Module

# Type alias for supported algorithms
AlgorithmType = Literal["ratio", "partial_ratio", "token_set_ratio"]


class RapidfuzzModule(Module[CompanySchema]):
    """String similarity matcher using rapidfuzz (Approach 1: Classical).

    This module compares company entities using rapidfuzz string similarity
    with configurable algorithms and field weights. All parameters are exposed
    for optimization.

    Parameters (tunable for optimization):
    - threshold: Minimum similarity for match consideration (default: 0.5)
    - algorithm: Which rapidfuzz function to use (default: "ratio")
    - field_weights: Relative importance of each field (default: name-heavy)

    Example:
        # Default configuration
        module = RapidfuzzModule()

        # Optimized configuration (from Optuna)
        module = RapidfuzzModule(
            threshold=0.75,
            algorithm="token_set_ratio",
            field_weights={"name": 0.7, "address": 0.2, "phone": 0.1}
        )

        for judgement in module.forward(candidates):
            if judgement.score >= module.threshold:
                print(f"Match: {judgement.left_id} <-> {judgement.right_id}")

    Note:
        Provenance includes: threshold, algorithm, field_weights, and per-field scores.
        This allows the optimizer to understand which parameters led to each decision.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        algorithm: AlgorithmType = "ratio",
        field_weights: dict[str, float] | None = None,
    ):
        """Initialize the rapidfuzz module with tunable parameters.

        Args:
            threshold: Minimum score to consider a match (0.0 to 1.0).
                Scores >= threshold are typically classified as matches.
            algorithm: Which rapidfuzz algorithm to use:
                - "ratio": Simple Levenshtein similarity
                - "partial_ratio": Best matching substring
                - "token_set_ratio": Order-independent token matching
            field_weights: Relative importance of each field.
                If None, uses default name-heavy weighting.
                Should sum to approximately 1.0 for interpretability.

        Raises:
            ValueError: If threshold is not in [0.0, 1.0].
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        self.threshold = threshold
        self.algorithm = algorithm

        # Default field weights (name is most important for company matching)
        if field_weights is None:
            self.field_weights = {
                "name": 0.6,
                "address": 0.2,
                "phone": 0.1,
                "website": 0.1,
            }
        else:
            self.field_weights = field_weights

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        """Compare entity pairs and yield match judgements.

        For each candidate pair:
        1. Compute per-field similarity using the configured algorithm
        2. Compute weighted average score across all fields
        3. Yield PairwiseJudgement with full provenance

        Args:
            candidates: Iterator of ERCandidate[CompanySchema] from blocker

        Yields:
            PairwiseJudgement with:
            - score: Weighted average of field similarities (0.0 to 1.0)
            - score_type: "heuristic" (classical string matching)
            - decision_step: Algorithm used (e.g., "rapidfuzz_ratio")
            - provenance: Full parameter set and per-field scores

        Note:
            Fields with None values (missing data) are skipped in scoring.
            The weighted average is computed only over non-None fields.
        """
        for candidate in candidates:
            # Compute per-field similarities
            field_scores = self._compute_field_scores(candidate.left, candidate.right)

            # Compute weighted average (skip None fields)
            weighted_score = self._compute_weighted_score(field_scores)

            # Build provenance for observability and optimization
            provenance = {
                "threshold": self.threshold,
                "algorithm": self.algorithm,
                "field_weights": self.field_weights,
                "field_scores": field_scores,
            }

            yield PairwiseJudgement(
                left_id=candidate.left.id,
                right_id=candidate.right.id,
                score=weighted_score,
                score_type="heuristic",
                decision_step=f"rapidfuzz_{self.algorithm}",
                provenance=provenance,
            )

    def _compute_field_scores(
        self, left: CompanySchema, right: CompanySchema
    ) -> dict[str, float | None]:
        """Compute similarity score for each field.

        Args:
            left: Left company entity
            right: Right company entity

        Returns:
            Dict mapping field names to similarity scores (0.0 to 1.0).
            Returns None for fields where either value is None.
        """
        field_scores: dict[str, float | None] = {}

        for field in ["name", "address", "phone", "website"]:
            left_value = getattr(left, field)
            right_value = getattr(right, field)

            # Skip comparison if either field is None
            if left_value is None or right_value is None:
                field_scores[field] = None
            else:
                # Compute similarity using configured algorithm
                similarity = self._compute_similarity(left_value, right_value)
                field_scores[field] = similarity

        return field_scores

    def _compute_similarity(self, left_str: str, right_str: str) -> float:
        """Compute string similarity using configured algorithm.

        Args:
            left_str: Left string
            right_str: Right string

        Returns:
            Similarity score in range [0.0, 1.0]
        """
        if self.algorithm == "ratio":
            score = fuzz.ratio(left_str, right_str)
        elif self.algorithm == "partial_ratio":
            score = fuzz.partial_ratio(left_str, right_str)
        elif self.algorithm == "token_set_ratio":
            score = fuzz.token_set_ratio(left_str, right_str)
        else:
            # Should not happen due to type hints, but be defensive
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # rapidfuzz returns scores in [0, 100], normalize to [0, 1]
        return score / 100.0

    def _compute_weighted_score(self, field_scores: dict[str, float | None]) -> float:
        """Compute weighted average score across fields.

        Args:
            field_scores: Dict of field names to scores (or None)

        Returns:
            Weighted average score in range [0.0, 1.0]

        Note:
            Only non-None fields are included in the weighted average.
            Weights are renormalized to sum to 1.0 over available fields.
        """
        # Filter out None scores and their weights
        available_fields = [
            field for field, score in field_scores.items() if score is not None
        ]

        if not available_fields:
            # No fields available for comparison (all None)
            # Return 0.0 as we can't determine similarity
            return 0.0

        # Compute weighted sum and total weight (for renormalization)
        weighted_sum = 0.0
        total_weight = 0.0

        for field in available_fields:
            score = field_scores[field]
            weight = self.field_weights.get(field, 0.0)
            # Note: score is guaranteed non-None here (filtered in available_fields)
            assert score is not None
            weighted_sum += score * weight
            total_weight += weight

        # Renormalize weights to sum to 1.0 over available fields
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            # No weight assigned to any available field
            return 0.0
