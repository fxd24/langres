"""RapidfuzzModule implementation for classical string matching.

This module computes weighted string similarity scores using rapidfuzz. It is
schema-agnostic, accepting field extractors and weights to work with any
Pydantic schema type.
"""

from collections.abc import Callable, Iterator
from typing import Any, Literal

from rapidfuzz import fuzz

from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.module import Module, SchemaT
from langres.core.modules.llm_judge import _inspect_scores_impl
from langres.core.reports import ScoreInspectionReport

# Supported rapidfuzz algorithms
Algorithm = Literal["ratio", "token_sort_ratio", "token_set_ratio"]


class RapidfuzzModule(Module[SchemaT]):
    """Schema-agnostic string similarity module using rapidfuzz.

    This module computes weighted string similarity scores across multiple fields
    using rapidfuzz algorithms. It is schema-agnostic: it works with ANY Pydantic
    schema by accepting field extractors and weights.

    Example:
        # For companies
        module = RapidfuzzModule(
            field_extractors={
                "name": (lambda x: x.name, 0.7),
                "address": (lambda x: x.address or "", 0.3),
            },
            threshold=0.7,
            algorithm="ratio"
        )

        # For products (different schema, same module!)
        module = RapidfuzzModule(
            field_extractors={
                "title": (lambda x: x.title, 0.8),
                "brand": (lambda x: x.brand or "", 0.2),
            },
            threshold=0.6,
            algorithm="token_sort_ratio"
        )

    Note:
        The field_extractors parameter allows maximum flexibility:
        - Keys are field names (used in provenance)
        - Values are tuples of (extractor_function, weight)
        - Extractors can be simple lambdas or complex functions
        - Weights are auto-normalized to sum to 1.0

    Note:
        Available algorithms:
        - "ratio": Basic character-level similarity (Levenshtein ratio)
        - "token_sort_ratio": Sorts tokens before comparison (order-insensitive)
        - "token_set_ratio": Compares unique token sets (handles duplicates)
    """

    def __init__(
        self,
        field_extractors: dict[str, tuple[Callable[[Any], str], float]],
        threshold: float = 0.5,
        algorithm: Algorithm = "ratio",
    ):
        """Initialize RapidfuzzModule.

        Args:
            field_extractors: Dictionary mapping field names to (extractor, weight)
                tuples. The extractor is a function that extracts a string from
                an entity, and the weight is the importance of this field in the
                final score.
            threshold: Minimum score to consider a match (0.0 to 1.0).
                This is stored for compatibility with Optimizer, but not used
                in forward() (that's the Clusterer's job).
            algorithm: Rapidfuzz algorithm to use for string comparison.

        Raises:
            ValueError: If threshold is not in range [0.0, 1.0] or algorithm
                is not supported.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        if algorithm not in ("ratio", "token_sort_ratio", "token_set_ratio"):
            raise ValueError(
                f"algorithm must be one of: ratio, token_sort_ratio, token_set_ratio. Got: {algorithm}"
            )

        # Normalize weights to sum to 1.0 for interpretable scores
        total_weight = sum(weight for _, weight in field_extractors.values())
        if total_weight > 0:
            self.field_extractors = {
                name: (extractor, weight / total_weight)
                for name, (extractor, weight) in field_extractors.items()
            }
        else:
            # If all weights are zero, distribute evenly
            num_fields = len(field_extractors)
            self.field_extractors = {
                name: (extractor, 1.0 / num_fields if num_fields > 0 else 0.0)
                for name, (extractor, _) in field_extractors.items()
            }

        self.threshold = threshold
        self.algorithm = algorithm

    def forward(self, candidates: Iterator[ERCandidate[SchemaT]]) -> Iterator[PairwiseJudgement]:
        """Compare entity pairs using weighted string similarity.

        Args:
            candidates: Stream of normalized entity pairs from a Blocker.

        Yields:
            PairwiseJudgement objects with scores and full provenance.
            Each judgement contains:
            - left_id: Identifier of the left entity
            - right_id: Identifier of the right entity
            - score: Weighted average of field similarities (0.0 to 1.0)
            - score_type: "heuristic" (string similarity is a heuristic)
            - decision_step: "rapidfuzz_weighted"
            - provenance: Field scores and algorithm used

        Note:
            The score is computed as a weighted average:
            score = sum(field_weight * similarity(left_field, right_field))

            Each field similarity is computed using the specified rapidfuzz
            algorithm and normalized to [0.0, 1.0].
        """
        # Get the rapidfuzz function based on algorithm
        fuzz_func = self._get_fuzz_function()

        for candidate in candidates:
            # Compute similarity for each field
            field_scores: dict[str, float] = {}

            for field_name, (extractor, weight) in self.field_extractors.items():
                left_value = extractor(candidate.left)
                right_value = extractor(candidate.right)

                # Compute similarity (normalized to 0.0-1.0)
                similarity = fuzz_func(left_value, right_value) / 100.0
                field_scores[field_name] = similarity

            # Compute weighted average score
            total_score = 0.0
            for field_name, (_, weight) in self.field_extractors.items():
                total_score += weight * field_scores[field_name]

            yield PairwiseJudgement(
                left_id=candidate.left.id,  # type: ignore[attr-defined]
                right_id=candidate.right.id,  # type: ignore[attr-defined]
                score=total_score,
                score_type="heuristic",
                decision_step="rapidfuzz_weighted",
                provenance={
                    "field_scores": field_scores,
                    "algorithm": self.algorithm,
                },
            )

    def _get_fuzz_function(self) -> Callable[[str, str], float]:
        """Get the rapidfuzz function based on the configured algorithm.

        Returns:
            Rapidfuzz comparison function that takes two strings and returns
            a similarity score in the range [0.0, 100.0].
        """
        if self.algorithm == "ratio":
            return fuzz.ratio
        elif self.algorithm == "token_sort_ratio":
            return fuzz.token_sort_ratio
        elif self.algorithm == "token_set_ratio":
            return fuzz.token_set_ratio
        else:  # pragma: no cover
            # This should never happen due to validation in __init__
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        """Explore scores without ground truth labels.

        This implementation delegates to a shared utility function that works
        for all Module types since they all return PairwiseJudgement objects.

        Args:
            judgements: List of PairwiseJudgement objects to analyze
            sample_size: Number of examples to include (default: 10)

        Returns:
            ScoreInspectionReport with statistics, examples, and recommendations
        """
        return _inspect_scores_impl(judgements, sample_size)
