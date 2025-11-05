"""
Module base class for entity comparison logic.

This module provides the abstract base class for all pairwise comparison
implementations in the langres framework.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Generic, TypeVar

from pydantic import BaseModel

from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.reports import ScoreInspectionReport

# Generic type variable for schema types (must be a Pydantic model)
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class Module(ABC, Generic[SchemaT]):
    """Abstract base class for entity comparison logic.

    The Module (also called "Flow") is the "brain" of the pipeline.
    It receives normalized entity pairs and yields match judgements.

    Design principles:
    - Operates on clean ERCandidate[SchemaT] (schema normalization is Blocker's job)
    - Yields rich PairwiseJudgement with provenance for observability
    - Reusable across tasks (dedup, linking, etc.)
    - Composable (can contain sub-modules, embeddings, models, etc.)

    The Module is the central Estimator in the langres architecture. It is
    responsible for comparing pairs of entities and producing match decisions
    with full provenance.

    Key architectural points:
    - **Separation of Concerns**: Module only compares; it doesn't load or
      normalize data. The Blocker handles candidate generation and schema
      normalization.
    - **Streaming First**: forward() is a generator to support lazy evaluation
      and memory-efficient processing of large datasets.
    - **Full Observability**: Every PairwiseJudgement includes provenance for
      debugging, optimization, and cost tracking.
    - **Composability**: Modules can contain other modules, classical similarity
      functions, embedding models, or LLM-based components.

    Example:
        class RapidfuzzModule(Module[CompanySchema]):
            '''Simple string-matching module using rapidfuzz.'''

            def forward(self, candidates):
                for pair in candidates:
                    score = fuzz.ratio(pair.left.name, pair.right.name) / 100.0
                    yield PairwiseJudgement(
                        left_id=pair.left.id,
                        right_id=pair.right.id,
                        score=score,
                        score_type="heuristic",
                        decision_step="rapidfuzz_name",
                        provenance={"method": "fuzz.ratio", "field": "name"}
                    )

    Example:
        class CascadeModule(Module[CompanySchema]):
            '''Multi-stage module with early exit optimization.'''

            def __init__(self):
                self.embed_sim = EmbeddingSimilarity()
                self.llm_judge = LLMJudge()

            def forward(self, candidates):
                for pair in candidates:
                    # Stage 1: Cheap embedding check
                    embed_score = self.embed_sim(pair.left.name, pair.right.name)

                    if embed_score < 0.3:
                        # Early exit: definitely not a match
                        yield PairwiseJudgement(
                            left_id=pair.left.id,
                            right_id=pair.right.id,
                            score=embed_score,
                            score_type="sim_cos",
                            decision_step="early_exit_low_similarity",
                            provenance={"embed_score": embed_score}
                        )
                    elif embed_score > 0.9:
                        # Early exit: definitely a match
                        yield PairwiseJudgement(
                            left_id=pair.left.id,
                            right_id=pair.right.id,
                            score=embed_score,
                            score_type="sim_cos",
                            decision_step="early_exit_high_similarity",
                            provenance={"embed_score": embed_score}
                        )
                    else:
                        # Stage 2: Expensive LLM judgment for uncertain cases
                        llm_result = self.llm_judge(pair)
                        yield PairwiseJudgement(
                            left_id=pair.left.id,
                            right_id=pair.right.id,
                            score=llm_result.score,
                            score_type="prob_llm",
                            decision_step="llm_judgment",
                            reasoning=llm_result.reasoning,
                            provenance={
                                "embed_score": embed_score,
                                "llm_model": "gpt-4",
                                "cost_usd": 0.002
                            }
                        )
    """

    @abstractmethod
    def forward(self, candidates: Iterator[ERCandidate[SchemaT]]) -> Iterator[PairwiseJudgement]:
        """Compare entity pairs and yield match judgements.

        This is the core method that all Module implementations must define.
        It processes a stream of normalized entity pairs and yields match
        decisions with full provenance.

        Args:
            candidates: Stream of normalized entity pairs from a Blocker.
                Each ERCandidate contains:
                - left: The left entity (SchemaT)
                - right: The right entity (SchemaT)
                - blocker_name: Name of the blocker that generated this pair

        Yields:
            PairwiseJudgement objects with scores and full provenance.
            Each judgement contains:
            - left_id: Identifier of the left entity
            - right_id: Identifier of the right entity
            - score: Match confidence in range [0.0, 1.0]
            - score_type: Type of score (e.g., "heuristic", "prob_llm", "sim_cos")
            - decision_step: Which logic branch made this decision
            - reasoning: Optional natural language explanation
            - provenance: Full audit trail with arbitrary metadata

        Note:
            Implementations should be generators (use yield) to support
            streaming/lazy evaluation for large datasets. This allows the
            pipeline to process millions of pairs without loading everything
            into memory.

        Note:
            Module implementations should NOT modify the input candidates.
            They are read-only consumers. All data normalization should
            happen in the Blocker before candidates reach the Module.

        Note:
            The SchemaT type variable ensures type safety when working with
            specific domain models (e.g., CompanySchema, ProductSchema).
            Subclasses can specialize this type for their specific use case.
        """
        pass  # pragma: no cover

    @abstractmethod
    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        """Explore scores without ground truth labels.

        Use this method to understand scoring output before labeling:
        - Score distribution statistics
        - High and low scoring examples with reasoning
        - Threshold recommendations based on distribution

        For quality evaluation with ground truth labels, use
        PipelineDebugger.analyze_scores() instead.

        Args:
            judgements: List of PairwiseJudgement objects to analyze
            sample_size: Number of examples to include (default: 10)

        Returns:
            ScoreInspectionReport with statistics, examples, and recommendations
        """
        pass  # pragma: no cover
