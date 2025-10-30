"""
Data contracts for langres entity resolution framework.

This module defines the core Pydantic models that serve as type-safe
interfaces between all components:

- CompanySchema: Test domain model for POC
- ERCandidate[SchemaT]: Generic normalized pair passed to Modules
- PairwiseJudgement: Rich decision output with full provenance
"""

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

# Generic type variable for ERCandidate
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class CompanySchema(BaseModel):
    """
    Domain model for company entities (POC test data).

    This schema represents a company with required identifier and name,
    plus optional contact information fields.
    """

    id: str
    name: str
    address: str | None = None
    phone: str | None = None
    website: str | None = None


class ERCandidate(BaseModel, Generic[SchemaT]):
    """
    Generic container for normalized entity pairs.

    This is the standardized input to all Module.forward() implementations.
    The Blocker is responsible for normalizing raw data into this schema
    and generating candidate pairs.

    Type Parameters:
        SchemaT: The Pydantic schema type for both entities (e.g., CompanySchema)

    Attributes:
        left: The left entity in the pair
        right: The right entity in the pair
        blocker_name: Name of the blocker that generated this candidate pair
    """

    left: SchemaT
    right: SchemaT
    blocker_name: str


class PairwiseJudgement(BaseModel):
    """
    Rich decision output from Module.forward() with full provenance.

    This model captures not just the match decision, but all metadata
    necessary for debugging, optimization, and cost tracking.

    Attributes:
        left_id: Identifier of the left entity
        right_id: Identifier of the right entity
        score: Match confidence score in range [0.0, 1.0]
        score_type: Type of score for proper interpretation
        decision_step: Which logic branch made this decision
        reasoning: Optional natural language explanation (e.g., from LLM)
        provenance: Full audit trail with arbitrary metadata
    """

    left_id: str
    right_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    score_type: Literal["sim_cos", "prob_llm", "heuristic", "calibrated_prob"]
    decision_step: str
    reasoning: str | None = None
    provenance: dict[str, Any]
