"""
Data contracts for langres entity resolution framework.

This module defines the core Pydantic models that serve as type-safe
interfaces between all components:

- EntityProtocol: Protocol defining required `id` attribute for all entities
- CompanySchema: Test domain model for POC
- ERCandidate[SchemaT]: Generic normalized pair passed to Modules
- PairwiseJudgement: Rich decision output with full provenance
"""

from typing import Any, Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, Field


class EntityProtocol(Protocol):
    """Protocol defining minimum requirements for entity schemas.

    All entity schemas used in langres must have an `id` attribute
    for identification and tracking. This enables type-safe generic
    programming while allowing any Pydantic schema that has an `id` field.

    Note:
        This Protocol is used in blocker.py and module.py with TYPE_CHECKING
        to provide type safety while maintaining Pydantic compatibility.
    """

    id: str


# Generic type variable for ERCandidate
# Note: Bound to BaseModel (not EntityProtocol) for Pydantic compatibility.
# Blocker and Module use TYPE_CHECKING to bind to EntityProtocol for type safety.
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
        similarity_score: Optional similarity score in [0, 1] for ranking evaluation
    """

    left: SchemaT
    right: SchemaT
    blocker_name: str
    similarity_score: float | None = Field(default=None, ge=0.0, le=1.0)


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
