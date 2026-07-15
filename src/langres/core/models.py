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

from langres.core.feature import ComparisonVector
from langres.core.registry import register_schema


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
# Blocker and Matcher use TYPE_CHECKING to bind to EntityProtocol for type safety.
SchemaT = TypeVar("SchemaT", bound=BaseModel)


@register_schema("CompanySchema")
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

    This is the standardized input to all Matcher.forward() implementations.
    The Blocker is responsible for normalizing raw data into this schema
    and generating candidate pairs.

    Note:
        ``SchemaT`` is the Pydantic schema type for both entities
        (e.g., ``CompanySchema``).

    Attributes:
        left: The left entity in the pair
        right: The right entity in the pair
        blocker_name: Name of the blocker that generated this candidate pair
        similarity_score: Optional similarity score in [0, 1] for ranking evaluation
        comparison: Per-feature ``ComparisonVector`` for the two-phase pipeline.
            In phase one a Comparator stage attaches it (one entry per feature);
            in phase two the judge runs. Comparison-aware judges (e.g.
            ``WeightedAverageMatcher``) consume this vector and raise if it is
            ``None`` (the Comparator stage was skipped). Self-contained judges
            (e.g. ``LLMMatcher``) ignore it and read the raw ``left``/``right``
            entities directly, so it stays ``None`` for them.
    """

    left: SchemaT
    right: SchemaT
    blocker_name: str
    similarity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    comparison: ComparisonVector | None = None


class PairwiseJudgement(BaseModel):
    """
    Rich decision output from Matcher.forward() with full provenance.

    This model captures not just the match decision, but all metadata
    necessary for debugging, optimization, and cost tracking.

    A judge is one of two shapes (and may be both):

    - A **ranker** emits a ``score`` (a confidence-ordered number); the caller's
      ``threshold`` turns it into a match/no-match.
    - A **decider** emits a boolean ``decision`` directly (a binary LLM says
      "Yes"/"No"); it has no meaningful score, and the threshold is irrelevant.

    Ask ":func:`predicted_match`" — never a raw ``score >= threshold`` — whether a
    judgement is a predicted match. It gives ``decision`` precedence over
    ``score`` (a decider that also ranked already decided), and returns ``None``
    for an abstention (neither set) rather than fabricating a "no".

    Attributes:
        left_id: Identifier of the left entity
        right_id: Identifier of the right entity
        decision: The judge's explicit match verdict, if it decides directly
            (a binary judge). ``None`` when the judge only ranks (emits a
            ``score``) or abstains. Takes precedence over ``score`` in
            :func:`predicted_match`.
        score: Confidence-ordered match score in range [0.0, 1.0] for a ranking
            judge, else ``None``. **Widened from a required float:** a decider
            has no score, so a fabricated ``0.0``/``1.0`` would lie. ``None``
            means "this judge does not rank", not "score of zero".
        score_type: Type of score for proper interpretation. Stays **required**;
            it doubles as the judge-family tag, so when ``score`` is ``None`` it
            describes the *family* (e.g. ``"prob_llm"`` for a binary LLM judge),
            not a score. This overloading is a known wart — do not widen it.
        confidence: Optional "how sure am I" in range [0.0, 1.0], orthogonal to
            the decision. ``None`` unless a judge supplies it (nothing does yet —
            that is Wave 2).
        confidence_source: Provenance of ``confidence``. ``"none"`` means this
            judge structurally has no confidence to give; ``"unrequested"`` means
            it could but was not asked (nothing sets this yet — Wave 2). The
            literal set is **provisional**, not a frozen API — expect it to grow.
        decision_step: Which logic branch made this decision
        reasoning: Optional natural language explanation (e.g., from LLM)
        provenance: Full audit trail with arbitrary metadata
    """

    left_id: str
    right_id: str
    decision: bool | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    score_type: Literal[
        "sim_cos",
        "prob_llm",
        "heuristic",
        "calibrated_prob",
        "prob_fs",
        "prob_rf",
        "prob_group_llm",
    ]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_source: Literal["none", "unrequested", "logprob", "calibrated", "heuristic"] = "none"
    decision_step: str
    reasoning: str | None = None
    provenance: dict[str, Any]

    @property
    def is_abstain(self) -> bool:
        """``True`` iff the judge gave no actionable signal (no decision, no score).

        An abstention is NOT a "no" — it carries no verdict at all. Distinct from
        a decider's explicit ``decision=False`` and from a ranker's low ``score``.
        """
        return self.decision is None and self.score is None


class MatcherAbstainedError(RuntimeError):
    """A judge abstained where the caller needs a verdict.

    Raised by :func:`~langres.link` when the judge neither decided nor scored
    (``PairwiseJudgement.is_abstain``) — e.g. an ``LLMMatcher`` whose response
    failed to parse under the default ``on_parse_error="abstain"``.

    This is deliberately an exception rather than a ``match=None`` verdict: a
    caller writing the obvious ``if verdict.match:`` would read ``None`` as "not
    a match", silently turning "I don't know" back into a confident no — the very
    conflation the decision/score split exists to remove. An exception cannot be
    ignored by accident. It subclasses ``RuntimeError`` (like
    ``NoMatcherAvailableError``) so existing ``except RuntimeError`` handlers still
    catch it.
    """


def predicted_match(judgement: PairwiseJudgement, threshold: float) -> bool | None:
    """The ONE place that answers 'is this pair a match'.

    A decider decided -> the threshold is irrelevant to it.
    A ranker ranked   -> the threshold decides.
    Neither           -> abstained. No actionable signal. NOT a "no".

    ``decision`` takes precedence over ``score``: a judge that both decided and
    ranked already made its call, so the threshold never overrides it.
    """
    if judgement.decision is not None:
        return judgement.decision
    if judgement.score is not None:
        return judgement.score >= threshold
    return None
