"""
Module base class for entity comparison logic.

This module provides the abstract base class for all pairwise comparison
implementations in the langres framework.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from langres.core.groups import ERCandidateGroup, derive_groups_from_pairs
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


class GroupwiseModule(Module[SchemaT], ABC):
    """A Module whose scoring logic naturally operates on GROUPS, not pairs (W1.0, E2).

    GroupwiseModule **IS-A** Module: its concrete :meth:`forward` derives
    ``ERCandidateGroup`` groups internally from the pairwise ``ERCandidate``
    stream it receives (via :func:`~langres.core.groups.derive_groups_from_pairs`)
    and dispatches to the abstract :meth:`forward_groups`, decomposing its
    output back to a flat ``Iterator[PairwiseJudgement]``. This is a
    deliberate design choice, not an accident: it means the existing Resolver
    execution spine (``Resolver._judgements`` -> ``module.forward``),
    :meth:`Module.inspect_scores`, the JudgementLog boundary, and benchmark
    dispatch (``BudgetedModuleRunner``, ``run_method``) all keep working with
    **zero changes** -- there is no parallel, group-aware execution path.

    Concrete set-wise judges (e.g. a future ComEM-style ``SelectJudge`` that
    asks one LLM call "which of these K candidates match the anchor?" instead
    of K separate pairwise calls) implement only :meth:`forward_groups` and
    :meth:`Module.inspect_scores`. Set-wise IN, pairwise OUT: the group
    structure never leaks past this class's boundary, so the clusterer, the
    metrics harness, and every other pairwise-only downstream consumer needs
    no changes either.

    Note:
        The default grouping this class applies to its ``forward()`` input is
        the same buffered/skew-prone derivation documented on
        ``Blocker.stream_groups()`` / ``derive_groups_from_pairs`` -- because
        ``forward()`` only ever sees a flat pairwise stream (whatever the
        Resolver's blocker produced via ``stream()``), never the blocker
        object itself, so it cannot reach a blocker's *native* per-anchor
        grouping (e.g. ``VectorBlocker.stream_groups()``). Benchmarking a
        set-wise judge against a blocker's true group structure requires
        driving it via that blocker's ``stream_groups()`` directly, not
        through this class's derived default.
    """

    def forward(self, candidates: Iterator[ERCandidate[SchemaT]]) -> Iterator[PairwiseJudgement]:
        """Group the pairwise stream internally, then dispatch to forward_groups().

        Args:
            candidates: Stream of normalized entity pairs, e.g. from a
                Blocker's ``stream()``.

        Yields:
            PairwiseJudgement objects, exactly as any other Module -- the
            set-wise grouping is an internal implementation detail invisible
            to callers of ``forward()``.
        """
        groups = derive_groups_from_pairs(candidates)
        yield from self.forward_groups(groups)

    @abstractmethod
    def forward_groups(
        self, groups: Iterator[ERCandidateGroup[SchemaT]]
    ) -> Iterator[PairwiseJudgement]:
        """Compare each group's anchor against its members and yield judgements.

        This is the set-wise counterpart to :meth:`Module.forward`. Concrete
        implementations should yield one ``PairwiseJudgement`` per
        (anchor, member) pair evaluated -- typically all members of a group,
        so the output covers the same pairs the input groups covered.

        Args:
            groups: Stream of anchor + K-candidate-member groups.

        Yields:
            PairwiseJudgement objects with scores and full provenance, same
            contract as :meth:`Module.forward`. When a single call produces
            judgements for a whole group, use
            :func:`stamp_group_cost` to apply the group-call cost convention.
        """
        pass  # pragma: no cover


def stamp_group_cost(
    judgements: list[PairwiseJudgement],
    call_cost_usd: float,
    group_id: str,
) -> list[PairwiseJudgement]:
    """Apply the group-call cost convention to one group's judgements (E5).

    One LLM call spans a whole group (K pairs); naively pricing each of the K
    resulting judgements at the call's full cost would silently overcount
    total spend by a factor of K. This convention avoids that: the FULL
    ``call_cost_usd`` is stamped onto the FIRST judgement's
    ``provenance["cost_usd"]``, every sibling gets ``$0``, and
    ``provenance["group_id"]`` is set on ALL of them (so results stay
    traceable back to their source call regardless of cost placement).
    Existing cost aggregation (``langres.core.benchmark._judgement_cost`` /
    ``_cost_track``, which already read ``provenance["cost_usd"]``) then sums
    a group to exactly one call's cost with no changes on their end.

    ``provenance["group_end"]`` is set ``True`` on (only) the LAST judgement
    of the group -- a boundary marker so a consumer draining a whole group
    from a lazy stream (:class:`~langres.core.presets._SpendCappedModule`,
    E9) knows exactly when to stop pulling without needing to peek at (and
    thereby trigger the computation of) the next group.

    Args:
        judgements: The judgements produced from ONE call spanning ONE group,
            in any order. Must be non-empty.
        call_cost_usd: The measured (or priced) cost of the single call that
            produced all of these judgements.
        group_id: Identifier of the source group, e.g. the group's
            ``ERCandidateGroup.group_id``.

    Returns:
        New ``PairwiseJudgement`` objects (the originals are not mutated)
        with ``provenance["cost_usd"]``/``provenance["group_id"]`` set per the
        convention above; all other provenance keys are preserved.

    Raises:
        ValueError: If ``judgements`` is empty (there is no "first" judgement
            to carry the cost).
    """
    if not judgements:
        raise ValueError("stamp_group_cost requires at least one judgement")

    stamped = []
    last_index = len(judgements) - 1
    for index, judgement in enumerate(judgements):
        cost = call_cost_usd if index == 0 else 0.0
        new_provenance: dict[str, Any] = {
            **judgement.provenance,
            "cost_usd": cost,
            "group_id": group_id,
        }
        if index == last_index:
            new_provenance["group_end"] = True
        stamped.append(judgement.model_copy(update={"provenance": new_provenance}))
    return stamped
