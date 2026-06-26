"""WeightedAverageJudge: the M0 heuristic scorer Module.

The judge is the scorer slot of the Resolver. It is **arg-free** to construct
(``WeightedAverageJudge()``); it owns the scoring *rule* but not the features.
The Resolver drives the pipeline:

    comparator = Comparator.from_schema(CompanySchema)
    judge = WeightedAverageJudge()
    for judgement in judge.forward(candidates, comparator=comparator):
        ...

For each candidate the judge asks the ``comparator`` for a
:class:`~langres.core.feature.ComparisonVector`, then combines the present
similarities with :func:`~langres.core.feature.combine_present` using the
FeatureSpec weights **normalized to sum to 1.0** (the 0.5 evidence floor is only
meaningful against a unit-weight total). It emits a
:class:`~langres.core.models.PairwiseJudgement` whose ``decision_step`` records
why a score was produced (``weighted_average``) or forced to zero
(``all_features_missing`` / ``below_evidence_floor``).

Scoring is also exposed directly via :meth:`score` so the Resolver (or a test)
can score a single ComparisonVector without going through ``forward``.
"""

from collections.abc import Iterator
from typing import ClassVar

from langres.core.comparator import Comparator
from langres.core.feature import ComparisonVector, FeatureSpec, combine_present
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.module import Module, SchemaT
from langres.core.modules.llm_judge import _inspect_scores_impl
from langres.core.registry import register
from langres.core.reports import ScoreInspectionReport


def _normalized_weights(specs: list[FeatureSpec]) -> dict[str, float]:
    """FeatureSpec weights normalized to sum to 1.0 (even split if all zero).

    ``combine_present`` defends against an empty/zero-weight map by returning
    ``0.0``, so an even split over zero specs (empty dict) is safe.
    """
    total = sum(spec.weight for spec in specs)
    if total > 0:
        return {spec.name: spec.weight / total for spec in specs}
    even = 1.0 / len(specs) if specs else 0.0
    return {spec.name: even for spec in specs}


@register("weighted_average_judge")
class WeightedAverageJudge(Module[SchemaT]):
    """Heuristic scorer: weighted average of present similarities + evidence floor.

    Arg-free to construct; the Resolver supplies the Comparator at ``forward``
    time. The score combiner and the over-merge evidence floor live in
    :func:`~langres.core.feature.combine_present`.
    """

    # Registry key, mirrored as a class attribute so the Resolver's uniform
    # serialization helper can discover the type name (see resolver.py).
    type_name: ClassVar[str] = "weighted_average_judge"

    def score(self, vector: ComparisonVector, specs: list[FeatureSpec]) -> float:
        """Combine a ComparisonVector into a score in ``[0, 1]``.

        Weights are taken from ``specs`` and normalized to sum to 1.0 before
        applying the evidence floor.
        """
        weights = _normalized_weights(specs)
        return combine_present(vector.similarities, weights)

    def forward(
        self,
        candidates: Iterator[ERCandidate[SchemaT]],
        *,
        comparator: Comparator[SchemaT] | None = None,
    ) -> Iterator[PairwiseJudgement]:
        """Score each candidate via ``comparator`` and yield PairwiseJudgements.

        Args:
            candidates: Stream of normalized pairs from a Blocker.
            comparator: The Comparator that turns each pair into a
                ComparisonVector. Supplied by the Resolver. Required.

        Yields:
            One PairwiseJudgement per candidate, with provenance carrying the
            per-feature levels and similarities.

        Raises:
            ValueError: If ``comparator`` is not provided.
        """
        if comparator is None:
            raise ValueError(
                "WeightedAverageJudge.forward requires a comparator "
                "(the Resolver supplies it: forward(candidates, comparator=...))."
            )
        specs = comparator.feature_specs
        weights = _normalized_weights(specs)

        for candidate in candidates:
            vector = comparator.compare(candidate.left, candidate.right)
            score = combine_present(vector.similarities, weights)
            decision_step = self._decision_step(vector, score)
            yield PairwiseJudgement(
                left_id=candidate.left.id,  # type: ignore[attr-defined]
                right_id=candidate.right.id,  # type: ignore[attr-defined]
                score=score,
                score_type="heuristic",
                decision_step=decision_step,
                provenance={
                    "levels": {name: level.value for name, level in vector.levels.items()},
                    "similarities": dict(vector.similarities),
                },
            )

    @staticmethod
    def _decision_step(vector: ComparisonVector, score: float) -> str:
        """Classify why a score was produced or forced to zero."""
        if score > 0.0:
            return "weighted_average"
        if not vector.similarities:
            return "all_features_missing"
        return "below_evidence_floor"

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        """Explore scores without ground truth (shared Module utility)."""
        return _inspect_scores_impl(judgements, sample_size)

    @property
    def config(self) -> dict[str, object]:
        """Serializable config. The judge is stateless, so config is empty."""
        return {}

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "WeightedAverageJudge[SchemaT]":
        """Reconstruct from :attr:`config` (stateless)."""
        return cls()
