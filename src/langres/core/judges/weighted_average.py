"""WeightedAverageJudge: the M0 heuristic scorer Module.

The judge is the scorer slot of the Resolver. It **owns its FeatureSpecs** (the
weights) and reads each candidate's attached
:class:`~langres.core.feature.ComparisonVector`. The Resolver drives the
pipeline: a Comparator attaches a ``comparison`` to each candidate, then the
judge scores it::

    comparator = Comparator.from_schema(CompanySchema)
    judge = WeightedAverageJudge(feature_specs=comparator.feature_specs)
    candidates = (
        c.model_copy(update={"comparison": comparator.compare(c.left, c.right)})
        for c in raw_candidates
    )
    for judgement in judge.forward(candidates):
        ...

For each candidate the judge reads ``candidate.comparison`` and combines the
present similarities with :func:`~langres.core.feature.combine_present` using the
FeatureSpec weights **normalized to sum to 1.0** (the 0.5 evidence floor is only
meaningful against a unit-weight total). It emits a
:class:`~langres.core.models.PairwiseJudgement` whose ``decision_step`` records
why a score was produced (``weighted_average``) or forced to zero
(``all_features_missing`` / ``below_evidence_floor``).

Scoring is also exposed directly via :meth:`score` so the Resolver (or a test)
can score a single ComparisonVector without going through ``forward``.
"""

from collections.abc import Iterator
from typing import ClassVar, cast

from langres.core.feature import ComparisonVector, FeatureSpec, combine_present
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.module import Module, SchemaT
from langres.core.registry import register
from langres.core.reports import ScoreInspectionReport, _inspect_scores_impl


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

    Owns its FeatureSpecs (the weights). Consumes each candidate's attached
    ``comparison`` (a :class:`~langres.core.feature.ComparisonVector`). The score
    combiner and the over-merge evidence floor live in
    :func:`~langres.core.feature.combine_present`.
    """

    # Registry key, mirrored as a class attribute so the Resolver's uniform
    # serialization helper can discover the type name (see resolver.py).
    type_name: ClassVar[str] = "weighted_average_judge"

    def __init__(self, feature_specs: list[FeatureSpec]) -> None:
        """Initialize with the features (and their weights) to score on.

        Args:
            feature_specs: The features to combine. Their weights are normalized
                to sum to 1.0 at scoring time. These should match the
                Comparator's features so the comparison vector and the weights
                line up.
        """
        self.feature_specs = feature_specs

    def score(self, vector: ComparisonVector) -> float:
        """Combine a ComparisonVector into a score in ``[0, 1]``.

        Weights are taken from :attr:`feature_specs` and normalized to sum to
        1.0 before applying the evidence floor.
        """
        weights = _normalized_weights(self.feature_specs)
        return combine_present(vector.similarities, weights)

    def forward(self, candidates: Iterator[ERCandidate[SchemaT]]) -> Iterator[PairwiseJudgement]:
        """Score each candidate's attached comparison and yield PairwiseJudgements.

        Args:
            candidates: Stream of normalized pairs from a Blocker, each carrying
                a ``comparison`` vector attached by a Comparator stage.

        Yields:
            One PairwiseJudgement per candidate, with provenance carrying the
            per-feature levels and similarities.

        Raises:
            ValueError: If a candidate carries no comparison vector.
        """
        weights = _normalized_weights(self.feature_specs)

        for candidate in candidates:
            vector = candidate.comparison
            if vector is None:
                raise ValueError(
                    "WeightedAverageJudge requires candidates carrying a comparison "
                    "vector — add a Comparator to the pipeline."
                )
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
        """Serializable config: the FeatureSpecs (names + weights) it scores on."""
        return {"feature_specs": [spec.model_dump() for spec in self.feature_specs]}

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "WeightedAverageJudge[SchemaT]":
        """Reconstruct from :attr:`config`, rebuilding the FeatureSpecs."""
        specs = [
            FeatureSpec.model_validate(s) for s in cast("list[object]", config["feature_specs"])
        ]
        return cls(feature_specs=specs)
