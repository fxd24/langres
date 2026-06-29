"""EmbeddingScoreJudge: the zero-spend embedding-cosine scorer Module.

This judge is the scorer slot for the ``embedding_cosine`` method. Unlike
:class:`~langres.core.judges.weighted_average.WeightedAverageJudge` (which needs
a Comparator upstream) it consumes nothing but each candidate's
``similarity_score`` — the cosine similarity a :class:`~langres.core.blockers.vector.VectorBlocker`
already attached during blocking. It emits a
:class:`~langres.core.models.PairwiseJudgement` whose ``score`` *is* that
similarity, so the Clusterer can threshold the raw embedding signal directly with
no second model call.

Because it reads ``candidate.similarity_score``, it requires a VectorBlocker
upstream: any blocker that leaves ``similarity_score`` as ``None`` (e.g.
``AllPairsBlocker``) makes the score undefined, and the judge raises a clear
``ValueError`` pointing the caller at the VectorBlocker.

It is registry-serializable exactly like ``WeightedAverageJudge``
(``@register("embedding_score_judge")`` + ``type_name`` + ``config`` /
``from_config``), so a Resolver with it in the ``module`` slot round-trips
through ``save`` / ``load``.
"""

from collections.abc import Iterator
from typing import ClassVar

from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.module import Module, SchemaT
from langres.core.registry import register
from langres.core.reports import ScoreInspectionReport, _inspect_scores_impl


@register("embedding_score_judge")
class EmbeddingScoreJudge(Module[SchemaT]):
    """Scorer that passes a VectorBlocker's cosine ``similarity_score`` through.

    Owns a single ``threshold`` used only to label each judgement's
    ``decision_step`` (``embedding_match`` / ``embedding_no_match``); the emitted
    ``score`` is always the raw similarity, never thresholded (the Clusterer owns
    the actual cut). Holds no model and makes no API call: it is fully zero-spend.
    """

    # Registry key, mirrored as a class attribute so the Resolver's uniform
    # serialization helper can discover the type name (see resolver.py).
    type_name: ClassVar[str] = "embedding_score_judge"

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize with the decision threshold.

        Args:
            threshold: Similarity at/above which a pair is labelled a match in
                ``decision_step``. Does not affect the emitted ``score`` (which is
                the raw similarity) — clustering is the Clusterer's job.

        Raises:
            ValueError: If ``threshold`` is not in ``[0.0, 1.0]``.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.threshold = threshold

    def forward(self, candidates: Iterator[ERCandidate[SchemaT]]) -> Iterator[PairwiseJudgement]:
        """Emit one PairwiseJudgement per candidate, scoring on ``similarity_score``.

        Args:
            candidates: Stream of normalized pairs from a ``VectorBlocker`` (each
                carrying the cosine ``similarity_score`` it computed at blocking).

        Yields:
            One PairwiseJudgement per candidate whose ``score`` is the candidate's
            ``similarity_score``.

        Raises:
            ValueError: If a candidate's ``similarity_score`` is ``None`` (the
                upstream blocker is not a VectorBlocker, so there is no similarity
                to score on).
        """
        for candidate in candidates:
            similarity = candidate.similarity_score
            if similarity is None:
                raise ValueError(
                    "EmbeddingScoreJudge requires candidates carrying a "
                    "similarity_score — use a VectorBlocker upstream (an AllPairs "
                    "or other non-vector blocker leaves similarity_score unset)."
                )
            decision_step = (
                "embedding_match" if similarity >= self.threshold else "embedding_no_match"
            )
            yield PairwiseJudgement(
                left_id=candidate.left.id,  # type: ignore[attr-defined]
                right_id=candidate.right.id,  # type: ignore[attr-defined]
                score=similarity,
                score_type="sim_cos",
                decision_step=decision_step,
                provenance={"similarity_score": similarity, "threshold": self.threshold},
            )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        """Explore scores without ground truth (shared Module utility)."""
        return _inspect_scores_impl(judgements, sample_size)

    @property
    def config(self) -> dict[str, object]:
        """Serializable config: the decision threshold."""
        return {"threshold": self.threshold}

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "EmbeddingScoreJudge[SchemaT]":
        """Reconstruct from :attr:`config`."""
        return cls(threshold=float(config["threshold"]))  # type: ignore[arg-type]
