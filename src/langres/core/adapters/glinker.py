"""GLinkerAdapter: contract-conformance stub for GLiNER-based entity resolution.

GLinker (gliner-linker) is an external entity resolution method that wraps
GLiNER (Generalist and Lightweight Model for Named Entity Recognition) for
zero-shot NER-driven entity matching. It can act as BOTH a Blocker (candidate
generation by identifying named entity spans) AND a Module (pairwise scoring
via entity span similarity).

ROADMAP M3 benchmarks GLinker behind langres's own Blocker and Module
interfaces to measure how an external NER-based approach compares to
langres's built-in methods on standard ER benchmarks.

This class is a CONTRACT-CONFORMANCE STUB:
- All four abstract-method bodies raise NotImplementedError.
- The class fully implements the Blocker[SchemaT] and Module[SchemaT] ABCs
  (isinstance checks pass), registers in the component registry, and
  type-checks under mypy --strict.
- The config + from_config round-trip is REAL (not stubbed).
- Method bodies are implemented in M3.
"""

from collections.abc import Iterator
from typing import Annotated, Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, Field

from langres.core.blocker import Blocker
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.module import Module
from langres.core.registry import register
from langres.core.reports import CandidateInspectionReport, ScoreInspectionReport

# Own TypeVar so mypy --strict sees both base-class parametrisations as
# consistent; explicitly listed in Generic[SchemaT] on the class.
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class GLinkerConfig(BaseModel):
    """Configuration for GLinkerAdapter.

    Attributes:
        model_name: HuggingFace model ID for the GLiNER model.
        threshold: Minimum entity-match confidence to emit a candidate (M3).
    """

    model_name: str = "urchade/gliner_medium-v2.1"
    threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5


@register("glinker_adapter")
class GLinkerAdapter(Blocker[SchemaT], Module[SchemaT], Generic[SchemaT]):
    """Contract-conformance stub for GLiNER-based entity resolution (M3 benchmark).

    Implements both Blocker[SchemaT] and Module[SchemaT] to prove the langres
    interfaces are implementable by an external NER-driven method.

    All method bodies raise NotImplementedError until M3.
    """

    type_name: ClassVar[str] = "glinker_adapter"

    def __init__(self, config: GLinkerConfig | None = None) -> None:
        self._config: GLinkerConfig = config or GLinkerConfig()

    @property
    def config(self) -> dict[str, object]:
        """Serializable construction config as a plain dict (component convention)."""
        return self._config.model_dump()

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "GLinkerAdapter[SchemaT]":
        """Construct from a plain-dict config (round-trippable via model_dump()).

        Args:
            config: Dict produced by ``GLinkerConfig.model_dump()``.

        Returns:
            GLinkerAdapter instance with validated config.
        """
        return cls(GLinkerConfig.model_validate(config))

    # ------------------------------------------------------------------
    # Blocker abstract methods
    # ------------------------------------------------------------------

    def stream(self, data: list[Any]) -> Iterator[ERCandidate[SchemaT]]:
        """Generate candidate pairs from input data.

        Not implemented until M3.

        Raises:
            NotImplementedError: Always — stub body.
        """
        raise NotImplementedError(
            "GLinkerAdapter is a benchmarking stub; implemented in M3"
        )  # pragma: no cover

    def inspect_candidates(
        self,
        candidates: list[ERCandidate[SchemaT]],
        entities: list[SchemaT],
        sample_size: int = 10,
    ) -> CandidateInspectionReport:
        """Explore candidate statistics without ground truth.

        Not implemented until M3.

        Raises:
            NotImplementedError: Always — stub body.
        """
        raise NotImplementedError(
            "GLinkerAdapter is a benchmarking stub; implemented in M3"
        )  # pragma: no cover

    # ------------------------------------------------------------------
    # Module abstract methods
    # ------------------------------------------------------------------

    def forward(self, candidates: Iterator[ERCandidate[SchemaT]]) -> Iterator[PairwiseJudgement]:
        """Score candidate pairs and yield pairwise judgements.

        Not implemented until M3.

        Raises:
            NotImplementedError: Always — stub body.
        """
        raise NotImplementedError(
            "GLinkerAdapter is a benchmarking stub; implemented in M3"
        )  # pragma: no cover

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        """Explore score distribution without ground truth.

        Not implemented until M3.

        Raises:
            NotImplementedError: Always — stub body.
        """
        raise NotImplementedError(
            "GLinkerAdapter is a benchmarking stub; implemented in M3"
        )  # pragma: no cover
