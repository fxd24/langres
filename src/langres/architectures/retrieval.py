"""Four named retrieval recipes over the shared resource Op algebra."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias, cast

from pydantic import BaseModel

from langres.core.clusterer import Clusterer
from langres.core.model_ref import ModelRef
from langres.core.op import Stage, ThresholdSelect, TopKSelect
from langres.core.op_adapters import ClustererStage
from langres.core.registry import register_model
from langres.core.resolver import ERModel
from langres.core.results import DedupeResult, LinkVerdict
from langres.resources import (
    CrossEncoderReranker,
    Embedder,
    Generate,
    LLM,
    Parse,
    Rerank,
    Reranker,
    Retrieve as RetrieveOp,
    SentenceTransformer,
    llm_from_model_ref,
)

if TYPE_CHECKING:
    from langres.tracking.judgement_log import JudgementLog

ResourceRef: TypeAlias = str | dict[str, str] | ModelRef
EmbedderLike: TypeAlias = Embedder | ResourceRef
RerankerLike: TypeAlias = Reranker | ResourceRef
LLMLike: TypeAlias = LLM | ResourceRef


def _embedder(value: EmbedderLike) -> Embedder:
    return value if isinstance(value, Embedder) else SentenceTransformer(value)


def _reranker(value: RerankerLike) -> Reranker:
    return value if isinstance(value, Reranker) else CrossEncoderReranker(value)


def _llm(value: LLMLike) -> LLM:
    return value if isinstance(value, LLM) else llm_from_model_ref(value)


def _cluster_stage(clusterer: Clusterer | None) -> ClustererStage[Any]:
    return ClustererStage(clusterer if clusterer is not None else Clusterer(threshold=0.0))


class _ResearchRecipe(ERModel):
    """Shared inspection sugar; each concrete class still spells out its topology."""

    accepted_method_kinds: ClassVar[frozenset[str] | None] = frozenset()

    def _initialize_recipe(
        self,
        resources: dict[str, Embedder | Reranker | LLM],
        *,
        schema: type[BaseModel] | None,
        budget_usd: float | None,
    ) -> None:
        self._declared_resources = {
            slot: resource.model_ref for slot, resource in resources.items()
        }
        self._resource_values = resources
        self._init_state(budget_usd=budget_usd)
        if schema is not None:
            self._bind(schema)

    def _adopt_topology(
        self,
        ops: list[Stage],
    ) -> None:
        built = type(self).from_topology(ops=ops, monitor=self._spend_monitor)
        self.__dict__ = built.__dict__

    def _recipe_ops(self, schema: type[BaseModel]) -> list[Stage]:
        raise NotImplementedError

    def _bind(self, schema: type[BaseModel]) -> None:
        """Build this recipe's explicit topology once schema is known."""
        if not self.is_bound:
            self._adopt_topology(self._recipe_ops(schema))

    def dedupe(
        self,
        records: list[dict[str, Any]],
        *,
        log: JudgementLog | str | Path | None = None,
    ) -> DedupeResult:
        """Infer a schema before the base front door chooses its topology path."""
        if len(records) >= 2 and self._ops is None:
            self._prepare(records)
        return super().dedupe(records, log=log)

    def compare(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
        *,
        log: JudgementLog | str | Path | None = None,
    ) -> LinkVerdict:
        """Infer a schema before the base front door chooses its topology path."""
        if self._ops is None:
            self._prepare([left, right])
        return super().compare(left, right, log=log)

    @property
    def resources(self) -> dict[str, ModelRef]:
        """Every model-bearing slot in this recipe, derived from the live Ops."""
        if self._ops is None:
            return dict(self._declared_resources)
        slots: dict[str, ModelRef] = {}
        for stage in self._require_ops():
            if isinstance(stage, RetrieveOp):
                slots["embedder"] = stage.resource.model_ref
            elif isinstance(stage, Rerank):
                slots["reranker"] = stage.resource.model_ref
            elif isinstance(stage, Generate):
                slots["llm"] = stage.resource.model_ref
        return slots

    @property
    def backbone(self) -> str | None:
        """Compatibility sugar only when the recipe has exactly one model slot."""
        resources = self.resources
        if len(resources) != 1:
            return None
        return next(iter(resources.values())).base


@register_model("retrieve")
class Retrieve(_ResearchRecipe):
    """Embed, retrieve nearest neighbours, threshold, and cluster."""

    def __init__(
        self,
        *,
        embedder: EmbedderLike,
        schema: type[BaseModel] | None = None,
        retrieve_k: int = 20,
        threshold: float = 0.5,
        text_field: str | None = None,
        clusterer: Clusterer | None = None,
        budget_usd: float | None = None,
    ) -> None:
        self.retrieve_k = retrieve_k
        self.threshold = threshold
        self.text_field = text_field
        self.clusterer_override = clusterer
        self._initialize_recipe(
            {"embedder": _embedder(embedder)},
            schema=schema,
            budget_usd=budget_usd,
        )

    def _recipe_ops(self, schema: type[BaseModel]) -> list[Stage]:
        embedder = cast(Embedder, self._resource_values["embedder"])
        return [
            RetrieveOp(
                embedder,
                schema=schema,
                k=self.retrieve_k,
                text_field=self.text_field,
            ),
            ThresholdSelect(self.threshold),
            _cluster_stage(self.clusterer_override),
        ]


@register_model("retrieve_rerank")
class RetrieveRerank(_ResearchRecipe):
    """Retrieve, rescore with one reusable Reranker, threshold, and cluster."""

    def __init__(
        self,
        *,
        embedder: EmbedderLike,
        reranker: RerankerLike,
        schema: type[BaseModel] | None = None,
        retrieve_k: int = 20,
        threshold: float = 0.5,
        text_field: str | None = None,
        clusterer: Clusterer | None = None,
        budget_usd: float | None = None,
    ) -> None:
        self.retrieve_k = retrieve_k
        self.threshold = threshold
        self.text_field = text_field
        self.clusterer_override = clusterer
        self._initialize_recipe(
            {
                "embedder": _embedder(embedder),
                "reranker": _reranker(reranker),
            },
            schema=schema,
            budget_usd=budget_usd,
        )

    def _recipe_ops(self, schema: type[BaseModel]) -> list[Stage]:
        embedder = cast(Embedder, self._resource_values["embedder"])
        reranker = cast(Reranker, self._resource_values["reranker"])
        return [
            RetrieveOp(
                embedder,
                schema=schema,
                k=self.retrieve_k,
                text_field=self.text_field,
            ),
            Rerank(reranker),
            ThresholdSelect(self.threshold),
            _cluster_stage(self.clusterer_override),
        ]


@register_model("retrieve_llm")
class RetrieveLLM(_ResearchRecipe):
    """Retrieve candidates, ask one LLM, parse decisions, and cluster."""

    def __init__(
        self,
        *,
        embedder: EmbedderLike,
        llm: LLMLike,
        schema: type[BaseModel] | None = None,
        retrieve_k: int = 20,
        llm_k: int = 5,
        text_field: str | None = None,
        clusterer: Clusterer | None = None,
        budget_usd: float | None = None,
    ) -> None:
        self.retrieve_k = retrieve_k
        self.llm_k = llm_k
        self.text_field = text_field
        self.clusterer_override = clusterer
        self._initialize_recipe(
            {
                "embedder": _embedder(embedder),
                "llm": _llm(llm),
            },
            schema=schema,
            budget_usd=budget_usd,
        )

    def _recipe_ops(self, schema: type[BaseModel]) -> list[Stage]:
        embedder = cast(Embedder, self._resource_values["embedder"])
        llm = cast(LLM, self._resource_values["llm"])
        return [
            RetrieveOp(
                embedder,
                schema=schema,
                k=self.retrieve_k,
                text_field=self.text_field,
            ),
            TopKSelect(self.llm_k),
            Generate(llm),
            Parse(),
            ThresholdSelect(0.5),
            _cluster_stage(self.clusterer_override),
        ]


@register_model("retrieve_rerank_llm")
class RetrieveRerankLLM(_ResearchRecipe):
    """Retrieve, rerank, prune, ask one LLM, parse decisions, and cluster."""

    def __init__(
        self,
        *,
        embedder: EmbedderLike,
        reranker: RerankerLike,
        llm: LLMLike,
        schema: type[BaseModel] | None = None,
        retrieve_k: int = 20,
        llm_k: int = 5,
        text_field: str | None = None,
        clusterer: Clusterer | None = None,
        budget_usd: float | None = None,
    ) -> None:
        self.retrieve_k = retrieve_k
        self.llm_k = llm_k
        self.text_field = text_field
        self.clusterer_override = clusterer
        self._initialize_recipe(
            {
                "embedder": _embedder(embedder),
                "reranker": _reranker(reranker),
                "llm": _llm(llm),
            },
            schema=schema,
            budget_usd=budget_usd,
        )

    def _recipe_ops(self, schema: type[BaseModel]) -> list[Stage]:
        embedder = cast(Embedder, self._resource_values["embedder"])
        reranker = cast(Reranker, self._resource_values["reranker"])
        llm = cast(LLM, self._resource_values["llm"])
        return [
            RetrieveOp(
                embedder,
                schema=schema,
                k=self.retrieve_k,
                text_field=self.text_field,
            ),
            Rerank(reranker),
            TopKSelect(self.llm_k),
            Generate(llm),
            Parse(),
            ThresholdSelect(0.5),
            _cluster_stage(self.clusterer_override),
        ]


__all__ = ["Retrieve", "RetrieveLLM", "RetrieveRerank", "RetrieveRerankLLM"]
