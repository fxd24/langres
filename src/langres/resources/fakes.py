"""Deterministic, zero-network resource implementations."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from langres.core.model_ref import ModelRef
from langres.resources.base import (
    EmbeddingBatch,
    EmbeddingFacts,
    GenerationBatch,
    GenerationEnvelope,
    GenerationRequest,
    GenerationUsage,
    LLMRuntimeConfig,
    RerankBatch,
    RerankRequest,
    RerankerRuntimeConfig,
    ResourceRuntimeConfig,
)


def _unit_score(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)


class FakeEmbedder:
    """Hash-based deterministic dense embedder."""

    def __init__(self, dimension: int = 16) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.model_ref = ModelRef(base="./fake/embedder", kind="local")
        self.runtime_config = ResourceRuntimeConfig(batch_size=1024, device="cpu")
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Return stable normalized vectors without loading a model."""
        vectors: list[np.ndarray] = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            vector = rng.uniform(-1.0, 1.0, size=self.dimension).astype(np.float32)
            norm = float(np.linalg.norm(vector))
            vectors.append(vector / norm if norm else vector)
        matrix = (
            np.stack(vectors).astype(np.float32)
            if vectors
            else np.zeros((0, self.dimension), dtype=np.float32)
        )
        return EmbeddingBatch(
            vectors=matrix,
            model_ref=self.model_ref,
            facts=EmbeddingFacts(
                dimension=self.dimension,
                dtype=str(matrix.dtype),
                normalized=True,
            ),
        )


class FakeReranker:
    """Deterministic pair scorer with optional exact score fixtures."""

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.model_ref = ModelRef(base="./fake/reranker", kind="local")
        self.runtime_config = RerankerRuntimeConfig(batch_size=1024, device="cpu")
        self._scores = dict(scores or {})
        self.calls = 0

    def rerank(self, pairs: Sequence[RerankRequest]) -> RerankBatch:
        """Return one stable score per pair without selecting rows."""
        self.calls += len(pairs)
        scores = tuple(
            self._scores.get(pair.pair_id, _unit_score(f"{pair.left}\0{pair.right}"))
            for pair in pairs
        )
        return RerankBatch(
            pair_ids=tuple(pair.pair_id for pair in pairs),
            scores=scores,
            model_ref=self.model_ref,
        )


class FakeLLM:
    """Deterministic local response table implementing the LLM protocol."""

    requires_cost_accounting = False

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        *,
        default_response: str = "NO_MATCH",
    ) -> None:
        self.model_ref = ModelRef(base="./fake/llm", kind="local")
        self.runtime_config = LLMRuntimeConfig(batch_size=1024, device="cpu")
        self._responses = dict(responses or {})
        self.default_response = default_response

    def generate(self, requests: Sequence[GenerationRequest]) -> GenerationBatch:
        """Return configured responses and zero token usage."""
        outputs = tuple(
            GenerationEnvelope.from_content(
                request_id=request.request_id,
                model_ref=self.model_ref,
                content=self._responses.get(request.request_id, self.default_response),
                usage=GenerationUsage(
                    input_tokens=0,
                    output_tokens=0,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                    reasoning_tokens=0,
                    model=self.model_ref.base,
                ),
            )
            for request in requests
        )
        return GenerationBatch(outputs=outputs, model_ref=self.model_ref)
