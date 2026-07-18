"""Lazy Sentence Transformers resources for embedding and reranking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from langres.core.embeddings import SentenceTransformerEmbedder
from langres.core.model_ref import ModelRef, UnsupportedBackboneError
from langres.resources._model_ref import normalize_inprocess_ref
from langres.resources.base import (
    EmbeddingBatch,
    EmbeddingFacts,
    RerankBatch,
    RerankRequest,
    RerankerRuntimeConfig,
    SentenceTransformerRuntimeConfig,
)


def _torch_dtype(name: str | None) -> Any:
    if name is None:
        return None
    import torch

    return getattr(torch, name)


class SentenceTransformer:
    """Dense embedding resource backed by the existing lazy embedder."""

    def __init__(
        self,
        model: str | dict[str, str] | ModelRef,
        *,
        runtime_config: SentenceTransformerRuntimeConfig | None = None,
    ) -> None:
        self.model_ref = normalize_inprocess_ref(model, slot="SentenceTransformer")
        self.runtime_config = runtime_config or SentenceTransformerRuntimeConfig()
        self._embedder = SentenceTransformerEmbedder(
            model_name=self.model_ref,
            batch_size=self.runtime_config.batch_size,
            show_progress_bar=self.runtime_config.show_progress_bar,
            normalize_embeddings=self.runtime_config.normalize_embeddings,
            device=self.runtime_config.device,
            dtype=self.runtime_config.dtype,
            backend=self.runtime_config.backend,
            local_files_only=self.runtime_config.local_files_only,
        )

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed ordered texts and attach measured vector facts."""
        vectors = self._embedder.encode(list(texts))
        return EmbeddingBatch(
            vectors=vectors,
            model_ref=self.model_ref,
            facts=EmbeddingFacts(
                dimension=int(vectors.shape[1]),
                dtype=str(vectors.dtype),
                normalized=self.runtime_config.normalize_embeddings,
            ),
        )


class CrossEncoderReranker:
    """Lazy one-score CrossEncoder resource for pair reranking."""

    def __init__(
        self,
        model: str | dict[str, str] | ModelRef,
        *,
        runtime_config: RerankerRuntimeConfig | None = None,
    ) -> None:
        self.model_ref = normalize_inprocess_ref(model, slot="CrossEncoderReranker")
        if self.model_ref.adapter is not None:
            raise UnsupportedBackboneError(
                "CrossEncoderReranker does not assemble PEFT adapters. Fix: merge the "
                "adapter into a CrossEncoder checkpoint or pass the merged model ref."
            )
        self.runtime_config = runtime_config or RerankerRuntimeConfig()
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder

            model_kwargs: dict[str, Any] = {}
            dtype = _torch_dtype(self.runtime_config.dtype)
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype
            self._model = CrossEncoder(
                self.model_ref.base,
                device=self.runtime_config.device,
                max_length=self.runtime_config.max_length,
                activation_fn=torch.nn.Sigmoid(),
                trust_remote_code=False,
                revision=self.model_ref.revision,
                local_files_only=self.runtime_config.local_files_only,
                model_kwargs=model_kwargs,
                backend=self.runtime_config.backend,
            )
        return self._model

    def rerank(self, pairs: Sequence[RerankRequest]) -> RerankBatch:
        """Score each pair without filtering or assigning a topology role."""
        if not pairs:
            return RerankBatch(pair_ids=(), scores=(), model_ref=self.model_ref)
        values = self._get_model().predict(
            [(pair.left, pair.right) for pair in pairs],
            batch_size=self.runtime_config.batch_size,
            show_progress_bar=self.runtime_config.show_progress_bar,
            convert_to_numpy=True,
        )
        scores = np.asarray(values, dtype=np.float64)
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]
        if scores.ndim != 1 or len(scores) != len(pairs):
            raise ValueError(
                "CrossEncoderReranker requires one scalar score per pair; the loaded "
                f"model returned shape {scores.shape}. Use a one-label checkpoint."
            )
        if np.any(~np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError(
                "CrossEncoderReranker requires finite scores in [0, 1]. The resource "
                "installs a sigmoid activation; verify the checkpoint emits one logit."
            )
        return RerankBatch(
            pair_ids=tuple(pair.pair_id for pair in pairs),
            scores=tuple(float(score) for score in scores),
            model_ref=self.model_ref,
        )
