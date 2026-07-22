"""Lazy Sentence Transformers resources for embedding and reranking."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from langres.core.embeddings import SentenceTransformerEmbedder
from langres.core.model_ref import ModelRef, UnsupportedBackboneError, to_config
from langres.core.registry import register
from langres.resources._model_ref import normalize_inprocess_ref
from langres.resources.base import (
    EmbeddingBatch,
    EmbeddingFacts,
    RerankBatch,
    RerankRequest,
    RerankerRuntimeConfig,
    SentenceTransformerRuntimeConfig,
    require_unique_ids,
)


def _torch_dtype(name: str | None) -> Any:
    if name is None:
        return None
    import torch

    return getattr(torch, name)


def _directory_bytes(path: Path) -> int:
    """Measure regular files below ``path``, following snapshot symlinks."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _local_artifact_bytes(model_ref: ModelRef) -> int | None:
    """Measure a local path or an already-cached Hub snapshot without networking."""
    local = Path(model_ref.base).expanduser()
    if local.exists():
        return _directory_bytes(local) if local.is_dir() else local.stat().st_size
    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            repo_id=model_ref.base,
            revision=model_ref.revision,
            local_files_only=True,
        )
    except (FileNotFoundError, OSError, ValueError):
        return None
    return _directory_bytes(Path(snapshot))


def _loaded_model_facts(model: Any, model_ref: ModelRef) -> dict[str, int | None]:
    """Best-effort parameter, resident tensor, and cached artifact sizes."""
    try:
        parameters = tuple(model.parameters())
    except (AttributeError, TypeError):
        parameters = ()
    try:
        buffers = tuple(model.buffers())
    except (AttributeError, TypeError):
        buffers = ()
    parameter_count = sum(int(parameter.numel()) for parameter in parameters) or None
    loaded_memory = (
        sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in (*parameters, *buffers))
        or None
    )
    return {
        "parameter_count": parameter_count,
        "artifact_bytes": _local_artifact_bytes(model_ref),
        "loaded_memory_bytes": loaded_memory,
    }


@register("resource_sentence_transformer")
class SentenceTransformer:
    """Dense embedding resource backed by the existing lazy embedder."""

    type_name: ClassVar[str] = "resource_sentence_transformer"

    def __init__(
        self,
        model: str | dict[str, str] | ModelRef,
        *,
        runtime_config: SentenceTransformerRuntimeConfig | None = None,
    ) -> None:
        self.model_ref = normalize_inprocess_ref(model, slot="SentenceTransformer")
        if self.model_ref.adapter is not None:
            raise UnsupportedBackboneError(
                "SentenceTransformer does not assemble PEFT adapters. Fix: merge the "
                "adapter into an embedding checkpoint or pass the merged model ref."
            )
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
        if len(vectors) != len(texts):
            raise ValueError(
                "SentenceTransformer requires one vector per input text; "
                f"got {len(vectors)} vectors for {len(texts)} texts."
            )
        loaded_model = getattr(self._embedder, "_model", None)
        model_facts = (
            _loaded_model_facts(loaded_model, self.model_ref)
            if loaded_model is not None
            else {
                "parameter_count": None,
                "artifact_bytes": None,
                "loaded_memory_bytes": None,
            }
        )
        return EmbeddingBatch(
            vectors=vectors,
            model_ref=self.model_ref,
            facts=EmbeddingFacts(
                dimension=int(vectors.shape[1]),
                dtype=str(vectors.dtype),
                normalized=self.runtime_config.normalize_embeddings,
                quantization=self.runtime_config.quantization,
                **model_facts,
            ),
        )

    @property
    def config(self) -> dict[str, object]:
        """Weightless model and runtime configuration."""
        return {
            "model": to_config(self.model_ref),
            "runtime_config": self.runtime_config.model_dump(mode="json"),
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "SentenceTransformer":
        """Rebuild with weights still unloaded."""
        return cls(
            config["model"],  # type: ignore[arg-type]
            runtime_config=SentenceTransformerRuntimeConfig.model_validate(
                config["runtime_config"]
            ),
        )


@register("resource_cross_encoder_reranker")
class CrossEncoderReranker:
    """Lazy one-score CrossEncoder resource for pair reranking."""

    type_name: ClassVar[str] = "resource_cross_encoder_reranker"

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

    @property
    def config(self) -> dict[str, object]:
        """Weightless model and runtime configuration."""
        return {
            "model": to_config(self.model_ref),
            "runtime_config": self.runtime_config.model_dump(mode="json"),
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "CrossEncoderReranker":
        """Rebuild with weights still unloaded."""
        return cls(
            config["model"],  # type: ignore[arg-type]
            runtime_config=RerankerRuntimeConfig.model_validate(config["runtime_config"]),
        )

    def rerank(self, pairs: Sequence[RerankRequest]) -> RerankBatch:
        """Score each pair without filtering or assigning a topology role."""
        require_unique_ids(
            [pair.pair_id for pair in pairs],
            field="pair_ids",
            operation="CrossEncoderReranker.rerank",
        )
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
