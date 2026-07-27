"""ModelRef/runtime propagation for the legacy sentence-transformer embedder."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from types import ModuleType
from typing import Any

from langres.core.embeddings import SentenceTransformerEmbedder
from langres.core.model_ref import ModelRef


def test_sentence_transformer_config_round_trips_full_model_ref_and_runtime() -> None:
    ref = ModelRef(base="sentence-transformers/all-MiniLM-L6-v2", kind="hf", revision="abc123")
    original = SentenceTransformerEmbedder(
        model_name=ref,
        batch_size=16,
        device="cpu",
        dtype="float32",
        backend="torch",
        local_files_only=True,
    )

    rebuilt = SentenceTransformerEmbedder.from_config(
        type(original.config()).model_validate_json(original.config().model_dump_json())
    )

    assert rebuilt.model_ref == ref
    assert rebuilt.model_name == ref.base
    assert rebuilt.batch_size == 16
    assert rebuilt.device == "cpu"
    assert rebuilt.dtype == "float32"
    assert rebuilt.local_files_only is True
    assert rebuilt._model is None


def test_sentence_transformer_loader_receives_revision_and_runtime(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class _FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            calls.append((model_name, kwargs))

        def modules(self) -> Iterator[Any]:
            """A real ``SentenceTransformer`` is an ``nn.Module``, so it has this.

            The config-honoured guard walks the submodules looking for HF configs;
            a double that lacks the method would make the guard's own failure look
            like a load failure. Empty: this fake carries no HF config to check.
            """
            return iter(())

    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    torch = ModuleType("torch")
    torch.float32 = "torch.float32"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)
    ref = ModelRef(base="org/model", kind="hf", revision="abc123")
    embedder = SentenceTransformerEmbedder(
        model_name=ref,
        device="cpu",
        dtype="float32",
        local_files_only=True,
    )

    embedder._get_model()

    assert calls[0][0] == "org/model"
    assert calls[0][1]["revision"] == "abc123"
    assert calls[0][1]["device"] == "cpu"
    assert calls[0][1]["local_files_only"] is True
    assert calls[0][1]["trust_remote_code"] is False
    assert calls[0][1]["model_kwargs"]["torch_dtype"] == "torch.float32"


def test_parameter_count_sums_the_loaded_weights(monkeypatch) -> None:
    """Size must come from the model that produced the vectors, not a table."""

    class _Parameter:
        def __init__(self, n: int) -> None:
            self._n = n

        def numel(self) -> int:
            return self._n

    class _FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            pass

        def modules(self) -> Iterator[Any]:
            return iter(())

        def parameters(self) -> Iterator[_Parameter]:
            return iter((_Parameter(1_000_000), _Parameter(234_567)))

    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    embedder = SentenceTransformerEmbedder(model_name="org/model")

    assert embedder.parameter_count == 1_234_567
