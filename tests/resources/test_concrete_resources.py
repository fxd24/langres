"""Fast production-resource tests with injected local backends."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from langres.core.model_ref import ModelRef
from langres.core.model_ref import UnsupportedBackboneError
from langres.resources import (
    CrossEncoderReranker,
    GenerationRequest,
    LLMRuntimeConfig,
    LiteLLM,
    RerankRequest,
    RerankerRuntimeConfig,
    SentenceTransformer,
    SentenceTransformerRuntimeConfig,
    TransformersLLM,
    llm_from_model_ref,
)


def test_sentence_transformer_resource_preserves_model_ref_and_runtime_config() -> None:
    ref = ModelRef(base="sentence-transformers/all-MiniLM-L6-v2", kind="hf", revision="abc123")
    runtime = SentenceTransformerRuntimeConfig(
        batch_size=16,
        device="cpu",
        dtype="float32",
        normalize_embeddings=False,
        local_files_only=True,
    )
    resource = SentenceTransformer(ref, runtime_config=runtime)

    assert resource.model_ref is ref
    assert resource.runtime_config is runtime
    assert resource._embedder._model is None
    assert resource.model_ref.revision == "abc123"
    assert resource.runtime_config.model_dump(mode="json") == {
        "backend": "torch",
        "batch_size": 16,
        "device": "cpu",
        "dtype": "float32",
        "local_files_only": True,
        "normalize_embeddings": False,
        "show_progress_bar": False,
    }


def test_sentence_transformer_resource_embeds_through_legacy_provider() -> None:
    resource = SentenceTransformer("sentence-transformers/test")

    class _Embedder:
        def encode(self, texts: list[str]) -> np.ndarray:
            assert texts == ["Acme", "Globex"]
            return np.ones((2, 3), dtype=np.float32)

    resource._embedder = _Embedder()
    batch = resource.embed(["Acme", "Globex"])
    assert batch.vectors.shape == (2, 3)
    assert batch.facts is not None
    assert batch.facts.dimension == 3


def test_inprocess_resources_reject_served_and_adapter_refs() -> None:
    with pytest.raises(UnsupportedBackboneError, match="in-process"):
        CrossEncoderReranker(ModelRef(base="openai/model", kind="api"))
    with pytest.raises(UnsupportedBackboneError, match="does not assemble PEFT"):
        CrossEncoderReranker(ModelRef(base="org/model", kind="hf", adapter="org/adapter"))


def test_cross_encoder_is_lazy_and_returns_bounded_pair_scores() -> None:
    resource = CrossEncoderReranker(
        ModelRef(base="cross-encoder/ms-marco-MiniLM-L6-v2", kind="hf", revision="deadbeef"),
        runtime_config=RerankerRuntimeConfig(batch_size=2, device="cpu"),
    )

    class _FakeCrossEncoder:
        def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> np.ndarray:
            assert kwargs["batch_size"] == 2
            return np.array([0.8, 0.1], dtype=np.float32)

    assert resource._model is None
    resource._model = _FakeCrossEncoder()
    batch = resource.rerank(
        [
            RerankRequest(pair_id="a:b", left="Acme", right="ACME"),
            RerankRequest(pair_id="a:c", left="Acme", right="Globex"),
        ]
    )

    assert batch.scores == pytest.approx((0.8, 0.1))
    assert batch.pair_ids == ("a:b", "a:c")


def test_cross_encoder_handles_empty_single_column_and_invalid_outputs() -> None:
    resource = CrossEncoderReranker("org/model")
    assert resource.rerank([]).scores == ()

    class _Model:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values

        def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> np.ndarray:
            return self.values

    request = [RerankRequest(pair_id="one", left="a", right="b")]
    resource._model = _Model(np.array([[0.4]], dtype=np.float32))
    assert resource.rerank(request).scores == pytest.approx((0.4,))

    resource._model = _Model(np.array([[0.4, 0.6]], dtype=np.float32))
    with pytest.raises(ValueError, match="one scalar"):
        resource.rerank(request)

    resource._model = _Model(np.array([1.2], dtype=np.float32))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        resource.rerank(request)


def test_cross_encoder_loader_receives_model_identity_and_runtime(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class _CrossEncoder:
        def __init__(self, model: str, **kwargs: Any) -> None:
            calls.append((model, kwargs))

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _CrossEncoder)
    resource = CrossEncoderReranker(
        ModelRef(base="org/model", kind="hf", revision="abc123"),
        runtime_config=RerankerRuntimeConfig(
            device="cpu",
            dtype="float32",
            local_files_only=True,
        ),
    )
    resource._get_model()

    assert calls[0][0] == "org/model"
    assert calls[0][1]["revision"] == "abc123"
    assert calls[0][1]["device"] == "cpu"
    assert calls[0][1]["local_files_only"] is True
    assert calls[0][1]["trust_remote_code"] is False


def test_litellm_resource_uses_injected_client_without_importing_backend() -> None:
    class _Client:
        def completion(self, **kwargs: Any) -> Any:
            assert kwargs["model"] == "openai/gpt-4o-mini"
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="MATCH"),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
            )

        def completion_cost(self, completion_response: Any) -> float:
            return 0.001

    resource = LiteLLM(
        ModelRef(base="openai/gpt-4o-mini", kind="api"),
        runtime_config=LLMRuntimeConfig(temperature=0.0, max_new_tokens=8),
        client=_Client(),
    )

    batch = resource.generate([GenerationRequest.user("a:b", "same?")])
    envelope = batch.outputs[0]

    assert envelope.content == "MATCH"
    assert envelope.usage is not None
    assert envelope.usage.input_tokens == 3
    assert envelope.cost_usd == pytest.approx(0.001)


def test_litellm_resource_forwards_endpoint_timeout_and_tolerates_unknown_cost() -> None:
    calls: list[dict[str, Any]] = []

    class _Client:
        def completion(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="x"))],
                usage=None,
            )

        def completion_cost(self, completion_response: Any) -> float:
            raise ValueError("price unknown")

    resource = LiteLLM(
        ModelRef(
            base="served-model",
            kind="endpoint",
            api_base="http://localhost:8000/v1",
        ),
        runtime_config=LLMRuntimeConfig(timeout_seconds=12.0),
        client=_Client(),
    )
    output = resource.generate([GenerationRequest.user("one", "prompt")]).outputs[0]

    assert calls[0]["api_base"] == "http://localhost:8000/v1"
    assert calls[0]["timeout"] == 12.0
    assert output.cost_usd is None
    assert output.finish_reason is None


def test_litellm_rejects_a_hub_revision_it_cannot_honor() -> None:
    with pytest.raises(UnsupportedBackboneError, match="cannot honor"):
        LiteLLM(ModelRef(base="org/model", kind="hf", revision="abc123"))


def test_transformers_llm_uses_the_same_generation_contract() -> None:
    class _Backend:
        def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="NO_MATCH"),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=4, completion_tokens=1),
            )

    resource = TransformersLLM(
        ModelRef(base="./local-model", kind="local"),
        runtime_config=LLMRuntimeConfig(max_new_tokens=8),
    )
    resource._backend = _Backend()

    batch = resource.generate([GenerationRequest.user("a:b", "same?")])

    assert batch.outputs[0].content == "NO_MATCH"
    assert batch.outputs[0].usage is not None
    assert batch.outputs[0].usage.output_tokens == 1


def test_llm_factory_routes_only_on_model_ref_kind() -> None:
    assert isinstance(
        llm_from_model_ref(ModelRef(base="./local", kind="local")),
        TransformersLLM,
    )
    assert isinstance(
        llm_from_model_ref(ModelRef(base="openai/gpt-4o-mini", kind="api")),
        LiteLLM,
    )
    with pytest.raises(UnsupportedBackboneError, match="requires kind"):
        TransformersLLM(ModelRef(base="openai/gpt-4o-mini", kind="api"))


def test_lazy_backend_construction_preserves_local_runtime() -> None:
    resource = TransformersLLM(
        ModelRef(base="./local", kind="local"),
        runtime_config=LLMRuntimeConfig(
            max_new_tokens=17,
            device="cpu",
            dtype="float32",
            local_files_only=True,
        ),
    )
    backend = resource._get_backend()
    assert backend._max_new_tokens == 17
    assert backend._device == "cpu"
    assert backend._dtype == "float32"
    assert backend._local_files_only is True


@pytest.mark.slow
def test_real_cross_encoder_smoke() -> None:
    resource = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L6-v2")
    batch = resource.rerank(
        [RerankRequest(pair_id="one", left="How many people live in Berlin?", right="Berlin")]
    )
    assert len(batch.scores) == 1
    assert 0.0 <= batch.scores[0] <= 1.0
