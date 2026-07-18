"""Contract tests for import-light model resources."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from pydantic import ValidationError

from langres.core.model_ref import ModelRef, to_config
from langres.resources import (
    Embedder,
    EmbeddingBatch,
    FakeEmbedder,
    FakeLLM,
    FakeReranker,
    GenerationBatch,
    GenerationEnvelope,
    GenerationRequest,
    GenerationUsage,
    LLM,
    RerankRequest,
    RerankBatch,
    Reranker,
    SentenceTransformerRuntimeConfig,
)


def test_resources_package_imports_without_optional_model_stacks() -> None:
    script = """
import sys
import langres.resources
heavy = {"torch", "transformers", "sentence_transformers", "litellm"}
assert not (heavy & sys.modules.keys()), heavy & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_fake_resources_satisfy_typed_protocols_without_network() -> None:
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(FakeReranker(), Reranker)
    assert isinstance(FakeLLM(), LLM)


def test_fake_embedder_is_deterministic_and_carries_facts() -> None:
    resource = FakeEmbedder(dimension=8)

    first = resource.embed(["Acme", "Globex"])
    second = resource.embed(["Acme", "Globex"])

    assert isinstance(first, EmbeddingBatch)
    np.testing.assert_array_equal(first.vectors, second.vectors)
    assert first.vectors.shape == (2, 8)
    assert first.facts is not None
    assert first.facts.dimension == 8
    assert first.model_ref == resource.model_ref


def test_embedding_batch_rejects_non_matrix_vectors() -> None:
    with pytest.raises(ValidationError, match="two-dimensional"):
        EmbeddingBatch(
            vectors=np.array([1.0, 2.0], dtype=np.float32),
            model_ref=ModelRef(base="./fake", kind="local"),
        )


def test_embedding_batch_rejects_dimension_fact_mismatch() -> None:
    from langres.resources import EmbeddingFacts

    with pytest.raises(ValidationError, match="facts.dimension"):
        EmbeddingBatch(
            vectors=np.zeros((2, 4), dtype=np.float32),
            model_ref=ModelRef(base="./fake", kind="local"),
            facts=EmbeddingFacts(dimension=3, dtype="float32"),
        )


def test_rerank_batch_requires_one_score_per_pair() -> None:
    with pytest.raises(ValidationError, match="one score per pair_id"):
        RerankBatch(
            pair_ids=("a:b", "a:c"),
            scores=(0.5,),
            model_ref=ModelRef(base="./fake", kind="local"),
        )


def test_resource_batches_reject_duplicate_identities() -> None:
    ref = ModelRef(base="./fake", kind="local")
    with pytest.raises(ValidationError, match="pair_ids must be unique"):
        RerankBatch(
            pair_ids=("same", "same"),
            scores=(0.2, 0.8),
            model_ref=ref,
        )
    with pytest.raises(ValidationError, match="request_ids must be unique"):
        GenerationBatch(
            outputs=(
                GenerationEnvelope.from_content(
                    request_id="same",
                    model_ref=ref,
                    content="first",
                ),
                GenerationEnvelope.from_content(
                    request_id="same",
                    model_ref=ref,
                    content="second",
                ),
            ),
            model_ref=ref,
        )


def test_resource_identity_is_complete_frozen_and_json_stable() -> None:
    ref = ModelRef(
        base="org/model",
        kind="hf",
        adapter="org/adapter",
        revision="abc123",
    )
    runtime = SentenceTransformerRuntimeConfig(
        batch_size=8,
        device="cpu",
        dtype="float32",
        backend="onnx",
        normalize_embeddings=False,
        local_files_only=True,
    )
    identity = {
        "model_ref": to_config(ref),
        "runtime": runtime.model_dump(mode="json"),
    }

    assert identity["model_ref"] == {
        "base": "org/model",
        "kind": "hf",
        "adapter": "org/adapter",
        "revision": "abc123",
    }
    assert identity["runtime"]["backend"] == "onnx"
    assert identity["runtime"]["local_files_only"] is True
    restored = SentenceTransformerRuntimeConfig.model_validate_json(runtime.model_dump_json())
    assert restored.model_dump_json() == runtime.model_dump_json()
    with pytest.raises(ValidationError, match="frozen"):
        setattr(runtime, "batch_size", 64)


def test_fake_reranker_preserves_request_identity_and_is_deterministic() -> None:
    pairs = [
        RerankRequest(pair_id="a:b", left="Acme", right="ACME"),
        RerankRequest(pair_id="a:c", left="Acme", right="Globex"),
    ]
    resource = FakeReranker(scores={"a:b": 0.9, "a:c": 0.1})

    batch = resource.rerank(pairs)

    assert batch.pair_ids == ("a:b", "a:c")
    assert batch.scores == (0.9, 0.1)
    assert batch.model_ref == resource.model_ref


def test_generation_envelope_is_versioned_and_raw_content_is_local_only() -> None:
    envelope = GenerationEnvelope.from_content(
        request_id="a:b",
        model_ref=ModelRef(base="./fake", kind="local"),
        content="private response",
    )

    assert envelope.version == "1"
    assert envelope.content == "private response"
    assert "private response" not in envelope.model_dump_json()
    assert envelope.local_payload()["raw_content"] == "private response"
    restored = GenerationEnvelope.from_local_payload(envelope.local_payload())
    assert restored == envelope
    assert restored.content == "private response"


def test_generation_envelope_rejects_an_unknown_version() -> None:
    with pytest.raises(ValidationError):
        GenerationEnvelope.model_validate(
            {
                "version": "99",
                "request_id": "a:b",
                "model_ref": {"base": "./fake", "kind": "local"},
            }
        )


def test_generation_batch_rejects_mixed_model_identity() -> None:
    first = ModelRef(base="./first", kind="local")
    second = ModelRef(base="./second", kind="local")
    with pytest.raises(ValidationError, match="share the batch model_ref"):
        GenerationBatch(
            outputs=(
                GenerationEnvelope.from_content(
                    request_id="one",
                    model_ref=second,
                    content="x",
                ),
            ),
            model_ref=first,
        )


def test_generation_usage_preserves_unknowns_and_nested_token_subsets() -> None:
    response = {
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 8,
            "prompt_tokens_details": {
                "cached_tokens": 20,
                "cache_creation_tokens": 5,
            },
            "completion_tokens_details": {"reasoning_tokens": 3},
        }
    }

    usage = GenerationUsage.from_response(response, model="provider/model")

    assert usage.input_tokens == 50
    assert usage.output_tokens == 8
    assert usage.cache_read_input_tokens == 20
    assert usage.cache_creation_input_tokens == 5
    assert usage.reasoning_tokens == 3


def test_generation_usage_reads_top_level_cache_fields_and_keeps_unknown_none() -> None:
    response = {
        "usage": {
            "input_tokens": 12,
            "output_tokens": 2,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 7,
        }
    }

    usage = GenerationUsage.from_response(response, model="anthropic/model")
    unknown = GenerationUsage.from_response({"usage": None}, model="unknown/model")

    assert usage.input_tokens == 12
    assert usage.output_tokens == 2
    assert usage.cache_read_input_tokens == 0
    assert usage.cache_creation_input_tokens == 7
    assert usage.reasoning_tokens is None
    assert unknown.input_tokens is None
    assert unknown.output_tokens is None
    assert unknown.cache_read_input_tokens is None


def test_fake_embedder_validates_dimension_and_handles_empty_batch() -> None:
    with pytest.raises(ValueError, match="positive"):
        FakeEmbedder(dimension=0)
    batch = FakeEmbedder(dimension=4).embed([])
    assert batch.vectors.shape == (0, 4)


def test_fake_llm_returns_one_envelope_per_request() -> None:
    requests = [
        GenerationRequest.user("a:b", "Are these the same?"),
        GenerationRequest.user("a:c", "Are these the same?"),
    ]
    resource = FakeLLM(responses={"a:b": "MATCH", "a:c": "NO_MATCH"})

    batch = resource.generate(requests)

    assert tuple(output.request_id for output in batch.outputs) == ("a:b", "a:c")
    assert tuple(output.content for output in batch.outputs) == ("MATCH", "NO_MATCH")
    assert batch.model_ref == resource.model_ref
