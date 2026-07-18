"""Zero-dependency tests for lazy Transformers model loading."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from langres.core.matchers.transformers_backend import TransformersBackend
from langres.core.model_ref import ModelRef


def test_loader_preserves_hf_revision_and_runtime(monkeypatch) -> None:
    calls: dict[str, dict[str, Any]] = {}

    class _LoadedModel:
        def eval(self) -> None:
            pass

        def to(self, device: str) -> "_LoadedModel":
            calls["device"] = {"value": device}
            return self

    class _Tokenizer:
        pad_token_id = 0

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model: str, **kwargs: Any) -> _Tokenizer:
            calls["tokenizer"] = {"model": model, **kwargs}
            return _Tokenizer()

    class _AutoModel:
        @staticmethod
        def from_pretrained(model: str, **kwargs: Any) -> _LoadedModel:
            calls["model"] = {"model": model, **kwargs}
            return _LoadedModel()

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = _AutoTokenizer  # type: ignore[attr-defined]
    transformers.AutoModelForCausalLM = _AutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    backend = TransformersBackend(
        ModelRef(base="org/model", kind="hf", revision="abc123"),
        device="cpu",
        local_files_only=True,
    )

    backend._ensure_loaded()

    assert calls["tokenizer"]["revision"] == "abc123"
    assert calls["tokenizer"]["local_files_only"] is True
    assert calls["tokenizer"]["trust_remote_code"] is False
    assert calls["model"]["revision"] == "abc123"
    assert calls["model"]["local_files_only"] is True
    assert calls["model"]["trust_remote_code"] is False
    assert calls["device"]["value"] == "cpu"
