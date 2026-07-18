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


def test_loader_pins_adapter_revision_and_offline_policy(monkeypatch) -> None:
    calls: dict[str, dict[str, Any]] = {}

    class _LoadedModel:
        def eval(self) -> None:
            pass

    class _Tokenizer:
        pad_token_id = 0

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model: str, **kwargs: Any) -> _Tokenizer:
            return _Tokenizer()

    class _AutoModel:
        @staticmethod
        def from_pretrained(model: str, **kwargs: Any) -> _LoadedModel:
            return _LoadedModel()

    class _PeftModel:
        @staticmethod
        def from_pretrained(
            model: _LoadedModel,
            adapter: str,
            **kwargs: Any,
        ) -> _LoadedModel:
            calls["adapter"] = {"adapter": adapter, **kwargs}
            return model

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = _AutoTokenizer  # type: ignore[attr-defined]
    transformers.AutoModelForCausalLM = _AutoModel  # type: ignore[attr-defined]
    peft = ModuleType("peft")
    peft.PeftModel = _PeftModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)
    backend = TransformersBackend(
        ModelRef(
            base="org/model",
            kind="hf",
            revision="base-sha",
            adapter="org/adapter",
            adapter_revision="adapter-sha",
        ),
        local_files_only=True,
    )

    backend._ensure_loaded()

    assert calls["adapter"] == {
        "adapter": "org/adapter",
        "revision": "adapter-sha",
        "local_files_only": True,
    }
