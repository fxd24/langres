"""Served and in-process LLM resources behind one generation contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langres.core.model_ref import (
    IN_PROCESS_KINDS,
    ModelRef,
    UnsupportedBackboneError,
    normalize_model_ref,
    require_litellm_routable,
)
from langres.resources.base import (
    GenerationBatch,
    GenerationEnvelope,
    GenerationRequest,
    GenerationUsage,
    LLM,
    LLMRuntimeConfig,
)


def _content(response: Any) -> str:
    return str(response.choices[0].message.content or "")


def _finish_reason(response: Any) -> str | None:
    value = getattr(response.choices[0], "finish_reason", None)
    return str(value) if value is not None else None


class LiteLLM:
    """Lazy LiteLLM-backed generation resource for API/endpoint refs."""

    def __init__(
        self,
        model: str | dict[str, str] | ModelRef,
        *,
        runtime_config: LLMRuntimeConfig | None = None,
        client: Any = None,
    ) -> None:
        ref = require_litellm_routable(normalize_model_ref(model), slot="LiteLLM")
        if ref.kind == "hf" and ref.revision is not None:
            raise UnsupportedBackboneError(
                "LiteLLM cannot honor a Hugging Face revision. Use TransformersLLM "
                "for a pinned in-process Hub model, or serve that revision and pass "
                "an API/endpoint ModelRef whose version is part of its model id."
            )
        self.model_ref = ref
        self.runtime_config = runtime_config or LLMRuntimeConfig()
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            import litellm

            self._client = litellm
        return self._client

    def generate(self, requests: Sequence[GenerationRequest]) -> GenerationBatch:
        """Generate in request order and retain raw content only locally."""
        client = self._get_client()
        outputs: list[GenerationEnvelope] = []
        for request in requests:
            kwargs: dict[str, Any] = {
                "model": self.model_ref.base,
                "messages": request.message_dicts(),
                "temperature": self.runtime_config.temperature,
                "max_tokens": self.runtime_config.max_new_tokens,
            }
            if self.model_ref.api_base is not None:
                kwargs["api_base"] = self.model_ref.api_base
            if self.runtime_config.timeout_seconds is not None:
                kwargs["timeout"] = self.runtime_config.timeout_seconds
            response = client.completion(**kwargs)
            usage = GenerationUsage.from_response(response, model=self.model_ref.base)
            cost_usd: float | None = None
            try:
                cost_usd = float(client.completion_cost(completion_response=response))
            except (AttributeError, TypeError, ValueError):
                pass
            outputs.append(
                GenerationEnvelope.from_content(
                    request_id=request.request_id,
                    model_ref=self.model_ref,
                    content=_content(response),
                    usage=usage,
                    finish_reason=_finish_reason(response),
                    cost_usd=cost_usd,
                    cost_basis="estimated" if cost_usd is not None else "none",
                )
            )
        return GenerationBatch(outputs=tuple(outputs), model_ref=self.model_ref)


class TransformersLLM:
    """Lazy local/Hugging Face causal-LM generation resource."""

    def __init__(
        self,
        model: str | dict[str, str] | ModelRef,
        *,
        runtime_config: LLMRuntimeConfig | None = None,
    ) -> None:
        self.model_ref = normalize_model_ref(model)
        if self.model_ref.kind not in IN_PROCESS_KINDS:
            raise UnsupportedBackboneError(
                "TransformersLLM requires kind='hf' or kind='local'. "
                "Use LiteLLM for API/endpoint references."
            )
        self.runtime_config = runtime_config or LLMRuntimeConfig()
        self._backend: Any = None

    def _get_backend(self) -> Any:
        if self._backend is None:
            from langres.core.matchers.transformers_backend import TransformersBackend

            self._backend = TransformersBackend(
                self.model_ref,
                max_new_tokens=self.runtime_config.max_new_tokens,
                device=self.runtime_config.device,
                dtype=self.runtime_config.dtype,
                local_files_only=self.runtime_config.local_files_only,
            )
        return self._backend

    def generate(self, requests: Sequence[GenerationRequest]) -> GenerationBatch:
        """Generate locally through the existing LiteLLM-shaped backend."""
        backend = self._get_backend()
        outputs = []
        for request in requests:
            response = backend.complete(
                request.message_dicts(),
                temperature=self.runtime_config.temperature,
                want_logprobs=False,
            )
            outputs.append(
                GenerationEnvelope.from_content(
                    request_id=request.request_id,
                    model_ref=self.model_ref,
                    content=_content(response),
                    usage=GenerationUsage.from_response(
                        response,
                        model=self.model_ref.base,
                    ),
                    finish_reason=_finish_reason(response),
                )
            )
        return GenerationBatch(outputs=tuple(outputs), model_ref=self.model_ref)


def llm_from_model_ref(
    model: str | dict[str, str] | ModelRef,
    *,
    runtime_config: LLMRuntimeConfig | None = None,
) -> LLM:
    """Route a typed ref to the corresponding generation resource."""
    ref = normalize_model_ref(model)
    if ref.kind in IN_PROCESS_KINDS:
        return TransformersLLM(ref, runtime_config=runtime_config)
    return LiteLLM(ref, runtime_config=runtime_config)
