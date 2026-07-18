"""Served and in-process LLM resources behind one generation contract."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from langres.clients.openrouter import parse_openrouter_billing
from langres.core.model_ref import (
    IN_PROCESS_KINDS,
    ModelRef,
    UnsupportedBackboneError,
    normalize_model_ref,
    require_litellm_routable,
    to_config,
)
from langres.core.registry import register
from langres.resources._model_ref import normalize_inprocess_ref
from langres.resources.base import (
    GenerationBatch,
    GenerationEnvelope,
    GenerationRequest,
    GenerationUsage,
    LLM,
    LLMRuntimeConfig,
    require_unique_ids,
)

logger = logging.getLogger(__name__)


class _LiteLLMOptions(BaseModel):
    """Strict JSON-only options safe to persist and replay."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )

    provider: dict[str, JsonValue] | None = None
    extra_body: dict[str, JsonValue] | None = None


def _content(response: Any) -> str:
    return str(response.choices[0].message.content or "")


def _finish_reason(response: Any) -> str | None:
    value = getattr(response.choices[0], "finish_reason", None)
    return str(value) if value is not None else None


def _response_string(response: Any, field: str) -> str | None:
    value = getattr(response, field, None)
    return str(value) if value is not None else None


def _nonnegative_cost(value: Any) -> float | None:
    """Normalize a usable USD observation; malformed pricing stays unknown."""
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None
    return cost if math.isfinite(cost) and cost >= 0.0 else None


@register("resource_litellm")
class LiteLLM:
    """Lazy LiteLLM-backed generation resource for API/endpoint refs."""

    type_name: ClassVar[str] = "resource_litellm"
    requires_cost_accounting: ClassVar[bool] = True

    def __init__(
        self,
        model: str | dict[str, str] | ModelRef,
        *,
        runtime_config: LLMRuntimeConfig | None = None,
        client: Any = None,
        provider: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        ref = require_litellm_routable(normalize_model_ref(model), slot="LiteLLM")
        if ref.kind == "hf" and ref.revision is not None:
            raise UnsupportedBackboneError(
                "LiteLLM cannot honor a Hugging Face revision. Use TransformersLLM "
                "for a pinned in-process Hub model, or serve that revision and pass "
                "an API/endpoint ModelRef whose version is part of its model id."
            )
        options = _LiteLLMOptions(provider=provider, extra_body=extra_body)
        self.model_ref = ref
        self.runtime_config = runtime_config or LLMRuntimeConfig()
        self.provider = deepcopy(options.provider)
        self.extra_body = deepcopy(options.extra_body)
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            import litellm

            self._client = litellm
        return self._client

    @property
    def config(self) -> dict[str, object]:
        """Weightless config; injected clients and credentials never persist."""
        return {
            "model": to_config(self.model_ref),
            "runtime_config": self.runtime_config.model_dump(mode="json"),
            "provider": deepcopy(self.provider),
            "extra_body": deepcopy(self.extra_body),
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "LiteLLM":
        """Rebuild lazily without retaining an injected client."""
        return cls(
            config["model"],  # type: ignore[arg-type]
            runtime_config=LLMRuntimeConfig.model_validate(config["runtime_config"]),
            provider=config.get("provider"),  # type: ignore[arg-type]
            extra_body=config.get("extra_body"),  # type: ignore[arg-type]
        )

    def _completion_extras(self, request: GenerationRequest) -> dict[str, Any]:
        """Return merged OpenRouter accounting/routing and run-correlation args."""
        extras: dict[str, Any] = {}
        extra_body = deepcopy(self.extra_body)
        if self.model_ref.base.startswith("openrouter/"):
            extra_body = extra_body or {}
            supplied_usage = extra_body.get("usage")
            usage = dict(supplied_usage) if isinstance(supplied_usage, dict) else {}
            usage["include"] = True
            extra_body["usage"] = usage
            if self.provider is not None:
                extra_body["provider"] = deepcopy(self.provider)
        if extra_body is not None:
            extras["extra_body"] = extra_body

        # Reuse the tracking ContextVar seam used by the legacy LLMMatcher.
        # LiteLLMResource accepts only LiteLLM-shaped clients, unlike the legacy
        # matcher's broader direct-OpenAI injection seam, so metadata is valid
        # for injected test/router clients too.
        from langres.tracking.runs import current_run

        attempt_id = current_run.get()
        if attempt_id is not None:
            extras["metadata"] = {
                "langres_attempt_id": attempt_id,
                "request_id": request.request_id,
                "decision_step": "llm_generation",
            }
        return extras

    def generate(self, requests: Sequence[GenerationRequest]) -> GenerationBatch:
        """Generate in request order and retain raw content only locally."""
        require_unique_ids(
            [request.request_id for request in requests],
            field="request_ids",
            operation="LiteLLM.generate",
        )
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
            if self.runtime_config.seed is not None:
                kwargs["seed"] = self.runtime_config.seed
            kwargs.update(self._completion_extras(request))
            response = client.completion(**kwargs)
            try:
                real_cost, provider = parse_openrouter_billing(response)
            except Exception as exc:
                # Billing/provider extraction is post-call observability. Keep
                # a successful response even if an SDK returns a novel shape.
                logger.warning(
                    "Could not read provider billing for %s: %s",
                    self.model_ref.base,
                    exc,
                )
                real_cost, provider = None, None
            served_model = _response_string(response, "model") or self.model_ref.base
            try:
                usage = GenerationUsage.from_response(
                    response,
                    model=served_model,
                    provider=provider,
                )
            except Exception as exc:
                logger.warning(
                    "Could not read completion usage for %s: %s",
                    self.model_ref.base,
                    exc,
                )
                usage = None
            cost_usd = _nonnegative_cost(real_cost)
            cost_basis: Literal["real", "estimated", "none"] = (
                "real" if cost_usd is not None else "none"
            )
            if cost_usd is None:
                try:
                    cost_usd = _nonnegative_cost(
                        client.completion_cost(completion_response=response)
                    )
                    cost_basis = "estimated" if cost_usd is not None else "none"
                except Exception as exc:
                    # Pricing is observability after a successful paid call. A
                    # broken/missing price table must not discard the response.
                    logger.warning(
                        "Could not estimate completion cost for %s: %s",
                        self.model_ref.base,
                        exc,
                    )
            outputs.append(
                GenerationEnvelope.from_content(
                    request_id=request.request_id,
                    model_ref=self.model_ref,
                    content=_content(response),
                    usage=usage,
                    provider=provider,
                    served_model=served_model,
                    provider_request_id=_response_string(response, "id"),
                    finish_reason=_finish_reason(response),
                    cost_usd=cost_usd,
                    cost_basis=cost_basis,
                )
            )
        return GenerationBatch(outputs=tuple(outputs), model_ref=self.model_ref)


@register("resource_transformers_llm")
class TransformersLLM:
    """Lazy local/Hugging Face causal-LM generation resource."""

    type_name: ClassVar[str] = "resource_transformers_llm"
    requires_cost_accounting: ClassVar[bool] = False

    def __init__(
        self,
        model: str | dict[str, str] | ModelRef,
        *,
        runtime_config: LLMRuntimeConfig | None = None,
    ) -> None:
        self.model_ref = normalize_inprocess_ref(model, slot="TransformersLLM")
        if self.model_ref.kind not in IN_PROCESS_KINDS:
            raise UnsupportedBackboneError(
                "TransformersLLM requires kind='hf' or kind='local'. "
                "Use LiteLLM for API/endpoint references."
            )
        self.runtime_config = runtime_config or LLMRuntimeConfig()
        self._backend: Any = None

    @property
    def config(self) -> dict[str, object]:
        """Weightless local/HF construction config."""
        return {
            "model": to_config(self.model_ref),
            "runtime_config": self.runtime_config.model_dump(mode="json"),
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "TransformersLLM":
        """Rebuild with weights still unloaded."""
        return cls(
            config["model"],  # type: ignore[arg-type]
            runtime_config=LLMRuntimeConfig.model_validate(config["runtime_config"]),
        )

    def _get_backend(self) -> Any:
        if self._backend is None:
            from langres.core.matchers.transformers_backend import TransformersBackend

            self._backend = TransformersBackend(
                self.model_ref,
                max_new_tokens=self.runtime_config.max_new_tokens,
                device=self.runtime_config.device,
                dtype=self.runtime_config.dtype,
                local_files_only=self.runtime_config.local_files_only,
                seed=self.runtime_config.seed,
            )
        return self._backend

    def generate(self, requests: Sequence[GenerationRequest]) -> GenerationBatch:
        """Generate locally through the existing LiteLLM-shaped backend."""
        require_unique_ids(
            [request.request_id for request in requests],
            field="request_ids",
            operation="TransformersLLM.generate",
        )
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
