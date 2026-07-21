"""Resource-specific ModelRef normalization helpers."""

from __future__ import annotations

from langres.core.model_ref import (
    IN_PROCESS_KINDS,
    ModelRef,
    UnsupportedBackboneError,
    normalize_model_ref,
)

_PATH_PREFIXES = ("./", "../", "/", "~")


def normalize_inprocess_ref(model: str | dict[str, str] | ModelRef, *, slot: str) -> ModelRef:
    """Normalize an HF/local reference and reject served-only forms.

    Bare model names are treated as Hugging Face ids in an in-process resource.
    This is intentionally narrower than the general ``ModelRef`` string
    inference, where a no-slash id denotes a LiteLLM API model.
    """
    if isinstance(model, str):
        kind = "local" if model.startswith(_PATH_PREFIXES) else "hf"
        ref = normalize_model_ref({"base": model, "kind": kind})
    else:
        ref = normalize_model_ref(model)
    if ref.kind not in IN_PROCESS_KINDS:
        raise UnsupportedBackboneError(
            f"{slot} requires an in-process Hugging Face or local model, but got "
            f"kind={ref.kind!r}. Fix: pass ModelRef(kind='hf'/'local'), or use "
            "the served resource for API/endpoint models."
        )
    return ref
