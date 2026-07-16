"""``ModelRef``: normalize a model reference for the serve / finetune paths.

A *model reference* names WHICH model an :class:`~langres.core.matchers.llm_judge.LLMMatcher`
(and, later, ``finetune()``) should run, in one of three forms:

- an **HF Hub id** (``"your-org/your-ft-model"``) or a **local directory path** —
  a self-contained (base or already-merged) model, and
- a **base + PEFT adapter** pair (``{"base": ..., "adapter": ...}``) — a QLoRA
  fine-tune served *without* merging the adapter into the base weights.

It is **weightless by construction**: a ``ModelRef`` is a small pair of
*reference strings*, never weight bytes, so it round-trips through
``Resolver.save`` / ``load`` as plain JSON config (:func:`to_config`). This
module is deliberately import-light (stdlib only — no torch / transformers) so a
bare ``import langres`` and the matcher's own import stay heavy-dependency free;
the actual weights are loaded lazily by the in-process backend on first use.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRef:
    """A normalized, weightless reference to a served/local model.

    ``base`` is the HF Hub id or local directory of the (base or merged) model;
    ``adapter`` is an optional PEFT-adapter HF id / local dir applied on top of
    ``base`` at load time (QLoRA served unmerged). ``adapter is None`` is the
    self-contained case.
    """

    base: str
    adapter: str | None = None


def normalize_model_ref(model: str | dict[str, str] | ModelRef) -> ModelRef:
    """Coerce a user-supplied model reference into a :class:`ModelRef`.

    Accepts the three surface forms and validates them:

    - ``str`` — an HF Hub id, a local dir, OR an API model name (e.g.
      ``"gpt-5-mini"``, ``"openrouter/..."``). All become ``ModelRef(base=model)``;
      the *routing* decision (in-process vs served API) is the matcher's, not
      this normalizer's — here we only carve out a canonical shape.
    - ``dict`` — must carry a non-empty ``"base"``; ``"adapter"`` is optional.
    - :class:`ModelRef` — returned unchanged (idempotent).

    Raises:
        ValueError: An empty string, a dict missing/with an empty ``"base"``, or a
            non-string ``"adapter"``.
        TypeError: Any other type.
    """
    if isinstance(model, ModelRef):
        return model
    if isinstance(model, str):
        if not model:
            raise ValueError("model string must be non-empty")
        return ModelRef(base=model)
    if isinstance(model, dict):
        base = model.get("base")
        if not isinstance(base, str) or not base:
            raise ValueError(f"model dict must carry a non-empty string 'base'; got {model!r}")
        adapter = model.get("adapter")
        if adapter is not None and not isinstance(adapter, str):
            raise ValueError(f"model dict 'adapter' must be a string or absent; got {adapter!r}")
        return ModelRef(base=base, adapter=adapter)
    raise TypeError(f"model must be a str, dict, or ModelRef; got {type(model).__name__}")


def to_config(ref: ModelRef) -> str | dict[str, str]:
    """Serialize a :class:`ModelRef` for ``config`` (the inverse of :func:`normalize_model_ref`).

    A self-contained ref (``adapter is None``) serializes to its bare ``base``
    string — **byte-identical to the pre-model_ref string config**, so existing
    saved artifacts and their round-trips are unchanged. Only the base+adapter
    case widens to a ``{"base": ..., "adapter": ...}`` dict.
    """
    if ref.adapter is None:
        return ref.base
    return {"base": ref.base, "adapter": ref.adapter}
