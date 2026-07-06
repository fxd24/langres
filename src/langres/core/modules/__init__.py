"""Module implementations for entity comparison.

``LLMJudge``/``LLMJudgeModule`` need litellm (the ``[llm]`` extra) and
``CascadeModule`` additionally needs sentence-transformers (``[semantic]``) --
resolved lazily via PEP 562 so importing this package (a side effect of e.g.
``from langres.core.modules.rapidfuzz import RapidfuzzModule``) never pulls
those in for a caller who only wants the light module. See
``langres.core``'s module docstring for the W0.4 rationale.
"""

import importlib
from typing import TYPE_CHECKING, Any

from langres.core.modules.cascade_judge import CascadeJudge
from langres.core.modules.rapidfuzz import RapidfuzzModule

if TYPE_CHECKING:
    from langres.core.modules.cascade import CascadeModule
    from langres.core.modules.llm_judge import LLMJudge, LLMJudgeModule

__all__ = ["CascadeJudge", "RapidfuzzModule", "LLMJudge", "LLMJudgeModule", "CascadeModule"]

_LAZY: dict[str, tuple[str, str]] = {
    "LLMJudge": ("langres.core.modules.llm_judge", "pip install 'langres[llm]'"),
    "LLMJudgeModule": ("langres.core.modules.llm_judge", "pip install 'langres[llm]'"),
    "CascadeModule": (
        "langres.core.modules.cascade",
        "pip install 'langres[llm,semantic]'",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, install_hint = _LAZY[name]
    try:
        value = getattr(importlib.import_module(module_path), name)
    except ImportError as exc:
        raise ImportError(
            f"langres.core.modules.{name} requires an optional dependency: {install_hint}"
        ) from exc
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value
