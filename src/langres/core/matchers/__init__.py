"""Matcher implementations for entity comparison.

``LLMMatcher``/``LLMMatcher`` need litellm (the ``[llm]`` extra) and
``CascadeChainMatcher`` additionally needs sentence-transformers (``[semantic]``) --
resolved lazily via PEP 562 so importing this package (a side effect of e.g.
``from langres.core.matchers.rapidfuzz import RapidfuzzMatcher``) never pulls
those in for a caller who only wants the light module. See
``langres.core``'s module docstring for the W0.4 rationale.
"""

import importlib
from typing import TYPE_CHECKING, Any

from langres.core.matchers.cascade_judge import CascadeMatcher
from langres.core.matchers.rapidfuzz import RapidfuzzMatcher

if TYPE_CHECKING:
    from langres.core.matchers.cascade import CascadeChainMatcher
    from langres.core.matchers.llm_judge import LLMMatcher, LLMMatcher

__all__ = ["CascadeMatcher", "RapidfuzzMatcher", "LLMMatcher", "LLMMatcher", "CascadeChainMatcher"]

_LAZY: dict[str, tuple[str, str]] = {
    "LLMMatcher": ("langres.core.matchers.llm_judge", "pip install 'langres[llm]'"),
    "LLMMatcher": ("langres.core.matchers.llm_judge", "pip install 'langres[llm]'"),
    "CascadeChainMatcher": (
        "langres.core.matchers.cascade",
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
            f"langres.core.matchers.{name} requires an optional dependency: {install_hint}"
        ) from exc
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value
