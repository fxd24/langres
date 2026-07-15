"""Matcher implementations for entity comparison.

``LLMMatcher`` needs litellm (the ``[llm]`` extra) and ``CascadeChainMatcher``
additionally needs sentence-transformers (``[semantic]``) -- resolved lazily via
PEP 562 so importing this package (a side effect of e.g.
``from langres.core.matchers.rapidfuzz import RapidfuzzMatcher``) never pulls
those in for a caller who only wants the light matcher. The eager scorers
(``WeightedAverageMatcher``, ``EmbeddingScoreMatcher``, ``CascadeMatcher``,
``RapidfuzzMatcher``) fire their ``@register`` decorators on import so a fresh
process can ``Resolver.load`` an artifact using any of them. See
``langres.core``'s module docstring for the W0.4 rationale.
"""

import importlib
from typing import TYPE_CHECKING, Any

from langres.core.matchers.cascade_judge import CascadeMatcher
from langres.core.matchers.embedding_score import EmbeddingScoreMatcher
from langres.core.matchers.rapidfuzz import RapidfuzzMatcher
from langres.core.matchers.weighted_average import WeightedAverageMatcher

if TYPE_CHECKING:
    from langres.core.matchers.cascade import CascadeChainMatcher
    from langres.core.matchers.llm_judge import LLMMatcher

__all__ = [
    "CascadeChainMatcher",
    "CascadeMatcher",
    "EmbeddingScoreMatcher",
    "LLMMatcher",
    "RapidfuzzMatcher",
    "WeightedAverageMatcher",
]

_LAZY: dict[str, tuple[str, str]] = {
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
