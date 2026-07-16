"""The single method/matcher-name registry (``core.method_registry``).

See ``langres.core._exports`` for the fragment contract.
"""

from langres.core.method_registry import (
    DEFAULT_EMBEDDING_MODEL,
    MethodSpec,
    UnknownMethodError,
    get_method,
    list_methods,
    register_method,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "get_method",
    "list_methods",
    "MethodSpec",
    "register_method",
    "UnknownMethodError",
]

LAZY_SYMBOLS: dict[str, str] = {}
EXTRA_BY_SYMBOL: dict[str, str] = {}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SYMBOLS)
