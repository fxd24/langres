"""Component and schema registry for serializable Resolvers.

A serialized Resolver references its components and schemas by string name (so
``resolver.json`` can say ``"type_name": "all_pairs_blocker"`` or
``"schema": "CompanySchema"``). This module provides the registration and
lookup machinery plus the typed errors that lookups raise.

Two namespaces:

- **Components** (``register`` / ``get_component``): blockers, comparators,
  scorer modules, clusterers — anything the Resolver composes.
- **Schemas** (``register_schema`` / ``get_schema``): Pydantic entity schemas
  referenced by name in a config.

No abstract base classes live here — only registration, lookup, and errors.
"""

import difflib
import importlib
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T")

_COMPONENT_REGISTRY: dict[str, type] = {}
_SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {}

# Lazy-registration map: ``type_name -> module path``. A component listed here is
# NOT eager-imported by ``langres.core`` — importing it would pull a heavy or
# side-effectful optional dependency into plain ``import langres.core``. Instead
# its module is imported on demand the first time :func:`get_component` is asked
# for that ``type_name`` (the import fires the module's ``@register`` decorator),
# so a fresh process doing ``Resolver.load`` on such an artifact still resolves
# the type. Keep in sync with any module kept off the eager-import path — e.g.
# ``dspy_judge``, which would otherwise import ``dspy`` (and open its disk cache)
# on plain ``import langres.core``.
_LAZY_COMPONENT_MODULES: dict[str, str] = {
    "dspy_judge": "langres.core.modules.dspy_judge",
}


class UnknownComponentType(KeyError):
    """Raised when a component ``type_name`` is not registered.

    Carries the available type names and a ``difflib`` did-you-mean suggestion
    so a typo in a config produces an actionable error.
    """


class SchemaNotRegistered(KeyError):
    """Raised when a schema name referenced by a config is not registered."""


def register(type_name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator: register a component under ``type_name``.

    The decorator is identity-typed (``type[T] -> type[T]``) so ``mypy --strict``
    sees the decorated class unchanged.

    Args:
        type_name: Unique registry key for the component.

    Raises:
        ValueError: If ``type_name`` is already registered.
    """

    def decorator(cls: type[T]) -> type[T]:
        if type_name in _COMPONENT_REGISTRY:
            raise ValueError(f"Component type '{type_name}' is already registered")
        _COMPONENT_REGISTRY[type_name] = cls
        return cls

    return decorator


def get_component(type_name: str) -> type:
    """Look up a registered component class by name.

    Raises:
        UnknownComponentType: If ``type_name`` is not registered. The message
            lists available types and a did-you-mean suggestion.
    """
    if type_name not in _COMPONENT_REGISTRY and type_name in _LAZY_COMPONENT_MODULES:
        # Import the owning module on demand — its ``@register`` populates the
        # registry — so an optional-dependency component (e.g. ``dspy_judge``)
        # resolves without being eager-imported by ``langres.core``.
        importlib.import_module(_LAZY_COMPONENT_MODULES[type_name])
    if type_name not in _COMPONENT_REGISTRY:
        available = sorted(_COMPONENT_REGISTRY)
        suggestions = difflib.get_close_matches(type_name, available, n=3)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise UnknownComponentType(
            f"Unknown component type '{type_name}'.{hint} "
            f"Available types: {', '.join(available) or '(none registered)'}"
        )
    return _COMPONENT_REGISTRY[type_name]


def register_schema(name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator: register a Pydantic schema under ``name``.

    Args:
        name: Unique registry key for the schema (e.g. ``"CompanySchema"``).

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def decorator(cls: type[T]) -> type[T]:
        if name in _SCHEMA_REGISTRY:
            raise ValueError(f"Schema '{name}' is already registered")
        # `cls` is a Pydantic model subclass at use sites; store as such.
        _SCHEMA_REGISTRY[name] = cls  # type: ignore[assignment]
        return cls

    return decorator


def get_schema(name: str) -> type[BaseModel]:
    """Look up a registered Pydantic schema by name.

    Raises:
        SchemaNotRegistered: If ``name`` is not registered.
    """
    if name not in _SCHEMA_REGISTRY:
        available = sorted(_SCHEMA_REGISTRY)
        raise SchemaNotRegistered(
            f"Schema '{name}' is not registered. "
            f"Available schemas: {', '.join(available) or '(none registered)'}"
        )
    return _SCHEMA_REGISTRY[name]
