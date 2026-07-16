"""The ``Resolver`` and the save/load seam it serializes through.

See ``langres.core._exports`` for the fragment contract.
"""

from langres.core.registry import (
    SchemaNotRegistered,
    UnknownComponentType,
    get_component,
    get_schema,
    register,
    register_schema,
)
from langres.core.resolver import Resolver
from langres.core.serialization import (
    ARTIFACT_VERSION,
    ArtifactManifest,
    ComponentSpec,
    SerializableState,
)

__all__ = [
    "ARTIFACT_VERSION",
    "ArtifactManifest",
    "ComponentSpec",
    "get_component",
    "get_schema",
    "register",
    "register_schema",
    "Resolver",
    "SchemaNotRegistered",
    "SerializableState",
    "UnknownComponentType",
]

LAZY_SYMBOLS: dict[str, str] = {}
EXTRA_BY_SYMBOL: dict[str, str] = {}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SYMBOLS)
