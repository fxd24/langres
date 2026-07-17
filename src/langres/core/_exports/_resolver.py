"""The ``Resolver`` and the save/load seam it serializes through.

See ``langres.core._exports`` for the fragment contract.
"""

from langres.core.registry import (
    SchemaNotRegistered,
    UnknownComponentType,
    UnknownModelType,
    get_component,
    get_model,
    get_schema,
    model_type_name,
    register,
    register_model,
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
    "get_model",
    "get_schema",
    "model_type_name",
    "register",
    "register_model",
    "register_schema",
    "Resolver",
    "SchemaNotRegistered",
    "SerializableState",
    "UnknownComponentType",
    "UnknownModelType",
]

LAZY_SYMBOLS: dict[str, str] = {}
EXTRA_BY_SYMBOL: dict[str, str] = {}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SYMBOLS)
