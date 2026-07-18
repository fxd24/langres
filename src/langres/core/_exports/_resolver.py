"""The ``ERModel`` and the save/load seam it serializes through.

Carries ``ERModel`` and its two result types, all three contracts a pipeline is
written *against*. The concrete architectures (``FuzzyString``,
``VectorLLMCascade``) are deliberately NOT here: they are implementations, and
re-exporting one from ``langres.core`` would put the floor above the components
it sits beneath — the rule ``test_import_budget.py``'s
``test_implementations_are_not_re_exported`` enforces. Import those from
``langres.architectures``.

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
from langres.core.resolver import ERModel, Resolver
from langres.core.results import DedupeResult, LinkVerdict
from langres.core.serialization import (
    ARTIFACT_VERSION,
    ArtifactManifest,
    ComponentSpec,
    OpSpec,
    SerializableState,
)

__all__ = [
    "ARTIFACT_VERSION",
    "ArtifactManifest",
    "ComponentSpec",
    "DedupeResult",
    "ERModel",
    "get_component",
    "get_model",
    "get_schema",
    "LinkVerdict",
    "model_type_name",
    "OpSpec",
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
