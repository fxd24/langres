"""Component and schema registry for serializable Resolvers.

A serialized Resolver references its components and schemas by string name (so
``resolver.json`` can say ``"type_name": "all_pairs_blocker"`` or
``"schema": "CompanySchema"``). This module provides the registration and
lookup machinery plus the typed errors that lookups raise.

Three namespaces:

- **Components** (``register`` / ``get_component``): blockers, comparators,
  scorer modules, clusterers — anything the Resolver composes.
- **Schemas** (``register_schema`` / ``get_schema``): Pydantic entity schemas
  referenced by name in a config.
- **Models** (``register_model`` / ``get_model``): Resolver *subclasses* — the
  named architectures — so ``save``/``load`` can round-trip which class a
  pipeline is, not just which parts it has.

Why models are their own namespace and not just more components: a component
*fills a slot* in a Resolver; a model *owns* the slots. Sharing one dict would
make ``"type_name": "fuzzy_string"`` resolvable in a blocker slot — a nonsense
config the registry would happily construct — and would cross-contaminate the
did-you-mean suggestions of both. The precedent is already here: schemas got
their own namespace for exactly this reason (a schema is not a component), so a
third one is the established pattern rather than a parallel registry.

No abstract base classes live here — only registration, lookup, and errors.
"""

import difflib
import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T")

_COMPONENT_REGISTRY: dict[str, type] = {}
_SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {}
_MODEL_REGISTRY: dict[str, type] = {}
_OP_SERIALIZERS_BY_ROLE: dict[str, "OpSerializer"] = {}
_OP_SERIALIZERS_BY_TYPE: dict[type, "OpSerializer"] = {}

# Lazy-registration map: ``type_name -> module path``. A component listed here is
# NOT eager-imported by ``langres.core`` — importing it would pull a heavy or
# side-effectful optional dependency into plain ``import langres.core``. Instead
# its module is imported on demand the first time :func:`get_component` is asked
# for that ``type_name`` (the import fires the module's ``@register`` decorator),
# so a fresh process doing ``Resolver.load`` on such an artifact still resolves
# the type. Keep in sync with any module kept off the eager-import path — e.g.
# ``dspy_judge``, which would otherwise import ``dspy`` (and open its disk cache)
# on plain ``import langres.core``.
#
# ``key_blocker``/``composite_blocker``/``correlation_clusterer``/
# ``fellegi_sunter_judge``/``random_forest`` were added to this map so a saved
# artifact referencing these types keeps resolving once W0.4's lazy-import
# refactor trimmed ``core/__init__.py``'s eager imports -- the same safety net
# ``select_judge``/``dspy_judge`` already relied on.
#
# W0.4 (extras split): ``llm_judge``/``vector_blocker``/``faiss_index``/
# ``sentence_transformer_embedder``/``fake_embedder`` joined this map when
# ``langres.core.__init__`` stopped eager-importing litellm (``llm_judge``) and
# faiss/sentence-transformers (``vector_blocker``, ``faiss_index``,
# ``*_embedder``) — those packages are now optional (``pip install
# langres[llm]`` / ``langres[semantic]``), so importing them must be deferred
# to the first actual access, exactly like ``dspy_judge``.
#
# ``calibrator`` (the Platt/isotonic ``Calibrator``) lives in
# ``langres.training.calibration``, which imports scikit-learn at module scope (the
# ``[trained]`` extra) -- so a saved Resolver carrying a fitted calibrator resolves
# its ``type_name`` here without ``calibration`` being on the eager-import path.
#
# ``comparator`` (the rapidfuzz ``StringComparator``) joined this map in W1, when it
# was split out of ``langres.core.comparator`` into ``langres.core.comparators.string``
# so the ABC could stop importing its own implementation. Nothing on the eager path
# imports the impl module any more, so without this entry a saved Resolver's
# ``"type_name": "comparator"`` would stop resolving in a fresh process. rapidfuzz is a
# core dep, so this is a layering deferral, not a dep deferral.
#
# ``cascade_judge`` is pure-core (no heavy deps) and IS eager-imported today, so
# this entry is redundant *right now* — it is kept deliberately as the same
# saved-artifact safety net as its ``random_forest``/``correlation_clusterer`` peers
# above: a ``CascadeMatcher`` wrapping a fitted student is exactly what lands in a
# saved ``Resolver`` artifact, so its ``type_name`` must keep resolving even if a
# future eager-import trim (like W0.4's) drops it from ``core/__init__.py``. It
# is here for parity with those peers, not because it needs dep deferral.
_LAZY_COMPONENT_MODULES: dict[str, str] = {
    "calibrator": "langres.training.calibration",
    "cascade_judge": "langres.core.matchers.cascade_judge",
    "comparator": "langres.core.comparators.string",
    "composite_blocker": "langres.core.blockers.composite",
    "correlation_clusterer": "langres.core.clusterers.correlation",
    "dspy_judge": "langres.core.matchers.dspy_judge",
    "faiss_index": "langres.core.indexes.vector_index",
    "fake_embedder": "langres.core.embeddings",
    "fastembed_late_interaction_embedder": "langres.core.embeddings",
    "fastembed_sparse_embedder": "langres.core.embeddings",
    "fellegi_sunter_judge": "langres.core.matchers.fellegi_sunter",
    "key_blocker": "langres.core.blockers.key",
    "llm_judge": "langres.core.matchers.llm_judge",
    "random_forest": "langres.core.matchers.random_forest_judge",
    "select_judge": "langres.core.matchers.select_judge",
    "sentence_transformer_embedder": "langres.core.embeddings",
    "vector_blocker": "langres.core.blockers.vector",
}


class UnknownComponentType(KeyError):
    """Raised when a component ``type_name`` is not registered.

    Carries the available type names and a ``difflib`` did-you-mean suggestion
    so a typo in a config produces an actionable error.
    """


class SchemaNotRegistered(KeyError):
    """Raised when a schema name referenced by a config is not registered."""


class UnknownModelType(KeyError):
    """Raised when an artifact's ``model_class`` is not a registered model.

    Carries the available model names and a ``difflib`` did-you-mean suggestion,
    like :class:`UnknownComponentType`. The usual cause is loading an artifact
    whose architecture lives in a module the process never imported.
    """


class UnknownOpType(KeyError):
    """Raised when an explicit-chain Op role or class has no safe serializer."""


@dataclass(frozen=True)
class OpSerializer:
    """One explicitly registered, fail-closed serializer for an Op class.

    ``dump`` returns the scalar params and optional nested legacy component
    carried by the stage. ``load`` reverses that data. The registry stores exact
    classes: a subclass must register itself rather than silently losing state
    through its parent's serializer.
    """

    role: str
    op_type: type
    dump: Callable[[object], tuple[dict[str, object], object | None]]
    load: Callable[[dict[str, object], object | None, Path], object]
    component_slot: str | None = None
    state_owner: Callable[[object], object | None] | None = None


def register_op_serializer(serializer: OpSerializer) -> None:
    """Register one trusted Op serializer by role and exact class.

    Registration is process-local and never imports code named by an artifact.
    Loading can therefore construct only serializers the application already
    imported and explicitly registered.
    """
    if serializer.role in _OP_SERIALIZERS_BY_ROLE:
        raise ValueError(f"Op role '{serializer.role}' is already registered")
    if serializer.op_type in _OP_SERIALIZERS_BY_TYPE:
        raise ValueError(
            f"Op type '{serializer.op_type.__name__}' already has a registered serializer"
        )
    _OP_SERIALIZERS_BY_ROLE[serializer.role] = serializer
    _OP_SERIALIZERS_BY_TYPE[serializer.op_type] = serializer


def get_op_serializer(role: str) -> OpSerializer:
    """Return the already-registered serializer for ``role``.

    Unlike component lookup, this has no lazy module import map: an artifact can
    never cause downloaded or otherwise untrusted Python to be imported.
    """
    try:
        return _OP_SERIALIZERS_BY_ROLE[role]
    except KeyError:
        available = sorted(_OP_SERIALIZERS_BY_ROLE)
        suggestions = difflib.get_close_matches(role, available, n=3)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise UnknownOpType(
            f"Unknown OpSpec role '{role}'.{hint} Registered roles: "
            f"{', '.join(available) or '(none registered)'}. Import and register the "
            "trusted Op implementation before loading this artifact."
        ) from None


def op_serializer_for_type(op_type: type) -> OpSerializer:
    """Return the serializer registered for the exact ``op_type``."""
    try:
        return _OP_SERIALIZERS_BY_TYPE[op_type]
    except KeyError:
        raise UnknownOpType(
            f"Op type '{op_type.__name__}' is not serializable. Register that exact class "
            "with @register_op(...) or register_op_serializer(...); parent-class serializers "
            "are deliberately not inherited because that could drop subclass state."
        ) from None


def _registered_op_config(op: object) -> dict[str, object]:
    """Read a custom registered Op's ``config`` property/method as plain data."""
    config = op.config() if callable(getattr(type(op), "config", None)) else op.config  # type: ignore[attr-defined]
    if isinstance(config, BaseModel):
        return config.model_dump()
    return dict(config)


def register_op(role: str) -> Callable[[type[T]], type[T]]:
    """Register a safe, component-free custom Op for artifact round-trips.

    The class must expose a JSON-compatible ``config`` property (or method) and
    ``from_config(config)`` classmethod, mirroring registered model components.
    Stateful Ops may additionally implement ``SerializableState``; the artifact
    layer detects that capability without broadening this leaf's dependencies.
    """

    def decorator(cls: type[T]) -> type[T]:
        if not callable(getattr(cls, "from_config", None)):
            raise TypeError(f"@register_op({role!r}) requires {cls.__name__}.from_config(config)")
        config_member = inspect.getattr_static(cls, "config", None)
        if config_member is None:
            raise TypeError(f"@register_op({role!r}) requires {cls.__name__}.config")

        def dump(op: object) -> tuple[dict[str, object], object | None]:
            return _registered_op_config(op), None

        def load(params: dict[str, object], component: object | None, _state_dir: Path) -> object:
            if component is not None:
                raise ValueError(
                    f"custom Op role {role!r} is component-free but its spec carries a component"
                )
            return cls.from_config(params)  # type: ignore[attr-defined]

        register_op_serializer(OpSerializer(role=role, op_type=cls, dump=dump, load=load))
        return cls

    return decorator


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


def register_model(type_name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator: register a Resolver subclass (an architecture) under ``type_name``.

    Registering is what makes a model's identity survive ``save``/``load``:
    :meth:`Resolver.save` records the registered name in the manifest's
    ``model_class`` and :meth:`Resolver.load` reconstructs *that* class. An
    unregistered subclass is not an error — it simply saves without a
    ``model_class`` and reloads as a plain ``Resolver``, exactly as every
    Resolver did before this field existed.

    Args:
        type_name: Unique registry key for the model (e.g. ``"fuzzy_string"``).

    Raises:
        ValueError: If ``type_name`` is already registered.
    """

    def decorator(cls: type[T]) -> type[T]:
        if type_name in _MODEL_REGISTRY:
            raise ValueError(f"Model type '{type_name}' is already registered")
        _MODEL_REGISTRY[type_name] = cls
        return cls

    return decorator


def get_model(type_name: str) -> type:
    """Look up a registered model (Resolver subclass) by name.

    Returns a bare ``type`` rather than ``type[Resolver]`` deliberately: this
    module sits *beneath* ``core.resolver`` (which imports it), so naming
    ``Resolver`` here — even under ``TYPE_CHECKING`` — would knot the two into an
    import cycle. ``tests/test_import_tangle.py`` counts that edge; ``get_component``
    returns a bare ``type`` for the same reason.

    .. note::
       There is no lazy ``type_name -> module`` map for models yet (contrast
       :data:`_LAZY_COMPONENT_MODULES`), because no model is registered anywhere
       yet — W4 lands the architectures. If W4 puts them off the eager-import
       path, it needs the same saved-artifact safety net, or a fresh process
       loading such an artifact will raise :class:`UnknownModelType` here.

    Raises:
        UnknownModelType: If ``type_name`` is not registered. The message lists
            the available models and a did-you-mean suggestion.
    """
    if type_name not in _MODEL_REGISTRY:
        available = sorted(_MODEL_REGISTRY)
        suggestions = difflib.get_close_matches(type_name, available, n=3)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise UnknownModelType(
            f"Unknown model type '{type_name}'.{hint} It may live in a module this "
            f"process never imported. Available models: "
            f"{', '.join(available) or '(none registered)'}"
        )
    return _MODEL_REGISTRY[type_name]


def model_type_name(cls: type) -> str | None:
    """Return the registered name for a model class, or ``None`` if unregistered.

    The reverse of :func:`register_model`, used by ``save`` to stamp the
    manifest. Matches ``cls`` **exactly** rather than walking the MRO: an
    unregistered subclass of a registered architecture is its own thing, and
    claiming its parent's name would make ``load`` hand back the wrong class.
    """
    for name, registered in _MODEL_REGISTRY.items():
        if registered is cls:
            return name
    return None


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
