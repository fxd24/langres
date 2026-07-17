"""Component <-> :class:`ComponentSpec` adapters: the unified serialization convention.

The mechanism behind ``ERModel.save``/``load``, extracted from ``resolver.py`` so
the model layer states *what* it persists (slots, in order) while this leaf owns
*how* a single component crosses the JSON boundary in both directions.

Wave 2 produced two component-config styles, and this module is the entire reason
the model layer never has to know which is which:

- A ``config`` **property** returning a plain ``dict`` (comparator, blockers,
  clusterer, judge).
- A ``config()`` **method** returning a Pydantic ``BaseModel`` plus
  ``type_name`` / ``config_model`` classvars (FAISSIndex, embedders).

langres does not pick one and rewrite the other. Instead every slot is adapted
behind two functions -- :func:`component_spec` (object -> ``ComponentSpec``) and
:func:`rebuild_component` (``ComponentSpec`` -> object) -- so every slot
serializes and reconstructs uniformly. Every model-slot component exposes a
``type_name`` class attribute so the spec adapter can discover its registry key.

Nothing here knows about ``ERModel``: these are free functions over the
:mod:`~langres.core.registry` and :mod:`~langres.core.serialization` contracts,
which is what keeps this a leaf (and keeps it out of the ``resolver`` import
knot). Rebuilding executes **no code and no pickle** -- a component is only ever
reconstructed by its registered ``type_name``.
"""

import inspect
from pathlib import Path
from typing import Any, TypeGuard

from pydantic import BaseModel

from langres.core.registry import get_component
from langres.core.serialization import ComponentSpec, SerializableState


def component_config_dict(obj: object) -> dict[str, object]:
    """Return a component's construction config as a plain JSON-able dict.

    Bridges the two Wave 2 conventions:

    - ``config`` **property** returning a ``dict`` -> returned as-is.
    - ``config()`` **method** returning a Pydantic ``BaseModel`` -> dumped.
    """
    # Inspect ``config`` on the *class* so a property descriptor reads as
    # non-callable while a real method reads as callable. (Checking the
    # resolved value on the instance would misclassify a config stored as a
    # plain instance attribute, e.g. a Pydantic model.)
    config = obj.config() if callable(getattr(type(obj), "config", None)) else obj.config  # type: ignore[attr-defined]
    if isinstance(config, BaseModel):
        return config.model_dump()
    return dict(config)


def component_spec(obj: object, slot: str) -> ComponentSpec:
    """Serialize any model-slot component into a :class:`ComponentSpec`.

    Reads the component's ``type_name`` class attribute (the registry key) and
    its construction config (via :func:`component_config_dict`), and records the
    ``slot`` name so ``ERModel.load`` can map the spec back self-describingly
    rather than by position or hard-coded ``type_name``.
    """
    type_name = getattr(obj, "type_name", None)
    if not isinstance(type_name, str):
        raise TypeError(
            f"{type(obj).__name__} is not serializable (no `type_name`/@register). "
            f"Use a registered component (e.g. LLMMatcher, WeightedAverageMatcher) in "
            f"the {slot!r} slot."
        )
    return ComponentSpec(type_name=type_name, slot=slot, config=component_config_dict(obj))


def state_owner(component: object) -> SerializableState | None:
    """Return the out-of-band-state owner for a slot component, if any.

    Two cases own persistable state in M0:

    - The component itself implements
      :class:`~langres.core.serialization.SerializableState` (e.g. a FAISS index
      used directly).
    - The component wraps a vector index that implements ``SerializableState``
      (e.g. a ``VectorBlocker`` holding a built ``FAISSIndex``). The nested index
      holds the heavy state; the blocker config only references it.

    Returns ``None`` for stateless components (AllPairs, comparator, judge,
    clusterer).
    """
    if isinstance(component, SerializableState):
        return component
    index = getattr(component, "vector_index", None)
    if isinstance(index, SerializableState):
        return index
    return None


def has_state(state_dir: Path | None) -> TypeGuard[Path]:
    """True iff ``state_dir`` exists and holds at least one persisted state file.

    An empty (or absent) sidecar dir signals "no out-of-band state to restore",
    so callers must not invoke ``load_state`` on it — that would try to read a
    missing state file (e.g. ``index.faiss``). Returning a ``TypeGuard`` narrows
    ``state_dir`` to ``Path`` in the truthy branch for the type checker.
    """
    return state_dir is not None and state_dir.is_dir() and any(state_dir.iterdir())


def rebuild_component(spec: ComponentSpec, state_dir: Path | None = None) -> Any:
    """Rebuild a component from its :class:`ComponentSpec` via the registry.

    Looks up the class by ``type_name`` and calls its ``from_config``. Components
    whose ``from_config`` takes a Pydantic model (the FAISS/embedder convention)
    declare a ``config_model`` classvar; we validate the dict into it first.
    Components whose ``from_config`` accepts a ``state_dir`` (e.g. ``VectorBlocker``,
    which restores its nested index's state) are given the slot's state dir.
    Finally, if the rebuilt component is itself a
    :class:`~langres.core.serialization.SerializableState` and a populated
    ``state_dir`` exists, its state is restored directly.
    """
    cls = get_component(spec.type_name)
    config_model = getattr(cls, "config_model", None)
    config_arg = (
        config_model.model_validate(spec.config) if config_model is not None else spec.config
    )

    # Pass state_dir only to from_config signatures that accept it, and only
    # when the sidecar actually holds state (an empty/absent dir means none).
    accepts_state_dir = "state_dir" in inspect.signature(cls.from_config).parameters  # type: ignore[attr-defined]
    if accepts_state_dir and has_state(state_dir):
        component = cls.from_config(config_arg, state_dir=state_dir)  # type: ignore[attr-defined]
    else:
        component = cls.from_config(config_arg)  # type: ignore[attr-defined]

    # Restore directly only when ``from_config`` did not already handle state
    # itself (guards against a double ``load_state`` for a component that both
    # accepts ``state_dir`` and implements SerializableState).
    if not accepts_state_dir and isinstance(component, SerializableState) and has_state(state_dir):
        component.load_state(state_dir)
    return component
