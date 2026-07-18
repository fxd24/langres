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
from typing import Any, TypeGuard, cast

from pydantic import BaseModel

from langres.core.op import OutSpace, Stage, ThresholdSelect, TopKSelect
from langres.core.op_adapters import (
    BlockerSource,
    ClustererStage,
    ComparatorScore,
    MatcherScore,
)
from langres.core.registry import get_component
from langres.core.serialization import ComponentSpec, OpSpec, SerializableState
from langres.core.spend_cap import SpendCappedMatcher


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


# ----------------------------------------------------------------------------------
# Explicit Op-chain adapters (#193, persist v2): the OpSpec analogue of
# component_spec / rebuild_component. A classic four-slot model serializes each
# slot with the pair above; an explicit-chain model
# (:meth:`~langres.core._model_state.ModelState.from_topology`) serializes each
# :class:`~langres.core.op.Stage` with the pair below. These import ``op`` /
# ``op_adapters`` / ``spend_cap``, which is why they live HERE (the ``architectures``
# tier that already imports those) and NOT in the ``serialization`` leaf that owns
# the pure :class:`~langres.core.serialization.OpSpec` data model.
# ----------------------------------------------------------------------------------

#: role -> the classic slot name stamped on a stage's *nested* ComponentSpec. The
#: slot is cosmetic in the ops path (``rebuild_op`` rebuilds by role, and
#: ``rebuild_component`` ignores ``ComponentSpec.slot``), but naming it after the
#: classic slot keeps the on-disk json legible.
_ROLE_COMPONENT_SLOT: dict[str, str] = {
    "blocker_source": "blocker",
    "comparator_score": "comparator",
    "matcher_score": "module",
    "clusterer_stage": "clusterer",
}


def _stage_serialization(stage: Stage) -> tuple[str, dict[str, object], object | None]:
    """The ``(role, params, nested_component)`` an explicit-chain stage serializes to.

    The ONE dispatch shared by :func:`op_spec` (which builds the
    :class:`~langres.core.serialization.OpSpec`) and :func:`op_state_owner` (which
    finds the sidecar owner) — so the two can never disagree on which stages are
    serializable, or on which component a stage carries.

    A :class:`~langres.core.op_adapters.MatcherScore` is **unwrapped**: its nested
    component is the *raw inner* matcher, never the
    :class:`~langres.core.spend_cap.SpendCappedMatcher` wrapper (which holds a live
    per-model :class:`~langres.core.spend.SpendMonitor` and has no registry
    ``type_name``). ``from_topology`` re-wraps the raw matcher against a fresh
    monitor on load, so the budget/monitor are deliberately not persisted.

    Raises:
        TypeError: (F3) for a stage this door cannot faithfully round-trip — a
            ``MatcherScore`` *subclass* (rebuilding it as a base ``MatcherScore``
            would silently drop the subclass's own state, exactly as
            ``from_topology`` rejects on re-secure), a ``GroupwiseMatcherScore`` or
            any other ``Spending`` Score ``from_topology`` also refuses, or a
            ``Finalize`` (``from_topology`` does not run one). Failing loud at save
            time beats writing a spec that load cannot rebuild.
    """
    if isinstance(stage, BlockerSource):
        return "blocker_source", {}, stage.blocker
    if isinstance(stage, ComparatorScore):
        return "comparator_score", {}, stage.comparator
    # ``type(stage) is MatcherScore`` (not isinstance): a subclass must be rejected
    # below, since rebuild would drop its extra state (mirrors _secure_chain_scores).
    if type(stage) is MatcherScore:
        matcher = stage.matcher
        inner = matcher._module if isinstance(matcher, SpendCappedMatcher) else matcher
        return "matcher_score", {"out_space": stage.out_space}, inner
    if isinstance(stage, ThresholdSelect):
        return "threshold_select", {"threshold": stage.threshold}, None
    if isinstance(stage, TopKSelect):
        return "topk_select", {"k": stage.k}, None
    if isinstance(stage, ClustererStage):
        return "clusterer_stage", {}, stage.clusterer
    raise TypeError(
        f"op_spec() cannot serialize a {type(stage).__name__}: only BlockerSource, "
        "ComparatorScore, a base MatcherScore, ThresholdSelect, TopKSelect and ClustererStage "
        "round-trip. A MatcherScore subclass / GroupwiseMatcherScore / Finalize is rejected "
        "(from_topology would reject or drop it on re-secure anyway). Fix: express the chain "
        "with the round-trippable stages, or pre-cap a paid matcher into a base MatcherScore."
    )


def op_spec(stage: Stage) -> OpSpec:
    """Serialize one explicit-chain :class:`~langres.core.op.Stage` into an
    :class:`~langres.core.serialization.OpSpec`.

    Records the stage's role, its role-specific scalar params, and (for a stage
    that adapts a legacy component) that component as a nested
    :class:`~langres.core.serialization.ComponentSpec` via :func:`component_spec`.
    A ``MatcherScore``'s matcher is serialized RAW (unwrapped past any spend cap);
    see :func:`_stage_serialization` for the unwrap and the F3 rejection rules.

    Raises:
        TypeError: If ``stage`` is not one of the round-trippable stage types, or
            its nested component lacks a registry ``type_name`` (both surface the
            same fail-loud-at-save contract as :func:`component_spec`).
    """
    role, params, component = _stage_serialization(stage)
    component_spec_obj = (
        component_spec(component, slot=_ROLE_COMPONENT_SLOT[role])
        if component is not None
        else None
    )
    return OpSpec(role=role, params=params, component=component_spec_obj)


def op_state_owner(stage: Stage) -> SerializableState | None:
    """The out-of-band-state owner of an explicit-chain stage, if any.

    The sidecar seam ``save`` walks per stage (keyed by ordinal ``op{i}``). It
    delegates to :func:`state_owner` on the SAME nested component :func:`op_spec`
    serializes, so a ``BlockerSource`` over a built ``VectorBlocker`` writes/reads
    its index sidecar exactly as the classic blocker slot does. A Select carries no
    component and owns no state; the shipped rerankers source from
    ``AllPairsBlocker`` and own none either. Called only after ``_build_manifest``
    has already validated every stage via :func:`op_spec`, so it never meets an
    unserializable stage.
    """
    _role, _params, component = _stage_serialization(stage)
    return state_owner(component) if component is not None else None


def _require_op_component(spec: OpSpec) -> ComponentSpec:
    """Return ``spec.component``, or raise on a malformed component-bearing OpSpec."""
    if spec.component is None:
        raise ValueError(
            f"OpSpec role {spec.role!r} requires a nested component to rebuild, but the spec "
            "carries none. Fix: this role must be written with component=component_spec(...)."
        )
    return spec.component


def rebuild_op(spec: OpSpec, *, state_dir: Path) -> Stage:
    """Rebuild an explicit-chain :class:`~langres.core.op.Stage` from its
    :class:`~langres.core.serialization.OpSpec`.

    The reverse of :func:`op_spec`: dispatches on ``spec.role``, rebuilds any
    nested component via :func:`rebuild_component` (restoring its sidecar state from
    ``state_dir``), and reconstructs the stage. A ``matcher_score`` is rebuilt
    around the **raw** inner matcher — ``from_topology`` re-wraps it in a
    :class:`~langres.core.spend_cap.SpendCappedMatcher` sharing the reloaded model's
    fresh ledger, so the cap is re-established on load rather than persisted.

    Args:
        spec: The stage spec to rebuild.
        state_dir: The per-stage sidecar directory (``op{i}``) any stateful nested
            component restores from.

    Raises:
        ValueError: If ``spec.role`` is unknown, or a component-bearing role's
            ``component`` is missing (a malformed manifest).
    """
    if spec.role == "blocker_source":
        return BlockerSource(rebuild_component(_require_op_component(spec), state_dir=state_dir))
    if spec.role == "comparator_score":
        return ComparatorScore(rebuild_component(_require_op_component(spec), state_dir=state_dir))
    if spec.role == "matcher_score":
        return MatcherScore(
            rebuild_component(_require_op_component(spec), state_dir=state_dir),
            out_space=cast(OutSpace, spec.params["out_space"]),
        )
    if spec.role == "threshold_select":
        return ThresholdSelect(cast(float, spec.params["threshold"]))
    if spec.role == "topk_select":
        return TopKSelect(cast(int, spec.params["k"]))
    if spec.role == "clusterer_stage":
        return ClustererStage(rebuild_component(_require_op_component(spec), state_dir=state_dir))
    raise ValueError(
        f"rebuild_op() got an unknown OpSpec role {spec.role!r}. Known roles: "
        "blocker_source, comparator_score, matcher_score, threshold_select, topk_select, "
        "clusterer_stage. The artifact was likely written by a newer langres."
    )
