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

from pydantic import BaseModel, ConfigDict, ValidationError

from langres.core.op import OutSpace, Stage, ThresholdSelect, TopKSelect
from langres.core.op_adapters import (
    BlockerSource,
    CalibratorScore,
    ClustererStage,
    ComparatorScore,
    MatcherScore,
)
from langres.core.registry import (
    OpSerializer,
    UnknownOpType,
    get_component,
    get_op_serializer,
    op_serializer_for_type,
    register_op_serializer,
)
from langres.core.serialization import ComponentSpec, OpSpec, SerializableState
from langres.core.spend_cap import SpendCappedMatcher


class _EmptyOpParams(BaseModel):
    """No-parameter role envelope; unknown keys are always malformed."""

    model_config = ConfigDict(extra="forbid", strict=True)


class _MatcherScoreParams(BaseModel):
    """Validated matcher-score parameter envelope."""

    model_config = ConfigDict(extra="forbid", strict=True)

    out_space: OutSpace


class _ThresholdSelectParams(BaseModel):
    """Validated threshold-select parameter envelope."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    threshold: float


class _TopKSelectParams(BaseModel):
    """Validated top-k parameter envelope."""

    model_config = ConfigDict(extra="forbid", strict=True)

    k: int


def _validate_params_as(model: type[BaseModel], params: dict[str, object]) -> dict[str, object]:
    """Validate one complete role envelope and return normalized plain data."""
    return model.model_validate(params).model_dump()


def _validate_empty_params(params: dict[str, object]) -> dict[str, object]:
    return _validate_params_as(_EmptyOpParams, params)


def _validate_matcher_score_params(params: dict[str, object]) -> dict[str, object]:
    return _validate_params_as(_MatcherScoreParams, params)


def _validate_threshold_select_params(params: dict[str, object]) -> dict[str, object]:
    return _validate_params_as(_ThresholdSelectParams, params)


def _validate_topk_select_params(params: dict[str, object]) -> dict[str, object]:
    return _validate_params_as(_TopKSelectParams, params)


def _validated_op_params(serializer: OpSerializer, params: dict[str, object]) -> dict[str, object]:
    """Fail closed on params before any nested component is reconstructed."""
    if serializer.validate_params is None:
        raise ValueError(
            f"OpSpec role {serializer.role!r} has no registered parameter schema. "
            "Register a validate_params callable before loading artifacts."
        )
    try:
        return serializer.validate_params(params)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError(
            f"OpSpec role {serializer.role!r} has invalid params; allowed keys and types are "
            f"defined by its registered serializer: {exc}"
        ) from None


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


def _dump_blocker_source(stage: object) -> tuple[dict[str, object], object | None]:
    assert isinstance(stage, BlockerSource)
    return {}, stage.blocker


def _load_blocker_source(
    _params: dict[str, object], component: object | None, _state_dir: Path
) -> object:
    if component is None:
        raise ValueError("OpSpec role 'blocker_source' requires a nested component")
    return BlockerSource(component)  # type: ignore[arg-type]


def _dump_comparator_score(stage: object) -> tuple[dict[str, object], object | None]:
    assert isinstance(stage, ComparatorScore)
    return {}, stage.comparator


def _load_comparator_score(
    _params: dict[str, object], component: object | None, _state_dir: Path
) -> object:
    if component is None:
        raise ValueError("OpSpec role 'comparator_score' requires a nested component")
    return ComparatorScore(component)  # type: ignore[arg-type]


def _dump_matcher_score(stage: object) -> tuple[dict[str, object], object | None]:
    assert type(stage) is MatcherScore
    matcher = stage.matcher
    inner = matcher._module if isinstance(matcher, SpendCappedMatcher) else matcher
    return {"out_space": stage.out_space}, inner


def _load_matcher_score(
    params: dict[str, object], component: object | None, _state_dir: Path
) -> object:
    if component is None:
        raise ValueError("OpSpec role 'matcher_score' requires a nested component")
    return MatcherScore(component, out_space=cast(OutSpace, params["out_space"]))  # type: ignore[arg-type]


def _dump_calibrator_score(stage: object) -> tuple[dict[str, object], object | None]:
    assert isinstance(stage, CalibratorScore)
    return {}, stage.calibrator


def _load_calibrator_score(
    _params: dict[str, object], component: object | None, _state_dir: Path
) -> object:
    if component is None:
        raise ValueError("OpSpec role 'calibrator_score' requires a nested component")
    return CalibratorScore(component)  # type: ignore[arg-type]


def _dump_threshold_select(stage: object) -> tuple[dict[str, object], object | None]:
    assert isinstance(stage, ThresholdSelect)
    return {"threshold": stage.threshold}, None


def _load_threshold_select(
    params: dict[str, object], _component: object | None, _state_dir: Path
) -> object:
    return ThresholdSelect(cast(float, params["threshold"]))


def _dump_topk_select(stage: object) -> tuple[dict[str, object], object | None]:
    assert isinstance(stage, TopKSelect)
    return {"k": stage.k}, None


def _load_topk_select(
    params: dict[str, object], _component: object | None, _state_dir: Path
) -> object:
    return TopKSelect(cast(int, params["k"]))


def _dump_clusterer_stage(stage: object) -> tuple[dict[str, object], object | None]:
    assert isinstance(stage, ClustererStage)
    return {}, stage.clusterer


def _load_clusterer_stage(
    _params: dict[str, object], component: object | None, _state_dir: Path
) -> object:
    if component is None:
        raise ValueError("OpSpec role 'clusterer_stage' requires a nested component")
    return ClustererStage(component)  # type: ignore[arg-type]


def _register_builtin_op_serializers() -> None:
    """Register the shipped stages through the same table custom Ops use."""
    serializers = (
        OpSerializer(
            "blocker_source",
            BlockerSource,
            _dump_blocker_source,
            _load_blocker_source,
            component_slot="blocker",
            validate_params=_validate_empty_params,
        ),
        OpSerializer(
            "comparator_score",
            ComparatorScore,
            _dump_comparator_score,
            _load_comparator_score,
            component_slot="comparator",
            validate_params=_validate_empty_params,
        ),
        OpSerializer(
            "matcher_score",
            MatcherScore,
            _dump_matcher_score,
            _load_matcher_score,
            component_slot="module",
            validate_params=_validate_matcher_score_params,
        ),
        OpSerializer(
            "calibrator_score",
            CalibratorScore,
            _dump_calibrator_score,
            _load_calibrator_score,
            component_slot="calibrator",
            validate_params=_validate_empty_params,
        ),
        OpSerializer(
            "threshold_select",
            ThresholdSelect,
            _dump_threshold_select,
            _load_threshold_select,
            validate_params=_validate_threshold_select_params,
        ),
        OpSerializer(
            "topk_select",
            TopKSelect,
            _dump_topk_select,
            _load_topk_select,
            validate_params=_validate_topk_select_params,
        ),
        OpSerializer(
            "clusterer_stage",
            ClustererStage,
            _dump_clusterer_stage,
            _load_clusterer_stage,
            component_slot="clusterer",
            validate_params=_validate_empty_params,
        ),
    )
    for serializer in serializers:
        register_op_serializer(serializer)


_register_builtin_op_serializers()


def _stage_serialization(stage: Stage) -> tuple[str, dict[str, object], object | None]:
    """The ``(role, params, nested_component)`` an explicit-chain stage serializes to.

    The ONE registered dispatch shared by :func:`op_spec` (which builds the
    :class:`~langres.core.serialization.OpSpec`) and :func:`op_state_owner`
    (which finds the sidecar owner) — so the two can never disagree on which
    stages are serializable, or on which component a stage carries. Built-ins
    and ``@register_op`` custom stages use the same fail-closed table.

    A :class:`~langres.core.op_adapters.MatcherScore` is **unwrapped**: its nested
    component is the *raw inner* matcher, never the
    :class:`~langres.core.spend_cap.SpendCappedMatcher` wrapper (which holds a live
    per-model :class:`~langres.core.spend.SpendMonitor` and has no registry
    ``type_name``). ``from_topology`` re-wraps the raw matcher against a fresh
    monitor on load, so the budget/monitor are deliberately not persisted.

    Raises:
        TypeError: For an exact stage class with no registered serializer.
            Parent serializers are not inherited, so a subclass cannot silently
            lose its own state.
    """
    try:
        serializer = op_serializer_for_type(type(stage))
    except UnknownOpType as exc:
        raise TypeError(f"op_spec() cannot serialize a {type(stage).__name__}: {exc}") from None
    params, component = serializer.dump(stage)
    return serializer.role, params, component


def op_spec(stage: Stage) -> OpSpec:
    """Serialize one explicit-chain :class:`~langres.core.op.Stage` into an
    :class:`~langres.core.serialization.OpSpec`.

    Records the stage's role, its role-specific scalar params, and (for a stage
    that adapts a legacy component) that component as a nested
    :class:`~langres.core.serialization.ComponentSpec` via :func:`component_spec`.
    A ``MatcherScore``'s matcher is serialized RAW (unwrapped past any spend
    cap). Custom component-free Scores/Selects register their config convention
    with :func:`langres.core.registry.register_op`.

    Raises:
        TypeError: If the exact stage class has no registered serializer, or its
            nested component lacks a registry ``type_name``.
    """
    role, params, component = _stage_serialization(stage)
    serializer = get_op_serializer(role)
    params = _validated_op_params(serializer, params)
    component_spec_obj = None
    if component is not None:
        if serializer.component_slot is None:
            raise TypeError(
                f"Op serializer {role!r} returned a nested component but declares no component_slot"
            )
        component_spec_obj = component_spec(component, slot=serializer.component_slot)
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
    role, _params, component = _stage_serialization(stage)
    serializer = get_op_serializer(role)
    candidate = (
        serializer.state_owner(stage)
        if serializer.state_owner is not None
        else component
        if component is not None
        else stage
    )
    return state_owner(candidate) if candidate is not None else None


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
    try:
        serializer = get_op_serializer(spec.role)
    except UnknownOpType as exc:
        raise ValueError(f"rebuild_op() got an unknown OpSpec role {spec.role!r}: {exc}") from None
    params = _validated_op_params(serializer, spec.params)

    # Validate the role/component envelope BEFORE looking up or constructing the
    # nested component. A stray component on a component-free role must not
    # trigger arbitrary registry work, and a component under the wrong slot must
    # never be reconstructed and only then rejected.
    if serializer.component_slot is None:
        if spec.component is not None:
            raise ValueError(
                f"OpSpec role {spec.role!r} does not accept a nested component, but the spec "
                f"carries {spec.component.type_name!r}. Fix: remove component= from this role."
            )
        component = None
    else:
        component_spec_obj = _require_op_component(spec)
        if component_spec_obj.slot != serializer.component_slot:
            raise ValueError(
                f"OpSpec role {spec.role!r} requires component slot "
                f"{serializer.component_slot!r}, but the spec carries "
                f"{component_spec_obj.slot!r}. Fix: serialize the component under the role's "
                "declared slot."
            )
        component = rebuild_component(component_spec_obj, state_dir=state_dir)

    stage = serializer.load(params, component, state_dir)
    if not isinstance(
        stage,
        (
            BlockerSource,
            CalibratorScore,
            ComparatorScore,
            MatcherScore,
            ThresholdSelect,
            TopKSelect,
            ClustererStage,
        ),
    ):
        # Custom stages still have to satisfy the public Stage contract. Importing
        # every concrete subclass is unnecessary; the four abstract roles cover it.
        from langres.core.op import ClusterStage, Finalize, Op, Source

        if not isinstance(stage, (Source, Op, ClusterStage, Finalize)):
            raise TypeError(
                f"registered Op serializer {spec.role!r} rebuilt {type(stage).__name__}, "
                "which is not a Source/Op/ClusterStage/Finalize"
            )
    if isinstance(stage, SerializableState) and has_state(state_dir):
        stage.load_state(state_dir)
    return stage
