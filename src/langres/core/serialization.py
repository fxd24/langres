"""Serialization contracts for the M0 Resolver artifact.

Two things live here:

1. :class:`SerializableState` — an **optional** capability that heavy
   components (e.g. the Wave 2d FAISS index) implement when they own
   out-of-band state (an index file) that cannot round-trip through plain
   JSON config. The Resolver checks for this capability with ``isinstance``.
   It is deliberately **not** added to the existing ``VectorIndex`` Protocol —
   doing so would force Qdrant/hybrid/reranking indexes to implement it.

2. The artifact manifest types — the typed shape of ``resolver.json`` so
   save/load has a contract: :class:`ComponentSpec` (a classic four-slot
   component), :class:`OpSpec` (an explicit-chain stage), :class:`ArtifactManifest`,
   and the :data:`ARTIFACT_VERSION` / :data:`CLASSIC_ARTIFACT_VERSION` constants.

This is a ``core`` leaf: it imports only pydantic + stdlib and MUST NOT import
:mod:`~langres.core.op` / :mod:`~langres.core.op_adapters`. The op↔spec *instance*
adapters (``op_spec`` / ``rebuild_op``, which reach into those modules) live in
:mod:`langres.core._artifacts`, a tier that already imports them — so the on-disk
data contract stays a dependency-free leaf.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

#: The **classic four-slot** on-disk layout, frozen at ``"1"``. A classic
#: (``self._ops is None``) save always stamps this, so its bytes — and the
#: ``recipe_id`` derived from its ``config_dict`` — never fork when the reader's
#: max advances. Do NOT bump this: bumping :data:`ARTIFACT_VERSION` (the reader's
#: max) is how a new *additive* layout ships without restamping classic artifacts.
CLASSIC_ARTIFACT_VERSION = "1"

#: The reader's **maximum** supported layout. Bumped ``"1"`` → ``"2"`` when the
#: explicit Op-chain layout (an ``ops`` list instead of ``components``) landed
#: (#193, persist v2). The reader accepts any layout in ``[1, ARTIFACT_VERSION]``;
#: only an explicit-chain (``self._ops is not None``) save stamps ``"2"``.
ARTIFACT_VERSION = "2"


@runtime_checkable
class SerializableState(Protocol):
    """Optional capability for components with out-of-band state to persist.

    A component implements this when its config alone cannot reconstruct it —
    for example a vector index that must persist its built index file. The
    Resolver detects the capability via ``isinstance(component, SerializableState)``
    and calls these methods with a component-specific subdirectory.

    Light components (no out-of-band state) do not implement this; their config
    fully reconstructs them.
    """

    def save_state(self, state_dir: Path) -> None:
        """Persist out-of-band state into ``state_dir`` (created by caller)."""
        ...  # pragma: no cover

    def load_state(self, state_dir: Path) -> None:
        """Restore out-of-band state previously written to ``state_dir``."""
        ...  # pragma: no cover


class ComponentSpec(BaseModel):
    """Serialized record of one Resolver component in the manifest.

    Attributes:
        type_name: Registry key (see :mod:`langres.core.registry`) used to look
            up the component class at load time.
        slot: The Resolver slot this component fills (``"blocker"``,
            ``"comparator"``, ``"module"``, or ``"clusterer"``). Lets
            :meth:`Resolver.load` map specs back to slots by name rather than by
            position or by a hard-coded ``type_name``, so a registered subclass
            with a custom ``type_name`` loads into the right slot. ``None`` for
            older/hand-written manifests, where load falls back to positional
            identification.
        config_version: Version of this component's config schema, for
            forward-compatible migration. Defaults to ``"1"``.
        config: The component's serializable construction config (pure data).
    """

    type_name: str
    slot: str | None = None
    config_version: str = "1"
    config: dict[str, object]


class OpSpec(BaseModel):
    """Serialized record of one **explicit-chain** stage in the manifest.

    The explicit-chain (:meth:`~langres.core._model_state.ModelState.from_topology`)
    analogue of :class:`ComponentSpec`: where a classic four-slot model serializes
    each slot as a ``ComponentSpec`` under ``ArtifactManifest.components``, an
    explicit-chain model serializes each :class:`~langres.core.op.Stage` as an
    ``OpSpec`` under :attr:`ArtifactManifest.ops`.

    Kept a pure data model beside ``ComponentSpec`` (this module is a leaf that
    imports only pydantic/stdlib). The op↔spec instance adapters that build one
    from a live stage and reverse it — ``op_spec`` / ``rebuild_op``, which import
    :mod:`~langres.core.op` / :mod:`~langres.core.op_adapters` — live in
    :mod:`langres.core._artifacts`.

    Attributes:
        role: The stage's role tag, e.g. ``"blocker_source"``,
            ``"comparator_score"``, ``"matcher_score"``, ``"threshold_select"``,
            ``"topk_select"``, ``"clusterer_stage"`` — how ``rebuild_op`` knows
            which adapter to reconstruct.
        params: Role-specific scalar parameters (e.g. ``{"threshold": 0.5}`` for a
            ThresholdSelect, ``{"k": 10}`` for a TopKSelect, ``{"out_space":
            "prob_llm"}`` for a MatcherScore). Empty for a role carrying no scalar
            of its own.
        component: The nested legacy component (blocker / comparator / matcher /
            clusterer) the stage adapts, as a :class:`ComponentSpec` — or ``None``
            for a component-free stage (a Select carries only ``params``). A
            spend-cap wrapper is deliberately never serialized here: a
            ``MatcherScore``'s spec records the *raw inner* matcher, and
            ``from_topology`` re-wraps it on load.
    """

    role: str
    params: dict[str, object] = Field(default_factory=dict)
    component: ComponentSpec | None = None


class ArtifactManifest(BaseModel):
    """Typed shape of ``resolver.json``.

    Carries **either** ``components`` (the classic four-slot layout) **or** ``ops``
    (the explicit Op-chain layout, #193 persist v2) — never meaningfully both. A
    classic save writes ``components`` and omits ``ops``; an explicit-chain save
    writes ``ops`` and omits ``components``. ``ops`` defaults to ``None`` so an old
    v1 ``resolver.json`` (which has no ``ops`` key) validates straight onto the
    classic read path.

    Attributes:
        artifact_version: Layout version of the artifact (see
            :data:`ARTIFACT_VERSION` / :data:`CLASSIC_ARTIFACT_VERSION`).
        langres_version: The ``langres.__version__`` that wrote the artifact.
        model_class: Registered name of the Resolver *subclass* that wrote the
            artifact (see ``langres.core.registry.register_model``), so a named
            architecture survives a round-trip instead of loading back as a
            plain ``Resolver``. **Optional by design**: ``None``/absent means a
            plain ``Resolver``, which is what every pre-0.4 artifact — and any
            unregistered subclass — is. That is why adding it does **not** bump
            :data:`ARTIFACT_VERSION`: a compatible read costs one ``if``, while a
            bump would make ``Resolver._check_versions`` reject existing 0.3.0
            artifacts in *both* directions (it raises on older *and* newer) for
            no gain.
        components: Ordered component specs composing a **classic four-slot**
            Resolver. Empty for an explicit-chain artifact (which uses ``ops``).
        ops: Ordered stage specs composing an **explicit Op chain**
            (:meth:`~langres.core._model_state.ModelState.from_topology`), or
            ``None`` for the classic four-slot layout. ``None`` by design so a
            pre-#193 v1 ``resolver.json`` — which has no ``ops`` key — validates to
            ``None`` and reads through the classic ``components`` path unchanged.
        checksums: Optional sidecar checksum map (filename -> checksum) for
            out-of-band state files written by :class:`SerializableState`
            components. Empty when no component has out-of-band state.
    """

    artifact_version: str
    langres_version: str
    model_class: str | None = None
    components: list[ComponentSpec] = Field(default_factory=list)
    ops: list[OpSpec] | None = None
    checksums: dict[str, str] = Field(default_factory=dict)
