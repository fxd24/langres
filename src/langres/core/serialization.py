"""Serialization contracts for the M0 Resolver artifact.

Two things live here:

1. :class:`SerializableState` — an **optional** capability that heavy
   components (e.g. the Wave 2d FAISS index) implement when they own
   out-of-band state (an index file) that cannot round-trip through plain
   JSON config. The Resolver checks for this capability with ``isinstance``.
   It is deliberately **not** added to the existing ``VectorIndex`` Protocol —
   doing so would force Qdrant/hybrid/reranking indexes to implement it.

2. The artifact manifest types — the typed shape of ``resolver.json`` so
   save/load (Wave 3) has a contract: :class:`ComponentSpec`,
   :class:`ArtifactManifest`, and the :data:`ARTIFACT_VERSION` constant.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

# Bump when the on-disk artifact layout changes incompatibly.
ARTIFACT_VERSION = "1"


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


class ArtifactManifest(BaseModel):
    """Typed shape of ``resolver.json``.

    Attributes:
        artifact_version: Layout version of the artifact (see
            :data:`ARTIFACT_VERSION`).
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
        components: Ordered component specs composing the Resolver.
        checksums: Optional sidecar checksum map (filename -> checksum) for
            out-of-band state files written by :class:`SerializableState`
            components. Empty when no component has out-of-band state.
    """

    artifact_version: str
    langres_version: str
    model_class: str | None = None
    components: list[ComponentSpec]
    checksums: dict[str, str] = Field(default_factory=dict)
