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
ARTIFACT_VERSION = "0"


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
        ...

    def load_state(self, state_dir: Path) -> None:
        """Restore out-of-band state previously written to ``state_dir``."""
        ...


class ComponentSpec(BaseModel):
    """Serialized record of one Resolver component in the manifest.

    Attributes:
        type_name: Registry key (see :mod:`langres.core.registry`) used to look
            up the component class at load time.
        config_version: Version of this component's config schema, for
            forward-compatible migration. Defaults to ``"1"``.
        config: The component's serializable construction config (pure data).
    """

    type_name: str
    config_version: str = "1"
    config: dict[str, object]


class ArtifactManifest(BaseModel):
    """Typed shape of ``resolver.json``.

    Attributes:
        artifact_version: Layout version of the artifact (see
            :data:`ARTIFACT_VERSION`).
        langres_version: The ``langres.__version__`` that wrote the artifact.
        components: Ordered component specs composing the Resolver.
        checksums: Optional sidecar checksum map (filename -> checksum) for
            out-of-band state files written by :class:`SerializableState`
            components. Empty when no component has out-of-band state.
    """

    artifact_version: str
    langres_version: str
    components: list[ComponentSpec]
    checksums: dict[str, str] = Field(default_factory=dict)
