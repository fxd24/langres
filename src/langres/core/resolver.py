"""Resolver: the M0 spine that composes a full entity-resolution pipeline.

The Resolver is the **top-level container** of langres.core. It wires four
slots into one runnable, serializable pipeline:

    blocker      -> candidate generation + schema normalization
    comparator   -> (optional) missing-aware per-feature comparison
    module       -> the scorer (a Module yielding PairwiseJudgements)
    clusterer    -> connected-components grouping of matched pairs

``resolve(records)`` orchestrates blocking -> (compare) -> score -> cluster.
``save(path)`` / ``load(path)`` round-trip the whole pipeline through a
human-readable ``resolver.json`` manifest plus per-component sidecar state for
any component that owns out-of-band state (e.g. a built FAISS index). Loading
executes **no code and no pickle** — every slot is rebuilt from the component
registry by its ``type_name``.

Unified serialization convention
---------------------------------
Wave 2 produced two component-config styles:

- A ``config`` **property** returning a plain ``dict`` (comparator, blockers,
  clusterer, judge).
- A ``config()`` **method** returning a Pydantic ``BaseModel`` plus
  ``type_name`` / ``config_model`` classvars (FAISSIndex, embedders).

The Resolver does not pick one and rewrite the other. Instead it adapts both
behind two tiny helpers — :func:`_component_spec` (object -> ``ComponentSpec``)
and :func:`_rebuild_component` (``ComponentSpec`` -> object) — so every slot is
serialized and reconstructed uniformly. Every Resolver-slot component exposes a
``type_name`` class attribute so the spec helper can discover its registry key;
the helpers normalize the dict-vs-model and property-vs-method differences.
"""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel

import langres
from langres.core.blocker import Blocker
from langres.core.clusterer import Clusterer
from langres.core.comparator import Comparator
from langres.core.judges.weighted_average import WeightedAverageJudge
from langres.core.models import PairwiseJudgement
from langres.core.module import Module
from langres.core.registry import get_component
from langres.core.serialization import (
    ARTIFACT_VERSION,
    ArtifactManifest,
    ComponentSpec,
    SerializableState,
)

logger = logging.getLogger(__name__)

# Slot names used as sidecar subdirectory names and for manifest ordering.
_MANIFEST_FILENAME = "resolver.json"


def _component_config_dict(obj: object) -> dict[str, object]:
    """Return a component's construction config as a plain JSON-able dict.

    Bridges the two Wave 2 conventions:

    - ``config`` **property** returning a ``dict`` -> returned as-is.
    - ``config()`` **method** returning a Pydantic ``BaseModel`` -> dumped.
    """
    config = obj.config() if callable(getattr(obj, "config")) else obj.config  # type: ignore[attr-defined]
    if isinstance(config, BaseModel):
        return config.model_dump()
    return dict(config)


def _component_spec(obj: object) -> ComponentSpec:
    """Serialize any Resolver-slot component into a :class:`ComponentSpec`.

    Reads the component's ``type_name`` class attribute (the registry key) and
    its construction config (via :func:`_component_config_dict`).
    """
    type_name = obj.type_name  # type: ignore[attr-defined]
    return ComponentSpec(type_name=type_name, config=_component_config_dict(obj))


def _rebuild_component(spec: ComponentSpec, state_dir: Path | None = None) -> Any:
    """Rebuild a component from its :class:`ComponentSpec` via the registry.

    Looks up the class by ``type_name`` and calls its ``from_config``. Components
    whose ``from_config`` takes a Pydantic model (the FAISS/embedder convention)
    declare a ``config_model`` classvar; we validate the dict into it first.
    Components whose ``from_config`` takes a plain dict receive the dict directly.
    After construction, if the component implements
    :class:`~langres.core.serialization.SerializableState` and a ``state_dir`` is
    given, its out-of-band state is restored.
    """
    cls = get_component(spec.type_name)
    config_model = getattr(cls, "config_model", None)
    if config_model is not None:
        component = cls.from_config(config_model.model_validate(spec.config))  # type: ignore[attr-defined]
    else:
        component = cls.from_config(spec.config)  # type: ignore[attr-defined]

    if isinstance(component, SerializableState) and state_dir is not None and state_dir.exists():
        component.load_state(state_dir)
    return component


class Resolver:
    """Composable entity-resolution pipeline: blocker -> compare -> score -> cluster.

    Args:
        blocker: Candidate generator + schema normalizer.
        comparator: Optional pre-stage turning each pair into a
            ComparisonVector. When ``None``, the module is called directly
            (e.g. a self-contained ``RapidfuzzModule``).
        module: The scorer Module that yields PairwiseJudgements.
        clusterer: Groups matched pairs into entity clusters.

    Example:
        resolver = Resolver(
            blocker=AllPairsBlocker(schema=CompanySchema),
            comparator=Comparator.from_schema(CompanySchema, weights={"name": 0.6, ...}),
            module=WeightedAverageJudge(),
            clusterer=Clusterer(threshold=0.7),
        )
        clusters = resolver.resolve(COMPANY_RECORDS)
        resolver.save("artifacts/company_v0")
        reloaded = Resolver.load("artifacts/company_v0")
    """

    def __init__(
        self,
        blocker: Blocker[Any],
        comparator: Comparator[Any] | None,
        module: Module[Any],
        clusterer: Clusterer,
    ) -> None:
        self.blocker = blocker
        self.comparator = comparator
        self.module = module
        self.clusterer = clusterer

    # ------------------------------------------------------------------
    # Construction convenience
    # ------------------------------------------------------------------

    @classmethod
    def from_schema(
        cls,
        schema: type[BaseModel],
        *,
        threshold: float = 0.7,
        weights: dict[str, float] | None = None,
        exclude: set[str] | None = None,
    ) -> "Resolver":
        """Build a default dedup Resolver from a Pydantic schema in one line.

        Defaults to an ``AllPairsBlocker`` over the schema, a missing-aware
        ``StringComparator`` auto-derived from the schema's string fields (with
        ``id`` excluded), a ``WeightedAverageJudge`` scorer, and a ``Clusterer``
        at ``threshold``.

        Args:
            schema: The Pydantic entity schema to resolve.
            threshold: Clusterer match threshold (default 0.7).
            weights: Optional per-feature weight overrides for the comparator.
                Defaults to equal weights; pass name-dominant weights (e.g.
                ``{"name": 0.6, "address": 0.2, ...}``) to recover name-only
                duplicates that equal weights would gate out via the evidence
                floor.
            exclude: Field names to skip when deriving features. Defaults to
                ``{"id"}`` (handled by the comparator).

        Returns:
            A ready-to-run Resolver.
        """
        from langres.core.blockers.all_pairs import AllPairsBlocker

        return cls(
            blocker=AllPairsBlocker(schema=schema),
            comparator=Comparator.from_schema(schema, exclude=exclude, weights=weights),
            module=WeightedAverageJudge(),
            clusterer=Clusterer(threshold=threshold),
        )

    # ------------------------------------------------------------------
    # Running the pipeline
    # ------------------------------------------------------------------

    def fit(self, data: list[Any]) -> Self:
        """No-op fit for sklearn-style symmetry; returns self.

        M0 components are not learned — the heuristic scorer and thresholds are
        fixed. Optimization (Optuna over thresholds/weights, DSPy over prompts)
        lands in M3+, at which point ``fit`` will tune the pipeline on labeled
        data. It exists now so callers can write ``resolver.fit(data).resolve(data)``
        without branching on whether the pipeline is learnable.
        """
        return self

    def _judgements(self, records: list[Any]) -> Iterator[PairwiseJudgement]:
        """Block records into candidates and score them into judgements.

        Builds an index-backed blocker's index transparently before streaming,
        so callers never call ``create_index`` themselves. Records are fed in
        the caller's stable list order.
        """
        self._ensure_index_built(records)
        candidates = self.blocker.stream(records)
        if self.comparator is not None:
            return self.module.forward(candidates, comparator=self.comparator)  # type: ignore[call-arg]
        return self.module.forward(candidates)

    def predict(self, records: list[Any]) -> list[PairwiseJudgement]:
        """Return the scored pairwise judgements before clustering.

        Useful for observability/tuning: inspect scores and provenance without
        committing to a clustering threshold.
        """
        return list(self._judgements(records))

    def resolve(self, records: list[Any]) -> list[set[str]]:
        """Resolve records into entity clusters (sets of IDs).

        Orchestrates blocking -> (compare) -> score -> cluster. Singletons are
        dropped by the Clusterer (it returns only connected components with an
        edge), so the result contains only multi-record clusters.

        Args:
            records: Raw records (dicts) in a stable list order.

        Returns:
            A list of clusters, each a set of entity IDs.
        """
        return self.clusterer.cluster(self._judgements(records))

    def _ensure_index_built(self, records: list[Any]) -> None:
        """Build/populate an index-backed blocker's index from ``records``.

        For a ``VectorBlocker`` whose index has not been built yet, embed the
        records' text field and create the index in place. For a blocker with no
        index (AllPairs), this is a no-op. Idempotent: an already-built index is
        left untouched (so a freshly loaded FAISS index keeps its restored state).
        """
        index = getattr(self.blocker, "vector_index", None)
        if index is None:
            return  # AllPairs and other index-free blockers.

        # Already built (e.g. restored via load_state) -> reuse, never re-embed.
        if getattr(self.blocker, "_index_is_built", None) is not None:
            if self.blocker._index_is_built():  # type: ignore[attr-defined]
                return

        entities = [self.blocker.schema_factory(record) for record in records]  # type: ignore[attr-defined]
        texts = [self.blocker.text_field_extractor(entity) for entity in entities]  # type: ignore[attr-defined]
        logger.info("Embedding %d records to build the blocker's vector index…", len(texts))
        index.create_index(texts)

    # ------------------------------------------------------------------
    # Linking / streaming (M5)
    # ------------------------------------------------------------------

    def link(self, left_records: list[Any], right_records: list[Any]) -> list[set[str]]:
        """Entity linking across two record sets. Not implemented until M5."""
        raise NotImplementedError(
            "Resolver.link (cross-source entity linking) lands in M5."
        )  # pragma: no cover

    def stream_against(self, records: list[Any]) -> Iterator[set[str]]:
        """Incremental resolution against a persisted index. Not implemented until M5."""
        raise NotImplementedError(
            "Resolver.stream_against (incremental resolution) lands in M5."
        )  # pragma: no cover

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _slots(self) -> list[tuple[str, object]]:
        """Ordered (slot_name, component) pairs, skipping an absent comparator.

        The slot name doubles as the sidecar subdirectory name for components
        that own out-of-band state.
        """
        slots: list[tuple[str, object]] = [("blocker", self.blocker)]
        if self.comparator is not None:
            slots.append(("comparator", self.comparator))
        slots.append(("module", self.module))
        slots.append(("clusterer", self.clusterer))
        return slots

    def save(self, path: str | Path) -> None:
        """Persist the whole pipeline to ``path`` as a self-describing artifact.

        Writes ``resolver.json`` (an :class:`ArtifactManifest`) plus, for any
        slot component that implements
        :class:`~langres.core.serialization.SerializableState`, a sidecar state
        directory named after the slot. The manifest records, per slot, the
        component ``type_name`` and config (the embedder persists by
        ``model_name`` only — no model bytes).

        Args:
            path: Directory to write the artifact into (created if absent).
        """
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        components: list[ComponentSpec] = []
        for slot_name, component in self._slots():
            components.append(_component_spec(component))
            if isinstance(component, SerializableState):
                state_dir = out_dir / slot_name
                state_dir.mkdir(parents=True, exist_ok=True)
                component.save_state(state_dir)

        manifest = ArtifactManifest(
            artifact_version=ARTIFACT_VERSION,
            langres_version=langres.__version__,
            components=components,
        )
        (out_dir / _MANIFEST_FILENAME).write_text(manifest.model_dump_json(indent=2))
        logger.info("Saved Resolver artifact to %s", out_dir)

    @classmethod
    def load(cls, path: str | Path) -> "Resolver":
        """Reconstruct a Resolver from an artifact directory written by :meth:`save`.

        Reads ``resolver.json``, validates the artifact version, and rebuilds
        each slot from the component registry by its ``type_name`` (no code
        execution, no pickle). Sidecar state is restored for any
        :class:`~langres.core.serialization.SerializableState` component.

        Args:
            path: Directory containing ``resolver.json`` and any sidecars.

        Returns:
            A Resolver equivalent to the one that was saved.

        Raises:
            ValueError: If the artifact's ``artifact_version`` is newer than this
                library understands.
        """
        in_dir = Path(path)
        manifest = ArtifactManifest.model_validate_json(
            (in_dir / _MANIFEST_FILENAME).read_text()
        )
        cls._check_versions(manifest)

        # Map specs back to slots. The comparator slot is present iff a spec has
        # type_name == "comparator"; everything else is positional.
        by_type = {spec.type_name: spec for spec in manifest.components}
        comparator_spec = by_type.get("comparator")

        # Identify the blocker, module, clusterer specs by elimination/order.
        ordered = list(manifest.components)
        blocker_spec = ordered[0]
        clusterer_spec = ordered[-1]
        module_spec = next(
            spec
            for spec in ordered
            if spec not in (blocker_spec, clusterer_spec, comparator_spec)
        )

        blocker = _rebuild_component(blocker_spec, state_dir=in_dir / "blocker")
        comparator = (
            _rebuild_component(comparator_spec, state_dir=in_dir / "comparator")
            if comparator_spec is not None
            else None
        )
        module = _rebuild_component(module_spec, state_dir=in_dir / "module")
        clusterer = _rebuild_component(clusterer_spec, state_dir=in_dir / "clusterer")

        return cls(
            blocker=blocker,
            comparator=comparator,
            module=module,
            clusterer=clusterer,
        )

    @staticmethod
    def _check_versions(manifest: ArtifactManifest) -> None:
        """Validate artifact compatibility; raise on an unreadably-new artifact.

        A newer-or-equal ``artifact_version`` than this library's
        :data:`~langres.core.serialization.ARTIFACT_VERSION` is fine to read
        (same major layout). A *strictly newer* layout we cannot understand is a
        hard error. A ``langres_version`` mismatch is logged as a warning, not a
        failure — configs are forward-compatible within a layout version.
        """
        try:
            artifact_v = int(manifest.artifact_version)
            current_v = int(ARTIFACT_VERSION)
        except ValueError:  # non-integer versions: fall back to string equality
            if manifest.artifact_version != ARTIFACT_VERSION:
                raise ValueError(
                    f"Artifact version {manifest.artifact_version!r} differs from "
                    f"supported {ARTIFACT_VERSION!r}; cannot load."
                ) from None
            return
        if artifact_v > current_v:
            raise ValueError(
                f"Artifact version {manifest.artifact_version!r} is newer than this "
                f"langres build supports ({ARTIFACT_VERSION!r}); upgrade langres to load it."
            )
        if manifest.langres_version != langres.__version__:
            logger.warning(
                "Loading artifact written by langres %s into langres %s; "
                "configs are forward-compatible within artifact version %s.",
                manifest.langres_version,
                langres.__version__,
                ARTIFACT_VERSION,
            )
