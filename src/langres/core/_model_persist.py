"""How an ER model persists: ``resolver.json`` + per-slot sidecars, no pickle.

The persistence half of the ``ERModel`` class chain, layered on
:mod:`langres.core._model_state` (which owns the slots this walks) and inherited
by the leaf in :mod:`langres.core.resolver`.

This module decides *what* is persisted -- the ordered slots, the version
envelope, the ``model_class`` stamp that lets a named architecture reload as
itself -- and delegates *how* each individual component crosses the JSON boundary
to the adapters in :mod:`langres.core._artifacts`. Loading executes **no code and
no pickle**: every slot is rebuilt from the component registry by its
``type_name``.
"""

import importlib
import logging
from pathlib import Path
from typing import Any, Callable, cast

from langres._version import __version__ as LANGRES_VERSION
from langres.core._artifacts import (
    component_spec,
    op_spec,
    op_state_owner,
    rebuild_component,
    rebuild_op,
    state_owner,
)
from langres.core._model_state import ModelState
from langres.core.registry import model_type_name
from langres.core.serialization import (
    ARTIFACT_VERSION,
    CLASSIC_ARTIFACT_VERSION,
    ArtifactManifest,
    ComponentSpec,
)

logger = logging.getLogger(__name__)

# Slot names double as sidecar subdirectory names and drive manifest ordering.
_MANIFEST_FILENAME = "resolver.json"


def _hub_callable(name: str) -> Callable[..., Any]:
    """Resolve one outer Hub adapter function without a core -> Hub import edge."""
    function = getattr(importlib.import_module("langres.hub"), name, None)
    if not callable(function):
        raise RuntimeError(f"langres.hub does not provide callable {name!r}")
    return cast(Callable[..., Any], function)


class ModelPersistence(ModelState):
    """``save`` / ``load`` / ``config_dict`` for an ``ERModel``."""

    def save_pretrained(
        self,
        path: str | Path,
        *,
        measurement_summary: object | None = None,
        model_card: object | None = None,
        claim_level: str = "reference-only",
        allow_sensitive_config: bool = False,
        overwrite: bool = False,
    ) -> Path:
        """Write a validated shareable bundle using the optional Hub lifecycle.

        Dispatch is resolved only when called, so local Resolver construction
        and ``import langres`` never import the Hub adapter or optional client.
        """
        save_pretrained = _hub_callable("save_pretrained")
        return cast(
            Path,
            save_pretrained(
                self,
                path,
                measurement_summary=measurement_summary,
                model_card=model_card,
                claim_level=claim_level,
                allow_sensitive_config=allow_sensitive_config,
                overwrite=overwrite,
            ),
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_or_path: str | Path,
        *,
        revision: str | None = None,
        token: str | None = None,
        transport: object | None = None,
    ) -> Any:
        """Load a local or immutable Hub bundle through the safe outer manifest."""
        from_pretrained = _hub_callable("from_pretrained")
        model = from_pretrained(
            repo_or_path,
            revision=revision,
            token=token,
            transport=transport,
        )
        if not isinstance(model, cls):
            raise TypeError(
                f"artifact reconstructed {type(model).__name__}, not requested {cls.__name__}"
            )
        return model

    def push_to_hub(
        self,
        repo_id: str,
        **kwargs: object,
    ) -> Any:
        """Build and upload a fresh validated bundle through a lazy Hub transport."""
        push_to_hub = _hub_callable("push_to_hub")
        return push_to_hub(self, repo_id, **kwargs)

    def _slots(self) -> list[tuple[str, object]]:
        """Ordered (slot_name, component) pairs, skipping absent optional slots.

        The slot name doubles as the sidecar subdirectory name for components
        that own out-of-band state. The comparator and calibrator are optional
        slots, emitted only when set; the clusterer stays last so the legacy
        positional load fallback (``ordered[-1]`` is the clusterer) still holds.
        """
        slots: list[tuple[str, object]] = [("blocker", self.blocker)]
        if self.comparator is not None:
            slots.append(("comparator", self.comparator))
        slots.append(("module", self.module))
        if self.calibrator is not None:
            slots.append(("calibrator", self.calibrator))
        slots.append(("clusterer", self.clusterer))
        return slots

    def _build_manifest(self) -> ArtifactManifest:
        """Assemble the in-memory :class:`ArtifactManifest` (no disk I/O).

        Shared by :meth:`save` (which writes it, plus sidecars) and
        :meth:`config_dict` (which returns it as a dict). Branches on the model's
        topology:

        - **Classic four-slot** (``self._ops is None``): serializes each slot into
          a :class:`ComponentSpec` via
          :func:`~langres.core._artifacts.component_spec` and stamps
          :data:`~langres.core.serialization.CLASSIC_ARTIFACT_VERSION` (``"1"``).
          Deliberately NOT :data:`~langres.core.serialization.ARTIFACT_VERSION`
          (now ``"2"``): a classic save's on-disk bytes — and the ``recipe_id``
          derived from its ``config_dict`` — must not restamp when the reader's max
          advances (F1).
        - **Explicit Op chain** (``self._ops is not None``): serializes each stage
          into an :class:`~langres.core.serialization.OpSpec` via
          :func:`~langres.core._artifacts.op_spec` and stamps ``ARTIFACT_VERSION``
          (``"2"``).

        Either builder raises :class:`TypeError` for a component/stage lacking a
        registry ``type_name`` (or an unserializable stage) — intentional, not
        swallowed here.

        Also stamps ``model_class`` with this class's registered model name so a
        named architecture survives ``save``/``load``. It is ``None`` for the
        base ``Resolver`` and for any unregistered subclass — see
        :func:`~langres.core.registry.model_type_name`.
        """
        model_class = model_type_name(type(self))
        if self._ops is not None:
            return ArtifactManifest(
                artifact_version=ARTIFACT_VERSION,
                langres_version=LANGRES_VERSION,
                model_class=model_class,
                ops=[op_spec(stage) for stage in self._ops],
                replay_boundary=self._replay_boundary_index,
            )
        components = [
            component_spec(component, slot=slot_name) for slot_name, component in self._slots()
        ]
        return ArtifactManifest(
            artifact_version=CLASSIC_ARTIFACT_VERSION,
            langres_version=LANGRES_VERSION,
            model_class=model_class,
            components=components,
        )

    def config_dict(self) -> dict[str, object]:
        """Return the resolver's hash-safe config snapshot, WITHOUT writing to disk.

        Returns only the reproducible *config* the manifest wraps — the ordered
        per-slot ``type_name`` + construction config under a ``components`` key —
        and deliberately **omits** the volatile version/provenance envelope
        (``artifact_version``, ``langres_version``) that :meth:`save` writes to
        ``resolver.json``.

        This is by design: the tracking layer feeds this dict to
        ``RunContext.resolver_config``, which is inside
        :func:`~langres.tracking.runs.compute_recipe_id`'s hash domain. Emitting the
        version fields would fork ``recipe_id`` on every package or
        artifact-schema bump, silently defeating idempotent replay. Version and
        provenance live on :class:`~langres.tracking.runs.RunContext` as separate,
        **unhashed** fields (e.g. ``RunContext.langres_version``); :meth:`save`
        still records them on disk for artifact reconstruction.

        Known limitation: this captures **declared** component config, not
        compiled/optimized in-memory state — e.g. a DSPy-compiled program's tuned
        prompts do not appear here. Persisting that state is out of scope for the
        config snapshot (it round-trips via
        :class:`~langres.core.serialization.SerializableState` sidecars in
        :meth:`save`, not through this dict).

        Returns:
            A plain, JSON-serializable dict with a single key: ``components`` for a
            classic four-slot model (the ordered slot specs, each a ``type_name`` +
            ``config``) or ``ops`` for an explicit-chain model (the ordered stage
            specs). No version fields — see above.

        Raises:
            TypeError: If a slot component / stage lacks a registry ``type_name``
                (same contract as :meth:`save`; not swallowed).
        """
        # The explicit-chain snapshot keys on ``ops``; the classic snapshot returns
        # the byte-identical ``{"components": ...}`` dict it always has, so a classic
        # model's recipe_id never forks (F2 — the crown invariant).
        if self._ops is not None:
            manifest = self._build_manifest()
            snapshot: dict[str, object] = {"ops": manifest.model_dump()["ops"]}
            if manifest.replay_boundary is not None:
                snapshot["replay_boundary"] = manifest.replay_boundary
            return snapshot
        return {"components": self._build_manifest().model_dump()["components"]}

    def save(self, path: str | Path) -> None:
        """Persist the whole pipeline to ``path`` as a self-describing artifact.

        Writes ``resolver.json`` (a full :class:`ArtifactManifest`, including the
        ``artifact_version`` + ``langres_version`` envelope that
        :meth:`config_dict` intentionally omits) plus, for any slot component that
        implements
        :class:`~langres.core.serialization.SerializableState`, a sidecar state
        directory named after the slot. The manifest records, per slot, the
        component ``type_name`` and config (the embedder persists by
        ``model_name`` only — no model bytes), plus this class's registered
        ``model_class`` when it has one, so :meth:`load` can rebuild the same
        architecture rather than a plain ``Resolver``.

        An **explicit-chain** model (``self._ops is not None``, epic #193 persist
        v2) writes the same ``resolver.json`` with an ``ops`` list instead of
        ``components``, and keys any sidecars by stage **ordinal** (``op{i}``)
        rather than slot name. Its paid Scores are serialized RAW (the spend cap is
        unwrapped, never written); ``load`` re-establishes the cap via
        ``from_topology``.

        Args:
            path: Directory to write the artifact into (created if absent).
        """
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build first: op_spec / component_spec raise on an unserializable
        # stage/component BEFORE any file is written (no half-written artifact).
        manifest = self._build_manifest()
        if self._ops is not None:
            # Explicit chain: a stage has no slot name, so sidecars key by ORDINAL.
            # Same state_owner/has_state seam via op_state_owner (a built
            # VectorBlocker Source is the one stateful case in practice).
            for index, stage in enumerate(self._ops):
                owner = op_state_owner(stage)
                if owner is not None:
                    state_dir = out_dir / f"op{index}"
                    state_dir.mkdir(parents=True, exist_ok=True)
                    owner.save_state(state_dir)
                    if not any(state_dir.iterdir()):
                        state_dir.rmdir()
            # Drop the (empty) ``components`` field from the explicit artifact.
            payload = manifest.model_dump_json(indent=2, exclude={"components"})
        else:
            for slot_name, component in self._slots():
                owner = state_owner(component)
                if owner is not None:
                    state_dir = out_dir / slot_name
                    state_dir.mkdir(parents=True, exist_ok=True)
                    owner.save_state(state_dir)
                    # A SerializableState owner with nothing to persist (e.g. a
                    # VectorBlocker whose index was never built) writes no files;
                    # drop the empty sidecar so load() doesn't later try to read a
                    # missing state file from a dir that only signals "has state".
                    if not any(state_dir.iterdir()):
                        state_dir.rmdir()
            # Exclude the ``ops`` field (defaults to None) so a classic
            # resolver.json stays byte-identical to pre-#193 — without this every
            # classic save would gain an ``"ops": null`` line and fork the frozen
            # legacy bytes (F1 / the crown invariant).
            payload = manifest.model_dump_json(
                indent=2,
                exclude={"ops", "replay_boundary"},
            )

        (out_dir / _MANIFEST_FILENAME).write_text(payload)
        logger.info("Saved Resolver artifact to %s", out_dir)

    @classmethod
    def _read_artifact(cls, path: str | Path) -> tuple[ArtifactManifest, dict[str, Any]]:
        """Read ``resolver.json`` and rebuild every slot component from it.

        The *mechanism* half of ``ERModel.load``: validate the artifact version,
        map each :class:`ComponentSpec` back to its slot, and rebuild the
        component from the registry by ``type_name`` (no code execution, no
        pickle), restoring any sidecar state.

        The *policy* half — which class to instantiate — stays at the leaf in
        :meth:`~langres.core.resolver.ERModel.load`, deliberately: choosing it
        means naming ``ERModel`` to check the registered ``model_class`` really
        is one, and this module sits *beneath* the leaf that defines ``ERModel``.
        Importing it back here (even under ``TYPE_CHECKING``) would re-knot the
        import graph that ``tests/test_import_tangle.py`` pins at zero runtime
        cycles. So this returns the parts; the leaf picks the class and assembles.

        Returns:
            The validated manifest, and the load kwargs. For a classic four-slot
            artifact that is the ``from_components`` kwargs
            (``blocker``/``comparator``/``matcher``/``clusterer``/``calibrator``);
            for an explicit-chain artifact (``manifest.ops is not None``, persist
            v2) it is ``{"ops": [<rebuilt Stage>, ...]}`` for ``from_topology`` —
            the leaf in :meth:`~langres.core.resolver.ERModel.load` dispatches on
            which key is present.

        Raises:
            ValueError: If the artifact's ``artifact_version`` is unreadable by
                this build, or the manifest is malformed.
        """
        in_dir = Path(path)
        manifest = ArtifactManifest.model_validate_json((in_dir / _MANIFEST_FILENAME).read_text())
        cls._check_versions(manifest)

        # Explicit Op chain (persist v2): rebuild each stage by ordinal, restoring
        # any per-stage sidecar (op{i}). Each MatcherScore comes back with its RAW
        # matcher; the leaf's from_topology re-wraps the cap against a fresh ledger.
        if manifest.ops is not None:
            ops = [
                rebuild_op(spec, state_dir=in_dir / f"op{index}")
                for index, spec in enumerate(manifest.ops)
            ]
            return manifest, {
                "ops": ops,
                "replay_boundary": manifest.replay_boundary,
            }

        # Map specs back to slots self-describingly. Each spec written by a
        # current ``save`` carries its ``slot`` name, so a registered subclass
        # with a custom ``type_name`` (e.g. a "phonetic_comparator" Comparator)
        # still loads into the right slot. Older/hand-written manifests have no
        # ``slot``; those fall back to positional + type_name identification.
        calibrator_spec: ComponentSpec | None = None
        by_slot = {spec.slot: spec for spec in manifest.components if spec.slot}
        if by_slot:
            blocker_spec = by_slot.get("blocker")
            comparator_spec = by_slot.get("comparator")
            module_spec = by_slot.get("module")
            clusterer_spec = by_slot.get("clusterer")
            calibrator_spec = by_slot.get("calibrator")
            if blocker_spec is None or module_spec is None or clusterer_spec is None:
                raise ValueError(
                    "Malformed artifact manifest: missing required slot among "
                    f"{[(c.slot, c.type_name) for c in manifest.components]}"
                )
        else:
            # Legacy fallback: the comparator slot is present iff a spec has
            # type_name == "comparator"; everything else is positional.
            by_type = {spec.type_name: spec for spec in manifest.components}
            comparator_spec = by_type.get("comparator")
            ordered = list(manifest.components)
            blocker_spec = ordered[0]
            clusterer_spec = ordered[-1]
            module_spec = next(
                (
                    spec
                    for spec in ordered
                    if spec not in (blocker_spec, clusterer_spec, comparator_spec)
                ),
                None,
            )
            if module_spec is None:
                raise ValueError(
                    f"Malformed artifact manifest: cannot identify a module spec among "
                    f"{[c.type_name for c in manifest.components]}"
                )

        blocker = rebuild_component(blocker_spec, state_dir=in_dir / "blocker")
        comparator = (
            rebuild_component(comparator_spec, state_dir=in_dir / "comparator")
            if comparator_spec is not None
            else None
        )
        module = rebuild_component(module_spec, state_dir=in_dir / "module")
        clusterer = rebuild_component(clusterer_spec, state_dir=in_dir / "clusterer")
        calibrator = (
            rebuild_component(calibrator_spec, state_dir=in_dir / "calibrator")
            if calibrator_spec is not None
            else None
        )

        return manifest, {
            "blocker": blocker,
            "comparator": comparator,
            "matcher": module,
            "clusterer": clusterer,
            "calibrator": calibrator,
        }

    @staticmethod
    def _check_versions(manifest: ArtifactManifest) -> None:
        """Validate artifact compatibility; raise outside the readable version window.

        ``ARTIFACT_VERSION`` (the reader's max, ``"2"``) is a monotonic
        integer-valued string. The reader accepts any layout in the **window**
        ``[CLASSIC_ARTIFACT_VERSION, ARTIFACT_VERSION]`` = ``[1, 2]``:

        - **Too new** (``> ARTIFACT_VERSION``): this build cannot read a layout it
          predates — a hard "upgrade langres" error.
        - **Below the window** (``< 1``, e.g. a pre-M0.5 ``"0"`` with incompatible
          configs): a hard "predates the readable layout" error, rather than
          falling through to a raw ``KeyError`` on the changed config.
        - **Malformed/non-integer**: a hard error.

        The change #193 made here: v1 is now *inside* the window (v2 is an
        **additive** layout — it only adds an ``ops`` list — so a v1
        ``components`` artifact still reads unchanged). Before, ``artifact_v <
        current_v`` rejected every older layout; that was correct only while each
        bump was config-breaking, which v1→v2 is not.

        A ``langres_version`` mismatch is logged as a warning, not a failure —
        configs are forward-compatible *within* the window.
        """
        try:
            artifact_v = int(manifest.artifact_version)
            current_v = int(ARTIFACT_VERSION)
            min_v = int(CLASSIC_ARTIFACT_VERSION)
        except ValueError:  # malformed/non-integer layout version -> incompatible.
            raise ValueError(
                f"Artifact version {manifest.artifact_version!r} differs from "
                f"supported {ARTIFACT_VERSION!r}; cannot load."
            ) from None
        if artifact_v > current_v:
            raise ValueError(
                f"Artifact version {manifest.artifact_version!r} is newer than this "
                f"langres build supports ({ARTIFACT_VERSION!r}); upgrade langres to load it."
            )
        if artifact_v < min_v:
            raise ValueError(
                f"Artifact version {manifest.artifact_version!r} predates the supported "
                f"layout (readable range {CLASSIC_ARTIFACT_VERSION!r}..{ARTIFACT_VERSION!r}) and "
                f"is no longer readable (the config schema changed incompatibly); re-save with "
                f"this langres build."
            )
        if manifest.langres_version != LANGRES_VERSION:
            logger.warning(
                "Loading artifact written by langres %s into langres %s; "
                "configs are forward-compatible within artifact version %s.",
                manifest.langres_version,
                LANGRES_VERSION,
                ARTIFACT_VERSION,
            )
