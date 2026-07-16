"""Resolver: the M0 spine that composes a full entity-resolution pipeline.

The Resolver is the **top-level container** of langres.core. It wires four
slots into one runnable, serializable pipeline:

    blocker      -> candidate generation + schema normalization
    comparator   -> (optional) missing-aware per-feature comparison
    module       -> the scorer (a Matcher yielding PairwiseJudgements)
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

import inspect
import logging
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, TypeGuard, cast

from pydantic import BaseModel

from langres._version import __version__ as LANGRES_VERSION
from langres.core.blocker import Blocker
from langres.core.blockers.composite import CompositeBlocker
from langres.core.clusterer import Clusterer
from langres.core.comparator import Comparator
from langres.core.comparators import StringComparator
from langres.core.fit import (
    BlockerFitMixin,
    CalibratorFitMixin,
    SupervisedFitMixin,
    UnsupervisedFitMixin,
)
from langres.core.fit_report import CalibrationDelta, FitReport
from langres.core.harvest import Correction, LabeledPair, align_pairs
from langres.core.methods_api import Method
from langres.core.matchers.weighted_average import WeightedAverageMatcher
from langres.core.metrics import (
    PairMetrics,
    brier_score,
    classify_pairs,
    expected_calibration_error,
)
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.matcher import Matcher
from langres.core.registry import get_component
from langres.core.runs import current_run
from langres.core.spend import SpendMonitor
from langres.core.spend_cap import SpendCappedMatcher, effective_budget
from langres.core.serialization import (
    ARTIFACT_VERSION,
    ArtifactManifest,
    ComponentSpec,
    SerializableState,
)

if TYPE_CHECKING:
    # [semantic] extra (faiss/sentence-transformers/torch) -- imported lazily
    # inside _build_embedding_blocker and _ensure_index_built (W0.4) so a
    # core-only `import langres` never pulls faiss/torch in for a Resolver
    # that never uses matcher="embedding".
    from langres.core.anchor_store import AnchorStore, ClusterDelta
    from langres.core.blockers.vector import VectorBlocker

logger = logging.getLogger(__name__)

# Slot names used as sidecar subdirectory names and for manifest ordering.
_MANIFEST_FILENAME = "resolver.json"

#: ``Resolver.from_schema``'s low-level judge switch. Deliberately narrower
#: than ``langres.core.presets.MatcherName`` -- no ``"auto"``, since resolving
#: that needs ``Settings``/env-var lookups, which is verb-layer magic (see
#: ``langres.core.presets.choose_auto_judge``); this stays a plain, explicit
#: constructor argument.
_FromSchemaJudge = Literal["string", "embedding", "zero_shot_llm", "prompt_llm", "random_forest"]


def _build_module_for_judge(
    judge: "_FromSchemaJudge | Matcher[Any]",
    schema: type[BaseModel],
    comparator: Comparator[Any],
    *,
    model: str | None,
    entity_noun: str,
    judge_params: dict[str, Any] | None = None,
) -> Matcher[Any]:
    """Build the scorer for ``Resolver.from_schema``'s ``matcher=`` slot.

    Construction is delegated to the one
    :mod:`~langres.core.method_registry` (a core leaf, so no
    ``Resolver -> presets`` cycle -- the pre-registry duplication this switch
    used to carry is gone); this function keeps only ``from_schema``'s policy:
    the allowed names (no ``"auto"``) and the uncapped-spend warning below.
    ``comparator`` is passed to the spec builder so custom
    ``weights=``/``exclude=`` flow into feature-spec-driven judges.
    """
    if isinstance(judge, Matcher):
        return judge
    if judge not in ("string", "embedding", "zero_shot_llm", "prompt_llm", "random_forest"):
        raise ValueError(
            f"unsupported judge {judge!r} for Resolver.from_schema; choose one of "
            "'string', 'embedding', 'zero_shot_llm', 'prompt_llm', 'random_forest', "
            "or pass a Matcher instance. 'auto' key-based resolution is a verbs-layer "
            "feature -- use langres.link/langres.dedupe for that."
        )
    from langres.core.method_registry import get_method

    if judge == "zero_shot_llm":
        from langres.clients.openrouter import DEFAULT_OPENROUTER_MODEL, dspy_price_per_1k

        resolved_model = model or DEFAULT_OPENROUTER_MODEL
        if dspy_price_per_1k(resolved_model) == 0.0:
            # An unpinned model self-reports $0/pair -- honest, not reassuring
            # (mirrors core.presets.notice_pre_scoring_cost's identical check).
            # The Resolver IS spend-capped now (B1), but a cap fed $0 costs can
            # never trip: the pipeline is capped on paper and blind in practice.
            warnings.warn(
                f"model {resolved_model!r} has no pinned price in "
                "langres.clients.openrouter.PRICES_PER_1M, so it self-reports "
                "$0/pair cost -- the Resolver's budget_usd spend cap tallies that "
                "same $0 and can NEVER trip, so it will not stop a runaway bill. "
                "Pin its price in PRICES_PER_1M, or use a model that already is, "
                "to get real spend-cap protection.",
                stacklevel=3,
            )
    return get_method(judge).build(
        schema,
        model=model,
        entity_noun=entity_noun,
        client=None,
        comparator=comparator,
        **(judge_params or {}),
    )


def _build_embedding_blocker(schema: type[BaseModel]) -> "VectorBlocker[Any]":
    """Build the ``VectorBlocker`` a ``matcher="embedding"`` pipeline needs.

    ``AllPairsBlocker``'s candidates never carry ``similarity_score``, which
    ``EmbeddingScoreMatcher`` requires to score -- ``matcher="embedding"`` must
    always be paired with a ``VectorBlocker``, mirroring the identical rule
    ``core.presets.build_resolver`` applies for the verb layer (same model,
    same k, same cosine metric). Duplicated here rather than imported from
    ``core.presets`` for the same layering reason as
    :func:`_build_module_for_judge`: ``core.presets`` sits ABOVE ``Resolver``
    and must not be imported back into it.
    """
    from langres.core.blockers.vector import VectorBlocker
    from langres.core.embeddings import SentenceTransformerEmbedder
    from langres.core.indexes.vector_index import FAISSIndex

    field_names = [spec.name for spec in StringComparator.from_schema(schema).feature_specs]

    def extract(entity: Any) -> str:
        parts = [str(getattr(entity, name)) for name in field_names if getattr(entity, name, None)]
        return " ".join(parts)

    from langres.core.method_registry import DEFAULT_EMBEDDING_MODEL

    embedder = SentenceTransformerEmbedder(DEFAULT_EMBEDDING_MODEL)
    index = FAISSIndex(embedder=embedder, metric="cosine")
    return VectorBlocker(
        vector_index=index, schema=schema, text_field_extractor=extract, k_neighbors=10
    )


def _iter_vector_blockers(blocker: object) -> "Iterator[VectorBlocker[Any]]":
    """Yield every ``VectorBlocker`` reachable from ``blocker``.

    A plain ``VectorBlocker`` yields itself. A ``CompositeBlocker`` (the
    blocking-algebra union/intersection/difference of child blockers) recurses
    into ``children`` at arbitrary depth, so a composite wrapping another
    composite still surfaces every nested ``VectorBlocker`` -- e.g. the
    recall-first pattern ``CompositeBlocker([KeyBlocker(...), VectorBlocker(...)],
    op="union")``. Index-free blockers (``AllPairsBlocker``, ``KeyBlocker``,
    ``GLinkerAdapter``) contribute nothing.

    Checks ``type_name`` rather than ``isinstance(blocker, VectorBlocker)``
    deliberately (W0.4, mirrors :meth:`Resolver._ensure_index_built`'s own
    docstring): this walks the blocker tree on every ``resolve()``/
    ``predict()`` call, so an ``isinstance`` check would need ``VectorBlocker``
    imported unconditionally, pulling faiss/sentence-transformers (the
    ``[semantic]`` extra) into a plain ``AllPairsBlocker``/``KeyBlocker``
    pipeline. ``CompositeBlocker`` itself has no heavy dependency, so it's
    safe to ``isinstance``-check directly.
    """
    if getattr(blocker, "type_name", None) == "vector_blocker":
        yield cast("VectorBlocker[Any]", blocker)
    elif isinstance(blocker, CompositeBlocker):
        for child in blocker.children:
            yield from _iter_vector_blockers(child)


def _component_config_dict(obj: object) -> dict[str, object]:
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


def _component_spec(obj: object, slot: str) -> ComponentSpec:
    """Serialize any Resolver-slot component into a :class:`ComponentSpec`.

    Reads the component's ``type_name`` class attribute (the registry key) and
    its construction config (via :func:`_component_config_dict`), and records the
    ``slot`` name so :meth:`Resolver.load` can map the spec back self-describingly
    rather than by position or hard-coded ``type_name``.
    """
    type_name = getattr(obj, "type_name", None)
    if not isinstance(type_name, str):
        raise TypeError(
            f"{type(obj).__name__} is not serializable (no `type_name`/@register). "
            f"Use a registered component (e.g. LLMMatcher, WeightedAverageMatcher) in "
            f"the {slot!r} slot."
        )
    return ComponentSpec(type_name=type_name, slot=slot, config=_component_config_dict(obj))


def _state_owner(component: object) -> SerializableState | None:
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


def _has_state(state_dir: Path | None) -> TypeGuard[Path]:
    """True iff ``state_dir`` exists and holds at least one persisted state file.

    An empty (or absent) sidecar dir signals "no out-of-band state to restore",
    so callers must not invoke ``load_state`` on it — that would try to read a
    missing state file (e.g. ``index.faiss``). Returning a ``TypeGuard`` narrows
    ``state_dir`` to ``Path`` in the truthy branch for the type checker.
    """
    return state_dir is not None and state_dir.is_dir() and any(state_dir.iterdir())


def _rebuild_component(spec: ComponentSpec, state_dir: Path | None = None) -> Any:
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
    if accepts_state_dir and _has_state(state_dir):
        component = cls.from_config(config_arg, state_dir=state_dir)  # type: ignore[attr-defined]
    else:
        component = cls.from_config(config_arg)  # type: ignore[attr-defined]

    # Restore directly only when ``from_config`` did not already handle state
    # itself (guards against a double ``load_state`` for a component that both
    # accepts ``state_dir`` and implements SerializableState).
    if not accepts_state_dir and isinstance(component, SerializableState) and _has_state(state_dir):
        component.load_state(state_dir)
    return component


def _is_prompt_compilable(module: object) -> bool:
    """Whether ``module`` is a prompt-optimizable (DSPy-style) matcher.

    Structural, import-light check (no ``dspy`` import): a compilable scorer
    exposes a ``compile(trainset, ...)`` method and a ``compiled`` flag -- the
    :class:`~langres.core.matchers.dspy_judge.DSPyMatcher` shape. Used by
    :meth:`Resolver.describe` to tag the matcher TRAINABLE and mirrors the
    matcher the ``method.kind == "prompt"`` fit path requires, without pulling
    ``dspy`` into a bare ``import langres``.
    """
    return callable(getattr(module, "compile", None)) and hasattr(module, "compiled")


class Resolver:
    """Composable entity-resolution pipeline: blocker -> compare -> score -> cluster.

    Args:
        blocker: Candidate generator + schema normalizer.
        comparator: Optional pre-stage turning each pair into a
            ComparisonVector. When ``None``, the module is called directly
            (e.g. a self-contained ``RapidfuzzMatcher``).
        matcher: The scorer Matcher that yields PairwiseJudgements.
        clusterer: Groups matched pairs into entity clusters.
        calibrator: Optional fitted
            :class:`~langres.core.fit.CalibratorFitMixin` that maps each
            judgement's raw ``score`` to a calibrated probability before
            clustering. ``None`` (the default) leaves scores untouched; set by
            ``fit(method=Platt()/Isotonic())``.

    Example:
        comparator = StringComparator.from_schema(CompanySchema, weights={"name": 0.6, ...})
        resolver = Resolver(
            blocker=AllPairsBlocker(schema=CompanySchema),
            comparator=comparator,
            matcher=WeightedAverageMatcher(feature_specs=comparator.feature_specs),
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
        matcher: Matcher[Any],
        clusterer: Clusterer,
        calibrator: CalibratorFitMixin | None = None,
        *,
        budget_usd: float | None = None,
    ) -> None:
        """Wire four components into one runnable pipeline.

        Args:
            blocker: Candidate generation + schema normalization.
            comparator: Optional missing-aware per-feature comparison.
            matcher: The scorer.
            clusterer: Connected-components grouping.
            calibrator: Optional score->probability map (set by ``fit``).
            budget_usd: **Spend cap for this Resolver's whole lifetime**, in
                USD. ``None`` (the default) resolves to
                :data:`~langres.core.spend_cap.DEFAULT_BUDGET_USD` -- it does
                NOT mean "uncapped"; pass
                :data:`~langres.core.spend_cap.UNCAPPED_BUDGET_USD`
                (``float("inf")``) for that, deliberately and in writing. A
                free matcher (string/embedding) meters $0 and never trips.
        """
        self.blocker = blocker
        self.comparator = comparator
        self.module = matcher
        self.clusterer = clusterer
        # ONE ledger for this Resolver's lifetime, so N resolve() calls share
        # one budget instead of getting a fresh one each (B1). The monitor --
        # not the wrapper -- is the durable thing: `self.module` is reassignable
        # (dedupe() wraps it in a LoggingMatcher; distil() replaces it), so
        # _judgements() re-wraps the CURRENT module around this same ledger.
        self._spend_monitor = SpendMonitor(budget_usd=effective_budget(budget_usd))
        # Optional score->probability map, set by fit(method=Platt()/Isotonic())
        # and applied in _judgements(); None leaves raw scores untouched.
        self.calibrator = calibrator
        # Set by fit(); the sklearn trailing-underscore "produced by fit" digest.
        # None until fit() runs (never serialized -- it is a fit-time artifact).
        self.fit_report_: FitReport | None = None
        # Set by build_anchor_store(); the incremental-assign state assign() uses.
        # Quoted: AnchorStore is a TYPE_CHECKING-only import (avoids an import cycle).
        self._anchor_store: "AnchorStore | None" = None

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
        matcher: "_FromSchemaJudge | Matcher[Any]" = "string",
        model: str | None = None,
        entity_noun: str = "entity",
        prompt_template: str | None = None,
        system_prompt: str | None = None,
        response_parser: str | None = None,
        budget_usd: float | None = None,
    ) -> "Resolver":
        """Build a default dedup Resolver from a Pydantic schema in one line.

        Defaults to an ``AllPairsBlocker`` over the schema, a missing-aware
        ``StringComparator`` auto-derived from the schema's string fields (with
        ``id`` excluded), a ``WeightedAverageMatcher`` scorer, and a ``Clusterer``
        at ``threshold``. ``matcher="embedding"`` is the one exception to the
        ``AllPairsBlocker`` default: it wires a ``VectorBlocker`` instead,
        since ``EmbeddingScoreMatcher`` scores off the blocker's
        ``similarity_score``, which only a ``VectorBlocker`` attaches.

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
            matcher: ``"string"`` (default -- identical to pre-existing
                behavior), ``"embedding"`` (wires a ``VectorBlocker``, see
                above), ``"zero_shot_llm"``, ``"prompt_llm"`` (the
                bring-your-own-prompt ``LLMMatcher`` -- with a *registered*
                ``response_parser`` name the whole judge, prompt included,
                ``save``/``load`` round-trips), ``"random_forest"`` (a
                supervised sklearn ``RandomForestMatcher`` over the comparator's
                per-feature similarities -- needs the ``[trained]`` extra and is
                TRAINABLE, so ``fit(records, pairs=...)``/``labels=...`` it with
                labeled data before it can score), or a ``Matcher`` instance. This
                is the low-level, explicit switch: no ``"auto"`` key-based
                resolution (that magic stays in ``langres.link``/``langres.dedupe``)
                -- but a paid matcher IS spend-capped here, via ``budget_usd``.
            model: Model id override for ``matcher="zero_shot_llm"``/``"prompt_llm"``.
            entity_noun: Domain noun woven into the LLM judge's prompt.
            prompt_template: ``matcher="prompt_llm"`` only: custom prompt with
                ``{left}``/``{right}`` placeholders (see
                :class:`~langres.core.matchers.llm_judge.LLMMatcher`).
            system_prompt: ``matcher="prompt_llm"`` only: optional system message.
            response_parser: ``matcher="prompt_llm"`` only: a *registered*
                parser name (``"score"`` / ``"binary_yes_no"`` -- see
                ``llm_judge.RESPONSE_PARSERS``), serialized in the artifact.
            budget_usd: Spend cap for the returned Resolver's whole lifetime
                (see :meth:`__init__`). ``None`` resolves to
                :data:`~langres.core.spend_cap.DEFAULT_BUDGET_USD`, NOT
                "uncapped".

        Returns:
            A ready-to-run Resolver.

        Raises:
            ValueError: For an unsupported ``judge`` name, or a prompt-seam
                kwarg with a non-``"prompt_llm"`` judge (never silently
                ignored).
        """
        from langres.core.blockers.all_pairs import AllPairsBlocker

        judge_params = {
            key: value
            for key, value in {
                "prompt_template": prompt_template,
                "system_prompt": system_prompt,
                "response_parser": response_parser,
            }.items()
            if value is not None
        }
        if judge_params and matcher != "prompt_llm":
            raise ValueError(
                f"{', '.join(sorted(judge_params))}: only valid with matcher='prompt_llm' "
                f"(got matcher={matcher!r})."
            )
        comparator: Comparator[Any] = StringComparator.from_schema(
            schema, exclude=exclude, weights=weights
        )
        module = _build_module_for_judge(
            matcher,
            schema,
            comparator,
            model=model,
            entity_noun=entity_noun,
            judge_params=judge_params or None,
        )
        blocker: Blocker[Any] = (
            _build_embedding_blocker(schema)
            if matcher == "embedding"
            else AllPairsBlocker(schema=schema)
        )
        return cls(
            blocker=blocker,
            comparator=comparator,
            matcher=module,
            clusterer=Clusterer(threshold=threshold),
            budget_usd=budget_usd,
        )

    # ------------------------------------------------------------------
    # Running the pipeline
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """Return a per-component "what would train vs what is frozen" digest.

        The honesty device the caller reads *before* ``fit``: one line per
        pipeline role naming the component and tagging it ``TRAINABLE`` (a fit
        hook or a prompt-compile would tune it) or ``frozen`` (nothing to train).
        A role is TRAINABLE when it implements the matching fit Protocol from
        :mod:`langres.core.fit` -- a :class:`~langres.core.fit.BlockerFitMixin`
        blocker, a :class:`~langres.core.fit.SupervisedFitMixin`/
        :class:`~langres.core.fit.UnsupervisedFitMixin` matcher (or a
        prompt-compilable :class:`~langres.core.matchers.dspy_judge.DSPyMatcher`,
        tuned by ``fit(method="prompt")``), or a
        :class:`~langres.core.fit.CalibratorFitMixin` calibrator. The clusterer is
        always frozen (a decision threshold, not a learned parameter).

        Pure string builder: it reads slots and reports, never trains, imports a
        backend, or mutates anything -- safe to call on a fresh Resolver. Example::

            blocker:    AllPairsBlocker         — frozen
            matcher:    DSPyMatcher             — TRAINABLE
            calibrator: <none>                  — frozen
            clusterer:  threshold=0.5           — frozen

        Returns:
            A newline-joined, column-aligned digest (no trailing newline).
        """
        calibrator = getattr(self, "calibrator", None)
        matcher_trainable = isinstance(
            self.module, (SupervisedFitMixin, UnsupervisedFitMixin)
        ) or _is_prompt_compilable(self.module)
        rows: list[tuple[str, str, bool]] = [
            ("blocker", type(self.blocker).__name__, isinstance(self.blocker, BlockerFitMixin)),
            ("matcher", type(self.module).__name__, matcher_trainable),
            (
                "calibrator",
                "<none>" if calibrator is None else type(calibrator).__name__,
                calibrator is not None and isinstance(calibrator, CalibratorFitMixin),
            ),
            ("clusterer", f"threshold={self.clusterer.threshold:g}", False),
        ]
        label_w = max(len(label) for label, _, _ in rows) + 1  # +1 for the trailing ":"
        desc_w = max(len(desc) for _, desc, _ in rows)
        return "\n".join(
            f"{label + ':':<{label_w}} {desc:<{desc_w}} — {'TRAINABLE' if trainable else 'frozen'}"
            for label, desc, trainable in rows
        )

    def fit(
        self,
        data: list[Any],
        labels: Sequence[bool] | None = None,
        *,
        pairs: str | Path | Sequence[LabeledPair] | Sequence[Correction] | None = None,
        split: float | None = None,
        seed: int = 0,
        method: Method | None = None,
    ) -> Self:
        """Fit the module when it supports a fit hook; sklearn-style no-op otherwise.

        Every non-raising path sets :attr:`fit_report_` (an sklearn
        trailing-underscore, produced-by-fit digest) and returns ``self`` so
        ``resolver.fit(...).resolve(...)`` still chains. Delegates to the module's
        fit hook when it implements one of the runtime-checkable Protocols in
        :mod:`langres.core.fit` (W1.0, E6):

        - :class:`~langres.core.fit.UnsupervisedFitMixin`
          (``fit_unlabeled(candidates)``): called with the blocked (and, if a
          comparator is configured, comparison-attached) candidate stream.
          ``labels``/``pairs`` are not used by this path (passing either raises).
        - :class:`~langres.core.fit.SupervisedFitMixin` (``fit(candidates,
          labels)``): trained from either pre-aligned ``labels`` or id-keyed
          ``pairs`` (see below); **raises** rather than silently skipping when
          neither is given -- a genuinely trainable module that never gets trained
          is exactly the silent-no-op footgun this hook exists to prevent.

        Two ways to supply supervision for a ``SupervisedFitMixin`` matcher:

        - ``labels``: a ``Sequence[bool]`` the caller has *already* positionally
          aligned with the blocked candidates (the pre-existing contract). No
          id-join happens, so the report carries no ``coverage``.
        - ``pairs``: id-keyed labels (a ``corrections.jsonl`` path, or a
          ``Sequence`` of :class:`~langres.core.harvest.LabeledPair` /
          :class:`~langres.core.harvest.Correction`) that
          :func:`~langres.core.harvest.align_pairs` joins to the candidates for
          you -- with an optional entity-disjoint ``split`` for held-out metrics
          and a :class:`~langres.core.harvest.GoldCoverage` guardrail. Pass at
          most one of ``labels``/``pairs``.

        When the module implements **neither** hook, this is a no-op that returns
        ``self`` (unchanged sklearn-style symmetry for non-learnable pipelines
        like ``WeightedAverageMatcher``) with a minimal ``fit_report_`` -- UNLESS
        ``labels``/``pairs`` was passed, in which case it raises rather than
        silently discarding them.

        Args:
            data: Raw records (dicts) in a stable list order, same shape as
                ``resolve()``/``predict()`` accept.
            labels: Gold labels pre-aligned with the blocked candidates. Only for
                a ``SupervisedFitMixin`` module; mutually exclusive with ``pairs``.
            pairs: Id-keyed labels ``align_pairs`` joins to the candidates. Only
                for a ``SupervisedFitMixin`` module; mutually exclusive with
                ``labels``.
            split: Held-out fraction for the entity-disjoint ``pairs`` split
                (``None`` = train on everything; only meaningful with ``pairs``).
            seed: Seed for the entity-disjoint split.
            method: An optional :class:`~langres.core.methods_api.Method` naming
                *how* to train (prompt-optimize / fine-tune / calibrate). When
                given, ``fit`` dispatches on ``method.kind`` to a per-kind handler
                (``_fit_prompt`` / ``_fit_finetune`` / ``_fit_calibrate``) instead
                of the isinstance-on-the-module default above; when ``None`` (the
                default), behavior is exactly the module-hook path described here.
                Prompt-optimization is implemented (:class:`~langres.core.methods_prompt.Bootstrap`
                / :class:`~langres.core.methods_prompt.MIPRO` compile a
                ``DSPyMatcher``'s prompt -- see :meth:`_fit_prompt`); the
                fine-tune (PR-F) and calibrate (PR-D) handlers are still stubs
                that raise a clear NotImplementedError naming their PR.

        Returns:
            ``self``, so ``resolver.fit(data).resolve(data)`` chains.

        Raises:
            ValueError: If both ``labels`` and ``pairs`` are given; if the module
                implements ``SupervisedFitMixin`` and neither is given; or if
                ``labels``/``pairs`` is given to a module that cannot use them.
            NotImplementedError: If ``method`` is given but its ``kind``'s fit
                path is not implemented yet (the seam is wired ahead of the
                concrete methods; the error names the PR that will land it).
        """
        if method is not None:
            # The ``method=`` object seam: a Method names *how* to train and
            # routes to its own per-kind handler below, so the concrete
            # strategies that fill these in later touch DISJOINT methods instead
            # of one shared branch. Each handler is a thin stub today, raising a
            # clear NotImplementedError naming its PR. Guarded by ``is not None``
            # so the ``method=None`` default leaves every existing fit path below
            # byte-for-byte unchanged.
            if method.kind == "prompt":
                return self._fit_prompt(
                    data, labels=labels, pairs=pairs, split=split, seed=seed, method=method
                )
            if method.kind == "finetune":
                return self._fit_finetune(
                    data, labels=labels, pairs=pairs, split=split, seed=seed, method=method
                )
            if method.kind == "calibrate":
                return self._fit_calibrate(
                    data, labels=labels, pairs=pairs, split=split, seed=seed, method=method
                )
            raise NotImplementedError(
                f"method kind {method.kind!r} is not recognized: no langres Method "
                f"implements it ({method.describe()})."
            )
        if labels is not None and pairs is not None:
            raise ValueError(
                "pass either labels= (a Sequence[bool] pre-aligned with the blocked "
                "candidates) or pairs= (id-keyed labels align_pairs() joins for you), "
                "not both."
            )
        if pairs is not None:
            self.fit_report_ = self._fit_from_pairs(data, pairs, split=split, seed=seed)
            return self

        matcher_name = type(self.module).__name__
        if isinstance(self.module, SupervisedFitMixin):
            if labels is None:
                raise ValueError(
                    f"{matcher_name} requires labeled data: pass "
                    "labels=<Sequence[bool] aligned with the blocked candidates> "
                    "(or pairs=<id-keyed labels>) to fit()."
                )
            self.module.fit(self._candidates(data), labels)
            self.fit_report_ = FitReport.build(
                trainable=f"{matcher_name} (SupervisedFitMixin)",
                trained=True,
                n_train=len(labels),
                threshold=self.clusterer.threshold,
                run_ref=current_run.get(),
            )
            return self
        if isinstance(self.module, UnsupervisedFitMixin):
            if labels is not None:
                raise ValueError(
                    f"{matcher_name} does not support fit(labels=...): "
                    "it implements UnsupervisedFitMixin, which trains without labels "
                    "(fit_unlabeled) -- drop the labels= argument."
                )
            candidates = self.candidates(data)
            self.module.fit_unlabeled(iter(candidates))
            self.fit_report_ = FitReport.build(
                trainable=f"{matcher_name} (UnsupervisedFitMixin)",
                trained=True,
                n_train=len(candidates),
                run_ref=current_run.get(),
            )
            return self
        if labels is not None:
            raise ValueError(
                f"{matcher_name} does not support fit(labels=...): "
                "it implements neither SupervisedFitMixin nor UnsupervisedFitMixin."
            )
        self.fit_report_ = FitReport.nothing_trainable(matcher_name)
        return self

    # ------------------------------------------------------------------
    # ``method=`` per-kind fit handlers (the object seam)
    #
    # Each ``Method.kind`` routes to its OWN handler so the concrete strategies
    # land in disjoint methods -- prompt-optimize in PR-C, fine-tune in PR-F,
    # calibrate in PR-D -- rather than colliding on one shared branch. Every
    # handler takes the full fit context (data + supervision + split/seed + the
    # Method itself) so its PR fills in only the body, not the call site.
    # ``_fit_prompt`` is implemented; ``_fit_finetune`` / ``_fit_calibrate``
    # remain thin stubs raising a clear, PR-naming NotImplementedError.
    # ------------------------------------------------------------------

    def _fit_prompt(
        self,
        data: list[Any],
        *,
        labels: Sequence[bool] | None,
        pairs: str | Path | Sequence[LabeledPair] | Sequence[Correction] | None,
        split: float | None,
        seed: int,
        method: Method,
    ) -> Self:
        """Fit via prompt-optimization (``method.kind == "prompt"``).

        Tunes a compilable :class:`~langres.core.matchers.dspy_judge.DSPyMatcher`'s
        prompt from labeled pairs by compiling its DSPy program against a gold set
        -- the optimizer named by ``method.optimizer`` (``BootstrapFewShot`` for
        :class:`~langres.core.methods_prompt.Bootstrap`, ``MIPROv2`` for
        :class:`~langres.core.methods_prompt.MIPRO`). Supervision comes from either
        id-keyed ``pairs`` (joined via :func:`~langres.core.harvest.align_pairs`,
        whose optional entity-disjoint ``split`` yields the ``valid`` fold
        ``MIPROv2`` uses as its valset) or pre-aligned ``labels``. Sets
        :attr:`fit_report_` naming the demos learned + teacher model + declared
        budget, and returns ``self`` so ``resolver.fit(...).resolve(...)`` chains.

        The budget seam: ``method.budget_usd`` caps the compile via the existing
        :class:`~langres.clients.openrouter.SpendMonitor`. DSPy-compile spend
        capture is deferred to issue #100 -- today the compile records ``$0`` (the
        ``DummyLM`` CI path is genuinely free; the paid ``MIPROv2`` path stays
        uncosted until #100 wires real spend through this same guard).

        Raises:
            ValueError: If the module is not a ``DSPyMatcher``
                (prompt-optimization needs a compilable scorer); if both
                ``labels`` and ``pairs`` are given, or neither.
        """
        # Lazy imports: keep ``dspy`` (and litellm-adjacent client code) out of a
        # bare ``import langres`` -- they load only when a prompt-optimize fit runs.
        from langres.clients.openrouter import SpendMonitor
        from langres.core.matchers.dspy_judge import DSPyMatcher

        matcher_name = type(self.module).__name__
        if not isinstance(self.module, DSPyMatcher):
            raise ValueError(
                f"method.kind='prompt' prompt-optimization needs a DSPyMatcher in "
                f"the module slot (a compilable DSPy scorer), but this Resolver's "
                f"matcher is {matcher_name}. Build it with matcher=DSPyMatcher(...) "
                f"to prompt-optimize, or drop method= to use {matcher_name}'s own "
                f"fit path."
            )
        if self.module.compiled:
            raise ValueError(
                "this DSPyMatcher is already compiled -- prompt-optimization "
                "compiles a fresh program once per matcher instance and DSPy "
                "cannot recompile in place. Build a new DSPyMatcher(...) for "
                "another prompt-optimize round."
            )
        if labels is not None and pairs is not None:
            raise ValueError(
                "pass either labels= (pre-aligned with the blocked candidates) or "
                "pairs= (id-keyed labels align_pairs() joins), not both."
            )
        optimizer = getattr(method, "optimizer", None)
        if optimizer is None:
            raise ValueError(
                f"method.kind='prompt' needs a PromptMethod exposing .optimizer "
                f"(e.g. Bootstrap()/MIPRO()); got {type(method).__name__} "
                f"({method.describe()})."
            )

        # Assemble labeled candidates (train + optional valid), reusing the same
        # id-join + entity-disjoint split as the SupervisedFitMixin pairs path.
        coverage = None
        valid_candidates: Sequence[ERCandidate[Any]] = []
        valid_labels: Sequence[bool] = []
        if pairs is not None:
            aligned = align_pairs(self.candidates(data), pairs, split=split, seed=seed)
            train_candidates: Sequence[ERCandidate[Any]] = aligned.train.candidates
            train_labels: Sequence[bool] = aligned.train.labels
            valid_candidates = aligned.valid.candidates
            valid_labels = aligned.valid.labels
            coverage = aligned.coverage
        elif labels is not None:
            train_candidates = self.candidates(data)
            train_labels = labels
        else:
            raise ValueError(
                f"prompt-optimization ({method.describe()}) needs gold labels to "
                "tune the prompt from: pass pairs=<id-keyed labels> or "
                "labels=<pre-aligned with the blocked candidates>."
            )

        trainset = self.module.examples_from_candidates(train_candidates, train_labels)
        valset = (
            self.module.examples_from_candidates(valid_candidates, valid_labels)
            if valid_candidates
            else None
        )

        budget_usd = getattr(method, "budget_usd", None)
        monitor = SpendMonitor(budget_usd=budget_usd) if budget_usd is not None else None
        compile_kwargs = method.compile_kwargs() if hasattr(method, "compile_kwargs") else {}
        self.module.compile(trainset, valset, optimizer=optimizer, **compile_kwargs)

        # See the docstring's budget note: DSPy-compile spend is not yet captured
        # (#100), so the monitor observes $0 today. The seam is wired so real
        # spend flows through this cap once #100 lands.
        spend_usd = 0.0
        if monitor is not None:
            monitor.add(spend_usd)
            monitor.check()

        self.fit_report_ = FitReport.build(
            trainable=(
                f"{matcher_name} ({method.describe()}; "
                f"teacher={self.module.model}, demos={self.module.n_demos})"
            ),
            trained=True,
            n_train=len(train_labels),
            n_valid=len(valid_labels),
            split=split,
            seed=seed,
            coverage=coverage,
            threshold=self.clusterer.threshold,
            cost=spend_usd if monitor is not None else None,
            run_ref=current_run.get(),
        )
        return self

    def _fit_finetune(
        self,
        data: list[Any],
        *,
        labels: Sequence[bool] | None,
        pairs: str | Path | Sequence[LabeledPair] | Sequence[Correction] | None,
        split: float | None,
        seed: int,
        method: Method,
    ) -> Self:
        """Fit via fine-tuning (``method.kind == "finetune"``): QLoRA train → serve.

        Aligns the labeled ``pairs`` to candidates, fine-tunes ``method.base`` on
        them (:func:`~langres.core.finetune.run_finetune`), repoints this
        Resolver's matcher at the resulting ``model_ref`` as an in-process,
        logprob-scoring :class:`~langres.core.matchers.llm_judge.LLMMatcher`, and
        evaluates held-out pair P/R/F1 on the entity-disjoint ``valid`` split.
        Records the GPU-seconds / derived-$ cost and the served ``model_ref`` in
        :attr:`fit_report_`.

        Heavy imports (``core.finetune`` → peft/trl on the training call;
        ``LLMMatcher`` → litellm) are deferred to here so the ``method=None`` and
        non-finetune fit paths never pay for them.

        Raises:
            TypeError: If ``method`` is not a :class:`~langres.core.finetune.QLoRA`.
            ValueError: If neither ``pairs`` nor ``labels`` is given (fine-tuning
                needs supervision), or both are.
        """
        from langres.core.finetune import FINETUNE_YES_NO_PROMPT, QLoRA, run_finetune
        from langres.core.matchers.llm_judge import LLMMatcher
        from langres.core.model_ref import to_config

        if not isinstance(method, QLoRA):
            raise TypeError(
                f"method kind 'finetune' requires a QLoRA method; got "
                f"{type(method).__name__} ({method.describe()})."
            )
        if labels is not None and pairs is not None:
            raise ValueError("pass either labels= or pairs= to a finetune fit, not both.")
        if labels is None and pairs is None:
            raise ValueError(
                "fine-tuning needs labeled supervision: pass pairs=<id-keyed labels> "
                "(align_pairs joins them + gives a held-out split) or "
                "labels=<Sequence[bool] aligned with the blocked candidates>."
            )

        candidates = self.candidates(data)
        coverage = None
        if pairs is not None:
            aligned = align_pairs(candidates, pairs, split=split, seed=seed)
            train_pairs = list(zip(aligned.train.candidates, aligned.train.labels, strict=True))
            valid_pairs = list(zip(aligned.valid.candidates, aligned.valid.labels, strict=True))
            coverage = aligned.coverage
            n_valid = len(aligned.valid.labels)
        else:
            train_pairs = list(zip(candidates, cast("Sequence[bool]", labels), strict=True))
            valid_pairs = []
            n_valid = 0
            split = None

        # Preserve the outgoing matcher's record rendering (so what the model is
        # trained on matches what it is served) when it is an LLMMatcher.
        render = self._llm_render_config()
        # Train AND serve on the SAME yes/no prompt: the model learns the
        # FINETUNE_YES_NO_PROMPT completion, so the served matcher must send that
        # prompt (not LLMMatcher's default "Score:" template) and read it with the
        # binary yes/no parser -- otherwise serving asks a differently-worded
        # question than training taught.
        outcome = run_finetune(
            train_pairs, method, prompt_template=FINETUNE_YES_NO_PROMPT, **render
        )

        # Repoint this Resolver at the fine-tuned model: an in-process,
        # logprob-scoring yes/no LLMMatcher over the produced model_ref.
        self.module = LLMMatcher(
            model=to_config(outcome.model_ref),
            confidence="logprob",
            response_parser="binary_yes_no",
            prompt_template=FINETUNE_YES_NO_PROMPT,
            **render,
        )

        metrics: PairMetrics | None = None
        if valid_pairs:
            judgements = list(self.module.forward(iter([c for c, _ in valid_pairs])))
            gold_pairs = {
                frozenset({str(c.left.id), str(c.right.id)}) for c, label in valid_pairs if label
            }
            metrics = classify_pairs(judgements, gold_pairs, self.clusterer.threshold)

        self.fit_report_ = FitReport.build(
            trainable=f"LLMMatcher (finetune: {outcome.method})",
            trained=True,
            n_train=outcome.n_train,
            n_valid=n_valid,
            split=split,
            seed=seed,
            coverage=coverage,
            threshold=self.clusterer.threshold,
            metrics=metrics,
            cost=outcome.dollars,
            gpu_seconds=outcome.gpu_seconds,
            model_ref=to_config(outcome.model_ref),
            run_ref=current_run.get(),
        )
        return self

    def _llm_render_config(self) -> dict[str, Any]:
        """The current matcher's record serializer, to keep train == serve.

        When ``self.module`` is an :class:`~langres.core.matchers.llm_judge.LLMMatcher`
        this carries its ``record_serializer`` (by registered name) so a fine-tune
        renders records the way they will be served. Only the serializer -- NOT the
        ``prompt_template``: ``_fit_finetune`` pins both training and serving to
        :data:`~langres.core.finetune.FINETUNE_YES_NO_PROMPT` (the outgoing matcher's
        prompt may be the ``Score:`` scoring template, which the yes/no fine-tune
        does not use). Empty for a non-LLM matcher (the finetune defaults apply).
        """
        from langres.core.matchers.llm_judge import LLMMatcher

        if isinstance(self.module, LLMMatcher):
            return {"record_serializer": self.module.config["record_serializer"]}
        return {}

    def _fit_calibrate(
        self,
        data: list[Any],
        *,
        labels: Sequence[bool] | None,
        pairs: str | Path | Sequence[LabeledPair] | Sequence[Correction] | None,
        split: float | None,
        seed: int,
        method: Method,
    ) -> Self:
        """Fit via score calibration (``method.kind == "calibrate"``): learn a score→prob map.

        Scores the labeled train candidates with the current matcher, fits a fresh
        :class:`~langres.core.calibration.Calibrator` (strategy from
        ``method.strategy``) on those ``(score, label)`` pairs, and attaches it as
        :attr:`calibrator` so :meth:`predict`/:meth:`resolve` map every raw score
        to a calibrated probability. Supervision comes from id-keyed ``pairs``
        (joined via :func:`~langres.core.harvest.align_pairs`, whose optional
        entity-disjoint ``split`` gives a held-out fold) or pre-aligned ``labels``.

        The honest test: when a ``valid`` split exists, the ``FitReport`` carries
        the Brier/ECE **before vs after** calibration on that held-out fold (raw
        matcher scores vs the fitted map) -- a real calibrator drives both down.
        Does NOT retrain or touch the matcher, and does NOT change the clusterer:
        calibration only makes the score a true probability so the existing
        threshold is meaningful.

        Raises:
            ImportError: If scikit-learn (the ``[trained]`` extra) is not installed.
            ValueError: If ``method`` exposes no ``.strategy`` (not a
                :class:`~langres.core.methods_calibrate.CalibrateMethod`); if both
                ``labels`` and ``pairs`` are given, or neither; or if the matcher
                emits no scores to calibrate (a pure decider).
        """
        try:
            from langres.core.calibration import Calibrator
        except ImportError as exc:  # pragma: no cover - core-only env
            raise ImportError(
                "score calibration (method='calibrate') needs scikit-learn (the "
                "'trained' extra): pip install 'langres[trained]' "
                "(or uv add 'langres[trained]')."
            ) from exc

        strategy = getattr(method, "strategy", None)
        if strategy is None:
            raise ValueError(
                f"method.kind='calibrate' needs a CalibrateMethod exposing .strategy "
                f"(e.g. Platt()/Isotonic()); got {type(method).__name__} "
                f"({method.describe()})."
            )
        if labels is not None and pairs is not None:
            raise ValueError(
                "pass either labels= (pre-aligned with the blocked candidates) or "
                "pairs= (id-keyed labels align_pairs() joins), not both."
            )

        coverage = None
        valid_candidates: Sequence[ERCandidate[Any]] = []
        valid_labels: Sequence[bool] = []
        if pairs is not None:
            aligned = align_pairs(self.candidates(data), pairs, split=split, seed=seed)
            train_candidates: Sequence[ERCandidate[Any]] = aligned.train.candidates
            train_labels: Sequence[bool] = aligned.train.labels
            valid_candidates = aligned.valid.candidates
            valid_labels = aligned.valid.labels
            coverage = aligned.coverage
        elif labels is not None:
            train_candidates = self.candidates(data)
            train_labels = labels
        else:
            raise ValueError(
                f"score calibration ({method.describe()}) needs gold labels: pass "
                "pairs=<id-keyed labels> or labels=<pre-aligned with the blocked "
                "candidates>."
            )

        train_scores, train_score_labels = self._scored_labeled_pairs(
            train_candidates, train_labels
        )
        if not train_scores:
            raise ValueError(
                f"{type(self.module).__name__} produced no scores to calibrate: score "
                "calibration needs a ranking matcher (one that emits "
                "PairwiseJudgement.score), not a pure decider."
            )
        calibrator = Calibrator(method=strategy)
        calibrator.fit_calibrator(train_scores, train_score_labels)
        self.calibrator = calibrator

        calibration = self._calibration_delta(strategy, calibrator, valid_candidates, valid_labels)

        self.fit_report_ = FitReport.build(
            trainable=f"Calibrator ({strategy})",
            trained=True,
            n_train=len(train_score_labels),
            n_valid=len(valid_labels),
            split=split,
            seed=seed,
            coverage=coverage,
            threshold=self.clusterer.threshold,
            calibration=calibration,
            run_ref=current_run.get(),
        )
        return self

    def _scored_labeled_pairs(
        self, candidates: Sequence[ERCandidate[Any]], labels: Sequence[bool]
    ) -> tuple[list[float], list[bool]]:
        """Score ``candidates`` with the matcher and join scores back to labels by id.

        Returns ``(scores, labels)`` for the ranking judgements only (``score is
        not None``), keyed by the unordered ``{left_id, right_id}`` pair so the
        join is robust to any reordering/filtering the matcher's ``forward`` does.
        """
        label_by_pair = {
            frozenset({str(c.left.id), str(c.right.id)}): bool(label)
            for c, label in zip(candidates, labels, strict=True)
        }
        scores: list[float] = []
        aligned_labels: list[bool] = []
        for judgement in self.module.forward(iter(candidates)):
            if judgement.score is None:
                continue
            key = frozenset({judgement.left_id, judgement.right_id})
            if key in label_by_pair:
                scores.append(judgement.score)
                aligned_labels.append(label_by_pair[key])
        return scores, aligned_labels

    def _calibration_delta(
        self,
        strategy: str,
        calibrator: CalibratorFitMixin,
        valid_candidates: Sequence[ERCandidate[Any]],
        valid_labels: Sequence[bool],
    ) -> CalibrationDelta | None:
        """Brier/ECE before-vs-after calibration on the held-out ``valid`` split.

        ``None`` when there is no valid split (nothing held out to measure on) or
        the matcher emits no scores over it. Raw matcher scores are already in
        ``[0, 1]`` (``PairwiseJudgement.score``'s contract), so both metrics are
        always defined on the "before" side.
        """
        if not valid_candidates:
            return None
        valid_scores, valid_score_labels = self._scored_labeled_pairs(
            valid_candidates, valid_labels
        )
        if not valid_scores:
            return None
        after = calibrator.transform(valid_scores)
        return CalibrationDelta(
            method=strategy,
            brier_before=brier_score(valid_scores, valid_score_labels),
            brier_after=brier_score(after, valid_score_labels),
            ece_before=expected_calibration_error(valid_scores, valid_score_labels),
            ece_after=expected_calibration_error(after, valid_score_labels),
        )

    def _fit_from_pairs(
        self,
        data: list[Any],
        pairs: str | Path | Sequence[LabeledPair] | Sequence[Correction],
        *,
        split: float | None,
        seed: int,
    ) -> FitReport:
        """Fit a ``SupervisedFitMixin`` matcher from id-keyed labels via ``align_pairs``.

        Runs the id-join + entity-disjoint split + coverage in one place, trains
        on the train split, and evaluates held-out pair P/R/F1 on the valid split
        (when a split was given, via :func:`~langres.core.metrics.classify_pairs`
        at the clusterer's threshold). Returns the assembled :class:`FitReport`.
        """
        matcher_name = type(self.module).__name__
        if not isinstance(self.module, SupervisedFitMixin):
            raise ValueError(
                f"{matcher_name} does not support fit(pairs=...): pairs= supplies "
                "labeled pairs for a SupervisedFitMixin matcher, and this matcher "
                "implements no supervised fit hook. Use fit() with no labels for an "
                "unsupervised/non-learnable matcher."
            )
        aligned = align_pairs(self.candidates(data), pairs, split=split, seed=seed)
        self.module.fit(iter(aligned.train.candidates), aligned.train.labels)

        metrics: PairMetrics | None = None
        if aligned.valid.candidates:
            judgements = list(self.module.forward(iter(aligned.valid.candidates)))
            gold_pairs = {
                frozenset({str(c.left.id), str(c.right.id)})
                for c, label in zip(aligned.valid.candidates, aligned.valid.labels, strict=True)
                if label
            }
            metrics = classify_pairs(judgements, gold_pairs, self.clusterer.threshold)

        return FitReport.build(
            trainable=f"{matcher_name} (SupervisedFitMixin)",
            trained=True,
            n_train=len(aligned.train.labels),
            n_valid=len(aligned.valid.labels),
            split=split,
            seed=seed,
            coverage=aligned.coverage,
            threshold=self.clusterer.threshold,
            metrics=metrics,
            run_ref=current_run.get(),
        )

    def _candidates(self, records: list[Any]) -> Iterator[ERCandidate[Any]]:
        """Block records into candidates, attaching comparisons if configured.

        Builds an index-backed blocker's index transparently before streaming,
        so callers never call ``create_index`` themselves. Records are fed in
        the caller's stable list order. Shared by ``_judgements`` (scoring)
        and ``fit`` (training) -- both need the same candidate stream.
        """
        self._ensure_index_built(records)
        candidates = self.blocker.stream(records)
        if self.comparator is not None:
            comparator = self.comparator
            candidates = (
                c.model_copy(update={"comparison": comparator.compare(c.left, c.right)})
                for c in candidates
            )
        return candidates

    def candidates(self, records: list[Any]) -> list[ERCandidate[Any]]:
        """Block records into a materialized list of judge-ready candidates.

        The public counterpart to :meth:`_candidates`: same blocking (building
        any index-backed blocker's index transparently) and the same
        comparison-attachment behavior, but returns a **list** rather than a
        generator. Comparison vectors ARE attached whenever this Resolver has a
        comparator configured (the default for ``Resolver.from_schema``) --
        a caller that instead reaches into e.g. ``bench.build_blocker().stream(records)``
        directly gets candidates WITHOUT comparison vectors, which silently
        changes what a comparison-reading judge (e.g. ``WeightedAverageMatcher``)
        sees.

        Prefer this over a raw ``blocker.stream(...)`` generator whenever the
        candidates are consumed more than once -- e.g.
        :func:`~langres.core.benchmark.evaluate_judge_on_candidates` both calls
        ``len(candidates)`` and iterates the sequence twice (once to judge, once
        to build the graded candidate pairs). Handing a generator to a caller
        that iterates twice makes ``len()`` fail and the second pass silently
        yield nothing.

        Args:
            records: Raw records (dicts) in a stable list order, same shape as
                ``resolve()``/``predict()`` accept.

        Returns:
            The blocked candidates, materialized as a list (never a generator).
        """
        return list(self._candidates(records))

    def _judgements(self, records: list[Any]) -> Iterator[PairwiseJudgement]:
        """Block records into candidates, score them, and calibrate if fitted.

        Scoring runs through this Resolver's spend cap (``budget_usd=``), whose
        ledger is shared across every :meth:`resolve`/:meth:`predict` call on
        this instance -- so two successive resolves cannot each spend a full
        budget. The wrapper is rebuilt per call but the
        :class:`~langres.core.spend.SpendMonitor` is not: that is what makes the
        cap cumulative while still metering whatever ``self.module`` is *now*
        (callers reassign it -- ``dedupe`` wraps it in a ``LoggingMatcher``).

        When :attr:`calibrator` is set (by ``fit(method=Platt()/Isotonic())``),
        every ranking judgement's raw ``score`` is mapped to a calibrated
        probability before it reaches :meth:`predict`/:meth:`resolve` -- so the
        clusterer thresholds on a real probability. Pure pass-through otherwise.
        """
        scorer = SpendCappedMatcher(self.module, monitor=self._spend_monitor)
        judgements = scorer.forward(self._candidates(records))
        if self.calibrator is None:
            return judgements
        return self._apply_calibrator(judgements, self.calibrator)

    def _apply_calibrator(
        self, judgements: Iterator[PairwiseJudgement], calibrator: CalibratorFitMixin
    ) -> Iterator[PairwiseJudgement]:
        """Map each ranking judgement's ``score`` through the fitted calibrator.

        Deciders (``score is None``) pass through untouched -- there is no score to
        calibrate. A mapped judgement keeps its ids/decision, retags
        ``score_type="calibrated_prob"``, and records the raw score under
        ``provenance["calibration"]`` for auditability.
        """
        for judgement in judgements:
            if judgement.score is None:
                yield judgement
                continue
            calibrated = calibrator.transform([judgement.score])[0]
            yield judgement.model_copy(
                update={
                    "score": calibrated,
                    "score_type": "calibrated_prob",
                    "provenance": {
                        **judgement.provenance,
                        "calibration": {
                            "method": getattr(calibrator, "method", None),
                            "raw_score": judgement.score,
                        },
                    },
                }
            )

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
        """Build/populate every reachable ``VectorBlocker``'s index from ``records``.

        Embeds the records' text field and creates the index in place for each
        index-backed blocker discovered via :func:`_iter_vector_blockers` --
        whether ``self.blocker`` is a ``VectorBlocker`` directly, or one is
        nested (at any depth) inside a ``CompositeBlocker``. A blocker with no
        index (AllPairs, GLinker, KeyBlocker) contributes nothing. When an
        index *is* already built (e.g. a freshly loaded FAISS index, or a
        Resolver reused on the same records), the would-be corpus is compared
        to the index's stored ``_corpus_texts``: identical -> reuse (never
        re-embed, so restore + same-records round-trips are cheap); different
        -> rebuild (so reusing the Resolver on a new record list scores
        against the right corpus rather than a stale one).

        No ``isinstance(..., VectorBlocker)`` anywhere in this walk (W0.4): see
        :func:`_iter_vector_blockers`'s docstring for why -- this method runs
        on every ``resolve()``/``predict()`` call regardless of blocker, so an
        ``isinstance`` check would need ``VectorBlocker`` imported
        unconditionally, pulling faiss/sentence-transformers (the
        ``[semantic]`` extra) into a plain ``AllPairsBlocker``/``KeyBlocker``
        pipeline.
        """
        for vector_blocker in _iter_vector_blockers(self.blocker):
            entities = [vector_blocker.schema_factory(record) for record in records]
            texts = [vector_blocker.text_field_extractor(entity) for entity in entities]

            index = vector_blocker.vector_index
            if vector_blocker._index_is_built() and getattr(index, "_corpus_texts", None) == texts:
                continue  # Same corpus already indexed -> reuse, never re-embed.

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

    def build_anchor_store(self, records: list[Any], *, entity_prefix: str = "e") -> "AnchorStore":
        """Anchor this resolver on a batch so :meth:`assign` can run (M5, S6).

        A dedicated build pass over ``records`` that mints a **stable** entity id
        for every record — including the singletons ``resolve()`` drops — and
        leaves the store on ``self`` for subsequent :meth:`assign` calls. Returns
        the store, which is independently serializable
        (:meth:`AnchorStore.save`).

        Args:
            records: The batch of raw record dicts to anchor on, same shape as
                ``resolve()`` accepts.
            entity_prefix: Prefix for minted entity ids (default ``"e"``).

        Returns:
            The built :class:`~langres.core.anchor_store.AnchorStore`.
        """
        from langres.core.anchor_store import AnchorStore

        self._anchor_store = AnchorStore.build(self, records, entity_prefix=entity_prefix)
        return self._anchor_store

    def assign(self, record: Any) -> "ClusterDelta":
        """Assign one new record to an existing entity, or mint a new one (M5, S6).

        Incremental single-record resolution against the anchor set built by
        :meth:`build_anchor_store`: returns a
        :class:`~langres.core.anchor_store.ClusterDelta` that either ``link``\\ s
        the record to an existing entity (with a stable id) or marks it ``new``.
        Distinct from the reserved cross-source :meth:`link` /
        :meth:`stream_against` stubs — ``assign`` is single-record incremental
        linking.

        Args:
            record: A raw record dict, same shape as ``resolve()`` accepts.

        Returns:
            A :class:`~langres.core.anchor_store.ClusterDelta`.

        Raises:
            RuntimeError: If :meth:`build_anchor_store` was not called first.
        """
        if self._anchor_store is None:
            raise RuntimeError(
                "call build_anchor_store(records) before assign(record): assign "
                "resolves a new record against a prior batch's anchor set."
            )
        return self._anchor_store.assign(record)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

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
        :meth:`config_dict` (which returns it as a dict). Serializes each slot
        component into a :class:`ComponentSpec` via :func:`_component_spec`,
        which raises :class:`TypeError` for a component lacking a registry
        ``type_name`` — that error is intentional and not swallowed here.
        """
        components = [
            _component_spec(component, slot=slot_name) for slot_name, component in self._slots()
        ]
        return ArtifactManifest(
            artifact_version=ARTIFACT_VERSION,
            langres_version=LANGRES_VERSION,
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
        :func:`~langres.core.runs.compute_recipe_id`'s hash domain. Emitting the
        version fields would fork ``recipe_id`` on every package or
        artifact-schema bump, silently defeating idempotent replay. Version and
        provenance live on :class:`~langres.core.runs.RunContext` as separate,
        **unhashed** fields (e.g. ``RunContext.langres_version``); :meth:`save`
        still records them on disk for artifact reconstruction.

        Known limitation: this captures **declared** component config, not
        compiled/optimized in-memory state — e.g. a DSPy-compiled program's tuned
        prompts do not appear here. Persisting that state is out of scope for the
        config snapshot (it round-trips via :class:`SerializableState` sidecars in
        :meth:`save`, not through this dict).

        Returns:
            A plain, JSON-serializable dict with a single ``components`` key: the
            ordered slot specs (each a ``type_name`` + ``config``). No version
            fields — see above.

        Raises:
            TypeError: If a slot component lacks a registry ``type_name`` (same
                contract as :meth:`save`; not swallowed).
        """
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
        ``model_name`` only — no model bytes).

        Args:
            path: Directory to write the artifact into (created if absent).
        """
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest = self._build_manifest()
        for slot_name, component in self._slots():
            owner = _state_owner(component)
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
        manifest = ArtifactManifest.model_validate_json((in_dir / _MANIFEST_FILENAME).read_text())
        cls._check_versions(manifest)

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

        blocker = _rebuild_component(blocker_spec, state_dir=in_dir / "blocker")
        comparator = (
            _rebuild_component(comparator_spec, state_dir=in_dir / "comparator")
            if comparator_spec is not None
            else None
        )
        module = _rebuild_component(module_spec, state_dir=in_dir / "module")
        clusterer = _rebuild_component(clusterer_spec, state_dir=in_dir / "clusterer")
        calibrator = (
            _rebuild_component(calibrator_spec, state_dir=in_dir / "calibrator")
            if calibrator_spec is not None
            else None
        )

        return cls(
            blocker=blocker,
            comparator=comparator,
            matcher=module,
            clusterer=clusterer,
            calibrator=calibrator,
        )

    @staticmethod
    def _check_versions(manifest: ArtifactManifest) -> None:
        """Validate artifact compatibility; raise on an unreadably-new artifact.

        ``ARTIFACT_VERSION`` is a monotonic integer-valued string bumped on an
        incompatible layout change. Each bump breaks the config schema, so only
        an artifact at the *exact* supported layout is readable: a *newer* layout
        (this build is too old), an *older* layout (predates an incompatible
        bump), or a malformed/non-integer layout are all hard errors — without
        this guard an older artifact would fall through to a raw ``KeyError`` on
        the changed config. A ``langres_version`` mismatch is logged as a
        warning, not a failure — configs are forward-compatible *within* a layout
        version.
        """
        try:
            artifact_v = int(manifest.artifact_version)
            current_v = int(ARTIFACT_VERSION)
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
        if artifact_v < current_v:
            raise ValueError(
                f"Artifact version {manifest.artifact_version!r} predates the supported "
                f"layout ({ARTIFACT_VERSION!r}) and is no longer readable (the config "
                f"schema changed incompatibly); re-save with this langres build."
            )
        if manifest.langres_version != LANGRES_VERSION:
            logger.warning(
                "Loading artifact written by langres %s into langres %s; "
                "configs are forward-compatible within artifact version %s.",
                manifest.langres_version,
                LANGRES_VERSION,
                ARTIFACT_VERSION,
            )
