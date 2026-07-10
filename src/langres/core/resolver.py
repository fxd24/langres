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

import inspect
import logging
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, TypeGuard, cast

from pydantic import BaseModel

import langres
from langres.core.blocker import Blocker
from langres.core.blockers.composite import CompositeBlocker
from langres.core.clusterer import Clusterer
from langres.core.comparator import Comparator
from langres.core.fit import SupervisedFitMixin, UnsupervisedFitMixin
from langres.core.judges.weighted_average import WeightedAverageJudge
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.module import Module
from langres.core.registry import get_component
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
    # that never uses judge="embedding".
    from langres.core.anchor_store import AnchorStore, ClusterDelta
    from langres.core.blockers.vector import VectorBlocker

logger = logging.getLogger(__name__)

# Slot names used as sidecar subdirectory names and for manifest ordering.
_MANIFEST_FILENAME = "resolver.json"

#: ``Resolver.from_schema``'s low-level judge switch. Deliberately narrower
#: than ``langres.core.presets.JudgeName`` -- no ``"auto"``, since resolving
#: that needs ``Settings``/env-var lookups, which is verb-layer magic (see
#: ``langres.core.presets.choose_auto_judge``); this stays a plain, explicit
#: constructor argument.
_FromSchemaJudge = Literal["string", "embedding", "zero_shot_llm"]


def _build_module_for_judge(
    judge: "_FromSchemaJudge | Module[Any]",
    comparator: Comparator[Any],
    *,
    model: str | None,
    entity_noun: str,
) -> Module[Any]:
    """Build the scorer for ``Resolver.from_schema``'s ``judge=`` slot.

    A small, deliberately self-contained switch: ``langres.core.presets``
    (which builds on top of ``Resolver``) is NOT imported here, since that
    would create a ``Resolver -> presets -> Resolver`` cycle -- see the
    dependency diagram in this module's docstring. The three branches below
    duplicate a little of ``presets.build_judge``'s logic; that duplication is
    the price of keeping the layering one-directional (``verbs -> presets ->
    Resolver``), not an oversight.
    """
    if isinstance(judge, Module):
        return judge
    if judge == "string":
        return WeightedAverageJudge(feature_specs=comparator.feature_specs)
    if judge == "embedding":
        from langres.core.judges.embedding_score import EmbeddingScoreJudge

        return EmbeddingScoreJudge()
    if judge == "zero_shot_llm":
        # Lazy: dspy must stay out of sys.modules unless this judge is chosen.
        from langres.clients.openrouter import DEFAULT_OPENROUTER_MODEL, dspy_price_per_1k
        from langres.core.modules.dspy_judge import DSPyJudge

        resolved_model = model or DEFAULT_OPENROUTER_MODEL
        dspy_module: DSPyJudge[Any] = DSPyJudge(model=resolved_model, entity_noun=entity_noun)
        price = dspy_price_per_1k(resolved_model)
        if price == 0.0:
            # An unpinned model self-reports $0/pair -- honest, not reassuring
            # (mirrors core.presets.notice_pre_scoring_cost's identical check;
            # duplicated here for the same layering reason as the rest of this
            # function). Resolver.from_schema has no spend cap at all (see its
            # judge= docstring), so this is strictly worse than the verbs'
            # blind-cap case: nothing would ever stop a runaway bill.
            warnings.warn(
                f"model {resolved_model!r} has no pinned price in "
                "langres.clients.openrouter.PRICES_PER_1M, so it self-reports "
                "$0/pair cost. Resolver.from_schema builds an UNCAPPED pipeline "
                "(no spend cap at all) -- pin its price in PRICES_PER_1M, or use "
                "langres.link/langres.dedupe for the built-in spend cap.",
                stacklevel=3,
            )
        dspy_module.price_per_1k_tokens = price
        return dspy_module
    raise ValueError(
        f"unsupported judge {judge!r} for Resolver.from_schema; choose one of "
        "'string', 'embedding', 'zero_shot_llm', or pass a Module instance. "
        "'auto' key-based resolution is a verbs-layer feature -- use "
        "langres.link/langres.dedupe for that."
    )


def _build_embedding_blocker(schema: type[BaseModel]) -> "VectorBlocker[Any]":
    """Build the ``VectorBlocker`` a ``judge="embedding"`` pipeline needs.

    ``AllPairsBlocker``'s candidates never carry ``similarity_score``, which
    ``EmbeddingScoreJudge`` requires to score -- ``judge="embedding"`` must
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

    field_names = [spec.name for spec in Comparator.from_schema(schema).feature_specs]

    def extract(entity: Any) -> str:
        parts = [str(getattr(entity, name)) for name in field_names if getattr(entity, name, None)]
        return " ".join(parts)

    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
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
            f"Use a registered component (e.g. LLMJudge, WeightedAverageJudge) in "
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
        comparator = Comparator.from_schema(CompanySchema, weights={"name": 0.6, ...})
        resolver = Resolver(
            blocker=AllPairsBlocker(schema=CompanySchema),
            comparator=comparator,
            module=WeightedAverageJudge(feature_specs=comparator.feature_specs),
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
        judge: "_FromSchemaJudge | Module[Any]" = "string",
        model: str | None = None,
        entity_noun: str = "entity",
    ) -> "Resolver":
        """Build a default dedup Resolver from a Pydantic schema in one line.

        Defaults to an ``AllPairsBlocker`` over the schema, a missing-aware
        ``StringComparator`` auto-derived from the schema's string fields (with
        ``id`` excluded), a ``WeightedAverageJudge`` scorer, and a ``Clusterer``
        at ``threshold``. ``judge="embedding"`` is the one exception to the
        ``AllPairsBlocker`` default: it wires a ``VectorBlocker`` instead,
        since ``EmbeddingScoreJudge`` scores off the blocker's
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
            judge: ``"string"`` (default -- identical to pre-existing
                behavior), ``"embedding"`` (wires a ``VectorBlocker``, see
                above), ``"zero_shot_llm"``, or a ``Module`` instance. This is
                the low-level, explicit switch (no ``"auto"`` key-based
                resolution and no spend cap -- that magic lives in
                ``langres.link``/``langres.dedupe``). **Caution**:
                ``judge="zero_shot_llm"`` (or any other paid ``Module``) built
                here runs UNCAPPED -- there is no ``budget_usd`` on this
                method and nothing stops a runaway bill. Use
                ``langres.link``/``langres.dedupe`` for the built-in
                ``SpendMonitor`` cap.
            model: Model id override for ``judge="zero_shot_llm"``.
            entity_noun: Domain noun woven into the LLM judge's prompt.

        Returns:
            A ready-to-run Resolver.
        """
        from langres.core.blockers.all_pairs import AllPairsBlocker

        comparator: Comparator[Any] = Comparator.from_schema(
            schema, exclude=exclude, weights=weights
        )
        module = _build_module_for_judge(judge, comparator, model=model, entity_noun=entity_noun)
        blocker: Blocker[Any] = (
            _build_embedding_blocker(schema)
            if judge == "embedding"
            else AllPairsBlocker(schema=schema)
        )
        return cls(
            blocker=blocker,
            comparator=comparator,
            module=module,
            clusterer=Clusterer(threshold=threshold),
        )

    # ------------------------------------------------------------------
    # Running the pipeline
    # ------------------------------------------------------------------

    def fit(self, data: list[Any], labels: Sequence[bool] | None = None) -> Self:
        """Fit the module when it supports a fit hook; sklearn-style no-op otherwise.

        Delegates to the module's fit hook when it implements one of the two
        runtime-checkable Protocols in :mod:`langres.core.fit` (W1.0, E6):

        - :class:`~langres.core.fit.UnsupervisedFitMixin`
          (``fit_unlabeled(candidates)``): called unconditionally with the
          blocked (and, if a comparator is configured, comparison-attached)
          candidate stream. ``labels`` is not used by this path.
        - :class:`~langres.core.fit.SupervisedFitMixin`
          (``fit(candidates, labels)``): called with ``labels`` when given;
          **raises** rather than silently skipping training when ``labels``
          is omitted -- a genuinely trainable module that never gets
          trained is exactly the silent-no-op footgun this hook exists to
          prevent.

        When the module implements **neither** hook, this is a no-op that
        returns ``self`` -- unchanged sklearn-style symmetry so callers can
        write ``resolver.fit(data).resolve(data)`` for non-learnable
        pipelines (e.g. ``WeightedAverageJudge``) without branching -- UNLESS
        ``labels`` was passed, in which case it raises rather than silently
        discarding them.

        Args:
            data: Raw records (dicts) in a stable list order, same shape as
                ``resolve()``/``predict()`` accept.
            labels: Gold match/non-match labels, positionally aligned with the
                blocked candidates. Required (and only used) when the module
                implements ``SupervisedFitMixin``.

        Returns:
            ``self``, so ``resolver.fit(data).resolve(data)`` chains.

        Raises:
            ValueError: If the module implements ``SupervisedFitMixin`` and
                ``labels`` is omitted, or if ``labels`` is given but the
                module implements neither fit hook.
        """
        if isinstance(self.module, SupervisedFitMixin):
            if labels is None:
                raise ValueError(
                    f"{type(self.module).__name__} requires labeled data: pass "
                    "labels=<Sequence[bool] aligned with the blocked candidates> "
                    "to fit()."
                )
            self.module.fit(self._candidates(data), labels)
            return self
        if isinstance(self.module, UnsupervisedFitMixin):
            if labels is not None:
                raise ValueError(
                    f"{type(self.module).__name__} does not support fit(labels=...): "
                    "it implements UnsupervisedFitMixin, which trains without labels "
                    "(fit_unlabeled) -- drop the labels= argument."
                )
            self.module.fit_unlabeled(self._candidates(data))
            return self
        if labels is not None:
            raise ValueError(
                f"{type(self.module).__name__} does not support fit(labels=...): "
                "it implements neither SupervisedFitMixin nor UnsupervisedFitMixin."
            )
        return self

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
        changes what a comparison-reading judge (e.g. ``WeightedAverageJudge``)
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
        """Block records into candidates and score them into judgements."""
        return self.module.forward(self._candidates(records))

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
            langres_version=langres.__version__,
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
        by_slot = {spec.slot: spec for spec in manifest.components if spec.slot}
        if by_slot:
            blocker_spec = by_slot.get("blocker")
            comparator_spec = by_slot.get("comparator")
            module_spec = by_slot.get("module")
            clusterer_spec = by_slot.get("clusterer")
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

        return cls(
            blocker=blocker,
            comparator=comparator,
            module=module,
            clusterer=clusterer,
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
        if manifest.langres_version != langres.__version__:
            logger.warning(
                "Loading artifact written by langres %s into langres %s; "
                "configs are forward-compatible within artifact version %s.",
                manifest.langres_version,
                langres.__version__,
                ARTIFACT_VERSION,
            )
