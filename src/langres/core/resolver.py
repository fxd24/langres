"""``ERModel`` (aliased ``Resolver``): the model layer that composes a full ER pipeline.

The **top-level container** of ``langres.core``. It wires four slots into one
runnable, serializable pipeline::

    blocker      -> candidate generation + schema normalization
    comparator   -> (optional) missing-aware per-feature comparison
    module       -> the scorer (a Matcher yielding PairwiseJudgements)
    clusterer    -> connected-components grouping of matched pairs

The class is assembled from a linear chain of layers, each owning one
responsibility and living in its own module, so no single file has to hold the
whole model:

===============================  ==========================================
:mod:`langres.core._model_state`   what a model *is*: slots, identity, the
                                   three construction doors, schema binding
:mod:`langres.core._model_run`     how it *runs*: block -> (compare) -> score
                                   -> cluster, plus ``dedupe``/``compare``
:mod:`langres.core._model_persist` how it *persists*: the ``resolver.json``
                                   manifest + per-slot sidecars (no pickle)
:mod:`langres.core._artifacts`     the component <-> ``ComponentSpec`` adapters
                                   ``_model_persist`` serializes each slot with
===============================  ==========================================

**What stays here, and why it is not arbitrary.** This module is the leaf: the
place where a *concrete backend* is named, built, or swapped in. ``from_schema``
and its two builders construct one; ``fit`` (with ``_fit_finetune``) trains one
and repoints the matcher slot at it; ``build_anchor_store``/``assign`` hold the
incremental state. Those are also, measurably, the only parts that cannot move:
each reaches ``core.method_registry`` / ``training.finetune`` /
``core.matchers.llm_judge`` / ``core.anchor_store``, every one of which reaches
back here via ``openrouter -> benchmark -> resolver``. Hosting them in a new
module would spread that existing knot across more modules -- the exact
regression ``tests/test_import_tangle.py`` was written to catch (see PR #169).
The layers above import strictly downward and add no cycle.

Unified serialization convention
---------------------------------
Wave 2 produced two component-config styles (a ``config`` property returning a
dict, and a ``config()`` method returning a Pydantic model). The model layer does
not pick one and rewrite the other -- :mod:`langres.core._artifacts` adapts both
behind ``component_spec`` / ``rebuild_component``, so every slot serializes and
reconstructs uniformly. See that module for the full convention.
"""

import logging
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, cast

from pydantic import BaseModel

from langres.core._model_persist import ModelPersistence
from langres.core._model_run import ModelRun
from langres.core.blocker import Blocker
from langres.core.clusterer import Clusterer
from langres.core.comparator import Comparator
from langres.core.comparators import StringComparator
from langres.core.fit import (
    BlockerFitMixin,
    CalibratorFitMixin,
    SupervisedFitMixin,
    UnsupervisedFitMixin,
)
from langres.training.fit_report import (
    CalibrationDelta,
    FitReport,
    ThresholdCandidate,
    ThresholdFit,
)
from langres.curation.harvest import (
    Correction,
    LabeledPair,
    align_pairs,
    derive_threshold_from_pairs,
    warn_if_silver_only as _warn_if_silver_only,
)
from langres.core.matcher import Matcher
from langres.core.methods_api import Method, UnsupportedMethodKind
from langres.core.metrics import (
    PairMetrics,
    brier_score,
    classify_pairs,
    expected_calibration_error,
)
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.pairs import PairRow
from langres.core.registry import get_model
from langres.tracking.runs import current_run
from langres.core.spend import SpendMonitor

if TYPE_CHECKING:
    # [semantic] extra (faiss/sentence-transformers/torch) -- imported lazily
    # inside _build_embedding_blocker so a core-only `import langres` never
    # pulls faiss/torch in for a Resolver that never uses matcher="embedding".
    from langres.curation.anchor_store import AnchorStore, ClusterDelta
    from langres.core.blockers.vector import VectorBlocker

logger = logging.getLogger(__name__)

#: ``ERModel.from_schema``'s low-level matcher switch. There is deliberately no
#: ``"auto"``: resolving one meant reading ``Settings``/env vars to find an API
#: key and then *spending* on whatever turned up. W4 deleted that path outright
#: (with its ``core.presets`` machinery) rather than moving it -- naming a model
#: is the user's job, not a heuristic's. This stays a plain, explicit argument.
_FromSchemaJudge = Literal["string", "embedding", "zero_shot_llm", "prompt_llm", "random_forest"]

#: The threshold-derivation method ``fit(derive_threshold=True)`` uses -- Youden's
#: J, ``derive_threshold``'s own default. Named here (rather than typed as
#: ``ThresholdMethod``) because importing that Literal would pull
#: ``training.calibration`` -- and with it scikit-learn -- into every
#: ``import langres``; the ``[trained]`` extra must stay lazy on this path.
_THRESHOLD_METHOD = "youden"


def _pair_key(candidate: ERCandidate[Any]) -> frozenset[str]:
    """The order-independent identity of a candidate pair, as ``align_pairs`` keys it."""
    return frozenset({str(candidate.left.id), str(candidate.right.id)})


def _row_key(row: PairRow[Any]) -> frozenset[str]:
    """The same identity for a carrier row, so row-side and candidate-side maps join."""
    return frozenset({row.left_id, row.right_id})


def _pair_f1(labeled: Sequence[LabeledPair], threshold: float) -> float:
    """Pair-F1 of ``score >= threshold`` against the gold labels -- the selection metric.

    Deliberately score-only, unlike
    :func:`~langres.core.metrics.classify_pairs`: every pair reaching a threshold
    fit carries a score (``derive_threshold_from_pairs`` refuses the input
    otherwise), so the decider/abstention branches cannot arise here. ``0.0`` for
    a cut that predicts nothing, which is the honest score for "matches nothing".
    """
    tp = sum(1 for p in labeled if p.score is not None and p.score >= threshold and p.label)
    fp = sum(1 for p in labeled if p.score is not None and p.score >= threshold and not p.label)
    fn = sum(1 for p in labeled if not (p.score is not None and p.score >= threshold) and p.label)
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


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
            f"unsupported matcher {judge!r} for ERModel.from_schema; choose one of "
            "'string', 'embedding', 'zero_shot_llm', 'prompt_llm', 'random_forest', "
            "or pass a Matcher instance. There is no 'auto': it resolved a paid model "
            "by sniffing env vars for an API key, and W4 deleted it. Name the model you "
            "want -- e.g. langres.architectures.FuzzyString() for the offline $0 path, "
            "or VectorLLMCascade(llm=...) for a paid one."
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


class ERModel(ModelRun, ModelPersistence):
    """A whole ER pipeline, named: blocker -> compare -> score -> cluster.

    **The thing W4 exists to give a name to.** Before it, nothing in langres
    named a *whole* pipeline: ``link()``/``dedupe()`` took a ``matcher=`` string
    x a ``model=`` string, and ``matcher="auto"`` sniffed the environment for an
    API key and **spent money** on whatever it found. An ``ERModel`` is that
    pipeline made explicit, constructed, and inspectable::

        FuzzyString().dedupe(records)                       # $0, offline, no key
        VectorLLMCascade(llm="openrouter/...").dedupe(records)   # paid, because you said so

    Two words, used precisely throughout:

    - **architecture** = the *topology* (which components, in what order). A new
      topology is a new class -- see :mod:`langres.architectures`.
    - **backbone** = what fills one model slot, named by a
      :class:`~langres.core.model_ref.ModelRef`. Swapping a backbone must never
      mint a new architecture.

    Two ways to get one, and they are not the same door:

    - ``ERModel(blocker=..., comparator=..., matcher=..., clusterer=...)`` -- the
      base, component-wired. Every slot given, nothing inferred.
    - ``FuzzyString(threshold=0.8)`` -- a named architecture, which stores
      *hyperparameters* and builds its own topology from the schema (given, or
      inferred from the records on first use). See :meth:`_topology`.

    :meth:`from_components` is the third, non-user-facing door -- how
    :meth:`load` rebuilds a saved model without replaying either ``__init__``.

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
        model = ERModel(
            blocker=AllPairsBlocker(schema=CompanySchema),
            comparator=comparator,
            matcher=WeightedAverageMatcher(feature_specs=comparator.feature_specs),
            clusterer=Clusterer(threshold=0.7),
        )
        clusters = model.resolve(COMPANY_RECORDS)
        model.save("artifacts/company_v0")
        reloaded = ERModel.load("artifacts/company_v0")
    """

    def _init_state(self, *, budget_usd: float | None) -> None:
        """Add the incremental-resolution state to the shared base state.

        Extends :meth:`~langres.core._model_state.ModelState._init_state` rather
        than ``__init__``, so all three construction doors -- including
        ``from_components``, which deliberately never runs ``__init__`` -- get the
        anchor store nulled exactly once.

        It lives here, with :meth:`build_anchor_store`/:meth:`assign` (the only
        things that touch it), instead of on the base: typing it needs
        ``AnchorStore``, and ``core.anchor_store`` imports this module back, so
        naming it from the base layer would knot the graph.
        """
        super()._init_state(budget_usd=budget_usd)
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
        derive_threshold: bool = False,
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
          ``Sequence`` of :class:`~langres.curation.harvest.LabeledPair` /
          :class:`~langres.curation.harvest.Correction`) that
          :func:`~langres.curation.harvest.align_pairs` joins to the candidates for
          you -- with an optional entity-disjoint ``split`` for held-out metrics
          and a :class:`~langres.curation.harvest.GoldCoverage` guardrail. Pass at
          most one of ``labels``/``pairs``.

        When the module implements **neither** hook, this is a no-op that returns
        ``self`` (unchanged sklearn-style symmetry for non-learnable pipelines
        like ``WeightedAverageMatcher``) with a minimal ``fit_report_`` -- UNLESS
        ``labels``/``pairs`` was passed, in which case it raises rather than
        silently discarding them.

        **``derive_threshold=True``: fit the decision cut from the labels.**
        Off by default. When set, ``fit`` scores the labeled ``train`` pairs
        through this model's own scoring path, derives a candidate match
        threshold from that score distribution via
        :func:`~langres.curation.harvest.derive_threshold_from_pairs`, and
        **races it against the threshold already in force** -- applying it only
        if it strictly beats the incumbent's pair-F1 on ``train``. A winner is
        written to wherever *this* model keeps its cut
        (``clusterer.threshold`` on a classic four-slot model, the terminal
        :class:`~langres.core.op.ThresholdSelect` on an explicit ``_ops`` chain,
        which has no clusterer slot at all); a loser is reported and discarded,
        leaving the threshold untouched. ``fit_report_.threshold_fit`` records
        which happened, both candidates, and how each scored.

        The race is not ceremony. A derived cut is not automatically a better
        one: Youden's J is flat across a wide separating gap, so on cleanly
        separated data it returns a cut that ties on ``train`` and is *worse* on
        unseen pairs -- measured during development at held-out pair-F1 1.0000 vs
        0.8000 (``tests/core/test_resolver_fit_threshold.py``). Selection runs on
        ``train``, never on ``valid``, so the held-out numbers stay clean
        estimates rather than becoming selection-set numbers. Four more things
        worth knowing before you switch it on:

        - **It needs ``pairs=``, not ``labels=``.** ``pairs=`` is what carries the
          entity-disjoint ``split``, and a threshold derived from the same rows
          you then score is *in-sample*: it will look like a large win and will
          not reproduce. With ``split=``, the cut is derived on ``train`` and the
          report's P/R/F1 grade it on ``valid``, which the cut never saw. Without
          ``split=`` the cut is still derived -- honestly, and the report says
          ``IN-SAMPLE`` -- but no metrics are computed, because an in-sample
          number here would only flatter the cut.
        - **It also fits a matcher with no fit hook.** A fixed scorer
          (``WeightedAverageMatcher``, ``RapidfuzzMatcher``, an LLM judge) has
          nothing to train, so ``fit(pairs=...)`` refuses it today. Its threshold
          *is* fittable, so with ``derive_threshold=True`` that call is accepted
          and fits exactly that one parameter. ``trained`` stays ``False`` (no
          matcher hook ran); ``threshold_fit.source`` is what says a cut was
          derived.
        - **Youden's J treats both errors as equally bad.** The default (and only)
          method maximizes ``tpr - fpr``, which is symmetric in cost. Entity
          resolution usually is not: a false merge propagates through transitive
          closure and can poison a whole cluster, while a false split leaves two
          records that can still be merged later. If your costs are that
          asymmetric, take the derived cut as a *starting point* and move it up,
          or call :func:`~langres.training.calibration.derive_threshold` yourself
          with ``method="percentile"``.
        - **On an explicit ``_ops`` chain it costs a full pass over ``data``.** A
          chain's Source owns retrieval, so it cannot score an arbitrary
          candidate subset the way a classic model can -- deriving runs the whole
          chain once over every record, not just the labeled pairs. Cheap for a
          local scorer; with a paid ``Score`` wired into a large chain, size the
          ``budget_usd`` for it.

        Needs scikit-learn (the ``[trained]`` extra). On a core-only install it
        raises a directed :class:`ImportError` rather than falling back to the
        constructed default -- identical code must not yield a different
        threshold depending on which extras happen to be installed.

        The derived threshold **survives** ``save``/``load`` (it lives on the
        clusterer / ``ThresholdSelect``); its provenance does **not**
        (``fit_report_`` is never serialized). See
        :class:`~langres.training.fit_report.ThresholdFit`.

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
                Prompt-optimization is implemented (:class:`~langres.training.methods_prompt.Bootstrap`
                / :class:`~langres.training.methods_prompt.MIPRO` compile a
                ``DSPyMatcher``'s prompt -- see :meth:`_fit_prompt`); the
                fine-tune (PR-F) and calibrate (PR-D) handlers are still stubs
                that raise a clear NotImplementedError naming their PR.
            derive_threshold: Derive the match threshold from ``pairs`` and apply
                it (see above). ``False`` (default) leaves the model's constructed
                threshold untouched -- the correct no-data fallback.

        Returns:
            ``self``, so ``resolver.fit(data).resolve(data)`` chains.

        Raises:
            ValueError: If both ``labels`` and ``pairs`` are given; if the module
                implements ``SupervisedFitMixin`` and neither is given; or if
                ``labels``/``pairs`` is given to a module that cannot use them.
                Also if ``derive_threshold=True`` without ``pairs=``, together
                with ``method=``, or on a model that keeps no match cut; and --
                for an explicit ``_ops`` topology, whose only fittable parameter
                is its ``ThresholdSelect`` -- for any ``fit()`` call that is not
                ``pairs=... , derive_threshold=True``. (That last case previously
                surfaced as a ``RuntimeError`` from the four-slot property
                accessors, which could not describe what such a model *can* fit.)
            ImportError: If ``derive_threshold=True`` and scikit-learn (the
                ``[trained]`` extra) is not installed.
            NotImplementedError: If ``method`` is given but its ``kind``'s fit
                path is not implemented yet (the seam is wired ahead of the
                concrete methods; the error names the PR that will land it).
            UnsupportedMethodKind: If this class declares
                :attr:`accepted_method_kinds` and ``method.kind`` is not among
                them. The base ``Resolver`` declares none and never raises it.
        """
        if method is not None:
            if derive_threshold:
                raise ValueError(
                    "fit(method=..., derive_threshold=True) is not supported: a "
                    f"method= fit ({method.describe()}) runs its own handler and "
                    "owns its own report. Fit the method first, then derive the "
                    "cut against the resulting pipeline in a second call -- "
                    "fit(data, method=...) then "
                    "fit(data, pairs=..., split=..., derive_threshold=True). "
                    "That ordering also matters for method='calibrate': the "
                    "threshold must be derived on the CALIBRATED score scale."
                )
            # Architectures that claim an identity refuse the kinds that would
            # change their topology out from under the class name. The base
            # Resolver declares no kinds and this is a no-op for it.
            self._check_method_accepted(method)
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
        if derive_threshold and pairs is None:
            raise ValueError(
                "fit(derive_threshold=True) needs pairs=<id-keyed labels>. A "
                "threshold is derived from a score distribution and its gold "
                "labels, and pairs= is the argument that carries the "
                "entity-disjoint split= -- which is what keeps the derived cut "
                "out of the number you then report. labels= is positional and "
                "carries no split, so it could only ever produce an in-sample "
                "cut. Pass pairs=[LabeledPair(...), ...] or a corrections.jsonl "
                "path (use split=None if you genuinely want the in-sample cut; "
                "the report will say so)."
            )
        if self._ops is not None:
            # An explicit Op chain has NO module/clusterer slot -- reading either
            # raises (see ``_require_bound``) -- so it can never reach the
            # slot-based dispatch below. Its one fittable parameter is the
            # chain's own ThresholdSelect.
            return self._fit_chain_threshold(
                data, pairs=pairs, split=split, seed=seed, derive_threshold=derive_threshold
            )
        if pairs is not None:
            self.fit_report_ = self._fit_from_pairs(
                data, pairs, split=split, seed=seed, derive_threshold=derive_threshold
            )
            return self

        matcher_name = type(self.module).__name__
        if isinstance(self.module, SupervisedFitMixin):
            if labels is None:
                raise ValueError(
                    f"{matcher_name} requires labeled data: pass "
                    "labels=<Sequence[bool] aligned with the blocked candidates> "
                    "(or pairs=<id-keyed labels>) to fit()."
                )
            self.module.fit(iter(self._candidates(data).to_candidates()), labels)
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

    @classmethod
    def _check_method_accepted(cls, method: Method) -> None:
        """Refuse a ``Method`` whose kind this architecture does not accept.

        The enforcement half of :attr:`accepted_method_kinds` (see there for the
        why and the rejected alternatives). A no-op when
        ``accepted_method_kinds`` is ``None`` -- i.e. always, for the base
        :class:`Resolver`.

        Raises:
            UnsupportedMethodKind: If this class declares
                ``accepted_method_kinds`` and ``method.kind`` is not among them.
                The message names the class, the offending kind, the method's
                own ``describe()``, and the kinds that *are* accepted.
        """
        accepted = cls.accepted_method_kinds
        if accepted is None or method.kind in accepted:
            return
        name = cls.__name__
        allowed = ", ".join(repr(k) for k in sorted(accepted)) or "(no method kinds)"
        raise UnsupportedMethodKind(
            f"{name} does not accept method kind {method.kind!r} ({method.describe()}); "
            f"it accepts: {allowed}. That fit would change the pipeline's topology, "
            f"leaving {name} naming a pipeline it no longer is. Use a plain Resolver "
            f"(which accepts every kind) or an architecture built for it."
        )

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
        :class:`~langres.training.methods_prompt.Bootstrap`, ``MIPROv2`` for
        :class:`~langres.training.methods_prompt.MIPRO`). Supervision comes from either
        id-keyed ``pairs`` (joined via :func:`~langres.curation.harvest.align_pairs`,
        whose optional entity-disjoint ``split`` yields the ``valid`` fold
        ``MIPROv2`` uses as its valset) or pre-aligned ``labels``. Sets
        :attr:`fit_report_` naming the demos learned + teacher model + declared
        budget, and returns ``self`` so ``resolver.fit(...).resolve(...)`` chains.

        The budget seam: ``method.budget_usd`` caps the compile via its own
        :class:`~langres.core.spend.SpendMonitor` -- deliberately NOT this
        Resolver's ``budget_usd`` ledger, because DSPy's compile calls never
        reach ``self.module.forward`` and so cannot be metered by
        :meth:`_scorer`. DSPy-compile spend capture is deferred to issue #100 --
        today the compile records ``$0`` (the ``DummyLM`` CI path is genuinely
        free; the paid ``MIPROv2`` path stays uncosted until #100 wires real
        spend through this same guard). This is the one scoring-adjacent path a
        Resolver's cap does not bound; see the CHANGELOG's known-gap note.

        Raises:
            ValueError: If the module is not a ``DSPyMatcher``
                (prompt-optimization needs a compilable scorer); if both
                ``labels`` and ``pairs`` are given, or neither.
        """
        # Lazy import: keep ``dspy`` out of a bare ``import langres`` -- it loads
        # only when a prompt-optimize fit runs. (``SpendMonitor`` needs no lazy
        # import: it is a dependency-free core leaf, imported at module scope.)
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
        them (:func:`~langres.training.finetune.run_finetune`), repoints this
        Resolver's matcher at the resulting ``model_ref`` as an in-process,
        logprob-scoring :class:`~langres.core.matchers.llm_judge.LLMMatcher`, and
        evaluates held-out pair P/R/F1 on the entity-disjoint ``valid`` split.
        Records the GPU-seconds / derived-$ cost and the served ``model_ref`` in
        :attr:`fit_report_`.

        Heavy imports (``training.finetune`` → peft/trl on the training call;
        ``LLMMatcher`` → litellm) are deferred to here so the ``method=None`` and
        non-finetune fit paths never pay for them.

        Raises:
            TypeError: If ``method`` is not a :class:`~langres.training.finetune.QLoRA`.
            ValueError: If neither ``pairs`` nor ``labels`` is given (fine-tuning
                needs supervision), or both are.
        """
        from langres.training.finetune import FINETUNE_YES_NO_PROMPT, QLoRA, run_finetune
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
            judgements = list(self._scorer().forward(iter([c for c, _ in valid_pairs])))
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
        :data:`~langres.training.finetune.FINETUNE_YES_NO_PROMPT` (the outgoing matcher's
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
        :class:`~langres.training.calibration.Calibrator` (strategy from
        ``method.strategy``) on those ``(score, label)`` pairs, and attaches it as
        :attr:`calibrator` so :meth:`predict`/:meth:`resolve` map every raw score
        to a calibrated probability. Supervision comes from id-keyed ``pairs``
        (joined via :func:`~langres.curation.harvest.align_pairs`, whose optional
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
                :class:`~langres.training.methods_calibrate.CalibrateMethod`); if both
                ``labels`` and ``pairs`` are given, or neither; or if the matcher
                emits no scores to calibrate (a pure decider).
        """
        try:
            from langres.training.calibration import Calibrator
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
        for judgement in self._scorer().forward(iter(candidates)):
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
        derive_threshold: bool = False,
    ) -> FitReport:
        """Fit a ``SupervisedFitMixin`` matcher from id-keyed labels via ``align_pairs``.

        Runs the id-join + entity-disjoint split + coverage in one place, trains
        on the train split, and evaluates held-out pair P/R/F1 on the valid split
        (when a split was given, via :func:`~langres.core.metrics.classify_pairs`
        at the clusterer's threshold). Returns the assembled :class:`FitReport`.

        With ``derive_threshold=True`` the cut is derived from the **train** split
        and applied *before* the valid split is graded, so the reported metrics
        measure the derived cut on pairs it never saw. A matcher with no
        supervised fit hook is accepted in that mode: its threshold is still a
        fittable parameter even though the matcher itself is fixed.
        """
        matcher_name = type(self.module).__name__
        supervised = isinstance(self.module, SupervisedFitMixin)
        if not supervised and not derive_threshold:
            raise ValueError(
                f"{matcher_name} does not support fit(pairs=...): pairs= supplies "
                "labeled pairs for a SupervisedFitMixin matcher, and this matcher "
                "implements no supervised fit hook. Use fit() with no labels for an "
                "unsupervised/non-learnable matcher, or pass derive_threshold=True "
                "to fit the decision threshold alone (the one parameter a fixed "
                "scorer does have)."
            )
        aligned = align_pairs(self.candidates(data), pairs, split=split, seed=seed)
        if supervised:
            cast("SupervisedFitMixin[Any]", self.module).fit(
                iter(aligned.train.candidates), aligned.train.labels
            )

        judgements: list[PairwiseJudgement] = []
        gold_pairs: set[frozenset[str]] = set()
        if aligned.valid.candidates:
            judgements = list(self._scorer().forward(iter(aligned.valid.candidates)))
            if derive_threshold:
                # A derived cut sits on the CUT's scale, which a fitted calibrator
                # moves (see :meth:`_cut_scale_scores`). Grading raw scores against
                # it would compare two different scales and report a number that
                # looks fine and means nothing. Only rescaled when something was
                # derived, so the non-deriving path stays byte-identical.
                judgements = self._on_cut_scale(judgements)
            gold_pairs = {
                frozenset({str(c.left.id), str(c.right.id)})
                for c, label in zip(aligned.valid.candidates, aligned.valid.labels, strict=True)
                if label
            }

        # Select BEFORE grading, so the reported metrics measure the cut actually
        # in force. Selection reads ``train`` only; the valid judgements are
        # passed in purely to be *scored* at each candidate, never to choose.
        threshold_fit = ThresholdFit(source="default")
        if derive_threshold:
            _warn_if_silver_only(pairs)
            threshold_fit = self._select_threshold(
                self._labeled_for_derivation(aligned.train.candidates, aligned.train.labels),
                valid_judgements=judgements,
                valid_gold=gold_pairs,
            )

        metrics: PairMetrics | None = None
        if aligned.valid.candidates:
            metrics = classify_pairs(judgements, gold_pairs, self.clusterer.threshold)

        return FitReport.build(
            trainable=(
                f"{matcher_name} (SupervisedFitMixin)"
                if supervised
                else f"{matcher_name} (no fit hook; threshold only)"
            ),
            # ``trained`` means "a matcher fit hook ran" and keeps that meaning: a
            # derived threshold is reported by ``threshold_fit``, not here.
            trained=supervised,
            n_train=len(aligned.train.labels),
            n_valid=len(aligned.valid.labels),
            split=split,
            seed=seed,
            coverage=aligned.coverage,
            threshold=self.clusterer.threshold,
            threshold_fit=threshold_fit,
            metrics=metrics,
            run_ref=current_run.get(),
        )

    # ------------------------------------------------------------------
    # Threshold fitting -- the ONE derivation seam, shared by both topologies.
    #
    # The derivation itself is NOT reimplemented here: it delegates to
    # ``curation.harvest.derive_threshold_from_pairs`` (which delegates in turn
    # to ``training.calibration.derive_threshold``), the same function the
    # harvest loop and the fixed-split pair benchmark already call. What lives
    # here is only the two things fit() adds: getting scores on the scale the cut
    # actually sees, and writing the result to whichever seam this model keeps
    # its cut in.
    # ------------------------------------------------------------------

    def _labeled_for_derivation(
        self, candidates: Sequence[ERCandidate[Any]], labels: Sequence[bool]
    ) -> list[LabeledPair]:
        """Score ``candidates`` and pair each score with its gold label.

        Scores exactly the labeled candidates -- not every blocked pair -- through
        :meth:`_scorer`, so a paid matcher bills for the labeled set alone and
        stays inside this model's ``budget_usd`` ledger.
        """
        judgements = list(self._scorer().forward(iter(candidates)))
        by_pair = {frozenset({j.left_id, j.right_id}): j for j in judgements}
        scores = self._cut_scale_scores([by_pair.get(_pair_key(c)) for c in candidates])
        return [
            LabeledPair(
                left_id=str(candidate.left.id),
                right_id=str(candidate.right.id),
                score=score,
                label=label,
                # These labels were asserted by the CALLER, not read back off the
                # judge's own verdicts, so they are gold rather than silver.
                # Stamping "verdict" would make derive_threshold_from_pairs warn
                # about a circularity that is not present; the genuinely-silver
                # case is caught by ``_warn_if_silver_only`` on the raw input.
                source="correction",
            )
            for candidate, label, score in zip(candidates, labels, scores, strict=True)
        ]

    def _cut_scale_scores(
        self, judgements: Sequence[PairwiseJudgement | None]
    ) -> list[float | None]:
        """Judgement scores on the scale the match cut actually sees.

        A fitted :attr:`calibrator` remaps every scored ranking row before
        clustering (:meth:`_scored_pairs` -> ``CalibratorScore``), so a threshold
        derived from RAW matcher scores would be a cut on a different scale than
        the one ``resolve()`` thresholds against. Mirror that mapping here.
        Score-less rows (a decider, or a candidate the matcher returned no
        judgement for) pass through as ``None``, exactly as
        :meth:`_apply_calibrator_to_pairs` leaves them --
        ``derive_threshold_from_pairs`` then refuses the input by name rather
        than silently calibrating on the scored subset.
        """
        scores: list[float | None] = [None if j is None else j.score for j in judgements]
        calibrator = self.calibrator
        if calibrator is None:
            return scores
        scored = [i for i, score in enumerate(scores) if score is not None]
        mapped = calibrator.transform([cast(float, scores[i]) for i in scored])
        for index, value in zip(scored, mapped, strict=True):
            scores[index] = value
        return scores

    def _on_cut_scale(self, judgements: Sequence[PairwiseJudgement]) -> list[PairwiseJudgement]:
        """Copies of ``judgements`` with their scores moved onto the match cut's scale.

        The judgement-shaped counterpart of :meth:`_cut_scale_scores`, for the
        graders (:func:`~langres.core.metrics.classify_pairs`) that want objects
        rather than a score list. Pure copies -- the caller's judgements are not
        mutated.
        """
        scores = self._cut_scale_scores(list(judgements))
        return [
            judgement.model_copy(update={"score": score})
            for judgement, score in zip(judgements, scores, strict=True)
        ]

    def _select_threshold(
        self,
        train: Sequence[LabeledPair],
        *,
        valid_judgements: list[PairwiseJudgement],
        valid_gold: set[frozenset[str]],
    ) -> ThresholdFit:
        """Derive a candidate cut, race it against the incumbent, keep the winner.

        **Deriving is not the same as keeping.** Youden's J maximizes
        ``tpr - fpr`` on ``train``; on a wide, cleanly-separated margin that can
        return a cut which ties there and loses on unseen pairs (measured on a
        toy fixture: held-out pair-F1 1.00 -> 0.80). So the derived cut has to
        earn the seat: it is applied only if it *strictly* beats the incumbent's
        pair-F1. A tie keeps the incumbent -- do not move a threshold without
        evidence that moving it helps.

        **The race runs on ``train``, never on ``valid``.** Picking the winner on
        ``valid`` would make ``valid`` a selection set and silently downgrade
        :attr:`FitReport.metrics` from a held-out estimate to an optimistic one.
        ``train`` is already spent (the candidate was derived from it), so
        selecting there costs no further honesty. ``valid_judgements`` are used
        only to *score* both cuts for the report -- they never influence the
        choice -- which is why both ``held_out_f1`` values stay clean estimates.

        Args:
            train: Scored + labeled train pairs; both the derivation input and
                the selection set.
            valid_judgements: Held-out judgements, already on the cut's scale.
                Empty when no split was given.
            valid_gold: Held-out gold pair keys, aligned with the above.

        Returns:
            The :class:`ThresholdFit` provenance, with the threshold already
            written to this model when the candidate won.

        Raises:
            ValueError: If this model keeps no match cut at all (an explicit chain
                with no ``ThresholdSelect``) -- deriving a number and writing it
                nowhere would be the silent no-op this whole seam exists to avoid.
        """
        seam = self._threshold_seam()
        if seam is None:
            raise ValueError(
                "fit(derive_threshold=True) found no decision threshold to fit: "
                "this model's explicit Op chain contains no ThresholdSelect, so "
                "there is no match cut to write. Add a ThresholdSelect to the "
                "topology (or use a classic four-slot model, whose cut lives on "
                "the clusterer)."
            )
        incumbent = cast(float, self._match_threshold())
        candidate = derive_threshold_from_pairs(train)

        def _described(threshold: float) -> ThresholdCandidate:
            return ThresholdCandidate(
                threshold=threshold,
                selection_f1=_pair_f1(train, threshold),
                held_out_f1=(
                    classify_pairs(valid_judgements, valid_gold, threshold).f1
                    if valid_judgements
                    else None
                ),
            )

        previous_described, candidate_described = _described(incumbent), _described(candidate)
        kept = cast(float, candidate_described.selection_f1) > cast(
            float, previous_described.selection_f1
        )
        if kept:
            self._set_match_threshold(candidate)
        return ThresholdFit(
            source="derived" if kept else "declined",
            method=_THRESHOLD_METHOD,
            n_pairs=len(train),
            held_out=bool(valid_judgements),
            # Nothing was written when the candidate lost, so naming a seam would
            # overclaim -- the threshold is still the one the model came with.
            applied_to=seam if kept else None,
            selected_on="train",
            previous=previous_described,
            candidate=candidate_described,
        )

    def _fit_chain_threshold(
        self,
        data: list[Any],
        *,
        pairs: str | Path | Sequence[LabeledPair] | Sequence[Correction] | None,
        split: float | None,
        seed: int,
        derive_threshold: bool,
    ) -> Self:
        """``fit`` for an explicit ``_ops`` chain: the threshold is all there is to fit.

        An explicit chain has no ``module`` slot to train and no ``clusterer``
        slot to read -- both properties raise -- so the slot-based dispatch in
        :meth:`fit` cannot run here at all. Its one fittable parameter is the
        terminal :class:`~langres.core.op.ThresholdSelect`.

        The chain is run ONCE, with that ThresholdSelect omitted
        (:meth:`_prethreshold_pairs`), so the derivation sees the rows below the
        current cut too. Both splits are then read off that single pass -- a
        chain's Source owns retrieval, so unlike a classic model it cannot score
        an arbitrary candidate subset.
        """
        if not derive_threshold or pairs is None:
            raise ValueError(
                "fit() on an explicit Op topology can only fit the chain's "
                "ThresholdSelect: there is no matcher slot to train (the chain's "
                "Scores are topology, not a fittable slot). Call "
                "fit(data, pairs=<id-keyed labels>, split=..., "
                "derive_threshold=True), or build a classic four-slot model if you "
                "need to train the matcher itself."
            )
        scored = self._prethreshold_pairs(data)
        score_by_pair = {_row_key(row): row.score for row in scored.rows}
        judgement_by_pair = {
            _row_key(row): row.to_judgement() for row in scored.rows if row.score_type is not None
        }
        aligned = align_pairs(scored.to_candidates(), pairs, split=split, seed=seed)

        judgements = [
            judgement
            for candidate in aligned.valid.candidates
            if (judgement := judgement_by_pair.get(_pair_key(candidate))) is not None
        ]
        gold_pairs = {
            _pair_key(c)
            for c, label in zip(aligned.valid.candidates, aligned.valid.labels, strict=True)
            if label
        }

        _warn_if_silver_only(pairs)
        threshold_fit = self._select_threshold(
            [
                LabeledPair(
                    left_id=str(candidate.left.id),
                    right_id=str(candidate.right.id),
                    score=score_by_pair.get(_pair_key(candidate)),
                    label=label,
                    source="correction",  # caller-asserted gold; see _labeled_for_derivation
                )
                for candidate, label in zip(
                    aligned.train.candidates, aligned.train.labels, strict=True
                )
            ],
            valid_judgements=judgements,
            valid_gold=gold_pairs,
        )

        metrics: PairMetrics | None = None
        if aligned.valid.candidates:
            metrics = classify_pairs(judgements, gold_pairs, cast(float, self._match_threshold()))

        matcher = self._chain_scoring_matcher()
        name = "explicit Op chain" if matcher is None else type(matcher).__name__
        self.fit_report_ = FitReport.build(
            trainable=f"{name} (threshold only; an Op chain has no matcher slot to fit)",
            trained=False,
            n_train=len(aligned.train.labels),
            n_valid=len(aligned.valid.labels),
            split=split,
            seed=seed,
            coverage=aligned.coverage,
            threshold=self._match_threshold(),
            threshold_fit=threshold_fit,
            metrics=metrics,
            run_ref=current_run.get(),
        )
        return self

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
            The built :class:`~langres.curation.anchor_store.AnchorStore`.
        """
        from langres.curation.anchor_store import AnchorStore

        self._anchor_store = AnchorStore.build(self, records, entity_prefix=entity_prefix)
        return self._anchor_store

    def assign(self, record: Any) -> "ClusterDelta":
        """Assign one new record to an existing entity, or mint a new one (M5, S6).

        Incremental single-record resolution against the anchor set built by
        :meth:`build_anchor_store`: returns a
        :class:`~langres.curation.anchor_store.ClusterDelta` that either ``link``\\ s
        the record to an existing entity (with a stable id) or marks it ``new``.
        Distinct from the reserved cross-source :meth:`link` /
        :meth:`stream_against` stubs — ``assign`` is single-record incremental
        linking.

        Args:
            record: A raw record dict, same shape as ``resolve()`` accepts.

        Returns:
            A :class:`~langres.curation.anchor_store.ClusterDelta`.

        Raises:
            RuntimeError: If :meth:`build_anchor_store` was not called first.
        """
        if self._anchor_store is None:
            raise RuntimeError(
                "call build_anchor_store(records) before assign(record): assign "
                "resolves a new record against a prior batch's anchor set."
            )
        return self._anchor_store.assign(record)

    @classmethod
    def load(cls, path: str | Path) -> "Resolver":
        """Reconstruct a Resolver from an artifact directory written by ``save``.

        Reads ``resolver.json``, validates the artifact version, and rebuilds
        each slot from the component registry by its ``type_name`` (no code
        execution, no pickle). Sidecar state is restored for any
        :class:`~langres.core.serialization.SerializableState` component. That
        whole mechanism lives in
        :meth:`~langres.core._model_persist.ModelPersistence._read_artifact`;
        what stays here is the one decision that needs the ``ERModel`` name in
        scope -- *which class* to build.

        Reconstructs the **class** the manifest names in ``model_class`` (a
        registered architecture), not merely the one ``load`` was called on -- so
        ``Resolver.load(<a FuzzyString artifact>)`` hands back a ``FuzzyString``.
        An artifact with no ``model_class`` (every pre-0.4 one, and any
        unregistered subclass) builds ``cls``, exactly as before.

        Reconstruction goes through
        :meth:`~langres.core._model_state.ModelState.from_components`, **not** the
        class's ``__init__`` -- so a named architecture is free to have an
        ergonomic constructor (``FuzzyString(threshold=0.8)``) that knows nothing
        about component keywords. This is W4 paying the debt #179 recorded right
        here: before ``model_class``, ``load`` always built the base class and the
        collision could not bite; after it, calling ``FuzzyString(blocker=...)``
        raised ``TypeError: unexpected keyword argument 'blocker'``. See
        ``from_components`` for why building-from-config beats replaying
        constructor args, and for the one invariant it asks of an architecture.

        An **explicit-chain** artifact (epic #193 persist v2, an ``ops`` list
        instead of ``components``) reconstructs through
        :meth:`~langres.core._model_state.ModelState.from_topology` instead, which
        re-secures each raw ``MatcherScore`` against a fresh default-budget
        :class:`~langres.core.spend.SpendMonitor` -- the spend cap is re-established
        on load, never persisted.

        Args:
            path: Directory containing ``resolver.json`` and any sidecars.

        Returns:
            A Resolver equivalent to the one that was saved -- of the saved
            architecture's class when the manifest names one.

        Raises:
            ValueError: If the artifact's ``artifact_version`` is newer than this
                library understands.
            UnknownModelType: If the manifest names a ``model_class`` this
                process has not registered (usually: its module was never
                imported).
        """
        manifest, payload = cls._read_artifact(path)

        # Reconstruct the class the artifact says it is, so a named architecture
        # round-trips as itself instead of decaying into a plain Resolver. Absent
        # ``model_class`` (every pre-0.4 artifact, and any unregistered subclass)
        # keeps the old behavior exactly: build ``cls``, whatever load was called on.
        target = cls if manifest.model_class is None else get_model(manifest.model_class)
        # The model registry cannot type-check its own entries: registry.py sits
        # beneath this module, so it can neither import Resolver nor annotate
        # ``get_model`` with it. Verify here, where Resolver IS in scope. Without
        # this, a ``model_class`` registered to a kwargs-swallowing non-Resolver
        # loads and is cast to Resolver, handing the caller a silently wrong
        # object instead of an error.
        if not issubclass(target, ERModel):
            raise TypeError(
                f"Artifact model_class {manifest.model_class!r} is registered to "
                f"{target.__module__}.{target.__qualname__}, which is not an ERModel "
                f"subclass; register_model is for ERModel subclasses (architectures)."
            )
        # No cast needed: the issubclass guard above narrows ``target`` to
        # ``type[ERModel]`` for the type checker.
        #
        # Explicit Op chain (persist v2): ``_read_artifact`` returns ``{"ops": [...]}``
        # of RAW-matcher stages -> from_topology, which re-secures every paid Score
        # against a FRESH default-budget monitor (the budget is a run policy, not
        # architecture, and is deliberately not persisted -- exactly as
        # from_components refuses to bake in a saved budget). Otherwise the classic
        # four-slot kwargs -> from_components (NOT the class's own __init__, so an
        # architecture keeps its ergonomic signature).
        if "ops" in payload:
            return target.from_topology(**payload)
        return target.from_components(**payload)


#: The pre-W4 name for :class:`ERModel`, kept as a plain alias.
#:
#: ``Resolver`` described a *thing that resolves*; ``ERModel`` describes *the
#: model you chose*, which is the distinction the whole refactor is about. The
#: class was reshaped in place rather than replaced (it already had
#: ``from_schema``/``resolve``/``save``/``load``/``fit`` -- what it lacked was
#: identity, typed slots, and a front door), so this is one class under two
#: names, not a compatibility shim wrapping another object: ``Resolver is
#: ERModel`` is True, and ``isinstance``/``issubclass`` behave identically.
#:
#: It stays because the name is load-bearing across the repo -- ``CLAUDE.md``
#: mandates ``langres.core`` carry "the ``Resolver``", ``resolver.json`` is the
#: artifact filename, and the docs/tests reference it heavily. Renaming those is
#: W5's documentation sweep, not this wave's.
Resolver = ERModel
