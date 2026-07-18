"""What an ER model *is*: four slots, an identity, and the doors that fill them.

The base of the ``ERModel`` class chain (:mod:`langres.core.resolver` assembles
the leaf). This module owns **construction and identity** and nothing else: the
three construction doors, the four typed slots behind their properties, the
schema binding a named architecture defers until first use, and the spend ledger
every scoring seam meters through.

It deliberately does NOT know how a model runs (:mod:`langres.core._model_run`),
trains, or persists (:mod:`langres.core._model_persist`). Those layers subclass
this one, so they inherit the slots as **non-Optional** types instead of
re-declaring them -- which is why this is a base class rather than a bag of
mixins: a mixin annotating ``blocker: Blocker[Any]`` would collide with the
property that actually implements it.
"""

import logging
from collections.abc import Sequence
from typing import Any, ClassVar, Self, cast

from pydantic import BaseModel

from langres.core.blocker import Blocker
from langres.core.clusterer import Clusterer
from langres.core.comparator import Comparator
from langres.core.fit import CalibratorFitMixin
from langres.training.fit_report import FitReport
from langres.core.inputs import normalize_records
from langres.core.matcher import Matcher
from langres.core.op import (
    ClusterStage,
    Op,
    Score,
    Sequential,
    Source,
    Stage,
    ThresholdSelect,
)
from langres.core.spend import SpendMonitor
from langres.core.spend_cap import SpendCappedMatcher, effective_budget

logger = logging.getLogger(__name__)


class ModelState:
    """The slots, the identity and the construction doors of an ``ERModel``.

    See :class:`~langres.core.resolver.ERModel` for the user-facing picture; this
    class is the half of it that answers "what is this model made of, and how did
    it get that way".
    """

    #: The ``Method.kind``s :meth:`~langres.core.resolver.ERModel.fit` accepts;
    #: ``None`` (the default) accepts every kind.
    #:
    #: This exists for the named architectures of W4 (``FuzzyString``,
    #: ``VectorLLMCascade``, ...), which make the claim *one architecture = one
    #: class = one identity*. That claim is falsifiable by ``fit`` itself:
    #: ``_fit_finetune`` deliberately **repoints the matcher slot** at the
    #: fine-tuned model, so ``FuzzyString().fit(method=QLoRA(...))`` would return
    #: an LLM-backed pipeline still calling itself ``FuzzyString``. A subclass
    #: declares the kinds it can absorb without ceasing to be itself; anything
    #: else is refused at the ``fit`` boundary with
    #: :class:`~langres.core.methods_api.UnsupportedMethodKind`.
    #:
    #: The base ``Resolver`` stays ``None`` -- **permissive on purpose**. It makes
    #: no identity claim ("a resolver" is not a topology), so there is nothing for
    #: a topology change to falsify, and every fit path it accepts today keeps
    #: working unchanged.
    #:
    #: Alternatives considered and rejected (recorded so this is cheap to flip):
    #:
    #: - *A topology-changing ``fit()`` returns a NEW architecture instance*
    #:   (sklearn ``clone``-like). More principled -- identity would follow the
    #:   topology instead of constraining it -- but ``fit`` mutates in place and
    #:   returns ``self``/metrics today, and the flywheel loop depends on that;
    #:   changing the return contract is far more churn than this gate.
    #: - *Downgrade the invariant to advisory* (document that ``fit`` may change
    #:   what the class name means). Zero code, but it guts the whole point of
    #:   naming architectures: a name that may silently describe something else is
    #:   not an identity.
    accepted_method_kinds: ClassVar[frozenset[str] | None] = None

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

                Scope, precisely: the cap meters **every seam that scores
                through the matcher** -- ``resolve``, ``predict``, ``fit``, and
                :meth:`~langres.curation.anchor_store.AnchorStore.assign` -- across
                every call on this instance, because they all route through
                ``_scorer``. It bounds spend at ``budget_usd`` plus at most
                one further call (see :mod:`langres.core.spend_cap`).

                The one exception, deliberately: ``fit(method=MIPRO())``.
                DSPy's compile calls never reach ``self.module.forward``, so
                this ledger cannot observe them; it caps them via its own
                ``method.budget_usd`` monitor instead (which records $0 until
                issue #100 captures compile spend). See ``_fit_prompt``.
        """
        self._init_state(budget_usd=budget_usd)
        self._wire(
            blocker=blocker,
            comparator=comparator,
            matcher=matcher,
            clusterer=clusterer,
            calibrator=calibrator,
        )

    def _init_state(self, *, budget_usd: float | None) -> None:
        """Set up the non-slot state every construction door needs.

        Split out of ``__init__`` so all three doors -- ``__init__``,
        :meth:`from_components`, and a named architecture's own ergonomic
        ``__init__`` -- share ONE definition of "a wired-up model's state",
        instead of each remembering to build a monitor and null four fields.

        Subclasses that own extra state extend this rather than ``__init__``
        (which :meth:`from_components` deliberately never runs): see
        :meth:`~langres.core.resolver.ERModel._init_state`, which adds the
        incremental-resolution anchor store on top of ``super()._init_state``.
        """
        # ONE ledger for this model's lifetime, so N resolve() calls share one
        # budget instead of getting a fresh one each (B1). The monitor -- not the
        # wrapper -- is the durable thing: `self.module` is reassignable
        # (fit(method=Finetune()) replaces it), so _scorer() re-wraps the CURRENT
        # module around this same ledger.
        self._spend_monitor = SpendMonitor(budget_usd=effective_budget(budget_usd))
        # Optional score->probability map, set by fit(method=Platt()/Isotonic())
        # and applied in _judgements(); None leaves raw scores untouched.
        self.calibrator: CalibratorFitMixin | None = None
        # Set by fit(); the sklearn trailing-underscore "produced by fit" digest.
        # None until fit() runs (never serialized -- it is a fit-time artifact).
        self.fit_report_: FitReport | None = None
        # Slots start empty so a named architecture can defer binding until it
        # knows the schema (see _topology). The base __init__ fills them at once.
        #
        # They are private + property-backed (below) rather than plain nullable
        # attributes for a measured reason: annotating `self.blocker` as
        # `Blocker | None` makes every one of the ~20 `self.blocker.stream(...)` /
        # `self.clusterer.threshold` call sites a mypy error, and
        # `assert self.blocker is not None` at each is noise that also lies (it
        # says "impossible" about a state that is reachable). The properties keep
        # every consumer's type non-Optional and turn the unbound case into ONE
        # directed message instead of `'NoneType' object has no attribute 'stream'`.
        self._blocker: Blocker[Any] | None = None
        self._comparator: Comparator[Any] | None = None
        self._module: Matcher[Any] | None = None
        self._clusterer: Clusterer | None = None
        # An OPTIONAL explicit Op chain (epic #193, PR-A). ``None`` -> this is a
        # classic four-slot topology and every run/read below takes its
        # slot-derived path unchanged; a list -> this model runs the explicit
        # chain instead (set ONLY by :meth:`from_topology`). A ``Stage`` is a
        # ``Source | Op | ClusterStage | Finalize`` (op.py's union): the Source
        # (records -> pairs), the body Ops (pairs -> pairs, INCLUDING Selects),
        # then the terminal ClusterStage (pairs -> clusters) and an optional
        # Finalize. All classic construction doors leave it ``None``.
        self._ops: list[Stage] | None = None

    @property
    def blocker(self) -> Blocker[Any]:
        """The candidate generator. Raises if this model is not bound yet."""
        self._require_bound("use the blocker")
        return cast(Blocker[Any], self._blocker)

    @blocker.setter
    def blocker(self, value: Blocker[Any]) -> None:
        self._blocker = value

    @property
    def comparator(self) -> Comparator[Any] | None:
        """The optional per-feature comparison stage (genuinely ``None`` when unset)."""
        return self._comparator

    @comparator.setter
    def comparator(self, value: Comparator[Any] | None) -> None:
        self._comparator = value

    @property
    def module(self) -> Matcher[Any]:
        """The scorer. Raises if this model is not bound yet.

        Public and reassignable (``fit(method=QLoRA())`` repoints it), which is
        exactly why ``_scorer`` rebuilds its wrapper per call rather than
        caching one -- see that method.
        """
        self._require_bound("use the matcher")
        return cast(Matcher[Any], self._module)

    @module.setter
    def module(self, value: Matcher[Any]) -> None:
        self._module = value

    @property
    def clusterer(self) -> Clusterer:
        """The grouping stage. Raises if this model is not bound yet."""
        self._require_bound("use the clusterer")
        return cast(Clusterer, self._clusterer)

    @clusterer.setter
    def clusterer(self, value: Clusterer) -> None:
        self._clusterer = value

    def _wire(
        self,
        *,
        blocker: Blocker[Any],
        comparator: Comparator[Any] | None,
        matcher: Matcher[Any],
        clusterer: Clusterer,
        calibrator: CalibratorFitMixin | None = None,
    ) -> None:
        """Fill the four slots. The ONE place a model's topology is set."""
        self.blocker = blocker
        self.comparator = comparator
        self.module = matcher
        self.clusterer = clusterer
        if calibrator is not None:
            self.calibrator = calibrator

    @classmethod
    def from_components(
        cls,
        *,
        blocker: Blocker[Any],
        comparator: Comparator[Any] | None,
        matcher: Matcher[Any],
        clusterer: Clusterer,
        calibrator: CalibratorFitMixin | None = None,
        budget_usd: float | None = None,
    ) -> Self:
        """Build an instance from already-built components, **bypassing ``__init__``**.

        The load-bearing decision of W4, and the reason
        :meth:`~langres.core.resolver.ERModel.load` works at all.

        The problem it solves, verified rather than assumed: ``load`` must
        reconstruct the class the artifact names (PR #179's ``model_class``), and
        it only has *components* to hand -- it rebuilt each slot from the registry.
        But a named architecture's whole point is an ergonomic ``__init__``
        (``FuzzyString(threshold=0.8)``, ``VectorLLMCascade(llm=...)``), which
        does not accept ``blocker=``/``comparator=``/``matcher=``/``clusterer=``.
        Calling it with those raises ``TypeError: unexpected keyword argument
        'blocker'``. So the two headline goals -- "a saved architecture reloads as
        itself" and "an architecture has an ergonomic constructor" -- collide, and
        exactly one of them can go through ``__init__``. (``load`` predates the
        collision: before ``model_class`` it always built the base class, so it
        could not bite. #179 spotted it and deferred the fix here, in writing.)

        The fix follows what every comparable framework does: **loading builds
        from config, it does not replay constructor args.** ``from_pretrained``
        does not re-run your ``__init__``'s argument parsing; ``load_state_dict``
        restores state into an already-shaped object. So this door skips
        ``__init__`` entirely (``__new__`` + the shared state/slot setup) and
        wires the saved components straight in.

        .. important::
           **The invariant this buys, and its price:** because ``__init__`` never
           runs, an architecture must keep **all** of its identity *in its slots*
           -- the threshold in the ``Clusterer``, the weights in the
           ``Comparator``, the backbone in the ``Matcher``. An architecture that
           stashed extra state on ``self`` in ``__init__`` would come back from
           ``load`` without it. That is not a limitation so much as the design:
           the slots are the single source of truth, which is also exactly what
           makes ``save`` able to write a *complete* config.
           ``TestProof4WeightlessRoundTrip`` in
           ``tests/architectures/test_w4_proofs.py`` pins it by re-saving a
           freshly loaded model and diffing the config.

        Args:
            blocker: Candidate generator + schema normalizer.
            comparator: Optional per-feature comparison stage.
            matcher: The scorer.
            clusterer: Connected-components grouping.
            calibrator: Optional fitted score->probability map.
            budget_usd: Spend cap for the new instance's lifetime (see
                :meth:`__init__`). Deliberately NOT read from the artifact: a
                budget is a *run policy*, not architecture, and is not in the
                manifest -- a reloaded model gets this call's cap (defaulting to
                :data:`~langres.core.spend_cap.DEFAULT_BUDGET_USD`), never a
                stale one baked in months ago by whoever saved it.

        Returns:
            A wired instance of ``cls``, ready to run.
        """
        model = cls.__new__(cls)
        model._init_state(budget_usd=budget_usd)
        model._wire(
            blocker=blocker,
            comparator=comparator,
            matcher=matcher,
            clusterer=clusterer,
            calibrator=calibrator,
        )
        return model

    @classmethod
    def from_topology(
        cls,
        *,
        ops: Sequence[Stage],
        budget_usd: float | None = None,
        monitor: SpendMonitor | None = None,
        calibrator: CalibratorFitMixin | None = None,
    ) -> Self:
        """Build a model that runs an EXPLICIT Op chain instead of the four slots.

        The breaking-change door of the final wave (epic #193, PR-A): where
        :meth:`from_components` wires a classic four-slot topology (which the run
        path *derives* an Op chain from), this door takes the chain **directly** --
        so a model can express a topology the four slots cannot (a Score after a
        Select, a second matcher, a retrieval-only prefix). Like
        :meth:`from_components` it bypasses ``__init__`` (``__new__`` + the shared
        state setup), then stores the chain in ``self._ops`` and leaves the four
        slots empty. Every method dispatches on ``self._ops``: ``None`` (every
        classic door) runs the slot-derived path unchanged; a list walks this
        chain.

        The chain's wiring is validated at construction via :class:`~langres.core.op.Sequential`
        (Source first, then Ops, then a ClusterStage, optionally a Finalize -- a
        Select over a vector score is rejected), and it must carry exactly one
        terminal :class:`~langres.core.op.ClusterStage` so ``resolve``/``dedupe``
        have a phase-1 exit.

        Args:
            ops: The explicit chain, in pipeline order (a Source, zero+ Ops
                including Selects, one terminal ClusterStage, optional Finalize).
                **Any paid Score in the chain must already wrap its matcher in a
                :class:`~langres.core.spend_cap.SpendCappedMatcher`** -- the
                ``.module.forward`` AST-ban only catches the literal slot call, so
                nothing mechanical stops an explicit-chain Score from billing
                off-ledger. Share this model's ledger by passing that same
                ``monitor=`` here.
            budget_usd: Spend cap for this instance's lifetime (see
                :meth:`__init__`). Mutually exclusive with ``monitor``.
            monitor: An existing :class:`~langres.core.spend.SpendMonitor` to adopt
                as this model's ledger -- pass the SAME monitor the chain's
                ``SpendCappedMatcher`` wraps, so the model's budget genuinely caps
                the chain. Mutually exclusive with ``budget_usd``.
            calibrator: **Rejected.** Calibration is a bespoke per-score transform
                that only the classic path applies; a ``CalibratorScore`` Op is a
                later wave, so an explicit chain must express any score mapping as
                its own Score. Passing one raises.

        Returns:
            A model wired to run ``ops``, ready to ``resolve``/``dedupe``/``compare``.

        Raises:
            ValueError: If ``calibrator`` is given, if both ``budget_usd`` and
                ``monitor`` are given, if the chain is mis-wired, or if it does not
                carry exactly one terminal ClusterStage.
        """
        if calibrator is not None:
            raise ValueError(
                "from_topology() does not accept a calibrator: calibration is a bespoke "
                "per-score transform only the classic four-slot path applies (a CalibratorScore "
                "Op is a later wave). Fix: express any score mapping as a Score in the chain, or "
                "build the classic path with from_components(...)."
            )
        if monitor is not None and budget_usd is not None:
            raise ValueError(
                "pass budget_usd= or monitor=, not both: a SpendMonitor already carries its own "
                f"budget (${monitor.budget_usd:.2f}), so budget_usd={budget_usd!r} would silently "
                "lose. Pass the monitor the chain's SpendCappedMatcher already shares."
            )
        chain = list(ops)
        # Reuse the one wiring guard rather than a second, decoupled check: a Source
        # first, Ops (pairs -> pairs) in the middle, a ClusterStage, an optional
        # Finalize -- and NO Select over a non-orderable vector score.
        Sequential(chain)
        cluster_stages = [stage for stage in chain if isinstance(stage, ClusterStage)]
        if len(cluster_stages) != 1:
            raise ValueError(
                f"from_topology() needs exactly one terminal ClusterStage (pairs -> clusters) so "
                f"resolve()/dedupe() have a phase-1 exit, but the chain has {len(cluster_stages)}. "
                "Fix: end the chain with exactly one ClusterStage."
            )
        model = cls.__new__(cls)
        model._init_state(budget_usd=budget_usd)
        if monitor is not None:
            # Adopt the caller's ledger so the model's cap IS the chain's cap.
            model._spend_monitor = monitor
        model._ops = chain
        return model

    def _topology(self, schema: type[BaseModel]) -> dict[str, Any]:
        """Build this architecture's components for ``schema``. The subclass hook.

        A named architecture stores *hyperparameters* in its ``__init__`` and
        implements this to turn them + a schema into the four slots -- which lets
        ``FuzzyString().dedupe(records)`` work with no schema named anywhere: the
        schema is inferred from the records, then handed here, on first use.

        Returns:
            The kwargs for :meth:`_wire` (``blocker``/``comparator``/
            ``matcher``/``clusterer``).

        Raises:
            NotImplementedError: On the base ``ERModel``, which is always
                component-wired at construction and so never needs to build a
                topology of its own.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no components and defines no _topology(schema) hook, "
            "so there is nothing to run. Either construct it with explicit components "
            f"({type(self).__name__}(blocker=..., comparator=..., matcher=..., clusterer=...)), "
            "or use a named architecture from langres.architectures (e.g. FuzzyString())."
        )

    @property
    def is_bound(self) -> bool:
        """Whether the four slots are filled and this model can run.

        ``False`` only for a named architecture constructed without a ``schema=``
        and not yet used -- it binds on its first ``dedupe``/``compare``.
        """
        return self._blocker is not None

    @property
    def schema(self) -> type[BaseModel] | None:
        """The schema this model is bound to, or ``None`` while unbound.

        Derived from the blocker slot rather than stored: keeping it as separate
        state would be a second source of truth that :meth:`from_components`
        (which never runs ``__init__``) could not restore.
        """
        return None if self._blocker is None else getattr(self._blocker, "schema", None)

    @property
    def backbone(self) -> str | None:
        """The underlying model that scores here -- an LLM id, an embedder -- or ``None``.

        Honest by default: it reports the matcher's own ``model`` attribute (both
        LLM matcher families expose one), and ``None`` when there is nothing with
        weights in the scoring slot (pure string similarity) rather than
        fabricating an identity. An architecture whose backbone lives elsewhere
        (e.g. in the blocker's embedder) overrides this to say so.

        For an explicit-chain model (``_ops`` set) there is no ``module`` slot, so
        it reads the chain's own scoring matcher instead (:meth:`_chain_scoring_matcher`,
        unwrapped past its spend cap), and ``None`` when that matcher exposes no
        ``model``.
        """
        if self._ops is not None:
            matcher = self._chain_scoring_matcher()
            candidate = getattr(matcher, "model", None) if matcher is not None else None
            return candidate if isinstance(candidate, str) else None
        candidate = getattr(self.module, "model", None)
        return candidate if isinstance(candidate, str) else None

    # ------------------------------------------------------------------
    # Explicit-chain readers (``_ops`` set) -- the chain is the source of
    # truth, so these parse it instead of reading the four slots. Every one
    # asserts ``_ops`` is set; callers guard on ``self._ops is not None``.
    # ``from_topology`` (via Sequential) guarantees the shape they assume:
    # a Source first, Ops in the middle, exactly one ClusterStage.
    # ------------------------------------------------------------------

    def _require_ops(self) -> list[Stage]:
        assert self._ops is not None  # callers guard on ``self._ops is not None``
        return self._ops

    def _chain_source(self) -> Source[Any]:
        """The chain's Source (records -> pairs) -- ``ops[0]``, guaranteed a Source."""
        source = self._require_ops()[0]
        assert isinstance(source, Source)  # Sequential requires a Source first
        return source

    def _chain_body(self) -> list[Op[Any]]:
        """The body Ops (Scores AND Selects). Every ``Op`` in the chain is a body Op:
        the Source, ClusterStage and Finalize are not ``Op``\\ s, so this filter picks
        out exactly the pairs -> pairs middle."""
        return [stage for stage in self._require_ops() if isinstance(stage, Op)]

    def _chain_body_scores(self) -> list[Score[Any]]:
        """The body's Score ops only (skips Selects) -- what ``compare`` folds on one pair."""
        return [op for op in self._chain_body() if isinstance(op, Score)]

    def _chain_cluster_stage(self) -> ClusterStage[Any]:
        """The chain's terminal ClusterStage (from_topology guarantees exactly one)."""
        for stage in self._require_ops():
            if isinstance(stage, ClusterStage):
                return stage
        raise AssertionError(  # pragma: no cover - from_topology guarantees exactly one
            "explicit chain has no ClusterStage"
        )

    def _chain_threshold(self) -> float | None:
        """The terminal :class:`~langres.core.op.ThresholdSelect`'s threshold -- the chain's
        match cut -- or ``None`` if the chain has no ThresholdSelect. The LAST one wins."""
        threshold: float | None = None
        for op in self._chain_body():
            if isinstance(op, ThresholdSelect):
                threshold = op.threshold
        return threshold

    def _chain_scoring_matcher(self) -> Matcher[Any] | None:
        """The chain's last matcher-``Score``'s matcher, unwrapped past its spend cap.

        The chain's paid Scores wrap their matcher in a
        :class:`~langres.core.spend_cap.SpendCappedMatcher` (that wrapper has no
        ``model``), so peel it to expose the real matcher for :attr:`backbone`. A
        matcher-Score is a Score carrying a ``matcher`` attribute (``MatcherScore`` /
        ``GroupwiseMatcherScore``); a ``ComparatorScore`` carries ``comparator``, not
        ``matcher``, so it is skipped.
        """
        matcher: Matcher[Any] | None = None
        for op in self._chain_body():
            candidate = getattr(op, "matcher", None)
            if isinstance(op, Score) and isinstance(candidate, Matcher):
                matcher = candidate
        if isinstance(matcher, SpendCappedMatcher):
            return matcher._module  # peel the cap; the wrapped matcher owns ``model``
        return matcher

    def _chain_source_schema(self) -> type[BaseModel] | None:
        """The schema the chain's Source blocks against (for front-door normalization),
        or ``None`` when the Source carries none (an opaque ``schema_factory`` blocker)."""
        blocker = getattr(self._chain_source(), "blocker", None)
        if isinstance(blocker, Blocker):
            return blocker.schema
        return None

    def _require_bound(self, action: str) -> None:
        """Raise a directed error if this model has no components yet."""
        if not self.is_bound:
            raise RuntimeError(
                f"cannot {action}: {type(self).__name__} has not been bound to a schema yet, "
                "so it has no components. Pass schema=<YourModel> to the constructor, or "
                "call dedupe()/compare() once (which infers a schema from the records and "
                "binds). An inferred schema is ephemeral and does not survive save/load in a "
                "fresh process -- pass schema= explicitly for anything you intend to persist."
            )

    def _bind(self, schema: type[BaseModel]) -> None:
        """Build and wire this architecture's topology for ``schema``, if not already bound."""
        if not self.is_bound:
            self._wire(**self._topology(schema))

    def _prepare(self, records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize ``records`` and bind this model, returning the ready records.

        The ONE front-door adapter (ported from the deleted verbs): a model bound
        to an explicit schema normalizes against it; an unbound one infers a
        schema from the records' own keys and binds to it here, on first use.
        """
        schema, normalized = normalize_records(records, self.schema)
        self._bind(schema)
        return normalized
