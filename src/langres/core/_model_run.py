"""How an ER model runs: block -> (compare) -> score -> cluster, and the front door.

The scoring half of the ``ERModel`` class chain, layered on
:mod:`langres.core._model_state` (which owns the slots this reads) and inherited
by the leaf in :mod:`langres.core.resolver`.

Two audiences, one pipeline:

- **The plumbing** -- :meth:`ModelRun.candidates` (block + attach comparisons),
  :meth:`ModelRun._scorer` (the ONE metered scoring seam),
  :meth:`ModelRun.predict` / :meth:`ModelRun.resolve`.
- **The front door** -- :meth:`ModelRun.dedupe` (a batch -> clusters) and
  :meth:`ModelRun.compare` (one pair -> a verdict), the replacements for the
  deleted ``langres.dedupe`` / ``langres.link`` verbs.

**The block -> (compare) -> score -> cluster pipeline runs through the Op
adapters** (:mod:`langres.core.op_adapters`), derived from the four slots
(W3-a, epic #193): :class:`~langres.core.op_adapters.BlockerSource` +
:class:`~langres.core.op_adapters.ComparatorScore` are the block/compare half
(:meth:`ModelRun._candidates`); :class:`~langres.core.op_adapters.MatcherScore`
wraps the spend-capped, optionally-logged matcher (:meth:`ModelRun._matcher_score`
over :meth:`ModelRun._scorer`) into the scoring half (:meth:`ModelRun._scored_pairs`);
and the exit (:meth:`ModelRun._cluster`, shared by :meth:`ModelRun.resolve` /
:meth:`ModelRun.dedupe`) is the two selections ``docs/THEORY.md`` separates -- an
explicit match cut (:class:`~langres.core.op.ThresholdSelect`, the selection π at
feasible-class THRESHOLD) THEN
:class:`~langres.core.op_adapters.ClustererStage` over a threshold-zeroed clone of
the clusterer (the equivalence π: pure transitive closure / pivot, no internal
cut). The one per-call factory
:meth:`ModelRun._stages` assembles the block -> (compare) -> score stages from the
slots (W3-c), and the generic driver :func:`_run_stages` folds ``forward`` across
them. The slots stay the source of truth -- the stages are assembled from them
per call, never cached, so the topology is an explicit chain of ``forward`` calls
over the one ``Pairs`` carrier.

**What "same behavior" means, precisely.** The *resolution* outputs are
byte-identical to the legacy direct-call spine -- clusters, ``DedupeResult``
metadata (``score_type``/``threshold``), metrics, and the judgement log all
match the frozen goldens (``tests/parity/test_behavior_parity_w0.py``). The one
deliberate change is the *order* of :meth:`ModelRun.predict` /
:meth:`ModelRun._judgements`: they now emit in deterministic carrier (blocker)
order, because :class:`~langres.core.op_adapters.MatcherScore` maps each
judgement back onto its row by ``(left_id, right_id)`` identity. For a pairwise
matcher that equals the legacy matcher-emission order (one judgement per
candidate, in candidate order), so it is invisible. For a *reordering* matcher --
a :class:`~langres.core.matcher.GroupwiseMatcher`, whose ``forward`` regroups the
stream by anchor -- the legacy spine returned that anchor-group emission order;
the new spine returns the blocker's carrier order. That intentional, more
deterministic contract is pinned by ``tests/parity/test_groupwise_spine_w3a.py``.

Everything that scores goes through :meth:`ModelRun._scorer`, so the spend cap
cannot be bypassed; ``tests/core/test_resolver_spend_cap.py`` AST-bans
``<any>.module.forward(...)`` in ``src/`` to keep it that way. The
:class:`~langres.core.op_adapters.MatcherScore` adapter wraps the *capped*
matcher (not the raw one), so draining its rescored rows still trips the cap
mid-pull.
"""

import hashlib
import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from langres.core._model_state import ModelState
from langres.core.blockers.composite import CompositeBlocker
from langres.core.clusterer import Clusterer
from langres.core.fit import CalibratorFitMixin
from langres.core.inputs import check_no_duplicate_ids, normalize_records
from langres.core.model_ref import ModelRef, to_config
from langres.core.models import (
    ERCandidate,
    MatcherAbstainedError,
    PairwiseJudgement,
    predicted_match,
)
from langres.core.op import (
    ClusterStage,
    ExecutionEvent,
    ExecutionObserver,
    ExecutionObserverError,
    ExecutionPlan,
    ExecutionResult,
    ExecutionStep,
    Op,
    Score,
    Select,
    Source,
    Stage,
    ThresholdSelect,
    TopKSelect,
)
from langres.core.op_adapters import (
    BlockerSource,
    CalibratorScore,
    ClustererStage,
    ComparatorScore,
    MatcherScore,
)
from langres.core.pairs import Pairs
from langres.core.results import DedupeResult, LinkVerdict
from langres.core.serialization import OpSpec
from langres.core.spend_cap import SpendCappedMatcher
from langres.tracking.judgement_log import JudgementLog, LoggingMatcher

if TYPE_CHECKING:
    # [semantic] extra (faiss/sentence-transformers/torch) -- type-only so a
    # core-only `import langres` never pulls the optional stack.
    from langres.core.blockers.vector import VectorBlocker


def iter_vector_blockers(blocker: object) -> "Iterator[VectorBlocker[Any]]":
    """Yield every ``VectorBlocker`` reachable from ``blocker``.

    A plain ``VectorBlocker`` yields itself. A ``CompositeBlocker`` (the
    blocking-algebra union/intersection/difference of child blockers) recurses
    into ``children`` at arbitrary depth, so a composite wrapping another
    composite still surfaces every nested ``VectorBlocker`` -- e.g. the
    recall-first pattern ``CompositeBlocker([KeyBlocker(...), VectorBlocker(...)],
    op="union")``. Index-free blockers (``AllPairsBlocker``, ``KeyBlocker``,
    ``GLinkerAdapter``) contribute nothing.

    Checks ``type_name`` rather than ``isinstance(blocker, VectorBlocker)``
    deliberately: callers outside the execution spine still use this helper,
    and importing ``VectorBlocker`` here would pull faiss/sentence-transformers
    (the ``[semantic]`` extra) into a plain
    ``AllPairsBlocker``/``KeyBlocker`` pipeline. ``CompositeBlocker`` itself has
    no heavy dependency, so it is safe to ``isinstance``-check directly.
    """
    if getattr(blocker, "type_name", None) == "vector_blocker":
        yield cast("VectorBlocker[Any]", blocker)
    elif isinstance(blocker, CompositeBlocker):
        for child in blocker.children:
            yield from iter_vector_blockers(child)


def _coerce_log(log: "JudgementLog | str | Path | None") -> JudgementLog | None:
    """Normalize a ``log=`` argument to a :class:`JudgementLog`, or ``None``.

    ``None`` stays ``None`` -- zero overhead, no wrap. A path is wrapped in a
    fresh, default (``features=False``) log; an existing ``JudgementLog`` (e.g.
    one built with ``features=True``) passes through unchanged.
    """
    if log is None or isinstance(log, JudgementLog):
        return log
    return JudgementLog(log)


def _run_stages(records: list[Any], stages: Sequence[Source[Any] | Op[Any]]) -> Pairs[Any]:
    """Fold ``forward`` across ``stages`` to turn ``records`` into a ``Pairs``.

    The generic driver behind :meth:`ModelRun._candidates`,
    :meth:`ModelRun._scored_pairs` (both derived-slot and explicit-chain). Every
    stage list it is handed leads with the :class:`~langres.core.op.Source`
    (records -> pairs) and continues with :class:`~langres.core.op.Op`\\ s
    (pairs -> pairs), so the fold walks ``Records -> Pairs -> ... -> Pairs``.

    The body element type is :class:`~langres.core.op.Op`, **not** ``Score``: an
    explicit chain (:meth:`~langres.core._model_state.ModelState.from_topology`)
    interleaves :class:`~langres.core.op.Select`\\ s among the Scores, and a
    ``Select`` is an ``Op`` but not a ``Score``. Narrowing the loop element to
    ``Score`` would exclude them; ``Op`` admits every body stage while keeping the
    fold precisely typed. The parameter is a covariant ``Sequence`` so the classic
    factories' ``list[Source | Score]`` still passes (``Score`` is an ``Op``).

    The two narrowing ``cast``\\ s encode the Source-first shape the factories
    guarantee -- ``Source.forward`` takes records while ``Op.forward`` takes a
    ``Pairs``, so the heterogeneous first stage cannot share one loosely-typed loop
    variable with the rest, and the cast is a typed assertion of the shape, not an
    ``Any``.
    """
    source, *body = stages
    source_stage = cast(Source[Any], source)
    source_stage.prepare(records)
    pairs = source_stage.forward(records)
    for op in body:
        pairs = cast(Op[Any], op).forward(pairs)
    return pairs


class ModelRun(ModelState):
    """The run path of an ``ERModel``: candidates, scoring, clustering, front door."""

    def _candidates(self, records: list[Any]) -> Pairs[Any]:
        """Block records into a :class:`~langres.core.pairs.Pairs`, comparing if configured.

        Builds an index-backed blocker's index transparently before streaming,
        so callers never call ``create_index`` themselves. Records are fed in
        the caller's stable list order. Shared by ``_judgements`` (scoring)
        and ``fit`` (training) -- both bridge back to ``ERCandidate`` at the
        matcher boundary via :meth:`~langres.core.pairs.Pairs.to_candidates`
        (matchers still consume the legacy stream; migrating them is W2).

        The blocker's per-pair ``similarity_score`` lands on each row as an
        *unscored* ``score`` (``score_type is None`` -- a blocker similarity,
        never a judge score, ``F-W1a``); the comparator's ``ComparisonVector``,
        when configured, is attached to each row's ``comparison`` before the
        carrier is built, so ``.left``/``.right`` stay bound to the shared
        store. Materializing the post-blocking set here (rather than streaming
        it lazily as this method used to) is the W1 carrier trade-off: rows are
        lightweight id-refs over one shared entity store, and behavior --
        clusters, scores -- is byte-identical.

        The block/compare half is the scoring pipeline's matcher-free prefix
        (W3-c, epic #193): it folds the driver (:func:`_run_stages`) over
        :meth:`_block_compare_stages` -- the
        :class:`~langres.core.op_adapters.BlockerSource` (bridging
        ``blocker.stream(records)`` into the ``Pairs``) plus, when a comparator is
        configured, the :class:`~langres.core.op_adapters.ComparatorScore` that
        attaches each row's ``ComparisonVector``. It depends on the
        blocker/comparator slots ALONE -- no matcher, no clusterer -- so
        blocking-only callers (``candidates`` on the hot fit/eval paths, and a
        future retrieval-only topology the four-slot core forbids) neither build
        nor require a scoring slot. ``_run_stages`` calls the Source's
        ``prepare`` lifecycle before streaming, so vector indexes build or reuse
        themselves consistently on classic and explicit paths.
        """
        return _run_stages(records, self._block_compare_stages())

    def candidates(self, records: list[Any]) -> list[ERCandidate[Any]]:
        """Block records into a materialized list of judge-ready candidates.

        The public counterpart to :meth:`_candidates`: same blocking (building
        any index-backed blocker's index transparently) and the same
        comparison-attachment behavior, but returns the legacy
        **``list[ERCandidate]``** (bridged off the ``Pairs`` carrier
        ``_candidates`` now returns) rather than the carrier itself -- external
        callers depend on this exact shape. Comparison vectors ARE attached
        whenever this Resolver has a
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
        return self._candidates(records).to_candidates()

    def _scorer(self, *, log: JudgementLog | None = None) -> SpendCappedMatcher:
        """This model's matcher, metered by its ONE spend ledger.

        **The only supported way to score through ``self.module``.** Reaching for
        the raw ``self.module.forward(...)`` gets you an *uncapped* scorer that
        bills straight past ``budget_usd`` and never touches this instance's
        ledger. That is not hypothetical: ``.module`` is a public attribute, and
        :class:`~langres.curation.anchor_store.AnchorStore` reached through it and
        scored uncapped -- inside the very change that added the cap. Route every
        new scoring path here and that hole cannot reopen;
        ``tests/core/test_resolver_spend_cap.py`` pins it with an AST sweep.

        Two properties this method exists to hold together:

        * **The monitor is per-instance, built once in ``__init__``** -- so
          ``resolve``, ``predict``, ``fit`` and an ``AnchorStore`` pass all draw
          down ONE cumulative budget instead of each getting a fresh one.
        * **The wrapper is rebuilt per call, deliberately.** Caching it in
          ``__init__`` would pin the matcher as it was *then*, and ``self.module``
          is reassignable: ``dedupe`` wraps it in a ``LoggingMatcher`` after
          construction and ``fit(method=Finetune())`` replaces it outright. A
          cached wrapper would silently meter -- and score through -- the stale
          matcher, emptying the judgement log. Rebuilding costs an object; the
          ledger it shares is what actually persists.

        The cap deliberately does NOT live in the ``module`` slot: ``save()``'s
        registry and ``fit()``'s isinstance checks both read ``self.module`` and
        must see the real component, not a wrapper.

        ``log`` composes INSIDE the cap, deliberately: the cap must meter every
        judgement the log records, so it stays outermost. This is the same
        nesting the deleted ``dedupe`` verb produced by reassigning
        ``resolver.module = LoggingMatcher(...)`` before scoring -- but per call
        and without permanently mutating the slot, so ``save()`` and ``fit()``'s
        isinstance checks still see the real matcher.

        Args:
            log: Optional per-call judgement sink (see :meth:`dedupe`).

        Returns:
            A :class:`~langres.core.spend_cap.SpendCappedMatcher` around the
            *current* ``self.module``, sharing this instance's ledger.
        """
        module = self.module
        if log is not None:
            module = LoggingMatcher(
                module, log=log, threshold=self.clusterer.threshold, model=self.backbone
            )
        return SpendCappedMatcher(module, monitor=self._spend_monitor)

    def _matcher_score(self, *, log: JudgementLog | None = None) -> MatcherScore[Any]:
        """This model's scoring :class:`~langres.core.op_adapters.MatcherScore` Op.

        Wraps :meth:`_scorer` -- the SAME ``SpendCappedMatcher(LoggingMatcher)``
        the legacy spine scored through -- so the cap stays lazy: the adapter
        drains the capped generator to rescore rows, and the cap trips mid-pull
        (budget + at most one call) exactly as before. Deliberately wraps the
        *capped* matcher, never the raw ``self.module``.

        ``out_space="unknown"`` is the honest sentinel (W3-b): a ``Matcher`` has no
        class-level ``score_type`` constant -- each judgement is stamped its own
        family per-row in :func:`~langres.core.op_adapters._rescore` -- so the
        family this generic ``self.module`` produces is genuinely not knowable at
        build time. ``"unknown"`` says exactly that (an orderable scalar of an
        unpinned family), rather than borrowing a real family like ``"heuristic"``
        as a stand-in. It is also **inert in the spine**: ``_rescore`` stamps each
        row with its judgement's OWN ``score_type`` (never this declared family),
        and nothing in the run path reads a ``Score.out_space``, so the declared
        value cannot change behavior. The authoritative per-row family is the
        matcher's own (also what :meth:`dedupe`'s ``DedupeResult.score_type``
        reports). If a later wave makes ``out_space`` load-bearing (e.g. a wiring
        check in the spine), derive it from the matcher's real family here.
        """
        return MatcherScore(self._scorer(log=log), out_space="unknown")

    def _block_compare_stages(self) -> list[Source[Any] | Score[Any]]:
        """The matcher-free block/compare prefix, from the blocker/comparator slots.

        A :class:`~langres.core.op_adapters.BlockerSource` (records -> pairs),
        then -- only when a comparator is configured -- a
        :class:`~langres.core.op_adapters.ComparatorScore` that attaches each
        row's ``ComparisonVector``. Shared by :meth:`_candidates` (which folds
        exactly this) and :meth:`_stages` (which appends the
        :class:`~langres.core.op_adapters.MatcherScore` to it), so the
        source/compare adapters are constructed in ONE place.

        Deliberately depends on ``self.blocker``/``self.comparator`` only -- it
        never reads the matcher or clusterer slot -- so the blocking half stays
        expressible without a scoring slot (a retrieval-only topology the
        four-slot core forbids today, epic #193's direction). Rebuilt every call:
        the adapters hold slot references, so caching would pin a stale
        blocker/comparator across a rebind.

        Returns:
            The source-first block/compare stages (one or two of them).
        """
        stages: list[Source[Any] | Score[Any]] = [BlockerSource(self.blocker)]
        if self.comparator is not None:
            stages.append(ComparatorScore(self.comparator))
        return stages

    def _stages(self, *, log: JudgementLog | None = None) -> list[Source[Any] | Score[Any]]:
        """The scoring pipeline as one ordered stage list, built from the slots.

        The single per-call factory for the full block -> (compare) -> score Op
        chain (W3-c, epic #193): :meth:`_block_compare_stages` (a
        :class:`~langres.core.op_adapters.BlockerSource`, then optionally a
        :class:`~langres.core.op_adapters.ComparatorScore`) with the
        :class:`~langres.core.op_adapters.MatcherScore` (:meth:`_matcher_score`)
        appended. :func:`_run_stages` folds ``forward`` across the returned list
        to turn ``records`` into scored ``Pairs``; :meth:`_candidates` folds the
        matcher-free prefix (:meth:`_block_compare_stages`) alone.

        **Rebuilt every call, never cached** -- exactly like :meth:`_scorer`,
        which it wraps. ``self.module`` is reassignable (``fit`` repoints it) and
        the ``log`` differs per call, so a cached stage list would pin a stale
        matcher and empty the judgement log. The stages hold id-refs and slot
        references only, so rebuilding is a handful of cheap allocations.

        The return type is a ``Source``/``Score`` union rather than one base: the
        first stage is a :class:`~langres.core.op.Source` (records -> pairs) and
        the rest are :class:`~langres.core.op.Score`\\ s (pairs -> pairs), which
        share no tighter common base than that (a ``Source`` is not an ``Op``).

        Args:
            log: Optional per-call judgement sink, threaded into the
                :class:`~langres.core.op_adapters.MatcherScore` (see :meth:`dedupe`).

        Returns:
            The ordered scoring stages, source-first.
        """
        stages = self._block_compare_stages()
        stages.append(self._matcher_score(log=log))
        if self.calibrator is not None:
            stages.append(CalibratorScore(self.calibrator))
        return stages

    def _explicit_body(self, *, log: JudgementLog | None = None) -> list[Op[Any]]:
        """Build the explicit body for one call, adding logging without mutation.

        Explicit Ops are durable topology objects. A per-call ``JudgementLog``
        therefore wraps each base ``MatcherScore`` in a fresh
        ``SpendCappedMatcher(LoggingMatcher(raw))`` sharing the model ledger,
        rather than replacing the stored stage. Every matcher Score is logged;
        component-free custom Scores continue unchanged.
        """
        body = self._chain_body()
        if log is None:
            return body

        wrapped: list[Op[Any]] = []
        for body_index, op in enumerate(body):
            if not isinstance(op, MatcherScore):
                wrapped.append(op)
                continue
            matcher = op.matcher
            raw = matcher._module if isinstance(matcher, SpendCappedMatcher) else matcher
            logged = LoggingMatcher(
                raw,
                log=log,
                threshold=self._direct_log_threshold(body, body_index),
                model=self._stage_resource_ref(op),
                # The Source is index 0, so body indices begin at one.
                stage_id=f"{body_index + 1:02d}-matcher_score",
            )
            wrapped.append(
                op.with_matcher(
                    SpendCappedMatcher(logged, monitor=self._spend_monitor),
                )
            )
        return wrapped

    @staticmethod
    def _direct_log_threshold(body: Sequence[Op[Any]], score_index: int) -> float | None:
        """Return the binary cut applied directly to one score stage.

        A later Score transforms or replaces the numbers, and a non-threshold
        Select (top-k/link/assignment) is not a binary match verdict. In either
        case the current stage logs ``verdict=None`` instead of borrowing the
        pipeline's final threshold.
        """
        for downstream in body[score_index + 1 :]:
            if isinstance(downstream, Score):
                return None
            if isinstance(downstream, Select):
                return downstream.threshold if isinstance(downstream, ThresholdSelect) else None
        return None

    def _scored_pairs(self, records: list[Any], *, log: JudgementLog | None = None) -> Pairs[Any]:
        """Block -> (compare) -> score as an Op chain, then calibrate: the scoring spine.

        Folds the driver (:func:`_run_stages`) over the full :meth:`_stages`
        pipeline -- :class:`~langres.core.op_adapters.BlockerSource`, then
        (optionally) :class:`~langres.core.op_adapters.ComparatorScore`, then the
        :class:`~langres.core.op_adapters.MatcherScore` (:meth:`_matcher_score`) --
        so the topology is one explicit chain of ``forward`` calls over the single
        ``Pairs`` carrier, built from one factory rather than scattered inline.
        ``_run_stages`` calls the Source's ``prepare`` lifecycle before
        ``forward`` so classic and explicit vector retrieval build/reuse indexes
        identically. Scoring runs through :meth:`_scorer`, so this model's
        spend cap (``budget_usd=``) and its ledger are shared across every
        :meth:`resolve`/:meth:`predict`/:meth:`dedupe` call on this instance -- two
        successive resolves cannot each spend a full budget, and a matcher that
        overruns the cap raises ``BudgetExceeded`` from the ``MatcherScore`` drain
        here.

        When :attr:`calibrator` is set (by ``fit(method=Platt()/Isotonic())``),
        every scored ranking row's raw ``score`` is mapped to a calibrated
        probability via ``CalibratorScore`` before clustering, so the clusterer
        thresholds on a real probability. Pure pass-through otherwise.

        **Explicit chain (``_ops`` set).** Folds the chain's Source + body Ops
        (Scores AND Selects, up to but excluding the terminal ClusterStage --
        :meth:`_cluster` runs that). Its Source is prepared before streaming, and
        per-call logging wraps each base ``MatcherScore`` without mutating the
        stored Ops. The legacy ``calibrator=`` slot is not accepted for explicit
        topology; score mapping belongs in the declared body as a Score.
        """
        if self._ops is not None:
            return _run_stages(records, [self._chain_source(), *self._explicit_body(log=log)])
        return _run_stages(records, self._stages(log=log))

    def _judgements(
        self, records: list[Any], *, log: JudgementLog | None = None
    ) -> Iterator[PairwiseJudgement]:
        """Block records into candidates, score them, and calibrate if fitted.

        Projects the scored :class:`~langres.core.pairs.Pairs`
        (:meth:`_scored_pairs`) back to the legacy judgement stream: a matcher
        yields exactly one judgement per candidate, so every scored row
        (``score_type is not None``) becomes one ``PairwiseJudgement`` in the
        carrier's row order (the blocker's stream order). A generator body keeps
        the projection lazy -- ``_scored_pairs`` runs (and any ``BudgetExceeded``
        raises) only when the stream is first drained, exactly as the legacy
        ``_scorer(...).forward`` stream did.
        """
        for row in self._scored_pairs(records, log=log).rows:
            if row.score_type is not None:
                yield row.to_judgement()

    def _apply_calibrator_to_pairs(
        self, pairs: Pairs[Any], calibrator: CalibratorFitMixin
    ) -> Pairs[Any]:
        """Map each scored ranking row's ``score`` through the fitted calibrator.

        The carrier counterpart to the legacy judgement-stream calibration.
        Deciders (a scored row with ``score is None``) pass through untouched --
        there is no score to calibrate -- as do unscored rows (``score_type is
        None``), whose ``score`` is a *blocker* similarity, never a judge score
        (``F-W1a``), and so must not be run through a score calibrator. A mapped
        row keeps its ids/decision, retags ``score_type="calibrated_prob"``, and
        records the raw score under ``provenance["calibration"]`` for
        auditability -- byte-identical to what the old ``_apply_calibrator``
        produced on the judgement stream (which only ever saw scored rows).
        """
        return CalibratorScore(calibrator).forward(pairs)

    def predict(self, records: list[Any]) -> list[PairwiseJudgement]:
        """Return the scored pairwise judgements before clustering.

        Useful for observability/tuning: inspect scores and provenance without
        committing to a clustering threshold.
        """
        return list(self._judgements(records))

    def _execution_stages(self) -> list[Stage]:
        """Return the full source-to-clusters topology for public execution."""
        if self._ops is not None:
            return [
                self._chain_source(),
                *self._explicit_body(),
                self._chain_cluster_stage(),
            ]
        return [
            *self._stages(),
            ThresholdSelect(self.clusterer.threshold),
            ClustererStage(self._closure_clusterer()),
        ]

    @staticmethod
    def _execution_spec(stage: Stage) -> OpSpec:
        """Build safe runtime metadata without requiring artifact serialization.

        Execution and inspection accept runnable custom stages and opaque schema
        factories. Only ``save()`` requires an explicitly registered serializer,
        so this deliberately does not call ``op_spec(stage)``.
        """
        if isinstance(stage, BlockerSource):
            role = "blocker_source"
        elif isinstance(stage, ComparatorScore):
            role = "comparator_score"
        elif isinstance(stage, MatcherScore):
            role = "matcher_score"
        elif isinstance(stage, CalibratorScore):
            role = "calibrator_score"
        elif isinstance(stage, ThresholdSelect):
            role = "threshold_select"
        elif isinstance(stage, TopKSelect):
            role = "topk_select"
        elif isinstance(stage, ClustererStage):
            role = "clusterer_stage"
        elif isinstance(stage, Source):
            role = "source"
        elif isinstance(stage, Score):
            role = "score"
        elif isinstance(stage, Select):
            role = "select"
        elif isinstance(stage, ClusterStage):
            role = "cluster_stage"
        else:
            role = "finalize"

        params: dict[str, object] = {"stage_type": type(stage).__name__}
        if isinstance(stage, Score):
            params.update(scope=stage.scope, out_space=stage.out_space)
        if isinstance(stage, Select):
            params.update(
                feasible=stage.feasible.name,
                algorithm=stage.algorithm,
            )
            if isinstance(stage, ThresholdSelect):
                params["threshold"] = stage.threshold
            elif isinstance(stage, TopKSelect):
                params["k"] = stage.k
        if isinstance(stage, ClusterStage):
            params["algorithm"] = stage.algorithm
        return OpSpec(role=role, params=params)

    @classmethod
    def _execution_step(cls, stage: Stage, index: int) -> ExecutionStep:
        """Build a stable step id from runtime metadata plus its ordinal."""
        spec = cls._execution_spec(stage)
        canonical = json.dumps(
            spec.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
        return ExecutionStep(
            stage_id=f"{index:02d}-{spec.role}-{digest}",
            index=index,
            spec=spec,
            resource_ref=cls._stage_resource_ref(stage),
        )

    @staticmethod
    def _stage_resource_ref(stage: object) -> str | None:
        """Return a stable model identity by structural inspection.

        The core does not import resource implementations. It recognizes the
        shared ``ModelRef`` leaf, legacy ``model``/``model_name`` attributes, and
        common composition attributes (resource, matcher, vector index,
        embedder). Unknown weightless stages report ``None``.
        """

        def visit(value: object, seen: set[int]) -> str | None:
            identity = id(value)
            if identity in seen:
                return None
            seen.add(identity)

            if isinstance(value, SpendCappedMatcher):
                return visit(value._module, seen)

            model_ref = getattr(value, "model_ref", None)
            if isinstance(model_ref, ModelRef):
                payload = to_config(model_ref)
                if isinstance(payload, str):
                    return payload
                return json.dumps(payload, sort_keys=True, separators=(",", ":"))

            for name in ("model", "model_name"):
                candidate = getattr(value, name, None)
                if isinstance(candidate, str) and candidate:
                    return candidate

            for name in (
                "resource",
                "matcher",
                "_module",
                "blocker",
                "vector_index",
                "embedder",
                "embedding_service",
            ):
                candidate = getattr(value, name, None)
                if candidate is not None:
                    found = visit(candidate, seen)
                    if found is not None:
                        return found
            return None

        return visit(stage, set())

    def execution_plan(self) -> ExecutionPlan:
        """Describe this model's runnable topology without reading legacy slots.

        Stage ids are content-derived from safe runtime metadata plus their
        ordinal, never from object identity or ``repr``.
        """
        if not self.is_bound:
            return ExecutionPlan(schema_name=None, is_bound=False, steps=())
        stages = self._execution_stages()
        schema = self.schema
        return ExecutionPlan(
            schema_name=schema.__name__ if schema is not None else None,
            is_bound=self.is_bound,
            steps=tuple(self._execution_step(stage, index) for index, stage in enumerate(stages)),
        )

    @staticmethod
    def _emit_execution_event(
        event: ExecutionEvent,
        events: list[ExecutionEvent],
        observer: ExecutionObserver | None,
        observer_errors: list[ExecutionObserverError],
    ) -> None:
        """Record/publish metadata without letting observers alter inference."""
        events.append(event)
        if observer is None:
            return
        try:
            observer(event)
        except Exception as exc:
            exception_type = re.sub(r"[^A-Za-z0-9_.-]", "_", type(exc).__name__)[:80]
            observer_errors.append(
                ExecutionObserverError(
                    event=event,
                    exception_type=exception_type or "Exception",
                    message="observer callback raised; exception details suppressed",
                )
            )

    def execute(
        self,
        records: list[dict[str, Any]],
        *,
        observer: ExecutionObserver | None = None,
    ) -> ExecutionResult:
        """Run the existing Op spine and return its slot-neutral intermediates.

        The observer receives immutable metadata only; its return value is
        ignored and callback exceptions are isolated in
        :attr:`ExecutionResult.observer_errors`, so observability cannot replace
        a carrier, abort a stage, or otherwise alter inference.
        """
        if self._ops is not None:
            _schema, normalized = normalize_records(records, self._chain_source_schema())
        else:
            normalized = self._prepare(records)
        check_no_duplicate_ids([record["id"] for record in normalized])

        stages = self._execution_stages()
        plan = ExecutionPlan(
            schema_name=self.schema.__name__ if self.schema is not None else None,
            is_bound=self.is_bound,
            steps=tuple(self._execution_step(stage, index) for index, stage in enumerate(stages)),
        )
        events: list[ExecutionEvent] = []
        observer_errors: list[ExecutionObserverError] = []
        pairs: Pairs[Any] | None = None
        clusters: list[set[str]] = []

        for step, stage in zip(plan.steps, stages, strict=True):
            input_count = len(normalized) if pairs is None else len(pairs.rows)
            start = ExecutionEvent(
                kind="start",
                stage_id=step.stage_id,
                index=step.index,
                role=step.spec.role,
                input_count=input_count,
            )
            self._emit_execution_event(start, events, observer, observer_errors)
            started = perf_counter()
            try:
                if isinstance(stage, Source):
                    stage.prepare(normalized)
                    pairs = stage.forward(normalized)
                    output_count = len(pairs.rows)
                elif isinstance(stage, Op):
                    assert pairs is not None
                    pairs = stage.forward(pairs)
                    output_count = len(pairs.rows)
                elif isinstance(stage, ClusterStage):
                    assert pairs is not None
                    clusters = stage.forward(pairs)
                    output_count = len(clusters)
                else:  # pragma: no cover - from_topology rejects Finalize
                    raise TypeError(f"execute() cannot run {type(stage).__name__}")
            except Exception as exc:
                failure = ExecutionEvent(
                    kind="failure",
                    stage_id=step.stage_id,
                    index=step.index,
                    role=step.spec.role,
                    input_count=input_count,
                    duration_seconds=perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self._emit_execution_event(failure, events, observer, observer_errors)
                raise
            finish = ExecutionEvent(
                kind="finish",
                stage_id=step.stage_id,
                index=step.index,
                role=step.spec.role,
                input_count=input_count,
                output_count=output_count,
                duration_seconds=perf_counter() - started,
            )
            self._emit_execution_event(finish, events, observer, observer_errors)

        assert pairs is not None
        return ExecutionResult(
            plan=plan,
            pairs=pairs,
            clusters=tuple(frozenset(cluster) for cluster in clusters),
            events=tuple(events),
            observer_errors=tuple(observer_errors),
        )

    def _closure_clusterer(self) -> Clusterer:
        """A pure-equivalence clone of this model's clusterer, its threshold zeroed.

        Clones the clusterer's OWN class through ``config``/``from_config`` -- NOT
        the base :class:`~langres.core.clusterer.Clusterer` -- so a
        :class:`~langres.core.clusterers.correlation.CorrelationClusterer` stays a
        *pivot* clusterer rather than being silently downgraded to transitive
        closure (which would merge pivot-split chains that must stay separate).
        The threshold is zeroed because the match cut already ran in
        :meth:`_cluster` (a :class:`~langres.core.op.ThresholdSelect`); this clone
        does pure equivalence over the survivors and must not re-threshold.

        It is a FRESH instance -- ``self.clusterer`` keeps its real threshold: the
        ThresholdSelect reads it, and ``dedupe``'s ``DedupeResult`` reports it.
        """
        return type(self.clusterer).from_config({**self.clusterer.config, "threshold": 0.0})

    def _cluster(self, scored_pairs: Pairs[Any]) -> list[set[str]]:
        """The pipeline exit: match cut (Select at THRESHOLD) THEN pure clustering.

        The two selections ``docs/THEORY.md`` separates, made explicit (epic #193,
        W3-d), shared by :meth:`resolve` and :meth:`dedupe`. The **match cut** is a
        :class:`~langres.core.op.ThresholdSelect` -- a ``Select`` at feasible-class
        THRESHOLD (the theory's selection π) keeping exactly the rows whose score
        clears ``self.clusterer.threshold`` (via
        :func:`~langres.core.models.predicted_match`: a decider's decision wins, an
        abstention is dropped). The **clustering** is then a separate equivalence π
        -- transitive closure, or pivot for a ``CorrelationClusterer`` -- run by a
        clusterer with NO threshold of its own (:meth:`_closure_clusterer`).

        Byte-identical to the legacy ``ClustererStage(self.clusterer)`` that kept
        the cut folded inside the clusterer's threshold: ``predicted_match(j, t) is
        True`` implies ``predicted_match(j, 0.0) is True`` (a decision is
        threshold-independent; else ``score >= t`` implies ``score >= 0``), so the
        zeroed clone's own filter is redundant given the ThresholdSelect and the
        surviving edge set is unchanged -- for transitive closure AND for pivot.
        ``scored_pairs`` must already be calibrated (callers pass
        :meth:`_scored_pairs`'s output), so the cut thresholds the CALIBRATED
        score, exactly as the clusterer did.

        **Explicit chain (``_ops`` set).** The chain's own terminal ClusterStage
        IS the exit -- its match cut is already a body ``ThresholdSelect`` and its
        ``ClustererStage`` already carries the (zeroed) clusterer -- so this runs
        that ClusterStage on ``scored_pairs`` directly, with **no** extra
        ThresholdSelect / :meth:`_closure_clusterer` wrap. Adding either would
        double-cut and silently diverge.
        """
        if self._ops is not None:
            return self._chain_cluster_stage().forward(scored_pairs)
        selected: Pairs[Any] = ThresholdSelect(self.clusterer.threshold).forward(scored_pairs)
        return ClustererStage(self._closure_clusterer()).forward(selected)

    def resolve(self, records: list[Any]) -> list[set[str]]:
        """Resolve records into entity clusters (sets of IDs).

        Orchestrates blocking -> (compare) -> score -> cluster through the Op
        adapters (:meth:`_scored_pairs`, then :meth:`_cluster` -- the explicit
        match cut, a :class:`~langres.core.op.ThresholdSelect` at the clusterer's
        threshold, followed by pure-equivalence
        :class:`~langres.core.op_adapters.ClustererStage` clustering over the
        survivors, per ``docs/THEORY.md``'s Select-π vs equivalence-π split).
        The output is whatever the (zeroed) equivalence clusterer returns over
        the selected edges: the base connected-components
        :class:`~langres.core.clusterer.Clusterer` emits only multi-record
        clusters (an isolated record never enters the graph), whereas a pivot
        :class:`~langres.core.clusterers.correlation.CorrelationClusterer` can
        leave a singleton behind -- a record with a qualifying edge whose only
        neighbours an earlier pivot already claimed.

        Args:
            records: Raw records (dicts) in a stable list order.

        Returns:
            A list of clusters, each a set of entity IDs.
        """
        return self._cluster(self._scored_pairs(records))

    # ------------------------------------------------------------------
    # The front door: dedupe a batch / compare one pair
    # ------------------------------------------------------------------

    def dedupe(
        self, records: list[dict[str, Any]], *, log: JudgementLog | str | Path | None = None
    ) -> DedupeResult:
        """Group a batch of records into entity clusters. **The front door.**

        The replacement for the deleted ``langres.dedupe`` verb, and the whole
        point of the wave: identical ergonomics, except *you* named the model, so
        nothing sniffs your environment for an API key and spends on what it
        finds. ``FuzzyString().dedupe(records)`` costs $0 and needs no key
        because ``FuzzyString`` cannot make a paid call, not because a heuristic
        guessed well.

        Schema-optional: an unbound architecture infers an ephemeral schema from
        the records' own keys and binds to it here, on first use (see
        :mod:`langres.core.inputs` for the normalization rules -- NaN, nested
        values, id resolution).

        Abstentions are left **unmerged** (the conservative reading -- the same
        as "not a match" for edge-building) rather than aborting the batch: one
        unparseable judgement among thousands should not sink a whole run. Pass
        ``log=`` to see which pairs the matcher declined. This differs from
        :meth:`compare`, which owes its single caller a verdict and so raises.

        Args:
            records: The records to dedupe (plain dicts). ``[]`` and a single
                record both return ``[]`` (no pair exists), short-circuiting
                before the model is ever bound or a matcher built -- so a
                zero-pair call cannot construct a backbone or spend.
                Every record needs a unique ``"id"`` (or none at all --
                positional ids are then assigned); a duplicate raises.
            log: Opt-in judgement sink -- a
                :class:`~langres.tracking.judgement_log.JudgementLog` or a path
                (wrapped in a default one). ``None`` (default): no logging, zero
                overhead, no wrap. The flywheel inlet later harvested into
                training pairs.

        Returns:
            A :class:`~langres.core.results.DedupeResult` -- a
            ``list[set[str]]`` of id clusters that also reports the
            ``architecture``, ``backbone``, ``score_type`` and effective
            ``threshold`` that produced it.

        Raises:
            ValueError: Duplicate ids, inconsistent id presence, or a nested
                value under an inferred schema.
            BudgetExceeded: If scoring would cross this model's spend cap; the
                exception carries the judgements already produced on
                ``.partial_judgements``.
        """
        if len(records) < 2:
            return DedupeResult(
                [],
                architecture=type(self).__name__,
                backbone=None,
                score_type="none",
                threshold=None,
            )
        # Explicit-chain model: no ``module``/``clusterer`` slot to read, so the
        # front-door reads (threshold/backbone) come from the chain (see
        # _dedupe_explicit).
        if self._ops is not None:
            return self._dedupe_explicit(records, log=_coerce_log(log))
        normalized = self._prepare(records)
        check_no_duplicate_ids([record["id"] for record in normalized])
        pairs = self._scored_pairs(normalized, log=_coerce_log(log))
        # The pipeline exit: an explicit match cut (ThresholdSelect at the
        # clusterer's threshold) THEN pure-equivalence clustering over the
        # survivors (THEORY.md's Select π vs equivalence π); byte-identical to the
        # legacy clusterer-owned cut -- see _cluster.
        clusters = self._cluster(pairs)
        # The scored rows' OWN score_type, not a per-name lookup table: the
        # matcher that ran is the only honest authority on what its scores mean.
        # The first scored row is the first candidate's judgement (one judgement
        # per candidate, in blocker order -- so this equals the legacy
        # ``judgements[0].score_type``). None only when the blocker yielded no
        # pair at all, which reports "unknown".
        score_type = next(
            (row.score_type for row in pairs.rows if row.score_type is not None), None
        )
        return DedupeResult(
            clusters,
            architecture=type(self).__name__,
            backbone=self.backbone,
            score_type=score_type if score_type is not None else "unknown",
            threshold=self.clusterer.threshold,
        )

    def compare(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
        *,
        log: JudgementLog | str | Path | None = None,
    ) -> LinkVerdict:
        """Decide whether two records refer to the same real-world entity.

        The replacement for the deleted ``langres.link`` verb (which scored ONE
        pair). It is deliberately **not** named ``link``: that name is reserved
        for cross-source linkage over two record *sets*, which
        :meth:`~langres.core.resolver.ERModel.link` still stubs out honestly. Two
        incompatible things were called ``link`` before this wave; ``compare`` is
        the pair one, under a name that says so.

        ``compare(a, a)`` (an entity against itself) is well-defined and does not
        raise -- the batch-uniqueness rule is :meth:`dedupe`'s, not this one's.

        **Blocking cannot veto the pair.** Blocking is a recall optimization for
        batches; a caller naming exactly two records has already decided this
        pair is worth judging. So the blocker runs only to *attach* what the
        matcher needs (e.g. a ``VectorBlocker``'s cosine ``similarity_score``,
        which an embedding-backed matcher scores off), and if it yields nothing
        the candidate is built directly instead. Silently answering "no match"
        because a blocker filtered the pair out would be a lie.

        Args:
            left: The first record (a plain dict).
            right: The second record.
            log: Opt-in judgement sink (see :meth:`dedupe`).

        Returns:
            A :class:`~langres.core.results.LinkVerdict` (truthy iff it matched).

        Raises:
            MatcherAbstainedError: If the matcher neither scored nor decided.
                ``compare`` owes its caller a verdict and will not fabricate one.
            BudgetExceeded: If scoring this pair would cross the spend cap.
        """
        # Explicit-chain model: run applicable body Ops on the named pair and
        # preserve compare's explicit abstention error (see _compare_explicit).
        if self._ops is not None:
            return self._compare_explicit(left, right, log=_coerce_log(log))
        normalized = self._prepare([left, right])
        pair = self._pair_candidate(normalized)
        scored = self._matcher_score(log=_coerce_log(log)).forward(pair)
        if self.calibrator is not None:
            scored = self._apply_calibrator_to_pairs(scored, self.calibrator)
        scored_rows = [row for row in scored.rows if row.score_type is not None]
        if not scored_rows:
            raise RuntimeError(
                f"the {type(self.module).__name__} matcher produced no judgement for this pair; "
                "every candidate must yield exactly one PairwiseJudgement. This indicates a bug "
                "in the matcher."
            )
        judgement = scored_rows[0].to_judgement()

        threshold = self.clusterer.threshold
        predicted = predicted_match(judgement, threshold)
        if predicted is None:
            raise MatcherAbstainedError(
                f"the {type(self.module).__name__} matcher abstained (no decision and no score) "
                "on this pair, so compare() cannot return a verdict. An LLMMatcher abstains when "
                "its response fails to parse (the default on_parse_error='abstain'); pass "
                "on_parse_error='raise' to surface the parse failure itself, or catch "
                "MatcherAbstainedError."
            )
        return LinkVerdict(
            match=predicted,
            score=judgement.score,
            reasoning=judgement.reasoning,
            architecture=type(self).__name__,
            backbone=self.backbone,
            score_type=judgement.score_type,
            threshold=threshold,
            judgement=judgement,
        )

    def _pair_candidate(self, records: list[dict[str, Any]]) -> Pairs[Any]:
        """The ONE pair for :meth:`compare` -- blocker-attached, never blocker-vetoed.

        Runs the real pipeline first, so anything the blocker attaches (a
        ``VectorBlocker``'s ``similarity_score``; the comparator's
        ``ComparisonVector``) is present exactly as it would be in a batch, and
        keeps only the *first* blocked pair (mirroring the pre-carrier
        "return the first candidate" behavior). Falls back to a directly-built
        pair when the blocker yields nothing -- see :meth:`compare` on why a
        veto must not decide the verdict.

        Returns a single-row :class:`~langres.core.pairs.Pairs`;
        :meth:`compare` bridges it back to an ``ERCandidate`` at the matcher
        boundary via :meth:`~langres.core.pairs.Pairs.to_candidates`.
        """
        pairs = self._candidates(records)
        if pairs.rows:
            return Pairs(store=pairs.store, rows=pairs.rows[:1])

        from langres.core.blockers.all_pairs import schema_to_factory

        schema = self.schema
        if schema is None:
            raise RuntimeError(
                f"{type(self.blocker).__name__} yielded no candidate for this pair and exposes "
                "no `schema`, so compare() cannot build the pair directly. Use a blocker that "
                "carries its schema (AllPairsBlocker, VectorBlocker), or compare via dedupe()."
            )
        factory = schema_to_factory(schema)
        left_entity, right_entity = (factory(record) for record in records)
        candidate = ERCandidate(left=left_entity, right=right_entity, blocker_name="compare")
        if self.comparator is not None:
            candidate = candidate.model_copy(
                update={"comparison": self.comparator.compare(left_entity, right_entity)}
            )
        return Pairs.from_candidates([candidate])

    # ------------------------------------------------------------------
    # The explicit-chain front door (``_ops`` set) -- the else branch of
    # dedupe/compare. Split out so the classic ``is None`` bodies above stay
    # byte-verbatim. Everything here reads the chain, never the four slots.
    # ------------------------------------------------------------------

    def _dedupe_explicit(
        self, records: list[dict[str, Any]], *, log: JudgementLog | None = None
    ) -> DedupeResult:
        """``dedupe`` over an explicit Op chain.

        Normalizes via the chain's Source schema (the ONE input contract, same as
        :meth:`_prepare`'s), runs the chain (Source + body Ops via
        :meth:`_scored_pairs`, then the terminal ClusterStage via :meth:`_cluster`,
        both dispatched on ``_ops``), and reports the chain's own effective cut
        (:meth:`_chain_threshold`) and :attr:`backbone` -- there is no single
        ``clusterer``/``module`` slot. ``score_type`` stays row-derived (the first
        scored row's own family), exactly as the classic path.
        """
        _schema, normalized = normalize_records(records, self._chain_source_schema())
        check_no_duplicate_ids([record["id"] for record in normalized])
        pairs = self._scored_pairs(normalized, log=log)
        clusters = self._cluster(pairs)
        score_type = next(
            (row.score_type for row in pairs.rows if row.score_type is not None), None
        )
        return DedupeResult(
            clusters,
            architecture=type(self).__name__,
            backbone=self.backbone,
            score_type=score_type if score_type is not None else "unknown",
            threshold=self._chain_threshold(),
        )

    def _compare_explicit(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
        *,
        log: JudgementLog | None = None,
    ) -> LinkVerdict:
        """``compare`` over an explicit Op chain.

        Runs Scores and applicable Selects in their declared order. A Select
        that drops the single row returns a negative verdict from the last
        judgement; an abstention is still surfaced before a ThresholdSelect can
        turn it into an indistinguishable empty relation.
        """
        _schema, normalized = normalize_records([left, right], self._chain_source_schema())
        pair = self._chain_pair_candidate(normalized)
        current = pair
        judgement: PairwiseJudgement | None = None
        backbone: str | None = None
        threshold: float | None = None
        for op in self._explicit_body(log=log):
            if isinstance(op, ThresholdSelect):
                threshold = op.threshold
                if judgement is not None and predicted_match(judgement, threshold) is None:
                    raise MatcherAbstainedError(
                        "the explicit chain's matcher abstained (no decision and no score) on "
                        "this pair, so compare() cannot return a verdict."
                    )
            current = op.forward(current)
            if isinstance(op, MatcherScore) or getattr(op, "resource", None) is not None:
                backbone = self._stage_resource_ref(op)
            scored_rows = [row for row in current.rows if row.score_type is not None]
            if scored_rows:
                judgement = scored_rows[0].to_judgement()
            if isinstance(op, Select) and not current.rows:
                if judgement is None:
                    raise RuntimeError(
                        "the explicit chain selected this pair before any Score produced a "
                        "judgement; compare() cannot explain that selection."
                    )
                effective_threshold = (
                    threshold if threshold is not None else self._chain_threshold()
                )
                if effective_threshold is None:
                    raise RuntimeError(
                        "compare() needs a match cut to report a verdict, but this explicit "
                        "chain has no ThresholdSelect."
                    )
                return self._explicit_verdict(
                    judgement=judgement,
                    match=False,
                    threshold=effective_threshold,
                    backbone=backbone,
                )

        if judgement is None:
            raise RuntimeError(
                "the explicit chain's Score ops produced no judgement for this pair; every "
                "candidate must yield exactly one PairwiseJudgement. This indicates a bug in the "
                "chain's matcher Score."
            )

        threshold = self._chain_threshold()
        if threshold is None:
            raise RuntimeError(
                "compare() needs a match cut to gate on, but this explicit chain has no terminal "
                "ThresholdSelect. Fix: add a ThresholdSelect before the ClusterStage, or compare "
                "via dedupe()."
            )
        predicted = predicted_match(judgement, threshold)
        if predicted is None:
            raise MatcherAbstainedError(
                "the explicit chain's matcher abstained (no decision and no score) on this pair, "
                "so compare() cannot return a verdict. An LLMMatcher abstains when its response "
                "fails to parse (the default on_parse_error='abstain'); pass "
                "on_parse_error='raise' to surface the parse failure, or catch "
                "MatcherAbstainedError."
            )
        return self._explicit_verdict(
            judgement=judgement,
            match=predicted,
            threshold=threshold,
            backbone=backbone,
        )

    def _explicit_verdict(
        self,
        *,
        judgement: PairwiseJudgement,
        match: bool,
        threshold: float,
        backbone: str | None,
    ) -> LinkVerdict:
        """Build explicit-chain compare metadata from one scored judgement."""
        return LinkVerdict(
            match=match,
            score=judgement.score,
            reasoning=judgement.reasoning,
            architecture=type(self).__name__,
            backbone=backbone,
            score_type=judgement.score_type,
            threshold=threshold,
            judgement=judgement,
        )

    def _chain_pair_candidate(self, records: list[dict[str, Any]]) -> Pairs[Any]:
        """The ONE pair for :meth:`_compare_explicit` -- the chain-Source analogue of
        :meth:`_pair_candidate`.

        Runs the chain's Source and keeps the first blocked pair; builds the pair
        directly (from the Source's schema) if the Source yields nothing, because
        blocking must never veto a ``compare`` verdict (see :meth:`compare`). The
        comparator vector is NOT attached here -- folding the chain's Score ops
        (which includes any ``ComparatorScore``) attaches it, unlike the slot-based
        :meth:`_pair_candidate`.
        """
        source = self._chain_source()
        source.prepare(records)
        pairs = source.forward(records)
        if pairs.rows:
            return Pairs(store=pairs.store, rows=pairs.rows[:1])

        from langres.core.blockers.all_pairs import schema_to_factory

        schema = self._chain_source_schema()
        if schema is None:
            raise RuntimeError(
                "the explicit chain's Source yielded no candidate for this pair and exposes no "
                "schema, so compare() cannot build the pair directly. Use a Source whose blocker "
                "carries a schema (AllPairsBlocker, VectorBlocker, KeyBlocker), or compare via "
                "dedupe()."
            )
        factory = schema_to_factory(schema)
        left_entity, right_entity = (factory(record) for record in records)
        candidate = ERCandidate(left=left_entity, right=right_entity, blocker_name="compare")
        return Pairs.from_candidates([candidate])

    def _ensure_index_built(self, records: list[Any]) -> None:
        """Backward-compatible delegate to the Source-owned prepare lifecycle.

        New execution paths call :meth:`Source.prepare` through
        :func:`_run_stages`; this method remains for older internal callers.
        """
        BlockerSource(self.blocker).prepare(records)
