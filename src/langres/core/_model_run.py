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

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from langres.core._model_state import ModelState
from langres.core.blockers.composite import CompositeBlocker
from langres.core.clusterer import Clusterer
from langres.core.fit import CalibratorFitMixin
from langres.core.inputs import check_no_duplicate_ids
from langres.tracking.judgement_log import JudgementLog, LoggingMatcher
from langres.core.models import (
    ERCandidate,
    MatcherAbstainedError,
    PairwiseJudgement,
    predicted_match,
)
from langres.core.op import Score, Source, ThresholdSelect
from langres.core.op_adapters import (
    BlockerSource,
    ClustererStage,
    ComparatorScore,
    MatcherScore,
)
from langres.core.pairs import PairRow, Pairs
from langres.core.results import DedupeResult, LinkVerdict
from langres.core.spend_cap import SpendCappedMatcher

if TYPE_CHECKING:
    # [semantic] extra (faiss/sentence-transformers/torch) -- imported lazily
    # inside _ensure_index_built (W0.4) so a core-only `import langres` never
    # pulls faiss/torch in for a model that never uses a VectorBlocker.
    from langres.core.blockers.vector import VectorBlocker

logger = logging.getLogger(__name__)


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
    deliberately (W0.4, mirrors :meth:`ModelRun._ensure_index_built`'s own
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


def _run_stages(records: list[Any], stages: list[Source[Any] | Score[Any]]) -> Pairs[Any]:
    """Fold ``forward`` across ``stages`` to turn ``records`` into a scored ``Pairs``.

    The generic driver behind :meth:`ModelRun._candidates` and
    :meth:`ModelRun._scored_pairs`. Both stage lists it is handed --
    :meth:`ModelRun._stages` (full) and its matcher-free prefix
    :meth:`ModelRun._block_compare_stages` -- lead with the
    :class:`~langres.core.op.Source` (records -> pairs) and continue with
    :class:`~langres.core.op.Score`\\ s (pairs -> pairs), so the fold walks
    ``Records -> Pairs -> ... -> Pairs``. The two narrowing ``cast``\\ s
    encode that Source-first shape (which the factories guarantee) precisely --
    ``Source.forward`` takes records while ``Score.forward`` takes a ``Pairs``, so
    the heterogeneous first stage cannot share one loosely-typed loop variable
    with the rest, and the cast is a typed assertion of the shape, not an ``Any``.
    """
    source, *scores = stages
    pairs = cast(Source[Any], source).forward(records)
    for score in scores:
        pairs = cast(Score[Any], score).forward(pairs)
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
        nor require a scoring slot. The index side-effect is deliberately NOT
        absorbed by the source adapter -- :meth:`_ensure_index_built` still runs
        here first, before the source streams.
        """
        self._ensure_index_built(records)
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
        return stages

    def _scored_pairs(self, records: list[Any], *, log: JudgementLog | None = None) -> Pairs[Any]:
        """Block -> (compare) -> score as an Op chain, then calibrate: the scoring spine.

        Folds the driver (:func:`_run_stages`) over the full :meth:`_stages`
        pipeline -- :class:`~langres.core.op_adapters.BlockerSource`, then
        (optionally) :class:`~langres.core.op_adapters.ComparatorScore`, then the
        :class:`~langres.core.op_adapters.MatcherScore` (:meth:`_matcher_score`) --
        so the topology is one explicit chain of ``forward`` calls over the single
        ``Pairs`` carrier, built from one factory rather than scattered inline.
        The index side-effect is not in the source adapter, so
        :meth:`_ensure_index_built` runs here first (as it did via
        :meth:`_candidates`). Scoring runs through :meth:`_scorer`, so this model's
        spend cap (``budget_usd=``) and its ledger are shared across every
        :meth:`resolve`/:meth:`predict`/:meth:`dedupe` call on this instance -- two
        successive resolves cannot each spend a full budget, and a matcher that
        overruns the cap raises ``BudgetExceeded`` from the ``MatcherScore`` drain
        here.

        When :attr:`calibrator` is set (by ``fit(method=Platt()/Isotonic())``),
        every scored ranking row's raw ``score`` is mapped to a calibrated
        probability (:meth:`_apply_calibrator_to_pairs`) before clustering, so the
        clusterer thresholds on a real probability. Pure pass-through otherwise.
        Calibration is a per-score transform applied in the spine, NOT an Op
        adapter (kept here through W3-c).
        """
        self._ensure_index_built(records)
        pairs = _run_stages(records, self._stages(log=log))
        if self.calibrator is not None:
            pairs = self._apply_calibrator_to_pairs(pairs, self.calibrator)
        return pairs

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
        rows: list[PairRow[Any]] = []
        for row in pairs.rows:
            if row.score_type is None or row.score is None:
                rows.append(row)
                continue
            calibrated = calibrator.transform([row.score])[0]
            rows.append(
                row.model_copy(
                    update={
                        "score": calibrated,
                        "score_type": "calibrated_prob",
                        "provenance": {
                            **row.provenance,
                            "calibration": {
                                "method": getattr(calibrator, "method", None),
                                "raw_score": row.score,
                            },
                        },
                    }
                )
            )
        return Pairs(store=pairs.store, rows=rows)

    def predict(self, records: list[Any]) -> list[PairwiseJudgement]:
        """Return the scored pairwise judgements before clustering.

        Useful for observability/tuning: inspect scores and provenance without
        committing to a clustering threshold.
        """
        return list(self._judgements(records))

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
        """
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
        Singletons are dropped by the Clusterer (it returns only connected
        components with an edge), so the result contains only multi-record
        clusters.

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

    def _ensure_index_built(self, records: list[Any]) -> None:
        """Build/populate every reachable ``VectorBlocker``'s index from ``records``.

        Embeds the records' text field and creates the index in place for each
        index-backed blocker discovered via :func:`iter_vector_blockers` --
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
        :func:`iter_vector_blockers`'s docstring for why -- this method runs
        on every ``resolve()``/``predict()`` call regardless of blocker, so an
        ``isinstance`` check would need ``VectorBlocker`` imported
        unconditionally, pulling faiss/sentence-transformers (the
        ``[semantic]`` extra) into a plain ``AllPairsBlocker``/``KeyBlocker``
        pipeline.
        """
        for vector_blocker in iter_vector_blockers(self.blocker):
            entities = [vector_blocker.schema_factory(record) for record in records]
            texts = [vector_blocker.text_field_extractor(entity) for entity in entities]

            index = vector_blocker.vector_index
            if vector_blocker._index_is_built() and getattr(index, "_corpus_texts", None) == texts:
                continue  # Same corpus already indexed -> reuse, never re-embed.

            logger.info("Embedding %d records to build the blocker's vector index…", len(texts))
            index.create_index(texts)
