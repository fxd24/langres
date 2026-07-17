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

Everything that scores goes through :meth:`ModelRun._scorer`, so the spend cap
cannot be bypassed; ``tests/core/test_resolver_spend_cap.py`` AST-bans
``<any>.module.forward(...)`` in ``src/`` to keep it that way.
"""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from langres.core._model_state import ModelState
from langres.core.blockers.composite import CompositeBlocker
from langres.core.fit import CalibratorFitMixin
from langres.core.inputs import check_no_duplicate_ids
from langres.core.judgement_log import JudgementLog, LoggingMatcher
from langres.core.models import (
    ERCandidate,
    MatcherAbstainedError,
    PairwiseJudgement,
    predicted_match,
)
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


class ModelRun(ModelState):
    """The run path of an ``ERModel``: candidates, scoring, clustering, front door."""

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

    def _scorer(self, *, log: JudgementLog | None = None) -> SpendCappedMatcher:
        """This model's matcher, metered by its ONE spend ledger.

        **The only supported way to score through ``self.module``.** Reaching for
        the raw ``self.module.forward(...)`` gets you an *uncapped* scorer that
        bills straight past ``budget_usd`` and never touches this instance's
        ledger. That is not hypothetical: ``.module`` is a public attribute, and
        :class:`~langres.core.anchor_store.AnchorStore` reached through it and
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

    def _judgements(
        self, records: list[Any], *, log: JudgementLog | None = None
    ) -> Iterator[PairwiseJudgement]:
        """Block records into candidates, score them, and calibrate if fitted.

        Scoring runs through :meth:`_scorer`, so this model's spend cap
        (``budget_usd=``) and its ledger are shared across every
        :meth:`resolve`/:meth:`predict` call on this instance -- two successive
        resolves cannot each spend a full budget.

        When :attr:`calibrator` is set (by ``fit(method=Platt()/Isotonic())``),
        every ranking judgement's raw ``score`` is mapped to a calibrated
        probability before it reaches :meth:`predict`/:meth:`resolve` -- so the
        clusterer thresholds on a real probability. Pure pass-through otherwise.
        """
        judgements = self._scorer(log=log).forward(self._candidates(records))
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
                :class:`~langres.core.judgement_log.JudgementLog` or a path
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
        judgements = list(self._judgements(normalized, log=_coerce_log(log)))
        clusters = self.clusterer.cluster(iter(judgements))
        return DedupeResult(
            clusters,
            architecture=type(self).__name__,
            backbone=self.backbone,
            # The judgements' OWN score_type, not a per-name lookup table: the
            # matcher that ran is the only honest authority on what its scores
            # mean. "unknown" only when the blocker yielded no pair at all.
            score_type=judgements[0].score_type if judgements else "unknown",
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
        candidate = self._pair_candidate(normalized)
        judgements = list(self._scorer(log=_coerce_log(log)).forward(iter([candidate])))
        if not judgements:
            raise RuntimeError(
                f"the {type(self.module).__name__} matcher produced no judgement for this pair; "
                "every candidate must yield exactly one PairwiseJudgement. This indicates a bug "
                "in the matcher."
            )
        judgement = judgements[0]
        if self.calibrator is not None:
            judgement = next(self._apply_calibrator(iter([judgement]), self.calibrator))

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

    def _pair_candidate(self, records: list[dict[str, Any]]) -> ERCandidate[Any]:
        """The ONE candidate for :meth:`compare` -- blocker-attached, never blocker-vetoed.

        Runs the real pipeline first, so anything the blocker attaches (a
        ``VectorBlocker``'s ``similarity_score``; the comparator's
        ``ComparisonVector``) is present exactly as it would be in a batch. Falls
        back to a directly-built candidate when the blocker yields nothing --
        see :meth:`compare` on why a veto must not decide the verdict.
        """
        for candidate in self._candidates(records):
            return candidate

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
        return candidate

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
