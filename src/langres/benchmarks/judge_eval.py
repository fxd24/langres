"""Benchmark harness: direct pair-level judge evaluation (fixed candidate set).

The scorer-isolating half of the harness: :func:`evaluate` /
:func:`evaluate_judge_on_candidates` score one :class:`~langres.core.matcher.Matcher`
over a *given* candidate set against gold pairs — no blocking, no clustering — so
the number is directly comparable to pairwise-F1 SOTA. :class:`BudgetedModuleRunner`
runs any matcher under a spend cap. Depends one-way on :mod:`langres.benchmarks.runner`
(cost tracks, ``_cost_track``, ``_validated_grid``) and on the spec in
:mod:`langres.data.benchmark` (:class:`~langres.data.benchmark.PairTrack`,
``DEFAULT_PAIR_GRID``, ``BlindCostError``).
"""

import logging
import math
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Literal

from pydantic import BaseModel

from langres.benchmarks.runner import (
    LatencyTrack,
    _cost_track,
    _judgement_cost,
    _validate_cut,
    _validated_grid,
)
from langres.core.matcher import Matcher
from langres.core.models import PairwiseJudgement
from langres.core.spend_cap import effective_budget
from langres.core.usage import CostTrack
from langres.data.benchmark import DEFAULT_PAIR_GRID, BlindCostError, PairTrack
from langres.metrics.metrics import classify_pairs, pair_pr_curve

logger = logging.getLogger(__name__)


class JudgePairEval(BaseModel):
    """Pair-level evaluation of one judge over a *given* candidate set.

    Unlike :func:`run_method` (which loads a benchmark, blocks, tunes a clusterer,
    and clusters), this is the scorer-isolating eval used when the candidate pairs
    are fixed up front — e.g. a literature train/valid/test pair split, or a single
    dataset's blocked band — so the number is directly comparable to pairwise-F1
    SOTA without any blocking-recall ceiling or clustering amplification.

    Attributes:
        pair: Pair-level P/R/F1 at ``graded_threshold`` — the grid's best-F1
            argmax when ``best_threshold`` is set, or the caller's fixed
            ``threshold=`` (:func:`evaluate` only) otherwise — plus the full PR
            curve (``pr_curve``, populated in both modes).
        cost: Spend accounting over the judgements actually produced.
        latency: Wall-clock per judged pair.
        n_candidates: Candidate pairs handed to the judge.
        n_judged: Judgements actually produced (``< n_candidates`` when a budget
            runner truncates or a call is skipped).
        best_threshold: The grid threshold maximizing pair-level F1, or ``None``
            when :func:`evaluate` was given a fixed ``threshold=`` — setting it
            to the argmax in that case would misrepresent an honest fixed cut as
            a tuned one.
        graded_threshold: The threshold ``pair`` (and ``slices``, if present)
            were actually graded at — ALWAYS populated, equal to
            ``best_threshold`` in argmax mode or the caller's fixed
            ``threshold=`` otherwise.
        truncated: ``True`` when fewer pairs were judged than supplied (a budget
            cap fired or calls were skipped) — the cell is partial.
        truncation_reason: Why ``truncated`` is ``True`` (``"none"`` otherwise):
            ``"budget_cap"`` / ``"budget_stop"`` for the two ways a
            :class:`BudgetedModuleRunner` can stop early (spend-caused), or
            ``"judge_skips"`` when the judge itself produced fewer judgements
            than candidates (a call raised or yielded nothing — NOT a spend
            issue). :func:`evaluate` treats the two very differently via
            ``on_truncation``.
        budget_exceeded: ``True`` when measured spend (:attr:`cost`'s
            ``usd_total``) crossed ``budget_usd`` DURING a call — detected only
            AFTER that call's real cost was tallied, since the runner's per-pair
            cap projects on a worst-case *estimate*, not the real cost a call
            turns out to have. This is deliberately independent of
            :attr:`truncated`: a breach on the LAST candidate still judges every
            pair (``n_judged == n_candidates``, ``truncated=False``) while
            ``budget_exceeded`` is ``True`` — the run is complete, but it cost
            more than the requested ceiling. :func:`evaluate` always warns
            loudly when this is ``True`` (never raises solely for it — the money
            is already spent). ``False`` when no :class:`BudgetedModuleRunner`
            was used (``runner=None`` in :func:`evaluate_judge_on_candidates`).
        slices: Optional per-slice pair tracks, each graded at the SAME fixed
            ``graded_threshold`` as ``pair`` (never a per-slice argmax). Populated
            only when :func:`evaluate_judge_on_candidates` is given a ``slice_fn``;
            ``None`` otherwise. This is the honest seen -> unseen view: one global
            cut, reported across data slices, so a degradation cannot be tuned
            away.
        n_parse_errors: How many judgements were abstentions, detected
            belt-and-suspenders — either the Wave-1 abstain shape
            (:attr:`~langres.core.models.PairwiseJudgement.is_abstain`, i.e.
            ``decision is None and score is None``) OR a judge that flagged
            ``provenance['parse_error']`` (truthy) — e.g. an ``LLMMatcher`` under
            ``on_parse_error='abstain'`` or a ``DSPyMatcher`` parse failure. Kept
            for backward compatibility; identical to :attr:`n_abstained`. ``0``
            for judges that never abstain.
        n_abstained: Same count as :attr:`n_parse_errors`, under the more
            general name — an abstention isn't necessarily a "parse" error for
            every judge kind (a judge may simply emit the ``None``/``None``
            abstain shape without setting ``parse_error``).
        abstention_rate: ``n_abstained / n_judged`` (``0.0`` when ``n_judged``
            is ``0``) — the normalized companion to the raw count.
    """

    pair: PairTrack
    cost: CostTrack
    latency: LatencyTrack
    n_candidates: int
    n_judged: int
    best_threshold: float | None
    graded_threshold: float
    truncated: bool
    truncation_reason: Literal["none", "budget_cap", "budget_stop", "judge_skips"] = "none"
    budget_exceeded: bool = False
    slices: dict[str, PairTrack] | None = None
    n_parse_errors: int = 0
    n_abstained: int = 0
    abstention_rate: float = 0.0


def _grade_slices(
    judgements: list[PairwiseJudgement],
    gold_in_scope: set[frozenset[str]],
    slice_fn: Callable[[frozenset[str]], str | None],
    threshold: float,
) -> dict[str, PairTrack]:
    """Grade each data slice at ONE fixed ``threshold`` (never a per-slice argmax).

    The load-bearing honesty rule of the sliced eval: the global best-F1
    ``threshold`` is chosen once over *all* judged pairs, then every slice is
    graded at that SAME cut via :func:`~langres.core.metrics.classify_pairs`. A
    per-slice argmax would re-tune the threshold within each slice and so hide the
    seen -> unseen degradation this report exists to expose.

    Judged pairs are grouped by ``slice_fn`` of their order-independent pair key,
    and ``gold_in_scope`` by the same tagger; a pair tagged ``None`` is excluded
    from every slice. The judged-tag and gold-tag sets are *unioned* so a slice
    holding gold pairs but no judged pair (e.g. a budget runner dropped its
    candidates, or the judge skipped them) still reports recall — ``0.0`` with no
    divide-by-zero — rather than silently vanishing.

    Args:
        judgements: The judgements actually produced by the judge.
        gold_in_scope: Gold pairs realizable from the candidate set (already
            intersected with the candidate pairs by the caller).
        slice_fn: Maps a pair key to a slice tag, or ``None`` to drop the pair.
        threshold: The single fixed cut every slice is graded at.

    Returns:
        ``tag -> PairTrack`` (``pr_curve=None`` — a single fixed cut has no curve).
    """
    judged_by_tag: dict[str, list[PairwiseJudgement]] = defaultdict(list)
    for judgement in judgements:
        tag = slice_fn(frozenset((judgement.left_id, judgement.right_id)))
        if tag is not None:
            judged_by_tag[tag].append(judgement)
    gold_by_tag: dict[str, set[frozenset[str]]] = defaultdict(set)
    for pair_key in gold_in_scope:
        tag = slice_fn(pair_key)
        if tag is not None:
            gold_by_tag[tag].add(pair_key)
    slices: dict[str, PairTrack] = {}
    for tag in set(judged_by_tag) | set(gold_by_tag):
        metrics = classify_pairs(judged_by_tag.get(tag, []), gold_by_tag.get(tag, set()), threshold)
        slices[tag] = PairTrack(precision=metrics.precision, recall=metrics.recall, f1=metrics.f1)
    return slices


def _gold_in_scope(
    candidates: Sequence[Any], gold_pairs: set[frozenset[str]]
) -> set[frozenset[str]]:
    """Restrict ``gold_pairs`` to pairs realizable from ``candidates``.

    When ``candidates`` is a subsample of a larger fixed-pair set, a gold pair
    whose candidate was not sampled is a blocking-style miss, not a judge error
    — counting it would cap recall artificially (e.g. a 600-pair subsample
    holding 61 of 234 gold pairs caps recall at 61/234≈0.26 for *every* method).
    Restricting gold to candidate-realizable pairs keeps grading a pure judge
    metric, and is a no-op when the candidates already cover all of
    ``gold_pairs``. Shared by :func:`evaluate_judge_on_candidates` and
    :func:`evaluate` so the two can never compute this differently.
    """
    candidate_pairs = {frozenset((cand.left.id, cand.right.id)) for cand in candidates}
    return gold_pairs & candidate_pairs


def _truncation_reason(
    n_judged: int, n_candidates: int, runner: "BudgetedModuleRunner | None"
) -> Literal["none", "budget_cap", "budget_stop", "judge_skips"]:
    """Classify WHY fewer pairs were judged than supplied (``"none"`` if not truncated).

    Spend-caused truncation (a runner's pre-flight cap, per-pair projected-spend
    stop, or post-call real-spend breach) is distinguished from the judge itself
    simply producing fewer judgements than candidates (an internal skip-continue
    with no runner at all, or a runner catching a per-call exception/empty
    yield) — :func:`evaluate`'s ``on_truncation`` policy treats the two very
    differently: spend causes may raise; a judge skip only ever warns.
    ``runner.budget_exceeded`` (the post-call breach a single call's real cost
    can cause -- see :class:`BudgetedModuleRunner`) is checked alongside
    ``runner.budget_stopped`` (the pre-call projected-spend stop): both are
    "the runner decided to stop because of money", so both classify as
    ``"budget_stop"`` here.
    """
    if n_judged >= n_candidates:
        return "none"
    if runner is not None:
        if runner.dropped_by_cap_count > 0:
            return "budget_cap"
        if runner.budget_stopped or runner.budget_exceeded:
            return "budget_stop"
    return "judge_skips"


def evaluate_judge_on_candidates(
    module: Matcher[Any],
    candidates: Sequence[Any],
    gold_pairs: set[frozenset[str]],
    grid: Sequence[float],
    *,
    slice_fn: Callable[[frozenset[str]], str | None] | None = None,
    runner: "BudgetedModuleRunner | None" = None,
    price_per_token_or_pair: float = 0.0,
    cost_track_fn: Callable[[list[PairwiseJudgement]], CostTrack] = _cost_track,
) -> tuple[JudgePairEval, list[PairwiseJudgement]]:
    """Score one judge over a fixed candidate set and grade it at the pair level.

    Runs the judge (optionally under a :class:`BudgetedModuleRunner` for paid
    judges), times it, and grades the resulting judgements against ``gold_pairs``
    with :func:`~langres.core.metrics.pair_pr_curve` — selecting the best-F1
    threshold on the grid and keeping the whole curve. Dataset-agnostic and
    blocking-free: the caller supplies the candidates (fixed pairs or a blocked
    band) and the order-independent gold match pairs.

    Args:
        module: The scorer to evaluate (any :class:`~langres.core.matcher.Matcher`).
        candidates: The fixed candidate pairs to judge (each an ``ERCandidate``).
        gold_pairs: True match pairs as order-independent ``frozenset`` pairs.
        grid: Score thresholds to sweep for the pair-level PR curve. Every point
            must lie in ``[0.0, 1.0]`` (``0.0`` is the curve's predict-all anchor).
            Validated — and materialised, so a generator is safe — BEFORE the judge
            runs, so a bad grid never costs an API call.
        slice_fn: Optional tagger mapping a pair key (``frozenset({left_id,
            right_id})``) to a slice tag, or ``None`` to exclude the pair. When
            given, the judged pairs and the in-scope gold are grouped by tag and
            each slice is graded at the ONE global ``best_threshold`` (never a
            per-slice argmax), populating ``JudgePairEval.slices``. When ``None``
            (default), ``slices`` stays ``None`` and the result is unchanged.
        runner: Optional budget runner. When given, the judge runs through it (its
            ``module`` must be ``module``) so spend is capped BETWEEN calls -- a
            single in-flight call can still overrun the cap by its own cost; the
            run may therefore judge fewer pairs than supplied. When ``None``, the
            judge is run directly (zero-spend path).
        price_per_token_or_pair: Worst-case price passed to ``runner.run`` (ignored
            without a runner). Must be ``> 0`` when a runner is given.
        cost_track_fn: Aggregator from judgements to a :class:`CostTrack`. Defaults
            to the flat :func:`_cost_track`; pass
            :func:`~langres.methods.cascade_cost_track` for cascade escalation
            diagnostics.

    Raises:
        ValueError: If ``grid`` is empty or holds a point outside ``(0.0, 1.0]``.

    Returns:
        ``(JudgePairEval, judgements)`` — the graded summary plus the raw
        judgements (kept in-process for error-map analysis; not part of the
        summary so a persisted result stays small).
    """
    # Validate BEFORE judging: this is the paid path, and a degenerate grid must
    # not cost money to discover. `grid=(0.0,)` would let classify_pairs count an
    # abstention (score=0.0) as a true match, inflating P/R/F1.
    grid = _validated_grid(grid)

    start = time.perf_counter()
    if runner is not None:
        judgements = runner.run(candidates, price_per_token_or_pair)
    else:
        judgements = list(module.forward(iter(candidates)))
    elapsed = time.perf_counter() - start

    # Grade the judge only on pairs it was actually given (see _gold_in_scope).
    gold_in_scope = _gold_in_scope(candidates, gold_pairs)
    curve = pair_pr_curve(judgements, gold_in_scope, grid)
    best = max(curve, key=lambda m: m.f1)
    pair = PairTrack(precision=best.precision, recall=best.recall, f1=best.f1, pr_curve=curve)

    # Sliced view (honest): grade every slice at the SAME global best.threshold.
    slices = (
        _grade_slices(judgements, gold_in_scope, slice_fn, best.threshold)
        if slice_fn is not None
        else None
    )

    n_candidates = len(candidates)
    n_judged = len(judgements)
    truncated = n_judged < n_candidates
    truncation_reason = _truncation_reason(n_judged, n_candidates, runner)

    # Abstention accounting: an abstention carries no actionable verdict, so
    # surface the count and warn loudly -- otherwise a table of P/R/F1 could be
    # built partly on non-verdicts (e.g. a naive replication of a paper prompt
    # the judge could not parse) with no visible signal. Detect an abstention
    # belt-and-suspenders: the Wave-1 abstain shape (``is_abstain`` -> decision
    # is None and score is None) OR a judge that flagged
    # ``provenance['parse_error']`` -- a judge may null the verdict, set the
    # flag, or (like a fixed DSPyMatcher parse failure) do both.
    n_abstained = sum(1 for j in judgements if j.is_abstain or j.provenance.get("parse_error"))
    abstention_rate = n_abstained / n_judged if n_judged > 0 else 0.0
    if n_abstained > 0:
        logger.warning(
            "%d of %d judgements (%.1f%%) were abstentions (is_abstain -- "
            "decision=None and score=None -- or provenance['parse_error']). A "
            "contract-correct abstention is EXCLUDED from the predicted set: "
            "classify_pairs() routes through predicted_match(), which returns "
            "None (never True) for an abstain, so it is counted neither as a "
            "confident match nor as a graded 'no'. This does NOT flatter recall "
            "-- an abstention on a true gold pair still lands in "
            "gold_pairs - predicted and is counted as a false negative, so the "
            "missing verdict costs recall exactly as it should. Inspect the "
            "judge's response_parser / prompt before trusting these numbers.",
            n_abstained,
            n_judged,
            abstention_rate * 100.0,
        )

    latency = LatencyTrack(seconds_per_pair=elapsed / n_judged if n_judged > 0 else 0.0)
    result = JudgePairEval(
        pair=pair,
        cost=cost_track_fn(judgements),
        latency=latency,
        n_candidates=n_candidates,
        n_judged=n_judged,
        best_threshold=best.threshold,
        graded_threshold=best.threshold,
        truncated=truncated,
        truncation_reason=truncation_reason,
        budget_exceeded=runner.budget_exceeded if runner is not None else False,
        slices=slices,
        n_parse_errors=n_abstained,
        n_abstained=n_abstained,
        abstention_rate=abstention_rate,
    )
    return result, judgements


class EvaluationTruncatedError(RuntimeError):
    """Raised by :func:`evaluate` when its internal spend cap truncated the run.

    Raised for spend-caused truncation (``truncation_reason`` in
    ``{"budget_cap", "budget_stop"}``) under the default ``on_truncation="raise"``
    — a ``"judge_skips"`` truncation only ever warns (see :func:`evaluate`) —
    and, regardless of the reason, when the judge produced **no judgements at
    all**: ``n_judged == 0`` over a non-empty candidate set is a failed run, not
    a partial one, and P/R/F1 of ``0.0`` would be indistinguishable from a judge
    that genuinely matched nothing.

    :attr:`partial` carries the truncated :class:`JudgePairEval` (set by the
    catcher immediately before re-raising, mirroring
    :class:`BlindCostError`.partial and
    :class:`~langres.clients.openrouter.BudgetExceeded`.partial_judgements) so a
    caller can recover the partial result instead of losing already-paid work.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.partial: JudgePairEval | None = None


#: Deliberately negligible "worst-case per pair" price for evaluate()'s internal
#: BudgetedModuleRunner. evaluate() accepts ANY Matcher and has no way to know its
#: real per-pair price (only evaluate_judge_on_candidates's explicit
#: price_per_token_or_pair= knob does), so the runner's pre-flight worst-case cap
#: is effectively disabled by this near-zero price -- it exists only to avoid
#: BlindCostError, never to model a real price. The per-pair "budget stop" still
#: works correctly because it tallies REAL, already-measured cost
#: (provenance["cost_usd"]) after each judged pair, so the cap tracks actual
#: spend, not a guessed one -- exactly why a free judge (string/embedding, which
#: never sets cost_usd) can never reach it.
_NEGLIGIBLE_WORST_CASE_PRICE = 1e-9


def _budget_truncation_message(result: JudgePairEval, budget_usd: float) -> str:
    """Actionable message for a spend-caused truncation: name the fix, not just the fault."""
    return (
        f"evaluate() stopped early ({result.truncation_reason}): judged "
        f"{result.n_judged}/{result.n_candidates} pairs before the ${budget_usd:.2f} "
        "budget_usd cap.\n"
        "Fix A: pass a larger budget_usd= to judge the full candidate set.\n"
        'Fix B: pass on_truncation="return" to accept the partial result silently '
        f"(.n_judged={result.n_judged}, .truncated=True) instead of raising.\n"
        'Fix C: pass on_truncation="warn" to log instead of raising.'
    )


def _budget_exceeded_message(result: JudgePairEval, budget_usd: float) -> str:
    """Actionable message when a single in-flight call pushed spend past the cap.

    Distinct from :func:`_budget_truncation_message`: this fires whenever
    ``result.budget_exceeded`` is ``True``, REGARDLESS of ``truncated`` -- a
    breach on the last candidate still judges every pair, so there is nothing
    to "fix" by raising the budget or retrying; this is purely informational
    (the money is already spent and the result is otherwise valid).
    """
    return (
        f"evaluate(): spend exceeded budget_usd mid-call -- measured usd_total="
        f"${result.cost.usd_total:.4f} against a ${budget_usd:.2f} cap. The runner's "
        "cap is enforced BETWEEN calls, so a single in-flight call can push total "
        "spend past it by that call's own cost. The run completed "
        f"({result.n_judged}/{result.n_candidates} pairs judged) and its metrics "
        "are valid -- this is a report, not an error."
    )


def _judge_skips_message(result: JudgePairEval) -> str:
    """Message for a non-spend truncation (the judge itself produced fewer judgements)."""
    return (
        f"evaluate(): the judge produced only {result.n_judged}/{result.n_candidates} "
        'judgements (truncation_reason="judge_skips") -- one or more candidates '
        "raised or yielded no verdict. This is not a spend issue, so evaluate() never "
        "raises for it; inspect the judge for the failing candidate(s)."
    )


def _empty_run_message(result: JudgePairEval) -> str:
    """Message for a run that produced no judgements at all — a failed run, not a partial one."""
    return (
        f"evaluate(): the judge produced 0 judgements over {result.n_candidates} "
        f'candidates (truncation_reason="{result.truncation_reason}"). Nothing was '
        "graded, so the reported precision/recall/F1 of 0.0 would mean 'the run "
        "never happened', not 'the judge matched nothing' -- and the two are "
        "indistinguishable in a results table.\n"
        "Fix A: inspect the judge -- every candidate raised, or it yielded no verdict.\n"
        "Fix B: raise budget_usd if the very first pair exhausted the spend cap.\n"
        'Fix C: pass on_truncation="return" to accept the empty result deliberately '
        "(.n_judged=0, .truncated=True) instead of raising."
    )


def _apply_truncation_policy(
    result: JudgePairEval,
    *,
    on_truncation: Literal["raise", "warn", "return"],
    budget_usd: float,
) -> None:
    """Warn or raise per ``on_truncation``, based on ``result.truncation_reason``.

    ``"return"`` is silent for every reason. Otherwise ``"judge_skips"`` only
    ever warns (never raises, even under ``on_truncation="raise"`` -- one bad
    candidate must not blow up a run); a spend-caused reason (``"budget_cap"`` /
    ``"budget_stop"``) warns under ``"warn"`` and raises
    :class:`EvaluationTruncatedError` under ``"raise"`` (the default).

    The one exception to "judge_skips only warns": if the judge produced *zero*
    judgements over a non-empty candidate set, there is no partial result to
    salvage -- ``pair.f1`` would be ``0.0``, indistinguishable from a judge that
    ran fine and matched nothing. That raises under both ``"raise"`` and
    ``"warn"``.
    """
    reason = result.truncation_reason
    if reason == "none" or on_truncation == "return":
        return
    if result.n_judged == 0 and result.n_candidates > 0:
        error = EvaluationTruncatedError(_empty_run_message(result))
        error.partial = result
        raise error
    if reason == "judge_skips":
        warnings.warn(_judge_skips_message(result), UserWarning, stacklevel=3)
        return
    if on_truncation == "warn":
        warnings.warn(_budget_truncation_message(result, budget_usd), UserWarning, stacklevel=3)
        return
    error = EvaluationTruncatedError(_budget_truncation_message(result, budget_usd))
    error.partial = result
    raise error


def evaluate(
    module: Matcher[Any],
    candidates: Sequence[Any],
    gold_pairs: set[frozenset[str]],
    *,
    grid: Sequence[float] = DEFAULT_PAIR_GRID,
    threshold: float | None = None,
    budget_usd: float | None = None,
    on_truncation: Literal["raise", "warn", "return"] = "raise",
    slice_fn: Callable[[frozenset[str]], str | None] | None = None,
) -> JudgePairEval:
    """Score any judge over a fixed candidate set against gold — the honest, spend-capped one-liner.

    A thin wrapper over :func:`evaluate_judge_on_candidates` for the common
    bring-your-own-data case: pass a module, its candidates, and gold pairs, and
    get back pair-level Precision/Recall/F1, the full PR curve, cost, and
    latency. Two things distinguish it from calling
    :func:`evaluate_judge_on_candidates` directly:

    - **Honest by construction.** With the default ``threshold=None`` the
      reported ``pair`` is graded at the grid's best-F1 threshold — but that
      argmax is fitted to the SAME ``gold_pairs`` the result reports F1
      against, so the number is optimistically biased (an upper bound, not a
      held-out estimate) — and a ``UserWarning`` says so on every such call.
      Pass ``threshold=<float>`` for an honest fixed cut instead: ``pair`` is
      graded ONCE at that threshold (no argmax), ``best_threshold`` is ``None``
      (setting it would be a lie), and ``graded_threshold`` names the cut used.
      ``pair.pr_curve`` stays populated in both modes (it is cheap, from the
      sweep, and a later PR needs it).
    - **Spend-capped by default, honestly.** Every call runs the judge through
      an internal :class:`BudgetedModuleRunner`, capped at ``budget_usd``
      (``None`` resolves to
      :data:`~langres.core.spend_cap.DEFAULT_BUDGET_USD`, the same default every
      ERModel uses). **This is not a hard ceiling**: the cap is enforced BETWEEN
      calls, so a single in-flight call can still push total spend past it by
      that call's own cost (a free judge — string/embedding, which never sets
      ``provenance["cost_usd"]`` — never approaches the cap regardless). When
      that happens, ``JudgePairEval.budget_exceeded`` is ``True`` and
      ``evaluate()`` always warns loudly naming the actual spend and the cap —
      it never raises solely for this, since the run already completed and the
      money is already spent. Separately, if the cap TRUNCATES the run (fewer
      pairs judged than supplied), ``on_truncation`` controls what happens:
      ``"raise"`` (default) raises :class:`EvaluationTruncatedError` carrying
      the partial result on ``.partial``; ``"warn"`` logs a ``UserWarning`` and
      returns the partial result; ``"return"`` returns it silently. A judge
      that itself produces fewer judgements than candidates (one call raised,
      or yielded nothing) is a DIFFERENT, non-spend truncation
      (``truncation_reason="judge_skips"``) and only ever warns — never raises,
      even under ``on_truncation="raise"`` — since one bad candidate must not
      blow up a run. The sole exception: **zero** judgements over a non-empty
      candidate set raises regardless of the reason (unless
      ``on_truncation="return"``). Nothing was graded, so a reported F1 of
      ``0.0`` would be indistinguishable from a healthy judge that matched
      nothing — the dishonest cell this policy exists to prevent. An EMPTY
      ``candidates`` sequence (zero pairs to grade at all — usually an upstream
      blocker that produced nothing) raises ``ValueError`` immediately, before
      any of the above.

    For a paid or compiled judge that needs a caller-supplied price, a
    caller-owned runner, custom cost accounting (e.g. cascade escalation
    diagnostics), or the raw judgements back, call
    :func:`evaluate_judge_on_candidates` directly — its ``runner`` /
    ``price_per_token_or_pair`` / ``cost_track_fn`` knobs stay there, not here.

    Args:
        module: The scorer to evaluate (any :class:`~langres.core.matcher.Matcher`).
        candidates: The fixed candidate pairs to judge (each an ``ERCandidate``).
            Must be non-empty.
        gold_pairs: True match pairs as order-independent ``frozenset`` pairs.
        grid: Score thresholds to sweep for the pair-level PR curve and (when
            ``threshold=None``) the best-F1 argmax. Defaults to
            :data:`DEFAULT_PAIR_GRID`, the fine ``0.05..0.95`` sweep. Every point
            must lie in ``[0.0, 1.0]``; ``0.0`` is allowed as the curve's
            predict-all anchor, but ``evaluate()`` warns if the argmax lands there.
        threshold: ``None`` (default) grades at the grid's best-F1 argmax
            (optimistically biased, warned about); a float grades ONCE at that
            fixed cut (honest, no warning). Must lie in ``(0.0, 1.0]`` — a cut of
            ``0.0`` would predict an abstention (``score=0.0``) a MATCH, and a
            cut above ``1.0`` is unreachable for a score in ``[0, 1]``.
        budget_usd: Spend ceiling for the internal runner, enforced BETWEEN
            calls (see above — NOT a hard ceiling on a single call's own cost);
            ``None`` (default) resolves to
            :data:`~langres.core.spend_cap.DEFAULT_BUDGET_USD`.
        on_truncation: What to do when the run is truncated — see above.
        slice_fn: Optional pair-key tagger forwarded to
            :func:`evaluate_judge_on_candidates`; when given, ``slices`` are
            graded at the SAME cut as ``pair`` (the fixed ``threshold`` in fixed
            mode, or the grid's best-F1 threshold in sweep mode).

    Returns:
        A :class:`JudgePairEval` — pair P/R/F1 at ``graded_threshold``, the PR
        curve, cost, latency, ``budget_exceeded``, and (when ``slice_fn`` is
        given) per-slice tracks.

    Raises:
        ValueError: If ``candidates`` is empty, if ``threshold`` falls outside
            ``(0.0, 1.0]``, or if ``grid`` is empty or holds a point outside
            ``[0.0, 1.0]``.
        EvaluationTruncatedError: If ``on_truncation="raise"`` (the default) and
            the internal spend cap truncated the run.
    """
    if not candidates:
        raise ValueError(
            "evaluate(): candidates is empty -- there is nothing to grade. "
            "precision/recall/F1 of 0.0 over zero candidates would be "
            "indistinguishable from 'the judge matched nothing', which is not "
            "what an empty candidate set means. This is almost always an "
            "upstream blocker producing zero candidates; inspect that before "
            "calling evaluate()."
        )

    if threshold is not None:
        _validate_cut("threshold", threshold)
    # Validate our own inputs up front, before the argmax warning fires -- a bad
    # grid should not be announced, it should be rejected. evaluate_judge_on_candidates
    # validates again for callers who reach it directly; _validated_grid is idempotent.
    grid = _validated_grid(grid)

    if threshold is None:
        warnings.warn(
            "evaluate() with threshold=None (the default) reports best_threshold "
            "chosen by argmax-F1 over `grid` -- fitted to the SAME gold_pairs this "
            "result's precision/recall/F1 are graded against. The reported F1 is "
            "therefore optimistically biased (an upper bound, not a held-out "
            "estimate). Pass threshold=<float> for an honest fixed cut.",
            UserWarning,
            stacklevel=2,
        )

    resolved_budget = effective_budget(budget_usd)
    runner = BudgetedModuleRunner(
        module,
        budget_usd=resolved_budget,
        budget_soft_usd=resolved_budget,
        worst_case_units_per_pair=1.0,
    )
    result, judgements = evaluate_judge_on_candidates(
        module,
        candidates,
        gold_pairs,
        grid,
        slice_fn=slice_fn,
        runner=runner,
        price_per_token_or_pair=_NEGLIGIBLE_WORST_CASE_PRICE,
    )

    if threshold is None and result.best_threshold == 0.0:
        # The argmax landed on the PR curve's predict-all anchor. Legal as a curve
        # point, useless as a decision rule: it calls EVERY pair a match, so recall
        # is 1.0 by construction and an abstention (score=0.0) is graded a YES.
        # A judge whose best F1 is here is not beating "say yes to everything".
        warnings.warn(
            "evaluate(): the best-F1 threshold is 0.0 -- the predict-all point, "
            "where every candidate (including an abstention at score=0.0) counts "
            "as a match and recall is trivially 1.0. This judge does not beat "
            "predicting every pair a match. Do not carry best_threshold=0.0 into "
            "production; drop 0.0 from `grid` to see the best non-trivial cut.",
            UserWarning,
            stacklevel=2,
        )

    if threshold is not None:
        gold_in_scope = _gold_in_scope(candidates, gold_pairs)
        graded = classify_pairs(judgements, gold_in_scope, threshold)
        fixed_slices = (
            _grade_slices(judgements, gold_in_scope, slice_fn, threshold)
            if slice_fn is not None
            else None
        )
        result = result.model_copy(
            update={
                "pair": PairTrack(
                    precision=graded.precision,
                    recall=graded.recall,
                    f1=graded.f1,
                    pr_curve=result.pair.pr_curve,
                ),
                "best_threshold": None,
                "graded_threshold": threshold,
                "slices": fixed_slices,
            }
        )

    # Always warn on a real breach, independent of on_truncation and of
    # whether the run was truncated (a last-pair breach still judges every
    # candidate) -- the money is spent either way, and the caller must not be
    # able to silence this the way they can silence a truncation warning.
    if result.budget_exceeded:
        warnings.warn(_budget_exceeded_message(result, resolved_budget), UserWarning, stacklevel=2)

    _apply_truncation_policy(result, on_truncation=on_truncation, budget_usd=resolved_budget)
    return result


class BudgetedModuleRunner:
    """Run any :class:`~langres.core.matcher.Matcher` under a spend cap.

    Wraps a module and scores candidate pairs one at a time, keeping the
    worst-case *projected* spend under ``budget_usd`` between calls, and
    stopping immediately once a call's *real*, measured cost pushes the running
    total past it. The same three-layer guarantee proven by
    :class:`~langres.curation.labelers.TeacherLabeler`, but as a clean,
    ``Matcher``-typed component returning :class:`PairwiseJudgement` (the teacher
    is welded to ``LLMMatcher`` and returns ``GoldPair``, so it cannot be reused
    directly):

    1. **Pre-flight cap.** Truncate the input to
       ``floor(budget_soft_usd / worst_case_per_pair)`` pairs, where
       ``worst_case_per_pair = worst_case_units_per_pair * price`` (``price`` is
       per token or per pair, set ``worst_case_units_per_pair=1`` for a flat
       per-pair price). A resolved ``price`` of ``$0`` makes the cap blind and
       raises :class:`BlindCostError` before any work.
    2. **Running tally + per-pair stop.** Spend is tallied from each judgement's
       ``provenance["cost_usd"]`` (falling back to ``provenance["llm_cost_usd"]``).
       Before scoring *each* pair, if the worst-case projected spend would cross
       ``budget_usd`` the run stops and returns what was scored so far.
    3. **Post-call breach detection.** The cap in point 2 projects on a
       worst-case *estimate*; the estimate can be wrong (or deliberately
       negligible, as :func:`evaluate` uses to avoid :class:`BlindCostError`
       while still measuring real cost). So immediately AFTER each call's real
       cost is added to the running total, the runner checks the total against
       ``budget_usd`` again and, if it now exceeds it, sets
       :attr:`budget_exceeded` and stops before starting another call. **This
       is not a hard ceiling**: a single in-flight call can still push total
       spend past ``budget_usd`` by that call's own cost — the cap is enforced
       *between* calls, never *within* one. :attr:`budget_exceeded` is how a
       caller learns a breach happened even when every candidate was still
       judged (see :attr:`~JudgePairEval.budget_exceeded` for the full
       semantics).
    4. **Per-call resilience.** Each pair is scored in its own ``forward`` call
       wrapped in ``try/except``; one failed call skips that pair and the loop
       continues, so a single error never discards already-paid results.

    Live run statistics are reset at the start of every :meth:`run` call and so
    describe only the most recent call: :attr:`total_spent_usd`,
    :attr:`labeled_count`, :attr:`skipped_count`, :attr:`dropped_by_cap_count`,
    :attr:`budget_stopped` (``True`` iff the pre-call projected-spend stop in
    point 2 fired -- the pre-flight cap in point 1 has no equivalent flag since
    ``dropped_by_cap_count > 0`` already says so), :attr:`budget_exceeded`
    (``True`` iff the post-call real-spend check in point 3 fired).

    Group-call atomicity (W1.0, E5): :meth:`run` scores exactly ONE candidate
    per ``module.forward()`` call (see point 3 above). A
    :class:`~langres.core.matcher.GroupwiseMatcher` given a single candidate
    derives a single, trivial, size-1 group from it -- so under this runner
    a group is NEVER split mid-call (there is never more than one pair per
    call to split), but a real multi-pair group is also never batched into
    one priced call: no cost amortization happens yet. See
    ``tests/benchmarks/test_benchmark.py::test_runner_never_splits_a_group_mid_call_but_also_never_batches_one``
    for the pinned-down behavior. Extending this runner (or adding a
    group-aware variant) to pre-flight and price whole groups atomically is
    deferred to W1.1, the branch that lands the first concrete
    ``GroupwiseMatcher`` (``SelectMatcher``) and can measure the real call-count
    reduction this is for.
    """

    def __init__(
        self,
        module: Matcher[Any],
        *,
        budget_usd: float = 20.0,
        budget_soft_usd: float = 15.0,
        worst_case_units_per_pair: float = 1.0,
    ) -> None:
        """Initialize the runner.

        Args:
            module: The wrapped scorer.
            budget_usd: Spend ceiling enforced BETWEEN calls (pre-call projected
                stop, point 2) and re-checked immediately AFTER each call's real
                cost lands (post-call breach, point 3) — not a hard ceiling: a
                single in-flight call can still push total spend past this by
                that call's own cost (see :attr:`budget_exceeded`).
            budget_soft_usd: Soft ceiling used to size the pre-flight cap, giving
                headroom below ``budget_usd``.
            worst_case_units_per_pair: Worst-case priced units (e.g. tokens) for
                one pair; ``1.0`` means ``price`` is a flat per-pair price.

        Raises:
            ValueError: If any budget/unit value is non-positive, or
                ``budget_soft_usd > budget_usd``.
        """
        if budget_usd <= 0.0 or budget_soft_usd <= 0.0:
            raise ValueError("budgets must be positive")
        if budget_soft_usd > budget_usd:
            raise ValueError("budget_soft_usd must not exceed budget_usd")
        if worst_case_units_per_pair <= 0.0:
            raise ValueError("worst_case_units_per_pair must be positive")

        self._module = module
        self.budget_usd = budget_usd
        self.budget_soft_usd = budget_soft_usd
        self.worst_case_units_per_pair = worst_case_units_per_pair

        self.total_spent_usd: float = 0.0
        self.labeled_count: int = 0
        self.skipped_count: int = 0
        self.dropped_by_cap_count: int = 0
        self.budget_stopped: bool = False
        self.budget_exceeded: bool = False

    def run(
        self,
        candidates: Sequence[Any],
        price_per_token_or_pair: float,
    ) -> list[PairwiseJudgement]:
        """Score candidates under the budget cap, returning the judgements made.

        Args:
            candidates: Candidate pairs to score (each scored via the module's
                ``forward``).
            price_per_token_or_pair: Price per worst-case unit (token, or pair
                when ``worst_case_units_per_pair == 1``). Must be ``> 0``.

        Returns:
            The judgements produced. May be fewer than the input: dropped by the
            pre-flight cap, skipped on a failed call, or truncated when the
            per-pair budget stop fires. Even when NOT fewer (every candidate was
            judged), check :attr:`budget_exceeded` — a breach on the last
            candidate still judges everything but still cost more than
            ``budget_usd``.

        Raises:
            BlindCostError: If the resolved worst-case per-pair price is ``$0``
                (a pre-flight blind cap), or if a wrapped module reports
                untrackable spend mid-run. In the mid-run case the judgements
                already produced (and paid for) are attached to the exception's
                ``partial`` so the caller can recover them.
        """
        self.total_spent_usd = 0.0
        self.labeled_count = 0
        self.skipped_count = 0
        self.dropped_by_cap_count = 0
        self.budget_stopped = False
        self.budget_exceeded = False

        worst_case_per_pair = self.worst_case_units_per_pair * price_per_token_or_pair
        if worst_case_per_pair <= 0.0:
            raise BlindCostError(
                f"resolved worst-case price is ${worst_case_per_pair:.6f}/pair; "
                "the budget cap would be blind"
            )

        capped = self._apply_preflight_cap(list(candidates), worst_case_per_pair)
        judgements: list[PairwiseJudgement] = []
        for candidate in capped:
            projected = self.total_spent_usd + worst_case_per_pair
            if projected > self.budget_usd:
                self.budget_stopped = True
                logger.info(
                    "Budget stop: spent=$%.4f + next worst-case $%.6f would exceed "
                    "budget $%.2f; returning %d judgements",
                    self.total_spent_usd,
                    worst_case_per_pair,
                    self.budget_usd,
                    len(judgements),
                )
                break
            try:
                judgement = self._score_one(candidate)
            except BlindCostError as exc:
                # Recover the already-paid judgements rather than discard them
                # (mirrors TeacherLabeler.label); the catcher sets ``partial``.
                exc.partial = judgements
                raise
            if judgement is not None:
                judgements.append(judgement)
            # Post-call breach: the pre-call check above projects on a
            # worst-case ESTIMATE, so it cannot catch a call whose REAL cost
            # (just tallied into total_spent_usd by _score_one) turns out to
            # exceed budget_usd on its own. Check again now, immediately, and
            # stop before starting another call -- this is the fix for the
            # "single $10 pair under a $1 cap sails through unnoticed" bug.
            if self.total_spent_usd > self.budget_usd:
                self.budget_exceeded = True
                logger.warning(
                    "Budget exceeded mid-call: spent=$%.4f > budget $%.2f after a "
                    "single call; stopping immediately (the cap is enforced "
                    "BETWEEN calls, not within one)",
                    self.total_spent_usd,
                    self.budget_usd,
                )
                break
        return judgements

    def _apply_preflight_cap(self, candidates: list[Any], worst_case_per_pair: float) -> list[Any]:
        """Truncate the input so even worst-case spend stays under the soft budget."""
        max_pairs = math.floor(self.budget_soft_usd / worst_case_per_pair)
        if len(candidates) > max_pairs:
            self.dropped_by_cap_count = len(candidates) - max_pairs
            logger.info(
                "Pre-flight cap: keeping %d of %d pairs (soft budget $%.2f, worst-case $%.6f/pair)",
                max_pairs,
                len(candidates),
                self.budget_soft_usd,
                worst_case_per_pair,
            )
            return candidates[:max_pairs]
        return candidates

    def _score_one(self, candidate: Any) -> PairwiseJudgement | None:
        """Score one pair, tally its spend, and return the judgement (or ``None``).

        Returns ``None`` (incrementing :attr:`skipped_count`) when the module call
        raises or yields nothing, so the caller's loop continues without losing
        already-paid results.
        """
        try:
            produced = list(self._module.forward(iter([candidate])))
        except BlindCostError:
            # Untrackable spend must abort the whole run, never be swallowed as a
            # skip — otherwise the budget cap would keep accruing unknowable cost
            # (mirrors TeacherLabeler, which re-raises BlindCostError).
            raise
        except Exception as exc:  # noqa: BLE001 — one bad call must not abort the run
            self.skipped_count += 1
            logger.warning("Matcher call failed for a candidate: %s; skipping", exc)
            return None

        if not produced:
            self.skipped_count += 1
            logger.warning("Matcher yielded no judgement for a candidate; skipping")
            return None

        judgement = produced[0]
        self.total_spent_usd += _judgement_cost(judgement)
        self.labeled_count += 1
        return judgement
