"""Benchmark harness: race resolution *methods* across *datasets* into a table.

The dataset-agnostic spine of the old ``langres.core.benchmark`` runner half:
:func:`run_method` / :func:`run_methods` load a :class:`~langres.data.benchmark.Benchmark`,
tune thresholds leakage-free, score both the pair-level and pipeline tracks, and
collect them in a :class:`BenchmarkTable`. The benchmark *spec* (the
:class:`~langres.data.benchmark.Benchmark` contract, :class:`~langres.data.benchmark.PairTrack`,
``gold_pairs_from_clusters``, ``complete_partition``) lives in
:mod:`langres.data.benchmark`; this module depends on it one-way. The method
registry (``langres.methods.make_resolver_factory``) is imported function-locally
in :func:`run_methods` to keep ``import langres.benchmarks.runner`` free of the
registry (and of ``langres.data``), exactly as the pre-split harness did.
"""

import logging
import math
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field

from langres.core.models import PairwiseJudgement
from langres.core.resolver import Resolver
from langres.core.usage import CostBasis, CostTrack, LLMUsage
from langres.data.benchmark import (
    Benchmark,
    PairTrack,
    RecordT,
    _Resolvable,
    complete_partition,
    gold_pairs_from_clusters,
)
from langres.metrics.metrics import (
    calculate_bcubed_metrics,
    calculate_pairwise_metrics,
    classify_pairs,
    pair_pr_curve,
)

logger = logging.getLogger(__name__)


class PipelineTrack(BaseModel):
    """Post-clustering pipeline scores against the closed-world truth partition.

    Attributes:
        bcubed_p: BCubed precision.
        bcubed_r: BCubed recall.
        bcubed_f1: BCubed F1.
        cluster_pairwise_f1: Pairwise F1 of the *completed* clustering (the
            transitive-closure view the pair track deliberately avoids).
        delta_above_floor: ``bcubed_f1 - sanity_floor_f1`` — value added over
            merging nothing. Negative means the method is worse than no merges.
        sanity_floor_f1: BCubed F1 of the all-singletons prediction.
    """

    bcubed_p: float
    bcubed_r: float
    bcubed_f1: float
    cluster_pairwise_f1: float
    delta_above_floor: float
    sanity_floor_f1: float


class LatencyTrack(BaseModel):
    """Wall-clock accounting for a method run.

    Attributes:
        seconds_per_pair: Mean seconds to score one candidate pair on the test
            split (block -> compare -> score -> cluster, amortized).
        throttle_seconds: Seconds spent sleeping on rate-limit throttles
            (API methods only); ``None`` when not throttled.
        embed_cache_hit_rate: Embedding cache hit rate in ``[0, 1]`` (embedding
            methods with a cache only); ``None`` otherwise.
    """

    seconds_per_pair: float
    throttle_seconds: float | None = None
    embed_cache_hit_rate: float | None = None


class MethodResult(BaseModel):
    """One method's result on one dataset/seed: both tracks, cost, and latency.

    Attributes:
        method: Method label (e.g. ``"weighted_average"``).
        dataset: Dataset label (e.g. ``"fodors_zagat"``).
        seed: Split seed used.
        threshold: Tuned pipeline (clusterer) threshold.
        pair: Pair-level track.
        pipeline: Post-clustering pipeline track.
        cost: Spend accounting.
        latency: Wall-clock accounting.
    """

    method: str
    dataset: str
    seed: int
    threshold: float
    pair: PairTrack
    pipeline: PipelineTrack
    cost: CostTrack = Field(default_factory=CostTrack)
    latency: LatencyTrack


#: Ranking metrics :meth:`BenchmarkTable.best` / :meth:`~BenchmarkTable.rank`
#: accept — the decision-relevant headline scores (higher is better for all
#: three), each a trivial accessor over an existing :class:`MethodResult` field.
_RANKABLE_METRICS: dict[str, Callable[["MethodResult"], float]] = {
    "pair_f1": lambda r: r.pair.f1,
    "bcubed_f1": lambda r: r.pipeline.bcubed_f1,
    "delta_above_floor": lambda r: r.pipeline.delta_above_floor,
}


def _rank_accessor(by: str) -> Callable[["MethodResult"], float]:
    """Resolve a ranking-metric name to its accessor, or raise on an unknown one."""
    try:
        return _RANKABLE_METRICS[by]
    except KeyError:
        raise ValueError(
            f"unknown ranking metric {by!r}; choose one of {sorted(_RANKABLE_METRICS)}"
        ) from None


class BenchmarkTable(BaseModel):
    """Collects :class:`MethodResult`s and renders a method×dataset×seed table.

    A thin, serializable accumulator: append results as runs complete, then call
    :meth:`to_markdown` for a compact headline view (BCubed F1, pair F1, cost,
    latency), or :meth:`best` / :meth:`rank` for a structured winner an agent
    driving experiments can read directly. Kept deliberately small — richer
    reporting (per-track breakdowns, plots) composes on top of the stored
    ``results`` rather than bloating this.
    """

    results: list[MethodResult] = Field(default_factory=list)

    def add(self, result: MethodResult) -> None:
        """Append one method result."""
        self.results.append(result)

    def rank(self, *, by: str = "pair_f1") -> list[MethodResult]:
        """Return the results sorted by ``by``, best (highest) first.

        The structured counterpart to :meth:`to_markdown`: instead of a rendered
        string, a list an experiment driver can index. The sort is stable, so ties
        keep insertion order. An empty table returns ``[]``.

        Args:
            by: A metric name in :data:`_RANKABLE_METRICS` (``"pair_f1"`` — the
                scorer-isolating default — ``"bcubed_f1"``, or
                ``"delta_above_floor"``).

        Returns:
            The stored results, highest ``by`` first.

        Raises:
            ValueError: If ``by`` is not a known ranking metric.
        """
        accessor = _rank_accessor(by)
        return sorted(self.results, key=accessor, reverse=True)

    def best(self, *, by: str = "pair_f1") -> MethodResult | None:
        """Return the single top result by ``by``, or ``None`` for an empty table.

        Args:
            by: A metric name in :data:`_RANKABLE_METRICS` (default ``"pair_f1"``).

        Returns:
            The highest-``by`` :class:`MethodResult` (first-added wins ties), or
            ``None`` when no results have been added.

        Raises:
            ValueError: If ``by`` is not a known ranking metric.
        """
        accessor = _rank_accessor(by)
        if not self.results:
            return None
        return max(self.results, key=accessor)

    def to_markdown(self) -> str:
        """Render the collected results as a Markdown table.

        Columns: method, dataset, seed, threshold, pipeline BCubed F1, pair F1,
        total USD, and seconds/pair. Rows are emitted in insertion order.

        Returns:
            A Markdown table string (header + one row per result). With no
            results, returns just the header row.
        """
        header = (
            "| method | dataset | seed | threshold | bcubed_f1 | pair_f1 "
            "| usd_total | s_per_pair |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- |"
        )
        rows = [
            f"| {r.method} | {r.dataset} | {r.seed} | {r.threshold:.2f} "
            f"| {r.pipeline.bcubed_f1:.4f} | {r.pair.f1:.4f} "
            f"| {r.cost.usd_total:.4f} | {r.latency.seconds_per_pair:.6f} |"
            for r in self.results
        ]
        return "\n".join([header, *rows])


if TYPE_CHECKING:
    # ``run_methods`` builds each method's factory via
    # ``langres.methods.make_resolver_factory``, which needs the *registry*
    # contract (``BlockingBenchmark``: ``schema`` + ``blocking_k`` +
    # ``build_blocker``) on top of the harness ``Benchmark`` contract. The two
    # cannot be a single runtime import here without closing the
    # ``core.benchmark -> methods -> core.benchmark`` cycle, so the combined
    # contract is expressed for the type checker only — the annotation is a
    # forward reference (quoted) and never evaluated at runtime.
    from langres.methods import BlockingBenchmark

    class _RunnableBenchmark(Benchmark[RecordT], BlockingBenchmark, Protocol):
        """A benchmark satisfying BOTH the harness and the registry contracts.

        The intersection ``run_methods`` requires — the same shape the two dataset
        conformers and the test fakes already expose. Mirrors the example's
        ``_RaceBenchmark`` so one benchmark can drive ``run_method`` (needs
        ``Benchmark``) and ``make_resolver_factory`` (needs ``BlockingBenchmark``).
        """


def _pipeline_track(
    predicted: list[set[str]],
    all_ids: list[str],
    truth_clusters: list[set[str]],
) -> PipelineTrack:
    """Score a predicted clustering into a :class:`PipelineTrack`.

    Shared by :func:`evaluate_resolver_bcubed` (which resolves first) and
    :func:`run_method` (which reuses an already-timed clustering), so the BCubed +
    floor + Δ computation lives in one place and cannot drift.
    """
    completed = complete_partition(predicted, all_ids)
    bcubed = calculate_bcubed_metrics(completed, truth_clusters)
    pairwise = calculate_pairwise_metrics(completed, truth_clusters)
    floor = calculate_bcubed_metrics([{rid} for rid in all_ids], truth_clusters)
    return PipelineTrack(
        bcubed_p=bcubed["precision"],
        bcubed_r=bcubed["recall"],
        bcubed_f1=bcubed["f1"],
        cluster_pairwise_f1=pairwise["f1"],
        delta_above_floor=bcubed["f1"] - floor["f1"],
        sanity_floor_f1=floor["f1"],
    )


def evaluate_resolver_bcubed(
    resolver: Resolver,
    test_records: Sequence[_Resolvable],
    test_truth_clusters: list[set[str]],
) -> PipelineTrack:
    """Score a built resolver on a held-out split (pipeline track).

    Runs ``resolver.resolve`` on the records, completes the predicted partition
    with singletons, and scores BCubed + pairwise against the closed-world
    ``test_truth_clusters``, plus the all-singletons sanity floor and the Δ above
    it. Dataset-agnostic: no cross-source / blocking diagnostics (those belong in
    a dataset-specific wrapper).

    Args:
        resolver: A built resolver (its index, if any, is built by ``resolve``).
        test_records: Held-out records (each exposing ``id`` and ``model_dump``).
        test_truth_clusters: Closed-world truth partition for the test split.

    Returns:
        A :class:`PipelineTrack`.
    """
    record_dicts = [r.model_dump() for r in test_records]
    all_ids = [r.id for r in test_records]
    predicted = resolver.resolve(record_dicts)
    return _pipeline_track(predicted, all_ids, test_truth_clusters)


def tune_threshold_on_train(
    resolver_factory: Callable[[float], Resolver],
    train_records: Sequence[_Resolvable],
    train_clusters: list[set[str]],
    *,
    thresholds: Sequence[float],
) -> float:
    """Select the clusterer threshold maximizing train BCubed F1 (no leakage).

    Builds a fresh resolver per candidate threshold via ``resolver_factory``,
    scores it on the TRAIN split, and returns the best threshold. The test split
    is never touched. Ties keep the first (lowest-index) threshold in order.

    Args:
        resolver_factory: ``threshold -> Resolver`` (a dataset wrapper injects its
            own builder, e.g. ``build_restaurant_resolver``).
        train_records: Training records.
        train_clusters: Closed-world truth partition for the train split.
        thresholds: Candidate clusterer thresholds to sweep.

    Returns:
        The best-performing threshold by train BCubed F1.

    Raises:
        ValueError: If ``thresholds`` is empty.
    """
    if not thresholds:
        raise ValueError("thresholds is empty; nothing to tune over")

    best_threshold = thresholds[0]
    best_f1 = -1.0
    for threshold in thresholds:
        resolver = resolver_factory(threshold)
        track = evaluate_resolver_bcubed(resolver, train_records, train_clusters)
        logger.info("threshold=%.2f -> train BCubed F1=%.4f", threshold, track.bcubed_f1)
        if track.bcubed_f1 > best_f1:
            best_f1 = track.bcubed_f1
            best_threshold = threshold
    return best_threshold


#: Provenance keys that carry a judgement's USD cost, in fallback order.
#: :func:`_judgement_cost` sums whichever is present; :func:`_judgement_cost_basis`
#: uses the SAME set to decide whether a judgement has a cost concept at all --
#: sharing one constant means the two can never disagree about what counts as
#: "this judgement has a cost" (previously `_judgement_cost_basis` only checked
#: ``"cost_usd"``, so real spend written under ``"llm_cost_usd"`` -- the key
#: ``CascadeChainMatcher`` writes -- was misclassified ``cost_basis="none"``).
def _validate_cut(label: str, value: float) -> None:
    """Reject a fixed grading cut outside ``(0.0, 1.0]``.

    ``classify_pairs`` predicts a match iff ``score >= cut``. A cut of ``0.0``
    (or below) therefore predicts EVERY pair a match -- including an abstention,
    which both ``LLMMatcher`` and ``DSPyMatcher`` emit as ``score=0.0``. The judge's
    "I could not decide" would silently become "yes, a match". A cut above
    ``1.0`` is unreachable for the ``ge=0.0, le=1.0`` score field: nothing ever
    matches, and F1 is a structural ``0.0`` rather than a measurement.

    This is the rule for a cut a caller *grades at* and may carry into
    production. It is deliberately STRICTER than :func:`_validate_grid_point` --
    a swept curve may legitimately anchor at ``0.0`` (see there).
    """
    if not math.isfinite(value) or not (0.0 < value <= 1.0):
        raise ValueError(
            f"{label} must be in (0.0, 1.0], got {value!r}. A cut of "
            "0.0 or less predicts every pair a match -- including an abstention "
            "(score=0.0), turning a judge's parse failure into a confident YES. "
            "A cut above 1.0 can never be reached by a score in [0, 1]."
        )


def _validate_grid_point(value: float) -> None:
    """Reject a swept threshold outside ``[0.0, 1.0]``.

    Looser than :func:`_validate_cut` by exactly one point: ``0.0`` is allowed.
    A PR curve's left anchor *is* the predict-all point (recall ``1.0``,
    precision equal to prevalence), and a full ``0.00..1.00`` sweep is a normal,
    honest way to draw one -- ``tests/data/test_wdc_computers.py`` does exactly
    that. Banning ``0.0`` here would outlaw a legitimate ranking-judge sweep in
    order to defend against an abstaining judge's convention, which is the
    layering mistake this module otherwise avoids (the generic pair primitives
    must not learn an LLM-specific dialect).

    What ``0.0`` must never become is a *reported* cut a caller then grades or
    ships at -- that is :func:`_validate_cut`'s job, and :func:`evaluate` warns
    when the argmax lands there.
    """
    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        raise ValueError(
            f"grid point must be in [0.0, 1.0], got {value!r}. A score is "
            "constrained to [0, 1], so a threshold outside it can never be a "
            "meaningful point on a PR curve."
        )


def _validated_grid(grid: Sequence[float]) -> list[float]:
    """Materialise ``grid`` and reject an empty or out-of-range one.

    Materialising first is load-bearing: ``grid`` is typed ``Sequence[float]``,
    but an untyped caller can hand over a generator, and validating it by
    iteration would consume it -- leaving the sweep below an empty grid. Same
    hazard :meth:`~langres.core.resolver.Resolver.candidates` exists to close.

    An empty grid is its own error rather than an opaque
    ``max() iterable argument is empty`` from deep inside the sweep.
    """
    points = list(grid)
    if not points:
        raise ValueError("grid is empty -- there are no thresholds to sweep")
    for point in points:
        _validate_grid_point(point)
    return points


_COST_KEYS: tuple[str, ...] = ("cost_usd", "llm_cost_usd")


def _judgement_cost(judgement: PairwiseJudgement) -> float:
    """Measured USD cost of one judgement from its provenance.

    Reads :data:`_COST_KEYS` in order (``"cost_usd"`` first, falling back to
    ``"llm_cost_usd"``, the key ``CascadeChainMatcher`` writes). Zero-spend judges set
    neither, so this returns ``0.0`` for them.
    """
    prov = judgement.provenance
    for key in _COST_KEYS:
        if key in prov:
            return float(prov[key])
    return 0.0


def _judgement_usage(judgement: PairwiseJudgement) -> LLMUsage:
    """The typed usage vector from one judgement's provenance, or all-zeros.

    Reads ``provenance["usage"]`` (the dict ``LLMUsage.model_dump()`` writes --
    see ``llm_judge.py`` / ``dspy_judge.py``). Absent (string/embedding judges
    never set it) or malformed (a foreign/corrupt payload) both fall back to an
    all-zero vector -- usage capture is observability, it must never make
    evaluation flake (mirrors ``usage.py``'s own ``_as_int`` philosophy).
    """
    raw = judgement.provenance.get("usage")
    if not isinstance(raw, dict):
        return LLMUsage()
    try:
        return LLMUsage(**raw)
    except (TypeError, ValueError):
        return LLMUsage()


def _sum_usage(usages: list[LLMUsage]) -> LLMUsage:
    """Sum token counts across ``usages`` into one vector.

    ``model`` / ``provider`` are left at their type defaults (``""`` / ``None``):
    a sum across judgements that may span several models has no single
    meaningful value for either.
    """
    return LLMUsage(
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        cache_read_input_tokens=sum(u.cache_read_input_tokens for u in usages),
        cache_creation_input_tokens=sum(u.cache_creation_input_tokens for u in usages),
        reasoning_tokens=sum(u.reasoning_tokens for u in usages),
    )


def _judgement_cost_basis(judgement: PairwiseJudgement) -> CostBasis:
    """Classify how ONE judgement's cost was determined (never returns ``"mixed"``).

    Checked in order: ``cost_untracked`` (DSPyMatcher's billed-but-unparseable
    call) wins regardless of what else is set; then none of :data:`_COST_KEYS`
    present means the judge has no cost concept at all (string/embedding); then
    ``cost_is_real`` (LLMMatcher's real-OpenRouter-billing vs litellm-estimate
    flag) distinguishes real from estimated -- a judge that sets a cost key
    without ``cost_is_real`` (DSPyMatcher's normal, non-error path: tokens times a
    pinned price) is also an estimate, never a real billed amount.
    """
    prov = judgement.provenance
    if prov.get("cost_untracked"):
        return "untracked"
    if not any(key in prov for key in _COST_KEYS):
        return "none"
    if prov.get("cost_is_real") is True:
        return "real"
    return "estimated"


def _combined_cost_basis(judgements: list[PairwiseJudgement]) -> CostBasis:
    """Combine every judgement's :func:`_judgement_cost_basis` into one label.

    ``"none"`` for an empty list; a single shared basis passes through; two or
    more DIFFERENT bases collapse to ``"mixed"`` -- a run that blends, say, real
    OpenRouter spend with a zero-cost local judge is neither wholly real nor
    wholly untracked, and pretending otherwise would misrepresent it.
    """
    if not judgements:
        return "none"
    bases = {_judgement_cost_basis(j) for j in judgements}
    if len(bases) == 1:
        return bases.pop()
    return "mixed"


def _cost_track(judgements: list[PairwiseJudgement]) -> CostTrack:
    """Aggregate per-judgement spend into a :class:`CostTrack`.

    ``escalation_rate`` / ``llm_calls_per_candidate`` stay ``None`` here: they are
    cascade-specific and not derivable from a flat judgement list.
    """
    n_pairs = len(judgements)
    usd_total = sum(_judgement_cost(j) for j in judgements)
    per_pair = usd_total / n_pairs if n_pairs > 0 else 0.0
    return CostTrack(
        usd_total=usd_total,
        usd_per_1k_pairs=per_pair * 1_000.0,
        est_usd_per_100k=per_pair * 100_000.0,
        usage=_sum_usage([_judgement_usage(j) for j in judgements]),
        cost_basis=_combined_cost_basis(judgements),
    )


def run_method(
    benchmark: Benchmark[RecordT],
    resolver_factory: Callable[[float], Resolver],
    *,
    seed: int,
    budget: float | None = None,
) -> MethodResult:
    """Run one resolver factory against one benchmark, computing both tracks.

    Pipeline:

    1. ``benchmark.load`` + ``benchmark.split(seed=seed)`` (leakage-free).
    2. Tune the clusterer threshold on TRAIN BCubed F1.
    3. Tune the pair-level threshold on TRAIN pair-level F1 (independent of the
       clusterer threshold — it classifies raw scores).
    4. Score TEST once: predict judgements (timed), then cluster.
    5. Compute the pipeline track (BCubed + floor) and the pair track (test
       judgements at the tuned pair threshold, plus the full PR curve), and the
       cost + latency tracks.

    Args:
        benchmark: A :class:`Benchmark` conformer.
        resolver_factory: ``threshold -> Resolver`` for the method under test.
        seed: Split seed.
        budget: Optional hard ceiling on measured spend; if set and the run's
            ``usd_total`` exceeds it, raises ``ValueError``. Zero-spend methods
            (W1) pass ``budget=None`` (no spend to bound) or ``0.0`` to assert
            they truly spent nothing. Per-pair *enforcement* of a budget is the
            job of :class:`BudgetedModuleRunner`; this is a post-run guard.

    Returns:
        A populated :class:`MethodResult`.

    Raises:
        ValueError: If ``budget`` is set and measured ``usd_total`` exceeds it.
    """
    corpus, gold_clusters, _gold_pairs = benchmark.load()
    train_records, test_records, train_clusters, test_clusters = benchmark.split(
        corpus, gold_clusters, seed=seed
    )

    # A dataset conformer supplies this grid; hold it to the same invariant as a
    # caller-supplied one (every bundled benchmark's grid starts at 0.3).
    grid = _validated_grid(benchmark.threshold_grid)

    # (2) Pipeline threshold on TRAIN.
    threshold = tune_threshold_on_train(
        resolver_factory, train_records, train_clusters, thresholds=grid
    )

    # (3) Pair threshold on TRAIN. The clusterer threshold is irrelevant to the
    # raw judgement scores, so any factory threshold yields the same judgements.
    train_gold_pairs = gold_pairs_from_clusters(train_clusters)
    train_judgements = resolver_factory(threshold).predict([r.model_dump() for r in train_records])
    train_curve = pair_pr_curve(train_judgements, train_gold_pairs, grid)
    best_pair = max(train_curve, key=lambda m: m.f1)

    # (4) Score TEST once (timed): judgements -> clusters.
    resolver = resolver_factory(threshold)
    test_dicts = [r.model_dump() for r in test_records]
    start = time.perf_counter()
    test_judgements = resolver.predict(test_dicts)
    predicted = resolver.clusterer.cluster(iter(test_judgements))
    elapsed = time.perf_counter() - start

    # (5a) Pipeline track (reuses the already-timed clustering).
    all_ids = [r.id for r in test_records]
    pipeline = _pipeline_track(predicted, all_ids, test_clusters)

    # (5b) Pair track on TEST at the train-tuned pair threshold.
    test_gold_pairs = gold_pairs_from_clusters(test_clusters)
    test_curve = pair_pr_curve(test_judgements, test_gold_pairs, grid)
    test_pair = classify_pairs(test_judgements, test_gold_pairs, best_pair.threshold)
    pair = PairTrack(
        precision=test_pair.precision,
        recall=test_pair.recall,
        f1=test_pair.f1,
        pr_curve=test_curve,
    )

    # (5c) Cost + latency.
    cost = _cost_track(test_judgements)
    n_pairs = len(test_judgements)
    latency = LatencyTrack(seconds_per_pair=elapsed / n_pairs if n_pairs > 0 else 0.0)

    if budget is not None and cost.usd_total > budget:
        raise ValueError(f"run_method spent ${cost.usd_total:.4f}, exceeding budget ${budget:.4f}")

    return MethodResult(
        method=getattr(resolver.module, "type_name", type(resolver.module).__name__),
        dataset=benchmark.name,
        seed=seed,
        threshold=threshold,
        pair=pair,
        pipeline=pipeline,
        cost=cost,
        latency=latency,
    )


def run_methods(
    benchmark: "_RunnableBenchmark[RecordT]",
    methods: Sequence[str],
    *,
    seed: int = 0,
    budget: float | None = None,
    **factory_kwargs: Any,
) -> BenchmarkTable:
    """Race several named methods on one benchmark into a :class:`BenchmarkTable`.

    The library-level experiment facade: the double loop
    ``examples/research/m3_zero_spend_race.py`` hand-coded (``for method: build factory ->
    run_method -> table.add``) becomes one call. For each name it builds a
    resolver factory via
    :func:`~langres.methods.make_resolver_factory(method, benchmark,
    **factory_kwargs)` and runs it through the existing :func:`run_method`,
    appending each :class:`MethodResult` in ``methods`` order. A thin orchestration
    wrapper: it owns no evaluation logic of its own, so both tracks, cost, and
    latency stay defined in exactly one place (``run_method``).

    ``make_resolver_factory`` lives in :mod:`langres.methods`, which imports
    ``CostTrack`` / ``_cost_track`` from *this* module. Importing it at module
    scope would close a ``core.benchmark -> methods -> core.benchmark`` cycle, so
    it is imported lazily inside the call. ``import langres.core.benchmark`` thus
    stays free of the method registry (and of ``langres.data``), preserving the
    harness's dataset-agnostic import graph; the method registry is pulled in only
    when an experiment is actually run.

    Args:
        benchmark: A conformer of BOTH the harness :class:`Benchmark` contract and
            the registry ``BlockingBenchmark`` contract (``schema`` +
            ``blocking_k`` + ``build_blocker``). The two dataset conformers and the
            test fakes already satisfy this intersection.
        methods: Method names to race (e.g. ``("rapidfuzz", "embedding_cosine")``);
            each must be a name :func:`~langres.methods.make_resolver_factory`
            accepts (raises ``ValueError`` otherwise).
        seed: Split seed shared by every method (identical leakage-free split).
        budget: Per-method spend ceiling forwarded to :func:`run_method`. ``0.0``
            asserts genuine zero spend (raises if any method is charged); ``None``
            bounds nothing.
        **factory_kwargs: Forwarded verbatim to ``make_resolver_factory`` (e.g.
            ``llm_client``, ``llm_model``) — ignored by the zero-spend methods.

    Returns:
        A :class:`BenchmarkTable` with one :class:`MethodResult` per method (in
        ``methods`` order).
    """
    from langres.methods import make_resolver_factory

    table = BenchmarkTable()
    for method in methods:
        factory = make_resolver_factory(method, benchmark, **factory_kwargs)
        result = run_method(benchmark, factory, seed=seed, budget=budget)
        # ``run_method`` stamps ``MethodResult.method`` from the module's
        # ``type_name`` (e.g. ``weighted_average_judge``), but callers race by the
        # *registry* key (``weighted_average``) and expect ``best().method`` to be
        # a name ``make_resolver_factory`` accepts. Overwrite it with the requested
        # key so the winner is directly re-runnable.
        table.add(result.model_copy(update={"method": method}))
    return table
