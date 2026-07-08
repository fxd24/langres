"""Dataset-agnostic entity-resolution benchmark harness (M3 Wave 1).

This module is the reusable spine for benchmarking resolution *methods* across
*datasets* with a single, honest methodology. It is deliberately free of any
domain or dataset import (no ``langres.data``): a dataset participates by
conforming to the :class:`Benchmark` protocol, and a method participates as a
``resolver_factory: Callable[[float], Resolver]``. Keeping the harness in
``langres.core`` and depending only on a *factory* (never on
``build_restaurant_resolver`` or any concrete loader) is what avoids the
``core -> data.er_benchmarks -> core`` import cycle.

Two evaluation tracks, computed for every method, are the core methodology fix:

- **Pair-level (pre-clustering).** :func:`~langres.core.metrics.classify_pairs`
  scores each candidate judgement against the gold pairs *before* clustering.
  Ranking judges only by post-clustering pairwise F1 is biased: transitive
  closure lets one false-positive edge chain-merge and tank precision, unfairly
  punishing high-recall judges. The pair track isolates the scorer.
- **Pipeline (post-clustering).** The familiar block -> judge -> cluster ->
  BCubed flow, plus the all-singletons sanity floor and the Δ above it.

The harness consumes a *full* :class:`~langres.core.resolver.Resolver` factory
rather than a raw ``Module`` factory on purpose: comparison-aware judges (e.g.
``WeightedAverageJudge``) require a ``Comparator`` upstream, so the only uniform
contract across methods is a complete resolver.
"""

import logging
import math
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, Field

from langres.core.metrics import (
    PairMetrics,
    calculate_bcubed_metrics,
    calculate_pairwise_metrics,
    classify_pairs,
    pair_pr_curve,
    pairs_from_clusters,
)
from langres.core.models import PairwiseJudgement
from langres.core.module import Module
from langres.core.resolver import Resolver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared errors and partition helpers
# ---------------------------------------------------------------------------


class BlindCostError(RuntimeError):
    """Raised when a budgeted runner cannot observe the cost of its work.

    A budget cap is only safe while spend can be measured. Two situations make
    the cap *blind* and so abort the run rather than risk unbounded spend:

    - The resolved per-pair worst-case price is ``$0`` (e.g. a price of zero was
      passed), so the pre-flight cap ``floor(budget / 0)`` is unbounded.
    - (For :class:`~langres.bootstrap.labelers.TeacherLabeler`) a judgement
      reports neither token counts nor a cost, so the running tally is untrusted.

    :attr:`partial` carries any results already produced (and paid for) before a
    *mid-loop* abort, so a caller can recover them rather than discard paid work.
    It is set by the catcher immediately before re-raising (e.g.
    :meth:`~langres.bootstrap.labelers.TeacherLabeler.label`), not at the raise
    site; for a *pre-flight* raise (``BudgetedModuleRunner`` rejecting a ``$0``
    price before any work) it stays empty. Typed ``list[Any]`` so the same error
    serves both the bootstrap teacher (``GoldPair`` results) and the core runner
    (``PairwiseJudgement`` results) without coupling ``core`` to ``bootstrap``.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        # Populated by the catcher immediately before re-raising, not at raise time.
        self.partial: list[Any] = []


def complete_partition(
    predicted_clusters: list[set[str]], all_ids: Sequence[str]
) -> list[set[str]]:
    """Complete a predicted clustering into a full partition over ``all_ids``.

    The :class:`~langres.core.clusterer.Clusterer` drops singletons, so a record
    that was never merged is simply absent from ``predicted_clusters``. BCubed
    must average over *every* item, so this appends a singleton ``{id}`` for each
    id not already in a predicted cluster. Even with the partition-safe metric
    fix, completing the partition is required so BCubed *precision* averages over
    all items rather than only the merged ones.

    Args:
        predicted_clusters: Multi-record clusters from ``Resolver.resolve``.
        all_ids: Every record id in the split (e.g. ``[r.id for r in records]``).

    Returns:
        ``predicted_clusters`` followed by one singleton per uncovered id (in
        ``all_ids`` order, so the result is deterministic).
    """
    clustered = {rid for cluster in predicted_clusters for rid in cluster}
    completed = list(predicted_clusters)
    completed.extend({rid} for rid in all_ids if rid not in clustered)
    return completed


def gold_pairs_from_clusters(clusters: list[set[str]]) -> set[frozenset[str]]:
    """Derive the order-independent gold match pairs from a cluster partition.

    Every within-cluster pair is a true match; singletons contribute none. Used
    to build the pair-level ground truth for one split (leakage-free, since the
    pairs come only from that split's clusters).

    Args:
        clusters: Gold clusters for one split (match sets + singletons).

    Returns:
        The set of true match pairs as ``frozenset`` pairs.
    """
    return {frozenset(pair) for pair in pairs_from_clusters(clusters)}


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class PairTrack(BaseModel):
    """Pair-level (pre-clustering) scores at the tuned pair threshold.

    Attributes:
        precision: Pair-level precision on the test split.
        recall: Pair-level recall on the test split.
        f1: Pair-level F1 on the test split.
        pr_curve: Optional precision/recall/F1 across the threshold grid (test
            split), for plotting the trade-off. ``None`` when not requested.
    """

    precision: float
    recall: float
    f1: float
    pr_curve: list[PairMetrics] | None = None


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


class CostTrack(BaseModel):
    """Spend accounting for a method run. Zero-spend methods leave the optionals empty.

    Attributes:
        usd_total: Total measured spend across all judged test pairs.
        usd_per_1k_pairs: Spend normalized per 1k candidate pairs.
        est_usd_per_100k: Linear extrapolation to 100k pairs.
        escalation_rate: Fraction of pairs escalated to the expensive stage
            (cascade methods only); ``None`` for single-stage methods.
        llm_calls_per_candidate: Mean LLM calls per candidate (cascade methods
            only); ``None`` for zero-LLM methods.
    """

    usd_total: float = 0.0
    usd_per_1k_pairs: float = 0.0
    est_usd_per_100k: float = 0.0
    escalation_rate: float | None = None
    llm_calls_per_candidate: float | None = None


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


# ---------------------------------------------------------------------------
# Benchmark protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class _Resolvable(Protocol):
    """Minimal record contract the harness needs: an id and a dict dump."""

    id: str

    def model_dump(self) -> dict[str, Any]:
        """Return a JSON-able dict of the record (the resolver's raw input)."""
        ...  # pragma: no cover


RecordT = TypeVar("RecordT", bound=_Resolvable)
"""The dataset's record type — any Pydantic schema exposing ``id`` + ``model_dump``."""


@runtime_checkable
class Benchmark(Protocol[RecordT]):
    """A dataset adapter the harness can run any resolver factory against.

    Conformers expose a stable ``name``, a ``load`` returning the corpus plus the
    closed-world gold partition and gold pairs, a leakage-free ``split``, and the
    ``threshold_grid`` to tune over. Generic over the dataset's record type so a
    conformer (e.g. the Fodors-Zagat adapter) keeps its concrete schema typing. A
    dataset's own schema and blocking config live behind its ``resolver_factory``
    (passed separately to :func:`run_method`), so this protocol stays free of any
    domain type.
    """

    name: str
    threshold_grid: tuple[float, ...]

    def load(self) -> tuple[list[RecordT], list[set[str]], set[frozenset[str]]]:
        """Return ``(corpus, gold_clusters, gold_pairs)`` for the full dataset."""
        ...  # pragma: no cover

    def split(
        self,
        corpus: list[RecordT],
        gold_clusters: list[set[str]],
        *,
        seed: int,
    ) -> tuple[list[RecordT], list[RecordT], list[set[str]], list[set[str]]]:
        """Split into ``(train_records, test_records, train_clusters, test_clusters)``."""
        ...  # pragma: no cover


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


# ---------------------------------------------------------------------------
# Generic evaluation primitives (resolver-factory based, no dataset imports)
# ---------------------------------------------------------------------------


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


def _judgement_cost(judgement: PairwiseJudgement) -> float:
    """Measured USD cost of one judgement from its provenance.

    Reads ``provenance["cost_usd"]`` first, falling back to
    ``provenance["llm_cost_usd"]`` (the key ``CascadeModule`` writes). Zero-spend
    judges set neither, so this returns ``0.0`` for them.
    """
    prov = judgement.provenance
    if "cost_usd" in prov:
        return float(prov["cost_usd"])
    if "llm_cost_usd" in prov:
        return float(prov["llm_cost_usd"])
    return 0.0


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

    grid = benchmark.threshold_grid

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


# ---------------------------------------------------------------------------
# Direct pair-level judge evaluation (no blocking, fixed candidate set)
# ---------------------------------------------------------------------------


class JudgePairEval(BaseModel):
    """Pair-level evaluation of one judge over a *given* candidate set.

    Unlike :func:`run_method` (which loads a benchmark, blocks, tunes a clusterer,
    and clusters), this is the scorer-isolating eval used when the candidate pairs
    are fixed up front — e.g. a literature train/valid/test pair split, or a single
    dataset's blocked band — so the number is directly comparable to pairwise-F1
    SOTA without any blocking-recall ceiling or clustering amplification.

    Attributes:
        pair: Pair-level P/R/F1 at the best-F1 threshold over the grid, plus the
            full PR curve.
        cost: Spend accounting over the judgements actually produced.
        latency: Wall-clock per judged pair.
        n_candidates: Candidate pairs handed to the judge.
        n_judged: Judgements actually produced (``< n_candidates`` when a budget
            runner truncates or a call is skipped).
        best_threshold: The grid threshold maximizing pair-level F1.
        truncated: ``True`` when fewer pairs were judged than supplied (a budget
            cap fired or calls were skipped) — the cell is partial.
        slices: Optional per-slice pair tracks, each graded at the SAME fixed
            ``best_threshold`` as ``pair`` (never a per-slice argmax). Populated
            only when :func:`evaluate_judge_on_candidates` is given a ``slice_fn``;
            ``None`` otherwise. This is the honest seen -> unseen view: one global
            cut, reported across data slices, so a degradation cannot be tuned
            away.
    """

    pair: PairTrack
    cost: CostTrack
    latency: LatencyTrack
    n_candidates: int
    n_judged: int
    best_threshold: float
    truncated: bool
    slices: dict[str, PairTrack] | None = None


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


def evaluate_judge_on_candidates(
    module: Module[Any],
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
        module: The scorer to evaluate (any :class:`~langres.core.module.Module`).
        candidates: The fixed candidate pairs to judge (each an ``ERCandidate``).
        gold_pairs: True match pairs as order-independent ``frozenset`` pairs.
        grid: Score thresholds to sweep for the pair-level PR curve.
        slice_fn: Optional tagger mapping a pair key (``frozenset({left_id,
            right_id})``) to a slice tag, or ``None`` to exclude the pair. When
            given, the judged pairs and the in-scope gold are grouped by tag and
            each slice is graded at the ONE global ``best_threshold`` (never a
            per-slice argmax), populating ``JudgePairEval.slices``. When ``None``
            (default), ``slices`` stays ``None`` and the result is unchanged.
        runner: Optional budget runner. When given, the judge runs through it (its
            ``module`` must be ``module``) so spend is hard-capped; the run may
            therefore judge fewer pairs than supplied. When ``None``, the judge is
            run directly (zero-spend path).
        price_per_token_or_pair: Worst-case price passed to ``runner.run`` (ignored
            without a runner). Must be ``> 0`` when a runner is given.
        cost_track_fn: Aggregator from judgements to a :class:`CostTrack`. Defaults
            to the flat :func:`_cost_track`; pass
            :func:`~langres.methods.cascade_cost_track` for cascade escalation
            diagnostics.

    Returns:
        ``(JudgePairEval, judgements)`` — the graded summary plus the raw
        judgements (kept in-process for error-map analysis; not part of the
        summary so a persisted result stays small).
    """
    start = time.perf_counter()
    if runner is not None:
        judgements = runner.run(candidates, price_per_token_or_pair)
    else:
        judgements = list(module.forward(iter(candidates)))
    elapsed = time.perf_counter() - start

    # Grade the judge only on pairs it was actually given. When ``candidates`` is a
    # subsample of a larger fixed-pair set, gold pairs whose candidate was not
    # sampled are blocking-style misses, not judge errors — counting them would cap
    # recall artificially (e.g. a 600-pair subsample holding 61 of 234 gold pairs
    # caps recall at 61/234≈0.26 for *every* method). Restricting gold to
    # candidate-realizable pairs keeps this a pure judge metric, and is a no-op when
    # the candidates already cover all of ``gold_pairs``.
    candidate_pairs = {frozenset((cand.left.id, cand.right.id)) for cand in candidates}
    gold_in_scope = gold_pairs & candidate_pairs
    curve = pair_pr_curve(judgements, gold_in_scope, grid)
    best = max(curve, key=lambda m: m.f1)
    pair = PairTrack(precision=best.precision, recall=best.recall, f1=best.f1, pr_curve=curve)

    # Sliced view (honest): grade every slice at the SAME global best.threshold.
    slices = (
        _grade_slices(judgements, gold_in_scope, slice_fn, best.threshold)
        if slice_fn is not None
        else None
    )

    n_judged = len(judgements)
    latency = LatencyTrack(seconds_per_pair=elapsed / n_judged if n_judged > 0 else 0.0)
    result = JudgePairEval(
        pair=pair,
        cost=cost_track_fn(judgements),
        latency=latency,
        n_candidates=len(candidates),
        n_judged=n_judged,
        best_threshold=best.threshold,
        truncated=n_judged < len(candidates),
        slices=slices,
    )
    return result, judgements


# ---------------------------------------------------------------------------
# Budgeted module runner
# ---------------------------------------------------------------------------


class BudgetedModuleRunner:
    """Run any :class:`~langres.core.module.Module` under a hard spend cap.

    Wraps a module and scores candidate pairs one at a time, never letting the
    worst-case spend cross ``budget_usd``. The same three-layer guarantee proven
    by :class:`~langres.bootstrap.labelers.TeacherLabeler`, but as a clean,
    ``Module``-typed component returning :class:`PairwiseJudgement` (the teacher
    is welded to ``LLMJudge`` and returns ``GoldPair``, so it cannot be reused
    directly):

    1. **Pre-flight hard cap.** Truncate the input to
       ``floor(budget_soft_usd / worst_case_per_pair)`` pairs, where
       ``worst_case_per_pair = worst_case_units_per_pair * price`` (``price`` is
       per token or per pair, set ``worst_case_units_per_pair=1`` for a flat
       per-pair price). A resolved ``price`` of ``$0`` makes the cap blind and
       raises :class:`BlindCostError` before any work.
    2. **Running tally + per-pair stop.** Spend is tallied from each judgement's
       ``provenance["cost_usd"]`` (falling back to ``provenance["llm_cost_usd"]``).
       Before scoring *each* pair, if the worst-case projected spend would cross
       ``budget_usd`` the run stops and returns what was scored so far.
    3. **Per-call resilience.** Each pair is scored in its own ``forward`` call
       wrapped in ``try/except``; one failed call skips that pair and the loop
       continues, so a single error never discards already-paid results.

    Live run statistics are reset at the start of every :meth:`run` call and so
    describe only the most recent call: :attr:`total_spent_usd`,
    :attr:`labeled_count`, :attr:`skipped_count`, :attr:`dropped_by_cap_count`.

    Group-call atomicity (W1.0, E5): :meth:`run` scores exactly ONE candidate
    per ``module.forward()`` call (see point 3 above). A
    :class:`~langres.core.module.GroupwiseModule` given a single candidate
    derives a single, trivial, size-1 group from it -- so under this runner
    a group is NEVER split mid-call (there is never more than one pair per
    call to split), but a real multi-pair group is also never batched into
    one priced call: no cost amortization happens yet. See
    ``tests/core/test_benchmark.py::test_runner_never_splits_a_group_mid_call_but_also_never_batches_one``
    for the pinned-down behavior. Extending this runner (or adding a
    group-aware variant) to pre-flight and price whole groups atomically is
    deferred to W1.1, the branch that lands the first concrete
    ``GroupwiseModule`` (``SelectJudge``) and can measure the real call-count
    reduction this is for.
    """

    def __init__(
        self,
        module: Module[Any],
        *,
        budget_usd: float = 20.0,
        budget_soft_usd: float = 15.0,
        worst_case_units_per_pair: float = 1.0,
    ) -> None:
        """Initialize the runner.

        Args:
            module: The wrapped scorer.
            budget_usd: Hard spend ceiling — the run stops before crossing it.
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
            per-pair budget stop fires.

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
            logger.warning("Module call failed for a candidate: %s; skipping", exc)
            return None

        if not produced:
            self.skipped_count += 1
            logger.warning("Module yielded no judgement for a candidate; skipping")
            return None

        judgement = produced[0]
        self.total_spent_usd += _judgement_cost(judgement)
        self.labeled_count += 1
        return judgement
