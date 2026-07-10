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
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, runtime_checkable

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

# ``_effective_budget`` resolves budget_usd=None -> presets.DEFAULT_BUDGET_USD, the
# same default the verbs' spend cap uses (DRY). No import-cycle risk despite
# presets.py sitting "above" benchmark.py in the module's own layering note:
# `langres/__init__.py` already imports `langres.core.presets` eagerly (and this
# module already imports `langres.core.resolver`, which does `import langres`,
# pulling presets in first regardless), and presets.py's own transitive imports
# never import `core.benchmark` at runtime (only a `TYPE_CHECKING`-guarded
# annotation inside `clients/openrouter.py`) -- verified empirically: importing
# `langres.core.presets` alone never inserts `langres.core.benchmark` into
# `sys.modules`.
from langres.core.presets import _effective_budget
from langres.core.resolver import Resolver
from langres.core.usage import LLMUsage

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


#: How a run's cost was determined. A single ``cost_is_real: bool`` cannot express
#: a run that mixes real OpenRouter-billed cost, a litellm/pinned-price estimate,
#: a zero-cost local judge (no cost concept at all), and DSPy's billed-but-
#: unparseable calls (``cost_untracked``) -- so :func:`_judgement_cost_basis`
#: classifies each judgement into one of the four leaves, and
#: :func:`_combined_cost_basis` collapses a run to ``"mixed"`` the moment two
#: judgements disagree.
CostBasis = Literal["real", "estimated", "mixed", "untracked", "none"]


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
        usage: Token-usage vector summed across every judgement (tokens are the
            fact; ``usd_total`` is derived from them where a real price is
            known). All-zero for judges that report no usage (string/embedding,
            or a judge that never populated ``provenance["usage"]``).
        cost_basis: How ``usd_total`` was determined -- see :data:`CostBasis`.
            ``"none"`` for an empty judgement list.
    """

    usd_total: float = 0.0
    usd_per_1k_pairs: float = 0.0
    est_usd_per_100k: float = 0.0
    escalation_rate: float | None = None
    llm_calls_per_candidate: float | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)
    cost_basis: CostBasis = "none"

    @property
    def cost_is_real(self) -> bool:
        """Whether ``usd_total`` is entirely real, billed spend (``cost_basis == "real"``).

        Kept for continuity with the pre-Task-3 boolean signal; prefer
        :attr:`cost_basis` for the full picture (a run can be ``"estimated"``,
        ``"mixed"``, ``"untracked"``, or ``"none"`` — all of which this reports
        as ``False``, exactly as the old bool-only signal would have wanted for
        anything short of "fully real").
        """
        return self.cost_basis == "real"


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


#: Provenance keys that carry a judgement's USD cost, in fallback order.
#: :func:`_judgement_cost` sums whichever is present; :func:`_judgement_cost_basis`
#: uses the SAME set to decide whether a judgement has a cost concept at all --
#: sharing one constant means the two can never disagree about what counts as
#: "this judgement has a cost" (previously `_judgement_cost_basis` only checked
#: ``"cost_usd"``, so real spend written under ``"llm_cost_usd"`` -- the key
#: ``CascadeModule`` writes -- was misclassified ``cost_basis="none"``).
_COST_KEYS: tuple[str, ...] = ("cost_usd", "llm_cost_usd")


def _judgement_cost(judgement: PairwiseJudgement) -> float:
    """Measured USD cost of one judgement from its provenance.

    Reads :data:`_COST_KEYS` in order (``"cost_usd"`` first, falling back to
    ``"llm_cost_usd"``, the key ``CascadeModule`` writes). Zero-spend judges set
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

    Checked in order: ``cost_untracked`` (DSPyJudge's billed-but-unparseable
    call) wins regardless of what else is set; then none of :data:`_COST_KEYS`
    present means the judge has no cost concept at all (string/embedding); then
    ``cost_is_real`` (LLMJudge's real-OpenRouter-billing vs litellm-estimate
    flag) distinguishes real from estimated -- a judge that sets a cost key
    without ``cost_is_real`` (DSPyJudge's normal, non-error path: tokens times a
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


#: Default score-threshold grid for :func:`evaluate` — the fine ``0.05..0.95``
#: sweep (19 points) a pair-level argmax wants when no fixed ``threshold=`` is
#: given (the default). ``core.benchmark`` is deliberately free of any
#: ``langres.data`` import (the harness is dataset-agnostic), so this mirrors
#: ``langres.data.fixed_split_pair_benchmark.DEFAULT_ARGMAX_GRID`` by value rather
#: than importing it.
DEFAULT_PAIR_GRID: tuple[float, ...] = tuple(round(i * 0.05, 2) for i in range(1, 20))


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
        n_parse_errors: How many judgements the judge flagged as a parse
            error / abstention (``provenance['parse_error']`` truthy — e.g. an
            ``LLMJudge`` under ``on_parse_error='abstain'``, or a ``DSPyJudge``
            parse failure). Kept for backward compatibility; identical to
            :attr:`n_abstained`. ``0`` for judges that never abstain.
        n_abstained: Same count as :attr:`n_parse_errors`, under the more
            general name — an abstention isn't necessarily a "parse" error for
            every judge kind.
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

    # Abstention accounting: judgements the judge flagged as a parse error /
    # abstention (``provenance['parse_error']``) are still graded at their
    # emitted score=0.0, so surface the count and warn loudly -- otherwise a
    # table of P/R/F1 could be built partly on non-verdicts (e.g. a naive
    # replication of a paper prompt the judge could not parse) with no visible
    # signal.
    n_abstained = sum(1 for j in judgements if j.provenance.get("parse_error"))
    abstention_rate = n_abstained / n_judged if n_judged > 0 else 0.0
    if n_abstained > 0:
        logger.warning(
            "%d of %d judgements (%.1f%%) were parse-error abstentions "
            "(provenance['parse_error']); each is graded at its abstained "
            "score=0.0, which classify_pairs() (predicted match iff "
            "score >= threshold) always resolves to a NON-match at any positive "
            "grid threshold -- so an abstention on a true gold pair silently "
            "becomes a false negative in the reported P/R/F1, never a real "
            "verdict. Inspect the judge's response_parser / prompt before "
            "trusting these numbers.",
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
#: BudgetedModuleRunner. evaluate() accepts ANY Module and has no way to know its
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
    """Actionable message for a spend-caused truncation (modelled on NoJudgeAvailableError)."""
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
    module: Module[Any],
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
      :data:`~langres.core.presets.DEFAULT_BUDGET_USD`, the same default the
      verbs use). **This is not a hard ceiling**: the cap is enforced BETWEEN
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
        module: The scorer to evaluate (any :class:`~langres.core.module.Module`).
        candidates: The fixed candidate pairs to judge (each an ``ERCandidate``).
            Must be non-empty.
        gold_pairs: True match pairs as order-independent ``frozenset`` pairs.
        grid: Score thresholds to sweep for the pair-level PR curve and (when
            ``threshold=None``) the best-F1 argmax. Defaults to
            :data:`DEFAULT_PAIR_GRID`, the fine ``0.05..0.95`` sweep.
        threshold: ``None`` (default) grades at the grid's best-F1 argmax
            (optimistically biased, warned about); a float grades ONCE at that
            fixed cut (honest, no warning).
        budget_usd: Spend ceiling for the internal runner, enforced BETWEEN
            calls (see above — NOT a hard ceiling on a single call's own cost);
            ``None`` (default) resolves to
            :data:`~langres.core.presets.DEFAULT_BUDGET_USD`.
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
        ValueError: If ``candidates`` is empty.
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

    resolved_budget = _effective_budget(budget_usd)
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


# ---------------------------------------------------------------------------
# Budgeted module runner
# ---------------------------------------------------------------------------


class BudgetedModuleRunner:
    """Run any :class:`~langres.core.module.Module` under a spend cap.

    Wraps a module and scores candidate pairs one at a time, keeping the
    worst-case *projected* spend under ``budget_usd`` between calls, and
    stopping immediately once a call's *real*, measured cost pushes the running
    total past it. The same three-layer guarantee proven by
    :class:`~langres.bootstrap.labelers.TeacherLabeler`, but as a clean,
    ``Module``-typed component returning :class:`PairwiseJudgement` (the teacher
    is welded to ``LLMJudge`` and returns ``GoldPair``, so it cannot be reused
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
