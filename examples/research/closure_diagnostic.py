"""B1 -- the closure diagnostic: does an output cluster contain a pair we REJECTED?

``Clusterer`` (transitive closure) is langres's default. Per ``docs/THEORY.md`` §7
it is correlation clustering with ``+inf`` on observed positive edges and **0 on
everything else** -- observed negatives and never-observed pairs priced alike. So
the default *discards evidence we already paid for*: every pair a matcher judged
and rejected, that nonetheless lands inside an output cluster, is a judgement --
sometimes a paid LLM judgement -- that the clusterer threw away.

This script turns that theoretical objection into a count. For each registered
benchmark it runs the real front door (``ERModel.dedupe(..., log=)``) at a
train-tuned threshold, then asks of the resulting clusters: **how many
same-cluster pairs were judged ``verdict=False``?** It then re-clusters the
identical judgement set with ``CorrelationClusterer`` -- a *pivot* clusterer,
which only merges on a DIRECT edge to a pivot -- and reports the same count, so
the two clusterers are compared on one scoring run. Note the shipped pivot order
is **deterministic** (``sorted`` by highest incident score, ties by node id --
``core/clusterers/correlation.py:_pivot_priority``), not the uniformly *random*
pivot of Ailon-Charikar-Newman that the class docstring cites; the 3-approximation
guarantee is a property of that randomization and does not transfer. For this
harness that is good news -- the correlation column is exactly reproducible -- but
the numbers below are not evidence for an approximation bound.

Everything here is **$0 in spend**: the ``rapidfuzz`` / ``embedding_cosine``
methods make no paid call. It is *not* dependency-free or offline on a cold
cache: blocking is always the benchmark's own ``VectorBlocker``
(``methods.py:build_blocker``), so a run needs the ``[semantic]`` extra and
downloads ``all-MiniLM-L6-v2`` once.

TWO TRAPS, both verified against the source before this was written. Either one
silently invalidates the result, and the naive version of this experiment is
worse than not running it:

1. **Do not scan ``score < threshold``.** The v4 ``JudgementLog`` row schema
   (``langres/tracking/judgement_log.py``) has **no ``threshold`` column** -- the
   threshold is consumed at write time to produce ``verdict`` and is never
   persisted. Edges are rebuilt from ``verdict``, which ``LoggingMatcher`` computes
   with the *same* ``predicted_match(judgement, threshold)`` predicate
   ``Clusterer.cluster()`` itself uses, so the reconstruction is exact. Rows with
   ``verdict = null`` (an earlier retrieval/reranking stage, or an abstention) are
   excluded -- they were never a rejection.
2. **``predicted_match`` gives ``decision`` precedence over ``score``**
   (``langres/core/models.py``). For a *decider* judge ``verdict`` can be ``True``
   while ``score`` is below the threshold, or ``None``. A naive score-vs-threshold
   scan mis-flags exactly those rows -- it manufactures the finding it is looking
   for. :attr:`BenchmarkFinding.decider_override_rows` counts them in **both**
   directions (a ``verdict=True`` the scan would miss, and a ``verdict=False``
   the scan would miss), so every run states how many rows the naive scan would
   have got wrong.

The instrument checks itself three times per run, and every check **raises**
rather than degrading to a plausible-looking number -- a diagnostic that reports
a figure it cannot vouch for is worse than one that stops:

* ``verdict_agreement`` -- re-deriving ``predicted_match`` from each row's
  ``score``/``decision`` must reproduce the logged ``verdict`` on every row.
* ``reconstruction_exact`` -- closing the accepted (``verdict is True``) edges
  transitively with a **local union-find** (``_components_from_verdicts``) must
  reproduce ``dedupe()``'s own output clusters exactly. If it does not, either the
  log lost edges or ``Clusterer`` no longer merges on ``predicted_match``.
  Deliberately NOT a ``Clusterer`` replay: re-running the implementation under
  test against its own output cannot observe that implementation drifting, and
  the other two checks are blind to the merge predicate as well (the first
  validates the *log*; the sweep re-clusters through the same ``Clusterer``).
  This is the one gate independent of ``core.clusterer``.
* the **sweep must agree with the tuned point**: the grid sweep re-clusters the
  same judgements, so its entry at the tuned threshold has to equal the tuned
  ``rejected_inside`` exactly. Two independent code paths, one number -- free,
  and it catches drift the first two cannot see.

Run (offline, $0)::

    uv run python examples/research/closure_diagnostic.py    # full portfolio -> tracked JSON

A narrowed run must name its own ``--out``: the write replaces the file wholesale,
so pointing ``--fast``/``--only`` at the canonical artifact would shrink the tracked
nine-benchmark result to the subset. The CLI refuses rather than warning::

    uv run python examples/research/closure_diagnostic.py --fast --out tmp/fast.json
    uv run python examples/research/closure_diagnostic.py --only dblp_scholar --out tmp/ds.json

``print`` is allowed in examples (this is an operator tool).
"""

import os

# Pin OpenMP / FAISS threading BEFORE importing anything that pulls torch/faiss
# (the dataset loaders import the embedding stack lazily; macOS libomp
# duplicate-load guard -- mirrors examples/research/portfolio_race.py).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Protocol, cast  # noqa: E402

from pydantic import BaseModel  # noqa: E402

from langres.core.clusterer import Clusterer  # noqa: E402
from langres.core.clusterers.correlation import CorrelationClusterer  # noqa: E402
from langres.core.models import PairwiseJudgement, predicted_match  # noqa: E402
from langres.core.score_type import ScoreType  # noqa: E402
from langres.data.benchmark import Benchmark, complete_partition  # noqa: E402
from langres.eval import get_benchmark, list_benchmarks  # noqa: E402
from langres.methods import BlockingBenchmark, make_resolver_factory  # noqa: E402
from langres.tracking.judgement_log import JudgementLog  # noqa: E402

logger = logging.getLogger("closure_diagnostic")

#: The zero-spend scorer this diagnostic runs by default. ``embedding_cosine`` is
#: the other $0 option (``--method``); both are rankers, so ``decision`` is always
#: ``None`` and trap 2 cannot bite -- which is exactly why the run reports the
#: would-have-been-mis-flagged count instead of asserting it is zero.
DEFAULT_METHOD = "rapidfuzz"

#: Small, in-repo datasets for a quick pass (``--fast``).
FAST_SUBSET: frozenset[str] = frozenset({"fodors_zagat", "dblp_acm", "tiny_fixture"})

#: The tracked full-portfolio artifact. Only a FULL run may write here by default;
#: ``--fast``/``--only`` must name their own ``--out``, because the write replaces
#: the file wholesale and a subset run would otherwise silently shrink it.
CANONICAL_OUT = Path("examples/research/results/closure_diagnostic.json")

SEED = 0

#: Largest output cluster the BCubed / pairwise metrics will be computed over.
#:
#: This is not a tuning knob, it is a tractability wall, and hitting it IS the
#: finding. ``calculate_bcubed_precision`` is O(sum(size**2)) and
#: ``calculate_pairwise_metrics`` materialises C(size, 2) pairs, so a single
#: 40k-record component costs ~1.6e9 comparisons and ~8e8 pairs -- it does not
#: finish. Transitive closure at a low threshold on a large corpus produces
#: exactly that: measured on ``dblp_scholar``, a grid point that chains most of
#: the corpus into one component. The cheap parts of the diagnostic -- the
#: rejected-inside COUNT, the cluster-size distribution -- are O(judgements) and
#: are always reported; only the quality metrics are skipped, as ``None``, with
#: the giant component's size recorded so the omission is legible.
MAX_SCORED_CLUSTER = 2_000

#: ``score_type`` is NOT persisted in the judgement log (see the v4 row schema).
#: The reconstruction fills this placeholder purely to satisfy the required
#: ``PairwiseJudgement`` field: nothing this script touches -- ``predicted_match``,
#: ``Clusterer.cluster``, ``CorrelationClusterer.cluster`` -- ever reads it.
_PLACEHOLDER_SCORE_TYPE: ScoreType = "heuristic"


class _DiagBenchmark(Benchmark[Any], BlockingBenchmark, Protocol):
    """A benchmark usable by BOTH the loader contract and the method registry.

    ``load``/``split`` come from :class:`~langres.data.benchmark.Benchmark`;
    ``schema`` + pinned blocking from
    :class:`~langres.methods.BlockingBenchmark`. Every registered loader satisfies
    this intersection (mirrors ``portfolio_race._RaceBenchmark``).
    """


class ClustererFinding(BaseModel):
    """One clusterer's view of the same judgement set.

    Attributes:
        clusterer: ``"closure"`` (transitive closure) or ``"correlation"`` (pivot).
        n_clusters: Output clusters, as the clusterer returned them. Records with
            no accepted edge are absent (neither clusterer emits a cluster per
            unmatched record), but **size-1 clusters do occur and are counted**:
            ``Clusterer`` adds an accepted pair as a graph edge, so a self-pair
            (``left_id == right_id``, which `VectorBlocker` does emit -- 43 of them
            on ``amazon_google``'s test split) becomes a self-loop and hence a
            one-node component; ``CorrelationClusterer`` strands a node whose only
            neighbour an earlier pivot already consumed. Singletons contribute
            ``C(1, 2) = 0`` to ``n_incluster_pairs``, so they cannot affect
            ``n_rejected_inside`` or the contamination rate -- they only widen the
            cluster-count denominators.
        largest_cluster: Size of the biggest output cluster.
        n_incluster_pairs: ``sum(C(size, 2))`` over the output clusters -- every
            pair the clustering asserts is the same entity.
        n_rejected_inside: Same-cluster pairs that were judged ``verdict=False``.
            **This is (a).**
        rejected_inside_rate: ``n_rejected_inside / n_rejected`` -- of everything
            the matcher rejected, the share the clusterer merged anyway.
        incluster_contamination: ``n_rejected_inside / n_incluster_pairs`` -- of
            everything the clustering asserts, the share it asserts against
            evidence.
        by_cluster_size: ``(size, n_clusters, n_rejected_inside)`` rows -- **(b)**,
            the distribution over cluster size.
        bcubed_f1: BCubed F1 of the completed partition against gold, or ``None``
            when the clustering exceeds :data:`MAX_SCORED_CLUSTER` (an
            intractable giant component -- itself the finding).
        pairwise_f1: Pairwise F1 of the completed partition, same ``None`` rule.
    """

    clusterer: str
    n_clusters: int
    largest_cluster: int
    n_incluster_pairs: int
    n_rejected_inside: int
    rejected_inside_rate: float | None
    incluster_contamination: float | None
    by_cluster_size: list[tuple[int, int, int]]
    bcubed_f1: float | None
    pairwise_f1: float | None


class SweepPoint(BaseModel):
    """The diagnostic at one threshold, for both clusterers.

    The tuned operating point alone is not enough to answer B1. ``THEORY.md`` §7.1
    sharpens the objection to closure into **threshold-fragility**, not badness:
    Hassanzadeh et al.'s partitioning collapses from F1 0.850 at t=0.4 to 0.177 at
    t=0.2, merging 500 true clusters into 51. A single well-tuned threshold can
    therefore report ``rejected_inside = 0`` while the pipeline sits one grid step
    away from a cliff -- and a cluster of size 2 has exactly one in-cluster pair,
    which was by construction accepted, so ``0`` is *forced* there rather than
    measured. Sweeping the benchmark's own threshold grid over the SAME judgements
    (the $0 scorers do not read the clusterer's threshold, so no re-scoring is
    needed) shows whether the zero is robust or an artifact of the operating point.

    Attributes:
        threshold: The grid point.
        n_rejected: Judgements with ``predicted_match(j, t) is False`` at this t.
        closure_clusters: Multi-record clusters under transitive closure.
        closure_largest: Largest closure cluster.
        closure_rejected_inside: **(a)** under closure at this t.
        closure_bcubed_f1: Closure BCubed F1 at this t.
        correlation_clusters: Multi-record clusters under the pivot algorithm.
        correlation_largest: Largest pivot cluster.
        correlation_rejected_inside: **(c)** -- the same count once negatives are
            priced.
        correlation_bcubed_f1: Pivot BCubed F1 at this t.
    """

    threshold: float
    n_rejected: int
    closure_clusters: int
    closure_largest: int
    closure_rejected_inside: int
    closure_bcubed_f1: float | None
    correlation_clusters: int
    correlation_largest: int
    correlation_rejected_inside: int
    correlation_bcubed_f1: float | None


class BenchmarkFinding(BaseModel):
    """The full diagnostic for one benchmark: both clusterers over one scoring run.

    Attributes:
        benchmark: Registry name.
        method: The $0 scorer that produced the judgements.
        threshold: The operating point, tuned on the TRAIN split (no leakage).
        n_test_records: Records in the held-out split the diagnostic ran on.
        n_logged: Judgement-log rows written by the run.
        n_judged: Rows carrying a real verdict (``verdict is not None``).
        n_accepted: ``verdict=True`` rows -- the edge set both clusterers see.
        n_rejected: ``verdict=False`` rows -- the evidence closure prices at 0.
        n_abstained: Rows with ``verdict=null`` (excluded; never a rejection).
        decider_override_rows: Rows a naive ``score < threshold`` scan would have
            mis-flagged as rejections -- ``verdict=True`` with a below-threshold or
            absent score. **Trap 2, counted.**
        verdict_agreement: Share of judged rows whose logged ``verdict`` is
            reproduced by re-deriving ``predicted_match`` from the row. Must be 1.0.
        reconstruction_exact: Whether re-clustering the reconstruction reproduces
            ``dedupe()``'s own clusters. Must be ``True`` for the numbers to mean
            anything.
        seconds: Wall-clock for this benchmark.
        closure: The default transitive-closure finding, at ``threshold``.
        correlation: The pivot-algorithm finding over the identical judgements.
        sweep: The same diagnostic across the benchmark's whole threshold grid --
            the threshold-fragility view (see :class:`SweepPoint`).
    """

    benchmark: str
    method: str
    threshold: float
    n_test_records: int
    n_logged: int
    n_judged: int
    n_accepted: int
    n_rejected: int
    n_abstained: int
    decider_override_rows: int
    verdict_agreement: float | None
    reconstruction_exact: bool
    seconds: float
    closure: ClustererFinding
    correlation: ClustererFinding
    sweep: list[SweepPoint]


def judgements_from_log(rows: list[dict[str, Any]]) -> list[PairwiseJudgement]:
    """Rebuild the judgements a run produced from its log rows.

    Keeps only rows carrying a real ``verdict`` (trap 1: ``verdict = null`` means
    an earlier retrieval/reranking stage, or an abstention -- neither is a
    rejection), and rebuilds each from its **own** ``score``/``decision``, never
    from the logged ``verdict``. Deriving the judgement from the verdict would
    make the agreement check tautological; deriving the verdict from the
    judgement is the check.

    ``confidence``/``confidence_source`` are carried through even though
    ``predicted_match`` never reads them: ``CorrelationClusterer`` falls back to
    ``confidence`` for the edge weight when a decider carries no ``score``
    (``core/clusterers/correlation.py:_build_adjacency``), and that weight sets
    the whole pivot order. Dropping them would silently change the correlation
    partition for any decider matcher while the base-``Clusterer`` reconstruction
    check still passed.

    Args:
        rows: Rows from :meth:`~langres.tracking.judgement_log.JudgementLog.read`.

    Returns:
        One :class:`~langres.core.models.PairwiseJudgement` per judged row, in log
        (scoring) order.
    """
    return [
        PairwiseJudgement(
            left_id=row["left_id"],
            right_id=row["right_id"],
            decision=row["decision"],
            score=row["score"],
            confidence=row.get("confidence"),
            confidence_source=row.get("confidence_source") or "none",
            score_type=_PLACEHOLDER_SCORE_TYPE,
            decision_step=row.get("decision_step") or "replayed",
            provenance={},
        )
        for row in rows
        if row["verdict"] is not None
    ]


def _cluster_index(clusters: list[set[str]]) -> dict[str, int]:
    """Map each clustered id to its cluster index (unclustered ids are absent)."""
    return {rid: idx for idx, cluster in enumerate(clusters) for rid in cluster}


def diagnose(
    label: str,
    clusters: list[set[str]],
    judgements: list[PairwiseJudgement],
    *,
    threshold: float,
    truth_clusters: list[set[str]],
    all_ids: list[str],
) -> ClustererFinding:
    """Count the rejected pairs sitting inside ``clusters`` and score the partition.

    A pair counts when BOTH ids land in the SAME output cluster and the matcher's
    verdict on it was ``False``. Self-pairs cannot occur (the blocker never emits
    one) and are ignored if they somehow do.

    Args:
        label: ``"closure"`` / ``"correlation"`` -- names the finding.
        clusters: The output clusters to interrogate.
        judgements: The judgements the clustering was built from.
        threshold: The operating point ``predicted_match`` is evaluated at.
        truth_clusters: Gold partition for the same split.
        all_ids: Every record id in the split (to complete the partition with
            singletons before scoring).

    Returns:
        A :class:`ClustererFinding`.
    """
    index = _cluster_index(clusters)
    sizes = [len(cluster) for cluster in clusters]
    n_incluster_pairs = sum(math.comb(size, 2) for size in sizes)

    rejected_by_cluster: Counter[int] = Counter()
    n_rejected = 0
    for judgement in judgements:
        if predicted_match(judgement, threshold) is not False:
            continue
        n_rejected += 1
        left = index.get(judgement.left_id)
        right = index.get(judgement.right_id)
        if left is not None and left == right and judgement.left_id != judgement.right_id:
            rejected_by_cluster[left] += 1

    n_rejected_inside = sum(rejected_by_cluster.values())

    # (b): fold the per-cluster counts into a per-SIZE distribution.
    clusters_by_size: Counter[int] = Counter(sizes)
    rejected_by_size: Counter[int] = Counter()
    for cluster_idx, count in rejected_by_cluster.items():
        rejected_by_size[len(clusters[cluster_idx])] += count
    by_cluster_size = [
        (size, clusters_by_size[size], rejected_by_size.get(size, 0))
        for size in sorted(clusters_by_size)
    ]

    largest = max(sizes, default=0)
    bcubed_f1: float | None = None
    pairwise_f1: float | None = None
    if largest <= MAX_SCORED_CLUSTER:
        metrics = Clusterer(threshold=threshold).evaluate(
            complete_partition(clusters, all_ids), truth_clusters
        )
        bcubed_f1 = metrics["bcubed"]["f1"]
        pairwise_f1 = metrics["pairwise"]["f1"]
    else:
        logger.warning(
            "  %s @ t=%.2f: largest cluster is %d records (> %d) -- BCubed/pairwise "
            "skipped (O(size^2)); the rejected-inside count is still exact.",
            label,
            threshold,
            largest,
            MAX_SCORED_CLUSTER,
        )
    return ClustererFinding(
        clusterer=label,
        n_clusters=len(clusters),
        largest_cluster=largest,
        n_incluster_pairs=n_incluster_pairs,
        n_rejected_inside=n_rejected_inside,
        rejected_inside_rate=n_rejected_inside / n_rejected if n_rejected else None,
        incluster_contamination=(
            n_rejected_inside / n_incluster_pairs if n_incluster_pairs else None
        ),
        by_cluster_size=by_cluster_size,
        bcubed_f1=bcubed_f1,
        pairwise_f1=pairwise_f1,
    )


def sweep_thresholds(
    judgements: list[PairwiseJudgement],
    *,
    grid: tuple[float, ...],
    truth_clusters: list[set[str]],
    all_ids: list[str],
) -> list[SweepPoint]:
    """Run the diagnostic at every grid point, for both clusterers.

    Costs no extra scoring: the $0 scorers ignore the clusterer's threshold, so
    one judgement set serves the whole grid. See :class:`SweepPoint` for why a
    single operating point is not enough to answer B1.

    Args:
        judgements: The scored judgements (test split).
        grid: The benchmark's own threshold grid.
        truth_clusters: Gold partition for the split.
        all_ids: Every record id in the split.

    Returns:
        One :class:`SweepPoint` per grid point, in grid order.
    """
    points: list[SweepPoint] = []
    for threshold in grid:
        closure = diagnose(
            "closure",
            Clusterer(threshold=threshold).cluster(judgements),
            judgements,
            threshold=threshold,
            truth_clusters=truth_clusters,
            all_ids=all_ids,
        )
        correlation = diagnose(
            "correlation",
            CorrelationClusterer(threshold=threshold).cluster(judgements),
            judgements,
            threshold=threshold,
            truth_clusters=truth_clusters,
            all_ids=all_ids,
        )
        points.append(
            SweepPoint(
                threshold=threshold,
                n_rejected=sum(1 for j in judgements if predicted_match(j, threshold) is False),
                closure_clusters=closure.n_clusters,
                closure_largest=closure.largest_cluster,
                closure_rejected_inside=closure.n_rejected_inside,
                closure_bcubed_f1=closure.bcubed_f1,
                correlation_clusters=correlation.n_clusters,
                correlation_largest=correlation.largest_cluster,
                correlation_rejected_inside=correlation.n_rejected_inside,
                correlation_bcubed_f1=correlation.bcubed_f1,
            )
        )
    return points


def tune_threshold(
    judgements: list[PairwiseJudgement],
    truth_clusters: list[set[str]],
    all_ids: list[str],
    grid: tuple[float, ...],
) -> float:
    """Pick the BCubed-F1-best threshold by re-clustering ONE scoring pass.

    The reuse is sound *structurally*, not just for these two methods:
    ``methods.make_resolver_factory`` builds the matcher ONCE outside its
    ``factory`` closure and threads ``threshold`` into ``Clusterer(...)`` and
    nowhere else, so no method reachable through it can produce
    threshold-dependent scores. One pass instead of ``len(grid)``.

    This is **not** identical to ``benchmarks.runner.tune_threshold_on_train``:
    that one rebuilds a resolver per point, does not disqualify giant clusterings
    (below), and falls back to the *lowest* threshold rather than the highest. On
    a corpus where disqualification bites, the two can pick different operating
    points -- deliberately, since that is the case disqualification exists for.
    Ties keep the lowest threshold. Run on the TRAIN split only -- the test split
    is never touched.

    Args:
        judgements: Train-split judgements, reconstructed from the log.
        truth_clusters: Gold partition for the train split.
        all_ids: Every train record id.
        grid: Candidate thresholds.

    A grid point whose clustering exceeds :data:`MAX_SCORED_CLUSTER` is
    **disqualified**, not scored: it is intractable to score *and* it is not an
    operating point anyone would choose (it declares thousands of records one
    entity). If every point is disqualified the highest threshold wins, being the
    most conservative merge.

    Returns:
        The best-scoring threshold.
    """
    best_threshold, best_f1 = max(grid), -1.0
    for threshold in grid:
        clusters = Clusterer(threshold=threshold).cluster(judgements)
        largest = max((len(cluster) for cluster in clusters), default=0)
        if largest > MAX_SCORED_CLUSTER:
            logger.info(
                "  threshold=%.2f -> DISQUALIFIED (largest train cluster %d > %d)",
                threshold,
                largest,
                MAX_SCORED_CLUSTER,
            )
            continue
        metrics = Clusterer(threshold=threshold).evaluate(
            complete_partition(clusters, all_ids), truth_clusters
        )
        f1 = metrics["bcubed"]["f1"]
        logger.info("  threshold=%.2f -> train BCubed F1=%.4f", threshold, f1)
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1
    return best_threshold


def _dedupe_with_log(
    resolver: Any, records: list[Any], log_path: Path
) -> tuple[list[set[str]], list[dict[str, Any]]]:
    """Run the real front door once and return ``(clusters, log rows)``.

    The log file is truncated first so a re-run never reads a previous run's
    rows appended beneath its own (``JudgementLog`` appends, by design).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)
    log = JudgementLog(log_path)
    clusters = resolver.dedupe([record.model_dump() for record in records], log=log)
    return list(clusters), log.read()


def run_benchmark(name: str, *, method: str, seed: int, log_dir: Path) -> BenchmarkFinding:
    """Run the whole diagnostic for one registered benchmark.

    Args:
        name: Registry name (must be ``loadable``).
        method: A $0 method name (``rapidfuzz`` / ``embedding_cosine``).
        seed: Split seed.
        log_dir: Where the (regenerable, gitignored) judgement logs are written.

    Returns:
        A :class:`BenchmarkFinding`.

    Raises:
        RuntimeError: If the pipeline has a fitted calibrator, which would make
            the logged ``verdict`` (computed on the RAW score) disagree with the
            match cut (applied to the CALIBRATED score). Nothing here fits one --
            this is a guard against a future change silently invalidating the run.
    """
    started = time.monotonic()
    bench = cast(_DiagBenchmark, get_benchmark(name))
    corpus, gold_clusters, _ = bench.load()
    train, test, train_clusters, test_clusters = bench.split(corpus, gold_clusters, seed=seed)
    factory = make_resolver_factory(method, bench)

    # 1. Operating point: tuned on TRAIN, one scoring pass, no test leakage.
    probe = factory(0.5)
    if probe.calibrator is not None:  # pragma: no cover - nothing here fits one
        raise RuntimeError(
            f"{name}: resolver carries a fitted calibrator; the logged verdict is "
            "computed on the raw score while the match cut thresholds the calibrated "
            "one, so the log would no longer reconstruct the clustering exactly."
        )
    _, train_rows = _dedupe_with_log(probe, train, log_dir / f"{name}_train.jsonl")
    threshold = tune_threshold(
        judgements_from_log(train_rows),
        train_clusters,
        [record.id for record in train],
        bench.threshold_grid,
    )
    logger.info("%s: tuned threshold=%.2f", name, threshold)

    # 2. The measured run: the real front door on the held-out split.
    resolver = factory(threshold)
    clusters, rows = _dedupe_with_log(resolver, test, log_dir / f"{name}_test.jsonl")
    # ONE filtered row list, shared with the reconstruction, so the two can never
    # drift out of alignment: zipping judgements against a separately-recomputed
    # filter would silently pair row i with judgement j the moment the predicate
    # in judgements_from_log changed.
    judged_rows = [row for row in rows if row["verdict"] is not None]
    judgements = judgements_from_log(judged_rows)

    # 3. Instrument check A -- the logged verdict is reproducible from the row.
    agree = sum(
        1
        for judgement, row in zip(judgements, judged_rows, strict=True)
        if predicted_match(judgement, threshold) is row["verdict"]
    )
    verdict_agreement = agree / len(judgements) if judgements else None
    if verdict_agreement is not None and verdict_agreement < 1.0:
        raise RuntimeError(
            f"{name}: only {agree}/{len(judgements)} rows reproduce their logged "
            "verdict from their own score/decision -- the reconstruction is not "
            "predicted_match, so every count below would be about something else."
        )

    # 4. Instrument check B -- the reconstruction reproduces dedupe()'s clusters.
    # Rebuilt by a LOCAL union-find over the logged verdicts, not by replaying
    # Clusterer: replaying the implementation under test cannot observe that
    # implementation drifting (see _components_from_verdicts). This is the only
    # one of the three gates that is independent of core.clusterer.
    replayed = _components_from_verdicts(judged_rows)
    if not _same_partition(replayed, clusters):
        raise RuntimeError(
            f"{name}: the reconstruction does NOT reproduce dedupe()'s clusters "
            f"({len(replayed)} replayed vs {len(clusters)} actual). Either the log "
            "lost edges or Clusterer no longer merges on predicted_match; the harness "
            "is measuring something other than the pipeline, so it refuses to report."
        )
    reconstruction_exact = True

    # Trap 2, counted in BOTH directions: rows a naive ``score < threshold`` scan
    # would mis-flag. A decider can say Yes below (or without) a score, AND No at
    # or above one -- the scan gets each wrong in the opposite direction, so
    # counting only the first would understate what the shortcut costs.
    decider_override_rows = sum(
        1
        for row in rows
        if (row["verdict"] is True and (row["score"] is None or row["score"] < threshold))
        or (row["verdict"] is False and row["score"] is not None and row["score"] >= threshold)
    )

    all_ids = [record.id for record in test]
    closure = diagnose(
        "closure",
        clusters,
        judgements,
        threshold=threshold,
        truth_clusters=test_clusters,
        all_ids=all_ids,
    )
    correlation = diagnose(
        "correlation",
        CorrelationClusterer(threshold=threshold).cluster(judgements),
        judgements,
        threshold=threshold,
        truth_clusters=test_clusters,
        all_ids=all_ids,
    )

    sweep = sweep_thresholds(
        judgements,
        grid=bench.threshold_grid,
        truth_clusters=test_clusters,
        all_ids=all_ids,
    )
    # 5. Instrument check C -- free, and independent of A and B. ``tune_threshold``
    # always returns a grid element, and the sweep re-clusters the SAME judgement
    # set, so the sweep's entry at the tuned threshold must equal the tuned-point
    # counts exactly. Two code paths, one number: a divergence means one of them
    # drifted, which neither A nor B can see.
    at_tuned = next((p for p in sweep if p.threshold == threshold), None)
    if at_tuned is None:
        raise RuntimeError(f"{name}: tuned threshold {threshold} is not a grid point")
    if (at_tuned.closure_rejected_inside, at_tuned.correlation_rejected_inside) != (
        closure.n_rejected_inside,
        correlation.n_rejected_inside,
    ):
        raise RuntimeError(
            f"{name}: the sweep disagrees with the tuned point at t={threshold} "
            f"(closure {at_tuned.closure_rejected_inside} vs "
            f"{closure.n_rejected_inside}, correlation "
            f"{at_tuned.correlation_rejected_inside} vs {correlation.n_rejected_inside})"
        )

    verdicts = Counter(row["verdict"] for row in rows)
    return BenchmarkFinding(
        benchmark=name,
        method=method,
        threshold=threshold,
        n_test_records=len(test),
        n_logged=len(rows),
        n_judged=len(judgements),
        n_accepted=verdicts[True],
        n_rejected=verdicts[False],
        n_abstained=verdicts[None],
        decider_override_rows=decider_override_rows,
        verdict_agreement=verdict_agreement,
        reconstruction_exact=reconstruction_exact,
        seconds=time.monotonic() - started,
        closure=closure,
        correlation=correlation,
        sweep=sweep,
    )


def _components_from_verdicts(rows: list[dict[str, Any]]) -> list[set[str]]:
    """Connected components of the ``verdict is True`` edges, by local union-find.

    Deliberately **not** ``Clusterer.cluster()``. Instrument check B compares this
    against the partition ``dedupe()`` actually returned, and *a check that replays
    the implementation under test cannot observe that implementation drifting*: if
    ``Clusterer`` ever merged on ``score >= threshold`` instead of
    ``predicted_match``, a ``Clusterer``-based replay would move in lockstep with
    the thing it is checking and stay green, while ``verdict_agreement`` (which
    validates the *log*, not the merge) and the sweep (same clusterer again) would
    also stay green -- three gates, one blind spot, and every count below silently
    describing a different predicate from the one that built the clusters.

    This path reads only the logged verdicts and closes them transitively itself,
    so it moves independently of ``core.clusterer`` and the comparison has a way
    to fail.

    Matches ``Clusterer``'s output convention exactly, including its edge cases: a
    node enters only via an accepted edge, so a record with no accepted pair is
    absent rather than emitted as a singleton -- but an accepted **self-pair**
    (``left_id == right_id``) does yield a one-node component, because ``Clusterer``
    feeds it to ``nx.add_edge(x, x)`` as a self-loop. That is not hypothetical:
    ``VectorBlocker`` emits 43 accepted self-pairs on ``amazon_google``'s test
    split. Reproducing it is what keeps this check exact rather than approximately
    right; a version that dropped singletons would fail on that benchmark alone.

    Args:
        rows: Judged log rows (``verdict`` not ``None``).

    Returns:
        The transitive closure of the accepted edges, as a list of id sets.
    """
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # path compression
            parent[node], node = root, parent[node]
        return root

    for row in rows:
        if row["verdict"] is True:
            left, right = find(row["left_id"]), find(row["right_id"])
            if left != right:
                parent[right] = left

    components: dict[str, set[str]] = {}
    for node in list(parent):
        components.setdefault(find(node), set()).add(node)
    return list(components.values())


def _same_partition(left: list[set[str]], right: list[set[str]]) -> bool:
    """Whether two clusterings are the same set of clusters (order-independent)."""
    return {frozenset(cluster) for cluster in left} == {frozenset(cluster) for cluster in right}


def select_benchmarks(*, fast: bool, only: list[str] | None) -> list[str]:
    """Registry-driven selection: every loadable entry, or a narrowed subset.

    Args:
        fast: Keep only :data:`FAST_SUBSET`.
        only: Explicit names (wins over ``fast``).

    Returns:
        Registered, loadable benchmark names in registry order.
    """
    names = [entry.name for entry in list_benchmarks() if entry.loadable]
    skipped = [entry.name for entry in list_benchmarks() if not entry.loadable]
    for name in skipped:
        print(f"[skip] {name}: external-only, not bundled")
    if only:
        # A typo'd name would otherwise yield an empty run, an empty table and an
        # empty tracked JSON -- silently overwriting a real result set with [].
        # A warning did NOT prevent that (it printed, then the run wrote anyway),
        # so selecting nothing is an error, not a message.
        unknown = sorted(set(only) - {entry.name for entry in list_benchmarks()})
        if unknown:
            print(f"[warn] --only names no registered benchmark: {', '.join(unknown)}")
        selected = [name for name in names if name in set(only)]
        if not selected:
            raise SystemExit(
                f"--only {' '.join(only)} selected no loadable benchmark; refusing to run. "
                f"Loadable: {', '.join(names)}"
            )
        return selected
    if fast:
        return [name for name in names if name in FAST_SUBSET]
    return names


def worst_sweep_point(finding: BenchmarkFinding) -> SweepPoint | None:
    """The grid point where closure discards the most evidence (ties: lowest t)."""
    if not finding.sweep:
        return None
    return max(finding.sweep, key=lambda p: (p.closure_rejected_inside, -p.threshold))


def to_markdown(findings: list[BenchmarkFinding]) -> str:
    """Render the headline table: (a) and (c) side by side, per benchmark.

    The last three columns are the threshold-fragility view -- **closure's** worst
    grid point, so a ``0`` at the tuned operating point is never read as "closure
    is safe here" without the evidence that it stays 0 across the grid. The final
    column is correlation's value *at that same threshold* (the like-for-like
    comparison), not correlation's own worst point.
    """
    header = (
        "| benchmark | t | judged | rejected | closure: rejected-inside | rate | "
        "corr: rejected-inside | closure BCubed F1 | corr BCubed F1 | "
        "closure worst t | closure rej-inside @ worst | corr rej-inside @ same t |"
    )
    lines = [header, "|" + "---|" * 12]
    for f in findings:
        rate = f.closure.rejected_inside_rate
        worst = worst_sweep_point(f)
        lines.append(
            f"| {f.benchmark} | {f.threshold:.2f} | {f.n_judged:,} | {f.n_rejected:,} | "
            f"{f.closure.n_rejected_inside:,} | "
            f"{'n/a' if rate is None else f'{rate:.4f}'} | "
            f"{f.correlation.n_rejected_inside:,} | "
            f"{_f1(f.closure.bcubed_f1)} | {_f1(f.correlation.bcubed_f1)} | "
            f"{'n/a' if worst is None else f'{worst.threshold:.2f}'} | "
            f"{'n/a' if worst is None else f'{worst.closure_rejected_inside:,}'} | "
            f"{'n/a' if worst is None else f'{worst.correlation_rejected_inside:,}'} |"
        )
    return "\n".join(lines)


def _f1(value: float | None) -> str:
    """Format an F1, or ``giant`` when the clustering was too large to score."""
    return "giant" if value is None else f"{value:.4f}"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="only the small in-repo subset")
    parser.add_argument("--only", nargs="+", help="explicit benchmark names")
    parser.add_argument("--method", default=DEFAULT_METHOD, help="a $0 method name")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "where to write the machine-readable findings (a TRACKED path). "
            f"Defaults to {CANONICAL_OUT} for a FULL run; REQUIRED with --fast/--only, "
            "which would otherwise replace the full-portfolio artifact with a subset"
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("tmp/closure_diagnostic"),
        help="where to write the (regenerable, gitignored) judgement logs",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # A narrowed run must not land on the canonical artifact. --fast/--only measure
    # a SUBSET, and the final write replaces the file wholesale, so the documented
    # `--fast` one-liner would silently reduce the tracked 9-benchmark result to 2.
    # Uncommitted/committed results are equally destroyed; refuse rather than warn.
    if args.out is None and (args.fast or args.only):
        parser.error(
            "--fast/--only measure a subset and the write replaces the whole file, "
            f"which would reduce {CANONICAL_OUT} to just those benchmarks. "
            "Pass an explicit --out (e.g. --out tmp/closure_subset.json), or run the "
            "full portfolio to refresh the canonical artifact."
        )
    out: Path = args.out if args.out is not None else CANONICAL_OUT

    findings: list[BenchmarkFinding] = []
    failures: list[str] = []
    for name in select_benchmarks(fast=args.fast, only=args.only):
        print(f"[run] {name}: {args.method} ($0 spend) ...", flush=True)
        try:
            finding = run_benchmark(name, method=args.method, seed=args.seed, log_dir=args.log_dir)
        except Exception as exc:  # noqa: BLE001 - a broken loader must not kill the sweep
            # ``exception`` (not ``error``): a benchmark that drops out is absent
            # from both the table and the tracked JSON, so the traceback in the run
            # log is the ONLY record of why. A one-line message makes a loader or
            # instrument-check failure near-undebuggable from the artifact.
            logger.exception("%s: run_benchmark raised", name)
            print(f"[fail] {name}: {type(exc).__name__}: {exc}", flush=True)
            failures.append(f"{name} ({type(exc).__name__})")
            continue
        findings.append(finding)
        print(
            f"       t={finding.threshold:.2f} judged={finding.n_judged:,} "
            f"rejected={finding.n_rejected:,} "
            f"rejected-inside closure={finding.closure.n_rejected_inside:,} "
            f"corr={finding.correlation.n_rejected_inside:,} "
            f"({finding.seconds:.1f}s)",
            flush=True,
        )
        # Persist after EVERY benchmark, not once at the end: the large datasets
        # take minutes, and a crash or a kill on the last one must not throw away
        # the ones already measured.
        write_findings(findings, out)

    print("\n" + to_markdown(findings))
    if failures:
        # Stated next to the table, not only in the log: a missing row otherwise
        # looks identical to a benchmark that was never selected.
        print(f"\nNOT MEASURED ({len(failures)}): {', '.join(failures)}")
    # ``out``, NOT ``args.out``: the latter is None on a full run (the default is
    # resolved into ``out`` above), so this line used to raise AttributeError on
    # ``out.parent`` -- after every benchmark had already been measured.
    write_findings(findings, out)
    print(f"\nwrote {out}")


def write_findings(findings: list[BenchmarkFinding], out: Path) -> None:
    """Persist the findings measured so far to a tracked JSON path.

    Refuses to replace a non-empty artifact with an empty one: if every selected
    benchmark raised, the run has learned nothing, and overwriting a good result
    set with ``[]`` would destroy it for no gain. Writing an empty file is only
    allowed when there was nothing there to lose.
    """
    if not findings and out.exists() and out.read_text().strip() not in ("", "[]"):
        raise SystemExit(
            f"every selected benchmark failed; refusing to overwrite {out} with []. "
            "The existing results are unchanged -- see the traceback(s) above."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([f.model_dump() for f in findings], indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
