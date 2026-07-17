"""Benchmark **spec**: the "a dataset *is* a benchmark" contract.

A dataset participates in the benchmark harness by conforming to the
:class:`Benchmark` protocol (its optional benchmark capability) — carrying a
``name``, a ``load`` returning the corpus + closed-world gold partition + gold
pairs, a leakage-free ``split``, and a ``threshold_grid``. This module is the
import-light half of the old ``langres.core.benchmark``: pydantic + typing + the
two leaf metric helpers a track type needs (:class:`~langres.metrics.metrics.PairMetrics`,
:func:`~langres.metrics.metrics.pairs_from_clusters`). It imports nothing from
``core.resolver`` / ``langres.methods`` / the harness, so a ``data`` module can
import the contract without pulling the runner — the edge the split exists to
kill. The generic *harness* (run a method, get a table; score a judge on a fixed
candidate set) lives in :mod:`langres.benchmarks`.
"""

from collections.abc import Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from langres.metrics.metrics import PairMetrics, pairs_from_clusters


class BlindCostError(RuntimeError):
    """Raised when a budgeted runner cannot observe the cost of its work.

    A budget cap is only safe while spend can be measured. Two situations make
    the cap *blind* and so abort the run rather than risk unbounded spend:

    - The resolved per-pair worst-case price is ``$0`` (e.g. a price of zero was
      passed), so the pre-flight cap ``floor(budget / 0)`` is unbounded.
    - (For :class:`~langres.curation.labelers.TeacherLabeler`) a judgement
      reports neither token counts nor a cost, so the running tally is untrusted.

    :attr:`partial` carries any results already produced (and paid for) before a
    *mid-loop* abort, so a caller can recover them rather than discard paid work.
    It is set by the catcher immediately before re-raising (e.g.
    :meth:`~langres.curation.labelers.TeacherLabeler.label`), not at the raise
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


#: Default score-threshold grid for :func:`evaluate` — the fine ``0.05..0.95``
#: sweep (19 points) a pair-level argmax wants when no fixed ``threshold=`` is
#: given (the default). Mirrored by value from
#: ``langres.data.fixed_split_pair_benchmark.DEFAULT_ARGMAX_GRID`` (an identical
#: grid) rather than imported from it: this module is deliberately import-light
#: (pydantic + typing + leaf metrics), while ``fixed_split_pair_benchmark`` pulls
#: the heavy ``[trained]`` chain (``langres.training.calibration`` -> scikit-learn)
#: plus ``langres.core`` matchers/comparators at module scope. Importing it here
#: would drag all of that into every ``import langres.data.benchmark`` (hence into
#: ``langres.eval`` / ``langres.benchmarks``), defeating the import-lightness this
#: benchmark-spec split exists to preserve. Not a cycle —
#: ``fixed_split_pair_benchmark`` does not import this module.
DEFAULT_PAIR_GRID: tuple[float, ...] = tuple(round(i * 0.05, 2) for i in range(1, 20))
