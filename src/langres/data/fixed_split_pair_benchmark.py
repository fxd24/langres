"""Fixed literature pair-split adapter + honest pair-level evaluation (#80 Phase 1, #55 C2).

The DeepMatcher/Magellan pair splits (``train.csv`` / ``valid.csv`` /
``test.csv``) that ship with Amazon-Google and Abt-Buy are rows of
``(id_a, id_b, label)``. To *fit* a supervised judge
(:class:`~langres.core.matchers.random_forest_judge.RandomForestMatcher`) or grade
any judge at the pair level against literature bands (Ditto, DeepMatcher,
Magellan), those rows must become :class:`~langres.core.models.ERCandidate`
objects **carrying a comparison vector** — the raw ``(id_a, id_b)`` rows do not,
and a supervised judge raises without one.

This module is that missing, reusable bridge:

- :class:`FixedSplitPairBenchmark` — given a corpus, a fixed pair split, and a
  :class:`~langres.core.comparator.Comparator`, it yields per split
  (train/valid/test) the :class:`ERCandidate` list *with a comparison vector
  attached*, the positionally-aligned ``list[bool]`` labels, and the gold set
  (``frozenset`` per ``label == 1`` row). It is dataset-agnostic: the schema and
  the two loaders are injected, so it works unchanged for Amazon-Google,
  Abt-Buy, and any future ``(id_a, id_b, label)`` split.

- :func:`evaluate_fixed_split_honest` — the *honest* pair-level grader. It
  derives the decision threshold from the judge's scores on the TRAIN (or VALID)
  split via :func:`~langres.training.calibration.derive_threshold`, then grades the
  judge on the FULL TEST split at that FIXED cut with
  :func:`~langres.core.metrics.classify_pairs`. It also reports the *leaky*
  "argmax-F1-on-test" number (what an evaluator that tunes the threshold on the
  test set itself would print) so the two can be compared directly — the
  honesty delta. It deliberately does **not** touch the argmax in
  :func:`~langres.core.benchmark.evaluate_judge_on_candidates`; this is the
  honest path *alongside* it.

Import direction: this lives in ``langres.data`` and depends on ``langres.core``
(calibration, metrics, comparator, models, module) — never the reverse, so the
dataset-agnostic ``core.benchmark`` harness stays free of any ``langres.data``
import.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

from pydantic import BaseModel

from langres.training.calibration import ThresholdMethod, derive_threshold
from langres.core.comparator import Comparator
from langres.core.comparators import StringComparator
from langres.core.feature import FeatureSpec
from langres.core.metrics import PairMetrics, classify_pairs, pair_pr_curve
from langres.core.models import ERCandidate
from langres.core.matcher import Matcher

SchemaT = TypeVar("SchemaT", bound=BaseModel)

#: Default threshold grid the leaky "argmax-on-test" comparison sweeps. Finer
#: than the datasets' clusterer grids (0.3..0.8) because a supervised judge's
#: ``prob_*`` scores can concentrate low on hard negatives, so the best-F1 cut
#: on test often lands below 0.3; a fine grid keeps the leaky *ceiling* honest.
DEFAULT_ARGMAX_GRID: tuple[float, ...] = tuple(round(i * 0.05, 2) for i in range(1, 20))


def _record_id(record: BaseModel) -> str:
    """Read the ``id`` off a schema record (bound to BaseModel, so via cast)."""
    return cast("str", record.id)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class SplitPairData(Generic[SchemaT]):
    """One split's fit/eval-ready bundle: candidates, labels, and gold pairs.

    A transient runtime container (not a serialized contract), so a frozen
    dataclass rather than a Pydantic model: it preserves the identity of the
    corpus records referenced by each candidate and avoids re-validating every
    ``ERCandidate`` on construction.

    Attributes:
        candidates: One :class:`ERCandidate` per split row, each carrying a
            ``comparison`` vector from the benchmark's Comparator (so a
            supervised judge can fit/score on it).
        labels: ``label == 1`` per row, positionally aligned with
            ``candidates`` (the contract ``RandomForestMatcher.fit`` requires).
        gold: The positive pairs as order-independent ``frozenset({id_a, id_b})``
            — exactly the ``label == 1`` rows.
    """

    candidates: list[ERCandidate[SchemaT]]
    labels: list[bool]
    gold: set[frozenset[str]] = field(default_factory=set)


class HonestPairEval(BaseModel):
    """Honest vs. leaky pair-level scores for one judge on one fixed test split.

    Attributes:
        dataset: The benchmark name (e.g. ``"amazon_google"``).
        derive_on: The split the honest threshold was derived from (``"train"``
            or ``"valid"``).
        derived_threshold: The data-driven cut derived from the judge's scores on
            ``derive_on`` via :func:`~langres.training.calibration.derive_threshold`.
        threshold_method: How the threshold was derived (``"youden"`` /
            ``"percentile"``).
        honest: Pair-level metrics on the FULL test split at
            ``derived_threshold`` (no test-label peeking).
        argmax_on_test: Pair-level metrics at the threshold that maximizes F1 on
            the test set itself — the *leaky* number, for comparison only.
        honesty_delta_f1: ``argmax_on_test.f1 - honest.f1`` — how much the leaky
            selection inflates F1 over the honest cut. Larger means more
            optimism baked into an argmax-on-test report.
    """

    dataset: str
    derive_on: str
    derived_threshold: float
    threshold_method: str
    honest: PairMetrics
    argmax_on_test: PairMetrics
    honesty_delta_f1: float


class FixedSplitPairBenchmark(Generic[SchemaT]):
    """Adapt a fixed ``(id_a, id_b, label)`` pair split into fit/eval-ready data.

    Given a corpus, the fixed literature splits, and a
    :class:`~langres.core.comparator.Comparator`, :meth:`build` turns each
    split's rows into a :class:`SplitPairData` whose candidates each carry a
    comparison vector. Dataset-agnostic: construct directly, or via
    :meth:`from_loaders` with the schema + the two loader functions injected (the
    same shape both :mod:`langres.data.amazon_google` and
    :mod:`langres.data.abt_buy` expose).
    """

    def __init__(
        self,
        *,
        name: str,
        corpus: Sequence[SchemaT],
        splits: dict[str, list[tuple[str, str, int]]],
        comparator: Comparator[SchemaT],
    ) -> None:
        """Initialize the adapter.

        Args:
            name: Benchmark name, used as each candidate's ``blocker_name``
                (``f"{name}_fixed_pairs"``) and echoed into results.
            corpus: The record objects the split ids reference. Indexed into an
                id -> record map once.
            splits: Fixed split rows keyed by split name (``"train"`` /
                ``"valid"`` / ``"test"``), each a list of ``(id_a, id_b, label)``.
            comparator: Attaches a comparison vector to every candidate; its
                :attr:`feature_specs` also drive a supervised judge's features.
        """
        self.name = name
        self._by_id: dict[str, SchemaT] = {_record_id(record): record for record in corpus}
        self._split_rows = splits
        self._comparator = comparator
        self._cache: dict[str, SplitPairData[SchemaT]] = {}

    @classmethod
    def from_loaders(
        cls,
        *,
        name: str,
        schema: type[SchemaT],
        corpus_loader: Callable[[], tuple[Sequence[SchemaT], object, object]],
        pair_split_loader: Callable[[], dict[str, list[tuple[str, str, int]]]],
        comparator: Comparator[SchemaT] | None = None,
        exclude: set[str] | None = None,
    ) -> "FixedSplitPairBenchmark[SchemaT]":
        """Build from a schema + the dataset's ``(corpus, ...)`` and split loaders.

        Args:
            name: Benchmark name (see :meth:`__init__`).
            schema: The record schema; used to auto-derive a
                :class:`~langres.core.comparator.StringComparator` when
                ``comparator`` is not given.
            corpus_loader: A loader returning ``(corpus, ...)`` — only element 0
                (the corpus) is used, matching the shared shape of
                ``load_amazon_google`` / ``load_abt_buy``.
            pair_split_loader: A loader returning the fixed split rows keyed by
                split name (e.g. ``load_amazon_google_pair_splits``).
            comparator: Optional explicit comparator. Defaults to
                ``StringComparator.from_schema(schema, exclude=exclude)``.
            exclude: Field names to exclude when auto-deriving the comparator
                (forwarded to ``StringComparator.from_schema``).

        Returns:
            A ready :class:`FixedSplitPairBenchmark`.
        """
        corpus = corpus_loader()[0]
        if comparator is None:
            comparator = StringComparator.from_schema(schema, exclude=exclude)
        return cls(
            name=name,
            corpus=corpus,
            splits=pair_split_loader(),
            comparator=comparator,
        )

    @property
    def split_names(self) -> list[str]:
        """The split names available (order as provided), e.g. train/valid/test."""
        return list(self._split_rows)

    @property
    def feature_specs(self) -> list[FeatureSpec]:
        """The comparator's declared features — what a supervised judge fits on."""
        return self._comparator.feature_specs

    def build(self, split_name: str) -> SplitPairData[SchemaT]:
        """Build one split's candidates (comparison attached), labels, and gold.

        Results are cached per split so repeated ``build("test")`` calls do not
        recompute comparisons.

        Args:
            split_name: The split to build (must be a key of the provided splits).

        Returns:
            The split's :class:`SplitPairData`.

        Raises:
            KeyError: If ``split_name`` is not one of :attr:`split_names`.
            ValueError: If a split row references an id absent from the corpus.
        """
        if split_name in self._cache:
            return self._cache[split_name]
        if split_name not in self._split_rows:
            raise KeyError(f"Unknown split {split_name!r}; available splits: {self.split_names}")

        candidates: list[ERCandidate[SchemaT]] = []
        labels: list[bool] = []
        gold: set[frozenset[str]] = set()
        for id_a, id_b, label in self._split_rows[split_name]:
            try:
                left = self._by_id[id_a]
                right = self._by_id[id_b]
            except KeyError as exc:
                raise ValueError(
                    f"Split {split_name!r} references id {exc.args[0]!r} which is not "
                    f"in the corpus of benchmark {self.name!r}."
                ) from exc
            is_match = label == 1
            candidates.append(
                ERCandidate(
                    left=left,
                    right=right,
                    blocker_name=f"{self.name}_fixed_pairs",
                    comparison=self._comparator.compare(left, right),
                )
            )
            labels.append(is_match)
            if is_match:
                gold.add(frozenset({id_a, id_b}))

        data = SplitPairData(candidates=candidates, labels=labels, gold=gold)
        self._cache[split_name] = data
        return data


def evaluate_fixed_split_honest(
    judge: Matcher[SchemaT],
    benchmark: FixedSplitPairBenchmark[SchemaT],
    *,
    derive_on: str = "train",
    argmax_grid: Sequence[float] = DEFAULT_ARGMAX_GRID,
    method: ThresholdMethod = "youden",
) -> HonestPairEval:
    """Grade a judge on the full test split at a threshold derived from train.

    The honest methodology, contrasted with an argmax-on-test report:

    1. Run ``judge`` over the ``derive_on`` split; derive a decision threshold
       from those scores + labels via
       :func:`~langres.training.calibration.derive_threshold` (no test labels seen).
    2. Run ``judge`` over the FULL test split and grade it at that FIXED
       threshold with :func:`~langres.core.metrics.classify_pairs`.
    3. Separately, sweep ``argmax_grid`` on the test judgements and take the
       best-F1 point — the *leaky* number an evaluator that tunes on test would
       report — so the caller can quote the honesty delta.

    Args:
        judge: The (already-fit, if supervised) scorer to grade.
        benchmark: The fixed-split adapter providing ``derive_on`` and ``test``.
        derive_on: Split to derive the threshold from (``"train"`` or
            ``"valid"``).
        argmax_grid: Thresholds swept for the leaky argmax-on-test comparison.
        method: Threshold-derivation method forwarded to ``derive_threshold``.

    Returns:
        A :class:`HonestPairEval` with the honest metrics, the leaky
        argmax-on-test metrics, the derived threshold, and the honesty delta.
    """
    derive_data = benchmark.build(derive_on)
    test_data = benchmark.build("test")

    derive_judgements = list(judge.forward(iter(derive_data.candidates)))
    # A threshold is derived from scores; a decision-only judge has none to derive
    # from, so name it and fail loudly rather than silently dropping its pairs.
    derive_scores: list[float] = []
    for j in derive_judgements:
        if j.score is None:
            raise ValueError(
                f"cannot derive a threshold for {benchmark.name!r}: judge "
                f"{type(judge).__name__} produced a score-less judgement for pair "
                f"{j.left_id}/{j.right_id}; a decision-only judge has no scores to "
                "derive a threshold from."
            )
        derive_scores.append(j.score)
    threshold = derive_threshold(derive_scores, derive_data.labels, method=method)

    test_judgements = list(judge.forward(iter(test_data.candidates)))
    honest = classify_pairs(test_judgements, test_data.gold, threshold)

    curve = pair_pr_curve(test_judgements, test_data.gold, argmax_grid)
    argmax = max(curve, key=lambda m: m.f1)

    return HonestPairEval(
        dataset=benchmark.name,
        derive_on=derive_on,
        derived_threshold=threshold,
        threshold_method=method,
        honest=honest,
        argmax_on_test=argmax,
        honesty_delta_f1=argmax.f1 - honest.f1,
    )
