"""Tests for FitReport + Resolver.fit() wiring (W1.x, PR-B).

Every non-raising Resolver.fit() path must set self.fit_report_ (the sklearn
trailing-underscore digest) and still return self. Covers the no-op branch, the
pre-aligned labels= branch, and the id-keyed pairs= branch (coverage + held-out
metrics), plus the import-light guarantee for the fit_report leaf.

Uses a lightweight fake SupervisedFitMixin matcher so these stay fast and pull
no sklearn/torch; the real RandomForestMatcher path is exercised by the PR-B
runtime smoke.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator, Sequence

import pytest

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.fit_report import FitReport
from langres.curation.harvest import GoldCoverage, LabeledPair
from langres.core.matcher import Matcher
from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.reports import ScoreInspectionReport
from langres.core.resolver import Resolver


class _FakeSupervised(Matcher[CompanySchema]):
    """A SupervisedFitMixin matcher: scores 1.0 when the two names are equal.

    Deterministic and dependency-free, so held-out metrics are predictable.
    """

    def __init__(self) -> None:
        self.fit_calls: list[tuple[int, list[bool]]] = []

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        for c in candidates:
            yield PairwiseJudgement(
                left_id=c.left.id,
                right_id=c.right.id,
                score=1.0 if c.left.name == c.right.name else 0.0,
                score_type="heuristic",
                decision_step="fake",
                provenance={},
            )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        raise NotImplementedError

    def fit(self, candidates: Iterator[ERCandidate[CompanySchema]], labels: Sequence[bool]) -> None:
        self.fit_calls.append((len(list(candidates)), list(labels)))


class _NoHook(Matcher[CompanySchema]):
    """Implements neither fit hook (non-learnable)."""

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        yield from ()

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        raise NotImplementedError


# Two matching pairs (same name) + one non-match, across two entity-components.
RECORDS = [
    {"id": "a1", "name": "Acme"},
    {"id": "a2", "name": "Acme"},
    {"id": "b1", "name": "Beta"},
    {"id": "b2", "name": "Beta"},
    {"id": "c1", "name": "Gamma"},
]


def _resolver(module: Matcher[CompanySchema], *, threshold: float = 0.5) -> Resolver:
    return Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        matcher=module,
        clusterer=Clusterer(threshold=threshold),
    )


# ---------------------------------------------------------------------------
# FitReport model basics.
# ---------------------------------------------------------------------------


def test_nothing_trainable_report_names_the_matcher_and_marks_untrained() -> None:
    report = FitReport.nothing_trainable("WeightedAverageMatcher")

    assert report.trained is False
    assert report.trainable == "WeightedAverageMatcher (no fit hook)"
    assert report.coverage is None
    assert "Fit Report" in report.to_markdown()
    assert report.render() == report.to_markdown()


def test_build_derives_entity_disjoint_from_split() -> None:
    with_split = FitReport.build(trainable="X", trained=True, n_train=1, split=0.3)
    no_split = FitReport.build(trainable="X", trained=True, n_train=1)

    assert with_split.entity_disjoint is True
    assert no_split.entity_disjoint is False


# ---------------------------------------------------------------------------
# Resolver.fit() sets fit_report_ on every non-raising path.
# ---------------------------------------------------------------------------


def test_noop_fit_sets_minimal_report_and_chains() -> None:
    resolver = _resolver(_NoHook())

    result = resolver.fit(RECORDS)

    assert result is resolver
    assert resolver.fit_report_ is not None
    assert resolver.fit_report_.trained is False
    # Still chains into resolve() (sklearn-style symmetry).
    assert isinstance(resolver.resolve(RECORDS), list)


def test_supervised_labels_path_sets_report_without_coverage() -> None:
    module = _FakeSupervised()
    resolver = _resolver(module)
    labels = [True, False, True, False, False, True, False, False, False, False]

    resolver.fit(RECORDS, labels=labels)

    report = resolver.fit_report_
    assert report is not None
    assert report.trained is True
    assert report.trainable == "_FakeSupervised (SupervisedFitMixin)"
    assert report.n_train == len(labels)
    assert report.coverage is None  # pre-aligned labels -> no id-join, no coverage
    assert report.metrics is None
    assert report.threshold == pytest.approx(0.5)


def test_pairs_path_sets_coverage_and_heldout_metrics() -> None:
    module = _FakeSupervised()
    resolver = _resolver(module)
    # Two DISJOINT entity-components (no cross pair to bridge them), so an
    # entity-disjoint split can hold one whole component out as valid.
    records = [
        {"id": "a1", "name": "Acme"},
        {"id": "a2", "name": "Acme"},
        {"id": "a3", "name": "Aardvark"},
        {"id": "b1", "name": "Beta"},
        {"id": "b2", "name": "Beta"},
        {"id": "b3", "name": "Bumble"},
    ]
    pairs = [
        LabeledPair(left_id="a1", right_id="a2", score=None, label=True, source="correction"),
        LabeledPair(left_id="a1", right_id="a3", score=None, label=False, source="correction"),
        LabeledPair(left_id="b1", right_id="b2", score=None, label=True, source="correction"),
        LabeledPair(left_id="b1", right_id="b3", score=None, label=False, source="correction"),
    ]

    resolver.fit(records, pairs=pairs, split=0.5, seed=0)

    report = resolver.fit_report_
    assert report is not None
    assert report.trained is True
    assert report.trainable == "_FakeSupervised (SupervisedFitMixin)"
    assert report.entity_disjoint is True
    assert report.n_train + report.n_valid == 4  # all four labeled pairs aligned
    assert report.n_train > 0 and report.n_valid > 0  # split held one component out
    assert isinstance(report.coverage, GoldCoverage)
    assert report.coverage.gold_coverage == pytest.approx(1.0)  # AllPairs keeps every pair
    # A split was given and a valid component exists, so held-out pair metrics exist.
    assert report.metrics is not None
    assert "Held-out pair metrics" in report.to_markdown()
    # The matcher was actually trained on the train split (label count == n_train).
    assert module.fit_calls
    assert len(module.fit_calls[0][1]) == report.n_train


def test_pairs_path_surfaces_blocker_dropped_positive() -> None:
    """A positive label for a pair with no candidate (unknown id) -> coverage < 1.0."""
    module = _FakeSupervised()
    resolver = _resolver(module)
    pairs = [
        LabeledPair(left_id="a1", right_id="a2", score=None, label=True, source="correction"),
        LabeledPair(left_id="a1", right_id="ghost", score=None, label=True, source="correction"),
    ]

    resolver.fit(RECORDS, pairs=pairs)

    report = resolver.fit_report_
    assert report is not None
    assert report.coverage is not None
    assert report.coverage.gold_coverage == pytest.approx(0.5)
    assert ("a1", "ghost") in report.coverage.dropped_positives


# ---------------------------------------------------------------------------
# Raise contracts around labels=/pairs=.
# ---------------------------------------------------------------------------


def test_both_labels_and_pairs_raises() -> None:
    resolver = _resolver(_FakeSupervised())
    with pytest.raises(ValueError, match="not both"):
        resolver.fit(RECORDS, labels=[True], pairs=[])


def test_pairs_given_to_non_supervised_matcher_raises() -> None:
    resolver = _resolver(_NoHook())
    pairs = [LabeledPair(left_id="a1", right_id="a2", score=None, label=True, source="correction")]
    with pytest.raises(ValueError, match="does not support fit\\(pairs="):
        resolver.fit(RECORDS, pairs=pairs)


def test_supervised_without_labels_or_pairs_still_raises() -> None:
    resolver = _resolver(_FakeSupervised())
    with pytest.raises(ValueError, match="requires labeled data"):
        resolver.fit(RECORDS)


# ---------------------------------------------------------------------------
# Import-light guarantee for the fit_report leaf.
# ---------------------------------------------------------------------------


def test_fit_report_module_stays_import_light() -> None:
    """importing langres.core.fit_report must not pull sklearn/torch/litellm."""
    script = (
        "import sys; import langres.core.fit_report; "
        "leaked = [m for m in ['sklearn', 'torch', 'litellm', 'faiss'] if m in sys.modules]; "
        "assert not leaked, f'fit_report leaked heavy modules: {leaked}'; "
        "print('OK')"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"fit_report import-budget check failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
