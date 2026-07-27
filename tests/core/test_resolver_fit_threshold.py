"""Tests for ``fit(derive_threshold=True)`` -- measuring the match cut from labels.

``derive_threshold`` was already wired into the harvest loop
(``curation/harvest.py``) and the fixed-split pair benchmark
(``data/fixed_split_pair_benchmark.py``). The gap this closes is narrower: **the
model's own ``fit()`` never called it**, so a user who supplied labels still
resolved at whatever constant the constructor set.

Two things get tested here that a single code path would get wrong:

- **Where the threshold lives.** A classic four-slot model keeps it on
  ``clusterer.threshold``. An explicit ``_ops`` chain has *no clusterer slot at
  all* -- the property raises ``_require_bound`` -- and keeps the cut in a
  ``ThresholdSelect``. Before this change ``fit()`` could not run on an ``_ops``
  model at all (it raises on ``self.module`` first). Both are exercised.
- **Leakage.** Deriving a cut on the rows you then score is in-sample and
  flatters the cut. ``test_derived_cut_beats_the_default_on_held_out_pairs``
  is the honest measurement: the cut comes from ``train`` only and is graded on
  the entity-disjoint ``valid`` split it never saw.

Everything here is ``$0`` and deterministic -- rapidfuzz over company names, no
LM calls, no network.
"""

import sys
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.matchers import RapidfuzzMatcher
from langres.core.metrics import classify_pairs
from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.op import ThresholdSelect, TopKSelect
from langres.core.op_adapters import BlockerSource, ClustererStage, MatcherScore
from langres.core.resolver import ERModel, Resolver
from langres.curation.harvest import LabeledPair, align_pairs, warn_if_silver_only
from langres.training.methods_calibrate import Platt

# Ten disconnected groups. Each is one entity-disjoint component: a twin pair
# that matches, plus a HARD negative sharing the leading token ("Acme
# Corporation" vs "Acme Holdings"). rapidfuzz scores those negatives well above
# 0.5, so the hand-set default cut is genuinely wrong on this data -- which is
# what makes a derived cut measurable rather than decorative.
_GROUPS = [
    ("acme", "Acme Corporation", "Acme Corporation Ltd", "Acme Holdings"),
    ("globex", "Globex Incorporated", "Globex Incorporated Co", "Globex Ventures"),
    ("initech", "Initech Systems", "Initech Systems Ltd", "Initech Capital"),
    ("umbrella", "Umbrella Holdings", "Umbrella Holdings Inc", "Umbrella Pharma"),
    ("soylent", "Soylent Industries", "Soylent Industries Co", "Soylent Foods"),
    ("tyrell", "Tyrell Corporation", "Tyrell Corporation Ltd", "Tyrell Genetics"),
    ("weyland", "Weyland Systems", "Weyland Systems Inc", "Weyland Mining"),
    ("stark", "Stark Industries", "Stark Industries Ltd", "Stark Aerospace"),
    ("oscorp", "Oscorp Laboratories", "Oscorp Laboratories Inc", "Oscorp Realty"),
    ("nakatomi", "Nakatomi Trading", "Nakatomi Trading Co", "Nakatomi Estates"),
]

_DEFAULT_THRESHOLD = 0.5
_SPLIT = 0.4
_SEED = 0


def _dataset() -> tuple[list[dict[str, str]], list[LabeledPair]]:
    records: list[dict[str, str]] = []
    pairs: list[LabeledPair] = []
    for key, twin_a, twin_b, other in _GROUPS:
        a, b, c = f"{key}1", f"{key}2", f"{key}3"
        records += [
            {"id": a, "name": twin_a},
            {"id": b, "name": twin_b},
            {"id": c, "name": other},
        ]
        pairs += [
            LabeledPair(left_id=a, right_id=b, score=None, label=True, source="correction"),
            LabeledPair(left_id=a, right_id=c, score=None, label=False, source="correction"),
            LabeledPair(left_id=b, right_id=c, score=None, label=False, source="correction"),
        ]
    return records, pairs


def _matcher() -> RapidfuzzMatcher[Any]:
    return RapidfuzzMatcher({"name": (lambda entity: entity.name, 1.0)})


def _classic(threshold: float = _DEFAULT_THRESHOLD) -> Resolver:
    """A classic four-slot model: the cut lives on ``clusterer.threshold``."""
    return Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        matcher=_matcher(),
        clusterer=Clusterer(threshold=threshold),
    )


def _explicit(threshold: float = _DEFAULT_THRESHOLD) -> ERModel:
    """An explicit Op chain: the cut lives in the terminal ``ThresholdSelect``.

    Its ``ClustererStage`` clusterer is threshold-free (0.0), matching how the
    shipped research recipes build one, so the ``ThresholdSelect`` is the single
    match cut rather than one of two gates.
    """
    return ERModel.from_topology(
        ops=[
            BlockerSource(AllPairsBlocker(schema=CompanySchema)),
            MatcherScore(_matcher(), out_space="heuristic"),
            ThresholdSelect(threshold),
            ClustererStage(Clusterer(threshold=0.0)),
        ]
    )


class _SupervisedMatcher:
    """A matcher that implements ``SupervisedFitMixin`` and records its training call.

    Structural typing: no subclassing needed for ``isinstance`` against the
    runtime-checkable Protocol, so this is deliberately minimal.
    """

    def __init__(self) -> None:
        self.fit_calls: list[int] = []

    def fit(
        self, candidates: Iterator[ERCandidate[Any]], labels: Sequence[bool]
    ) -> None:
        self.fit_calls.append(len(list(candidates)))

    def forward(self, candidates: Iterator[ERCandidate[Any]]) -> Iterator[PairwiseJudgement]:
        for candidate in candidates:
            left = candidate.left.name
            right = candidate.right.name
            shared = len(set(left.split()) & set(right.split()))
            yield PairwiseJudgement(
                left_id=candidate.left.id,
                right_id=candidate.right.id,
                score=min(1.0, shared / 3.0),
                score_type="heuristic",
                decision_step="supervised_stub",
                provenance={},
            )


# --- The classic seam: clusterer.threshold ----------------------------------


def test_derive_threshold_moves_the_classic_cut_off_its_default() -> None:
    """A derived cut replaces the constructor's constant on ``clusterer.threshold``."""
    records, pairs = _dataset()
    model = _classic()

    model.fit(records, pairs=pairs, split=_SPLIT, seed=_SEED, derive_threshold=True)

    assert model.clusterer.threshold != _DEFAULT_THRESHOLD
    assert 0.0 <= model.clusterer.threshold <= 1.0


def test_report_records_derived_provenance_for_the_classic_seam() -> None:
    """``threshold_fit`` says derived, by what, from how many, held-out, and where."""
    records, pairs = _dataset()
    model = _classic()

    model.fit(records, pairs=pairs, split=_SPLIT, seed=_SEED, derive_threshold=True)
    report = model.fit_report_
    assert report is not None and report.threshold_fit is not None
    fit = report.threshold_fit

    assert fit.source == "derived"
    assert fit.method == "youden"
    assert fit.n_pairs == report.n_train
    assert fit.held_out is True
    assert fit.applied_to == "clusterer"
    assert fit.previous == _DEFAULT_THRESHOLD
    assert report.threshold == model.clusterer.threshold


def test_default_leaves_the_threshold_untouched_and_says_so() -> None:
    """``derive_threshold=False`` (the default) is a no-op the report still reports.

    The constructed threshold is the correct no-data fallback; what would be
    wrong is letting a reader mistake it for a measurement.
    """
    records, pairs = _dataset()
    model = _classic()

    # A fixed scorer needs derive_threshold=True to accept pairs= at all, so the
    # honest "not derived" case is read off a supervised matcher.
    supervised = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        matcher=_SupervisedMatcher(),  # type: ignore[arg-type]
        clusterer=Clusterer(threshold=_DEFAULT_THRESHOLD),
    )
    supervised.fit(records, pairs=pairs, split=_SPLIT, seed=_SEED)

    assert supervised.clusterer.threshold == _DEFAULT_THRESHOLD
    assert supervised.fit_report_ is not None
    assert supervised.fit_report_.threshold_fit is not None
    assert supervised.fit_report_.threshold_fit.source == "default"
    assert supervised.fit_report_.threshold_fit.n_pairs == 0
    assert model.clusterer.threshold == _DEFAULT_THRESHOLD  # untouched, never fitted


# --- Leakage: the number that matters is the one the cut never saw ----------


def test_derived_cut_beats_the_default_on_held_out_pairs() -> None:
    """The measured claim, held-out: derived on ``train``, graded on ``valid``.

    The cut is derived from the train split ONLY and applied before valid is
    graded, so this comparison contains no in-sample leakage. On this fixture the
    hand-set 0.5 admits the hard negatives; the derived cut excludes them.
    """
    records, pairs = _dataset()
    model = _classic()

    model.fit(records, pairs=pairs, split=_SPLIT, seed=_SEED, derive_threshold=True)
    derived_cut = model.clusterer.threshold

    # Rebuild the exact same deterministic split and score it at both cuts.
    reference = _classic()
    aligned = align_pairs(reference.candidates(records), pairs, split=_SPLIT, seed=_SEED)
    assert aligned.valid.candidates, "fixture must produce a non-empty held-out split"
    judgements = list(reference._scorer().forward(iter(aligned.valid.candidates)))
    gold = {
        frozenset({str(c.left.id), str(c.right.id)})
        for c, label in zip(aligned.valid.candidates, aligned.valid.labels, strict=True)
        if label
    }

    before = classify_pairs(judgements, gold, _DEFAULT_THRESHOLD)
    after = classify_pairs(judgements, gold, derived_cut)

    assert after.f1 > before.f1
    # The report's own held-out metrics ARE the after-cut number, not a
    # separately computed one.
    assert model.fit_report_ is not None and model.fit_report_.metrics is not None
    assert model.fit_report_.metrics.f1 == pytest.approx(after.f1)
    assert model.fit_report_.metrics.threshold == pytest.approx(derived_cut)


def test_without_a_split_the_cut_is_in_sample_and_no_metrics_are_reported() -> None:
    """No ``split`` -> the cut is derived honestly, flagged IN-SAMPLE, and ungraded.

    Reporting P/R/F1 on the same rows the cut came from would manufacture a large
    fake improvement; refusing to compute it is the point.
    """
    records, pairs = _dataset()
    model = _classic()

    model.fit(records, pairs=pairs, split=None, derive_threshold=True)
    report = model.fit_report_
    assert report is not None and report.threshold_fit is not None

    assert report.threshold_fit.source == "derived"
    assert report.threshold_fit.held_out is False
    assert report.metrics is None
    assert "IN-SAMPLE" in report.to_markdown()


def test_markdown_never_shows_a_bare_number_for_a_derived_or_default_cut() -> None:
    """The rendered threshold carries its provenance in the same line."""
    records, pairs = _dataset()
    model = _classic()
    model.fit(records, pairs=pairs, split=_SPLIT, seed=_SEED, derive_threshold=True)
    assert model.fit_report_ is not None
    line = next(
        row for row in model.fit_report_.to_markdown().splitlines() if row.startswith("- Threshold")
    )
    assert "derived" in line and "youden" in line and "held-out" in line


# --- The explicit _ops seam: ThresholdSelect --------------------------------


def test_derive_threshold_writes_the_chains_threshold_select() -> None:
    """An explicit chain has no clusterer slot; the cut is its ``ThresholdSelect``."""
    records, pairs = _dataset()
    model = _explicit()

    model.fit(records, pairs=pairs, split=_SPLIT, seed=_SEED, derive_threshold=True)

    assert model._chain_threshold() != _DEFAULT_THRESHOLD
    assert model.fit_report_ is not None and model.fit_report_.threshold_fit is not None
    assert model.fit_report_.threshold_fit.applied_to == "threshold_select"
    assert model.fit_report_.threshold_fit.previous == _DEFAULT_THRESHOLD
    # The seam a classic model uses still raises here -- which is exactly why a
    # single code path would have been wrong.
    with pytest.raises(RuntimeError, match="explicit, slot-neutral Op topology"):
        _ = model.clusterer


def test_explicit_chain_derivation_sees_rows_below_the_current_cut() -> None:
    """Derivation runs the chain with the fitted ``ThresholdSelect`` omitted.

    Deriving from ``_scored_pairs`` instead would only ever see rows that already
    cleared today's cut, so the "derived" threshold could not move below it. Start
    from a cut high enough to delete every row and check the fit still recovers a
    lower one.
    """
    records, pairs = _dataset()
    model = _explicit(threshold=0.99)
    assert model._prethreshold_pairs(records).rows, "the pre-cut pass must keep rows"

    model.fit(records, pairs=pairs, split=_SPLIT, seed=_SEED, derive_threshold=True)

    assert model._chain_threshold() is not None
    assert model._chain_threshold() < 0.99


def test_explicit_chain_keeps_upstream_selects_when_deriving() -> None:
    """Only the fitted ThresholdSelect is omitted -- an upstream TopKSelect still runs."""
    records, _ = _dataset()
    model = ERModel.from_topology(
        ops=[
            BlockerSource(AllPairsBlocker(schema=CompanySchema)),
            MatcherScore(_matcher(), out_space="heuristic"),
            TopKSelect(1),
            ThresholdSelect(_DEFAULT_THRESHOLD),
            ClustererStage(Clusterer(threshold=0.0)),
        ]
    )
    rows = model._prethreshold_pairs(records).rows
    by_left: dict[str, int] = {}
    for row in rows:
        by_left[row.left_id] = by_left.get(row.left_id, 0) + 1
    assert by_left and max(by_left.values()) == 1


def test_explicit_chain_fit_without_derive_threshold_raises_a_directed_error() -> None:
    """``fit()`` on an Op chain says what it CAN fit instead of failing obscurely."""
    records, pairs = _dataset()
    with pytest.raises(ValueError, match="can only fit the chain's ThresholdSelect"):
        _explicit().fit(records, pairs=pairs)


def test_explicit_chain_without_a_threshold_select_refuses_to_derive() -> None:
    """No cut to write -> raise, rather than derive a number and drop it."""
    records, pairs = _dataset()
    model = ERModel.from_topology(
        ops=[
            BlockerSource(AllPairsBlocker(schema=CompanySchema)),
            MatcherScore(_matcher(), out_space="heuristic"),
            ClustererStage(Clusterer(threshold=0.0)),
        ]
    )
    with pytest.raises(ValueError, match="contains no ThresholdSelect"):
        model.fit(records, pairs=pairs, split=_SPLIT, seed=_SEED, derive_threshold=True)


def test_threshold_seam_reports_where_each_topology_keeps_its_cut() -> None:
    """The reader half of the seam, so a caller need not branch on ``_ops`` itself."""
    assert _classic()._threshold_seam() == "clusterer"
    assert _classic()._match_threshold() == _DEFAULT_THRESHOLD
    assert _explicit()._threshold_seam() == "threshold_select"
    assert _explicit()._match_threshold() == _DEFAULT_THRESHOLD


def test_set_match_threshold_refuses_an_unbound_model() -> None:
    """Writing nowhere and returning would be the silent no-op the seam prevents."""
    model = ERModel.__new__(ERModel)
    model._init_state(budget_usd=None)
    assert model._threshold_seam() is None
    with pytest.raises(RuntimeError, match="not bound to a schema"):
        model._set_match_threshold(0.7)


def test_set_match_threshold_refuses_a_chain_with_no_threshold_select() -> None:
    """Same guard on the other seam: an Op chain with no cut has nowhere to write."""
    model = ERModel.from_topology(
        ops=[
            BlockerSource(AllPairsBlocker(schema=CompanySchema)),
            MatcherScore(_matcher(), out_space="heuristic"),
            ClustererStage(Clusterer(threshold=0.0)),
        ]
    )
    assert model._threshold_seam() is None
    assert model._match_threshold() is None
    with pytest.raises(RuntimeError, match="contains no ThresholdSelect"):
        model._set_match_threshold(0.7)


def test_set_match_threshold_writes_the_last_threshold_select() -> None:
    """Reader and writer agree on WHICH select owns the cut when a chain has two."""
    first, last = ThresholdSelect(0.1), ThresholdSelect(0.2)
    model = ERModel.from_topology(
        ops=[
            BlockerSource(AllPairsBlocker(schema=CompanySchema)),
            MatcherScore(_matcher(), out_space="heuristic"),
            first,
            last,
            ClustererStage(Clusterer(threshold=0.0)),
        ]
    )
    assert model._match_threshold() == 0.2

    model._set_match_threshold(0.75)

    assert last.threshold == 0.75
    assert first.threshold == 0.1
    assert model._match_threshold() == 0.75


# --- Which matchers may fit a threshold -------------------------------------


def test_a_matcher_with_no_fit_hook_can_still_fit_its_threshold() -> None:
    """The case that matters most: a fixed scorer has nothing else to fit.

    ``fit(pairs=...)`` alone still refuses it (unchanged), because without
    ``derive_threshold`` there genuinely is nothing to do with those labels.
    """
    records, pairs = _dataset()
    with pytest.raises(ValueError, match="does not support fit\\(pairs=\\.\\.\\.\\)"):
        _classic().fit(records, pairs=pairs)

    model = _classic()
    model.fit(records, pairs=pairs, split=_SPLIT, seed=_SEED, derive_threshold=True)
    report = model.fit_report_
    assert report is not None and report.threshold_fit is not None
    # ``trained`` keeps its meaning -- "a matcher fit hook ran" -- and no hook
    # ran here. The derived cut is reported by ``threshold_fit``, not by lying.
    assert report.trained is False
    assert report.trainable is not None and "threshold only" in report.trainable
    assert report.threshold_fit.source == "derived"


def test_a_supervised_matcher_trains_and_derives_in_one_fit() -> None:
    """Deriving does not displace the matcher fit hook; both run, hook first."""
    records, pairs = _dataset()
    matcher = _SupervisedMatcher()
    model = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        matcher=matcher,  # type: ignore[arg-type]
        clusterer=Clusterer(threshold=_DEFAULT_THRESHOLD),
    )

    model.fit(records, pairs=pairs, split=_SPLIT, seed=_SEED, derive_threshold=True)

    assert matcher.fit_calls, "the supervised hook must still run"
    assert model.fit_report_ is not None
    assert model.fit_report_.trained is True
    assert model.fit_report_.trainable == "_SupervisedMatcher (SupervisedFitMixin)"
    assert model.fit_report_.threshold_fit is not None
    assert model.fit_report_.threshold_fit.source == "derived"


def test_resolve_actually_uses_the_derived_cut() -> None:
    """End to end: the derived threshold changes what ``resolve()`` merges.

    At 0.5 the hard negatives merge into their twins' clusters; at the derived cut
    they do not.
    """
    records, pairs = _dataset()
    before = _classic().resolve(records)

    model = _classic()
    model.fit(records, pairs=pairs, split=_SPLIT, seed=_SEED, derive_threshold=True)
    after = model.resolve(records)

    assert before != after
    assert all(len(cluster) == 2 for cluster in after)


# --- Guards ------------------------------------------------------------------


def test_derive_threshold_requires_pairs_not_labels() -> None:
    """``labels=`` carries no split, so it could only ever give an in-sample cut."""
    records, _ = _dataset()
    with pytest.raises(ValueError, match="needs pairs="):
        _classic().fit(records, derive_threshold=True)
    with pytest.raises(ValueError, match="needs pairs="):
        _classic().fit(records, labels=[True], derive_threshold=True)


def test_derive_threshold_is_refused_alongside_method() -> None:
    """A ``method=`` fit owns its own report -- and calibrate changes the scale."""
    records, pairs = _dataset()
    with pytest.raises(ValueError, match="not supported"):
        _classic().fit(records, pairs=pairs, method=Platt(), derive_threshold=True)


def test_silver_only_labels_still_warn_on_the_fit_path() -> None:
    """The circularity guard survives the trip through ``fit``.

    ``fit`` restamps its labeled pairs as gold before handing them to
    ``derive_threshold_from_pairs`` (they were asserted by the caller, not read
    off the judge's verdicts), which would otherwise silently suppress the
    silver-only warning -- so ``fit`` checks the raw input itself.
    """
    records, pairs = _dataset()
    silver = [pair.model_copy(update={"source": "verdict"}) for pair in pairs]
    with pytest.warns(UserWarning, match="silver-only calibration is circular"):
        _classic().fit(records, pairs=silver, split=_SPLIT, seed=_SEED, derive_threshold=True)


def test_gold_labels_do_not_trigger_the_silver_warning() -> None:
    """Caller-asserted labels are not the judge's own verdicts."""
    records, pairs = _dataset()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _classic().fit(records, pairs=pairs, split=_SPLIT, seed=_SEED, derive_threshold=True)


@pytest.mark.parametrize(
    "supplied",
    [
        pytest.param("corrections.jsonl", id="path-str"),
        pytest.param(Path("corrections.jsonl"), id="path-object"),
        pytest.param([], id="empty"),
    ],
)
def test_silver_guard_ignores_inputs_that_carry_no_source(supplied: Any) -> None:
    """Only ``LabeledPair`` carries ``source``; a file or a Correction has no silver case."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warn_if_silver_only(supplied)


def test_missing_sklearn_raises_a_directed_error_and_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A core-only install must fail loudly, not silently keep the default cut.

    ``training.calibration`` imports scikit-learn at MODULE scope, so on a
    core-only install the failure is an ImportError at that import, not a
    numerical one later. ``None`` in ``sys.modules`` reproduces exactly that.
    """
    records, pairs = _dataset()
    model = _classic()
    monkeypatch.setitem(sys.modules, "langres.training.calibration", None)

    with pytest.raises(ImportError) as excinfo:
        model.fit(records, pairs=pairs, split=_SPLIT, seed=_SEED, derive_threshold=True)

    message = str(excinfo.value)
    assert "scikit-learn" in message  # problem
    assert "'trained' extra" in message  # cause
    assert "pip install 'langres[trained]'" in message  # fix
    # No silent fallback: the threshold is exactly what it was.
    assert model.clusterer.threshold == _DEFAULT_THRESHOLD


# --- Score scale --------------------------------------------------------------


def test_derivation_uses_the_calibrated_scale_when_a_calibrator_is_fitted() -> None:
    """``resolve()`` thresholds calibrated scores, so the cut must be derived on them.

    Deriving from raw matcher scores would place the cut on a different scale than
    the one it is later compared against.
    """
    pytest.importorskip("sklearn", reason="requires the [trained] extra")
    records, pairs = _dataset()
    model = _classic()
    model.fit(records, pairs=pairs, split=_SPLIT, seed=_SEED, method=Platt())
    assert model.calibrator is not None

    raw = [j.score for j in model._scorer().forward(iter(model.candidates(records)))]
    calibrated = model._cut_scale_scores(
        list(model._scorer().forward(iter(model.candidates(records))))
    )
    assert calibrated != raw

    model.fit(records, pairs=pairs, split=_SPLIT, seed=_SEED, derive_threshold=True)
    derived = model.clusterer.threshold
    assert min(s for s in calibrated if s is not None) <= derived
    assert derived <= max(s for s in calibrated if s is not None)


def test_cut_scale_scores_passes_through_without_a_calibrator() -> None:
    """No calibrator -> the raw judgement scores are already the cut's scale."""
    records, _ = _dataset()
    model = _classic()
    judgements = list(model._scorer().forward(iter(model.candidates(records))))
    assert model._cut_scale_scores(judgements) == [j.score for j in judgements]


def test_cut_scale_scores_keeps_missing_judgements_as_none() -> None:
    """A candidate the matcher returned nothing for stays ``None``, never a fake 0.0."""
    assert _classic()._cut_scale_scores([None]) == [None]


def test_held_out_metrics_are_graded_on_the_same_scale_as_the_derived_cut() -> None:
    """A calibrator moves the scale; the cut and the grader must move together.

    Grading raw scores against a cut derived on calibrated ones compares two
    different scales and reports a number that looks fine and means nothing.
    """
    pytest.importorskip("sklearn", reason="requires the [trained] extra")
    records, pairs = _dataset()
    model = _classic()
    model.fit(records, pairs=pairs, split=_SPLIT, seed=_SEED, method=Platt())
    model.fit(records, pairs=pairs, split=_SPLIT, seed=_SEED, derive_threshold=True)

    assert model.fit_report_ is not None and model.fit_report_.metrics is not None
    assert model.fit_report_.metrics.threshold == pytest.approx(model.clusterer.threshold)

    raw = list(model._scorer().forward(iter(model.candidates(records))))
    rescaled = model._on_cut_scale(raw)
    # Copies, on the calibrated scale -- the caller's judgements are untouched.
    assert [j.score for j in rescaled] != [j.score for j in raw]
    assert [j.left_id for j in rescaled] == [j.left_id for j in raw]


# --- Persistence: what survives, and what does not ---------------------------


def test_the_derived_threshold_survives_save_load_but_its_provenance_does_not(
    tmp_path: Path,
) -> None:
    """The value is model state; the provenance is a fit-time artifact.

    ``fit_report_`` is deliberately never serialized, so a reloaded model carries
    a measured cut with no record that it was measured. Stated here rather than
    left undefined. (Built via ``from_schema`` because the raw ``RapidfuzzMatcher``
    used elsewhere in this file carries no registry ``type_name`` and so cannot be
    saved at all.)
    """
    records, pairs = _dataset()
    model = Resolver.from_schema(CompanySchema, matcher="string", threshold=_DEFAULT_THRESHOLD)
    model.fit(records, pairs=pairs, split=_SPLIT, seed=_SEED, derive_threshold=True)
    derived = model.clusterer.threshold

    model.save(tmp_path / "model")
    reloaded = Resolver.load(tmp_path / "model")

    assert reloaded.clusterer.threshold == derived
    assert reloaded.fit_report_ is None
