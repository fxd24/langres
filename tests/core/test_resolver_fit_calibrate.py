"""Tests for the score-calibration fit path + ``Platt``/``Isotonic`` methods (PR-D).

Two seams land here on top of the PR-M ``method=`` dispatch:

- **``fit(method=Platt()/Isotonic())``** learns a score→probability
  :class:`~langres.core.calibration.Calibrator` from labeled pairs (the
  ``method.kind == "calibrate"`` branch, now real), attaches it to the Resolver,
  and applies it in ``predict()``/``resolve()``. Every test is deterministic and
  ``$0`` (a ``WeightedAverageMatcher`` over company names -- no LM calls).
- The concrete ``Platt``/``Isotonic`` :class:`Method` objects are unit-tested
  here too (kind, strategy identity, ``describe()``).

The *quality* proof (calibration lowers Brier/ECE) lives on the ``Calibrator``
itself in ``tests/core/test_calibration.py``; these tests exercise the Resolver
wiring: dispatch, ``describe()``, the calibrated predict path, the ``FitReport``
delta, and the save/load round-trip of a fitted calibrator.
"""

from pathlib import Path

import pytest

from langres.core.harvest import LabeledPair
from langres.core.methods_api import Method
from langres.core.methods_calibrate import CalibrateMethod, Isotonic, Platt
from langres.core.models import CompanySchema
from langres.core.resolver import Resolver

# Six disconnected groups; each is one entity-disjoint component carrying two
# positives (X~X, Y~Y) and one negative (X!Y). An entity-disjoint ``split`` holds
# out WHOLE groups, so both train and valid see both classes -- the shape the
# calibrate fit and its held-out Brier/ECE delta need.
_BASES = [
    ("Acme", "Beta"),
    ("Gamma", "Delta"),
    ("Epsilon", "Zeta"),
    ("Eta", "Theta"),
    ("Iota", "Kappa"),
    ("Lambda", "Mu"),
]


def _dataset() -> tuple[list[dict[str, str]], list[LabeledPair]]:
    records: list[dict[str, str]] = []
    pairs: list[LabeledPair] = []
    for g, (x, y) in enumerate(_BASES):
        x0, x1, y0, y1 = f"g{g}x0", f"g{g}x1", f"g{g}y0", f"g{g}y1"
        records += [
            {"id": x0, "name": f"{x} Corp"},
            {"id": x1, "name": f"{x} Corporation"},
            {"id": y0, "name": f"{y} Inc"},
            {"id": y1, "name": f"{y} Incorporated"},
        ]
        pairs += [
            LabeledPair(left_id=x0, right_id=x1, score=None, label=True, source="correction"),
            LabeledPair(left_id=y0, right_id=y1, score=None, label=True, source="correction"),
            LabeledPair(left_id=x0, right_id=y0, score=None, label=False, source="correction"),
        ]
    return records, pairs


def _resolver() -> Resolver:
    return Resolver.from_schema(CompanySchema, matcher="string", threshold=0.5)


# --- Method objects: kind, strategy identity, describe() --------------------


def test_calibrate_methods_share_kind_calibrate() -> None:
    """Both concrete calibrate methods route through the ``kind == "calibrate"`` branch."""
    assert Platt.kind == "calibrate"
    assert Isotonic.kind == "calibrate"
    assert issubclass(Platt, CalibrateMethod) and issubclass(Isotonic, Method)


def test_strategy_is_classvar_not_a_field() -> None:
    """``strategy`` is strategy-type identity, not serialized per-instance config."""
    assert Platt.strategy == "platt"
    assert Isotonic.strategy == "isotonic"
    assert "strategy" not in Platt.model_fields
    assert "kind" not in Platt.model_fields


def test_describe_names_the_strategy() -> None:
    assert Platt().describe() == "calibrate (Platt scaling)"
    assert Isotonic().describe() == "calibrate (isotonic regression)"


# --- describe(): the calibrator row -----------------------------------------


def test_describe_shows_calibrator_none_before_fit() -> None:
    """A fresh Resolver reports no calibrator, frozen."""
    row = next(
        line for line in _resolver().describe().splitlines() if line.startswith("calibrator")
    )
    assert "<none>" in row and "frozen" in row


def test_describe_shows_calibrator_trainable_after_fit() -> None:
    """Once fitted, the calibrator row names the component and is TRAINABLE."""
    records, pairs = _dataset()
    resolver = _resolver().fit(records, pairs=pairs, method=Platt())
    row = next(line for line in resolver.describe().splitlines() if line.startswith("calibrator"))
    assert "Calibrator" in row and "TRAINABLE" in row


# --- The calibrate fit path --------------------------------------------------


@pytest.mark.parametrize("method_cls", [Platt, Isotonic])
def test_fit_calibrate_attaches_calibrator_and_report(method_cls: type[CalibrateMethod]) -> None:
    """``fit(method=...)`` sets ``calibrator`` + a trained ``fit_report_``."""
    records, pairs = _dataset()
    resolver = _resolver()
    returned = resolver.fit(records, pairs=pairs, method=method_cls())

    assert returned is resolver  # chains
    assert resolver.calibrator is not None
    assert resolver.calibrator.method == method_cls.strategy
    report = resolver.fit_report_
    assert report is not None and report.trained
    assert report.trainable == f"Calibrator ({method_cls.strategy})"
    assert report.n_train > 0


def test_fit_calibrate_reports_brier_ece_before_after_on_valid() -> None:
    """With a split, the report carries the held-out Brier/ECE before-vs-after."""
    records, pairs = _dataset()
    resolver = _resolver().fit(records, pairs=pairs, method=Platt(), split=0.4, seed=0)

    delta = resolver.fit_report_.calibration
    assert delta is not None
    assert delta.method == "platt"
    # The synthetic set is over-confident, so calibration must not worsen it.
    assert delta.brier_after <= delta.brier_before
    assert delta.ece_after <= delta.ece_before
    assert "Calibration (platt" in resolver.fit_report_.to_markdown()


def test_no_split_leaves_calibration_delta_none() -> None:
    """Without a held-out split there is nothing to measure before/after on."""
    records, pairs = _dataset()
    resolver = _resolver().fit(records, pairs=pairs, method=Platt())
    assert resolver.fit_report_.calibration is None


def test_predict_returns_calibrated_probabilities() -> None:
    """After a calibrate fit, ``predict()`` scores are calibrated probs, not raw."""
    records, pairs = _dataset()
    resolver = _resolver()
    raw = {(j.left_id, j.right_id): j.score for j in resolver.predict(records)}

    resolver.fit(records, pairs=pairs, method=Platt())
    calibrated = resolver.predict(records)

    assert calibrated  # non-empty
    changed = 0
    for j in calibrated:
        assert j.score is not None and 0.0 <= j.score <= 1.0
        assert j.score_type == "calibrated_prob"
        assert j.provenance["calibration"]["method"] == "platt"
        if abs(raw[(j.left_id, j.right_id)] - j.score) > 1e-9:
            changed += 1
    assert changed > 0  # calibration actually moved the scores


def test_save_load_round_trips_a_fitted_calibrator(tmp_path: Path) -> None:
    """A fitted calibrator round-trips through save/load with identical predictions."""
    records, pairs = _dataset()
    resolver = _resolver().fit(records, pairs=pairs, method=Isotonic())
    before = {(j.left_id, j.right_id): j.score for j in resolver.predict(records)}

    resolver.save(tmp_path / "artifact")
    reloaded = Resolver.load(tmp_path / "artifact")

    assert reloaded.calibrator is not None
    assert reloaded.calibrator.method == "isotonic"
    after = {(j.left_id, j.right_id): j.score for j in reloaded.predict(records)}
    assert before.keys() == after.keys()
    assert all(abs(before[k] - after[k]) < 1e-12 for k in before)


def test_uncalibrated_resolver_saves_without_calibrator_slot(tmp_path: Path) -> None:
    """No calibrator => no calibrator slot in the manifest (optional slot stays absent)."""
    resolver = _resolver()
    resolver.save(tmp_path / "artifact")
    manifest = (tmp_path / "artifact" / "resolver.json").read_text()
    assert "calibrator" not in manifest
    assert Resolver.load(tmp_path / "artifact").calibrator is None


# --- Error paths -------------------------------------------------------------


def test_calibrate_without_labels_raises() -> None:
    """Calibration needs supervision; neither labels nor pairs is an error."""
    records, _ = _dataset()
    with pytest.raises(ValueError, match="needs gold labels"):
        _resolver().fit(records, method=Platt())


def test_calibrate_with_both_labels_and_pairs_raises() -> None:
    """Passing both supervision channels is a clear error."""
    records, pairs = _dataset()
    with pytest.raises(ValueError, match="either labels|not both"):
        _resolver().fit(records, labels=[True], pairs=pairs, method=Platt())


def test_calibrate_with_non_calibrate_method_raises() -> None:
    """A method whose kind is not 'calibrate' never reaches _fit_calibrate; a
    bare CalibrateMethod (no .strategy) is rejected with an actionable message."""
    records, pairs = _dataset()

    class _Bare(CalibrateMethod):
        """A CalibrateMethod subclass that forgot to set .strategy."""

    with pytest.raises(ValueError, match="needs a CalibrateMethod exposing .strategy"):
        _resolver().fit(records, pairs=pairs, method=_Bare())
