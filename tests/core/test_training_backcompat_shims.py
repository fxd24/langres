"""Back-compat: the W2-sweep shims at the old ``langres.core.*`` fit-family paths.

The fit family moved to ``langres.training`` (see CHANGELOG ``[Unreleased]``).
Temporary shims at the old ``langres.core.*`` paths keep deep imports working
until the W2 sweep deletes them. This proves each old path still resolves to the
*same object* as its new home -- the shim's whole contract -- and, because
nothing in-repo imports the old paths anymore, it is also what covers the shim
modules under the ``core`` contract-coverage gate.

Mirrors the ``langres.core.matchers.model_ref`` shim's own back-compat test.
"""

from __future__ import annotations

import importlib

import pytest

#: ``(old langres.core path, new langres.training path, re-exported names)``.
#: ``calibration`` is excluded here -- it imports scikit-learn at module scope,
#: so it needs the ``[trained]`` extra and gets its own skip-guarded test below.
_SHIMS = [
    (
        "langres.core.fit_report",
        "langres.training.fit_report",
        ["CalibrationDelta", "FitReport"],
    ),
    (
        "langres.core.methods_prompt",
        "langres.training.methods_prompt",
        ["Bootstrap", "GEPA", "MIPRO", "PromptMethod"],
    ),
    (
        "langres.core.methods_calibrate",
        "langres.training.methods_calibrate",
        ["CalibrateMethod", "Isotonic", "Platt"],
    ),
    (
        "langres.core.finetune",
        "langres.training.finetune",
        [
            "Conversation",
            "FINETUNE_YES_NO_PROMPT",
            "FinetuneOutcome",
            "FinetuneTrainer",
            "LabeledCandidate",
            "QLoRA",
            "QLoRATrainer",
            "TrainOutcome",
            "finetune",
            "run_finetune",
        ],
    ),
]


@pytest.mark.parametrize(("old", "new", "names"), _SHIMS)
def test_shim_reexports_are_the_new_objects(old: str, new: str, names: list[str]) -> None:
    """Every name on the old ``core`` path is identity-equal to the new one."""
    old_mod = importlib.import_module(old)
    new_mod = importlib.import_module(new)
    for name in names:
        assert getattr(old_mod, name) is getattr(new_mod, name), name
    assert sorted(old_mod.__all__) == sorted(names)


def test_calibration_shim_reexports_the_new_objects() -> None:
    """The calibration shim needs ``[trained]`` (scikit-learn at module scope)."""
    pytest.importorskip("sklearn", reason="langres.core.calibration needs the [trained] extra")
    old_mod = importlib.import_module("langres.core.calibration")
    new_mod = importlib.import_module("langres.training.calibration")
    for name in ("Calibrator", "derive_threshold", "ThresholdMethod"):
        assert getattr(old_mod, name) is getattr(new_mod, name), name
    assert sorted(old_mod.__all__) == ["Calibrator", "ThresholdMethod", "derive_threshold"]
