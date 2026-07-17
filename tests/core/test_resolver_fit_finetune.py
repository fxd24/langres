"""Tests for ``Resolver.fit(..., method=QLoRA(...))`` -- the fine-tune surface (PR-F).

``fit`` dispatches ``kind="finetune"`` to ``_fit_finetune``, which aligns the
labeled supervision, delegates training to
:func:`~langres.training.finetune.run_finetune` (here a *monkeypatched fake trainer*,
so no GPU / no peft/trl), repoints the Resolver's matcher at the produced
``model_ref`` as an in-process logprob :class:`LLMMatcher`, and records the
``model_ref`` + GPU-seconds + held-out pair metrics in ``fit_report_``.

These lock that wiring with a fake trainer; the REAL peft/trl training is the CPU
dry-run in ``tests/core/test_finetune.py`` (the ``test-finetune`` job).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.training.finetune import TrainOutcome
from langres.curation.harvest import LabeledPair
from langres.core.matchers.llm_judge import LLMMatcher
from langres.core.methods_api import Method
from langres.core.models import CompanySchema, PairwiseJudgement
from langres.core.resolver import Resolver

# a-b, c-d, e-f are three entity-disjoint matching pairs (all-pairs blocking
# proposes each), so an entity-disjoint held-out split has whole pairs to hold.
RECORDS = [
    {"id": "a", "name": "Acme Corp"},
    {"id": "b", "name": "Acme Corporation"},
    {"id": "c", "name": "Beta Inc"},
    {"id": "d", "name": "Beta Incorporated"},
    {"id": "e", "name": "Cyan LLC"},
    {"id": "f", "name": "Cyan Limited"},
]


def _resolver() -> Resolver:
    """A Resolver whose matcher is an LLMMatcher (so ``_llm_render_config`` fires)."""
    return Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        matcher=LLMMatcher(client=object(), model="gpt-5-mini"),
        clusterer=Clusterer(threshold=0.5),
    )


@pytest.fixture
def fake_train_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the default ``QLoRATrainer`` with a fake -- captures calls, no GPU.

    ``run_finetune`` instantiates the module-global ``QLoRATrainer`` when no
    trainer is injected (the ``Resolver._fit_finetune`` path), so patching it here
    exercises the real orchestration end-to-end without importing peft/trl.
    """
    calls: list[dict[str, Any]] = []

    class _FakeTrainer:
        def train(
            self, base: str, conversations: list[Any], method: Any, output_dir: str
        ) -> TrainOutcome:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            calls.append({"base": base, "conversations": conversations, "output_dir": output_dir})
            return TrainOutcome(
                adapter_dir=output_dir,
                train_seconds=8.0,
                n_train=len(conversations),
                merged=False,
                device="cpu",
            )

    monkeypatch.setattr("langres.training.finetune.QLoRATrainer", _FakeTrainer)
    return calls


def _predict_all_match(self: Any, candidates: Any) -> Any:
    """A fake ``LLMMatcher.forward`` that scores every valid candidate a match."""
    for cand in candidates:
        yield PairwiseJudgement(
            left_id=str(cand.left.id),
            right_id=str(cand.right.id),
            score=1.0,
            score_type="prob_llm",
            decision_step="fake",
            provenance={},
        )


class _NotQLoRA(Method):
    """A ``kind="finetune"`` method that is NOT a ``QLoRA`` -- the type-guard input."""

    kind: ClassVar[str] = "finetune"

    def describe(self) -> str:
        return "not-a-qlora finetune"


# --- labels= path: no split, repoint the matcher, report ref + GPU-seconds ----


def test_labels_path_trains_repoints_and_reports(
    fake_train_calls: list[dict[str, Any]],
) -> None:
    """``fit(labels=...)`` fine-tunes, repoints the matcher at the ref, fills the report."""
    from langres.training.finetune import QLoRA

    resolver = _resolver()

    result = resolver.fit(
        [{"id": "a", "name": "Acme Corp"}, {"id": "b", "name": "Acme Corporation"}],
        labels=[True],  # the single a-b candidate is a match
        method=QLoRA(base="tiny/model", gpu_hourly_usd=4.0),
    )

    assert result is resolver
    report = resolver.fit_report_
    assert report is not None
    assert report.trained is True
    assert "finetune" in report.trainable and "tiny/model" in report.trainable
    assert report.n_train == 1
    # unmerged QLoRA -> a base+adapter dict ref
    assert isinstance(report.model_ref, dict)
    assert report.model_ref["base"] == "tiny/model"
    assert report.gpu_seconds == 8.0
    assert report.cost == pytest.approx(8.0 / 3600.0 * 4.0)
    # matcher is now an in-process logprob yes/no LLMMatcher over the produced ref
    assert isinstance(resolver.module, LLMMatcher)
    assert resolver.module.model_ref.base == "tiny/model"
    assert resolver.module.model_ref.adapter is not None
    assert resolver.module.confidence == "logprob"
    assert resolver.module.config["response_parser"] == "binary_yes_no"
    # the fake trainer actually saw the one rendered conversation
    assert len(fake_train_calls) == 1
    assert fake_train_calls[0]["base"] == "tiny/model"
    assert len(fake_train_calls[0]["conversations"]) == 1


# --- pairs= path: entity-disjoint held-out split + scored valid metrics -------


def test_pairs_path_holds_out_valid_and_scores_it(
    fake_train_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``fit(pairs=, split=)`` reports coverage, a held-out split, and valid metrics."""
    from langres.training.finetune import QLoRA

    monkeypatch.setattr(LLMMatcher, "forward", _predict_all_match)
    resolver = _resolver()
    pairs = [
        LabeledPair(left_id="a", right_id="b", score=None, label=True, source="correction"),
        LabeledPair(left_id="c", right_id="d", score=None, label=True, source="correction"),
        LabeledPair(left_id="e", right_id="f", score=None, label=True, source="correction"),
    ]

    resolver.fit(RECORDS, pairs=pairs, split=0.5, seed=0, method=QLoRA(base="tiny/model"))

    report = resolver.fit_report_
    assert report is not None
    assert report.split == 0.5
    assert report.seed == 0
    assert report.n_valid >= 1
    assert report.coverage is not None
    assert report.coverage.gold_coverage == pytest.approx(1.0)  # every positive has a candidate
    assert report.metrics is not None
    assert report.metrics.recall == pytest.approx(1.0)  # fake scores every valid pair a match


# --- guards: QLoRA required; supervision required and exclusive ---------------


def test_non_qlora_finetune_method_raises_typeerror() -> None:
    """A ``kind="finetune"`` method that is not a ``QLoRA`` is a clear TypeError."""
    with pytest.raises(TypeError, match=r"requires a QLoRA method.*not-a-qlora finetune"):
        _resolver().fit([{"id": "a", "name": "A"}], method=_NotQLoRA())


def test_both_labels_and_pairs_raises() -> None:
    """Fine-tuning takes labels= XOR pairs=, never both."""
    from langres.training.finetune import QLoRA

    with pytest.raises(ValueError, match="not both"):
        _resolver().fit(
            [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
            labels=[True],
            pairs=[
                LabeledPair(left_id="a", right_id="b", score=None, label=True, source="correction")
            ],
            method=QLoRA(base="tiny/model"),
        )


def test_neither_labels_nor_pairs_raises() -> None:
    """Fine-tuning needs supervision -- no labels and no pairs is a clear error."""
    from langres.training.finetune import QLoRA

    with pytest.raises(ValueError, match="needs labeled supervision"):
        _resolver().fit([{"id": "a", "name": "A"}], method=QLoRA(base="tiny/model"))
