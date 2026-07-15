"""Acceptance test for the fine-tune CAPSTONE: the full PUBLIC train->serve->evaluate flow.

Where ``test_finetune.py::test_finetune_overfit_train_serve_evaluate`` is the
go/no-go for the trainer INTERNALS (hand-built pairs -> run_finetune -> serve),
this is the acceptance of the user-facing CAPSTONE (``examples/finetune_capstone.py``):
every step goes through the PUBLIC surface --

  * ``candidates_for(get_benchmark("fodors_zagat"), split=...)`` for real blocked
    candidate pairs + gold (the eval facade + a real benchmark, not fixtures),
  * ``run_finetune(pairs, QLoRA(...))`` for the model_ref + GPU-seconds,
  * a JSON round-trip of ``to_config(model_ref)`` -> ``normalize_model_ref`` (the
    "weightless ref survives save/reload" claim),
  * ``LLMMatcher(model=ref, ...)`` in-process serve with the finetune prompt,
  * ``evaluate(matcher, held_out, gold, threshold=0.5)`` for the honest fixed cut.

**Why the gate is "beats the untrained base on ranking", not an F1 floor.**
Measured on this real, hard slice (SmolLM2-135M, held-out Fodors-Zagat): the
UNTRAINED base is the trivial always-"Yes" predictor -- F1@0.5 = 0.667 on a
balanced slice, but match/non-match separation ~0.004 and ROC-AUC ~0.583.
Fine-tuning pulls matches and non-matches apart (separation ~0.031, AUC ~0.64)
yet its F1 *at the fixed 0.5 cut* DROPS to ~0.54, because the small model's
Yes-mass clusters near 0.5 and it starts (correctly) saying "No" to some pairs.
So an F1@0.5 floor is a DISHONEST gate here: the do-nothing base clears it while
the model that actually learned to discriminate fails it -- a fixed-threshold
artifact, exactly what a calibrator fixes. The honest, robust learning signal is
threshold-free ranking that BEATS the same untrained base. That is what we gate.

Marked ``slow`` + ``finetune`` so it runs only in the dedicated ``test-finetune``
job / on demand, never in the fast suite (real model downloads + real training).
"""

from __future__ import annotations

# faiss (candidates_for's blocker) and torch (the fine-tune) each ship their own
# OpenMP runtime; loading both in one process can hard-crash on macOS. Allow the
# duplicate load before either is imported. Harmless where it does not apply.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from typing import Any

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.finetune]

BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _balanced(cands: list[Any], gold: set[frozenset[str]], k: int) -> list[tuple[Any, bool]]:
    """Up to ``k`` gold-positive candidates + ``k`` negatives, labeled, stable order."""
    pos = [c for c in cands if frozenset({str(c.left.id), str(c.right.id)}) in gold]
    neg = [c for c in cands if frozenset({str(c.left.id), str(c.right.id)}) not in gold]
    return [(c, True) for c in pos[:k]] + [(c, False) for c in neg[:k]]


def _rank_signal(model_cfg: Any, pairs: list[tuple[Any, bool]]) -> tuple[float, float]:
    """Serve ``model_cfg`` on the held-out slice; return (mean-p_yes separation, ROC-AUC).

    Both are THRESHOLD-FREE ranking signals -- how well the served model pulls
    matches above non-matches -- so they measure learning without the fixed-0.5-cut
    artifact that makes F1 reward the trivial always-"Yes" base (see module docstring).
    """
    from langres.core import LLMMatcher
    from langres.core.finetune import FINETUNE_YES_NO_PROMPT
    from langres.eval import roc_auc_score

    matcher: LLMMatcher[Any] = LLMMatcher(
        model=model_cfg,
        confidence="logprob",
        response_parser="binary_yes_no",
        prompt_template=FINETUNE_YES_NO_PROMPT,
    )
    judgements = list(matcher.forward(iter([c for c, _ in pairs])))
    p_yes = [j.provenance["p_yes"] for j in judgements]
    labels = [y for _, y in pairs]
    pos = [p for p, y in zip(p_yes, labels, strict=True) if y]
    neg = [p for p, y in zip(p_yes, labels, strict=True) if not y]
    separation = sum(pos) / len(pos) - sum(neg) / len(neg)
    return separation, roc_auc_score(labels, p_yes)


def test_capstone_train_serve_evaluate_public_flow() -> None:
    """The whole public capstone runs and LEARNS: fine-tune beats the untrained base.

    Real QLoRA train on a small Fodors-Zagat TRAIN-split slice -> JSON round-trip
    the weightless model_ref -> in-process serve -> rank a disjoint TEST-split
    slice. Asserts the run reports real GPU-seconds AND the fine-tuned model
    separates matches from non-matches *better than the same untrained base* on two
    threshold-free signals (mean-p_yes separation and ROC-AUC). Also runs the public
    ``evaluate(..., threshold=0.5)`` to prove that path composes at the honest cut.
    """
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")

    from langres.core import LLMMatcher
    from langres.core.finetune import FINETUNE_YES_NO_PROMPT, QLoRA, run_finetune
    from langres.core.matchers.model_ref import normalize_model_ref, to_config
    from langres.eval import candidates_for, evaluate, get_benchmark

    bench = get_benchmark("fodors_zagat")

    # TRAIN: block the train split, label from gold, balance a small slice.
    train_cands, train_gold = candidates_for(bench, split="train")
    train_pairs = _balanced(train_cands, train_gold, k=24)
    assert sum(1 for _, y in train_pairs if y) >= 8  # enough positive signal to learn

    # Held-out ranking slice from the disjoint TEST split (entity-disjoint by
    # benchmark construction -> genuine generalization, not memorization).
    test_cands, test_gold = candidates_for(bench, split="test")
    eval_pairs = _balanced(test_cands, test_gold, k=20)

    # BASELINE: the SAME base, untrained, on the same held-out slice. This is the
    # honest floor -- training must beat THIS, not a magic constant.
    base_sep, base_auc = _rank_signal(BASE, eval_pairs)

    method = QLoRA(base=BASE, epochs=8, batch_size=8)
    outcome = run_finetune(train_pairs, method)

    # COST is real and honest: GPU-seconds > 0 on a recorded device; a base+adapter
    # ref (nothing merged); n_train matches what we trained on.
    assert outcome.gpu_seconds > 0.0
    assert outcome.device in {"cuda", "mps", "cpu"}
    assert outcome.model_ref.adapter is not None
    assert outcome.n_train == len(train_pairs)

    # SAVE -> RELOAD the weightless model_ref through JSON (base id + adapter path,
    # no weight blob) and serve the reloaded ref -- proving the ref round-trips.
    reloaded = normalize_model_ref(json.loads(json.dumps(to_config(outcome.model_ref))))
    tuned_sep, tuned_auc = _rank_signal(to_config(reloaded), eval_pairs)

    # LEARNED: the fine-tuned model out-ranks the untrained base on BOTH signals.
    # (Observed: base sep~0.004 auc~0.583 -> tuned sep~0.031 auc~0.64; margins in
    # the PR report. The base barely separates; training pulls the classes apart.)
    assert tuned_sep > base_sep
    assert tuned_auc > base_auc

    # The public evaluate() path also composes at the honest fixed cut (F1@0.5 is
    # low here -- the small model needs a calibrated threshold; see the module
    # docstring -- so we assert the path/threshold, not an F1 magnitude).
    result = evaluate(
        LLMMatcher(
            model=to_config(reloaded),
            confidence="logprob",
            response_parser="binary_yes_no",
            prompt_template=FINETUNE_YES_NO_PROMPT,
        ),
        [c for c, _ in eval_pairs],
        {frozenset({str(c.left.id), str(c.right.id)}) for c, y in eval_pairs if y},
        threshold=0.5,
    )
    assert result.graded_threshold == 0.5
