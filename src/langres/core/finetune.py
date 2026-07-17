"""Back-compat shim: ``langres.core.finetune`` moved to ``langres.training.finetune``.

# TEMPORARY: deleted by the W2 sweep

Fitting a matcher is what *produces* a tuned model, not entity-resolution
modelling itself, so QLoRA/LoRA training now lives in ``langres.training``
beside ``core`` rather than inside it. Import from ``langres.training.finetune``
(or the ``langres`` facade, which still resolves ``QLoRA`` / ``run_finetune`` /
``finetune`` lazily).

peft/trl/bitsandbytes/torch stay imported lazily by the real module -- inside
``QLoRATrainer.train`` -- never by this shim, so a bare ``import langres`` (and
even importing this shim) pulls none of the ``[finetune]`` stack.
"""

from langres.training.finetune import (
    FINETUNE_YES_NO_PROMPT,
    Conversation,
    FinetuneOutcome,
    FinetuneTrainer,
    LabeledCandidate,
    QLoRA,
    QLoRATrainer,
    TrainOutcome,
    finetune,
    run_finetune,
)

__all__ = [
    "Conversation",
    "FINETUNE_YES_NO_PROMPT",
    "finetune",
    "FinetuneOutcome",
    "FinetuneTrainer",
    "LabeledCandidate",
    "QLoRA",
    "QLoRATrainer",
    "run_finetune",
    "TrainOutcome",
]
