"""The training surface at the root: the pieces that make ``Resolver.fit`` legible.

The method objects ``Method`` / ``Bootstrap`` / ``MIPRO`` / ``GEPA`` (prompt)
and ``Platt`` / ``Isotonic`` (calibrate), the ``align_pairs``
pairs->candidates bridge, and the ``FitReport`` digest are all import-light
config/primitives (dspy/scikit-learn stay lazy inside their fit paths), so they
are eager.

See ``langres._exports`` for the fragment contract.
"""

from typing import TYPE_CHECKING

from langres.core import (
    Bootstrap,
    FitReport,
    GEPA,
    Isotonic,
    MIPRO,
    Method,
    Platt,
    align_pairs,
)

if TYPE_CHECKING:
    # Never executed at runtime -- keeps the lazy names visible to `mypy --strict`
    # without pulling scikit-learn into a bare `import langres`.
    from langres.core.calibration import derive_threshold

__all__ = [
    "align_pairs",
    "Bootstrap",
    "FitReport",
    "GEPA",
    "Isotonic",
    "Method",
    "MIPRO",
    "Platt",
]

#: ``derive_threshold`` imports scikit-learn at module scope (the ``[trained]``
#: extra). The finetune surface is import-light instead (peft/trl/torch import
#: lazily inside ``QLoRATrainer.train``), so those symbols carry no extra here
#: -- an ImportError from importing them is a genuine bug, and the actionable
#: "pip install langres[finetune]" hint is raised at train time.
LAZY_SYMBOLS: dict[str, str] = {
    "derive_threshold": "langres.core.calibration",
    "QLoRA": "langres.core.finetune",
    "finetune": "langres.core.finetune",
    "run_finetune": "langres.core.finetune",
}

EXTRA_BY_SYMBOL: dict[str, str] = {
    "derive_threshold": "trained",
}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SYMBOLS)
