"""The flywheel loop, end to end at the root.

``JudgementLog`` (the inlet -- wire via ``log=`` on ``link``/``dedupe``),
``select_for_review`` / ``ReviewQueue`` (pick the uncertain margin),
``Correction`` / ``CorrectionLog`` (the human labels the ``langres import-csv``
CLI writes), ``harvest_labeled_pairs`` + ``derive_threshold_from_pairs``
(labels -> a tuned threshold), and ``gold_pairs_from_clusters`` +
``EvalReport`` (grade a run against gold at $0). See
``examples/flywheel_min.py`` for the whole loop in one script.

The harvest surface is already in the eager import graph (via ``langres.core``)
so it is eager here for free. ``EvalReport`` / ``gold_pairs_from_clusters`` are
import-light but live in modules kept out of the eager graph on purpose
(``report.eval_report`` pulls ``core.benchmark``/``core.metrics``), so they stay
lazy -- they need no extra, and an ImportError from them is a genuine bug that
must propagate unchanged.

See ``langres._exports`` for the fragment contract.
"""

from typing import TYPE_CHECKING

from langres.core import (
    Correction,
    CorrectionLog,
    JudgementLog,
    ReviewQueue,
    derive_threshold_from_pairs,
    harvest_labeled_pairs,
    select_for_review,
)

if TYPE_CHECKING:
    # Never executed at runtime -- keeps the lazy names visible to `mypy --strict`
    # without pulling the eval-report/benchmark modules into a bare
    # `import langres`.
    from langres.core.benchmark import gold_pairs_from_clusters
    from langres.report.eval_report import EvalReport

__all__ = [
    "Correction",
    "CorrectionLog",
    "derive_threshold_from_pairs",
    "harvest_labeled_pairs",
    "JudgementLog",
    "ReviewQueue",
    "select_for_review",
]

LAZY_SYMBOLS: dict[str, str] = {
    "EvalReport": "langres.report.eval_report",
    "gold_pairs_from_clusters": "langres.core.benchmark",
}

#: Neither needs an extra -- see the module docstring.
EXTRA_BY_SYMBOL: dict[str, str] = {}

NAMES: tuple[str, ...] = (*__all__, *LAZY_SYMBOLS)
