"""The flywheel's harvest half: logged verdicts + human corrections -> labeled pairs.

``JudgementLog`` (W0.2, :mod:`langres.core.judgement_log`) is the flywheel's
*inlet* -- an opt-in JSONL log of every judge call (ids, score, verdict, model,
cost, ``"v": 1``). This module is the *outlet*: it turns those logged judgements,
plus a review tool's human corrections, into **labeled pairs** that feed
:func:`langres.core.calibration.derive_threshold` (its first production caller)
and, where applicable, a judge's ``fit()``.

The division of labor is deliberate: langres owns the **contract** and the
**harvest**; the human-review UX (the queue a reviewer clicks through) stays in
the downstream application (e.g. brainsquad). :class:`Correction` is the stable
line schema that review tool writes to ``corrections.jsonl``; :class:`CorrectionLog`
is the reference reader/writer for that file (mirroring ``JudgementLog`` so the
two flywheel files are handled the same way); :func:`harvest_labeled_pairs` is
the merge -- verdicts as weak labels, corrections overriding them where a human
has reviewed the pair.

Weak-label provenance survives onto every :class:`LabeledPair` via ``source``
(``"verdict"`` vs ``"correction"``), so a caller can weight, filter, or audit the
human-reviewed subset instead of trusting all labels equally.

Positioning: like :mod:`langres.core.calibration`, this is eval/calibration-tier
code, not part of the ``link()``/``dedupe()`` runtime. Importing the module is
cheap (Pydantic only); the one function that needs scikit-learn,
:func:`derive_threshold_from_pairs`, imports it lazily so the *contract* models
(:class:`Correction`, :class:`LabeledPair`) a review tool depends on never pull a
heavy dependency. Held-out evaluation of a derived threshold is intentionally
*not* here -- it belongs to :mod:`langres.core.metrics` (``classify_pairs``),
which pulls ``ranx``; keeping harvest light means emitting a ``corrections.jsonl``
never drags eval tooling in.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from langres.core.calibration import ThresholdMethod

__all__ = [
    "Correction",
    "CorrectionLog",
    "LabeledPair",
    "derive_threshold_from_pairs",
    "harvest_labeled_pairs",
]

logger = logging.getLogger(__name__)

#: Schema-version tag written into every ``corrections.jsonl`` line -- mirrors
#: ``JudgementLog``'s ``"v": 1`` so a future format migration can branch on it
#: instead of guessing.
_CORRECTION_SCHEMA_VERSION = 1


class Correction(BaseModel):
    """One human verdict-correction for a judged pair: the ``corrections.jsonl`` line.

    This is the stable contract an external review queue (e.g. brainsquad) writes
    when a reviewer overrides a judge's verdict. Only ``left_id``/``right_id`` and
    the corrected ``label`` are required; the rest is optional audit context. A
    pair is identified order-independently on harvest (see
    :func:`harvest_labeled_pairs`), so which record is ``left`` need not match the
    judgement log's ordering.

    Attributes:
        v: Schema-version tag (mirrors ``JudgementLog``'s ``"v"``). Default ``1``.
        left_id: Identifier of one entity in the corrected pair.
        right_id: Identifier of the other entity in the corrected pair.
        label: The reviewer's corrected verdict -- ``True`` for a match,
            ``False`` for a non-match. This overrides the judge's logged verdict.
        original_score: The judge's score for this pair, if the review tool
            carried it through (audit only; harvest reads the score from the
            judgement log, not from here).
        original_verdict: The judge's verdict being overridden, if recorded
            (audit only).
        reviewer: Who made the correction, if recorded (audit only).
        timestamp: ISO-8601 timestamp of the correction, if recorded (audit only).
    """

    v: int = _CORRECTION_SCHEMA_VERSION
    left_id: str
    right_id: str
    label: bool
    original_score: float | None = None
    original_verdict: bool | None = None
    reviewer: str | None = None
    timestamp: str | None = None


class LabeledPair(BaseModel):
    """A harvested labeled pair: a score plus a label ready for calibration/fit.

    The output unit of :func:`harvest_labeled_pairs`. ``score`` comes from the
    judgement log; ``label`` is the judge's verdict unless a human correction
    overrode it, which ``source`` records so a caller can weight or audit the
    human-reviewed subset.

    Attributes:
        left_id: Identifier of the left entity (as logged by the judge).
        right_id: Identifier of the right entity (as logged by the judge).
        score: The judge's score for the pair (from the judgement log).
        label: The weak/corrected match label -- ``True`` for a match.
        source: Where ``label`` came from -- ``"verdict"`` (the judge's logged
            verdict, a weak label) or ``"correction"`` (a human override).
    """

    left_id: str
    right_id: str
    score: float
    label: bool
    source: Literal["verdict", "correction"]


class CorrectionLog:
    """JSONL-file-backed reader/writer for ``corrections.jsonl`` -- the harvest inlet.

    The reference implementation of the :class:`Correction` file contract,
    mirroring :class:`~langres.core.judgement_log.JudgementLog` so both flywheel
    files are appended and reloaded the same way. A downstream review tool may
    write the file itself (any JSONL of :class:`Correction`-shaped lines is
    valid); this class is what langres-side tooling and tests use to produce and
    consume it.

    Args:
        path: The ``corrections.jsonl`` file to append to / read from. Parent
            directories are created on first :meth:`append` if missing.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, correction: Correction) -> None:
        """Append one JSON line for ``correction`` (creating parent dirs if needed)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(correction.model_dump()) + "\n")

    def read(self) -> list[Correction]:
        """Reload every correction written so far, in write order.

        Returns ``[]`` if the file was never created (no corrections yet). Blank
        lines are skipped; each non-blank line is validated into a
        :class:`Correction`.
        """
        if not self.path.exists():
            return []
        corrections: list[Correction] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    corrections.append(Correction.model_validate(json.loads(stripped)))
        return corrections


def harvest_labeled_pairs(
    judgement_rows: Sequence[Mapping[str, Any]],
    corrections: Sequence[Correction],
) -> list[LabeledPair]:
    """Merge logged judgements with human corrections into labeled pairs.

    Each judgement row (a ``JudgementLog``-format dict, e.g. from
    :meth:`JudgementLog.read <langres.core.judgement_log.JudgementLog.read>`)
    yields one :class:`LabeledPair`. Its label is the logged ``verdict`` (a weak
    label) unless a :class:`Correction` covers the same pair, in which case the
    human ``label`` overrides it and ``source`` is ``"correction"``. Pairs are
    matched order-independently by ``frozenset({left_id, right_id})``, so a
    correction need not repeat the log's left/right ordering.

    Corrections referencing a pair that appears in no judgement row are skipped
    with a warning: a correction carries no score, so without a logged score for
    the pair it cannot become a calibration example.

    Args:
        judgement_rows: Logged judge calls as mappings with at least ``left_id``,
            ``right_id``, ``score`` and ``verdict`` keys.
        corrections: Human corrections to overlay. Later corrections for the same
            pair win (last-write-wins).

    Returns:
        One :class:`LabeledPair` per judgement row, in row order.
    """
    corrections_by_pair: dict[frozenset[str], Correction] = {}
    for correction in corrections:
        key = frozenset({correction.left_id, correction.right_id})
        if key in corrections_by_pair:
            logger.debug("Duplicate correction for pair %s; last one wins", tuple(key))
        corrections_by_pair[key] = correction

    matched: set[frozenset[str]] = set()
    pairs: list[LabeledPair] = []
    for row in judgement_rows:
        left_id = str(row["left_id"])
        right_id = str(row["right_id"])
        key = frozenset({left_id, right_id})
        correction = corrections_by_pair.get(key)
        if correction is not None:
            matched.add(key)
            label = correction.label
            source: Literal["verdict", "correction"] = "correction"
        else:
            label = bool(row["verdict"])
            source = "verdict"
        pairs.append(
            LabeledPair(
                left_id=left_id,
                right_id=right_id,
                score=float(row["score"]),
                label=label,
                source=source,
            )
        )

    unmatched = len(corrections_by_pair) - len(matched)
    if unmatched:
        logger.warning(
            "%d correction(s) reference a pair absent from the judgement log and "
            "were skipped (no logged score to attach a label to).",
            unmatched,
        )
    return pairs


def derive_threshold_from_pairs(
    pairs: Sequence[LabeledPair],
    *,
    method: ThresholdMethod = "youden",
    percentile: float | None = None,
) -> float:
    """Derive a decision threshold from harvested labeled pairs.

    The flywheel's payoff and :func:`~langres.core.calibration.derive_threshold`'s
    first production caller: it reads the score distribution and labels straight
    off :func:`harvest_labeled_pairs`' output and returns a data-driven cut,
    replacing a hand-set constant.

    Args:
        pairs: Harvested labeled pairs (from :func:`harvest_labeled_pairs`).
        method: Passed through to
            :func:`~langres.core.calibration.derive_threshold`.
        percentile: Passed through (required only for ``method="percentile"``).

    Returns:
        The derived threshold as a plain ``float``.

    Raises:
        ValueError: Propagated from
            :func:`~langres.core.calibration.derive_threshold` (empty input,
            single-class labels under ``"youden"``, bad ``percentile``, ...).
    """
    from langres.core.calibration import derive_threshold

    return derive_threshold(
        [pair.score for pair in pairs],
        [pair.label for pair in pairs],
        method=method,
        percentile=percentile,
    )
