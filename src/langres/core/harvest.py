"""The flywheel's harvest half: logged verdicts + human corrections -> labeled pairs.

``JudgementLog`` (W0.2, :mod:`langres.core.judgement_log`) is the flywheel's
*inlet* -- an opt-in JSONL log of every judge call (ids, score, decision, verdict,
model, cost, ``"v": 3``). This module is the *outlet*: it turns those logged
judgements,
plus a review tool's human corrections, into **labeled pairs** that feed
:func:`langres.core.calibration.derive_threshold` (its first production caller)
and, where applicable, a judge's ``fit()``.

The division of labor is deliberate: langres owns the **contract**, the
**harvest**, and the headless + terminal review surfaces (the ``langres review``
CLI and its CSV round-trip); anything with a rendering loop (a web review UI)
stays in the downstream application. :class:`Correction` is the
stable line schema a review tool writes to ``corrections.jsonl``; :class:`CorrectionLog`
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
import warnings
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

    This is the stable contract an external review queue writes
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
        score: The judge's score for the pair (from the judgement log), or
            ``None`` for a decision-only judge that logged no score. The label is
            still usable, but a score-less pair cannot contribute to a *score*
            threshold -- :func:`derive_threshold_from_pairs` rejects such input
            rather than calibrating on the self-selected scored subset.
        label: The weak/corrected match label -- ``True`` for a match.
        source: Where ``label`` came from -- ``"verdict"`` (the judge's logged
            verdict, a weak label) or ``"correction"`` (a human override).
    """

    left_id: str
    right_id: str
    score: float | None
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

    A decision-only row (``score: null`` -- a judge that decided but did not
    rank) yields a :class:`LabeledPair` with ``score=None``: the label is still
    usable, but the pair carries no score for a *score* threshold, so
    :func:`derive_threshold_from_pairs` rejects it rather than dropping it into a
    biased scored-only subset. The score is carried as-is (never coerced to
    ``0.0``), so a genuine ``score == 0.0`` and a missing score stay distinct.

    An *abstention* row (``verdict: null`` -- the judge neither decided nor
    scored) carries no usable label and is **skipped** unless a human correction
    supplies one. Coercing a null verdict to a ``False`` non-match would seed
    silver-label training with a label the judge never gave -- the label-side
    twin of never coercing a null score to ``0.0``. So the output has one
    :class:`LabeledPair` per *labeled* row, not necessarily per input row.

    Args:
        judgement_rows: Logged judge calls as mappings with at least ``left_id``,
            ``right_id``, ``score`` and ``verdict`` keys. ``score`` may be
            ``None`` for a decision-only judge.
        corrections: Human corrections to overlay. Later corrections for the same
            pair win (last-write-wins).

    Returns:
        One :class:`LabeledPair` per *labeled* judgement row, in row order. An
        abstention row (``verdict`` null) with no correction carries no usable
        label and is omitted, so the output may be shorter than the input.
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
        override = corrections_by_pair.get(key)
        if override is not None:
            matched.add(key)
            label = override.label
            source: Literal["verdict", "correction"] = "correction"
        else:
            verdict = row["verdict"]
            if verdict is None:
                # An abstention (decision=None -> verdict=None): the judge gave
                # no verdict, so there is no label to harvest. Skip it rather
                # than coerce None to a False non-match -- a fabricated
                # "not a match" would poison silver labels exactly as a
                # fabricated 0.0 would poison a score threshold.
                continue
            label = bool(verdict)
            source = "verdict"
        raw_score = row["score"]
        pairs.append(
            LabeledPair(
                left_id=left_id,
                right_id=right_id,
                # Carry a decision-only row's null score as None (never coerce to
                # 0.0): the label is usable, but calibration must see the score is
                # absent, not a real 0.0.
                score=None if raw_score is None else float(raw_score),
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

    Calibrating on silver labels alone is circular: the judge's own verdicts
    were produced *by* a cut, so they are the only signal a threshold search
    can recover. A :class:`UserWarning` flags that case -- overlay human
    corrections (``source="correction"``) before calibrating. (Training a
    *different* model on silver labels is legitimate, which is why
    :func:`harvest_labeled_pairs` itself stays warning-free.)

    Args:
        pairs: Harvested labeled pairs (from :func:`harvest_labeled_pairs`).
        method: Passed through to
            :func:`~langres.core.calibration.derive_threshold`.
        percentile: Passed through (required only for ``method="percentile"``).

    Returns:
        The derived threshold as a plain ``float``.

    Raises:
        ValueError: If any pair has ``score is None`` (a decision-only judge has
            no scores to derive a *score* threshold from) -- raised here, naming
            the offending pair, rather than dropping the score-less pairs into a
            biased scored-only subset. Also propagated from
            :func:`~langres.core.calibration.derive_threshold` (empty input,
            single-class labels under ``"youden"``, bad ``percentile``, ...).

    Warns:
        UserWarning: If ``pairs`` is non-empty and every label is silver
            (``source == "verdict"``) -- silver-only calibration is circular
            (see above). Suppress deliberately via :mod:`warnings` filters.
    """
    # A score-less pair (decision-only judge) has no score to calibrate on;
    # collect the scores while checking, so mypy sees a list[float] below.
    scores: list[float] = []
    for pair in pairs:
        if pair.score is None:
            raise ValueError(
                "cannot derive a score threshold: pair "
                f"{pair.left_id}/{pair.right_id} has no score (a decision-only "
                "judge logged score=null); a decision-only judge has no scores to "
                "derive a score threshold from. Drop these pairs or calibrate on "
                "a scoring judge's output."
            )
        scores.append(pair.score)

    if pairs and all(pair.source == "verdict" for pair in pairs):
        warnings.warn(
            "silver-only calibration is circular -- deriving a threshold from "
            "a judge's own verdicts can only recover the cut that produced "
            "them; overlay human corrections (source='correction') before "
            "calibrating. (Training a DIFFERENT model on silver labels is "
            "fine.)",
            UserWarning,
            stacklevel=2,
        )

    from langres.core.calibration import derive_threshold

    return derive_threshold(
        scores,
        [pair.label for pair in pairs],
        method=method,
        percentile=percentile,
    )
