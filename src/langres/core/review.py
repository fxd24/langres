"""The flywheel's review half: pick which judged pairs a human should look at.

:mod:`langres.core.judgement_log` writes the signal (one JSONL line per judge
call, ids only); :mod:`langres.core.harvest` turns answered reviews back into
labeled pairs. This module is the piece between them: :func:`select_for_review`
reads logged judgement rows and selects the pairs *worth* a human's attention,
and :class:`ReviewQueue` writes that selection as a ``review_queue.jsonl``
snapshot a labeling surface (``langres review``, a spreadsheet export, a
downstream web UI) can render.

Three selection strategies:

- ``"uncertainty"`` -- pairs where the judge itself was least sure, most
  uncertain first. The bread-and-butter strategy. Two signals, in order of
  preference: a logged ``confidence`` (credence in the judge's own answer,
  ``0.5`` = maximally uncertain -- ranked by ``|confidence - 0.5|``), else a
  ``score`` within ``margin`` of the decision ``threshold`` (ranked by
  ``|score - threshold|``). A decision-only/binary log carrying *neither* -- a
  raw Yes/No judge with no ``confidence="logprob"`` signal -- has nothing to
  rank by uncertainty and raises ``ValueError`` naming the fix, rather than
  silently returning ``[]`` as if the loop had finished.
- ``"disagreement"`` -- pairs where two judgement logs (e.g. a cheap student
  vs a frontier teacher) reached opposite verdicts, largest score gap first.
- ``"audit"`` -- a seeded random governance sample over all judged pairs, no
  threshold needed. Unbiased trust measurement: uncertainty-only sampling
  biases precision/recall estimates and never surfaces *confident false
  merges*; the audit slice is the mechanism that catches them.

The uncertainty and disagreement strategies mix a small audit slice into every
batch (``audit_fraction``, default 10% of ``limit``) for the same reason; pass
``audit_fraction=0.0`` to opt out. An exhausted selection returns ``[]`` -- the
stop signal that there is nothing left worth asking -- rather than padding the
batch with audits.

Privacy posture: judgement logs store ids only, and review items are likewise
ids-only unless the caller explicitly passes ``records=`` to join the record
content back on -- record content is never copied out of the caller's data
unless asked.

Import weight: pydantic + stdlib only (mirrors :mod:`langres.core.harvest`),
so this module stays eagerly importable from ``langres.core`` without touching
any optional extra (see ``tests/test_import_budget.py``).
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

__all__ = ["ReviewItem", "ReviewQueue", "select_for_review"]

logger = logging.getLogger(__name__)

#: Schema-version tag written into every ``review_queue.jsonl`` line -- mirrors
#: ``JudgementLog``'s and ``CorrectionLog``'s ``"v": 1`` so a future format
#: migration can branch on it instead of guessing.
_REVIEW_SCHEMA_VERSION = 1

#: The selection strategies :func:`select_for_review` understands.
ReviewReason = Literal["uncertainty", "disagreement", "audit"]

_STRATEGIES: tuple[ReviewReason, ...] = ("uncertainty", "disagreement", "audit")


class _PairIds(Protocol):
    """Anything naming a judged pair by its two ids (e.g. ``harvest.Correction``)."""

    left_id: str
    right_id: str


class ReviewItem(BaseModel):
    """One pair queued for human review: the ``review_queue.jsonl`` line contract.

    Field names ``left_id``/``right_id`` deliberately match the
    ``JudgementLog`` line format and ``langres.core.harvest.Correction``, so a
    labeling surface can answer an item by writing a ``Correction`` with the
    same ids and the harvest step joins everything back up.

    Attributes:
        v: Schema-version tag (mirrors ``JudgementLog``'s ``"v"``). Default ``1``.
        left_id: Identifier of one entity in the judged pair.
        right_id: Identifier of the other entity in the judged pair.
        score: The judge's logged score for the pair, or ``None`` for a decider
            (a binary judge that emits a ``decision`` and no score -- widened
            from a required float to match ``PairwiseJudgement.score``; a
            fabricated ``0.0``/``1.0`` would lie about a judge that does not rank).
        verdict: The judge's logged verdict being reviewed (``decision`` when the
            row carries one, else the thresholded ``verdict``). ``None`` only for
            a row usable by its score alone that recorded no verdict/decision.
        reason: Why this pair was selected -- the strategy that picked it
            (``"audit"`` items also appear inside uncertainty/disagreement
            batches via the audit mix-in).
        decision_step: The logged ``decision_step``, if present (which pipeline
            step produced the judgement -- e.g. a cascade tier).
        model: The logged model name, if present.
        reasoning: The judge's logged natural-language explanation, if the log
            recorded one (``features=True``). ``None`` for the default ids-only
            rows and for v1/v2 rows that predate the field -- so the reviewer can
            see *why* the judge answered as it did.
        confidence: The judge's logged credence in its own answer (``[0, 1]``,
            ``0.5`` = maximally uncertain), if the log carries one (the
            ``confidence="logprob"`` path). ``None`` for judges that give no
            confidence and for rows that predate the field.
        confidence_source: Provenance of ``confidence`` (e.g. ``"logprob"``),
            if the row records it -- so the reviewer can see *how sure* and on
            what basis.
        left_record: The joined content of the left record, or ``None`` when
            ``records=`` was not passed to :func:`select_for_review` (the
            ids-only privacy posture) or the id was not found.
        right_record: Same as ``left_record``, for the right record.
        details: Strategy-specific context -- uncertainty: ``distance`` (and
            ``threshold`` on the score path); disagreement: ``against_*`` (the
            second log's score, verdict, model, decision_step).
    """

    v: int = _REVIEW_SCHEMA_VERSION
    left_id: str
    right_id: str
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    verdict: bool | None = None
    reason: ReviewReason
    decision_step: str | None = None
    model: str | None = None
    reasoning: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_source: str | None = None
    left_record: dict[str, Any] | None = None
    right_record: dict[str, Any] | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ReviewQueue:
    """JSONL-file-backed **snapshot** of a review selection.

    A review queue is a derived snapshot -- regenerate it from the judgement
    log (via :func:`select_for_review`), never hand-edit it. Unlike the two
    append-only source-of-truth logs (``JudgementLog``, ``CorrectionLog``),
    :meth:`write` truncates the file: the queue always reflects exactly one
    selection, so re-running the selector after new corrections replaces stale
    items instead of stacking batches.

    Args:
        path: The ``review_queue.jsonl`` file to write to / read from. Parent
            directories are created on :meth:`write` if missing.

    Example:
        >>> queue = ReviewQueue("review_queue.jsonl")
        >>> queue.write(items)  # snapshot: replaces any previous queue
        >>> items = queue.read()
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, items: Sequence[ReviewItem]) -> None:
        """Write ``items`` as the queue's new content (snapshot: truncates the file)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item.model_dump()) + "\n")

    def read(self) -> list[ReviewItem]:
        """Reload the queue's current snapshot, in write order.

        Returns ``[]`` if the file was never written. Blank lines are skipped.

        Raises:
            ValueError: A line is not valid JSON or not a valid
                :class:`ReviewItem` -- the message names the line number. The
                queue is derived state: regenerate it with
                :func:`select_for_review` instead of repairing it by hand.
        """
        if not self.path.exists():
            return []
        items: list[ReviewItem] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    items.append(ReviewItem.model_validate(json.loads(stripped)))
                except (json.JSONDecodeError, ValidationError) as exc:
                    raise ValueError(
                        f"Corrupt review-queue line {lineno} in {self.path}. The queue "
                        "is a derived snapshot -- regenerate it with select_for_review() "
                        "instead of hand-editing."
                    ) from exc
        return items


def select_for_review(
    judgement_rows: Sequence[Mapping[str, Any]],
    *,
    strategy: ReviewReason,
    threshold: float | None = None,
    margin: float = 0.1,
    against: Sequence[Mapping[str, Any]] | None = None,
    records: Sequence[Any] | None = None,
    corrections: Sequence[_PairIds] = (),
    limit: int = 20,
    audit_fraction: float = 0.1,
    seed: int = 0,
) -> list[ReviewItem]:
    """Select the judged pairs most worth a human's review.

    Reads ``JudgementLog``-format rows (e.g. from ``JudgementLog.read()``),
    keys them order-independently by ``frozenset({left_id, right_id})``, with
    the row LATEST IN THE INPUT SEQUENCE winning on duplicates -- positional,
    by the order rows arrive in ``judgement_rows``, NOT by any logged
    ``timestamp`` field. For the common case of a single append-only log this
    is the same as chronological order, but a caller concatenating multiple
    logs or replaying rows out of order should sort by ``timestamp`` first.
    It then drops every pair already answered in ``corrections`` (a corrected
    pair is never re-asked), and applies ``strategy``:

    - ``"uncertainty"``: the pairs the judge was least sure about, sorted
      most-uncertain first. Prefers a logged ``confidence`` (ranked by
      ``|confidence - 0.5| <= margin``); falls back to ``score`` (ranked by
      ``|score - threshold| <= margin``, unchanged). Requires ``threshold``.
      A decision-only/binary log with neither a usable ``confidence`` nor a
      non-degenerate ``score`` raises ``ValueError`` (see below) instead of
      returning ``[]``.
    - ``"disagreement"``: pairs whose verdict differs between
      ``judgement_rows`` and ``against``, sorted by largest score gap first
      (scores, ids and verdicts are taken from ``judgement_rows``; the second
      log's side lands in ``details["against_*"]``). Requires ``against``.
    - ``"audit"``: a seeded random sample of up to ``limit`` judged pairs --
      pure governance/trust measurement, no threshold needed.

    For the first two strategies a small audit slice is mixed into the batch:
    ``int(limit * audit_fraction)`` seeded-random items (``reason="audit"``)
    drawn from the judged pairs not already selected, and the primary strategy
    fills the remaining ``limit`` slots. Pass ``audit_fraction=0.0`` to disable
    the mix-in. If the primary strategy finds *nothing*, the result is ``[]``
    with no audit padding -- the stop signal that the loop is exhausted.

    Deterministic throughout: same inputs and ``seed`` produce the same items
    in the same order (all randomness flows through ``random.Random(seed)``).

    A row is usable when it names a pair (``left_id`` and ``right_id``) AND
    carries at least one actionable signal: a bool ``decision``/``verdict`` OR a
    finite ``score`` in ``[0, 1]``. This admits a binary decider (``decision``
    set, ``score`` ``None``) as well as a pure ranker. Rows failing that -- no
    ids, or nothing but non-bool verdicts and a non-finite/out-of-range score --
    are skipped with one summary ``logger.warning``, so a hand-edited JSONL line
    degrades the batch, never crashes it.

    Privacy posture: with ``records=`` omitted (the default) items carry ids
    only -- record content is never copied into the queue unless asked. Pass
    ``records=`` to join content on: each record may be a mapping with an
    ``"id"`` key or an object with an ``id`` attribute (both matched against
    the log's ids as strings).

    Args:
        judgement_rows: Logged judge calls -- mappings with at least
            ``left_id``, ``right_id``, ``score`` and ``verdict`` keys.
        strategy: ``"uncertainty"``, ``"disagreement"`` or ``"audit"``.
        threshold: The decision cut to sample around (uncertainty only).
        margin: Scalar half-width of the uncertainty band around ``threshold``
            (a score qualifies when ``|score - threshold| <= margin``).
        against: A second judgement log to compare verdicts with
            (disagreement only).
        records: Optional records to join content from (see privacy posture
            above). Records without an id are skipped with one warning; logged
            ids absent from ``records`` leave their items ids-only.
        corrections: Already-answered pairs (e.g.
            ``langres.core.harvest.Correction`` objects, or anything with
            ``left_id``/``right_id`` attributes) -- excluded from selection.
        limit: Maximum number of items in the batch (audit slice included).
        audit_fraction: Fraction of ``limit`` reserved for the audit mix-in
            (``0.0`` disables it; ignored for ``strategy="audit"``).
        seed: Seed for the audit sampling (determinism).

    Returns:
        Selected :class:`ReviewItem` objects -- primary items first (in
        strategy order), then the audit slice. ``[]`` when the strategy is
        exhausted.

    Raises:
        ValueError: Unknown ``strategy``, ``strategy="uncertainty"`` without
            ``threshold``, ``strategy="disagreement"`` without ``against``, or
            ``strategy="uncertainty"`` over a decision-only/binary log with no
            rankable uncertainty signal (no usable ``confidence`` and every
            ``score`` ``None`` or a ``0``/``1`` decision). That last case used to
            silently return ``[]`` -- indistinguishable from a finished loop --
            so it now fails loud, naming ``strategy="disagreement"`` or
            ``LLMJudge(confidence="logprob")`` as the fix.

    Example:
        >>> rows = [
        ...     {"left_id": "a", "right_id": "b", "score": 0.62, "verdict": True},
        ...     {"left_id": "a", "right_id": "c", "score": 0.05, "verdict": False},
        ... ]
        >>> items = select_for_review(
        ...     rows, strategy="uncertainty", threshold=0.6, audit_fraction=0.0
        ... )
        >>> [(i.left_id, i.right_id, i.reason) for i in items]
        [('a', 'b', 'uncertainty')]
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}: expected 'uncertainty', 'disagreement' or 'audit'."
        )
    if not 0.0 <= audit_fraction <= 1.0:
        raise ValueError(
            f"audit_fraction must be in [0.0, 1.0], got {audit_fraction!r}. It is the "
            "share of the batch spent on random audit items (0.0 = no audit slice); "
            "a value > 1 would let the audit slice exceed limit."
        )
    if limit < 0:
        raise ValueError(
            f"limit must be >= 0, got {limit!r}. It is the maximum number of items "
            "in the batch; a negative value would make random.Random.sample's "
            "count negative and raise instead of returning an empty selection."
        )

    rows = _clean_rows(judgement_rows, source="judgement_rows")
    corrected = {_pair_key(c.left_id, c.right_id) for c in corrections}
    eligible = {key: row for key, row in _dedupe_by_pair(rows).items() if key not in corrected}
    rng = random.Random(seed)

    if strategy == "uncertainty":
        if threshold is None:
            raise ValueError(
                "strategy='uncertainty' requires threshold= (the decision cut to sample "
                "around, e.g. from derive_threshold_from_pairs)."
            )
        primary = _select_uncertainty(eligible, threshold=threshold, margin=margin)
    elif strategy == "disagreement":
        if against is None:
            raise ValueError(
                "strategy='disagreement' requires against= (a second judgement log to "
                "compare verdicts with)."
            )
        against_by_pair = _dedupe_by_pair(_clean_rows(against, source="against"))
        primary = _select_disagreement(eligible, against_by_pair)
    else:  # strategy == "audit": first-class governance sample, no mix-in on top
        pool = list(eligible.values())
        sampled = rng.sample(pool, min(limit, len(pool)))
        return _join_records([_build_item(row, reason="audit") for row in sampled], records)

    if not primary:
        return []  # stop signal: the strategy is exhausted -- no audit padding

    audit_count = int(limit * audit_fraction)
    primary_items = primary[: max(limit - audit_count, 0)]
    taken = {_pair_key(item.left_id, item.right_id) for item in primary_items}
    audit_pool = [row for key, row in eligible.items() if key not in taken]
    audit_items = [
        _build_item(row, reason="audit")
        for row in rng.sample(audit_pool, min(audit_count, len(audit_pool)))
    ]
    return _join_records(primary_items + audit_items, records)


def _pair_key(left_id: str, right_id: str) -> frozenset[str]:
    """Order-independent pair key (kept local -- harvest.py has its own copy)."""
    return frozenset({left_id, right_id})


def _clean_rows(rows: Sequence[Mapping[str, Any]], *, source: str) -> list[Mapping[str, Any]]:
    """Drop malformed rows, warning once with the count (never per row)."""
    clean: list[Mapping[str, Any]] = []
    skipped = 0
    for row in rows:
        if _is_well_formed(row):
            clean.append(row)
        else:
            skipped += 1
    if skipped:
        logger.warning(
            "Skipped %d malformed judgement row(s) from %s (each row needs left_id, "
            "right_id and at least one actionable signal: a bool decision/verdict "
            "or a finite score in [0, 1]).",
            skipped,
            source,
        )
    return clean


def _is_well_formed(row: Mapping[str, Any]) -> bool:
    """True if ``row`` names a pair AND carries at least one actionable signal.

    Actionable = a bool ``decision``/``verdict`` (a decider's or ranker's
    call) OR a finite ``score`` in ``[0, 1]`` (a ranker's number). A binary
    decider (``score`` ``None``, ``decision``/``verdict`` bool) passes; a row
    with no ids, or with nothing but non-bool verdicts and a
    non-finite/out-of-range score, fails.
    """
    if row.get("left_id") is None or row.get("right_id") is None:
        return False
    return _row_verdict(row) is not None or _finite_unit(row.get("score")) is not None


def _finite_unit(value: Any) -> float | None:
    """``value`` as a float in ``[0, 1]`` if it is a finite real number, else ``None``.

    The single sanitizer for both ``score`` and ``confidence``: ``bool`` is not
    a number here (``True``/``False`` are decisions, not scores), and ``NaN``,
    ``inf`` and out-of-range values degrade to ``None`` rather than crashing a
    later ``float(...)`` or Pydantic ``ge/le`` bound.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and 0.0 <= number <= 1.0 else None


def _row_verdict(row: Mapping[str, Any]) -> bool | None:
    """The row's bool verdict: ``decision`` (v3) when present, else ``verdict``.

    ``decision`` is the judge's own call (A3b logs it first-class); ``verdict``
    is the thresholded predicted-match the older format recorded. Prefer the
    former, fall back to the latter, and return ``None`` when neither is a bool
    (an abstain, or a score-only/foreign row) so callers can skip it.
    """
    decision = row.get("decision")
    if isinstance(decision, bool):
        return decision
    verdict = row.get("verdict")
    if isinstance(verdict, bool):
        return verdict
    return None


def _dedupe_by_pair(rows: Sequence[Mapping[str, Any]]) -> dict[frozenset[str], Mapping[str, Any]]:
    """Key rows by unordered pair; a later row for the same pair wins."""
    by_pair: dict[frozenset[str], Mapping[str, Any]] = {}
    for row in rows:
        by_pair[_pair_key(str(row["left_id"]), str(row["right_id"]))] = row
    return by_pair


def _uncertainty_by_score(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    margin: float,
) -> list[ReviewItem]:
    """Score-distance uncertainty band: rows within ``margin`` of ``threshold``.

    The fallback ranking when a row carries no credence -- ``|score - threshold|``,
    least sure first. Rows with no usable score contribute nothing. Raises
    nothing: the "no gradient at all" decision is the caller's (see
    :func:`_select_uncertainty`).
    """
    scored = [(s, row) for row in rows if (s := _finite_unit(row.get("score"))) is not None]
    in_band = [(distance, row) for s, row in scored if (distance := abs(s - threshold)) <= margin]
    in_band.sort(key=lambda entry: entry[0])
    return [
        _build_item(
            row, reason="uncertainty", details={"threshold": threshold, "distance": distance}
        )
        for distance, row in in_band
    ]


def _select_uncertainty(
    eligible: Mapping[frozenset[str], Mapping[str, Any]],
    *,
    threshold: float,
    margin: float,
) -> list[ReviewItem]:
    """The pairs the judge was least sure about, most uncertain first.

    Prefers a logged ``confidence`` (credence in the judge's own answer, ranked
    by ``|confidence - 0.5|`` -- ``0.5`` is maximally uncertain); falls back to
    ``score`` (ranked by ``|score - threshold|``, exactly as before). A
    decision-only/binary log carrying neither a usable confidence nor a
    non-degenerate score has *nothing* to rank by uncertainty and raises,
    rather than silently returning ``[]`` (see :func:`select_for_review`).

    A *mixed* log (some rows carry confidence, some only a score -- e.g. a
    :class:`~langres.core.modules.cascade_judge.CascadeJudge` whose cheap-student
    rows are score-only and whose escalated rows carry a logprob confidence)
    yields both bands: credence-ranked rows first (a real self-reported
    uncertainty), then score-ranked rows for the rows that lacked confidence, so
    an uncertain score-only pair is never silently dropped just because some
    other row happens to carry a confidence.
    """
    rows = list(eligible.values())
    if not rows:
        # Nothing judged yet (or every pair already corrected): a genuinely
        # finished loop, not a broken one -- return [], never raise.
        return []

    # A real credence signal wins when the log has one (the confidence="logprob"
    # path): rank by distance from 0.5 -- maximal uncertainty -- least sure first.
    confident = [(c, row) for row in rows if (c := _finite_unit(row.get("confidence"))) is not None]
    if confident:
        in_band = [(abs(c - 0.5), row) for c, row in confident if abs(c - 0.5) <= margin]
        in_band.sort(key=lambda entry: entry[0])
        items = [
            _build_item(row, reason="uncertainty", details={"distance": distance})
            for distance, row in in_band
        ]
        # Fold in the score-only rows (those with no usable confidence) via the
        # score-distance fallback -- otherwise they vanish from the queue the
        # instant one row carries a confidence, the same silent no-op this
        # function exists to kill, just relocated to a mixed log.
        score_only = [row for row in rows if _finite_unit(row.get("confidence")) is None]
        items.extend(_uncertainty_by_score(score_only, threshold=threshold, margin=margin))
        return items

    # No confidence: fall back to score-distance. A binary/decision log has no
    # continuous score to rank (every score is None or a 0/1 decision), so
    # |score - threshold| is a constant and the band is *always* empty -- the
    # silent no-op this function used to hide behind a "[]". Fail loud instead.
    scored = [(s, row) for row in rows if (s := _finite_unit(row.get("score"))) is not None]
    if not any(0.0 < s < 1.0 for s, _ in scored):
        raise ValueError(
            "strategy='uncertainty' has no signal to rank this judgement log by: "
            "no row carries a usable confidence and every score is None or a 0/1 "
            "decision, so the uncertainty band is always empty (this used to "
            "silently return [], indistinguishable from a finished loop). For a "
            "binary/decision judge, either use strategy='disagreement' to compare "
            'it against a second judge, or run LLMJudge(confidence="logprob") so '
            "each judgement carries a real uncertainty signal."
        )
    return _uncertainty_by_score(rows, threshold=threshold, margin=margin)


def _select_disagreement(
    eligible: Mapping[frozenset[str], Mapping[str, Any]],
    against_by_pair: Mapping[frozenset[str], Mapping[str, Any]],
) -> list[ReviewItem]:
    """Pairs whose verdict differs across the two logs, largest score gap first.

    Compares each side's ``decision`` (v3) or ``verdict`` via
    :func:`_row_verdict`, so a binary/decision log -- the fallback the
    uncertainty raise points at -- disagrees correctly. A row with no bool
    verdict on either side (an abstain, a score-only row) cannot disagree and is
    skipped. The largest-gap sort uses ``_score_gap``, which is ``0.0`` when
    either score is ``None`` (a decider has no score), so a binary log keeps
    insertion order (a stable sort) instead of crashing on ``float(None)``.
    """
    differing: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for key, row in eligible.items():
        other = against_by_pair.get(key)
        if other is None:
            continue
        row_verdict = _row_verdict(row)
        other_verdict = _row_verdict(other)
        if row_verdict is None or other_verdict is None or row_verdict == other_verdict:
            continue
        differing.append((row, other))
    differing.sort(key=lambda pair: _score_gap(pair[0], pair[1]), reverse=True)
    return [
        _build_item(
            row,
            reason="disagreement",
            details={
                "against_score": _finite_unit(other.get("score")),
                "against_verdict": _row_verdict(other),
                "against_model": _opt_str(other.get("model")),
                "against_decision_step": _opt_str(other.get("decision_step")),
            },
        )
        for row, other in differing
    ]


def _score_gap(row: Mapping[str, Any], other: Mapping[str, Any]) -> float:
    """Absolute score gap between two rows, or ``0.0`` when either has no score."""
    left = _finite_unit(row.get("score"))
    right = _finite_unit(other.get("score"))
    if left is None or right is None:
        return 0.0
    return abs(left - right)


def _build_item(
    row: Mapping[str, Any],
    *,
    reason: ReviewReason,
    details: Mapping[str, Any] | None = None,
) -> ReviewItem:
    """One :class:`ReviewItem` from a well-formed judgement row (ids-only).

    Reads ``score``/``verdict``/``confidence`` through the same sanitizers the
    selection used, so a decider row (``score`` ``None``) and v1/v2 rows lacking
    the ``reasoning``/``confidence``/``confidence_source`` columns build a valid
    item instead of tripping a ``float(None)`` or a Pydantic bound.
    """
    return ReviewItem(
        left_id=str(row["left_id"]),
        right_id=str(row["right_id"]),
        score=_finite_unit(row.get("score")),
        verdict=_row_verdict(row),
        reason=reason,
        decision_step=_opt_str(row.get("decision_step")),
        model=_opt_str(row.get("model")),
        reasoning=_opt_str(row.get("reasoning")),
        confidence=_finite_unit(row.get("confidence")),
        confidence_source=_opt_str(row.get("confidence_source")),
        details=dict(details) if details is not None else {},
    )


def _opt_str(value: Any) -> str | None:
    """``None`` stays ``None``; anything else is coerced to ``str``."""
    return None if value is None else str(value)


def _record_id(record: Any) -> str | None:
    """The record's id as a string -- mapping ``"id"`` key or ``id`` attribute."""
    value = record.get("id") if isinstance(record, Mapping) else getattr(record, "id", None)
    return None if value is None else str(value)


def _record_content(record: Any) -> dict[str, Any]:
    """The record's fields as a plain dict (for the joined ``*_record`` fields)."""
    if isinstance(record, Mapping):
        return dict(record)
    if isinstance(record, BaseModel):
        return record.model_dump()
    return dict(vars(record))


def _join_records(items: list[ReviewItem], records: Sequence[Any] | None) -> list[ReviewItem]:
    """Attach record content to ``items`` in place (no-op when ``records`` is omitted)."""
    if records is None:
        return items

    by_id: dict[str, Any] = {}
    unidentified = 0
    for record in records:
        record_id = _record_id(record)
        if record_id is None:
            unidentified += 1
        else:
            by_id[record_id] = record
    if unidentified:
        logger.warning(
            "%d record(s) in records= carry no id (mapping 'id' key or id attribute) and "
            "were skipped from the record join; affected review items stay ids-only.",
            unidentified,
        )
    if not by_id:
        return items

    misses = 0
    for item in items:
        left = by_id.get(item.left_id)
        if left is None:
            misses += 1
        else:
            item.left_record = _record_content(left)
        right = by_id.get(item.right_id)
        if right is None:
            misses += 1
        else:
            item.right_record = _record_content(right)
    if misses:
        logger.warning(
            "%d logged id(s) were not found in records=; those review items stay ids-only "
            "on that side.",
            misses,
        )
    return items
