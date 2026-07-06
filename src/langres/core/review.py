"""The flywheel's review half: pick which judged pairs a human should look at.

:mod:`langres.core.judgement_log` writes the signal (one JSONL line per judge
call, ids only); :mod:`langres.core.harvest` turns answered reviews back into
labeled pairs. This module is the piece between them: :func:`select_for_review`
reads logged judgement rows and selects the pairs *worth* a human's attention,
and :class:`ReviewQueue` writes that selection as a ``review_queue.jsonl``
snapshot a labeling surface (``langres review``, a spreadsheet export, a
downstream web UI) can render.

Three selection strategies:

- ``"uncertainty"`` -- pairs whose score falls within ``margin`` of the
  decision ``threshold``, most uncertain first. The bread-and-butter strategy:
  label where the judge itself was least sure.
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
        score: The judge's logged score for the pair.
        verdict: The judge's logged verdict being reviewed.
        reason: Why this pair was selected -- the strategy that picked it
            (``"audit"`` items also appear inside uncertainty/disagreement
            batches via the audit mix-in).
        decision_step: The logged ``decision_step``, if present (which pipeline
            step produced the judgement -- e.g. a cascade tier).
        model: The logged model name, if present.
        left_record: The joined content of the left record, or ``None`` when
            ``records=`` was not passed to :func:`select_for_review` (the
            ids-only privacy posture) or the id was not found.
        right_record: Same as ``left_record``, for the right record.
        details: Strategy-specific context -- uncertainty: ``threshold`` and
            ``distance``; disagreement: ``against_*`` (the second log's score,
            verdict, model, decision_step).
    """

    v: int = _REVIEW_SCHEMA_VERSION
    left_id: str
    right_id: str
    score: float
    verdict: bool
    reason: ReviewReason
    decision_step: str | None = None
    model: str | None = None
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
    keys them order-independently by ``frozenset({left_id, right_id})`` with
    last-write-wins on duplicates, drops every pair already answered in
    ``corrections`` (a corrected pair is never re-asked), and applies
    ``strategy``:

    - ``"uncertainty"``: pairs with ``|score - threshold| <= margin``, sorted
      most-uncertain first. Requires ``threshold``.
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

    Malformed rows (missing ``left_id``/``right_id``/``score``/``verdict``, a
    non-finite or out-of-``[0, 1]`` score, a non-bool verdict) are skipped with
    one summary ``logger.warning`` -- a hand-edited JSONL line degrades the
    batch, never crashes it.

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
            ``threshold``, or ``strategy="disagreement"`` without ``against``.

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
            "right_id, a finite score in [0, 1] and a bool verdict).",
            skipped,
            source,
        )
    return clean


def _is_well_formed(row: Mapping[str, Any]) -> bool:
    """True if ``row`` carries usable ids, score and verdict."""
    if row.get("left_id") is None or row.get("right_id") is None:
        return False
    if not isinstance(row.get("verdict"), bool):
        return False
    score = row.get("score")
    if isinstance(score, bool) or not isinstance(score, int | float):
        return False
    return math.isfinite(score) and 0.0 <= score <= 1.0


def _dedupe_by_pair(rows: Sequence[Mapping[str, Any]]) -> dict[frozenset[str], Mapping[str, Any]]:
    """Key rows by unordered pair; a later row for the same pair wins."""
    by_pair: dict[frozenset[str], Mapping[str, Any]] = {}
    for row in rows:
        by_pair[_pair_key(str(row["left_id"]), str(row["right_id"]))] = row
    return by_pair


def _select_uncertainty(
    eligible: Mapping[frozenset[str], Mapping[str, Any]],
    *,
    threshold: float,
    margin: float,
) -> list[ReviewItem]:
    """Pairs within ``margin`` of ``threshold``, most uncertain first."""
    in_band: list[tuple[float, Mapping[str, Any]]] = []
    for row in eligible.values():
        distance = abs(float(row["score"]) - threshold)
        if distance <= margin:
            in_band.append((distance, row))
    in_band.sort(key=lambda entry: entry[0])
    return [
        _build_item(
            row, reason="uncertainty", details={"threshold": threshold, "distance": distance}
        )
        for distance, row in in_band
    ]


def _select_disagreement(
    eligible: Mapping[frozenset[str], Mapping[str, Any]],
    against_by_pair: Mapping[frozenset[str], Mapping[str, Any]],
) -> list[ReviewItem]:
    """Pairs whose verdict differs across the two logs, largest score gap first."""
    differing: list[tuple[float, Mapping[str, Any], Mapping[str, Any]]] = []
    for key, row in eligible.items():
        other = against_by_pair.get(key)
        if other is None or bool(other["verdict"]) == bool(row["verdict"]):
            continue
        gap = abs(float(row["score"]) - float(other["score"]))
        differing.append((gap, row, other))
    differing.sort(key=lambda entry: entry[0], reverse=True)
    return [
        _build_item(
            row,
            reason="disagreement",
            details={
                "against_score": float(other["score"]),
                "against_verdict": bool(other["verdict"]),
                "against_model": _opt_str(other.get("model")),
                "against_decision_step": _opt_str(other.get("decision_step")),
            },
        )
        for _gap, row, other in differing
    ]


def _build_item(
    row: Mapping[str, Any],
    *,
    reason: ReviewReason,
    details: Mapping[str, Any] | None = None,
) -> ReviewItem:
    """One :class:`ReviewItem` from a well-formed judgement row (ids-only)."""
    return ReviewItem(
        left_id=str(row["left_id"]),
        right_id=str(row["right_id"]),
        score=float(row["score"]),
        verdict=bool(row["verdict"]),
        reason=reason,
        decision_step=_opt_str(row.get("decision_step")),
        model=_opt_str(row.get("model")),
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
