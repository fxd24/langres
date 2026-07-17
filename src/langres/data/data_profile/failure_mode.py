"""Failure-mode profile section: where a matcher's errors concentrate.

The post-hoc, error-analysis companion to mining readiness. Where mining
readiness profiles a labeled set *before* training, this section profiles a
matcher's *judgements after* it ran: it joins a precomputed
:class:`~langres.tracking.judgement_log.JudgementLog` against gold, splits the
confident verdicts into correct / false-positive / false-negative, and then asks
the question that drives the next curation round -- **which slice of the data do
the errors concentrate in?**

It answers that two ways, both a *diff of the error distribution against the
success distribution*:

- a **score-band overlay** -- the density histogram of error scores vs correct
  scores, so the uncertain middle band (where errors pile up) is visible;
- a **slice table** -- error rate + *lift* (slice error rate / overall error
  rate) per data slice: each score band, each content field's emptiness, and the
  cross- vs same-source split. A lift well above ``1.0`` names a slice the next
  mining round should target.

Like :class:`~langres.data.data_profile.mining_readiness.MiningReadinessSection`
it is a **pure consumer of precomputed inputs**: it runs no matcher and pulls no
scikit-learn (keeping ``data_profile`` import-light). The caller runs the matcher
(logging to a :class:`JudgementLog`), reads the rows back, and hands them plus the
gold pairs (and, for the field/source slices, the id->record mapping) to
:func:`profile_failure_mode`. Every input beyond the judgements + gold is optional
and degrades to fewer slices -- the same graceful-degradation contract the rest of
the report keeps.
"""

from __future__ import annotations

import html
import logging
import math
from collections.abc import Collection, Hashable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from langres.report import _report_html, _svg
from langres.data.data_profile.base import ProfileSection

logger = logging.getLogger(__name__)

#: Error / success series colors for the score overlay (explicit for light+dark;
#: the eval_report palette -- errors warm, successes cool).
_COLOR_ERROR = "#e76f51"
_COLOR_SUCCESS = "#2a9d8f"

#: Default number of equal-width score bands over ``[0, 1]`` (histogram + slices).
_DEFAULT_SCORE_BANDS = 5

#: Default cap on how many slices the table shows (the most concentrated first).
_DEFAULT_MAX_SLICES = 12


class FailureSlice(BaseModel):
    """Error concentration in one data slice -- one row of the failure table.

    A frozen record: how a single slice of the judged pairs (a score band, a
    field's emptiness, the source split) fares against the overall error rate.

    Attributes:
        dimension: The slicing axis (e.g. ``"score_band"``, ``"empty:price"``,
            ``"source"``).
        value: The human label of this slice's bucket (e.g. ``"0.40-0.60"``,
            ``"either side empty"``, ``"cross-source"``).
        n: Confident (non-abstain) judged pairs in this slice -- the error-rate
            denominator.
        n_errors: Errors (verdict != gold) among those ``n`` pairs.
        error_rate: ``n_errors / n``; ``None`` for an empty slice.
        lift: ``error_rate`` relative to the overall error rate -- ``> 1`` means
            errors concentrate here. ``None`` when the overall rate is ``0`` (no
            errors anywhere) or the slice is empty.
    """

    model_config = ConfigDict(frozen=True)

    dimension: str
    value: str
    n: int
    n_errors: int
    error_rate: float | None
    lift: float | None


class FailureModeSection(ProfileSection):
    """Where a matcher's errors concentrate: FP/FN split + sliced error rates.

    A frozen :class:`ProfileSection` holding the confusion headline (correct /
    false-positive / false-negative / abstain), the error-vs-success score
    histogram, and the per-slice :class:`FailureSlice` table. Build it with
    :func:`profile_failure_mode`; it computes nothing that needs a model.

    Attributes:
        n_judged: Judged pairs joined against gold (all rows).
        n_correct: Confident verdicts that matched gold.
        n_errors: Confident verdicts that disagreed with gold.
        n_abstain: Rows with no confident verdict (``verdict`` null) -- excluded
            from the error rate (an abstain is not a wrong answer).
        n_false_positive: Predicted match, gold non-match.
        n_false_negative: Predicted non-match, gold match.
        score_edges: Bin edges (``B + 1``) of the score overlay; empty when no
            confident pair carried a score.
        error_counts: Error scores per band (``B`` values).
        success_counts: Correct scores per band (``B`` values).
        slices: Failure slices, most concentrated (highest lift) first.
        n_slices_hidden: Slices dropped by the display cap (0 when none).
    """

    kind: Literal["failure_mode"] = "failure_mode"

    n_judged: int
    n_correct: int
    n_errors: int
    n_abstain: int
    n_false_positive: int
    n_false_negative: int
    score_edges: list[float]
    error_counts: list[float]
    success_counts: list[float]
    slices: list[FailureSlice]
    n_slices_hidden: int

    # ------------------------------------------------------------- derived stats
    @property
    def n_confident(self) -> int:
        """Confident verdicts (correct + errors) -- the error-rate denominator."""
        return self.n_correct + self.n_errors

    @property
    def error_rate(self) -> float | None:
        """Overall error rate over confident verdicts; ``None`` when none are confident."""
        return self.n_errors / self.n_confident if self.n_confident else None

    @property
    def abstain_rate(self) -> float | None:
        """Fraction of judged pairs with no confident verdict; ``None`` for an empty set."""
        return self.n_abstain / self.n_judged if self.n_judged else None

    # ------------------------------------------------------------- shared render
    def _metrics_kv(self) -> list[tuple[str, str]]:
        """The headline metrics as ``(label, display)`` pairs (markdown + HTML share this)."""
        return [
            ("judged pairs", f"{self.n_judged:,}"),
            ("confident verdicts", f"{self.n_confident:,}"),
            ("errors", _fmt_count_share(self.n_errors, self.error_rate)),
            ("false positives", f"{self.n_false_positive:,}"),
            ("false negatives", f"{self.n_false_negative:,}"),
            ("abstentions", _fmt_count_share(self.n_abstain, self.abstain_rate)),
        ]

    # ------------------------------------------------------------ text surfaces
    def to_markdown(self) -> str:
        """Markdown: the confusion headline, then the concentrated-slice table."""
        lines = [f"## {self.title}", "", "| metric | value |", "|---|---|"]
        lines += [
            f"| {_report_html._md_cell(k)} | {_report_html._md_cell(v)} |"
            for k, v in self._metrics_kv()
        ]
        if self.n_judged == 0:
            lines += ["", "_No judged pairs: nothing to analyze yet._"]
            return "\n".join(lines)
        lines += [
            "",
            "### Error concentration by slice",
            "",
            "| slice | value | pairs | errors | error rate | lift |",
            "|---|---|---|---|---|---|",
        ]
        if self.slices:
            for s in self.slices:
                lines.append(
                    f"| {_report_html._md_cell(s.dimension)} "
                    f"| {_report_html._md_cell(s.value)} | {s.n:,} | {s.n_errors:,} "
                    f"| {_fmt_pct(s.error_rate)} | {_report_html._num(s.lift, 2)} |"
                )
        else:
            lines.append("| _(no slices)_ | | 0 | 0 | n/a | n/a |")
        if self.n_slices_hidden:
            lines += ["", f"_+{self.n_slices_hidden} more slice(s) below the display cap._"]
        return "\n".join(lines)

    @property
    def summary(self) -> dict[str, Any]:
        """Headline numbers as a flat, title-namespaced dict (collision-free per report)."""
        return {
            f"{self.title}.n_judged": self.n_judged,
            f"{self.title}.n_errors": self.n_errors,
            f"{self.title}.error_rate": self.error_rate,
            f"{self.title}.n_false_positive": self.n_false_positive,
            f"{self.title}.n_false_negative": self.n_false_negative,
            f"{self.title}.n_abstain": self.n_abstain,
        }

    def rows(self) -> list[dict[str, Any]]:
        """One row per failure slice -- ``pd.DataFrame(section.rows())``-ready."""
        return [
            {
                "dimension": s.dimension,
                "value": s.value,
                "n": s.n,
                "n_errors": s.n_errors,
                "error_rate": s.error_rate,
                "lift": s.lift,
            }
            for s in self.slices
        ]

    # -------------------------------------------------------------- html panel
    def panels(self) -> list[str]:
        """A single ``<section>``: the KV headline, the score overlay, the slice table."""
        kv = "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>"
            for k, v in self._metrics_kv()
        )
        body = f'<table class="kv">{kv}</table>'
        if self.error_counts or self.success_counts:
            body += _svg.bar_chart(
                self.score_edges,
                [
                    ("errors", _COLOR_ERROR, self.error_counts),
                    ("correct", _COLOR_SUCCESS, self.success_counts),
                ],
                x_label="score",
                y_label="density",
                normalize="density",
            )
        body += self._slices_table_html()
        return [_report_html.section(self.title, body)]

    def _slices_table_html(self) -> str:
        """The per-slice error-concentration table as an HTML ``<table>``."""
        if not self.slices:
            return '<p class="empty">No slices to report.</p>'
        head = (
            "<tr><th>slice</th><th>value</th><th>pairs</th><th>errors</th>"
            "<th>error rate</th><th>lift</th></tr>"
        )
        body_rows = "".join(
            f"<tr><td>{html.escape(s.dimension)}</td><td>{html.escape(s.value)}</td>"
            f"<td>{s.n:,}</td><td>{s.n_errors:,}</td>"
            f"<td>{html.escape(_fmt_pct(s.error_rate))}</td>"
            f"<td>{html.escape(_report_html._num(s.lift, 2))}</td></tr>"
            for s in self.slices
        )
        return f'<table class="errors">{head}{body_rows}</table>'


# ---------------------------------------------------------------------------
# One joined + labeled judged pair (internal working shape)
# ---------------------------------------------------------------------------


class _Judged:
    """A judgement row joined to its gold label -- the working unit of the profiler."""

    __slots__ = ("key", "left_id", "right_id", "score", "verdict", "gold")

    def __init__(
        self,
        *,
        left_id: str,
        right_id: str,
        score: float | None,
        verdict: bool | None,
        gold: bool,
    ) -> None:
        self.key = frozenset({left_id, right_id})
        self.left_id = left_id
        self.right_id = right_id
        self.score = score
        self.verdict = verdict
        self.gold = gold

    @property
    def confident(self) -> bool:
        """A confident verdict is a real match/non-match call (not an abstention)."""
        return self.verdict is not None

    @property
    def error(self) -> bool:
        """A confident verdict that disagrees with gold."""
        return self.verdict is not None and self.verdict != self.gold


def profile_failure_mode(
    judgements: Sequence[Mapping[str, Any]],
    *,
    gold_pairs: Collection[Collection[Hashable]] | None,
    records: Mapping[Hashable, Mapping[str, Any]] | None = None,
    id_key: str = "id",
    source_key: str | None = "source",
    n_score_bands: int = _DEFAULT_SCORE_BANDS,
    max_slices: int = _DEFAULT_MAX_SLICES,
    title: str = "Failure modes",
) -> FailureModeSection | None:
    """Profile where a matcher's logged errors concentrate.

    Joins each logged judgement to its gold label, splits the confident verdicts
    into correct / false-positive / false-negative, and diffs the error
    distribution against the success distribution -- a score overlay plus a
    slice table (error rate + lift per score band, per content-field emptiness,
    and the cross- vs same-source split).

    Pure assembly over inputs the caller already produced -- it runs no matcher
    and pulls no scikit-learn (keeping ``data_profile`` import-light).

    Args:
        judgements: Logged judge calls as mappings with at least ``left_id``,
            ``right_id``, ``score`` and ``verdict`` keys -- e.g.
            :meth:`JudgementLog.read() <langres.tracking.judgement_log.JudgementLog.read>`.
            ``verdict`` is the caller's predicted-match decision (``None`` for an
            abstention); ``score`` may be ``None`` for a decision-only judge.
        gold_pairs: The gold positive pairs -- a collection of 2-id collections
            (order-independent). A judged pair is a true match iff it is in this
            set (closed-world: any pair not listed is a true non-match). ``None``
            means no gold is available, and the profiler returns ``None`` (the
            section is omitted, logged) -- errors cannot be identified without it.
        records: Optional ``id -> field mapping`` for the two records of each
            pair. Enables the field-emptiness and source slices; omit it and only
            the score-band slices are produced.
        id_key: Record field holding the id (default ``"id"``); excluded from the
            field-emptiness slices.
        source_key: Record field naming the linkage side for the cross/same-source
            slice (default ``"source"``); pass ``None`` to skip that slice, and it
            is skipped automatically when the field is absent. Excluded from the
            field-emptiness slices.
        n_score_bands: Equal-width score bands over ``[0, 1]`` for the overlay and
            the score-band slices.
        max_slices: Cap on how many slices the table shows (most concentrated
            first); the remainder are counted in ``n_slices_hidden``.
        title: Section heading; also the report lookup key and the namespace for
            this section's :attr:`~FailureModeSection.summary` keys.

    Returns:
        A :class:`FailureModeSection`, or ``None`` when there is nothing to
        analyze (no judgements, or no gold) -- logged, never raised.
    """
    if not judgements:
        logger.warning("profile_failure_mode: no judgements to analyze; returning None.")
        return None
    if gold_pairs is None:
        logger.warning(
            "profile_failure_mode: gold_pairs is None; cannot identify errors without gold. "
            "Returning None."
        )
        return None

    gold = _normalize_gold(gold_pairs)
    judged = _join_gold(judgements, gold)
    confident = [j for j in judged if j.confident]

    n_false_positive = sum(1 for j in confident if j.verdict and not j.gold)
    n_false_negative = sum(1 for j in confident if not j.verdict and j.gold)
    n_errors = sum(1 for j in confident if j.error)
    n_correct = len(confident) - n_errors
    overall_rate = n_errors / len(confident) if confident else None

    score_edges, error_counts, success_counts = _score_overlay(confident, n_score_bands)

    # Normalize record keys to str ONCE here (symmetric with _normalize_gold), so
    # the slice lookups -- keyed by the stringified judgement ids -- hit even when
    # the caller passed an int-keyed map.
    slice_records = _normalize_records(records) if records is not None else None
    all_slices = _build_slices(
        confident,
        overall_rate,
        records=slice_records,
        id_key=id_key,
        source_key=source_key,
        n_score_bands=n_score_bands,
    )
    ranked = _rank_slices(all_slices)
    shown = ranked[:max_slices]

    return FailureModeSection(
        title=title,
        n_judged=len(judged),
        n_correct=n_correct,
        n_errors=n_errors,
        n_abstain=len(judged) - len(confident),
        n_false_positive=n_false_positive,
        n_false_negative=n_false_negative,
        score_edges=score_edges,
        error_counts=error_counts,
        success_counts=success_counts,
        slices=shown,
        n_slices_hidden=len(ranked) - len(shown),
    )


# ---------------------------------------------------------------------------
# Join + slice helpers
# ---------------------------------------------------------------------------


def _normalize_gold(gold_pairs: Collection[Collection[Hashable]]) -> set[frozenset[str]]:
    """Normalize any gold-pair collection to a set of 2-id ``frozenset``\\ s of strings.

    Accepts the ``set[frozenset[str]]`` shape :func:`evaluate` uses as well as a
    plain sequence of ``(left, right)`` tuples, keying ids as strings so the join
    matches the judgement log (which logs ids as strings). Malformed pairs (not
    exactly two distinct ids) are dropped.
    """
    normalized: set[frozenset[str]] = set()
    n_dropped = 0
    for pair in gold_pairs:
        key = frozenset(str(identifier) for identifier in pair)
        if len(key) == 2:
            normalized.add(key)
        else:
            n_dropped += 1
    if n_dropped:
        logger.warning(
            "profile_failure_mode: dropped %d malformed gold pair(s) (not exactly two "
            "distinct ids); this section logs omissions, never drops silently.",
            n_dropped,
        )
    return normalized


def _normalize_records(
    records: Mapping[Hashable, Mapping[str, Any]],
) -> dict[Hashable, Mapping[str, Any]]:
    """Re-key a records map by ``str`` id, symmetric with :func:`_normalize_gold`.

    The join stringifies judgement ids (the log stores ids as strings), so the
    slice lookups (:func:`_either_empty`, :func:`_source_slices`) must find each
    record by that same string key. A caller passing an ``int``-keyed map would
    otherwise miss every lookup silently -- reporting every field-emptiness slice
    as all-empty and skipping the source slice. Normalizing keys once at entry
    keeps the str-conversion out of every per-pair helper.
    """
    normalized: dict[Hashable, Mapping[str, Any]] = {
        str(key): value for key, value in records.items()
    }
    return normalized


def _join_gold(judgements: Sequence[Mapping[str, Any]], gold: set[frozenset[str]]) -> list[_Judged]:
    """Join each judgement row to its gold label (in the log's row order)."""
    joined: list[_Judged] = []
    for row in judgements:
        left_id = str(row["left_id"])
        right_id = str(row["right_id"])
        raw_score = row.get("score")
        joined.append(
            _Judged(
                left_id=left_id,
                right_id=right_id,
                score=None if raw_score is None else float(raw_score),
                verdict=row.get("verdict"),
                gold=frozenset({left_id, right_id}) in gold,
            )
        )
    return joined


def _in_unit_range(score: float | None) -> bool:
    """Whether a score is a real number in ``[0, 1]`` -- the only binnable range.

    The overlay counts (via :func:`~langres.report._report_html._histogram`) drop
    scores outside the ``[0, 1]`` edges, and a raw/signed-score judge can emit
    such values. Both the overlay and the score-band slices exclude out-of-range
    (and ``None``) scores through this one predicate, so the two views stay
    consistent and a negative score can never index a negative band.
    """
    return score is not None and 0.0 <= score <= 1.0


def _score_overlay(
    confident: Sequence[_Judged], n_bands: int
) -> tuple[list[float], list[float], list[float]]:
    """Bin error vs correct scores into ``[0, 1]`` bands; ``([], [], [])`` when no scores.

    The diff of the error distribution against the success distribution along the
    score axis: two histograms sharing fixed ``[0, 1]`` edges so the bands line up
    (density normalization happens at render time). Confident pairs without a
    score (a decision-only judge) or with an out-of-``[0, 1]`` score contribute to
    neither -- consistent with the score-band slices (see :func:`_in_unit_range`).
    """
    error_scores = [j.score for j in confident if j.error and _in_unit_range(j.score)]
    success_scores = [j.score for j in confident if not j.error and _in_unit_range(j.score)]
    if not error_scores and not success_scores:
        return [], [], []
    edges = [i / n_bands for i in range(n_bands + 1)]
    error_counts = _report_html._histogram([s for s in error_scores if s is not None], edges)
    success_counts = _report_html._histogram([s for s in success_scores if s is not None], edges)
    return edges, error_counts, success_counts


def _build_slices(
    confident: Sequence[_Judged],
    overall_rate: float | None,
    *,
    records: Mapping[Hashable, Mapping[str, Any]] | None,
    id_key: str,
    source_key: str | None,
    n_score_bands: int,
) -> list[FailureSlice]:
    """Build every failure slice: score bands, then (with records) field / source."""
    slices: list[FailureSlice] = []
    slices += _score_band_slices(confident, overall_rate, n_score_bands)
    if records is not None:
        slices += _field_empty_slices(
            confident, overall_rate, records=records, id_key=id_key, source_key=source_key
        )
        if source_key is not None:
            slices += _source_slices(
                confident, overall_rate, records=records, source_key=source_key
            )
    return slices


def _slice(
    dimension: str, value: str, bucket: Sequence[_Judged], overall_rate: float | None
) -> FailureSlice:
    """Aggregate one bucket of confident pairs into a :class:`FailureSlice`."""
    n = len(bucket)
    n_errors = sum(1 for j in bucket if j.error)
    rate = n_errors / n if n else None
    lift = rate / overall_rate if (rate is not None and overall_rate) else None
    return FailureSlice(
        dimension=dimension, value=value, n=n, n_errors=n_errors, error_rate=rate, lift=lift
    )


def _score_band_slices(
    confident: Sequence[_Judged], overall_rate: float | None, n_bands: int
) -> list[FailureSlice]:
    """One slice per score band; empty bands and out-of-``[0, 1]`` scores are dropped."""
    # Exclude out-of-range scores (consistent with the overlay), so a negative
    # score cannot produce a negative band index -> buckets[-1] KeyError.
    scored = [j for j in confident if _in_unit_range(j.score)]
    if not scored:
        return []
    buckets: dict[int, list[_Judged]] = {b: [] for b in range(n_bands)}
    for j in scored:
        assert j.score is not None  # narrowed by the _in_unit_range filter above
        band = min(int(j.score * n_bands), n_bands - 1)  # score==1.0 lands in the last band
        buckets[band].append(j)
    out: list[FailureSlice] = []
    for b in range(n_bands):
        if not buckets[b]:
            continue
        label = f"{b / n_bands:.2f}-{(b + 1) / n_bands:.2f}"
        out.append(_slice("score_band", label, buckets[b], overall_rate))
    return out


def _field_empty_slices(
    confident: Sequence[_Judged],
    overall_rate: float | None,
    *,
    records: Mapping[Hashable, Mapping[str, Any]],
    id_key: str,
    source_key: str | None,
) -> list[FailureSlice]:
    """One slice per content field: error rate among pairs missing that field on either side.

    A field is "empty on either side" when at least one of the pair's two records
    has no non-empty value for it (absent, ``None``, or blank string). Fields never
    empty across the confident pairs contribute no slice (nothing to concentrate).
    ``id_key`` and ``source_key`` are structural, not content, so both are skipped.
    """
    skip = {id_key} | ({source_key} if source_key is not None else set())
    fields = _content_fields(records, skip)
    out: list[FailureSlice] = []
    for field in fields:
        bucket = [j for j in confident if _either_empty(j, field, records)]
        if bucket:
            out.append(_slice(f"empty:{field}", "either side empty", bucket, overall_rate))
    return out


def _source_slices(
    confident: Sequence[_Judged],
    overall_rate: float | None,
    *,
    records: Mapping[Hashable, Mapping[str, Any]],
    source_key: str,
) -> list[FailureSlice]:
    """Cross-source vs same-source slices (skipped when no pair carries the source key)."""
    buckets: dict[str, list[_Judged]] = {"cross-source": [], "same-source": []}
    for j in confident:
        left = records.get(j.left_id)
        right = records.get(j.right_id)
        if left is None or right is None:
            continue
        if source_key not in left or source_key not in right:
            continue
        same = left.get(source_key) == right.get(source_key)
        buckets["same-source" if same else "cross-source"].append(j)
    return [
        _slice("source", value, bucket, overall_rate) for value, bucket in buckets.items() if bucket
    ]


def _content_fields(records: Mapping[Hashable, Mapping[str, Any]], skip: set[str]) -> list[str]:
    """The sorted union of the records' field names, minus the structural ``skip`` set."""
    names: set[str] = set()
    for record in records.values():
        names.update(record.keys())
    return sorted(names - skip)


def _either_empty(
    judged: _Judged, field: str, records: Mapping[Hashable, Mapping[str, Any]]
) -> bool:
    """``True`` if ``field`` is empty on either record of the pair (or a record is absent)."""
    left = records.get(judged.left_id)
    right = records.get(judged.right_id)
    if left is None or right is None:
        return True
    return _is_empty(left.get(field)) or _is_empty(right.get(field))


def _is_empty(value: Any) -> bool:
    """A field value counts as empty when absent, ``None``, or a blank/whitespace string."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _rank_slices(slices: Sequence[FailureSlice]) -> list[FailureSlice]:
    """Sort slices most-concentrated first: by lift (then error rate, then size), desc.

    A slice with an undefined lift (no errors anywhere, or an empty overall rate)
    sorts last -- there is no concentration to surface.
    """
    return sorted(
        slices,
        key=lambda s: (
            s.lift if s.lift is not None else -1.0,
            s.error_rate if s.error_rate is not None else -1.0,
            s.n,
        ),
        reverse=True,
    )


def _fmt_pct(value: float | None) -> str:
    """A fraction as a scannable percentage (3 sig figs); ``None``/NaN/Inf -> ``"n/a"``.

    ``%.3g`` (not ``%.2g``) so a full ``100%`` error rate -- common in a small
    slice -- renders as ``"100%"`` rather than ``.2g``'s scientific ``"1e+02%"``.
    """
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value * 100:.3g}%"


def _fmt_count_share(count: int, share: float | None) -> str:
    """``"<count> (<pct>)"`` when a share is defined, else the bare count."""
    if share is None:
        return f"{count:,}"
    return f"{count:,} ({_fmt_pct(share)})"
