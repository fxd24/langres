"""Corpus-field profile section: per-field completeness, cardinality, and length.

The second tabular block of the data-profile report. Given the records, it walks
every field and reports the three signals that decide whether a field is worth
comparing on: how often it is populated (non-null rate), how discriminative it is
(cardinality -- distinct / total), and how long its values run (mean/median
length, plus a compact length sparkline). Fields are ordered **most-missing
first** so the holes in the data surface at the top, and an all-null field is
**flagged, never dropped** -- a silently missing field is exactly the kind of
data problem a profile exists to expose.

Generic by construction: :func:`profile_corpus_fields` takes a plain
``Sequence`` of record mappings (``id -> value`` field dicts), never a
benchmark-coupled type. No records means nothing to profile, and the profiler
returns ``None`` so the section is simply absent (graceful degradation).
"""

from __future__ import annotations

import html
import logging
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from langres.core import _report_html
from langres.core.data_profile.base import ProfileSection

logger = logging.getLogger(__name__)

#: Default cap on how many field rows the table shows. Fields beyond the cap
#: (after the most-missing-first sort) are dropped from the table, and the
#: truncation is logged -- never silent.
_DEFAULT_TOP_N = 50

#: Width (characters) of the per-field length sparkline.
_SPARK_WIDTH = 8

#: Unicode block-elements low->high, for the length sparkline.
_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def _is_present(value: Any) -> bool:
    """True when a field value counts as populated (not null / not blank).

    ``None`` and whitespace-only strings are missing (mirroring the Comparator's
    MISSING rule, where an empty string is not a comparable value). Any non-string
    value (an ``int``, a ``bool``) is present.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def _length_histogram(lengths: Sequence[int], width: int = _SPARK_WIDTH) -> list[int]:
    """Bucket value lengths into ``width`` equal-width bins over ``[min, max]``.

    Returns ``width`` counts. When every value has the same length (a degenerate
    range) all mass lands in the first bucket -- the sparkline then shows a single
    spike, honestly conveying "one length".
    """
    if not lengths:
        return []
    lo, hi = min(lengths), max(lengths)
    counts = [0] * width
    if hi == lo:
        counts[0] = len(lengths)
        return counts
    span = hi - lo
    for length in lengths:
        idx = min(int((length - lo) / span * width), width - 1)
        counts[idx] += 1
    return counts


def _sparkline(counts: Sequence[int]) -> str:
    """Render bucket ``counts`` as a unicode block sparkline (empty string if all zero).

    A pure-unicode indicator so the same string renders in Markdown *and* HTML
    (no inline SVG needed, no escaping hazard). Each bucket maps to a block by its
    height relative to the tallest bucket.
    """
    peak = max(counts) if counts else 0
    if peak <= 0:
        return ""
    top = len(_SPARK_BLOCKS) - 1
    return "".join(_SPARK_BLOCKS[round(count / peak * top)] for count in counts)


class FieldStat(BaseModel):
    """Per-field completeness / cardinality / length stats for one corpus field.

    Attributes:
        name: The field key.
        n_present: Records with a populated value for this field.
        n_total: Total records profiled (the denominator for ``non_null_rate``).
        non_null_rate: ``n_present / n_total``.
        n_distinct: Distinct populated values (by string form).
        uniqueness: ``n_distinct / n_present`` (1.0 == a key); ``None`` when the
            field is all-null.
        mean_len: Mean populated-value length; ``None`` when all-null.
        median_len: Median populated-value length; ``None`` when all-null.
        all_null: True when the field is populated in no record (flagged, not
            dropped).
        len_hist: Length-distribution bucket counts backing the sparkline (empty
            when all-null).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    n_present: int
    n_total: int
    non_null_rate: float
    n_distinct: int
    uniqueness: float | None
    mean_len: float | None
    median_len: float | None
    all_null: bool
    len_hist: list[int]


class CorpusFieldSection(ProfileSection):
    """Per-field data-quality table over a record corpus.

    A frozen :class:`ProfileSection` holding one :class:`FieldStat` per shown
    field (sorted most-missing-first, capped at ``top_n``) plus the total field
    count so any truncation stays visible. Build it with
    :func:`profile_corpus_fields`.

    Attributes:
        n_records: Records profiled.
        n_fields_total: Distinct field keys seen across all records (before the
            ``top_n`` cap).
        fields: The shown per-field stats, most-missing first.
    """

    kind: Literal["corpus_field"] = "corpus_field"

    n_records: int
    n_fields_total: int
    fields: list[FieldStat]

    @property
    def _n_all_null(self) -> int:
        return sum(1 for f in self.fields if f.all_null)

    @property
    def _truncated(self) -> int:
        return self.n_fields_total - len(self.fields)

    # ------------------------------------------------------------ row rendering
    _HEADERS = ("field", "non-null", "distinct", "unique", "mean len", "median len", "length", "flags")

    def _display_cells(self, field: FieldStat) -> tuple[str, ...]:
        """One field's cells as display strings (markdown + HTML share this)."""
        return (
            field.name,
            _report_html._num(field.non_null_rate),
            f"{field.n_distinct:,}",
            _report_html._num(field.uniqueness),
            _report_html._num(field.mean_len, digits=1),
            _report_html._num(field.median_len, digits=1),
            _sparkline(field.len_hist),
            "all-null" if field.all_null else "",
        )

    # ------------------------------------------------------------ text surfaces
    def to_markdown(self) -> str:
        """Markdown: the field table, most-missing first, with a truncation note."""
        lines = [
            f"## {self.title}",
            "",
            f"{self.n_records:,} records; {self.n_fields_total:,} fields"
            + (f"; {self._n_all_null} all-null" if self._n_all_null else ""),
            "",
            "| " + " | ".join(self._HEADERS) + " |",
            "|" + "---|" * len(self._HEADERS),
        ]
        for field in self.fields:
            cells = [_report_html._md_cell(c) for c in self._display_cells(field)]
            lines.append("| " + " | ".join(cells) + " |")
        if not self.fields:
            lines.append("| _(no fields)_ |" + " |" * (len(self._HEADERS) - 1))
        if self._truncated > 0:
            lines += ["", f"_Showing {len(self.fields)} of {self.n_fields_total} fields "
                      f"(most-missing first); {self._truncated} truncated._"]
        return "\n".join(lines)

    @property
    def summary(self) -> dict[str, Any]:
        """Headline counts as a flat, title-namespaced dict."""
        return {
            f"{self.title}.n_records": self.n_records,
            f"{self.title}.n_fields": self.n_fields_total,
            f"{self.title}.n_fields_shown": len(self.fields),
            f"{self.title}.n_all_null_fields": self._n_all_null,
        }

    def rows(self) -> list[dict[str, Any]]:
        """One row per shown field -- ``pd.DataFrame(section.rows())``-ready (no ``len_hist``)."""
        return [
            {
                "field": f.name,
                "non_null_rate": f.non_null_rate,
                "n_distinct": f.n_distinct,
                "uniqueness": f.uniqueness,
                "mean_len": f.mean_len,
                "median_len": f.median_len,
                "all_null": f.all_null,
            }
            for f in self.fields
        ]

    # -------------------------------------------------------------- html panel
    def panels(self) -> list[str]:
        """A single ``<section>``: the field table (all names HTML-escaped)."""
        head = "<tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in self._HEADERS) + "</tr>"
        body_rows = []
        for field in self.fields:
            cells = "".join(f"<td>{html.escape(c)}</td>" for c in self._display_cells(field))
            body_rows.append(f"<tr>{cells}</tr>")
        table = f'<table class="errors">{head}{"".join(body_rows)}</table>'
        note = ""
        if self._truncated > 0:
            note = (
                f'<p class="empty">Showing {len(self.fields)} of {self.n_fields_total} '
                f"fields (most-missing first); {self._truncated} truncated.</p>"
            )
        return [_report_html.section(self.title, table + note)]


def profile_corpus_fields(
    records: Sequence[Mapping[str, Any]] | None,
    *,
    top_n: int = _DEFAULT_TOP_N,
    title: str = "Corpus fields",
) -> CorpusFieldSection | None:
    """Profile a record corpus into a :class:`CorpusFieldSection`.

    Args:
        records: The corpus -- a sequence of field mappings (``id -> value``).
            ``None`` or empty means nothing to profile, and the profiler returns
            ``None`` (the section is omitted). Field keys are the union across all
            records; a record missing a key counts as null for that field.
        top_n: Maximum field rows to show. After the most-missing-first sort, any
            fields beyond ``top_n`` are dropped from the table and the truncation
            is logged (never silent). :attr:`~CorpusFieldSection.n_fields_total`
            still reports the full count.
        title: Section heading; also namespaces this section's :attr:`summary` keys.

    Returns:
        A :class:`CorpusFieldSection`, or ``None`` when ``records`` is falsy.
    """
    if not records:
        return None

    n_records = len(records)
    field_names: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                seen.add(key)
                field_names.append(key)

    stats = [_profile_field(name, records, n_records) for name in field_names]
    # Most-missing first (highest null rate == lowest non-null rate); name
    # tie-break keeps the order deterministic.
    stats.sort(key=lambda f: (f.non_null_rate, f.name))

    n_fields_total = len(stats)
    if n_fields_total > top_n:
        logger.warning(
            "profile_corpus_fields: showing %d of %d fields (most-missing first); "
            "%d truncated by top_n=%d",
            top_n,
            n_fields_total,
            n_fields_total - top_n,
            top_n,
        )
        stats = stats[:top_n]

    return CorpusFieldSection(
        title=title,
        n_records=n_records,
        n_fields_total=n_fields_total,
        fields=stats,
    )


def _profile_field(name: str, records: Sequence[Mapping[str, Any]], n_records: int) -> FieldStat:
    """Compute one field's :class:`FieldStat` over the corpus."""
    present = [record[name] for record in records if name in record and _is_present(record[name])]
    n_present = len(present)
    distinct = {str(value) for value in present}
    lengths = [len(str(value)) for value in present]

    return FieldStat(
        name=name,
        n_present=n_present,
        n_total=n_records,
        non_null_rate=n_present / n_records,
        n_distinct=len(distinct),
        uniqueness=(len(distinct) / n_present) if n_present else None,
        mean_len=statistics.fmean(lengths) if lengths else None,
        median_len=statistics.median(lengths) if lengths else None,
        all_null=n_present == 0,
        len_hist=_length_histogram(lengths),
    )
