"""Hero (KPI) profile section: the at-a-glance headline of the whole tearsheet.

The first block a reader sees. It is *derivative* -- it computes nothing from raw
data; it distils the numbers the other sections already produced into a compact
row of big-number KPI cards: how many records and clusters, how rare a true match
is (positive-pair prevalence + the ``1:N`` class-imbalance ratio), and how cleanly
*some* signal separates matches from non-matches (separability AUC). Those five
numbers frame every downstream ER decision, so they belong at the top.

Built with :func:`build_hero`, which reads the headline numbers off a
:class:`~langres.data.data_profile.label_structure.LabelStructureSection` and a
:class:`~langres.data.data_profile.separability.SeparabilitySection` when present.
Every KPI is optional: a missing source section leaves that card as an honest
``"n/a"`` (via :func:`~langres.report._report_html._num`), never a raise -- the same
graceful-degradation contract the rest of the report keeps.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from typing import Any, Literal

from langres.report import _report_html
from langres.data.data_profile.base import ProfileSection
from langres.data.data_profile.label_structure import LabelStructureSection
from langres.data.data_profile.separability import SeparabilitySection


def _fmt_count(value: int | None) -> str:
    """Thousands-separated integer, or ``"n/a"`` when the count is undefined."""
    if value is None:
        return "n/a"
    return f"{value:,}"


def _fmt_prevalence(value: float | None) -> str:
    """Positive-pair prevalence as a scannable percentage (2 sig figs).

    The hero card trades the raw float (``0.00638742``) for the percentage a
    reader actually scans (``0.64%``). ``None``/NaN/Inf and an exactly-zero
    prevalence (a degenerate gold with no positive pairs) render as ``"n/a"`` --
    the report's honest-degradation contract -- never a crash. The Label-structure
    table keeps full ``%g`` precision (its own ``_fmt_g``); only the hero abridges.
    """
    if value is None or not math.isfinite(value) or value == 0:
        return "n/a"
    return f"{value * 100:.2g}%"


def _fmt_imbalance(value: float | None) -> str:
    """The class-imbalance ratio as ``"1:N"``; ``None``/NaN/Inf -> ``"n/a"``."""
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"1:{value:,.0f}"


class HeroSection(ProfileSection):
    """The KPI-card headline: records, clusters, prevalence, imbalance, AUC.

    A frozen :class:`ProfileSection` holding five optional headline numbers lifted
    from the other sections. Every field is ``| None`` so a report missing a source
    section (no gold, no separability signal) still renders a valid hero with
    ``"n/a"`` cards. Build it with :func:`build_hero`.

    Attributes:
        n_records: Total records profiled (from the label-structure section).
        n_clusters: Number of gold clusters (from the label-structure section).
        prevalence: Positive-pair prevalence -- the share of all candidate pairs
            that are true matches (from the label-structure section).
        imbalance_ratio: Negatives per positive, the ``N`` in ``1:N`` (from the
            label-structure section).
        separability_auc: How cleanly the (first) separability signal ranks
            positives above negatives (from the separability section).
    """

    kind: Literal["hero"] = "hero"

    n_records: int | None
    n_clusters: int | None
    prevalence: float | None
    imbalance_ratio: float | None
    separability_auc: float | None

    # ------------------------------------------------------------- shared render
    def _cards(self) -> list[tuple[str, str]]:
        """The KPI cards as ``(label, display)`` pairs (markdown + HTML share this)."""
        return [
            ("records", _fmt_count(self.n_records)),
            ("clusters", _fmt_count(self.n_clusters)),
            ("positive-pair prevalence", _fmt_prevalence(self.prevalence)),
            ("class imbalance (pos:neg)", _fmt_imbalance(self.imbalance_ratio)),
            ("separability AUC", _report_html._num(self.separability_auc)),
        ]

    # ------------------------------------------------------------ text surfaces
    def to_markdown(self) -> str:
        """Markdown: a compact metric table of the five KPIs."""
        lines = [f"## {self.title}", "", "| metric | value |", "|---|---|"]
        lines += [
            f"| {_report_html._md_cell(label)} | {_report_html._md_cell(value)} |"
            for label, value in self._cards()
        ]
        return "\n".join(lines)

    @property
    def summary(self) -> dict[str, Any]:
        """Headline numbers as a flat, title-namespaced dict (log it, assert on it)."""
        return {
            f"{self.title}.n_records": self.n_records,
            f"{self.title}.n_clusters": self.n_clusters,
            f"{self.title}.prevalence": self.prevalence,
            f"{self.title}.imbalance_ratio": self.imbalance_ratio,
            f"{self.title}.separability_auc": self.separability_auc,
        }

    def rows(self) -> list[dict[str, Any]]:
        """A single row of the raw KPI values -- ``pd.DataFrame(section.rows())``-ready."""
        return [
            {
                "n_records": self.n_records,
                "n_clusters": self.n_clusters,
                "prevalence": self.prevalence,
                "imbalance_ratio": self.imbalance_ratio,
                "separability_auc": self.separability_auc,
            }
        ]

    # -------------------------------------------------------------- html panel
    def panels(self) -> list[str]:
        """A single ``<section>``: a big-number KPI card grid (inline styles only).

        The cards sit in a CSS grid of ``auto-fill`` tracks
        (``repeat(auto-fill, minmax(150px, 1fr))``): ``auto-fill`` keeps the empty
        trailing tracks, so a lone last card stays one cell wide and left-aligned
        instead of stretching across the row (as ``auto-fit`` or a flex row would).
        It rides on the shared :mod:`langres.report._report_html` CSS variables
        (``--line``/``--muted``/``--fg``), inheriting the report's light/dark
        palette without a new stylesheet, and reflows to fewer columns on narrow
        widths.
        """
        cards = "".join(
            f'<div style="border:1px solid var(--line);'
            f'border-radius:8px;padding:12px 14px;">'
            f'<div style="font-size:0.78rem;color:var(--muted);">{html.escape(label)}</div>'
            f'<div style="font-size:1.6rem;font-weight:600;font-variant-numeric:tabular-nums;">'
            f"{html.escape(value)}</div>"
            f"</div>"
            for label, value in self._cards()
        )
        grid = (
            '<div style="display:grid;'
            'grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;">'
            f"{cards}</div>"
        )
        return [_report_html.section(self.title, grid)]


def build_hero(
    sections: Sequence[ProfileSection], *, title: str = "Overview"
) -> HeroSection | None:
    """Distil a hero KPI section from already-built profile sections.

    Reads the headline numbers off the first
    :class:`~langres.data.data_profile.label_structure.LabelStructureSection`
    (records / clusters / prevalence / imbalance) and the first
    :class:`~langres.data.data_profile.separability.SeparabilitySection` (AUC) in
    ``sections``. A KPI whose source section is absent stays ``None`` and renders
    as ``"n/a"``.

    Args:
        sections: The other report sections to summarise (any order; unrelated
            kinds are ignored).
        title: Section heading; also namespaces this section's :attr:`summary` keys.

    Returns:
        A :class:`HeroSection`, or ``None`` when *neither* a label-structure nor a
        separability section is present (there is nothing to headline, so the hero
        is simply omitted -- graceful degradation).
    """
    label = next((s for s in sections if isinstance(s, LabelStructureSection)), None)
    sep = next((s for s in sections if isinstance(s, SeparabilitySection)), None)
    if label is None and sep is None:
        return None
    return HeroSection(
        title=title,
        n_records=label.n_records if label is not None else None,
        n_clusters=label.n_clusters if label is not None else None,
        prevalence=label.prevalence if label is not None else None,
        imbalance_ratio=label.imbalance_ratio if label is not None else None,
        separability_auc=sep.auc if sep is not None else None,
    )
