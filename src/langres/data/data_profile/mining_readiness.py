"""Mining-readiness profile section: is this labeled set ready to train on?

The training-preparation companion to the label-structure block. Where label
structure profiles the *gold clustering*, this section profiles a *labeled pair
set* the way the training wave will consume it: how balanced the classes are
(the AnyMatch 2:1 lever), how many **hard positives** a miner surfaced, how many
pairs a denoiser flagged as likely mislabeled, and -- when per-pair margins /
``|1 − p|`` scores are supplied -- how confident a model is across the set.

It is a **pure consumer of precomputed stats**: it never runs a matcher or
featurizes anything (that would drag scikit-learn into the import-light
``data_profile`` package). The caller runs the miners
(:mod:`langres.data.mining`) and hands the resulting counts (and optional
margins) to :func:`profile_mining_readiness`. Every input beyond the two class
counts is optional and degrades to an honest ``"n/a"`` card when absent -- the
same graceful-degradation contract the rest of the report keeps.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from typing import Any, Literal

from langres.core import _report_html, _svg
from langres.data.data_profile.base import ProfileSection

#: Single-series bar color for the margin histogram (explicit, not
#: ``currentColor``, so it reads in light and dark themes -- matches the
#: label-structure / eval_report palette).
_COLOR_MARGIN = "#8338ec"

#: Default number of bins for the margin / ``|1 − p|`` histogram.
_DEFAULT_MARGIN_BINS = 20


class MiningReadinessSection(ProfileSection):
    """Class balance + hard-positive / label-noise yields of a labeled pair set.

    A frozen :class:`ProfileSection` holding the headline mining-readiness counts
    and (optionally) a precomputed per-pair margin histogram. Build it with
    :func:`profile_mining_readiness`; it computes nothing that needs a model.

    Attributes:
        n_positive: Positive (match) labeled pairs.
        n_negative: Negative (non-match) labeled pairs.
        n_hard_positive: Hard positives a miner surfaced (e.g.
            :func:`~langres.data.mining.mine_misclassified_pairs`); ``None`` when
            not measured.
        n_flagged_noise: Pairs a denoiser flagged as likely mislabeled (e.g.
            :func:`~langres.data.mining.denoise_pairs`); ``None`` when not measured.
        margin_label: Human label of the per-pair margin scores (e.g.
            ``"|1 − p|"``); shown on the histogram x-axis.
        margin_edges: Bin edges (``B + 1`` values) of the margin histogram; empty
            when no margins were supplied.
        margin_counts: Per-bin counts (``B`` values); empty when no margins.
    """

    kind: Literal["mining_readiness"] = "mining_readiness"

    n_positive: int
    n_negative: int
    n_hard_positive: int | None
    n_flagged_noise: int | None
    margin_label: str
    margin_edges: list[float]
    margin_counts: list[float]

    # ------------------------------------------------------------- derived stats
    @property
    def total(self) -> int:
        """Total labeled pairs (positives + negatives)."""
        return self.n_positive + self.n_negative

    @property
    def positive_share(self) -> float | None:
        """Fraction of pairs that are positive; ``None`` for an empty set."""
        return self.n_positive / self.total if self.total else None

    @property
    def imbalance_ratio(self) -> float | None:
        """Negatives per positive (the ``N`` in ``1:N``); ``None`` with no positives."""
        return self.n_negative / self.n_positive if self.n_positive else None

    @property
    def hard_positive_share(self) -> float | None:
        """Hard positives as a fraction of positives; ``None`` when unmeasured/empty."""
        if self.n_hard_positive is None or self.n_positive == 0:
            return None
        return self.n_hard_positive / self.n_positive

    @property
    def flagged_noise_share(self) -> float | None:
        """Flagged pairs as a fraction of all pairs; ``None`` when unmeasured/empty."""
        if self.n_flagged_noise is None or self.total == 0:
            return None
        return self.n_flagged_noise / self.total

    # ------------------------------------------------------------- shared render
    def _metrics_kv(self) -> list[tuple[str, str]]:
        """The headline metrics as ``(label, display)`` pairs (markdown + HTML share this)."""
        imbalance = f"1:{self.imbalance_ratio:,.1f}" if self.imbalance_ratio is not None else "n/a"
        return [
            ("positive pairs", f"{self.n_positive:,}"),
            ("negative pairs", f"{self.n_negative:,}"),
            ("total pairs", f"{self.total:,}"),
            ("positive share", _fmt_pct(self.positive_share)),
            ("class imbalance (pos:neg)", imbalance),
            ("hard positives", _fmt_count_share(self.n_hard_positive, self.hard_positive_share)),
            (
                "flagged label noise",
                _fmt_count_share(self.n_flagged_noise, self.flagged_noise_share),
            ),
        ]

    # ------------------------------------------------------------ text surfaces
    def to_markdown(self) -> str:
        """Markdown: a metrics table (and a note when the set is empty)."""
        lines = [f"## {self.title}", "", "| metric | value |", "|---|---|"]
        lines += [
            f"| {_report_html._md_cell(k)} | {_report_html._md_cell(v)} |"
            for k, v in self._metrics_kv()
        ]
        if self.total == 0:
            lines += ["", "_No labeled pairs: nothing to train on yet._"]
        return "\n".join(lines)

    @property
    def summary(self) -> dict[str, Any]:
        """Headline numbers as a flat, title-namespaced dict (collision-free per report)."""
        return {
            f"{self.title}.n_positive": self.n_positive,
            f"{self.title}.n_negative": self.n_negative,
            f"{self.title}.imbalance_ratio": self.imbalance_ratio,
            f"{self.title}.n_hard_positive": self.n_hard_positive,
            f"{self.title}.n_flagged_noise": self.n_flagged_noise,
        }

    def rows(self) -> list[dict[str, Any]]:
        """A single row of the raw readiness counts -- ``pd.DataFrame(section.rows())``-ready."""
        return [
            {
                "n_positive": self.n_positive,
                "n_negative": self.n_negative,
                "imbalance_ratio": self.imbalance_ratio,
                "n_hard_positive": self.n_hard_positive,
                "n_flagged_noise": self.n_flagged_noise,
            }
        ]

    # -------------------------------------------------------------- html panel
    def panels(self) -> list[str]:
        """A single ``<section>``: the metrics KV table, plus the margin histogram when present."""
        kv = "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>"
            for k, v in self._metrics_kv()
        )
        body = f'<table class="kv">{kv}</table>'
        if self.margin_counts:
            body += _svg.bar_chart(
                self.margin_edges,
                [(self.margin_label, _COLOR_MARGIN, self.margin_counts)],
                x_label=self.margin_label,
                y_label="count",
            )
        return [_report_html.section(self.title, body)]


def profile_mining_readiness(
    *,
    n_positive: int,
    n_negative: int,
    n_hard_positive: int | None = None,
    n_flagged_noise: int | None = None,
    margins: Sequence[float] | None = None,
    n_bins: int = _DEFAULT_MARGIN_BINS,
    margin_label: str = "|1 − p|",
    title: str = "Mining readiness",
) -> MiningReadinessSection:
    """Build a :class:`MiningReadinessSection` from precomputed mining stats.

    Pure assembly over counts the caller already computed -- it runs no matcher and
    featurizes nothing (keeping ``data_profile`` import-light). When ``margins`` is
    supplied, it is binned into a histogram here (a stat computation, not a model
    call); otherwise the histogram is omitted and the section still renders.

    Args:
        n_positive: Positive (match) labeled pairs.
        n_negative: Negative (non-match) labeled pairs.
        n_hard_positive: Optional count of hard positives a miner surfaced.
        n_flagged_noise: Optional count of pairs a denoiser flagged.
        margins: Optional per-pair confidence margins (e.g. ``|1 − p|`` or
            ``|p − threshold|``); binned into the section's histogram. Non-finite
            values are dropped.
        n_bins: Number of histogram bins over the observed margin range.
        margin_label: Human label of the margin scores (histogram x-axis).
        title: Section heading; also the report lookup key and the namespace for
            this section's :attr:`~MiningReadinessSection.summary` keys.

    Returns:
        A :class:`MiningReadinessSection` (always -- there is no ``None`` path; an
        all-zero set renders honestly).
    """
    edges, counts = _margin_histogram(margins, n_bins)
    return MiningReadinessSection(
        title=title,
        n_positive=n_positive,
        n_negative=n_negative,
        n_hard_positive=n_hard_positive,
        n_flagged_noise=n_flagged_noise,
        margin_label=margin_label,
        margin_edges=edges,
        margin_counts=counts,
    )


def _margin_histogram(
    margins: Sequence[float] | None, n_bins: int
) -> tuple[list[float], list[float]]:
    """Bin finite ``margins`` into ``(edges, counts)``; ``([], [])`` when absent/empty.

    A degenerate range (all margins equal) is widened to unit width so the bars
    keep real pixel extent instead of collapsing onto one line.
    """
    if not margins:
        return [], []
    finite = [float(m) for m in margins if math.isfinite(m)]
    if not finite:
        return [], []
    lo, hi = min(finite), max(finite)
    if hi <= lo:
        hi = lo + 1.0
    edges = [lo + (hi - lo) * i / n_bins for i in range(n_bins + 1)]
    counts = _report_html._histogram(finite, edges)
    return edges, counts


def _fmt_pct(value: float | None) -> str:
    """A fraction as a scannable percentage (2 sig figs); ``None``/NaN/Inf -> ``"n/a"``."""
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value * 100:.2g}%"


def _fmt_count_share(count: int | None, share: float | None) -> str:
    """``"<count> (<pct>)"`` for a measured count, else ``"n/a"`` (never measured)."""
    if count is None:
        return "n/a"
    if share is None:
        return f"{count:,}"
    return f"{count:,} ({_fmt_pct(share)})"
