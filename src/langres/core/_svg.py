"""Pure-stdlib inline-SVG chart primitives for the EvalReport HTML tearsheet.

This module is the *only* place raw SVG strings and data->pixel coordinate math
live. Callers pass values in **data space**; every scaling step and the SVG
y-axis flip happen here, so the flip lives in exactly one place and cannot be
duplicated (or forgotten) by callers.

Design constraints:
- Pure Python standard library only (``math``, ``html``, ``typing``). No numpy,
  no matplotlib, no external assets -- an import-budget test enforces this.
- Output is deterministic: identical inputs produce a byte-identical string
  (no timestamps, no randomness).
- Non-finite coordinates (NaN/Infinity) are dropped before emission, so the
  output never contains a broken numeric attribute.
- Every caller-supplied string (labels, colors, annotations) is HTML-escaped.

Public API:
- ``Series``: one plotted series in data coordinates.
- ``line_chart``: a PR/ROC/reliability line-or-marker chart with an optional
  chance diagonal and top-right annotations.
- ``bar_chart``: an overlaid grouped bar chart (e.g. a score histogram split
  gold vs non-gold).
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Literal

# Plot padding (pixels). Left/bottom are generous so tick labels never clip.
_LEFT_PAD = 44
_RIGHT_PAD = 12
_TOP_PAD = 12
_BOTTOM_PAD = 28


@dataclass(frozen=True)
class Series:
    """One plotted series in data space.

    Attributes:
        points: (x, y) pairs in DATA coordinates (not pixels). Non-finite
            points are dropped at render time.
        stroke: any CSS color string (html-escaped on output).
        label: legend label (html-escaped on output). Pass ``""`` to omit the
            series from the legend.
        kind: ``"line"`` renders a single ``<polyline>``; ``"markers"`` renders
            one ``<circle>`` per point.
    """

    points: list[tuple[float, float]]
    stroke: str
    label: str
    kind: Literal["line", "markers"] = "line"


def _scale(value: float, dmin: float, dmax: float, rmin: float, rmax: float) -> float:
    """Linearly map ``value`` from a data range onto a pixel range.

    Args:
        value: The data-space value to map.
        dmin: Minimum of the data range.
        dmax: Maximum of the data range.
        rmin: Pixel value that ``dmin`` maps to.
        rmax: Pixel value that ``dmax`` maps to.

    Returns:
        The mapped value. When ``dmax == dmin`` (a degenerate range) the range
        midpoint ``(rmin + rmax) / 2`` is returned instead of dividing by zero.
        Callers guard non-finite inputs before calling this.
    """
    if dmax == dmin:
        return (rmin + rmax) / 2.0
    return rmin + (value - dmin) * (rmax - rmin) / (dmax - dmin)


def _esc(text: str) -> str:
    """HTML-escape a caller-supplied string for safe SVG embedding."""
    return html.escape(text, quote=True)


def _fmt(value: float) -> str:
    """Format a number to at most 2 decimals, trailing zeros stripped."""
    text = f"{value:.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _ticks(dmin: float, dmax: float, n: int) -> list[float]:
    """Return ``n`` evenly spaced tick values across ``[dmin, dmax]``."""
    if n < 2:
        return [float(dmin), float(dmax)]
    step = (dmax - dmin) / (n - 1)
    return [dmin + step * i for i in range(n)]


def _svg_open(width: int, height: int) -> str:
    """Open a responsive, theme-neutral ``<svg>`` element."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" role="img" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'style="max-width:100%;height:auto">'
    )


def _axes(
    *,
    plot_left: float,
    plot_right: float,
    plot_top: float,
    plot_bottom: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    x_label: str,
    y_label: str,
    n_ticks: int,
) -> list[str]:
    """Render the L-shaped axis frame, tick marks, tick labels, and axis titles.

    Uses ``currentColor`` throughout so the frame is legible in both light and
    dark themes. The y ticks apply the same bottom->top flip as the data.
    """
    parts: list[str] = [
        f'<path d="M {_fmt(plot_left)} {_fmt(plot_top)} '
        f"L {_fmt(plot_left)} {_fmt(plot_bottom)} "
        f'L {_fmt(plot_right)} {_fmt(plot_bottom)}" '
        'fill="none" stroke="currentColor" stroke-opacity="0.6" stroke-width="1"/>'
    ]
    for tick in _ticks(x0, x1, n_ticks):
        px = _scale(tick, x0, x1, plot_left, plot_right)
        parts.append(
            f'<line x1="{_fmt(px)}" y1="{_fmt(plot_bottom)}" '
            f'x2="{_fmt(px)}" y2="{_fmt(plot_bottom + 4)}" '
            'stroke="currentColor" stroke-opacity="0.6"/>'
        )
        parts.append(
            f'<text x="{_fmt(px)}" y="{_fmt(plot_bottom + 15)}" '
            'font-size="9" text-anchor="middle" fill="currentColor">'
            f"{_esc(_fmt(tick))}</text>"
        )
    for tick in _ticks(y0, y1, n_ticks):
        py = _scale(tick, y0, y1, plot_bottom, plot_top)
        parts.append(
            f'<line x1="{_fmt(plot_left - 4)}" y1="{_fmt(py)}" '
            f'x2="{_fmt(plot_left)}" y2="{_fmt(py)}" '
            'stroke="currentColor" stroke-opacity="0.6"/>'
        )
        parts.append(
            f'<text x="{_fmt(plot_left - 6)}" y="{_fmt(py + 3)}" '
            'font-size="9" text-anchor="end" fill="currentColor">'
            f"{_esc(_fmt(tick))}</text>"
        )
    if x_label:
        parts.append(
            f'<text x="{_fmt((plot_left + plot_right) / 2)}" y="{_fmt(plot_bottom + 26)}" '
            'font-size="10" text-anchor="middle" fill="currentColor">'
            f"{_esc(x_label)}</text>"
        )
    if y_label:
        cy = (plot_top + plot_bottom) / 2
        parts.append(
            f'<text x="10" y="{_fmt(cy)}" font-size="10" text-anchor="middle" '
            f'fill="currentColor" transform="rotate(-90 10 {_fmt(cy)})">'
            f"{_esc(y_label)}</text>"
        )
    return parts


def _legend(entries: list[tuple[str, str]], x: float, y: float) -> list[str]:
    """Render a compact top-left legend; entries with an empty label are skipped.

    Args:
        entries: ``(label, stroke)`` pairs.
        x: Left edge (pixels) of the plot area.
        y: Top edge (pixels) of the plot area.
    """
    parts: list[str] = []
    row = 0
    for label, stroke in entries:
        if not label:
            continue
        ly = y + 12 + row * 14
        parts.append(
            f'<line x1="{_fmt(x + 6)}" y1="{_fmt(ly - 3)}" '
            f'x2="{_fmt(x + 20)}" y2="{_fmt(ly - 3)}" '
            f'stroke="{_esc(stroke)}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{_fmt(x + 24)}" y="{_fmt(ly)}" font-size="9" '
            f'text-anchor="start" fill="currentColor">{_esc(label)}</text>'
        )
        row += 1
    return parts


def line_chart(
    series: list[Series],
    *,
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
    width: int = 340,
    height: int = 260,
    x_label: str = "",
    y_label: str = "",
    diagonal: bool = False,
    annotations: list[str] | None = None,
    n_ticks: int = 5,
) -> str:
    """Render a line/marker chart as a self-contained ``<svg>`` string.

    Args:
        series: Series to plot. ``"line"`` series need >= 2 finite points to
            emit a ``<polyline>``; ``"markers"`` series emit one ``<circle>``
            per finite point.
        x_domain: ``(min, max)`` of the x data range.
        y_domain: ``(min, max)`` of the y data range. The data max is placed at
            the TOP of the plot (the SVG y-axis flip happens here).
        width: SVG width in pixels.
        height: SVG height in pixels.
        x_label: X-axis title (html-escaped).
        y_label: Y-axis title (html-escaped).
        diagonal: If True, draw a dashed chance line from
            ``(x_domain[0], y_domain[0])`` to ``(x_domain[1], y_domain[1])``.
        annotations: Short strings drawn top-right (each html-escaped), e.g.
            ``"AUC = 0.950"``.
        n_ticks: Number of ticks per axis.

    Returns:
        A complete ``<svg ...>...</svg>`` string.
    """
    x0, x1 = x_domain
    y0, y1 = y_domain
    plot_left = float(_LEFT_PAD)
    plot_right = float(width - _RIGHT_PAD)
    plot_top = float(_TOP_PAD)
    plot_bottom = float(height - _BOTTOM_PAD)

    parts = [_svg_open(width, height)]
    parts.extend(
        _axes(
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            x_label=x_label,
            y_label=y_label,
            n_ticks=n_ticks,
        )
    )

    if diagonal:
        dx1 = _scale(x0, x0, x1, plot_left, plot_right)
        dy1 = _scale(y0, y0, y1, plot_bottom, plot_top)
        dx2 = _scale(x1, x0, x1, plot_left, plot_right)
        dy2 = _scale(y1, y0, y1, plot_bottom, plot_top)
        parts.append(
            f'<line x1="{_fmt(dx1)}" y1="{_fmt(dy1)}" '
            f'x2="{_fmt(dx2)}" y2="{_fmt(dy2)}" '
            'stroke="currentColor" stroke-opacity="0.4" stroke-dasharray="5,5"/>'
        )

    for s in series:
        pixels = [
            (_scale(x, x0, x1, plot_left, plot_right), _scale(y, y0, y1, plot_bottom, plot_top))
            for x, y in s.points
            if math.isfinite(x) and math.isfinite(y)
        ]
        stroke = _esc(s.stroke)
        if s.kind == "line":
            if len(pixels) >= 2:
                coords = " ".join(f"{_fmt(px)},{_fmt(py)}" for px, py in pixels)
                parts.append(
                    f'<polyline fill="none" stroke="{stroke}" stroke-width="2" points="{coords}"/>'
                )
        else:
            for px, py in pixels:
                parts.append(f'<circle cx="{_fmt(px)}" cy="{_fmt(py)}" r="3" fill="{stroke}"/>')

    parts.extend(_legend([(s.label, s.stroke) for s in series], plot_left, plot_top))

    for i, note in enumerate(annotations or []):
        ay = plot_top + 12 + i * 13
        parts.append(
            f'<text x="{_fmt(plot_right - 4)}" y="{_fmt(ay)}" font-size="10" '
            f'text-anchor="end" fill="currentColor">{_esc(note)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def bar_chart(
    bin_edges: list[float],
    series: list[tuple[str, str, list[float]]],
    *,
    width: int = 340,
    height: int = 260,
    x_label: str = "",
    y_label: str = "",
) -> str:
    """Render an overlaid grouped bar chart as a self-contained ``<svg>`` string.

    One bar group per bin; series are drawn overlaid (semi-transparent) so a
    two-series split (e.g. a score histogram of gold vs non-gold) reads clearly.
    Bars are anchored to a zero baseline. Non-finite or non-positive counts
    emit no ``<rect>`` (the axes still render), so an empty/degenerate histogram
    renders as bare axes rather than broken attributes.

    Args:
        bin_edges: ``B + 1`` ascending edges defining ``B`` bins.
        series: ``(label, stroke, counts)`` tuples, where ``counts`` has length
            ``B`` (one bar per bin). ``label`` populates the legend.
        width: SVG width in pixels.
        height: SVG height in pixels.
        x_label: X-axis title (html-escaped).
        y_label: Y-axis title (html-escaped).

    Returns:
        A complete ``<svg ...>...</svg>`` string.
    """
    plot_left = float(_LEFT_PAD)
    plot_right = float(width - _RIGHT_PAD)
    plot_top = float(_TOP_PAD)
    plot_bottom = float(height - _BOTTOM_PAD)

    x0 = bin_edges[0] if bin_edges else 0.0
    x1 = bin_edges[-1] if bin_edges else 1.0

    finite_counts = [c for _, _, counts in series for c in counts if math.isfinite(c)]
    y_max = max(finite_counts) if finite_counts else 0.0
    if y_max <= 0.0:
        y_max = 1.0  # avoid a degenerate y range; no bars are drawn either way

    parts = [_svg_open(width, height)]
    parts.extend(
        _axes(
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            x0=x0,
            x1=x1,
            y0=0.0,
            y1=y_max,
            x_label=x_label,
            y_label=y_label,
            n_ticks=5,
        )
    )

    n_bins = max(len(bin_edges) - 1, 0)
    for _label, stroke, counts in series:
        fill = _esc(stroke)
        for i in range(min(n_bins, len(counts))):
            count = counts[i]
            if not math.isfinite(count) or count <= 0.0:
                continue
            left_px = _scale(bin_edges[i], x0, x1, plot_left, plot_right)
            right_px = _scale(bin_edges[i + 1], x0, x1, plot_left, plot_right)
            top_px = _scale(count, 0.0, y_max, plot_bottom, plot_top)
            bar_w = max(right_px - left_px - 1.0, 0.0)
            bar_h = max(plot_bottom - top_px, 0.0)
            parts.append(
                f'<rect x="{_fmt(left_px)}" y="{_fmt(top_px)}" '
                f'width="{_fmt(bar_w)}" height="{_fmt(bar_h)}" '
                f'fill="{fill}" fill-opacity="0.55"/>'
            )

    parts.extend(_legend([(label, stroke) for label, stroke, _ in series], plot_left, plot_top))
    parts.append("</svg>")
    return "".join(parts)
