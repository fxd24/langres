"""Tests for the pure-stdlib inline-SVG chart primitives (``langres.core._svg``).

Tested by STRUCTURE (element counts, parsed attributes, escaped substrings),
never by full-string equality, so cosmetic tweaks don't churn the suite.
"""

import re

from langres.core._svg import Series, _scale, bar_chart, line_chart


def _polyline_points(svg: str) -> list[tuple[float, float]]:
    """Parse the first ``<polyline points="...">`` into (x, y) float pairs."""
    match = re.search(r'<polyline[^>]*points="([^"]*)"', svg)
    assert match is not None, "expected a <polyline> in the output"
    return [(float(x), float(y)) for x, y in (p.split(",") for p in match.group(1).split())]


class TestScale:
    def test_endpoints(self) -> None:
        assert _scale(0.0, 0.0, 1.0, 10.0, 20.0) == 10.0
        assert _scale(1.0, 0.0, 1.0, 10.0, 20.0) == 20.0

    def test_midpoint(self) -> None:
        assert _scale(0.5, 0.0, 1.0, 0.0, 100.0) == 50.0

    def test_degenerate_range_returns_range_midpoint_no_zerodiv(self) -> None:
        # dmax == dmin must not raise ZeroDivisionError; returns range midpoint.
        assert _scale(5.0, 3.0, 3.0, 10.0, 20.0) == 15.0

    def test_y_flip_via_scale(self) -> None:
        # With the bottom->top pixel range, larger data-y maps to smaller pixel-y.
        y_top = _scale(1.0, 0.0, 1.0, 232.0, 12.0)
        y_bottom = _scale(0.0, 0.0, 1.0, 232.0, 12.0)
        assert y_top < y_bottom


class TestLineChartHappyPath:
    def test_returns_a_single_svg(self) -> None:
        out = line_chart(
            [Series([(0.0, 0.0), (1.0, 1.0)], "#f00", "curve")],
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
        )
        assert out.count("<svg") == 1
        assert out.rstrip().endswith("</svg>")
        assert 'role="img"' in out
        assert 'viewBox="0 0 340 260"' in out
        assert "max-width:100%" in out

    def test_two_point_line_yields_one_polyline_with_two_points(self) -> None:
        out = line_chart(
            [Series([(0.0, 0.0), (1.0, 1.0)], "#f00", "curve")],
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
        )
        assert out.count("<polyline") == 1
        assert len(_polyline_points(out)) == 2

    def test_markers_kind_yields_one_circle_per_point_and_no_polyline(self) -> None:
        out = line_chart(
            [Series([(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)], "#00f", "rel", "markers")],
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
        )
        assert out.count("<circle") == 3
        assert "<polyline" not in out


class TestYFlip:
    def test_data_max_maps_above_data_min(self) -> None:
        # Two points at the same x, one at y=0 (data min), one at y=1 (data max).
        out = line_chart(
            [Series([(0.0, 0.0), (0.0, 1.0)], "#f00", "c")],
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
        )
        pts = _polyline_points(out)
        (_, y_at_data_min), (_, y_at_data_max) = pts
        # SVG y grows downward: data y=1 (max) sits at the TOP -> smaller pixel-y.
        assert y_at_data_max < y_at_data_min


class TestNaNGuard:
    def test_nonfinite_points_are_dropped_and_no_nan_or_infinity_leaks(self) -> None:
        out = line_chart(
            [
                Series(
                    [
                        (0.0, 0.0),
                        (0.5, float("nan")),
                        (float("inf"), 0.2),
                        (1.0, 1.0),
                    ],
                    "#f00",
                    "c",
                )
            ],
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
        )
        assert "NaN" not in out and "Infinity" not in out
        # Only the two finite points survive.
        assert len(_polyline_points(out)) == 2

    def test_all_nonfinite_line_emits_no_polyline_but_svg_still_returns(self) -> None:
        out = line_chart(
            [Series([(float("nan"), 0.0), (float("inf"), 1.0)], "#f00", "c")],
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
        )
        assert "<polyline" not in out
        assert out.count("<svg") == 1
        assert "NaN" not in out and "Infinity" not in out

    def test_single_finite_point_line_emits_no_polyline(self) -> None:
        # A "line" needs >= 2 finite points.
        out = line_chart(
            [Series([(0.5, 0.5), (float("nan"), 0.2)], "#f00", "c")],
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
        )
        assert "<polyline" not in out
        assert out.count("<svg") == 1


class TestEscaping:
    def test_axis_label_is_html_escaped_and_does_not_leak_a_tag(self) -> None:
        out = line_chart(
            [Series([(0.0, 0.0), (1.0, 1.0)], "#f00", "c")],
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
            x_label="a & b <c>",
        )
        assert "a &amp; b &lt;c&gt;" in out
        assert "<c>" not in out

    def test_annotation_appears_escaped(self) -> None:
        out = line_chart(
            [Series([(0.0, 0.0), (1.0, 1.0)], "#f00", "c")],
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
            annotations=["AUC = 0.950 & <x>"],
        )
        assert "AUC = 0.950 &amp; &lt;x&gt;" in out
        assert "<x>" not in out


class TestDiagonal:
    def test_diagonal_adds_a_dashed_chance_line(self) -> None:
        kwargs = {"x_domain": (0.0, 1.0), "y_domain": (0.0, 1.0)}
        base = line_chart([Series([(0.0, 0.0), (1.0, 1.0)], "#f00", "c")], **kwargs)
        diag = line_chart([Series([(0.0, 0.0), (1.0, 1.0)], "#f00", "c")], diagonal=True, **kwargs)
        assert "stroke-dasharray" not in base
        assert "stroke-dasharray" in diag
        # Chance line runs corner-to-corner: bottom-left (44,232) -> top-right (328,12).
        assert 'x1="44" y1="232"' in diag
        assert 'x2="328" y2="12"' in diag


class TestBarChart:
    def test_two_overlaid_series_render_one_rect_per_positive_count(self) -> None:
        out = bar_chart(
            [0.0, 0.5, 1.0],
            [("gold", "#0a0", [3.0, 1.0]), ("non-gold", "#a00", [1.0, 4.0])],
        )
        assert out.count("<svg") == 1
        assert out.count("<rect") == 4  # 2 bins x 2 series, all positive
        assert "NaN" not in out and "Infinity" not in out

    def test_empty_counts_render_axes_only(self) -> None:
        out = bar_chart([0.0, 1.0], [("gold", "#0a0", []), ("non-gold", "#a00", [])])
        assert out.count("<svg") == 1
        assert "<rect" not in out
        assert "NaN" not in out and "Infinity" not in out

    def test_zero_counts_draw_no_bars(self) -> None:
        out = bar_chart([0.0, 0.5, 1.0], [("gold", "#0a0", [0.0, 0.0])])
        assert "<rect" not in out

    def test_nonfinite_count_is_dropped(self) -> None:
        out = bar_chart([0.0, 0.5, 1.0], [("gold", "#0a0", [float("nan"), 2.0])])
        assert "NaN" not in out and "Infinity" not in out
        assert out.count("<rect") == 1

    def test_series_label_is_escaped_in_legend(self) -> None:
        out = bar_chart([0.0, 1.0], [("g & <n>", "#0a0", [2.0])])
        assert "g &amp; &lt;n&gt;" in out
        assert "<n>" not in out
