"""Tests for the shared render scaffold ``langres.core._report_html``.

Structure-based asserts (doctype prefix, escaping, ``"n/a"`` sentinel, absence
of literal ``NaN``/``Infinity``), never byte-for-byte equality, so cosmetic
tweaks don't churn the suite. The AUC/AP guards are checked against the
underlying ``core.metrics`` primitives on the finite-scored subset.
"""

from __future__ import annotations

import math

from langres.core import _report_html
from langres.core.metrics import average_precision_score, roc_auc_score


class TestDocument:
    def test_is_a_self_contained_html_document(self) -> None:
        out = _report_html.document("My title", "<section>body</section>")
        assert out.startswith("<!doctype html>")
        assert out.rstrip().endswith("</html>")
        # Inline style, no external asset.
        assert "<style>" in out
        assert "http://" not in out.replace('xmlns="http://www.w3.org/2000/svg"', "")
        # Title lands in both <title> and <h1>.
        assert "<title>My title</title>" in out
        assert "<h1>My title</h1>" in out
        # Body is emitted verbatim.
        assert "<section>body</section>" in out

    def test_title_is_html_escaped_in_head_and_heading(self) -> None:
        out = _report_html.document("a & b <x>", "")
        assert "a &amp; b &lt;x&gt;" in out
        assert "<x>" not in out

    def test_summary_html_absent_by_default(self) -> None:
        out = _report_html.document("t", "<p>b</p>")
        assert 'class="summary"' not in out

    def test_summary_html_is_emitted_verbatim_when_given(self) -> None:
        out = _report_html.document("t", "<p>b</p>", summary_html="P=<b>0.9</b>")
        assert '<p class="summary">P=<b>0.9</b></p>' in out

    def test_empty_body_still_valid_document(self) -> None:
        out = _report_html.document("t", "")
        assert out.startswith("<!doctype html>")
        assert out.rstrip().endswith("</html>")


class TestSection:
    def test_wraps_body_in_a_section_with_escaped_heading(self) -> None:
        out = _report_html.section("Fields & <stuff>", "<table></table>")
        assert out.startswith("<section><h2>")
        assert out.endswith("</section>")
        assert "Fields &amp; &lt;stuff&gt;" in out
        assert "<stuff>" not in out
        # Body HTML passes through unescaped.
        assert "<table></table>" in out


class TestNum:
    def test_formats_to_requested_digits(self) -> None:
        assert _report_html._num(0.12345) == "0.123"
        assert _report_html._num(0.12345, digits=1) == "0.1"

    def test_none_and_nonfinite_map_to_na(self) -> None:
        assert _report_html._num(None) == "n/a"
        assert _report_html._num(float("nan")) == "n/a"
        assert _report_html._num(float("inf")) == "n/a"
        assert _report_html._num(float("-inf")) == "n/a"


class TestMdCell:
    def test_escapes_pipe_and_flattens_newlines(self) -> None:
        assert _report_html._md_cell("a|b") == r"a\|b"
        assert _report_html._md_cell("a\nb\rc") == "a b c"

    def test_coerces_non_str(self) -> None:
        assert _report_html._md_cell(42) == "42"  # type: ignore[arg-type]


class TestHistogram:
    def test_counts_into_half_open_bins_last_bin_closed(self) -> None:
        edges = [0.0, 0.5, 1.0]
        # 0.5 -> second bin; 1.0 (== final edge) counted in the last (closed) bin.
        assert _report_html._histogram([0.1, 0.5, 1.0], edges) == [1.0, 2.0]

    def test_non_finite_values_ignored(self) -> None:
        edges = [0.0, 1.0]
        assert _report_html._histogram([0.5, float("nan"), float("inf")], edges) == [1.0]

    def test_no_bins_when_single_edge(self) -> None:
        assert _report_html._histogram([0.5], [0.0]) == []


class TestSafeAuc:
    def test_none_on_empty(self) -> None:
        assert _report_html.safe_auc([], []) is None

    def test_none_when_all_scores_non_finite(self) -> None:
        assert _report_html.safe_auc([True, False], [float("nan"), float("inf")]) is None

    def test_matches_metric_on_finite_subset(self) -> None:
        # One non-finite pair is dropped; the guard must equal the metric on the rest.
        labels = [False, True, False, True]
        scores = [1.0, 2.0, float("nan"), 4.0]
        expected = roc_auc_score([False, True, True], [1.0, 2.0, 4.0])
        got = _report_html.safe_auc(labels, scores)
        assert got is not None
        assert got == expected

    def test_perfect_separation(self) -> None:
        assert _report_html.safe_auc([False, False, True, True], [1.0, 2.0, 3.0, 4.0]) == 1.0

    def test_single_class_returns_underlying_nan_never_raises(self) -> None:
        got = _report_html.safe_auc([True, True], [0.2, 0.8])
        assert got is not None and math.isnan(got)


class TestSafeAp:
    def test_none_on_empty(self) -> None:
        assert _report_html.safe_ap([], []) is None

    def test_none_when_all_scores_non_finite(self) -> None:
        assert _report_html.safe_ap([True, False], [float("inf"), float("nan")]) is None

    def test_matches_metric_on_finite_subset(self) -> None:
        labels = [False, True, False, True]
        scores = [1.0, 2.0, float("inf"), 4.0]
        expected = average_precision_score([False, True, True], [1.0, 2.0, 4.0])
        got = _report_html.safe_ap(labels, scores)
        assert got is not None
        assert got == expected

    def test_no_positives_returns_underlying_nan_never_raises(self) -> None:
        got = _report_html.safe_ap([False, False], [0.2, 0.8])
        assert got is not None and math.isnan(got)
