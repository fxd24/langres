"""Tests for the separability profile section + the string signal.

Behavior + edges: the ``SimilaritySignal`` bridge, ``string_signal`` (absent id
and no-shared-feature -> ``None``, real similarity otherwise), the density-
normalized overlaid histogram staying legible under 1:1000 class imbalance, the
capped-sample truncation (logged), ``None``/non-finite score dropping, the
no-usable-scores -> absent section path, the one-sided keep-with-hint path
(never a NaN panel), and the render invariants (no ``NaN``/``Infinity``, name
escaping).
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Hashable

import pytest
from pydantic import BaseModel

from langres.core.data_profile import ProfileSection
from langres.core.data_profile.separability import (
    SeparabilitySection,
    SimilaritySignal,
    profile_separability,
    string_signal,
)


class _Schema(BaseModel):
    id: str
    name: str | None = None
    city: str | None = None


_RECORDS = {
    "1": {"name": "Acme Corporation", "city": "New York"},
    "2": {"name": "Acme Corp", "city": "New York City"},
    "3": {"name": "Zeta Foods Ltd", "city": "Los Angeles"},
    "4": {"name": "Omega Holdings", "city": "San Francisco"},
}


def _rect_heights_by_fill(svg: str, fill: str) -> list[float]:
    """Extract every ``<rect>`` height for a given fill color from an SVG string."""
    heights: list[float] = []
    for rect in re.findall(r"<rect [^>]*/>", svg):
        if f'fill="{fill}"' in rect:
            match = re.search(r'height="([0-9.]+)"', rect)
            if match:
                heights.append(float(match.group(1)))
    return heights


class TestStringSignal:
    def test_returns_similarity_for_present_ids(self) -> None:
        signal = string_signal(_RECORDS, _Schema)
        score = signal("1", "2")
        assert score is not None
        assert 0.0 <= score <= 1.0
        # Near-duplicate scores higher than an unrelated pair.
        assert signal("1", "2") > signal("1", "3")  # type: ignore[operator]

    def test_absent_id_returns_none(self) -> None:
        signal = string_signal(_RECORDS, _Schema)
        assert signal("1", "missing") is None
        assert signal("missing", "1") is None

    def test_no_shared_feature_returns_none(self) -> None:
        # Two records with no populated comparable field -> unscorable (not 0.0).
        records: dict[Hashable, dict[str, object]] = {"a": {}, "b": {}}
        signal = string_signal(records, _Schema)
        assert signal("a", "b") is None

    def test_matches_similaritysignal_shape(self) -> None:
        signal: SimilaritySignal = string_signal(_RECORDS, _Schema)
        assert callable(signal)


class TestProfileSeparability:
    def test_both_classes_scored_yield_auc(self) -> None:
        signal = string_signal(_RECORDS, _Schema)
        section = profile_separability(
            [("1", "2")], [("1", "3"), ("1", "4"), ("3", "4")], signal, name="string"
        )
        assert section is not None
        assert section.n_positive == 1
        assert section.n_negative == 3
        assert section.auc == 1.0  # the match pair outscores every non-match
        assert section.note == ""

    def test_no_usable_scores_returns_none(self) -> None:
        # An all-None signal (every pair unscorable) -> the section is absent.
        def none_signal(a: Hashable, b: Hashable) -> float | None:
            return None

        assert (
            profile_separability([("1", "2")], [("3", "4")], none_signal, name="string") is None
        )

    def test_one_sided_negative_empty_keeps_with_hint(self) -> None:
        signal = string_signal(_RECORDS, _Schema)
        section = profile_separability([("1", "2")], [], signal, name="string")
        assert section is not None
        assert section.n_positive == 1
        assert section.n_negative == 0
        assert section.auc is None  # single-class -> undefined, normalized to None
        assert "negative" in section.note.lower()
        # The hint surfaces in Markdown too, and the AUC reads "n/a" (never NaN).
        md = section.to_markdown()
        assert "negative" in md.lower()
        assert "n/a" in md
        # Never a NaN panel.
        html = "".join(section.panels())
        assert "NaN" not in html and "Infinity" not in html

    def test_one_sided_positive_empty_keeps_with_hint(self) -> None:
        signal = string_signal(_RECORDS, _Schema)
        section = profile_separability([], [("3", "4")], signal, name="string")
        assert section is not None
        assert section.auc is None
        assert "positive" in section.note.lower()

    def test_none_and_nonfinite_scores_dropped(self) -> None:
        # A signal emitting None / inf / nan -> those pairs are dropped as unscorable.
        scores = {("1", "2"): 0.9, ("1", "3"): None, ("1", "4"): float("inf")}

        def flaky(a: Hashable, b: Hashable) -> float | None:
            return scores.get((a, b), 0.1)  # type: ignore[return-value]

        section = profile_separability(
            [("1", "2"), ("1", "3")], [("1", "4"), ("3", "4")], flaky, name="flaky"
        )
        assert section is not None
        assert section.n_positive == 1  # ("1","3") dropped (None)
        assert section.n_negative == 1  # ("1","4") dropped (inf)

    def test_degenerate_equal_scores_do_not_crash(self) -> None:
        # All scores identical -> the edge range is widened so bars still render.
        def constant(a: Hashable, b: Hashable) -> float | None:
            return 0.5

        section = profile_separability(
            [("1", "2")], [("3", "4")], constant, name="const", n_bins=5
        )
        assert section is not None
        assert section.hist_edges[0] < section.hist_edges[-1]
        html = "".join(section.panels())
        assert "NaN" not in html and "Infinity" not in html


class TestCapping:
    def test_cap_truncates_and_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        signal = string_signal(_RECORDS, _Schema)
        negatives = [("1", "3"), ("1", "4"), ("3", "4"), ("2", "3"), ("2", "4")]
        with caplog.at_level(logging.WARNING):
            section = profile_separability([("1", "2")], negatives, signal, name="string", cap=2)
        assert section is not None
        assert section.n_negative == 2  # capped
        assert any("sampled" in r.message.lower() for r in caplog.records)

    def test_no_cap_scores_everything(self) -> None:
        signal = string_signal(_RECORDS, _Schema)
        negatives = [("1", "3"), ("1", "4"), ("3", "4")]
        section = profile_separability([("1", "2")], negatives, signal, name="string", cap=None)
        assert section is not None
        assert section.n_negative == 3


class TestDensityLegibility:
    def test_tiny_positive_class_stays_visible_under_imbalance(self) -> None:
        # 3 positives vs 3000 negatives: without density normalization the
        # positive bars would be < 1px; with it, both classes are comparable.
        pos_pairs = [(f"p{i}a", f"p{i}b") for i in range(3)]
        neg_pairs = [(f"n{i}a", f"n{i}b") for i in range(3000)]
        pos_set = set(pos_pairs)

        def signal(a: Hashable, b: Hashable) -> float | None:
            return 0.95 if (a, b) in pos_set else 0.05

        section = profile_separability(pos_pairs, neg_pairs, signal, name="string", n_bins=10)
        assert section is not None
        assert section.n_positive == 3
        assert section.n_negative == 3000
        assert section.auc == 1.0

        svg = "".join(section.panels())
        pos_heights = _rect_heights_by_fill(svg, "#2a9d8f")
        neg_heights = _rect_heights_by_fill(svg, "#e76f51")
        assert pos_heights and neg_heights
        # Density-normalized: the tiny positive class reaches a comparable height,
        # not a sub-pixel sliver dwarfed by the 3000 negatives.
        assert max(pos_heights) > 0.1 * max(neg_heights)


class TestRenderSurfaces:
    def _section(self) -> SeparabilitySection:
        signal = string_signal(_RECORDS, _Schema)
        section = profile_separability(
            [("1", "2")], [("1", "3"), ("3", "4")], signal, name="string"
        )
        assert section is not None
        return section

    def test_rows_are_per_bin_distributions(self) -> None:
        section = self._section()
        rows = section.rows()
        assert len(rows) == len(section.pos_counts)
        assert set(rows[0]) == {"bin_lo", "bin_hi", "positives", "negatives"}

    def test_summary_is_title_namespaced(self) -> None:
        section = self._section()
        summary = section.summary
        assert "Separability (string).auc" in summary
        assert summary["Separability (string).n_positive"] == 1

    def test_markdown_and_panels_render(self) -> None:
        section = self._section()
        md = section.to_markdown()
        assert md.startswith("## Separability (string)")
        assert "AUC" in md
        panels = section.panels()
        assert len(panels) == 1
        assert "<svg" in panels[0]

    def test_name_escaped_in_html(self) -> None:
        signal = string_signal(_RECORDS, _Schema)
        section = profile_separability([("1", "2")], [("3", "4")], signal, name="a & <b>")
        assert section is not None
        html = "".join(section.panels())
        assert "a &amp; &lt;b&gt;" in html
        # The raw, unescaped name must not leak into the markup.
        assert "a & <b>" not in html


class TestContract:
    def test_kind_and_type(self) -> None:
        section = profile_separability(
            [("1", "2")], [("3", "4")], string_signal(_RECORDS, _Schema), name="string"
        )
        assert isinstance(section, SeparabilitySection)
        assert isinstance(section, ProfileSection)
        assert section.kind == "separability"

    def test_is_frozen(self) -> None:
        section = profile_separability(
            [("1", "2")], [("3", "4")], string_signal(_RECORDS, _Schema), name="string"
        )
        assert section is not None
        with pytest.raises(Exception):  # noqa: B017 - frozen model
            section.auc = 0.0  # type: ignore[misc]
