"""Tests for the ``MiningReadinessSection`` + ``profile_mining_readiness``.

Covers the render ladder (markdown / summary / rows / panels), the derived
class-balance stats, the graceful-degrade path (unmeasured counts -> ``"n/a"``,
absent margins -> no chart, empty set -> honest note, never a raise or literal
NaN), the margin histogram, and the section's wiring into ``DataProfileReport``
(``from_records(mining_readiness=...)`` + the ``include=`` kind filter).
"""

from __future__ import annotations

from langres.data.data_profile import (
    DataProfileReport,
    MiningReadinessSection,
    from_records,
    profile_mining_readiness,
)


def _full_section() -> MiningReadinessSection:
    return profile_mining_readiness(
        n_positive=100,
        n_negative=400,
        n_hard_positive=25,
        n_flagged_noise=10,
        margins=[0.05, 0.1, 0.1, 0.9, 0.95, 0.5],
    )


class TestDerivedStats:
    def test_class_balance_and_imbalance(self) -> None:
        section = profile_mining_readiness(n_positive=100, n_negative=400)
        assert section.total == 500
        assert section.positive_share == 100 / 500
        assert section.imbalance_ratio == 4.0

    def test_hard_positive_and_noise_shares(self) -> None:
        section = _full_section()
        assert section.hard_positive_share == 25 / 100
        assert section.flagged_noise_share == 10 / 500

    def test_shares_are_none_when_unmeasured(self) -> None:
        section = profile_mining_readiness(n_positive=10, n_negative=10)
        assert section.hard_positive_share is None
        assert section.flagged_noise_share is None

    def test_empty_set_stats_are_none(self) -> None:
        section = profile_mining_readiness(n_positive=0, n_negative=0)
        assert section.total == 0
        assert section.positive_share is None
        assert section.imbalance_ratio is None


class TestRenderLadder:
    def test_markdown_has_all_metrics(self) -> None:
        md = _full_section().to_markdown()
        assert md.startswith("## Mining readiness")
        assert "class imbalance (pos:neg)" in md
        assert "1:4" in md
        assert "hard positives" in md
        assert "NaN" not in md

    def test_summary_is_title_namespaced(self) -> None:
        section = _full_section()
        assert section.summary == {
            "Mining readiness.n_positive": 100,
            "Mining readiness.n_negative": 400,
            "Mining readiness.imbalance_ratio": 4.0,
            "Mining readiness.n_hard_positive": 25,
            "Mining readiness.n_flagged_noise": 10,
        }

    def test_rows_single_raw_row(self) -> None:
        rows = _full_section().rows()
        assert rows == [
            {
                "n_positive": 100,
                "n_negative": 400,
                "imbalance_ratio": 4.0,
                "n_hard_positive": 25,
                "n_flagged_noise": 10,
            }
        ]

    def test_panels_render_kv_table_and_margin_chart(self) -> None:
        panels = _full_section().panels()
        assert len(panels) == 1
        panel = panels[0]
        assert panel.startswith("<section><h2>Mining readiness</h2>")
        assert 'table class="kv"' in panel
        assert "<svg" in panel  # margins supplied -> histogram
        assert "NaN" not in panel and "Infinity" not in panel

    def test_panels_without_margins_have_no_chart(self) -> None:
        section = profile_mining_readiness(n_positive=10, n_negative=10)
        panel = section.panels()[0]
        assert 'table class="kv"' in panel
        assert "<svg" not in panel


class TestGracefulDegradation:
    def test_unmeasured_counts_render_na(self) -> None:
        section = profile_mining_readiness(n_positive=10, n_negative=10)
        kv = dict(section._metrics_kv())
        assert kv["hard positives"] == "n/a"
        assert kv["flagged label noise"] == "n/a"

    def test_measured_counts_render_count_and_share(self) -> None:
        kv = dict(_full_section()._metrics_kv())
        assert kv["hard positives"] == "25 (25%)"
        assert kv["flagged label noise"] == "10 (2%)"

    def test_count_without_share_renders_bare_count(self) -> None:
        """A measured count with an undefined share (no positives) renders just the count."""
        section = profile_mining_readiness(n_positive=0, n_negative=0, n_hard_positive=3)
        assert section.hard_positive_share is None
        assert dict(section._metrics_kv())["hard positives"] == "3"

    def test_empty_set_renders_note_never_raises(self) -> None:
        section = profile_mining_readiness(n_positive=0, n_negative=0)
        md = section.to_markdown()
        assert "nothing to train on" in md
        assert "NaN" not in md
        kv = dict(section._metrics_kv())
        assert kv["positive share"] == "n/a"
        assert kv["class imbalance (pos:neg)"] == "n/a"


class TestMarginHistogram:
    def test_margins_are_binned(self) -> None:
        section = profile_mining_readiness(
            n_positive=3, n_negative=3, margins=[0.0, 0.5, 1.0], n_bins=4
        )
        assert len(section.margin_edges) == 5  # n_bins + 1
        assert len(section.margin_counts) == 4
        assert sum(section.margin_counts) == 3.0

    def test_no_margins_leave_empty_histogram(self) -> None:
        section = profile_mining_readiness(n_positive=3, n_negative=3)
        assert section.margin_edges == []
        assert section.margin_counts == []

    def test_non_finite_margins_dropped(self) -> None:
        section = profile_mining_readiness(
            n_positive=1, n_negative=1, margins=[float("nan"), float("inf"), 0.5]
        )
        assert sum(section.margin_counts) == 1.0

    def test_all_non_finite_margins_leave_empty_histogram(self) -> None:
        section = profile_mining_readiness(
            n_positive=1, n_negative=1, margins=[float("nan"), float("inf")]
        )
        assert section.margin_edges == []
        assert section.margin_counts == []


class TestReportWiring:
    def _records(self) -> tuple[list[dict[str, str]], list[set[str]]]:
        records = [
            {"id": "1", "name": "Acme"},
            {"id": "2", "name": "Acme Inc"},
            {"id": "3", "name": "Zephyr"},
        ]
        gold = [{"1", "2"}, {"3"}]
        return records, gold

    def test_section_included_when_supplied(self) -> None:
        records, gold = self._records()
        section = profile_mining_readiness(n_positive=1, n_negative=2, n_hard_positive=1)
        report = from_records(records, gold=gold, mining_readiness=section)
        assert isinstance(report["Mining readiness"], MiningReadinessSection)

    def test_section_absent_when_not_supplied(self) -> None:
        records, gold = self._records()
        report = from_records(records, gold=gold)
        kinds = {section.kind for section in report.sections}
        assert "mining_readiness" not in kinds

    def test_include_selects_the_kind(self) -> None:
        records, gold = self._records()
        section = profile_mining_readiness(n_positive=1, n_negative=2)
        report = from_records(
            records, gold=gold, mining_readiness=section, include=["mining_readiness"]
        )
        kinds = {section.kind for section in report.sections}
        assert kinds == {"mining_readiness"}

    def test_report_renders_with_section(self) -> None:
        records, gold = self._records()
        section = profile_mining_readiness(n_positive=1, n_negative=2, n_flagged_noise=0)
        report = from_records(records, gold=gold, mining_readiness=section)
        assert "Mining readiness" in report.to_markdown()
        assert "<section><h2>Mining readiness</h2>" in report.to_html()

    def test_report_is_a_data_profile_report(self) -> None:
        records, gold = self._records()
        section = profile_mining_readiness(n_positive=1, n_negative=2)
        report = from_records(records, gold=gold, mining_readiness=section)
        assert isinstance(report, DataProfileReport)
