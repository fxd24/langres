"""Tests for the KPI ``HeroSection`` + ``build_hero``.

Covers the render ladder (markdown / summary / rows / panels), the degenerate
all-``None`` case (``"n/a"`` cards, never a raise, no literal NaN/Infinity), and
``build_hero``'s extraction from the label-structure + separability sections
(including the graceful ``None`` when neither is present).
"""

from __future__ import annotations

from collections.abc import Hashable

from langres.core.data_profile.hero import HeroSection, build_hero
from langres.core.data_profile.label_structure import profile_label_structure
from langres.core.data_profile.separability import profile_separability


def _perfect_signal(left: Hashable, right: Hashable) -> float | None:
    """Score the one positive pair high, everything else low (AUC == 1.0)."""
    return 1.0 if frozenset((left, right)) == frozenset(("1", "2")) else 0.0


def _label_and_sep() -> tuple[object, object]:
    label = profile_label_structure([{"1", "2"}, {"3"}], n_records=3)
    sep = profile_separability(
        [("1", "2")], [("1", "3")], _perfect_signal, name="string"
    )
    return label, sep


class TestHeroSectionRender:
    def test_full_cards(self) -> None:
        hero = HeroSection(
            title="Overview",
            n_records=1000,
            n_clusters=400,
            prevalence=0.0003,
            imbalance_ratio=2940.0,
            separability_auc=0.812,
        )
        labels = dict(hero._cards())
        assert labels["records"] == "1,000"
        assert labels["clusters"] == "400"
        assert labels["positive-pair prevalence"] == "0.0003"
        assert labels["class imbalance (pos:neg)"] == "1:2,940"
        assert labels["separability AUC"] == "0.812"

    def test_markdown_has_all_kpis(self) -> None:
        hero = HeroSection(
            title="Overview",
            n_records=10,
            n_clusters=4,
            prevalence=0.1,
            imbalance_ratio=9.0,
            separability_auc=0.75,
        )
        md = hero.to_markdown()
        assert md.startswith("## Overview")
        assert "separability AUC" in md
        assert "1:9" in md
        assert "NaN" not in md

    def test_summary_is_title_namespaced(self) -> None:
        hero = HeroSection(
            title="Overview",
            n_records=10,
            n_clusters=4,
            prevalence=0.1,
            imbalance_ratio=9.0,
            separability_auc=0.75,
        )
        assert hero.summary == {
            "Overview.n_records": 10,
            "Overview.n_clusters": 4,
            "Overview.prevalence": 0.1,
            "Overview.imbalance_ratio": 9.0,
            "Overview.separability_auc": 0.75,
        }

    def test_rows_single_raw_row(self) -> None:
        hero = HeroSection(
            title="Overview",
            n_records=10,
            n_clusters=4,
            prevalence=0.1,
            imbalance_ratio=9.0,
            separability_auc=0.75,
        )
        rows = hero.rows()
        assert rows == [
            {
                "n_records": 10,
                "n_clusters": 4,
                "prevalence": 0.1,
                "imbalance_ratio": 9.0,
                "separability_auc": 0.75,
            }
        ]

    def test_panels_render_card_grid(self) -> None:
        hero = HeroSection(
            title="Overview",
            n_records=10,
            n_clusters=4,
            prevalence=0.1,
            imbalance_ratio=9.0,
            separability_auc=0.75,
        )
        panels = hero.panels()
        assert len(panels) == 1
        panel = panels[0]
        assert panel.startswith("<section><h2>Overview</h2>")
        # Rides on the shared CSS vocabulary (no new external stylesheet).
        assert "var(--line)" in panel
        assert "var(--muted)" in panel
        assert "<svg" not in panel  # KPI cards, not a chart

    def test_degenerate_all_none_renders_na_never_raises(self) -> None:
        hero = HeroSection(
            title="Overview",
            n_records=None,
            n_clusters=None,
            prevalence=None,
            imbalance_ratio=None,
            separability_auc=None,
        )
        assert [value for _, value in hero._cards()] == ["n/a"] * 5
        panel = hero.panels()[0]
        assert "n/a" in panel
        assert "NaN" not in panel and "Infinity" not in panel
        md = hero.to_markdown()
        assert "n/a" in md and "NaN" not in md


class TestBuildHero:
    def test_extracts_from_label_and_separability(self) -> None:
        label, sep = _label_and_sep()
        hero = build_hero([label, sep])  # type: ignore[list-item]
        assert hero is not None
        assert hero.n_records == 3
        assert hero.n_clusters == 2
        assert hero.separability_auc == 1.0
        # prevalence / imbalance flow from the label section unchanged.
        assert hero.prevalence == label.prevalence  # type: ignore[attr-defined]
        assert hero.imbalance_ratio == label.imbalance_ratio  # type: ignore[attr-defined]

    def test_label_only_leaves_auc_none(self) -> None:
        label, _ = _label_and_sep()
        hero = build_hero([label])  # type: ignore[list-item]
        assert hero is not None
        assert hero.n_records == 3
        assert hero.separability_auc is None

    def test_separability_only_leaves_counts_none(self) -> None:
        _, sep = _label_and_sep()
        hero = build_hero([sep])  # type: ignore[list-item]
        assert hero is not None
        assert hero.separability_auc == 1.0
        assert hero.n_records is None
        assert hero.prevalence is None

    def test_none_when_no_source_sections(self) -> None:
        assert build_hero([]) is None

    def test_custom_title_namespaces_summary(self) -> None:
        label, sep = _label_and_sep()
        hero = build_hero([label, sep], title="KPIs")  # type: ignore[list-item]
        assert hero is not None
        assert hero.title == "KPIs"
        assert "KPIs.n_records" in hero.summary
