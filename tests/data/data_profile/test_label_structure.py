"""Tests for the label-structure profile section.

Behavior + edges: None gold -> absent section, empty / single-record inputs
render without raising, the single-cluster entropy log(0) guard, cluster-size
distribution + histogram (including the overflow-bin path), pair prevalence /
imbalance with an implied-singleton denominator, and the render invariants (no
literal ``NaN``/``Infinity`` in HTML, title escaping).
"""

from __future__ import annotations

import math

import pytest

from langres.data.data_profile import ProfileSection
from langres.data.data_profile.label_structure import (
    LabelStructureSection,
    profile_label_structure,
)


class TestGracefulDegradation:
    def test_none_clusters_returns_none(self) -> None:
        # No gold available -> the section is simply absent from the report.
        assert profile_label_structure(None) is None

    def test_empty_clusters_render_without_raising(self) -> None:
        section = profile_label_structure([])
        assert section is not None
        assert section.n_records == 0
        assert section.n_clusters == 0
        assert section.mean_cluster_size is None
        assert section.prevalence is None
        assert section.imbalance_ratio is None
        assert section.entropy_bits is None
        # Every surface renders on the degenerate input.
        assert section.to_markdown().startswith("## Label structure")
        assert section.rows() == []
        html = "".join(section.panels())
        assert "<svg" in html
        assert "NaN" not in html and "Infinity" not in html

    def test_single_record_renders(self) -> None:
        section = profile_label_structure([{"a"}])
        assert section is not None
        assert section.n_records == 1
        assert section.n_clusters == 1
        assert section.n_singletons == 1
        assert section.positive_pairs == 0
        assert section.total_pairs == 0
        assert section.prevalence is None  # C(1, 2) == 0 -> undefined
        assert section.imbalance_ratio is None
        assert "NaN" not in "".join(section.panels())


class TestClusterMetrics:
    def test_basic_counts_and_pairs(self) -> None:
        section = profile_label_structure([{"a", "b"}, {"c"}, {"d", "e", "f"}])
        assert section is not None
        assert section.n_records == 6
        assert section.n_clusters == 3
        assert section.n_singletons == 1
        assert section.n_multi == 2
        assert section.max_cluster_size == 3
        assert section.mean_cluster_size == 6 / 3
        # positives = C(2,2) + C(1,2) + C(3,2) = 1 + 0 + 3 = 4; total = C(6,2) = 15.
        assert section.positive_pairs == 4
        assert section.total_pairs == 15
        assert math.isclose(section.prevalence, 4 / 15)  # type: ignore[arg-type]
        # negatives = 15 - 4 = 11; imbalance N = 11 / 4.
        assert math.isclose(section.imbalance_ratio, 11 / 4)  # type: ignore[arg-type]

    def test_size_distribution_and_rows(self) -> None:
        section = profile_label_structure([{"a", "b"}, {"c"}, {"d"}, {"e", "f", "g"}])
        assert section is not None
        assert section.size_distribution == [(1, 2), (2, 1), (3, 1)]
        assert section.rows() == [
            {"cluster_size": 1, "n_clusters": 2},
            {"cluster_size": 2, "n_clusters": 1},
            {"cluster_size": 3, "n_clusters": 1},
        ]
        # The non-empty distribution renders as a Markdown table.
        md = section.to_markdown()
        assert "### Cluster-size distribution" in md
        assert "| 3 | 1 |" in md

    def test_n_records_folds_implied_singletons(self) -> None:
        # Gold lists only the matched clusters; the true corpus has 100 records.
        section = profile_label_structure([{"a", "b"}, {"c", "d"}], n_records=100)
        assert section is not None
        # 4 clustered records + 96 implied singletons.
        assert section.n_records == 100
        assert section.n_singletons == 96
        assert section.n_clusters == 2 + 96
        # positives = 2 * C(2,2) = 2; total = C(100, 2) = 4950.
        assert section.positive_pairs == 2
        assert section.total_pairs == math.comb(100, 2)
        assert section.prevalence == 2 / math.comb(100, 2)

    def test_n_records_smaller_than_clustered_is_ignored(self) -> None:
        # A too-small n_records never shrinks the corpus below what the clusters
        # already contain (the clustering always wins).
        section = profile_label_structure([{"a", "b", "c"}], n_records=1)
        assert section is not None
        assert section.n_records == 3
        assert section.n_singletons == 0

    def test_large_imbalance_ratio(self) -> None:
        # One match-pair in a 1000-record corpus: extreme ER class imbalance.
        section = profile_label_structure([{"a", "b"}], n_records=1000)
        assert section is not None
        # positives = 1; negatives = C(1000,2) - 1; imbalance ~ 499499.
        assert section.positive_pairs == 1
        assert section.imbalance_ratio == math.comb(1000, 2) - 1
        # KV render shows a grouped 1:N ratio, not a raw float / NaN.
        html = "".join(section.panels())
        assert "1:499,499" in html
        assert "NaN" not in html and "Infinity" not in html


class TestEntropy:
    def test_single_cluster_entropy_is_zero(self) -> None:
        # The log(0) guard: a single cluster has p == 1, log2(1) == 0 -> entropy
        # exactly 0.0 (a fully predictable partition), never a crash.
        section = profile_label_structure([{"a", "b", "c"}])
        assert section is not None
        assert section.entropy_bits == 0.0
        assert section.prevalence == 1.0

    def test_uniform_clusters_entropy_is_log2_k(self) -> None:
        # k equal-size clusters -> maximal entropy log2(k).
        section = profile_label_structure([{"a", "b"}, {"c", "d"}, {"e", "f"}, {"g", "h"}])
        assert section is not None
        assert math.isclose(section.entropy_bits, math.log2(4))  # type: ignore[arg-type]

    def test_empty_cluster_is_skipped_by_log_guard(self) -> None:
        # A degenerate size-0 cluster must not crash the entropy sum (size <= 0
        # skip branch) and leaves a finite, defined entropy.
        section = profile_label_structure([set(), {"a", "b"}])
        assert section is not None
        assert section.entropy_bits is not None
        assert math.isfinite(section.entropy_bits)


class TestHistogramRendering:
    def test_overflow_bin_folds_large_sizes(self) -> None:
        # A cluster larger than the bar cap must not emit one bar per member; it
        # folds into a single overflow bar (exercised via the panel not raising).
        big = set(range(30))
        section = profile_label_structure([big, {100, 101}, {102}])
        assert section is not None
        assert section.max_cluster_size == 30
        html = "".join(section.panels())
        assert "<svg" in html
        assert "NaN" not in html and "Infinity" not in html

    def test_panels_contain_kv_table_and_chart(self) -> None:
        section = profile_label_structure([{"a", "b"}, {"c"}])
        assert section is not None
        panels = section.panels()
        assert len(panels) == 1
        html = panels[0]
        assert html.startswith("<section><h2>Label structure</h2>")
        assert '<table class="kv">' in html
        assert "<svg" in html

    def test_title_is_escaped_in_html(self) -> None:
        section = profile_label_structure([{"a", "b"}], title="Gold & <labels>")
        assert section is not None
        html = "".join(section.panels())
        assert "Gold &amp; &lt;labels&gt;" in html
        assert "<labels>" not in html


class TestContract:
    def test_kind_and_type(self) -> None:
        section = profile_label_structure([{"a", "b"}])
        assert isinstance(section, LabelStructureSection)
        assert isinstance(section, ProfileSection)
        assert section.kind == "label_structure"

    def test_summary_is_title_namespaced(self) -> None:
        section = profile_label_structure([{"a", "b"}, {"c"}], title="Labels")
        assert section is not None
        summary = section.summary
        assert summary["Labels.n_records"] == 3
        assert summary["Labels.n_clusters"] == 2
        assert "Labels.prevalence" in summary

    def test_is_frozen(self) -> None:
        section = profile_label_structure([{"a", "b"}])
        assert section is not None
        with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError on frozen set
            section.n_records = 99  # type: ignore[misc]

    def test_markdown_has_no_nan(self) -> None:
        section = profile_label_structure([])
        assert section is not None
        assert "NaN" not in section.to_markdown()
