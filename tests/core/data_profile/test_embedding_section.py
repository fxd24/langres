"""Tests for the embedding profile sections (norm distribution + comparison).

Vectors are synthesised in-memory (no ML stack); sections are exercised through
their full render ladder (markdown / summary / rows / panels), the degenerate
keep-with-hint paths (empty, pre-normalized, constant norm), the drop counting
(zero-norm / non-finite), the ``cap`` truncation, and the comparison's
small-multiples / <2-source placeholder / dim-mismatch caveat. Every render is
asserted to contain no literal ``NaN`` / ``Infinity``.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from langres.core.data_profile.base import ProfileSection
from langres.core.data_profile.embedding_section import (
    EmbeddingComparisonSection,
    EmbeddingSection,
    _good_norms,
    _median_from_hist,
    profile_embedding,
    profile_embedding_comparison,
)
from langres.core.data_profile.embedding_source import ArraySource


def _spread_source(name: str = "m", n: int = 200, dim: int = 8) -> ArraySource:
    """A source whose row norms vary over a real range (norms grow with the row)."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((n, dim)).astype(np.float32)
    scales = np.linspace(0.5, 5.0, n, dtype=np.float32).reshape(-1, 1)
    unit = base / np.linalg.norm(base, axis=1, keepdims=True)
    return ArraySource(name, [f"r{i}" for i in range(n)], unit * scales)


def _ids(n: int) -> list[str]:
    return [f"r{i}" for i in range(n)]


# ------------------------------------------------------------------ single section
class TestProfileEmbedding:
    def test_returns_embedding_section(self) -> None:
        section = profile_embedding(_spread_source(), _ids(200))
        assert isinstance(section, EmbeddingSection)
        assert isinstance(section, ProfileSection)
        assert section.kind == "embedding"

    def test_basic_stats_and_histogram(self) -> None:
        section = profile_embedding(_spread_source(n=200, dim=8), _ids(200))
        assert section.dim == 8
        assert section.n_vectors == 200
        assert section.n_dropped == 0
        assert not section.degenerate
        assert len(section.hist_counts) == len(section.hist_edges) - 1
        assert sum(section.hist_counts) == pytest.approx(200)
        # Mean norm sits inside the [0.5, 5.0] scale band.
        assert 0.5 < section.mean_norm < 5.0
        assert section.median_norm is not None
        assert section.min_norm is not None and section.max_norm is not None

    def test_median_matches_numpy_reference_roughly(self) -> None:
        src = _spread_source(n=400, dim=8)
        section = profile_embedding(src, _ids(400))
        true_norms = np.linalg.norm(src.vectors_for(_ids(400)).astype(np.float64), axis=1)
        assert section.median_norm == pytest.approx(float(np.median(true_norms)), abs=0.15)

    def test_batching_matches_single_batch(self) -> None:
        src = _spread_source(n=200, dim=8)
        big = profile_embedding(src, _ids(200), batch=10_000)
        small = profile_embedding(src, _ids(200), batch=7)
        assert big.n_vectors == small.n_vectors
        assert big.mean_norm == pytest.approx(small.mean_norm)
        assert big.hist_counts == small.hist_counts

    def test_zero_norm_and_non_finite_rows_dropped_and_counted(self) -> None:
        matrix = np.array(
            [[3.0, 4.0], [0.0, 0.0], [np.nan, 1.0], [1.0, 0.0], [np.inf, 0.0]],
            dtype=np.float32,
        )
        src = ArraySource("m", _ids(5), matrix)
        section = profile_embedding(src, _ids(5))
        assert section.n_vectors == 2  # rows 0 and 3
        assert section.n_dropped == 3  # zero, nan, inf

    def test_cap_truncates_and_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        src = _spread_source(n=200, dim=8)
        with caplog.at_level(logging.INFO):
            section = profile_embedding(src, _ids(200), cap=50)
        assert section.n_vectors == 50
        assert section.n_truncated == 150
        assert "cap=50" in caplog.text

    def test_empty_corpus_is_degenerate(self) -> None:
        src = ArraySource("m", [], np.zeros((0, 4), dtype=np.float32))
        section = profile_embedding(src, [])
        assert section.degenerate
        assert section.n_vectors == 0
        assert section.median_norm is None
        assert section.hist_counts == []

    def test_pre_normalized_is_degenerate_with_caveat(self) -> None:
        rng = np.random.default_rng(1)
        base = rng.standard_normal((50, 16)).astype(np.float32)
        unit = base / np.linalg.norm(base, axis=1, keepdims=True)
        src = ArraySource("cos", _ids(50), unit, pre_normalized=True, metric="cosine")
        section = profile_embedding(src, _ids(50))
        assert section.degenerate  # all norms ~1.0
        assert section.pre_normalized
        assert "pre-normalized" in section.to_markdown()

    def test_constant_norm_is_degenerate(self) -> None:
        matrix = np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]], dtype=np.float32)
        src = ArraySource("m", _ids(3), matrix)  # all norms == 2.0
        section = profile_embedding(src, _ids(3))
        assert section.degenerate
        assert not section.pre_normalized
        assert "constant norm" in section.to_markdown()

    def test_custom_title(self) -> None:
        section = profile_embedding(_spread_source(), _ids(200), title="My embeddings")
        assert section.title == "My embeddings"

    def test_rejects_bare_ndarray(self) -> None:
        with pytest.raises(TypeError, match="bare numpy ndarray"):
            profile_embedding(np.zeros((3, 4)), _ids(3))  # type: ignore[arg-type]


class TestEmbeddingSectionRender:
    def test_markdown_has_no_nan_literal(self) -> None:
        # An empty section has nan mean/std -> must render as n/a, never "NaN".
        src = ArraySource("m", [], np.zeros((0, 4), dtype=np.float32))
        md = profile_embedding(src, []).to_markdown()
        assert "NaN" not in md and "Infinity" not in md
        assert "n/a" in md

    def test_panels_render_section_html(self) -> None:
        section = profile_embedding(_spread_source(), _ids(200))
        panels = section.panels()
        assert len(panels) == 2
        assert all(panel.startswith("<section>") for panel in panels)
        assert "<svg" in panels[0]  # histogram
        assert 'class="kv"' in panels[1]

    def test_degenerate_panel_is_caveat_not_chart(self) -> None:
        src = ArraySource("m", [], np.zeros((0, 4), dtype=np.float32))
        panels = profile_embedding(src, []).panels()
        assert "<svg" not in panels[0]  # no broken chart
        assert 'class="empty"' in panels[0]

    def test_summary_keyed_by_model(self) -> None:
        section = profile_embedding(_spread_source("minilm"), _ids(200))
        summary = section.summary
        assert summary["embedding.minilm.n_vectors"] == 200
        assert "embedding.minilm.mean_norm" in summary

    def test_rows_single_row(self) -> None:
        section = profile_embedding(_spread_source("m"), _ids(200))
        rows = section.rows()
        assert len(rows) == 1
        assert rows[0]["model"] == "m"
        assert rows[0]["n_vectors"] == 200

    def test_html_render_via_container_has_no_nan(self) -> None:
        from langres.core.data_profile.base import DataProfileReport

        section = profile_embedding(_spread_source(), _ids(200))
        out = DataProfileReport([section]).to_html()
        assert out.startswith("<!doctype html>")
        assert "NaN" not in out and "Infinity" not in out
        assert out.count("<svg") == 1


# ------------------------------------------------------------------ comparison
class TestProfileEmbeddingComparison:
    def test_two_models_small_multiples(self) -> None:
        a = _spread_source("a", n=150, dim=8)
        b = _spread_source("b", n=150, dim=8)
        section = profile_embedding_comparison([a, b], _ids(150))
        assert isinstance(section, EmbeddingComparisonSection)
        assert section.kind == "embedding_comparison"
        assert not section.placeholder
        assert not section.degenerate
        assert len(section.models) == 2
        assert section.shared_edges  # shared axis exists
        panels = section.panels()
        assert len(panels) == 3  # table + 2 minis
        assert panels[1].count("<svg") == 1 and panels[2].count("<svg") == 1

    def test_shared_edges_are_identical_axis(self) -> None:
        a = _spread_source("a", n=150, dim=8)
        b = _spread_source("b", n=150, dim=8)
        section = profile_embedding_comparison([a, b], _ids(150))
        # Both minis are drawn against section.shared_edges (one shared x-axis);
        # the per-model counts align to those edges.
        for model in section.models:
            assert len(model.hist_counts) == len(section.shared_edges) - 1

    def test_fewer_than_two_is_placeholder_not_raise(self) -> None:
        section = profile_embedding_comparison([_spread_source("only")], _ids(200))
        assert section.placeholder
        assert section.n_sources == 1
        panels = section.panels()
        assert len(panels) == 1
        assert "at least 2" in panels[0]
        assert "<svg" not in panels[0]

    def test_zero_sources_is_placeholder(self) -> None:
        section = profile_embedding_comparison([], _ids(200))
        assert section.placeholder
        assert section.models == []

    def test_dim_mismatch_still_renders_with_caveat(self) -> None:
        a = _spread_source("a8", n=100, dim=8)
        b = _spread_source("b16", n=100, dim=16)
        section = profile_embedding_comparison([a, b], _ids(100))
        assert section.dims_differ
        assert not section.placeholder
        assert "different dimensionalities" in section.to_markdown()
        assert "different dimensionalities" in section._panel_table()

    def test_all_pre_normalized_is_degenerate(self) -> None:
        rng = np.random.default_rng(3)

        def _unit(name: str) -> ArraySource:
            base = rng.standard_normal((40, 12)).astype(np.float32)
            unit = base / np.linalg.norm(base, axis=1, keepdims=True)
            return ArraySource(name, _ids(40), unit, pre_normalized=True, metric="cosine")

        section = profile_embedding_comparison([_unit("a"), _unit("b")], _ids(40))
        assert section.degenerate
        assert section.shared_edges == []
        panels = section.panels()
        # Table + per-model caveats, no charts.
        assert all("<svg" not in panel for panel in panels)

    def test_mixed_variance_uses_varied_range(self) -> None:
        varied = _spread_source("varied", n=100, dim=8)
        rng = np.random.default_rng(4)
        base = rng.standard_normal((100, 8)).astype(np.float32)
        flat = ArraySource(
            "flat",
            _ids(100),
            base / np.linalg.norm(base, axis=1, keepdims=True),  # norms ~1.0
            pre_normalized=True,
        )
        section = profile_embedding_comparison([varied, flat], _ids(100))
        assert not section.degenerate  # 'varied' anchors the shared range
        # The pre-normalized model still gets a spike histogram on the shared axis.
        flat_model = next(m for m in section.models if m.name == "flat")
        assert sum(flat_model.hist_counts) > 0

    def test_summary_and_rows(self) -> None:
        a = _spread_source("a", n=100, dim=8)
        b = _spread_source("b", n=100, dim=8)
        section = profile_embedding_comparison([a, b], _ids(100))
        assert section.summary["embedding_comparison.n_models"] == 2
        assert len(section.rows()) == 2

    def test_cap_applies_to_comparison(self, caplog: pytest.LogCaptureFixture) -> None:
        a = _spread_source("a", n=200, dim=8)
        b = _spread_source("b", n=200, dim=8)
        with caplog.at_level(logging.INFO):
            section = profile_embedding_comparison([a, b], _ids(200), cap=40)
        assert section.n_truncated == 160
        assert all(model.n_vectors == 40 for model in section.models)

    def test_render_via_container_has_no_nan(self) -> None:
        from langres.core.data_profile.base import DataProfileReport

        a = _spread_source("a", n=100, dim=8)
        b = _spread_source("b", n=100, dim=8)
        section = profile_embedding_comparison([a, b], _ids(100))
        out = DataProfileReport([section]).to_html()
        assert "NaN" not in out and "Infinity" not in out

    def test_rejects_bare_ndarray(self) -> None:
        good = _spread_source("a")
        with pytest.raises(TypeError, match="bare numpy ndarray"):
            profile_embedding_comparison([good, np.zeros((3, 4))], _ids(200))  # type: ignore[list-item]

    def test_markdown_placeholder(self) -> None:
        section = profile_embedding_comparison([_spread_source("only")], _ids(50))
        assert "at least 2" in section.to_markdown()

    def test_markdown_same_dim_has_no_caveat(self) -> None:
        a = _spread_source("a", n=80, dim=8)
        b = _spread_source("b", n=80, dim=8)
        md = profile_embedding_comparison([a, b], _ids(80)).to_markdown()
        assert "different dimensionalities" not in md
        assert "| a |" in md and "| b |" in md


# --------------------------------------------------------------------- internals
class TestSectionInternals:
    def test_good_norms_empty_matrix(self) -> None:
        norms, dropped = _good_norms(np.zeros((0, 4), dtype=np.float32))
        assert norms.shape == (0,)
        assert dropped == 0

    def test_all_dropped_batch_via_batch_one(self) -> None:
        # batch=1 makes each zero/nan/inf row its own fully-dropped batch,
        # exercising the good.size == 0 path in both streaming passes.
        matrix = np.array([[3, 4], [0, 0], [np.nan, 1], [1, 0], [np.inf, 0]], dtype=np.float32)
        section = profile_embedding(ArraySource("m", _ids(5), matrix), _ids(5), batch=1)
        assert section.n_vectors == 2  # norms 5 and 1
        assert section.n_dropped == 3
        assert not section.degenerate

    def test_median_from_hist_no_mass(self) -> None:
        assert _median_from_hist([0.0, 0.0], [0.0, 0.5, 1.0], 0, 0) is None

    def test_median_from_hist_tails_majority(self) -> None:
        assert _median_from_hist([1.0], [0.0, 1.0], 10, 0) is None

    def test_median_from_hist_underflow_at_half_zero_first_bin(self) -> None:
        # underflow == half with a zero first bin -> the count==0 fraction path,
        # returns the bin's lower edge.
        assert _median_from_hist([0.0, 2.0], [0.0, 1.0, 2.0], 2, 0) == pytest.approx(0.0)

    def test_markdown_non_degenerate_shows_cap_row(self) -> None:
        section = profile_embedding(_spread_source(n=200, dim=8), _ids(200), cap=50)
        md = section.to_markdown()
        assert "truncated by cap: 150" in md  # non-degenerate + capped

    def test_panels_show_cap_and_pre_normalized_rows(self) -> None:
        capped = profile_embedding(_spread_source(n=200, dim=8), _ids(200), cap=50)
        assert "truncated by cap" in capped.panels()[1]
        rng = np.random.default_rng(2)
        base = rng.standard_normal((30, 8)).astype(np.float32)
        unit = base / np.linalg.norm(base, axis=1, keepdims=True)
        pn = profile_embedding(ArraySource("cos", _ids(30), unit, pre_normalized=True), _ids(30))
        assert "pre-normalized" in pn.panels()[1]
