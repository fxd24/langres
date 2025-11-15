"""Tests for blocker visualization functions."""

from unittest.mock import MagicMock, call, patch

import pytest

from langres.core.reports import (
    BlockerEvaluationReport,
    CandidateMetrics,
    RankingMetrics,
    RankMetrics,
    RecallCurveStats,
    ScoreMetrics,
)


@pytest.fixture
def sample_report():
    """Create sample BlockerEvaluationReport for testing."""
    return BlockerEvaluationReport(
        candidates=CandidateMetrics(
            recall=0.95,
            precision=0.87,
            total=100,
            avg_per_entity=20.0,
            missed_matches=5,
            false_positives=13,
        ),
        ranking=RankingMetrics(
            map=0.85,
            mrr=0.92,
            ndcg_at_10=0.88,
            ndcg_at_20=0.90,
            recall_at_5=0.80,
            recall_at_10=0.90,
            recall_at_20=0.95,
        ),
        scores=ScoreMetrics(
            separation=0.45,
            true_median=0.85,
            true_mean=0.82,
            true_std=0.12,
            false_median=0.40,
            false_mean=0.38,
            false_std=0.15,
            overlap_fraction=0.15,
            histogram={
                "true": {0.7: 5, 0.8: 15, 0.9: 25},
                "false": {0.3: 20, 0.4: 30, 0.5: 10},
            },
        ),
        ranks=RankMetrics(
            median=3.0,
            percentile_95=12.0,
            percent_in_top_5=78.0,
            percent_in_top_10=92.0,
            percent_in_top_20=98.0,
            rank_counts={1: 15, 2: 10, 3: 8, 5: 5, 10: 2},
        ),
        recall_curve=RecallCurveStats(
            k_values=[1, 5, 10, 20],
            recall_values=[0.60, 0.85, 0.92, 0.98],
            avg_pairs_values=[1.0, 5.0, 10.0, 20.0],
        ),
    )


@patch("langres.plotting.blockers.plt")
def test_plot_score_distribution_creates_histogram(mock_plt, sample_report):
    """Test plot_score_distribution creates overlaid histograms."""
    from langres.plotting.blockers import plot_score_distribution

    mock_ax = MagicMock()
    result = plot_score_distribution(sample_report, ax=mock_ax)

    # Verify bar() called for both true and false scores
    assert mock_ax.bar.call_count >= 2  # At least 2 bar calls

    # Verify median lines added (2 axvline calls)
    assert mock_ax.axvline.call_count == 2

    # Verify labels set
    mock_ax.set_xlabel.assert_called_once()
    mock_ax.set_ylabel.assert_called_once()
    mock_ax.set_title.assert_called_once()

    # Verify legend called
    mock_ax.legend.assert_called_once()

    # Verify grid enabled
    mock_ax.grid.assert_called_once()

    # Returns the axes
    assert result == mock_ax


@patch("langres.plotting.blockers.plt")
def test_plot_score_distribution_creates_figure_if_no_ax(mock_plt, sample_report):
    """Test creates new figure when ax=None."""
    from langres.plotting.blockers import plot_score_distribution

    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    result = plot_score_distribution(sample_report, ax=None)

    # Verify subplots called
    mock_plt.subplots.assert_called_once()

    # Should return the created axes
    assert result == mock_ax


@patch("langres.plotting.blockers.plt")
def test_plot_rank_distribution_creates_bar_chart(mock_plt, sample_report):
    """Test plot_rank_distribution creates bar chart of ranks."""
    from langres.plotting.blockers import plot_rank_distribution

    mock_ax = MagicMock()
    result = plot_rank_distribution(sample_report, ax=mock_ax)

    # Verify bar() called
    mock_ax.bar.assert_called_once()

    # Verify median line added
    mock_ax.axvline.assert_called_once()

    # Verify labels set
    mock_ax.set_xlabel.assert_called_once()
    mock_ax.set_ylabel.assert_called_once()
    mock_ax.set_title.assert_called_once()

    # Verify legend called
    mock_ax.legend.assert_called_once()

    # Verify grid enabled
    mock_ax.grid.assert_called_once()

    # Returns the axes
    assert result == mock_ax


@patch("langres.plotting.blockers.plt")
def test_plot_rank_distribution_creates_figure_if_no_ax(mock_plt, sample_report):
    """Test creates new figure when ax=None."""
    from langres.plotting.blockers import plot_rank_distribution

    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    result = plot_rank_distribution(sample_report, ax=None)

    # Verify subplots called
    mock_plt.subplots.assert_called_once()

    # Should return the created axes
    assert result == mock_ax


@patch("langres.plotting.blockers.plt")
def test_plot_recall_curve_creates_line_plot(mock_plt, sample_report):
    """Test plot_recall_curve creates recall vs k plot with cost proxy."""
    from langres.plotting.blockers import plot_recall_curve

    mock_ax = MagicMock()
    mock_ax2 = MagicMock()
    mock_ax.twinx.return_value = mock_ax2

    result = plot_recall_curve(sample_report, ax=mock_ax)

    # Verify primary plot (recall)
    assert mock_ax.plot.call_count == 1

    # Verify secondary axis created
    mock_ax.twinx.assert_called_once()

    # Verify secondary plot (cost)
    assert mock_ax2.plot.call_count == 1

    # Verify target line added
    assert mock_ax.axhline.call_count == 1

    # Verify labels set
    mock_ax.set_xlabel.assert_called_once()
    mock_ax.set_ylabel.assert_called_once()
    mock_ax2.set_ylabel.assert_called_once()

    # Verify title
    mock_ax.set_title.assert_called_once()

    # Verify legend (combines both axes)
    mock_ax.legend.assert_called_once()

    # Returns the primary axes
    assert result == mock_ax


@patch("langres.plotting.blockers.plt")
def test_plot_recall_curve_creates_figure_if_no_ax(mock_plt, sample_report):
    """Test creates new figure when ax=None."""
    from langres.plotting.blockers import plot_recall_curve

    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_ax2 = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    mock_ax.twinx.return_value = mock_ax2

    result = plot_recall_curve(sample_report, ax=None)

    # Verify subplots called
    mock_plt.subplots.assert_called_once()

    # Should return the created axes
    assert result == mock_ax


@patch("langres.plotting.blockers.plt")
def test_plot_evaluation_summary_creates_4_panels(mock_plt, sample_report):
    """Test plot_evaluation_summary creates 2x2 subplot grid."""
    from langres.plotting.blockers import plot_evaluation_summary

    mock_fig = MagicMock()
    mock_axes = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_axes)

    # Mock axes indexing to return MagicMock objects
    mock_ax_00 = MagicMock()
    mock_ax_01 = MagicMock()
    mock_ax_10 = MagicMock()
    mock_ax_11 = MagicMock()
    mock_ax_11.bar.return_value = [MagicMock(), MagicMock()]  # For bar chart value labels

    def axes_getitem(index):
        if index == (0, 0):
            return mock_ax_00
        elif index == (0, 1):
            return mock_ax_01
        elif index == (1, 0):
            return mock_ax_10
        elif index == (1, 1):
            return mock_ax_11
        raise IndexError

    mock_axes.__getitem__ = MagicMock(side_effect=axes_getitem)

    result = plot_evaluation_summary(sample_report)

    # Verify 2x2 subplots created
    mock_plt.subplots.assert_called_once()
    call_args = mock_plt.subplots.call_args
    assert call_args[0] == (2, 2)  # 2x2 grid

    # Verify suptitle set
    mock_fig.suptitle.assert_called_once()

    # Verify tight_layout called
    mock_plt.tight_layout.assert_called_once()

    # Returns the figure
    assert result == mock_fig


@patch("langres.plotting.blockers.plt")
def test_plot_evaluation_summary_saves_figure(mock_plt, sample_report, tmp_path):
    """Test save_path parameter saves figure."""
    from langres.plotting.blockers import plot_evaluation_summary

    mock_fig = MagicMock()
    mock_axes = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_axes)

    # Mock axes indexing
    mock_ax_00 = MagicMock()
    mock_ax_01 = MagicMock()
    mock_ax_10 = MagicMock()
    mock_ax_11 = MagicMock()
    mock_ax_11.bar.return_value = [MagicMock(), MagicMock()]

    def axes_getitem(index):
        if index == (0, 0):
            return mock_ax_00
        elif index == (0, 1):
            return mock_ax_01
        elif index == (1, 0):
            return mock_ax_10
        elif index == (1, 1):
            return mock_ax_11
        raise IndexError

    mock_axes.__getitem__ = MagicMock(side_effect=axes_getitem)

    save_path = str(tmp_path / "test_plot.png")
    result = plot_evaluation_summary(sample_report, save_path=save_path)

    # Verify savefig called
    mock_fig.savefig.assert_called_once_with(save_path, dpi=300, bbox_inches="tight")


@patch("langres.plotting.blockers.plt")
def test_plot_evaluation_summary_custom_figsize(mock_plt, sample_report):
    """Test custom figsize parameter works."""
    from langres.plotting.blockers import plot_evaluation_summary

    mock_fig = MagicMock()
    mock_axes = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_axes)

    # Mock axes indexing
    mock_ax_00 = MagicMock()
    mock_ax_01 = MagicMock()
    mock_ax_10 = MagicMock()
    mock_ax_11 = MagicMock()
    mock_ax_11.bar.return_value = [MagicMock(), MagicMock()]

    def axes_getitem(index):
        if index == (0, 0):
            return mock_ax_00
        elif index == (0, 1):
            return mock_ax_01
        elif index == (1, 0):
            return mock_ax_10
        elif index == (1, 1):
            return mock_ax_11
        raise IndexError

    mock_axes.__getitem__ = MagicMock(side_effect=axes_getitem)

    custom_figsize = (20, 15)
    plot_evaluation_summary(sample_report, figsize=custom_figsize)

    # Verify figsize passed to subplots
    call_args = mock_plt.subplots.call_args
    assert call_args[1]["figsize"] == custom_figsize


@patch("langres.plotting.blockers.plt")
def test_plot_metrics_bars_is_called_by_summary(mock_plt, sample_report):
    """Test _plot_metrics_bars is called as part of summary."""
    from langres.plotting.blockers import plot_evaluation_summary

    mock_fig = MagicMock()
    mock_axes = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_axes)

    # Mock axes indexing
    mock_ax_00 = MagicMock()
    mock_ax_01 = MagicMock()
    mock_ax_10 = MagicMock()
    mock_ax_11 = MagicMock()

    # Mock bar() to return list of bar objects
    mock_bar_1 = MagicMock()
    mock_bar_2 = MagicMock()
    mock_bar_1.get_height.return_value = 0.95
    mock_bar_2.get_height.return_value = 0.85
    mock_bar_1.get_x.return_value = 0.0
    mock_bar_2.get_x.return_value = 1.0
    mock_bar_1.get_width.return_value = 0.8
    mock_bar_2.get_width.return_value = 0.8
    mock_ax_11.bar.return_value = [mock_bar_1, mock_bar_2]

    def axes_getitem(index):
        if index == (0, 0):
            return mock_ax_00
        elif index == (0, 1):
            return mock_ax_01
        elif index == (1, 0):
            return mock_ax_10
        elif index == (1, 1):
            return mock_ax_11
        raise IndexError

    mock_axes.__getitem__ = MagicMock(side_effect=axes_getitem)

    plot_evaluation_summary(sample_report)

    # Verify bar chart created (panel 4)
    assert mock_ax_11.bar.call_count == 1

    # Verify text labels added to bars
    assert mock_ax_11.text.call_count >= 2  # At least 2 value labels
