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
        rank_distribution=RankMetrics(
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


@patch("matplotlib.pyplot")
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


@patch("matplotlib.pyplot")
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


@patch("matplotlib.pyplot")
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


@patch("matplotlib.pyplot")
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


@patch("matplotlib.pyplot")
def test_plot_recall_curve_creates_line_plot(mock_plt, sample_report):
    """Test plot_recall_curve creates recall vs k plot with cost proxy."""
    from langres.plotting.blockers import plot_recall_curve

    mock_ax = MagicMock()
    mock_ax2 = MagicMock()
    mock_ax.twinx.return_value = mock_ax2

    # Mock get_legend_handles_labels to return proper data
    mock_ax.get_legend_handles_labels.return_value = (
        [MagicMock(), MagicMock()],
        ["Recall@k", "95% Target"],
    )
    mock_ax2.get_legend_handles_labels.return_value = ([MagicMock()], ["Cost"])

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


@patch("matplotlib.pyplot")
def test_plot_recall_curve_creates_figure_if_no_ax(mock_plt, sample_report):
    """Test creates new figure when ax=None."""
    from langres.plotting.blockers import plot_recall_curve

    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_ax2 = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    mock_ax.twinx.return_value = mock_ax2

    # Mock get_legend_handles_labels to return proper data
    mock_ax.get_legend_handles_labels.return_value = (
        [MagicMock(), MagicMock()],
        ["Recall@k", "95% Target"],
    )
    mock_ax2.get_legend_handles_labels.return_value = ([MagicMock()], ["Cost"])

    result = plot_recall_curve(sample_report, ax=None)

    # Verify subplots called
    mock_plt.subplots.assert_called_once()

    # Should return the created axes
    assert result == mock_ax


@patch("matplotlib.pyplot")
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

    # Mock get_legend_handles_labels for recall curve (axes[1, 0])
    mock_ax_10.get_legend_handles_labels.return_value = (
        [MagicMock(), MagicMock()],
        ["Recall@k", "95% Target"],
    )
    mock_ax_10_secondary = MagicMock()
    mock_ax_10_secondary.get_legend_handles_labels.return_value = ([MagicMock()], ["Cost"])
    mock_ax_10.twinx.return_value = mock_ax_10_secondary

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


@patch("matplotlib.pyplot")
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

    # Mock get_legend_handles_labels for recall curve (axes[1, 0])
    mock_ax_10.get_legend_handles_labels.return_value = (
        [MagicMock(), MagicMock()],
        ["Recall@k", "95% Target"],
    )
    mock_ax_10_secondary = MagicMock()
    mock_ax_10_secondary.get_legend_handles_labels.return_value = ([MagicMock()], ["Cost"])
    mock_ax_10.twinx.return_value = mock_ax_10_secondary

    save_path = str(tmp_path / "test_plot.png")
    result = plot_evaluation_summary(sample_report, save_path=save_path)

    # Verify savefig called
    mock_fig.savefig.assert_called_once_with(save_path, dpi=300, bbox_inches="tight")


@patch("matplotlib.pyplot")
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

    # Mock get_legend_handles_labels for recall curve (axes[1, 0])
    mock_ax_10.get_legend_handles_labels.return_value = (
        [MagicMock(), MagicMock()],
        ["Recall@k", "95% Target"],
    )
    mock_ax_10_secondary = MagicMock()
    mock_ax_10_secondary.get_legend_handles_labels.return_value = ([MagicMock()], ["Cost"])
    mock_ax_10.twinx.return_value = mock_ax_10_secondary

    custom_figsize = (20, 15)
    plot_evaluation_summary(sample_report, figsize=custom_figsize)

    # Verify figsize passed to subplots
    call_args = mock_plt.subplots.call_args
    assert call_args[1]["figsize"] == custom_figsize


@patch("matplotlib.pyplot")
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

    # Mock get_legend_handles_labels for recall curve (axes[1, 0])
    mock_ax_10.get_legend_handles_labels.return_value = (
        [MagicMock(), MagicMock()],
        ["Recall@k", "95% Target"],
    )
    mock_ax_10_secondary = MagicMock()
    mock_ax_10_secondary.get_legend_handles_labels.return_value = ([MagicMock()], ["Cost"])
    mock_ax_10.twinx.return_value = mock_ax_10_secondary

    plot_evaluation_summary(sample_report)

    # Verify bar chart created (panel 4)
    assert mock_ax_11.bar.call_count == 1

    # Verify text labels added to bars
    assert mock_ax_11.text.call_count >= 2  # At least 2 value labels


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


@patch("matplotlib.pyplot")
def test_plot_score_distribution_with_empty_true_histogram(mock_plt):
    """Test plotting when no true matches exist (empty true histogram)."""
    from langres.plotting.blockers import plot_score_distribution

    # Create report with no true matches (recall=0.0, empty true histogram)
    report = BlockerEvaluationReport(
        candidates=CandidateMetrics(
            recall=0.0,
            precision=0.0,
            total=50,
            avg_per_entity=10.0,
            missed_matches=20,
            false_positives=50,
        ),
        ranking=RankingMetrics(
            map=0.0,
            mrr=0.0,
            ndcg_at_10=0.0,
            ndcg_at_20=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            recall_at_20=0.0,
        ),
        scores=ScoreMetrics(
            separation=0.0,
            true_median=0.0,
            true_mean=0.0,
            true_std=0.0,
            false_median=0.35,
            false_mean=0.33,
            false_std=0.12,
            overlap_fraction=0.0,
            histogram={
                "true": {},  # Empty - no true matches
                "false": {0.2: 10, 0.3: 20, 0.4: 15, 0.5: 5},
            },
        ),
        rank_distribution=RankMetrics(
            median=1.0,  # Min valid value when no ranks exist
            percentile_95=1.0,  # Min valid value when no ranks exist
            percent_in_top_5=0.0,
            percent_in_top_10=0.0,
            percent_in_top_20=0.0,
            rank_counts={},  # No ranks when no true matches
        ),
        recall_curve=RecallCurveStats(
            k_values=[1, 5, 10, 20],
            recall_values=[0.0, 0.0, 0.0, 0.0],
            avg_pairs_values=[1.0, 5.0, 10.0, 20.0],
        ),
    )

    mock_ax = MagicMock()
    result = plot_score_distribution(report, ax=mock_ax)

    # Should not crash
    assert result is not None

    # Should still create the plot with false histogram only
    # Only 1 bar() call for false histogram (true is empty)
    assert mock_ax.bar.call_count == 1

    # Should still add median lines (2 axvline calls)
    assert mock_ax.axvline.call_count == 2

    # Should still set labels
    mock_ax.set_xlabel.assert_called_once()
    mock_ax.set_ylabel.assert_called_once()
    mock_ax.set_title.assert_called_once()


@patch("matplotlib.pyplot")
def test_plot_score_distribution_with_single_bin(mock_plt):
    """Test plotting when histogram has single bin (all scores identical)."""
    from langres.plotting.blockers import plot_score_distribution

    # All true matches have same score, all false candidates have same score
    report = BlockerEvaluationReport(
        candidates=CandidateMetrics(
            recall=1.0,
            precision=0.5,
            total=40,
            avg_per_entity=8.0,
            missed_matches=0,
            false_positives=20,
        ),
        ranking=RankingMetrics(
            map=0.95,
            mrr=1.0,
            ndcg_at_10=0.98,
            ndcg_at_20=0.98,
            recall_at_5=1.0,
            recall_at_10=1.0,
            recall_at_20=1.0,
        ),
        scores=ScoreMetrics(
            separation=0.55,
            true_median=0.95,
            true_mean=0.95,
            true_std=0.0,  # No variance
            false_median=0.40,
            false_mean=0.40,
            false_std=0.0,  # No variance
            overlap_fraction=0.0,
            histogram={
                "true": {0.95: 20},  # Single bin - all scores identical
                "false": {0.40: 20},  # Single bin - all scores identical
            },
        ),
        rank_distribution=RankMetrics(
            median=1.0,
            percentile_95=1.0,
            percent_in_top_5=100.0,
            percent_in_top_10=100.0,
            percent_in_top_20=100.0,
            rank_counts={1: 20},
        ),
        recall_curve=RecallCurveStats(
            k_values=[1, 5, 10],
            recall_values=[1.0, 1.0, 1.0],
            avg_pairs_values=[1.0, 5.0, 10.0],
        ),
    )

    mock_ax = MagicMock()
    result = plot_score_distribution(report, ax=mock_ax)

    # Should not crash
    assert result is not None

    # Should create 2 bar charts (true and false)
    assert mock_ax.bar.call_count == 2

    # Verify both bar() calls were made (for single bins)
    calls = mock_ax.bar.call_args_list
    # First call: true histogram with single bin at 0.95
    assert calls[0][0][0] == [0.95]
    assert calls[0][0][1] == [20]
    # Second call: false histogram with single bin at 0.40
    assert calls[1][0][0] == [0.40]
    assert calls[1][0][1] == [20]


@patch("matplotlib.pyplot")
def test_plot_rank_distribution_with_extreme_ranks(mock_plt):
    """Test plotting when ranks span very wide range (1 to 10000)."""
    from langres.plotting.blockers import plot_rank_distribution

    # Create report with ranks spanning wide range
    report = BlockerEvaluationReport(
        candidates=CandidateMetrics(
            recall=0.80,
            precision=0.10,
            total=1000,
            avg_per_entity=200.0,
            missed_matches=20,
            false_positives=900,
        ),
        ranking=RankingMetrics(
            map=0.25,
            mrr=0.35,
            ndcg_at_10=0.40,
            ndcg_at_20=0.45,
            recall_at_5=0.15,
            recall_at_10=0.30,
            recall_at_20=0.55,
        ),
        scores=ScoreMetrics(
            separation=0.15,
            true_median=0.60,
            true_mean=0.58,
            true_std=0.20,
            false_median=0.55,
            false_mean=0.52,
            false_std=0.22,
            overlap_fraction=0.65,
            histogram={
                "true": {0.5: 20, 0.6: 30, 0.7: 25, 0.8: 5},
                "false": {0.4: 200, 0.5: 300, 0.6: 250, 0.7: 150},
            },
        ),
        rank_distribution=RankMetrics(
            median=25.0,
            percentile_95=5000.0,  # Extreme 95th percentile
            percent_in_top_5=15.0,
            percent_in_top_10=30.0,
            percent_in_top_20=55.0,
            rank_counts={
                1: 5,
                2: 3,
                5: 7,
                10: 10,
                25: 15,
                100: 20,
                1000: 15,
                5000: 10,
                10000: 5,  # Extreme rank
            },
        ),
        recall_curve=RecallCurveStats(
            k_values=[1, 5, 10, 20, 50],
            recall_values=[0.05, 0.15, 0.30, 0.55, 0.75],
            avg_pairs_values=[1.0, 5.0, 10.0, 20.0, 50.0],
        ),
    )

    mock_ax = MagicMock()
    result = plot_rank_distribution(report, ax=mock_ax)

    # Should not crash with extreme values
    assert result is not None

    # Should create bar chart
    mock_ax.bar.assert_called_once()

    # Verify bar chart uses all rank values including extremes
    call_args = mock_ax.bar.call_args
    ranks_plotted = call_args[0][0]
    assert min(ranks_plotted) == 1
    assert max(ranks_plotted) == 10000

    # Should still add median line
    mock_ax.axvline.assert_called_once()


@patch("matplotlib.pyplot")
def test_plot_recall_curve_with_single_k(mock_plt):
    """Test recall curve with only one k value."""
    from langres.plotting.blockers import plot_recall_curve

    # Create report with only single k value
    report = BlockerEvaluationReport(
        candidates=CandidateMetrics(
            recall=0.75,
            precision=0.85,
            total=30,
            avg_per_entity=6.0,
            missed_matches=5,
            false_positives=5,
        ),
        ranking=RankingMetrics(
            map=0.80,
            mrr=0.88,
            ndcg_at_10=0.82,
            ndcg_at_20=0.84,
            recall_at_5=0.75,
            recall_at_10=0.75,
            recall_at_20=0.75,
        ),
        scores=ScoreMetrics(
            separation=0.40,
            true_median=0.80,
            true_mean=0.78,
            true_std=0.10,
            false_median=0.35,
            false_mean=0.33,
            false_std=0.12,
            overlap_fraction=0.10,
            histogram={
                "true": {0.7: 5, 0.8: 10, 0.9: 5},
                "false": {0.3: 2, 0.4: 3},
            },
        ),
        rank_distribution=RankMetrics(
            median=2.0,
            percentile_95=8.0,
            percent_in_top_5=85.0,
            percent_in_top_10=100.0,
            percent_in_top_20=100.0,
            rank_counts={1: 8, 2: 7, 3: 3, 5: 2},
        ),
        recall_curve=RecallCurveStats(
            k_values=[5],  # Only single k value
            recall_values=[0.75],
            avg_pairs_values=[6.0],
        ),
    )

    mock_ax = MagicMock()
    mock_ax2 = MagicMock()
    mock_ax.twinx.return_value = mock_ax2

    # Mock get_legend_handles_labels
    mock_ax.get_legend_handles_labels.return_value = (
        [MagicMock(), MagicMock()],
        ["Recall@k", "95% Target"],
    )
    mock_ax2.get_legend_handles_labels.return_value = ([MagicMock()], ["Cost"])

    result = plot_recall_curve(report, ax=mock_ax)

    # Should not crash with single data point
    assert result is not None

    # Should still plot both lines (recall and cost)
    assert mock_ax.plot.call_count == 1
    assert mock_ax2.plot.call_count == 1

    # Verify single point plotted
    recall_call = mock_ax.plot.call_args
    assert recall_call[0][0] == [5]  # Single k value
    assert recall_call[0][1] == [0.75]  # Single recall value


@patch("matplotlib.pyplot")
def test_plot_score_distribution_with_empty_false_histogram(mock_plt):
    """Test plotting when no false candidates exist (empty false histogram)."""
    from langres.plotting.blockers import plot_score_distribution

    # Create report with perfect precision (all candidates are true matches)
    report = BlockerEvaluationReport(
        candidates=CandidateMetrics(
            recall=1.0,
            precision=1.0,  # Perfect precision - no false candidates
            total=20,
            avg_per_entity=4.0,
            missed_matches=0,
            false_positives=0,
        ),
        ranking=RankingMetrics(
            map=1.0,
            mrr=1.0,
            ndcg_at_10=1.0,
            ndcg_at_20=1.0,
            recall_at_5=1.0,
            recall_at_10=1.0,
            recall_at_20=1.0,
        ),
        scores=ScoreMetrics(
            separation=1.0,
            true_median=0.90,
            true_mean=0.88,
            true_std=0.08,
            false_median=0.0,  # No false candidates
            false_mean=0.0,
            false_std=0.0,
            overlap_fraction=0.0,
            histogram={
                "true": {0.75: 3, 0.85: 8, 0.95: 9},  # Multiple bins
                "false": {},  # Empty - no false candidates
            },
        ),
        rank_distribution=RankMetrics(
            median=1.0,
            percentile_95=2.0,
            percent_in_top_5=100.0,
            percent_in_top_10=100.0,
            percent_in_top_20=100.0,
            rank_counts={1: 15, 2: 5},
        ),
        recall_curve=RecallCurveStats(
            k_values=[1, 5, 10],
            recall_values=[1.0, 1.0, 1.0],
            avg_pairs_values=[1.0, 5.0, 10.0],
        ),
    )

    mock_ax = MagicMock()
    result = plot_score_distribution(report, ax=mock_ax)

    # Should not crash
    assert result is not None

    # Should create the plot with true histogram only
    # Only 1 bar() call for true histogram (false is empty)
    assert mock_ax.bar.call_count == 1

    # Should still add median lines (2 axvline calls)
    assert mock_ax.axvline.call_count == 2

    # Should still set labels
    mock_ax.set_xlabel.assert_called_once()
    mock_ax.set_ylabel.assert_called_once()
    mock_ax.set_title.assert_called_once()


@patch("matplotlib.pyplot")
def test_plot_evaluation_summary_saves_to_file(mock_plt, sample_report, tmp_path):
    """Test that plot_evaluation_summary saves file successfully."""
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

    # Mock get_legend_handles_labels for recall curve
    mock_ax_10.get_legend_handles_labels.return_value = (
        [MagicMock(), MagicMock()],
        ["Recall@k", "95% Target"],
    )
    mock_ax_10_secondary = MagicMock()
    mock_ax_10_secondary.get_legend_handles_labels.return_value = ([MagicMock()], ["Cost"])
    mock_ax_10.twinx.return_value = mock_ax_10_secondary

    save_path = str(tmp_path / "summary.png")
    result = plot_evaluation_summary(sample_report, save_path=save_path)

    # Verify savefig was called with correct path
    mock_fig.savefig.assert_called_once_with(save_path, dpi=300, bbox_inches="tight")

    # Returns the figure
    assert result == mock_fig
