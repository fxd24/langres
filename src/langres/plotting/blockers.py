"""Blocker evaluation visualizations."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from langres.core.reports import BlockerEvaluationReport


def plot_score_distribution(
    report: "BlockerEvaluationReport",
    ax: "matplotlib.axes.Axes | None" = None,
    **kwargs: Any,
) -> "matplotlib.axes.Axes":
    """Plot score distribution histogram for true vs false candidates.

    Args:
        report: Blocker evaluation report
        ax: Matplotlib axes (creates new if None)
        **kwargs: Additional arguments (bins, alpha, colors)

    Returns:
        Axes with histogram plot

    Example:
        >>> report = blocker.evaluate(candidates, gold_clusters)
        >>> report.plot_score_distribution()
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    # Extract histogram data (already computed in report)
    true_hist = report.scores.histogram["true"]
    false_hist = report.scores.histogram["false"]

    # Convert histogram dicts to arrays for plotting
    true_bins = sorted(true_hist.keys())
    true_counts = [true_hist[b] for b in true_bins]
    false_bins = sorted(false_hist.keys())
    false_counts = [false_hist[b] for b in false_bins]

    # Extract kwargs with defaults
    alpha = kwargs.pop("alpha", 0.6)
    colors = kwargs.pop("colors", ["green", "red"])

    # Plot as bar charts (since we have pre-computed histograms)
    # Use bin width as approximation based on range
    if true_bins:
        true_width = (
            (max(true_bins) - min(true_bins)) / len(true_bins) if len(true_bins) > 1 else 0.02
        )
        ax.bar(
            true_bins,
            true_counts,
            width=true_width,
            alpha=alpha,
            color=colors[0],
            label="True Matches",
        )

    if false_bins:
        false_width = (
            (max(false_bins) - min(false_bins)) / len(false_bins) if len(false_bins) > 1 else 0.02
        )
        ax.bar(
            false_bins,
            false_counts,
            width=false_width,
            alpha=alpha,
            color=colors[1],
            label="False Candidates",
        )

    # Add median lines
    ax.axvline(
        report.scores.true_median,
        color="darkgreen",
        linestyle="--",
        linewidth=2,
        label=f"True Median: {report.scores.true_median:.3f}",
    )
    ax.axvline(
        report.scores.false_median,
        color="darkred",
        linestyle="--",
        linewidth=2,
        label=f"False Median: {report.scores.false_median:.3f}",
    )

    # Labels and styling
    ax.set_xlabel("Similarity Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Score Distribution: True Matches vs False Candidates", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return ax


def plot_rank_distribution(
    report: "BlockerEvaluationReport",
    ax: "matplotlib.axes.Axes | None" = None,
    **kwargs: Any,
) -> "matplotlib.axes.Axes":
    """Plot histogram of ranks where true matches appear.

    Args:
        report: Blocker evaluation report
        ax: Matplotlib axes (creates new if None)
        **kwargs: Additional plotting arguments

    Returns:
        Axes with rank distribution plot

    Example:
        >>> report = blocker.evaluate(candidates, gold_clusters)
        >>> report.plot_rank_distribution()
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    # Extract rank data
    ranks = sorted(report.rank_distribution.rank_counts.keys())
    counts = [report.rank_distribution.rank_counts[r] for r in ranks]

    # Bar chart
    ax.bar(ranks, counts, **kwargs)

    # Add median line
    ax.axvline(
        report.rank_distribution.median,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median Rank: {report.rank_distribution.median:.1f}",
    )

    # Labels
    ax.set_xlabel("Rank Position", fontsize=12)
    ax.set_ylabel("Count of True Matches", fontsize=12)
    ax.set_title("Rank Distribution of True Matches", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    return ax


def plot_recall_curve(
    report: "BlockerEvaluationReport",
    ax: "matplotlib.axes.Axes | None" = None,
    **kwargs: Any,
) -> "matplotlib.axes.Axes":
    """Plot recall@k vs k with cost proxy on secondary axis.

    Args:
        report: Blocker evaluation report
        ax: Matplotlib axes (creates new if None)
        **kwargs: Additional plotting arguments

    Returns:
        Axes with recall curve plot

    Example:
        >>> report = blocker.evaluate(candidates, gold_clusters)
        >>> report.plot_recall_curve()
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    k_values = report.recall_curve.k_values
    recall_values = report.recall_curve.recall_values
    avg_pairs = report.recall_curve.avg_pairs_values

    # Primary axis: Recall
    color_recall = "tab:blue"
    ax.set_xlabel("k (Candidates per Entity)", fontsize=12)
    ax.set_ylabel("Recall", fontsize=12, color=color_recall)
    ax.plot(k_values, recall_values, marker="o", linewidth=2, color=color_recall, label="Recall@k")
    ax.tick_params(axis="y", labelcolor=color_recall)
    ax.grid(True, alpha=0.3)

    # Secondary axis: Cost
    ax2 = ax.twinx()
    color_cost = "tab:orange"
    ax2.set_ylabel("Avg Candidates per Entity", fontsize=12, color=color_cost)
    ax2.plot(
        k_values,
        avg_pairs,
        marker="s",
        linewidth=2,
        linestyle="--",
        color=color_cost,
        label="Cost",
    )
    ax2.tick_params(axis="y", labelcolor=color_cost)

    # Target line
    ax.axhline(0.95, color="green", linestyle=":", linewidth=1, label="95% Target")

    # Title
    ax.set_title("Recall@k vs Computational Cost", fontsize=14)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10)

    return ax


def _plot_metrics_bars(
    report: "BlockerEvaluationReport",
    ax: "matplotlib.axes.Axes",
) -> None:
    """Plot bar chart of key metrics.

    Args:
        report: Blocker evaluation report
        ax: Matplotlib axes

    Note:
        This is an internal helper function for plot_evaluation_summary().
    """
    metrics = {
        "Recall": report.candidates.recall,
        "Precision": report.candidates.precision,
        "MAP": report.ranking.map,
        "MRR": report.ranking.mrr,
        "NDCG@20": report.ranking.ndcg_at_20,
    }

    names = list(metrics.keys())
    values = list(metrics.values())

    bars = ax.bar(names, values, color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"])

    # Add value labels on bars
    for bar, value in zip(bars, values, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Key Metrics", fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")


def plot_evaluation_summary(
    report: "BlockerEvaluationReport",
    save_path: str | None = None,
    figsize: tuple[int, int] = (16, 12),
) -> "matplotlib.figure.Figure":
    """Create comprehensive 4-panel evaluation summary.

    Layout:
        ┌─────────────────┬─────────────────┐
        │ Score Dist      │ Rank Dist       │
        ├─────────────────┼─────────────────┤
        │ Recall Curve    │ Metrics Bars    │
        └─────────────────┴─────────────────┘

    Args:
        report: Blocker evaluation report
        save_path: Path to save figure (optional)
        figsize: Figure size in inches (default: (16, 12))

    Returns:
        Matplotlib figure

    Example:
        >>> report = blocker.evaluate(candidates, gold_clusters)
        >>> report.plot_all(save_path="blocker_eval.png")
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel 1: Score distribution
    plot_score_distribution(report, ax=axes[0, 0])

    # Panel 2: Rank distribution
    plot_rank_distribution(report, ax=axes[0, 1])

    # Panel 3: Recall curve
    plot_recall_curve(report, ax=axes[1, 0])

    # Panel 4: Metrics bar chart
    _plot_metrics_bars(report, ax=axes[1, 1])

    # Overall title
    fig.suptitle("Blocker Evaluation Summary", fontsize=16, fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
