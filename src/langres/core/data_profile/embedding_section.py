"""Embedding profile sections: the norm distribution of a model, and a compare.

Two :class:`~langres.core.data_profile.base.ProfileSection` subclasses plus their
profiler functions:

- :class:`EmbeddingSection` / :func:`profile_embedding` -- streams a corpus's
  vectors through an :class:`~langres.core.data_profile.embedding_source.EmbeddingSource`
  and reports the **L2-norm distribution** (a histogram + mean/median/std). Norm
  is the cheapest, most revealing embedding health signal: a spike of zero norms
  means empty inputs, a fat tail means outliers, and an all-``1.0`` spike means
  the store is *pre-normalized* (a cosine index) -- in which case a norm chart is
  uninformative and the section renders a caveat instead of a flat bar (a
  keep-with-hint, never a broken chart).
- :class:`EmbeddingComparisonSection` / :func:`profile_embedding_comparison` --
  the same norm distribution for several models as **small multiples on one
  shared axis**, so their norm scales are visually comparable. With fewer than
  two models it degrades to a keep-with-hint placeholder rather than vanishing.

**Memory.** Every pass streams ``corpus_ids`` in fixed batches and holds at most
one batch of vectors (``O(batch * dim)``), never the whole matrix -- the whole
point of the :class:`EmbeddingSource` seam. The histogram range is derived in a
first stats pass and filled in a second, so both the range and the shared
comparison axis are exact while memory stays bounded.

Leaf module: numpy + stdlib + pydantic + the shared render scaffold; no heavy
imports.
"""

from __future__ import annotations

import html
import logging
import math
from collections.abc import Hashable, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from langres.core import _report_html, _svg
from langres.core.data_profile.accumulators import OnlineHistogram, RunningStats
from langres.core.data_profile.base import ProfileSection
from langres.core.data_profile.embedding_source import EmbeddingSource, _ensure_source

logger = logging.getLogger(__name__)

# Norm-histogram bar color -- explicit (not currentColor) so it renders in light
# and dark themes, matching the eval-report panel palette.
_COLOR_NORM = "#4361ee"

#: Default number of in-range histogram bins for a norm distribution.
_DEFAULT_BINS = 30


# --------------------------------------------------------------------- streaming
def _good_norms(matrix: NDArray[np.floating]) -> tuple[NDArray[np.float64], int]:
    """L2 norms of the usable rows of a batch, plus the count of dropped rows.

    A row is *dropped* (and counted) when its norm is non-finite (a ``NaN``/
    ``Inf`` vector) or exactly zero (an all-zero vector) -- both are degenerate
    for a norm distribution and for cosine, so they are excluded from the stats
    and surfaced as a count instead.

    Returns:
        ``(good_norms, n_dropped)`` where ``good_norms`` holds the finite,
        strictly-positive norms.
    """
    if matrix.shape[0] == 0:
        return np.empty(0, dtype=np.float64), 0
    norms = np.linalg.norm(np.asarray(matrix, dtype=np.float64), axis=1)
    usable = np.isfinite(norms) & (norms > 0.0)
    n_dropped = int((~usable).sum())
    return norms[usable], n_dropped


def _apply_cap(corpus_ids: Sequence[Hashable], cap: int | None) -> tuple[list[Hashable], int]:
    """Truncate ``corpus_ids`` to at most ``cap`` ids, logging when it bites.

    ``cap`` is an explicit, logged bound on how many ids are streamed -- the lever
    for profiling a slice of a very large corpus. Returns the (possibly
    truncated) id list and the number of ids dropped.
    """
    ids = list(corpus_ids)
    if cap is not None and len(ids) > cap:
        dropped = len(ids) - cap
        logger.info(
            "profile_embedding: cap=%d truncated the corpus %d -> %d (dropped %d)",
            cap,
            len(ids),
            cap,
            dropped,
        )
        return ids[:cap], dropped
    return ids, 0


def _stream_stats(
    source: EmbeddingSource, ids: Sequence[Hashable], batch: int
) -> tuple[RunningStats, float, float, int]:
    """First pass: running mean/std, global min/max norm, and dropped-row count.

    Streams ``ids`` in fixed batches (``O(batch * dim)`` memory). The exact global
    min/max is what lets the second pass build a histogram over the true data
    range (rather than guessing one from the first batch).

    Returns:
        ``(stats, global_min, global_max, n_dropped)``. ``global_min`` is ``+inf``
        and ``global_max`` is ``-inf`` when no usable row was seen.
    """
    stats = RunningStats()
    n_dropped = 0
    global_min = math.inf
    global_max = -math.inf
    for start in range(0, len(ids), batch):
        good, dropped = _good_norms(source.vectors_for(ids[start : start + batch]))
        n_dropped += dropped
        if good.size:
            stats.update(good)
            global_min = min(global_min, float(good.min()))
            global_max = max(global_max, float(good.max()))
    return stats, global_min, global_max, n_dropped


def _stream_histogram(
    source: EmbeddingSource,
    ids: Sequence[Hashable],
    batch: int,
    lo: float,
    hi: float,
    n_bins: int,
) -> OnlineHistogram:
    """Second pass: fill a fixed-range norm histogram (``O(batch * dim)`` memory)."""
    hist = OnlineHistogram(lo, hi, n_bins)
    for start in range(0, len(ids), batch):
        good, _ = _good_norms(source.vectors_for(ids[start : start + batch]))
        if good.size:
            hist.update(good)
    return hist


def _is_degenerate_norms(n_vectors: int, lo: float, hi: float, pre_normalized: bool) -> bool:
    """Whether a norm distribution has nothing to plot (render the caveat instead).

    Degenerate when there are no usable vectors, the source is flagged
    pre-normalized (a cosine store -- norms are ~1.0 by construction), or the norm
    range is negligible. The relative-spread floor (``1e-6``) catches
    float32-rounding noise around a constant norm (a pre-normalized store that was
    *not* flagged), while leaving a genuinely narrow-but-real distribution intact.
    """
    if n_vectors == 0 or pre_normalized:
        return True
    if not (hi > lo):
        return True
    scale = max(abs(hi), abs(lo), 1.0)
    return (hi - lo) <= 1e-6 * scale


def _median_from_hist(
    counts: list[float], edges: list[float], underflow: int, overflow: int
) -> float | None:
    """Approximate the median from in-range bin counts (linear within the bin).

    Returns ``None`` when there is no mass, or when the under/overflow tails hold
    the majority (the median then falls outside the histogram range and cannot be
    located from it -- honest ``n/a`` beats a wrong number).
    """
    total = underflow + overflow + sum(counts)
    if total <= 0:
        return None
    if (underflow + overflow) * 2 > total:
        return None
    half = total / 2.0
    cumulative = float(underflow)
    for i, count in enumerate(counts):
        if cumulative + count >= half:
            lo, hi = edges[i], edges[i + 1]
            fraction = (half - cumulative) / count if count > 0 else 0.0
            return float(lo + fraction * (hi - lo))
        cumulative += count
    # Unreachable given the tail-majority guard above (the in-range mass then
    # holds at least half the total, so the loop always returns first).
    return float(edges[-1])  # pragma: no cover


# ----------------------------------------------------------------- single section
class EmbeddingSection(ProfileSection):
    """The L2-norm distribution of one embedding model over a corpus.

    Attributes:
        source_name: The model label the vectors came from.
        dim: Embedding dimensionality.
        n_vectors: Usable rows profiled (finite, strictly-positive norm).
        n_dropped: Rows dropped as zero-norm or non-finite.
        n_truncated: Ids dropped by an explicit ``cap``.
        mean_norm / std_norm: Norm mean and population std (``nan`` when empty).
        median_norm: Histogram-approximate median (``None`` when unavailable).
        min_norm / max_norm: Global norm extremes (``None`` when empty).
        pre_normalized: The vectors are already unit-norm (a cosine store); the
            norm chart is then replaced by a caveat.
        degenerate: No usable vectors, or all norms equal -- render the caveat,
            not a chart.
        hist_edges / hist_counts: In-range histogram edges and counts (empty when
            degenerate).
    """

    kind: str = "embedding"
    source_name: str
    dim: int
    n_vectors: int
    n_dropped: int
    n_truncated: int = 0
    mean_norm: float
    std_norm: float
    median_norm: float | None
    min_norm: float | None
    max_norm: float | None
    pre_normalized: bool = False
    degenerate: bool = False
    hist_edges: list[float] = Field(default_factory=list)
    hist_counts: list[float] = Field(default_factory=list)

    # ------------------------------------------------------------- text surfaces
    def to_markdown(self) -> str:
        """Markdown heading + a norm summary, with a caveat line when degenerate."""
        lines = [
            f"## {self.title}",
            "",
            f"- model: `{_report_html._md_cell(self.source_name)}`",
            f"- dimensionality: {self.dim}",
            f"- vectors profiled: {self.n_vectors}",
            f"- mean ‖v‖: {_report_html._num(self.mean_norm)} | "
            f"median ‖v‖: {_report_html._num(self.median_norm)} | "
            f"std ‖v‖: {_report_html._num(self.std_norm)}",
            f"- dropped (zero/non-finite): {self.n_dropped}",
        ]
        if self.n_truncated:
            lines.append(f"- truncated by cap: {self.n_truncated}")
        caveat = self._caveat_text()
        if caveat:
            lines += ["", f"_{caveat}_"]
        return "\n".join(lines)

    @property
    def summary(self) -> dict[str, object]:
        """Headline norm numbers, keyed by model so several sections do not clash."""
        prefix = f"embedding.{self.source_name}"
        return {
            f"{prefix}.dim": self.dim,
            f"{prefix}.n_vectors": self.n_vectors,
            f"{prefix}.mean_norm": self.mean_norm,
            f"{prefix}.median_norm": self.median_norm,
            f"{prefix}.n_dropped": self.n_dropped,
            f"{prefix}.pre_normalized": self.pre_normalized,
        }

    def rows(self) -> list[dict[str, object]]:
        """One tabular row summarising this model's norm profile."""
        return [
            {
                "model": self.source_name,
                "dim": self.dim,
                "n_vectors": self.n_vectors,
                "mean_norm": self.mean_norm,
                "median_norm": self.median_norm,
                "std_norm": self.std_norm,
                "n_dropped": self.n_dropped,
                "pre_normalized": self.pre_normalized,
            }
        ]

    # -------------------------------------------------------------- html surface
    def panels(self) -> list[str]:
        """A norm-histogram panel (or caveat) plus a key/value summary panel."""
        return [self._panel_histogram(), self._panel_summary()]

    def _caveat_text(self) -> str:
        """The keep-with-hint note for a degenerate norm distribution (else ``""``)."""
        if not self.degenerate:
            return ""
        if self.n_vectors == 0:
            return (
                "No vectors to profile: every requested id was missing, zero-norm, or non-finite."
            )
        if self.pre_normalized:
            return (
                "Vectors are pre-normalized (a cosine store): every norm is ~1.0, so "
                "a norm distribution is uninformative -- compare these embeddings by "
                "direction, not magnitude."
            )
        return (
            f"All {self.n_vectors} vectors share a constant norm "
            f"(~{_report_html._num(self.mean_norm)}); there is no distribution to plot."
        )

    def _panel_histogram(self) -> str:
        heading = f"{self.title} — norm distribution"
        if self.degenerate or not self.hist_edges:
            note = f'<p class="empty">{html.escape(self._caveat_text())}</p>'
            return _report_html.section(heading, note)
        svg = _svg.bar_chart(
            self.hist_edges,
            [("‖v‖", _COLOR_NORM, self.hist_counts)],
            x_label="L2 norm",
            y_label="count",
        )
        return _report_html.section(heading, svg)

    def _panel_summary(self) -> str:
        rows = [
            ("dimensionality", str(self.dim)),
            ("vectors profiled", str(self.n_vectors)),
            ("mean ‖v‖", _report_html._num(self.mean_norm)),
            ("median ‖v‖", _report_html._num(self.median_norm)),
            ("std ‖v‖", _report_html._num(self.std_norm)),
            ("dropped (zero/non-finite)", str(self.n_dropped)),
        ]
        if self.n_truncated:
            rows.append(("truncated by cap", str(self.n_truncated)))
        if self.pre_normalized:
            rows.append(("pre-normalized", "yes (cosine store — norms ~1.0)"))
        cells = "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in rows
        )
        return _report_html.section(f"{self.title} — summary", f'<table class="kv">{cells}</table>')


def profile_embedding(
    source: EmbeddingSource,
    corpus_ids: Sequence[Hashable],
    *,
    batch: int = 4096,
    cap: int | None = None,
    n_bins: int = _DEFAULT_BINS,
    title: str | None = None,
) -> EmbeddingSection:
    """Profile one model's L2-norm distribution over ``corpus_ids`` (streaming).

    Two streaming passes over ``corpus_ids`` (each ``O(batch * dim)`` memory): the
    first derives the exact norm range + mean/std/dropped counts, the second fills
    the histogram over that range. A pre-normalized (cosine) store, or any corpus
    whose norms are constant, is flagged ``degenerate`` so the section renders a
    caveat rather than a flat chart.

    Args:
        source: An :class:`EmbeddingSource` (a bare ndarray raises ``TypeError``).
        corpus_ids: The record ids to profile (a subset of the source's ids;
            unknown ones are dropped and logged by the source).
        batch: Ids per streaming batch (caps the transient vector memory).
        cap: Optional hard limit on ids streamed; logged when it truncates.
        n_bins: In-range histogram bins.
        title: Section title (defaults to ``"Embeddings ({name})"``).

    Returns:
        An :class:`EmbeddingSection`.
    """
    _ensure_source(source)
    ids, n_truncated = _apply_cap(corpus_ids, cap)
    stats, global_min, global_max, n_dropped = _stream_stats(source, ids, batch)
    n_vectors = stats.count
    pre_normalized = bool(getattr(source, "pre_normalized", False))
    degenerate = _is_degenerate_norms(n_vectors, global_min, global_max, pre_normalized)

    hist_edges: list[float] = []
    hist_counts: list[float] = []
    median: float | None = None
    if not degenerate:
        hist = _stream_histogram(source, ids, batch, global_min, global_max, n_bins)
        hist_edges = [float(edge) for edge in hist.edges]
        hist_counts = [float(count) for count in hist.bin_counts]
        median = _median_from_hist(hist_counts, hist_edges, hist.underflow, hist.overflow)

    return EmbeddingSection(
        title=title or f"Embeddings ({source.name})",
        source_name=source.name,
        dim=int(source.dim),
        n_vectors=n_vectors,
        n_dropped=n_dropped,
        n_truncated=n_truncated,
        mean_norm=stats.mean,
        std_norm=stats.std,
        median_norm=median,
        min_norm=global_min if n_vectors else None,
        max_norm=global_max if n_vectors else None,
        pre_normalized=pre_normalized,
        degenerate=degenerate,
        hist_edges=hist_edges,
        hist_counts=hist_counts,
    )


# ----------------------------------------------------------- comparison section
class EmbeddingModelSummary(BaseModel):
    """One model's norm profile inside an :class:`EmbeddingComparisonSection`.

    Carries the per-model headline norms plus the histogram counts binned on the
    comparison's **shared** edges, so every model's mini-chart shares one axis.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    dim: int
    n_vectors: int
    n_dropped: int
    mean_norm: float
    std_norm: float
    median_norm: float | None
    min_norm: float | None
    max_norm: float | None
    pre_normalized: bool
    hist_counts: list[float] = Field(default_factory=list)


class EmbeddingComparisonSection(ProfileSection):
    """Norm distributions of several models as shared-axis small multiples.

    Attributes:
        models: Per-model norm summaries (empty on the placeholder).
        shared_edges: The bin edges every model's mini-histogram shares (empty
            when degenerate/placeholder).
        n_sources: How many sources were passed (drives the placeholder message).
        n_truncated: Ids dropped by an explicit ``cap``.
        placeholder: Fewer than two sources -- render a keep-with-hint, not charts.
        degenerate: No source had norm variance (all pre-normalized/constant/empty)
            -- render caveats instead of charts.
        dims_differ: Models have different dimensionalities (norm magnitudes are
            then not directly comparable -- surfaced as a caveat).
    """

    kind: str = "embedding_comparison"
    models: list[EmbeddingModelSummary] = Field(default_factory=list)
    shared_edges: list[float] = Field(default_factory=list)
    n_sources: int = 0
    n_truncated: int = 0
    placeholder: bool = False
    degenerate: bool = False
    dims_differ: bool = False

    # ------------------------------------------------------------- text surfaces
    def to_markdown(self) -> str:
        """Markdown table of the compared models (or the placeholder note)."""
        if self.placeholder:
            return f"## {self.title}\n\n_{self._placeholder_text()}_"
        lines = [
            f"## {self.title}",
            "",
            "| model | dim | n_vectors | mean ‖v‖ | median ‖v‖ | std ‖v‖ | dropped | pre-norm |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for model in self.models:
            lines.append(
                f"| {_report_html._md_cell(model.name)} | {model.dim} | {model.n_vectors} | "
                f"{_report_html._num(model.mean_norm)} | {_report_html._num(model.median_norm)} | "
                f"{_report_html._num(model.std_norm)} | {model.n_dropped} | "
                f"{'yes' if model.pre_normalized else 'no'} |"
            )
        if self.dims_differ:
            lines += ["", f"_{self._dims_differ_text()}_"]
        return "\n".join(lines)

    @property
    def summary(self) -> dict[str, object]:
        """Headline: how many models, and whether the comparison is renderable."""
        return {
            "embedding_comparison.n_models": len(self.models),
            "embedding_comparison.placeholder": self.placeholder,
            "embedding_comparison.degenerate": self.degenerate,
            "embedding_comparison.dims_differ": self.dims_differ,
        }

    def rows(self) -> list[dict[str, object]]:
        """One tabular row per compared model."""
        return [
            {
                "model": model.name,
                "dim": model.dim,
                "n_vectors": model.n_vectors,
                "mean_norm": model.mean_norm,
                "median_norm": model.median_norm,
                "std_norm": model.std_norm,
                "n_dropped": model.n_dropped,
                "pre_normalized": model.pre_normalized,
            }
            for model in self.models
        ]

    # -------------------------------------------------------------- html surface
    def panels(self) -> list[str]:
        """A comparison table plus one shared-axis mini-histogram per model."""
        if self.placeholder:
            note = f'<p class="empty">{html.escape(self._placeholder_text())}</p>'
            return [_report_html.section(self.title, note)]
        panels = [self._panel_table()]
        for model in self.models:
            panels.append(self._panel_model(model))
        return panels

    def _placeholder_text(self) -> str:
        return (
            f"Add at least 2 embeddings to compare (only {self.n_sources} given). "
            "A comparison needs two or more models to place side by side."
        )

    def _dims_differ_text(self) -> str:
        return (
            "Models have different dimensionalities; norm magnitudes are not directly "
            "comparable across dimensions (higher-dimensional vectors tend to have "
            "larger norms). Compare shapes, not absolute scales."
        )

    def _panel_table(self) -> str:
        head = (
            "<tr><th>model</th><th>dim</th><th>n</th><th>mean ‖v‖</th>"
            "<th>median ‖v‖</th><th>std ‖v‖</th><th>dropped</th><th>pre-norm</th></tr>"
        )
        body = "".join(
            f"<tr><td>{html.escape(model.name)}</td><td>{model.dim}</td>"
            f"<td>{model.n_vectors}</td><td>{_report_html._num(model.mean_norm)}</td>"
            f"<td>{_report_html._num(model.median_norm)}</td>"
            f"<td>{_report_html._num(model.std_norm)}</td><td>{model.n_dropped}</td>"
            f"<td>{'yes' if model.pre_normalized else 'no'}</td></tr>"
            for model in self.models
        )
        note = (
            f'<p class="empty">{html.escape(self._dims_differ_text())}</p>'
            if self.dims_differ
            else ""
        )
        return _report_html.section(self.title, f"<table>{head}{body}</table>{note}")

    def _panel_model(self, model: EmbeddingModelSummary) -> str:
        heading = f"{model.name} — norm distribution"
        if self.degenerate or not self.shared_edges or not model.hist_counts:
            reason = (
                "pre-normalized (norms ~1.0)"
                if model.pre_normalized
                else "no norm variance to plot"
            )
            note = f'<p class="empty">{html.escape(reason)}</p>'
            return _report_html.section(heading, note)
        # Shared edges + per-series max-normalisation => every mini-chart shares
        # one x-axis and a comparable [0, 1] y-axis despite different corpus sizes.
        svg = _svg.bar_chart(
            self.shared_edges,
            [(model.name, _COLOR_NORM, model.hist_counts)],
            width=260,
            height=180,
            x_label="L2 norm",
            y_label="rel. count",
            normalize="max",
        )
        return _report_html.section(heading, svg)


def profile_embedding_comparison(
    sources: Sequence[EmbeddingSource],
    corpus_ids: Sequence[Hashable],
    *,
    batch: int = 4096,
    cap: int | None = None,
    n_bins: int = _DEFAULT_BINS,
    title: str = "Embedding comparison",
) -> EmbeddingComparisonSection:
    """Compare several models' norm distributions on one shared axis (small multiples).

    Each source is streamed twice (``O(batch * dim)`` memory): a stats pass per
    source yields the per-model norms and the union norm range; a second pass fills
    each model's histogram over the **shared** edges of that union range, so the
    mini-charts are directly comparable. Fewer than two sources yields a
    keep-with-hint placeholder (never a raise or a vanish).

    Args:
        sources: Two or more :class:`EmbeddingSource` s (bare ndarrays raise
            ``TypeError``).
        corpus_ids: Record ids to profile through every source.
        batch: Ids per streaming batch.
        cap: Optional hard limit on ids streamed; logged when it truncates.
        n_bins: In-range histogram bins (shared across models).
        title: Section title.

    Returns:
        An :class:`EmbeddingComparisonSection`.
    """
    source_list = list(sources)
    for source in source_list:
        _ensure_source(source)

    if len(source_list) < 2:
        return EmbeddingComparisonSection(title=title, n_sources=len(source_list), placeholder=True)

    ids, n_truncated = _apply_cap(corpus_ids, cap)

    # Pass 1 per source: stats + per-model norm range.
    profiles: list[tuple[EmbeddingSource, RunningStats, float, float, int]] = []
    for source in source_list:
        stats, global_min, global_max, n_dropped = _stream_stats(source, ids, batch)
        profiles.append((source, stats, global_min, global_max, n_dropped))

    # Shared range = union over the sources that actually have norm variance
    # (a pre-normalized / constant source contributes no range of its own).
    varied = [
        (lo, hi)
        for source, stats, lo, hi, _ in profiles
        if not _is_degenerate_norms(
            stats.count, lo, hi, bool(getattr(source, "pre_normalized", False))
        )
    ]
    degenerate = not varied
    shared_edges: list[float] = []
    shared_lo = shared_hi = 0.0
    if not degenerate:
        shared_lo = min(lo for lo, _ in varied)
        shared_hi = max(hi for _, hi in varied)
        shared_edges = [float(edge) for edge in np.linspace(shared_lo, shared_hi, n_bins + 1)]

    # Pass 2 per source: fill each model's histogram on the shared edges.
    models: list[EmbeddingModelSummary] = []
    for source, stats, global_min, global_max, n_dropped in profiles:
        hist_counts: list[float] = []
        median: float | None = None
        if not degenerate:
            hist = _stream_histogram(source, ids, batch, shared_lo, shared_hi, n_bins)
            hist_counts = [float(count) for count in hist.bin_counts]
            median = _median_from_hist(hist_counts, shared_edges, hist.underflow, hist.overflow)
        n_vectors = stats.count
        models.append(
            EmbeddingModelSummary(
                name=source.name,
                dim=int(source.dim),
                n_vectors=n_vectors,
                n_dropped=n_dropped,
                mean_norm=stats.mean,
                std_norm=stats.std,
                median_norm=median,
                min_norm=global_min if n_vectors else None,
                max_norm=global_max if n_vectors else None,
                pre_normalized=bool(getattr(source, "pre_normalized", False)),
                hist_counts=hist_counts,
            )
        )

    dims_differ = len({model.dim for model in models}) > 1
    return EmbeddingComparisonSection(
        title=title,
        models=models,
        shared_edges=shared_edges,
        n_sources=len(source_list),
        n_truncated=n_truncated,
        placeholder=False,
        degenerate=degenerate,
        dims_differ=dims_differ,
    )
