"""Label-structure profile section: what the gold clustering *looks like*.

The first tabular block of the data-profile report. Given a gold clustering (an
equivalence partition of record ids into match-clusters), it reports the shape
that governs every downstream ER decision: how many records and clusters, how
lopsided the cluster sizes are, and -- the number that sets the whole difficulty
of the task -- the **positive-pair prevalence** / class-imbalance ratio. A
handful of large clusters vs a long tail of singletons is a very different
problem from uniform pairs, and this section makes that visible before a single
matcher runs.

Generic by construction: :func:`profile_label_structure` takes a plain
``Sequence`` of clusters (each a ``Collection`` of hashable record ids), never a
benchmark-coupled type -- Wave 2 adapts benchmark gold into this shape. A
``None`` clustering means "no gold available", and the profiler returns ``None``
so the section is simply absent from the report (graceful degradation).
"""

from __future__ import annotations

import html
import math
from collections import Counter
from collections.abc import Collection, Hashable, Sequence
from typing import Any, Literal

from langres.core import _report_html, _svg
from langres.core.data_profile.base import ProfileSection

#: Cap on the number of bars in the cluster-size histogram. Sizes at or above the
#: cap fold into a single overflow bin so a pathological dedup dataset (one
#: 10k-member cluster) cannot emit 10k bars.
_MAX_SIZE_BINS = 20

#: Single-series bar color for the cluster-size histogram (explicit, not
#: ``currentColor``, so it reads in light and dark themes -- matches the
#: eval_report palette).
_COLOR_CLUSTERS = "#4361ee"


def _shannon_entropy_bits(
    size_distribution: Sequence[tuple[int, int]], n_records: int
) -> float | None:
    """Shannon entropy (bits) of the cluster-size distribution, with a log(0) guard.

    Treats a record's cluster membership as the random variable: a cluster of
    size ``s`` carries probability ``p = s / n_records``, and there are ``count``
    such clusters. Entropy is ``-sum(count * p * log2(p))``.

    The **explicit log(0) guard** is the ``p > 0`` test: a probability of zero
    (an empty cluster, or the empty-corpus case) contributes ``0`` to the sum by
    the ``0 * log(0) = 0`` convention, never ``log2(0) = -inf``. A single cluster
    yields ``p = 1`` and ``log2(1) = 0``, so entropy is exactly ``0.0`` (a fully
    predictable partition), not a crash.

    Returns:
        Entropy in bits, or ``None`` when there are no records (undefined).
    """
    if n_records <= 0:
        return None
    entropy = 0.0
    for size, count in size_distribution:
        p = size / n_records
        # log(0) guard: a zero-probability term (a degenerate empty cluster, or
        # the empty corpus) contributes 0 by the ``0 * log(0) == 0`` convention,
        # never ``log2(0) == -inf``.
        if p > 0.0:
            entropy -= count * p * math.log2(p)
    return entropy


def _size_histogram(
    size_distribution: Sequence[tuple[int, int]],
) -> tuple[list[float], list[float]]:
    """Build ``(edges, counts)`` for the cluster-size bar chart from a size map.

    One bar per integer cluster size ``1..max_size``; when ``max_size`` exceeds
    :data:`_MAX_SIZE_BINS`, sizes at or above the cap fold into a single overflow
    bar. Returns two empty lists for an empty distribution (the chart then draws
    bare axes rather than raising).
    """
    counts_by_size = dict(size_distribution)
    if not counts_by_size:
        return [], []
    max_size = max(counts_by_size)
    if max_size <= _MAX_SIZE_BINS:
        edges = [float(i) for i in range(1, max_size + 2)]
        counts = [float(counts_by_size.get(i, 0)) for i in range(1, max_size + 1)]
        return edges, counts
    # Overflow: bars for sizes 1..cap-1, then one bar for everything >= cap.
    edges = [float(i) for i in range(1, _MAX_SIZE_BINS + 1)]
    edges.append(float(max_size + 1))
    counts = [float(counts_by_size.get(i, 0)) for i in range(1, _MAX_SIZE_BINS)]
    counts.append(float(sum(c for s, c in counts_by_size.items() if s >= _MAX_SIZE_BINS)))
    return edges, counts


class LabelStructureSection(ProfileSection):
    """Cluster-shape + class-imbalance metrics of a gold clustering.

    A frozen :class:`ProfileSection` holding the headline label-structure numbers
    plus the exact cluster-size distribution (so :meth:`rows` and the histogram
    both derive from one stored field). Build it with
    :func:`profile_label_structure`.

    Attributes:
        n_records: Total records covered by the clustering (clustered ids plus any
            implied singletons declared via ``n_records=``).
        n_clusters: Number of equivalence classes (implied singletons included).
        n_singletons: Clusters of size 1.
        n_multi: Clusters of size >= 2.
        max_cluster_size: Largest cluster size (``0`` for an empty clustering).
        mean_cluster_size: ``n_records / n_clusters``; ``None`` when empty.
        positive_pairs: In-cluster pairs -- ``sum(C(size, 2))`` (the ER positives).
        total_pairs: All candidate pairs -- ``C(n_records, 2)``.
        prevalence: ``positive_pairs / total_pairs``; ``None`` when no pairs exist.
        imbalance_ratio: Negatives per positive (the ``N`` in ``1:N``); ``None``
            when there are no positive pairs.
        entropy_bits: Shannon entropy (bits) of the cluster-size distribution;
            ``None`` when empty.
        size_distribution: Sorted ``(cluster_size, n_clusters)`` pairs -- the exact
            distribution the histogram and :meth:`rows` render from.
    """

    kind: Literal["label_structure"] = "label_structure"

    n_records: int
    n_clusters: int
    n_singletons: int
    n_multi: int
    max_cluster_size: int
    mean_cluster_size: float | None
    positive_pairs: int
    total_pairs: int
    prevalence: float | None
    imbalance_ratio: float | None
    entropy_bits: float | None
    size_distribution: list[tuple[int, int]]

    # ------------------------------------------------------------- shared render
    def _metrics_kv(self) -> list[tuple[str, str]]:
        """The headline metrics as ``(label, display)`` pairs (markdown + HTML share this)."""
        imbalance = f"1:{self.imbalance_ratio:,.0f}" if self.imbalance_ratio is not None else "n/a"
        return [
            ("records", f"{self.n_records:,}"),
            ("clusters", f"{self.n_clusters:,}"),
            ("singletons", f"{self.n_singletons:,}"),
            ("multi-record clusters", f"{self.n_multi:,}"),
            ("max cluster size", f"{self.max_cluster_size:,}"),
            ("mean cluster size", _report_html._num(self.mean_cluster_size)),
            ("positive pairs", f"{self.positive_pairs:,}"),
            ("total pairs", f"{self.total_pairs:,}"),
            ("positive-pair prevalence", _fmt_g(self.prevalence)),
            ("class imbalance (pos:neg)", imbalance),
            ("cluster-size entropy (bits)", _report_html._num(self.entropy_bits)),
        ]

    # ------------------------------------------------------------ text surfaces
    def to_markdown(self) -> str:
        """Markdown: a metrics table plus the cluster-size distribution table."""
        lines = [f"## {self.title}", "", "| metric | value |", "|---|---|"]
        lines += [
            f"| {_report_html._md_cell(k)} | {_report_html._md_cell(v)} |"
            for k, v in self._metrics_kv()
        ]
        lines += [
            "",
            "### Cluster-size distribution",
            "",
            "| cluster size | clusters |",
            "|---|---|",
        ]
        if self.size_distribution:
            lines += [f"| {size} | {count} |" for size, count in self.size_distribution]
        else:
            lines.append("| _(none)_ | 0 |")
        return "\n".join(lines)

    @property
    def summary(self) -> dict[str, Any]:
        """Headline numbers as a flat, title-namespaced dict (collision-free per report)."""
        return {
            f"{self.title}.n_records": self.n_records,
            f"{self.title}.n_clusters": self.n_clusters,
            f"{self.title}.n_singletons": self.n_singletons,
            f"{self.title}.prevalence": self.prevalence,
            f"{self.title}.imbalance_ratio": self.imbalance_ratio,
            f"{self.title}.entropy_bits": self.entropy_bits,
        }

    def rows(self) -> list[dict[str, Any]]:
        """One row per distinct cluster size -- ``pd.DataFrame(section.rows())``-ready."""
        return [
            {"cluster_size": size, "n_clusters": count} for size, count in self.size_distribution
        ]

    # -------------------------------------------------------------- html panel
    def panels(self) -> list[str]:
        """A single ``<section>``: the metrics KV table above the size histogram."""
        kv = "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>"
            for k, v in self._metrics_kv()
        )
        edges, counts = _size_histogram(self.size_distribution)
        chart = _svg.bar_chart(
            edges,
            [("clusters", _COLOR_CLUSTERS, counts)],
            x_label="cluster size",
            y_label="count",
        )
        body = f'<table class="kv">{kv}</table>{chart}'
        return [_report_html.section(self.title, body)]


def profile_label_structure(
    clusters: Sequence[Collection[Hashable]] | None,
    *,
    n_records: int | None = None,
    title: str = "Label structure",
) -> LabelStructureSection | None:
    """Profile a gold clustering into a :class:`LabelStructureSection`.

    Args:
        clusters: The gold equivalence partition -- a sequence of clusters, each a
            collection of record ids. ``None`` means no gold is available, and the
            profiler returns ``None`` (the section is omitted). An empty sequence
            is a valid, degenerate clustering and renders (all-zero metrics).
        n_records: Optional total record count. Cross-source ER gold often lists
            only the matched clusters and omits singletons; passing the true
            record count folds the ``n_records - clustered`` unmatched records in
            as size-1 clusters, so prevalence and the imbalance ratio use the
            honest denominator. Ignored when smaller than the ids already covered
            by ``clusters`` (the clustering always wins -- we never report fewer
            records than it contains).
        title: Section heading; also the key the report looks it up by. Namespaces
            this section's :attr:`summary` keys, so distinct titles never collide.

    Returns:
        A :class:`LabelStructureSection`, or ``None`` when ``clusters is None``.
    """
    if clusters is None:
        return None

    sizes = [len(cluster) for cluster in clusters]
    clustered = sum(sizes)
    total_records = clustered if n_records is None else max(int(n_records), clustered)
    extra_singletons = total_records - clustered

    size_counts: Counter[int] = Counter(sizes)
    if extra_singletons:
        size_counts[1] += extra_singletons

    n_clusters = sum(size_counts.values())
    n_singletons = size_counts.get(1, 0)
    n_multi = sum(count for size, count in size_counts.items() if size >= 2)
    max_cluster_size = max(size_counts) if size_counts else 0
    mean_cluster_size = total_records / n_clusters if n_clusters else None

    positive_pairs = sum(math.comb(size, 2) * count for size, count in size_counts.items())
    total_pairs = math.comb(total_records, 2)
    prevalence = positive_pairs / total_pairs if total_pairs else None
    negatives = total_pairs - positive_pairs
    imbalance_ratio = negatives / positive_pairs if positive_pairs else None

    size_distribution = sorted(size_counts.items())
    entropy_bits = _shannon_entropy_bits(size_distribution, total_records)

    return LabelStructureSection(
        title=title,
        n_records=total_records,
        n_clusters=n_clusters,
        n_singletons=n_singletons,
        n_multi=n_multi,
        max_cluster_size=max_cluster_size,
        mean_cluster_size=mean_cluster_size,
        positive_pairs=positive_pairs,
        total_pairs=total_pairs,
        prevalence=prevalence,
        imbalance_ratio=imbalance_ratio,
        entropy_bits=entropy_bits,
        size_distribution=size_distribution,
    )


def _fmt_g(value: float | None) -> str:
    """Format a small ratio/probability with ``%g`` sig-figs; ``None``/NaN/Inf -> ``"n/a"``.

    Prevalence is often tiny (``3e-04`` under ER imbalance); ``%g`` keeps the
    significant digits that :func:`_report_html._num`'s fixed 3 decimals would
    round to ``0.000``.
    """
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:g}"
