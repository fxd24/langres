"""Separability profile section: the chart that justifies the whole tearsheet.

Every ER pipeline lives or dies on one question: does *some* signal separate the
matching pairs from the non-matching ones? This section answers it directly.
Given labeled positive and negative pairs and a :data:`SimilaritySignal` that
scores a pair of record ids, it builds the overlaid similarity histogram of the
two classes and a **separability AUC** -- the single number saying how cleanly
the signal ranks positives above negatives.

Two design choices make it honest under real ER class imbalance (positives are
often 1:1000+ of the negatives):

- the histogram is **density-normalized** (each class scaled to sum 1), so the
  tiny positive class stays legible instead of being flattened by the negatives;
- a **capped sample** per class keeps the scan bounded on a huge negative set,
  and any truncation is logged, never silent.

The A<->B bridge is the :data:`SimilaritySignal` alias: a plain callable over two
record ids. :func:`string_signal` is the default (rapidfuzz string similarity);
Unit B supplies a cosine variant of the *same shape*, so the profiler never
depends on which signal it is handed.
"""

from __future__ import annotations

import html
import logging
import math
import random
from collections.abc import Callable, Hashable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, Literal

from pydantic import BaseModel

from langres.core import _report_html, _svg
from langres.core.comparator import StringComparator
from langres.core.data_profile.base import ProfileSection
from langres.core.feature import combine_present

logger = logging.getLogger(__name__)

#: A pair-similarity signal: score two record ids, or ``None`` if unscorable
#: (an id is absent, or the pair shares no comparable evidence). This is the
#: A<->B bridge the separability profiler consumes -- :func:`string_signal` is the
#: default; a cosine-embedding variant of the same shape lives in Unit B.
SimilaritySignal = Callable[[Hashable, Hashable], float | None]

#: Seed for the deterministic per-class sample when capping (reproducible profiles).
_SAMPLE_SEED = 0

#: Positive / negative series colors (explicit for light+dark; the eval_report palette).
_COLOR_POSITIVE = "#2a9d8f"
_COLOR_NEGATIVE = "#e76f51"


class SeparabilitySection(ProfileSection):
    """Positives-vs-negatives similarity separation for one signal.

    A frozen :class:`ProfileSection` holding the two raw class histograms (density
    normalization happens at render time) and the separability AUC. Build it with
    :func:`profile_separability`.

    Attributes:
        signal_name: Human label of the scoring signal (e.g. ``"string"``).
        n_positive: Positive pairs that yielded a usable (finite, non-``None``)
            score.
        n_negative: Negative pairs that yielded a usable score.
        auc: Separability AUC (positives labeled ``True``); ``None`` when
            undefined (a class has no usable scores).
        hist_edges: Shared bin edges of the two histograms (``B + 1`` values).
        pos_counts: Raw positive-class counts per bin (``B`` values).
        neg_counts: Raw negative-class counts per bin.
        note: An honest one-line hint when the result is degenerate (a class is
            empty, or AUC is undefined); ``""`` when both classes scored.
    """

    kind: Literal["separability"] = "separability"

    signal_name: str
    n_positive: int
    n_negative: int
    auc: float | None
    hist_edges: list[float]
    pos_counts: list[float]
    neg_counts: list[float]
    note: str

    # ------------------------------------------------------------ text surfaces
    def to_markdown(self) -> str:
        """Markdown: the AUC + class counts, and the hint when degenerate."""
        lines = [
            f"## {self.title}",
            "",
            f"- separability AUC: {_report_html._num(self.auc)}",
            f"- positives scored: {self.n_positive:,}",
            f"- negatives scored: {self.n_negative:,}",
        ]
        if self.note:
            lines += ["", f"_{_report_html._md_cell(self.note)}_"]
        return "\n".join(lines)

    @property
    def summary(self) -> dict[str, Any]:
        """Headline numbers as a flat, title-namespaced dict."""
        return {
            f"{self.title}.auc": self.auc,
            f"{self.title}.n_positive": self.n_positive,
            f"{self.title}.n_negative": self.n_negative,
        }

    def rows(self) -> list[dict[str, Any]]:
        """One row per histogram bin -- the two class distributions side by side."""
        rows: list[dict[str, Any]] = []
        for i in range(len(self.pos_counts)):
            rows.append(
                {
                    "bin_lo": self.hist_edges[i],
                    "bin_hi": self.hist_edges[i + 1],
                    "positives": self.pos_counts[i],
                    "negatives": self.neg_counts[i],
                }
            )
        return rows

    # -------------------------------------------------------------- html panel
    def panels(self) -> list[str]:
        """A single ``<section>``: the density-overlaid histogram + AUC line + hint."""
        chart = _svg.bar_chart(
            self.hist_edges,
            [
                ("positives", _COLOR_POSITIVE, self.pos_counts),
                ("negatives", _COLOR_NEGATIVE, self.neg_counts),
            ],
            x_label=f"{self.signal_name} similarity",
            y_label="density",
            normalize="density",
        )
        auc_line = f"<p>separability AUC <b>{_report_html._num(self.auc)}</b></p>"
        note = f'<p class="empty">{html.escape(self.note)}</p>' if self.note else ""
        return [_report_html.section(self.title, chart + auc_line + note)]


def _edges(scores: Sequence[float], n_bins: int) -> list[float]:
    """``n_bins + 1`` ascending edges spanning the observed score range.

    A degenerate range (all scores equal) is widened to unit width so the bars
    have real pixel extent rather than collapsing onto one line.
    """
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        hi = lo + 1.0
    return [lo + (hi - lo) * i / n_bins for i in range(n_bins + 1)]


def _sampled_scores(
    pairs: Sequence[tuple[Hashable, Hashable]],
    signal: SimilaritySignal,
    cap: int | None,
    label: str,
) -> list[float]:
    """Score a (capped) sample of ``pairs``, dropping ``None`` / non-finite results.

    Sampling is a seeded, deterministic draw so a profile is reproducible; any
    truncation is logged. ``label`` names the class only for that log line.
    """
    sample: Sequence[tuple[Hashable, Hashable]] = pairs
    if cap is not None and len(pairs) > cap:
        logger.warning(
            "profile_separability: sampled %d of %d %s pairs (cap=%d)",
            cap,
            len(pairs),
            label,
            cap,
        )
        sample = random.Random(_SAMPLE_SEED).sample(list(pairs), cap)
    scores: list[float] = []
    for left, right in sample:
        value = signal(left, right)
        if value is not None and math.isfinite(value):
            scores.append(float(value))
    return scores


def profile_separability(
    positive_pairs: Sequence[tuple[Hashable, Hashable]],
    negative_pairs: Sequence[tuple[Hashable, Hashable]],
    signal: SimilaritySignal,
    *,
    name: str,
    n_bins: int = 20,
    cap: int | None = None,
) -> SeparabilitySection | None:
    """Profile how well ``signal`` separates positive from negative pairs.

    Args:
        positive_pairs: Matching id-pairs (the positive class).
        negative_pairs: Non-matching id-pairs (the negative class).
        signal: The :data:`SimilaritySignal` to score each pair with. ``None`` /
            non-finite scores are dropped as unscorable.
        name: Human label of the signal (e.g. ``"string"``), shown on the chart
            axis and used to build the section title.
        n_bins: Number of histogram bins over the observed score range.
        cap: Optional per-class sample cap. When a class has more than ``cap``
            pairs, a seeded sample of ``cap`` is scored and the truncation is
            logged (never silent). ``None`` scores every pair.

    Returns:
        A :class:`SeparabilitySection`, or ``None`` when *no* pair in either class
        produced a usable score (nothing to plot). A one-sided result (only one
        class scored) is kept with an honest ``note`` and an ``n/a`` AUC rather
        than dropped.
    """
    pos_scores = _sampled_scores(positive_pairs, signal, cap, "positive")
    neg_scores = _sampled_scores(negative_pairs, signal, cap, "negative")

    if not pos_scores and not neg_scores:
        return None

    edges = _edges(pos_scores + neg_scores, n_bins)
    pos_counts = _report_html._histogram(pos_scores, edges)
    neg_counts = _report_html._histogram(neg_scores, edges)

    auc = _report_html.safe_auc(
        [True] * len(pos_scores) + [False] * len(neg_scores),
        pos_scores + neg_scores,
    )
    # safe_auc returns the underlying metric's NaN for a single-class vector;
    # normalize that to None so no panel/summary ever carries a NaN.
    if auc is not None and not math.isfinite(auc):
        auc = None

    note = _degenerate_note(pos_scores, neg_scores)

    return SeparabilitySection(
        title=f"Separability ({name})",
        signal_name=name,
        n_positive=len(pos_scores),
        n_negative=len(neg_scores),
        auc=auc,
        hist_edges=edges,
        pos_counts=pos_counts,
        neg_counts=neg_counts,
        note=note,
    )


def _degenerate_note(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> str:
    """Honest one-line hint for a one-sided result (``""`` when both classes scored).

    Reached only after the both-empty case has already returned ``None``, so
    exactly one of these branches fires for a one-sided result; with both classes
    present the AUC is always defined (never single-class), so no hint is needed.
    """
    if not neg_scores:
        return "No usable negative-pair scores: AUC is undefined; only the positive class is shown."
    if not pos_scores:
        return "No usable positive-pair scores: AUC is undefined; only the negative class is shown."
    return ""


def string_signal(
    records: Mapping[Hashable, Mapping[str, Any]],
    schema: type[BaseModel],
) -> SimilaritySignal:
    """Build the default rapidfuzz string-similarity :data:`SimilaritySignal`.

    Closes over a :class:`~langres.core.comparator.StringComparator` derived from
    ``schema`` and the :func:`~langres.core.feature.combine_present` evidence
    floor: the returned callable looks each id up in ``records``, compares the two
    field dicts feature-by-feature, and combines the present-feature similarities
    into one score.

    Args:
        records: ``id -> field mapping`` for every record the pairs reference.
        schema: The Pydantic entity schema; its ``str | None`` fields become the
            comparable features (via
            :meth:`StringComparator.from_schema <langres.core.comparator.StringComparator.from_schema>`).

    Returns:
        A :data:`SimilaritySignal` returning the combined similarity, or ``None``
        when either id is absent or the pair shares no comparable (present)
        feature.
    """
    # ``StringComparator[Any]``: ``from_schema``'s TypeVar is unbound at this call
    # site, and ``compare`` is fed attribute-shim objects (``SimpleNamespace``),
    # not the schema type -- ``Any`` keeps the missing-aware ``getattr`` path honest
    # without forcing a concrete model on the shim.
    comparator: StringComparator[Any] = StringComparator.from_schema(schema)
    weights = {spec.name: spec.weight for spec in comparator.feature_specs}

    def signal(left_id: Hashable, right_id: Hashable) -> float | None:
        left = records.get(left_id)
        right = records.get(right_id)
        if left is None or right is None:
            return None
        vector = comparator.compare(SimpleNamespace(**left), SimpleNamespace(**right))
        if not vector.similarities:
            # No feature present on both sides -> the pair is unscorable, not 0.
            return None
        return combine_present(vector.similarities, weights)

    return signal
