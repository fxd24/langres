"""``EvalReport``: a $0 evaluation tearsheet computed from persisted judgements.

Judging is expensive; *analysing* persisted judgements is free. Once a judge has
scored a candidate set (paying for LLM calls, embeddings, whatever), every number
below -- pair precision/recall/F1, the PR and ROC curves, the score distribution,
the confidence-calibration diagram, the most-confident errors -- is recoverable
at **zero additional cost** from the logged rows plus the gold pairs. This
generalises what ``peeters_llm_em_replication.py --report-only`` already does for
one dataset: recompute the full table from committed JSONL at $0.

Two constructors, both free:

- :meth:`EvalReport.from_log` -- the default mental model. Consumes
  :meth:`~langres.core.judgement_log.JudgementLog.read` output (plain dicts) and
  the gold pairs (gold is never in the log). No judge, no candidates, no API.
- :meth:`EvalReport.from_judgements` -- the in-process path: you already hold the
  :class:`~langres.core.models.PairwiseJudgement` list, so pass it straight in.

**Layering.** This is a *leaf* module. It imports from ``core.metrics``,
``core.benchmark``, ``core.models`` and ``core._svg`` -- all import-light -- and
**nothing in ``reports.py`` / ``module.py`` may import it**, which would reverse
the dependency arrow and pull SVG/HTML into a bare ``import langres``. A dedicated
import-budget test locks that this module never drags in a heavy dependency
(torch/litellm/faiss/sklearn/…): the tearsheet is dependency-free by construction.
"""

from __future__ import annotations

import html
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from langres.core import _svg
from langres.core.benchmark import DEFAULT_PAIR_GRID, PairTrack
from langres.core.metrics import (
    PairMetrics,
    ReliabilityBin,
    average_precision_score,
    brier_score,
    classify_pairs,
    expected_calibration_error,
    pair_pr_curve,
    reliability_bins,
    roc_auc_score,
)
from langres.core.models import PairwiseJudgement, predicted_match

# Panel series colors -- explicit (not currentColor) so they render in light AND
# dark themes. Chosen for contrast against a neutral background either way.
_COLOR_CURVE = "#4361ee"  # PR / ROC line
_COLOR_GOLD = "#2a9d8f"  # gold (true-match) distribution
_COLOR_NONGOLD = "#e76f51"  # non-match distribution
_COLOR_RELIABILITY = "#4361ee"


def _pair_key(left_id: str, right_id: str) -> frozenset[str]:
    """Order-independent identity of a candidate pair."""
    return frozenset({left_id, right_id})


def _num(value: float | None, digits: int = 3) -> str:
    """Format a number for display, mapping ``None``/NaN/Inf to ``"n/a"``.

    A blank or ``"n/a"`` cell is the honest render of an undefined metric (a
    single-class ROC-AUC, calibration with no confidence signal); it must never
    surface as the literal ``NaN``/``Infinity`` string in the HTML.
    """
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _histogram(values: Sequence[float], edges: Sequence[float]) -> list[float]:
    """Count ``values`` into the half-open bins defined by ascending ``edges``.

    ``len(edges) == B + 1`` yields ``B`` counts. The last bin is closed on the
    right so a value equal to the final edge (e.g. ``1.0``) is counted, not
    dropped. Non-finite values are ignored.
    """
    n_bins = len(edges) - 1
    counts = [0.0] * max(n_bins, 0)
    for value in values:
        if not math.isfinite(value):
            continue
        for b in range(n_bins):
            lo = edges[b]
            hi = edges[b + 1]
            if lo <= value < hi or (b == n_bins - 1 and value == hi):
                counts[b] += 1.0
                break
    return counts


def _roc_curve(labels: Sequence[bool], scores: Sequence[float]) -> list[tuple[float, float]]:
    """ROC-curve points ``(fpr, tpr)`` from ``(0,0)`` to ``(1,1)``.

    Ties are handled by advancing through every equal score before recording a
    point, so a run of tied scores contributes a single diagonal step rather
    than a staircase that would misstate the curve. Returns ``[]`` when the
    labels are single-class (the curve, like AUC, is then undefined).
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    n_pos = sum(1 for label in labels if label)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return []
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    true_pos = 0
    false_pos = 0
    i = 0
    m = len(order)
    while i < m:
        s = scores[order[i]]
        while i < m and scores[order[i]] == s:
            if labels[order[i]]:
                true_pos += 1
            else:
                false_pos += 1
            i += 1
        points.append((false_pos / n_neg, true_pos / n_pos))
    return points


def _judgement_from_row(row: Mapping[str, Any]) -> PairwiseJudgement:
    """Rebuild a :class:`PairwiseJudgement` from one ``JudgementLog.read()`` row.

    ``decision`` is read directly (v3 rows carry it; :meth:`JudgementLog.read`
    backfills it from ``verdict`` for legacy v1/v2 rows), falling back to
    ``verdict`` only when the ``decision`` key is entirely absent -- so a genuine
    abstention (``decision`` present and ``None``) stays an abstention and is
    never coerced to a verdict.
    """
    decision = row["decision"] if "decision" in row else row.get("verdict")
    score = row.get("score")
    return PairwiseJudgement(
        left_id=str(row["left_id"]),
        right_id=str(row["right_id"]),
        decision=decision,
        score=None if score is None else float(score),
        score_type=row.get("score_type", "prob_llm"),
        confidence=row.get("confidence"),
        confidence_source=row.get("confidence_source", "none"),
        decision_step=str(row.get("decision_step", "log")),
        provenance={},
    )


class EvalError(BaseModel):
    """One most-confident classification error, for the tearsheet's error table.

    An *error* is a judged pair whose prediction disagrees with gold: a predicted
    match that is not gold (false positive) or a predicted non-match that is gold
    (false negative). Abstentions are not errors (no prediction was made). Only
    *judged* pairs appear -- a gold pair the blocker never surfaced is a false
    negative in the counts but has no judgement to show here.

    Attributes:
        left_id: Left entity id.
        right_id: Right entity id.
        predicted: The judge's match prediction at the report threshold.
        is_gold: Whether the pair is a true match.
        score: The judge's score, if it ranked (else ``None``).
        confidence: The judge's self-reported confidence, if any (else ``None``).
    """

    model_config = ConfigDict(frozen=True)

    left_id: str
    right_id: str
    predicted: bool
    is_gold: bool
    score: float | None
    confidence: float | None


class EvalReport(BaseModel):
    """A frozen, $0 evaluation tearsheet over a set of judged pairs.

    Every field is derived from the judgements plus the gold pairs -- no judge,
    no API, no re-run. Build it with :meth:`from_log` (from persisted rows) or
    :meth:`from_judgements` (in-process), then read :attr:`summary`, call
    :meth:`to_markdown` / :meth:`to_dict`, or render :meth:`to_html` for a
    self-contained visual report.

    Attributes:
        threshold: The match threshold every pair-level number is graded at.
        n_candidates: Number of judged candidate pairs.
        n_gold: Number of gold (true-match) pairs.
        n_ranked: Judgements carrying a usable score (rankers). Deciders that
            emit only a decision contribute 0 here, so the ROC/PR/histogram
            panels -- which need a continuous signal -- are empty for a pure
            binary log.
        n_abstained: Judgements that neither decided nor scored.
        tp, fp, fn, tn: Confusion counts at :attr:`threshold`. ``tp``/``fp``/``fn``
            match :func:`~langres.core.metrics.classify_pairs` exactly (``fn``
            includes gold pairs the blocker never surfaced); ``tn`` counts judged
            non-gold pairs the judge explicitly predicted non-match (abstentions
            are excluded from ``tn`` -- no verdict is not a correct negative).
        pair: Pair-level precision/recall/F1 at :attr:`threshold` plus the PR
            curve across :data:`~langres.core.benchmark.DEFAULT_PAIR_GRID`.
        roc_curve: ``(fpr, tpr)`` points; ``[]`` when the ranking signal is
            single-class or absent.
        roc_auc: ROC-AUC of the ranking signal vs gold; ``nan`` when undefined.
        average_precision: AP of the ranking signal vs gold; ``nan`` when
            undefined.
        hist_edges, hist_gold, hist_nongold: Score histogram of the ranking
            signal split by gold membership (``hist_edges`` has one more entry
            than each count list).
        reliability: Per-bin calibration points of ``confidence`` vs the judge's
            own correctness; empty when no judgement carried a confidence.
        brier, ece: Calibration scores over the confidence signal; ``None`` when
            no confidence was present.
        n_with_confidence: Judged, non-abstained pairs carrying a confidence.
        confidence_source_counts: Count of judgements per ``confidence_source``.
        total_cost_usd: Summed logged cost of producing these judgements.
        top_errors: The most-confident errors, most-confident first.
    """

    model_config = ConfigDict(frozen=True)

    threshold: float
    n_candidates: int
    n_gold: int
    n_ranked: int
    n_abstained: int
    tp: int
    fp: int
    fn: int
    tn: int
    pair: PairTrack
    roc_curve: list[tuple[float, float]]
    roc_auc: float
    average_precision: float
    hist_edges: list[float]
    hist_gold: list[float]
    hist_nongold: list[float]
    reliability: list[ReliabilityBin]
    brier: float | None
    ece: float | None
    n_with_confidence: int
    confidence_source_counts: dict[str, int]
    total_cost_usd: float
    top_errors: list[EvalError]

    # ---------------------------------------------------------------- constructors
    @classmethod
    def from_judgements(
        cls,
        judgements: Sequence[PairwiseJudgement],
        gold_pairs: set[frozenset[str]],
        *,
        threshold: float = 0.5,
        grid: Sequence[float] = DEFAULT_PAIR_GRID,
        hist_bins: int = 10,
        top_n: int = 10,
        costs: Sequence[float] | None = None,
    ) -> EvalReport:
        """Build a report from an in-process judgement list plus gold pairs.

        Args:
            judgements: The scorer's output (pre-clustering).
            gold_pairs: True match pairs as order-independent ``frozenset`` pairs.
            threshold: Match cut every pair-level number is graded at.
            grid: Threshold grid for the PR curve.
            hist_bins: Number of score-histogram bins over ``[0, 1]``.
            top_n: How many most-confident errors to keep.
            costs: Optional per-judgement cost (aligned with ``judgements``); the
                sum becomes :attr:`total_cost_usd`. ``None`` -> ``0.0``.

        Returns:
            A frozen :class:`EvalReport`.
        """
        judged = list(judgements)
        n_candidates = len(judged)

        base = classify_pairs(judged, gold_pairs, threshold)
        pr_curve: list[PairMetrics] = pair_pr_curve(judged, gold_pairs, grid)

        # tn / abstain: walk once, classifying each judged pair. tp/fp/fn come
        # from classify_pairs (the tested primitive); tn is the judged non-gold
        # pairs the judge explicitly said "no" to -- abstentions are neither.
        tn = 0
        n_abstained = 0
        ranked_scores: list[float] = []
        ranked_labels: list[bool] = []
        conf_values: list[float] = []
        conf_outcomes: list[bool] = []
        errors: list[tuple[float, EvalError]] = []
        source_counter: Counter[str] = Counter()

        for judgement in judged:
            source_counter[judgement.confidence_source] += 1
            is_gold = _pair_key(judgement.left_id, judgement.right_id) in gold_pairs
            predicted = predicted_match(judgement, threshold)

            if predicted is None:
                n_abstained += 1
            elif predicted is False and not is_gold:
                tn += 1

            if judgement.score is not None:
                ranked_scores.append(judgement.score)
                ranked_labels.append(is_gold)

            if judgement.confidence is not None and predicted is not None:
                conf_values.append(judgement.confidence)
                conf_outcomes.append(predicted == is_gold)

            if predicted is not None and predicted != is_gold:
                # "how confident in the wrong answer": a self-reported confidence
                # if present, else distance of the score past the threshold.
                if judgement.confidence is not None:
                    sort_key = judgement.confidence
                elif judgement.score is not None:
                    sort_key = abs(judgement.score - threshold)
                else:
                    sort_key = 0.0
                errors.append(
                    (
                        sort_key,
                        EvalError(
                            left_id=judgement.left_id,
                            right_id=judgement.right_id,
                            predicted=predicted,
                            is_gold=is_gold,
                            score=judgement.score,
                            confidence=judgement.confidence,
                        ),
                    )
                )

        roc_auc = _safe_auc(ranked_labels, ranked_scores)
        avg_precision = _safe_ap(ranked_labels, ranked_scores)
        roc_curve = _roc_curve(ranked_labels, ranked_scores)

        edges = [i / hist_bins for i in range(hist_bins + 1)]
        hist_gold = _histogram(
            [s for s, g in zip(ranked_scores, ranked_labels, strict=True) if g], edges
        )
        hist_nongold = _histogram(
            [s for s, g in zip(ranked_scores, ranked_labels, strict=True) if not g], edges
        )

        if conf_values:
            reliability = reliability_bins(conf_values, conf_outcomes)
            brier: float | None = brier_score(conf_values, conf_outcomes)
            ece: float | None = expected_calibration_error(conf_values, conf_outcomes)
        else:
            reliability = []
            brier = None
            ece = None

        errors.sort(key=lambda entry: entry[0], reverse=True)
        top_errors = [error for _, error in errors[:top_n]]

        return cls(
            threshold=threshold,
            n_candidates=n_candidates,
            n_gold=len(gold_pairs),
            n_ranked=len(ranked_scores),
            n_abstained=n_abstained,
            tp=base.tp,
            fp=base.fp,
            fn=base.fn,
            tn=tn,
            pair=PairTrack(
                precision=base.precision,
                recall=base.recall,
                f1=base.f1,
                pr_curve=pr_curve,
            ),
            roc_curve=roc_curve,
            roc_auc=roc_auc,
            average_precision=avg_precision,
            hist_edges=edges,
            hist_gold=hist_gold,
            hist_nongold=hist_nongold,
            reliability=reliability,
            brier=brier,
            ece=ece,
            n_with_confidence=len(conf_values),
            confidence_source_counts=dict(source_counter),
            total_cost_usd=float(sum(costs)) if costs else 0.0,
            top_errors=top_errors,
        )

    @classmethod
    def from_log(
        cls,
        rows: Sequence[Mapping[str, Any]],
        gold_pairs: set[frozenset[str]],
        *,
        threshold: float = 0.5,
        grid: Sequence[float] = DEFAULT_PAIR_GRID,
        hist_bins: int = 10,
        top_n: int = 10,
    ) -> EvalReport:
        """Build a report from persisted ``JudgementLog.read()`` rows plus gold.

        The default mental model: judging already happened and was logged; this
        recomputes the whole tearsheet at $0. Per-row ``cost_usd`` (falling back
        to ``llm_cost_usd``, the cascade key) is summed into
        :attr:`total_cost_usd`.

        Args:
            rows: Rows as produced by
                :meth:`~langres.core.judgement_log.JudgementLog.read`.
            gold_pairs: True match pairs (never present in the log).
            threshold: Match cut every pair-level number is graded at.
            grid: Threshold grid for the PR curve.
            hist_bins: Number of score-histogram bins over ``[0, 1]``.
            top_n: How many most-confident errors to keep.

        Returns:
            A frozen :class:`EvalReport`.
        """
        judgements = [_judgement_from_row(row) for row in rows]
        costs = [float(row.get("cost_usd") or row.get("llm_cost_usd") or 0.0) for row in rows]
        return cls.from_judgements(
            judgements,
            gold_pairs,
            threshold=threshold,
            grid=grid,
            hist_bins=hist_bins,
            top_n=top_n,
            costs=costs,
        )

    # -------------------------------------------------------------------- summary
    @property
    def summary(self) -> str:
        """A one-line headline of the report's most important numbers."""
        return (
            f"{self.n_candidates} pairs @ threshold {self.threshold:g}: "
            f"P={_num(self.pair.precision)} R={_num(self.pair.recall)} "
            f"F1={_num(self.pair.f1)} | ROC-AUC={_num(self.roc_auc)} "
            f"AP={_num(self.average_precision)} | "
            f"Brier={_num(self.brier)} ECE={_num(self.ece)}"
        )

    def to_dict(self) -> dict[str, Any]:
        """The report as a plain dict (``model_dump()``)."""
        return self.model_dump()

    def to_markdown(self) -> str:
        """Render the report as a Markdown document."""
        lines: list[str] = [
            "# Evaluation report",
            "",
            self.summary,
            "",
            "## Counts",
            "",
            f"- candidates judged: {self.n_candidates}",
            f"- gold pairs: {self.n_gold}",
            f"- ranked (with a score): {self.n_ranked}",
            f"- abstained: {self.n_abstained}",
            f"- with confidence: {self.n_with_confidence}",
            f"- total cost: ${self.total_cost_usd:.4f}",
            "",
            f"## Confusion @ threshold {self.threshold:g}",
            "",
            "| | predicted match | predicted non-match |",
            "|---|---|---|",
            f"| **gold match** | {self.tp} (tp) | {self.fn} (fn) |",
            f"| **gold non-match** | {self.fp} (fp) | {self.tn} (tn) |",
            "",
            f"- precision {_num(self.pair.precision)} | "
            f"recall {_num(self.pair.recall)} | f1 {_num(self.pair.f1)}",
            f"- ROC-AUC {_num(self.roc_auc)} | average precision {_num(self.average_precision)}",
        ]
        if self.reliability:
            lines += [
                "",
                "## Calibration",
                "",
                f"- Brier {_num(self.brier)} | ECE {_num(self.ece)} "
                f"(over {self.n_with_confidence} confidence-bearing judgements)",
            ]
        if self.top_errors:
            lines += [
                "",
                "## Most-confident errors",
                "",
                "| left | right | predicted | gold | score | confidence |",
                "|---|---|---|---|---|---|",
            ]
            for error in self.top_errors:
                kind = "match" if error.predicted else "non-match"
                gold = "match" if error.is_gold else "non-match"
                lines.append(
                    f"| {error.left_id} | {error.right_id} | {kind} | {gold} | "
                    f"{_num(error.score)} | {_num(error.confidence)} |"
                )
        return "\n".join(lines)

    # ----------------------------------------------------------------------- html
    def to_html(self, *, title: str = "Evaluation report") -> str:
        """Render a single self-contained HTML document (no external assets).

        The whole tearsheet -- styles inline, every chart an inline ``<svg>`` from
        :mod:`langres.core._svg` -- fits in one string with zero network requests,
        no CDN, no matplotlib. Themes: neutral ``currentColor`` axes plus a
        ``prefers-color-scheme`` block, so it reads in light and dark.
        """
        panels: list[str] = [self._panel_summary(), self._panel_confusion()]
        panels.append(self._panel_pr())
        panels.append(self._panel_roc())
        panels.append(self._panel_histogram())
        panels.append(self._panel_reliability())
        panels.append(self._panel_errors())
        body = "\n".join(panels)
        return (
            "<!doctype html>\n"
            '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{html.escape(title)}</title>\n"
            f"<style>{_CSS}</style>\n</head>\n<body>\n"
            f"<h1>{html.escape(title)}</h1>\n"
            f'<p class="summary">{html.escape(self.summary)}</p>\n'
            f"{body}\n"
            "</body>\n</html>\n"
        )

    # ---- panels -----------------------------------------------------------
    def _panel_summary(self) -> str:
        rows = [
            ("candidates judged", str(self.n_candidates)),
            ("gold pairs", str(self.n_gold)),
            ("ranked (score)", str(self.n_ranked)),
            ("abstained", str(self.n_abstained)),
            ("with confidence", str(self.n_with_confidence)),
            ("total cost", f"${self.total_cost_usd:.4f}"),
        ]
        cells = "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in rows
        )
        return f'<section><h2>Counts</h2><table class="kv">{cells}</table></section>'

    def _panel_confusion(self) -> str:
        return (
            f"<section><h2>Confusion @ threshold {html.escape(f'{self.threshold:g}')}</h2>"
            '<table class="confusion">'
            "<tr><th></th><th>predicted match</th><th>predicted non-match</th></tr>"
            f'<tr><th>gold match</th><td class="tp">{self.tp}</td>'
            f'<td class="fn">{self.fn}</td></tr>'
            f'<tr><th>gold non-match</th><td class="fp">{self.fp}</td>'
            f'<td class="tn">{self.tn}</td></tr>'
            "</table>"
            f"<p>precision <b>{_num(self.pair.precision)}</b> &middot; "
            f"recall <b>{_num(self.pair.recall)}</b> &middot; "
            f"F1 <b>{_num(self.pair.f1)}</b></p></section>"
        )

    def _panel_pr(self) -> str:
        points = [(m.recall, m.precision) for m in (self.pair.pr_curve or [])]
        svg = _svg.line_chart(
            [_svg.Series(points=points, stroke=_COLOR_CURVE, label="")],
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
            x_label="recall",
            y_label="precision",
            annotations=[f"AP = {_num(self.average_precision)}"],
        )
        return f"<section><h2>Precision–Recall</h2>{svg}</section>"

    def _panel_roc(self) -> str:
        svg = _svg.line_chart(
            [_svg.Series(points=list(self.roc_curve), stroke=_COLOR_CURVE, label="")],
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
            x_label="false positive rate",
            y_label="true positive rate",
            diagonal=True,
            annotations=[f"AUC = {_num(self.roc_auc)}"],
        )
        note = (
            ""
            if self.roc_curve
            else '<p class="empty">No ranking signal (a pure decider log has no '
            "continuous score to trace a ROC curve).</p>"
        )
        return f"<section><h2>ROC</h2>{svg}{note}</section>"

    def _panel_histogram(self) -> str:
        svg = _svg.bar_chart(
            self.hist_edges,
            [
                ("gold", _COLOR_GOLD, self.hist_gold),
                ("non-gold", _COLOR_NONGOLD, self.hist_nongold),
            ],
            x_label="score",
            y_label="count",
        )
        return f"<section><h2>Score distribution</h2>{svg}</section>"

    def _panel_reliability(self) -> str:
        if not self.reliability:
            return (
                "<section><h2>Calibration</h2>"
                '<p class="empty">No confidence signal — run '
                'LLMJudge(confidence="logprob") to populate this panel.</p></section>'
            )
        points = [(b.mean_confidence, b.observed_frequency) for b in self.reliability]
        svg = _svg.line_chart(
            [_svg.Series(points=points, stroke=_COLOR_RELIABILITY, label="", kind="markers")],
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
            x_label="mean confidence",
            y_label="observed accuracy",
            diagonal=True,
            annotations=[f"Brier = {_num(self.brier)}", f"ECE = {_num(self.ece)}"],
        )
        return (
            f"<section><h2>Calibration</h2>{svg}"
            f"<p>over {self.n_with_confidence} confidence-bearing judgements</p></section>"
        )

    def _panel_errors(self) -> str:
        if not self.top_errors:
            return "<section><h2>Most-confident errors</h2><p>None.</p></section>"
        head = (
            "<tr><th>left</th><th>right</th><th>predicted</th>"
            "<th>gold</th><th>score</th><th>confidence</th></tr>"
        )
        body_rows = []
        for error in self.top_errors:
            kind = "match" if error.predicted else "non-match"
            gold = "match" if error.is_gold else "non-match"
            body_rows.append(
                f"<tr><td>{html.escape(error.left_id)}</td>"
                f"<td>{html.escape(error.right_id)}</td>"
                f"<td>{kind}</td><td>{gold}</td>"
                f"<td>{_num(error.score)}</td><td>{_num(error.confidence)}</td></tr>"
            )
        return (
            "<section><h2>Most-confident errors</h2>"
            f'<table class="errors">{head}{"".join(body_rows)}</table></section>'
        )


def _safe_auc(labels: list[bool], scores: list[float]) -> float:
    """ROC-AUC that returns ``nan`` (not a raise) on an empty/degenerate input."""
    if not scores:
        return float("nan")
    return roc_auc_score(labels, scores)


def _safe_ap(labels: list[bool], scores: list[float]) -> float:
    """Average precision that returns ``nan`` (not a raise) on empty input."""
    if not scores:
        return float("nan")
    return average_precision_score(labels, scores)


# Inline stylesheet: neutral, theme-aware, no external fonts or assets.
_CSS = """
:root { color-scheme: light dark; --fg: #1a1a1a; --bg: #ffffff; --muted: #666;
  --line: #ddd; --tp: #2a9d8f; --fp: #e76f51; --fn: #e9c46a; --tn: #8ecae6; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e6e6e6; --bg: #16181d; --muted: #9aa; --line: #333; } }
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  color: var(--fg); background: var(--bg); margin: 0 auto; max-width: 820px;
  padding: 24px; line-height: 1.5; }
h1 { font-size: 1.5rem; margin: 0 0 4px; }
h2 { font-size: 1.05rem; margin: 0 0 8px; border-bottom: 1px solid var(--line);
  padding-bottom: 4px; }
section { margin: 24px 0; }
.summary { color: var(--muted); font-variant-numeric: tabular-nums; margin: 0 0 8px; }
.empty { color: var(--muted); font-style: italic; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 10px; text-align: left; border: 1px solid var(--line); }
table.kv th { color: var(--muted); font-weight: 500; }
table.confusion td { text-align: right; font-weight: 600; }
table.confusion td.tp { color: var(--tp); } table.confusion td.fp { color: var(--fp); }
table.confusion td.fn { color: var(--fn); } table.confusion td.tn { color: var(--tn); }
table.errors { width: 100%; font-size: 0.9rem; }
svg { display: block; margin: 4px 0; }
""".strip()
