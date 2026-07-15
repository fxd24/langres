"""Shared, dependency-free HTML/render scaffold for the ``$0`` report family.

This is the leaf both :mod:`langres.core.eval_report` and the data-profile
report (:mod:`langres.core.data_profile`) render through. It carries the same
guarantee ``eval_report`` already proves: inline styles, no CDN, no matplotlib,
no ML stack -- a self-contained HTML page buildable on a bare core-only install.
An import-budget test (``tests/test_import_budget.py``) locks that this module
never drags in a heavy dependency.

**Why this exists (and why it duplicates ``eval_report`` for now).**
``EvalReport`` keeps its ``_CSS`` + panel idiom *inside* the class. Rather than
churn that shipped module, this arc *lifts* the reusable pieces (the stylesheet,
the number/markdown formatters, the histogram binner, the never-raise AUC/AP
guards) here so the profile report can share them. ``eval_report`` is left
untouched (a later, optional migration -- surgical-changes rule), so the small
overlap is a deliberate, transient duplicate.

Contents:
- :func:`document` -- the doctype + ``<head>`` + ``<body>`` shell (mirrors
  ``EvalReport.to_html``'s wrapper).
- :func:`section` -- one ``<section><h2>...</h2>...</section>`` block.
- :func:`_num` -- format a number, mapping ``None``/NaN/Inf to ``"n/a"``.
- :func:`_md_cell` -- make a value safe inside a Markdown table cell.
- :func:`_histogram` -- count values into ascending half-open bins.
- :func:`safe_auc` / :func:`safe_ap` -- ROC-AUC / average-precision guards that
  return ``None`` (never raise) on empty input, dropping non-finite
  ``(score, label)`` pairs before calling the underlying metric.

Pure standard library plus (optionally) numpy; ``core.metrics`` is the only
langres import, and it is itself import-light.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence

from langres.core.metrics import average_precision_score, roc_auc_score

# Explicit shared surface. ``_num`` / ``_md_cell`` / ``_histogram`` are
# underscore-named (module-internal by convention) but are *intentionally*
# provided for the profiler sections that land in Wave 1 -- listing them here
# documents that they are exported-for-reuse, not dead code.
__all__ = [
    "document",
    "section",
    "safe_auc",
    "safe_ap",
    "_num",
    "_md_cell",
    "_histogram",
]

# Inline stylesheet: neutral, theme-aware, no external fonts or assets. Copied
# verbatim from ``eval_report._CSS`` so both reports look identical; the two
# copies converge if ``eval_report`` is later migrated onto this scaffold.
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


def document(title: str, body: str, *, summary_html: str = "") -> str:
    """Wrap ``body`` in a self-contained HTML document (no external assets).

    Mirrors the shell ``EvalReport.to_html`` builds: a ``<!doctype html>`` page
    with the shared inline :data:`_CSS`, an escaped ``<h1>`` title, an optional
    summary line, and the caller-supplied ``body`` (already-rendered
    ``<section>`` HTML). Zero network requests, no CDN, no matplotlib.

    Args:
        title: Page title -- HTML-escaped into both ``<title>`` and ``<h1>``.
        body: Pre-rendered HTML for the page body (typically joined
            ``<section>`` blocks). Emitted as-is; callers are responsible for
            escaping any untrusted text inside it (use :func:`section`).
        summary_html: Optional pre-rendered HTML for a one-line summary under
            the title. Emitted as-is (hence the ``_html`` suffix); pass ``""``
            (default) to omit the summary paragraph entirely.

    Returns:
        A complete ``<!doctype html>...`` string.
    """
    summary = f'<p class="summary">{summary_html}</p>\n' if summary_html else ""
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n"
        f"<h1>{html.escape(title)}</h1>\n"
        f"{summary}"
        f"{body}\n"
        "</body>\n</html>\n"
    )


def section(title: str, body_html: str) -> str:
    """Render one ``<section><h2>title</h2>body_html</section>`` block.

    The ``title`` is HTML-escaped; ``body_html`` is emitted as-is (it is
    pre-rendered HTML -- a table, an inline ``<svg>``, a paragraph). This is the
    single place the ``<section>`` idiom lives, so every profiler section renders
    a consistent frame.

    Args:
        title: Section heading -- HTML-escaped.
        body_html: Pre-rendered HTML for the section body (emitted verbatim).

    Returns:
        A ``<section>...</section>`` string.
    """
    return f"<section><h2>{html.escape(title)}</h2>{body_html}</section>"


def _num(value: float | None, digits: int = 3) -> str:
    """Format a number for display, mapping ``None``/NaN/Inf to ``"n/a"``.

    A blank or ``"n/a"`` cell is the honest render of an undefined metric; it
    must never surface as the literal ``NaN``/``Infinity`` string in the HTML.
    """
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _md_cell(text: str) -> str:
    """Make a value safe inside a Markdown table cell.

    A value may contain a ``|`` (the column delimiter) or a newline; either
    silently corrupts table alignment. Escape the pipe and flatten newlines so
    the Markdown and HTML render paths stay symmetric.
    """
    return str(text).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


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


def safe_auc(y_true: Sequence[bool], scores: Sequence[float]) -> float | None:
    """ROC-AUC that never raises: ``None`` on empty, non-finite scores dropped.

    :func:`~langres.core.metrics.roc_auc_score` *raises* on an empty input or a
    non-finite score (a broken ranking). This guard drops every
    ``(score, label)`` pair whose score is non-finite *before* calling it, and
    returns ``None`` when nothing usable remains -- so a section can always
    compute a headline number without a try/except at every call site.

    A single-class label vector (all positive or all negative) still yields the
    underlying metric's documented ``nan`` "undefined statistic" return, which
    :func:`_num` renders as ``"n/a"``.

    Args:
        y_true: Ground-truth boolean labels (positive class is ``True``).
        scores: Continuous scores aligned with ``y_true``.

    Returns:
        The ROC-AUC over the finite-scored pairs, or ``None`` when none remain.
    """
    labels, clean = _finite_pairs(y_true, scores)
    if not clean:
        return None
    return roc_auc_score(labels, clean)


def safe_ap(y_true: Sequence[bool], scores: Sequence[float]) -> float | None:
    """Average precision that never raises: ``None`` on empty, non-finite dropped.

    The AP counterpart of :func:`safe_auc` -- same never-raise contract over
    :func:`~langres.core.metrics.average_precision_score`.

    Args:
        y_true: Ground-truth boolean labels (positive class is ``True``).
        scores: Continuous scores aligned with ``y_true``.

    Returns:
        The average precision over the finite-scored pairs, or ``None`` when
        none remain.
    """
    labels, clean = _finite_pairs(y_true, scores)
    if not clean:
        return None
    return average_precision_score(labels, clean)


def _finite_pairs(
    y_true: Sequence[bool], scores: Sequence[float]
) -> tuple[list[bool], list[float]]:
    """Drop ``(label, score)`` pairs whose score is non-finite; return the rest.

    The shared filter behind :func:`safe_auc` / :func:`safe_ap`: a non-finite
    score makes a ranking undefined, so it is removed before the metric sees it.
    """
    labels: list[bool] = []
    clean: list[float] = []
    for label, score in zip(y_true, scores, strict=True):
        if math.isfinite(score):
            labels.append(bool(label))
            clean.append(float(score))
    return labels, clean
