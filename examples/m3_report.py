"""Render the M3 race comparison table from the committed per-cell JSON results.

Reads every ``data/benchmarks/m3/results/*.json`` cell written by
``examples/m3_race.py`` and renders ``data/benchmarks/m3/M3_RESULTS.md``: the
5-method × {Fodors-Zagat, Amazon-Google} comparison with pair-level P/R/F1
(primary; AG labelled literature-comparable on the fixed test pairs), the
zero-spend pipeline BCubed / cluster-F1 / Δ-floor (mean ± std over 5 seeds), the
LLM FZ-band pipeline (single seed @0.5), cost ($ total, $/1k, est $/100k), latency
(s/pair), and cascade escalation diagnostics. Missing cells render as ``—`` so a
partial run still produces a (clearly partial) table. Pure read-render — it never
calls a model or spends.

Usage::

    uv run python examples/m3_report.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("data/benchmarks/m3/results")
OUT_PATH = Path("data/benchmarks/m3/M3_RESULTS.md")

ZERO_SPEND = ("rapidfuzz", "weighted_average", "embedding_cosine")
DASH = "—"


def load(cell_id: str) -> dict[str, Any] | None:
    """Load one cell's JSON, or ``None`` if it has not been produced yet."""
    path = RESULTS_DIR / f"{cell_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _f(value: float | None, places: int = 4) -> str:
    return DASH if value is None else f"{value:.{places}f}"


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


# ---------------------------------------------------------------------------
# Pair-level tables (primary judge ranking)
# ---------------------------------------------------------------------------


def _pair_row(label: str, model: str, cell: dict[str, Any] | None) -> str:
    if cell is None:
        return (
            f"| {label} | {model} | {DASH} | {DASH} | {DASH} | {DASH} | {DASH} "
            f"| {DASH} | {DASH} | {DASH} | {DASH} | {DASH} |"
        )
    ev = cell["eval"]
    pair, cost, lat = ev["pair"], ev["cost"], ev["latency"]
    n = ev["n_judged"]
    note = []
    if cell.get("subsample_n"):
        note.append(f"subsample {cell['subsample_n']}/{cell['subsample_of']}")
    if ev.get("truncated"):
        note.append(f"TRUNCATED to {ev['n_judged']}")
    notes = "; ".join(note) or ""
    return (
        f"| {label} | {model} | {n} | {_f(pair['precision'])} | {_f(pair['recall'])} "
        f"| **{_f(pair['f1'])}** | {_f(ev['best_threshold'], 2)} | {_f(cost['usd_total'])} "
        f"| {_f(cost['usd_per_1k_pairs'])} | {_f(cost['est_usd_per_100k'], 2)} "
        f"| {_f(lat['seconds_per_pair'], 4)} | {notes} |"
    )


def pair_table(dataset_tag: str, surface: str) -> str:
    """Render a pair-level table for one dataset (``agfixed`` or ``fzband`` cells)."""
    header = (
        "| method | model | n_judged | P | R | **F1** | best_thr | $total | $/1k "
        "| est$/100k | s/pair | notes |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    rows = [_pair_row(m, "zero-spend", load(f"{surface}_{m}")) for m in ZERO_SPEND]
    rows.append(_pair_row("llm_judge", "GLM-5.2", load(f"{surface}_llm_judge")))
    rows.append(_pair_row("cascade", "GLM-5.2", load(f"{surface}_cascade")))
    rows.append(_pair_row("llm_judge", "gpt-4o (frontier)", load(f"{surface}_llm_judge_frontier")))
    return f"### {dataset_tag} — pair-level (best-F1 threshold over the wide grid)\n\n" + "\n".join(
        [header, *rows]
    )


# ---------------------------------------------------------------------------
# Pipeline tables
# ---------------------------------------------------------------------------


def _zero_pipeline_row(method: str, dataset: str) -> str:
    cell = load(f"pipeline_{dataset}_{method}")
    if cell is None:
        return f"| {method} | zero-spend (5 seeds) | {DASH} | {DASH} | {DASH} | {DASH} | {DASH} |"
    pls = [r["pipeline"] for r in cell["results"]]
    bf1_m, bf1_s = _mean_std([p["bcubed_f1"] for p in pls])
    cf1_m, cf1_s = _mean_std([p["cluster_pairwise_f1"] for p in pls])
    dfl_m, dfl_s = _mean_std([p["delta_above_floor"] for p in pls])
    floor = pls[0]["sanity_floor_f1"]
    return (
        f"| {method} | zero-spend (5 seeds) | {bf1_m:.4f} ± {bf1_s:.4f} "
        f"| {cf1_m:.4f} ± {cf1_s:.4f} | {dfl_m:+.4f} ± {dfl_s:.4f} | {floor:.4f} | free |"
    )


def _llm_pipeline_row(label: str, model: str, surface_cell: dict[str, Any] | None) -> str:
    if surface_cell is None or "pipeline" not in surface_cell:
        return f"| {label} | {model} | {DASH} | {DASH} | {DASH} | {DASH} | {DASH} |"
    p = surface_cell["pipeline"]
    thr = surface_cell.get("pipeline_cluster_threshold", 0.5)
    return (
        f"| {label} | {model} @thr={thr} | {p['bcubed_f1']:.4f} | {p['cluster_pairwise_f1']:.4f} "
        f"| {p['delta_above_floor']:+.4f} | {p['sanity_floor_f1']:.4f} | "
        f"${surface_cell['usd_total']:.4f} |"
    )


def pipeline_table_fz() -> str:
    header = (
        "| method | setup | BCubed F1 | cluster pairwise-F1 | Δ above floor | floor F1 | cost |\n"
        "| --- | --- | --- | --- | --- | --- | --- |"
    )
    rows = [_zero_pipeline_row(m, "fodors_zagat") for m in ZERO_SPEND]
    rows.append(_llm_pipeline_row("llm_judge", "GLM-5.2", load("fzband_llm_judge")))
    rows.append(_llm_pipeline_row("cascade", "GLM-5.2", load("fzband_cascade")))
    rows.append(_llm_pipeline_row("llm_judge", "gpt-4o", load("fzband_llm_judge_frontier")))
    return "### Fodors-Zagat — pipeline (post-clustering)\n\n" + "\n".join([header, *rows])


def pipeline_table_ag() -> str:
    header = (
        "| method | setup | BCubed F1 | cluster pairwise-F1 | Δ above floor | floor F1 | cost |\n"
        "| --- | --- | --- | --- | --- | --- | --- |"
    )
    rows = [_zero_pipeline_row(m, "amazon_google") for m in ZERO_SPEND]
    return (
        "### Amazon-Google — pipeline (post-clustering)\n\n"
        "> The end-to-end **LLM** pipeline on Amazon-Google is deliberately SKIPPED: the "
        "blocked TEST band is ~48,854 pairs (un-affordable to LLM-judge in full under $15) AND "
        "blocking Pair-Completeness caps recall at ~0.84. The AG LLM signal is the fixed-pair "
        "pair-level eval above, not a clustered pipeline.\n\n" + "\n".join([header, *rows])
    )


# ---------------------------------------------------------------------------
# Cascade diagnostics + cost summary
# ---------------------------------------------------------------------------


def cascade_diagnostics() -> str:
    rows = []
    for surface, tag in (
        ("agfixed_cascade", "Amazon-Google (fixed pairs)"),
        ("fzband_cascade", "Fodors-Zagat (band)"),
    ):
        cell = load(surface)
        if cell is None:
            rows.append(f"| {tag} | {DASH} | {DASH} | {DASH} |")
            continue
        cost = cell["eval"]["cost"]
        rows.append(
            f"| {tag} | {_f(cost.get('escalation_rate'), 4)} "
            f"| {_f(cost.get('llm_calls_per_candidate'), 4)} | {_f(cost['usd_total'])} |"
        )
    header = (
        "| dataset | escalation rate | LLM calls / candidate | $total |\n| --- | --- | --- | --- |"
    )
    return "### Cascade (embedding→LLM) escalation diagnostics\n\n" + "\n".join([header, *rows])


def total_spend() -> float:
    return sum(
        float(json.loads(p.read_text()).get("usd_total", 0.0)) for p in RESULTS_DIR.glob("*.json")
    )


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


def render() -> str:
    committed = sorted(p.stem for p in RESULTS_DIR.glob("*.json"))
    spent = total_spend()
    parts = [
        "# M3 results — multi-method benchmark race (Wave 4, paid EXIT run)",
        "",
        "Five resolution methods raced on two benchmarks under a hard $15 cap. **Pair-level "
        "P/R/F1 is the primary judge-ranking metric.** For Amazon-Google it is measured on the "
        "FIXED literature `test` pair split (2293 pairs, 234 positives) with **no blocking**, so "
        "it is directly comparable to DeepMatcher/Ditto pairwise-F1 SOTA (~0.5-0.75). For "
        "Fodors-Zagat it is measured on the blocked TEST band. The best-F1 threshold is selected "
        "over a wide grid (0.05-0.99) — the shared race grid caps at 0.80, which unfairly "
        "collapses score-based judges.",
        "",
        f"**Total measured spend: ${spent:.4f} / $15.00 cap.**  "
        f"Models: GLM-5.2 (`openrouter/z-ai/glm-5.2`) + gpt-4o (`openrouter/openai/gpt-4o`, frontier).",
        "",
        f"Cells committed ({len(committed)}): {', '.join(committed)}",
        "",
        "## Primary: pair-level judge ranking",
        "",
        pair_table("Amazon-Google (FIXED literature test pairs — SOTA-comparable)", "agfixed"),
        "",
        pair_table("Fodors-Zagat (blocked TEST band)", "fzband"),
        "",
        "## Pipeline (post-clustering)",
        "",
        pipeline_table_fz(),
        "",
        "> LLM FZ-band pipeline clusters at threshold 0.5 (the natural LLM-probability boundary); "
        "zero-spend rows are tuned per-seed on TRAIN by `run_method`, so the two are not strictly "
        "apples-to-apples — pair-level (above) is the headline, pipeline is the production-context "
        "secondary.",
        "",
        pipeline_table_ag(),
        "",
        "## Cost & cascade",
        "",
        cascade_diagnostics(),
        "",
    ]
    return "\n".join(parts)


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    md = render()
    OUT_PATH.write_text(md)
    print(md)
    print(f"\n[report] wrote {OUT_PATH} (total spend ${total_spend():.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
