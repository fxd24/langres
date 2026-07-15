"""Shared plumbing for the two flywheel paid-validation scripts (T8).

Both :mod:`examples.research.flywheel_fz_smoke` (the <=$2 Fodors-Zagat wiring
smoke) and :mod:`examples.research.flywheel_amazon_google` (the <=$10 economics
run) drive the SAME closed loop (:func:`examples.flywheel_closed_loop.run_closed_loop`)
with a REAL frontier teacher under a hard
:class:`~langres.clients.openrouter.SpendMonitor` cap, then write the SAME shape
of result doc. This module lifts the genuinely-shared pieces -- building the real
teacher, the model/budget pre-flight, and serializing a
:class:`~examples.flywheel_closed_loop.ClosedLoopReport` into a JSON + Markdown
result doc -- so neither script re-implements them.

**$0 / import-safety:** nothing here imports ``dspy`` at module load;
:func:`build_real_teacher` imports :class:`~langres.core.matchers.dspy_judge.DSPyMatcher`
lazily (only on the real path), so the scripts and their ``simulated=True``
verification import cleanly in a lean env with no ``[llm]`` extra. ``print`` is
allowed under ``examples/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langres.clients.openrouter import (
    PRICES_PER_1M,
    dspy_price_per_1k,
    register_runtime_model_price,
)

if TYPE_CHECKING:
    from langres.core.matcher import Matcher

    from examples.flywheel_closed_loop import ClosedLoopReport, FZRecord

#: Default frontier model for both paid runs: PRICED (a key of PRICES_PER_1M, so
#: its spend cap is not blind) and cheap ($0.20/$0.80 per 1M).
DEFAULT_MODEL = "openrouter/openai/gpt-4o-mini"

#: Where both scripts write their committed result docs (the data/benchmarks/ pattern).
RESULTS_DIR = Path("data/benchmarks/flywheel")


def build_real_teacher(model: str, *, entity_noun: str) -> Matcher[FZRecord]:
    """Build a REAL DSPy frontier teacher with an honest per-1k price wired in.

    Lazily imports :class:`~langres.core.matchers.dspy_judge.DSPyMatcher` (the
    ``[llm]`` extra) so the simulated path never needs ``dspy``. The pinned
    price makes ``provenance["cost_usd"]`` honest, which is what the outer
    :class:`~examples.flywheel_closed_loop._OuterSpendCap` meters and enforces.

    Args:
        model: A ``PRICES_PER_1M``-pinned OpenRouter model id.
        entity_noun: Domain noun woven into the judge's prompt
            (``"restaurant"`` for Fodors-Zagat, ``"product"`` for Amazon-Google).

    Returns:
        A price-wired ``DSPyMatcher`` ready to score ``FZRecord`` candidates.
    """
    from langres.core.matchers.dspy_judge import DSPyMatcher

    judge: DSPyMatcher[FZRecord] = DSPyMatcher(model=model, entity_noun=entity_noun)
    judge.price_per_1k_tokens = dspy_price_per_1k(model)
    return judge


def preflight_real_model(model: str, *, budget_usd: float, ceiling_usd: float) -> str | None:
    """Refuse to spend unless the run is safe; pin the model's dated runtime price.

    Mirrors ``examples/research/w3_paid_smoke.py``'s ``main`` guards: the budget
    must be at or under ``ceiling_usd``; ``OPENROUTER_API_KEY`` must be set; the
    model must be priced in ``PRICES_PER_1M`` (else the cap is blind); and the
    model id must actually resolve/respond once (never guess-and-spend).

    Args:
        model: The frontier model id to validate + price-pin.
        budget_usd: The requested hard cap.
        ceiling_usd: The task's hard ceiling for this run (<=$2 FZ / <=$10 AG).

    Returns:
        ``None`` if the run is cleared to proceed, else a human-readable reason
        the caller should print before refusing (non-zero exit).
    """
    from dotenv import load_dotenv

    if budget_usd > ceiling_usd:
        return f"--budget ${budget_usd:.2f} exceeds the ${ceiling_usd:.2f} ceiling; refusing."
    load_dotenv(".env")  # OPENROUTER_API_KEY lives in .env, not Settings.
    if "OPENROUTER_API_KEY" not in os.environ:
        return "OPENROUTER_API_KEY not set; refusing to proceed (export it or use --simulated)."
    if model not in PRICES_PER_1M:
        return (
            f"model {model!r} is not in PRICES_PER_1M, so its spend cap would be blind. "
            f"Pin its price first, or pick one of: {sorted(PRICES_PER_1M)}."
        )
    if register_runtime_model_price(model) is None:
        return f"{model} did not resolve/respond; STOP (never guess-and-spend)."
    return None


def report_to_results(
    report: ClosedLoopReport,
    *,
    dataset: str,
    model: str,
    budget_usd: float,
    simulated: bool,
    notes: str = "",
) -> dict[str, Any]:
    """Serialize a closed-loop report into the committed result-doc dict.

    Args:
        report: The finished :class:`ClosedLoopReport`.
        dataset: Human label of the dataset (e.g. ``"fodors-zagat"``).
        model: The teacher model id (or ``"simulated-frontier"`` in a dry run).
        budget_usd: The spend cap the run was placed under.
        simulated: ``True`` when the teacher was the deterministic $0 simulation
            (so ``teacher_spend_usd`` is FICTIONAL, not real spend).
        notes: Optional free-text caveats to carry into the doc.

    Returns:
        A JSON-able dict with the headline economics + the full report dump.
    """
    return {
        "dataset": dataset,
        "model": model,
        "simulated": simulated,
        "budget_usd": budget_usd,
        "teacher_spend_usd": report.teacher_spend_usd,
        "escalation_rate": report.escalation_rate,
        "frontier_call_reduction": report.frontier_call_reduction,
        "n_escalated": report.n_escalated,
        "n_heldout": report.n_heldout,
        "escalated_accuracy": report.escalated_accuracy,
        "audit_disagreement_rate": report.audit_disagreement_rate,
        "teacher_f1": report.teacher.f1,
        "student_f1": report.student.f1,
        "cascade_f1": report.cascade.f1,
        "notes": notes,
        "report": report.model_dump(),
    }


def render_result_markdown(results: dict[str, Any], *, title: str) -> str:
    """Render the human-readable result doc (a small Markdown summary + table)."""
    spend_kind = "FICTIONAL (simulated teacher, no API call)" if results["simulated"] else "REAL"
    rep = results["report"]
    lines = [
        f"# {title}",
        "",
        f"- Dataset: **{results['dataset']}**",
        f"- Teacher model: `{results['model']}`",
        f"- Mode: **{'SIMULATED ($0)' if results['simulated'] else 'REAL (paid)'}**",
        f"- Spend cap (budget_usd): **${results['budget_usd']:.2f}**",
        f"- Teacher spend: **${results['teacher_spend_usd']:.4f}** ({spend_kind})",
        "",
        "## Economics",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| escalation rate | {results['escalation_rate']:.1%} "
        f"({results['n_escalated']}/{results['n_heldout']} held-out pairs) |",
        f"| frontier-call reduction | {results['frontier_call_reduction']:.1%} |",
        f"| escalated-pair accuracy | {results['escalated_accuracy']:.1%} |",
        f"| audit-slice disagreement | {results['audit_disagreement_rate']:.1%} |",
        "",
        "## Pairwise F1 on the held-out split (one threshold cuts every stream)",
        "",
        "| judge | F1 | precision | recall |",
        "| --- | --- | --- | --- |",
        f"| teacher (frontier) | {rep['teacher']['f1']:.3f} | "
        f"{rep['teacher']['precision']:.3f} | {rep['teacher']['recall']:.3f} |",
        f"| student (cheap) | {rep['student']['f1']:.3f} | "
        f"{rep['student']['precision']:.3f} | {rep['student']['recall']:.3f} |",
        f"| cascade | {rep['cascade']['f1']:.3f} | "
        f"{rep['cascade']['precision']:.3f} | {rep['cascade']['recall']:.3f} |",
        "",
    ]
    if results["notes"]:
        lines += ["## Notes", "", results["notes"], ""]
    return "\n".join(lines)


def write_result_docs(
    results: dict[str, Any], *, out_dir: Path, stem: str, title: str
) -> tuple[Path, Path]:
    """Write the JSON + Markdown result docs; return their paths.

    Args:
        results: The dict from :func:`report_to_results`.
        out_dir: Directory to write into (created if missing).
        stem: File stem (``<stem>.json`` + ``<stem>.md``).
        title: Markdown H1 title.

    Returns:
        ``(json_path, md_path)``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_path.write_text(render_result_markdown(results, title=title), encoding="utf-8")
    return json_path, md_path
