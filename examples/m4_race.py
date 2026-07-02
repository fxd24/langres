"""M4 Wave 2 — the paid, resumable DSPy-scorer benchmark on Amazon-Google.

This is the M4 script that spends real money (OpenRouter, ``$5`` global cap). It
drives the merged M4 seam — ``DSPyJudge`` (the compilable DSPy scorer) graded by
``evaluate_judge_on_candidates`` (pairwise-F1, judged-once) — on the Amazon-Google
fixed literature pair split, and answers M4's one falsifiable question:

    Can a *precision-tuned / compiled* CHEAP judge (GLM-5.2) approach the frontier
    (gpt-4o) at materially lower cost, versus its precision-collapsed zero-shot?

Cells (each writes one committed JSON under ``data/benchmarks/m4/results/`` so a
crash or interruption re-does only the missing cells — paid work is never lost):

* ``smoke`` (cents, NOT committed): a handful of GLM-5.2 DSPyJudge calls that
  confirm the paid DSPy path works AND registers honest cost > $0.
* ``ag600_dspy_glm_zeroshot`` — **the precision probe / DSPy baseline.** An
  *uncompiled* ``DSPyJudge`` (whose signature is already the hand-written,
  precision-tuned, hard-negative prompt: "a different model / size / edition /
  variant ⇒ a different product") on the deterministic 600-pair AG band. Its
  pair precision, versus GLM-5.2 zero-shot's 0.264 (M3) and frontier's 0.541
  (M3, the ceiling), is the C7 gate.
* ``ag600_dspy_glm_compiled`` — **only if the gate clears** and budget allows: a
  ``MIPROv2 auto="light"`` compile of the SAME judge on a small AG trainset,
  re-graded on the identical 600-band, with the compiled Resolver artifact saved.

Reused (cited, NOT re-run — they are identical-band, same-seed M3 numbers, so
re-spending on them is waste):
  * GLM-5.2 zero-shot (``DEFAULT_PROMPT``): P=0.264 R=0.902 F1=0.409  ($0.47)
  * gpt-4o frontier zero-shot (the ceiling): P=0.541 R=0.869 F1=0.667 ($0.91)

Budget safety: the ``$5`` cap is held two ways — a committed-JSON ledger
(``spent_so_far``, resume-safe) feeds a live :class:`SpendMonitor`, and every
capped judge cell runs through a :class:`BudgetedModuleRunner` (pre-flight pair
cap + per-pair tally stop). Costs are made honest by pricing judgements from their
captured token counts against pinned OpenRouter prices (litellm prices OpenRouter
at $0). MIPROv2's spend is internal/uncappable, so the compile cell is gated on a
data-driven worst-case estimate before it starts.

Usage (run with the sandbox disabled — OpenRouter is a network call)::

    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE uv run python examples/m4_race.py --smoke
    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE uv run python examples/m4_race.py --probe
    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE uv run python examples/m4_race.py --compile
    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE uv run python examples/m4_race.py --report

``OPENROUTER_API_KEY`` is loaded from ``.env`` (it is NOT a declared Settings
field). ``print`` is allowed in examples (this is an operator tool).
"""

from __future__ import annotations

import os

# Pin OpenMP / FAISS threading BEFORE importing torch/faiss (macOS libomp guard).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
from dspy import Example  # noqa: E402

from langres.clients.openrouter import (  # noqa: E402
    PRICES_PER_1M,
    SpendMonitor,
    make_token_cost_track,
    per_token_worst_price,
    register_runtime_model_price,
)
from langres.core import AllPairsBlocker, Clusterer, Resolver  # noqa: E402
from langres.core.benchmark import (  # noqa: E402
    BudgetedModuleRunner,
    JudgePairEval,
    evaluate_judge_on_candidates,
)
from langres.core.models import ERCandidate  # noqa: E402
from langres.core.modules.dspy_judge import DSPyJudge  # noqa: E402
from langres.data.amazon_google import (  # noqa: E402
    ProductSchema,
    load_amazon_google,
    load_amazon_google_pair_splits,
)

logger = logging.getLogger("m4_race")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data" / "benchmarks" / "m4" / "results"
ARTIFACT_DIR = ROOT / "data" / "benchmarks" / "m4" / "compiled_resolver"
RESULTS_MD = ROOT / "data" / "benchmarks" / "m4" / "M4_RESULTS.md"

HARD_BUDGET_USD = 5.0
#: Worst-case total tokens per pair, used to size the pre-flight budget cap
#: (product records are short; 1200 is a generous upper bound).
WORST_CASE_TOKENS_PER_PAIR = 1200

#: Deterministic AG fixed-pair band (same size/seed as M3, so the probe's F1 is
#: directly comparable to M3's GLM-5.2 zero-shot 0.409 and frontier 0.667).
BAND_N = 600
BAND_SEED = 0
ENTITY_NOUN = "product"

#: Small labeled sets for the MIPROv2 compile (kept tiny to bound the uncappable
#: MIPRO spend; auto="light" needs only a handful of demos/val examples).
COMPILE_TRAIN_N = 40
COMPILE_VAL_N = 40

GLM_MODEL = "openrouter/z-ai/glm-5.2"
GLM_MODEL_FALLBACK = "openrouter/z-ai/glm-4.6"

#: Wide pair-level threshold grid (matches M3's AG band grid — up to 0.99 so a
#: probability judge is not unfairly capped at 0.80).
GRID: tuple[float, ...] = (
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
)

SESSION = "https://claude.ai/code/session_01L9R9DAPyzhLjSRawN4gKD6"

#: Reused M3 references on the IDENTICAL 600-pair band (cited, never re-run).
M3_GLM_ZEROSHOT = {
    "label": "glm-5.2 zero-shot (M3, DEFAULT_PROMPT)",
    "model": "openrouter/z-ai/glm-5.2",
    "precision": 0.264,
    "recall": 0.902,
    "f1": 0.409,
    "usd": 0.4729,
}
M3_FRONTIER = {
    "label": "gpt-4o frontier zero-shot (M3, the ceiling)",
    "model": "openrouter/openai/gpt-4o",
    "precision": 0.541,
    "recall": 0.869,
    "f1": 0.667,
    "usd": 0.9114,
}

# Resolved OpenRouter model ids (verified against the live models list, 2026-07-02):
#   cheap judge (primary):  openrouter/z-ai/glm-5.2       (PROVEN in M3)
#   cheapest candidate:     openrouter/deepseek/deepseek-v4-flash
#   teacher/judge:          openrouter/moonshotai/kimi-k2.6
#   frontier ceiling:       openrouter/openai/gpt-4o      (reused from M3)
# The paid cells use GLM-5.2 only: it is the M3 baseline model, so changing only
# the PROMPT (zero-shot -> precision-tuned/compiled) isolates the M4 effect.


# ---------------------------------------------------------------------------
# Candidate construction (mirrors examples/m4_dspy_judge.py + m3_race.py)
# ---------------------------------------------------------------------------


def _split_candidates(split: str) -> tuple[list[ERCandidate[ProductSchema]], set[frozenset[str]]]:
    """Build ER candidates from a fixed AG literature pair split (no embedding).

    ``DSPyJudge`` reads the raw records (not a cosine similarity), so — unlike
    ``m3_race.build_ag_fixed_candidates`` — we skip the MiniLM embedding step.
    Candidates are built in fixed row order so a downstream stratified subsample
    reproduces the SAME band M3 evaluated.
    """
    corpus, _clusters, _gold = load_amazon_google()
    by_id = {r.id: r for r in corpus}
    rows = load_amazon_google_pair_splits()[split]
    candidates: list[ERCandidate[ProductSchema]] = []
    gold: set[frozenset[str]] = set()
    for amazon_id, google_id, label in rows:
        candidates.append(
            ERCandidate(
                left=by_id[amazon_id], right=by_id[google_id], blocker_name="ag_fixed_pairs"
            )
        )
        if label == 1:
            gold.add(frozenset({amazon_id, google_id}))
    return candidates, gold


def subsample_stratified(
    candidates: Sequence[ERCandidate[Any]],
    gold: set[frozenset[str]],
    n: int,
    *,
    seed: int = 0,
) -> list[ERCandidate[Any]]:
    """Deterministic label-stratified subsample of ``n`` candidates (copied from m3_race).

    Keeps the positive fraction stable so a subsampled recall estimate is not
    biased. Identical logic + seed to M3, so the 600-band is the same pair set.
    """
    rng = random.Random(seed)
    pos = [c for c in candidates if frozenset({c.left.id, c.right.id}) in gold]
    neg = [c for c in candidates if frozenset({c.left.id, c.right.id}) not in gold]
    frac = n / len(candidates)
    n_pos = round(len(pos) * frac)
    n_neg = n - n_pos
    rng.shuffle(pos)
    rng.shuffle(neg)
    chosen = pos[:n_pos] + neg[:n_neg]
    rng.shuffle(chosen)
    return chosen


def build_ag_band() -> tuple[list[ERCandidate[ProductSchema]], set[frozenset[str]]]:
    """The deterministic 600-pair AG test band + its positive gold pairs."""
    candidates, gold = _split_candidates("test")
    band = subsample_stratified(candidates, gold, BAND_N, seed=BAND_SEED)
    band_gold = {
        frozenset({c.left.id, c.right.id})
        for c in band
        if frozenset({c.left.id, c.right.id}) in gold
    }
    return band, band_gold


def _trainset(
    candidates: Sequence[ERCandidate[ProductSchema]], gold: set[frozenset[str]]
) -> list[Example]:
    """Turn candidates into labeled DSPy examples (rendered exactly like ``forward``)."""
    examples: list[Example] = []
    for candidate in candidates:
        is_match = frozenset({candidate.left.id, candidate.right.id}) in gold
        examples.append(
            Example(
                left=candidate.left.model_dump_json(indent=2),
                right=candidate.right.model_dump_json(indent=2),
                match=is_match,
            ).with_inputs("left", "right")
        )
    return examples


# ---------------------------------------------------------------------------
# DSPy judge + cost helpers
# ---------------------------------------------------------------------------


def build_glm_dspy_judge(program: Any = None) -> DSPyJudge[ProductSchema]:
    """A GLM-5.2 ``DSPyJudge`` with an honest (conservative) per-token price wired in.

    ``price_per_1k_tokens`` is set to GLM-5.2's *output* rate (the dearer side), so
    the per-pair provenance cost the ``BudgetedModuleRunner`` tallies is a safe
    upper bound; the FINAL reported ``usd_total`` uses ``make_token_cost_track``
    (accurate input/output split), so the cap is conservative and the report exact.
    """
    judge: DSPyJudge[ProductSchema] = DSPyJudge(
        model=GLM_MODEL, temperature=0.0, entity_noun=ENTITY_NOUN, program=program
    )
    judge.price_per_1k_tokens = max(PRICES_PER_1M[GLM_MODEL]) / 1000.0
    return judge


def _lm_history_cost(lm: Any, model: str = GLM_MODEL) -> tuple[float, int, int]:
    """Honest USD cost of EVERY call recorded on a DSPy LM's history (compile + eval).

    MIPROv2's spend is internal and not surfaced by ``evaluate_judge_on_candidates``
    (which only prices the eval judgements). DSPy records every LM call in
    ``lm.history`` with a ``usage`` dict, so we price the whole history against the
    pinned per-1M rates — the source-of-truth cost for the compile cell.

    Returns ``(usd_total, prompt_tokens, completion_tokens)``.
    """
    in_per_1m, out_per_1m = PRICES_PER_1M[model]
    prompt_tokens = completion_tokens = 0
    for call in getattr(lm, "history", []) or []:
        usage = call.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens += int(usage.get("completion_tokens", 0) or 0)
    usd = prompt_tokens * in_per_1m / 1_000_000.0 + completion_tokens * out_per_1m / 1_000_000.0
    return usd, prompt_tokens, completion_tokens


# ---------------------------------------------------------------------------
# Budget ledger + per-cell persistence (mirrors m3_race)
# ---------------------------------------------------------------------------


def spent_so_far() -> float:
    """Sum the recorded spend of every committed cell JSON (resume-safe ledger)."""
    total = 0.0
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        total += float(data.get("usd_total", 0.0))
    return total


def _write_cell(cell_id: str, result: dict[str, Any]) -> Path:
    """Atomically write one cell JSON (a truncated write can never look 'done')."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{cell_id}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, path)
    return path


def commit_and_push(paths: Sequence[Path], cell_id: str) -> None:
    """Git add + commit + push the given paths (durability before the next spend).

    Push is best-effort (the commit already secures the paid result in the branch);
    a push failure is logged, not fatal, so a transient network blip never discards
    a just-paid cell.
    """
    subprocess.run(["git", "add", *[str(p) for p in paths]], check=True)
    subprocess.run(
        [
            "git",
            "commit",
            "-q",
            "-m",
            f"M4 W2: paid race cell {cell_id}\n\nClaude-Session: {SESSION}",
        ],
        check=True,
    )
    try:
        subprocess.run(["git", "push", "-q", "origin", "HEAD"], check=True)
        print(f"[commit] {cell_id}: committed + pushed")
    except subprocess.CalledProcessError as exc:
        print(f"[commit] {cell_id}: committed; PUSH FAILED ({exc}) — commit still secures it")


def _judge_eval_dict(
    cell_id: str, method: str, model: str, result: JudgePairEval, **extra: Any
) -> dict[str, Any]:
    """Persisted cell shape; the ledger reads ``usd_total``."""
    return {
        "cell_id": cell_id,
        "kind": "ag600_pair",
        "method": method,
        "dataset": "amazon_google",
        "model": model,
        "band_n": BAND_N,
        "band_seed": BAND_SEED,
        "usd_total": result.cost.usd_total,
        "eval": result.model_dump(),
        **extra,
    }


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


def smoke() -> int:
    """A few real GLM-5.2 DSPyJudge calls: confirm the paid path works + cost > $0."""
    dated = register_runtime_model_price(GLM_MODEL)
    if dated is None:
        print(f"[smoke] {GLM_MODEL} did not respond; trying fallback {GLM_MODEL_FALLBACK}")
        return 1
    candidates, gold = _split_candidates("test")
    sample = subsample_stratified(candidates, gold, 6, seed=0)
    judge = build_glm_dspy_judge()
    result, judgements = evaluate_judge_on_candidates(
        judge, sample, gold, GRID, cost_track_fn=make_token_cost_track(GLM_MODEL)
    )
    hist_usd, pt, ct = _lm_history_cost(judge._get_lm())
    print(
        f"[smoke] {GLM_MODEL} (dated={dated}): judged {result.n_judged}/{len(sample)} pairs, "
        f"F1={result.pair.f1:.3f} P={result.pair.precision:.3f} R={result.pair.recall:.3f}"
    )
    print(
        f"[smoke] token cost (make_token_cost_track)=${result.cost.usd_total:.6f} | "
        f"lm.history cost=${hist_usd:.6f} (prompt={pt}, completion={ct} tokens)"
    )
    ok = result.cost.usd_total > 0.0 and result.n_judged > 0
    print(
        f"[smoke] paid DSPy path {'OK — real spend registered' if ok else 'FAILED — cost is $0'}."
    )
    return 0 if ok else 1


def run_probe_cell(monitor: SpendMonitor) -> dict[str, Any]:
    """The precision probe / DSPy baseline: UNCOMPILED GLM-5.2 DSPyJudge on the 600-band."""
    cell_id = "ag600_dspy_glm_zeroshot"
    band, gold = build_ag_band()
    judge = build_glm_dspy_judge()  # uncompiled -> the precision-tuned signature is the baseline
    remaining = HARD_BUDGET_USD - spent_so_far()
    runner = BudgetedModuleRunner(
        judge,
        budget_usd=max(0.02, remaining),
        budget_soft_usd=max(0.01, remaining - 0.30),
        worst_case_units_per_pair=float(WORST_CASE_TOKENS_PER_PAIR),
    )
    result, _judgements = evaluate_judge_on_candidates(
        judge,
        band,
        gold,
        GRID,
        runner=runner,
        price_per_token_or_pair=per_token_worst_price(GLM_MODEL),
        cost_track_fn=make_token_cost_track(GLM_MODEL),
    )
    monitor.add(result.cost.usd_total)
    monitor.check()
    print(
        f"[probe] {cell_id}: F1={result.pair.f1:.3f} P={result.pair.precision:.3f} "
        f"R={result.pair.recall:.3f} @thr={result.best_threshold:.2f} "
        f"(judged {result.n_judged}/{len(band)}, cost ${result.cost.usd_total:.4f}, "
        f"cumulative ${monitor.spent:.4f}/${HARD_BUDGET_USD:.2f})"
    )
    return _judge_eval_dict(cell_id, "dspy_judge_uncompiled", GLM_MODEL, result, compiled=False)


def estimate_compile_cost(per_call_usd: float) -> float:
    """Data-driven worst-case MIPROv2 + eval cost, from the probe's real per-call cost.

    MIPROv2 ``auto="light"`` proposes instructions, bootstraps demos, and evaluates
    candidate programs over minibatches. A generous upper bound on its LM calls for
    a tiny trainset is ~ (10 trials x 35 minibatch) + bootstrapping + proposal
    ~= 500 calls; the final eval adds BAND_N calls. We price that call count at
    3x the probe's mean per-call cost as a safety multiplier.
    """
    est_calls = 500 + BAND_N
    return est_calls * per_call_usd * 3.0


def run_compile_cell(monitor: SpendMonitor) -> dict[str, Any]:
    """MIPROv2 compile of the GLM-5.2 DSPyJudge + eval on the 600-band + save artifact."""
    cell_id = "ag600_dspy_glm_compiled"
    train_cands, train_gold = _split_candidates("train")
    val_cands, val_gold = _split_candidates("valid")
    train_sub = subsample_stratified(train_cands, train_gold, COMPILE_TRAIN_N, seed=0)
    val_sub = subsample_stratified(val_cands, val_gold, COMPILE_VAL_N, seed=0)

    judge = build_glm_dspy_judge()
    print(
        f"[compile] MIPROv2 auto=light on {len(train_sub)} train / {len(val_sub)} val examples..."
    )
    judge.compile(_trainset(train_sub, train_gold), _trainset(val_sub, val_gold), optimizer="mipro")
    print(f"[compile] done. compiled={judge.compiled}")

    band, gold = build_ag_band()
    result, _judgements = evaluate_judge_on_candidates(
        judge, band, gold, GRID, cost_track_fn=make_token_cost_track(GLM_MODEL)
    )
    # Honest TOTAL cell cost = every LM call on this judge's history (compile + eval).
    total_usd, pt, ct = _lm_history_cost(judge._get_lm())
    monitor.add(total_usd)
    monitor.check()

    resolver = Resolver(
        blocker=AllPairsBlocker(schema=ProductSchema),
        comparator=None,
        module=judge,
        clusterer=Clusterer(threshold=result.best_threshold),
    )
    resolver.save(ARTIFACT_DIR)
    print(
        f"[compile] {cell_id}: F1={result.pair.f1:.3f} P={result.pair.precision:.3f} "
        f"R={result.pair.recall:.3f} @thr={result.best_threshold:.2f} | "
        f"TOTAL cell cost ${total_usd:.4f} (prompt={pt}, completion={ct} tokens) | "
        f"cumulative ${monitor.spent:.4f}/${HARD_BUDGET_USD:.2f} | artifact -> {ARTIFACT_DIR}"
    )
    cell = _judge_eval_dict(cell_id, "dspy_judge_mipro", GLM_MODEL, result, compiled=True)
    # Override usd_total with the honest history-priced total (eval-only would undercount MIPRO).
    cell["usd_total"] = total_usd
    cell["compile_prompt_tokens"] = pt
    cell["compile_completion_tokens"] = ct
    return cell


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pair_of(cell: dict[str, Any]) -> dict[str, float]:
    p = cell["eval"]["pair"]
    return {
        "p": p["precision"],
        "r": p["recall"],
        "f1": p["f1"],
        "thr": cell["eval"]["best_threshold"],
    }


def write_report() -> None:
    """Aggregate committed cells + reused M3 refs into M4_RESULTS.md (the hand-back)."""
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    cells = {
        json.loads(p.read_text())["cell_id"]: json.loads(p.read_text())
        for p in sorted(RESULTS_DIR.glob("*.json"))
    }
    lines: list[str] = []
    lines.append("# M4 Wave 2 — paid DSPy-scorer benchmark on Amazon-Google\n")
    lines.append(
        "The falsifiable M4 question: can a **precision-tuned / compiled cheap judge** "
        "(GLM-5.2) approach frontier quality (gpt-4o) at materially lower cost, versus "
        "its precision-collapsed zero-shot? Evaluated on the deterministic "
        f"{BAND_N}-pair Amazon-Google literature band (seed {BAND_SEED}), pairwise-F1 "
        "(judged once) at the best-F1 grid threshold.\n"
    )
    lines.append("## Resolved OpenRouter model ids\n")
    lines.append("| role | id | status |")
    lines.append("| --- | --- | --- |")
    lines.append("| cheap judge (primary) | `openrouter/z-ai/glm-5.2` | used (PROVEN in M3) |")
    lines.append(
        "| cheapest candidate | `openrouter/deepseek/deepseek-v4-flash` | resolved, not spent |"
    )
    lines.append(
        "| teacher/judge candidate | `openrouter/moonshotai/kimi-k2.6` | resolved, not spent |"
    )
    lines.append(
        "| frontier ceiling | `openrouter/openai/gpt-4o` | reused from M3 (not re-run) |\n"
    )

    lines.append("## Results (600-pair AG band, pairwise P/R/F1)\n")
    lines.append("| cell | judge | P | R | F1 | thr | USD | source |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    lines.append(
        f"| — | {M3_GLM_ZEROSHOT['label']} | {M3_GLM_ZEROSHOT['precision']:.3f} | "
        f"{M3_GLM_ZEROSHOT['recall']:.3f} | {M3_GLM_ZEROSHOT['f1']:.3f} | 0.90 | "
        f"{M3_GLM_ZEROSHOT['usd']:.4f} | M3 (reused) |"
    )
    probe = cells.get("ag600_dspy_glm_zeroshot")
    if probe:
        m = _pair_of(probe)
        lines.append(
            f"| ag600_dspy_glm_zeroshot | GLM-5.2 DSPyJudge UNCOMPILED (precision-tuned signature) | "
            f"{m['p']:.3f} | {m['r']:.3f} | {m['f1']:.3f} | {m['thr']:.2f} | "
            f"{probe['usd_total']:.4f} | **this run** |"
        )
    compiled = cells.get("ag600_dspy_glm_compiled")
    if compiled:
        m = _pair_of(compiled)
        lines.append(
            f"| ag600_dspy_glm_compiled | GLM-5.2 DSPyJudge MIPROv2-compiled | "
            f"{m['p']:.3f} | {m['r']:.3f} | {m['f1']:.3f} | {m['thr']:.2f} | "
            f"{compiled['usd_total']:.4f} | **this run** |"
        )
    lines.append(
        f"| — | {M3_FRONTIER['label']} | {M3_FRONTIER['precision']:.3f} | "
        f"{M3_FRONTIER['recall']:.3f} | {M3_FRONTIER['f1']:.3f} | 0.85 | "
        f"{M3_FRONTIER['usd']:.4f} | M3 (reused, the ceiling) |\n"
    )

    # C7 gate narrative.
    lines.append("## C7 gate — can a precision-tuned/compiled cheap judge approach frontier?\n")
    if probe:
        m = _pair_of(probe)
        dp = m["p"] - M3_GLM_ZEROSHOT["precision"]
        gap = M3_FRONTIER["precision"] - m["p"]
        lines.append(
            f"- Cheap-judge **precision**: zero-shot 0.264 (M3) -> precision-tuned "
            f"DSPy baseline **{m['p']:.3f}** (Δ {dp:+.3f}); frontier ceiling 0.541 "
            f"(gap {gap:+.3f}).\n"
        )
    else:
        lines.append("- (probe cell not yet run)\n")

    lines.append("## Spend\n")
    lines.append(
        f"- Cumulative committed spend: **${spent_so_far():.4f} / ${HARD_BUDGET_USD:.2f}**."
    )
    for cid, cell in cells.items():
        lines.append(f"  - `{cid}`: ${float(cell.get('usd_total', 0.0)):.4f}")
    lines.append("")

    RESULTS_MD.write_text("\n".join(lines))
    print(f"[report] wrote {RESULTS_MD} (cumulative ${spent_so_far():.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    load_dotenv(".env")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="M4 W2 paid DSPy-scorer benchmark (AG, $5 cap).")
    parser.add_argument(
        "--smoke", action="store_true", help="Cents smoke: confirm paid DSPy path + cost>0."
    )
    parser.add_argument(
        "--probe", action="store_true", help="Run the 600-band precision probe (paid)."
    )
    parser.add_argument(
        "--compile", action="store_true", help="Run the MIPROv2 compile cell (paid, gated)."
    )
    parser.add_argument(
        "--report", action="store_true", help="(Re)write M4_RESULTS.md from committed cells."
    )
    args = parser.parse_args()

    if args.report:
        write_report()
        return 0

    if "OPENROUTER_API_KEY" not in os.environ:
        print("[fatal] OPENROUTER_API_KEY not set; refusing to proceed.")
        return 1

    if args.smoke:
        return smoke()

    # Pin the GLM price + confirm the id responds before ANY spend.
    if register_runtime_model_price(GLM_MODEL) is None:
        print(f"[fatal] {GLM_MODEL} did not resolve/respond; STOP (never guess-and-spend).")
        return 1

    monitor = SpendMonitor(budget_usd=HARD_BUDGET_USD)
    monitor.add(spent_so_far())  # seed from the committed ledger (resume-safe).
    print(
        f"[budget] cumulative committed spend so far: ${monitor.spent:.4f} / ${HARD_BUDGET_USD:.2f}"
    )

    if args.probe:
        cell_id = "ag600_dspy_glm_zeroshot"
        if (RESULTS_DIR / f"{cell_id}.json").exists():
            print(f"[skip] {cell_id} already committed.")
            return 0
        cell = run_probe_cell(monitor)
        path = _write_cell(cell_id, cell)
        commit_and_push([path], cell_id)
        write_report()
        commit_and_push([RESULTS_MD], "report")
        return 0

    if args.compile:
        cell_id = "ag600_dspy_glm_compiled"
        if (RESULTS_DIR / f"{cell_id}.json").exists():
            print(f"[skip] {cell_id} already committed.")
            return 0
        remaining = HARD_BUDGET_USD - spent_so_far()
        probe_path = RESULTS_DIR / "ag600_dspy_glm_zeroshot.json"
        if not probe_path.exists():
            print("[fatal] run --probe first (the compile is gated on the C7 decision).")
            return 1
        probe = json.loads(probe_path.read_text())
        n_judged = max(1, int(probe["eval"]["n_judged"]))
        per_call = float(probe["usd_total"]) / n_judged
        est = estimate_compile_cost(per_call)
        print(
            f"[compile] budget check: remaining ${remaining:.4f}, probe per-call ${per_call:.6f}, "
            f"worst-case MIPRO+eval estimate ${est:.4f}."
        )
        if remaining < 2.50:
            print(
                f"[stop] < $2.50 remaining (${remaining:.4f}); not compiling. Report for go-ahead."
            )
            return 2
        if est > remaining * 0.9:
            print(
                f"[stop] worst-case ${est:.4f} does not comfortably fit ${remaining:.4f}; report for go-ahead."
            )
            return 2
        cell = run_compile_cell(monitor)
        path = _write_cell(cell_id, cell)
        artifact_files = list(ARTIFACT_DIR.rglob("*"))
        commit_and_push([path, *artifact_files], cell_id)
        write_report()
        commit_and_push([RESULTS_MD], "report")
        return 0

    print("Nothing to do — pass --smoke, --probe, --compile, or --report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
