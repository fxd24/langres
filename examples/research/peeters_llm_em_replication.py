"""Peeters, Steiner & Bizer (EDBT 2025) LLM-EM replication — offline replay + live paid run.

Three modes over the SAME abt-buy `domain-complex-force` slice (arXiv 2310.11244
v4 Table 2), sharing one prompt template + Peeters serializer so what the paid
run pays for is exactly what the `$0` replay validated:

* ``--mode replay`` (default, **$0**, no key): replays the authors' *archived*
  raw model answers through langres — downloads their public answer archive to a
  temp cache, parses each stored answer with our unified parser, aligns to the
  gold labels of our regenerated pair-set slice, and scores pairwise P/R/F1 with
  ``langres.core.metrics.classify_pairs`` (validating *our* metric code). It also
  verifies the prompt round-trip byte-for-byte. Target: ``abt-buy`` /
  ``gpt-4-0613`` / ``domain-complex-force`` → **F1 95.15**.

* ``--mode dry-run`` (**$0**, no key): renders every one of the 1206 pairs
  through the *live* path's template + serializer and reports the token counts +
  a cost estimate — **zero API calls**. Use it to preview cost and confirm the
  rendered prompt matches the archived one before spending anything.

* ``--mode live`` (**PAID**, off by default): runs a real ``LLMJudge`` (the
  paper's ``domain-complex-force`` template, the Peeters per-dataset
  ``record_serializer``, ``response_parser=parse_binary_yes_no``,
  ``temperature=0.0``) over the 1206 regenerated Abt-Buy pairs, under a hard
  :class:`~langres.clients.openrouter.SpendMonitor` cap. It is guarded three
  ways: an explicit ``--yes-spend-money`` flag, a priced-model assertion, and the
  cap. It reports F1 + the aggregated :class:`~langres.core.usage.LLMUsage`
  vector + the **real OpenRouter-billed** cost (``cost_is_real``) per model.

The paid run races exactly two dated snapshots (Abt-Buy domain-complex-force):

* ``openrouter/openai/gpt-4o-mini-2024-07-18`` — the paper's "GPT-mini", published
  F1 **90.95**.
* ``openrouter/openai/gpt-4o-2024-08-06`` — the paper's "GPT-4o", published F1
  **89.33**.

``gpt-4-0613`` (the F1 **95.15** cell the offline replay reproduces) would cost
~$3.15 to run live and was **deliberately declined** — not worth the spend, and
it retires 2026-10-23. ``gpt-3.5-turbo-0613`` / ``-0301`` were shut down
2024-09-13, so neither is a live option.

Spend safety (read twice):

* ``--mode live`` makes **no** network call until AFTER it prints the cost
  estimate and sees ``--yes-spend-money``. Each pair's real cost is charged to
  one :class:`SpendMonitor` and ``check()``\\ ed, so cumulative spend cannot
  cross ``--budget`` (default **$1.00** for both models combined; measured total
  ≈ $0.29, a ~3.4x margin).
* Every model MUST be priced in
  :data:`~langres.clients.openrouter.PRICES_PER_1M` — an unpriced model silently
  contributes $0 to the cap, so the script refuses to start without a price entry.
* ``OPENROUTER_API_KEY`` is **required** for ``--mode live`` and is **NOT present
  in this environment**; the paid path fails fast with a clear message rather
  than silently falling back to another provider or an unpriced model.

Data licensing: MatchGPT ships no LICENSE (``license: null``); langres is
Apache-2.0. Nothing from MatchGPT is vendored — the ~186 MB answer archive is
downloaded transiently to a cache dir (``--cache-dir``) and never committed. Our
pair-set slice is regenerated from our own already-vendored DeepMatcher CSVs.

Usage::

    # $0, no key:
    uv run python examples/research/peeters_llm_em_replication.py
    uv run python examples/research/peeters_llm_em_replication.py --mode dry-run
    # PAID (run with the sandbox disabled — OpenRouter is a network call):
    uv run python examples/research/peeters_llm_em_replication.py --mode live --yes-spend-money

``print`` is allowed in examples (this is an operator tool).
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langres.clients.openrouter import (
    PRICES_PER_1M,
    BudgetExceeded,
    SpendMonitor,
)
from langres.core.metrics import classify_pairs
from langres.core.modules.llm_judge import LLMJudge, parse_binary_yes_no
from langres.core.usage import LLMUsage
from langres.data.peeters import (
    PeetersReplicationSpec,
    build_candidates,
    build_llm_prompt_template,
    get_peeters_replication,
    gold_match_pairs,
    judgements_from_answers,
    list_peeters_replications,
    load_peeters_sample,
    make_record_serializer,
    render_sample_prompts,
)

logger = logging.getLogger("peeters_llm_em")

# ---------------------------------------------------------------------------
# Offline replay (archived answers) — constants + helpers
# ---------------------------------------------------------------------------

#: The authors' Git-LFS answer archive (real bytes, not an LFS pointer).
ARCHIVE_URL = (
    "https://media.githubusercontent.com/media/wbsg-uni-mannheim/MatchGPT/"
    "main/LLMForEM/prompts-and-answers/prompts_and_answers.zip"
)

#: Published arXiv v4 Table 2 F1 (%) per (dataset, model, prompt-design) — the
#: *offline replay* cells this harness can assert against (their archived answers).
PUBLISHED_F1 = {
    ("abt-buy", "gpt-4-0613", "domain-complex-force"): 95.15,
}

#: 2-decimal reporting tolerance for the offline replay: our exact F1 (e.g.
#: 95.1456) must round to the published 2-dp value; 0.05 absorbs that rounding.
F1_TOLERANCE = 0.05


def member_name(dataset: str, prompt_design: str, model: str) -> str:
    """Archive member for one (dataset, prompt-design, model) run."""
    return f"{dataset}-sampled-gs_{prompt_design}_default_{model}_run-1.jsonl"


def ensure_answers(cache_dir: Path, dataset: str, prompt_design: str, model: str) -> Path:
    """Return the extracted JSONL for a run, downloading/extracting on demand.

    The 186 MB archive is downloaded once (skipped if already cached), and each
    requested member is extracted once.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    member = member_name(dataset, prompt_design, model)
    extracted = cache_dir / member
    if extracted.exists():
        print(f"[cache] using extracted answers: {extracted}")
        return extracted

    archive = cache_dir / "prompts_and_answers.zip"
    if not archive.exists():
        print(f"[download] {ARCHIVE_URL}\n           -> {archive} (~186 MB, one time)")
        urllib.request.urlretrieve(ARCHIVE_URL, archive)  # noqa: S310 (trusted host)
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        if member not in names:
            available = "\n  ".join(n for n in names if dataset in n) or "(none for dataset)"
            raise SystemExit(
                f"member {member!r} not in archive. Available for {dataset}:\n  {available}"
            )
        print(f"[extract] {member}")
        zf.extract(member, cache_dir)
    return extracted


def replay(dataset: str, model: str, prompt_design: str, cache_dir: Path) -> None:
    """Replay one archived run and report prompt round-trip + pairwise F1 ($0)."""
    spec = get_peeters_replication(dataset)
    prompts = render_sample_prompts(spec)  # our records + serializer, in sample order

    answers_path = ensure_answers(cache_dir, dataset, prompt_design, model)
    archived = [json.loads(line) for line in answers_path.read_text().splitlines() if line.strip()]

    if len(archived) != len(prompts):
        raise SystemExit(
            f"archive has {len(archived)} lines but our sample has {len(prompts)} pairs — "
            "alignment would be wrong; aborting."
        )

    # --- Prompt round-trip: our rendered prompt vs their archived prompt -------
    exact = sum(1 for p, rec in zip(prompts, archived, strict=True) if p.prompt == rec["prompt"])
    round_trip = 100.0 * exact / len(prompts)

    # --- Replay: parse their answers, score with OUR metric code ---------------
    raw_answers = [rec["answer"] for rec in archived]
    judgements = judgements_from_answers(prompts, raw_answers)
    metrics = classify_pairs(judgements, gold_match_pairs(prompts), threshold=0.5)
    f1_pct = metrics.f1 * 100.0

    print("\n" + "=" * 72)
    print(f"Peeters LLM-EM replay — {dataset} / {model} / {prompt_design}")
    print("=" * 72)
    print(f"pairs                : {len(prompts)}  ({len(gold_match_pairs(prompts))} positive)")
    print(f"prompt round-trip    : {exact}/{len(prompts)} = {round_trip:.2f}% byte-exact")
    print(
        f"pairwise (via classify_pairs): P={metrics.precision * 100:.2f}  "
        f"R={metrics.recall * 100:.2f}  F1={f1_pct:.2f}"
    )
    print(f"  tp={metrics.tp} fp={metrics.fp} fn={metrics.fn}")

    target = PUBLISHED_F1.get((dataset, model, prompt_design))
    if target is not None:
        delta = abs(f1_pct - target)
        ok = delta <= F1_TOLERANCE
        print(f"published Table 2 F1 : {target:.2f}  (Δ={delta:.4f}, tol={F1_TOLERANCE})")
        if not ok:
            raise SystemExit(f"FAIL: replayed F1 {f1_pct:.2f} != published {target:.2f}")
        print("REPRODUCED ✓")
    else:
        print("(no published-F1 assertion registered for this cell — reported only)")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Live (paid) run — constants + the testable core
# ---------------------------------------------------------------------------

#: The dated snapshots the paid run races, keyed by OpenRouter routing id (each
#: MUST be a key of PRICES_PER_1M). Value = the published Abt-Buy
#: domain-complex-force F1 (arXiv v4 Table 2), shown as a column so the delta is
#: visible at a glance.
PAID_MODELS: dict[str, float] = {
    "openrouter/openai/gpt-4o-mini-2024-07-18": 90.95,  # paper "GPT-mini"
    "openrouter/openai/gpt-4o-2024-08-06": 89.33,  # paper "GPT-4o"
}

#: Hard spend cap (USD) for the two-model run combined. Measured total ≈ $0.29,
#: so $1.00 is a ~3.4x margin. ``main`` refuses anything above the ceiling.
DEFAULT_BUDGET_USD = 1.00
BUDGET_CEILING_USD = 2.00

#: Estimated output tokens per pair for the dry-run cost estimate: the binary
#: protocol answers with a single "Yes"/"No" word (~1–2 o200k tokens). The live
#: run meters the REAL output cost; this only sizes the pre-flight estimate.
_EST_OUTPUT_TOKENS_PER_PROMPT = 2


def _tokenizer_model(model: str) -> str:
    """Bare model id for tokenization (strip routing/provider prefixes).

    ``litellm.token_counter`` only resolves the correct **o200k_base** encoding
    from the bare OpenAI id (``gpt-4o-mini-2024-07-18``); the ``openrouter/…`` and
    ``openai/…`` prefixed forms silently fall back to cl100k_base and over-count
    (~1.5% high). Taking the last path segment recovers the id that bills
    correctly — verified to match a direct o200k_base count (100,256 tokens over
    the 1206 abt-buy prompts).
    """
    return model.split("/")[-1]


def _litellm_token_counter(model: str) -> Callable[[str], int]:
    """A ``prompt -> input_token_count`` closure for ``model`` (local tiktoken, $0)."""
    import litellm

    tok_model = _tokenizer_model(model)

    def count(prompt: str) -> int:
        return int(
            litellm.token_counter(model=tok_model, messages=[{"role": "user", "content": prompt}])
        )

    return count


def dry_run(
    spec: PeetersReplicationSpec,
    model: str,
    *,
    prices: dict[str, tuple[float, float]] = PRICES_PER_1M,
    count_tokens: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Render every pair through the LIVE path and count tokens — ZERO API calls.

    Renders each candidate with the exact ``build_llm_prompt_template`` +
    ``make_record_serializer`` the live ``LLMJudge`` uses, so the input-token
    total is the true billed input. Output tokens are estimated (the single-word
    binary answer). ``count_tokens`` is injectable for a dependency-free test;
    ``None`` uses the real litellm/tiktoken counter.

    Returns a dict with the pair count, input/output token totals, mean/max input,
    and a cost estimate priced against ``prices[model]``.
    """
    template = build_llm_prompt_template(spec)
    serializer = make_record_serializer(spec)
    candidates = build_candidates(spec)
    counter = count_tokens if count_tokens is not None else _litellm_token_counter(model)

    input_tokens = 0
    max_input = 0
    for candidate in candidates:
        prompt = template.replace("{left}", serializer(candidate.left)).replace(
            "{right}", serializer(candidate.right)
        )
        n = counter(prompt)
        input_tokens += n
        max_input = max(max_input, n)

    n_pairs = len(candidates)
    output_tokens = n_pairs * _EST_OUTPUT_TOKENS_PER_PROMPT
    in_per_1m, out_per_1m = prices[model]
    est_usd = input_tokens * in_per_1m / 1_000_000.0 + output_tokens * out_per_1m / 1_000_000.0
    return {
        "model": model,
        "n_pairs": n_pairs,
        "input_tokens": input_tokens,
        "output_tokens_est": output_tokens,
        "mean_input_tokens": input_tokens / n_pairs if n_pairs else 0.0,
        "max_input_tokens": max_input,
        "est_usd": est_usd,
    }


def _aggregate_usage(judgements: Sequence[Any], model: str) -> LLMUsage:
    """Sum the per-judgement ``provenance["usage"]`` vectors into one ``LLMUsage``."""
    totals = [0, 0, 0, 0, 0]
    provider: str | None = None
    for judgement in judgements:
        raw = judgement.provenance.get("usage") or {}
        usage = LLMUsage(**raw) if raw else LLMUsage()
        totals[0] += usage.input_tokens
        totals[1] += usage.output_tokens
        totals[2] += usage.cache_read_input_tokens
        totals[3] += usage.cache_creation_input_tokens
        totals[4] += usage.reasoning_tokens
        provider = provider or usage.provider
    return LLMUsage(
        input_tokens=totals[0],
        output_tokens=totals[1],
        cache_read_input_tokens=totals[2],
        cache_creation_input_tokens=totals[3],
        reasoning_tokens=totals[4],
        provider=provider,
        model=model,
    )


def run_live(
    spec: PeetersReplicationSpec,
    model: str,
    *,
    budget_usd: float,
    prices: dict[str, tuple[float, float]] = PRICES_PER_1M,
    client: Any = None,
) -> dict[str, Any]:
    """Judge every sampled pair with a live ``LLMJudge`` under a hard spend cap.

    Builds the judge with the paper's ``domain-complex-force`` template, the
    Peeters ``record_serializer``, ``response_parser=parse_binary_yes_no`` and
    ``temperature=0.0``, then streams ``forward`` over the candidates — charging
    each judgement's REAL (OpenRouter-billed) cost to a
    :class:`SpendMonitor` and stopping the moment cumulative spend crosses
    ``budget_usd``. Computes pairwise P/R/F1 at threshold 0.5 (no sweep — the
    protocol is binary) and aggregates the token-usage vector + billed cost.

    ``client`` is injectable (a fake returning "Yes"/"No") so the whole flow is
    verified at **$0** in tests; ``None`` lets ``LLMJudge`` build a real litellm
    client from the environment (the paid path).
    """
    template = build_llm_prompt_template(spec)
    serializer = make_record_serializer(spec)
    candidates = build_candidates(spec)
    gold = {
        frozenset({left_id, right_id})
        for left_id, right_id, label in load_peeters_sample(spec)
        if label == 1
    }

    judge: LLMJudge[Any] = LLMJudge(
        client=client,
        model=model,
        temperature=0.0,
        prompt_template=template,
        record_serializer=serializer,
        response_parser=parse_binary_yes_no,
    )

    monitor = SpendMonitor(budget_usd=budget_usd)
    judgements: list[Any] = []
    real_cost = 0.0
    cost_is_real = True
    budget_hit = False
    for judgement in judge.forward(iter(candidates)):
        judgements.append(judgement)
        cost = float(judgement.provenance.get("cost_usd") or 0.0)
        real_cost += cost
        cost_is_real = cost_is_real and bool(judgement.provenance.get("cost_is_real"))
        monitor.add(cost)
        try:
            monitor.check()
        except BudgetExceeded:
            budget_hit = True
            logger.warning("budget cap hit after %d/%d pairs", len(judgements), len(candidates))
            break

    metrics = classify_pairs(judgements, gold, threshold=0.5)
    usage = _aggregate_usage(judgements, model)
    n_judged = len(judgements)
    return {
        "model": model,
        "published_f1": PAID_MODELS.get(model),
        "n_pairs": len(candidates),
        "n_judged": n_judged,
        "budget_hit": budget_hit,
        "f1": metrics.f1 * 100.0,
        "precision": metrics.precision * 100.0,
        "recall": metrics.recall * 100.0,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "usage": usage.model_dump(),
        "real_cost_usd": real_cost,
        "cost_is_real": cost_is_real and n_judged > 0,
        "usd_per_1k_pairs": (real_cost / n_judged * 1000.0) if n_judged else 0.0,
    }


def _print_comparison(results: Sequence[dict[str, Any]]) -> None:
    """Print the per-model comparison table (F1 vs published, usage, real cost)."""
    print("\n" + "=" * 96)
    print("Peeters LLM-EM LIVE — abt-buy / domain-complex-force")
    print("=" * 96)
    header = (
        f"{'model':40} {'F1':>7} {'pub':>7} {'Δ':>6} {'P':>7} {'R':>7} "
        f"{'in_tok':>9} {'out_tok':>8} {'cost$':>8} {'$/1k':>8} real"
    )
    print(header)
    print("-" * 96)
    for r in results:
        pub = r["published_f1"]
        delta = f"{r['f1'] - pub:+.2f}" if pub is not None else "—"
        pub_s = f"{pub:.2f}" if pub is not None else "—"
        usage = r["usage"]
        print(
            f"{r['model']:40} {r['f1']:7.2f} {pub_s:>7} {delta:>6} "
            f"{r['precision']:7.2f} {r['recall']:7.2f} "
            f"{usage['input_tokens']:9d} {usage['output_tokens']:8d} "
            f"{r['real_cost_usd']:8.4f} {r['usd_per_1k_pairs']:8.4f} {str(r['cost_is_real']):>5}"
        )
        if r["budget_hit"]:
            print(f"  ! budget cap hit after {r['n_judged']}/{r['n_pairs']} pairs (partial)")
    print("=" * 96)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_paid_models(requested: str | None) -> list[str]:
    """Resolve the ``--model`` selection to a list of PAID_MODELS ids ('both' = all)."""
    if requested in (None, "both"):
        return list(PAID_MODELS)
    if requested not in PAID_MODELS:
        raise SystemExit(
            f"[fatal] {requested!r} is not a paid-run model. Choose one of "
            f"{sorted(PAID_MODELS)} or 'both'."
        )
    return [requested]


def _assert_priced(models: Sequence[str]) -> None:
    """Refuse to start unless every model has a PRICES_PER_1M entry (else the cap is blind)."""
    unpriced = [m for m in models if m not in PRICES_PER_1M]
    if unpriced:
        raise SystemExit(
            f"[fatal] no PRICES_PER_1M entry for {unpriced} — an unpriced model silently "
            f"contributes $0 to the spend cap. Pin its price in "
            f"langres.clients.openrouter.PRICES_PER_1M first. Known: {sorted(PRICES_PER_1M)}"
        )


def _run_replay_mode(args: argparse.Namespace) -> int:
    print("Offline replay — NO API key, NO LLM call, $0 spend. Replaying archived answers.\n")
    replay(args.dataset, args.model or "gpt-4-0613", args.prompt_design, args.cache_dir)
    return 0


def _run_dry_run_mode(args: argparse.Namespace) -> int:
    spec = get_peeters_replication(args.dataset)
    models = _resolve_paid_models(args.model)
    _assert_priced(models)
    print("Dry run — NO API key, NO LLM call, $0 spend. Rendering prompts + counting tokens.\n")
    total_est = 0.0
    for model in models:
        report = dry_run(spec, model)
        total_est += report["est_usd"]
        print(
            f"{model:40}  pairs={report['n_pairs']}  "
            f"input_tokens={report['input_tokens']}  "
            f"(mean {report['mean_input_tokens']:.1f}, max {report['max_input_tokens']})  "
            f"output_tokens≈{report['output_tokens_est']}  est=${report['est_usd']:.4f}"
        )
    print(
        f"\nestimated total for {len(models)} model(s): ${total_est:.4f}  "
        f"(budget ${args.budget:.2f})"
    )
    return 0


def _run_live_mode(args: argparse.Namespace) -> int:
    import os

    from dotenv import load_dotenv

    spec = get_peeters_replication(args.dataset)
    models = _resolve_paid_models(args.model)
    _assert_priced(models)

    # Print the estimate BEFORE any network call, so the operator sees the cost.
    print("Estimating cost (dry run, $0) before any spend...\n")
    total_est = 0.0
    for model in models:
        report = dry_run(spec, model)
        total_est += report["est_usd"]
        print(f"  {model:40}  input_tokens={report['input_tokens']}  est=${report['est_usd']:.4f}")
    print(f"\nestimated total: ${total_est:.4f}  |  hard cap: ${args.budget:.2f}\n")

    if not args.yes_spend_money:
        print(
            "[refused] --mode live is a PAID run. Re-run with --yes-spend-money to proceed "
            "(no network call was made)."
        )
        return 1

    load_dotenv(".env")  # OPENROUTER_API_KEY lives in .env, not Settings.
    if "OPENROUTER_API_KEY" not in os.environ:
        print(
            "[fatal] OPENROUTER_API_KEY is not set. The paid path fails fast rather than "
            "falling back to another provider or an unpriced model. Set it and retry."
        )
        return 1

    results: list[dict[str, Any]] = []
    per_model_budget = args.budget / len(models)
    try:
        for model in models:
            print(
                f"[live] judging {spec.name} with {model} "
                f"(per-model cap ${per_model_budget:.2f})..."
            )
            results.append(run_live(spec, model, budget_usd=per_model_budget))
    except BudgetExceeded as exc:
        print(f"[stopped] budget cap fired: {exc}")
        return 2

    _print_comparison(results)
    if args.results_path:
        out = Path(args.results_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"[report] wrote {out}")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--mode",
        choices=["replay", "dry-run", "live"],
        default="replay",
        help="replay archived answers ($0), dry-run the live path ($0), or run live (PAID).",
    )
    parser.add_argument(
        "--dataset",
        default="abt-buy",
        choices=list_peeters_replications(),
        help="Which replication slice (default: abt-buy).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "replay: archived model id (default gpt-4-0613). live/dry-run: one of "
            f"{sorted(PAID_MODELS)} or 'both' (default both)."
        ),
    )
    parser.add_argument(
        "--prompt-design",
        default="domain-complex-force",
        help="Archived prompt design (replay only; default domain-complex-force = Table 2 target).",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=DEFAULT_BUDGET_USD,
        help=f"Hard spend cap (USD) for live mode (default ${DEFAULT_BUDGET_USD:.2f}).",
    )
    parser.add_argument(
        "--yes-spend-money",
        action="store_true",
        help="Required to actually spend in --mode live (off by default).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "langres-peeters-replication",
        help="Where to download/extract the answer archive (replay mode; gitignored).",
    )
    parser.add_argument(
        "--results-path", default=None, help="Optional JSON path to write live results to."
    )
    args = parser.parse_args()

    if args.mode == "live" and args.budget > BUDGET_CEILING_USD:
        print(f"[fatal] --budget ${args.budget:.2f} exceeds the ${BUDGET_CEILING_USD:.2f} ceiling.")
        return 1

    if args.mode == "replay":
        return _run_replay_mode(args)
    if args.mode == "dry-run":
        return _run_dry_run_mode(args)
    return _run_live_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
