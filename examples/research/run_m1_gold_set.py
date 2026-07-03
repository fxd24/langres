"""M1 Wave 5 — the paid EXIT run: label the Fodors-Zagat cross-source band with a
real budget-capped GLM teacher, then commit the gold set + report.

This is the one script in the repo that spends money. It is intentionally a
two-step tool so the price/token assumptions behind the budget cap are *verified*
before any band-sized spend (design-review B1/W1, and the "never burn a paid run
blind" lesson):

    # 1. Smoke-test: ONE real call. Prints the model id, token usage, and the
    #    cost litellm/OpenRouter reports, so we can pin honest prices + a safe
    #    worst-case token count. Costs a fraction of a cent.
    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=1 uv run python examples/research/run_m1_gold_set.py \
        --smoke --model openrouter/z-ai/glm-5.2

    # 2. Full run: labels the whole cross-source band under a hard $20 cap, using
    #    the prices pinned from step 1, and writes data/gold_sets/fodors_zagat/.
    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=1 uv run python examples/research/run_m1_gold_set.py \
        --model openrouter/z-ai/glm-5.2 \
        --price-prompt 0.95 --price-completion 3.00 \
        --worst-case-tokens 1500 --budget 20 --budget-soft 15

The teacher client is built with ``enable_langfuse=False`` (B4), so no Langfuse
creds are needed. ``OPENROUTER_API_KEY`` is loaded from ``.env`` into the process
environment (via ``load_dotenv``) so LiteLLM picks it up — it is NOT a declared
``Settings`` field. Run with the sandbox disabled — openrouter.ai is a network call.

``print`` is allowed in examples (this is an operator tool, not library code).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from langres.bootstrap import Bootstrapper, GoldSet, HardNegativeMiner, TeacherLabeler
from langres.core.blockers.vector import VectorBlocker
from langres.core.embeddings import SentenceTransformerEmbedder
from langres.core.indexes.vector_index import FAISSIndex
from langres.core.models import ERCandidate
from langres.data.er_benchmarks import (
    DEFAULT_BLOCKING_K,
    RestaurantSchema,
    load_fodors_zagat,
)

logger = logging.getLogger(__name__)

OUT_DIR = Path("data/gold_sets/fodors_zagat")


def _cross_source(candidate: ERCandidate[RestaurantSchema]) -> bool:
    """Keep only cross-source candidate pairs (Fodors x Zagat) — design-review B2."""
    return bool(candidate.left.source != candidate.right.source)


def _make_blocker(k_neighbors: int) -> VectorBlocker[RestaurantSchema]:
    """Real-embedding vector blocker (MiniLM + FAISS, cosine) — same wiring as the
    deterministic example, so blocking pair-completeness is honest."""
    return VectorBlocker(
        vector_index=FAISSIndex(
            embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
            metric="cosine",
        ),
        schema=RestaurantSchema,
        text_field="embed_text",
        k_neighbors=k_neighbors,
    )


def smoke(model: str) -> int:
    """Make ONE real teacher call on a real cross-source pair and print diagnostics.

    Prints token usage and the cost litellm reports, so the operator can pin
    honest ``--price-prompt``/``--price-completion`` and a safe ``--worst-case-tokens``
    for the full run. Returns a process exit code.
    """
    from langres.clients import Settings, create_llm_client
    from langres.core.modules.llm_judge import LLMJudge

    corpus_records, _ = load_fodors_zagat()
    fodors = next(r for r in corpus_records if r.source == "fodors")
    zagat = next(r for r in corpus_records if r.source == "zagat")
    candidate: ERCandidate[RestaurantSchema] = ERCandidate(
        left=fodors, right=zagat, blocker_name="smoke"
    )

    client = create_llm_client(Settings(), enable_langfuse=False)
    judge: LLMJudge[Any] = LLMJudge(client=client, model=model, entity_noun="restaurant")

    print(f"[smoke] model={model}")
    print(f"[smoke] pair: {fodors.name!r} (fodors)  vs  {zagat.name!r} (zagat)")
    judgements = list(judge.forward(iter([candidate])))
    if not judgements:
        print("[smoke] FAIL: judge yielded no judgement.")
        return 1
    j = judgements[0]
    prov = j.provenance
    pt = int(prov.get("prompt_tokens", 0) or 0)
    ct = int(prov.get("completion_tokens", 0) or 0)
    cost = float(prov.get("cost_usd", 0.0) or 0.0)
    print(f"[smoke] score={j.score:.3f}  prompt_tokens={pt}  completion_tokens={ct}")
    print(f"[smoke] litellm reported cost_usd={cost:.6f}")
    if pt == 0 and ct == 0:
        print("[smoke] WARN: no token usage reported — the budget tally would be blind.")
        return 1
    suggested = max(1000, (pt + ct) * 2)
    print(
        f"[smoke] OK. Suggested --worst-case-tokens >= {suggested} "
        f"(2x measured {pt + ct}). If reported cost_usd is 0, pin prices from the "
        "provider's published per-1M rates; the tally then uses those pinned prices."
    )
    return 0


def run(
    *,
    model: str,
    price_prompt: float,
    price_completion: float,
    worst_case_tokens: int,
    budget: float,
    budget_soft: float,
    k_neighbors: int,
) -> int:
    """Run the full paid band-labeling and write the gold set + report. Returns an exit code."""
    corpus_records, gold_clusters = load_fodors_zagat()
    corpus = [r.model_dump() for r in corpus_records]

    teacher = TeacherLabeler.from_env(
        price_per_1m_prompt_tokens=price_prompt,
        price_per_1m_completion_tokens=price_completion,
        worst_case_tokens_per_pair=worst_case_tokens,
        model=model,
        entity_noun="restaurant",
        budget_usd=budget,
        budget_soft_usd=budget_soft,
    )
    bootstrapper = Bootstrapper(_make_blocker(k_neighbors), HardNegativeMiner(seed=0), teacher)

    print(f"[run] model={model}  budget=${budget:.2f} (soft ${budget_soft:.2f})")
    gold, report = bootstrapper.build(
        corpus, candidate_filter=_cross_source, gold_clusters=gold_clusters
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gold_path = OUT_DIR / "gold_set.json"
    report_path = OUT_DIR / "report.md"
    gold.save(gold_path)
    report_path.write_text(report.to_markdown(), encoding="utf-8")
    # Round-trip proof on the artifact we just wrote.
    assert GoldSet.load(gold_path).model_dump() == gold.model_dump(), "gold-set round-trip differs"

    cov = report.coverage
    print(
        f"[run] labeled={cov.labeled} skipped={teacher.skipped_count} "
        f"dropped_by_cap={teacher.dropped_by_cap_count}"
    )
    print(f"[run] honest cost=${teacher.total_spent_usd:.4f}  (cap ${budget:.2f})")
    print(f"[run] Pair-Completeness={report.blocking.pair_completeness:.4f}")
    if report.agreement is not None:
        a = report.agreement
        print(
            f"[run] teacher-vs-truth: acc={a.accuracy:.4f} F1={a.f1:.4f} "
            f"kappa={a.cohens_kappa:.4f} MCC={a.mcc:.4f} (n={a.n_evaluated})"
        )
    if report.calibration is not None:
        c = report.calibration
        print(f"[run] calibration: Brier={c.brier:.4f} ECE={c.ece:.4f} (n={c.n_evaluated})")
    print(f"[run] wrote {gold_path} and {report_path}")
    if cov.labeled == 0:
        print("[run] FAIL: 0 labeled — check the model id / API key.")
        return 1
    return 0


def main() -> int:
    # Load .env into the process environment so LiteLLM sees OPENROUTER_API_KEY
    # (it is not a declared Settings field). Explicit, not reliant on import side
    # effects.
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("langres").setLevel(logging.INFO)

    p = argparse.ArgumentParser(description="M1 Wave 5 paid GLM-teacher gold-set run.")
    p.add_argument("--smoke", action="store_true", help="One real call + diagnostics, then exit.")
    p.add_argument("--model", default="openrouter/z-ai/glm-5.2")
    p.add_argument("--price-prompt", type=float, default=0.95, help="USD per 1M prompt tokens.")
    p.add_argument(
        "--price-completion", type=float, default=3.00, help="USD per 1M completion tokens."
    )
    p.add_argument("--worst-case-tokens", type=int, default=1500)
    p.add_argument("--budget", type=float, default=20.0)
    p.add_argument("--budget-soft", type=float, default=15.0)
    p.add_argument("--k-neighbors", type=int, default=DEFAULT_BLOCKING_K)
    args = p.parse_args()

    if args.smoke:
        return smoke(args.model)
    return run(
        model=args.model,
        price_prompt=args.price_prompt,
        price_completion=args.price_completion,
        worst_case_tokens=args.worst_case_tokens,
        budget=args.budget,
        budget_soft=args.budget_soft,
        k_neighbors=args.k_neighbors,
    )


if __name__ == "__main__":
    raise SystemExit(main())
