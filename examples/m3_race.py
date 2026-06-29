"""M3 Wave 4 — the paid, resumable multi-method benchmark race (the EXIT run).

This is the one M3 script that spends money. It races all five resolution methods
(rapidfuzz · weighted_average · embedding_cosine · llm_judge · cascade) on two
benchmarks — Fodors-Zagat (saturated/easy) and Amazon-Google (hard, literature
SOTA pairwise-F1 ~0.5-0.75) — under a HARD $15 cap, and writes one JSON per
"cell" under ``data/benchmarks/m3/results/`` so a crash or interruption re-does
only the missing cells (paid work is never lost). Each cell is git-committed as
soon as it completes.

Two evaluation surfaces (see ``docs/POC.md`` / the M3 plan):

* **Primary pair-level judge ranking.** For Amazon-Google we judge the FIXED
  literature ``test`` pair split (2293 pairs, 234 positives) directly — no
  blocking, so the number is literature-comparable to DeepMatcher/Ditto. For
  Fodors-Zagat we judge the blocked TEST band. Both grade with
  ``evaluate_judge_on_candidates`` (pair P/R/F1 at the best-F1 grid threshold +
  the PR curve).
* **Pipeline (production-realistic).** Zero-spend methods run the full
  ``run_method`` on BOTH datasets across 5 seeds (free → mean ± std). The LLM
  judge additionally clusters its FZ-band judgements (single seed) for the
  production story. The AG end-to-end LLM pipeline is deliberately SKIPPED (the
  48k blocked band is un-affordable AND recall-capped ~0.84); the AG fixed-pair
  eval is the AG LLM signal.

Budget safety: every paid judge runs through ``BudgetedModuleRunner`` (pre-flight
pair cap + per-pair tally stop), and each cell's hard ceiling is the REMAINING
real budget (``$15`` minus what committed cells already spent), so the global cap
holds across resumes. Costs are made honest by patching ``litellm.model_cost``
for the OpenRouter models (litellm otherwise prices them at $0).

Usage (run with the sandbox disabled — OpenRouter + HF-hub are network calls)::

    # 0. Verify the model ids respond and report honest cost — ONE call each.
    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE uv run python examples/m3_race.py --smoke

    # 1. Run the race (resumable; skips cells whose JSON already exists).
    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE uv run python examples/m3_race.py

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
import time  # noqa: E402
from collections.abc import Callable, Sequence  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from langres.core.benchmark import (  # noqa: E402
    BudgetedModuleRunner,
    CostTrack,
    JudgePairEval,
    _cost_track,
    _pipeline_track,
    evaluate_judge_on_candidates,
    gold_pairs_from_clusters,
    run_method,
)
from langres.core.embeddings import SentenceTransformerEmbedder  # noqa: E402
from langres.core.models import ERCandidate, PairwiseJudgement  # noqa: E402
from langres.data.amazon_google import (  # noqa: E402
    AmazonGoogleBenchmark,
    ProductSchema,
    load_amazon_google,
    load_amazon_google_pair_splits,
)
from langres.data.er_benchmarks import FodorsZagatBenchmark, RestaurantSchema  # noqa: E402
from langres.methods import (  # noqa: E402
    ZERO_SPEND_METHODS,
    cascade_cost_track,
    make_resolver_factory,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("data/benchmarks/m3/results")
HARD_BUDGET_USD = 15.0
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: Wide pair-level threshold grid (the shared race grid caps at 0.80, which
#: unfairly collapses score-based judges whose raw scores exceed 0.80; we sweep
#: up to 0.99 and report the full PR curve).
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

#: Clusterer threshold the LLM FZ-band pipeline cell clusters at. LLM scores are
#: probabilities, so 0.5 is the natural decision boundary; we avoid spending on
#: train-tuning (the production story only needs one test pass).
LLM_PIPELINE_CLUSTER_THRESHOLD = 0.5

GLM_MODEL = "openrouter/z-ai/glm-5.2"
GLM_MODEL_FALLBACK = "openrouter/z-ai/glm-4.6"  # M1-proven fallback.

#: Frontier candidates tried in order (first that responds wins; recorded).
FRONTIER_CANDIDATES: tuple[str, ...] = (
    "openrouter/openai/gpt-4o",
    "openrouter/anthropic/claude-3.7-sonnet",
    "openrouter/anthropic/claude-3.5-sonnet",
    "openrouter/google/gemini-2.0-flash-001",
)

#: Published per-1M-token (input, output) USD prices, pinned so cost is honest
#: even when litellm prices a model at $0. GLM rates are the M1-pinned OpenRouter
#: anchors; frontier rates are each provider's published list price.
PRICES_PER_1M: dict[str, tuple[float, float]] = {
    "openrouter/z-ai/glm-5.2": (0.95, 3.00),
    "openrouter/z-ai/glm-4.6": (0.60, 2.20),
    "openrouter/openai/gpt-4o": (2.50, 10.00),
    "openrouter/anthropic/claude-3.7-sonnet": (3.00, 15.00),
    "openrouter/anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "openrouter/google/gemini-2.0-flash-001": (0.10, 0.40),
}

#: Worst-case total tokens per pair, used to size the pre-flight budget cap.
WORST_CASE_TOKENS_PER_PAIR = 1500

#: Frontier AG fixed-pair subsample size (frontier is pricier — bound it).
FRONTIER_AG_SUBSAMPLE = 700

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: Per-call timeout (s) and retry count for the LLM clients. litellm defaults to a
#: 6000s timeout, so a stalled keep-alive socket can hang the whole sequential run;
#: a bounded timeout + retries makes a stalled call fail fast and recover.
LLM_TIMEOUT_S = 60.0
LLM_NUM_RETRIES = 2


# ---------------------------------------------------------------------------
# Pricing patch + clients
# ---------------------------------------------------------------------------


def patch_litellm_prices(model: str) -> None:
    """Patch ``litellm.model_cost`` so ``completion_cost`` is honest for ``model``.

    litellm prices many OpenRouter models at $0 (unknown), which would silently
    report $0 spend and (for the budget tally) hide real cost. We write the pinned
    per-token price under both the litellm-routing key (``openrouter/...``) and the
    bare provider key (``z-ai/glm-5.2``) that an OpenAI/OpenRouter response carries
    in ``response.model``, so both the LiteLLM (llm_judge) and OpenAI-client
    (cascade) cost paths resolve.
    """
    import litellm

    in_per_1m, out_per_1m = PRICES_PER_1M[model]
    entry = {
        "input_cost_per_token": in_per_1m / 1_000_000.0,
        "output_cost_per_token": out_per_1m / 1_000_000.0,
        "litellm_provider": "openrouter",
        "mode": "chat",
    }
    litellm.model_cost[model] = entry
    bare = model.split("/", 1)[1] if "/" in model else model
    litellm.model_cost[bare] = entry


def register_runtime_model_price(model: str) -> str | None:
    """Probe ``model`` once and register its DATED response id into ``model_cost``.

    OpenRouter returns a *dated* model id (e.g. ``z-ai/glm-5.2-20260616``) in
    ``response.model``, and ``litellm.completion_cost(completion_response=r)`` (the
    no-override call both ``LLMJudge`` and ``CascadeModule`` make) resolves the
    price against that dated id — which is absent from litellm's table, so cost
    silently reports ``$0``. We make ONE cheap call, read the dated id, and pin the
    published price under it so both cost paths become honest. Returns the dated id
    (or ``None`` if the probe failed — the caller then tries a fallback model).
    """
    import litellm

    if model not in PRICES_PER_1M:
        return None
    patch_litellm_prices(model)
    try:
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0,
            max_tokens=1,
        )
    except Exception as exc:  # noqa: BLE001 — probe; caller falls back on None
        print(f"[price] probe failed for {model}: {type(exc).__name__}: {exc}")
        return None
    dated = str(resp.model)
    in_per_1m, out_per_1m = PRICES_PER_1M[model]
    litellm.model_cost[dated] = {
        "input_cost_per_token": in_per_1m / 1_000_000.0,
        "output_cost_per_token": out_per_1m / 1_000_000.0,
        "litellm_provider": "openrouter",
        "mode": "chat",
    }
    print(f"[price] {model}: registered dated id {dated!r} at ${in_per_1m}/${out_per_1m} per 1M")
    return dated


class _TimeoutLiteLLM:
    """Wrap the LiteLLM module so every ``completion`` gets a timeout + retries.

    ``LLMJudge`` calls ``client.completion(model=, messages=, temperature=)`` with
    no timeout, and ``litellm.request_timeout`` defaults to 6000s — so a single
    stalled keep-alive socket hangs the whole sequential run indefinitely (observed
    in the first paid attempt). This proxy injects a bounded ``timeout`` and
    ``num_retries`` on every call, so a stalled call fails fast, retries on a fresh
    connection, and (if it still fails) is skipped by the BudgetedModuleRunner —
    the run can never wedge on one bad socket. All other attributes pass through.
    """

    def __init__(self, inner: Any, *, timeout: float, num_retries: int) -> None:
        self._inner = inner
        self._timeout = timeout
        self._num_retries = num_retries

    def completion(self, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", self._timeout)
        kwargs.setdefault("num_retries", self._num_retries)
        return self._inner.completion(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def make_litellm_client() -> Any:
    """LiteLLM client (for ``llm_judge``), Langfuse off, with a bounded timeout."""
    from langres.clients import Settings, create_llm_client

    inner = create_llm_client(Settings(), enable_langfuse=False)
    return _TimeoutLiteLLM(inner, timeout=LLM_TIMEOUT_S, num_retries=LLM_NUM_RETRIES)


def make_openai_client() -> Any:
    """OpenAI-shaped client pointed at OpenRouter (for ``cascade``), bounded timeout."""
    from openai import OpenAI

    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=LLM_TIMEOUT_S,
        max_retries=LLM_NUM_RETRIES,
    )


def per_token_worst_price(model: str) -> float:
    """Worst-case per-token price (the dearer of input/output) for the budget cap."""
    in_per_1m, out_per_1m = PRICES_PER_1M[model]
    return max(in_per_1m, out_per_1m) / 1_000_000.0


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def _cosine_01(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of two (already-normalized) embeddings, mapped to [0, 1].

    Mirrors ``FAISSIndex.to_similarities`` for the cosine metric (``(ip + 1) / 2``)
    so a fixed-pair ``embedding_cosine`` score matches what the VectorBlocker would
    have attached during blocking.
    """
    ip = float(np.dot(a, b))
    return max(0.0, min(1.0, (ip + 1.0) / 2.0))


def build_ag_fixed_candidates(
    split: str = "test",
) -> tuple[list[ERCandidate[ProductSchema]], set[frozenset[str]]]:
    """Build ER candidates from the FIXED Amazon-Google literature pair split.

    Each split row ``(amazon_id, google_id, label)`` becomes one candidate whose
    ``similarity_score`` is the MiniLM cosine of the two records' ``embed_text``
    (so ``embedding_cosine`` works without a blocker). Returns the candidates plus
    the positive gold pairs (``label == 1``). No blocking is involved, so pair-level
    P/R/F1 over these is the literature-comparable AG number.
    """
    corpus, _gc, _gp = load_amazon_google()
    by_id: dict[str, ProductSchema] = {r.id: r for r in corpus}
    rows = load_amazon_google_pair_splits()[split]

    # Embed every record that appears in the split once (normalized MiniLM).
    referenced = {rid for a, g, _ in rows for rid in (a, g)}
    ordered_ids = sorted(referenced)
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    vecs = embedder.encode([by_id[rid].embed_text for rid in ordered_ids])
    emb: dict[str, np.ndarray] = {rid: vecs[i] for i, rid in enumerate(ordered_ids)}

    candidates: list[ERCandidate[ProductSchema]] = []
    gold: set[frozenset[str]] = set()
    for a, g, label in rows:
        candidates.append(
            ERCandidate(
                left=by_id[a],
                right=by_id[g],
                blocker_name="ag_fixed_pairs",
                similarity_score=_cosine_01(emb[a], emb[g]),
            )
        )
        if label == 1:
            gold.add(frozenset({a, g}))
    return candidates, gold


def build_fz_band_candidates(
    seed: int = 0,
) -> tuple[list[ERCandidate[RestaurantSchema]], set[frozenset[str]], list[set[str]], list[str]]:
    """Build the blocked Fodors-Zagat TEST band (vector blocker, k pinned).

    Returns ``(candidates, gold_pairs, test_truth_clusters, all_test_ids)``: the
    full blocker output over the test split (intra- and cross-source, matching
    ``run_method``'s band), the cross-source gold match pairs, the closed-world
    truth partition for the test split, and every test record id (for the
    pipeline's singleton completion). The candidates carry ``similarity_score`` so
    ``embedding_cosine`` works.
    """
    bench = FodorsZagatBenchmark()
    corpus, gold_clusters, _gp = bench.load()
    _train, test, _trc, test_clusters = bench.split(corpus, gold_clusters, seed=seed)

    blocker = bench.build_blocker(bench.blocking_k)
    test_dicts = [r.model_dump() for r in test]
    entities = [blocker.schema_factory(d) for d in test_dicts]
    blocker.vector_index.create_index([blocker.text_field_extractor(e) for e in entities])
    candidates = list(blocker.stream(test_dicts))

    gold_pairs = gold_pairs_from_clusters(test_clusters)
    all_ids = [r.id for r in test]
    return candidates, gold_pairs, test_clusters, all_ids


def _attach_comparison(
    candidates: Sequence[ERCandidate[Any]], comparator: Any
) -> list[ERCandidate[Any]]:
    """Attach a Comparator's per-feature vector (needed by ``weighted_average``)."""
    return [
        c.model_copy(update={"comparison": comparator.compare(c.left, c.right)}) for c in candidates
    ]


def subsample_stratified(
    candidates: Sequence[ERCandidate[Any]],
    gold: set[frozenset[str]],
    n: int,
    *,
    seed: int = 0,
) -> list[ERCandidate[Any]]:
    """Deterministic label-stratified subsample of ``n`` candidates.

    Keeps the positive fraction stable so a subsampled recall estimate is not
    biased. A candidate is positive iff its ``frozenset({left.id, right.id})`` is
    in ``gold``.
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


# ---------------------------------------------------------------------------
# Budget ledger
# ---------------------------------------------------------------------------


def spent_so_far() -> float:
    """Sum the recorded spend of every committed cell JSON (resume-safe ledger)."""
    total = 0.0
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        total += float(data.get("usd_total", 0.0))
    return total


# ---------------------------------------------------------------------------
# Cell runners — each returns a JSON-able dict (the persisted cell result)
# ---------------------------------------------------------------------------


def _budget_runner(module: Any, model: str) -> BudgetedModuleRunner:
    """A budget runner sized from the REMAINING real budget (global cap held)."""
    remaining = HARD_BUDGET_USD - spent_so_far()
    hard = max(0.02, remaining)
    soft = max(0.01, hard - 0.25)
    return BudgetedModuleRunner(
        module,
        budget_usd=hard,
        budget_soft_usd=soft,
        worst_case_units_per_pair=float(WORST_CASE_TOKENS_PER_PAIR),
    )


def _judge_eval_to_dict(
    cell_id: str,
    kind: str,
    method: str,
    dataset: str,
    model: str | None,
    result: JudgePairEval,
) -> dict[str, Any]:
    """Wrap a JudgePairEval as a persisted cell dict (ledger reads ``usd_total``)."""
    return {
        "cell_id": cell_id,
        "kind": kind,
        "method": method,
        "dataset": dataset,
        "model": model,
        "usd_total": result.cost.usd_total,
        "eval": result.model_dump(),
    }


def run_agfixed_cell(
    method: str,
    *,
    model: str | None,
    candidates: list[ERCandidate[ProductSchema]],
    gold: set[frozenset[str]],
) -> dict[str, Any]:
    """Pair-level eval of one method's judge on the AG fixed candidate set."""
    llm_client = _client_for(method, model)
    factory = make_resolver_factory(
        method, AmazonGoogleBenchmark(), llm_client=llm_client, llm_model=model or "unused"
    )
    resolver = factory(0.5)
    cands: Sequence[ERCandidate[Any]] = candidates
    if resolver.comparator is not None:
        cands = _attach_comparison(candidates, resolver.comparator)

    runner = _budget_runner(resolver.module, model) if model else None
    cost_fn: Callable[[list[PairwiseJudgement]], CostTrack] = (
        cascade_cost_track if method == "cascade" else _cost_track
    )
    result, _judgements = evaluate_judge_on_candidates(
        resolver.module,
        cands,
        gold,
        GRID,
        runner=runner,
        price_per_token_or_pair=per_token_worst_price(model) if model else 0.0,
        cost_track_fn=cost_fn,
    )
    cell_id = _agfixed_cell_id(method, model)
    return _judge_eval_to_dict(cell_id, "agfixed", method, "amazon_google", model, result)


def run_fzband_cell(
    method: str,
    *,
    model: str | None,
    band: list[ERCandidate[RestaurantSchema]],
    gold: set[frozenset[str]],
    test_clusters: list[set[str]],
    all_ids: list[str],
) -> dict[str, Any]:
    """Pair-level (+ pipeline for the judge) eval of one method on the FZ band."""
    llm_client = _client_for(method, model)
    factory = make_resolver_factory(
        method, FodorsZagatBenchmark(), llm_client=llm_client, llm_model=model or "unused"
    )
    resolver = factory(LLM_PIPELINE_CLUSTER_THRESHOLD)
    cands: Sequence[ERCandidate[Any]] = band
    if resolver.comparator is not None:
        cands = _attach_comparison(band, resolver.comparator)

    runner = _budget_runner(resolver.module, model) if model else None
    cost_fn: Callable[[list[PairwiseJudgement]], CostTrack] = (
        cascade_cost_track if method == "cascade" else _cost_track
    )
    result, judgements = evaluate_judge_on_candidates(
        resolver.module,
        cands,
        gold,
        GRID,
        runner=runner,
        price_per_token_or_pair=per_token_worst_price(model) if model else 0.0,
        cost_track_fn=cost_fn,
    )

    cell_id = _fzband_cell_id(method, model)
    out = _judge_eval_to_dict(cell_id, "fzband", method, "fodors_zagat", model, result)

    # Pipeline track only for the LLM judge: clustering at 0.5 is the natural
    # probability boundary for it, whereas zero-spend methods get their tuned
    # pipeline from ``run_method`` (clustering them at 0.5 would be meaningless).
    if model is not None:
        predicted = resolver.clusterer.cluster(iter(judgements))
        pipeline = _pipeline_track(predicted, all_ids, test_clusters)
        out["pipeline"] = pipeline.model_dump()
        out["pipeline_cluster_threshold"] = LLM_PIPELINE_CLUSTER_THRESHOLD
    return out


def run_pipeline_cell(method: str, bench: Any, dataset: str) -> dict[str, Any]:
    """Zero-spend full-pipeline eval via ``run_method`` across all seeds (free)."""
    results = []
    for seed in SEEDS:
        factory = make_resolver_factory(method, bench)
        results.append(run_method(bench, factory, seed=seed, budget=0.0).model_dump())
    cell_id = f"pipeline_{dataset}_{method}"
    return {
        "cell_id": cell_id,
        "kind": "pipeline",
        "method": method,
        "dataset": dataset,
        "model": None,
        "usd_total": 0.0,
        "seeds": list(SEEDS),
        "results": results,
    }


def _client_for(method: str, model: str | None) -> Any:
    """Pick the correctly-shaped LLM client for a method (None for zero-spend)."""
    if model is None or method in ZERO_SPEND_METHODS:
        return None
    return make_openai_client() if method == "cascade" else make_litellm_client()


def _agfixed_cell_id(method: str, model: str | None) -> str:
    suffix = "_frontier" if (model and model not in (GLM_MODEL, GLM_MODEL_FALLBACK)) else ""
    return f"agfixed_{method}{suffix}"


def _fzband_cell_id(method: str, model: str | None) -> str:
    suffix = "_frontier" if (model and model not in (GLM_MODEL, GLM_MODEL_FALLBACK)) else ""
    return f"fzband_{method}{suffix}"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def smoke(glm_model: str) -> int:
    """One real call per model id; verify they respond and report honest cost.

    Picks the first frontier candidate that responds. Prints token usage and the
    (patched) litellm cost so the operator can confirm cost is non-zero before any
    band-sized spend. Returns a process exit code.
    """
    from langres.core.modules.llm_judge import LLMJudge

    corpus, _gc, _gp = load_amazon_google()
    a = next(r for r in corpus if r.source == "amazon")
    g = next(r for r in corpus if r.source == "google")
    candidate: ERCandidate[ProductSchema] = ERCandidate(left=a, right=g, blocker_name="smoke")

    client = make_litellm_client()

    def _one(model: str) -> float | None:
        register_runtime_model_price(model)
        judge: LLMJudge[Any] = LLMJudge(client=client, model=model, entity_noun="product")
        try:
            js = list(judge.forward(iter([candidate])))
        except Exception as exc:  # noqa: BLE001 — smoke probe, report and continue
            print(f"[smoke] {model}: FAILED ({type(exc).__name__}: {exc})")
            return None
        if not js:
            print(f"[smoke] {model}: no judgement")
            return None
        prov = js[0].provenance
        pt = int(prov.get("prompt_tokens", 0) or 0)
        ct = int(prov.get("completion_tokens", 0) or 0)
        cost = float(prov.get("cost_usd", 0.0) or 0.0)
        print(
            f"[smoke] {model}: score={js[0].score:.3f} prompt_tokens={pt} "
            f"completion_tokens={ct} cost_usd={cost:.6f}"
        )
        return cost

    print(f"[smoke] GLM probe: {glm_model}")
    glm_cost = _one(glm_model)
    if glm_cost is None and glm_model == GLM_MODEL:
        print(f"[smoke] retrying GLM fallback: {GLM_MODEL_FALLBACK}")
        glm_cost = _one(GLM_MODEL_FALLBACK)
    if glm_cost is None:
        print("[smoke] FAIL: no GLM model responded.")
        return 1

    print("[smoke] frontier probe (first that responds wins):")
    frontier = pick_frontier(client, candidate)
    if frontier is None:
        print("[smoke] FAIL: no frontier model responded.")
        return 1
    print(f"[smoke] OK. GLM and frontier ({frontier}) both responded.")
    return 0


def pick_frontier(client: Any, candidate: ERCandidate[ProductSchema]) -> str | None:
    """Return the first frontier candidate that responds to a single probe call."""
    from langres.core.modules.llm_judge import LLMJudge

    for model in FRONTIER_CANDIDATES:
        patch_litellm_prices(model)
        judge: LLMJudge[Any] = LLMJudge(client=client, model=model, entity_noun="product")
        try:
            js = list(judge.forward(iter([candidate])))
        except Exception as exc:  # noqa: BLE001 — probe, try the next candidate
            print(f"[frontier] {model}: FAILED ({type(exc).__name__})")
            continue
        if js:
            prov = js[0].provenance
            print(
                f"[frontier] {model}: OK score={js[0].score:.3f} "
                f"cost_usd={float(prov.get('cost_usd', 0.0) or 0.0):.6f}"
            )
            return model
    return None


# ---------------------------------------------------------------------------
# Cell orchestration
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    """One unit of evaluation: an id, whether it spends, and a thunk producing its dict."""

    cell_id: str
    paid: bool
    run: Callable[[], dict[str, Any]]


def commit_cell(path: Path, cell_id: str, paid: bool) -> None:
    """Git-commit a freshly-written cell JSON (durability before the next spend)."""
    subprocess.run(["git", "add", str(path)], check=True)
    tag = "paid" if paid else "free"
    subprocess.run(
        [
            "git",
            "commit",
            "-q",
            "-m",
            f"M3 W4: race cell {cell_id} ({tag})\n\n"
            "Claude-Session: https://claude.ai/code/session_01Qik2ZVTbXtYt6rx2uLkYSd",
        ],
        check=True,
    )


def build_cells(glm_model: str, frontier_model: str | None) -> list[Cell]:
    """Assemble every race cell in priority order (free first, then paid)."""
    fz = FodorsZagatBenchmark()
    ag = AmazonGoogleBenchmark()

    # Lazily-built shared inputs (only constructed if a cell that needs them runs).
    cache: dict[str, Any] = {}

    def ag_fixed() -> tuple[list[ERCandidate[ProductSchema]], set[frozenset[str]]]:
        if "ag_fixed" not in cache:
            cache["ag_fixed"] = build_ag_fixed_candidates("test")
        return cache["ag_fixed"]

    def fz_band() -> tuple[Any, ...]:
        if "fz_band" not in cache:
            cache["fz_band"] = build_fz_band_candidates(seed=0)
        return cache["fz_band"]

    cells: list[Cell] = []

    # --- FREE (fast): zero-spend full pipeline on Fodors-Zagat, 5 seeds ---
    for method in ZERO_SPEND_METHODS:
        cells.append(
            Cell(
                f"pipeline_fodors_zagat_{method}",
                False,
                lambda m=method: run_pipeline_cell(m, fz, "fodors_zagat"),
            )
        )

    # --- FREE (fast): zero-spend pair-level on the AG fixed pairs + FZ band ---
    for method in ZERO_SPEND_METHODS:
        cells.append(
            Cell(
                f"agfixed_{method}",
                False,
                lambda m=method: run_agfixed_cell(
                    m, model=None, candidates=ag_fixed()[0], gold=ag_fixed()[1]
                ),
            )
        )
    for method in ZERO_SPEND_METHODS:
        cells.append(
            Cell(
                f"fzband_{method}",
                False,
                lambda m=method: run_fzband_cell(
                    m,
                    model=None,
                    band=fz_band()[0],
                    gold=fz_band()[1],
                    test_clusters=fz_band()[2],
                    all_ids=fz_band()[3],
                ),
            )
        )

    # --- PAID: GLM judges (priority order) ---
    cells.append(
        Cell(
            "agfixed_llm_judge",
            True,
            lambda: run_agfixed_cell(
                "llm_judge", model=glm_model, candidates=ag_fixed()[0], gold=ag_fixed()[1]
            ),
        )
    )
    cells.append(
        Cell(
            "fzband_llm_judge",
            True,
            lambda: run_fzband_cell(
                "llm_judge",
                model=glm_model,
                band=fz_band()[0],
                gold=fz_band()[1],
                test_clusters=fz_band()[2],
                all_ids=fz_band()[3],
            ),
        )
    )
    cells.append(
        Cell(
            "agfixed_cascade",
            True,
            lambda: run_agfixed_cell(
                "cascade", model=glm_model, candidates=ag_fixed()[0], gold=ag_fixed()[1]
            ),
        )
    )
    cells.append(
        Cell(
            "fzband_cascade",
            True,
            lambda: run_fzband_cell(
                "cascade",
                model=glm_model,
                band=fz_band()[0],
                gold=fz_band()[1],
                test_clusters=fz_band()[2],
                all_ids=fz_band()[3],
            ),
        )
    )

    # --- PAID: frontier llm_judge (AG subsampled, then FZ band) ---
    if frontier_model is not None:

        def _ag_frontier() -> dict[str, Any]:
            cands, gold = ag_fixed()
            sub = subsample_stratified(cands, gold, FRONTIER_AG_SUBSAMPLE, seed=0)
            out = run_agfixed_cell("llm_judge", model=frontier_model, candidates=sub, gold=gold)
            out["subsample_n"] = len(sub)
            out["subsample_of"] = len(cands)
            return out

        cells.append(Cell("agfixed_llm_judge_frontier", True, _ag_frontier))
        cells.append(
            Cell(
                "fzband_llm_judge_frontier",
                True,
                lambda: run_fzband_cell(
                    "llm_judge",
                    model=frontier_model,
                    band=fz_band()[0],
                    gold=fz_band()[1],
                    test_clusters=fz_band()[2],
                    all_ids=fz_band()[3],
                ),
            )
        )

    # --- FREE (slow): zero-spend full pipeline on Amazon-Google, 5 seeds ---
    # Last: each call re-embeds the ~3.2k-record train corpus several times, so
    # this is the slow tail — it must not delay the paid cells above.
    for method in ZERO_SPEND_METHODS:
        cells.append(
            Cell(
                f"pipeline_amazon_google_{method}",
                False,
                lambda m=method: run_pipeline_cell(m, ag, "amazon_google"),
            )
        )

    return cells


def run_race(glm_model: str, frontier_model: str | None, only: set[str] | None) -> int:
    """Run every missing cell, committing each as it lands. Returns an exit code."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cells = build_cells(glm_model, frontier_model)
    for cell in cells:
        if only is not None and cell.cell_id not in only:
            continue
        path = RESULTS_DIR / f"{cell.cell_id}.json"
        if path.exists():
            print(f"[skip] {cell.cell_id} (already committed)")
            continue
        if cell.paid:
            remaining = HARD_BUDGET_USD - spent_so_far()
            if remaining < 0.05:
                print(
                    f"[stop] budget exhausted (${spent_so_far():.4f} spent); skipping {cell.cell_id}"
                )
                continue
            print(f"[run] {cell.cell_id} — remaining budget ${remaining:.2f}")
        else:
            print(f"[run] {cell.cell_id} (free)")
        t0 = time.perf_counter()
        result = cell.run()
        path.write_text(json.dumps(result, indent=2))
        print(
            f"[done] {cell.cell_id} in {time.perf_counter() - t0:.1f}s "
            f"(cell cost ${float(result.get('usd_total', 0.0)):.4f}, "
            f"total ${spent_so_far():.4f})"
        )
        commit_cell(path, cell.cell_id, cell.paid)
    print(f"\n[race] complete. Total spend: ${spent_so_far():.4f} / ${HARD_BUDGET_USD:.2f}")
    return 0


def main() -> int:
    load_dotenv(".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("langres").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    p = argparse.ArgumentParser(description="M3 W4 paid multi-method benchmark race.")
    p.add_argument("--smoke", action="store_true", help="One call per model id, then exit.")
    p.add_argument("--glm-model", default=GLM_MODEL)
    p.add_argument("--frontier-model", default=None, help="Override frontier model id.")
    p.add_argument("--only", default=None, help="Comma-separated cell ids to run (others skipped).")
    args = p.parse_args()

    if "OPENROUTER_API_KEY" not in os.environ:
        print("[fatal] OPENROUTER_API_KEY not in environment (.env not loaded?).")
        return 1

    # Backstop the per-call proxy/client timeouts with a bounded global default
    # (litellm ships a 6000s default that hangs on a stalled socket).
    import litellm

    litellm.request_timeout = LLM_TIMEOUT_S

    if args.smoke:
        return smoke(args.glm_model)

    # Register the GLM model (probe + dated-id price patch) so cost is honest; fall
    # back to glm-4.6 if the primary id does not respond.
    glm_model = args.glm_model
    if register_runtime_model_price(glm_model) is None:
        print(f"[price] {glm_model} unavailable; falling back to {GLM_MODEL_FALLBACK}")
        glm_model = GLM_MODEL_FALLBACK
        if register_runtime_model_price(glm_model) is None:
            print("[fatal] no GLM model responded.")
            return 1

    frontier = args.frontier_model
    if frontier is None:
        corpus, _gc, _gp = load_amazon_google()
        a = next(r for r in corpus if r.source == "amazon")
        g = next(r for r in corpus if r.source == "google")
        probe: ERCandidate[ProductSchema] = ERCandidate(left=a, right=g, blocker_name="probe")
        frontier = pick_frontier(make_litellm_client(), probe)
    if frontier is not None:
        register_runtime_model_price(frontier)

    only = set(args.only.split(",")) if args.only else None
    return run_race(glm_model, frontier, only)


if __name__ == "__main__":
    raise SystemExit(main())
