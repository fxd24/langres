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
  F1 **90.95** (P=89.25, R=92.72).
* ``openrouter/openai/gpt-4o-2024-08-06`` — the paper's "GPT-4o", published F1
  **90.47** (P=83.27, R=99.03). (An earlier draft of this harness carried a wrong
  89.33 for this cell; arXiv v4 Table 2 and the authors' ``results.xlsx`` agree on
  90.47.)

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
* The one deviation from the paper's setup is that we route the same dated model
  snapshot through **OpenRouter** instead of calling OpenAI directly. The live
  judge therefore pins :data:`LIVE_PROVIDER`
  (``{"order": ["OpenAI"], "allow_fallbacks": False}``) so OpenRouter must serve
  the request from OpenAI's own backend and cannot silently substitute a
  different provider/quantization of the model.

Cheaper trials + per-pair agreement against the archive:

* ``--limit N`` runs a **stratified** subset of ``N`` pairs (preserving the
  17.1% positive ratio, deterministic under ``--seed``, default 0) instead of all
  1206 — so a live trial can be sized to a few cents. Applies to
  ``dry-run``/``live``/``replay``.
* ``--compare-archived`` (``--mode live`` only) judges each pair live **and**
  compares our parsed verdict to the authors' archived answer for the same model,
  reporting the per-pair agreement rate, a 2x2 confusion of ours-vs-theirs, up to
  10 concrete disagreeing pairs, our F1/P/R on the judged subset next to *their*
  F1/P/R recomputed on that same subset (and the published full-set number), plus
  the usage vector + real billed cost. It fails loudly if our rendered prompt does
  not match the archived one (a mismatch means the alignment is off).

Data licensing: MatchGPT ships no LICENSE (``license: null``); langres is
Apache-2.0. Nothing from MatchGPT is vendored — the ~186 MB answer archive is
downloaded transiently to a cache dir (``--cache-dir``) and never committed. Our
pair-set slice is regenerated from our own already-vendored DeepMatcher CSVs.

Usage::

    # $0, no key:
    uv run python examples/research/peeters_llm_em_replication.py
    uv run python examples/research/peeters_llm_em_replication.py --mode dry-run
    uv run python examples/research/peeters_llm_em_replication.py --mode dry-run --limit 150
    # PAID (run with the sandbox disabled — OpenRouter is a network call):
    uv run python examples/research/peeters_llm_em_replication.py --mode live --yes-spend-money
    # PAID, sized to a 150-pair stratified subset, with per-pair archive agreement:
    uv run python examples/research/peeters_llm_em_replication.py --mode live \\
        --model openrouter/openai/gpt-4o-mini-2024-07-18 --limit 150 \\
        --compare-archived --yes-spend-money

``print`` is allowed in examples (this is an operator tool).
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import random
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
    parse_binary_answer,
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


def replay(
    dataset: str,
    model: str,
    prompt_design: str,
    cache_dir: Path,
    *,
    limit: int | None = None,
    seed: int = 0,
) -> None:
    """Replay one archived run and report prompt round-trip + pairwise F1 ($0).

    ``limit`` runs a stratified subset (via :func:`stratified_subset_indices`) —
    handy for a quick look; the published-F1 assertion is a full-set claim, so it
    is reported (not asserted) when the run is limited.
    """
    spec = get_peeters_replication(dataset)
    prompts = render_sample_prompts(spec)  # our records + serializer, in sample order

    answers_path = ensure_answers(cache_dir, dataset, prompt_design, model)
    archived = [json.loads(line) for line in answers_path.read_text().splitlines() if line.strip()]

    if len(archived) != len(prompts):
        raise SystemExit(
            f"archive has {len(archived)} lines but our sample has {len(prompts)} pairs — "
            "alignment would be wrong; aborting."
        )

    if limit is not None:
        indices = stratified_subset_indices([p.label for p in prompts], limit, seed)
        prompts = [prompts[i] for i in indices]
        archived = [archived[i] for i in indices]

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
    if target is not None and limit is not None:
        print(
            f"published Table 2 F1 : {target:.2f}  (full-set claim — not asserted on a --limit subset)"
        )
    elif target is not None:
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
    "openrouter/openai/gpt-4o-2024-08-06": 90.47,  # paper "GPT-4o" (corrected from 89.33)
}

#: OpenRouter provider-routing block pinned on the live judge. Our sole deviation
#: from the paper's setup is routing the same dated snapshot through OpenRouter
#: rather than calling OpenAI directly; pinning ``order: ["OpenAI"]`` +
#: ``allow_fallbacks: False`` forces OpenRouter to serve the request from OpenAI's
#: own backend, so a different provider/quantization can't silently be substituted
#: and change the F1/cost we report. Sent as ``LLMJudge(provider=...)`` ->
#: ``extra_body["provider"]``.
LIVE_PROVIDER: dict[str, Any] = {"order": ["OpenAI"], "allow_fallbacks": False}

#: Hard spend cap (USD) for the two-model run combined. Measured total ≈ $0.29,
#: so $1.00 is a ~3.4x margin. ``main`` refuses anything above the ceiling.
DEFAULT_BUDGET_USD = 1.00
BUDGET_CEILING_USD = 2.00

#: Estimated output tokens per pair for the dry-run cost estimate: the binary
#: protocol answers with a single "Yes"/"No" word (~1–2 o200k tokens). The live
#: run meters the REAL output cost; this only sizes the pre-flight estimate.
_EST_OUTPUT_TOKENS_PER_PROMPT = 2


def stratified_subset_indices(labels: Sequence[int], limit: int, seed: int = 0) -> list[int]:
    """Pick ``limit`` sample-order indices, preserving the positive ratio, deterministically.

    The Peeters pair set is a *concatenation* of the positive block followed by the
    negative block (Abt-Buy: 206 positives at indices 0–205, then 1000 negatives),
    so naively taking the first ``N`` rows would return an all-positive (or, past
    206, positive-heavy) subset. This instead keeps ``round(limit * pos_ratio)``
    positives and the rest negatives — reproducing the full set's ~17.1% positive
    ratio — sampling each class with a seeded :class:`random.Random` so the same
    ``(limit, seed)`` always yields the same subset. Indices are returned in
    **ascending** (sample) order, so they can subset any sample-aligned sequence
    (prompts, candidates, archived answers) without breaking alignment.

    Args:
        labels: The full pair set's gold labels, in sample order (``1`` = match).
        limit: Target subset size. ``>= len(labels)`` returns every index.
        seed: RNG seed (default ``0``) making the choice reproducible.

    Returns:
        The chosen indices, ascending. Its length is ``limit`` unless a class is
        exhausted first (not the case for the shipped Abt-Buy/Amazon-Google sets
        at any ``limit <= 1206``).
    """
    n = len(labels)
    if limit >= n:
        return list(range(n))
    positives = [i for i, label in enumerate(labels) if label == 1]
    negatives = [i for i, label in enumerate(labels) if label == 0]
    n_pos = min(round(limit * len(positives) / n), len(positives), limit)
    n_neg = min(limit - n_pos, len(negatives))
    rng = random.Random(seed)
    return sorted(rng.sample(positives, n_pos) + rng.sample(negatives, n_neg))


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
    indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Render every pair through the LIVE path and count tokens — ZERO API calls.

    Renders each candidate with the exact ``build_llm_prompt_template`` +
    ``make_record_serializer`` the live ``LLMJudge`` uses, so the input-token
    total is the true billed input. Output tokens are estimated (the single-word
    binary answer). ``count_tokens`` is injectable for a dependency-free test;
    ``None`` uses the real litellm/tiktoken counter. ``indices`` (from
    :func:`stratified_subset_indices`) prices only that ``--limit`` subset;
    ``None`` prices all pairs.

    Returns a dict with the pair count, input/output token totals, mean/max input,
    and a cost estimate priced against ``prices[model]``.
    """
    template = build_llm_prompt_template(spec)
    serializer = make_record_serializer(spec)
    candidates = build_candidates(spec)
    if indices is not None:
        candidates = [candidates[i] for i in indices]
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


def _build_live_judge(
    spec: PeetersReplicationSpec, model: str, client: Any = None
) -> LLMJudge[Any]:
    """The live ``LLMJudge`` for a Peeters slice — the single build site.

    Wires the paper's ``domain-complex-force`` template, the Peeters per-dataset
    ``record_serializer``, ``response_parser=parse_binary_yes_no`` and
    ``temperature=0.0``, and pins :data:`LIVE_PROVIDER` so OpenRouter routes to
    OpenAI's own backend (our only deviation from the paper is the OpenRouter
    hop). Shared by :func:`run_live` and :func:`run_compare_archived` so the two
    paid paths judge with a byte-identical judge. ``client`` is injectable (a
    fake) so the whole flow is exercised at **$0** in tests.
    """
    return LLMJudge(
        client=client,
        model=model,
        temperature=0.0,
        prompt_template=build_llm_prompt_template(spec),
        record_serializer=make_record_serializer(spec),
        response_parser=parse_binary_yes_no,
        provider=LIVE_PROVIDER,
    )


def _judge_under_budget(
    judge: LLMJudge[Any], candidates: Sequence[Any], budget_usd: float
) -> tuple[list[Any], float, bool, bool]:
    """Stream ``judge.forward`` over ``candidates`` under a hard :class:`SpendMonitor`.

    Charges each judgement's REAL (OpenRouter-billed) cost to the monitor and
    stops the moment cumulative spend crosses ``budget_usd``. Returns
    ``(judgements, real_cost_usd, cost_is_real, budget_hit)`` — the judgements are
    in candidate order and truncated at the pair that tripped the cap. Shared by
    :func:`run_live` and :func:`run_compare_archived`.
    """
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
    return judgements, real_cost, cost_is_real, budget_hit


def run_live(
    spec: PeetersReplicationSpec,
    model: str,
    *,
    budget_usd: float,
    prices: dict[str, tuple[float, float]] = PRICES_PER_1M,
    client: Any = None,
    indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Judge every sampled pair with a live ``LLMJudge`` under a hard spend cap.

    Builds the judge via :func:`_build_live_judge` (paper template, Peeters
    serializer, ``parse_binary_yes_no``, ``temperature=0.0``, provider pinned),
    then streams it under a :class:`SpendMonitor` via :func:`_judge_under_budget`.
    Computes pairwise P/R/F1 at threshold 0.5 (no sweep — the protocol is binary)
    over the pairs actually judged and aggregates the token-usage vector + billed
    cost. ``indices`` (from :func:`stratified_subset_indices`) restricts the run to
    a ``--limit`` subset; ``None`` judges all pairs.

    ``client`` is injectable (a fake returning "Yes"/"No") so the whole flow is
    verified at **$0** in tests; ``None`` lets ``LLMJudge`` build a real litellm
    client from the environment (the paid path).
    """
    candidates = build_candidates(spec)
    sample = load_peeters_sample(spec)
    if indices is not None:
        candidates = [candidates[i] for i in indices]
        sample = [sample[i] for i in indices]

    judge = _build_live_judge(spec, model, client)
    judgements, real_cost, cost_is_real, budget_hit = _judge_under_budget(
        judge, candidates, budget_usd
    )
    n_judged = len(judgements)

    # Gold is restricted to the pairs actually judged (subset ∩ budget prefix), so
    # ``fn`` counts only positives the model was asked about and answered "no" to.
    gold = {
        frozenset({left_id, right_id})
        for left_id, right_id, label in sample[:n_judged]
        if label == 1
    }
    metrics = classify_pairs(judgements, gold, threshold=0.5)
    usage = _aggregate_usage(judgements, model)
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


# ---------------------------------------------------------------------------
# Per-pair agreement against the authors' archived answers (--compare-archived).
#
# The paid run above scores *our* F1 against the paper's published number. This
# goes finer-grained: for the exact model we run, the authors archived their raw
# per-pair answer, so we can check pair-by-pair whether our live verdict matches
# theirs — the rows a human reads when the aggregate F1 diverges from the paper.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparedPair:
    """One pair judged both ways: our live verdict vs. the authors' archived verdict.

    Attributes:
        left_id / right_id: Source-prefixed record ids.
        left / right: The serialized records the model actually saw.
        label: Gold label (``1`` = match, ``0`` = non-match).
        our_answer / our_verdict: Our raw model answer and its
            :func:`parse_binary_yes_no` verdict (``1``/``0``).
        their_answer / their_verdict: The authors' archived raw answer and its
            :func:`parse_binary_yes_no` verdict — both sides go through the one
            canonical parser so the comparison is apples-to-apples.
    """

    left_id: str
    right_id: str
    left: str
    right: str
    label: int
    our_answer: str
    our_verdict: int
    their_answer: str
    their_verdict: int


def load_archived_answers(
    spec: PeetersReplicationSpec,
    model: str,
    prompt_design: str,
    cache_dir: Path,
) -> list[dict[str, Any]]:
    """Fetch + parse the authors' archived ``{prompt, answer}`` rows for ``model``.

    Reuses the offline replay's :func:`ensure_answers` download/extract/cache path
    (no duplication) — mapping the OpenRouter routing id to the bare model id the
    archive member is keyed on (``openrouter/openai/gpt-4o-mini-2024-07-18`` →
    ``gpt-4o-mini-2024-07-18``). Rows are returned in the archive's line order,
    which matches the sampled pair set (hence :func:`stratified_subset_indices`
    can subset both consistently).
    """
    archived_model = model.split("/")[-1]  # archive members use the bare OpenAI id
    path = ensure_answers(cache_dir, spec.name, prompt_design, archived_model)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_compared_pairs(
    prompts: Sequence[Any],
    candidates: Sequence[Any],
    our_judgements: Sequence[Any],
    archived: Sequence[dict[str, Any]],
    serializer: Callable[[Any], str],
) -> list[ComparedPair]:
    """Zip the four sample-aligned streams into :class:`ComparedPair`\\ s, failing loud on drift.

    For each pair it asserts our rendered prompt equals the archived ``prompt``
    field — a mismatch means the alignment between our pair set and their archive
    is off, which would make *every* downstream comparison meaningless, so it
    raises :class:`SystemExit` with a unified diff rather than reporting garbage.
    Both verdicts come from :func:`parse_binary_yes_no` (ours already applied by
    the live judge; theirs via :func:`parse_binary_answer`).

    Raises:
        ValueError: If the four inputs are not the same length.
        SystemExit: On the first prompt that does not match its archived prompt.
    """
    n = len(prompts)
    if not (len(candidates) == n and len(our_judgements) == n and len(archived) == n):
        raise ValueError(
            f"compare inputs must align 1:1 (prompts={n}, candidates={len(candidates)}, "
            f"judgements={len(our_judgements)}, archived={len(archived)})"
        )
    compared: list[ComparedPair] = []
    for i in range(n):
        rendered = prompts[i].prompt
        archived_prompt = str(archived[i]["prompt"])
        if rendered != archived_prompt:
            diff = "\n".join(
                difflib.unified_diff(
                    archived_prompt.splitlines(),
                    rendered.splitlines(),
                    fromfile="archived",
                    tofile="ours",
                    lineterm="",
                )
            )
            raise SystemExit(
                f"[fatal] prompt mismatch at compared row {i} "
                f"({prompts[i].left_id} vs {prompts[i].right_id}) — alignment is off, so "
                f"every downstream comparison would be meaningless. Aborting.\n{diff}"
            )
        their_answer = str(archived[i]["answer"])
        our_answer = our_judgements[i].reasoning or ""
        compared.append(
            ComparedPair(
                left_id=prompts[i].left_id,
                right_id=prompts[i].right_id,
                left=serializer(candidates[i].left),
                right=serializer(candidates[i].right),
                label=prompts[i].label,
                our_answer=our_answer,
                our_verdict=int(our_judgements[i].score),
                their_answer=their_answer,
                their_verdict=parse_binary_answer(their_answer),
            )
        )
    return compared


def confusion_and_agreement(compared: Sequence[ComparedPair]) -> dict[str, Any]:
    """The per-pair agreement rate + the 2x2 confusion of our verdicts vs. theirs."""
    both_yes = sum(1 for c in compared if c.our_verdict == 1 and c.their_verdict == 1)
    both_no = sum(1 for c in compared if c.our_verdict == 0 and c.their_verdict == 0)
    we_yes_they_no = sum(1 for c in compared if c.our_verdict == 1 and c.their_verdict == 0)
    we_no_they_yes = sum(1 for c in compared if c.our_verdict == 0 and c.their_verdict == 1)
    n = len(compared)
    return {
        "n_compared": n,
        "agreement_rate": (both_yes + both_no) / n if n else 0.0,
        "confusion": {
            "both_yes": both_yes,
            "both_no": both_no,
            "we_yes_they_no": we_yes_they_no,
            "we_no_they_yes": we_no_they_yes,
        },
    }


def run_compare_archived(
    spec: PeetersReplicationSpec,
    model: str,
    *,
    prompt_design: str = "domain-complex-force",
    budget_usd: float,
    client: Any = None,
    archived: Sequence[dict[str, Any]] | None = None,
    cache_dir: Path | None = None,
    limit: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Judge a (optionally ``--limit``-ed) subset live and compare pair-by-pair to the archive.

    Runs the live judge over the stratified subset (:func:`_build_live_judge` +
    :func:`_judge_under_budget`), loads the authors' archived answers for the same
    model (:func:`load_archived_answers`, or the injected ``archived`` in tests),
    and reports: the per-pair agreement rate, the 2x2 confusion of ours vs. theirs,
    up to 10 concrete disagreeing pairs, our F1/P/R on the judged subset next to
    *their* F1/P/R recomputed on that same subset (plus the published full-set
    number), and the usage vector + real billed cost.

    The archived JSONL row count is asserted equal to the *full* pair-set count
    before any subsetting; per-compared-pair the rendered prompt is asserted equal
    to the archived one (:func:`build_compared_pairs` raises on a mismatch). The
    ``archived`` / ``client`` seams let the whole path run at **$0** in tests.

    Raises:
        SystemExit: On an archived-vs-pair-set row-count mismatch (or, via
            :func:`build_compared_pairs`, a per-pair prompt mismatch).
        ValueError: If ``archived is None`` and no ``cache_dir`` was given.
    """
    full_prompts = render_sample_prompts(spec)
    if archived is None:
        if cache_dir is None:
            raise ValueError("run_compare_archived needs either `archived` or a `cache_dir`")
        archived = load_archived_answers(spec, model, prompt_design, cache_dir)
    if len(archived) != len(full_prompts):
        raise SystemExit(
            f"[fatal] archived JSONL has {len(archived)} rows but the {spec.name} pair set "
            f"has {len(full_prompts)} — alignment would be wrong; aborting."
        )

    labels = [p.label for p in full_prompts]
    indices = (
        stratified_subset_indices(labels, limit, seed)
        if limit is not None
        else list(range(len(labels)))
    )
    full_candidates = build_candidates(spec)
    sub_prompts = [full_prompts[i] for i in indices]
    sub_candidates = [full_candidates[i] for i in indices]
    sub_archived = [archived[i] for i in indices]

    judge = _build_live_judge(spec, model, client)
    our_judgements, real_cost, cost_is_real, budget_hit = _judge_under_budget(
        judge, sub_candidates, budget_usd
    )
    n_judged = len(our_judgements)

    # Budget may have truncated the run: compare only the pairs actually judged.
    judged_prompts = sub_prompts[:n_judged]
    judged_candidates = sub_candidates[:n_judged]
    judged_archived = sub_archived[:n_judged]

    serializer = make_record_serializer(spec)
    compared = build_compared_pairs(
        judged_prompts, judged_candidates, our_judgements, judged_archived, serializer
    )
    summary = confusion_and_agreement(compared)

    gold = {frozenset({p.left_id, p.right_id}) for p in judged_prompts if p.label == 1}
    our_metrics = classify_pairs(list(our_judgements), gold, threshold=0.5)
    their_judgements = judgements_from_answers(
        judged_prompts, [str(a["answer"]) for a in judged_archived]
    )
    their_metrics = classify_pairs(their_judgements, gold, threshold=0.5)
    usage = _aggregate_usage(our_judgements, model)
    disagreements = [c for c in compared if c.our_verdict != c.their_verdict][:10]

    return {
        "model": model,
        "n_pairs": len(sub_candidates),
        "n_judged": n_judged,
        "budget_hit": budget_hit,
        "agreement_rate": summary["agreement_rate"],
        "confusion": summary["confusion"],
        "disagreements": [
            {
                "left_id": c.left_id,
                "right_id": c.right_id,
                "left": c.left,
                "right": c.right,
                "gold_label": c.label,
                "their_answer": c.their_answer,
                "our_answer": c.our_answer,
            }
            for c in disagreements
        ],
        "ours": {
            "f1": our_metrics.f1 * 100.0,
            "precision": our_metrics.precision * 100.0,
            "recall": our_metrics.recall * 100.0,
            "tp": our_metrics.tp,
            "fp": our_metrics.fp,
            "fn": our_metrics.fn,
        },
        "theirs_subset": {
            "f1": their_metrics.f1 * 100.0,
            "precision": their_metrics.precision * 100.0,
            "recall": their_metrics.recall * 100.0,
            "tp": their_metrics.tp,
            "fp": their_metrics.fp,
            "fn": their_metrics.fn,
        },
        "published_f1": PAID_MODELS.get(model),
        "usage": usage.model_dump(),
        "real_cost_usd": real_cost,
        "cost_is_real": cost_is_real and n_judged > 0,
        "usd_per_1k_pairs": (real_cost / n_judged * 1000.0) if n_judged else 0.0,
    }


def _print_archived_comparison(report: dict[str, Any]) -> None:
    """Print the per-pair archive-agreement report (agreement, confusion, disagreements)."""
    conf = report["confusion"]
    ours = report["ours"]
    theirs = report["theirs_subset"]
    pub = report["published_f1"]
    print("\n" + "=" * 96)
    print(f"Peeters LLM-EM ARCHIVE AGREEMENT — {report['model']}")
    print("=" * 96)
    print(
        f"pairs judged         : {report['n_judged']}/{report['n_pairs']}"
        + ("  (budget cap hit — partial)" if report["budget_hit"] else "")
    )
    print(f"per-pair agreement   : {report['agreement_rate'] * 100:.2f}%  (ours vs. theirs)")
    print("confusion (ours×theirs):")
    print(f"  both YES : {conf['both_yes']:5d}    both NO       : {conf['both_no']:5d}")
    print(
        f"  we-Y they-N: {conf['we_yes_they_no']:3d}    we-N they-Y   : {conf['we_no_they_yes']:5d}"
    )
    print("-" * 96)
    print(
        f"OURS   (judged subset): F1={ours['f1']:.2f}  P={ours['precision']:.2f}  R={ours['recall']:.2f}"
    )
    print(
        f"THEIRS (same subset)  : F1={theirs['f1']:.2f}  P={theirs['precision']:.2f}  "
        f"R={theirs['recall']:.2f}"
    )
    pub_s = f"{pub:.2f}" if pub is not None else "—"
    print(f"THEIRS (published, full set) : F1={pub_s}")
    usage = report["usage"]
    print(
        f"usage: in={usage['input_tokens']} out={usage['output_tokens']}  "
        f"real_cost=${report['real_cost_usd']:.4f}  cost_is_real={report['cost_is_real']}  "
        f"$/1k={report['usd_per_1k_pairs']:.4f}"
    )
    disagreements = report["disagreements"]
    print("-" * 96)
    print("disagreeing pairs (up to 10 of the divergences):")
    if not disagreements:
        print("  (none — every judged pair agreed with the archive)")
    for d in disagreements:
        print(
            f"  [{d['left_id']} vs {d['right_id']}] gold={d['gold_label']}  "
            f"theirs={d['their_answer']!r}  ours={d['our_answer']!r}"
        )
        print(f"      left : {d['left']}")
        print(f"      right: {d['right']}")
    print("=" * 96)


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


def _subset_indices(spec: PeetersReplicationSpec, limit: int | None, seed: int) -> list[int] | None:
    """The ``--limit`` stratified sample indices for ``spec`` (``None`` = all pairs)."""
    if limit is None:
        return None
    labels = [label for _left, _right, label in load_peeters_sample(spec)]
    return stratified_subset_indices(labels, limit, seed)


def _run_replay_mode(args: argparse.Namespace) -> int:
    print("Offline replay — NO API key, NO LLM call, $0 spend. Replaying archived answers.\n")
    replay(
        args.dataset,
        args.model or "gpt-4-0613",
        args.prompt_design,
        args.cache_dir,
        limit=args.limit,
        seed=args.seed,
    )
    return 0


def _run_dry_run_mode(args: argparse.Namespace) -> int:
    spec = get_peeters_replication(args.dataset)
    models = _resolve_paid_models(args.model)
    _assert_priced(models)
    indices = _subset_indices(spec, args.limit, args.seed)
    print("Dry run — NO API key, NO LLM call, $0 spend. Rendering prompts + counting tokens.\n")
    total_est = 0.0
    for model in models:
        report = dry_run(spec, model, indices=indices)
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
    indices = _subset_indices(spec, args.limit, args.seed)

    # Print the estimate BEFORE any network call, so the operator sees the cost.
    print("Estimating cost (dry run, $0) before any spend...\n")
    total_est = 0.0
    for model in models:
        report = dry_run(spec, model, indices=indices)
        total_est += report["est_usd"]
        print(
            f"  {model:40}  pairs={report['n_pairs']}  "
            f"input_tokens={report['input_tokens']}  est=${report['est_usd']:.4f}"
        )
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
            if args.compare_archived:
                print(
                    f"[live+archive] judging {spec.name} with {model} "
                    f"(per-model cap ${per_model_budget:.2f})..."
                )
                report = run_compare_archived(
                    spec,
                    model,
                    prompt_design=args.prompt_design,
                    budget_usd=per_model_budget,
                    cache_dir=args.cache_dir,
                    limit=args.limit,
                    seed=args.seed,
                )
                _print_archived_comparison(report)
            else:
                print(
                    f"[live] judging {spec.name} with {model} "
                    f"(per-model cap ${per_model_budget:.2f})..."
                )
                report = run_live(spec, model, budget_usd=per_model_budget, indices=indices)
            results.append(report)
    except BudgetExceeded as exc:
        print(f"[stopped] budget cap fired: {exc}")
        return 2

    if not args.compare_archived:
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
        "--limit",
        type=int,
        default=None,
        help=(
            "Run a stratified subset of N pairs (preserving the ~17.1%% positive ratio, "
            "deterministic under --seed) instead of all 1206. Applies to dry-run/live/replay."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the --limit stratified subset (default 0).",
    )
    parser.add_argument(
        "--compare-archived",
        action="store_true",
        help=(
            "(--mode live) also compare each judged pair to the authors' archived answer for "
            "the same model: per-pair agreement, a 2x2 confusion, up to 10 disagreements, and "
            "our vs. their F1/P/R on the judged subset."
        ),
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
        help="Where to download/extract the answer archive (replay/compare-archived; gitignored).",
    )
    parser.add_argument(
        "--results-path", default=None, help="Optional JSON path to write live results to."
    )
    args = parser.parse_args()

    if args.mode == "live" and args.budget > BUDGET_CEILING_USD:
        print(f"[fatal] --budget ${args.budget:.2f} exceeds the ${BUDGET_CEILING_USD:.2f} ceiling.")
        return 1

    if args.compare_archived and args.mode != "live":
        print("[fatal] --compare-archived only applies to --mode live (it runs a paid judge).")
        return 1

    if args.limit is not None and args.limit <= 0:
        print(f"[fatal] --limit must be a positive integer (got {args.limit}).")
        return 1

    if args.mode == "replay":
        return _run_replay_mode(args)
    if args.mode == "dry-run":
        return _run_dry_run_mode(args)
    return _run_live_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
