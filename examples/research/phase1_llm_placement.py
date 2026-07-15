"""Phase 1 (#80): the honest, full-split LLM placement next to the RF floor.

"Prove the seam" Phase 1 has an RF-judge $0 *floor* (see
:mod:`examples.research.phase1_rf_floor`). This script produces the companion
*paid* data point: a prompted :class:`~langres.core.matchers.llm_judge.LLMMatcher`
run over the FULL standard Amazon-Google / Abt-Buy literature **test** split under
the SAME honest protocol, recording the **real OpenRouter cost**.

The methodology mirrors the RF floor exactly, so the two numbers are comparable:

    1. :class:`~langres.data.fixed_split_pair_benchmark.FixedSplitPairBenchmark`
       (REUSED, not re-implemented) turns each dataset's fixed
       ``(id_a, id_b, label)`` splits into ``ERCandidate`` objects. For an LLM
       judge the comparison vector is ignored, but reusing the same benchmark
       guarantees the records / gold / splits match the RF floor pair-for-pair.
    2. The LLM judge scores each pair. ``--derive-on fixed:0.5`` (the default)
       grades the FULL test split at a constant 0.5 cut in a SINGLE judging pass --
       leakage-free and, crucially, single-event-loop safe (see the hard cap note).
       ``--derive-on valid``/``train`` instead derive the threshold on that split
       via
       :func:`~langres.data.fixed_split_pair_benchmark.evaluate_fixed_split_honest`
       and apply the FIXED cut to test; because that judges a SECOND split it opens
       a second ``asyncio.run`` and so is refused on the real (paid) path -- offered
       only under ``--dry-run`` (the mock judge has no ``forward_async``). Either
       way the leaky "argmax-F1-on-test" number is also reported so the honesty
       delta is explicit.

**Hard spend cap.** The real ``LLMMatcher`` is judged **concurrently in chunks**
(``--max-concurrent`` calls in flight via its ``forward_async`` -- ~an order of
magnitude faster than the ~4s/call sequential path, and retry-backed). Every
paid judgement's honest ``provenance["cost_usd"]`` is metered through ONE
:class:`~langres.clients.openrouter.SpendMonitor` shared across the valid + test
passes (and across datasets in a ``--dataset both`` run), and the cap is checked
after **each chunk**, so cumulative spend can never cross ``--max-usd`` by more
than a single chunk. A breach raises
:class:`~langres.clients.openrouter.BudgetExceeded` carrying the judgements
already paid for on ``.partial_judgements`` -- the run stops cleanly, the money
already spent is reported, nothing is lost. A conservative worst-case projection
also refuses *before* spending when a full run would obviously blow the cap.

Artifacts land in ``data/benchmarks/phase1/``: a per-``(model, dataset)`` JSON
plus an appended row in ``PHASE1_LLM_PLACEMENT.md`` (regenerated from every
committed JSON, so re-runs update rather than duplicate).

Staged operation (recommended): run ONE dataset per process at ``fixed:0.5`` (the
real-path shape -- exactly one ``asyncio.run``), observe the printed cost, then
scale up the model or run the other dataset in a fresh process. ``--derive-on
valid``/``train`` and ``--dataset both`` each open a second ``asyncio.run`` and
are refused on the real path (available under ``--dry-run``). Example (run WITH a
real key, sandbox off)::

    uv run python examples/research/phase1_llm_placement.py \\
        --model openrouter/deepseek/deepseek-v4-flash \\
        --dataset amazon_google --derive-on fixed:0.5 --max-usd 1.0

``--dry-run`` swaps in a deterministic $0 mock judge to exercise the full wiring
without a key or any spend. ``print`` is allowed under ``examples/``.
"""

from __future__ import annotations

import os

# Pin OpenMP / FAISS threading BEFORE the (lazy) dataset loaders pull torch/faiss
# transitively via VectorBlocker -- mirrors the sibling research scripts and
# dodges the macOS libomp crash. Harmless when the loaders are never imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import tempfile  # noqa: E402
from collections.abc import Iterator, Sequence  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from langres.clients.openrouter import (  # noqa: E402
    PRICES_PER_1M,
    BudgetExceeded,
    SpendMonitor,
    per_token_worst_price,
    register_runtime_model_price,
)
from langres.core.metrics import classify_pairs, pair_pr_curve  # noqa: E402
from langres.core.models import ERCandidate, PairwiseJudgement  # noqa: E402
from langres.core.matcher import Matcher  # noqa: E402
from langres.data.fixed_split_pair_benchmark import (  # noqa: E402
    DEFAULT_ARGMAX_GRID,
    FixedSplitPairBenchmark,
    HonestPairEval,
    evaluate_fixed_split_honest,
)

logger = logging.getLogger("phase1_llm_placement")

# ---------------------------------------------------------------------------
# Reference bands (comparison targets for the placement)
# ---------------------------------------------------------------------------

#: The RF-judge honest pair-F1 *floor* per dataset (the number this LLM run is
#: placed against), read from the committed
#: ``data/benchmarks/phase1/phase1_rf_floor_<dataset>.json`` artifacts. Rounded:
#: 0.360 AG / 0.404 Abt-Buy.
RF_FLOOR_F1: dict[str, float] = {
    "amazon_google": 0.3596330275229358,
    "abt_buy": 0.40362811791383213,
}

#: Ditto pairwise-F1 SOTA band per dataset (the far target above the floor).
DITTO_F1: dict[str, float] = {"amazon_google": 0.756, "abt_buy": 0.893}

#: Both datasets are products, so the judge's prompt uses this entity noun.
_ENTITY_NOUN: dict[str, str] = {"amazon_google": "product", "abt_buy": "product"}

#: Generous worst-case tokens per pair, sizing the pre-flight projection only.
#: The live SpendMonitor meters and enforces the REAL cost as scoring happens.
WORST_CASE_TOKENS_PER_PAIR = 1200.0

#: Where the JSON + Markdown artifacts land (repo-root ``data/benchmarks``).
_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "phase1"
_MARKDOWN_NAME = "PHASE1_LLM_PLACEMENT.md"


# ---------------------------------------------------------------------------
# Metered judge: reuse the SpendMonitor / BudgetExceeded cap, one shared monitor
# ---------------------------------------------------------------------------


class _MeteredJudge(Matcher[Any]):
    """Wrap a judge, metering each judgement's real cost through a shared monitor.

    Reuses the exact spend-cap pattern of
    :class:`~langres.core.presets._SpendCappedMatcher` and
    ``examples/research/w3_paid_smoke.py``'s ``_charge`` -- add each judgement's
    honest ``provenance["cost_usd"]`` to a :class:`SpendMonitor`, ``check()`` it,
    and re-raise :class:`BudgetExceeded` with every already-paid-for judgement on
    ``.partial_judgements``. The one difference (and the reason this is not just
    ``_SpendCappedMatcher``): the monitor is **injected**, so ONE cumulative
    ledger spans the valid-derivation pass, the test pass, and every dataset in
    the run -- a genuinely run-wide hard cap.

    It also tallies this judge's own cost, call count, real-cost fraction, and
    served providers, so the caller can put the honest economics in the artifact.

    A judge exposing an async ``forward_async`` (the real ``LLMMatcher``) is driven
    **concurrently in chunks**: each chunk of ``chunk_size`` pairs is judged in
    parallel (up to ``max_concurrent`` calls in flight), then metered and
    cap-checked as a unit. That is ~an order of magnitude faster than the
    ~4s/call sequential path and retry-backed, while bounding any spend overshoot
    to a single chunk. Judges without ``forward_async`` (the dry-run / mock
    judges) keep the original sequential, per-judgement loop.
    """

    def __init__(
        self,
        inner: Matcher[Any],
        *,
        monitor: SpendMonitor,
        max_concurrent: int = 50,
        chunk_size: int = 100,
    ) -> None:
        """Wrap ``inner``, metering it through the shared ``monitor``.

        Args:
            inner: The judge to meter.
            monitor: The shared, run-wide spend monitor enforcing the hard cap.
            max_concurrent: Parallel LLM calls in flight per chunk on the async
                path (ignored by the sequential fallback).
            chunk_size: Pairs judged (concurrently) before each cap ``check()`` on
                the async path -- the overshoot bound. Internal, not CLI-exposed.
        """
        self._inner = inner
        self._monitor = monitor
        self._max_concurrent = max_concurrent
        self._chunk_size = chunk_size
        self.cost_usd = 0.0
        self.n_judged = 0
        self.n_real_cost = 0
        self.providers: dict[str, int] = {}

    def forward(self, candidates: Iterator[ERCandidate[Any]]) -> Iterator[PairwiseJudgement]:
        """Yield each judgement after charging its cost and enforcing the cap.

        Dispatches on the wrapped judge: a real ``LLMMatcher`` (anything exposing
        ``forward_async``) takes the concurrent, chunked path; everything else
        (the dry-run / mock judges) takes the unchanged sequential path. Both
        preserve the run-wide hard cap and carry the already-paid-for judgements
        on :class:`BudgetExceeded`'s ``.partial_judgements`` when it fires.
        """
        materialized = list(candidates)
        if hasattr(self._inner, "forward_async"):
            yield from self._forward_async(materialized)
        else:
            yield from self._forward_sync(materialized)

    def _meter(self, judgement: PairwiseJudgement) -> None:
        """Tally one judgement's cost / provider and add it to the shared monitor.

        The single source of truth for the money accounting, shared by both the
        sequential and concurrent paths (only the cap-``check()`` granularity
        differs between them -- per judgement vs. per chunk).
        """
        prov = judgement.provenance
        cost = prov.get("cost_usd", 0.0)
        cost = float(cost) if cost is not None else 0.0
        self.cost_usd += cost
        self.n_judged += 1
        if prov.get("cost_is_real"):
            self.n_real_cost += 1
        provider = prov.get("provider")
        if isinstance(provider, str) and provider:
            self.providers[provider] = self.providers.get(provider, 0) + 1
        self._monitor.add(cost)

    def _forward_sync(self, candidates: list[ERCandidate[Any]]) -> Iterator[PairwiseJudgement]:
        """Sequential path (unchanged): meter + cap-check after every judgement.

        The dry-run / mock judges without ``forward_async`` take exactly this
        loop -- one flat charge per judged pair, the cap enforced immediately.
        """
        produced: list[PairwiseJudgement] = []
        for judgement in self._inner.forward(iter(candidates)):
            produced.append(judgement)
            self._meter(judgement)
            try:
                self._monitor.check()
            except BudgetExceeded as exc:
                exc.partial_judgements = list(produced)
                raise
            yield judgement

    def _forward_async(self, candidates: list[ERCandidate[Any]]) -> Iterator[PairwiseJudgement]:
        """Concurrent path: judge ALL chunks under a SINGLE event loop.

        litellm caches a global async logging worker bound to the FIRST event
        loop created in the process; a *second* ``asyncio.run()`` orphans that
        worker's queue and floods ``RuntimeError: <Queue ...> is bound to a
        different event loop`` (one per call). So the whole judging for this call
        runs under ONE ``asyncio.run`` (:meth:`_judge_chunks_async`) -- never one
        per chunk. The corollary the CLI must honour: run ONE dataset per process
        with a single-``forward`` eval (``--derive-on fixed:0.5``), so the process
        makes exactly one ``asyncio.run`` total. ``--dataset both`` or a
        ``valid``/``train`` derive would issue a second ``asyncio.run`` and hit
        the same litellm bug.
        """
        return iter(asyncio.run(self._judge_chunks_async(candidates)))

    async def _judge_chunks_async(
        self, candidates: list[ERCandidate[Any]]
    ) -> list[PairwiseJudgement]:
        """Judge ``candidates`` chunk-by-chunk in one loop; meter + cap-check each.

        Within the loop each chunk of ``chunk_size`` pairs fans out up to
        ``max_concurrent`` concurrent calls (retry-backed), is metered, then the
        cap is ``check()``-ed once -- the mid-run hard stop at chunk granularity
        (a breach stops within one chunk of overshoot, carrying the already-paid
        judgements on :attr:`BudgetExceeded.partial_judgements`).
        """
        produced: list[PairwiseJudgement] = []
        for start in range(0, len(candidates), self._chunk_size):
            chunk = candidates[start : start + self._chunk_size]
            chunk_judgements: list[PairwiseJudgement] = await self._inner.forward_async(  # type: ignore[attr-defined]
                iter(chunk), max_concurrent=self._max_concurrent
            )
            for judgement in chunk_judgements:
                produced.append(judgement)
                self._meter(judgement)
            try:
                self._monitor.check()
            except BudgetExceeded as exc:
                exc.partial_judgements = list(produced)
                raise
        return produced

    def inspect_scores(self, judgements: list[PairwiseJudgement], sample_size: int = 10) -> Any:
        """Delegate score inspection to the wrapped judge."""
        return self._inner.inspect_scores(judgements, sample_size)


# ---------------------------------------------------------------------------
# The run (testable core)
# ---------------------------------------------------------------------------


def run_placement(
    benchmark: FixedSplitPairBenchmark[Any],
    judge: Matcher[Any],
    *,
    derive_on: str,
    monitor: SpendMonitor,
    argmax_grid: Sequence[float] = DEFAULT_ARGMAX_GRID,
    max_concurrent: int = 50,
    chunk_size: int = 100,
) -> tuple[HonestPairEval, _MeteredJudge]:
    """Score the full test split honestly, metered by a shared spend monitor.

    Args:
        benchmark: The fixed-split adapter (records / gold / splits).
        judge: The scorer to grade (an ``LLMMatcher`` on the real path; any
            ``Matcher`` in a test / dry run).
        derive_on: ``"valid"`` / ``"train"`` -> derive the threshold on that
            split via :func:`evaluate_fixed_split_honest`; ``"fixed:<x>"`` ->
            skip the derive calls and grade test at the constant ``x``.
        monitor: The shared :class:`SpendMonitor` enforcing ``--max-usd`` across
            every pass and dataset of the run.
        argmax_grid: Thresholds swept for the leaky argmax-on-test comparison.
        max_concurrent: Parallel LLM calls in flight per chunk when the judge
            exposes ``forward_async`` (the real ``LLMMatcher``); ignored otherwise.
        chunk_size: Pairs judged concurrently before each cap ``check()`` on the
            async path -- the spend-overshoot bound.

    Returns:
        ``(result, meter)`` -- the :class:`HonestPairEval` and the
        :class:`_MeteredJudge` carrying this run's cost / provider tallies.

    Raises:
        BudgetExceeded: If cumulative spend crosses the monitor's budget; the
            exception carries the already-paid-for judgements on
            ``.partial_judgements``.
    """
    metered = _MeteredJudge(
        judge, monitor=monitor, max_concurrent=max_concurrent, chunk_size=chunk_size
    )
    if derive_on.startswith("fixed:"):
        threshold = float(derive_on.split(":", 1)[1])
        result = _eval_fixed_threshold(metered, benchmark, threshold, argmax_grid)
    else:
        result = evaluate_fixed_split_honest(
            metered, benchmark, derive_on=derive_on, argmax_grid=argmax_grid
        )
    return result, metered


def _eval_fixed_threshold(
    judge: Matcher[Any],
    benchmark: FixedSplitPairBenchmark[Any],
    threshold: float,
    argmax_grid: Sequence[float],
) -> HonestPairEval:
    """Grade the full test split at a FIXED threshold (no derive-split calls).

    The leakage-free, cheapest option: the cut is a constant (not tuned on any
    split), so only the test pairs are judged. The leaky argmax-on-test number is
    still reported for the honesty delta.
    """
    test = benchmark.build("test")
    test_judgements = list(judge.forward(iter(test.candidates)))
    honest = classify_pairs(test_judgements, test.gold, threshold)
    curve = pair_pr_curve(test_judgements, test.gold, argmax_grid)
    argmax = max(curve, key=lambda m: m.f1)
    return HonestPairEval(
        dataset=benchmark.name,
        derive_on=f"fixed:{threshold}",
        derived_threshold=threshold,
        threshold_method="fixed",
        honest=honest,
        argmax_on_test=argmax,
        honesty_delta_f1=argmax.f1 - honest.f1,
    )


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def build_artifact(
    result: HonestPairEval,
    meter: _MeteredJudge,
    *,
    dataset: str,
    model: str,
    provider: dict[str, Any] | None,
    n_test: int,
    n_test_pos: int,
    n_derive: int,
) -> dict[str, Any]:
    """Assemble the JSON-ready artifact for one ``(model, dataset)`` run."""
    floor = RF_FLOOR_F1[dataset]
    ditto = DITTO_F1[dataset]
    return {
        "dataset": dataset,
        "method": "LLMMatcher",
        "model": model,
        "provider_requested": provider,
        "provider_served": sorted(meter.providers),
        "provider_served_counts": meter.providers,
        "derive_on": result.derive_on,
        "threshold_method": result.threshold_method,
        "derived_threshold": result.derived_threshold,
        "n_derive": n_derive,
        "n_test": n_test,
        "n_test_pos": n_test_pos,
        "n_judged": meter.n_judged,
        "n_real_cost": meter.n_real_cost,
        "cost_is_real_frac": (meter.n_real_cost / meter.n_judged) if meter.n_judged else 0.0,
        "real_cost_usd": meter.cost_usd,
        "honest": _metrics_dict(result.honest),
        "argmax_on_test": _metrics_dict(result.argmax_on_test),
        "honesty_delta_f1": result.honesty_delta_f1,
        "rf_floor_f1": floor,
        "gap_to_rf_floor_f1": result.honest.f1 - floor,
        "ditto_f1": ditto,
        "gap_to_ditto_f1": ditto - result.honest.f1,
        "notes": (
            "Prompted LLMMatcher on the FULL fixed test split. HONEST f1 applies a "
            "threshold derived on the derive-split (or a fixed constant) to the "
            "whole test split; argmax_on_test tunes the cut on test itself "
            "(honesty_delta_f1 = argmax_on_test.f1 - honest.f1). gap_to_rf_floor_f1 "
            "is honest.f1 minus the $0 RandomForestMatcher floor (positive = the LLM "
            "beats the local floor); gap_to_ditto_f1 is the remaining distance to "
            "the Ditto SOTA band. real_cost_usd is OpenRouter's actual billed cost "
            "(cost_is_real_frac of judgements) metered through the SpendMonitor."
        ),
    }


def _metrics_dict(metrics: Any) -> dict[str, float | int]:
    """Flatten a :class:`~langres.core.metrics.PairMetrics` into a JSON dict."""
    return {
        "threshold": metrics.threshold,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
    }


def _artifact_stem(model: str, dataset: str) -> str:
    """Deterministic per-run JSON stem (model slug + dataset)."""
    slug = model.replace("/", "-")
    return f"phase1_llm_placement_{slug}_{dataset}"


def write_artifacts(artifact: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Write the per-run JSON and regenerate the combined Markdown table.

    The Markdown is rebuilt from EVERY ``phase1_llm_placement_*.json`` in
    ``out_dir`` (this run's included), so a staged sequence of runs appends rows,
    and a re-run of the same ``(model, dataset)`` updates its row rather than
    duplicating it.

    Returns:
        ``(json_path, markdown_path)``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{_artifact_stem(artifact['model'], artifact['dataset'])}.json"
    json_path.write_text(json.dumps(artifact, indent=2))

    artifacts = [
        json.loads(p.read_text()) for p in sorted(out_dir.glob("phase1_llm_placement_*.json"))
    ]
    md_path = out_dir / _MARKDOWN_NAME
    # Atomic replace: the documented run pattern is one dataset per process, and
    # several such processes regenerate this shared file concurrently. A bare
    # ``write_text`` can be read half-written (torn) by a sibling; writing to a
    # per-process temp in the SAME dir and ``os.replace``-ing it swaps the file in
    # atomically so a reader always sees a complete table.
    fd, tmp_name = tempfile.mkstemp(dir=out_dir, prefix=".phase1_md_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(format_report(artifacts) + "\n")
        os.replace(tmp_name, md_path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)  # no-op after a successful replace
    return json_path, md_path


def format_report(artifacts: list[dict[str, Any]]) -> str:
    """Render the combined Markdown table over all placement artifacts."""
    lines = [
        "# Phase 1 -- LLMMatcher honest full-split placement (paid)",
        "",
        "Prompted `LLMMatcher` on the **full standard test split**, threshold "
        "derived honestly (on VALID, or a fixed constant) and applied to all of "
        "test. `argmax-F1` is the leaky ceiling (cut tuned on test) shown only for "
        "the honesty delta. `real cost` is OpenRouter's actual billed spend. The "
        "gap columns place the honest F1 against the $0 RandomForestMatcher floor "
        "(0.360 AG / 0.404 Abt-Buy) and the Ditto SOTA band (0.756 / 0.893).",
        "",
        "| model | dataset | honest F1 | honest P | honest R | threshold | "
        "argmax-F1 | real cost USD | provider | gap to RF floor | gap to Ditto |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for art in artifacts:
        honest = art["honest"]
        served = art.get("provider_served") or []
        provider = ", ".join(served) if served else "-"
        lines.append(
            f"| `{art['model']}` | {art['dataset']} "
            f"| {honest['f1']:.4f} | {honest['precision']:.4f} | {honest['recall']:.4f} "
            f"| {art['derived_threshold']:.4f} | {art['argmax_on_test']['f1']:.4f} "
            f"| ${art['real_cost_usd']:.4f} | {provider} "
            f"| {art['gap_to_rf_floor_f1']:+.4f} | {art['gap_to_ditto_f1']:+.4f} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- **honest F1** is the placement number: no test-label peeking.",
        "- **gap to RF floor** > 0 means the paid LLM judge beats the $0 local "
        "baseline; < 0 means the thin single-metric floor is (surprisingly) ahead.",
        "- **gap to Ditto** is the distance still open to the SOTA band.",
        "- **real cost USD** is the actual OpenRouter spend for that cell "
        "(per-`(model, dataset)` JSON has the served-provider breakdown).",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Real-run plumbing (lazy heavy imports live here)
# ---------------------------------------------------------------------------


def build_benchmark(dataset: str) -> FixedSplitPairBenchmark[Any]:
    """Build the fixed-split benchmark for ``dataset`` (lazy loader imports).

    The dataset loaders pull the ``[semantic]`` embedding stack at import, so
    they are imported HERE (not at module top) -- keeping this module and
    :func:`run_placement` importable for the $0 mocked tests, which never touch a
    real dataset.
    """
    if dataset == "amazon_google":
        from langres.data.amazon_google import (
            ProductSchema,
            load_amazon_google,
            load_amazon_google_pair_splits,
        )

        return FixedSplitPairBenchmark.from_loaders(
            name=dataset,
            schema=ProductSchema,
            corpus_loader=load_amazon_google,
            pair_split_loader=load_amazon_google_pair_splits,
        )
    if dataset == "abt_buy":
        from langres.data.abt_buy import (
            AbtBuySchema,
            load_abt_buy,
            load_abt_buy_pair_splits,
        )

        return FixedSplitPairBenchmark.from_loaders(
            name=dataset,
            schema=AbtBuySchema,
            corpus_loader=load_abt_buy,
            pair_split_loader=load_abt_buy_pair_splits,
        )
    raise ValueError(f"unknown dataset {dataset!r}; choose amazon_google | abt_buy")


def _build_llm_judge(model: str, provider: dict[str, Any] | None, entity_noun: str) -> Matcher[Any]:
    """Build the real ``LLMMatcher`` (lazy import -- pulls the ``[llm]`` extra)."""
    from langres.core.matchers.llm_judge import LLMMatcher

    return LLMMatcher(model=model, provider=provider, entity_noun=entity_noun)


class _DryRunJudge(Matcher[Any]):
    """A deterministic, $0 stand-in for ``--dry-run`` (no key, no network).

    Scores each pair from its comparison vector (mean similarity when present,
    else 0.5) so the honest/argmax machinery sees a real distribution, and stamps
    ``cost_usd = 0.0`` so the SpendMonitor never moves.
    """

    def forward(self, candidates: Iterator[ERCandidate[Any]]) -> Iterator[PairwiseJudgement]:
        """Yield a $0, comparison-derived judgement per candidate."""
        for candidate in candidates:
            sims: list[float] = []
            if candidate.comparison is not None:
                sims = [v for v in candidate.comparison.similarities.values() if v is not None]
            score = sum(sims) / len(sims) if sims else 0.5
            yield PairwiseJudgement(
                left_id=candidate.left.id,
                right_id=candidate.right.id,
                score=max(0.0, min(1.0, score)),
                score_type="prob_llm",
                decision_step="dry_run",
                provenance={"cost_usd": 0.0, "cost_is_real": False, "provider": None},
            )

    def inspect_scores(self, judgements: list[PairwiseJudgement], sample_size: int = 10) -> Any:
        """Not used on the dry-run path."""
        raise NotImplementedError("dry-run judge does not inspect scores")


def project_worst_case_usd(model: str, n_calls: int) -> float:
    """Conservative worst-case USD for ``n_calls`` at the model's dearer rate."""
    return n_calls * WORST_CASE_TOKENS_PER_PAIR * per_token_worst_price(model)


# ---------------------------------------------------------------------------
# Config + CLI
# ---------------------------------------------------------------------------


@dataclass
class PlacementConfig:
    """Everything a run needs; defaults are the safe real-run defaults."""

    model: str
    datasets: list[str]
    provider: dict[str, Any] | None = None
    max_usd: float = 5.0
    derive_on: str = "valid"
    dry_run: bool = False
    max_concurrent: int = 50
    out_dir: Path = field(default_factory=lambda: _OUTPUT_DIR)


def _run_one_dataset(
    cfg: PlacementConfig, dataset: str, monitor: SpendMonitor
) -> dict[str, Any] | None:
    """Build, project-check, judge, and write artifacts for one dataset.

    Returns the artifact dict, or ``None`` if the worst-case projection refused
    the dataset before any spend (the shared monitor is left untouched).
    """
    benchmark = build_benchmark(dataset)
    n_test = len(benchmark.build("test").candidates)
    n_derive = (
        0 if cfg.derive_on.startswith("fixed:") else len(benchmark.build(cfg.derive_on).candidates)
    )
    n_calls = n_test + n_derive

    if not cfg.dry_run:
        projected = project_worst_case_usd(cfg.model, n_calls)
        logger.info(
            "[%s] %d judge calls (%d derive + %d test); worst-case projection $%.4f "
            "vs remaining budget $%.4f",
            dataset,
            n_calls,
            n_derive,
            n_test,
            projected,
            monitor.remaining,
        )
        if projected > monitor.remaining:
            print(
                f"[refused] {dataset}: worst-case projection ${projected:.4f} exceeds "
                f"remaining budget ${monitor.remaining:.4f}. Raise --max-usd, pick a "
                f"cheaper --model, or use --derive-on fixed:0.5 to skip the derive calls."
            )
            return None

    judge: Matcher[Any] = (
        _DryRunJudge()
        if cfg.dry_run
        else _build_llm_judge(cfg.model, cfg.provider, _ENTITY_NOUN[dataset])
    )
    result, meter = run_placement(
        benchmark,
        judge,
        derive_on=cfg.derive_on,
        monitor=monitor,
        max_concurrent=cfg.max_concurrent,
    )

    artifact = build_artifact(
        result,
        meter,
        dataset=dataset,
        model=cfg.model,
        provider=cfg.provider,
        n_test=n_test,
        n_test_pos=len(benchmark.build("test").gold),
        n_derive=n_derive,
    )
    json_path, md_path = write_artifacts(artifact, cfg.out_dir)
    _print_dataset_summary(artifact, result, meter)
    print(f"[report] wrote {json_path} and {md_path}")
    return artifact


def _print_dataset_summary(
    artifact: dict[str, Any], result: HonestPairEval, meter: _MeteredJudge
) -> None:
    """Print the headline placement + cost line for one dataset."""
    served = ", ".join(sorted(meter.providers)) or "-"
    print(f"\n## {artifact['dataset']}  ({artifact['model']})")
    print(f"derive_on:            {result.derive_on} (thr={result.derived_threshold:.4f})")
    print(
        f"HONEST:               P={result.honest.precision:.4f} R={result.honest.recall:.4f} "
        f"F1={result.honest.f1:.4f}"
    )
    print(
        f"argmax-on-test (leaky): F1={result.argmax_on_test.f1:.4f} "
        f"(honesty delta {result.honesty_delta_f1:+.4f})"
    )
    print(
        f"gap to RF floor:      {artifact['gap_to_rf_floor_f1']:+.4f}   "
        f"gap to Ditto: {artifact['gap_to_ditto_f1']:+.4f}"
    )
    print(
        f"real cost:            ${meter.cost_usd:.4f} over {meter.n_judged} calls "
        f"({artifact['cost_is_real_frac']:.0%} real-cost) | served: {served}"
    )


def _parse_derive_on(value: str) -> str:
    """Validate ``--derive-on`` (``valid`` | ``train`` | ``fixed:<0..1>``)."""
    if value in ("valid", "train"):
        return value
    if value.startswith("fixed:"):
        threshold = float(value.split(":", 1)[1])  # ValueError -> argparse reports it
        if not 0.0 <= threshold <= 1.0:
            raise argparse.ArgumentTypeError(f"fixed threshold must be in [0, 1], got {threshold}")
        return value
    raise argparse.ArgumentTypeError(
        f"--derive-on must be 'valid', 'train', or 'fixed:<0..1>', got {value!r}"
    )


def _parse_provider(value: str | None) -> dict[str, Any] | None:
    """Parse the optional ``--provider`` JSON routing block into a dict (or None)."""
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"--provider must be a JSON object, got {type(parsed)}")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1 honest full-split LLM placement (paid, SpendMonitor-capped)."
    )
    parser.add_argument(
        "--model",
        default="openrouter/deepseek/deepseek-v4-flash",
        help="A PRICES_PER_1M-pinned OpenRouter model (else its spend cap is blind).",
    )
    parser.add_argument(
        "--dataset",
        default="amazon_google",
        choices=["amazon_google", "abt_buy", "both"],
        help="Which fixed-split benchmark(s) to place.",
    )
    parser.add_argument(
        "--provider",
        type=_parse_provider,
        default=None,
        help='Optional OpenRouter provider-pin JSON, e.g. \'{"order":["DeepSeek"],'
        '"allow_fallbacks":false}\'. Default None -> OpenRouter default routing.',
    )
    parser.add_argument(
        "--max-usd", type=float, default=5.0, help="Hard cumulative spend cap (USD)."
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        help="Parallel LLM calls in flight per chunk on the async judge path (default 50).",
    )
    parser.add_argument(
        "--derive-on",
        type=_parse_derive_on,
        default="fixed:0.5",
        help="Threshold source: 'fixed:<0..1>' (default, single-loop safe) | 'valid' | 'train'. "
        "valid/train each judge a second split, so the real LLMMatcher path refuses them "
        "(litellm allows one asyncio.run per process).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a deterministic $0 mock judge (no key, no network, no spend).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: place the LLM judge under a hard, run-wide spend cap."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    args = _parse_args(argv)

    datasets = ["amazon_google", "abt_buy"] if args.dataset == "both" else [args.dataset]
    cfg = PlacementConfig(
        model=args.model,
        datasets=datasets,
        provider=args.provider,  # already parsed by argparse (type=_parse_provider)
        max_usd=args.max_usd,
        derive_on=args.derive_on,
        dry_run=args.dry_run,
        max_concurrent=args.max_concurrent,
    )

    if not cfg.dry_run:
        from dotenv import load_dotenv

        load_dotenv(".env")  # OPENROUTER_API_KEY lives in .env, not Settings.
        if "OPENROUTER_API_KEY" not in os.environ:
            print("[fatal] OPENROUTER_API_KEY not set; export it or use --dry-run.")
            return 1
        if cfg.model not in PRICES_PER_1M:
            print(
                f"[fatal] model {cfg.model!r} is not in PRICES_PER_1M, so its spend cap "
                f"would be blind. Pin its price first, or pick one of: {sorted(PRICES_PER_1M)}."
            )
            return 1
        # Single-asyncio.run-per-process invariant. The real LLMMatcher exposes
        # forward_async, so every judge.forward() opens one asyncio.run. A
        # valid/train derive judges a SECOND split (derive + test), and --dataset
        # both judges a SECOND dataset -- either issues a second asyncio.run, which
        # trips litellm's cross-loop logging-worker bug AFTER the first pass has
        # already paid. Refuse before spending (before the price probe below).
        if cfg.derive_on in ("valid", "train") or len(datasets) > 1:
            print(
                f"[fatal] the real LLMMatcher path allows exactly ONE asyncio.run per "
                f"process, but derive_on={cfg.derive_on!r} + datasets={datasets} would "
                f"issue more (valid/train judges a second split; --dataset both a second "
                f"dataset) and trip litellm's cross-loop worker bug after the first pass "
                f"has already paid. Run ONE dataset per process at a fixed threshold, e.g. "
                f"--dataset {datasets[0]} --derive-on fixed:0.5."
            )
            return 1
        # Pin the model's price (incl. its dated OpenRouter runtime id) and confirm
        # the id actually responds before the full run's spend. Without the pin an
        # unpinned dated id makes litellm's completion_cost return 0.0, so the cap
        # would under-meter to $0 whenever usage-accounting cost is absent. Mirrors
        # examples/research/w3_paid_smoke.py.
        if register_runtime_model_price(cfg.model) is None:
            print(f"[fatal] {cfg.model} did not resolve/respond; STOP (never guess-and-spend).")
            return 1

    print("=" * 78)
    print(f"Phase 1 -- LLMMatcher placement | model={cfg.model} | datasets={datasets}")
    print(f"derive_on={cfg.derive_on} | max_usd=${cfg.max_usd:.2f} | dry_run={cfg.dry_run}")
    print("=" * 78)

    monitor = SpendMonitor(budget_usd=cfg.max_usd)
    exit_code = 0
    try:
        for dataset in datasets:
            _run_one_dataset(cfg, dataset, monitor)
    except BudgetExceeded as exc:
        print(
            f"\n[stopped] hard spend cap fired: {exc} "
            f"(partial judgements recovered: {len(exc.partial_judgements)})"
        )
        exit_code = 2
    finally:
        # Always report what was actually spent -- on success, on a BudgetExceeded
        # stop, OR on a hard error escaping a chunk's forward_async gather. Money
        # spent must never go unreported just because a run crashed.
        print("\n" + "=" * 78)
        print(
            f"[done] total honest spend ${monitor.spent:.4f} / ${cfg.max_usd:.2f} cap "
            f"(remaining ${monitor.remaining:.4f})"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
