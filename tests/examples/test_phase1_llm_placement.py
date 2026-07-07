"""$0 behavior tests for the Phase 1 LLM placement script (#80).

Every test runs at **$0** with a mocked judge -- no key, no network, no
``litellm``. They drive ``run_placement`` on a tiny FAKE
:class:`~langres.data.fixed_split_pair_benchmark.FixedSplitPairBenchmark` (built
from an in-memory corpus + splits, no dataset loader, no ``[semantic]`` extra),
so this suite never imports ``LLMJudge`` or a real dataset. They assert the
script:

- computes honest P/R/F1 and writes the JSON + Markdown artifacts;
- records the (mocked) real cost and served provider;
- ``--derive-on fixed:<x>`` skips the derive-split calls;
- respects ``--max-usd`` -- a tiny budget aborts cleanly with the partial
  judgements the money already bought;
- the ``--dry-run`` judge is genuinely $0.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from examples.research.phase1_llm_placement import (
    RF_FLOOR_F1,
    PlacementConfig,
    _DryRunJudge,
    _parse_derive_on,
    _parse_provider,
    build_artifact,
    run_placement,
    write_artifacts,
)
from langres.clients.openrouter import BudgetExceeded, SpendMonitor
from langres.core.comparator import StringComparator
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.module import Module
from langres.data.fixed_split_pair_benchmark import FixedSplitPairBenchmark


class _Rec(BaseModel):
    """A tiny record schema for the fake benchmark."""

    id: str
    name: str


#: A fake corpus: two obvious match pairs + three distractors.
_CORPUS = [
    _Rec(id="a1", name="apple ipod nano"),
    _Rec(id="a2", name="apple ipod nano 8gb"),
    _Rec(id="b1", name="samsung galaxy s"),
    _Rec(id="b2", name="samsung galaxy s phone"),
    _Rec(id="z1", name="sony walkman"),
]

#: valid has both classes (youden needs a positive AND a negative); test likewise.
_SPLITS = {
    "valid": [("a1", "a2", 1), ("a1", "z1", 0), ("b1", "z1", 0)],
    "test": [("a1", "a2", 1), ("b1", "b2", 1), ("a1", "z1", 0), ("b1", "z1", 0)],
}

#: The gold-match id pairs (label == 1) across ALL splits, so the mock judge can
#: score a known match high and everything else low -- a clean separation.
_MATCHES = {frozenset({"a1", "a2"}), frozenset({"b1", "b2"})}


def _benchmark() -> FixedSplitPairBenchmark[_Rec]:
    """A tiny fixed-split benchmark over the fake corpus (no dataset loader)."""
    return FixedSplitPairBenchmark(
        name="amazon_google",  # a real key so RF_FLOOR_F1 / DITTO_F1 resolve
        corpus=_CORPUS,
        splits=_SPLITS,
        comparator=StringComparator.from_schema(_Rec),
    )


class _MockJudge(Module[_Rec]):
    """A deterministic $0..$N mock: known matches score high, else low.

    Args:
        cost_usd: Flat honest cost stamped on every judgement (independent of any
            real tokens), so a test can make the SpendMonitor fire at will.
        provider: The served provider recorded in provenance.
    """

    def __init__(self, *, cost_usd: float = 0.0, provider: str = "MockProvider") -> None:
        self._cost = cost_usd
        self._provider = provider

    def forward(self, candidates: Iterator[ERCandidate[_Rec]]) -> Iterator[PairwiseJudgement]:
        for candidate in candidates:
            pair = frozenset({candidate.left.id, candidate.right.id})
            yield PairwiseJudgement(
                left_id=candidate.left.id,
                right_id=candidate.right.id,
                score=0.9 if pair in _MATCHES else 0.1,
                score_type="prob_llm",
                decision_step="mock",
                provenance={
                    "cost_usd": self._cost,
                    "cost_is_real": True,
                    "provider": self._provider,
                    "model": "mock",
                },
            )

    def inspect_scores(self, judgements: list[PairwiseJudgement], sample_size: int = 10) -> object:
        raise NotImplementedError


class _AsyncMockJudge(Module[_Rec]):
    """A judge exposing BOTH a sync ``forward`` and an async ``forward_async``.

    ``forward_async`` returns the same deterministic, flat-cost judgements as
    :class:`_MockJudge` (mocked -- no real await, no network), and records the
    size of every chunk it was handed plus the ``max_concurrent`` it was called
    with, so a test can prove :class:`_MeteredJudge` drove the *async* path in
    chunks. Its sync ``forward`` raises: if it ever runs, the wrapper wrongly
    fell back to the sequential path instead of using ``forward_async``.
    """

    def __init__(self, *, cost_usd: float = 0.0, provider: str = "AsyncProvider") -> None:
        self._cost = cost_usd
        self._provider = provider
        self.chunk_sizes: list[int] = []
        self.max_concurrent_seen: list[int] = []

    def _judge(self, candidate: ERCandidate[_Rec]) -> PairwiseJudgement:
        pair = frozenset({candidate.left.id, candidate.right.id})
        return PairwiseJudgement(
            left_id=candidate.left.id,
            right_id=candidate.right.id,
            score=0.9 if pair in _MATCHES else 0.1,
            score_type="prob_llm",
            decision_step="async_mock",
            provenance={
                "cost_usd": self._cost,
                "cost_is_real": True,
                "provider": self._provider,
                "model": "async-mock",
            },
        )

    async def forward_async(
        self, candidates: Iterator[ERCandidate[_Rec]], max_concurrent: int = 50
    ) -> list[PairwiseJudgement]:
        chunk = list(candidates)
        self.chunk_sizes.append(len(chunk))
        self.max_concurrent_seen.append(max_concurrent)
        return [self._judge(candidate) for candidate in chunk]

    def forward(self, candidates: Iterator[ERCandidate[_Rec]]) -> Iterator[PairwiseJudgement]:
        raise AssertionError("sync forward must not run when forward_async exists")

    def inspect_scores(self, judgements: list[PairwiseJudgement], sample_size: int = 10) -> object:
        raise NotImplementedError


def test_async_path_meters_in_chunks_and_yields_all() -> None:
    """A judge with forward_async is driven concurrently in chunks, fully metered."""
    benchmark = _benchmark()
    monitor = SpendMonitor(budget_usd=100.0)
    judge = _AsyncMockJudge(cost_usd=0.003, provider="DeepSeek")

    result, meter = run_placement(
        benchmark, judge, derive_on="fixed:0.5", monitor=monitor, chunk_size=2
    )

    # fixed:0.5 judges only the 4 test pairs; chunk_size=2 -> two chunks of 2,
    # each run through forward_async (the sync forward would have raised).
    n_test = len(_SPLITS["test"])
    assert judge.chunk_sizes == [2, 2]
    assert judge.max_concurrent_seen == [50, 50]  # default max_concurrent threaded through
    assert meter.n_judged == n_test

    # Cumulative cost metered through the shared monitor; every judgement yielded.
    assert meter.cost_usd == pytest.approx(0.003 * n_test)
    assert monitor.spent == pytest.approx(meter.cost_usd)
    assert meter.providers == {"DeepSeek": n_test}
    assert result.honest.f1 == pytest.approx(1.0)  # clean separation, same as sync path


def test_async_max_concurrent_is_threaded() -> None:
    """run_placement's max_concurrent reaches the inner judge's forward_async."""
    benchmark = _benchmark()
    monitor = SpendMonitor(budget_usd=100.0)
    judge = _AsyncMockJudge(cost_usd=0.0)

    run_placement(
        benchmark, judge, derive_on="fixed:0.5", monitor=monitor, max_concurrent=7, chunk_size=2
    )

    assert judge.max_concurrent_seen == [7, 7]


def test_async_spend_cap_aborts_after_one_chunk_with_partials() -> None:
    """The async cap is checked per chunk: a chunk that crosses stops the run.

    Overshoot is bounded to a single chunk -- the next chunk is never judged --
    and every already-paid-for judgement rides out on ``.partial_judgements``.
    """
    benchmark = _benchmark()
    monitor = SpendMonitor(budget_usd=0.05)
    judge = _AsyncMockJudge(cost_usd=0.05)  # one chunk of 2 = $0.10 > $0.05 cap

    with pytest.raises(BudgetExceeded) as excinfo:
        run_placement(benchmark, judge, derive_on="fixed:0.5", monitor=monitor, chunk_size=2)

    partials = excinfo.value.partial_judgements
    assert len(partials) == 2, "only the first chunk was judged before the hard stop"
    assert judge.chunk_sizes == [2], "the second chunk must never be judged"
    assert monitor.spent > monitor.budget_usd


def test_computes_prf_records_cost_and_writes_artifacts(tmp_path: Path) -> None:
    """derive_on=valid: clean separation -> honest F1=1.0, cost + artifacts recorded."""
    benchmark = _benchmark()
    monitor = SpendMonitor(budget_usd=100.0)
    judge = _MockJudge(cost_usd=0.002, provider="DeepSeek")

    result, meter = run_placement(benchmark, judge, derive_on="valid", monitor=monitor)

    # Honest P/R/F1 computed from the derived threshold applied to the full test.
    assert result.honest.f1 == pytest.approx(1.0)
    assert result.honest.precision == pytest.approx(1.0)
    assert result.honest.recall == pytest.approx(1.0)
    assert result.honest.tp == 2 and result.honest.fp == 0 and result.honest.fn == 0

    # Cost recorded: one flat charge per judged pair (valid + test), served provider seen.
    expected_calls = len(_SPLITS["valid"]) + len(_SPLITS["test"])
    assert meter.n_judged == expected_calls
    assert meter.cost_usd == pytest.approx(0.002 * expected_calls)
    assert monitor.spent == pytest.approx(meter.cost_usd)
    assert meter.providers == {"DeepSeek": expected_calls}

    # Artifact: JSON on disk + a row in the Markdown table.
    artifact = build_artifact(
        result,
        meter,
        dataset="amazon_google",
        model="openrouter/deepseek/deepseek-v4-flash",
        provider={"order": ["DeepSeek"]},
        n_test=len(_SPLITS["test"]),
        n_test_pos=2,
        n_derive=len(_SPLITS["valid"]),
    )
    assert artifact["real_cost_usd"] == pytest.approx(0.002 * expected_calls)
    assert artifact["provider_served"] == ["DeepSeek"]
    assert artifact["gap_to_rf_floor_f1"] == pytest.approx(1.0 - RF_FLOOR_F1["amazon_google"])

    json_path, md_path = write_artifacts(artifact, tmp_path)
    assert json_path.exists() and md_path.exists()
    on_disk = json.loads(json_path.read_text())
    assert on_disk["honest"]["f1"] == pytest.approx(1.0)
    table = md_path.read_text()
    assert "openrouter/deepseek/deepseek-v4-flash" in table
    assert "amazon_google" in table
    assert "| honest F1 |" in table  # header present


def test_fixed_threshold_skips_derive_calls(tmp_path: Path) -> None:
    """derive_on=fixed:0.5: only the TEST split is judged (no derive-split spend)."""
    benchmark = _benchmark()
    monitor = SpendMonitor(budget_usd=100.0)
    judge = _MockJudge(cost_usd=0.01)

    result, meter = run_placement(benchmark, judge, derive_on="fixed:0.5", monitor=monitor)

    assert result.threshold_method == "fixed"
    assert result.derived_threshold == pytest.approx(0.5)
    assert result.honest.f1 == pytest.approx(1.0)  # 0.9 >= 0.5 matches, 0.1 < 0.5 rejected
    # Cheapest mode: NO valid calls, only the test pairs are judged.
    assert meter.n_judged == len(_SPLITS["test"])


def test_spend_cap_aborts_with_partials() -> None:
    """A tiny budget + nonzero cost -> BudgetExceeded carrying the paid partials."""
    benchmark = _benchmark()
    monitor = SpendMonitor(budget_usd=0.01)
    judge = _MockJudge(cost_usd=0.05)  # 0.05 > 0.01 -> the first metered call trips it

    with pytest.raises(BudgetExceeded) as excinfo:
        run_placement(benchmark, judge, derive_on="valid", monitor=monitor)

    assert excinfo.value.partial_judgements, "cap must carry the judgements already paid for"
    assert monitor.spent > monitor.budget_usd


def test_dry_run_judge_is_zero_cost() -> None:
    """The --dry-run judge scores from the comparison vector at exactly $0."""
    benchmark = _benchmark()
    monitor = SpendMonitor(budget_usd=100.0)

    result, meter = run_placement(benchmark, _DryRunJudge(), derive_on="fixed:0.5", monitor=monitor)

    assert meter.cost_usd == pytest.approx(0.0)
    assert monitor.spent == pytest.approx(0.0)
    assert meter.n_judged == len(_SPLITS["test"])
    assert 0.0 <= result.honest.f1 <= 1.0


def test_dry_run_config_defaults() -> None:
    """PlacementConfig carries the safe defaults the CLI relies on."""
    cfg = PlacementConfig(model="openrouter/deepseek/deepseek-v4-flash", datasets=["amazon_google"])
    assert cfg.max_usd == 5.0
    assert cfg.derive_on == "valid"
    assert cfg.provider is None
    assert cfg.dry_run is False
    assert cfg.max_concurrent == 50


def test_parse_derive_on_accepts_and_rejects() -> None:
    """--derive-on validation: valid/train/fixed:<0..1> accepted, junk rejected."""
    assert _parse_derive_on("valid") == "valid"
    assert _parse_derive_on("train") == "train"
    assert _parse_derive_on("fixed:0.5") == "fixed:0.5"
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_derive_on("bogus")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_derive_on("fixed:1.5")  # out of [0, 1]


def test_parse_provider_json_or_none() -> None:
    """--provider parses an OpenRouter routing JSON object, or None; rejects non-objects."""
    assert _parse_provider(None) is None
    assert _parse_provider('{"order": ["DeepSeek"], "allow_fallbacks": false}') == {
        "order": ["DeepSeek"],
        "allow_fallbacks": False,
    }
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_provider('["not", "an", "object"]')
