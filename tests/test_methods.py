"""Tests for the competing-method registry (``langres.methods``).

Covers, with zero real LLM spend:

- the five method factories each build a valid ``Resolver`` and run end-to-end on
  a tiny synthetic corpus (LLM/cascade with a MOCK client returning canned
  judgements + synthetic provenance cost);
- ``BudgetedModuleRunner`` integration for the LLM/cascade modules (pre-flight
  cap, ``BlindCostError`` on a $0 price, per-call isolation, and the cascade
  ``llm_cost_usd`` provenance key flowing through the runner's fallback);
- ``cascade_cost_track`` surfacing escalation-rate + llm-calls-per-candidate;
- a fast ``run_method`` race for the three deterministic methods on a synthetic
  ``Benchmark`` (FakeVectorIndex, no real embeddings) asserting both tracks
  populate; and a slow real-embedding race on ``FodorsZagatBenchmark``.

The un-fakeable real-network LLM glue lives behind an ``OPENROUTER_API_KEY``
skipif, so ``--cov`` stays green without a live key.
"""

import os
from types import SimpleNamespace
from typing import Any

import pytest

from langres.core.benchmark import (
    Benchmark,
    BlindCostError,
    BudgetedModuleRunner,
    MethodResult,
    _cost_track,
    gold_pairs_from_clusters,
    run_method,
)
from langres.core.indexes.vector_index import FakeVectorIndex
from langres.core.judges.embedding_score import EmbeddingScoreJudge
from langres.core.judges.weighted_average import WeightedAverageJudge
from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.modules.cascade import CascadeModule
from langres.core.modules.dspy_judge import DSPyJudge
from langres.core.modules.llm_judge import LLMJudge
from langres.core.modules.rapidfuzz import RapidfuzzModule
from langres.core.resolver import Resolver
from langres.methods import (
    ALL_METHODS,
    ZERO_SPEND_METHODS,
    cascade_cost_track,
    make_resolver_factory,
)
from langres.core.blockers.vector import VectorBlocker

# ---------------------------------------------------------------------------
# Synthetic, embedding-free benchmark (FakeVectorIndex) for fast tests
# ---------------------------------------------------------------------------


def _company_factory(record: dict[str, Any]) -> CompanySchema:
    return CompanySchema(**{f: record.get(f) for f in CompanySchema.model_fields})


class _FakeBlockingBenchmark(Benchmark[CompanySchema]):
    """A tiny CompanySchema benchmark whose blocker uses a FakeVectorIndex.

    Conforms to both the core ``Benchmark`` protocol and
    ``langres.methods.BlockingBenchmark`` (``schema`` + ``blocking_k`` +
    ``build_blocker``), so it drives both the factory tests and a *fast*
    ``run_method`` race with no real embeddings.
    """

    name = "fake"
    threshold_grid = (0.3, 0.5, 0.7, 0.9)
    schema = CompanySchema
    blocking_k = 2

    _CORPUS = [
        CompanySchema(id="c1", name="Acme Corporation", address="1 Main St"),
        CompanySchema(id="c1b", name="Acme Corporation", address="1 Main St"),
        CompanySchema(id="c2", name="Zeta Holdings", address="9 Pine Rd"),
        CompanySchema(id="c3", name="Beta Incorporated", address="2 Oak Ave"),
        CompanySchema(id="c3b", name="Beta Incorporated", address="2 Oak Ave"),
        CompanySchema(id="c4", name="Omega Limited", address="7 Elm Blvd"),
    ]
    _GOLD = [{"c1", "c1b"}, {"c2"}, {"c3", "c3b"}, {"c4"}]

    def build_blocker(self, k_neighbors: int) -> VectorBlocker[CompanySchema]:
        return VectorBlocker(
            schema_factory=_company_factory,
            text_field_extractor=lambda e: e.name,
            vector_index=FakeVectorIndex(),
            k_neighbors=k_neighbors,
        )

    def load(self) -> tuple[list[CompanySchema], list[set[str]], set[frozenset[str]]]:
        gold = [set(c) for c in self._GOLD]
        return list(self._CORPUS), gold, gold_pairs_from_clusters(gold)

    def split(
        self,
        corpus: list[CompanySchema],
        gold_clusters: list[set[str]],
        *,
        seed: int,
    ) -> tuple[list[CompanySchema], list[CompanySchema], list[set[str]], list[set[str]]]:
        by_id = {r.id: r for r in corpus}
        train_clusters = [{"c1", "c1b"}, {"c2"}]
        test_clusters = [{"c3", "c3b"}, {"c4"}]
        train = [by_id[i] for c in train_clusters for i in sorted(c)]
        test = [by_id[i] for c in test_clusters for i in sorted(c)]
        return train, test, train_clusters, test_clusters


def _records() -> list[dict[str, Any]]:
    return [r.model_dump() for r in _FakeBlockingBenchmark._CORPUS]


# ---------------------------------------------------------------------------
# Mock LLM clients (no network, no spend) + a scripted embedding model
# ---------------------------------------------------------------------------


def _fake_response(content: str) -> SimpleNamespace:
    """An OpenAI/LiteLLM-shaped response carrying ``content`` + token usage."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[choice], usage=usage)


class _MockLiteLLMClient:
    """LiteLLM-shaped client for ``LLMJudge`` (``client.completion(...)``)."""

    def __init__(
        self, content: str = "MATCH\nScore: 0.90\nReasoning: same", boom_on_call: int | None = None
    ) -> None:
        self._content = content
        self._boom_on_call = boom_on_call
        self.calls = 0

    def completion(
        self, *, model: str, messages: list[dict[str, str]], temperature: float
    ) -> SimpleNamespace:
        self.calls += 1
        if self._boom_on_call is not None and self.calls == self._boom_on_call:
            raise RuntimeError("simulated LLM failure")
        return _fake_response(self._content)


class _MockOpenAIClient:
    """OpenAI-shaped client for ``CascadeModule`` (``client.chat.completions.create``)."""

    def __init__(self, content: str = "MATCH\nScore: 0.80\nReasoning: x") -> None:
        self._content = content
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(
        self, *, model: str, messages: list[dict[str, str]], temperature: float
    ) -> SimpleNamespace:
        self.calls += 1
        return _fake_response(self._content)


def _dummy_lm() -> Any:
    """A DSPy ``DummyLM`` for ``dspy_judge`` — the factory injects it as the LM (no spend)."""
    from dspy.utils.dummies import DummyLM

    return DummyLM([{"reasoning": "same", "match": "True", "match_probability": "0.9"}] * 20)


class _ScriptedEmbeddingModel:
    """Cascade embedding double: pops a preset ``[left_vec, right_vec]`` per encode.

    Lets a test force each pair's stage-1 cosine into the early-exit-low,
    early-exit-high, or LLM-escalation band deterministically (no real MiniLM).
    """

    def __init__(self, pairs: list[list[list[float]]]) -> None:
        self._pairs = pairs
        self._i = 0

    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> Any:
        import numpy as np

        pair = self._pairs[self._i]
        self._i += 1
        return np.array(pair, dtype=np.float32)


class _ConstantEmbeddingModel:
    """Cascade embedding double: every encode returns the same ``[left, right]`` pair."""

    def __init__(self, pair: list[list[float]]) -> None:
        self._pair = pair

    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> Any:
        import numpy as np

        return np.array(self._pair, dtype=np.float32)


# Vector pairs whose cosine lands in a known band for low=0.3, high=0.9.
_ESCALATE = [[1.0, 0.0], [1.0, 1.0]]  # cosine ~0.707 -> uncertain -> LLM
_EXIT_LOW = [[1.0, 0.0], [0.0, 1.0]]  # cosine 0.0 -> early exit (no match)


@pytest.fixture(autouse=True)
def _patch_completion_cost(mocker: Any) -> None:
    """Make ``litellm.completion_cost`` deterministic (so no real pricing call)."""
    mocker.patch("litellm.completion_cost", return_value=0.002)


def _candidates(n: int) -> list[ERCandidate[CompanySchema]]:
    return [
        ERCandidate(
            left=CompanySchema(id=f"l{i}", name=f"Acme {i}"),
            right=CompanySchema(id=f"r{i}", name=f"Acme {i} Inc"),
            blocker_name="test",
            similarity_score=0.8,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Factory wiring
# ---------------------------------------------------------------------------


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        make_resolver_factory("does_not_exist", _FakeBlockingBenchmark())


@pytest.mark.parametrize(
    ("method", "module_type", "needs_comparator", "client"),
    [
        ("rapidfuzz", RapidfuzzModule, False, None),
        ("weighted_average", WeightedAverageJudge, True, None),
        ("embedding_cosine", EmbeddingScoreJudge, False, None),
        # The injected client must match each scorer's call shape: LLMJudge calls
        # ``client.completion(...)`` (LiteLLM-shaped); CascadeModule calls
        # ``client.chat.completions.create(...)`` (OpenAI-shaped).
        ("llm_judge", LLMJudge, False, _MockLiteLLMClient()),
        ("cascade", CascadeModule, False, _MockOpenAIClient()),
        # ``dspy_judge`` takes a DSPy LM as its injected client (a ``DummyLM``
        # here), distinct from the LiteLLM/OpenAI clients above.
        ("dspy_judge", DSPyJudge, False, _dummy_lm()),
    ],
)
def test_factory_builds_valid_resolver(
    method: str, module_type: type, needs_comparator: bool, client: object
) -> None:
    """Each method builds a Resolver with the right scorer + comparator wiring."""
    factory = make_resolver_factory(method, _FakeBlockingBenchmark(), llm_client=client)
    resolver = factory(0.6)

    assert isinstance(resolver, Resolver)
    assert isinstance(resolver.blocker, VectorBlocker)
    assert isinstance(resolver.module, module_type)
    assert resolver.clusterer.threshold == pytest.approx(0.6)
    assert (resolver.comparator is not None) is needs_comparator


def test_cascade_factory_without_client_does_not_require_live_key() -> None:
    """Building the cascade resolver with no injected client still succeeds.

    The placeholder key satisfies CascadeModule's constructor; the real client is
    injected later (W4) — never a live-key requirement at build time. Here no
    client is injected, so the module's client stays unset.
    """
    resolver = make_resolver_factory("cascade", _FakeBlockingBenchmark())(0.5)
    assert isinstance(resolver.module, CascadeModule)
    assert resolver.module._llm_client is None


def test_dspy_judge_factory_prices_from_pinned_table() -> None:
    """The ``dspy_judge`` factory wires ``price_per_1k_tokens`` from the pinned table.

    Closes the $0-spend gap: on a KNOWN paid model the factory-built judge carries
    the worst-case per-1k price (dearer of input/output), so its per-pair
    ``provenance["cost_usd"]`` is real — and flows through the DEFAULT ``_cost_track``
    surface (no custom ``cost_track_fn``) that ``run_method`` uses.
    """
    from langres.clients.openrouter import per_token_worst_price

    model = "openrouter/z-ai/glm-5.2"
    factory = make_resolver_factory(
        "dspy_judge", _FakeBlockingBenchmark(), llm_client=_dummy_lm(), llm_model=model
    )
    judge = factory(0.5).module
    assert isinstance(judge, DSPyJudge)
    expected = per_token_worst_price(model) * 1_000.0
    assert judge.price_per_1k_tokens == pytest.approx(expected)
    assert judge.price_per_1k_tokens > 0.0

    # Fake token counts (as a real paid call would carry) now cost real money, and
    # that honest cost lands in the ``cost_usd`` key the default ``_cost_track`` reads.
    cost = judge._cost_usd(1000, 500)
    assert cost > 0.0
    priced = PairwiseJudgement(
        left_id="a",
        right_id="b",
        score=0.9,
        score_type="prob_llm",
        decision_step="dspy_judgment",
        provenance={"cost_usd": cost, "prompt_tokens": 1000, "completion_tokens": 500},
    )
    assert _cost_track([priced]).usd_total == pytest.approx(cost)
    assert _cost_track([priced]).usd_total > 0.0


def test_dspy_judge_factory_unknown_model_keeps_zero_price() -> None:
    """An unknown model id keeps ``price_per_1k_tokens = 0.0`` (zero-spend/test runs).

    Mirrors ``register_runtime_model_price`` returning ``None`` for unknown ids: no
    crash, cost stays $0 rather than guessing a price.
    """
    factory = make_resolver_factory(
        "dspy_judge",
        _FakeBlockingBenchmark(),
        llm_client=_dummy_lm(),
        llm_model="unknown/model-not-in-table",
    )
    judge = factory(0.5).module
    assert isinstance(judge, DSPyJudge)
    assert judge.price_per_1k_tokens == 0.0


def test_dspy_price_per_1k_known_and_unknown() -> None:
    """``_dspy_price_per_1k`` maps a known model to its worst-case per-1k, unknown to 0.0."""
    from langres.clients.openrouter import per_token_worst_price
    from langres.methods import _dspy_price_per_1k

    model = "openrouter/z-ai/glm-5.2"
    assert _dspy_price_per_1k(model) == pytest.approx(per_token_worst_price(model) * 1_000.0)
    assert _dspy_price_per_1k("unknown/model-not-in-table") == 0.0


def test_factory_yields_fresh_independent_blockers() -> None:
    """Each ``factory(threshold)`` builds a NEW blocker (no shared, pre-built index)."""
    factory = make_resolver_factory("embedding_cosine", _FakeBlockingBenchmark())
    a = factory(0.5)
    b = factory(0.5)
    assert a.blocker is not b.blocker
    assert a.module is not b.module


def test_blocking_held_constant_uses_pinned_k() -> None:
    """The factory pins blocking to ``benchmark.blocking_k`` for every method."""
    bench = _FakeBlockingBenchmark()
    for method in ALL_METHODS:
        factory = make_resolver_factory(method, bench, llm_client=_MockOpenAIClient())
        resolver = factory(0.5)
        assert isinstance(resolver.blocker, VectorBlocker)
        assert resolver.blocker.k_neighbors == bench.blocking_k


# ---------------------------------------------------------------------------
# Deterministic methods: end-to-end predict on the tiny corpus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ZERO_SPEND_METHODS)
def test_deterministic_method_runs_end_to_end(method: str) -> None:
    factory = make_resolver_factory(method, _FakeBlockingBenchmark())
    judgements = factory(0.5).predict(_records())

    assert judgements  # candidates were generated and scored
    assert all(isinstance(j, PairwiseJudgement) for j in judgements)
    # Zero-spend: no cost recorded in provenance.
    assert all(
        "cost_usd" not in j.provenance and "llm_cost_usd" not in j.provenance for j in judgements
    )


def test_rapidfuzz_and_weighted_average_score_the_same_fields() -> None:
    """rapidfuzz extractors mirror Comparator.from_schema (same fields raced)."""
    from langres.core.comparator import Comparator
    from langres.methods import _rapidfuzz_extractors

    extractors = _rapidfuzz_extractors(CompanySchema)
    comparator_fields = {s.name for s in Comparator.from_schema(CompanySchema).feature_specs}
    assert set(extractors) == comparator_fields
    assert "id" not in extractors  # id is excluded from comparison


# ---------------------------------------------------------------------------
# LLM / cascade: end-to-end predict with a MOCK client (zero spend)
# ---------------------------------------------------------------------------


def test_llm_judge_runs_end_to_end_with_mock_client() -> None:
    client = _MockLiteLLMClient(content="MATCH\nScore: 0.95\nReasoning: same co")
    factory = make_resolver_factory("llm_judge", _FakeBlockingBenchmark(), llm_client=client)
    judgements = factory(0.5).predict(_records())

    assert judgements
    assert client.calls == len(judgements)  # one LLM call per candidate
    assert all(j.score == pytest.approx(0.95) for j in judgements)
    assert all(j.provenance["cost_usd"] == pytest.approx(0.002) for j in judgements)


def test_cascade_runs_end_to_end_with_mock_client() -> None:
    """Cascade escalates the uncertain band to the (mock) LLM, recording llm_cost_usd."""
    bench = _FakeBlockingBenchmark()
    client = _MockOpenAIClient(content="MATCH\nScore: 0.70\nReasoning: maybe")
    resolver = make_resolver_factory("cascade", bench, llm_client=client)(0.5)

    # Force every pair into the uncertain band so each escalates to the LLM.
    cascade = resolver.module
    assert isinstance(cascade, CascadeModule)
    cascade._embedding_model = _ConstantEmbeddingModel(_ESCALATE)

    judgements = resolver.predict(_records())
    assert judgements
    assert client.calls == len(judgements)
    assert all(j.decision_step == "llm_judgment" for j in judgements)
    assert all(j.provenance["llm_cost_usd"] == pytest.approx(0.002) for j in judgements)


# ---------------------------------------------------------------------------
# BudgetedModuleRunner integration (LLM + cascade modules)
# ---------------------------------------------------------------------------


def test_budgeted_runner_preflight_cap_on_llm_module() -> None:
    module: LLMJudge[CompanySchema] = LLMJudge(client=_MockLiteLLMClient(), model="mock")
    runner = BudgetedModuleRunner(module, budget_usd=1.0, budget_soft_usd=1.0)
    # worst_case_per_pair = 1 * 0.5 = 0.5 -> floor(1.0/0.5) = 2 kept of 5.
    out = runner.run(_candidates(5), price_per_token_or_pair=0.5)
    assert len(out) == 2
    assert runner.dropped_by_cap_count == 3


def test_budgeted_runner_blind_cost_error_on_zero_price() -> None:
    module: LLMJudge[CompanySchema] = LLMJudge(client=_MockLiteLLMClient(), model="mock")
    runner = BudgetedModuleRunner(module, budget_usd=10.0, budget_soft_usd=10.0)
    with pytest.raises(BlindCostError, match="blind"):
        runner.run(_candidates(3), price_per_token_or_pair=0.0)


def test_budgeted_runner_isolates_per_call_failure() -> None:
    """A single failed LLM call is skipped; already-paid judgements survive."""
    module: LLMJudge[CompanySchema] = LLMJudge(
        client=_MockLiteLLMClient(boom_on_call=2), model="mock"
    )
    runner = BudgetedModuleRunner(module, budget_usd=100.0, budget_soft_usd=100.0)
    out = runner.run(_candidates(3), price_per_token_or_pair=0.001)
    assert len(out) == 2  # the 2nd candidate's call raised -> skipped
    assert runner.skipped_count == 1
    assert runner.labeled_count == 2


def test_budgeted_runner_tallies_cascade_llm_cost_usd_key() -> None:
    """Cascade's ``llm_cost_usd`` provenance flows through the runner's fallback tally."""
    cascade: CascadeModule[CompanySchema] = CascadeModule(llm_api_key="injected", llm_model="mock")
    cascade._llm_client = _MockOpenAIClient()
    cascade._embedding_model = _ConstantEmbeddingModel(_ESCALATE)

    runner = BudgetedModuleRunner(cascade, budget_usd=100.0, budget_soft_usd=100.0)
    out = runner.run(_candidates(3), price_per_token_or_pair=0.001)
    assert len(out) == 3
    # 3 escalated pairs * $0.002 each, read from the llm_cost_usd fallback key.
    assert runner.total_spent_usd == pytest.approx(0.006)


# ---------------------------------------------------------------------------
# cascade_cost_track: escalation-rate + llm-calls-per-candidate
# ---------------------------------------------------------------------------


def test_cascade_cost_track_surfaces_escalation() -> None:
    cascade: CascadeModule[CompanySchema] = CascadeModule(llm_api_key="injected", llm_model="mock")
    cascade._llm_client = _MockOpenAIClient()
    # 2 escalate (LLM), 1 early-exits low -> escalation rate 2/3.
    cascade._embedding_model = _ScriptedEmbeddingModel([_ESCALATE, _EXIT_LOW, _ESCALATE])

    judgements = list(cascade.forward(iter(_candidates(3))))
    track = cascade_cost_track(judgements)

    assert track.escalation_rate == pytest.approx(2 / 3)
    assert track.llm_calls_per_candidate == pytest.approx(2 / 3)
    assert track.usd_total == pytest.approx(0.004)  # 2 escalations * $0.002


def test_cascade_cost_track_empty_is_zero() -> None:
    track = cascade_cost_track([])
    assert track.escalation_rate == 0.0
    assert track.llm_calls_per_candidate == 0.0
    assert track.usd_total == 0.0


# ---------------------------------------------------------------------------
# run_method races
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ZERO_SPEND_METHODS)
def test_fast_run_method_race_populates_both_tracks(method: str) -> None:
    """A fast (FakeVectorIndex) race: each deterministic method yields full tracks."""
    bench = _FakeBlockingBenchmark()
    result = run_method(bench, make_resolver_factory(method, bench), seed=0)

    assert isinstance(result, MethodResult)
    assert result.dataset == "fake"
    # Pair track populated.
    assert 0.0 <= result.pair.f1 <= 1.0
    assert result.pair.pr_curve is not None
    assert len(result.pair.pr_curve) == len(bench.threshold_grid)
    # Pipeline track populated (quality is not asserted here: the FakeVectorIndex
    # emits content-blind similarities, so embedding_cosine can sit below the
    # all-singletons floor — that is a property of the synthetic blocker, not a
    # bug. The slow Fodors-Zagat race exercises real embeddings).
    assert 0.0 <= result.pipeline.bcubed_f1 <= 1.0
    assert 0.0 <= result.pipeline.sanity_floor_f1 <= 1.0
    assert result.pipeline.delta_above_floor == pytest.approx(
        result.pipeline.bcubed_f1 - result.pipeline.sanity_floor_f1
    )
    # Zero-spend.
    assert result.cost.usd_total == 0.0


@pytest.mark.slow
@pytest.mark.parametrize("method", ZERO_SPEND_METHODS)
def test_slow_fodors_zagat_race(method: str) -> None:
    """The zero-spend mini-race on real Fodors-Zagat embeddings: full tracks per method."""
    from langres.data.er_benchmarks import FodorsZagatBenchmark

    bench = FodorsZagatBenchmark()
    result = run_method(bench, make_resolver_factory(method, bench), seed=0)

    assert result.dataset == "fodors_zagat"
    assert 0.0 <= result.pair.f1 <= 1.0
    assert result.pair.pr_curve is not None
    assert 0.0 <= result.pipeline.bcubed_f1 <= 1.0
    assert result.cost.usd_total == 0.0  # zero-spend methods


# ---------------------------------------------------------------------------
# Real-network glue (un-fakeable) — skipped without a live key
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="needs OPENROUTER_API_KEY for a real LLM call",
)
def test_llm_judge_real_network_smoke() -> None:  # pragma: no cover - requires live key
    """A single real LLM judgement via the injected env client (W4 path)."""
    from langres.clients import Settings, create_llm_client

    bench = _FakeBlockingBenchmark()
    client = create_llm_client(Settings())
    factory = make_resolver_factory(
        "llm_judge", bench, llm_client=client, llm_model="openrouter/openai/gpt-4o-mini"
    )
    judgements = factory(0.5).predict(_records()[:2])
    assert all(0.0 <= j.score <= 1.0 for j in judgements)
