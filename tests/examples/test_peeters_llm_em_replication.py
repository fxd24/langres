"""$0 smoke tests for the Peeters LLM-EM live/dry-run harness.

Every test runs at **$0** — no API key, no network, no real model call. A fake
client (returning canned "Yes"/"No" answers with a fake usage/cost) drives the
live path so the whole flow — build candidates, judge, charge the SpendMonitor,
aggregate usage, score pairwise F1 — is verified without spending. Proves:

1. The dry-run renders + counts tokens with zero API calls, priced from the table.
2. The live core runs end-to-end and reports F1 + the aggregated usage vector +
   the real billed cost.
3. The hard SpendMonitor cap FIRES (partial run) when cost crosses the budget.
4. The safety guards (priced-model assertion, model resolution) behave.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from examples.research.peeters_llm_em_replication import (
    LIVE_PROVIDER,
    PAID_MODELS,
    _aggregate_usage,
    _assert_priced,
    _build_live_judge,
    _resolve_paid_models,
    build_compared_pairs,
    confusion_and_agreement,
    dry_run,
    run_compare_archived,
    run_live,
    stratified_subset_indices,
)
from langres.clients.openrouter import PRICES_PER_1M
from langres.data.peeters import (
    get_peeters_replication,
    load_peeters_sample,
    render_sample_prompts,
)

_MODEL = "openrouter/openai/gpt-4o-mini-2024-07-18"


# --------------------------------------------------------------------------- #
# Fake client: canned answers + a fake usage/cost on every response.
# --------------------------------------------------------------------------- #


def _response(content: str, *, cost: float, in_tok: int = 80, out_tok: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=in_tok,
            completion_tokens=out_tok,
            cost=cost,  # parse_openrouter_billing reads usage.cost -> cost_is_real
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
        _hidden_params={},
        provider="fake-provider",
        model="fake",
    )


class _FakeClient:
    """A stand-in for litellm: hands back canned answers with a fixed per-call cost.

    Records the kwargs of the most recent ``completion`` call in ``last_kwargs`` so
    a test can assert what actually reached the wire (e.g. the pinned provider
    routing block), without spending anything.
    """

    def __init__(self, answers: list[str], *, cost_per_call: float) -> None:
        self._answers = answers
        self._i = 0
        self._cost = cost_per_call
        self.last_kwargs: dict[str, Any] | None = None

    def completion(self, **kwargs: Any) -> SimpleNamespace:
        self.last_kwargs = kwargs
        content = self._answers[self._i]
        self._i += 1
        return _response(content, cost=self._cost)


def _perfect_answers(spec: Any) -> list[str]:
    """ "Yes" for the gold positives, "No" for the rest — a perfect-F1 answer set."""
    return ["Yes" if label == 1 else "No" for _l, _r, label in load_peeters_sample(spec)]


# --------------------------------------------------------------------------- #
# Priced-model guards
# --------------------------------------------------------------------------- #


def test_paid_models_are_all_priced() -> None:
    """Every paid-run model must have a PRICES_PER_1M entry (else the cap is blind)."""
    for model in PAID_MODELS:
        assert model in PRICES_PER_1M


def test_assert_priced_rejects_unpriced_model() -> None:
    with pytest.raises(SystemExit, match="no PRICES_PER_1M entry"):
        _assert_priced(["openrouter/openai/not-a-real-model"])


def test_resolve_paid_models_defaults_to_both() -> None:
    assert _resolve_paid_models(None) == list(PAID_MODELS)
    assert _resolve_paid_models("both") == list(PAID_MODELS)
    assert _resolve_paid_models(_MODEL) == [_MODEL]


def test_resolve_paid_models_rejects_unknown() -> None:
    with pytest.raises(SystemExit, match="not a paid-run model"):
        _resolve_paid_models("gpt-4-0613")


# --------------------------------------------------------------------------- #
# Dry run ($0, injected counter — no litellm/tiktoken needed)
# --------------------------------------------------------------------------- #


def test_dry_run_counts_and_prices_with_injected_counter() -> None:
    spec = get_peeters_replication("abt-buy")
    report = dry_run(spec, _MODEL, count_tokens=lambda _prompt: 10)
    n = report["n_pairs"]
    assert n == 1206
    assert report["input_tokens"] == 10 * n
    assert report["output_tokens_est"] == 2 * n
    assert report["max_input_tokens"] == 10
    in_1m, out_1m = PRICES_PER_1M[_MODEL]
    expected = (10 * n) * in_1m / 1e6 + (2 * n) * out_1m / 1e6
    assert report["est_usd"] == pytest.approx(expected)


@pytest.mark.slow
def test_dry_run_real_token_total_matches_measurement() -> None:
    """The real tiktoken count over all 1206 rendered prompts is 100,256 input tokens.

    Pins prompt-rendering fidelity: a drift in the live template/serializer would
    move this away from the value measured with o200k_base before any paid run.
    """
    pytest.importorskip("litellm")
    spec = get_peeters_replication("abt-buy")
    report = dry_run(spec, _MODEL)
    assert report["input_tokens"] == 100256
    assert report["output_tokens_est"] == 2412


# --------------------------------------------------------------------------- #
# Live core ($0 via the fake client)
# --------------------------------------------------------------------------- #


def test_run_live_end_to_end_with_perfect_answers() -> None:
    spec = get_peeters_replication("abt-buy")
    answers = _perfect_answers(spec)
    client = _FakeClient(answers, cost_per_call=0.0)
    result = run_live(spec, _MODEL, budget_usd=1.0, client=client)

    assert result["n_judged"] == 1206
    assert result["budget_hit"] is False
    assert result["f1"] == pytest.approx(100.0)
    assert result["precision"] == pytest.approx(100.0)
    assert result["recall"] == pytest.approx(100.0)
    assert result["fp"] == 0 and result["fn"] == 0
    # Usage aggregated across all 1206 fake calls (80 in / 2 out each).
    assert result["usage"]["input_tokens"] == 1206 * 80
    assert result["usage"]["output_tokens"] == 1206 * 2
    assert result["published_f1"] == 90.95


def test_run_live_aggregates_real_billed_cost() -> None:
    spec = get_peeters_replication("abt-buy")
    answers = _perfect_answers(spec)
    client = _FakeClient(answers, cost_per_call=0.0001)
    result = run_live(spec, _MODEL, budget_usd=1.0, client=client)

    assert result["cost_is_real"] is True
    assert result["real_cost_usd"] == pytest.approx(1206 * 0.0001)
    assert result["usd_per_1k_pairs"] == pytest.approx(0.0001 * 1000.0)


def test_run_live_spend_cap_fires_and_returns_partial() -> None:
    """The hard cap stops the run: high per-call cost + tiny budget => partial."""
    spec = get_peeters_replication("abt-buy")
    answers = _perfect_answers(spec)
    client = _FakeClient(answers, cost_per_call=0.5)
    result = run_live(spec, _MODEL, budget_usd=1.0, client=client)

    assert result["budget_hit"] is True
    assert result["n_judged"] == 3  # 0.5*3 = 1.5 > 1.0 budget; stops on the 3rd
    assert result["n_pairs"] == 1206


def test_aggregate_usage_sums_vectors() -> None:
    judgements = [
        SimpleNamespace(provenance={"usage": {"input_tokens": 5, "output_tokens": 1}}),
        SimpleNamespace(provenance={"usage": {"input_tokens": 7, "output_tokens": 2}}),
        SimpleNamespace(provenance={}),  # a judgement with no usage vector
    ]
    usage = _aggregate_usage(judgements, _MODEL)
    assert usage.input_tokens == 12
    assert usage.output_tokens == 3
    assert usage.model == _MODEL


# --------------------------------------------------------------------------- #
# Task 1 — the live judge pins OpenRouter -> OpenAI provider routing
# --------------------------------------------------------------------------- #


def test_gpt4o_published_f1_corrected_to_90_47() -> None:
    """The gpt-4o cell is the corrected 90.47 (was a wrong 89.33), gpt-4o-mini stays 90.95."""
    assert PAID_MODELS["openrouter/openai/gpt-4o-2024-08-06"] == 90.47
    assert PAID_MODELS["openrouter/openai/gpt-4o-mini-2024-07-18"] == 90.95
    assert 89.33 not in PAID_MODELS.values()


def test_build_live_judge_pins_openai_provider() -> None:
    spec = get_peeters_replication("abt-buy")
    judge = _build_live_judge(spec, _MODEL)
    assert judge.provider == {"order": ["OpenAI"], "allow_fallbacks": False}
    assert judge.provider == LIVE_PROVIDER


def test_live_run_sends_pinned_provider_on_the_wire() -> None:
    """The pinned provider reaches the client as ``extra_body['provider']`` — verified at $0."""
    spec = get_peeters_replication("abt-buy")
    client = _FakeClient(["Yes", "No", "Yes"], cost_per_call=0.0)
    run_live(spec, _MODEL, budget_usd=1.0, client=client, indices=[0, 1, 2])
    assert client.last_kwargs is not None
    extra_body = client.last_kwargs["extra_body"]
    assert extra_body["provider"] == LIVE_PROVIDER
    # usage accounting still requested alongside the provider pin.
    assert extra_body["usage"] == {"include": True}


# --------------------------------------------------------------------------- #
# Task 2 — --limit stratified subset (ratio-preserving + deterministic)
# --------------------------------------------------------------------------- #


def test_stratified_subset_preserves_positive_ratio_on_real_slice() -> None:
    """A 150-pair subset of Abt-Buy keeps the ~17.1% positive ratio (within one pair)."""
    spec = get_peeters_replication("abt-buy")
    labels = [label for _l, _r, label in load_peeters_sample(spec)]
    full_ratio = sum(labels) / len(labels)

    idx = stratified_subset_indices(labels, 150, seed=0)
    assert len(idx) == 150
    assert idx == sorted(idx)  # ascending -> keeps sample alignment
    assert len(set(idx)) == 150  # no duplicates
    sub_labels = [labels[i] for i in idx]
    sub_ratio = sum(sub_labels) / len(sub_labels)
    # Preserved to within one pair's worth of rounding (1/150).
    assert abs(sub_ratio - full_ratio) <= 1.0 / 150
    # Concretely: round(150 * 206/1206) = 26 positives, 124 negatives.
    assert sum(sub_labels) == 26


def test_stratified_subset_first_n_would_be_all_positive() -> None:
    """Guard the motivation: naive first-N is all-positive; stratified is not."""
    spec = get_peeters_replication("abt-buy")
    labels = [label for _l, _r, label in load_peeters_sample(spec)]
    # The file is positives-block then negatives-block, so first 150 are all matches.
    assert sum(labels[:150]) == 150
    idx = stratified_subset_indices(labels, 150, seed=0)
    assert 0 < sum(labels[i] for i in idx) < 150  # a real mix


def test_stratified_subset_is_deterministic_and_seed_sensitive() -> None:
    labels = [1] * 206 + [0] * 1000
    a = stratified_subset_indices(labels, 150, seed=0)
    b = stratified_subset_indices(labels, 150, seed=0)
    c = stratified_subset_indices(labels, 150, seed=7)
    assert a == b  # same seed -> identical subset
    assert a != c  # a different seed picks a different subset
    # ratio preserved for both seeds
    for idx in (a, c):
        assert sum(labels[i] for i in idx) == 26


def test_stratified_subset_limit_ge_n_returns_all() -> None:
    labels = [1, 1, 0, 0, 0]
    assert stratified_subset_indices(labels, 5, seed=0) == [0, 1, 2, 3, 4]
    assert stratified_subset_indices(labels, 99, seed=0) == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# Task 3/4 — per-pair agreement vs the archive (comparison core, $0)
# --------------------------------------------------------------------------- #


def _pp(left_id: str, right_id: str, label: int, prompt: str) -> SimpleNamespace:
    return SimpleNamespace(left_id=left_id, right_id=right_id, label=label, prompt=prompt)


def _cand(left: str, right: str) -> SimpleNamespace:
    return SimpleNamespace(left=left, right=right)


def _judg(score: float, reasoning: str) -> SimpleNamespace:
    # A binary live judge DECIDES; ``build_compared_pairs`` reads ``.decision``
    # (its ``score`` is now None/p_yes). Derive the decision from the intended
    # verdict so existing call sites (_judg(1.0, "Yes") / _judg(0.0, "No")) hold.
    return SimpleNamespace(decision=score >= 0.5, score=None, reasoning=reasoning)


def test_build_compared_pairs_and_confusion_with_deliberate_disagreement() -> None:
    prompts = [
        _pp("a0", "b0", 1, "P0"),
        _pp("a1", "b1", 0, "P1"),
        _pp("a2", "b2", 1, "P2"),  # gold match; ours=Yes, theirs=No -> disagreement
    ]
    candidates = [_cand("L0", "R0"), _cand("L1", "R1"), _cand("L2", "R2")]
    our_judgements = [_judg(1.0, "Yes"), _judg(0.0, "No"), _judg(1.0, "Yes")]
    archived = [
        {"prompt": "P0", "answer": "Yes"},
        {"prompt": "P1", "answer": "No"},
        {"prompt": "P2", "answer": "No"},
    ]

    compared = build_compared_pairs(prompts, candidates, our_judgements, archived, str)
    assert [c.our_verdict for c in compared] == [1, 0, 1]
    assert [c.their_verdict for c in compared] == [1, 0, 0]

    summary = confusion_and_agreement(compared)
    assert summary["agreement_rate"] == pytest.approx(2 / 3)
    assert summary["confusion"] == {
        "both_yes": 1,
        "both_no": 1,
        "we_yes_they_no": 1,
        "we_no_they_yes": 0,
    }

    disagreements = [c for c in compared if c.our_verdict != c.their_verdict]
    assert len(disagreements) == 1
    d = disagreements[0]
    assert (d.left_id, d.right_id, d.label) == ("a2", "b2", 1)
    assert d.their_answer == "No" and d.our_answer == "Yes"
    assert d.left == "L2" and d.right == "R2"  # serialized records shown to a human


def test_build_compared_pairs_raises_on_prompt_mismatch() -> None:
    prompts = [_pp("a0", "b0", 1, "P0"), _pp("a1", "b1", 0, "OUR-PROMPT")]
    candidates = [_cand("L0", "R0"), _cand("L1", "R1")]
    our_judgements = [_judg(1.0, "Yes"), _judg(0.0, "No")]
    archived = [
        {"prompt": "P0", "answer": "Yes"},
        {"prompt": "ARCHIVED-PROMPT", "answer": "No"},  # != our rendered prompt
    ]
    with pytest.raises(SystemExit, match="prompt mismatch at compared row 1"):
        build_compared_pairs(prompts, candidates, our_judgements, archived, str)


def _perfect_archive(spec: Any) -> list[dict[str, str]]:
    """Their archive over the FULL slice: the rendered prompt + a perfect Yes/No answer."""
    return [
        {"prompt": p.prompt, "answer": "Yes" if p.label == 1 else "No"}
        for p in render_sample_prompts(spec)
    ]


def test_run_compare_archived_end_to_end_at_zero_dollars() -> None:
    """Full comparison path on Abt-Buy, fake client + injected archive, one planted disagreement."""
    spec = get_peeters_replication("abt-buy")
    limit, seed = 20, 0
    archived = _perfect_archive(spec)

    labels = [label for _l, _r, label in load_peeters_sample(spec)]
    idx = stratified_subset_indices(labels, limit, seed)
    subset_labels = [labels[i] for i in idx]  # 3 positives (idx<206) then 17 negatives
    assert (sum(subset_labels), len(subset_labels)) == (3, 20)

    # Our answers over the subset in idx order: perfect, except flip subset position 0
    # (a gold positive) to "No" -> exactly one we-No / they-Yes disagreement.
    our_answers = ["Yes" if lbl == 1 else "No" for lbl in subset_labels]
    assert subset_labels[0] == 1
    our_answers[0] = "No"
    client = _FakeClient(our_answers, cost_per_call=0.0001)

    report = run_compare_archived(
        spec, _MODEL, budget_usd=1.0, client=client, archived=archived, limit=limit, seed=seed
    )

    assert report["n_judged"] == 20
    assert report["budget_hit"] is False
    assert report["agreement_rate"] == pytest.approx(19 / 20)
    assert report["confusion"] == {
        "both_yes": 2,
        "both_no": 17,
        "we_yes_they_no": 0,
        "we_no_they_yes": 1,
    }

    # The single disagreement is a gold positive we said No to, they said Yes.
    assert len(report["disagreements"]) == 1
    d = report["disagreements"][0]
    assert d["gold_label"] == 1
    assert d["their_answer"] == "Yes" and d["our_answer"] == "No"

    # Our metrics on the judged subset: tp=2, fp=0, fn=1 (the flipped positive).
    assert report["ours"]["tp"] == 2 and report["ours"]["fp"] == 0 and report["ours"]["fn"] == 1
    assert report["ours"]["precision"] == pytest.approx(100.0)
    assert report["ours"]["recall"] == pytest.approx(2 / 3 * 100.0)
    # Their (perfect) verdicts on the same subset: F1 100.
    assert report["theirs_subset"]["f1"] == pytest.approx(100.0)
    # Published full-set number carried through for reference.
    assert report["published_f1"] == 90.95

    # Usage + real billed cost aggregated over the 20 judged pairs.
    assert report["usage"]["input_tokens"] == 20 * 80
    assert report["real_cost_usd"] == pytest.approx(20 * 0.0001)
    assert report["cost_is_real"] is True


def test_run_compare_archived_rejects_archived_row_count_mismatch() -> None:
    """A wrong archived row count => the alignment is off => fail loud, before any judging."""
    spec = get_peeters_replication("abt-buy")
    short_archive = [{"prompt": "x", "answer": "Yes"}] * 10  # != 1206
    client = _FakeClient(["Yes"], cost_per_call=0.0)
    with pytest.raises(SystemExit, match="rows but the abt-buy pair set"):
        run_compare_archived(spec, _MODEL, budget_usd=1.0, client=client, archived=short_archive)
