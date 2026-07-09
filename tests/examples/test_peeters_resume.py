"""$0 crash-safety + resume tests for the Peeters LLM-EM paid harness.

The paid run (``--mode live``) races two models over 1206 pairs each. An earlier
run was killed partway and lost ~$0.187 of already-billed calls because results
were only persisted at the very end. These tests pin the resumable, crash-safe
rewrite (mirroring ``examples/research/m3_race.py``):

* every judged pair is durably appended (flush + ``os.fsync``) to a per-(model,
  dataset, prompt-design) JSONL under a gitignored ``--results-dir``;
* a resumed run skips already-judged pairs (re-running a completed model costs
  ``$0`` and makes ZERO API calls);
* the hard spend cap accounts for spend already recorded, so the aggregate cap
  holds across resumes;
* the final report is computed purely from the persisted rows (identical whether
  the run completed in one pass or three), and ``--report-only`` recomputes it
  with zero API calls;
* a truncated JSONL (a kill mid-write) is recovered from — intact rows are kept
  and skipped, the interrupted pair is re-judged.

Every test runs at **$0** with an injected fake client; the client is never a
real litellm client and no network call is made.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from examples.research.peeters_llm_em_replication import (
    PeetersResultStore,
    report_compare_from_store,
    report_live_from_store,
    results_path_for,
    run_compare_archived,
    run_live,
    stratified_subset_indices,
)
from langres.data.peeters import (
    get_peeters_replication,
    load_peeters_sample,
    render_sample_prompts,
)

_MODEL = "openrouter/openai/gpt-4o-mini-2024-07-18"
_PROMPT_DESIGN = "domain-complex-force"


# --------------------------------------------------------------------------- #
# Fake clients: canned answers + a fixed per-call cost, counting the calls.
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


class _CountingFakeClient:
    """Stand-in for litellm that counts ``completion`` calls (0 == no spend).

    Answers are consumed in order; running out raises so a test that expects
    ZERO calls fails loudly rather than silently succeeding on an empty client.
    """

    def __init__(self, answers: list[str], *, cost_per_call: float) -> None:
        self._answers = answers
        self._i = 0
        self._cost = cost_per_call
        self.calls = 0
        self.last_kwargs: dict[str, Any] | None = None

    def completion(self, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        self.last_kwargs = kwargs
        content = self._answers[self._i]
        self._i += 1
        return _response(content, cost=self._cost)


class _PromptKeyedFakeClient:
    """Perfect Yes/No keyed to the rendered prompt, so verdicts are order-independent.

    Unlike ``_CountingFakeClient`` (answers by call order), this returns the right
    answer for whichever pair is judged — so a resumed run produces byte-identical
    verdicts to a single-pass run regardless of where it resumes.
    """

    def __init__(self, answer_by_prompt: dict[str, str], *, cost_per_call: float) -> None:
        self._answer_by_prompt = answer_by_prompt
        self._cost = cost_per_call
        self.calls = 0

    def completion(self, *, messages: Any, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        return _response(self._answer_by_prompt[messages[-1]["content"]], cost=self._cost)


def _perfect_answer_by_prompt(spec: Any) -> dict[str, str]:
    return {p.prompt: "Yes" if p.label == 1 else "No" for p in render_sample_prompts(spec)}


def _spec() -> Any:
    return get_peeters_replication("abt-buy")


def _subset_indices(limit: int, seed: int = 0) -> list[int]:
    labels = [label for _l, _r, label in load_peeters_sample(_spec())]
    return stratified_subset_indices(labels, limit, seed)


def _perfect_answers_for_indices(indices: list[int]) -> list[str]:
    """ "Yes"/"No" matching each subset pair's gold label, in subset order."""
    sample = load_peeters_sample(_spec())
    return ["Yes" if sample[i][2] == 1 else "No" for i in indices]


# --------------------------------------------------------------------------- #
# PeetersResultStore — round-trip, ledger, crash-tolerant reads
# --------------------------------------------------------------------------- #


def _row(left: str, right: str, *, gold: int, verdict: int, cost: float) -> dict[str, Any]:
    return {
        "v": 1,
        "model": _MODEL,
        "dataset": "abt-buy",
        "prompt_design": _PROMPT_DESIGN,
        "left_id": left,
        "right_id": right,
        "gold": gold,
        "response_text": "Yes" if verdict == 1 else "No",
        "verdict": verdict,
        "score": float(verdict),
        "cost_usd": cost,
        "cost_is_real": True,
        "provider": "fake-provider",
        "usage": {"input_tokens": 80, "output_tokens": 2},
    }


def test_store_append_and_rows_roundtrip(tmp_path: Path) -> None:
    store = PeetersResultStore(tmp_path / "r.jsonl")
    assert store.rows() == []  # nothing written yet
    r0 = _row("a0", "b0", gold=1, verdict=1, cost=0.001)
    r1 = _row("a1", "b1", gold=0, verdict=0, cost=0.002)
    store.append(r0)
    store.append(r1)
    assert store.rows() == [r0, r1]


def test_store_judged_pairs_and_spent(tmp_path: Path) -> None:
    store = PeetersResultStore(tmp_path / "r.jsonl")
    store.append(_row("a0", "b0", gold=1, verdict=1, cost=0.001))
    store.append(_row("a1", "b1", gold=0, verdict=0, cost=0.002))
    assert store.judged_pairs() == {frozenset({"a0", "b0"}), frozenset({"a1", "b1"})}
    assert store.spent() == pytest.approx(0.003)


def test_store_tolerates_truncated_trailing_line(tmp_path: Path) -> None:
    """A kill mid-write leaves a partial final line — it must be skipped, not crash."""
    store = PeetersResultStore(tmp_path / "r.jsonl")
    store.append(_row("a0", "b0", gold=1, verdict=1, cost=0.001))
    store.append(_row("a1", "b1", gold=0, verdict=0, cost=0.002))
    # Append a partial JSON fragment with no trailing newline (an interrupted write).
    with store.path.open("a", encoding="utf-8") as fh:
        fh.write('{"v": 1, "left_id": "a2", "right_id": "b2", "cost_usd": 0.5')
    rows = store.rows()
    assert len(rows) == 2  # the corrupt fragment is skipped
    assert store.judged_pairs() == {frozenset({"a0", "b0"}), frozenset({"a1", "b1"})}
    assert store.spent() == pytest.approx(0.003)  # the 0.5 fragment is NOT counted


def test_store_append_after_truncation_does_not_corrupt(tmp_path: Path) -> None:
    """Appending after a partial line repairs the newline so no valid row is lost."""
    store = PeetersResultStore(tmp_path / "r.jsonl")
    store.append(_row("a0", "b0", gold=1, verdict=1, cost=0.001))
    with store.path.open("a", encoding="utf-8") as fh:
        fh.write('{"v": 1, "left_id": "a1"')  # interrupted write, no newline
    store.append(_row("a2", "b2", gold=1, verdict=1, cost=0.004))
    rows = store.rows()
    # The intact first row and the freshly-appended row both survive; the fragment
    # is isolated on its own (skippable) line rather than fused to the new row.
    assert {r["left_id"] for r in rows} == {"a0", "a2"}
    assert store.spent() == pytest.approx(0.005)


def test_results_path_for_is_per_model_dataset_prompt(tmp_path: Path) -> None:
    p1 = results_path_for(tmp_path, "abt-buy", _PROMPT_DESIGN, _MODEL)
    p2 = results_path_for(
        tmp_path, "abt-buy", _PROMPT_DESIGN, "openrouter/openai/gpt-4o-2024-08-06"
    )
    assert p1 != p2  # different models -> different files
    assert p1.parent == tmp_path and p1.suffix == ".jsonl"
    # No path separators from the model id leak into the filename.
    assert "/" not in p1.name


# --------------------------------------------------------------------------- #
# run_live with a store — per-pair persistence
# --------------------------------------------------------------------------- #


def test_run_live_persists_every_judged_pair(tmp_path: Path) -> None:
    spec, idx = _spec(), _subset_indices(20)
    store = PeetersResultStore(tmp_path / "live.jsonl")
    client = _CountingFakeClient(_perfect_answers_for_indices(idx), cost_per_call=0.0001)

    report = run_live(spec, _MODEL, budget_usd=1.0, client=client, indices=idx, store=store)

    rows = store.rows()
    assert len(rows) == 20
    assert client.calls == 20
    assert report["n_judged"] == 20
    required = {
        "left_id",
        "right_id",
        "gold",
        "response_text",
        "verdict",
        "cost_usd",
        "cost_is_real",
        "model",
        "provider",
        "usage",
    }
    for r in rows:
        assert required <= set(r)
        assert r["model"] == _MODEL
    # Persisted cost matches the report's aggregate (the ledger reads the rows).
    assert store.spent() == pytest.approx(report["real_cost_usd"])


def test_run_live_full_resume_costs_zero_and_makes_zero_calls(tmp_path: Path) -> None:
    spec, idx = _spec(), _subset_indices(20)
    store = PeetersResultStore(tmp_path / "live.jsonl")
    first = run_live(
        spec,
        _MODEL,
        budget_usd=1.0,
        client=_CountingFakeClient(_perfect_answers_for_indices(idx), cost_per_call=0.0001),
        indices=idx,
        store=store,
    )
    # A completed model re-runs at $0 with zero API calls.
    empty_client = _CountingFakeClient([], cost_per_call=0.0001)
    second = run_live(spec, _MODEL, budget_usd=1.0, client=empty_client, indices=idx, store=store)

    assert empty_client.calls == 0
    assert len(store.rows()) == 20  # unchanged — nothing re-judged
    assert second["n_judged"] == 20
    assert second["real_cost_usd"] == pytest.approx(first["real_cost_usd"])
    assert second["f1"] == pytest.approx(first["f1"])


def test_run_live_resume_skips_committed_and_judges_only_the_rest(tmp_path: Path) -> None:
    spec, idx = _spec(), _subset_indices(20)
    store = PeetersResultStore(tmp_path / "live.jsonl")
    answers = _perfect_answers_for_indices(idx)

    # First pass stops early via a tiny budget: 5 calls at 0.0001 crosses 0.0004.
    partial = run_live(
        spec,
        _MODEL,
        budget_usd=0.00045,
        client=_CountingFakeClient(answers, cost_per_call=0.0001),
        indices=idx,
        store=store,
    )
    n_first = len(store.rows())
    assert partial["budget_hit"] is True
    assert 0 < n_first < 20

    # Resume with a full budget: only the remaining pairs are judged.
    resume_client = _CountingFakeClient(answers, cost_per_call=0.0001)
    final = run_live(spec, _MODEL, budget_usd=1.0, client=resume_client, indices=idx, store=store)

    assert resume_client.calls == 20 - n_first
    assert len(store.rows()) == 20
    assert final["n_judged"] == 20
    assert final["budget_hit"] is False


def test_run_live_budget_ledger_holds_across_resumes(tmp_path: Path) -> None:
    """The hard cap accounts for prior recorded spend, so a resume can't exceed it."""
    spec, idx = _spec(), _subset_indices(20)
    store = PeetersResultStore(tmp_path / "live.jsonl")
    answers = _perfect_answers_for_indices(idx)

    # Pass 1: cost 0.25/call, cap 0.6 -> fires after 3 calls (0.75 > 0.6).
    run_live(
        spec,
        _MODEL,
        budget_usd=0.6,
        client=_CountingFakeClient(answers, cost_per_call=0.25),
        indices=idx,
        store=store,
    )
    assert len(store.rows()) == 3
    assert store.spent() == pytest.approx(0.75)

    # Pass 2: cap 1.0. WITHOUT the ledger it would judge 4 more (its own $1.0);
    # WITH prior 0.75 seeded it fires after 2 (0.75 + 0.5 = 1.25 > 1.0).
    resume_client = _CountingFakeClient(answers, cost_per_call=0.25)
    final = run_live(spec, _MODEL, budget_usd=1.0, client=resume_client, indices=idx, store=store)

    assert resume_client.calls == 2
    assert final["budget_hit"] is True
    assert store.spent() == pytest.approx(1.25)
    assert len(store.rows()) == 5


def test_run_live_report_identical_whether_one_pass_or_three(tmp_path: Path) -> None:
    spec, idx = _spec(), _subset_indices(20)
    # Prompt-keyed client so a resumed pair still gets its correct answer.
    answer_by_prompt = _perfect_answer_by_prompt(spec)

    single = run_live(
        spec,
        _MODEL,
        budget_usd=1.0,
        client=_PromptKeyedFakeClient(answer_by_prompt, cost_per_call=0.0001),
        indices=idx,
        store=PeetersResultStore(tmp_path / "single.jsonl"),
    )

    multi_store = PeetersResultStore(tmp_path / "multi.jsonl")
    for cap in (0.00035, 0.0007, 1.0):  # three partial passes, then complete
        run_live(
            spec,
            _MODEL,
            budget_usd=cap,
            client=_PromptKeyedFakeClient(answer_by_prompt, cost_per_call=0.0001),
            indices=idx,
            store=multi_store,
        )
    multi = report_live_from_store(multi_store, spec=spec, model=_MODEL, limit=20, seed=0)

    for key in ("n_judged", "f1", "precision", "recall", "tp", "fp", "fn", "real_cost_usd"):
        assert multi[key] == pytest.approx(single[key]), key
    assert multi["usage"] == single["usage"]


# --------------------------------------------------------------------------- #
# --report-only: recompute from the JSONL with zero API calls
# --------------------------------------------------------------------------- #


def test_report_live_from_store_needs_no_client(tmp_path: Path) -> None:
    spec, idx = _spec(), _subset_indices(20)
    store = PeetersResultStore(tmp_path / "live.jsonl")
    run_live(
        spec,
        _MODEL,
        budget_usd=1.0,
        client=_CountingFakeClient(_perfect_answers_for_indices(idx), cost_per_call=0.0001),
        indices=idx,
        store=store,
    )
    # Zero client: the report is a pure read of the JSONL.
    report = report_live_from_store(store, spec=spec, model=_MODEL, limit=20, seed=0)
    assert report["n_judged"] == 20
    assert report["n_pairs"] == 20
    assert report["f1"] == pytest.approx(100.0)
    assert report["real_cost_usd"] == pytest.approx(20 * 0.0001)
    assert report["published_f1"] == 90.95


def test_report_live_from_store_reports_partial_as_incomplete(tmp_path: Path) -> None:
    spec, idx = _spec(), _subset_indices(20)
    store = PeetersResultStore(tmp_path / "live.jsonl")
    run_live(
        spec,
        _MODEL,
        budget_usd=0.00025,  # 3 calls at 0.0001 -> fires at 0.0003
        client=_CountingFakeClient(_perfect_answers_for_indices(idx), cost_per_call=0.0001),
        indices=idx,
        store=store,
    )
    report = report_live_from_store(store, spec=spec, model=_MODEL, limit=20, seed=0)
    assert report["n_judged"] < 20
    assert report["n_pairs"] == 20
    assert report["budget_hit"] is True  # incomplete == cap fired


# --------------------------------------------------------------------------- #
# run_compare_archived with a store + report-only agreement report
# --------------------------------------------------------------------------- #


def _perfect_archive(spec: Any) -> list[dict[str, str]]:
    return [
        {"prompt": p.prompt, "answer": "Yes" if p.label == 1 else "No"}
        for p in render_sample_prompts(spec)
    ]


def test_run_compare_archived_persists_and_report_from_store_matches(tmp_path: Path) -> None:
    spec = _spec()
    limit, seed = 20, 0
    idx = _subset_indices(limit, seed)
    archived = _perfect_archive(spec)

    # Perfect answers except flip subset position 0 (a gold positive) -> 1 disagreement.
    answers = _perfect_answers_for_indices(idx)
    assert answers[0] == "Yes"
    answers[0] = "No"
    client = _CountingFakeClient(answers, cost_per_call=0.0001)
    store = PeetersResultStore(tmp_path / "cmp.jsonl")

    run_report = run_compare_archived(
        spec,
        _MODEL,
        budget_usd=1.0,
        client=client,
        archived=archived,
        limit=limit,
        seed=seed,
        store=store,
    )
    assert len(store.rows()) == 20
    assert run_report["agreement_rate"] == pytest.approx(19 / 20)

    # Report-only: reproduce the agreement report from rows + archive, ZERO calls.
    reread = report_compare_from_store(
        store, spec=spec, model=_MODEL, archived=archived, limit=limit, seed=seed
    )
    assert reread["agreement_rate"] == pytest.approx(run_report["agreement_rate"])
    assert reread["confusion"] == run_report["confusion"]
    assert reread["ours"] == run_report["ours"]
    assert reread["theirs_subset"] == run_report["theirs_subset"]
    assert reread["n_judged"] == 20
    assert len(reread["disagreements"]) == 1
    assert reread["disagreements"][0]["gold_label"] == 1


def test_run_compare_archived_full_resume_makes_zero_calls(tmp_path: Path) -> None:
    spec = _spec()
    limit, seed = 20, 0
    idx = _subset_indices(limit, seed)
    archived = _perfect_archive(spec)
    answers = _perfect_answers_for_indices(idx)
    store = PeetersResultStore(tmp_path / "cmp.jsonl")

    first = run_compare_archived(
        spec,
        _MODEL,
        budget_usd=1.0,
        client=_CountingFakeClient(answers, cost_per_call=0.0001),
        archived=archived,
        limit=limit,
        seed=seed,
        store=store,
    )
    empty = _CountingFakeClient([], cost_per_call=0.0001)
    second = run_compare_archived(
        spec,
        _MODEL,
        budget_usd=1.0,
        client=empty,
        archived=archived,
        limit=limit,
        seed=seed,
        store=store,
    )
    assert empty.calls == 0
    assert len(store.rows()) == 20
    assert second["agreement_rate"] == pytest.approx(first["agreement_rate"])
    assert second["confusion"] == first["confusion"]


# --------------------------------------------------------------------------- #
# End-to-end crash recovery: truncate the JSONL mid-file, then resume
# --------------------------------------------------------------------------- #


def test_resume_recovers_from_a_truncated_results_file(tmp_path: Path) -> None:
    spec, idx = _spec(), _subset_indices(20)
    store = PeetersResultStore(tmp_path / "live.jsonl")
    answers = _perfect_answers_for_indices(idx)
    run_live(
        spec,
        _MODEL,
        budget_usd=1.0,
        client=_CountingFakeClient(answers, cost_per_call=0.0001),
        indices=idx,
        store=store,
    )
    assert len(store.rows()) == 20

    # Simulate a kill mid-write: truncate the file inside the 15th row.
    data = store.path.read_bytes()
    newline_offsets = [i for i, b in enumerate(data) if b == ord("\n")]
    cut = newline_offsets[13] + 1 + 15  # 14 whole rows + a partial 15th, no newline
    store.path.write_bytes(data[:cut])
    assert len(store.rows()) == 14  # the partial 15th row is dropped

    # Resume: only the 6 un-persisted pairs are re-judged; recovery is clean.
    resume_client = _CountingFakeClient(answers, cost_per_call=0.0001)
    final = run_live(spec, _MODEL, budget_usd=1.0, client=resume_client, indices=idx, store=store)
    assert resume_client.calls == 6
    assert final["n_judged"] == 20
    # Every subset pair is present exactly once in the recovered store.
    judged = store.judged_pairs()
    sample = load_peeters_sample(spec)
    assert judged == {frozenset({sample[i][0], sample[i][1]}) for i in idx}


def test_no_results_file_left_behind_when_store_omitted(tmp_path: Path) -> None:
    """The store is opt-in: the classic in-memory path writes nothing to disk."""
    spec, idx = _spec(), _subset_indices(10)
    run_live(
        spec,
        _MODEL,
        budget_usd=1.0,
        client=_CountingFakeClient(_perfect_answers_for_indices(idx), cost_per_call=0.0),
        indices=idx,
    )
    assert list(tmp_path.iterdir()) == []


def test_each_judged_pair_is_exactly_one_json_line(tmp_path: Path) -> None:
    """Each judged pair is exactly one JSON line (grep-able, one-object-per-line)."""
    spec, idx = _spec(), _subset_indices(20)
    store = PeetersResultStore(tmp_path / "live.jsonl")
    run_live(
        spec,
        _MODEL,
        budget_usd=1.0,
        client=_CountingFakeClient(_perfect_answers_for_indices(idx), cost_per_call=0.0001),
        indices=idx,
        store=store,
    )
    lines = [ln for ln in store.path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 20
    for ln in lines:
        json.loads(ln)  # each line independently valid
