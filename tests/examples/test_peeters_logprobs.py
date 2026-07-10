"""$0 tests for the Peeters harness `--logprobs` credence probe + v2 rows.

Every test runs at **$0** with an injected fake client (returning canned answers
WITH a logprobs block) — no API key, no network, no real model call. Covers:

* the ``results_path_for`` variant token (contamination firewall),
* ``_build_live_judge(confidence="logprob")`` staying byte-identical apart from
  the logprob request,
* the v2 ``_row_from_judgement`` columns (``correct`` always; credence keys only
  when the probe was on),
* the spend cap still firing with logprobs on,
* resume skipping already-committed pairs,
* ``--report-only`` still reading the old **v1** rows (the key regression), and
* the committed rows reproducing our published-run F1 (92.09 / 90.71) with no
  network at all.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from examples.research.peeters_llm_em_replication import (
    COMMITTED_RESULTS_DIR,
    LOGPROBS_VARIANT,
    PeetersResultStore,
    _build_live_judge,
    _live_report_from_rows,
    _metrics_from_rows,
    _row_from_judgement,
    report_live_from_store,
    results_path_for,
    run_live,
)
from langres.data.peeters import get_peeters_replication

_MODEL = "openrouter/openai/gpt-4o-mini-2024-07-18"


# --------------------------------------------------------------------------- #
# Fake client: canned answer + a logprobs block on every response.
# --------------------------------------------------------------------------- #


def _logprob_response(answer: str, *, cost: float, in_tok: int = 80) -> SimpleNamespace:
    """A response whose first token is ``answer`` with a dominant logprob + one rival."""
    other = "No" if answer.strip().lower().startswith("y") else "Yes"
    content = [
        SimpleNamespace(
            token=answer,
            logprob=math.log(0.9),
            bytes=None,
            top_logprobs=[
                SimpleNamespace(token=answer, logprob=math.log(0.9), bytes=None),
                SimpleNamespace(token=other, logprob=math.log(0.08), bytes=None),
            ],
        )
    ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=answer),
                logprobs=SimpleNamespace(content=content),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=in_tok,
            completion_tokens=1,  # single output token — logprobs add zero output cost
            cost=cost,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
        _hidden_params={},
        provider="fake",
        model="fake",
    )


class _FakeLogprobClient:
    """Returns a fixed answer + logprobs block; counts calls; records last kwargs."""

    def __init__(self, *, answer: str = "Yes", cost_per_call: float = 0.0) -> None:
        self._answer = answer
        self._cost = cost_per_call
        self.calls = 0
        self.last_kwargs: dict[str, Any] | None = None

    def completion(self, **kwargs: Any) -> SimpleNamespace:
        self.last_kwargs = kwargs
        self.calls += 1
        return _logprob_response(self._answer, cost=self._cost)


# --------------------------------------------------------------------------- #
# Contamination firewall: the variant token.
# --------------------------------------------------------------------------- #


def test_results_path_variant_is_a_distinct_file() -> None:
    plain = results_path_for("d", "abt-buy", "domain-complex-force", _MODEL)
    probe = results_path_for(
        "d", "abt-buy", "domain-complex-force", _MODEL, variant=LOGPROBS_VARIANT
    )
    assert plain != probe
    assert probe.name.endswith("__logprobs.jsonl")
    assert "__logprobs" not in plain.name  # the replication file is untouched


def test_results_path_variant_composes_with_subset() -> None:
    p = results_path_for(
        "d", "abt-buy", "domain-complex-force", _MODEL, limit=150, seed=3, variant=LOGPROBS_VARIANT
    )
    assert p.name.endswith("__limit150-seed3__logprobs.jsonl")


# --------------------------------------------------------------------------- #
# The probe judge = the replication judge + the logprob request only.
# --------------------------------------------------------------------------- #


def test_build_live_judge_confidence_default_off() -> None:
    spec = get_peeters_replication("abt-buy")
    assert _build_live_judge(spec, _MODEL).confidence == "none"


def test_build_live_judge_confidence_logprob() -> None:
    spec = get_peeters_replication("abt-buy")
    judge = _build_live_judge(spec, _MODEL, confidence="logprob")
    assert judge.confidence == "logprob"
    # Everything else is the replication judge (provider pin, temperature).
    assert judge.provider == {"order": ["OpenAI"], "allow_fallbacks": False}
    assert judge.temperature == 0.0


# --------------------------------------------------------------------------- #
# v2 row schema: `correct` always; credence keys only when the probe was on.
# --------------------------------------------------------------------------- #


def test_row_from_judgement_v2_with_credence() -> None:
    judgement = SimpleNamespace(
        left_id="a1",
        right_id="b1",
        score=1.0,
        reasoning="Yes",
        provenance={
            "cost_usd": 1e-5,
            "cost_is_real": True,
            "provider": "OpenAI",
            "usage": {"input_tokens": 70, "output_tokens": 1},
            "p_yes": 0.84,
            "confidence_leaked_mass": 0.02,
            "p_yes_is_bound": False,
        },
    )
    row = _row_from_judgement(
        judgement, model=_MODEL, dataset="abt-buy", prompt_design="domain-complex-force", gold=1
    )
    assert row["v"] == 2
    assert row["verdict"] == 1
    assert row["correct"] == 1  # verdict == gold
    assert row["p_yes"] == 0.84
    assert row["leaked_mass"] == 0.02
    assert row["p_yes_is_bound"] is False


def test_row_from_judgement_correct_flag_tracks_gold() -> None:
    judgement = SimpleNamespace(
        left_id="a1", right_id="b1", score=0.0, reasoning="No", provenance={"cost_usd": 0.0}
    )
    row = _row_from_judgement(
        judgement, model=_MODEL, dataset="abt-buy", prompt_design="dc", gold=1
    )
    assert row["verdict"] == 0 and row["correct"] == 0  # said No on a gold match => wrong
    # Probe off => NO credence keys written (not null), only `correct`.
    assert "p_yes" not in row and "leaked_mass" not in row and "p_yes_is_bound" not in row


# --------------------------------------------------------------------------- #
# run_live with the probe on: logprobs reach the wire, credence is persisted,
# and the spend cap still fires.
# --------------------------------------------------------------------------- #


def test_run_live_logprobs_persists_credence(tmp_path: Path) -> None:
    spec = get_peeters_replication("abt-buy")
    store = PeetersResultStore(tmp_path / "probe.jsonl")
    client = _FakeLogprobClient(answer="Yes", cost_per_call=0.0)
    run_live(
        spec,
        _MODEL,
        budget_usd=1.0,
        client=client,
        indices=[0, 1, 2],
        store=store,
        confidence="logprob",
    )
    assert client.last_kwargs is not None
    assert client.last_kwargs["logprobs"] is True and client.last_kwargs["top_logprobs"] == 20
    rows = store.rows()
    assert len(rows) == 3
    for r in rows:
        assert r["v"] == 2
        assert "p_yes" in r and "leaked_mass" in r and "correct" in r
        # p_yes = 0.9 / (0.9 + 0.08) for the "Yes"-dominant fake.
        assert r["p_yes"] == pytest.approx(0.9 / 0.98)


def test_spend_cap_fires_with_logprobs_on() -> None:
    """The hard cap stops the run even with the probe on (high per-call cost, tiny budget)."""
    spec = get_peeters_replication("abt-buy")
    client = _FakeLogprobClient(answer="Yes", cost_per_call=0.5)
    result = run_live(spec, _MODEL, budget_usd=1.0, client=client, confidence="logprob")
    assert result["budget_hit"] is True
    assert result["n_judged"] == 3  # 0.5*3 = 1.5 > 1.0; stops on the 3rd pair
    assert client.calls == 3  # exactly three paid calls made, then stop


# --------------------------------------------------------------------------- #
# Resume: a second pass over a full store makes ZERO calls.
# --------------------------------------------------------------------------- #


def test_resume_skips_already_committed_pairs(tmp_path: Path) -> None:
    spec = get_peeters_replication("abt-buy")
    store = PeetersResultStore(tmp_path / "resume.jsonl")

    first = _FakeLogprobClient(answer="Yes", cost_per_call=0.0)
    run_live(spec, _MODEL, budget_usd=1.0, client=first, indices=[0, 1, 2], store=store)
    assert first.calls == 3 and len(store.rows()) == 3

    # Second pass, same store, same subset: every pair is already committed.
    second = _FakeLogprobClient(answer="No", cost_per_call=99.0)
    report = run_live(spec, _MODEL, budget_usd=1.0, client=second, indices=[0, 1, 2], store=store)
    assert second.calls == 0  # zero API calls on resume
    assert len(store.rows()) == 3  # no new rows
    assert report["n_judged"] == 3  # report still computed from the whole store


# --------------------------------------------------------------------------- #
# The KEY regression: --report-only still reads the old v1 rows.
# --------------------------------------------------------------------------- #


def _v1_row(left: str, right: str, gold: int, verdict: int) -> dict[str, Any]:
    """A hand-written v1 row: has score/verdict, NO p_yes/correct (the committed shape)."""
    return {
        "v": 1,
        "model": _MODEL,
        "dataset": "abt-buy",
        "prompt_design": "domain-complex-force",
        "left_id": left,
        "right_id": right,
        "gold": gold,
        "response_text": "Yes" if verdict else "No",
        "verdict": verdict,
        "score": float(verdict),
        "cost_usd": 1e-5,
        "cost_is_real": True,
        "provider": "OpenAI",
        "usage": {"input_tokens": 70, "output_tokens": 1},
    }


def test_report_only_reads_v1_rows(tmp_path: Path) -> None:
    """A pre-v2 row (no p_yes, no `correct`) loads and yields the expected metrics."""
    spec = get_peeters_replication("abt-buy")
    rows = [
        _v1_row("a0", "b0", gold=1, verdict=1),  # tp
        _v1_row("a1", "b1", gold=1, verdict=0),  # fn
        _v1_row("a2", "b2", gold=0, verdict=0),  # tn
        _v1_row("a3", "b3", gold=0, verdict=1),  # fp
    ]
    store = PeetersResultStore(tmp_path / "v1.jsonl")
    for r in rows:
        store.append(r)

    report = report_live_from_store(store, spec=spec, model=_MODEL)
    assert report["n_judged"] == 4
    assert (report["tp"], report["fp"], report["fn"]) == (1, 1, 1)
    # F1 = 2*1 / (2*1 + 1 + 1) = 0.5.
    assert report["f1"] == pytest.approx(50.0)

    # And the raw metric helper agrees (backward read is byte-stable).
    m = _metrics_from_rows(rows)
    assert m.tp == 1 and m.fp == 1 and m.fn == 1


def test_v1_and_v2_rows_coexist_in_one_store(tmp_path: Path) -> None:
    """A store mixing a v1 row and a v2 (probe) row still reads cleanly."""
    v2 = _v1_row("a4", "b4", gold=1, verdict=1)
    v2.update({"v": 2, "correct": 1, "p_yes": 0.9, "leaked_mass": 0.02, "p_yes_is_bound": False})
    store = PeetersResultStore(tmp_path / "mixed.jsonl")
    store.append(_v1_row("a0", "b0", gold=1, verdict=1))
    store.append(v2)
    report = _live_report_from_rows(store.rows(), model=_MODEL, n_pairs=2)
    assert report["n_judged"] == 2 and report["tp"] == 2


# --------------------------------------------------------------------------- #
# The committed rows reproduce our published-run F1 — NO network, NO archive.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "slug,expected_f1",
    [
        ("openrouter_openai_gpt-4o-mini-2024-07-18", 92.09),
        ("openrouter_openai_gpt-4o-2024-08-06", 90.71),
    ],
)
def test_committed_rows_reproduce_our_f1(slug: str, expected_f1: float) -> None:
    """The committed v1 rows recompute to our published OURS-F1 (92.09 / 90.71) at $0.

    This is the same OURS number `--report-only --compare-archived` prints; the
    99.25% archive agreement additionally needs the (uncommitted, ~186 MB) answer
    archive, so it is verified out-of-band, not in CI.
    """
    path = COMMITTED_RESULTS_DIR / f"abt-buy__domain-complex-force__{slug}.jsonl"
    if not path.exists():  # pragma: no cover - committed in-repo, guard for odd checkouts
        pytest.skip(f"committed results not present: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1206
    assert rows[0]["v"] == 1  # committed rows are the pre-probe schema
    m = _metrics_from_rows(rows)
    assert m.f1 * 100 == pytest.approx(expected_f1, abs=0.01)
