"""Zero-spend tests for DSPyMatcher (the M4 learnable scorer seam).

Every test runs at $0 with DSPy's ``DummyLM`` — no network, no key. They cover
the serializable-Matcher contract (forward shape, honest cost, parse-failure
guard, lazy LM, global-state isolation), compilation (``BootstrapFewShot``
populates demos), state round-trips (``save_state``/``load_state`` and a full
fresh-process ``Resolver`` round-trip for both compiled and uncompiled judges),
and import-safety (``import langres.core`` must not import ``dspy``).
"""

import logging
import subprocess
import sys
from pathlib import Path

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from langres.core.metrics import classify_pairs
from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.matchers.dspy_judge import (
    DSPyMatcher,
    _clamp01,
    _gepa_metric,
    _pair_metric,
    _salvage_usage,
)
from langres.core.registry import get_component
from langres.tracking.runs import RunStore


def _answers(n: int, *, match: str = "True", prob: str = "0.9") -> list[dict[str, str]]:
    """``n`` canned JSONAdapter answers carrying the three signature output fields."""
    return [{"reasoning": "same company", "match": match, "match_probability": prob}] * n


def _candidate(left_id: str = "l1", right_id: str = "r1") -> ERCandidate[CompanySchema]:
    return ERCandidate(
        left=CompanySchema(id=left_id, name="Acme"),
        right=CompanySchema(id=right_id, name="Acme Inc"),
        blocker_name="test",
    )


def _dummy_judge(answers: list[dict[str, str]] | None = None) -> DSPyMatcher[CompanySchema]:
    return DSPyMatcher(lm=DummyLM(answers or _answers(10)), entity_noun="company")


# ---------------------------------------------------------------------------
# Registration & config
# ---------------------------------------------------------------------------


def test_registered_under_dspy_judge_via_lazy_lookup() -> None:
    """``get_component('dspy_judge')`` lazily imports+registers the class."""
    assert get_component("dspy_judge") is DSPyMatcher
    assert DSPyMatcher.type_name == "dspy_judge"


def test_config_is_pure_and_excludes_lm_and_program() -> None:
    """``config`` carries only model/temperature/entity_noun — never the LM/program."""
    judge = DSPyMatcher(lm=DummyLM([]), model="openrouter/z-ai/glm-5.2", entity_noun="product")
    config = judge.config
    assert set(config) == {"model", "temperature", "entity_noun"}
    assert config == {
        "model": "openrouter/z-ai/glm-5.2",
        "temperature": 0.0,
        "entity_noun": "product",
    }
    import json

    json.dumps(config)  # must be JSON-serializable


def test_explicit_program_is_used_as_is() -> None:
    """An injected ``program`` is adopted verbatim (no fresh ChainOfThought built)."""
    program = dspy.ChainOfThought("left, right -> match")
    judge = DSPyMatcher(lm=DummyLM([]), program=program)
    assert judge._program is program


def test_from_config_builds_fresh_uncompiled_judge() -> None:
    """``from_config`` rebuilds an equivalent judge with a fresh, uncompiled program."""
    original = _dummy_judge()
    rebuilt = DSPyMatcher.from_config(original.config)
    assert rebuilt.config == original.config
    assert rebuilt._lm is None  # LM is not persisted — built lazily
    assert rebuilt._compiled is False


def test_entity_noun_woven_into_signature_instructions() -> None:
    """The domain noun is substituted into the program's signature instructions."""
    judge = DSPyMatcher(lm=DummyLM([]), entity_noun="restaurant")
    instructions = judge._program.predict.signature.instructions
    assert "restaurant" in instructions
    assert "{entity_noun}" not in instructions


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


def test_forward_yields_judgement_with_score_equal_to_probability() -> None:
    """forward emits a PairwiseJudgement whose score is the parsed match_probability."""
    judge = _dummy_judge(_answers(4, prob="0.83"))
    [judgement] = list(judge.forward(iter([_candidate("a", "b")])))
    assert isinstance(judgement, PairwiseJudgement)
    assert judgement.left_id == "a"
    assert judgement.right_id == "b"
    assert judgement.score == pytest.approx(0.83)
    assert judgement.score_type == "prob_llm"
    assert judgement.decision_step == "dspy_judgment"
    assert judgement.reasoning == "same company"


def test_forward_provenance_has_cost_and_token_keys() -> None:
    """Provenance carries honest cost + token counts (0 under DummyLM = $0)."""
    [judgement] = list(_dummy_judge().forward(iter([_candidate()])))
    assert set(judgement.provenance) == {
        "model",
        "cost_usd",
        "prompt_tokens",
        "completion_tokens",
        "usage",
    }
    assert judgement.provenance["cost_usd"] == 0.0
    assert judgement.provenance["prompt_tokens"] == 0
    assert judgement.provenance["completion_tokens"] == 0


def test_forward_provenance_carries_usage_vector() -> None:
    """The typed usage vector is captured (all zeros under DummyLM = no billed usage)."""
    [judgement] = list(_dummy_judge().forward(iter([_candidate()])))
    usage = judgement.provenance["usage"]
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0
    assert usage["reasoning_tokens"] == 0
    assert usage["model"] == judgement.provenance["model"]


def test_forward_parses_bool_and_float_outputs() -> None:
    """DSPy typed outputs give a real bool ``match`` and float probability."""
    judge = _dummy_judge(_answers(2, match="False", prob="0.12"))
    [judgement] = list(judge.forward(iter([_candidate()])))
    assert judgement.score == pytest.approx(0.12)


def test_forward_clamps_probability_into_unit_range() -> None:
    """An out-of-range probability is clamped to [0, 1] so PairwiseJudgement validates."""
    judge = _dummy_judge(_answers(2, prob="1.7"))
    [judgement] = list(judge.forward(iter([_candidate()])))
    assert judgement.score == 1.0


def test_forward_leaves_global_dspy_lm_untouched() -> None:
    """``dspy.context`` is per-call — the global ``dspy.settings.lm`` is never set."""
    before = dspy.settings.lm
    list(_dummy_judge().forward(iter([_candidate()])))
    assert dspy.settings.lm is before


def test_forward_warns_on_uncompiled_program(caplog: pytest.LogCaptureFixture) -> None:
    """Scoring with an uncompiled program warns so an untuned judge isn't silent."""
    with caplog.at_level(logging.WARNING):
        list(_dummy_judge().forward(iter([_candidate()])))
    assert any("UNCOMPILED" in r.message for r in caplog.records)


def test_forward_parse_failure_emits_abstention_judgement(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A DSPy parse error emits a Wave-1 abstention — is_abstain, excluded from predicted.

    Regression chain: DSPyMatcher first abstained at score=0.5 with NO
    ``provenance["parse_error"]`` flag (``classify_pairs`` called it a MATCH at
    any threshold <= 0.5), then at score=0.0 with the flag — better, but still
    not the Wave-1 contract: score=0.0 is a *confident non-match*, so
    ``is_abstain`` was ``False`` and the pair was graded a "no" rather than
    excluded. The contract-correct abstention nulls the verdict
    (``decision=None, score=None``): ``is_abstain`` is ``True``,
    ``predicted_match`` returns ``None``, and ``classify_pairs`` *excludes* the
    pair from the predicted set — a false negative on a gold pair (recall is not
    flattered), a true negative otherwise, never a match. The oracle below
    asserts those confusion-matrix counts move, not merely that the flag exists.
    """
    judge = DSPyMatcher(lm=DummyLM([{"unexpected": "shape"}]), entity_noun="company")
    with caplog.at_level(logging.WARNING):
        [judgement] = list(judge.forward(iter([_candidate("x", "y")])))
    # Wave-1 abstain shape: no decision, no score -> is_abstain (score=0.0 would
    # NOT satisfy is_abstain and would be graded a confident non-match).
    assert judgement.decision is None
    assert judgement.score is None
    assert judgement.is_abstain is True
    assert judgement.decision_step == "dspy_parse_error"
    assert judgement.reasoning is None
    assert "error" in judgement.provenance
    assert judgement.provenance["parse_error"] is True
    # The billed-but-unparseable call is flagged so cost tracking never silently
    # undercounts it (tokens salvaged from history are 0 under DummyLM => $0).
    assert judgement.provenance["cost_untracked"] is True
    assert judgement.provenance["cost_usd"] == 0.0
    assert any("parse failure" in r.message for r in caplog.records)

    # Oracle — the confusion matrix must move, not just carry a flag. The pair is
    # {x, y}; grade it once as gold and once as non-gold at a positive threshold.
    pair = frozenset({"x", "y"})
    when_gold = classify_pairs([judgement], gold_pairs={pair}, threshold=0.5)
    assert (when_gold.tp, when_gold.fp, when_gold.fn) == (0, 0, 1)  # excluded -> FN
    when_not_gold = classify_pairs([judgement], gold_pairs=set(), threshold=0.5)
    assert (when_not_gold.tp, when_not_gold.fp, when_not_gold.fn) == (0, 0, 0)  # excluded -> TN


# ---------------------------------------------------------------------------
# Lazy LM + cost seam
# ---------------------------------------------------------------------------


def test_get_lm_builds_dspy_lm_lazily(mocker) -> None:  # type: ignore[no-untyped-def]
    """With no injected LM, ``_get_lm`` builds a ``dspy.LM(model, cache=False)`` once."""
    sentinel = object()
    build = mocker.patch("dspy.LM", return_value=sentinel)
    judge: DSPyMatcher[CompanySchema] = DSPyMatcher(model="openrouter/z-ai/glm-5.2")
    assert judge._lm is None  # not built at construction
    assert judge._get_lm() is sentinel
    assert judge._get_lm() is sentinel  # cached
    build.assert_called_once_with("openrouter/z-ai/glm-5.2", cache=False, temperature=0.0)


def test_get_lm_forwards_temperature(mocker) -> None:  # type: ignore[no-untyped-def]
    """A non-default temperature reaches the lazily-constructed ``dspy.LM``."""
    build = mocker.patch("dspy.LM", return_value=object())
    judge: DSPyMatcher[CompanySchema] = DSPyMatcher(
        model="openrouter/z-ai/glm-5.2", temperature=0.7
    )
    judge._get_lm()
    build.assert_called_once_with("openrouter/z-ai/glm-5.2", cache=False, temperature=0.7)


def test_injected_lm_is_used_and_never_rebuilt(mocker) -> None:  # type: ignore[no-untyped-def]
    """An injected LM wins — ``dspy.LM`` is never constructed."""
    build = mocker.patch("dspy.LM")
    lm = DummyLM([])
    judge = DSPyMatcher(lm=lm)
    assert judge._get_lm() is lm
    build.assert_not_called()


def test_cost_usd_uses_pinned_price_seam() -> None:
    """The honest-cost seam multiplies tokens by the injectable per-1k price."""
    judge = _dummy_judge()
    assert judge._cost_usd(1000, 500) == 0.0  # default price 0.0 -> $0
    judge.price_per_1k_tokens = 2.0
    assert judge._cost_usd(1000, 500) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Small unit helpers
# ---------------------------------------------------------------------------


def test_clamp01_bounds() -> None:
    assert _clamp01(-0.5) == 0.0
    assert _clamp01(0.4) == 0.4
    assert _clamp01(2.0) == 1.0


class _StubLM:
    """Minimal LM stub carrying only a DSPy-shaped ``history`` for usage salvage."""

    def __init__(self, history: list[dict[str, object]]) -> None:
        self.history = history


def test_salvage_usage_reads_last_history_entry() -> None:
    """When the LM recorded a billed call, its token counts are salvaged."""
    lm = _StubLM([{"usage": {"prompt_tokens": 12, "completion_tokens": 3}}])
    assert _salvage_usage(lm) == (12, 3)


def test_salvage_usage_returns_zeros_without_history() -> None:
    """A stub LM with no history yields ``(0, 0)`` — a $0 no-op, never a crash."""
    assert _salvage_usage(object()) == (0, 0)
    assert _salvage_usage(_StubLM([])) == (0, 0)


def test_salvage_usage_handles_missing_usage_key() -> None:
    """A history entry without a ``usage`` key is treated as zero tokens."""
    assert _salvage_usage(_StubLM([{"model": "x"}])) == (0, 0)


def test_pair_metric_compares_match_bool() -> None:
    example = dspy.Example(match=True)
    assert _pair_metric(example, dspy.Prediction(match=True)) is True
    assert _pair_metric(example, dspy.Prediction(match=False)) is False


def test_inspect_scores_returns_report() -> None:
    """``inspect_scores`` delegates to the shared report helper."""
    judgements = list(_dummy_judge().forward(iter([_candidate()])))
    report = _dummy_judge().inspect_scores(judgements)
    assert report.total_judgements == 1


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def _trainset(n: int = 4) -> list[dspy.Example]:
    return [
        dspy.Example(left="Acme", right="Acme Inc", match=True).with_inputs("left", "right")
        for _ in range(n)
    ]


def _other_trainset(n: int = 4) -> list[dspy.Example]:
    """A trainset with DIFFERENT labeled content than :func:`_trainset`."""
    return [
        dspy.Example(left="Beta", right="Beta LLC", match=False).with_inputs("left", "right")
        for _ in range(n)
    ]


def test_compile_bootstrap_populates_demos_and_sets_flag() -> None:
    """``compile(optimizer='bootstrap')`` tunes the program (adds demos) in place."""
    judge = _dummy_judge(_answers(50))
    returned = judge.compile(_trainset(), optimizer="bootstrap")
    assert returned is judge  # chainable
    assert judge._compiled is True
    demos = sum(len(p.demos) for _, p in judge._program.named_predictors())
    assert demos > 0


def test_compiled_property_reflects_state_across_build_compile_and_reload(tmp_path: Path) -> None:
    """The public ``compiled`` flag mirrors ``_compiled`` through build -> compile -> reload."""
    judge = _dummy_judge(_answers(50))
    assert judge.compiled is False  # a fresh program is uncompiled
    judge.compile(_trainset(), optimizer="bootstrap")
    assert judge.compiled is True  # compile tunes it

    judge.save_state(tmp_path)
    fresh = DSPyMatcher.from_config(judge.config)
    assert fresh.compiled is False  # a fresh rebuild is uncompiled
    fresh.load_state(tmp_path)
    assert fresh.compiled is True  # the reloaded (compiled) program restores the flag


def test_compile_unknown_optimizer_raises() -> None:
    judge = _dummy_judge(_answers(10))
    with pytest.raises(ValueError, match="unknown optimizer"):
        judge.compile(_trainset(), optimizer="nope")


# ---------------------------------------------------------------------------
# Compilation — GEPA reflective optimizer (zero-spend under DummyLM)
# ---------------------------------------------------------------------------


def _mixed_trainset() -> list[dspy.Example]:
    """A trainset with both a positive and a negative (GEPA reflects over both)."""
    return [
        dspy.Example(left="Acme", right="Acme Inc", match=True).with_inputs("left", "right"),
        dspy.Example(left="Beta", right="Gamma", match=False).with_inputs("left", "right"),
        dspy.Example(left="Acme", right="Acme LLC", match=True).with_inputs("left", "right"),
        dspy.Example(left="Delta", right="Omega", match=False).with_inputs("left", "right"),
    ]


def test_gepa_metric_is_five_arg_and_wraps_pair_metric() -> None:
    """``dspy.GEPA`` requires a 5-arg metric; ``_gepa_metric`` adapts the decision.

    It must (a) accept ``(gold, pred, trace, pred_name, pred_trace)`` -- exactly
    what ``dspy.GEPA.__init__`` binds to validate the metric -- and (b) return the
    scalar ``1.0``/``0.0`` mirror of :func:`_pair_metric`.
    """
    import inspect

    inspect.signature(_gepa_metric).bind(None, None, None, None, None)  # 5-arg contract

    gold = dspy.Example(match=True)
    assert _gepa_metric(gold, dspy.Prediction(match=True)) == 1.0
    assert _gepa_metric(gold, dspy.Prediction(match=False)) == 0.0
    # pred_name / pred_trace are accepted and ignored (per-predictor feedback is
    # a future enhancement).
    assert _gepa_metric(gold, dspy.Prediction(match=True), None, "predict", None) == 1.0


def test_compile_gepa_sets_flag_zero_spend() -> None:
    """``compile(optimizer='gepa')`` runs the full reflective loop at $0 under DummyLM.

    Both the student and the reflection LM are the injected ``DummyLM``
    (``reflection_model=None`` reuses the matcher's own LM), so GEPA reflects and
    evolves without any network/paid call. A tight ``max_metric_calls`` keeps it
    fast and deterministic.
    """
    judge = _dummy_judge(_answers(500))
    returned = judge.compile(
        _mixed_trainset(),
        optimizer="gepa",
        max_metric_calls=8,
        reflection_minibatch_size=2,
    )
    assert returned is judge  # chainable
    assert judge.compiled is True
    # The reflective optimizer leaves the program tuned (an instruction it can
    # score/forward with) -- forward still works after a GEPA compile.
    judgements = list(judge.forward(iter([_candidate()])))
    assert len(judgements) == 1


def test_compile_gepa_builds_named_reflection_lm(mocker) -> None:  # type: ignore[no-untyped-def]
    """A ``reflection_model`` builds a dedicated ``dspy.LM`` passed as GEPA's reflection_lm.

    Covers the named-model branch without a paid call: ``dspy.GEPA`` is stubbed so
    only the wiring (reflection LM construction + kwargs) is asserted.
    """
    reflection_sentinel = object()

    def _fake_lm(model_id: str, *args: object, **kw: object) -> object:
        return reflection_sentinel if model_id == "openrouter/openai/gpt-4o" else object()

    mocker.patch("dspy.LM", side_effect=_fake_lm)
    fake_program = object()
    gepa_ctor = mocker.patch("dspy.GEPA")
    gepa_ctor.return_value.compile.return_value = fake_program

    judge = _dummy_judge(_answers(10))
    judge.compile(
        _mixed_trainset(),
        optimizer="gepa",
        reflection_model="openrouter/openai/gpt-4o",
        auto="medium",
    )

    _, ctor_kwargs = gepa_ctor.call_args
    assert ctor_kwargs["reflection_lm"] is reflection_sentinel
    assert ctor_kwargs["metric"] is _gepa_metric
    assert ctor_kwargs["auto"] == "medium"  # no max_metric_calls => auto preset
    assert "max_metric_calls" not in ctor_kwargs  # exactly one budget knob
    assert judge._program is fake_program
    assert judge.compiled is True


# ---------------------------------------------------------------------------
# Compile-run lineage (S6 tracking seam)
# ---------------------------------------------------------------------------


def test_fresh_judge_has_no_compile_run_id() -> None:
    """The lineage carrier starts unset — the read contract for a later capture_run."""
    assert _dummy_judge()._compile_run_id is None


def test_compile_records_run_and_stamps_compile_run_id(tmp_path: Path) -> None:
    """``compile(store=...)`` persists one completed compile run and stamps its id.

    Read contract for Stream C: ``judge._compile_run_id`` equals the recorded
    ``attempt_id``, which a later ``capture_run`` threads into ``parent_run_id``.
    """
    store = RunStore(tmp_path / "runs.jsonl")
    judge = _dummy_judge(_answers(50))
    judge.compile(_trainset(), optimizer="bootstrap", store=store)

    records = store.read()
    assert len(records) == 1  # the running + terminal lines collapse (last-wins)
    record = records[0]
    assert record.status == "completed"
    assert record.context.method == "dspy_compile"
    assert record.context.experiment == "dspy_compile"
    assert record.context.llm_model == judge.model
    assert record.context.resolver_config is not None
    assert record.context.resolver_config["optimizer"] == "bootstrap"
    # The stamped carrier is exactly the persisted run's PK.
    assert judge._compile_run_id == record.attempt_id


def test_compile_fingerprints_trainset_into_recipe_id(tmp_path: Path) -> None:
    """Different labeled trainsets get different recipe_ids; an identical one, the same.

    Regression: ``compile`` used a constant ``dataset_name`` and left
    ``dataset_fingerprint`` unset, so two compiles on DIFFERENT labels collapsed to
    the SAME ``recipe_id`` (a store-based replay guard could treat them as one run).
    The trainset now feeds ``dataset_fingerprint`` -> ``compute_recipe_id``.
    """
    store_a = RunStore(tmp_path / "a.jsonl")
    store_b = RunStore(tmp_path / "b.jsonl")
    store_c = RunStore(tmp_path / "c.jsonl")

    _dummy_judge(_answers(50)).compile(_trainset(), optimizer="bootstrap", store=store_a)
    _dummy_judge(_answers(50)).compile(_other_trainset(), optimizer="bootstrap", store=store_b)
    _dummy_judge(_answers(50)).compile(_trainset(), optimizer="bootstrap", store=store_c)

    [ra] = store_a.read()
    [rb] = store_b.read()
    [rc] = store_c.read()

    # A content fingerprint is now stamped (no longer left None).
    assert ra.context.dataset_fingerprint is not None
    # Different labeled trainsets -> different fingerprint -> different recipe_id.
    assert ra.context.dataset_fingerprint != rb.context.dataset_fingerprint
    assert ra.recipe_id != rb.recipe_id
    # An identical trainset -> identical fingerprint -> identical recipe_id (only
    # the timestamped attempt_id differs), so genuine replays still dedup.
    assert ra.context.dataset_fingerprint == rc.context.dataset_fingerprint
    assert ra.recipe_id == rc.recipe_id
    assert ra.attempt_id != rc.attempt_id


def test_compile_threads_parent_run_id_onto_the_run(tmp_path: Path) -> None:
    """A ``parent_run_id`` passed to compile is recorded on the compile run's context."""
    store = RunStore(tmp_path / "runs.jsonl")
    judge = _dummy_judge(_answers(50))
    judge.compile(_trainset(), store=store, parent_run_id="sweep-abc")
    [record] = store.read()
    assert record.context.parent_run_id == "sweep-abc"


def test_compile_stamps_run_id_even_without_a_store() -> None:
    """The carrier is stamped even when nothing is persisted (default ``store=None``)."""
    judge = _dummy_judge(_answers(50))
    judge.compile(_trainset(), optimizer="bootstrap")
    assert isinstance(judge._compile_run_id, str)
    assert judge._compiled is True
    # Default path is behavior-unchanged: demos were still bootstrapped.
    assert sum(len(p.demos) for _, p in judge._program.named_predictors()) > 0


def test_compile_tracking_params_do_not_leak_into_optimizer_kwargs(  # type: ignore[no-untyped-def]
    tmp_path: Path,
    mocker,
) -> None:
    """tracker/store/parent_run_id are bound params — never forwarded to the optimizer.

    A real ``**kwargs`` (``max_bootstrapped_demos``) DOES reach the optimizer's
    ``compile``; the tracking params do NOT (they are explicit, before ``**kwargs``).
    """
    fake_optimizer = mocker.MagicMock()
    fake_optimizer.compile.return_value = dspy.ChainOfThought("left, right -> match")
    mocker.patch("dspy.BootstrapFewShot", return_value=fake_optimizer)

    judge = _dummy_judge(_answers(10))
    judge.compile(
        _trainset(),
        store=RunStore(tmp_path / "runs.jsonl"),
        parent_run_id="p",
        max_bootstrapped_demos=2,
    )

    _, kwargs = fake_optimizer.compile.call_args
    assert "store" not in kwargs
    assert "tracker" not in kwargs
    assert "parent_run_id" not in kwargs
    assert kwargs["max_bootstrapped_demos"] == 2  # real optimizer kwargs still forwarded
    # Even with the optimizer mocked, the run seam still fired and stamped the carrier.
    assert isinstance(judge._compile_run_id, str)


# ---------------------------------------------------------------------------
# State round-trip
# ---------------------------------------------------------------------------


def test_save_state_load_state_restores_compiled_program(tmp_path: Path) -> None:
    """A compiled program persists to ``program.json`` and reloads with its demos."""
    judge = _dummy_judge(_answers(50))
    judge.compile(_trainset(), optimizer="bootstrap")
    trained_demos = sum(len(p.demos) for _, p in judge._program.named_predictors())

    judge.save_state(tmp_path)
    assert (tmp_path / "program.json").exists()

    fresh = DSPyMatcher.from_config(judge.config)
    assert fresh._compiled is False
    fresh.load_state(tmp_path)
    assert fresh._compiled is True  # a loaded program is a tuned program
    restored_demos = sum(len(p.demos) for _, p in fresh._program.named_predictors())
    assert restored_demos == trained_demos


def test_save_before_compile_reloads_as_uncompiled(tmp_path: Path) -> None:
    """A judge saved BEFORE compile must reload UNCOMPILED (flag reflects reality).

    Regression: ``load_state`` used to hard-set ``_compiled = True``, so a judge
    persisted before ``compile`` was wrongly marked tuned on reload — suppressing
    the "uncompiled judge" warning. The sidecar marker now restores the real flag.
    """
    judge = _dummy_judge(_answers(10))
    assert judge._compiled is False  # never compiled
    judge.save_state(tmp_path)
    assert (tmp_path / "compiled").read_text() == "false"

    fresh = DSPyMatcher.from_config(judge.config)
    fresh.load_state(tmp_path)
    assert fresh._compiled is False  # stays uncompiled — the marker said so


def test_save_after_compile_reloads_as_compiled(tmp_path: Path) -> None:
    """A judge saved AFTER compile reloads compiled (marker says ``true``)."""
    judge = _dummy_judge(_answers(50))
    judge.compile(_trainset(), optimizer="bootstrap")
    judge.save_state(tmp_path)
    assert (tmp_path / "compiled").read_text() == "true"

    fresh = DSPyMatcher.from_config(judge.config)
    fresh.load_state(tmp_path)
    assert fresh._compiled is True


def test_load_state_without_marker_infers_from_demos(tmp_path: Path) -> None:
    """An older artifact (no marker) infers compilation from bootstrapped demos."""
    judge = _dummy_judge(_answers(50))
    judge.compile(_trainset(), optimizer="bootstrap")
    judge.save_state(tmp_path)
    (tmp_path / "compiled").unlink()  # simulate a pre-marker artifact

    fresh = DSPyMatcher.from_config(judge.config)
    fresh.load_state(tmp_path)
    assert fresh._compiled is True  # inferred from the restored demos


def test_resolver_with_dspy_judge_saves_and_loads(tmp_path: Path) -> None:
    """A Resolver with a DSPyMatcher in the module slot round-trips in-process."""
    from langres.core import Clusterer, Resolver
    from langres.core.blockers import AllPairsBlocker

    judge = _dummy_judge(_answers(50))
    judge.compile(_trainset(), optimizer="bootstrap")
    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        matcher=judge,
        clusterer=Clusterer(threshold=0.7),
    )
    resolver.save(tmp_path)

    import json

    manifest = json.loads((tmp_path / "resolver.json").read_text())
    module_spec = next(c for c in manifest["components"] if c["slot"] == "module")
    assert module_spec["type_name"] == "dspy_judge"
    assert "lm" not in module_spec["config"]
    assert (tmp_path / "module" / "program.json").exists()

    reloaded = Resolver.load(tmp_path)
    assert isinstance(reloaded.module, DSPyMatcher)
    assert reloaded.module.config == judge.config
    assert reloaded.module._compiled is True


@pytest.mark.slow
@pytest.mark.parametrize("compiled", [True, False])
def test_resolver_load_dspy_judge_in_fresh_process(tmp_path: Path, compiled: bool) -> None:
    """A clean process can ``Resolver.load`` a dspy_judge artifact via ``langres.core`` alone.

    Regression for the lazy-registration path: ``@register('dspy_judge')`` only
    fires when ``langres.core.matchers.dspy_judge`` is imported, and ``langres.core``
    deliberately does NOT import it (that would import ``dspy``). ``get_component``
    imports it on demand, so a fresh process that only does ``from langres.core
    import Resolver`` still resolves the type. Covers both a compiled and an
    uncompiled judge.
    """
    from langres.core import Clusterer, Resolver
    from langres.core.blockers import AllPairsBlocker

    judge = _dummy_judge(_answers(50))
    if compiled:
        judge.compile(_trainset(), optimizer="bootstrap")
    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        matcher=judge,
        clusterer=Clusterer(threshold=0.7),
    )
    resolver.save(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from langres.core import Resolver; "
                f"r = Resolver.load(r'{tmp_path}'); "
                "assert type(r.module).__name__ == 'DSPyMatcher'; "
                "print('OK')"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "fresh-process Resolver.load failed for a dspy_judge artifact.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "UnknownComponentType" not in result.stderr


# ---------------------------------------------------------------------------
# Import-safety
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_import_langres_core_does_not_import_dspy() -> None:
    """``import langres.core`` must NOT pull in ``dspy`` (it opens a disk cache).

    Run in a fresh subprocess so a prior in-process ``dspy`` import can't mask the
    leak (this test module imports dspy itself).
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, langres.core; "
            "assert 'dspy' not in sys.modules, 'dspy leaked into import langres.core'; "
            "print('OK')",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"import-safety check failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
