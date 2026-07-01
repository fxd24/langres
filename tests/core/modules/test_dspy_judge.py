"""Zero-spend tests for DSPyJudge (the M4 learnable scorer seam).

Every test runs at $0 with DSPy's ``DummyLM`` — no network, no key. They cover
the serializable-Module contract (forward shape, honest cost, parse-failure
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

from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.modules.dspy_judge import (
    DSPyJudge,
    _clamp01,
    _pair_metric,
    _salvage_usage,
)
from langres.core.registry import get_component


def _answers(n: int, *, match: str = "True", prob: str = "0.9") -> list[dict[str, str]]:
    """``n`` canned JSONAdapter answers carrying the three signature output fields."""
    return [{"reasoning": "same company", "match": match, "match_probability": prob}] * n


def _candidate(left_id: str = "l1", right_id: str = "r1") -> ERCandidate[CompanySchema]:
    return ERCandidate(
        left=CompanySchema(id=left_id, name="Acme"),
        right=CompanySchema(id=right_id, name="Acme Inc"),
        blocker_name="test",
    )


def _dummy_judge(answers: list[dict[str, str]] | None = None) -> DSPyJudge[CompanySchema]:
    return DSPyJudge(lm=DummyLM(answers or _answers(10)), entity_noun="company")


# ---------------------------------------------------------------------------
# Registration & config
# ---------------------------------------------------------------------------


def test_registered_under_dspy_judge_via_lazy_lookup() -> None:
    """``get_component('dspy_judge')`` lazily imports+registers the class."""
    assert get_component("dspy_judge") is DSPyJudge
    assert DSPyJudge.type_name == "dspy_judge"


def test_config_is_pure_and_excludes_lm_and_program() -> None:
    """``config`` carries only model/temperature/entity_noun — never the LM/program."""
    judge = DSPyJudge(lm=DummyLM([]), model="openrouter/z-ai/glm-5.2", entity_noun="product")
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
    judge = DSPyJudge(lm=DummyLM([]), program=program)
    assert judge._program is program


def test_from_config_builds_fresh_uncompiled_judge() -> None:
    """``from_config`` rebuilds an equivalent judge with a fresh, uncompiled program."""
    original = _dummy_judge()
    rebuilt = DSPyJudge.from_config(original.config)
    assert rebuilt.config == original.config
    assert rebuilt._lm is None  # LM is not persisted — built lazily
    assert rebuilt._compiled is False


def test_entity_noun_woven_into_signature_instructions() -> None:
    """The domain noun is substituted into the program's signature instructions."""
    judge = DSPyJudge(lm=DummyLM([]), entity_noun="restaurant")
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
    }
    assert judgement.provenance["cost_usd"] == 0.0
    assert judgement.provenance["prompt_tokens"] == 0
    assert judgement.provenance["completion_tokens"] == 0


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


def test_forward_parse_failure_emits_low_confidence_judgement(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A DSPy parse error yields a 0.5 judgement tagged ``dspy_parse_error`` (never a skip)."""
    judge = DSPyJudge(lm=DummyLM([{"unexpected": "shape"}]), entity_noun="company")
    with caplog.at_level(logging.WARNING):
        [judgement] = list(judge.forward(iter([_candidate("x", "y")])))
    assert judgement.score == 0.5
    assert judgement.decision_step == "dspy_parse_error"
    assert judgement.reasoning is None
    assert "error" in judgement.provenance
    # The billed-but-unparseable call is flagged so cost tracking never silently
    # undercounts it (tokens salvaged from history are 0 under DummyLM => $0).
    assert judgement.provenance["cost_untracked"] is True
    assert judgement.provenance["cost_usd"] == 0.0
    assert any("parse failure" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Lazy LM + cost seam
# ---------------------------------------------------------------------------


def test_get_lm_builds_dspy_lm_lazily(mocker) -> None:  # type: ignore[no-untyped-def]
    """With no injected LM, ``_get_lm`` builds a ``dspy.LM(model, cache=False)`` once."""
    sentinel = object()
    build = mocker.patch("dspy.LM", return_value=sentinel)
    judge: DSPyJudge[CompanySchema] = DSPyJudge(model="openrouter/z-ai/glm-5.2")
    assert judge._lm is None  # not built at construction
    assert judge._get_lm() is sentinel
    assert judge._get_lm() is sentinel  # cached
    build.assert_called_once_with("openrouter/z-ai/glm-5.2", cache=False, temperature=0.0)


def test_get_lm_forwards_temperature(mocker) -> None:  # type: ignore[no-untyped-def]
    """A non-default temperature reaches the lazily-constructed ``dspy.LM``."""
    build = mocker.patch("dspy.LM", return_value=object())
    judge: DSPyJudge[CompanySchema] = DSPyJudge(model="openrouter/z-ai/glm-5.2", temperature=0.7)
    judge._get_lm()
    build.assert_called_once_with("openrouter/z-ai/glm-5.2", cache=False, temperature=0.7)


def test_injected_lm_is_used_and_never_rebuilt(mocker) -> None:  # type: ignore[no-untyped-def]
    """An injected LM wins — ``dspy.LM`` is never constructed."""
    build = mocker.patch("dspy.LM")
    lm = DummyLM([])
    judge = DSPyJudge(lm=lm)
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


def test_compile_bootstrap_populates_demos_and_sets_flag() -> None:
    """``compile(optimizer='bootstrap')`` tunes the program (adds demos) in place."""
    judge = _dummy_judge(_answers(50))
    returned = judge.compile(_trainset(), optimizer="bootstrap")
    assert returned is judge  # chainable
    assert judge._compiled is True
    demos = sum(len(p.demos) for _, p in judge._program.named_predictors())
    assert demos > 0


def test_compile_unknown_optimizer_raises() -> None:
    judge = _dummy_judge(_answers(10))
    with pytest.raises(ValueError, match="unknown optimizer"):
        judge.compile(_trainset(), optimizer="nope")


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

    fresh = DSPyJudge.from_config(judge.config)
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

    fresh = DSPyJudge.from_config(judge.config)
    fresh.load_state(tmp_path)
    assert fresh._compiled is False  # stays uncompiled — the marker said so


def test_save_after_compile_reloads_as_compiled(tmp_path: Path) -> None:
    """A judge saved AFTER compile reloads compiled (marker says ``true``)."""
    judge = _dummy_judge(_answers(50))
    judge.compile(_trainset(), optimizer="bootstrap")
    judge.save_state(tmp_path)
    assert (tmp_path / "compiled").read_text() == "true"

    fresh = DSPyJudge.from_config(judge.config)
    fresh.load_state(tmp_path)
    assert fresh._compiled is True


def test_load_state_without_marker_infers_from_demos(tmp_path: Path) -> None:
    """An older artifact (no marker) infers compilation from bootstrapped demos."""
    judge = _dummy_judge(_answers(50))
    judge.compile(_trainset(), optimizer="bootstrap")
    judge.save_state(tmp_path)
    (tmp_path / "compiled").unlink()  # simulate a pre-marker artifact

    fresh = DSPyJudge.from_config(judge.config)
    fresh.load_state(tmp_path)
    assert fresh._compiled is True  # inferred from the restored demos


def test_resolver_with_dspy_judge_saves_and_loads(tmp_path: Path) -> None:
    """A Resolver with a DSPyJudge in the module slot round-trips in-process."""
    from langres.core import AllPairsBlocker, Clusterer, Resolver

    judge = _dummy_judge(_answers(50))
    judge.compile(_trainset(), optimizer="bootstrap")
    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        module=judge,
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
    assert isinstance(reloaded.module, DSPyJudge)
    assert reloaded.module.config == judge.config
    assert reloaded.module._compiled is True


@pytest.mark.slow
@pytest.mark.parametrize("compiled", [True, False])
def test_resolver_load_dspy_judge_in_fresh_process(tmp_path: Path, compiled: bool) -> None:
    """A clean process can ``Resolver.load`` a dspy_judge artifact via ``langres.core`` alone.

    Regression for the lazy-registration path: ``@register('dspy_judge')`` only
    fires when ``langres.core.modules.dspy_judge`` is imported, and ``langres.core``
    deliberately does NOT import it (that would import ``dspy``). ``get_component``
    imports it on demand, so a fresh process that only does ``from langres.core
    import Resolver`` still resolves the type. Covers both a compiled and an
    uncompiled judge.
    """
    from langres.core import AllPairsBlocker, Clusterer, Resolver

    judge = _dummy_judge(_answers(50))
    if compiled:
        judge.compile(_trainset(), optimizer="bootstrap")
    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        module=judge,
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
                "assert type(r.module).__name__ == 'DSPyJudge'; "
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
