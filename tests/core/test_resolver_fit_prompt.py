"""Tests for the prompt-optimize fit path + ``Resolver.describe()`` (PR-C).

Two seams land here on top of the PR-M ``method=`` dispatch:

- **``fit(method=Bootstrap()/MIPRO())``** compiles a ``DSPyMatcher``'s prompt
  from labeled pairs (the ``method.kind == "prompt"`` branch, now real). Every
  test runs at ``$0`` with DSPy's ``DummyLM`` on the deterministic
  ``BootstrapFewShot`` optimizer -- never a paid ``MIPROv2`` compile.
- **``Resolver.describe()``** -- the pre-fit "what trains vs what is frozen"
  digest, a pure string builder (no training, no backend import).

The concrete ``MIPRO``/``Bootstrap`` :class:`Method` objects are unit-tested
here too (kind, optimizer identity, budget-aware ``describe()``, compile-arg
mapping).
"""

from collections.abc import Iterator

import pytest
from dspy.utils.dummies import DummyLM

from langres.core.harvest import LabeledPair
from langres.core.matcher import Matcher
from langres.core.matchers.dspy_judge import DSPyMatcher
from langres.core.methods_api import Method
from langres.training.methods_prompt import GEPA, MIPRO, Bootstrap, PromptMethod
from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.reports import ScoreInspectionReport
from langres.core.resolver import Resolver

RECORDS = [
    {"id": "a", "name": "Acme Corp"},
    {"id": "b", "name": "Acme Corporation"},
    {"id": "c", "name": "Beta Inc"},
    {"id": "d", "name": "Beta Incorporated"},
]

# Two positives (a~b, c~d) + one negative (a!c): enough for BootstrapFewShot to
# select demos that reproduce the labeled match decision.
PAIRS = [
    LabeledPair(left_id="a", right_id="b", score=None, label=True, source="correction"),
    LabeledPair(left_id="c", right_id="d", score=None, label=True, source="correction"),
    LabeledPair(left_id="a", right_id="c", score=None, label=False, source="correction"),
]

_ANSWER = {"reasoning": "same", "match": "True", "match_probability": "0.9"}


def _dspy_resolver() -> tuple[Resolver, DSPyMatcher]:
    """A Resolver whose matcher is a ``DummyLM``-backed (zero-spend) DSPyMatcher."""
    matcher: DSPyMatcher[CompanySchema] = DSPyMatcher(
        lm=DummyLM([_ANSWER] * 40), entity_noun="company"
    )
    resolver = Resolver.from_schema(CompanySchema, matcher=matcher)
    return resolver, matcher


# --- Method objects: kind, optimizer identity, describe(), compile-arg map ---


def test_prompt_methods_share_kind_prompt() -> None:
    """All concrete prompt methods route through the ``kind == "prompt"`` branch."""
    assert Bootstrap.kind == "prompt"
    assert MIPRO.kind == "prompt"
    assert GEPA.kind == "prompt"
    assert issubclass(Bootstrap, PromptMethod) and issubclass(MIPRO, Method)
    assert issubclass(GEPA, PromptMethod)


def test_optimizer_is_classvar_not_a_field() -> None:
    """``optimizer`` is strategy-type identity, not serialized per-instance config."""
    assert Bootstrap.optimizer == "bootstrap"
    assert MIPRO.optimizer == "mipro"
    assert GEPA.optimizer == "gepa"
    assert "optimizer" not in MIPRO.model_fields
    assert "optimizer" not in GEPA.model_fields
    assert "kind" not in MIPRO.model_fields


def test_budget_usd_is_a_field_and_defaults_none() -> None:
    """``budget_usd`` IS serialized config (unlike kind/optimizer) and defaults off."""
    assert "budget_usd" in Bootstrap.model_fields
    assert Bootstrap().budget_usd is None
    assert MIPRO(budget_usd=5.0).budget_usd == 5.0


def test_describe_renders_optimizer_and_budget() -> None:
    """``describe()`` names the optimizer and appends the budget only when set."""
    assert Bootstrap().describe() == "prompt-optimize (BootstrapFewShot)"
    assert Bootstrap(budget_usd=5.0).describe() == "prompt-optimize (BootstrapFewShot), budget $5"
    assert MIPRO().describe() == "prompt-optimize (MIPROv2, auto=light)"
    assert MIPRO(auto="heavy", budget_usd=2.5).describe() == (
        "prompt-optimize (MIPROv2, auto=heavy), budget $2.5"
    )


def test_gepa_describe_renders_budget_knob_and_dollar_cap() -> None:
    """GEPA's describe() shows the auto preset, or the precise call cap when set."""
    assert GEPA().describe() == "prompt-optimize (GEPA reflective, auto=light)"
    assert GEPA(auto="heavy", budget_usd=2.5).describe() == (
        "prompt-optimize (GEPA reflective, auto=heavy), budget $2.5"
    )
    # max_metric_calls (when set) is the budget GEPA actually runs under, so it
    # supersedes the auto preset in the descriptor.
    assert GEPA(max_metric_calls=40).describe() == (
        "prompt-optimize (GEPA reflective, max_metric_calls=40)"
    )


def test_compile_kwargs_maps_auto_for_mipro_only() -> None:
    """MIPRO threads its ``auto`` preset onto compile; Bootstrap adds nothing."""
    assert Bootstrap().compile_kwargs() == {}
    assert MIPRO(auto="medium").compile_kwargs() == {"auto": "medium"}


def test_gepa_compile_kwargs_carries_budget_and_reflection_config() -> None:
    """GEPA threads its budget + reflection knobs onto ``DSPyMatcher.compile``."""
    assert GEPA().compile_kwargs() == {
        "auto": "light",
        "max_metric_calls": None,
        "reflection_model": None,
        "reflection_minibatch_size": 3,
    }
    assert GEPA(
        auto="medium",
        max_metric_calls=25,
        reflection_model="openrouter/openai/gpt-4o",
        reflection_minibatch_size=5,
    ).compile_kwargs() == {
        "auto": "medium",
        "max_metric_calls": 25,
        "reflection_model": "openrouter/openai/gpt-4o",
        "reflection_minibatch_size": 5,
    }


# --- fit(method=Bootstrap()): the zero-spend compile-under-fit happy path ------


def test_fit_prompt_compiles_and_sets_report() -> None:
    """``fit(pairs, method=Bootstrap())`` compiles the DSPy program and reports it."""
    resolver, matcher = _dspy_resolver()
    assert matcher.compiled is False

    out = resolver.fit(RECORDS, pairs=PAIRS, method=Bootstrap(budget_usd=5.0))

    assert out is resolver  # chains
    assert matcher.compiled is True
    report = resolver.fit_report_
    assert report is not None
    assert report.trained is True
    assert report.n_train == len(PAIRS)
    # The descriptor names the method, teacher model, and demos learned.
    assert "prompt-optimize (BootstrapFewShot)" in report.trainable
    assert f"teacher={matcher.model}" in report.trainable
    assert f"demos={matcher.n_demos}" in report.trainable
    # Zero-spend under DummyLM: a declared budget reports $0 observed (see #100).
    assert report.cost == 0.0


def test_fit_prompt_reports_gold_coverage_from_pairs() -> None:
    """The pairs id-join is reused, so blocking coverage lands in the report."""
    resolver, _ = _dspy_resolver()
    resolver.fit(RECORDS, pairs=PAIRS, method=Bootstrap())
    coverage = resolver.fit_report_.coverage
    assert coverage is not None
    assert coverage.gold_coverage == 1.0  # both positives survive AllPairs blocking


def test_fit_prompt_no_budget_leaves_cost_unset() -> None:
    """Without a budget there is no SpendMonitor, so cost stays ``None`` (unmeasured)."""
    resolver, _ = _dspy_resolver()
    resolver.fit(RECORDS, pairs=PAIRS, method=Bootstrap())
    assert resolver.fit_report_.cost is None


def test_fit_prompt_accepts_prealigned_labels() -> None:
    """Pre-aligned ``labels`` feed the trainset directly (no id-join, no coverage)."""
    resolver, matcher = _dspy_resolver()
    candidates = resolver.candidates(RECORDS)
    labels = [True] * len(candidates)

    resolver.fit(RECORDS, labels=labels, method=Bootstrap())

    assert matcher.compiled is True
    assert resolver.fit_report_.n_train == len(candidates)
    assert resolver.fit_report_.coverage is None


def test_fit_prompt_compiles_with_gepa_zero_spend() -> None:
    """``fit(pairs, method=GEPA())`` runs GEPA's reflective loop at $0 under DummyLM.

    GEPA needs many more LM calls than BootstrapFewShot (student rollouts +
    reflection), so this resolver is built with a generous ``DummyLM`` answer
    pool and a tight ``max_metric_calls`` budget -- still fully offline (the
    reflection LM defaults to the matcher's own DummyLM).
    """
    matcher: DSPyMatcher[CompanySchema] = DSPyMatcher(
        lm=DummyLM([_ANSWER] * 500), entity_noun="company"
    )
    resolver = Resolver.from_schema(CompanySchema, matcher=matcher)

    out = resolver.fit(
        RECORDS, pairs=PAIRS, method=GEPA(max_metric_calls=8, reflection_minibatch_size=2)
    )

    assert out is resolver
    assert matcher.compiled is True
    report = resolver.fit_report_
    assert report is not None and report.trained is True
    assert report.n_train == len(PAIRS)
    assert "prompt-optimize (GEPA reflective, max_metric_calls=8)" in report.trainable


# --- fit(method=...) error contracts -----------------------------------------


def test_prompt_method_on_non_dspy_matcher_raises() -> None:
    """A prompt method needs a compilable DSPyMatcher; a string matcher is rejected."""
    resolver = Resolver.from_schema(CompanySchema, matcher="string")
    with pytest.raises(ValueError, match=r"prompt-optimization needs a DSPyMatcher"):
        resolver.fit(RECORDS, pairs=PAIRS, method=MIPRO(budget_usd=5.0))


def test_refitting_compiled_matcher_raises_clear_error() -> None:
    """DSPy cannot recompile in place -- the second fit gets a clear langres error."""
    resolver, _ = _dspy_resolver()
    resolver.fit(RECORDS, pairs=PAIRS, method=Bootstrap())
    with pytest.raises(ValueError, match=r"already compiled"):
        resolver.fit(RECORDS, pairs=PAIRS, method=Bootstrap())


def test_prompt_fit_rejects_both_labels_and_pairs() -> None:
    """Supervision is one of labels/pairs, not both -- mirrors the module-hook path."""
    resolver, _ = _dspy_resolver()
    with pytest.raises(ValueError, match=r"either labels= .* or pairs="):
        resolver.fit(RECORDS, labels=[True], pairs=PAIRS, method=Bootstrap())


def test_prompt_fit_requires_supervision() -> None:
    """Prompt-optimization has nothing to tune from without gold labels."""
    resolver, _ = _dspy_resolver()
    with pytest.raises(ValueError, match=r"needs gold labels"):
        resolver.fit(RECORDS, method=Bootstrap())


# --- Resolver.describe(): the pre-fit trainable/frozen honesty device ---------


class _FitHookMatcher(Matcher[CompanySchema]):
    """A structural ``SupervisedFitMixin`` matcher (has ``fit(candidates, labels)``)."""

    def fit(self, candidates: Iterator[ERCandidate[CompanySchema]], labels: list[bool]) -> None:
        pass

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        yield from ()

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        raise NotImplementedError


def test_describe_tags_frozen_string_pipeline() -> None:
    """A non-learnable string pipeline: every role frozen, clusterer named by threshold."""
    resolver = Resolver.from_schema(CompanySchema, matcher="string", threshold=0.5)
    text = resolver.describe()
    lines = text.splitlines()
    assert len(lines) == 4  # blocker / matcher / calibrator / clusterer
    assert text.count("frozen") == 4
    assert "TRAINABLE" not in text
    assert any(line.startswith("matcher:") and "frozen" in line for line in lines)
    assert any(line.startswith("clusterer:") and "threshold=0.5" in line for line in lines)


def test_describe_tags_dspy_matcher_trainable_before_fit() -> None:
    """A compilable DSPyMatcher reads TRAINABLE even before fit -- the point of describe()."""
    resolver, _ = _dspy_resolver()
    matcher_line = next(
        line for line in resolver.describe().splitlines() if line.startswith("matcher:")
    )
    assert "DSPyMatcher" in matcher_line
    assert "TRAINABLE" in matcher_line


def test_describe_tags_supervised_fit_matcher_trainable() -> None:
    """A ``SupervisedFitMixin`` matcher is TRAINABLE (structural Protocol detection)."""
    resolver = Resolver.from_schema(CompanySchema, matcher=_FitHookMatcher())
    matcher_line = next(
        line for line in resolver.describe().splitlines() if line.startswith("matcher:")
    )
    assert "TRAINABLE" in matcher_line
