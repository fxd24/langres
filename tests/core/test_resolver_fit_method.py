"""Tests for the ``method=`` object seam on ``Resolver.fit`` (PR-M).

``fit(..., method=<Method>)`` dispatches on ``method.kind`` to the matching fit
path *before* the module-hook isinstance chain. The concrete per-kind training
bodies land in later PRs (MIPRO/PR-C, QLoRA/PR-F, Platt/PR-D), so every wired
kind currently raises a clear ``NotImplementedError`` naming that PR -- but the
dispatch + param plumbing is real, which is what these tests lock:

- distinct ``kind`` values route to distinct branches (distinct error text);
- an unrecognized ``kind`` falls through to a "no Method implements it" error;
- ``method=None`` preserves today's exact module-hook behavior (no-op here);
- ``Method`` carries ``kind`` as a ClassVar (identity, not serialized config)
  and a ``describe()`` "what + cost" one-liner.
"""

from collections.abc import Iterator
from typing import ClassVar

import pytest

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.matcher import Matcher
from langres.core.methods_api import Method
from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.reports import ScoreInspectionReport
from langres.core.resolver import Resolver


class _NoHookModule(Matcher[CompanySchema]):
    """Implements neither fit hook -- the ``method=None`` no-op baseline."""

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        yield from ()

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        raise NotImplementedError


class _PromptMethod(Method):
    """Fake prompt-optimize strategy (kind wired to PR-C)."""

    kind: ClassVar[str] = "prompt"


class _FinetuneMethod(Method):
    """Fake fine-tune strategy (kind wired to PR-F) with a custom describe()."""

    kind: ClassVar[str] = "finetune"

    def describe(self) -> str:
        return "fine-tune (QLoRA, ~GPU-seconds)"


class _CalibrateMethod(Method):
    """Fake calibrate strategy (kind wired to PR-D)."""

    kind: ClassVar[str] = "calibrate"


class _UnknownMethod(Method):
    """A kind no concrete Method implements -- the fall-through branch."""

    kind: ClassVar[str] = "bogus"


RECORDS = [
    {"id": "a", "name": "Acme Corp"},
    {"id": "b", "name": "Acme Corporation"},
    {"id": "c", "name": "Beta Inc"},
]


def _resolver() -> Resolver:
    return Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        matcher=_NoHookModule(),
        clusterer=Clusterer(threshold=0.5),
    )


# --- dispatch: kind routes to its branch, all stubbed for now ---------------


def test_finetune_kind_routes_to_pr_f_stub() -> None:
    """A ``kind="finetune"`` method routes to the wired-but-stubbed PR-F branch."""
    with pytest.raises(NotImplementedError, match=r"method kind 'finetune'.*lands in PR-F"):
        _resolver().fit(RECORDS, method=_FinetuneMethod())


def test_prompt_kind_routes_to_pr_c_stub() -> None:
    """A ``kind="prompt"`` method routes to a *different* branch (PR-C) -- proves routing."""
    with pytest.raises(NotImplementedError, match=r"method kind 'prompt'.*lands in PR-C"):
        _resolver().fit(RECORDS, method=_PromptMethod())


def test_calibrate_kind_routes_to_pr_d_stub() -> None:
    """A ``kind="calibrate"`` method routes to the PR-D branch."""
    with pytest.raises(NotImplementedError, match=r"method kind 'calibrate'.*lands in PR-D"):
        _resolver().fit(RECORDS, method=_CalibrateMethod())


def test_unknown_kind_falls_through_to_no_impl_error() -> None:
    """A kind no concrete Method implements raises the distinct fall-through error."""
    with pytest.raises(NotImplementedError, match=r"method kind 'bogus' is not recognized"):
        _resolver().fit(RECORDS, method=_UnknownMethod())


def test_error_carries_the_describe_string() -> None:
    """The dispatch surfaces ``method.describe()`` at the call site (kind + cost)."""
    with pytest.raises(NotImplementedError, match=r"fine-tune \(QLoRA, ~GPU-seconds\)"):
        _resolver().fit(RECORDS, method=_FinetuneMethod())


# --- backward compatibility: method=None is byte-for-byte today's behavior --


def test_method_none_preserves_noop_behavior() -> None:
    """``method=None`` on a no-hook module stays the existing no-op that returns self."""
    resolver = _resolver()

    result = resolver.fit(RECORDS, method=None)

    assert result is resolver


def test_method_default_is_none() -> None:
    """Omitting ``method`` entirely is identical to the module-hook path (no-op here)."""
    resolver = _resolver()

    assert resolver.fit(RECORDS) is resolver


# --- Method contract: describe() + kind is a ClassVar, not a field ----------


def test_describe_defaults_to_kind() -> None:
    """The base ``describe()`` returns the bare ``kind`` when not overridden."""
    assert _PromptMethod().describe() == "prompt"


def test_describe_can_be_overridden() -> None:
    """A subclass overriding ``describe()`` returns its own 'what + cost' string."""
    assert _FinetuneMethod().describe() == "fine-tune (QLoRA, ~GPU-seconds)"


def test_kind_is_classvar_not_a_model_field() -> None:
    """``kind`` is strategy-type identity, so it must not become a serialized field."""
    assert "kind" not in _PromptMethod.model_fields
    assert _PromptMethod.kind == "prompt"
