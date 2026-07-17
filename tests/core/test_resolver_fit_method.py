"""Tests for the ``method=`` object seam on ``Resolver.fit`` (PR-M).

``fit(..., method=<Method>)`` dispatches on ``method.kind`` to the matching fit
path *before* the module-hook isinstance chain. All three concrete kinds now
route to real handlers (``prompt``/PR-C, ``finetune``/PR-F, ``calibrate``/PR-D):
each rejects a malformed method for its kind with a distinct, handler-specific
error (a non-``DSPyMatcher`` module, a non-``QLoRA`` method, a calibrate method
missing ``.strategy``) that still surfaces ``describe()``. That distinct error
text is what proves the routing; the dispatch + param plumbing is real, which is
what these tests lock:

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
from langres.core.methods_api import Method, UnsupportedMethodKind
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
    """A malformed calibrate strategy: right kind, but no ``.strategy`` set."""

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


def test_finetune_kind_routes_to_finetune_handler() -> None:
    """A ``kind="finetune"`` method routes to the real ``_fit_finetune`` handler (PR-F).

    A non-``QLoRA`` finetune method reaches the handler and gets its QLoRA
    type-guard TypeError -- distinct from the ``prompt``/``calibrate``
    NotImplementedError branches, which is what proves the routing.
    """
    with pytest.raises(TypeError, match=r"method kind 'finetune' requires a QLoRA method"):
        _resolver().fit(RECORDS, method=_FinetuneMethod())


def test_prompt_kind_routes_to_pr_c_prompt_fit() -> None:
    """A ``kind="prompt"`` method routes to the real PR-C ``_fit_prompt`` branch.

    ``_fit_prompt`` requires a compilable DSPyMatcher; the no-hook module here is
    not one, so it raises that clear error -- which *proves routing* (a different
    branch than finetune/calibrate) without a paid DSPy compile.
    """
    with pytest.raises(ValueError, match=r"prompt-optimization needs a DSPyMatcher"):
        _resolver().fit(RECORDS, method=_PromptMethod())


def test_calibrate_kind_routes_to_calibrate_handler() -> None:
    """A ``kind="calibrate"`` method routes to the real PR-D ``_fit_calibrate`` branch.

    ``_fit_calibrate`` needs a ``CalibrateMethod`` exposing ``.strategy``; the fake
    here has none, so it raises that clear error -- which *proves routing* (a
    different branch than prompt/finetune) without needing gold pairs.
    """
    with pytest.raises(ValueError, match=r"needs a CalibrateMethod exposing .strategy"):
        _resolver().fit(RECORDS, method=_CalibrateMethod())


def test_unknown_kind_falls_through_to_no_impl_error() -> None:
    """A kind no concrete Method implements raises the distinct fall-through error."""
    with pytest.raises(NotImplementedError, match=r"method kind 'bogus' is not recognized"):
        _resolver().fit(RECORDS, method=_UnknownMethod())


def test_error_carries_the_describe_string() -> None:
    """The dispatch surfaces ``method.describe()`` at the call site (kind + cost).

    The finetune handler's QLoRA type-guard error still embeds ``describe()``, so
    the "what + cost" string reaches the caller exactly like the stub branches do.
    """
    with pytest.raises(TypeError, match=r"fine-tune \(QLoRA, ~GPU-seconds\)"):
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


# --- accepted_method_kinds: an architecture refuses topology-changing fits ---
#
# W4's named architectures claim "one architecture = one class = one identity".
# ``_fit_finetune`` repoints the matcher slot, so an architecture that let a
# ``finetune`` through would keep its class name while becoming a different
# pipeline. A subclass declares what it accepts; the base stays permissive.


class _PromptOnlyArchitecture(Resolver):
    """A stand-in for a W4 architecture that only accepts prompt-optimization."""

    accepted_method_kinds: ClassVar[frozenset[str] | None] = frozenset({"prompt"})


class _FrozenArchitecture(Resolver):
    """An architecture whose topology is fixed: it accepts no ``method=`` at all."""

    accepted_method_kinds: ClassVar[frozenset[str] | None] = frozenset()


def _prompt_only() -> _PromptOnlyArchitecture:
    return _PromptOnlyArchitecture(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        matcher=_NoHookModule(),
        clusterer=Clusterer(threshold=0.5),
    )


def test_base_resolver_declares_no_accepted_kinds() -> None:
    """The base Resolver makes no identity claim, so it constrains no kind."""
    assert Resolver.accepted_method_kinds is None


def test_base_resolver_still_reaches_the_finetune_handler() -> None:
    """No regression: the permissive base routes ``finetune`` exactly as before.

    The gate must not intercept the base Resolver -- reaching ``_fit_finetune``'s
    own QLoRA type-guard proves the method got all the way to its handler.
    """
    with pytest.raises(TypeError, match=r"method kind 'finetune' requires a QLoRA method"):
        _resolver().fit(RECORDS, method=_FinetuneMethod())


def test_architecture_rejects_kind_outside_its_declaration() -> None:
    """A restricted architecture refuses a topology-changing kind at the fit boundary."""
    with pytest.raises(UnsupportedMethodKind) as excinfo:
        _prompt_only().fit(RECORDS, method=_FinetuneMethod())

    message = str(excinfo.value)
    assert "_PromptOnlyArchitecture" in message  # names the architecture
    assert "'finetune'" in message  # names the offending kind
    assert "fine-tune (QLoRA, ~GPU-seconds)" in message  # carries describe()
    assert "it accepts: 'prompt'" in message  # names what IS accepted


def test_unsupported_method_kind_is_a_typeerror() -> None:
    """The typed error is catchable as TypeError, like its _fit_finetune sibling."""
    assert issubclass(UnsupportedMethodKind, TypeError)
    with pytest.raises(TypeError):
        _prompt_only().fit(RECORDS, method=_FinetuneMethod())


def test_architecture_still_accepts_its_declared_kind() -> None:
    """The gate blocks only undeclared kinds -- a declared one reaches its handler."""
    with pytest.raises(ValueError, match=r"prompt-optimization needs a DSPyMatcher"):
        _prompt_only().fit(RECORDS, method=_PromptMethod())


def test_gate_runs_before_the_unknown_kind_fallthrough() -> None:
    """A restricted architecture rejects an unknown kind as unaccepted, not unrecognized."""
    with pytest.raises(UnsupportedMethodKind, match=r"does not accept method kind 'bogus'"):
        _prompt_only().fit(RECORDS, method=_UnknownMethod())


def test_architecture_accepting_no_kinds_reports_that_readably() -> None:
    """An empty declaration means "no method= at all" and must not print an empty list."""
    frozen = _FrozenArchitecture(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        matcher=_NoHookModule(),
        clusterer=Clusterer(threshold=0.5),
    )

    with pytest.raises(UnsupportedMethodKind, match=r"it accepts: \(no method kinds\)"):
        frozen.fit(RECORDS, method=_PromptMethod())


def test_gate_does_not_touch_the_method_none_path() -> None:
    """A restricted architecture still no-ops on ``method=None`` (the default path)."""
    resolver = _prompt_only()

    assert resolver.fit(RECORDS) is resolver
