"""Tests for Resolver.fit() delegating to the E6 fit-hook Protocols.

Resolver.fit() was a no-op stub (``return self``) for sklearn-style symmetry
with non-learnable pipelines. This branch wires it to the new
SupervisedFitMixin / UnsupervisedFitMixin Protocols (core/fit.py) WITHOUT
breaking that existing no-op contract for modules that implement neither hook
-- see tests/test_resolver_roundtrip.py::test_resolver_fit_is_noop_returns_self,
which must keep passing unmodified.

Reconciliation with E6 ("raise a clear error, not a silent no-op, when the
module lacks the hook"): the no-op is preserved ONLY for the zero-argument
call (``resolver.fit(data)``) on a module implementing neither mixin -- the
existing, tested, backward-compatible case. The failure mode E6 actually
guards against -- a caller believing fit() trained something when it silently
didn't -- is caught here instead: passing ``labels=`` to a module that can't
use them raises, and calling fit() on a SupervisedFitMixin module WITHOUT
labels raises (silent no-op on a genuinely trainable module is exactly the
footgun E6 names).
"""

from collections.abc import Iterator, Sequence

import pytest
from pydantic import BaseModel

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.module import Module
from langres.core.reports import ScoreInspectionReport
from langres.core.resolver import Resolver


class _NoHookModule(Module[CompanySchema]):
    """Implements neither fit hook (mirrors WeightedAverageJudge/heuristic judges)."""

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        yield from ()

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        raise NotImplementedError


class _SupervisedModule(Module[CompanySchema]):
    """Implements SupervisedFitMixin."""

    def __init__(self) -> None:
        self.fit_calls: list[tuple[int, Sequence[bool]]] = []

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        yield from ()

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        raise NotImplementedError

    def fit(
        self, candidates: Iterator[ERCandidate[CompanySchema]], labels: Sequence[bool]
    ) -> None:
        self.fit_calls.append((len(list(candidates)), labels))


class _UnsupervisedModule(Module[CompanySchema]):
    """Implements UnsupervisedFitMixin."""

    def __init__(self) -> None:
        self.fit_unlabeled_calls: list[int] = []

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        yield from ()

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        raise NotImplementedError

    def fit_unlabeled(self, candidates: Iterator[ERCandidate[CompanySchema]]) -> None:
        self.fit_unlabeled_calls.append(len(list(candidates)))


class ProductSchema(BaseModel):
    """Second schema for schema-agnostic verification."""

    id: str
    title: str


RECORDS = [
    {"id": "a", "name": "Acme Corp"},
    {"id": "b", "name": "Acme Corporation"},
    {"id": "c", "name": "Beta Inc"},
]


def _resolver(module: Module[CompanySchema]) -> Resolver:
    return Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        module=module,
        clusterer=Clusterer(threshold=0.5),
    )


def test_fit_is_noop_for_module_without_hooks_and_no_labels() -> None:
    """Backward-compat: a module with neither hook + no labels -> no-op, returns self."""
    resolver = _resolver(_NoHookModule())

    result = resolver.fit(RECORDS)

    assert result is resolver


def test_fit_delegates_to_unsupervised_hook() -> None:
    """A module implementing UnsupervisedFitMixin gets fit_unlabeled() called."""
    module = _UnsupervisedModule()
    resolver = _resolver(module)

    result = resolver.fit(RECORDS)

    assert result is resolver
    assert module.fit_unlabeled_calls == [3]  # AllPairs over 3 records -> 3 pairs


def test_fit_delegates_to_supervised_hook_with_labels() -> None:
    """A module implementing SupervisedFitMixin gets fit(candidates, labels) called."""
    module = _SupervisedModule()
    resolver = _resolver(module)
    labels = [True, False, True]

    result = resolver.fit(RECORDS, labels=labels)

    assert result is resolver
    assert module.fit_calls == [(3, labels)]


def test_fit_raises_when_supervised_module_gets_no_labels() -> None:
    """A genuinely trainable module called without labels RAISES (not a silent no-op)."""
    resolver = _resolver(_SupervisedModule())

    with pytest.raises(ValueError, match="requires labeled data"):
        resolver.fit(RECORDS)


def test_fit_raises_when_labels_given_to_a_module_without_supervised_hook() -> None:
    """Passing labels to a module that can't use them raises (not silently ignored)."""
    resolver = _resolver(_NoHookModule())

    with pytest.raises(ValueError, match="does not support fit"):
        resolver.fit(RECORDS, labels=[True, False])


def test_fit_raises_when_labels_given_to_unsupervised_module() -> None:
    """An UnsupervisedFitMixin module doesn't take labels either -- also raises."""
    resolver = _resolver(_UnsupervisedModule())

    with pytest.raises(ValueError, match="does not support fit"):
        resolver.fit(RECORDS, labels=[True, False])


def test_fit_is_schema_agnostic_with_product_schema() -> None:
    """The fit() delegation logic works identically for a ProductSchema pipeline."""

    class _ProductUnsupervisedModule(Module[ProductSchema]):
        def __init__(self) -> None:
            self.called = False

        def forward(
            self, candidates: Iterator[ERCandidate[ProductSchema]]
        ) -> Iterator[PairwiseJudgement]:
            yield from ()

        def inspect_scores(
            self, judgements: list[PairwiseJudgement], sample_size: int = 10
        ) -> ScoreInspectionReport:
            raise NotImplementedError

        def fit_unlabeled(self, candidates: Iterator[ERCandidate[ProductSchema]]) -> None:
            self.called = True
            list(candidates)

    module = _ProductUnsupervisedModule()
    resolver = Resolver(
        blocker=AllPairsBlocker(schema=ProductSchema),
        comparator=None,
        module=module,
        clusterer=Clusterer(threshold=0.5),
    )
    products = [{"id": "p1", "title": "iPhone"}, {"id": "p2", "title": "iPhone Pro"}]

    result = resolver.fit(products)

    assert result is resolver
    assert module.called is True
