"""Tests for the fit-hook Protocols: SupervisedFitMixin, UnsupervisedFitMixin.

These are runtime-checkable structural Protocols (W1.0 contracts-only; E6): a
Module opts in by implementing the method, not by subclassing. No abstract fit
method is added to the base Module ABC (that would break every existing
module), so these tests verify isinstance()-based structural detection instead
of ABC-enforced instantiation failures.
"""

from collections.abc import Iterator, Sequence

from pydantic import BaseModel

from langres.core.fit import SupervisedFitMixin, UnsupervisedFitMixin
from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.module import Module
from langres.core.reports import ScoreInspectionReport


class ProductSchema(BaseModel):
    """Second schema for schema-agnostic verification."""

    id: str
    title: str


class _NonLearnableModule(Module[CompanySchema]):
    """A module implementing neither fit hook (e.g. WeightedAverageJudge-like)."""

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        for candidate in candidates:
            yield PairwiseJudgement(
                left_id=candidate.left.id,
                right_id=candidate.right.id,
                score=0.5,
                score_type="heuristic",
                decision_step="test",
                provenance={},
            )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        raise NotImplementedError  # not exercised in these tests


class _SupervisedModule(Module[CompanySchema]):
    """A module implementing SupervisedFitMixin (e.g. a future RandomForestJudge)."""

    def __init__(self) -> None:
        self.fit_calls: list[tuple[list[ERCandidate[CompanySchema]], Sequence[bool]]] = []

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        yield from ()

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        raise NotImplementedError

    def fit(self, candidates: Iterator[ERCandidate[CompanySchema]], labels: Sequence[bool]) -> None:
        self.fit_calls.append((list(candidates), labels))


class _UnsupervisedModule(Module[CompanySchema]):
    """A module implementing UnsupervisedFitMixin (e.g. a future FellegiSunterJudge)."""

    def __init__(self) -> None:
        self.fit_unlabeled_calls: list[list[ERCandidate[CompanySchema]]] = []

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        yield from ()

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        raise NotImplementedError

    def fit_unlabeled(self, candidates: Iterator[ERCandidate[CompanySchema]]) -> None:
        self.fit_unlabeled_calls.append(list(candidates))


def _candidate() -> ERCandidate[CompanySchema]:
    return ERCandidate(
        left=CompanySchema(id="1", name="Acme"),
        right=CompanySchema(id="2", name="Acme Corp"),
        blocker_name="test_blocker",
    )


def test_non_learnable_module_satisfies_neither_mixin() -> None:
    """A module with only forward()/inspect_scores() is not fit-capable."""
    module = _NonLearnableModule()
    assert not isinstance(module, SupervisedFitMixin)
    assert not isinstance(module, UnsupervisedFitMixin)


def test_supervised_module_satisfies_supervised_mixin_only() -> None:
    """A module with fit(candidates, labels) matches SupervisedFitMixin."""
    module = _SupervisedModule()
    assert isinstance(module, SupervisedFitMixin)
    assert not isinstance(module, UnsupervisedFitMixin)


def test_unsupervised_module_satisfies_unsupervised_mixin_only() -> None:
    """A module with fit_unlabeled(candidates) matches UnsupervisedFitMixin."""
    module = _UnsupervisedModule()
    assert isinstance(module, UnsupervisedFitMixin)
    assert not isinstance(module, SupervisedFitMixin)


def test_supervised_mixin_fit_receives_candidates_and_labels() -> None:
    """The supervised hook is callable with (candidates, labels) as specified."""
    module = _SupervisedModule()
    labels = [True, False]

    module.fit(iter([_candidate(), _candidate()]), labels)

    assert len(module.fit_calls) == 1
    candidates, received_labels = module.fit_calls[0]
    assert len(candidates) == 2
    assert received_labels == labels


def test_unsupervised_mixin_fit_unlabeled_receives_candidates() -> None:
    """The unsupervised hook is callable with just candidates as specified."""
    module = _UnsupervisedModule()

    module.fit_unlabeled(iter([_candidate()]))

    assert len(module.fit_unlabeled_calls) == 1
    assert len(module.fit_unlabeled_calls[0]) == 1


def test_fit_mixins_are_schema_agnostic_with_product_schema() -> None:
    """The Protocols are generic over SchemaT — a ProductSchema module also matches."""

    class _ProductSupervisedModule(Module[ProductSchema]):
        def forward(
            self, candidates: Iterator[ERCandidate[ProductSchema]]
        ) -> Iterator[PairwiseJudgement]:
            yield from ()

        def inspect_scores(
            self, judgements: list[PairwiseJudgement], sample_size: int = 10
        ) -> ScoreInspectionReport:
            raise NotImplementedError

        def fit(
            self, candidates: Iterator[ERCandidate[ProductSchema]], labels: Sequence[bool]
        ) -> None:
            pass

    assert isinstance(_ProductSupervisedModule(), SupervisedFitMixin)
