"""Tests for CascadeJudge — the two-tier student/escalation judge (T3).

Covers: ctor validation (band + GroupwiseModule rejection), inclusive band
edges, per-pair escalation laziness, escalation-wins fields, provenance merge
(cost_usd + model MUST survive on escalated pairs), the exactly-one-judgement
contract guard, the one-time score_type contract warning, mixed
prob_rf/prob_llm streams through one Clusterer threshold, the spend-cap +
logging composition (an escalation-side BudgetExceeded must not drop paid
judgements from the log), registry config round-trip, and SerializableState
sidecar layout — including a full Resolver save/load round trip with a fitted
RandomForestJudge student.
"""

import json
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from langres.clients.openrouter import BudgetExceeded
from langres.core.clusterer import Clusterer
from langres.core.feature import FeatureSpec
from langres.core.groups import ERCandidateGroup
from langres.core.judgement_log import JudgementLog, LoggingModule
from langres.core.judges.weighted_average import WeightedAverageJudge
from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement
from langres.core.module import GroupwiseModule, Module
from langres.core.modules.cascade_judge import (
    CASCADE_ESCALATED_STEP,
    CASCADE_STUDENT_STEP,
    CascadeJudge,
)
from langres.core.presets import _SpendCappedModule
from langres.core.registry import get_component
from langres.core.reports import ScoreInspectionReport, _inspect_scores_impl

# ---------------------------------------------------------------------------
# Helpers: candidates + stub judges
# ---------------------------------------------------------------------------


def _pair(left_id: str, right_id: str) -> ERCandidate[CompanySchema]:
    return ERCandidate(
        left=CompanySchema(id=left_id, name=f"Company {left_id}"),
        right=CompanySchema(id=right_id, name=f"Company {right_id}"),
        blocker_name="test",
    )


class ScriptedJudge(Module[CompanySchema]):
    """Stub pairwise judge: one judgement per pair with a scripted score.

    Records every pair it sees in ``seen`` (the escalation-laziness spy).
    """

    def __init__(
        self,
        scores: dict[tuple[str, str], float],
        *,
        score_type: str = "prob_rf",
        decision_step: str = "scripted",
        reasoning: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        self.scores = scores
        self.score_type = score_type
        self.decision_step = decision_step
        self.reasoning = reasoning
        self.provenance = provenance or {}
        self.seen: list[tuple[str, str]] = []

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        for candidate in candidates:
            key = (candidate.left.id, candidate.right.id)
            self.seen.append(key)
            yield PairwiseJudgement(
                left_id=candidate.left.id,
                right_id=candidate.right.id,
                score=self.scores[key],
                score_type=self.score_type,  # type: ignore[arg-type]
                decision_step=self.decision_step,
                reasoning=self.reasoning,
                provenance=dict(self.provenance),
            )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        return _inspect_scores_impl(judgements, sample_size)


class ZeroJudge(ScriptedJudge):
    """Contract violator: yields NO judgement for a candidate."""

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        for _ in candidates:
            pass
        yield from ()


class DoubleJudge(ScriptedJudge):
    """Contract violator: yields TWO judgements for a candidate."""

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        for candidate in candidates:
            for _ in range(2):
                yield PairwiseJudgement(
                    left_id=candidate.left.id,
                    right_id=candidate.right.id,
                    score=0.5,
                    score_type="prob_llm",
                    decision_step="double",
                    provenance={},
                )


class RaisingJudge(ScriptedJudge):
    """Stub escalation judge: raises a plain (non-BudgetExceeded) exception.

    Represents a network/rate-limit/API failure -- unlike BudgetExceeded, it
    carries no ``partial_judgements`` field, so it must propagate unchanged.
    """

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        raise RuntimeError("escalation backend unavailable")


class GroupwiseStub(GroupwiseModule[CompanySchema]):
    """Minimal GroupwiseModule — must be rejected by the CascadeJudge ctor."""

    def forward_groups(
        self, groups: Iterator[ERCandidateGroup[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        yield from ()

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        return _inspect_scores_impl(judgements, sample_size)


class StatefulStubJudge(ScriptedJudge):
    """ScriptedJudge that also implements SerializableState (a ``state.txt`` file).

    An empty ``value`` writes nothing — mirrors an unfit RandomForestJudge's
    "nothing to save" behavior, exercising the empty-sidecar branches.
    """

    def __init__(self, *args: Any, value: str = "", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.value = value

    def save_state(self, state_dir: Path) -> None:
        if self.value:
            (state_dir / "state.txt").write_text(self.value)

    def load_state(self, state_dir: Path) -> None:
        self.value = (state_dir / "state.txt").read_text()


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestCtorValidation:
    @pytest.mark.parametrize(
        "band",
        [(0.5, 0.5), (0.7, 0.3), (-0.1, 0.5), (0.5, 1.1)],
        ids=["equal", "inverted", "low_below_zero", "high_above_one"],
    )
    def test_invalid_band_raises(self, band: tuple[float, float]) -> None:
        with pytest.raises(ValueError, match="band"):
            CascadeJudge(student=ScriptedJudge({}), escalation=ScriptedJudge({}), band=band)

    def test_valid_band_is_stored_as_tuple(self) -> None:
        judge = CascadeJudge(
            student=ScriptedJudge({}), escalation=ScriptedJudge({}), band=(0.0, 1.0)
        )
        assert judge.band == (0.0, 1.0)

    @pytest.mark.parametrize("slot", ["student", "escalation"])
    def test_groupwise_child_rejected_with_actionable_error(self, slot: str) -> None:
        children: dict[str, Module[CompanySchema]] = {
            "student": ScriptedJudge({}),
            "escalation": ScriptedJudge({}),
        }
        children[slot] = GroupwiseStub()
        with pytest.raises(ValueError, match=f"GroupwiseModule as its {slot}"):
            CascadeJudge(
                student=children["student"], escalation=children["escalation"], band=(0.3, 0.7)
            )


# ---------------------------------------------------------------------------
# Band routing: inclusive edges, laziness, escalation-wins
# ---------------------------------------------------------------------------


class TestBandRouting:
    def test_band_edges_are_inclusive(self) -> None:
        scores = {("a", "b"): 0.3, ("c", "d"): 0.7, ("e", "f"): 0.29, ("g", "h"): 0.71}
        student = ScriptedJudge(scores)
        escalation = ScriptedJudge(
            {("a", "b"): 0.9, ("c", "d"): 0.1}, score_type="prob_llm", decision_step="frontier"
        )
        cascade = CascadeJudge(student=student, escalation=escalation, band=(0.3, 0.7))

        pairs = [_pair(*key) for key in scores]
        steps = [j.decision_step for j in cascade.forward(iter(pairs))]

        assert steps == [
            CASCADE_ESCALATED_STEP,  # score == low: escalates
            CASCADE_ESCALATED_STEP,  # score == high: escalates
            CASCADE_STUDENT_STEP,  # just below low: student
            CASCADE_STUDENT_STEP,  # just above high: student
        ]

    def test_escalation_is_lazy_and_per_pair(self) -> None:
        student = ScriptedJudge({("a", "b"): 0.9, ("c", "d"): 0.5, ("e", "f"): 0.1})
        escalation = ScriptedJudge({("c", "d"): 0.8}, score_type="prob_llm")
        cascade = CascadeJudge(student=student, escalation=escalation, band=(0.3, 0.7))

        list(cascade.forward(iter([_pair("a", "b"), _pair("c", "d"), _pair("e", "f")])))

        assert student.seen == [("a", "b"), ("c", "d"), ("e", "f")]
        assert escalation.seen == [("c", "d")]  # ONLY the band pair

    def test_escalation_judgement_wins_fields(self) -> None:
        student = ScriptedJudge({("a", "b"): 0.5}, decision_step="stub_student")
        escalation = ScriptedJudge(
            {("a", "b"): 0.92},
            score_type="prob_llm",
            decision_step="frontier",
            reasoning="same company, different branding",
        )
        cascade = CascadeJudge(student=student, escalation=escalation, band=(0.3, 0.7))

        (judgement,) = list(cascade.forward(iter([_pair("a", "b")])))

        assert judgement.score == 0.92
        assert judgement.score_type == "prob_llm"
        assert judgement.reasoning == "same company, different branding"
        assert judgement.decision_step == CASCADE_ESCALATED_STEP
        assert judgement.left_id == "a" and judgement.right_id == "b"

    def test_student_judgement_passes_through(self) -> None:
        student = ScriptedJudge({("a", "b"): 0.95}, decision_step="stub_student")
        escalation = ScriptedJudge({}, score_type="prob_llm")
        cascade = CascadeJudge(student=student, escalation=escalation, band=(0.3, 0.7))

        (judgement,) = list(cascade.forward(iter([_pair("a", "b")])))

        assert judgement.score == 0.95
        assert judgement.score_type == "prob_rf"
        assert judgement.decision_step == CASCADE_STUDENT_STEP
        assert judgement.left_id == "a" and judgement.right_id == "b"


# ---------------------------------------------------------------------------
# Provenance merge (cost_usd + model MUST survive)
# ---------------------------------------------------------------------------


class TestProvenanceMerge:
    def test_escalated_provenance_merges_into_childs_dict(self) -> None:
        student = ScriptedJudge({("a", "b"): 0.5}, decision_step="stub_student")
        escalation = ScriptedJudge(
            {("a", "b"): 0.9},
            score_type="prob_llm",
            decision_step="frontier",
            provenance={"cost_usd": 0.002, "model": "frontier-x", "tokens": 42},
        )
        cascade = CascadeJudge(student=student, escalation=escalation, band=(0.3, 0.7))

        (judgement,) = list(cascade.forward(iter([_pair("a", "b")])))

        # The child's provenance survives (cost_usd feeds _SpendCappedModule,
        # model feeds the JudgementLog model column) ...
        assert judgement.provenance["cost_usd"] == 0.002
        assert judgement.provenance["model"] == "frontier-x"
        assert judgement.provenance["tokens"] == 42
        # ... and the cascade keys merge in on top.
        assert judgement.provenance["cascade_tier"] == "escalated"
        assert judgement.provenance["student_score"] == 0.5
        assert judgement.provenance["student_decision_step"] == "stub_student"
        assert judgement.provenance["escalation_decision_step"] == "frontier"

    def test_student_provenance_carries_cascade_keys(self) -> None:
        student = ScriptedJudge(
            {("a", "b"): 0.9}, decision_step="stub_student", provenance={"n_estimators": 10}
        )
        cascade = CascadeJudge(student=student, escalation=ScriptedJudge({}), band=(0.3, 0.7))

        (judgement,) = list(cascade.forward(iter([_pair("a", "b")])))

        assert judgement.provenance["n_estimators"] == 10  # original key preserved
        assert judgement.provenance["cascade_tier"] == "student"
        assert judgement.provenance["student_score"] == 0.9
        assert judgement.provenance["student_decision_step"] == "stub_student"


# ---------------------------------------------------------------------------
# The exactly-one-judgement contract guard
# ---------------------------------------------------------------------------


class TestOneGuard:
    def test_student_yielding_zero_judgements_raises(self) -> None:
        cascade = CascadeJudge(student=ZeroJudge({}), escalation=ScriptedJudge({}), band=(0.3, 0.7))
        with pytest.raises(RuntimeError, match="student judge produced 0 judgements"):
            list(cascade.forward(iter([_pair("a", "b")])))

    def test_escalation_yielding_two_judgements_raises(self) -> None:
        cascade = CascadeJudge(
            student=ScriptedJudge({("a", "b"): 0.5}),
            escalation=DoubleJudge({}),
            band=(0.3, 0.7),
        )
        with pytest.raises(RuntimeError, match="escalation judge produced 2 judgements"):
            list(cascade.forward(iter([_pair("a", "b")])))


# ---------------------------------------------------------------------------
# One-time score_type contract warning
# ---------------------------------------------------------------------------


class TestScoreTypeWarning:
    def test_non_probability_student_warns_once(self) -> None:
        student = ScriptedJudge({("a", "b"): 0.9, ("c", "d"): 0.95}, score_type="heuristic")
        cascade = CascadeJudge(student=student, escalation=ScriptedJudge({}), band=(0.3, 0.7))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            list(cascade.forward(iter([_pair("a", "b"), _pair("c", "d")])))
        contract_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning) and "score_type" in str(w.message)
        ]
        assert len(contract_warnings) == 1  # one-time, not per pair

        # A second forward() on the SAME instance stays silent too.
        with warnings.catch_warnings(record=True) as caught_again:
            warnings.simplefilter("always")
            list(cascade.forward(iter([_pair("a", "b")])))
        assert not [w for w in caught_again if "score_type" in str(w.message)]

    def test_non_probability_escalation_warns(self) -> None:
        student = ScriptedJudge({("a", "b"): 0.5})
        escalation = ScriptedJudge({("a", "b"): 0.8}, score_type="sim_cos")
        cascade = CascadeJudge(student=student, escalation=escalation, band=(0.3, 0.7))

        with pytest.warns(UserWarning, match="sim_cos"):
            list(cascade.forward(iter([_pair("a", "b")])))

    def test_probability_score_types_stay_silent(self) -> None:
        student = ScriptedJudge({("a", "b"): 0.5, ("c", "d"): 0.9}, score_type="prob_rf")
        escalation = ScriptedJudge({("a", "b"): 0.8}, score_type="prob_llm")
        cascade = CascadeJudge(student=student, escalation=escalation, band=(0.3, 0.7))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            list(cascade.forward(iter([_pair("a", "b"), _pair("c", "d")])))
        assert not [w for w in caught if "score_type" in str(w.message)]


# ---------------------------------------------------------------------------
# Mixed score_type stream through one Clusterer threshold
# ---------------------------------------------------------------------------


class TestMixedStreamClustering:
    def test_mixed_prob_rf_and_prob_llm_cut_by_one_threshold(self) -> None:
        student = ScriptedJudge(
            {("a", "b"): 0.95, ("c", "d"): 0.5, ("e", "f"): 0.05}, score_type="prob_rf"
        )
        escalation = ScriptedJudge({("c", "d"): 0.9}, score_type="prob_llm")
        cascade = CascadeJudge(student=student, escalation=escalation, band=(0.3, 0.7))

        judgements = list(
            cascade.forward(iter([_pair("a", "b"), _pair("c", "d"), _pair("e", "f")]))
        )
        assert {j.score_type for j in judgements} == {"prob_rf", "prob_llm"}

        clusters = Clusterer(threshold=0.7).cluster(judgements)
        assert {frozenset(c) for c in clusters} == {
            frozenset({"a", "b"}),  # student-tier match
            frozenset({"c", "d"}),  # escalated match, prob_llm score
        }


# ---------------------------------------------------------------------------
# Spend-cap / logging composition (BudgetExceeded partials)
# ---------------------------------------------------------------------------


class TestBudgetExceededComposition:
    def test_capped_escalation_under_logging_module_keeps_paid_judgements(
        self, tmp_path: Path
    ) -> None:
        """The exact composition from the plan: a _SpendCappedModule-wrapped
        escalation child under LoggingModule must NOT drop the paid judgement
        from the log (judgement_log.py's ``[logged:]`` slice)."""
        student = ScriptedJudge(
            {("a", "b"): 0.9, ("c", "d"): 0.1, ("e", "f"): 0.5}, decision_step="stub_student"
        )
        expensive = ScriptedJudge(
            {("e", "f"): 0.8},
            score_type="prob_llm",
            decision_step="frontier",
            provenance={"cost_usd": 1.0, "model": "frontier-x"},
        )
        capped: _SpendCappedModule = _SpendCappedModule(expensive, budget_usd=0.5)
        cascade: CascadeJudge[CompanySchema] = CascadeJudge(
            student=student, escalation=capped, band=(0.3, 0.7)
        )
        log = JudgementLog(tmp_path / "log.jsonl")
        logging_module = LoggingModule(cascade, log=log, threshold=0.5)

        with pytest.raises(BudgetExceeded) as excinfo:
            list(logging_module.forward(iter([_pair("a", "b"), _pair("c", "d"), _pair("e", "f")])))

        # partial_judgements is the cascade's own full produced list, in
        # cascade form: two student pass-throughs + the paid escalated pair.
        partials = excinfo.value.partial_judgements
        assert [j.decision_step for j in partials] == [
            CASCADE_STUDENT_STEP,
            CASCADE_STUDENT_STEP,
            CASCADE_ESCALATED_STEP,
        ]
        assert partials[2].provenance["cost_usd"] == 1.0
        assert partials[2].provenance["model"] == "frontier-x"

        # ... so the log keeps the paid judgement (the flywheel's most
        # valuable row) instead of slicing it away.
        rows = log.read()
        assert len(rows) == 3
        assert rows[2]["decision_step"] == CASCADE_ESCALATED_STEP
        assert rows[2]["cost_usd"] == 1.0
        assert rows[2]["model"] == "frontier-x"

    def test_outer_spend_cap_sees_escalated_cost(self) -> None:
        """cost_usd passes through the merge, so an OUTER _SpendCappedModule
        wrapping the whole cascade trips on escalation spend unchanged."""
        student = ScriptedJudge({("a", "b"): 0.9, ("c", "d"): 0.5})
        escalation = ScriptedJudge(
            {("c", "d"): 0.8},
            score_type="prob_llm",
            provenance={"cost_usd": 1.0, "model": "frontier-x"},
        )
        cascade: CascadeJudge[CompanySchema] = CascadeJudge(
            student=student, escalation=escalation, band=(0.3, 0.7)
        )
        capped: _SpendCappedModule = _SpendCappedModule(cascade, budget_usd=0.5)

        with pytest.raises(BudgetExceeded) as excinfo:
            list(capped.forward(iter([_pair("a", "b"), _pair("c", "d")])))

        partials = excinfo.value.partial_judgements
        assert partials[-1].decision_step == CASCADE_ESCALATED_STEP
        assert partials[-1].provenance["cost_usd"] == 1.0


# ---------------------------------------------------------------------------
# Non-BudgetExceeded escalation failures propagate unchanged
# ---------------------------------------------------------------------------


class TestNonBudgetExceededPropagation:
    def test_generic_escalation_exception_propagates_unmodified(self) -> None:
        """Only BudgetExceeded gets partial_judgements rewritten; a generic
        failure (e.g. network/rate-limit/API error) from the escalation child
        must propagate as-is -- not swallowed, not converted."""
        student = ScriptedJudge({("a", "b"): 0.5})  # in-band -> escalates
        cascade = CascadeJudge(student=student, escalation=RaisingJudge({}), band=(0.3, 0.7))

        with pytest.raises(RuntimeError, match="escalation backend unavailable"):
            list(cascade.forward(iter([_pair("a", "b")])))


# ---------------------------------------------------------------------------
# Serialization: registry config round-trip + SerializableState sidecars
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_config_roundtrip_via_registry(self) -> None:
        specs = [FeatureSpec(name="name")]
        cascade: CascadeJudge[CompanySchema] = CascadeJudge(
            student=WeightedAverageJudge(feature_specs=specs),
            escalation=WeightedAverageJudge(feature_specs=specs),
            band=(0.3, 0.7),
        )

        config = cascade.config
        json.dumps(config)  # registry configs must be plain JSON data
        assert config["band"] == [0.3, 0.7]

        cls = get_component("cascade_judge")
        assert cls is CascadeJudge
        rebuilt = cls.from_config(config)
        assert isinstance(rebuilt, CascadeJudge)
        assert rebuilt.band == (0.3, 0.7)
        assert isinstance(rebuilt.student, WeightedAverageJudge)
        assert isinstance(rebuilt.escalation, WeightedAverageJudge)
        assert rebuilt.config == config

    def test_config_rejects_unregistered_child(self) -> None:
        cascade: CascadeJudge[CompanySchema] = CascadeJudge(
            student=ScriptedJudge({}), escalation=ScriptedJudge({}), band=(0.3, 0.7)
        )
        with pytest.raises(ValueError, match="type_name"):
            _ = cascade.config

    def test_registry_lazy_map_entry(self) -> None:
        """Pin the lazy-map wiring so a fresh-process ``Resolver.load`` on a
        cascade_judge artifact resolves even if ``langres.core``'s eager import
        of the module is ever trimmed (mirrors the w1_2/w1_3 wiring tests)."""
        from langres.core.registry import _LAZY_COMPONENT_MODULES

        assert _LAZY_COMPONENT_MODULES["cascade_judge"] == "langres.core.modules.cascade_judge"

    def test_save_state_writes_per_child_subdirs(self, tmp_path: Path) -> None:
        cascade: CascadeJudge[CompanySchema] = CascadeJudge(
            student=StatefulStubJudge({}, value="fitted-student"),
            escalation=ScriptedJudge({}),  # no SerializableState: skipped
            band=(0.3, 0.7),
        )
        cascade.save_state(tmp_path)

        assert (tmp_path / "student" / "state.txt").read_text() == "fitted-student"
        assert not (tmp_path / "escalation").exists()

    def test_save_state_drops_empty_child_dir(self, tmp_path: Path) -> None:
        """A stateful child with nothing to persist (e.g. unfit) leaves no dir."""
        cascade: CascadeJudge[CompanySchema] = CascadeJudge(
            student=StatefulStubJudge({}, value=""),  # writes nothing
            escalation=ScriptedJudge({}),
            band=(0.3, 0.7),
        )
        cascade.save_state(tmp_path)
        assert not (tmp_path / "student").exists()

    def test_load_state_restores_stateful_child(self, tmp_path: Path) -> None:
        (tmp_path / "student").mkdir()
        (tmp_path / "student" / "state.txt").write_text("fitted-student")

        student = StatefulStubJudge({}, value="")
        cascade: CascadeJudge[CompanySchema] = CascadeJudge(
            student=student, escalation=ScriptedJudge({}), band=(0.3, 0.7)
        )
        cascade.load_state(tmp_path)
        assert student.value == "fitted-student"

    def test_load_state_tolerates_missing_and_stateless(self, tmp_path: Path) -> None:
        """Absent subdirs and stateless children are both no-ops (never raise)."""
        student = StatefulStubJudge({}, value="untouched")
        cascade: CascadeJudge[CompanySchema] = CascadeJudge(
            student=student, escalation=ScriptedJudge({}), band=(0.3, 0.7)
        )
        cascade.load_state(tmp_path)  # empty dir: nothing to restore
        assert student.value == "untouched"

    def test_resolver_save_load_roundtrip_with_fitted_rf_student(self, tmp_path: Path) -> None:
        """Full Resolver round trip: fitted RandomForestJudge student survives save/load
        via the cascade's SerializableState sidecar (zero Resolver changes)."""
        pytest.importorskip("sklearn")
        from langres.core import AllPairsBlocker, Resolver
        from langres.core.comparator import StringComparator
        from langres.core.modules.random_forest_judge import RandomForestJudge

        comparator = StringComparator.from_schema(CompanySchema)
        candidates: list[ERCandidate[CompanySchema]] = []
        labels: list[bool] = []
        for i in range(10):
            left = CompanySchema(id=f"m{i}L", name=f"Acme Corporation {i}")
            right = CompanySchema(id=f"m{i}R", name=f"Acme Corporation {i}")
            candidates.append(
                ERCandidate(
                    left=left,
                    right=right,
                    blocker_name="test",
                    comparison=comparator.compare(left, right),
                )
            )
            labels.append(True)
            left = CompanySchema(id=f"n{i}L", name=f"Zephyr Holdings {i}")
            right = CompanySchema(id=f"n{i}R", name=f"Quasar Industries {i}")
            candidates.append(
                ERCandidate(
                    left=left,
                    right=right,
                    blocker_name="test",
                    comparison=comparator.compare(left, right),
                )
            )
            labels.append(False)

        student: RandomForestJudge[CompanySchema] = RandomForestJudge(
            feature_specs=comparator.feature_specs, n_estimators=10, random_state=0
        )
        student.fit(iter(candidates), labels)
        # Unfit escalation: persists nothing, so its sidecar subdir is dropped
        # (the tolerate-empty-dirs contract in a real Resolver round trip).
        escalation: RandomForestJudge[CompanySchema] = RandomForestJudge(
            feature_specs=comparator.feature_specs, n_estimators=5, random_state=1
        )
        cascade: CascadeJudge[CompanySchema] = CascadeJudge(
            student=student, escalation=escalation, band=(0.45, 0.55)
        )
        resolver = Resolver(
            blocker=AllPairsBlocker(schema=CompanySchema),
            comparator=comparator,
            module=cascade,
            clusterer=Clusterer(threshold=0.5),
        )
        resolver.save(tmp_path)

        # The ACTUAL sidecar layout on disk: per-child subdirs under the
        # module slot; the unfit escalation child leaves no dir at all.
        manifest = json.loads((tmp_path / "resolver.json").read_text())
        module_spec = next(c for c in manifest["components"] if c["slot"] == "module")
        assert module_spec["type_name"] == "cascade_judge"
        assert module_spec["config"]["student"]["type_name"] == "random_forest"
        assert module_spec["config"]["escalation"]["type_name"] == "random_forest"
        assert (tmp_path / "module" / "student" / "forest.json").exists()
        assert not (tmp_path / "module" / "escalation").exists()

        reloaded = Resolver.load(tmp_path)
        assert isinstance(reloaded.module, CascadeJudge)
        assert reloaded.module.band == (0.45, 0.55)

        records = [
            {"id": "a", "name": "Acme Corporation 1"},
            {"id": "b", "name": "Acme Corporation 1"},
        ]
        original = resolver.predict(records)
        rebuilt = reloaded.predict(records)
        assert [j.score for j in rebuilt] == pytest.approx([j.score for j in original])
        assert [j.decision_step for j in rebuilt] == [CASCADE_STUDENT_STEP]


# ---------------------------------------------------------------------------
# inspect_scores
# ---------------------------------------------------------------------------


class TestInspectScores:
    def test_inspect_scores_returns_report(self) -> None:
        student = ScriptedJudge({("a", "b"): 0.9})
        cascade = CascadeJudge(student=student, escalation=ScriptedJudge({}), band=(0.3, 0.7))
        judgements = list(cascade.forward(iter([_pair("a", "b")])))
        report = cascade.inspect_scores(judgements)
        assert isinstance(report, ScoreInspectionReport)
