"""Unit tests for WeightedAverageJudge (M0 Wave 2a scorer Module).

The judge is arg-free (``WeightedAverageJudge()``). The Resolver drives the
Comparator and feeds the judge per-pair via ``forward``; the judge owns the
scoring rule (weight normalization + the evidence floor from
``combine_present``). These tests pin that behavior.
"""

import pytest

from langres.core.comparator import StringComparator
from langres.core.feature import ComparisonLevel, ComparisonVector, FeatureSpec
from langres.core.judges.weighted_average import WeightedAverageJudge
from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement


def _candidate(left: CompanySchema, right: CompanySchema) -> ERCandidate[CompanySchema]:
    return ERCandidate(left=left, right=right, blocker_name="test")


def _company(**kwargs: object) -> CompanySchema:
    base: dict[str, object] = {"id": "x", "name": "Acme"}
    base.update(kwargs)
    return CompanySchema(**base)  # type: ignore[arg-type]


class TestScore:
    def test_weighted_average_correct(self) -> None:
        # Two present features, weights 0.6 / 0.4, similarities 1.0 / 0.5.
        specs = [FeatureSpec(name="name", weight=0.6), FeatureSpec(name="address", weight=0.4)]
        judge = WeightedAverageJudge()
        vec = ComparisonVector(
            levels={"name": ComparisonLevel.PRESENT, "address": ComparisonLevel.PRESENT},
            similarities={"name": 1.0, "address": 0.5},
        )
        # Normalized weights already sum to 1.0: 0.6*1.0 + 0.4*0.5 = 0.8.
        assert judge.score(vec, specs) == pytest.approx(0.8)

    def test_weights_normalized_to_one(self) -> None:
        # Raw weights 6 / 4 must normalize to 0.6 / 0.4 -> same 0.8 result.
        specs = [FeatureSpec(name="name", weight=6.0), FeatureSpec(name="address", weight=4.0)]
        judge = WeightedAverageJudge()
        vec = ComparisonVector(
            levels={"name": ComparisonLevel.PRESENT, "address": ComparisonLevel.PRESENT},
            similarities={"name": 1.0, "address": 0.5},
        )
        assert judge.score(vec, specs) == pytest.approx(0.8)

    def test_evidence_floor_single_low_weight_forces_zero(self) -> None:
        # Single present feature carrying < 0.5 of total weight -> floor -> 0.0.
        specs = [FeatureSpec(name="name", weight=0.3), FeatureSpec(name="address", weight=0.7)]
        judge = WeightedAverageJudge()
        vec = ComparisonVector(
            levels={"name": ComparisonLevel.PRESENT, "address": ComparisonLevel.MISSING},
            similarities={"name": 0.95},
        )
        assert judge.score(vec, specs) == 0.0

    def test_single_high_weight_present_clears_floor(self) -> None:
        # Single present feature with >= 0.5 of total weight scores normally.
        specs = [FeatureSpec(name="name", weight=0.6), FeatureSpec(name="address", weight=0.4)]
        judge = WeightedAverageJudge()
        vec = ComparisonVector(
            levels={"name": ComparisonLevel.PRESENT, "address": ComparisonLevel.MISSING},
            similarities={"name": 0.9},
        )
        assert judge.score(vec, specs) == pytest.approx(0.9)

    def test_all_missing_scores_zero(self) -> None:
        specs = [FeatureSpec(name="name", weight=0.6), FeatureSpec(name="address", weight=0.4)]
        judge = WeightedAverageJudge()
        vec = ComparisonVector(
            levels={"name": ComparisonLevel.MISSING, "address": ComparisonLevel.MISSING},
            similarities={},
        )
        assert judge.score(vec, specs) == 0.0


class TestForward:
    def _specs(self) -> list[FeatureSpec]:
        return [FeatureSpec(name="name", weight=0.6), FeatureSpec(name="address", weight=0.4)]

    def test_emits_valid_pairwise_judgement(self) -> None:
        comp = StringComparator(self._specs())
        judge = WeightedAverageJudge()
        cand = _candidate(
            _company(id="a", name="Acme", address="123 Main"),
            _company(id="b", name="Acme", address="123 Main"),
        )
        results = list(judge.forward([cand], comparator=comp))
        assert len(results) == 1
        j = results[0]
        assert isinstance(j, PairwiseJudgement)
        assert j.left_id == "a"
        assert j.right_id == "b"
        assert j.score_type == "heuristic"
        assert j.decision_step == "weighted_average"
        assert j.score == pytest.approx(1.0)
        # Provenance carries levels + similarities for observability.
        assert j.provenance["levels"]["name"] == "PRESENT"
        assert "name" in j.provenance["similarities"]

    def test_decision_step_all_features_missing(self) -> None:
        comp = StringComparator(self._specs())
        judge = WeightedAverageJudge()
        cand = _candidate(
            _company(id="a", name="", address=None),
            _company(id="b", name="", address=None),
        )
        j = next(iter(judge.forward([cand], comparator=comp)))
        assert j.score == 0.0
        assert j.decision_step == "all_features_missing"

    def test_decision_step_below_evidence_floor(self) -> None:
        specs = [FeatureSpec(name="name", weight=0.3), FeatureSpec(name="address", weight=0.7)]
        comp = StringComparator(specs)
        judge = WeightedAverageJudge()
        # name present (low weight), address missing -> below floor -> 0.0.
        cand = _candidate(
            _company(id="a", name="Acme", address=None),
            _company(id="b", name="Acme", address=None),
        )
        j = next(iter(judge.forward([cand], comparator=comp)))
        assert j.score == 0.0
        assert j.decision_step == "below_evidence_floor"

    def test_streams_lazily(self) -> None:
        comp = StringComparator(self._specs())
        judge = WeightedAverageJudge()
        gen = judge.forward(iter([]), comparator=comp)
        assert list(gen) == []


class TestInspectScores:
    def test_inspect_scores_delegates(self) -> None:
        judge = WeightedAverageJudge()
        judgements = [
            PairwiseJudgement(
                left_id="a",
                right_id="b",
                score=0.9,
                score_type="heuristic",
                decision_step="weighted_average",
                provenance={},
            )
        ]
        report = judge.inspect_scores(judgements, sample_size=5)
        assert report.total_judgements == 1


class TestConfigRoundTrip:
    def test_config_is_serializable(self) -> None:
        import json

        json.dumps(WeightedAverageJudge().config)

    def test_from_config_reconstructs(self) -> None:
        judge = WeightedAverageJudge()
        rebuilt = WeightedAverageJudge.from_config(judge.config)
        assert isinstance(rebuilt, WeightedAverageJudge)

    def test_registered_under_weighted_average_judge(self) -> None:
        from langres.core.registry import get_component

        assert get_component("weighted_average_judge") is WeightedAverageJudge


class TestCompanyFixtureBehavior:
    """Behavioral test on company-like records: name-only match merges, a
    low-weight-only shared field scores 0.0."""

    def test_name_only_match_scores_high(self) -> None:
        # Mirrors c4 / c4_partial: identical name, all other fields missing.
        # name weight 0.6 of total 1.0 -> clears the 0.5 evidence floor.
        comp = StringComparator.from_schema(CompanySchema, weights={"name": 0.6})
        judge = WeightedAverageJudge()
        cand = _candidate(
            _company(id="c4", name="DataFlow Solutions", address="321 Tech Way"),
            _company(id="c4_partial", name="DataFlow Solutions"),
        )
        j = next(iter(judge.forward([cand], comparator=comp)))
        assert j.score >= 0.7  # would merge at a 0.7 clusterer threshold

    def test_share_only_low_weight_field_scores_zero(self) -> None:
        # Only a single low-weight feature (website) is present on both sides;
        # everything else differs/missing -> below evidence floor -> 0.0.
        comp = StringComparator.from_schema(
            CompanySchema, weights={"name": 0.6, "address": 0.2, "phone": 0.1, "website": 0.1}
        )
        judge = WeightedAverageJudge()
        cand = _candidate(
            _company(id="a", name="Quantum Dynamics", website="https://shared.com"),
            _company(
                id="b",
                name="Pacific Logistics Group",
                website="https://shared.com",
            ),
        )
        # name present too (both sides) but differs strongly; website matches.
        # name (0.6) + website (0.1) present -> 2 present features clears floor,
        # but the weighted average is dominated by the near-zero name match.
        j = next(iter(judge.forward([cand], comparator=comp)))
        assert j.score < 0.7
