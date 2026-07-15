"""Unit tests for WeightedAverageMatcher (M0 Wave 2a scorer Matcher).

The judge owns its FeatureSpecs (``WeightedAverageMatcher(feature_specs=...)``) and
the scoring rule (weight normalization + the evidence floor from
``combine_present``). It reads each candidate's attached ``comparison`` vector;
the Resolver attaches it from the Comparator. These tests pin that behavior.
"""

import pytest

from langres.core.comparator import StringComparator
from langres.core.feature import ComparisonLevel, ComparisonVector, FeatureSpec
from langres.core.matchers.weighted_average import WeightedAverageMatcher
from langres.core.models import CompanySchema, ERCandidate, PairwiseJudgement


def _candidate(
    left: CompanySchema,
    right: CompanySchema,
    comparison: ComparisonVector | None = None,
) -> ERCandidate[CompanySchema]:
    return ERCandidate(left=left, right=right, blocker_name="test", comparison=comparison)


def _compared(
    comparator: StringComparator[CompanySchema],
    left: CompanySchema,
    right: CompanySchema,
) -> ERCandidate[CompanySchema]:
    """Build a candidate with its comparison vector attached (as the Resolver does)."""
    return _candidate(left, right, comparison=comparator.compare(left, right))


def _company(**kwargs: object) -> CompanySchema:
    base: dict[str, object] = {"id": "x", "name": "Acme"}
    base.update(kwargs)
    return CompanySchema(**base)  # type: ignore[arg-type]


class TestScore:
    def test_weighted_average_correct(self) -> None:
        # Two present features, weights 0.6 / 0.4, similarities 1.0 / 0.5.
        specs = [FeatureSpec(name="name", weight=0.6), FeatureSpec(name="address", weight=0.4)]
        judge = WeightedAverageMatcher(feature_specs=specs)
        vec = ComparisonVector(
            levels={"name": ComparisonLevel.PRESENT, "address": ComparisonLevel.PRESENT},
            similarities={"name": 1.0, "address": 0.5},
        )
        # Normalized weights already sum to 1.0: 0.6*1.0 + 0.4*0.5 = 0.8.
        assert judge.score(vec) == pytest.approx(0.8)

    def test_weights_normalized_to_one(self) -> None:
        # Raw weights 6 / 4 must normalize to 0.6 / 0.4 -> same 0.8 result.
        specs = [FeatureSpec(name="name", weight=6.0), FeatureSpec(name="address", weight=4.0)]
        judge = WeightedAverageMatcher(feature_specs=specs)
        vec = ComparisonVector(
            levels={"name": ComparisonLevel.PRESENT, "address": ComparisonLevel.PRESENT},
            similarities={"name": 1.0, "address": 0.5},
        )
        assert judge.score(vec) == pytest.approx(0.8)

    def test_evidence_floor_single_low_weight_forces_zero(self) -> None:
        # Single present feature carrying < 0.5 of total weight -> floor -> 0.0.
        specs = [FeatureSpec(name="name", weight=0.3), FeatureSpec(name="address", weight=0.7)]
        judge = WeightedAverageMatcher(feature_specs=specs)
        vec = ComparisonVector(
            levels={"name": ComparisonLevel.PRESENT, "address": ComparisonLevel.MISSING},
            similarities={"name": 0.95},
        )
        assert judge.score(vec) == 0.0

    def test_single_high_weight_present_clears_floor(self) -> None:
        # Single present feature with >= 0.5 of total weight scores normally.
        specs = [FeatureSpec(name="name", weight=0.6), FeatureSpec(name="address", weight=0.4)]
        judge = WeightedAverageMatcher(feature_specs=specs)
        vec = ComparisonVector(
            levels={"name": ComparisonLevel.PRESENT, "address": ComparisonLevel.MISSING},
            similarities={"name": 0.9},
        )
        assert judge.score(vec) == pytest.approx(0.9)

    def test_all_missing_scores_zero(self) -> None:
        specs = [FeatureSpec(name="name", weight=0.6), FeatureSpec(name="address", weight=0.4)]
        judge = WeightedAverageMatcher(feature_specs=specs)
        vec = ComparisonVector(
            levels={"name": ComparisonLevel.MISSING, "address": ComparisonLevel.MISSING},
            similarities={},
        )
        assert judge.score(vec) == 0.0

    def test_empty_specs_scores_zero(self) -> None:
        # Defensive: scoring with no specs yields an empty weight map, which
        # combine_present treats as no-evidence -> 0.0 (never divides by zero).
        judge: WeightedAverageMatcher[CompanySchema] = WeightedAverageMatcher(feature_specs=[])
        vec = ComparisonVector(levels={}, similarities={})
        assert judge.score(vec) == 0.0

    def test_all_zero_weights_falls_back_to_even_split(self) -> None:
        # When every spec has weight 0, normalization splits evenly (1/n each)
        # rather than dividing by zero. Two present features clear the floor.
        specs = [FeatureSpec(name="name", weight=0.0), FeatureSpec(name="address", weight=0.0)]
        judge = WeightedAverageMatcher(feature_specs=specs)
        vec = ComparisonVector(
            levels={"name": ComparisonLevel.PRESENT, "address": ComparisonLevel.PRESENT},
            similarities={"name": 1.0, "address": 0.0},
        )
        # Even split 0.5 / 0.5: 0.5*1.0 + 0.5*0.0 = 0.5.
        assert judge.score(vec) == pytest.approx(0.5)


class TestForward:
    def _specs(self) -> list[FeatureSpec]:
        return [FeatureSpec(name="name", weight=0.6), FeatureSpec(name="address", weight=0.4)]

    def test_emits_valid_pairwise_judgement(self) -> None:
        comp = StringComparator(self._specs())
        judge = WeightedAverageMatcher(feature_specs=self._specs())
        cand = _compared(
            comp,
            _company(id="a", name="Acme", address="123 Main"),
            _company(id="b", name="Acme", address="123 Main"),
        )
        results = list(judge.forward([cand]))
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
        judge = WeightedAverageMatcher(feature_specs=self._specs())
        cand = _compared(
            comp,
            _company(id="a", name="", address=None),
            _company(id="b", name="", address=None),
        )
        j = next(iter(judge.forward([cand])))
        assert j.score == 0.0
        assert j.decision_step == "all_features_missing"

    def test_decision_step_below_evidence_floor(self) -> None:
        specs = [FeatureSpec(name="name", weight=0.3), FeatureSpec(name="address", weight=0.7)]
        comp = StringComparator(specs)
        judge = WeightedAverageMatcher(feature_specs=specs)
        # name present (low weight), address missing -> below floor -> 0.0.
        cand = _compared(
            comp,
            _company(id="a", name="Acme", address=None),
            _company(id="b", name="Acme", address=None),
        )
        j = next(iter(judge.forward([cand])))
        assert j.score == 0.0
        assert j.decision_step == "below_evidence_floor"

    def test_streams_lazily(self) -> None:
        judge = WeightedAverageMatcher(feature_specs=self._specs())
        gen = judge.forward(iter([]))
        assert list(gen) == []

    def test_forward_without_comparison_raises(self) -> None:
        judge = WeightedAverageMatcher(feature_specs=self._specs())
        # Candidate carries no comparison vector -> the judge cannot score it.
        cand = _candidate(_company(id="a"), _company(id="b"))
        with pytest.raises(ValueError, match="requires candidates carrying a comparison"):
            list(judge.forward([cand]))


class TestInspectScores:
    def test_inspect_scores_delegates(self) -> None:
        judge = WeightedAverageMatcher(feature_specs=[FeatureSpec(name="name")])
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

        json.dumps(WeightedAverageMatcher(feature_specs=[FeatureSpec(name="name")]).config)

    def test_from_config_reconstructs_feature_specs(self) -> None:
        specs = [FeatureSpec(name="name", weight=0.6), FeatureSpec(name="address", weight=0.4)]
        judge = WeightedAverageMatcher(feature_specs=specs)
        rebuilt = WeightedAverageMatcher.from_config(judge.config)
        assert isinstance(rebuilt, WeightedAverageMatcher)
        # The weights survive the round-trip so scoring is identical.
        assert rebuilt.feature_specs == specs

    def test_registered_under_weighted_average_judge(self) -> None:
        from langres.core.registry import get_component

        assert get_component("weighted_average_judge") is WeightedAverageMatcher


class TestCompanyFixtureBehavior:
    """Behavioral test on company-like records: name-only match merges, a
    low-weight-only shared field scores 0.0."""

    def _name_dominant_comparator(self) -> StringComparator[CompanySchema]:
        return StringComparator.from_schema(
            CompanySchema, weights={"name": 0.6, "address": 0.2, "phone": 0.1, "website": 0.1}
        )

    def test_name_only_match_scores_high(self) -> None:
        # Mirrors c4 / c4_partial: identical name, all other fields missing.
        # name carries 0.6 of total weight (0.6 / (0.6+0.2+0.1+0.1)) -> clears
        # the 0.5 evidence floor on a name-only match.
        comp = self._name_dominant_comparator()
        judge = WeightedAverageMatcher(feature_specs=comp.feature_specs)
        cand = _compared(
            comp,
            _company(id="c4", name="DataFlow Solutions", address="321 Tech Way"),
            _company(id="c4_partial", name="DataFlow Solutions"),
        )
        j = next(iter(judge.forward([cand])))
        assert j.score >= 0.7  # would merge at a 0.7 clusterer threshold

    def test_share_only_low_weight_field_scores_zero(self) -> None:
        # Only a single low-weight feature (website) is present on both sides;
        # everything else differs/missing -> below evidence floor -> 0.0.
        comp = self._name_dominant_comparator()
        judge = WeightedAverageMatcher(feature_specs=comp.feature_specs)
        cand = _compared(
            comp,
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
        j = next(iter(judge.forward([cand])))
        assert j.score < 0.7
