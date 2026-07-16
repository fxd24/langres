"""Tests for FellegiSunterMatcher — the first "learn with no labels" judge (W1.2, S2).

FellegiSunterMatcher fits per-feature m/u agreement probabilities via EM
(``UnsupervisedFitMixin.fit_unlabeled``) and scores candidates with the
classical Fellegi-Sunter posterior. Critical design constraint (E4, both Eng
voices in the M4.5/M5 plan): it must NOT rely on ``ComparisonLevel.MISMATCH``
(StringComparator never emits it, and emitting it would silently change
``combine_present`` scoring for every other judge). Instead FS binarizes
PRESENT similarities into agree/disagree *inside the judge itself*, using its
own ``agreement_threshold`` — the comparator and ``combine_present`` stay
completely untouched.

u-probabilities are estimated from RANDOM pairs drawn from the entity pool
seen in the candidate stream (not from the blocked candidate pairs themselves,
which are match-enriched by the blocker and would bias u upward) — mirrors
Splink's separate random-sampling u-estimation step.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

from langres.core.comparator import StringComparator
from langres.core.feature import ComparisonLevel, ComparisonVector, FeatureSpec
from langres.core.matchers.fellegi_sunter import _GUARD_EPS, _PROB_EPS, FellegiSunterMatcher
from langres.core.models import CompanySchema, ERCandidate
from langres.core.registry import get_component


class ProductSchema(BaseModel):
    """Second schema for schema-agnostic verification (per TDD-agent mandate)."""

    id: str
    title: str
    brand: str | None = None


def _company(id: str, name: str, address: str | None = None) -> CompanySchema:
    return CompanySchema(id=id, name=name, address=address)


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
    return _candidate(left, right, comparison=comparator.compare(left, right))


def _company_comparator(**kwargs: object) -> StringComparator[CompanySchema]:
    return StringComparator.from_schema(CompanySchema, **kwargs)  # type: ignore[arg-type]


def _clustered_pairs(
    comparator: StringComparator[CompanySchema],
    n_matches: int = 15,
    n_nonmatches: int = 15,
) -> list[ERCandidate[CompanySchema]]:
    """A synthetic candidate set with a genuine agree/disagree separation.

    "Matches" share name and address (both high similarity); "non-matches" have
    unrelated names/addresses (low similarity) — gives EM real signal to learn
    from, unlike the degenerate all-agree/all-disagree fixtures below.
    """
    candidates = []
    for i in range(n_matches):
        left = _company(f"m{i}L", f"Acme Corporation {i}", f"{i} Main Street")
        right = _company(f"m{i}R", f"Acme Corporation {i}", f"{i} Main Street")
        candidates.append(_compared(comparator, left, right))
    for i in range(n_nonmatches):
        left = _company(f"n{i}L", f"Zephyr Holdings {i}", f"{i} Ocean Avenue")
        right = _company(f"n{i}R", f"Quasar Industries {i}", f"{i} Mountain Road")
        candidates.append(_compared(comparator, left, right))
    return candidates


# ---------------------------------------------------------------------------
# fit_unlabeled: EM basics
# ---------------------------------------------------------------------------


class TestFitUnlabeledBasics:
    def test_fit_unlabeled_sets_prior_m_and_u(self) -> None:
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, random_state=0
        )
        judge.fit_unlabeled(iter(_clustered_pairs(comparator)))

        assert judge.prior is not None
        assert judge.m_prob is not None
        assert judge.u_prob is not None
        assert 0.0 < judge.prior < 1.0
        for name in comparator.feature_specs:
            assert 0.0 < judge.m_prob[name.name] < 1.0
            assert 0.0 < judge.u_prob[name.name] < 1.0

    def test_fit_unlabeled_separates_matches_from_nonmatches(self) -> None:
        """With real separating signal, EM should learn m > u for discriminative features."""
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, random_state=0, n_random_pairs=200
        )
        candidates = _clustered_pairs(comparator, n_matches=20, n_nonmatches=20)
        judge.fit_unlabeled(iter(candidates))

        assert judge.m_prob is not None and judge.u_prob is not None
        # "name" is a clean discriminator in this fixture (identical for matches,
        # different for non-matches): m should end up clearly higher than u.
        assert judge.m_prob["name"] > judge.u_prob["name"]

    def test_m_prob_never_below_u_prob(self) -> None:
        """The m>=u guard (label-switch protection) always holds after fit."""
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, random_state=1
        )
        judge.fit_unlabeled(iter(_clustered_pairs(comparator)))
        assert judge.m_prob is not None and judge.u_prob is not None
        for name in judge.m_prob:
            assert judge.m_prob[name] >= judge.u_prob[name]

    def test_u_prob_estimated_from_random_pairs_not_blocked_candidates(self) -> None:
        """u must come from the random-pair pool, not the (match-enriched) candidate stream.

        Construct a candidate stream that is ENTIRELY matches (100% agreement on
        every feature) — if u were estimated from these biased candidates
        directly, u would also be ~1.0 for every feature (the classic
        match-enriched bias the plan calls out). Since the entity pool still
        contains distinct-enough records, random cross-pairing breaks that
        correlation and u should land meaningfully below 1.0.
        """
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, random_state=0, n_random_pairs=200
        )
        # Distinct, UNRELATED name pools per company (no shared template/prefix) so
        # random cross-pairs are genuinely dissimilar under token_sort_ratio.
        names = [
            "Zenith Holdings",
            "Umbrella Industries",
            "Wayne Enterprises",
            "Stark Technologies",
            "Globex Corporation",
            "Initech Solutions",
            "Hooli Systems",
            "Soylent Foods",
            "Aperture Science",
            "Cyberdyne Systems",
            "Massive Dynamic",
            "Oscorp Industries",
            "Wonka Confections",
            "Gringotts Bank",
            "Weyland Yutani",
            "Tyrell Corporation",
            "Vandelay Industries",
            "Prestige Worldwide",
            "Pied Piper Inc",
            "Buy N Large",
        ]
        all_match_candidates = [
            _compared(
                comparator,
                _company(f"{i}L", names[i], f"{i} Unique Street"),
                _company(f"{i}R", names[i], f"{i} Unique Street"),
            )
            for i in range(len(names))
        ]
        judge.fit_unlabeled(iter(all_match_candidates))
        assert judge.u_prob is not None
        # Random cross-pairs of distinct companies should rarely agree on name.
        assert judge.u_prob["name"] < 0.5

    def test_fit_unlabeled_raises_without_comparison_vector(self) -> None:
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(comparator=comparator)
        bare = _candidate(_company("a", "Acme"), _company("b", "Acme Inc"), comparison=None)
        with pytest.raises(ValueError, match="comparison vector"):
            judge.fit_unlabeled(iter([bare]))

    def test_fit_unlabeled_is_schema_agnostic_with_product_schema(self) -> None:
        """The exact same judge class works against a second, unrelated schema."""
        comparator: StringComparator[ProductSchema] = StringComparator.from_schema(ProductSchema)
        judge: FellegiSunterMatcher[ProductSchema] = FellegiSunterMatcher(
            comparator=comparator, random_state=0
        )
        candidates = [
            ERCandidate(
                left=ProductSchema(id=f"{i}L", title=f"Widget {i}", brand="Acme"),
                right=ProductSchema(id=f"{i}R", title=f"Widget {i}", brand="Acme"),
                blocker_name="test",
                comparison=comparator.compare(
                    ProductSchema(id=f"{i}L", title=f"Widget {i}", brand="Acme"),
                    ProductSchema(id=f"{i}R", title=f"Widget {i}", brand="Acme"),
                ),
            )
            for i in range(10)
        ]
        judge.fit_unlabeled(iter(candidates))
        assert judge.prior is not None
        [judgement] = list(judge.forward(iter(candidates[:1])))
        assert judgement.score_type == "prob_fs"


# ---------------------------------------------------------------------------
# EM numerics: convergence + fallback
# ---------------------------------------------------------------------------


class TestEMConvergence:
    def test_converged_flag_true_on_normal_fit(self) -> None:
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, random_state=0, max_em_iter=50
        )
        judge.fit_unlabeled(iter(_clustered_pairs(comparator)))
        assert judge.converged is True

    def test_max_iter_zero_falls_back_to_priors_and_marks_unconverged(self) -> None:
        """max_em_iter=0 runs no EM steps at all -> fallback to the safe initial priors."""
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, random_state=0, max_em_iter=0
        )
        judge.fit_unlabeled(iter(_clustered_pairs(comparator)))

        assert judge.converged is False
        assert judge.prior == pytest.approx(0.5)
        assert judge.m_prob is not None
        for name in judge.m_prob:
            assert judge.m_prob[name] == pytest.approx(0.9)

    def test_max_iter_zero_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, random_state=0, max_em_iter=0
        )
        with caplog.at_level("WARNING"):
            judge.fit_unlabeled(iter(_clustered_pairs(comparator)))
        assert any("did not converge" in message for message in caplog.messages)

    def test_tol_controls_convergence_speed(self) -> None:
        """A very loose tolerance converges immediately (delta always < tol)."""
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, random_state=0, max_em_iter=50, tol=1.0
        )
        judge.fit_unlabeled(iter(_clustered_pairs(comparator)))
        assert judge.converged is True


class TestEMFallbackGuard:
    """Regression: the non-convergence fallback must not violate the m>=u guard.

    Bug found in W1.2 PR review (Codex, P2): ``_run_em``'s fallback branch
    returned the raw ``init_m=0.9`` for every feature WITHOUT clamping against
    the (separately-estimated) ``u_prob`` -- unlike the converged path, whose
    ``_m_step`` always floors ``m`` at ``u_prob[name] + _GUARD_EPS``. For any
    low-entropy feature whose random-pair ``u_prob`` exceeds the fixed 0.9
    (common — e.g. a field that is near-constant across the corpus), the
    fallback would silently invert evidence: ``forward()`` computes
    ``log(m/u)`` per agreeing feature, and ``log(m/u) < 0`` when ``m < u``
    means AGREEING on that feature lowers the match score instead of raising
    it. The fallback is not a corner case here — the W1.2 replication
    (``docs/research/20260702_w1_trained_family_results.md``) shows it fires
    on every real benchmark dataset at the class-default ``max_em_iter=20``.
    """

    def test_fallback_m_respects_m_geq_u_guard_directly(self) -> None:
        """Unit-level: ``_run_em``'s fallback clamps m against a high u_prob."""
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, max_em_iter=0
        )
        # "name"'s u_prob (0.95) exceeds the raw fallback init_m (0.9) --
        # exactly the scenario that inverts evidence without the guard.
        u_prob = {"name": 0.95, "address": 0.2, "phone": 0.2, "website": 0.2}

        prior, m_prob, converged = judge._run_em(patterns=[], u_prob=u_prob)

        assert converged is False
        assert prior == pytest.approx(0.5)  # the fallback prior is untouched
        for name, u in u_prob.items():
            assert m_prob[name] >= u + _GUARD_EPS - 1e-9

    def test_fallback_never_inverts_evidence_for_low_entropy_feature(self) -> None:
        """End-to-end (fit_unlabeled -> forward): agreement is never anti-evidence.

        "name" is identical across the whole entity pool, so every random pair
        "agrees" on it -> u_prob["name"] lands well above 0.9. max_em_iter=0
        forces the fallback path unconditionally.
        """
        comparator = _company_comparator()
        candidates = [
            _compared(
                comparator,
                _company(f"m{i}L", "Acme Corp", f"{i} Main St"),
                _company(f"m{i}R", "Acme Corp", f"{i} Main St"),
            )
            for i in range(10)
        ]
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, max_em_iter=0, random_state=0
        )
        judge.fit_unlabeled(iter(candidates))

        assert judge.converged is False
        assert judge.u_prob is not None and judge.m_prob is not None
        assert judge.u_prob["name"] > 0.9  # confirms the low-entropy setup
        assert judge.m_prob["name"] >= judge.u_prob["name"]  # the guard holds

        left = _company("x", "Acme Corp", "1 Test Ave")
        right = _company("y", "Acme Corp", "1 Test Ave")
        vector = comparator.compare(left, right)
        [judgement] = list(judge.forward(iter([_candidate(left, right, vector)])))

        # Agreeing on "name" must never be evidence AGAINST a match.
        assert math.log(judge.m_prob["name"] / judge.u_prob["name"]) >= 0.0
        assert judgement.score_type == "prob_fs"


class TestUProbClampLeavesGuardMargin:
    """Regression: u_prob estimation must leave headroom for the m>=u margin.

    Review finding (claude-review, low severity, found on this PR's own P2 fix
    commit): ``_estimate_u_probs`` clamps its raw Laplace-smoothed estimate to
    ``[_PROB_EPS, 1 - _PROB_EPS]`` -- the same ceiling the *final* guarded ``m``
    is clamped to. For an extremely low-entropy feature (every sampled random
    pair "agrees"), enough pairs push the raw estimate arbitrarily close to
    that same ceiling (Laplace smoothing never reaches exactly 1.0, but gets
    arbitrarily close as the sample grows). When ``u_prob[name] + _GUARD_EPS``
    then exceeds the ceiling, the trailing ``_clamp()`` on ``m`` silently pulls
    it back down toward ``u`` -- ``m`` stays ``>= u`` (never inverted, so this
    is not the P2 class of bug), but the *margin* the guard promises
    (``_GUARD_EPS``) is shaved, in the limit to zero.
    """

    def test_estimate_u_probs_leaves_headroom_for_guard_margin(self) -> None:
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(comparator=comparator)

        # Every pair "agrees" on every feature (identical name/address) --
        # with enough pairs, Laplace smoothing pushes the raw estimate
        # arbitrarily close to 1.0 (2001/2002 ~= 0.9995 for 2000 pairs).
        pairs = [
            (
                _company(f"L{i}", "Acme Corp", "1 Main St"),
                _company(f"R{i}", "Acme Corp", "1 Main St"),
            )
            for i in range(2000)
        ]
        u_prob = judge._estimate_u_probs(pairs)

        for u in u_prob.values():
            assert u <= 1.0 - _PROB_EPS - _GUARD_EPS + 1e-9

    def test_fallback_preserves_full_guard_margin_at_extreme_u(self) -> None:
        """End-to-end: even at the extreme, the fallback m keeps the FULL margin."""
        comparator = _company_comparator()
        candidates = [
            _compared(
                comparator,
                _company(f"m{i}L", "Acme Corp", "1 Main St"),
                _company(f"m{i}R", "Acme Corp", "1 Main St"),
            )
            for i in range(70)
        ]
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, max_em_iter=0, random_state=0, n_random_pairs=2000
        )
        judge.fit_unlabeled(iter(candidates))

        assert judge.u_prob is not None and judge.m_prob is not None
        for name in judge.u_prob:
            assert judge.m_prob[name] - judge.u_prob[name] >= _GUARD_EPS - 1e-9


# ---------------------------------------------------------------------------
# Degenerate inputs (explicitly required by the branch spec)
# ---------------------------------------------------------------------------


class TestDegenerateInputs:
    def test_all_missing_features_no_crash(self) -> None:
        """Every candidate has an empty comparison vector (all features MISSING)."""
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, random_state=0
        )
        empty_vector = ComparisonVector(levels={}, similarities={})
        candidates = [
            _candidate(_company(f"{i}L", "X"), _company(f"{i}R", "Y"), comparison=empty_vector)
            for i in range(10)
        ]
        judge.fit_unlabeled(iter(candidates))
        assert judge.prior is not None

        judgements = list(judge.forward(iter(candidates)))
        assert len(judgements) == 10
        for judgement in judgements:
            assert 0.0 <= judgement.score <= 1.0
            assert not math.isnan(judgement.score)
            # No evidence at all -> score collapses to the learned prior.
            assert judgement.score == pytest.approx(judge.prior, abs=1e-6)

    def test_all_agree_pattern_no_crash(self) -> None:
        """Every candidate agrees on every present feature — an extreme, one-sided fixture."""
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, random_state=0, n_random_pairs=50
        )
        candidates = [
            _compared(
                comparator,
                _company(f"{i}L", "Acme Corp", "1 Main St"),
                _company(f"{i}R", "Acme Corp", "1 Main St"),
            )
            for i in range(10)
        ]
        judge.fit_unlabeled(iter(candidates))
        judgements = list(judge.forward(iter(candidates)))
        assert len(judgements) == 10
        for judgement in judgements:
            assert 0.0 <= judgement.score <= 1.0
            assert not math.isnan(judgement.score)

    def test_one_field_schema_no_crash(self) -> None:
        """A comparator with only a single declared feature (F=1) works fine."""
        specs = [FeatureSpec(name="name")]
        comparator: StringComparator[CompanySchema] = StringComparator(specs, schema=CompanySchema)
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, random_state=0
        )
        candidates = _clustered_pairs(comparator, n_matches=10, n_nonmatches=10)
        judge.fit_unlabeled(iter(candidates))

        assert judge.m_prob is not None and set(judge.m_prob) == {"name"}
        judgements = list(judge.forward(iter(candidates)))
        assert len(judgements) == 20
        for judgement in judgements:
            assert 0.0 <= judgement.score <= 1.0

    def test_no_positive_signal_all_disagree_no_crash(self) -> None:
        """Every candidate disagrees on every present feature — no positive evidence anywhere."""
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, random_state=0, n_random_pairs=50
        )
        candidates = [
            _compared(
                comparator,
                _company(f"{i}L", f"Totally Unrelated Name {i}", f"{i} Nowhere Lane"),
                _company(f"{i}R", f"Completely Different Corp {i}", f"{i} Elsewhere Blvd"),
            )
            for i in range(10)
        ]
        judge.fit_unlabeled(iter(candidates))
        judgements = list(judge.forward(iter(candidates)))
        assert len(judgements) == 10
        for judgement in judgements:
            assert 0.0 <= judgement.score <= 1.0
            assert not math.isnan(judgement.score)

    def test_fit_unlabeled_empty_candidate_stream_no_crash(self) -> None:
        """Zero candidates -> EM has no evidence to work with, but must not crash.

        With no patterns, the M-step is data-independent (purely Laplace/guard
        floor), so it settles at a fixed point in a couple of trivial
        iterations — the important guarantee is a valid, finite result, not a
        specific converged flag.
        """
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, random_state=0
        )
        judge.fit_unlabeled(iter([]))
        assert judge.prior is not None
        assert 0.0 < judge.prior < 1.0
        assert judge.m_prob is not None
        for value in judge.m_prob.values():
            assert 0.0 < value < 1.0

    def test_entity_pool_of_one_record_no_crash(self) -> None:
        """A pool with < 2 distinct entities can't form any random pair for u-estimation."""
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, random_state=0, n_random_pairs=50
        )
        solo = _company("x", "Acme")
        candidates = [_candidate(solo, solo, comparison=comparator.compare(solo, solo))]
        judge.fit_unlabeled(iter(candidates))
        assert judge.u_prob is not None
        for value in judge.u_prob.values():
            assert value == pytest.approx(0.5)  # pure Laplace default, no observations


# ---------------------------------------------------------------------------
# forward(): scoring + provenance
# ---------------------------------------------------------------------------


class TestForward:
    def test_forward_raises_before_fit(self) -> None:
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(comparator=comparator)
        candidate = _compared(comparator, _company("a", "Acme"), _company("b", "Acme Inc"))
        with pytest.raises(ValueError, match="fit_unlabeled"):
            list(judge.forward(iter([candidate])))

    def test_forward_raises_without_comparison_vector(self) -> None:
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(comparator=comparator)
        judge.fit_unlabeled(iter(_clustered_pairs(comparator)))
        bare = _candidate(_company("a", "Acme"), _company("b", "Acme Inc"), comparison=None)
        with pytest.raises(ValueError, match="comparison vector"):
            list(judge.forward(iter([bare])))

    def test_forward_score_matches_hand_computed_posterior(self) -> None:
        """Pin the exact FS posterior formula against a hand-computed value."""
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(comparator=comparator)
        # Bypass fit_unlabeled: set fitted state directly to known, simple values.
        judge.prior = 0.5
        judge.m_prob = {"name": 0.9, "address": 0.9, "phone": 0.9, "website": 0.9}
        judge.u_prob = {"name": 0.1, "address": 0.1, "phone": 0.1, "website": 0.1}
        judge.converged = True

        vector = ComparisonVector(
            levels={"name": ComparisonLevel.PRESENT, "address": ComparisonLevel.MISSING},
            similarities={"name": 0.95},  # >= default agreement_threshold=0.5 -> agree
        )
        candidate = _candidate(_company("a", "X"), _company("b", "Y"), comparison=vector)
        [judgement] = list(judge.forward(iter([candidate])))

        # log_odds = log(0.5/0.5) + log(0.9/0.1) = 0 + log(9)
        expected_log_odds = math.log(0.9 / 0.1)
        expected_score = 1.0 / (1.0 + math.exp(-expected_log_odds))
        assert judgement.score == pytest.approx(expected_score)
        assert judgement.score_type == "prob_fs"
        assert judgement.decision_step == "fellegi_sunter_em"

    def test_forward_disagree_lowers_score_below_prior(self) -> None:
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(comparator=comparator)
        judge.prior = 0.5
        judge.m_prob = {"name": 0.9, "address": 0.9, "phone": 0.9, "website": 0.9}
        judge.u_prob = {"name": 0.1, "address": 0.1, "phone": 0.1, "website": 0.1}
        judge.converged = True

        vector = ComparisonVector(
            levels={"name": ComparisonLevel.PRESENT},
            similarities={"name": 0.05},  # below threshold -> disagree
        )
        candidate = _candidate(_company("a", "X"), _company("b", "Y"), comparison=vector)
        [judgement] = list(judge.forward(iter([candidate])))
        assert judgement.score < 0.5

    def test_forward_provenance_includes_pattern(self) -> None:
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(comparator=comparator)
        judge.fit_unlabeled(iter(_clustered_pairs(comparator)))
        candidate = _compared(
            comparator,
            _company("a", "Acme Corp", "1 Main St"),
            _company("b", "Acme Corp", "1 Main St"),
        )
        [judgement] = list(judge.forward(iter([candidate])))
        assert "pattern" in judgement.provenance
        assert "log_odds" in judgement.provenance

    def test_inspect_scores_delegates_to_shared_util(self) -> None:
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(comparator=comparator)
        judge.fit_unlabeled(iter(_clustered_pairs(comparator)))
        judgements = list(judge.forward(iter(_clustered_pairs(comparator))))
        report = judge.inspect_scores(judgements, sample_size=5)
        assert report.total_judgements == len(judgements)

    def test_forward_left_id_right_id(self) -> None:
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(comparator=comparator)
        judge.fit_unlabeled(iter(_clustered_pairs(comparator)))
        candidate = _compared(comparator, _company("left-1", "Acme"), _company("right-1", "Acme"))
        [judgement] = list(judge.forward(iter([candidate])))
        assert judgement.left_id == "left-1"
        assert judgement.right_id == "right-1"


# ---------------------------------------------------------------------------
# agreement_threshold: owned by the judge, not the comparator
# ---------------------------------------------------------------------------


class TestAgreementThreshold:
    def test_custom_agreement_threshold_changes_binarization(self) -> None:
        """A stricter threshold reclassifies a mid-similarity feature as disagree."""
        comparator = _company_comparator()
        lenient: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, agreement_threshold=0.3
        )
        strict: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, agreement_threshold=0.9
        )
        for judge in (lenient, strict):
            judge.prior = 0.5
            judge.m_prob = {"name": 0.9, "address": 0.9, "phone": 0.9, "website": 0.9}
            judge.u_prob = {"name": 0.1, "address": 0.1, "phone": 0.1, "website": 0.1}
            judge.converged = True

        vector = ComparisonVector(
            levels={"name": ComparisonLevel.PRESENT}, similarities={"name": 0.6}
        )
        candidate = _candidate(_company("a", "X"), _company("b", "Y"), comparison=vector)
        [lenient_j] = list(lenient.forward(iter([candidate])))
        [strict_j] = list(strict.forward(iter([candidate])))
        # 0.6 agrees at threshold=0.3, disagrees at threshold=0.9.
        assert lenient_j.score > 0.5
        assert strict_j.score < 0.5


# ---------------------------------------------------------------------------
# Serialization: config / from_config / registry / fresh-process
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_is_registered_with_type_name(self) -> None:
        assert get_component("fellegi_sunter_judge") is FellegiSunterMatcher
        assert FellegiSunterMatcher.type_name == "fellegi_sunter_judge"

    def test_config_is_json_serializable_before_fit(self) -> None:
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, agreement_threshold=0.4, n_random_pairs=123, random_state=7
        )
        config = judge.config
        json.dumps(config)  # must be pure JSON-serializable data
        assert config["agreement_threshold"] == pytest.approx(0.4)
        assert config["n_random_pairs"] == 123
        assert config["random_state"] == 7
        assert config["m_prob"] is None
        assert config["u_prob"] is None
        assert config["prior"] is None
        assert config["converged"] is None

    def test_config_includes_fitted_state_after_fit(self) -> None:
        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, random_state=0
        )
        judge.fit_unlabeled(iter(_clustered_pairs(comparator)))
        config = judge.config
        json.dumps(config)
        assert config["m_prob"] is not None
        assert config["u_prob"] is not None
        assert config["prior"] is not None
        assert config["converged"] is True

    def test_from_config_round_trips_unfit_judge(self) -> None:
        comparator = _company_comparator()
        original: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, agreement_threshold=0.42
        )
        rebuilt = FellegiSunterMatcher.from_config(original.config)
        assert rebuilt.agreement_threshold == pytest.approx(0.42)
        assert rebuilt.m_prob is None
        with pytest.raises(ValueError, match="fit_unlabeled"):
            list(
                rebuilt.forward(
                    iter([_compared(comparator, _company("a", "X"), _company("b", "Y"))])
                )
            )

    def test_from_config_round_trips_fitted_judge_and_scores_identically(self) -> None:
        comparator = _company_comparator()
        original: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, random_state=0
        )
        candidates = _clustered_pairs(comparator)
        original.fit_unlabeled(iter(candidates))
        original_scores = [j.score for j in original.forward(iter(candidates))]

        rebuilt = FellegiSunterMatcher.from_config(original.config)
        rebuilt_scores = [j.score for j in rebuilt.forward(iter(candidates))]

        assert rebuilt_scores == pytest.approx(original_scores)

    def test_resolver_with_fs_judge_saves_and_loads(self, tmp_path: Path) -> None:
        from langres.core import Clusterer, Resolver
        from langres.core.blockers import AllPairsBlocker

        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, random_state=0
        )
        judge.fit_unlabeled(iter(_clustered_pairs(comparator)))
        resolver = Resolver(
            blocker=AllPairsBlocker(schema=CompanySchema),
            comparator=comparator,
            matcher=judge,
            clusterer=Clusterer(threshold=0.5),
        )
        resolver.save(tmp_path)

        manifest = json.loads((tmp_path / "resolver.json").read_text())
        module_spec = next(c for c in manifest["components"] if c["slot"] == "module")
        assert module_spec["type_name"] == "fellegi_sunter_judge"

        reloaded = Resolver.load(tmp_path)
        assert isinstance(reloaded.module, FellegiSunterMatcher)
        assert reloaded.module.converged is True

    @pytest.mark.slow
    def test_resolver_load_fs_judge_in_fresh_process(self, tmp_path: Path) -> None:
        """Fresh-process save/load round trip (the M2 lesson — E12).

        Saves a fitted FellegiSunterMatcher inside a Resolver, then reloads it in a
        brand-new Python subprocess that imports ONLY ``langres.core`` (never this
        test module or the judge module directly) — proving eager registration
        actually fires on plain ``import langres.core``, not just because this
        test file happened to import the class first.
        """
        from langres.core import Clusterer, Resolver
        from langres.core.blockers import AllPairsBlocker

        comparator = _company_comparator()
        judge: FellegiSunterMatcher[CompanySchema] = FellegiSunterMatcher(
            comparator=comparator, random_state=0
        )
        judge.fit_unlabeled(iter(_clustered_pairs(comparator)))
        resolver = Resolver(
            blocker=AllPairsBlocker(schema=CompanySchema),
            comparator=comparator,
            matcher=judge,
            clusterer=Clusterer(threshold=0.5),
        )
        resolver.save(tmp_path)

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from langres.core import Resolver; "
                    f"r = Resolver.load(r'{tmp_path}'); "
                    "assert type(r.module).__name__ == 'FellegiSunterMatcher'; "
                    "assert r.module.converged is True; "
                    "print('OK')"
                ),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            "fresh-process Resolver.load failed for a fellegi_sunter_judge artifact.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        assert "UnknownComponentType" not in result.stderr


# ---------------------------------------------------------------------------
# Regression: existing comparator / WeightedAverageMatcher semantics untouched (E4)
# ---------------------------------------------------------------------------


class TestComparatorSemanticsUnchanged:
    def test_string_comparator_never_emits_mismatch(self) -> None:
        """StringComparator.compare must still only ever emit PRESENT/MISSING."""
        comparator = _company_comparator()
        vector = comparator.compare(
            _company("a", "Acme", address="1 Main St"),
            _company("b", "Zephyr", address=None),
        )
        assert set(vector.levels.values()) <= {ComparisonLevel.PRESENT, ComparisonLevel.MISSING}
        assert ComparisonLevel.MISMATCH not in vector.levels.values()

    def test_weighted_average_judge_score_matches_pre_fs_baseline(self) -> None:
        """Pin WeightedAverageMatcher's score on a fixed fixture (unaffected by this branch).

        Asserted via ``pytest.approx`` against the same expected value as the
        pre-existing pinned fixture below -- not a literal byte-for-byte
        comparison of two runs (nit from review: earlier PR-description
        wording said "byte-identical", which overstated what this checks).
        """
        from langres.core.matchers.weighted_average import WeightedAverageMatcher

        specs = [FeatureSpec(name="name", weight=0.6), FeatureSpec(name="address", weight=0.4)]
        judge = WeightedAverageMatcher(feature_specs=specs)
        vector = ComparisonVector(
            levels={"name": ComparisonLevel.PRESENT, "address": ComparisonLevel.PRESENT},
            similarities={"name": 1.0, "address": 0.5},
        )
        # Same expected value as tests/core/test_weighted_average_judge.py's
        # pre-existing pinned regression fixture — 0.6*1.0 + 0.4*0.5 = 0.8.
        assert judge.score(vector) == pytest.approx(0.8)
