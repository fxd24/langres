"""FellegiSunterJudge: the first "learn with no labels" judge (W1.2, S2).

Classical Fellegi-Sunter probabilistic record linkage, fit via EM with **no
labels**. The judge rides the existing :class:`~langres.core.feature.ComparisonVector`
seam (``candidate.comparison.similarities``) but deliberately does **not**
change comparator semantics to get there (E4, both Eng voices in the M4.5/M5
plan):

- ``StringComparator`` never emits ``ComparisonLevel.MISMATCH`` and this judge
  does not ask it to -- teaching the comparator a discrete 3-way
  agree/disagree/missing taxonomy would change what "PRESENT" means for every
  other judge (``combine_present`` treats *any* PRESENT feature as
  evidence-in-favour; a MISMATCH-as-PRESENT-with-low-similarity would silently
  raise scores for conflicting pairs -- an over-merge regression).
- Instead, FS **binarizes similarities into agree/disagree itself**, feature by
  feature, using its own :attr:`agreement_threshold`. Missing features (absent
  from ``similarities``) stay missing and contribute no evidence, exactly like
  every other consumer of :class:`~langres.core.feature.ComparisonVector`.

u-probabilities (the per-feature agreement rate under a **random**, mostly
non-matching pair) are estimated from a random sample of the entity pool seen
in the fitted candidate stream -- **not** from the blocked candidate pairs
themselves. A blocker's output is match-enriched (a kNN/vector blocker
specifically surfaces near-duplicates), so estimating u directly from it would
bias u upward and understate how discriminative each feature really is
(Splink runs an explicit random-sampling u step for the same reason). The
entity pool is limited to entities that appeared in the fitted stream (the
``fit_unlabeled(candidates)`` contract never sees the raw corpus) -- a
documented approximation, not full-corpus random sampling.

m-probabilities (the per-feature agreement rate under a true match) and the
match prior are learned via **EM in log-space**, holding u fixed. See
:meth:`FellegiSunterJudge._run_em` for the exact numerics: Laplace smoothing,
probability clamping, an m>=u guard against the classical two-cluster
label-switch failure mode, and a convergence check with a max-iteration
fallback to the safe initial priors (never returns numerically degenerate
parameters).
"""

import logging
import math
import random
from collections.abc import Iterator, Sequence
from typing import ClassVar, cast

from langres.core.comparator import StringComparator
from langres.core.feature import ComparisonVector
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.module import Module, SchemaT
from langres.core.registry import register
from langres.core.reports import ScoreInspectionReport, _inspect_scores_impl

logger = logging.getLogger(__name__)

# Laplace (add-alpha) smoothing constant applied to every probability estimate
# (m, u, and the prior) so a small or one-sided sample never yields a
# degenerate 0.0/1.0 estimate (which would make log(m)/log(1-m) blow up).
_LAPLACE_ALPHA = 1.0

# Hard clamp on every learned probability, so log() is always finite and a
# feature can never be treated as perfectly deterministic.
_PROB_EPS = 1e-6

# Minimum margin enforced between a feature's m and u probability (the m>=u
# guard) -- prevents the classical EM label-switch pathology where a feature
# ends up "agreeing is evidence AGAINST a match", which is never the intended
# semantics of a user-declared comparable feature.
_GUARD_EPS = 1e-3

# Safe bound for math.exp() arguments (float64 overflows around 709); used to
# clamp the sigmoid's input so an extreme (very confident) log-odds sum never
# raises OverflowError.
_SIGMOID_CLAMP = 700.0


def _clamp(value: float, low: float = _PROB_EPS, high: float = 1.0 - _PROB_EPS) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return min(max(value, low), high)


def _sigmoid(log_odds: float) -> float:
    """Numerically-safe logistic sigmoid."""
    x = min(max(log_odds, -_SIGMOID_CLAMP), _SIGMOID_CLAMP)
    return 1.0 / (1.0 + math.exp(-x))


def _binarize(vector: ComparisonVector, threshold: float) -> dict[str, bool]:
    """Binarize a ComparisonVector's PRESENT similarities into agree/disagree.

    Only features present in ``vector.similarities`` are included (MISSING
    features contribute no evidence, exactly as they do for every other
    consumer of a ComparisonVector). This is the judge-owned binarization step
    that keeps ``ComparisonLevel``/``combine_present`` semantics untouched.
    """
    return {name: sim >= threshold for name, sim in vector.similarities.items()}


def _sample_random_pairs(
    pool: Sequence[object], n_pairs: int, rng: random.Random
) -> list[tuple[object, object]]:
    """Sample up to ``n_pairs`` distinct, unordered pairs from ``pool``.

    Caps at the number of distinct pairs actually available (``len(pool)
    choose 2``) so a small pool never causes an infinite/slow retry loop.
    Returns an empty list when ``pool`` has fewer than 2 entities.
    """
    n = len(pool)
    if n < 2:
        return []
    max_possible = n * (n - 1) // 2
    target = min(n_pairs, max_possible)

    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[object, object]] = []
    max_attempts = target * 20 + 100
    attempts = 0
    while len(pairs) < target and attempts < max_attempts:
        attempts += 1
        i, j = rng.sample(range(n), 2)
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((pool[i], pool[j]))
    return pairs


@register("fellegi_sunter_judge")
class FellegiSunterJudge(Module[SchemaT]):
    """Classical Fellegi-Sunter judge, fit via EM with no labels.

    Consumes each candidate's attached ``comparison`` (a
    :class:`~langres.core.feature.ComparisonVector`), same as
    ``WeightedAverageJudge``, but requires its own :class:`Comparator`
    instance at construction time -- unlike a purely-scoring judge, FS needs
    the ability to *compute new comparisons* for synthetic random pairs during
    :meth:`fit_unlabeled` (the u-probability estimation step).
    """

    type_name: ClassVar[str] = "fellegi_sunter_judge"

    def __init__(
        self,
        comparator: StringComparator[SchemaT],
        *,
        agreement_threshold: float = 0.5,
        n_random_pairs: int = 1000,
        max_em_iter: int = 20,
        tol: float = 1e-4,
        random_state: int = 0,
    ) -> None:
        """Initialize an (unfit) FellegiSunterJudge.

        Args:
            comparator: Computes ComparisonVectors for both the fitted
                candidates (via their pre-attached ``comparison``) and the
                synthetic random pairs used to estimate u. Its
                ``feature_specs`` define the features FS learns m/u for.
            agreement_threshold: Per-feature similarity cutoff (owned by this
                judge, not the comparator) above which a PRESENT feature is
                binarized to "agree".
            n_random_pairs: Target number of random pairs sampled from the
                fitted stream's entity pool for u-estimation (capped by the
                pool's actual pair count).
            max_em_iter: Maximum EM iterations. ``0`` means "run no EM steps"
                -- fit_unlabeled immediately falls back to the safe initial
                priors.
            tol: Convergence threshold on the max absolute change across the
                prior and every feature's m-probability between iterations.
            random_state: Seed for the random-pair sampler (deterministic
                fits given the same candidate stream).
        """
        self.comparator: StringComparator[SchemaT] = comparator
        self.feature_specs = comparator.feature_specs
        self.agreement_threshold = agreement_threshold
        self.n_random_pairs = n_random_pairs
        self.max_em_iter = max_em_iter
        self.tol = tol
        self.random_state = random_state

        self.prior: float | None = None
        self.m_prob: dict[str, float] | None = None
        self.u_prob: dict[str, float] | None = None
        self.converged: bool | None = None

    # ------------------------------------------------------------------
    # Fitting (UnsupervisedFitMixin)
    # ------------------------------------------------------------------

    def fit_unlabeled(self, candidates: Iterator[ERCandidate[SchemaT]]) -> None:
        """Fit m/u/prior via a random-pair u-estimate + EM over the candidates.

        Args:
            candidates: The blocked, comparison-attached candidate stream to
                learn from. Consumed fully (materialized) -- both the
                u-estimation step and EM need to see the whole set, and the
                entity pool for random pairing is derived from it.

        Raises:
            ValueError: If any candidate carries no comparison vector.
        """
        materialized = list(candidates)

        pool: dict[str, SchemaT] = {}
        for candidate in materialized:
            left_id = candidate.left.id  # type: ignore[attr-defined]
            right_id = candidate.right.id  # type: ignore[attr-defined]
            pool[left_id] = candidate.left
            pool[right_id] = candidate.right
        pool_list = list(pool.values())

        rng = random.Random(self.random_state)
        random_pairs = _sample_random_pairs(pool_list, self.n_random_pairs, rng)
        self.u_prob = self._estimate_u_probs(cast("list[tuple[SchemaT, SchemaT]]", random_pairs))

        patterns: list[dict[str, bool]] = []
        for candidate in materialized:
            vector = candidate.comparison
            if vector is None:
                raise ValueError(
                    "FellegiSunterJudge requires candidates carrying a comparison "
                    "vector — add a Comparator to the pipeline."
                )
            patterns.append(_binarize(vector, self.agreement_threshold))

        self.prior, self.m_prob, self.converged = self._run_em(patterns, self.u_prob)

    def _estimate_u_probs(self, random_pairs: list[tuple[SchemaT, SchemaT]]) -> dict[str, float]:
        """Estimate per-feature u (agreement rate under a random, ~non-match pair)."""
        feature_names = [spec.name for spec in self.feature_specs]
        agree_counts = dict.fromkeys(feature_names, 0.0)
        total_counts = dict.fromkeys(feature_names, 0.0)

        for left, right in random_pairs:
            vector = self.comparator.compare(left, right)
            for name, sim in vector.similarities.items():
                total_counts[name] += 1
                if sim >= self.agreement_threshold:
                    agree_counts[name] += 1

        return {
            name: _clamp(
                (agree_counts[name] + _LAPLACE_ALPHA) / (total_counts[name] + 2 * _LAPLACE_ALPHA)
            )
            for name in feature_names
        }

    def _run_em(
        self, patterns: list[dict[str, bool]], u_prob: dict[str, float]
    ) -> tuple[float, dict[str, float], bool]:
        """Log-space EM for the match prior + per-feature m, holding u fixed.

        Returns ``(prior, m_prob, converged)``. On non-convergence within
        ``max_em_iter`` iterations, falls back to the initial priors
        (``prior=0.5``, ``m=0.9`` for every feature) rather than returning a
        possibly-diverged partial result.
        """
        feature_names = [spec.name for spec in self.feature_specs]
        init_prior = 0.5
        init_m = dict.fromkeys(feature_names, 0.9)

        prior = init_prior
        m_prob = dict(init_m)

        converged = False
        for _iteration in range(self.max_em_iter):
            responsibilities = self._e_step(patterns, prior, m_prob, u_prob)
            new_prior, new_m = self._m_step(patterns, responsibilities, u_prob, feature_names)

            delta = abs(new_prior - prior)
            for name in feature_names:
                delta = max(delta, abs(new_m[name] - m_prob[name]))

            prior, m_prob = new_prior, new_m
            if delta < self.tol:
                converged = True
                break

        if not converged:
            logger.warning(
                "FellegiSunterJudge EM did not converge within max_em_iter=%d "
                "(tol=%g); falling back to the safe initial priors "
                "(prior=%.2f, m=%.2f per feature) for a usable judge.",
                self.max_em_iter,
                self.tol,
                init_prior,
                0.9,
            )
            return init_prior, init_m, False
        return prior, m_prob, True

    def _e_step(
        self,
        patterns: list[dict[str, bool]],
        prior: float,
        m_prob: dict[str, float],
        u_prob: dict[str, float],
    ) -> list[float]:
        """Compute each pattern's P(match | pattern) via log-space Bayes."""
        responsibilities = []
        for pattern in patterns:
            log_match = math.log(prior)
            log_nonmatch = math.log(1.0 - prior)
            for name, agree in pattern.items():
                m = m_prob[name]
                u = u_prob[name]
                log_match += math.log(m) if agree else math.log(1.0 - m)
                log_nonmatch += math.log(u) if agree else math.log(1.0 - u)
            # Stable normalization (subtract the max before exponentiating).
            peak = max(log_match, log_nonmatch)
            p_match = math.exp(log_match - peak)
            p_nonmatch = math.exp(log_nonmatch - peak)
            responsibilities.append(p_match / (p_match + p_nonmatch))
        return responsibilities

    def _m_step(
        self,
        patterns: list[dict[str, bool]],
        responsibilities: list[float],
        u_prob: dict[str, float],
        feature_names: list[str],
    ) -> tuple[float, dict[str, float]]:
        """Update the prior and per-feature m from E-step responsibilities.

        Laplace-smoothed, and each feature's m is floored at ``u_prob[name] +
        _GUARD_EPS`` (the m>=u guard) before being clamped into a valid
        probability range.
        """
        n = len(patterns)
        total_r = sum(responsibilities)
        new_prior = _clamp((total_r + _LAPLACE_ALPHA) / (n + 2 * _LAPLACE_ALPHA))

        new_m: dict[str, float] = {}
        for name in feature_names:
            numerator = _LAPLACE_ALPHA
            denominator = 2 * _LAPLACE_ALPHA
            for pattern, r in zip(patterns, responsibilities, strict=True):
                if name in pattern:
                    denominator += r
                    if pattern[name]:
                        numerator += r
            m_value = numerator / denominator
            m_value = max(m_value, u_prob[name] + _GUARD_EPS)  # m>=u guard
            new_m[name] = _clamp(m_value)
        return new_prior, new_m

    # ------------------------------------------------------------------
    # Scoring (Module)
    # ------------------------------------------------------------------

    def forward(self, candidates: Iterator[ERCandidate[SchemaT]]) -> Iterator[PairwiseJudgement]:
        """Score each candidate with the Fellegi-Sunter posterior.

        Yields:
            One PairwiseJudgement per candidate, ``score_type="prob_fs"``.

        Raises:
            ValueError: If the judge has not been fit yet, or a candidate
                carries no comparison vector.
        """
        if self.prior is None or self.m_prob is None or self.u_prob is None:
            raise ValueError(
                "FellegiSunterJudge must be fit before forward(): call "
                "fit_unlabeled(candidates) directly, or resolver.fit(records) "
                "on a Resolver whose module is this judge."
            )
        prior, m_prob, u_prob = self.prior, self.m_prob, self.u_prob
        base_log_odds = math.log(prior / (1.0 - prior))

        for candidate in candidates:
            vector = candidate.comparison
            if vector is None:
                raise ValueError(
                    "FellegiSunterJudge requires candidates carrying a comparison "
                    "vector — add a Comparator to the pipeline."
                )
            pattern = _binarize(vector, self.agreement_threshold)
            log_odds = base_log_odds
            for name, agree in pattern.items():
                m = m_prob.get(name, 0.5)
                u = u_prob.get(name, 0.5)
                log_odds += math.log(m / u) if agree else math.log((1.0 - m) / (1.0 - u))
            score = _sigmoid(log_odds)

            yield PairwiseJudgement(
                left_id=candidate.left.id,  # type: ignore[attr-defined]
                right_id=candidate.right.id,  # type: ignore[attr-defined]
                score=score,
                score_type="prob_fs",
                decision_step="fellegi_sunter_em",
                provenance={"pattern": pattern, "log_odds": log_odds},
            )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        """Explore scores without ground truth (shared Module utility)."""
        return _inspect_scores_impl(judgements, sample_size)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @property
    def config(self) -> dict[str, object]:
        """Serializable config: construction args + fitted state (plain JSON).

        The fitted m/u/prior are small dicts of floats -- unlike a compiled
        DSPy program, there is no out-of-band state file needed; everything
        round-trips through this plain ``dict``.
        """
        return {
            "comparator_config": self.comparator.config,
            "agreement_threshold": self.agreement_threshold,
            "n_random_pairs": self.n_random_pairs,
            "max_em_iter": self.max_em_iter,
            "tol": self.tol,
            "random_state": self.random_state,
            "prior": self.prior,
            "m_prob": self.m_prob,
            "u_prob": self.u_prob,
            "converged": self.converged,
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "FellegiSunterJudge[SchemaT]":
        """Reconstruct from :attr:`config`, restoring any fitted state."""
        comparator: StringComparator[SchemaT] = StringComparator.from_config(
            cast("dict[str, object]", config["comparator_config"])
        )
        judge = cls(
            comparator=comparator,
            agreement_threshold=cast("float", config["agreement_threshold"]),
            n_random_pairs=cast("int", config["n_random_pairs"]),
            max_em_iter=cast("int", config["max_em_iter"]),
            tol=cast("float", config["tol"]),
            random_state=cast("int", config["random_state"]),
        )
        judge.prior = cast("float | None", config["prior"])
        judge.m_prob = cast("dict[str, float] | None", config["m_prob"])
        judge.u_prob = cast("dict[str, float] | None", config["u_prob"])
        judge.converged = cast("bool | None", config["converged"])
        return judge
