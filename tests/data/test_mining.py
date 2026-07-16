"""Tests for the training-pair miners (:mod:`langres.data.mining`).

Covers each miner's behaviour and edges: the featurizer-parity guarantee, the
cap + top-up of hard-positive mining, the 2:1 negative-sampling lever, multi-
attribute augmentation, flip orientation/symmetry, the single-class / no-
comparison guards, and the confident-learning denoiser. The denoise gate is a
seeded synthetic-noise injection with known ground truth (recall + precision
thresholds), plus a cleanlab cross-check on the same out-of-fold probabilities.
"""

from __future__ import annotations

import random
import subprocess
import sys

import numpy as np
import pytest

from langres.core.comparators import StringComparator
from langres.core.feature import FeatureSpec
from langres.core.finetune import LabeledCandidate
from langres.core.models import CompanySchema, ERCandidate
from langres.data.mining import (
    _feature_matrix,
    _oof_predictions,
    _resolve_feature_specs,
    augment_by_attribute,
    denoise_pairs,
    flip_pairs,
    mine_misclassified_pairs,
    sample_negative_pairs,
)

_COMPARATOR: StringComparator[CompanySchema] = StringComparator.from_schema(CompanySchema)


def _company(id: str, name: str, address: str | None = None) -> CompanySchema:
    return CompanySchema(id=id, name=name, address=address)


def _compared(left: CompanySchema, right: CompanySchema) -> ERCandidate[CompanySchema]:
    """A candidate with its comparison vector attached (what a Comparator pass yields)."""
    candidate = ERCandidate(left=left, right=right, blocker_name="test")
    return candidate.model_copy(update={"comparison": _COMPARATOR.compare(left, right)})


def _bare(left: CompanySchema, right: CompanySchema) -> ERCandidate[CompanySchema]:
    """A candidate WITHOUT a comparison vector (blocker-only, no Comparator pass)."""
    return ERCandidate(left=left, right=right, blocker_name="test")


def _separable_dataset(n_positive: int = 30, n_negative: int = 30) -> list[LabeledCandidate]:
    """A cleanly separable labeled set: matches share strings, non-matches don't."""
    labeled: list[LabeledCandidate] = []
    for i in range(n_positive):
        left = _company(f"m{i}L", f"Acme Corporation {i}", f"{i} Main Street")
        right = _company(f"m{i}R", f"Acme Corporation {i}", f"{i} Main Street")
        labeled.append((_compared(left, right), True))
    for i in range(n_negative):
        left = _company(f"n{i}L", f"Zephyr Holdings {i}", f"{i} Ocean Avenue")
        right = _company(f"n{i}R", f"Quasar Industries {i}", f"{i} Mountain Road")
        labeled.append((_compared(left, right), False))
    return labeled


# ---------------------------------------------------------------------------
# Featurization parity + resolution
# ---------------------------------------------------------------------------


class TestFeaturization:
    def test_feature_matrix_matches_random_forest_matcher(self) -> None:
        """The X the miners build is byte-identical to RandomForestMatcher._feature_vector."""
        from langres.core.matchers.random_forest_judge import RandomForestMatcher

        labeled = _separable_dataset(n_positive=5, n_negative=5)
        specs = _resolve_feature_specs(labeled, None)
        x = _feature_matrix(labeled, specs)

        matcher: RandomForestMatcher[CompanySchema] = RandomForestMatcher(feature_specs=specs)
        x_rf = [matcher._feature_vector(candidate) for candidate, _ in labeled]
        assert x == x_rf

    def test_derived_specs_span_comparison_levels(self) -> None:
        """With feature_specs=None, specs are the sorted union of comparison levels."""
        labeled = _separable_dataset(n_positive=2, n_negative=2)
        names = [spec.name for spec in _resolve_feature_specs(labeled, None)]
        # CompanySchema's comparable string fields, id excluded, sorted.
        assert names == ["address", "name", "phone", "website"]

    def test_missing_feature_uses_minus_one_sentinel(self) -> None:
        """A MISSING feature (absent from similarities) featurizes to the -1.0 sentinel."""
        # address is None on the left -> MISSING -> not in similarities.
        candidate = _compared(_company("a", "Acme", None), _company("b", "Acme", "1 St"))
        specs = [FeatureSpec(name="address")]
        assert _feature_matrix([(candidate, True)], specs) == [[-1.0]]

    def test_explicit_feature_specs_are_honoured(self) -> None:
        """Passing feature_specs pins the featurization order/columns."""
        labeled = _separable_dataset(n_positive=2, n_negative=2)
        specs = [FeatureSpec(name="name")]
        x = _feature_matrix(labeled, specs)
        assert all(len(row) == 1 for row in x)

    def test_resolve_feature_specs_passes_explicit_through(self) -> None:
        """An explicit spec list is used verbatim (comparisons are not consulted)."""
        specs = [FeatureSpec(name="name"), FeatureSpec(name="phone")]
        assert _resolve_feature_specs([], specs) == specs

    def test_feature_matrix_missing_comparison_with_explicit_specs_raises(self) -> None:
        """The _feature_matrix guard fires even when specs are explicit (no derive step)."""
        bare = (_bare(_company("a", "Acme"), _company("b", "Acme")), True)
        with pytest.raises(ValueError, match="comparison"):
            _feature_matrix([bare], [FeatureSpec(name="name")])


# ---------------------------------------------------------------------------
# mine_misclassified_pairs
# ---------------------------------------------------------------------------


class TestMineMisclassified:
    def test_returns_only_positives_capped(self) -> None:
        labeled = _separable_dataset(n_positive=20, n_negative=20)
        mined = mine_misclassified_pairs(labeled, cap=5, cv=4)
        assert len(mined) == 5
        assert all(label for _, label in mined)

    def test_surfaces_injected_hard_positives_first(self) -> None:
        """Positives that look like non-matches are surfaced before easy positives."""
        labeled = _separable_dataset(n_positive=25, n_negative=25)
        # Hard positives: label True but strings look like a non-match (low similarity).
        hard_ids = set()
        for i in range(3):
            left = _company(f"hard{i}L", f"Aardvark Systems {i}", f"{i} Alpha Way")
            right = _company(f"hard{i}R", f"Zenith Partners {i}", f"{i} Omega Blvd")
            labeled.append((_compared(left, right), True))
            hard_ids.add(f"hard{i}L")

        mined = mine_misclassified_pairs(labeled, cap=3, cv=4)
        assert {candidate.left.id for candidate, _ in mined} == hard_ids

    def test_tops_up_with_easy_positives_when_few_hard(self) -> None:
        """Fewer hard positives than cap -> topped up to min(cap, n_positive)."""
        labeled = _separable_dataset(n_positive=15, n_negative=15)
        mined = mine_misclassified_pairs(labeled, cap=10, cv=4)
        assert len(mined) == 10  # separable: ~0 hard, filled with easy positives

    def test_cap_larger_than_positives_returns_all_positives(self) -> None:
        labeled = _separable_dataset(n_positive=8, n_negative=8)
        mined = mine_misclassified_pairs(labeled, cap=100, cv=4)
        assert len(mined) == 8

    def test_single_class_input_raises(self) -> None:
        labeled = _separable_dataset(n_positive=10, n_negative=0)
        with pytest.raises(ValueError, match="both positive and negative"):
            mine_misclassified_pairs(labeled)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="both positive and negative"):
            mine_misclassified_pairs([])

    def test_missing_comparison_raises(self) -> None:
        labeled: list[LabeledCandidate] = [
            (_bare(_company("a", "Acme"), _company("b", "Acme")), True),
            (_bare(_company("c", "Zed"), _company("d", "Yak")), False),
        ]
        with pytest.raises(ValueError, match="comparison"):
            mine_misclassified_pairs(labeled)

    def test_tiny_minority_class_degrades_without_oof(self) -> None:
        """A minority class of one can't be stratified -> unranked fallback, no raise."""
        labeled = _separable_dataset(n_positive=1, n_negative=10)
        mined = mine_misclassified_pairs(labeled, cap=5, cv=5)
        assert len(mined) == 1
        assert all(label for _, label in mined)


# ---------------------------------------------------------------------------
# sample_negative_pairs
# ---------------------------------------------------------------------------


class TestSampleNegatives:
    def test_two_to_one_ratio_exact(self) -> None:
        labeled = _separable_dataset(n_positive=10, n_negative=40)
        sampled = sample_negative_pairs(labeled, ratio=2.0, seed=0)
        assert len(sampled) == 20
        assert all(not label for _, label in sampled)

    def test_shortfall_returns_all_negatives(self) -> None:
        labeled = _separable_dataset(n_positive=10, n_negative=5)
        sampled = sample_negative_pairs(labeled, ratio=2.0)  # wants 20, has 5
        assert len(sampled) == 5

    def test_seeded_determinism(self) -> None:
        labeled = _separable_dataset(n_positive=10, n_negative=40)
        first = sample_negative_pairs(labeled, ratio=2.0, seed=7)
        second = sample_negative_pairs(labeled, ratio=2.0, seed=7)
        assert [c.left.id for c, _ in first] == [c.left.id for c, _ in second]

    def test_no_positives_samples_nothing(self) -> None:
        labeled = _separable_dataset(n_positive=0, n_negative=10)
        assert sample_negative_pairs(labeled, ratio=2.0) == []

    def test_target_equal_to_available_returns_all(self) -> None:
        """Exactly enough negatives (target == available): return all, no shortfall log."""
        labeled = _separable_dataset(n_positive=10, n_negative=20)
        sampled = sample_negative_pairs(labeled, ratio=2.0)  # target 20 == 20
        assert len(sampled) == 20


# ---------------------------------------------------------------------------
# augment_by_attribute
# ---------------------------------------------------------------------------


class TestAugmentByAttribute:
    def test_blanks_each_string_field_of_each_positive(self) -> None:
        """One variant per (positive, non-empty string field); comparison reset to None."""
        left = _company("a", "Acme", "1 Main St")
        right = _company("b", "Acme", "1 Main St")
        labeled: list[LabeledCandidate] = [(_compared(left, right), True)]
        augmented = augment_by_attribute(labeled)
        # name + address are the non-empty string fields -> 2 variants.
        assert len(augmented) == 2
        assert all(label for _, label in augmented)
        assert all(candidate.comparison is None for candidate, _ in augmented)

    def test_blanked_field_is_emptied_on_both_sides(self) -> None:
        left = _company("a", "Acme", "1 Main St")
        right = _company("b", "Acme", "1 Main St")
        augmented = augment_by_attribute([(_compared(left, right), True)])
        blanked_fields = set()
        for candidate, _ in augmented:
            for field in ("name", "address"):
                if getattr(candidate.left, field) == "":
                    assert getattr(candidate.right, field) == ""
                    blanked_fields.add(field)
        assert blanked_fields == {"name", "address"}

    def test_cap_bounds_output(self) -> None:
        labeled = _separable_dataset(n_positive=20, n_negative=5)
        augmented = augment_by_attribute(labeled, cap=7)
        assert len(augmented) == 7

    def test_no_positives_yields_nothing(self) -> None:
        labeled = _separable_dataset(n_positive=0, n_negative=5)
        assert augment_by_attribute(labeled) == []


# ---------------------------------------------------------------------------
# flip_pairs
# ---------------------------------------------------------------------------


class TestFlipPairs:
    def test_swaps_left_right_keeps_label_and_comparison(self) -> None:
        left = _company("a", "Acme", "1 Main St")
        right = _company("b", "Acme Inc", "1 Main Street")
        original = _compared(left, right)
        ((flipped, label),) = flip_pairs([(original, True)])
        assert flipped.left.id == "b" and flipped.right.id == "a"
        assert label is True
        # Comparison kept: string comparison is symmetric and feature-keyed.
        assert flipped.comparison is not None
        assert flipped.comparison.similarities == original.comparison.similarities

    def test_flip_of_flip_restores_orientation(self) -> None:
        labeled = _separable_dataset(n_positive=2, n_negative=2)
        once = flip_pairs(labeled)
        twice = flip_pairs(once)
        assert [c.left.id for c, _ in twice] == [c.left.id for c, _ in labeled]

    def test_empty_input(self) -> None:
        assert flip_pairs([]) == []


# ---------------------------------------------------------------------------
# denoise_pairs
# ---------------------------------------------------------------------------


def _inject_noise(
    clean: list[LabeledCandidate], k: int, seed: int
) -> tuple[list[LabeledCandidate], set[int]]:
    """Flip ``k`` known labels; return the noisy set and the flipped indices."""
    rng = random.Random(seed)
    noisy = list(clean)
    flipped = set(rng.sample(range(len(noisy)), k))
    for i in flipped:
        candidate, label = noisy[i]
        noisy[i] = (candidate, not label)
    return noisy, flipped


def _flagged_indices(noisy: list[LabeledCandidate], flagged: list[LabeledCandidate]) -> set[int]:
    """Recover the input indices of the flagged pairs by identity."""
    flagged_ids = {id(candidate) for candidate, _ in flagged}
    return {i for i, (candidate, _) in enumerate(noisy) if id(candidate) in flagged_ids}


class TestDenoisePairs:
    def test_ground_truth_noise_injection_recall_and_precision(self) -> None:
        """The hard gate: flag injected label errors with high recall + precision."""
        clean = _separable_dataset(n_positive=40, n_negative=40)
        noisy, injected = _inject_noise(clean, k=8, seed=0)

        cleaned, flagged = denoise_pairs(noisy, cv=5, seed=0)
        assert len(cleaned) + len(flagged) == len(noisy)

        found = _flagged_indices(noisy, flagged)
        true_positive = len(found & injected)
        recall = true_positive / len(injected)
        precision = true_positive / len(found) if found else 1.0
        assert recall >= 0.75, f"recall={recall}"
        assert precision >= 0.75, f"precision={precision}"

    def test_class_imbalanced_noise(self) -> None:
        """Confident learning stays honest under 1:4 class imbalance."""
        clean = _separable_dataset(n_positive=16, n_negative=64)
        noisy, injected = _inject_noise(clean, k=6, seed=3)

        _cleaned, flagged = denoise_pairs(noisy, cv=5, seed=0)
        found = _flagged_indices(noisy, flagged)
        recall = len(found & injected) / len(injected)
        assert recall >= 0.5, f"recall={recall}"

    def test_clean_data_flags_few(self) -> None:
        """A clean, separable set should flag (almost) nothing."""
        clean = _separable_dataset(n_positive=40, n_negative=40)
        _cleaned, flagged = denoise_pairs(clean, cv=5, seed=0)
        assert len(flagged) <= 2

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown denoise method"):
            denoise_pairs(_separable_dataset(4, 4), method="magic")

    def test_single_class_input_raises(self) -> None:
        with pytest.raises(ValueError, match="both positive and negative"):
            denoise_pairs(_separable_dataset(n_positive=6, n_negative=0))

    def test_missing_comparison_raises(self) -> None:
        labeled: list[LabeledCandidate] = [
            (_bare(_company("a", "Acme"), _company("b", "Acme")), True),
            (_bare(_company("c", "Zed"), _company("d", "Yak")), False),
        ]
        with pytest.raises(ValueError, match="comparison"):
            denoise_pairs(labeled)

    def test_degenerate_folds_flag_nothing(self) -> None:
        """A minority class of one can't be stratified -> (all, []), no raise."""
        labeled = _separable_dataset(n_positive=1, n_negative=10)
        cleaned, flagged = denoise_pairs(labeled, cv=5)
        assert len(cleaned) == 11
        assert flagged == []

    def test_honours_feature_specs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """feature_specs pins denoise's feature space (parity with mine_misclassified_pairs).

        Without it, a matcher trained on a custom feature SUBSET would be denoised
        in the full default feature space -- a mismatch. Spy on _feature_matrix to
        assert the explicit subset is threaded through, not the derived default.
        """
        import langres.data.mining as mining_mod

        labeled = _separable_dataset(n_positive=20, n_negative=20)
        subset = [FeatureSpec(name="name")]
        captured: list[list[FeatureSpec]] = []
        real_feature_matrix = mining_mod._feature_matrix

        def spy(labeled_arg: list[LabeledCandidate], specs: list[FeatureSpec]) -> list[list[float]]:
            captured.append(list(specs))
            return real_feature_matrix(labeled_arg, specs)

        monkeypatch.setattr(mining_mod, "_feature_matrix", spy)
        denoise_pairs(labeled, feature_specs=subset, cv=4, seed=0)
        assert captured and captured[0] == subset

    def test_agrees_with_cleanlab_on_same_probabilities(self) -> None:
        """Cross-check: our confident-joint (Northcutt et al.) core overlaps cleanlab
        on the same OOF probs -- partial overlap (jaccard >= 0.5), not byte-parity:
        we run the confident-joint core, not Cleanlab's full prune-by-noise-rate pipeline.
        """
        cleanlab_filter = pytest.importorskip("cleanlab.filter")

        clean = _separable_dataset(n_positive=40, n_negative=40)
        noisy, _injected = _inject_noise(clean, k=8, seed=0)
        _cleaned, flagged = denoise_pairs(noisy, cv=5, seed=0)
        ours = _flagged_indices(noisy, flagged)

        specs = _resolve_feature_specs(noisy, None)
        x = _feature_matrix(noisy, specs)
        y = [int(label) for _, label in noisy]
        proba = np.asarray(
            _oof_predictions(x, y, cv=5, seed=0, method="predict_proba"), dtype=float
        )
        mask = cleanlab_filter.find_label_issues(
            labels=np.asarray(y), pred_probs=proba, return_indices_ranked_by=None
        )
        cleanlab_flags = {int(i) for i in np.where(mask)[0]}

        overlap = ours & cleanlab_flags
        union = ours | cleanlab_flags
        jaccard = len(overlap) / len(union) if union else 1.0
        assert jaccard >= 0.5, f"ours={sorted(ours)} cleanlab={sorted(cleanlab_flags)}"


class TestConfidentJoint:
    def test_flags_confident_disagreement_and_skips_unconfident(self) -> None:
        """The confident-joint rule on a hand-built OOF probability matrix.

        Columns are ``[p(neg), p(pos)]``. Per-class thresholds are the mean
        self-confidence over each class's members: t0 = mean(0.85, 0.7) = 0.775,
        t1 = mean(0.1, 0.9) = 0.5. Row 0 is labeled positive but confidently
        negative (flagged); row 3 clears neither threshold (unconfident -> the
        ``continue`` branch, keeps its label).
        """
        from langres.data.mining import _confident_label_errors

        proba = np.array(
            [
                [0.9, 0.1],  # y=1, confidently neg -> flagged
                [0.1, 0.9],  # y=1, confidently pos -> kept
                [0.85, 0.15],  # y=0, confidently neg -> kept
                [0.7, 0.3],  # y=0, clears neither threshold -> unconfident, kept
            ]
        )
        assert _confident_label_errors(proba, [1, 1, 0, 0]) == {0}


# ---------------------------------------------------------------------------
# Import-light budget: sklearn must stay lazy behind the featurizing miners.
# ---------------------------------------------------------------------------


def test_importing_data_and_mining_does_not_pull_sklearn() -> None:
    """A bare ``import langres.data`` / ``import langres.data.mining`` stays sklearn-free."""
    script = (
        "import sys; import langres.data; import langres.data.mining; "
        "leaked = [m for m in ['sklearn', 'torch', 'litellm', 'faiss'] if m in sys.modules]; "
        "assert not leaked, f'mining pulled heavy modules: {leaked}'; "
        "print('OK')"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
