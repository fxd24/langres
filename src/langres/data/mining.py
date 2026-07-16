"""Training-pair miners: the data-preparation substrate for the training wave.

A leaf module of small, composable functions over the pair currency
:data:`~langres.core.finetune.LabeledCandidate` (``(ERCandidate, is_match)``) --
the working shape both the ``fit`` surface and ``finetune`` consume. Each miner
takes labeled candidates in and returns labeled candidates out, so they compose
freely and a Wave-B harness can chain them (mine hard positives, balance the
negatives, augment, denoise) before one comparator pass over the assembled set.

Two miners are *featurizing* -- :func:`mine_misclassified_pairs` and
:func:`denoise_pairs` train an out-of-fold RandomForest to find, respectively,
the AnyMatch-style **hard positives** and the confident-learning **label noise**.
They featurize each candidate with the *same* logic the served
:class:`~langres.core.matchers.random_forest_judge.RandomForestMatcher` uses
(a throwaway matcher is the single source of truth, so there is no drift), which
means they require a ``comparison`` vector on every candidate and raise a clear
:class:`ValueError` otherwise. The remaining three
(:func:`sample_negative_pairs`, :func:`augment_by_attribute`, :func:`flip_pairs`)
are pure record/label transforms and need no model.

**Import-light by construction.** Module scope is numpy + stdlib + core data
contracts only; scikit-learn (the ``[trained]`` extra) is imported **lazily
inside** the two featurizing miners, never at module load -- so ``import
langres`` / ``import langres.data`` stay sklearn-free (locked by
``tests/test_import_budget.py`` and a guard in ``tests/data/test_mining.py``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from langres.core.feature import FeatureSpec
from langres.core.finetune import LabeledCandidate

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)

#: The clear error every featurizing miner raises when a candidate carries no
#: comparison vector -- mirrors ``RandomForestMatcher._feature_vector``'s
#: behaviour, but names the mining-side fix (run a comparator pass first).
_NO_COMPARISON_MSG = (
    "mining requires candidates carrying a comparison vector, but a candidate "
    "had comparison=None. Run a Comparator pass first (e.g. "
    "Resolver.from_schema(Schema).candidates(records)) so every candidate "
    "carries a ComparisonVector, then mine."
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _class_counts(labeled: Sequence[LabeledCandidate]) -> tuple[int, int]:
    """``(n_positive, n_negative)`` over the labels."""
    n_positive = sum(1 for _, label in labeled if label)
    return n_positive, len(labeled) - n_positive


def _require_both_classes(labeled: Sequence[LabeledCandidate], *, fn: str) -> tuple[int, int]:
    """Guard: a featurizing miner needs both a positive and a negative example.

    A single-class (or empty) input cannot train a discriminative fold, so the
    RandomForest would learn nothing -- fail loudly rather than return a
    misleading empty/degenerate result.
    """
    n_positive, n_negative = _class_counts(labeled)
    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            f"{fn} needs both positive and negative labeled pairs to train an "
            f"out-of-fold classifier; got {n_positive} positive and "
            f"{n_negative} negative."
        )
    return n_positive, n_negative


def _resolve_feature_specs(
    labeled: Sequence[LabeledCandidate],
    feature_specs: Sequence[FeatureSpec] | None,
) -> list[FeatureSpec]:
    """The feature set to featurize with: explicit, or derived from the comparisons.

    When ``feature_specs`` is ``None``, derive one :class:`FeatureSpec` per
    feature name in the union of every candidate's ``comparison.levels`` keys
    (sorted for determinism) -- so featurization spans exactly the features the
    comparator declared, mirroring what a served matcher fit on the same
    candidates would see. Raises when a candidate carries no comparison.
    """
    if feature_specs is not None:
        return list(feature_specs)
    names: set[str] = set()
    for candidate, _ in labeled:
        if candidate.comparison is None:
            raise ValueError(_NO_COMPARISON_MSG)
        names.update(candidate.comparison.levels)
    return [FeatureSpec(name=name) for name in sorted(names)]


def _feature_matrix(
    labeled: Sequence[LabeledCandidate],
    feature_specs: Sequence[FeatureSpec],
) -> list[list[float]]:
    """Featurize candidates exactly as ``RandomForestMatcher`` serves them.

    A throwaway ``RandomForestMatcher`` is the single source of truth for the
    feature vector (mirrors ``finetune._render_conversation``'s throwaway
    ``LLMMatcher``), so the mined-on features cannot drift from the served ones:
    each row is ``[similarities.get(spec.name, -1.0) for spec in feature_specs]``
    with the ``-1.0`` MISSING sentinel. Pre-checks for a missing comparison so
    the error names the mining fix rather than the pipeline one.
    """
    for candidate, _ in labeled:
        if candidate.comparison is None:
            raise ValueError(_NO_COMPARISON_MSG)
    # Lazy: importing random_forest_judge pulls scikit-learn ([trained]); keep it
    # out of module load so ``import langres.data`` stays import-light.
    from langres.core.matchers.random_forest_judge import RandomForestMatcher

    featurizer: RandomForestMatcher[Any] = RandomForestMatcher(feature_specs=list(feature_specs))
    return [featurizer._feature_vector(candidate) for candidate, _ in labeled]


def _oof_predictions(
    x: list[list[float]],
    y: list[int],
    *,
    cv: int,
    seed: int,
    method: str,
) -> npt.NDArray[Any] | None:
    """Out-of-fold RandomForest predictions; ``None`` when a fold can't hold both classes.

    A model fit to completion has ~zero *in-sample* error, so mining on in-sample
    predictions finds nothing -- these MUST be out-of-fold. Uses stratified
    ``cross_val_predict`` with ``n_splits = min(cv, n_positive, n_negative)`` so no
    fold is single-class; when that floor drops below 2 (a minority class of one)
    out-of-fold prediction is impossible, so we return ``None`` and let the caller
    degrade gracefully. ``method`` is ``"predict"`` (0/1 labels) or
    ``"predict_proba"`` (an ``(n, 2)`` probability matrix).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict

    n_positive = sum(y)
    n_negative = len(y) - n_positive
    n_splits = min(cv, n_positive, n_negative)
    if n_splits < 2:
        return None
    classifier = RandomForestClassifier(random_state=seed)
    predictions = cross_val_predict(classifier, x, y, cv=n_splits, method=method)
    return np.asarray(predictions)


# ---------------------------------------------------------------------------
# Featurizing miners (scikit-learn, lazy)
# ---------------------------------------------------------------------------


def mine_misclassified_pairs(
    labeled: Sequence[LabeledCandidate],
    *,
    cap: int = 400,
    feature_specs: Sequence[FeatureSpec] | None = None,
    cv: int = 5,
    seed: int = 0,
) -> list[LabeledCandidate]:
    """Mine AnyMatch-style **hard positives**: positives an RF gets wrong out-of-fold.

    Trains a RandomForest and takes its *out-of-fold* predictions (a fit-to-
    completion forest has ~0 in-sample error, so in-sample mining returns nothing).
    Positives whose out-of-fold prediction is negative are the hard ones -- the
    boundary examples worth over-weighting in a training set. They are returned
    first (up to ``cap``); if there are fewer than ``cap`` hard positives, the set
    is topped up with correctly-classified positives so the caller gets a stable
    ``min(cap, n_positive)`` positives to train on.

    Args:
        labeled: Comparison-attached labeled candidates. Must contain both classes.
        cap: Maximum positives to return.
        feature_specs: Features to featurize with; when ``None``, derived from the
            union of the candidates' comparison levels (see module docs).
        cv: Requested number of out-of-fold folds (reduced automatically when a
            class is too small to stratify; when the minority class has a single
            member, out-of-fold prediction is skipped and the first ``cap``
            positives are returned unranked, logged).
        seed: RandomForest seed (deterministic folds/fit).

    Returns:
        Up to ``cap`` positive :data:`LabeledCandidate`\\ s, hard ones first.

    Raises:
        ValueError: If the input lacks either class, or a candidate has no
            comparison vector.
    """
    materialized = list(labeled)
    _require_both_classes(materialized, fn="mine_misclassified_pairs")
    specs = _resolve_feature_specs(materialized, feature_specs)
    x = _feature_matrix(materialized, specs)
    y = [int(label) for _, label in materialized]

    positives = [i for i, label in enumerate(y) if label == 1]
    predictions = _oof_predictions(x, y, cv=cv, seed=seed, method="predict")
    if predictions is None:
        logger.warning(
            "mine_misclassified_pairs: minority class too small for %d-fold "
            "out-of-fold prediction; returning the first %d positives unranked "
            "(cannot identify hard positives).",
            cv,
            min(cap, len(positives)),
        )
        return [materialized[i] for i in positives[:cap]]

    hard = [i for i in positives if int(predictions[i]) == 0]
    easy = [i for i in positives if int(predictions[i]) == 1]
    selected = hard[:cap]
    if len(selected) < cap:
        selected = selected + easy[: cap - len(selected)]
    logger.info(
        "mine_misclassified_pairs: %d hard positive(s) of %d positives; returning %d (cap=%d).",
        len(hard),
        len(positives),
        len(selected),
        cap,
    )
    return [materialized[i] for i in selected]


def denoise_pairs(
    labeled: Sequence[LabeledCandidate],
    *,
    method: str = "confident_learning",
    feature_specs: Sequence[FeatureSpec] | None = None,
    cv: int = 5,
    seed: int = 0,
) -> tuple[list[LabeledCandidate], list[LabeledCandidate]]:
    """Flag likely-mislabeled pairs via confident learning; return ``(clean, flagged)``.

    Trains an out-of-fold RandomForest for predicted class probabilities, then
    applies the Northcutt et al. *confident learning* rule: for each class, the
    self-confidence threshold is the mean predicted probability of that class over
    the examples labeled with it; a pair is flagged when it is *confidently*
    assigned (its probability clears the threshold) to a class different from its
    given label. Robust to class imbalance because the threshold is per-class, not
    a global 0.5.

    Caveat -- confident learning cannot perfectly separate label noise from
    *genuinely hard* positives (both look alike to the model), so it may flag a
    real hard positive as suspected noise. Running :func:`denoise_pairs` *before*
    :func:`mine_misclassified_pairs` therefore competes with hard-positive mining:
    it can strip the very boundary positives mining targets. Order accordingly.

    Args:
        labeled: Comparison-attached labeled candidates. Must contain both classes.
        method: Denoise strategy; only ``"confident_learning"`` is supported.
        feature_specs: Features to featurize with; when ``None``, derived from the
            union of the candidates' comparison levels (see module docs).
        cv: Requested out-of-fold folds (reduced when a class is small; when a
            fold cannot hold both classes, nothing is flagged and the split is
            ``(all, [])``, logged).
        seed: RandomForest seed.

    Returns:
        ``(clean, flagged)`` -- the pairs kept as-is and the pairs whose label the
        confident joint disputes, both preserving input order.

    Raises:
        ValueError: For an unknown ``method``, an input lacking either class, or a
            candidate with no comparison vector.
    """
    if method != "confident_learning":
        raise ValueError(
            f"unknown denoise method {method!r}; only 'confident_learning' is supported."
        )
    materialized = list(labeled)
    _require_both_classes(materialized, fn="denoise_pairs")
    specs = _resolve_feature_specs(materialized, feature_specs=feature_specs)
    x = _feature_matrix(materialized, specs)
    y = [int(label) for _, label in materialized]

    proba = _oof_predictions(x, y, cv=cv, seed=seed, method="predict_proba")
    if proba is None:
        logger.warning(
            "denoise_pairs: minority class too small for %d-fold out-of-fold "
            "probabilities; flagging nothing.",
            cv,
        )
        return list(materialized), []

    flagged = _confident_label_errors(np.asarray(proba, dtype=float), y)
    clean = [materialized[i] for i in range(len(materialized)) if i not in flagged]
    flagged_pairs = [materialized[i] for i in sorted(flagged)]
    logger.info(
        "denoise_pairs: flagged %d of %d pairs as likely mislabeled (confident learning).",
        len(flagged),
        len(materialized),
    )
    return clean, flagged_pairs


def _confident_label_errors(proba: npt.NDArray[Any], y: list[int]) -> set[int]:
    """Indices confidently assigned to a class other than their label (confident joint).

    ``proba`` is the out-of-fold ``(n, 2)`` probability matrix; column ``j`` is the
    model's probability of class ``j``. ``y`` is the given binary label per row.
    Both classes are guaranteed present by the caller, so each per-class threshold
    averages over a non-empty set (no divide-by-zero).
    """
    n = len(y)
    thresholds = [float(np.mean([proba[i, j] for i in range(n) if y[i] == j])) for j in (0, 1)]
    flagged: set[int] = set()
    for i in range(n):
        eligible = [j for j in (0, 1) if proba[i, j] >= thresholds[j]]
        if not eligible:
            continue  # unconfident about either class -> trust the given label
        suggested = max(eligible, key=lambda j: proba[i, j])
        if suggested != y[i]:
            flagged.add(i)
    return flagged


# ---------------------------------------------------------------------------
# Pure record/label transforms (no model)
# ---------------------------------------------------------------------------


def sample_negative_pairs(
    labeled: Sequence[LabeledCandidate],
    *,
    ratio: float = 2.0,
    seed: int = 0,
) -> list[LabeledCandidate]:
    """Seeded down-sample of negatives to ``ratio × (#positives)`` -- the AnyMatch balance lever.

    Returns a reproducible random subset of the negative-labeled pairs sized to
    ``round(ratio × n_positive)``. When fewer negatives exist than the target, all
    of them are returned and the shortfall is logged (never silent). The sampled
    subset keeps input order for determinism.

    Args:
        labeled: Labeled candidates (positives are counted; negatives are sampled).
        ratio: Negatives-per-positive target (AnyMatch uses ``2.0``).
        seed: Sampling seed.

    Returns:
        The sampled negative :data:`LabeledCandidate`\\ s.
    """
    materialized = list(labeled)
    n_positive = sum(1 for _, label in materialized if label)
    negatives = [pair for pair in materialized if not pair[1]]
    target = int(round(ratio * n_positive))

    if target >= len(negatives):
        if target > len(negatives):
            logger.info(
                "sample_negative_pairs: only %d negative(s) available for a "
                "%.2f×%d = %d target; returning all.",
                len(negatives),
                ratio,
                n_positive,
                target,
            )
        return list(negatives)

    rng = np.random.default_rng(seed)
    chosen = sorted(int(i) for i in rng.choice(len(negatives), size=target, replace=False))
    return [negatives[i] for i in chosen]


def augment_by_attribute(
    labeled: Sequence[LabeledCandidate],
    *,
    cap: int = 800,
    seed: int = 0,
) -> list[LabeledCandidate]:
    """AnyMatch attribute augmentation: for each positive, blank one attribute at a time.

    For every positive pair, emit new positive :data:`LabeledCandidate`\\ s each
    with a single string attribute blanked (set to ``""``) on **both** records --
    teaching a matcher/finetune robustness to a missing field while the pair stays
    a genuine positive (same real-world entity). One variant per (positive,
    blankable field); the field set is the union of each side's non-empty string
    fields (excluding ``id``). Capped, seeded when it bites.

    Because the emitted candidates mutate the underlying **records**, their
    ``comparison`` is set to ``None``: a featurizing consumer MUST re-run the
    comparator over the augmented set before using it (the Wave-B harness does one
    comparator pass over the whole assembled set).

    Args:
        labeled: Labeled candidates (only the positives are augmented).
        cap: Maximum augmented pairs to return.
        seed: Sampling seed for the cap.

    Returns:
        Up to ``cap`` augmented positive :data:`LabeledCandidate`\\ s, each with
        ``comparison=None``.
    """
    positives = [candidate for candidate, label in labeled if label]
    augmented: list[LabeledCandidate] = []
    for candidate in positives:
        fields = _nonempty_string_fields(candidate.left) | _nonempty_string_fields(candidate.right)
        for field in sorted(fields):
            blanked = candidate.model_copy(
                update={
                    "left": candidate.left.model_copy(update={field: ""}),
                    "right": candidate.right.model_copy(update={field: ""}),
                    "comparison": None,
                }
            )
            augmented.append((blanked, True))

    if len(augmented) > cap:
        rng = np.random.default_rng(seed)
        keep = sorted(int(i) for i in rng.choice(len(augmented), size=cap, replace=False))
        logger.info(
            "augment_by_attribute: generated %d augmentation(s), capped to %d.",
            len(augmented),
            cap,
        )
        augmented = [augmented[i] for i in keep]
    return augmented


def _nonempty_string_fields(record: Any) -> set[str]:
    """Names of the record's non-empty string fields, excluding ``id``.

    These are the record's content attributes -- what a string comparator
    compares -- so blanking one is a meaningful "missing field" signal. ``id``
    is excluded because it is a primary key, never a content field to blank
    (a text serializer may still render it).
    """
    return {
        field
        for field, value in record.model_dump().items()
        if field != "id" and isinstance(value, str) and value.strip()
    }


def flip_pairs(pairs: Sequence[LabeledCandidate]) -> list[LabeledCandidate]:
    """Swap ``left`` <-> ``right`` on each candidate, keeping the label.

    Order-invariance augmentation, primarily for the finetune/text path (which
    renders ``left`` then ``right`` into the prompt): a matcher should judge a pair
    the same regardless of which record comes first. The existing ``comparison`` is
    **kept**: string comparison is symmetric and a ``ComparisonVector`` is keyed by
    feature name (not by side), so a flip leaves the per-feature similarities
    unchanged -- the flipped candidates stay featurizable without a re-compare.

    Orientation is preserved because these tuples feed ``Matcher.fit`` directly and
    never round-trip through ``align_pairs``' frozenset dedup (which would collapse
    a pair and its flip).

    Args:
        pairs: The labeled candidates to flip.

    Returns:
        One flipped :data:`LabeledCandidate` per input, same labels.
    """
    return [
        (candidate.model_copy(update={"left": candidate.right, "right": candidate.left}), label)
        for candidate, label in pairs
    ]
