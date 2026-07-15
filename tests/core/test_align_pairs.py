"""Tests for align_pairs: the id-join bridge (W1.x, PR-B correctness gate).

align_pairs joins id-keyed labels to blocked candidates, emits
positionally-aligned (candidates, labels) for SupervisedFitMixin.fit, splits
entity-disjointly (NOT row-random -- a row-random split leaks entities across
train/valid and inflates held-out metrics), and reports a GoldCoverage
guardrail surfacing blocker-dropped positives.

These are pure, dependency-light unit tests: candidates are constructed
directly (no blocker/comparator needed), so nothing here pulls sklearn/torch.
"""

from __future__ import annotations

import pytest

from langres.core.harvest import (
    AlignedPairs,
    Correction,
    CorrectionLog,
    GoldCoverage,
    LabeledPair,
    align_pairs,
)
from langres.core.models import CompanySchema, ERCandidate


def _cand(left_id: str, right_id: str, *, left_name: str | None = None,
          right_name: str | None = None) -> ERCandidate[CompanySchema]:
    """A minimal candidate keyed by (left_id, right_id)."""
    return ERCandidate(
        left=CompanySchema(id=left_id, name=left_name or left_id),
        right=CompanySchema(id=right_id, name=right_name or right_id),
        blocker_name="test",
    )


def _labeled(left_id: str, right_id: str, label: bool) -> LabeledPair:
    return LabeledPair(left_id=left_id, right_id=right_id, score=None, label=label, source="correction")


# ---------------------------------------------------------------------------
# Gold coverage: blocker-dropped positives are surfaced, not silently lost.
# ---------------------------------------------------------------------------


def test_blocker_miss_drops_coverage_below_one_and_names_the_pair() -> None:
    """A positive label whose pair the blocker never proposed -> coverage < 1.0."""
    candidates = [_cand("a", "b"), _cand("c", "d")]
    labels = [_labeled("a", "b", True), _labeled("x", "y", True)]  # (x,y) has no candidate

    aligned = align_pairs(candidates, labels)

    assert isinstance(aligned, AlignedPairs)
    assert isinstance(aligned.coverage, GoldCoverage)
    assert aligned.coverage.gold_coverage == pytest.approx(0.5)  # 1 of 2 positives kept
    assert aligned.coverage.dropped_positives == [("x", "y")]
    assert aligned.coverage.n_positive_labels == 2


def test_full_coverage_when_every_positive_has_a_candidate() -> None:
    candidates = [_cand("a", "b"), _cand("c", "d")]
    labels = [_labeled("a", "b", True), _labeled("c", "d", True)]

    aligned = align_pairs(candidates, labels)

    assert aligned.coverage.gold_coverage == pytest.approx(1.0)
    assert aligned.coverage.dropped_positives == []


def test_coverage_is_one_when_there_are_no_positive_labels() -> None:
    """No positives -> nothing to miss -> coverage 1.0 (not 0.0 from an empty ratio)."""
    candidates = [_cand("a", "b")]
    labels = [_labeled("a", "b", False)]

    aligned = align_pairs(candidates, labels)

    assert aligned.coverage.gold_coverage == pytest.approx(1.0)
    assert aligned.coverage.n_positive_labels == 0


def test_negative_labels_do_not_count_toward_coverage() -> None:
    """A dropped *negative* label is not a coverage miss (only positives matter)."""
    candidates = [_cand("a", "b")]
    labels = [_labeled("a", "b", True), _labeled("x", "y", False)]  # (x,y) negative, no candidate

    aligned = align_pairs(candidates, labels)

    assert aligned.coverage.gold_coverage == pytest.approx(1.0)
    assert aligned.coverage.dropped_positives == []
    assert aligned.coverage.n_labeled == 2  # both labels counted for the total


# ---------------------------------------------------------------------------
# Order-independent id-join + duplicate / conflicting labels.
# ---------------------------------------------------------------------------


def test_id_join_is_order_independent() -> None:
    """A candidate (a,b) matches a label keyed (b,a) -- unordered frozenset join."""
    candidates = [_cand("a", "b")]
    labels = [_labeled("b", "a", True)]

    aligned = align_pairs(candidates, labels)

    assert len(aligned.train.candidates) == 1
    assert aligned.train.labels == [True]
    assert aligned.coverage.n_aligned == 1


def test_duplicate_label_for_same_pair_is_deduplicated() -> None:
    candidates = [_cand("a", "b")]
    labels = [_labeled("a", "b", True), _labeled("a", "b", True)]

    aligned = align_pairs(candidates, labels)

    assert aligned.coverage.n_labeled == 1
    assert aligned.train.labels == [True]


def test_conflicting_label_is_last_write_wins() -> None:
    """Same pair labeled both ways (order flipped) -> deterministic last-write-wins."""
    candidates = [_cand("a", "b")]
    labels = [_labeled("a", "b", True), _labeled("b", "a", False)]  # last one (False) wins

    aligned = align_pairs(candidates, labels)

    assert aligned.coverage.n_labeled == 1
    assert aligned.train.labels == [False]


def test_self_referential_label_does_not_crash_and_matches_nothing() -> None:
    """A degenerate (a,a) label is harmless: it can never match a real candidate."""
    candidates = [_cand("a", "b")]
    labels = [_labeled("a", "a", True), _labeled("a", "b", True)]

    aligned = align_pairs(candidates, labels)

    assert aligned.train.labels == [True]  # only (a,b) aligns
    assert aligned.coverage.dropped_positives == []  # (a,a) is not a real positive pair


# ---------------------------------------------------------------------------
# Entity-disjoint split (NOT row-random).
# ---------------------------------------------------------------------------


def _two_component_setup() -> tuple[list[ERCandidate[CompanySchema]], list[LabeledPair]]:
    # Component 1: {a,b,c} via pairs (a,b),(b,c). Component 2: {d,e,f} via (d,e),(e,f).
    candidates = [_cand("a", "b"), _cand("b", "c"), _cand("d", "e"), _cand("e", "f")]
    labels = [
        _labeled("a", "b", True),
        _labeled("b", "c", False),
        _labeled("d", "e", True),
        _labeled("e", "f", False),
    ]
    return candidates, labels


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
def test_split_is_entity_disjoint_no_id_in_both_train_and_valid(seed: int) -> None:
    """The subtle trap: no entity id may appear in both train and valid."""
    candidates, labels = _two_component_setup()

    aligned = align_pairs(candidates, labels, split=0.5, seed=seed)

    train_ids = {c.left.id for c in aligned.train.candidates} | {
        c.right.id for c in aligned.train.candidates
    }
    valid_ids = {c.left.id for c in aligned.valid.candidates} | {
        c.right.id for c in aligned.valid.candidates
    }
    assert train_ids and valid_ids  # both non-empty for this 2-component, 0.5 split
    assert train_ids.isdisjoint(valid_ids)
    # Every aligned pair lands in exactly one split.
    assert len(aligned.train.candidates) + len(aligned.valid.candidates) == len(candidates)


def test_split_keeps_whole_components_together() -> None:
    """A whole entity-component goes to one side (the mechanism behind disjointness)."""
    candidates, labels = _two_component_setup()

    aligned = align_pairs(candidates, labels, split=0.5, seed=0)

    # Each split holds a whole 3-id component -> exactly 2 aligned pairs each.
    assert len(aligned.train.candidates) == 2
    assert len(aligned.valid.candidates) == 2


def test_split_is_deterministic_for_a_fixed_seed() -> None:
    candidates, labels = _two_component_setup()

    a1 = align_pairs(candidates, labels, split=0.5, seed=3)
    a2 = align_pairs(candidates, labels, split=0.5, seed=3)

    assert [c.left.id for c in a1.valid.candidates] == [c.left.id for c in a2.valid.candidates]


def test_single_connected_component_cannot_split_keeps_train_nonempty() -> None:
    """All labeled entities connected -> valid empty, train keeps everything (never empties train)."""
    # (a,b),(b,c),(c,d) chain everything into one component.
    candidates = [_cand("a", "b"), _cand("b", "c"), _cand("c", "d")]
    labels = [_labeled("a", "b", True), _labeled("b", "c", False), _labeled("c", "d", True)]

    aligned = align_pairs(candidates, labels, split=0.5, seed=0)

    assert len(aligned.train.candidates) == 3  # everything stays in train
    assert aligned.valid.candidates == []  # no entity-disjoint valid is possible


def test_split_none_gives_empty_valid_and_all_train() -> None:
    candidates = [_cand("a", "b"), _cand("c", "d")]
    labels = [_labeled("a", "b", True), _labeled("c", "d", False)]

    aligned = align_pairs(candidates, labels)  # split defaults to None

    assert aligned.valid.candidates == []
    assert aligned.valid.labels == []
    assert len(aligned.train.candidates) == 2
    assert aligned.labels == aligned.train.labels  # .labels convenience == train labels


@pytest.mark.parametrize("bad_split", [0.0, 1.0, -0.1, 1.5])
def test_split_out_of_range_raises(bad_split: float) -> None:
    candidates = [_cand("a", "b")]
    labels = [_labeled("a", "b", True)]

    with pytest.raises(ValueError, match="split must be in the open interval"):
        align_pairs(candidates, labels, split=bad_split)


# ---------------------------------------------------------------------------
# Path-vs-in-memory equivalence + Correction input.
# ---------------------------------------------------------------------------


def test_path_and_in_memory_labels_give_identical_alignment(tmp_path) -> None:
    """A corrections.jsonl path and the equivalent in-memory Corrections align identically."""
    candidates = [_cand("a", "b"), _cand("c", "d"), _cand("e", "f")]
    corrections = [
        Correction(left_id="a", right_id="b", label=True),
        Correction(left_id="d", right_id="c", label=False),  # order flipped on purpose
        Correction(left_id="e", right_id="f", label=True),
    ]

    path = tmp_path / "corrections.jsonl"
    log = CorrectionLog(path)
    for c in corrections:
        log.append(c)

    from_path = align_pairs(candidates, path, split=0.5, seed=1)
    from_memory = align_pairs(candidates, corrections, split=0.5, seed=1)

    assert [c.left.id for c in from_path.train.candidates] == [
        c.left.id for c in from_memory.train.candidates
    ]
    assert from_path.train.labels == from_memory.train.labels
    assert [c.left.id for c in from_path.valid.candidates] == [
        c.left.id for c in from_memory.valid.candidates
    ]
    assert from_path.coverage.model_dump() == from_memory.coverage.model_dump()


def test_correction_input_is_accepted_like_labeled_pair() -> None:
    candidates = [_cand("a", "b")]
    corrections = [Correction(left_id="a", right_id="b", label=True)]

    aligned = align_pairs(candidates, corrections)

    assert aligned.train.labels == [True]
    assert aligned.coverage.n_aligned == 1


def test_unlabeled_candidates_are_dropped_from_the_fit_set() -> None:
    """Candidates with no matching label are not part of the training set."""
    candidates = [_cand("a", "b"), _cand("c", "d")]  # (c,d) is unlabeled
    labels = [_labeled("a", "b", True)]

    aligned = align_pairs(candidates, labels)

    assert len(aligned.train.candidates) == 1
    assert aligned.train.candidates[0].left.id == "a"
