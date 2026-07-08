"""Tests for the WDC-computers benchmark loader + its derived seen/unseen slice.

Runs the shared loader contract (``tests/data/_loader_contract.py``) against the
factory-built ``WdcComputersBenchmark`` and checks the Wave D ``wdc_slice_map``
seen/unseen tagging. The third test wires the slice map through
``evaluate_judge_on_candidates`` end-to-end (a rapidfuzz judge over the WDC test
pairs) to prove the honest fixed-threshold sliced eval on real data. All are fast
(CSV parse + rapidfuzz over titles; no embeddings).
"""

import logging

import pytest

from langres.core.benchmark import evaluate_judge_on_candidates
from langres.core.metrics import classify_pairs
from langres.core.models import ERCandidate
from langres.core.modules.rapidfuzz import RapidfuzzModule
from langres.data.wdc_computers import (
    WdcComputersBenchmark,
    WdcComputersSchema,
    load_wdc_computers,
    load_wdc_computers_pair_splits,
    wdc_slice_map,
)
from tests.data._loader_contract import assert_loader_contract

logger = logging.getLogger(__name__)

#: Pinned as evidence (see ``datasets/wdc_computers/ATTRIBUTION.md``): 2204 tableA
#: + 2443 tableB records, and 1111 transitive-closure within-cluster gold pairs
#: (the many-to-many closed-world partition of 986 pooled positive pairs).
_N_CORPUS = 4647
_N_GOLD_PAIRS = 1111


def test_wdc_computers_satisfies_the_loader_contract() -> None:
    assert_loader_contract(
        WdcComputersBenchmark(),
        expected_corpus_size=_N_CORPUS,
        expected_gold_pairs=_N_GOLD_PAIRS,
    )


def test_wdc_slice_map_tags_test_pairs_seen_and_unseen() -> None:
    """The derived seen/unseen slice is non-empty, well-tagged, and spans the range.

    Every value is a valid tag, and the test split genuinely exhibits both
    ``"seen"`` and ``"unseen"`` pairs (measured: seen=86, half_seen=423,
    unseen=572) — the precondition Wave D's honest seen -> unseen F1-drop demo
    depends on.
    """
    tags = wdc_slice_map("test")
    assert tags, "wdc_slice_map('test') is empty"
    assert set(tags.values()) <= {"seen", "half_seen", "unseen"}, "unexpected slice tag"
    present = set(tags.values())
    assert "seen" in present, "no fully-seen test pairs (Wave D needs both endpoints)"
    assert "unseen" in present, "no fully-unseen test pairs (Wave D needs both endpoints)"


def test_wdc_sliced_judge_eval_grades_every_slice_at_one_fixed_threshold() -> None:
    """End-to-end honest seen->unseen eval: one judge, one threshold, many slices.

    Builds candidates from WDC's labelled TEST pairs, judges them with an offline
    rapidfuzz module over ``title``, and evaluates with a ``slice_fn`` closed over
    ``wdc_slice_map("test")``. The mechanism is the assertion: the single global
    best-F1 threshold is chosen once, then every slice (seen / half_seen / unseen)
    is graded at that SAME cut — reconstructed exactly via ``classify_pairs`` at
    ``result.best_threshold``. The observed per-slice F1s are reported (not forced
    into a fixed inequality: WDC is title-only and noisy).
    """
    corpus, _gold_clusters, _gold_pairs = load_wdc_computers()
    by_id: dict[str, WdcComputersSchema] = {r.id: r for r in corpus}
    test_pairs = load_wdc_computers_pair_splits()["test"]

    candidates: list[ERCandidate[WdcComputersSchema]] = [
        ERCandidate(left=by_id[left], right=by_id[right], blocker_name="wdc_test_pairs")
        for (left, right, _label) in test_pairs
    ]
    gold_pairs = {frozenset({left, right}) for (left, right, label) in test_pairs if label == 1}

    judge: RapidfuzzModule[WdcComputersSchema] = RapidfuzzModule(
        field_extractors={"title": (lambda r: r.title, 1.0)},
        algorithm="token_set_ratio",
    )
    slice_map = wdc_slice_map("test")

    def slice_fn(pair_key: frozenset[str]) -> str | None:
        return slice_map.get(pair_key)

    grid = tuple(round(0.05 * i, 2) for i in range(21))  # 0.00 .. 1.00 step 0.05
    result, judgements = evaluate_judge_on_candidates(
        judge, candidates, gold_pairs, grid, slice_fn=slice_fn
    )

    assert result.slices is not None
    assert set(result.slices) <= {"seen", "half_seen", "unseen"}
    assert len(result.slices) >= 2, "expected at least two distinct slices"

    # Mechanism: every slice is graded at the ONE global threshold. Reconstruct
    # each slice's track via classify_pairs at result.best_threshold and require
    # an exact match — a per-slice argmax would diverge here.
    candidate_pairs = {frozenset({c.left.id, c.right.id}) for c in candidates}
    gold_in_scope = gold_pairs & candidate_pairs
    for tag, track in result.slices.items():
        tag_judged = [
            j for j in judgements if slice_map.get(frozenset({j.left_id, j.right_id})) == tag
        ]
        tag_gold = {pk for pk in gold_in_scope if slice_map.get(pk) == tag}
        expected = classify_pairs(tag_judged, tag_gold, result.best_threshold)
        assert track.precision == pytest.approx(expected.precision)
        assert track.recall == pytest.approx(expected.recall)
        assert track.f1 == pytest.approx(expected.f1)
        assert track.pr_curve is None

    # Report the observed seen/half_seen/unseen F1 at the fixed threshold. The
    # slices are genuinely distinct (not one number copied across tags).
    observed = {tag: round(t.f1, 4) for tag, t in result.slices.items()}
    logger.info("WDC test sliced F1 @ fixed threshold %.2f: %s", result.best_threshold, observed)
    assert result.slices["seen"] != result.slices["unseen"]
