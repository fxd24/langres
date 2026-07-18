"""Proof that the Op-algebra expressiveness example runs and shows the reranker win.

Drives the importable core of ``examples/op_reranker_expressiveness.py``: two $0
rapidfuzz pipelines built from the additive Op adapters. The crux is the
**reranker** — a ``Score`` AFTER a ``Select``, a topology the old four-slot core
(blocker -> comparator -> matcher -> clusterer) could not place. This test pins
that the algebra both *accepts* that topology (``Sequential.check()``) and *runs*
it end-to-end to a tighter clustering than the four-slot baseline.

Guards the W3 spine flip: if the additive adapters ever stop composing without an
``ERModel``, or a ``Select`` refuses a carrier a prior ``Score`` emits, these fail.
"""

from __future__ import annotations

import pytest

from examples.op_reranker_expressiveness import (
    RECORDS,
    THRESHOLD,
    TOPK,
    ThresholdSelect,
    TopKSelect,
    _merge_groups,
    _cheap_scorer,
    _clusterer_stage,
    _sharp_scorer,
    _source,
    run_baseline,
    run_reranker,
)
from langres.core.models import CompanySchema
from langres.core.op import Op, Score, Select, Sequential

_TRUE_MERGES = [["a1", "a2"], ["g1", "g2"]]


def test_baseline_runs_and_over_merges_same_name_branches() -> None:
    """The four-slot shape (as Ops) fuses the same-name / different-address branches."""
    merges = _merge_groups(run_baseline())
    assert merges == [["a1", "a2"], ["g1", "g2"], ["i1", "i2"], ["u1", "u2"]]


def test_reranker_recovers_the_true_duplicate_structure() -> None:
    """A Score-after-Select rescores the survivors and separates the branches."""
    assert _merge_groups(run_reranker()) == _TRUE_MERGES


def test_reranker_is_strictly_tighter_than_the_baseline() -> None:
    """Same threshold + clusterer; the only variable is the inserted rescoring Score."""
    baseline = _merge_groups(run_baseline())
    reranker = _merge_groups(run_reranker())
    assert reranker != baseline
    # Every reranker merge is also a baseline merge -> the reranker only DROPPED
    # (wrong) merges, never invented one.
    assert all(group in baseline for group in reranker)
    assert len(reranker) < len(baseline)


def test_selects_are_ops_in_the_select_role() -> None:
    """The concrete THRESHOLD / TOPK selects are Ops (Pairs -> Pairs) in the Select role."""
    for select in (ThresholdSelect[CompanySchema](THRESHOLD), TopKSelect[CompanySchema](TOPK)):
        assert isinstance(select, Op)
        assert isinstance(select, Select)


def test_topk_select_actually_prunes_and_threshold_gates() -> None:
    """TOPK removes low-score candidates; THRESHOLD keeps only rows clearing the price."""
    scored = _cheap_scorer().forward(_source().forward(RECORDS))
    pruned = TopKSelect[CompanySchema](TOPK).forward(scored)
    assert 0 < len(pruned.rows) < len(scored.rows)  # genuinely pruned, not empty

    gated = ThresholdSelect[CompanySchema](THRESHOLD).forward(scored)
    assert all(row.predicted_match(THRESHOLD) is True for row in gated.rows)
    assert len(gated.rows) < len(scored.rows)


def test_wiring_guard_accepts_a_score_after_a_select() -> None:
    """The expressiveness crux: Sequential.check() admits Score-after-Select.

    ``BlockerSource -> Score -> Select(TOPK) -> Score -> Select(THRESHOLD) ->
    ClustererStage`` — the second Score sits AFTER the first Select. The four-slot
    core has one matcher position before the clusterer, so this is structurally new.
    """
    source, cheap, topk, sharp, thr, cluster = (
        _source(),
        _cheap_scorer(),
        TopKSelect[CompanySchema](TOPK),
        _sharp_scorer(),
        ThresholdSelect[CompanySchema](THRESHOLD),
        _clusterer_stage(),
    )
    seq: Sequential[CompanySchema] = Sequential(
        [source, cheap, topk, sharp, thr, cluster]
    )  # no raise == accepted
    # The Score-after-Select really is present in the validated topology.
    roles = [type(stage).__name__ for stage in seq.stages]
    assert roles.index("MatcherScore", roles.index("TopKSelect")) > roles.index("TopKSelect")
    assert isinstance(sharp, Score)


def test_wiring_guard_rejects_a_score_after_the_cluster_stage() -> None:
    """The guard is real: an Op after the ClusterStage (pairs gone) is refused."""
    source, cheap, cluster, sharp = (
        _source(),
        _cheap_scorer(),
        _clusterer_stage(),
        _sharp_scorer(),
    )
    with pytest.raises(ValueError, match="carrier"):
        Sequential([source, cheap, cluster, sharp])
