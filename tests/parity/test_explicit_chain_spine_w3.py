"""The generalized Op-chain spine's *else* branch (epic #193, PR-A).

``ERModel`` can now run an EXPLICIT Op chain the four legacy slots cannot express,
built via :meth:`~langres.core._model_state.ModelState.from_topology`. Every
shipped architecture leaves ``_ops`` ``None`` and hits the unchanged classic
branch (pinned by the ``tests/parity/test_behavior_parity_*`` goldens); this file
covers the new ``else`` branch end to end.

All ``$0``/offline: a fake cost-stamping matcher (no network, no key), so the
spend ledger moves under a real budget without real spend. The chains are built
by :mod:`tests.parity._explicit_chain_fixture`, which the later persist PR reuses.
"""

from __future__ import annotations

from typing import cast

import pytest

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.fit import CalibratorFitMixin
from langres.core.models import MatcherAbstainedError
from langres.core.op import Stage, ThresholdSelect
from langres.core.op_adapters import BlockerSource, ClustererStage, MatcherScore
from langres.core.resolver import ERModel
from langres.core.spend import SpendMonitor
from langres.core.spend_cap import SpendCappedMatcher

from tests.parity._explicit_chain_fixture import (
    RECORDS,
    THRESHOLD,
    ChainCo,
    CostedNameMatcher,
    build_explicit_chain_model,
    build_factory_source_model,
    build_key_source_model,
    build_no_threshold_chain_model,
    build_score_after_select_model,
    chain_ops,
)


def _record(record_id: str) -> dict[str, object]:
    return next(record for record in RECORDS if record["id"] == record_id)


def _canonical(clusters: list[set[str]]) -> list[list[str]]:
    return sorted(sorted(cluster) for cluster in clusters)


# ----------------------------------------------------------------------
# resolve / dedupe / compare run correctly through the explicit chain
# ----------------------------------------------------------------------


def test_resolve_runs_the_explicit_chain() -> None:
    """resolve() folds Source + body Ops, then the terminal ClusterStage."""
    model, _monitor, _matcher = build_explicit_chain_model()
    assert _canonical(model.resolve(RECORDS)) == [["a1", "a2"]]


def test_dedupe_reports_chain_threshold_and_backbone() -> None:
    """dedupe()'s self-describing metadata comes from the chain, not the (empty) slots.

    threshold = the terminal ThresholdSelect's; backbone = the chain's scoring
    matcher's (None -- the fake advertises no model); score_type stays row-derived.
    """
    model, _monitor, _matcher = build_explicit_chain_model()
    result = model.dedupe(RECORDS)

    assert _canonical(list(result)) == [["a1", "a2"]]
    assert result.architecture == "ERModel"
    assert result.backbone is None
    assert result.score_type == "prob_llm"
    assert result.threshold == THRESHOLD


def test_dedupe_backbone_reads_chain_matcher_model() -> None:
    """backbone unwraps the chain's SpendCappedMatcher to report the real matcher's model."""
    model, _monitor, _matcher = build_explicit_chain_model(model="fake/model")
    assert model.dedupe(RECORDS).backbone == "fake/model"


def test_compare_matches_and_nonmatches() -> None:
    """compare() folds only the chain's Score ops and gates on its ThresholdSelect."""
    model, _monitor, _matcher = build_explicit_chain_model()

    match = model.compare(_record("a1"), _record("a2"))
    assert match.match is True
    assert match.score == 1.0
    assert match.threshold == THRESHOLD
    assert match.architecture == "ERModel"

    nonmatch = model.compare(_record("a1"), _record("b1"))
    assert nonmatch.match is False
    assert nonmatch.score == 0.0


def test_dedupe_short_circuits_below_two_records() -> None:
    """The < 2 short-circuit runs before any chain walk (no scoring, threshold None)."""
    model, _monitor, matcher = build_explicit_chain_model()
    result = model.dedupe([_record("a1")])
    assert list(result) == []
    assert result.threshold is None
    assert matcher.produced == 0


# ----------------------------------------------------------------------
# the spend cap is a semantic guard on the explicit chain
# ----------------------------------------------------------------------


def test_spend_monitor_is_the_models_ledger_and_is_hit() -> None:
    """An explicit-chain paid Score routes through the model's shared SpendMonitor.

    4 records -> 6 AllPairs candidates at $0.05 -> $0.30 on the SAME ledger the
    model caps with. Mutation check: drop the SpendCappedMatcher wrap in the
    fixture and ``spent`` stays 0.
    """
    model, monitor, matcher = build_explicit_chain_model(cost_each=0.05)
    model.resolve(RECORDS)

    assert model._spend_monitor is monitor
    assert matcher.produced == 6
    assert monitor.spent == pytest.approx(0.30)


def test_explicit_chain_budget_trips_and_carries_partials() -> None:
    """The shared budget bounds the explicit chain: a run past it raises BudgetExceeded."""
    from langres.core.spend import BudgetExceeded

    # 6 pairs at $0.60 -> the first trips a $1.00 budget after two paid calls.
    model, _monitor, _matcher = build_explicit_chain_model(budget_usd=1.0, cost_each=0.6)
    with pytest.raises(BudgetExceeded):
        model.resolve(RECORDS)


# ----------------------------------------------------------------------
# Score-after-Select, and the no-single-slot reads
# ----------------------------------------------------------------------


def test_score_after_select_chain_runs() -> None:
    """A Score after a Select (TopKSelect between two MatcherScores) is walked correctly."""
    model = build_score_after_select_model()
    assert _canonical(model.resolve(RECORDS)) == [["a1", "a2"]]
    # The last MatcherScore's family wins the row-derived report.
    assert model.dedupe(RECORDS).score_type == "prob_llm"


def test_chain_without_threshold_select_reports_none_threshold() -> None:
    """A chain whose cut lives in the ClusterStage clusterer still resolves, and
    dedupe honestly reports threshold=None (there is no terminal ThresholdSelect)."""
    model = build_no_threshold_chain_model()
    assert _canonical(model.resolve(RECORDS)) == [["a1", "a2"]]
    assert model.dedupe(RECORDS).threshold is None


def test_compare_without_a_cut_raises() -> None:
    """compare() needs a match cut; a chain with no ThresholdSelect raises a directed error."""
    model = build_no_threshold_chain_model()
    with pytest.raises(RuntimeError, match="no terminal ThresholdSelect"):
        model.compare(_record("a1"), _record("a2"))


def test_compare_builds_the_pair_when_the_source_vetoes_it() -> None:
    """Blocking must not veto a compare verdict: a KeyBlocker source that yields no
    candidate for two different-name records still gets a scored, gated verdict."""
    model = build_key_source_model()
    verdict = model.compare(_record("a1"), _record("b1"))  # different names -> no key bucket
    assert verdict.match is False
    assert verdict.score == 0.0


def test_factory_source_chain_infers_schema() -> None:
    """A Source blocker with no schema (opaque schema_factory) -> the front door infers one."""
    model = build_factory_source_model()
    assert _canonical(model.resolve(RECORDS)) == [["a1", "a2"]]
    assert _canonical(list(model.dedupe(RECORDS))) == [["a1", "a2"]]


# ----------------------------------------------------------------------
# from_topology door: the directed rejections
# ----------------------------------------------------------------------


def test_from_topology_rejects_a_calibrator() -> None:
    """Calibration is a classic-path-only bespoke transform; an explicit chain rejects it."""
    monitor = SpendMonitor(budget_usd=1.0)
    ops, _matcher = chain_ops(monitor)
    with pytest.raises(ValueError, match="does not accept a calibrator"):
        ERModel.from_topology(ops=ops, calibrator=cast(CalibratorFitMixin, object()))


def test_from_topology_rejects_monitor_and_budget_together() -> None:
    """A monitor already carries its budget; passing budget_usd too is ambiguous."""
    monitor = SpendMonitor(budget_usd=1.0)
    ops, _matcher = chain_ops(monitor)
    with pytest.raises(ValueError, match="not both"):
        ERModel.from_topology(ops=ops, monitor=monitor, budget_usd=2.0)


def test_from_topology_requires_exactly_one_cluster_stage() -> None:
    """resolve()/dedupe() need a phase-1 exit: no ClusterStage (or two) is refused."""
    monitor = SpendMonitor(budget_usd=1.0)
    no_stage: list[Stage] = [
        BlockerSource(AllPairsBlocker(schema=ChainCo)),
        MatcherScore(SpendCappedMatcher(CostedNameMatcher(), monitor=monitor), out_space="prob_llm"),
    ]
    with pytest.raises(ValueError, match="exactly one terminal ClusterStage"):
        ERModel.from_topology(ops=no_stage, monitor=monitor)

    two_stages: list[Stage] = [
        BlockerSource(AllPairsBlocker(schema=ChainCo)),
        MatcherScore(SpendCappedMatcher(CostedNameMatcher(), monitor=monitor), out_space="prob_llm"),
        ThresholdSelect(THRESHOLD),
        ClustererStage(Clusterer(threshold=0.0)),
        ClustererStage(Clusterer(threshold=0.0)),
    ]
    with pytest.raises(ValueError, match="exactly one terminal ClusterStage"):
        ERModel.from_topology(ops=two_stages, monitor=SpendMonitor(budget_usd=1.0))


def test_from_topology_rejects_a_miswired_chain() -> None:
    """The Sequential wiring guard runs at construction (here: no Source first)."""
    monitor = SpendMonitor(budget_usd=1.0)
    miswired: list[Stage] = [
        MatcherScore(SpendCappedMatcher(CostedNameMatcher(), monitor=monitor), out_space="prob_llm"),
        ClustererStage(Clusterer(threshold=0.0)),
    ]
    with pytest.raises(ValueError, match="must start with a Source"):
        ERModel.from_topology(ops=miswired, monitor=monitor)


# ----------------------------------------------------------------------
# the classic path is untouched (the is-None branch still works)
# ----------------------------------------------------------------------


def test_classic_slot_model_still_runs_and_leaves_ops_none() -> None:
    """A component-wired ERModel keeps _ops None and takes the unchanged classic path."""
    model = ERModel.from_schema(ChainCo, matcher="string", threshold=THRESHOLD)
    assert model._ops is None
    assert _canonical(model.resolve(RECORDS)) == [["a1", "a2"]]
    assert model.compare(_record("a1"), _record("a2")).match is True
