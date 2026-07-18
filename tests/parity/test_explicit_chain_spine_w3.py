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

from collections.abc import Iterator
from typing import Any

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.fit import CalibratorFitMixin
from langres.core.groups import ERCandidateGroup
from langres.core.matcher import GroupwiseMatcher
from langres.core.models import MatcherAbstainedError, PairwiseJudgement
from langres.core.op import Clusters, Finalize, Score, Spending, Stage, ThresholdSelect
from langres.core.op_adapters import (
    BlockerSource,
    ClustererStage,
    GroupwiseMatcherScore,
    MatcherScore,
)
from langres.core.resolver import ERModel
from langres.core.spend import SpendMonitor
from langres.core.spend_cap import SpendCappedMatcher

from tests.parity._explicit_chain_fixture import (
    RECORDS,
    THRESHOLD,
    ChainCo,
    CostedNameMatcher,
    build_abstaining_chain_model,
    build_explicit_chain_model,
    build_factory_source_model,
    build_key_source_model,
    build_no_threshold_chain_model,
    build_precapped_chain_model,
    build_score_after_select_model,
    chain_ops,
)


def _record(record_id: str) -> dict[str, object]:
    return next(record for record in RECORDS if record["id"] == record_id)


def _canonical(clusters: list[set[str]]) -> list[list[str]]:
    return sorted(sorted(cluster) for cluster in clusters)


def _chain_matcher_scores(model: ERModel) -> list[MatcherScore[Any]]:
    """The MatcherScores stored on the model's explicit chain (post from_topology)."""
    assert model._ops is not None
    return [stage for stage in model._ops if isinstance(stage, MatcherScore)]


# ----------------------------------------------------------------------
# resolve / dedupe / compare run correctly through the explicit chain
# ----------------------------------------------------------------------


def test_resolve_runs_the_explicit_chain() -> None:
    """resolve() folds Source + body Ops, then the terminal ClusterStage."""
    model, _monitor, _matcher = build_explicit_chain_model()
    assert _canonical(model.resolve(RECORDS)) == [["a1", "a2"]]


def test_predict_runs_the_explicit_chain() -> None:
    """predict() reaches the explicit ``_scored_pairs`` branch (via ``_judgements``).

    On a chain whose cut lives in the ClusterStage (no body ThresholdSelect to
    prune), predict returns one judgement per candidate -- 6 AllPairs pairs -- and
    the two Acmes are the matching pair.
    """
    model = build_no_threshold_chain_model()
    judgements = model.predict(RECORDS)

    assert len(judgements) == 6  # 4 records -> 6 AllPairs candidates, all scored
    a1_a2 = next(j for j in judgements if {j.left_id, j.right_id} == {"a1", "a2"})
    assert a1_a2.score == 1.0


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


def test_from_topology_caps_a_raw_matcher_via_the_budget_door() -> None:
    """The ``budget_usd`` door (no shared ``monitor``) auto-caps a RAW matcher-Score.

    The caller passes an UNWRAPPED matcher; ``from_topology`` must wrap it in a
    ``SpendCappedMatcher`` on this model's ledger, so the stored Score is capped and
    backbone still peels the cap to report the real model. This is the enforcement
    the ``.module.forward`` AST-ban cannot see on the explicit path.
    """
    raw = CostedNameMatcher(model="raw/model")  # NOT wrapped -- the door must wrap it
    ops: list[Stage] = [
        BlockerSource(AllPairsBlocker(schema=ChainCo)),
        MatcherScore(raw, out_space="prob_llm"),
        ThresholdSelect(THRESHOLD),
        ClustererStage(Clusterer(threshold=0.0)),
    ]
    model = ERModel.from_topology(ops=ops, budget_usd=1.0)

    # The door enforced the cap: the stored Score now holds a SpendCappedMatcher
    # wrapping the caller's raw matcher, sharing the model's ledger.
    (stored,) = _chain_matcher_scores(model)
    assert isinstance(stored.matcher, SpendCappedMatcher)
    assert stored.matcher._module is raw
    assert stored.matcher.monitor is model._spend_monitor

    assert _canonical(model.resolve(RECORDS)) == [["a1", "a2"]]
    assert model.dedupe(RECORDS).backbone == "raw/model"  # backbone peels the door's cap


def test_from_topology_enforces_the_cap_on_a_raw_matcher() -> None:
    """The money guarantee holds even when the caller forgets to wrap: a chain built
    with a RAW paid matcher still trips BudgetExceeded (the door capped it)."""
    from langres.core.spend import BudgetExceeded

    raw = CostedNameMatcher(cost_each=0.6)  # 6 pairs at $0.60 overruns a $1.00 budget
    ops: list[Stage] = [
        BlockerSource(AllPairsBlocker(schema=ChainCo)),
        MatcherScore(raw, out_space="prob_llm"),
        ThresholdSelect(THRESHOLD),
        ClustererStage(Clusterer(threshold=0.0)),
    ]
    model = ERModel.from_topology(ops=ops, budget_usd=1.0)
    with pytest.raises(BudgetExceeded):
        model.resolve(RECORDS)


def test_from_topology_leaves_an_already_capped_matcher_alone() -> None:
    """A matcher the caller already wrapped is NOT double-wrapped: the stored Score
    holds the caller's exact SpendCappedMatcher object."""
    model, monitor, capped = build_precapped_chain_model()
    (stored,) = _chain_matcher_scores(model)
    assert stored.matcher is capped
    assert model._spend_monitor is monitor


def test_from_topology_leaves_a_precapped_matcherscore_subclass_alone() -> None:
    """A MatcherScore SUBCLASS whose matcher is already capped through the model
    monitor takes the leave-alone branch (no rebuild, so its subclass state survives)
    -- the rebuild-reject fires only for a subclass with a RAW matcher."""

    class _TaggedMatcherScore(MatcherScore[Any]):
        def __init__(self, matcher: Any, *, out_space: Any, tag: str) -> None:
            super().__init__(matcher, out_space=out_space)
            self.tag = tag

    monitor = SpendMonitor(budget_usd=1.0)
    capped = SpendCappedMatcher(CostedNameMatcher(), monitor=monitor)
    subclass_score = _TaggedMatcherScore(capped, out_space="prob_llm", tag="keep-me")
    ops: list[Stage] = [
        BlockerSource(AllPairsBlocker(schema=ChainCo)),
        subclass_score,
        ThresholdSelect(THRESHOLD),
        ClustererStage(Clusterer(threshold=0.0)),
    ]
    model = ERModel.from_topology(ops=ops, monitor=monitor)

    (stored,) = _chain_matcher_scores(model)
    assert stored is subclass_score  # same object -- not rebuilt, subclass state intact
    assert subclass_score.tag == "keep-me"  # subclass state survived (no base rebuild)
    assert stored.matcher is capped


def test_from_topology_rejects_a_spending_score_it_cannot_cap() -> None:
    """A Spending Score the door cannot wrap (a GroupwiseMatcherScore, whose
    ``forward_groups`` bypasses a SpendCappedMatcher's per-judgement metering) is
    rejected rather than allowed to bill off-ledger."""

    class _NullGroupwiseMatcher(GroupwiseMatcher[Any]):
        def forward_groups(
            self, groups: Iterator[ERCandidateGroup[Any]]
        ) -> Iterator[PairwiseJudgement]:
            return iter(())  # never called: from_topology rejects the chain first

    monitor = SpendMonitor(budget_usd=1.0)
    ops: list[Stage] = [
        BlockerSource(AllPairsBlocker(schema=ChainCo)),
        GroupwiseMatcherScore(_NullGroupwiseMatcher()),
        ThresholdSelect(THRESHOLD),
        ClustererStage(Clusterer(threshold=0.0)),
    ]
    with pytest.raises(ValueError, match="cannot cap a GroupwiseMatcherScore"):
        ERModel.from_topology(ops=ops, monitor=monitor)


def test_from_topology_rejects_a_foreign_monitor_precap() -> None:
    """A pre-capped MatcherScore whose cap uses a DIFFERENT monitor than the model
    ledger is rejected: its spend would never count against the model budget, so
    'one model ledger' would be false and total spend could exceed budget_usd."""
    model_ledger = SpendMonitor(budget_usd=1.0)
    foreign = SpendMonitor(budget_usd=1.0)  # a different ledger the model never sees
    ops: list[Stage] = [
        BlockerSource(AllPairsBlocker(schema=ChainCo)),
        MatcherScore(
            SpendCappedMatcher(CostedNameMatcher(), monitor=foreign), out_space="prob_llm"
        ),
        ThresholdSelect(THRESHOLD),
        ClustererStage(Clusterer(threshold=0.0)),
    ]
    with pytest.raises(ValueError, match="monitor other than this model's ledger"):
        ERModel.from_topology(ops=ops, monitor=model_ledger)


def test_from_topology_rejects_a_spending_score_that_bills_via_a_hidden_attr() -> None:
    """The allowlist is fail-safe against a custom Spending Score that holds its paid
    matcher under a differently-named attribute (the denylist that sniffed ``.matcher``
    would have let this bill uncapped): declared Spending + not a MatcherScore -> reject."""

    class _HiddenBillingScore(Score[Any], Spending):
        """A paid Score whose matcher lives at ``.judge`` (not ``.matcher``)."""

        def __init__(self) -> None:
            super().__init__(scope="pair", out_space="prob_llm")
            self.judge = CostedNameMatcher(cost_each=0.6)  # would bill uncapped if run

        def forward(self, pairs: Any) -> Any:  # never called: rejected at from_topology
            raise NotImplementedError

    ops: list[Stage] = [
        BlockerSource(AllPairsBlocker(schema=ChainCo)),
        _HiddenBillingScore(),
        ThresholdSelect(THRESHOLD),
        ClustererStage(Clusterer(threshold=0.0)),
    ]
    with pytest.raises(ValueError, match="cannot cap a _HiddenBillingScore"):
        ERModel.from_topology(ops=ops, budget_usd=1.0)


def test_from_topology_admits_a_free_non_spending_custom_score() -> None:
    """A legitimate FREE custom Score (does not declare Spending) passes untouched and
    the chain runs -- the allowlist must not over-restrict free scalarizers."""

    class _FreePassthroughScore(Score[Any]):  # NOT Spending -> trusted free
        def __init__(self) -> None:
            super().__init__(scope="pair", out_space="prob_llm")

        def forward(self, pairs: Any) -> Any:
            return pairs  # a no-op transform on the scored rows

    free = _FreePassthroughScore()
    ops: list[Stage] = [
        BlockerSource(AllPairsBlocker(schema=ChainCo)),
        MatcherScore(CostedNameMatcher(), out_space="prob_llm"),
        free,
        ThresholdSelect(THRESHOLD),
        ClustererStage(Clusterer(threshold=0.0)),
    ]
    model = ERModel.from_topology(ops=ops, budget_usd=1.0)
    # The door must PRESERVE the free stage (not silently drop it): same object, same
    # position (index 2). A no-op forward() alone would pass even if it were dropped.
    assert model._ops is not None
    assert model._ops[2] is free
    assert _canonical(model.resolve(RECORDS)) == [["a1", "a2"]]


def test_from_topology_rejects_a_raw_matcherscore_subclass() -> None:
    """A MatcherScore SUBCLASS carrying a raw matcher is rejected: rebuilding it as a
    base MatcherScore to inject the cap would silently drop the subclass's own state
    (codex Q3). A pre-capped subclass sharing the model monitor is fine (covered
    elsewhere); only the raw-matcher rebuild is refused."""

    class _TaggedMatcherScore(MatcherScore[Any]):
        """A MatcherScore subclass with extra state the base rebuild would lose."""

        def __init__(self, matcher: Any, *, out_space: Any, tag: str) -> None:
            super().__init__(matcher, out_space=out_space)
            self.tag = tag

    ops: list[Stage] = [
        BlockerSource(AllPairsBlocker(schema=ChainCo)),
        _TaggedMatcherScore(CostedNameMatcher(), out_space="prob_llm", tag="keep-me"),
        ThresholdSelect(THRESHOLD),
        ClustererStage(Clusterer(threshold=0.0)),
    ]
    with pytest.raises(ValueError, match="cannot faithfully spend-cap a MatcherScore subclass"):
        ERModel.from_topology(ops=ops, budget_usd=1.0)


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
    model caps with. The fixture passes a RAW matcher, so this also proves
    ``from_topology`` wired the door's auto-cap onto the model's ledger (no manual
    wrap in the fixture at all).
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


def test_compare_raises_on_an_abstaining_chain_matcher() -> None:
    """compare() owes a verdict: an abstaining chain matcher raises MatcherAbstainedError
    (the Selects are skipped, so the abstaining pair reaches the gate rather than being cut)."""
    model = build_abstaining_chain_model()
    with pytest.raises(MatcherAbstainedError):
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
    ops, _matcher = chain_ops()
    with pytest.raises(ValueError, match="does not accept a calibrator"):
        ERModel.from_topology(ops=ops, calibrator=cast(CalibratorFitMixin, object()))


def test_from_topology_rejects_monitor_and_budget_together() -> None:
    """A monitor already carries its budget; passing budget_usd too is ambiguous."""
    monitor = SpendMonitor(budget_usd=1.0)
    ops, _matcher = chain_ops()
    with pytest.raises(ValueError, match="not both"):
        ERModel.from_topology(ops=ops, monitor=monitor, budget_usd=2.0)


def test_from_topology_rejects_a_finalize() -> None:
    """A terminal Finalize would be silently dropped by the explicit exit (which stops
    at the ClusterStage), so from_topology rejects it loud (deferred, not silent)."""

    class _IdentityFinalize(Finalize):
        def forward(self, clusters: Clusters) -> Clusters:
            return clusters  # never reached: from_topology rejects the chain first

    ops, _matcher = chain_ops()
    ops.append(_IdentityFinalize())
    with pytest.raises(ValueError, match="does not run a terminal Finalize"):
        ERModel.from_topology(ops=ops, budget_usd=1.0)


def test_from_topology_requires_a_terminal_cluster_stage() -> None:
    """resolve()/dedupe() need a phase-1 exit: a Sequential-valid chain with no
    ClusterStage (it ends in Scores) is refused by from_topology's own count check
    -- the one thing Sequential does not enforce."""
    monitor = SpendMonitor(budget_usd=1.0)
    no_stage: list[Stage] = [
        BlockerSource(AllPairsBlocker(schema=ChainCo)),
        MatcherScore(
            SpendCappedMatcher(CostedNameMatcher(), monitor=monitor), out_space="prob_llm"
        ),
    ]
    with pytest.raises(ValueError, match="exactly one terminal ClusterStage"):
        ERModel.from_topology(ops=no_stage, monitor=monitor)


def test_from_topology_rejects_a_second_cluster_stage_at_the_wiring_guard() -> None:
    """Two ClusterStages can't both sit terminally -- Sequential catches the carrier
    mismatch (a ClusterStage consumes 'pairs', the first produced 'clusters') before
    the count check ever runs."""
    monitor = SpendMonitor(budget_usd=1.0)
    two_stages: list[Stage] = [
        BlockerSource(AllPairsBlocker(schema=ChainCo)),
        MatcherScore(
            SpendCappedMatcher(CostedNameMatcher(), monitor=monitor), out_space="prob_llm"
        ),
        ThresholdSelect(THRESHOLD),
        ClustererStage(Clusterer(threshold=0.0)),
        ClustererStage(Clusterer(threshold=0.0)),
    ]
    with pytest.raises(ValueError, match="out of pipeline order"):
        ERModel.from_topology(ops=two_stages, monitor=monitor)


def test_from_topology_rejects_a_miswired_chain() -> None:
    """The Sequential wiring guard runs at construction (here: no Source first)."""
    monitor = SpendMonitor(budget_usd=1.0)
    miswired: list[Stage] = [
        MatcherScore(
            SpendCappedMatcher(CostedNameMatcher(), monitor=monitor), out_space="prob_llm"
        ),
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
