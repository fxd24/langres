"""CorrelationClusterer match-cut parity net for epic #193 -- the W3-d trap guard.

W3-d splits the fused match-cut-and-cluster in the ``resolve()`` / ``dedupe()``
spine into the two selections ``docs/THEORY.md`` separates: an explicit
:class:`~langres.core.op.ThresholdSelect` (the selection π at feasible-class
THRESHOLD) THEN pure-equivalence clustering over the survivors by a clusterer
with NO threshold of its own (:meth:`~langres.core._model_run.ModelRun._cluster`
over :meth:`~langres.core._model_run.ModelRun._closure_clusterer`).

The two W0/cascade goldens (``test_behavior_parity_w0`` /
``test_behavior_parity_cascade_w3d``) exercise only the DEFAULT transitive-closure
:class:`~langres.core.clusterer.Clusterer`, so they cannot catch the one thing the
rewire is most likely to get wrong: the closure clusterer must be a clone of the
clusterer's OWN class, not a hardcoded base ``Clusterer``. A
:class:`~langres.core.clusterers.correlation.CorrelationClusterer` does PIVOT
clustering, not transitive closure; downgrading it to the base class would merge
pivot-split chains that must stay separate -- and it would still pass both goldens
(they use the default clusterer), failing only a test that actually runs a
``CorrelationClusterer`` through the spine. This is that test.

The fixture is the classic over-merge chain: ``a-b`` and ``b-c`` both clear the
threshold, but there is NO direct ``a-c`` edge. Pivot clustering keeps ``{a, b}``
and ``{c}`` apart; transitive closure (the trap) would merge all three into
``{a, b, c}``. The test asserts the real spine helper reproduces the pivot split
and NOT the transitive-closure merge, so a revert of ``_closure_clusterer`` to a
hardcoded ``Clusterer`` turns it red.
"""

from __future__ import annotations

import pytest

from langres.core.clusterer import Clusterer
from langres.core.clusterers.correlation import CorrelationClusterer
from langres.core.models import CompanySchema
from langres.core.op import ThresholdSelect
from langres.core.op_adapters import ClustererStage
from langres.core.pairs import PairRow, Pairs
from langres.core.resolver import Resolver

_THRESHOLD = 0.8


def _sorted(clusters: list[set[str]]) -> list[list[str]]:
    """Order-independent view of a clustering, for equality assertions."""
    return sorted(sorted(cluster) for cluster in clusters)


def _chain_pairs() -> Pairs[CompanySchema]:
    """A scored ``Pairs`` for the over-merge chain a-b, b-c (NO direct a-c edge).

    Both edges score 0.9 (>= the 0.8 threshold). Pivot clustering splits this into
    ``{a, b}`` / ``{c}``; transitive closure merges it into ``{a, b, c}`` -- the
    exact input that distinguishes the two clustering algorithms.
    """
    store = {rid: CompanySchema(id=rid, name=rid) for rid in ("a", "b", "c")}
    rows = [
        PairRow(left_id="a", right_id="b", blocker_name="test", score=0.9, score_type="heuristic"),
        PairRow(left_id="b", right_id="c", blocker_name="test", score=0.9, score_type="heuristic"),
    ]
    return Pairs(store=store, rows=rows)


@pytest.fixture
def correlation_model() -> Resolver:
    """A real spine (blocker+comparator+matcher wired) with a CorrelationClusterer slot.

    Built through the ordinary ``from_schema`` door, then its clusterer slot is
    swapped for a ``CorrelationClusterer`` -- so ``_cluster`` / ``_closure_clusterer``
    run exactly as they do in production, over a pivot clusterer.
    """
    model = Resolver.from_schema(CompanySchema, matcher="string", threshold=_THRESHOLD)
    model.clusterer = CorrelationClusterer(threshold=_THRESHOLD)
    return model


def test_closure_clusterer_clones_the_clusterers_own_class(correlation_model: Resolver) -> None:
    """``_closure_clusterer`` returns a threshold-zeroed clone of the OWN class.

    The core of the trap guard: for a ``CorrelationClusterer`` slot the closure
    clusterer must itself be a ``CorrelationClusterer`` (a pivot clusterer), not a
    base ``Clusterer`` -- and its threshold is zeroed because the match cut already
    ran. ``self.clusterer`` is untouched (still holds its real threshold).
    """
    closure = correlation_model._closure_clusterer()

    assert type(closure) is CorrelationClusterer  # the OWN class, not base Clusterer
    assert closure.threshold == 0.0
    # self.clusterer is not mutated -- it keeps its real threshold.
    assert correlation_model.clusterer.threshold == _THRESHOLD
    assert type(correlation_model.clusterer) is CorrelationClusterer


def test_cluster_matches_legacy_clusterer_owned_cut(correlation_model: Resolver) -> None:
    """The W3-d ``_cluster`` path == the legacy ``ClustererStage(CorrelationClusterer(t))``.

    Legacy folded the match cut inside the clusterer's threshold; W3-d makes it an
    explicit ThresholdSelect then runs a zeroed pivot clone. The clusters must be
    byte-identical -- and specifically the pivot split ``{a, b}`` / ``{c}``.
    """
    pairs = _chain_pairs()

    # Legacy oracle: the clusterer owned the cut (threshold=_THRESHOLD), no Select.
    legacy = ClustererStage(CorrelationClusterer(threshold=_THRESHOLD)).forward(pairs)
    # W3-d: the real spine helper (ThresholdSelect -> zeroed pivot clone).
    got = correlation_model._cluster(pairs)

    assert _sorted(got) == _sorted(legacy)
    assert _sorted(got) == [["a", "b"], ["c"]]  # pivot split, NOT the merged chain


def test_the_trap_would_produce_the_wrong_merged_chain(correlation_model: Resolver) -> None:
    """Documents the trap: a hardcoded base ``Clusterer(0.0)`` merges the whole chain.

    The teeth of the test made explicit -- if ``_closure_clusterer`` were reverted
    to a hardcoded ``Clusterer``, the closure step would be transitive closure and
    merge ``{a, b, c}``. The real path must differ from that wrong answer.
    """
    pairs = _chain_pairs()
    selected = ThresholdSelect(_THRESHOLD).forward(pairs)

    # What the hardcoded-Clusterer trap would yield: transitive closure over-merges.
    trap = ClustererStage(Clusterer(threshold=0.0)).forward(selected)
    assert _sorted(trap) == [["a", "b", "c"]]

    # The real W3-d path avoids it (stays a pivot clusterer).
    got = correlation_model._cluster(pairs)
    assert _sorted(got) != _sorted(trap)
    assert _sorted(got) == [["a", "b"], ["c"]]
