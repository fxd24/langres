"""Proofs for the ``Reranker`` architecture -- epic #193 PR-C, the expressiveness payoff.

``Reranker`` (``src/langres/architectures/reranker.py``) exists to demonstrate ONE
claim: a topology the four fixed slots cannot express -- a ``Score`` AFTER a
``Select`` (a rerank) -- is now expressible AND persistable through the PUBLIC
:meth:`~langres.core._model_state.ModelState.from_topology` door, with **zero core
change**. Every test here is written to fail if that claim breaks:

- **golden** -- the two-pass chain recovers the true duplicate structure a
  single-pass (name-only) matcher over-merges. Fails if the rerank stops running
  or the chain mis-wires.
- **persist** -- save -> load (fresh) round-trips as a ``Reranker`` (not a plain
  ``Resolver``) with an identical resolution and a RE-SECURED spend cap. Fails if
  persist v2 drops the ``model_class`` stamp, the ops, or the cap.
- **score-after-select** -- the rerank (pass 2, AFTER the ``TopKSelect``)
  DEMOTES a cheap-pass winner the four-slot core would have merged. This is the
  topology doing something four fixed slots could not.
- **wiring** -- constructing the reranker does not raise from ``Sequential``: the
  cheap scalar ``Score`` overwrites the comparator's vector before the ``Select``,
  so the ``Select`` is over an orderable scalar (a Select on a vector is illegal).

All $0, deterministic, offline -- rapidfuzz string similarity only, no LLM, no
network. ``CompanySchema`` has five fields (``id`` excluded); the first comparable
field is ``name``, so the cheap pass scores on name alone and the rerank adds
``address`` -- exactly the same-name / different-address trap that separates a
real duplicate from a coincidental name clash.
"""

from __future__ import annotations

from pathlib import Path

from langres.architectures import Reranker
from langres.architectures.reranker import Reranker as RerankerDirect
from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.comparators import StringComparator
from langres.core.matchers.weighted_average import WeightedAverageMatcher
from langres.core.models import CompanySchema
from langres.core.op import Score, Select, ThresholdSelect, TopKSelect
from langres.core.op_adapters import BlockerSource, ClustererStage, ComparatorScore, MatcherScore
from langres.core.registry import model_type_name
from langres.core.resolver import ERModel
from langres.core.serialization import ArtifactManifest
from langres.core.spend_cap import SpendCappedMatcher

# --------------------------------------------------------------------------------------
# Fixed $0 dataset. Acme/Globex are true duplicates (same name AND same address);
# Initech is the TRAP -- same name, DIFFERENT address, so a name-only pass merges it
# but a name+address rerank keeps it apart. Soylent is a singleton.
# --------------------------------------------------------------------------------------

RECORDS: list[dict[str, object]] = [
    {"id": "a1", "name": "Acme Inc", "address": "1 Main St"},
    {"id": "a2", "name": "Acme Inc", "address": "1 Main St"},
    {"id": "g1", "name": "Globex Corp", "address": "500 Oak Ave"},
    {"id": "g2", "name": "Globex Corp", "address": "500 Oak Ave"},
    {"id": "i1", "name": "Initech", "address": "5 North Ave"},
    {"id": "i2", "name": "Initech", "address": "820 South Blvd"},
    {"id": "s1", "name": "Soylent Foods", "address": "7 Green Way"},
]

#: The only true duplicate merges. The trap ``{i1, i2}`` is deliberately NOT here.
TRUE_MERGES = [["a1", "a2"], ["g1", "g2"]]

K = 3
THRESHOLD = 0.85


def _merges(clusters: list[set[str]]) -> list[list[str]]:
    """Order-independent view of the non-singleton clusters (the actual merges)."""
    return sorted(sorted(cluster) for cluster in clusters if len(cluster) > 1)


def _pass1_only_baseline() -> ERModel:
    """The name-only single-pass baseline, as an explicit chain (no rerank).

    ``BlockerSource -> ComparatorScore -> MatcherScore(name only) -> ThresholdSelect
    -> ClustererStage`` -- the four-slot shape (one matcher position). It is the
    control the reranker is measured against: same blocking, comparator, threshold
    and clusterer, the ONLY difference being the missing ``TopKSelect`` + second
    (full-evidence) ``Score``.
    """
    comparator = StringComparator.from_schema(CompanySchema)
    cheap = [comparator.feature_specs[0]]
    ops = [
        BlockerSource(AllPairsBlocker(schema=CompanySchema)),
        ComparatorScore(comparator),
        MatcherScore(WeightedAverageMatcher(feature_specs=cheap), out_space="heuristic"),
        ThresholdSelect(THRESHOLD),
        ClustererStage(Clusterer(threshold=0.0)),
    ]
    return ERModel.from_topology(ops=ops)


def test_reranker_is_an_explicit_seven_op_chain() -> None:
    """The reranker runs an explicit Op chain (``_ops`` set), not the four slots.

    The seven roles are exactly the Score-after-Select topology the four-slot core
    cannot place: the SECOND ``MatcherScore`` sits AFTER the ``TopKSelect``.
    """
    model = Reranker.for_schema(CompanySchema, k=K, threshold=THRESHOLD)
    assert model._ops is not None  # explicit chain, not slot-derived
    roles = [type(stage).__name__ for stage in model._ops]
    assert roles == [
        "BlockerSource",
        "ComparatorScore",
        "MatcherScore",
        "TopKSelect",
        "MatcherScore",
        "ThresholdSelect",
        "ClustererStage",
    ]
    # A Score genuinely AFTER a Select -- the crux.
    topk_index = roles.index("TopKSelect")
    assert roles.index("MatcherScore", topk_index) > topk_index


def test_golden_dedupe_recovers_true_duplicate_structure() -> None:
    """GOLDEN: the reranker recovers exactly the true duplicates, dropping the trap.

    The name-only baseline over-merges Initech (same name, different address); the
    two-pass reranker rescores the survivors on name+address and keeps them apart.
    """
    result = Reranker.for_schema(CompanySchema, k=K, threshold=THRESHOLD).dedupe(RECORDS)
    assert _merges(result) == TRUE_MERGES
    # Self-describing result metadata (from the chain, not a slot).
    assert result.architecture == "Reranker"
    assert result.score_type == "heuristic"
    assert result.threshold == THRESHOLD
    assert result.backbone is None  # free string matchers -- nothing with weights


def test_score_after_select_rerank_demotes_a_cheap_pass_winner() -> None:
    """The topology does something four fixed slots could NOT: a Score after a Select.

    The trap pair ``{i1, i2}`` is a *cheap-pass winner* -- name-only similarity 1.0,
    which clears the threshold -- so the single-pass baseline merges it. The reranker
    keeps it as a top-k survivor and then the SECOND ``Score`` (name+address, after
    the ``TopKSelect``) demotes it below the cut. The two pipelines share every knob
    but that inserted rescore, so the divergence IS the Score-after-Select payoff.
    """
    baseline = _merges(_pass1_only_baseline().dedupe(RECORDS))
    reranker = _merges(Reranker.for_schema(CompanySchema, k=K, threshold=THRESHOLD).dedupe(RECORDS))

    # The single-pass baseline over-merges the trap; the reranker does not.
    assert ["i1", "i2"] in baseline
    assert ["i1", "i2"] not in reranker
    # The reranker only DROPPED the wrong merge -- it invented none.
    assert all(group in baseline for group in reranker)
    assert len(reranker) < len(baseline)
    assert reranker == TRUE_MERGES


def test_construction_wires_a_legal_sequential_scalar_before_the_select() -> None:
    """WIRING: constructing does not raise from ``Sequential``.

    ``Sequential`` rejects a ``Select`` over a vector score (a ``ComparisonVector``
    is not orderable). The reranker is legal because the cheap ``MatcherScore``
    (a scalar ``"heuristic"``) overwrites the ``ComparatorScore``'s vector BEFORE the
    ``TopKSelect``. This test pins that ordering: the vector Score precedes the
    scalarizing Score, which precedes the first Select.
    """
    model = Reranker.for_schema(CompanySchema, k=K, threshold=THRESHOLD)  # no raise == wired
    assert model._ops is not None
    ops = model._ops

    comparator_score = next(i for i, s in enumerate(ops) if isinstance(s, ComparatorScore))
    first_scalarizer = next(
        i for i, s in enumerate(ops) if isinstance(s, Score) and not isinstance(s, ComparatorScore)
    )
    first_select = next(i for i, s in enumerate(ops) if isinstance(s, Select))

    # vector Score  ->  scalar Score  ->  Select : the scalar cut the vector first.
    assert comparator_score < first_scalarizer < first_select
    assert isinstance(ops[comparator_score], ComparatorScore)  # out_space="vector"
    assert isinstance(ops[first_select], TopKSelect)


def test_persist_round_trips_as_a_reranker_with_a_resecured_cap(tmp_path: Path) -> None:
    """PERSIST v2: save -> load (fresh) is a ``Reranker`` with an identical resolution.

    Exercises the whole persist-v2 path for a real architecture: the manifest stamps
    ``model_class="reranker"`` and an ``ops`` list, ``load`` rebuilds the class the
    artifact names, and ``from_topology`` re-establishes the spend cap on the loaded
    chain (the cap is a run policy, re-wrapped fresh, never persisted).
    """
    model = Reranker.for_schema(CompanySchema, k=K, threshold=THRESHOLD)
    original = _merges(model.dedupe(RECORDS))
    assert original == TRUE_MERGES  # sanity: we saved a working model

    out = tmp_path / "reranker_artifact"
    model.save(out)

    # The on-disk manifest names the architecture and carries an explicit chain.
    manifest = ArtifactManifest.model_validate_json((out / "resolver.json").read_text())
    assert manifest.model_class == "reranker"
    assert manifest.ops is not None and manifest.components == []

    # A FRESH load (via the base class) reconstructs the Reranker, not a plain Resolver.
    loaded = ERModel.load(out)
    assert isinstance(loaded, Reranker)
    assert model_type_name(type(loaded)) == "reranker"
    assert loaded._ops is not None

    # Identical resolution after the round-trip.
    assert _merges(loaded.dedupe(RECORDS)) == original
    assert _merges(loaded.resolve(RECORDS)) == original

    # The cap was re-secured: the loaded scoring MatcherScore holds a
    # SpendCappedMatcher sharing THIS loaded model's ledger (not a persisted one).
    matcher_scores = [s for s in loaded._ops if isinstance(s, MatcherScore)]
    scoring = matcher_scores[-1]  # the rerank Score
    assert isinstance(scoring.matcher, SpendCappedMatcher)
    assert scoring.matcher.monitor is loaded._spend_monitor


def test_compare_gates_on_the_reranked_score(tmp_path: Path) -> None:
    """``compare`` folds the chain's Scores (both passes) and gates on the rerank cut.

    A true duplicate matches; the same-name / different-address trap does not --
    because ``compare`` scores on the FULL evidence (name+address), exactly as the
    rerank does in ``dedupe``. Blocking never vetoes the pair.
    """
    model = Reranker.for_schema(CompanySchema, k=K, threshold=THRESHOLD)
    dup = model.compare(
        {"id": "a1", "name": "Acme Inc", "address": "1 Main St"},
        {"id": "a2", "name": "Acme Inc", "address": "1 Main St"},
    )
    trap = model.compare(
        {"id": "i1", "name": "Initech", "address": "5 North Ave"},
        {"id": "i2", "name": "Initech", "address": "820 South Blvd"},
    )
    assert dup.match is True
    assert trap.match is False
    assert dup.score_type == "heuristic" and dup.threshold == THRESHOLD


def test_reranker_is_exported_from_the_package() -> None:
    """The architecture is reachable both ways it is documented (package + module)."""
    assert Reranker is RerankerDirect
    from langres.architectures import __all__ as arch_all

    assert "Reranker" in arch_all
