"""Tests for AnchorStore / ClusterDelta — incremental single-record assign (W2.2).

Two pipelines are exercised:

- The **string** pipeline (``AllPairsBlocker`` + ``WeightedAverageJudge``) — pure
  core, no embeddings — drives the store logic, singleton coverage, stable ids,
  and the all-pairs candidate path, plus a fresh-in-process save/load round-trip.
- The **vector** pipeline (``VectorBlocker`` + ``FAISSIndex`` + ``FakeEmbedder``)
  is fast and deterministic (no model download) and exercises the single-record
  kNN candidate path over ``index.search`` and the FAISS-sidecar round-trip.

Records carry name + address + phone so the WeightedAverageJudge's evidence floor
is cleared (a single present feature scores 0 as ``below_evidence_floor``).
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from langres.core import (
    AnchorStore,
    Clusterer,
    ClusterDelta,
    CompanySchema,
    Comparator,
    ERCandidate,
    Module,
    PairwiseJudgement,
    Resolver,
    WeightedAverageJudge,
)
from langres.core.blockers.vector import VectorBlocker
from langres.core.embeddings import FakeEmbedder
from langres.core.indexes.vector_index import FAISSIndex
from langres.core.reports import ScoreInspectionReport

# --- Test fixtures: a duplicate pair (1, 2) plus two uniques (3, 4). ---------

APPLE_1 = {
    "id": "1",
    "name": "Apple Inc",
    "address": "1 Infinite Loop Cupertino",
    "phone": "408-996-1010",
}
APPLE_2 = {
    "id": "2",
    "name": "Apple Incorporated",
    "address": "1 Infinite Loop, Cupertino",
    "phone": "408-996-1010",
}
MICROSOFT = {
    "id": "3",
    "name": "Microsoft Corporation",
    "address": "1 Microsoft Way Redmond",
    "phone": "425-882-8080",
}
UMBRELLA = {
    "id": "4",
    "name": "Umbrella Corporation",
    "address": "Raccoon City Center",
    "phone": "202-555-0100",
}
RECORDS = [APPLE_1, APPLE_2, MICROSOFT, UMBRELLA]

# A new mention that clearly matches the Apple entity.
APPLE_NEW = {
    "id": "9",
    "name": "Apple Inc.",
    "address": "1 Infinite Loop Cupertino",
    "phone": "408-996-1010",
}
# A new mention that matches nothing in the anchor set.
NOVEL = {
    "id": "10",
    "name": "Nintendo Company",
    "address": "11-1 Kamitoba Kyoto",
    "phone": "075-541-6111",
}


def _string_resolver(threshold: float = 0.6) -> Resolver:
    """AllPairs + WeightedAverageJudge string pipeline (pure core, no embeddings)."""
    return Resolver.from_schema(CompanySchema, judge="string", threshold=threshold)


def _vector_resolver(threshold: float = 0.6, k_neighbors: int = 10) -> Resolver:
    """VectorBlocker + FAISS + FakeEmbedder pipeline (fast, deterministic, serializable)."""
    comparator = Comparator.from_schema(CompanySchema)
    index = FAISSIndex(embedder=FakeEmbedder(), metric="cosine")
    blocker: VectorBlocker[CompanySchema] = VectorBlocker(
        vector_index=index, schema=CompanySchema, text_field="name", k_neighbors=k_neighbors
    )
    return Resolver(
        blocker=blocker,
        comparator=comparator,
        module=WeightedAverageJudge(feature_specs=comparator.feature_specs),
        clusterer=Clusterer(threshold=threshold),
    )


class _NameJudge(Module[CompanySchema]):
    """Self-contained judge (no comparator): score 1.0 on exact name match, else 0.0.

    Reads the raw ``left``/``right`` entities directly, so a Resolver using it
    needs no comparator — exercising AnchorStore's ``comparator is None`` path.
    ``flip=True`` emits each judgement with left/right ids swapped, exercising the
    orientation-robust anchor-id extraction in ``assign``.
    """

    def __init__(self, flip: bool = False) -> None:
        self.flip = flip

    def forward(
        self, candidates: Iterator[ERCandidate[CompanySchema]]
    ) -> Iterator[PairwiseJudgement]:
        for pair in candidates:
            score = 1.0 if pair.left.name == pair.right.name else 0.0
            left_id, right_id = pair.left.id, pair.right.id
            if self.flip:
                left_id, right_id = right_id, left_id
            yield PairwiseJudgement(
                left_id=left_id,
                right_id=right_id,
                score=score,
                score_type="heuristic",
                decision_step="name_exact",
                provenance={},
            )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:  # pragma: no cover - not exercised
        raise NotImplementedError


def _name_resolver(threshold: float = 0.6) -> Resolver:
    """Resolver with NO comparator and a self-contained name judge."""
    return Resolver(
        blocker=Resolver.from_schema(CompanySchema).blocker,
        comparator=None,
        module=_NameJudge(),
        clusterer=Clusterer(threshold=threshold),
    )


# --- Build pass: stable ids + singleton coverage (D2, D3) -------------------


def test_build_covers_every_record_including_singletons() -> None:
    store = AnchorStore.build(_string_resolver(), RECORDS)
    # Every input record has an assignment (singletons NOT dropped).
    assert set(store._assignments) == {"1", "2", "3", "4"}
    # The duplicate pair collapses to one entity; the two uniques stand alone.
    assert store.entity_id_of("1") == store.entity_id_of("2")
    assert store.entity_id_of("3") != store.entity_id_of("1")
    assert store.entity_id_of("4") != store.entity_id_of("1")
    assert store.entity_id_of("3") != store.entity_id_of("4")
    # 3 distinct entities: {1,2}, {3}, {4}.
    assert len(store.entity_ids) == 3


def test_build_mints_monotonic_ids_from_prefix() -> None:
    store = AnchorStore.build(_string_resolver(), RECORDS, entity_prefix="ent")
    assert all(eid.startswith("ent") for eid in store.entity_ids)
    # Contiguous ordinals 0..2 minted in input order.
    assert store.entity_id_of("1") == "ent0"
    assert store._next_ordinal == 3


def test_build_singleton_is_assignable_against() -> None:
    """A known unique (Microsoft) must be present AND linkable, not spuriously new."""
    resolver = _string_resolver()
    resolver.build_anchor_store(RECORDS)
    delta = resolver.assign(dict(MICROSOFT, id="99"))
    assert delta.type == "link"
    assert delta.entity_id == resolver._anchor_store.entity_id_of("3")
    assert "3" in delta.matched_anchor_ids


# --- assign: link vs new (D5) ----------------------------------------------


def test_assign_links_matching_record() -> None:
    store = AnchorStore.build(_string_resolver(), RECORDS)
    delta = store.assign(APPLE_NEW)
    assert isinstance(delta, ClusterDelta)
    assert delta.type == "link"
    assert delta.entity_id == store.entity_id_of("1")
    assert set(delta.matched_anchor_ids) == {"1", "2"}
    assert delta.score is not None and delta.score >= 0.6


def test_assign_new_when_nothing_matches() -> None:
    store = AnchorStore.build(_string_resolver(), RECORDS)
    before = set(store.entity_ids)
    delta = store.assign(NOVEL)
    assert delta.type == "new"
    assert delta.entity_id not in before
    assert delta.matched_anchor_ids == []
    # A score was computed (candidates existed) but stayed below threshold.
    assert delta.score is not None and delta.score < 0.6


def test_assign_links_to_oldest_entity_on_multi_entity_match() -> None:
    """A record matching two DIFFERENT entities links to the lowest-ordinal one.

    Constructed directly so ``a`` and ``b`` are genuinely distinct entities
    (``build``'s clusterer would have merged same-name records) while a new
    record ``c`` matches both — the merge-signal path that links to the oldest.
    """
    a = {"id": "a", "name": "Globex", "address": "100 Main St Springfield", "phone": "111"}
    b = {"id": "b", "name": "Globex", "address": "999 Other Rd Ogdenville", "phone": "222"}
    store = AnchorStore(
        resolver=_name_resolver(),
        records={"a": a, "b": b},
        assignments={"a": "e0", "b": "e1"},
        anchor_ids=["a", "b"],
        next_ordinal=2,
    )
    delta = store.assign({"id": "c", "name": "Globex", "address": "x", "phone": "9"})
    assert delta.type == "link"
    assert set(delta.matched_anchor_ids) == {"a", "b"}
    # Links to the oldest (lowest-ordinal) of the two matched entities.
    assert delta.entity_id == "e0"


# --- Stable ids across repeated / interleaved assigns (D5) ------------------


def test_assign_same_record_twice_is_stable() -> None:
    store = AnchorStore.build(_string_resolver(), RECORDS)
    first = store.assign(NOVEL)
    second = store.assign(NOVEL)
    assert first.entity_id == second.entity_id
    assert second.type == "link"
    assert second.reasoning == "record id already assigned"


def test_interleaved_assigns_never_renumber_prior_ids() -> None:
    store = AnchorStore.build(_string_resolver(), RECORDS)
    a = store.assign({"id": "n1", "name": "Aaa Co", "address": "1 A St Town", "phone": "1"})
    b = store.assign({"id": "n2", "name": "Bbb Co", "address": "2 B St City", "phone": "2"})
    assert a.type == b.type == "new"
    assert a.entity_id != b.entity_id
    # Re-assigning the first, after the second was minted, returns its ORIGINAL id.
    a_again = store.assign({"id": "n1", "name": "Aaa Co", "address": "1 A St Town", "phone": "1"})
    assert a_again.entity_id == a.entity_id


def test_assign_empty_store_returns_new_with_no_score() -> None:
    """No anchors -> no candidates -> a new entity with score None."""
    store = AnchorStore.build(_string_resolver(), [])
    assert store.entity_ids == set()
    delta = store.assign(APPLE_NEW)
    assert delta.type == "new"
    assert delta.matched_anchor_ids == []
    assert delta.score is None
    assert delta.entity_id == "e0"


# --- Judge-seam robustness: no comparator + flipped pair orientation --------


def test_assign_without_comparator() -> None:
    store = AnchorStore.build(_name_resolver(), RECORDS)
    # Exact-name new record links; a novel name is new.
    linked = store.assign(dict(MICROSOFT, id="m2"))
    assert linked.type == "link"
    assert linked.entity_id == store.entity_id_of("3")
    assert store.assign(NOVEL).type == "new"


def test_assign_identifies_anchor_regardless_of_pair_orientation() -> None:
    resolver = Resolver(
        blocker=Resolver.from_schema(CompanySchema).blocker,
        comparator=None,
        module=_NameJudge(flip=True),
        clusterer=Clusterer(threshold=0.6),
    )
    store = AnchorStore.build(resolver, RECORDS)
    delta = store.assign(dict(MICROSOFT, id="m3"))
    assert delta.type == "link"
    assert "3" in delta.matched_anchor_ids
    assert delta.entity_id == store.entity_id_of("3")


# --- Resolver.assign wiring -------------------------------------------------


def test_resolver_assign_requires_build_first() -> None:
    resolver = _string_resolver()
    with pytest.raises(RuntimeError, match="build_anchor_store"):
        resolver.assign(APPLE_NEW)


def test_resolver_build_and_assign_delegates() -> None:
    resolver = _string_resolver()
    returned = resolver.build_anchor_store(RECORDS)
    assert isinstance(returned, AnchorStore)
    delta = resolver.assign(APPLE_NEW)
    assert delta.type == "link"
    assert delta.entity_id == returned.entity_id_of("1")


def test_link_and_stream_against_stubs_untouched() -> None:
    """D1: assign is a NEW method; the cross-source M5 stubs stay NotImplemented."""
    resolver = _string_resolver()
    with pytest.raises(NotImplementedError):
        resolver.link([], [])
    with pytest.raises(NotImplementedError):
        list(resolver.stream_against([]))


# --- Persistence round-trips (config-registry seam; no pickle) --------------


def test_save_load_round_trip_string_pipeline(tmp_path: Path) -> None:
    store = AnchorStore.build(_string_resolver(), RECORDS)
    before = store.assign(APPLE_NEW)

    path = tmp_path / "anchors"
    AnchorStore.build(_string_resolver(), RECORDS).save(path)
    assert (path / "anchor_store.json").exists()
    assert (path / "resolver" / "resolver.json").exists()

    loaded = AnchorStore.load(path)
    after = loaded.assign(APPLE_NEW)
    assert after.type == before.type == "link"
    assert after.entity_id == before.entity_id
    assert set(after.matched_anchor_ids) == set(before.matched_anchor_ids)


def test_save_load_round_trip_vector_pipeline(tmp_path: Path) -> None:
    store = AnchorStore.build(_vector_resolver(), RECORDS)
    before = store.assign(APPLE_NEW)
    assert before.type == "link"  # vector kNN surfaced the true anchors

    path = tmp_path / "anchors"
    AnchorStore.build(_vector_resolver(), RECORDS).save(path)
    # The built FAISS index persisted as a resolver sidecar.
    assert (path / "resolver" / "blocker").is_dir()

    loaded = AnchorStore.load(path)
    after = loaded.assign(APPLE_NEW)
    assert after.entity_id == before.entity_id
    assert after.type == "link"


# --- ClusterDelta contract --------------------------------------------------


def test_cluster_delta_reserves_future_types() -> None:
    """D5: merge/split/reject are reserved in the enum so the contract is stable."""
    for reserved in ("merge", "split", "reject"):
        delta = ClusterDelta(type=reserved, record_id="x", entity_id="e0")  # type: ignore[arg-type]
        assert delta.type == reserved
