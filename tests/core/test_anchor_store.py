"""Tests for AnchorStore / ClusterDelta — incremental single-record assign (W2.2).

Two pipelines are exercised:

- The **string** pipeline (``AllPairsBlocker`` + ``WeightedAverageMatcher``) — pure
  core, no embeddings — drives the store logic, singleton coverage, stable ids,
  and the all-pairs candidate path, plus a fresh-in-process save/load round-trip.
- The **vector** pipeline (``VectorBlocker`` + ``FAISSIndex`` + ``FakeEmbedder``)
  is fast and deterministic (no model download) and exercises the single-record
  kNN candidate path over ``index.search`` and the FAISS-sidecar round-trip.

Records carry name + address + phone so the WeightedAverageMatcher's evidence floor
is cleared (a single present feature scores 0 as ``below_evidence_floor``).
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from langres.core import (
    Clusterer,
    CompanySchema,
    Comparator,
    ERCandidate,
    Matcher,
    PairwiseJudgement,
    Resolver,
)
from langres.core.anchor_store import AnchorStore, ClusterDelta
from langres.core.blockers import AllPairsBlocker, CompositeBlocker, KeyBlocker
from langres.core.matchers import EmbeddingScoreMatcher, WeightedAverageMatcher
from langres.core.anchor_store import _schema_factory
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
    """AllPairs + WeightedAverageMatcher string pipeline (pure core, no embeddings)."""
    return Resolver.from_schema(CompanySchema, matcher="string", threshold=threshold)


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
        matcher=WeightedAverageMatcher(feature_specs=comparator.feature_specs),
        clusterer=Clusterer(threshold=threshold),
    )


class _NameJudge(Matcher[CompanySchema]):
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
        matcher=_NameJudge(),
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
    # The idempotent path returns no fresh evidence (documented contract).
    assert second.matched_anchor_ids == []
    assert second.score is None


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
        matcher=_NameJudge(flip=True),
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


def test_assignments_property_is_a_read_only_copy() -> None:
    store = AnchorStore.build(_string_resolver(), RECORDS)
    snapshot = store.assignments
    assert snapshot == {"1": "e0", "2": "e0", "3": "e1", "4": "e2"}
    snapshot["1"] = "TAMPERED"  # mutating the copy must not touch the store
    assert store.entity_id_of("1") == "e0"


# --- Review fixes: unsupported blocker, -1 padding, documented boundary ------


def test_schema_factory_unreachable_raises() -> None:
    """A blocker with neither a schema_factory nor schema-bearing children errors."""

    class _NoFactoryBlocker:
        type_name = "no_factory"

    with pytest.raises(NotImplementedError, match="schema_factory"):
        _schema_factory(_NoFactoryBlocker())


def test_schema_factory_skips_children_without_factory() -> None:
    """Recursion skips a factory-less child and returns a later child's factory."""

    class _Stub:
        pass

    good = AllPairsBlocker(schema=CompanySchema)

    class _FakeComposite:
        children = [_Stub(), good]

    assert _schema_factory(_FakeComposite()) is good.schema_factory


def test_composite_blocker_reaches_child_schema_factory() -> None:
    """Recall-first CompositeBlocker (KeyBlocker union AllPairs) works via a child."""
    composite: CompositeBlocker[CompanySchema] = CompositeBlocker(
        [
            KeyBlocker(schema=CompanySchema, key_field="phone"),
            AllPairsBlocker(schema=CompanySchema),
        ],
        op="union",
    )
    comparator = Comparator.from_schema(CompanySchema)
    resolver = Resolver(
        blocker=composite,
        comparator=comparator,
        matcher=WeightedAverageMatcher(feature_specs=comparator.feature_specs),
        clusterer=Clusterer(threshold=0.6),
    )
    store = resolver.build_anchor_store(RECORDS)
    assert set(store.assignments) == {"1", "2", "3", "4"}
    assert resolver.assign(APPLE_NEW).type == "link"


def test_nested_composite_blocker_supports_build_and_vector_assign() -> None:
    """A CompositeBlocker-of-CompositeBlocker with a nested VectorBlocker works.

    Mirrors #66's nested-composite index build: both the schema-factory
    resolution (``build``) and the vector-candidate-source lookup (``assign``)
    must recurse through composite-of-composites, not just one level. The
    VectorBlocker sits at depth 2, so a single-level lookup would miss it —
    ``build`` would raise NotImplementedError and ``assign`` would silently fall
    back to the all-anchors (``similarity=None``) path.
    """
    index = FAISSIndex(embedder=FakeEmbedder(), metric="cosine")
    vblocker: VectorBlocker[CompanySchema] = VectorBlocker(
        vector_index=index, schema=CompanySchema, text_field="name", k_neighbors=10
    )
    # depth 2: outer( inner( vblocker, KeyBlocker ), KeyBlocker )
    inner: CompositeBlocker[CompanySchema] = CompositeBlocker(
        [vblocker, KeyBlocker(schema=CompanySchema, key_field="phone")], op="union"
    )
    outer: CompositeBlocker[CompanySchema] = CompositeBlocker(
        [inner, KeyBlocker(schema=CompanySchema, key_field="address")], op="union"
    )
    comparator = Comparator.from_schema(CompanySchema)
    resolver = Resolver(
        blocker=outer,
        comparator=comparator,
        matcher=WeightedAverageMatcher(feature_specs=comparator.feature_specs),
        clusterer=Clusterer(threshold=0.6),
    )

    # schema_factory recursion: every record is normalized and anchored.
    store = resolver.build_anchor_store(RECORDS)
    assert set(store.assignments) == {"1", "2", "3", "4"}

    # vector-source recursion: candidate anchors come from the nested index and
    # therefore carry a real similarity (the None fallback would mean the walk
    # never reached the depth-2 VectorBlocker).
    entity = _schema_factory(outer)(APPLE_NEW)
    pairs = store._candidate_anchors(entity)
    assert pairs and all(similarity is not None for _, similarity in pairs)

    # And end-to-end assign still links through the nesting.
    assert resolver.assign(APPLE_NEW).type == "link"


def test_assign_does_not_store_non_anchor_records(tmp_path: Path) -> None:
    """assign() must not accumulate raw non-anchor records (unbounded-growth guard).

    ``_records`` holds the anchor corpus fixed at build time; a newly-assigned
    record's payload is never read back (``_candidate`` only reads anchors), so
    assign() must grow only the ``record_id -> entity_id`` map, not the record
    store — otherwise a long incremental-ingestion stream bloats memory and
    ``anchor_store.json`` without bound.
    """
    import json

    store = AnchorStore.build(_string_resolver(), RECORDS)
    anchors = set(store._records)
    assert anchors == {"1", "2", "3", "4"}

    for i in range(5):
        delta = store.assign(
            {"id": f"new-{i}", "name": f"Zzz {i} Co", "address": f"{i} Z St Ville", "phone": str(i)}
        )
        assert delta.type == "new"

    # The raw-record store did NOT grow by the assigned records...
    assert set(store._records) == anchors
    # ...but the (cheap) id map did — idempotency is preserved.
    assert {"new-0", "new-4"} <= set(store.assignments)

    # Persisted manifest carries anchors only, not the assign stream.
    path = tmp_path / "anchors"
    store.save(path)
    payload = json.loads((path / "anchor_store.json").read_text())
    assert set(payload["records"]) == anchors
    assert "new-0" in payload["assignments"]  # id map still round-trips


def test_embedding_judge_assign_gets_similarity_score() -> None:
    """assign() must attach similarity_score so EmbeddingScoreMatcher can score."""
    anchors = [
        {"id": "1", "name": "Apple", "address": "a", "phone": "1"},
        {"id": "2", "name": "Microsoft", "address": "b", "phone": "2"},
        {"id": "3", "name": "Umbrella", "address": "c", "phone": "3"},
    ]
    index = FAISSIndex(embedder=FakeEmbedder(), metric="cosine")
    blocker: VectorBlocker[CompanySchema] = VectorBlocker(
        vector_index=index, schema=CompanySchema, text_field="name", k_neighbors=10
    )
    # EmbeddingScoreMatcher scores purely off similarity_score (no comparator).
    resolver = Resolver(
        blocker=blocker,
        comparator=None,
        matcher=EmbeddingScoreMatcher(threshold=0.9),
        clusterer=Clusterer(threshold=0.9),
    )
    store = resolver.build_anchor_store(anchors)
    # Same text as anchor "1" -> FakeEmbedder yields an identical vector ->
    # cosine similarity 1.0 -> links (would raise ValueError if score were None).
    delta = resolver.assign({"id": "9", "name": "Apple", "address": "z", "phone": "9"})
    assert delta.type == "link"
    assert delta.matched_anchor_ids == ["1"]
    assert delta.entity_id == store.entity_id_of("1")
    assert delta.score is not None and delta.score >= 0.9


class _MinusOnePaddingIndex:
    """Fake vector index whose ``search`` pads with -1, like a Qdrant fusion query."""

    type_name = "faiss_index"

    def __init__(self) -> None:
        self._corpus_texts: list[str] | None = None

    def create_index(self, texts: list[str]) -> None:
        self._corpus_texts = list(texts)

    def search(
        self, query_texts: Any, k: int, query_prompt: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        # First slot is -1 padding; second points at anchor position 0.
        return np.array([0.9, 0.1]), np.array([-1, 0])

    def to_similarities(self, distances: np.ndarray) -> np.ndarray:
        return np.clip(distances, 0.0, 1.0)


def test_vector_search_skips_minus_one_padding() -> None:
    """-1 padding must be skipped, not wrapped to the last anchor (Qdrant safety)."""
    anchor_a = {"id": "a", "name": "Globex", "address": "1 A St", "phone": "1"}
    anchor_b = {"id": "b", "name": "Initech", "address": "2 B St", "phone": "2"}
    blocker: VectorBlocker[CompanySchema] = VectorBlocker(
        vector_index=_MinusOnePaddingIndex(),  # type: ignore[arg-type]
        schema=CompanySchema,
        text_field="name",
        k_neighbors=2,
    )
    resolver = Resolver(
        blocker=blocker, comparator=None, matcher=_NameJudge(), clusterer=Clusterer(threshold=0.6)
    )
    store = AnchorStore(
        resolver=resolver,
        records={"a": anchor_a, "b": anchor_b},
        assignments={"a": "e0", "b": "e1"},
        anchor_ids=["a", "b"],
        next_ordinal=2,
    )
    # Query name matches anchor "a" (position 0); the -1 slot must be dropped,
    # NOT wrapped to anchor_ids[-1] == "b".
    delta = store.assign({"id": "c", "name": "Globex", "address": "x", "phone": "9"})
    assert delta.type == "link"
    assert delta.matched_anchor_ids == ["a"]
    assert delta.entity_id == "e0"


def test_build_rejects_duplicate_ids() -> None:
    dup = [APPLE_1, dict(APPLE_2, id="1")]  # both id "1"
    with pytest.raises(ValueError, match="unique record ids"):
        AnchorStore.build(_string_resolver(), dup)


def test_load_rejects_incompatible_store_version(tmp_path: Path) -> None:
    import json

    path = tmp_path / "anchors"
    AnchorStore.build(_string_resolver(), RECORDS).save(path)
    manifest_path = path / "anchor_store.json"
    payload = json.loads(manifest_path.read_text())
    payload["store_version"] = "999"
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="version"):
        AnchorStore.load(path)


def test_new_records_are_not_added_to_the_searchable_set() -> None:
    """Documented W2.2 boundary: two distinct new ids that duplicate each other
    each mint a NEW entity (assign matches only the build-time anchor set)."""
    store = AnchorStore.build(_string_resolver(), RECORDS)
    first = store.assign(dict(NOVEL, id="dup-1"))
    second = store.assign(dict(NOVEL, id="dup-2"))  # same content, different id
    assert first.type == second.type == "new"
    assert first.entity_id != second.entity_id
