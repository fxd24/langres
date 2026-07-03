"""Tests for the FEBRL person benchmark adapter (M5 W2.1 — second entity type)."""

import pytest

from langres.core.benchmark import gold_pairs_from_clusters
from langres.core.blockers.vector import VectorBlocker
from langres.core.metrics import evaluate_blocking
from langres.core.models import ERCandidate
from langres.data.febrl_person import (
    ACHIEVED_PC_AT_DEFAULT_K,
    DEFAULT_PERSON_BLOCKING_K,
    GATE_MET,
    PERSON_RECALL_GATE,
    PERSON_THRESHOLD_GRID,
    FebrlPersonBenchmark,
    PersonSchema,
    _cross_source,
    _record_from_row,
    build_person_blocker,
    load_febrl_person,
    pick_blocking_k,
    sweep_blocking_k,
)

# Deterministic ground-truth counts for the vendored 500/side FEBRL4 subset: a
# clean 1:1 linkage, so 500 originals + 500 duplicates = 1000 records, 500 gold
# pairs, and (2-element match sets) 500 match clusters with 0 singletons (every
# record is matched).
_N_A = 500
_N_B = 500
_N_CORPUS = _N_A + _N_B
_N_GOLD_PAIRS = 500
_N_MATCH_CLUSTERS = 500
_N_SINGLETONS = 0


# --- loader: fast, no embeddings -------------------------------------------------


def test_load_returns_full_corpus() -> None:
    corpus, _gold, _pairs = load_febrl_person()
    assert len(corpus) == _N_CORPUS
    assert sum(1 for r in corpus if r.source == "a") == _N_A
    assert sum(1 for r in corpus if r.source == "b") == _N_B


def test_corpus_ids_are_globally_unique_and_source_prefixed() -> None:
    corpus, _gold, _pairs = load_febrl_person()
    ids = [r.id for r in corpus]
    assert len(set(ids)) == len(ids)
    assert all(r.id.startswith("a") for r in corpus if r.source == "a")
    assert all(r.id.startswith("b") for r in corpus if r.source == "b")


def test_gold_pairs_count_and_all_cross_source() -> None:
    corpus, _gold, gold_pairs = load_febrl_person()
    assert len(gold_pairs) == _N_GOLD_PAIRS
    id_to_source = {r.id: r.source for r in corpus}
    for pair in gold_pairs:
        left, right = sorted(pair)
        assert left in id_to_source and right in id_to_source
        assert id_to_source[left] != id_to_source[right]


def test_gold_clusters_are_complete_partition() -> None:
    corpus, gold_clusters, _pairs = load_febrl_person()
    covered = [rid for cluster in gold_clusters for rid in cluster]
    assert len(covered) == len(set(covered))
    assert sorted(covered) == sorted(r.id for r in corpus)
    match_clusters = [c for c in gold_clusters if len(c) >= 2]
    singletons = [c for c in gold_clusters if len(c) == 1]
    assert len(match_clusters) == _N_MATCH_CLUSTERS
    assert len(singletons) == _N_SINGLETONS


def test_gold_clusters_cover_every_gold_pair() -> None:
    _corpus, gold_clusters, gold_pairs = load_febrl_person()
    cluster_of = {rid: i for i, cluster in enumerate(gold_clusters) for rid in cluster}
    for pair in gold_pairs:
        left, right = tuple(pair)
        assert cluster_of[left] == cluster_of[right]


def test_gold_pairs_equal_partition_closure() -> None:
    _corpus, gold_clusters, gold_pairs = load_febrl_person()
    # For a 1:1 linkage the loader's gold_pairs equal the within-cluster closure.
    assert gold_pairs == gold_pairs_from_clusters(gold_clusters)


# --- schema / embed_text ---------------------------------------------------------


def test_embed_text_composition_order() -> None:
    r = PersonSchema(id="a1", given_name="rachael", surname="dent", suburb="byford", source="a")
    assert r.embed_text == "rachael dent byford"


def test_embed_text_omits_missing_fields() -> None:
    r = PersonSchema(id="b1", given_name="elton", source="b")
    assert r.embed_text == "elton"


def test_embed_text_serializes_as_computed_field() -> None:
    r = PersonSchema(id="a9", given_name="ana", surname="lee", suburb="perth", source="a")
    assert r.model_dump()["embed_text"] == "ana lee perth"


# --- private helpers: edge branches not exercised by clean real data -------------


def test_record_from_row_handles_empty_and_missing_fields() -> None:
    row = {"rec_id": " 7 ", "given_name": "", "surname": "  "}
    rec = _record_from_row(row, "b", "b")
    assert rec.id == "b7"
    assert rec.given_name is None
    assert rec.surname is None
    # Fields absent from the row map to None too.
    assert rec.soc_sec_id is None
    assert rec.source == "b"


def test_record_from_row_preserves_present_fields() -> None:
    row = {
        "rec_id": "3",
        "given_name": "michaela",
        "surname": "neumann",
        "street_number": "8",
        "address_1": "stanley street",
        "address_2": "miami",
        "suburb": "winston hills",
        "postcode": "4223",
        "state": "nsw",
        "date_of_birth": "19151111",
        "soc_sec_id": "5304218",
    }
    rec = _record_from_row(row, "a", "a")
    assert rec.id == "a3"
    assert rec.given_name == "michaela"
    assert rec.surname == "neumann"
    assert rec.street_number == "8"
    assert rec.address_1 == "stanley street"
    assert rec.address_2 == "miami"
    assert rec.suburb == "winston hills"
    assert rec.postcode == "4223"
    assert rec.state == "nsw"
    assert rec.date_of_birth == "19151111"
    assert rec.soc_sec_id == "5304218"
    assert rec.source == "a"


def test_cross_source_filters_intra_source_pairs() -> None:
    a1 = PersonSchema(id="a1", given_name="x", source="a")
    a2 = PersonSchema(id="a2", given_name="y", source="a")
    b1 = PersonSchema(id="b1", given_name="z", source="b")
    cross = ERCandidate(left=a1, right=b1, blocker_name="t")
    same = ERCandidate(left=a1, right=a2, blocker_name="t")
    assert _cross_source([cross, same]) == [cross]


def test_cross_source_all_same_source_returns_empty() -> None:
    a1 = PersonSchema(id="a1", given_name="x", source="a")
    a2 = PersonSchema(id="a2", given_name="y", source="a")
    same = ERCandidate(left=a1, right=a2, blocker_name="t")
    assert _cross_source([same]) == []


# --- FebrlPersonBenchmark conformer: fast contract tests (no embeddings) ----------


def test_benchmark_exposes_pinned_config() -> None:
    bench = FebrlPersonBenchmark()
    assert bench.name == "febrl_person"
    assert bench.threshold_grid == PERSON_THRESHOLD_GRID
    assert bench.schema is PersonSchema
    assert bench.blocking_k == DEFAULT_PERSON_BLOCKING_K


def test_benchmark_load_matches_loader_with_closure_pairs() -> None:
    bench = FebrlPersonBenchmark()
    corpus, gold_clusters, gold_pairs = bench.load()

    base_corpus, base_clusters, _base_pairs = load_febrl_person()
    assert [r.id for r in corpus] == [r.id for r in base_corpus]
    assert gold_clusters == base_clusters
    assert gold_pairs == gold_pairs_from_clusters(gold_clusters)


def test_benchmark_split_is_leakage_free_and_partitions() -> None:
    bench = FebrlPersonBenchmark()
    corpus, gold_clusters, _pairs = bench.load()
    train_recs, test_recs, train_cls, test_cls = bench.split(corpus, gold_clusters, seed=0)

    train_ids = {r.id for r in train_recs}
    test_ids = {r.id for r in test_recs}
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == {r.id for r in corpus}
    assert {rid for c in train_cls for rid in c} == train_ids
    assert {rid for c in test_cls for rid in c} == test_ids
    assert len(train_cls) + len(test_cls) == len(gold_clusters)


def test_benchmark_split_is_deterministic() -> None:
    bench = FebrlPersonBenchmark()
    corpus, gold_clusters, _pairs = bench.load()
    a = bench.split(corpus, gold_clusters, seed=0)
    b = bench.split(corpus, gold_clusters, seed=0)
    assert [r.id for r in a[0]] == [r.id for r in b[0]]
    assert [r.id for r in a[1]] == [r.id for r in b[1]]


def test_benchmark_build_blocker_returns_fresh_unbuilt_vector_blocker() -> None:
    bench = FebrlPersonBenchmark()
    b1 = bench.build_blocker(7)
    b2 = bench.build_blocker(7)
    assert isinstance(b1, VectorBlocker)
    assert b1.k_neighbors == 7
    assert b1 is not b2


# --- pick_blocking_k: pure, fast branches ---------------------------------------


def test_pick_blocking_k_returns_min_passing() -> None:
    assert pick_blocking_k({5: 0.96, 10: 0.97, 20: 0.98}) == 5


def test_pick_blocking_k_falls_back_to_best_when_none_pass() -> None:
    assert pick_blocking_k({5: 0.80, 10: 0.88, 20: 0.90}) == 20


def test_pick_blocking_k_custom_threshold() -> None:
    assert pick_blocking_k({5: 0.90, 10: 0.96}, threshold=0.96) == 10


def test_pick_blocking_k_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        pick_blocking_k({})


def test_default_blocking_k_is_pinned_with_honest_gate_outcome() -> None:
    assert DEFAULT_PERSON_BLOCKING_K == 5
    assert PERSON_RECALL_GATE == 0.95
    assert ACHIEVED_PC_AT_DEFAULT_K == pytest.approx(0.9660, abs=1e-4)
    # Like Fodors-Zagat, the person k-sweep clears the 0.95 gate at k=5.
    assert GATE_MET is True


# --- slow: real embeddings, runs in CI --------------------------------------------


@pytest.mark.slow
def test_build_person_blocker_returns_default_k() -> None:
    blocker = build_person_blocker()
    assert blocker.k_neighbors == DEFAULT_PERSON_BLOCKING_K


@pytest.mark.slow
def test_sweep_blocking_k_pins_documented_pair_completeness() -> None:
    corpus, gold, _pairs = load_febrl_person()
    ks = (5, 10, 20, 30, 50)
    recalls = sweep_blocking_k(corpus, gold, ks=ks)

    assert set(recalls) == set(ks)
    assert all(0.0 <= v <= 1.0 for v in recalls.values())

    chosen = pick_blocking_k(recalls, PERSON_RECALL_GATE)
    assert chosen == DEFAULT_PERSON_BLOCKING_K
    assert recalls[chosen] == pytest.approx(ACHIEVED_PC_AT_DEFAULT_K, abs=5e-3)
    # The gate is honestly met: the chosen k is the smallest one clearing it.
    assert recalls[chosen] >= PERSON_RECALL_GATE


@pytest.mark.slow
def test_pair_completeness_is_computable_via_evaluate_blocking() -> None:
    corpus, gold, _pairs = load_febrl_person()
    recalls = sweep_blocking_k(corpus, gold, ks=(DEFAULT_PERSON_BLOCKING_K,))
    blocker = build_person_blocker(DEFAULT_PERSON_BLOCKING_K)
    records = [r.model_dump() for r in corpus]
    blocker.vector_index.create_index([r.embed_text for r in corpus])
    candidates = _cross_source(list(blocker.stream(records)))
    pc = evaluate_blocking(candidates, gold).candidate_recall
    assert pc == pytest.approx(recalls[DEFAULT_PERSON_BLOCKING_K], abs=1e-6)
