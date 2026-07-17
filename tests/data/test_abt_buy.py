"""Tests for the Abt-Buy benchmark adapter (M4.5/W1.2 — the textual-hard dataset)."""

import pytest

from langres.data.benchmark import gold_pairs_from_clusters
from langres.core.blockers.vector import VectorBlocker
from langres.core.metrics import evaluate_blocking
from langres.core.models import ERCandidate
from langres.data.abt_buy import (
    ABT_BUY_RECALL_GATE,
    ABT_BUY_THRESHOLD_GRID,
    ACHIEVED_PC_AT_DEFAULT_K,
    DEFAULT_ABT_BUY_BLOCKING_K,
    GATE_MET,
    AbtBuyBenchmark,
    AbtBuySchema,
    _cross_source,
    _record_from_row,
    build_abt_buy_blocker,
    load_abt_buy,
    load_abt_buy_pair_splits,
    pick_blocking_k,
    sweep_blocking_k,
)

# Deterministic ground-truth counts for the vendored benchmark (verified against
# the raw CSVs): tableA=1081 Abt + tableB=1092 Buy = 2173 records; 1028 positive
# pairs across train+valid+test; connected components yield 1012 match clusters
# (sizes 2-3) + 133 singletons.
_N_ABT = 1081
_N_BUY = 1092
_N_CORPUS = _N_ABT + _N_BUY
_N_GOLD_PAIRS = 1028
_N_MATCH_CLUSTERS = 1012
_N_SINGLETONS = 133


# --- loader: fast, no embeddings -------------------------------------------------


def test_load_returns_full_corpus() -> None:
    corpus, _gold, _pairs = load_abt_buy()
    assert len(corpus) == _N_CORPUS
    assert sum(1 for r in corpus if r.source == "abt") == _N_ABT
    assert sum(1 for r in corpus if r.source == "buy") == _N_BUY


def test_corpus_ids_are_globally_unique_and_source_prefixed() -> None:
    corpus, _gold, _pairs = load_abt_buy()
    ids = [r.id for r in corpus]
    assert len(set(ids)) == len(ids)
    assert all(r.id.startswith("a") for r in corpus if r.source == "abt")
    assert all(r.id.startswith("b") for r in corpus if r.source == "buy")


def test_gold_pairs_count_equals_positive_labels() -> None:
    _corpus, _gold, gold_pairs = load_abt_buy()
    assert len(gold_pairs) == _N_GOLD_PAIRS
    splits = load_abt_buy_pair_splits()
    positives = {
        frozenset({left, right})
        for split in splits.values()
        for left, right, label in split
        if label == 1
    }
    assert positives == gold_pairs


def test_all_gold_matches_are_cross_source() -> None:
    corpus, _gold, gold_pairs = load_abt_buy()
    id_to_source = {r.id: r.source for r in corpus}
    for pair in gold_pairs:
        left, right = sorted(pair)
        assert left in id_to_source and right in id_to_source
        assert id_to_source[left] != id_to_source[right]


def test_gold_clusters_are_complete_partition() -> None:
    corpus, gold_clusters, _pairs = load_abt_buy()
    covered = [rid for cluster in gold_clusters for rid in cluster]
    assert len(covered) == len(set(covered))
    assert sorted(covered) == sorted(r.id for r in corpus)
    match_clusters = [c for c in gold_clusters if len(c) >= 2]
    singletons = [c for c in gold_clusters if len(c) == 1]
    assert len(match_clusters) == _N_MATCH_CLUSTERS
    assert len(singletons) == _N_SINGLETONS


def test_gold_clusters_cover_every_gold_pair() -> None:
    _corpus, gold_clusters, gold_pairs = load_abt_buy()
    cluster_of = {rid: i for i, cluster in enumerate(gold_clusters) for rid in cluster}
    for pair in gold_pairs:
        left, right = tuple(pair)
        assert cluster_of[left] == cluster_of[right]


# --- pair splits -----------------------------------------------------------------


def test_pair_splits_have_three_named_splits_with_prefixed_ids() -> None:
    splits = load_abt_buy_pair_splits()
    assert set(splits) == {"train", "valid", "test"}
    for rows in splits.values():
        assert rows
        for left, right, label in rows:
            assert left.startswith("a")
            assert right.startswith("b")
            assert label in (0, 1)
        assert any(label == 0 for _, _, label in rows)
        assert any(label == 1 for _, _, label in rows)


# --- schema / embed_text ---------------------------------------------------------


def test_embed_text_composition_order() -> None:
    r = AbtBuySchema(id="a1", name="sony turntable", description="belt drive", source="abt")
    assert r.embed_text == "sony turntable belt drive"


def test_embed_text_omits_missing_description() -> None:
    r = AbtBuySchema(id="b1", name="solo name", source="buy")
    assert r.embed_text == "solo name"


def test_embed_text_serializes_as_computed_field() -> None:
    r = AbtBuySchema(id="a9", name="widget", description="a fine widget", source="abt")
    assert r.model_dump()["embed_text"] == "widget a fine widget"


# --- private helpers: edge branches not exercised by clean real data -------------


def test_record_from_row_handles_empty_and_missing_fields() -> None:
    row = {"id": " 7 ", "name": "", "description": "", "price": ""}
    rec = _record_from_row(row, "buy", "b")
    assert rec.id == "b7"
    assert rec.name == ""
    assert rec.description is None
    assert rec.price is None
    assert rec.source == "buy"


def test_record_from_row_preserves_present_fields() -> None:
    row = {"id": "3", "name": "sony turntable", "description": "belt drive", "price": "199.99"}
    rec = _record_from_row(row, "abt", "a")
    assert rec.id == "a3"
    assert rec.name == "sony turntable"
    assert rec.description == "belt drive"
    assert rec.price == "199.99"


def test_cross_source_filters_intra_source_pairs() -> None:
    a1 = AbtBuySchema(id="a1", name="x", source="abt")
    a2 = AbtBuySchema(id="a2", name="y", source="abt")
    b1 = AbtBuySchema(id="b1", name="z", source="buy")
    cross = ERCandidate(left=a1, right=b1, blocker_name="t")
    same = ERCandidate(left=a1, right=a2, blocker_name="t")
    assert _cross_source([cross, same]) == [cross]


def test_cross_source_all_same_source_returns_empty() -> None:
    a1 = AbtBuySchema(id="a1", name="x", source="abt")
    a2 = AbtBuySchema(id="a2", name="y", source="abt")
    same = ERCandidate(left=a1, right=a2, blocker_name="t")
    assert _cross_source([same]) == []


# --- AbtBuyBenchmark conformer: fast contract tests (no embeddings) --------------


def test_benchmark_exposes_pinned_config() -> None:
    bench = AbtBuyBenchmark()
    assert bench.name == "abt_buy"
    assert bench.threshold_grid == ABT_BUY_THRESHOLD_GRID
    assert bench.schema is AbtBuySchema
    assert bench.blocking_k == DEFAULT_ABT_BUY_BLOCKING_K


def test_benchmark_load_matches_loader_with_closure_pairs() -> None:
    bench = AbtBuyBenchmark()
    corpus, gold_clusters, gold_pairs = bench.load()

    base_corpus, base_clusters, _base_pairs = load_abt_buy()
    assert [r.id for r in corpus] == [r.id for r in base_corpus]
    assert gold_clusters == base_clusters
    assert gold_pairs == gold_pairs_from_clusters(gold_clusters)


def test_benchmark_split_is_leakage_free_and_partitions() -> None:
    bench = AbtBuyBenchmark()
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
    bench = AbtBuyBenchmark()
    corpus, gold_clusters, _pairs = bench.load()
    a = bench.split(corpus, gold_clusters, seed=0)
    b = bench.split(corpus, gold_clusters, seed=0)
    assert [r.id for r in a[0]] == [r.id for r in b[0]]
    assert [r.id for r in a[1]] == [r.id for r in b[1]]


def test_benchmark_build_blocker_returns_fresh_unbuilt_vector_blocker() -> None:
    bench = AbtBuyBenchmark()
    b1 = bench.build_blocker(7)
    b2 = bench.build_blocker(7)
    assert isinstance(b1, VectorBlocker)
    assert b1.k_neighbors == 7
    assert b1 is not b2


# --- pick_blocking_k: pure, fast branches ---------------------------------------


def test_pick_blocking_k_returns_min_passing() -> None:
    assert pick_blocking_k({5: 0.85, 10: 0.91, 20: 0.95}) == 10


def test_pick_blocking_k_falls_back_to_best_when_none_pass() -> None:
    assert pick_blocking_k({5: 0.76, 10: 0.81, 30: 0.84}) == 30


def test_pick_blocking_k_custom_threshold() -> None:
    assert pick_blocking_k({5: 0.80, 10: 0.85}, threshold=0.85) == 10


def test_pick_blocking_k_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        pick_blocking_k({})


def test_default_blocking_k_is_pinned_with_honest_gate_outcome() -> None:
    assert DEFAULT_ABT_BUY_BLOCKING_K == 20
    assert ABT_BUY_RECALL_GATE == 0.90
    # Unlike Amazon-Google, the Abt-Buy k-sweep clears the 0.90 gate at k=20.
    assert ACHIEVED_PC_AT_DEFAULT_K == pytest.approx(0.9301, abs=1e-4)
    assert GATE_MET is True


# --- slow: real embeddings, runs in CI --------------------------------------------


@pytest.mark.slow
def test_build_abt_buy_blocker_returns_default_k() -> None:
    blocker = build_abt_buy_blocker()
    assert blocker.k_neighbors == DEFAULT_ABT_BUY_BLOCKING_K


@pytest.mark.slow
def test_sweep_blocking_k_pins_documented_pair_completeness() -> None:
    corpus, gold, _pairs = load_abt_buy()
    ks = (5, 10, 20, 30, 50)
    recalls = sweep_blocking_k(corpus, gold, ks=ks)

    assert set(recalls) == set(ks)
    assert all(0.0 <= v <= 1.0 for v in recalls.values())

    chosen = pick_blocking_k(recalls, ABT_BUY_RECALL_GATE)
    assert chosen == DEFAULT_ABT_BUY_BLOCKING_K
    assert recalls[chosen] == pytest.approx(ACHIEVED_PC_AT_DEFAULT_K, abs=5e-3)
    # The gate is honestly met: the chosen k is the smallest one clearing it.
    assert recalls[chosen] >= ABT_BUY_RECALL_GATE


@pytest.mark.slow
def test_pair_completeness_is_computable_via_evaluate_blocking() -> None:
    corpus, gold, _pairs = load_abt_buy()
    recalls = sweep_blocking_k(corpus, gold, ks=(DEFAULT_ABT_BUY_BLOCKING_K,))
    # The shared build_abt_buy_blocker factory must measure the same
    # Pair-Completeness as the sweep. Its FAISSIndex ships unbuilt (embedding is
    # deferred to resolve-time), so build it here, then measure PC directly via
    # evaluate_blocking -- confirming PC is a plain candidate_recall measurement.
    blocker = build_abt_buy_blocker(DEFAULT_ABT_BUY_BLOCKING_K)
    records = [r.model_dump() for r in corpus]
    blocker.vector_index.create_index([r.embed_text for r in corpus])
    candidates = _cross_source(list(blocker.stream(records)))
    pc = evaluate_blocking(candidates, gold).candidate_recall
    assert pc == pytest.approx(recalls[DEFAULT_ABT_BUY_BLOCKING_K], abs=1e-6)
