"""Tests for the Amazon-Google benchmark adapter and blocking k-sweep."""

import pytest

from langres.core.benchmark import gold_pairs_from_clusters
from langres.core.blockers.vector import VectorBlocker
from langres.core.metrics import evaluate_blocking
from langres.core.models import ERCandidate
from langres.data.amazon_google import (
    ACHIEVED_PC_AT_DEFAULT_K,
    AG_RECALL_GATE,
    DEFAULT_AG_BLOCKING_K,
    DEFAULT_AG_THRESHOLD_GRID,
    GATE_MET,
    AmazonGoogleBenchmark,
    ProductSchema,
    _clusters_from_pairs,
    _cross_source,
    _record_from_row,
    build_product_blocker,
    load_amazon_google,
    load_amazon_google_pair_splits,
    pick_blocking_k,
    sweep_blocking_k,
)

# Deterministic ground-truth counts for the vendored benchmark (verified against
# the raw CSVs): tableA=1363 Amazon + tableB=3226 Google = 4589 records; 1167
# positive pairs across train+valid+test; connected components yield 995 match
# clusters (sizes 2-6, since the task is many-to-many) + 2428 singletons.
_N_AMAZON = 1363
_N_GOOGLE = 3226
_N_CORPUS = _N_AMAZON + _N_GOOGLE
_N_GOLD_PAIRS = 1167
_N_MATCH_CLUSTERS = 995
_N_SINGLETONS = 2428


# --- loader: fast, no embeddings -------------------------------------------------


def test_load_returns_full_corpus() -> None:
    corpus, _gold, _pairs = load_amazon_google()
    assert len(corpus) == _N_CORPUS
    assert sum(1 for r in corpus if r.source == "amazon") == _N_AMAZON
    assert sum(1 for r in corpus if r.source == "google") == _N_GOOGLE


def test_corpus_ids_are_globally_unique_and_source_prefixed() -> None:
    corpus, _gold, _pairs = load_amazon_google()
    ids = [r.id for r in corpus]
    assert len(set(ids)) == len(ids)
    assert all(r.id.startswith("a") for r in corpus if r.source == "amazon")
    assert all(r.id.startswith("g") for r in corpus if r.source == "google")


def test_gold_pairs_count_equals_positive_labels() -> None:
    _corpus, _gold, gold_pairs = load_amazon_google()
    assert len(gold_pairs) == _N_GOLD_PAIRS
    # Pooled positive labels across the three splits must match the pair count.
    splits = load_amazon_google_pair_splits()
    positives = {
        frozenset({left, right})
        for split in splits.values()
        for left, right, label in split
        if label == 1
    }
    assert positives == gold_pairs


def test_all_gold_matches_are_cross_source() -> None:
    corpus, _gold, gold_pairs = load_amazon_google()
    id_to_source = {r.id: r.source for r in corpus}
    for pair in gold_pairs:
        left, right = sorted(pair)
        assert left in id_to_source and right in id_to_source
        assert id_to_source[left] != id_to_source[right]


def test_gold_clusters_are_complete_partition() -> None:
    corpus, gold_clusters, _pairs = load_amazon_google()
    # Every corpus id appears in exactly one cluster (a true closed-world partition).
    covered = [rid for cluster in gold_clusters for rid in cluster]
    assert len(covered) == len(set(covered))  # no id in two clusters
    assert sorted(covered) == sorted(r.id for r in corpus)
    match_clusters = [c for c in gold_clusters if len(c) >= 2]
    singletons = [c for c in gold_clusters if len(c) == 1]
    assert len(match_clusters) == _N_MATCH_CLUSTERS
    assert len(singletons) == _N_SINGLETONS
    # Amazon-Google is many-to-many: at least one cluster exceeds a simple pair.
    assert max(len(c) for c in match_clusters) > 2


def test_gold_clusters_cover_every_gold_pair() -> None:
    _corpus, gold_clusters, gold_pairs = load_amazon_google()
    cluster_of = {rid: i for i, cluster in enumerate(gold_clusters) for rid in cluster}
    # Both endpoints of every gold pair land in the same cluster (transitive closure).
    for pair in gold_pairs:
        left, right = tuple(pair)
        assert cluster_of[left] == cluster_of[right]


# --- pair splits -----------------------------------------------------------------


def test_pair_splits_have_three_named_splits_with_prefixed_ids() -> None:
    splits = load_amazon_google_pair_splits()
    assert set(splits) == {"train", "valid", "test"}
    for rows in splits.values():
        assert rows  # non-empty
        for left, right, label in rows:
            assert left.startswith("a")
            assert right.startswith("g")
            assert label in (0, 1)
        # Each split must retain both classes; a loader regression that silently
        # dropped all negatives (or all positives) would otherwise pass.
        assert any(label == 0 for _, _, label in rows)
        assert any(label == 1 for _, _, label in rows)


# --- schema / embed_text ---------------------------------------------------------


def test_embed_text_composition_order() -> None:
    r = ProductSchema(id="a1", title="quickbooks pro 2007", manufacturer="intuit", source="amazon")
    assert r.embed_text == "quickbooks pro 2007 intuit"


def test_embed_text_omits_missing_manufacturer() -> None:
    r = ProductSchema(id="g1", title="solo title", source="google")
    assert r.embed_text == "solo title"


def test_embed_text_serializes_as_computed_field() -> None:
    r = ProductSchema(id="a9", title="photoshop", manufacturer="adobe", source="amazon")
    assert r.model_dump()["embed_text"] == "photoshop adobe"


# --- private helpers: edge branches not exercised by clean real data -------------


def test_record_from_row_handles_empty_and_missing_fields() -> None:
    row = {"id": " 7 ", "title": "", "manufacturer": "", "price": ""}
    rec = _record_from_row(row, "google", "g")
    assert rec.id == "g7"  # surrounding whitespace stripped
    assert rec.title == ""  # empty title -> "" fallback
    assert rec.manufacturer is None  # empty cell -> None
    assert rec.price is None  # missing 'price' key tolerated -> None
    assert rec.source == "google"


def test_record_from_row_preserves_present_fields() -> None:
    row = {"id": "3", "title": "norton 360", "manufacturer": "symantec", "price": "49.99"}
    rec = _record_from_row(row, "amazon", "a")
    assert rec.id == "a3"
    assert rec.title == "norton 360"
    assert rec.manufacturer == "symantec"
    assert rec.price == "49.99"


def test_clusters_from_pairs_merges_transitively_and_completes_singletons() -> None:
    # a1-g1 and a1-g2 share a1, so all three merge; a2/g3 are unmatched singletons.
    pairs = {frozenset({"a1", "g1"}), frozenset({"a1", "g2"})}
    clusters = _clusters_from_pairs(pairs, ["a1", "a2", "g1", "g2", "g3"])
    match = [c for c in clusters if len(c) >= 2]
    singletons = [c for c in clusters if len(c) == 1]
    # Index rather than compare the whole list: cluster order depends on frozenset
    # iteration, which isn't guaranteed across Python versions / hash seeds.
    assert len(match) == 1
    assert match[0] == {"a1", "g1", "g2"}
    assert sorted(tuple(c)[0] for c in singletons) == ["a2", "g3"]


def test_clusters_from_pairs_empty_pairs_yields_all_singletons() -> None:
    # No matches: every id becomes its own singleton (no components).
    clusters = _clusters_from_pairs(set(), ["a1", "g1", "g2"])
    assert sorted(tuple(c)[0] for c in clusters) == ["a1", "g1", "g2"]
    assert all(len(c) == 1 for c in clusters)


def test_clusters_from_pairs_empty_ids_yields_components_only() -> None:
    # No corpus ids to complete against: only the match component is returned.
    clusters = _clusters_from_pairs({frozenset({"a1", "g1"})}, [])
    assert clusters == [{"a1", "g1"}]


# --- pick_blocking_k: pure, fast branches ---------------------------------------


def test_pick_blocking_k_returns_min_passing() -> None:
    assert pick_blocking_k({5: 0.85, 10: 0.91, 20: 0.95}) == 10


def test_pick_blocking_k_falls_back_to_best_when_none_pass() -> None:
    # Honest fallback (the real Amazon-Google case): no k clears 0.90 -> best k.
    assert pick_blocking_k({5: 0.76, 10: 0.81, 50: 0.84}) == 50


def test_pick_blocking_k_custom_threshold() -> None:
    assert pick_blocking_k({5: 0.80, 10: 0.85}, threshold=0.85) == 10


def test_pick_blocking_k_single_entry_passing() -> None:
    assert pick_blocking_k({5: 0.95}) == 5


def test_pick_blocking_k_single_entry_failing_falls_back() -> None:
    # One entry, below the gate: fallback returns it as the best available.
    assert pick_blocking_k({5: 0.80}) == 5


def test_pick_blocking_k_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        pick_blocking_k({})


def test_default_blocking_k_is_pinned_with_honest_gate_outcome() -> None:
    assert DEFAULT_AG_BLOCKING_K == 50
    assert AG_RECALL_GATE == 0.90
    # The gate is honestly NOT met by title+manufacturer vector blocking.
    assert ACHIEVED_PC_AT_DEFAULT_K == pytest.approx(0.8388, abs=1e-4)
    assert GATE_MET is False


def test_cross_source_filters_intra_source_pairs() -> None:
    a1 = ProductSchema(id="a1", title="x", source="amazon")
    a2 = ProductSchema(id="a2", title="y", source="amazon")
    g1 = ProductSchema(id="g1", title="z", source="google")
    cross = ERCandidate(left=a1, right=g1, blocker_name="t")
    same = ERCandidate(left=a1, right=a2, blocker_name="t")
    assert _cross_source([cross, same]) == [cross]


def test_cross_source_all_same_source_returns_empty() -> None:
    a1 = ProductSchema(id="a1", title="x", source="amazon")
    a2 = ProductSchema(id="a2", title="y", source="amazon")
    same = ERCandidate(left=a1, right=a2, blocker_name="t")
    assert _cross_source([same]) == []


# --- slow: real embeddings, runs in CI ------------------------------------------


@pytest.mark.slow
def test_sweep_blocking_k_pins_documented_pair_completeness() -> None:
    corpus, gold, _pairs = load_amazon_google()
    ks = (5, 10, 20, 30, 50)
    recalls = sweep_blocking_k(corpus, gold, ks=ks)

    assert set(recalls) == set(ks)
    assert all(0.0 <= v <= 1.0 for v in recalls.values())

    chosen = pick_blocking_k(recalls)
    assert chosen == DEFAULT_AG_BLOCKING_K
    # The pinned k achieves the documented (honest) Pair-Completeness.
    assert recalls[chosen] == pytest.approx(ACHIEVED_PC_AT_DEFAULT_K, abs=5e-3)
    # No k clears the 0.90 gate: pick falls back to the best-recall k.
    assert max(recalls.values()) < AG_RECALL_GATE
    assert recalls[chosen] == max(recalls.values())


@pytest.mark.slow
def test_pair_completeness_is_computable_via_evaluate_blocking() -> None:
    corpus, gold, _pairs = load_amazon_google()
    recalls = sweep_blocking_k(corpus, gold, ks=(DEFAULT_AG_BLOCKING_K,))
    # The shared build_product_blocker factory must measure the same
    # Pair-Completeness as the sweep. Its FAISSIndex ships unbuilt (embedding is
    # deferred to resolve-time), so build it here, then measure PC directly via
    # evaluate_blocking — confirming PC is a plain candidate_recall measurement.
    blocker = build_product_blocker(DEFAULT_AG_BLOCKING_K)
    records = [r.model_dump() for r in corpus]
    blocker.vector_index.create_index([r.embed_text for r in corpus])
    candidates = _cross_source(list(blocker.stream(records)))
    pc = evaluate_blocking(candidates, gold).candidate_recall
    assert pc == pytest.approx(recalls[DEFAULT_AG_BLOCKING_K], abs=5e-3)


# --- AmazonGoogleBenchmark conformer: fast contract tests (no embeddings) --------


def test_benchmark_exposes_pinned_config() -> None:
    """The conformer surfaces the dataset-agnostic + blocking-registry contracts."""
    bench = AmazonGoogleBenchmark()
    assert bench.name == "amazon_google"
    assert bench.threshold_grid == DEFAULT_AG_THRESHOLD_GRID
    assert bench.schema is ProductSchema
    assert bench.blocking_k == DEFAULT_AG_BLOCKING_K


def test_benchmark_load_matches_loader_with_closure_pairs() -> None:
    """``load`` returns the corpus/partition plus within-cluster (closure) pairs."""
    bench = AmazonGoogleBenchmark()
    corpus, gold_clusters, gold_pairs = bench.load()

    base_corpus, base_clusters, _base_pairs = load_amazon_google()
    assert [r.id for r in corpus] == [r.id for r in base_corpus]
    assert gold_clusters == base_clusters
    # Conformer gold_pairs are the within-cluster closure (mirrors FZ), a superset
    # of the raw labelled pairs because Amazon-Google is many-to-many.
    assert gold_pairs == gold_pairs_from_clusters(gold_clusters)


def test_benchmark_split_is_leakage_free_and_partitions() -> None:
    """No gold cluster straddles the split; train/test ids are disjoint + complete."""
    bench = AmazonGoogleBenchmark()
    corpus, gold_clusters, _pairs = bench.load()
    train_recs, test_recs, train_cls, test_cls = bench.split(corpus, gold_clusters, seed=0)

    train_ids = {r.id for r in train_recs}
    test_ids = {r.id for r in test_recs}
    # Disjoint splits that together cover the whole corpus.
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == {r.id for r in corpus}
    # Every returned cluster lives entirely on its own side (no match pair leaks).
    assert {rid for c in train_cls for rid in c} == train_ids
    assert {rid for c in test_cls for rid in c} == test_ids
    # The full gold partition is split whole (many-to-many clusters kept intact).
    assert len(train_cls) + len(test_cls) == len(gold_clusters)


def test_benchmark_split_is_deterministic() -> None:
    """Same seed → identical split (deterministic shuffle)."""
    bench = AmazonGoogleBenchmark()
    corpus, gold_clusters, _pairs = bench.load()
    a = bench.split(corpus, gold_clusters, seed=0)
    b = bench.split(corpus, gold_clusters, seed=0)
    assert [r.id for r in a[0]] == [r.id for r in b[0]]
    assert [r.id for r in a[1]] == [r.id for r in b[1]]


def test_benchmark_build_blocker_returns_fresh_unbuilt_vector_blocker() -> None:
    """``build_blocker`` yields a fresh VectorBlocker at the requested k each call."""
    bench = AmazonGoogleBenchmark()
    b1 = bench.build_blocker(7)
    b2 = bench.build_blocker(7)
    assert isinstance(b1, VectorBlocker)
    assert b1.k_neighbors == 7
    assert b1 is not b2  # fresh instance per call (independent index)
