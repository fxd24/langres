"""M2 Wave 1 — shared restaurant blocker/resolver factories + Fodors split.

Fast, network-free unit tests for the three public helpers that wire the M0/M1
primitives into a reusable M2 baseline:

- ``build_restaurant_blocker`` — the shared MiniLM + FAISS-cosine VectorBlocker.
- ``build_restaurant_resolver`` — VectorBlocker + StringComparator + the
  zero-spend WeightedAverageMatcher + connected-components Clusterer, serializable.
- ``split_restaurant_corpus`` — leakage-free stratified train/test split over
  FULL ``RestaurantSchema`` records (preserving ``source`` + ``f``/``z`` ids).

The wiring / type / split-shape assertions need no embeddings and stay fast; the
one assertion that ``.save()`` writes an artifact runs on an unbuilt index (the
blocker never embeds), so it is fast too.
"""

from pathlib import Path

import pytest

from langres.core.blockers.vector import VectorBlocker
from langres.core.clusterer import Clusterer
from langres.core.comparator import StringComparator
from langres.core.indexes.vector_index import FAISSIndex
from langres.core.matchers.weighted_average import WeightedAverageMatcher
from langres.core.resolver import Resolver
from langres.data.er_benchmarks import (
    DEFAULT_BLOCKING_K,
    RestaurantSchema,
    build_restaurant_blocker,
    build_restaurant_resolver,
    load_fodors_zagat,
    split_restaurant_corpus,
)

# --- build_restaurant_blocker ----------------------------------------------------


def test_build_restaurant_blocker_defaults() -> None:
    blocker = build_restaurant_blocker()
    assert isinstance(blocker, VectorBlocker)
    assert blocker.k_neighbors == DEFAULT_BLOCKING_K
    # Declarative (serializable) wiring is asserted via the save() round-trip in
    # test_build_restaurant_resolver_is_serializable; don't reach into private
    # attrs here. MiniLM + FAISS-cosine — the M1 config, reused.
    assert isinstance(blocker.vector_index, FAISSIndex)
    assert blocker.vector_index.metric == "cosine"
    assert blocker.vector_index.embedder.model_name == "all-MiniLM-L6-v2"


def test_build_restaurant_blocker_custom_k() -> None:
    blocker = build_restaurant_blocker(k_neighbors=17)
    assert blocker.k_neighbors == 17


# --- build_restaurant_resolver ---------------------------------------------------


def test_build_restaurant_resolver_wires_expected_components() -> None:
    resolver = build_restaurant_resolver(threshold=0.7)
    assert isinstance(resolver, Resolver)
    assert isinstance(resolver.blocker, VectorBlocker)
    assert isinstance(resolver.comparator, StringComparator)
    assert isinstance(resolver.module, WeightedAverageMatcher)
    assert isinstance(resolver.clusterer, Clusterer)
    assert resolver.clusterer.threshold == 0.7
    assert resolver.blocker.k_neighbors == DEFAULT_BLOCKING_K


def test_build_restaurant_resolver_custom_k() -> None:
    resolver = build_restaurant_resolver(threshold=0.5, k_neighbors=9)
    assert isinstance(resolver.blocker, VectorBlocker)
    assert resolver.blocker.k_neighbors == 9


def test_build_restaurant_resolver_excludes_source_feature() -> None:
    # Cross-source true matches mean comparing ``source`` would penalise every
    # positive; the comparator (and the judge sharing its specs) must omit it.
    resolver = build_restaurant_resolver(threshold=0.7)
    assert resolver.comparator is not None
    feature_names = {s.name for s in resolver.comparator.feature_specs}
    assert "source" not in feature_names
    assert "embed_text" not in feature_names
    assert "id" not in feature_names
    assert feature_names == {"name", "addr", "city", "phone", "type"}
    # Judge scores on the SAME specs the comparator compares on.
    assert {s.name for s in resolver.module.feature_specs} == feature_names


def test_build_restaurant_resolver_is_serializable(tmp_path: Path) -> None:
    # The whole point of the M2 artifact contract: save() must not raise. The
    # blocker's index is never built here, so this stays fast (no embeddings).
    resolver = build_restaurant_resolver(threshold=0.7)
    resolver.save(tmp_path)
    # round-trips through the registry (no pickle, no code execution).
    reloaded = Resolver.load(tmp_path)
    assert isinstance(reloaded.blocker, VectorBlocker)
    assert isinstance(reloaded.module, WeightedAverageMatcher)
    assert isinstance(reloaded.comparator, StringComparator)
    assert reloaded.clusterer.threshold == 0.7


# --- split_restaurant_corpus -----------------------------------------------------


def _synthetic_corpus() -> tuple[list[RestaurantSchema], list[set[str]]]:
    """A small closed-world corpus: 4 cross-source match pairs + 4 singletons.

    Mirrors ``load_fodors_zagat``'s shape (source-prefixed ids, complete
    partition) without touching disk or embeddings.
    """
    corpus: list[RestaurantSchema] = []
    clusters: list[set[str]] = []
    for i in range(4):  # matched cross-source pairs f{i}/z{i}
        corpus.append(RestaurantSchema(id=f"f{i}", name=f"resto {i}", source="fodors"))
        corpus.append(RestaurantSchema(id=f"z{i}", name=f"resto {i}", source="zagat"))
        clusters.append({f"f{i}", f"z{i}"})
    for i in range(4):  # unmatched singletons
        corpus.append(RestaurantSchema(id=f"f{100 + i}", name=f"solo {i}", source="fodors"))
        clusters.append({f"f{100 + i}"})
    return corpus, clusters


def test_split_returns_full_records_with_source_and_ids() -> None:
    corpus, clusters = _synthetic_corpus()
    train, test, _, _ = split_restaurant_corpus(corpus, clusters, test_size=0.5, seed=0)
    # Records survive as RestaurantSchema with source + f/z ids preserved.
    for rec in train + test:
        assert isinstance(rec, RestaurantSchema)
        assert rec.source in ("fodors", "zagat")
        assert rec.id[0] in ("f", "z")


def test_split_is_disjoint_and_covers_corpus() -> None:
    corpus, clusters = _synthetic_corpus()
    train, test, _, _ = split_restaurant_corpus(corpus, clusters, test_size=0.5, seed=0)
    train_ids = {r.id for r in train}
    test_ids = {r.id for r in test}
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == {r.id for r in corpus}
    assert len(train) + len(test) == len(corpus)


def test_split_keeps_every_match_pair_in_one_side() -> None:
    # The leakage guard: no gold match pair may straddle train/test.
    corpus, clusters = _synthetic_corpus()
    train, test, _, _ = split_restaurant_corpus(corpus, clusters, test_size=0.5, seed=0)
    train_ids = {r.id for r in train}
    test_ids = {r.id for r in test}
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        in_train = cluster <= train_ids
        in_test = cluster <= test_ids
        assert in_train ^ in_test, f"match pair {cluster} leaked across the split"


def test_split_clusters_restricted_to_each_side() -> None:
    corpus, clusters = _synthetic_corpus()
    train, test, train_cls, test_cls = split_restaurant_corpus(
        corpus, clusters, test_size=0.5, seed=0
    )
    train_ids = {r.id for r in train}
    test_ids = {r.id for r in test}
    # Returned clusters partition exactly their own split's ids.
    assert {rid for c in train_cls for rid in c} == train_ids
    assert {rid for c in test_cls for rid in c} == test_ids
    # Each returned cluster is a subset of its split (never mixes sides).
    for c in train_cls:
        assert c <= train_ids
    for c in test_cls:
        assert c <= test_ids


def test_split_is_deterministic_under_fixed_seed() -> None:
    corpus, clusters = _synthetic_corpus()
    a = split_restaurant_corpus(corpus, clusters, test_size=0.5, seed=7)
    b = split_restaurant_corpus(corpus, clusters, test_size=0.5, seed=7)
    assert [r.id for r in a[0]] == [r.id for r in b[0]]
    assert [r.id for r in a[1]] == [r.id for r in b[1]]
    assert a[2] == b[2]
    assert a[3] == b[3]


def test_split_real_fodors_corpus_no_leakage_and_complete() -> None:
    # The loader is fast (CSV only, no embeddings) — keep this in the fast suite.
    corpus, gold = load_fodors_zagat()
    train, test, train_cls, test_cls = split_restaurant_corpus(corpus, gold, test_size=0.3, seed=0)
    train_ids = {r.id for r in train}
    test_ids = {r.id for r in test}
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == {r.id for r in corpus}
    # No cross-source match pair straddles the split (the leakage invariant).
    for cluster in (c for c in gold if len(c) == 2):
        in_train = cluster <= train_ids
        in_test = cluster <= test_ids
        assert in_train ^ in_test
    # Roughly 30% of records land in test (stratified, so approximate).
    assert 0.2 < len(test) / len(corpus) < 0.4
    # Restricted clusters cover exactly their split.
    assert {rid for c in train_cls for rid in c} == train_ids
    assert {rid for c in test_cls for rid in c} == test_ids


def test_split_rejects_out_of_range_test_size() -> None:
    corpus, clusters = _synthetic_corpus()
    with pytest.raises(ValueError, match="test_size"):
        split_restaurant_corpus(corpus, clusters, test_size=1.5, seed=0)
