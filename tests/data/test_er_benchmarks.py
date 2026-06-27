"""Tests for the Fodors-Zagat benchmark adapter and blocking k-sweep."""

import pytest

from langres.data.er_benchmarks import (
    DEFAULT_BLOCKING_K,
    RECALL_GATE,
    RestaurantSchema,
    _record_from_row,
    _unquote,
    load_fodors_zagat,
    pick_blocking_k,
    sweep_blocking_k,
)

# --- loader: fast, no embeddings -------------------------------------------------


def test_load_returns_full_corpus_and_gold() -> None:
    corpus, gold = load_fodors_zagat()
    assert len(corpus) == 864
    assert sum(1 for r in corpus if r.source == "fodors") == 533
    assert sum(1 for r in corpus if r.source == "zagat") == 331
    assert len(gold) == 112


def test_corpus_ids_are_globally_unique_and_source_prefixed() -> None:
    corpus, _ = load_fodors_zagat()
    ids = [r.id for r in corpus]
    assert len(set(ids)) == len(ids)
    assert all(r.id.startswith("f") for r in corpus if r.source == "fodors")
    assert all(r.id.startswith("z") for r in corpus if r.source == "zagat")


def test_gold_clusters_are_cross_source_pairs_present_in_corpus() -> None:
    corpus, gold = load_fodors_zagat()
    id_to_source = {r.id: r.source for r in corpus}
    for cluster in gold:
        assert len(cluster) == 2
        left, right = sorted(cluster)
        assert left in id_to_source and right in id_to_source
        assert id_to_source[left] != id_to_source[right]


def test_loader_strips_quotes_and_unescapes() -> None:
    corpus, _ = load_fodors_zagat()
    by_id = {r.id: r for r in corpus}
    # 'arnie morton\'s of chicago' -> arnie morton's of chicago
    assert by_id["f534"].name == "arnie morton's of chicago"
    assert "'" not in by_id["f534"].name[:1]  # no leading wrapping quote


# --- schema / embed_text ---------------------------------------------------------


def test_embed_text_composition_order() -> None:
    r = RestaurantSchema(
        id="f1", name="Masa's", addr="648 bush st.", city="san francisco", source="fodors"
    )
    assert r.embed_text == "Masa's san francisco 648 bush st."


def test_embed_text_omits_missing_fields() -> None:
    r = RestaurantSchema(id="z1", name="Solo", source="zagat")
    assert r.embed_text == "Solo"
    r2 = RestaurantSchema(id="z2", name="No Addr", city="paris", source="zagat")
    assert r2.embed_text == "No Addr paris"


def test_embed_text_serializes_as_computed_field() -> None:
    r = RestaurantSchema(id="f9", name="Cafe", city="rome", source="fodors")
    assert r.model_dump()["embed_text"] == "Cafe rome"


# --- private helpers: edge branches not hit by clean real data -------------------


def test_unquote_variants() -> None:
    assert _unquote("'art\\'s deli'") == "art's deli"
    assert _unquote("534") == "534"  # unquoted numeric: no wrapping quotes
    assert _unquote("'paris'") == "paris"
    assert _unquote("") == ""
    assert _unquote("'") == "'"  # single lone quote: too short to be a wrapper


def test_record_from_row_handles_empty_and_missing_fields() -> None:
    row = {"id": "7", "name": "", "addr": "", "city": "'nyc'", "phone": "", "type": ""}
    rec = _record_from_row(row, "zagat", "z")
    assert rec.id == "z7"
    assert rec.name == ""  # empty name -> "" fallback
    assert rec.addr is None  # empty cell -> None
    assert rec.city == "nyc"
    assert rec.phone is None
    assert rec.type is None  # missing 'class' tolerated; empty type -> None


# --- pick_blocking_k: pure, both branches fast ----------------------------------


def test_pick_blocking_k_returns_min_passing() -> None:
    assert pick_blocking_k({5: 0.90, 10: 0.96, 20: 0.99}) == 10


def test_pick_blocking_k_falls_back_to_best_when_none_pass() -> None:
    # Honest fallback: no k clears the gate -> best (highest) recall k.
    assert pick_blocking_k({5: 0.80, 10: 0.91, 20: 0.88}) == 10


def test_pick_blocking_k_custom_threshold() -> None:
    assert pick_blocking_k({5: 0.80, 10: 0.85}, threshold=0.85) == 10


def test_pick_blocking_k_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        pick_blocking_k({})


def test_default_blocking_k_is_pinned_and_positive() -> None:
    assert DEFAULT_BLOCKING_K == 5
    assert RECALL_GATE == 0.95


# --- slow: real embeddings, runs in CI ------------------------------------------


@pytest.mark.slow
def test_sweep_blocking_k_pins_gate() -> None:
    corpus, gold = load_fodors_zagat()
    ks = (5, 10, 20, 30, 50)
    recalls = sweep_blocking_k(corpus, gold, ks=ks)

    assert set(recalls) == set(ks)
    assert all(0.0 <= v <= 1.0 for v in recalls.values())

    chosen = pick_blocking_k(recalls)
    # The gate is met on this benchmark; if an honest fallback ever triggered,
    # this asserts the documented best recall instead of a faked 0.95.
    if max(recalls.values()) >= RECALL_GATE:
        assert recalls[chosen] >= RECALL_GATE
        assert chosen == DEFAULT_BLOCKING_K
    else:  # pragma: no cover - benchmark currently clears the gate
        assert chosen == max(recalls, key=lambda k: recalls[k])
