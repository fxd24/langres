"""Proof that the enrichment loop example runs end-to-end and actually enriches.

Drives the importable core of ``examples/canonicalizer_enrichment.py`` (``run_demo``):
a sparse mention links to an anchored entity (`ClusterDelta(type="link")`) and,
after canonicalization, the entity's golden record is MORE complete than either
the sparse mention or the pre-enrichment golden record — with the mention filling
a field (``website``) the anchors never had. Deterministic string judge, $0.
"""

from __future__ import annotations

from examples.canonicalizer_enrichment import (
    ANCHOR_RECORDS,
    SPARSE_MENTION,
    _completeness,
    run_demo,
)


def test_sparse_mention_links_to_existing_entity() -> None:
    results = run_demo()
    delta = results["delta"]
    assert delta.type == "link"
    assert delta.record_id == SPARSE_MENTION["id"]
    assert delta.entity_id == results["acme_entity_id"]
    assert "c1" in delta.matched_anchor_ids  # linked to Acme's anchor


def test_enrichment_makes_golden_record_more_complete() -> None:
    results = run_demo()
    before, after = results["golden_before"], results["golden_after"]

    # The mention filled the website the anchors never carried.
    assert before["website"] is None
    assert after["website"] == SPARSE_MENTION["website"]

    # The golden record is strictly richer than the pre-enrichment record AND
    # than the sparse mention itself — the point of progressive enrichment.
    assert _completeness(after) > _completeness(before)
    assert _completeness(after) > _completeness(SPARSE_MENTION)

    # Established rich fields are retained (not clobbered by the sparse mention).
    anchor = next(r for r in ANCHOR_RECORDS if r["id"] == "c1")
    assert after["name"] == anchor["name"]
    assert after["address"] == anchor["address"]
    assert after["id"] == results["acme_entity_id"]  # stable master id


def test_canonicalizer_config_round_trip_is_identical() -> None:
    assert run_demo()["identical"] is True


def test_run_demo_is_repeatable() -> None:
    assert run_demo()["golden_after"] == run_demo()["golden_after"]
