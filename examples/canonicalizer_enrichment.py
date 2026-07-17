"""The enrichment loop: a sparse mention links to an entity and enriches its golden record.

This is the flagship W2.3 flow, end-to-end and deterministic (string judge, no
embeddings, no network, $0). It shows the two M5 seams composing:

1. **`AnchorStore`** (W2.2): anchor a `Resolver` on a batch of rich company
   records, minting a stable entity id per record.
2. **`Resolver.assign`** (W2.2): a **sparse** new mention (only a name and a
   website) links to the matching existing entity — `ClusterDelta(type="link")`.
3. **`Canonicalizer`** (W2.3): fold that mention into the entity's golden record
   via survivorship. The mention fills a field the golden record *lacked* (its
   ``website``), so the golden record ends up MORE complete than either input —
   the point of progressive enrichment.

Run it::

    uv run python examples/canonicalizer_enrichment.py

``print`` is allowed in examples (this is demonstration, not library code).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from langres.core import Resolver
from langres.curation.anchor_store import AnchorStore
from langres.curation.canonicalizer import Canonicalizer
from langres.core.models import CompanySchema

# ----------------------------------------------------------------------------
# A prior batch of RICH company records (the anchors) + one SPARSE new mention.
# CompanySchema is already registered; we reuse it (no new schema name).
# ----------------------------------------------------------------------------

# Rich anchors: full name/address/phone, but NONE carries a website.
ANCHOR_RECORDS: list[dict[str, Any]] = [
    {
        "id": "c1",
        "name": "Acme Corporation",
        "address": "123 Main St, Springfield",
        "phone": "555-0100",
        "website": None,
    },
    {
        "id": "c2",
        "name": "Globex Industries",
        "address": "9 Enterprise Way, Ogdenville",
        "phone": "555-0199",
        "website": None,
    },
]

# A SPARSE new mention of Acme: just a (shortened) name and a website. It is the
# ONLY source that knows Acme's website — the field the anchors are missing.
SPARSE_MENTION: dict[str, Any] = {
    "id": "m-new",
    "name": "Acme Corp",
    "website": "https://acme.example.com",
}


def _completeness(record: dict[str, Any]) -> int:
    """Count non-empty attribute fields (id excluded) — a simple richness measure."""
    return sum(
        1
        for key, value in record.items()
        if key != "id" and value is not None and str(value).strip() != ""
    )


def run_demo() -> dict[str, Any]:
    """Anchor -> assign the sparse mention -> canonicalize -> enriched golden record.

    Returns the pieces the test asserts on: the assign `ClusterDelta`, the golden
    record before and after enrichment, and a flag that the save/reload of the
    Canonicalizer config reproduced the same golden record in a reused process.
    """
    # 1. Anchor a string-judge Resolver on the rich batch (deterministic, free).
    resolver = Resolver.from_schema(
        CompanySchema,
        matcher="string",
        threshold=0.5,
        weights={"name": 0.7, "address": 0.1, "phone": 0.1, "website": 0.1},
    )
    store: AnchorStore = resolver.build_anchor_store(ANCHOR_RECORDS)

    # 2. The sparse mention links to Acme's entity (name alone clears threshold).
    delta = resolver.assign(SPARSE_MENTION)
    acme_entity_id = store.entity_id_of("c1")

    # 3. Build the golden record for Acme's entity from its anchor member(s), then
    #    ENRICH it with the freshly-linked sparse mention. Same survivorship path.
    canon = Canonicalizer()  # most_complete default: prefer the richest source
    acme_anchor_ids = [rid for rid, eid in store.assignments.items() if eid == acme_entity_id]
    acme_records = [r for r in ANCHOR_RECORDS if r["id"] in acme_anchor_ids]
    golden_before = canon.canonicalize(acme_records, entity_id=acme_entity_id)
    golden_after = canon.enrich(golden_before, SPARSE_MENTION)

    # 4. The survivorship policy round-trips through its config artifact (no
    #    pickle): a reloaded Canonicalizer produces the identical golden record.
    with tempfile.TemporaryDirectory() as tmp:
        canon_dir = Path(tmp) / "canon"
        canon.save(canon_dir)
        reloaded = Canonicalizer.load(canon_dir)
        reloaded_after = reloaded.enrich(golden_before, SPARSE_MENTION)
    identical = reloaded_after == golden_after

    return {
        "delta": delta,
        "acme_entity_id": acme_entity_id,
        "golden_before": golden_before,
        "golden_after": golden_after,
        "identical": identical,
    }


def main() -> None:
    logging.getLogger("langres").setLevel(logging.ERROR)

    print("=" * 78)
    print("Enrichment loop — sparse mention links to an entity and enriches it")
    print("=" * 78)

    results = run_demo()
    delta = results["delta"]

    print("\n1. Anchored 2 rich companies; assigned the SPARSE new mention:")
    print(f"   mention: {SPARSE_MENTION}")
    print(
        f"   -> ClusterDelta(type={delta.type!r}, entity_id={delta.entity_id!r}, "
        f"matched={delta.matched_anchor_ids})"
    )
    assert delta.type == "link", "expected the sparse mention to LINK to Acme's entity"
    assert delta.entity_id == results["acme_entity_id"]

    print("\n2. Golden record BEFORE enrichment (from the anchor member(s)):")
    print(
        f"   {results['golden_before']}   [completeness={_completeness(results['golden_before'])}]"
    )

    print("\n3. Golden record AFTER folding in the sparse mention:")
    print(f"   {results['golden_after']}   [completeness={_completeness(results['golden_after'])}]")

    before, after = results["golden_before"], results["golden_after"]
    filled = [k for k in after if after.get(k) and not before.get(k)]
    print(f"\n   Enrichment filled: {filled}  (website the anchors never had)")
    assert after["website"] == SPARSE_MENTION["website"], "website should be enriched in"
    assert _completeness(after) > _completeness(before), "golden must get MORE complete"
    assert _completeness(after) > _completeness(SPARSE_MENTION), "golden richer than the mention"

    print(
        f"\n4. Canonicalizer config save/reload reproduced the golden record: "
        f"{'IDENTICAL' if results['identical'] else 'DIFFERENT'}"
    )
    assert results["identical"], "reloaded Canonicalizer produced a different golden record"

    print("\n" + "=" * 78)
    print("Enrichment loop ran end-to-end. A sparse mention made the entity richer. ✓")
    print("=" * 78)


if __name__ == "__main__":
    main()
