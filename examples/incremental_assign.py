"""Incremental single-record assignment with ``AnchorStore`` (M5 / W2.2).

A batch ``resolve()`` answers "which of these records group together?". But once
that batch is settled, new records keep arriving one at a time — and the question
becomes *incremental*: "here is one NEW record; which existing entity does it
belong to, or is it new?".

``AnchorStore`` answers that. You anchor a ``Resolver`` on the batch once
(minting a stable entity id for every record, singletons included), then
``assign()`` each new record against it — getting back a ``ClusterDelta`` that
either ``link``\\ s to an existing entity (with a stable id) or marks it ``new``.

This example is fully offline (the string judge — no embeddings, no API calls).
Run it with:  ``uv run python examples/incremental_assign.py``
"""

import tempfile
from pathlib import Path

from langres.core import CompanySchema, Resolver
from langres.core.anchor_store import AnchorStore

# A prior batch: records 1 and 2 are the same company; 3 and 4 are unique.
BATCH = [
    {
        "id": "1",
        "name": "Apple Inc",
        "address": "1 Infinite Loop Cupertino",
        "phone": "408-996-1010",
    },
    {
        "id": "2",
        "name": "Apple Incorporated",
        "address": "1 Infinite Loop, Cupertino",
        "phone": "408-996-1010",
    },
    {
        "id": "3",
        "name": "Microsoft Corporation",
        "address": "1 Microsoft Way Redmond",
        "phone": "425-882-8080",
    },
    {
        "id": "4",
        "name": "Umbrella Corporation",
        "address": "Raccoon City Center",
        "phone": "202-555-0100",
    },
]


def main() -> None:
    resolver = Resolver.from_schema(CompanySchema, matcher="string", threshold=0.6)

    # 1. Anchor the resolver on the batch — one stable entity id per record,
    #    including the singletons a batch resolve() would have dropped.
    store = resolver.build_anchor_store(BATCH)
    print("Anchored entities (record_id -> entity_id):")
    for record_id, entity_id in sorted(store.assignments.items()):
        print(f"  record {record_id} -> {entity_id}")

    # 2. A new mention of an existing company -> LINK to its stable entity id.
    delta = resolver.assign(
        {
            "id": "9",
            "name": "Apple Inc.",
            "address": "1 Infinite Loop Cupertino",
            "phone": "408-996-1010",
        }
    )
    print(
        f"\nassign(Apple Inc.)   -> {delta.type} to {delta.entity_id} "
        f"(matched anchors {delta.matched_anchor_ids})"
    )

    # 3. A brand-new company -> NEW freshly minted entity id.
    delta = resolver.assign(
        {
            "id": "10",
            "name": "Nintendo Company",
            "address": "11-1 Kamitoba Kyoto",
            "phone": "075-541-6111",
        }
    )
    print(f"assign(Nintendo)     -> {delta.type} to {delta.entity_id}")

    # 4. A new mention of a record that was a SINGLETON in the batch -> LINK
    #    (singletons are first-class anchors, not dropped).
    delta = resolver.assign(
        {
            "id": "11",
            "name": "Microsoft Corporation",
            "address": "1 Microsoft Way Redmond",
            "phone": "425-882-8080",
        }
    )
    print(f"assign(Microsoft)    -> {delta.type} to {delta.entity_id} (singleton was assignable)")

    # 5. Persist the whole store (pipeline + id bookkeeping) and reload it —
    #    the config-registry artifact seam, no pickle.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "anchors"
        store.save(path)
        reloaded = AnchorStore.load(path)
        delta = reloaded.assign(
            {
                "id": "12",
                "name": "Apple Inc.",
                "address": "1 Infinite Loop Cupertino",
                "phone": "408-996-1010",
            }
        )
        print(f"\nafter save/load: assign(Apple Inc.) -> {delta.type} to {delta.entity_id}")


if __name__ == "__main__":
    main()
