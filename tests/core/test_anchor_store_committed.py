"""AnchorStore on committed data + a fresh-process round-trip (W2.2 exit).

Grounds :class:`~langres.curation.anchor_store.AnchorStore` in the committed
Fodors-Zagat benchmark (no embeddings — the offline string pipeline) and proves
the M2 lesson: an artifact saved in one process reloads and assigns identically
in a **clean** subprocess (catching any registry/import side-effect a single
process would hide).

The anchored slice includes Fodor's record ``f534`` but holds out its Zagat gold
partner ``z219``; assigning ``z219`` must link it back to ``f534``'s entity.
"""

import json
import subprocess
import sys
from pathlib import Path

from langres.core import Resolver
from langres.curation.anchor_store import AnchorStore
from langres.data.er_benchmarks import RestaurantSchema, load_fodors_zagat

# A Fodors-Zagat gold match pair: Fodor's f534 <-> Zagat z219.
_ANCHORED_ID = "f534"
_HELD_OUT_ID = "z219"


def _anchor_slice() -> tuple[list[dict[str, object]], dict[str, object]]:
    """80-record anchor slice (includes f534, excludes z219) + the z219 record."""
    corpus, _gold = load_fodors_zagat()
    by_id = {record.id: record for record in corpus}
    anchors = [r.model_dump() for r in corpus if r.id != _HELD_OUT_ID][:80]
    assert any(a["id"] == _ANCHORED_ID for a in anchors)
    assert all(a["id"] != _HELD_OUT_ID for a in anchors)
    return anchors, by_id[_HELD_OUT_ID].model_dump()


def test_assign_links_held_out_partner_on_committed_data() -> None:
    anchors, held_out = _anchor_slice()
    resolver = Resolver.from_schema(RestaurantSchema, matcher="string", threshold=0.5)
    store = resolver.build_anchor_store(anchors)

    delta = resolver.assign(held_out)
    assert delta.type == "link"
    assert delta.entity_id == store.entity_id_of(_ANCHORED_ID)
    assert _ANCHORED_ID in delta.matched_anchor_ids


def test_assign_novel_record_is_new_on_committed_data() -> None:
    anchors, _held_out = _anchor_slice()
    resolver = Resolver.from_schema(RestaurantSchema, matcher="string", threshold=0.5)
    resolver.build_anchor_store(anchors)
    novel = {
        "id": "novel-1",
        "name": "Totally Nonexistent Diner XYZ",
        "addr": "999 Nowhere Blvd",
        "city": "Voidtown",
        "phone": "000-000-0000",
        "type": "none",
        "source": "zagat",
    }
    assert resolver.assign(novel).type == "new"


def test_fresh_process_round_trip(tmp_path: Path) -> None:
    """Build+save here; load+assign in a CLEAN subprocess -> identical result."""
    anchors, held_out = _anchor_slice()

    # In-process reference result.
    reference = Resolver.from_schema(
        RestaurantSchema, matcher="string", threshold=0.5
    ).build_anchor_store(anchors)
    before = reference.assign(held_out)

    # Persist a fresh store (assign above mutates; save an untouched one).
    path = tmp_path / "anchors"
    Resolver.from_schema(RestaurantSchema, matcher="string", threshold=0.5).build_anchor_store(
        anchors
    ).save(path)

    # Reload + assign in a brand-new interpreter (no shared imports/registry).
    script = (
        "import json, sys\n"
        "from langres.curation.anchor_store import AnchorStore\n"
        "from langres.data.er_benchmarks import load_fodors_zagat\n"
        "corpus, _ = load_fodors_zagat()\n"
        "held = next(r for r in corpus if r.id == 'z219').model_dump()\n"
        f"store = AnchorStore.load({str(path)!r})\n"
        "delta = store.assign(held)\n"
        "sys.stdout.write(json.dumps({'type': delta.type, 'entity_id': delta.entity_id}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    result = json.loads(proc.stdout)
    assert result["type"] == before.type == "link"
    assert result["entity_id"] == before.entity_id
