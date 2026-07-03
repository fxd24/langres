"""Survivorship correctness for :class:`~langres.core.canonicalizer.Canonicalizer`.

Per-strategy "known winner" cases, the enrichment loop, edge cases (all-null,
single-record identity, tie-break determinism), config validation, and a
config save/load round-trip (in-process + a fresh subprocess — the M2 lesson).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from langres.core import Canonicalizer
from langres.core.canonicalizer import (
    CANONICALIZER_VERSION,
    DEFAULT_STRATEGY,
    FieldContext,
    _is_missing,
)


# ----------------------------------------------------------------------------
# Per-strategy correctness: a group with a KNOWN winner per field.
# ----------------------------------------------------------------------------


def test_most_complete_prefers_richest_source_record() -> None:
    """The value comes from the record carrying the most non-missing fields."""
    records = [
        {"id": "1", "name": "Ann", "phone": None, "city": None},  # completeness 1
        {"id": "2", "name": "Ann Lee", "phone": "555", "city": "NY"},  # completeness 3
    ]
    golden = Canonicalizer(default_strategy="most_complete").canonicalize(records)
    assert golden["name"] == "Ann Lee"  # richer record wins even though both present


def test_most_complete_fills_gap_from_sparser_record() -> None:
    """A field only the sparser record has is still taken (present beats absent)."""
    records = [
        {"id": "1", "name": "Ann Lee", "phone": "555"},  # no email
        {"id": "2", "name": "Ann", "email": "a@x.io"},  # only email holder
    ]
    golden = Canonicalizer().canonicalize(records)
    assert golden["email"] == "a@x.io"
    assert golden["phone"] == "555"


def test_longest_picks_longest_string() -> None:
    records = [{"id": "1", "name": "Bob"}, {"id": "2", "name": "Robert William"}]
    golden = Canonicalizer(field_strategies={"name": "longest"}).canonicalize(records)
    assert golden["name"] == "Robert William"


def test_most_frequent_picks_mode() -> None:
    records = [
        {"id": "1", "phone": "111"},
        {"id": "2", "phone": "222"},
        {"id": "3", "phone": "222"},
    ]
    golden = Canonicalizer(field_strategies={"phone": "most_frequent"}).canonicalize(records)
    assert golden["phone"] == "222"


def test_first_and_source_priority_are_first_non_missing() -> None:
    records = [{"id": "1", "name": None}, {"id": "2", "name": "Real"}, {"id": "3", "name": "Other"}]
    assert Canonicalizer(default_strategy="first").canonicalize(records)["name"] == "Real"
    assert Canonicalizer(default_strategy="source_priority").canonicalize(records)["name"] == "Real"


def test_most_recent_picks_latest_timestamp() -> None:
    records = [
        {"id": "1", "name": "Old Name", "updated_at": "2020-01-01"},
        {"id": "2", "name": "New Name", "updated_at": "2023-06-15"},
        {"id": "3", "name": "Undated", "updated_at": None},  # ignored (no timestamp)
    ]
    canon = Canonicalizer(default_strategy="most_recent", timestamp_field="updated_at")
    assert canon.canonicalize(records)["name"] == "New Name"


def test_most_recent_returns_none_when_no_record_is_dated() -> None:
    records = [{"id": "1", "name": "A", "ts": None}, {"id": "2", "name": "B", "ts": None}]
    canon = Canonicalizer(default_strategy="most_recent", timestamp_field="ts")
    assert canon.canonicalize(records)["name"] is None


# ----------------------------------------------------------------------------
# The enrichment loop.
# ----------------------------------------------------------------------------


def test_enrich_fills_missing_field_from_mention() -> None:
    """A sparse mention enriches the golden record: fills a field it lacked."""
    golden = {"id": "e0", "name": "Acme Corporation", "address": "123 Main St", "website": None}
    mention = {"id": "m9", "name": "Acme Corp", "website": "acme.com"}
    canon = Canonicalizer()
    enriched = canon.enrich(golden, mention)
    assert enriched["website"] == "acme.com"  # gap filled
    assert enriched["name"] == "Acme Corporation"  # richer golden retained
    assert enriched["address"] == "123 Main St"
    assert enriched["id"] == "e0"  # master id preserved


def test_enrich_is_canonicalize_over_two_records() -> None:
    """enrich(g, m) is exactly canonicalize([g, m]) with g's id kept."""
    golden = {"id": "e0", "name": "A", "phone": "555"}
    mention = {"id": "m1", "name": "A Inc", "phone": None, "email": "a@x.io"}
    canon = Canonicalizer()
    assert canon.enrich(golden, mention) == canon.canonicalize([golden, mention], entity_id="e0")


def test_enrich_can_restamp_entity_id() -> None:
    golden = {"id": "e0", "name": "A"}
    assert Canonicalizer().enrich(golden, {"id": "m1", "name": "A"}, entity_id="e7")["id"] == "e7"


# ----------------------------------------------------------------------------
# Edge cases.
# ----------------------------------------------------------------------------


def test_single_record_group_is_identity() -> None:
    record = {"id": "1", "name": "Solo", "phone": "555", "email": None}
    assert Canonicalizer().canonicalize([record]) == record


def test_all_null_field_resolves_to_none() -> None:
    records = [{"id": "1", "name": "A", "fax": None}, {"id": "2", "name": "B", "fax": ""}]
    golden = Canonicalizer().canonicalize(records)
    assert "fax" in golden and golden["fax"] is None


def test_longest_and_most_frequent_return_none_when_all_missing() -> None:
    """A field null across the whole group resolves to None under any strategy."""
    records = [{"id": "1", "name": "A", "x": None}, {"id": "2", "name": "B", "x": ""}]
    assert Canonicalizer(field_strategies={"x": "longest"}).canonicalize(records)["x"] is None
    assert Canonicalizer(field_strategies={"x": "most_frequent"}).canonicalize(records)["x"] is None


def test_empty_group_raises() -> None:
    with pytest.raises(ValueError, match="at least one record"):
        Canonicalizer().canonicalize([])


def test_entity_id_stamps_master_id() -> None:
    golden = Canonicalizer().canonicalize([{"id": "1", "name": "A"}], entity_id="master-7")
    assert golden["id"] == "master-7"


def test_default_entity_id_is_first_record_id() -> None:
    golden = Canonicalizer().canonicalize([{"id": "first", "name": "A"}, {"id": "second"}])
    assert golden["id"] == "first"


def test_records_without_id_field_produce_idless_golden() -> None:
    """A group carrying no id field yields a golden without one (no crash)."""
    golden = Canonicalizer().canonicalize([{"name": "A"}, {"name": "A Inc", "city": "NY"}])
    assert "id" not in golden and golden["name"] == "A Inc"


def test_id_field_is_not_survivorshipped() -> None:
    """The id is stamped, never chosen by the (e.g. longest) attribute strategy."""
    records = [{"id": "short", "name": "A"}, {"id": "much-longer-id", "name": "A"}]
    golden = Canonicalizer(default_strategy="longest").canonicalize(records)
    assert golden["id"] == "short"  # first record's id, not the longest string


def test_custom_id_field() -> None:
    records = [{"pk": "p1", "name": "A"}, {"pk": "p2", "name": "A Inc", "city": "NY"}]
    golden = Canonicalizer(id_field="pk").canonicalize(records)
    assert golden["pk"] == "p1" and golden["name"] == "A Inc"


def test_zero_and_false_are_present_values() -> None:
    """0 / False are real values, not missing — a later None must not overwrite them."""
    records = [{"id": "1", "count": 0, "active": False}, {"id": "2", "count": None}]
    golden = Canonicalizer(default_strategy="first").canonicalize(records)
    assert golden["count"] == 0
    assert golden["active"] is False


def test_tie_break_is_deterministic_first_seen() -> None:
    """Equal-length / equal-frequency ties resolve to the first-seen value."""
    records = [{"id": "1", "name": "Bob"}, {"id": "2", "name": "Ann"}]  # both length 3
    assert (
        Canonicalizer(field_strategies={"name": "longest"}).canonicalize(records)["name"] == "Bob"
    )
    freq = [{"id": "1", "v": "x"}, {"id": "2", "v": "y"}]  # each once
    assert Canonicalizer(field_strategies={"v": "most_frequent"}).canonicalize(freq)["v"] == "x"


# ----------------------------------------------------------------------------
# Helpers / validation.
# ----------------------------------------------------------------------------


def test_is_missing_semantics() -> None:
    assert _is_missing(None)
    assert _is_missing("")
    assert _is_missing("   ")
    assert not _is_missing(0)
    assert not _is_missing(False)
    assert not _is_missing("x")


def test_field_context_present_skips_missing_and_absent() -> None:
    ctx = FieldContext(
        field="name",
        records=[{"name": "A"}, {"name": None}, {}, {"name": "B"}],
        id_field="id",
        timestamp_field=None,
    )
    assert ctx.present() == [(0, "A"), (3, "B")]


def test_unknown_strategy_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown survivorship strategy"):
        Canonicalizer(default_strategy="bogus")
    with pytest.raises(ValueError, match="Unknown survivorship strategy"):
        Canonicalizer(field_strategies={"name": "bogus"})


def test_most_recent_without_timestamp_field_rejected() -> None:
    with pytest.raises(ValueError, match="needs a `timestamp_field`"):
        Canonicalizer(default_strategy="most_recent")


# ----------------------------------------------------------------------------
# Config round-trip (config-registry seam; no pickle).
# ----------------------------------------------------------------------------


def test_config_round_trip_in_process() -> None:
    canon = Canonicalizer(
        default_strategy="longest",
        field_strategies={"phone": "most_frequent", "updated": "most_recent"},
        id_field="pk",
        timestamp_field="updated",
    )
    rebuilt = Canonicalizer.from_config(canon.config)
    assert rebuilt.config == canon.config


def test_default_config_shape() -> None:
    assert Canonicalizer().config == {
        "default_strategy": DEFAULT_STRATEGY,
        "field_strategies": {},
        "id_field": "id",
        "timestamp_field": None,
    }


def test_save_load_round_trip(tmp_path: Path) -> None:
    canon = Canonicalizer(default_strategy="most_frequent", field_strategies={"name": "longest"})
    canon.save(tmp_path / "canon")
    reloaded = Canonicalizer.load(tmp_path / "canon")
    records = [{"id": "1", "name": "AB", "v": "x"}, {"id": "2", "name": "ABC", "v": "x"}]
    assert reloaded.canonicalize(records) == canon.canonicalize(records)


def test_load_rejects_incompatible_version(tmp_path: Path) -> None:
    path = tmp_path / "canon"
    Canonicalizer().save(path)
    manifest = path / "canonicalizer.json"
    doc = json.loads(manifest.read_text())
    doc["version"] = "999"
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="differs from supported"):
        Canonicalizer.load(path)


def test_registry_resolves_canonicalizer() -> None:
    from langres.core.registry import get_component

    assert get_component("canonicalizer") is Canonicalizer


def test_fresh_process_config_round_trip(tmp_path: Path) -> None:
    """Save here; load + canonicalize in a CLEAN subprocess -> identical golden."""
    canon = Canonicalizer(default_strategy="longest", field_strategies={"phone": "most_frequent"})
    path = tmp_path / "canon"
    canon.save(path)
    records = [
        {"id": "1", "name": "Bob", "phone": "222"},
        {"id": "2", "name": "Robert William", "phone": "222"},
        {"id": "3", "name": "Rob", "phone": "111"},
    ]
    reference = canon.canonicalize(records, entity_id="e0")

    script = (
        "import json, sys\n"
        "from langres.core import Canonicalizer\n"
        f"canon = Canonicalizer.load({str(path)!r})\n"
        f"records = {records!r}\n"
        "golden = canon.canonicalize(records, entity_id='e0')\n"
        "sys.stdout.write(json.dumps(golden))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert json.loads(proc.stdout) == reference
