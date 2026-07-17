"""Frozen, hand-authored fixture for the W0 behavior-parity net (epic #193).

A fixed, offline, ``$0`` record set with a KNOWN duplicate structure, plus its
hand-labeled ground truth. Everything downstream of this module -- the golden
snapshots in ``goldens/`` and the committed legacy artifact in
``legacy_artifact_v1/`` -- is pinned to *these exact records*. Do not edit the
records or the gold clusters without regenerating every golden (that is the
whole point: a change here is a deliberate re-baseline, never an accident).

The schema is registered under a distinctive name (``"ParityBusinessW0"``) at
import time, so the committed legacy artifact -- whose ``AllPairsBlocker`` config
stores only ``schema_type_name`` -- can be ``Resolver.load``-ed in a fresh
process just by importing this module. An *inferred* schema (what a bare
``FuzzyString().dedupe(records)`` mints) is an ephemeral class a fresh process
cannot import back, so anything meant to ``save`` binds this concrete class.
"""

from __future__ import annotations

from pydantic import BaseModel

from langres.core.registry import register_schema


# The registry key MUST equal the class ``__name__``: an ``AllPairsBlocker``
# built with ``schema=`` registers itself (idempotently) under ``schema.__name__``
# and persists THAT name as ``schema_type_name``. If the decorator key and the
# class name diverged, the committed artifact would store one name while a fresh
# process registered the other, and ``Resolver.load`` would raise
# ``SchemaNotRegistered``. Keeping them identical is what makes the legacy artifact
# loadable simply by importing this module.
@register_schema("ParityBusinessW0")
class ParityBusinessW0(BaseModel):
    """A business entity: id + name + city + phone. Deliberately tiny and typed."""

    id: str
    name: str
    city: str
    phone: str | None = None


# 14 records: four clear duplicate pairs + six clear singletons. The dup pairs
# span the canonical fuzzy-match cases -- possessive/spelling drift ("Bob's
# Diner"/"Bob Diner"), a corporate suffix ("Acme Hardware"/"Acme Hardware Inc"),
# a street abbreviation ("123 Main Street"/"123 Main St"), and a word-form drift
# ("Books"/"Book Store") -- each with matching city + phone so the pair is
# unambiguous. The singletons share no name, city, or phone with anything.
#
# Phones are fully distinct across DIFFERENT businesses (no shared "555-" middle
# digits): a shared exchange makes every phone token-similar, which at the low
# default threshold silently drags unrelated records into one over-merged
# cluster (the token_sort_ratio on "555-1000" vs "555-8000" is high). Distinct
# numbers keep the four dup pairs -- which alone share an *identical* phone --
# the only phone matches, so the clustering is legible at threshold 0.5 and 0.7.
RECORDS: list[dict[str, object]] = [
    # -- duplicate pair 1: possessive / spelling drift
    {"id": "b01", "name": "Bob's Diner", "city": "Springfield", "phone": "217-340-1187"},
    {"id": "b02", "name": "Bob Diner", "city": "Springfield", "phone": "217-340-1187"},
    # -- duplicate pair 2: corporate suffix
    {"id": "b03", "name": "Acme Hardware", "city": "Portland", "phone": "503-782-6421"},
    {"id": "b04", "name": "Acme Hardware Inc", "city": "Portland", "phone": "503-782-6421"},
    # -- duplicate pair 3: street abbreviation
    {"id": "b05", "name": "123 Main Street Cafe", "city": "Austin", "phone": "512-916-2038"},
    {"id": "b06", "name": "123 Main St Cafe", "city": "Austin", "phone": "512-916-2038"},
    # -- duplicate pair 4: word-form drift
    {"id": "b07", "name": "Riverside Books", "city": "Denver", "phone": "303-274-8890"},
    {"id": "b08", "name": "Riverside Book Store", "city": "Denver", "phone": "303-274-8890"},
    # -- six clear non-duplicates
    {"id": "b09", "name": "Quantum Dynamics LLC", "city": "Seattle", "phone": "206-118-5501"},
    {"id": "b10", "name": "Sunflower Bakery", "city": "Miami", "phone": "305-449-7712"},
    {"id": "b11", "name": "Peak Summit Gear", "city": "Boulder", "phone": "720-863-0294"},
    {"id": "b12", "name": "Harbor Point Marina", "city": "Boston", "phone": "617-528-3345"},
    {"id": "b13", "name": "Ironclad Security", "city": "Chicago", "phone": "312-905-6677"},
    {"id": "b14", "name": "Velvet Note Music", "city": "Nashville", "phone": "615-737-9931"},
]

# Hand-labeled ground truth: the four duplicate pairs. Singletons are omitted --
# BCubed/pairwise helpers and the Clusterer both treat an unlisted id as its own
# singleton, so this is the canonical multi-record-cluster gold set.
GOLD_CLUSTERS: list[set[str]] = [
    {"b01", "b02"},
    {"b03", "b04"},
    {"b05", "b06"},
    {"b07", "b08"},
]
