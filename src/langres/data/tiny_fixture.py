"""Tiny synthetic cross-source fixture benchmark, built via the loader factory (Wave B).

A ~40-line dataset module that proves :func:`~langres.data._deepmatcher_loader.make_deepmatcher_benchmark`
end-to-end: a dataset-namespaced schema (:class:`TinyFixtureSchema`) plus one
factory call yields the ``load`` / ``load_pair_splits`` / :class:`TinyFixtureBenchmark`
triple, exactly the shape a real DeepMatcher dataset (WDC, DBLP-ACM, …) uses.

The data is **fully synthetic** (invented product names, no PII, no third-party
data — see ``datasets/tiny_fixture/ATTRIBUTION.md``), so there is no license
concern. It exists to validate the factory + registry and to give CI a fast,
offline end-to-end (blocked with an ``AllPairsBlocker``, judged with rapidfuzz —
no embeddings needed). It is registered ``loadable=True`` in
:mod:`langres.data.registry`.
"""

from typing import Literal

from pydantic import BaseModel, computed_field

from langres.data._deepmatcher_loader import SourceTable, make_deepmatcher_benchmark

__all__ = [
    "TINY_FIXTURE_ACHIEVED_PC",
    "TINY_FIXTURE_BLOCKING_K",
    "TINY_FIXTURE_GATE_MET",
    "TINY_FIXTURE_THRESHOLD_GRID",
    "TinyFixtureBenchmark",
    "TinyFixtureSchema",
    "load_tiny_fixture",
    "load_tiny_fixture_pair_splits",
]

_DATASET_PACKAGE = "langres.data.datasets.tiny_fixture"

TinyFixtureSource = Literal["a", "b"]

#: Pinned blocking k. The fixture is trivially small (12 records); every match is
#: a near neighbour, so a small k already yields perfect Pair-Completeness.
TINY_FIXTURE_BLOCKING_K = 5

#: Clusterer thresholds swept when racing methods (mirrors the other adapters).
TINY_FIXTURE_THRESHOLD_GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

#: Cross-source Pair-Completeness at :data:`TINY_FIXTURE_BLOCKING_K` — the three
#: matches are near-duplicate names, so vector blocking captures all of them.
TINY_FIXTURE_ACHIEVED_PC = 1.0

#: Whether :data:`TINY_FIXTURE_ACHIEVED_PC` clears a nominal recall gate.
TINY_FIXTURE_GATE_MET = True


class TinyFixtureSchema(BaseModel):
    """A single synthetic product record from the tiny fixture.

    ``id`` is globally unique across both sources (the loader prefixes the raw
    source id with ``a``/``b``). ``embed_text`` is the serializable blocking text.

    Attributes:
        id: Globally-unique record id (e.g. ``"a1"`` / ``"b3"``).
        name: Product name (always present).
        description: Free-text description, if present.
        source: Originating table (``"a"`` for ``tableA``, ``"b"`` for ``tableB``).
    """

    id: str
    name: str
    description: str | None = None
    source: TinyFixtureSource

    @computed_field  # type: ignore[prop-decorator]
    @property
    def embed_text(self) -> str:
        """Blocking text: name and description joined by a space."""
        return " ".join(p for p in [self.name, self.description] if p)


load_tiny_fixture, load_tiny_fixture_pair_splits, TinyFixtureBenchmark = make_deepmatcher_benchmark(
    name="tiny_fixture",
    schema=TinyFixtureSchema,
    dataset_package=_DATASET_PACKAGE,
    table_a=SourceTable(file="tableA.csv", source="a", id_prefix="a"),
    table_b=SourceTable(file="tableB.csv", source="b", id_prefix="b"),
    split_files={"train": "train.csv", "valid": "valid.csv", "test": "test.csv"},
    blocking_k=TINY_FIXTURE_BLOCKING_K,
    threshold_grid=TINY_FIXTURE_THRESHOLD_GRID,
    achieved_pc=TINY_FIXTURE_ACHIEVED_PC,
    gate_met=TINY_FIXTURE_GATE_MET,
)
