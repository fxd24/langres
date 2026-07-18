"""Shared zero-network fixtures for the research-foundation examples."""

from __future__ import annotations

import random
from collections.abc import Callable

from pydantic import BaseModel

from langres.architectures import Retrieve, RetrieveRerank
from langres.core.resolver import ERModel
from langres.core.spend import SpendMonitor
from langres.data.benchmark import gold_pairs_from_clusters
from langres.data.registry import BenchmarkEntry, list_benchmarks, register
from langres.experiments import ArchitectureFactory
from langres.resources import FakeEmbedder, FakeReranker


class ResearchRecord(BaseModel):
    """Common display-text projection for the local research fixtures."""

    id: str
    name: str | None = None
    title: str | None = None
    description: str | None = None
    manufacturer: str | None = None


class _LocalBenchmark:
    """Small deterministic benchmark used only by the copy-paste examples."""

    threshold_grid = (0.3, 0.5, 0.7)
    records: tuple[ResearchRecord, ...]
    clusters: tuple[frozenset[str], ...]

    def load(
        self,
    ) -> tuple[list[ResearchRecord], list[set[str]], set[frozenset[str]]]:
        clusters = [set(cluster) for cluster in self.clusters]
        return list(self.records), clusters, gold_pairs_from_clusters(clusters)

    def split(
        self,
        corpus: list[ResearchRecord],
        gold_clusters: list[set[str]],
        *,
        seed: int,
    ) -> tuple[
        list[ResearchRecord],
        list[ResearchRecord],
        list[set[str]],
        list[set[str]],
    ]:
        clusters = [set(cluster) for cluster in gold_clusters]
        random.Random(seed).shuffle(clusters)
        midpoint = len(clusters) // 2
        train_clusters, test_clusters = clusters[:midpoint], clusters[midpoint:]
        train_ids = {record_id for cluster in train_clusters for record_id in cluster}
        test_ids = {record_id for cluster in test_clusters for record_id in cluster}
        train = [record for record in corpus if record.id in train_ids]
        test = [record for record in corpus if record.id in test_ids]
        return train, test, train_clusters, test_clusters


class LocalCompaniesBenchmark(_LocalBenchmark):
    """Two-source company-name fixture with four match clusters."""

    name = "local_companies"
    records = (
        ResearchRecord(id="co-1a", name="Acme Corporation"),
        ResearchRecord(id="co-1b", name="ACME Corp"),
        ResearchRecord(id="co-2a", name="Globex LLC"),
        ResearchRecord(id="co-2b", name="Globex Limited"),
        ResearchRecord(id="co-3a", name="Initech Incorporated"),
        ResearchRecord(id="co-3b", name="Initech Inc"),
        ResearchRecord(id="co-4a", name="Umbrella Health"),
        ResearchRecord(id="co-4b", name="Umbrella Healthcare"),
    )
    clusters = (
        frozenset(("co-1a", "co-1b")),
        frozenset(("co-2a", "co-2b")),
        frozenset(("co-3a", "co-3b")),
        frozenset(("co-4a", "co-4b")),
    )


class LocalProductsBenchmark(_LocalBenchmark):
    """Two-source product-title fixture with four match clusters."""

    name = "local_products"
    records = (
        ResearchRecord(id="pr-1a", title="Atlas Mechanical Keyboard"),
        ResearchRecord(id="pr-1b", title="Atlas mech keyboard"),
        ResearchRecord(id="pr-2a", title="Nimbus Wireless Mouse"),
        ResearchRecord(id="pr-2b", title="Nimbus cordless mouse"),
        ResearchRecord(id="pr-3a", title="Helios 27 inch Monitor"),
        ResearchRecord(id="pr-3b", title="Helios 27in display"),
        ResearchRecord(id="pr-4a", title="Orion USB C Dock"),
        ResearchRecord(id="pr-4b", title="Orion USB-C docking station"),
    )
    clusters = (
        frozenset(("pr-1a", "pr-1b")),
        frozenset(("pr-2a", "pr-2b")),
        frozenset(("pr-3a", "pr-3b")),
        frozenset(("pr-4a", "pr-4b")),
    )


def _register_local_benchmarks() -> None:
    existing = {entry.name for entry in list_benchmarks()}
    for entry in (
        BenchmarkEntry(
            name=LocalCompaniesBenchmark.name,
            task="linkage",
            domain="company",
            loadable=True,
            module_path="_research_foundation",
            loader_symbol="LocalCompaniesBenchmark",
        ),
        BenchmarkEntry(
            name=LocalProductsBenchmark.name,
            task="linkage",
            domain="product",
            loadable=True,
            module_path="_research_foundation",
            loader_symbol="LocalProductsBenchmark",
        ),
    ):
        if entry.name not in existing:
            register(entry)


_register_local_benchmarks()


def retrieve_factory(
    *,
    name: str = "Retrieve",
    rerank: bool = False,
) -> ArchitectureFactory:
    """Build a runner factory over deterministic local resources."""

    def build(threshold: float, monitor: SpendMonitor) -> ERModel:
        if rerank:
            return RetrieveRerank(
                embedder=FakeEmbedder(dimension=16),
                reranker=FakeReranker(),
                schema=ResearchRecord,
                retrieve_k=5,
                threshold=threshold,
                monitor=monitor,
            )
        return Retrieve(
            embedder=FakeEmbedder(dimension=16),
            schema=ResearchRecord,
            retrieve_k=5,
            threshold=threshold,
            monitor=monitor,
        )

    typed_build: Callable[[float, SpendMonitor], ERModel] = build
    return ArchitectureFactory(name=name, factory=typed_build)
