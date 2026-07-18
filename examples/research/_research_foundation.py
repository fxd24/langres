"""Shared zero-network fixtures for the research-foundation examples."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from langres.architectures import Retrieve, RetrieveRerank
from langres.core.resolver import ERModel
from langres.core.spend import SpendMonitor
from langres.experiments import ArchitectureFactory
from langres.resources import FakeEmbedder, FakeReranker


class ResearchRecord(BaseModel):
    """Common display-text projection for the bundled product benchmarks."""

    id: str
    name: str | None = None
    title: str | None = None
    description: str | None = None
    manufacturer: str | None = None

def retrieve_factory(
    *,
    name: str = "Retrieve",
    rerank: bool = False,
) -> ArchitectureFactory:
    """Build a runner factory over deterministic local resources."""

    def build(threshold: float, monitor: SpendMonitor) -> ERModel:
        common = {
            "embedder": FakeEmbedder(dimension=16),
            "schema": ResearchRecord,
            "retrieve_k": 5,
            "threshold": threshold,
            "budget_usd": monitor.budget_usd,
        }
        if rerank:
            return RetrieveRerank(
                **common,
                reranker=FakeReranker(),
            )
        return Retrieve(**common)

    typed_build: Callable[[float, SpendMonitor], ERModel] = build
    return ArchitectureFactory(name=name, factory=typed_build)
