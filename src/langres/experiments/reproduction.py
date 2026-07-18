"""Safe, weightless experiment handoff bundles and verification."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TextIO

from pydantic import BaseModel, ConfigDict

from langres.experiments.identity import SourceState
from langres.experiments.report import ExperimentReport


class ReproductionArchitecture(BaseModel):
    """One architecture's safe execution-plan snapshot."""

    model_config = ConfigDict(frozen=True)

    name: str
    variant_id: str
    cache_semantics: str
    estimated_usd: float | None = None
    execution_plan: dict[str, Any]


class ReproductionBundle(BaseModel):
    """A validated local handoff with no executable Python callables."""

    model_config = ConfigDict(frozen=True)

    version: Literal[1] = 1
    source: SourceState
    architectures: tuple[ReproductionArchitecture, ...]
    report: ExperimentReport


def write_reproduction_bundle(
    path: str | Path,
    *,
    source: SourceState,
    architectures: Sequence[ReproductionArchitecture],
    report: ExperimentReport,
) -> Path:
    """Atomically write the JSON bundle consumed by the CLI verifier."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    bundle = ReproductionBundle(
        source=source,
        architectures=tuple(architectures),
        report=report,
    )
    temporary.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_reproduction_bundle(path: str | Path) -> ReproductionBundle:
    """Validate a reproduction bundle without importing resource backends."""
    source = Path(path)
    try:
        payload = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(
            f"Cannot load reproduction bundle {source}. "
            "Cause: the artifact is missing or unreadable. "
            "Fix: pass the JSON path emitted by ExperimentReport.reproduce_command."
        ) from exc
    try:
        return ReproductionBundle.model_validate_json(payload)
    except ValueError as exc:
        raise ValueError(
            f"Cannot validate reproduction bundle {source}. "
            "Cause: its schema or embedded protocol/report is incompatible. "
            "Fix: regenerate the bundle with the current langres version."
        ) from exc


def verify_reproduction_bundle(path: str | Path, *, output: TextIO) -> ReproductionBundle:
    """Load a handoff and print a bounded, content-free verification summary."""
    bundle = load_reproduction_bundle(path)
    names = ", ".join(architecture.name for architecture in bundle.architectures)
    output.write(
        "Verified reproduction bundle "
        f"{Path(path)}: {len(bundle.report.runs)} run(s); architectures: {names}\n"
    )
    return bundle


__all__ = [
    "ReproductionArchitecture",
    "ReproductionBundle",
    "load_reproduction_bundle",
    "verify_reproduction_bundle",
    "write_reproduction_bundle",
]
