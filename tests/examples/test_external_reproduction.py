"""Offline behaviour coverage for the external-reproduction research harness.

The harness produces four **tracked** artifacts (a rows JSONL, a crosscheck
JSON, a hand-transcribed reference JSON and the rendered report). Nothing here
measures anything -- these tests exercise the parts that decide *which measured
rows a report is allowed to describe*, because those are the parts whose failure
mode is a plausible-looking report built from the wrong data.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).parents[2]
HARNESS = ROOT / "examples" / "research" / "external_reproduction.py"


def _load() -> ModuleType:
    name = "example_external_reproduction"
    sys.path.insert(0, str(HARNESS.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, HARNESS)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # Register BEFORE executing: the harness declares a frozen dataclass under
        # `from __future__ import annotations`, and dataclasses resolves those
        # string annotations through `sys.modules[cls.__module__]`. An unregistered
        # module makes that lookup return None and the class definition raises.
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(HARNESS.parent))


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    return _load()


@pytest.fixture(scope="module")
def rows(harness: ModuleType) -> list[dict[str, Any]]:
    return harness._load_rows(harness.ROWS_PATH)


def test_every_default_model_is_revision_pinned(harness: ModuleType) -> None:
    """An unpinned checkpoint silently drifts as the Hub's ``main`` moves."""
    for model in harness.DEFAULT_MODELS:
        revision = harness.MODEL_REVISIONS.get(model)
        assert revision is not None, f"{model} is not pinned"
        assert len(revision) == 40, f"{model} is not pinned to a full commit sha"
        assert set(revision) <= set("0123456789abcdef")


def test_committed_rows_are_all_at_the_current_metric_revision(
    harness: ModuleType, rows: list[dict[str, Any]]
) -> None:
    assert rows, "the tracked rows file is empty"
    assert {r["metric_revision"] for r in rows} == {harness.METRIC_REVISION}


def test_stale_metric_revisions_are_dropped_not_rendered(harness: ModuleType) -> None:
    """The gate must be able to observe the failure it exists to catch.

    ``_write_row`` keys on the metric revision so bumping it preserves the older
    generation. That is only safe if rendering then refuses the stale rows.
    """
    current = {"benchmark": "abt_buy", "model": "m", "metric_revision": harness.METRIC_REVISION}
    stale = {"benchmark": "abt_buy", "model": "m", "metric_revision": harness.METRIC_REVISION - 1}
    missing = {"benchmark": "abt_buy", "model": "m"}

    kept = harness._current_revision_only([current, stale, missing], "probe")

    assert kept == [current]


def test_render_reproduces_the_tracked_report(
    harness: ModuleType, rows: list[dict[str, Any]]
) -> None:
    """The committed report is exactly what the committed rows render to."""
    reference = json.loads(harness.REFERENCE_PATH.read_text())

    rendered = harness.render(rows, reference)

    assert rendered == harness.REPORT_PATH.read_text()


def test_crosscheck_section_is_dropped_when_its_cells_are_not_rendered(
    harness: ModuleType, rows: list[dict[str, Any]]
) -> None:
    """Section F must not describe an experiment the rest of the report is not about.

    The crosscheck artifact is written by a separate run, so it can outlive the
    rows it was paired with. Rendering a single unrelated row must therefore drop
    the section rather than quietly attribute someone else's measurement.
    """
    reference = json.loads(harness.REFERENCE_PATH.read_text())
    unrelated = [r for r in rows if r["benchmark"] == "febrl_person"]
    assert unrelated, "expected a febrl_person row, which no crosscheck entry covers"

    rendered = harness.render(unrelated, reference)

    assert "## F. Where the published protocol" not in rendered


def test_crosscheck_entries_cover_only_benchmarks_the_report_measures(
    harness: ModuleType, rows: list[dict[str, Any]]
) -> None:
    checks = json.loads(
        (harness.RESEARCH_DIR / "20260728_external_reproduction_crosscheck.json").read_text()
    )
    cells = {(r["benchmark"], r["model"]) for r in rows}

    assert checks
    for entry in checks:
        assert entry["metric_revision"] == harness.METRIC_REVISION
        assert (entry["benchmark"], entry["model"]) in cells
