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


def test_crosscheck_is_dropped_when_the_checkpoint_revision_moved(
    harness: ModuleType, rows: list[dict[str, Any]]
) -> None:
    """Section F claims "one set of embeddings"; a moved pin makes that false."""
    reference = json.loads(harness.REFERENCE_PATH.read_text())
    remeasured = [dict(r, model_revision="0" * 40) for r in rows]

    rendered = harness.render(remeasured, reference)

    assert "## F. Where the published protocol" not in rendered


def test_partial_renders_omit_the_fixed_verdict(
    harness: ModuleType, rows: list[dict[str, Any]]
) -> None:
    """The verdict is prose about the full sweep, so a subset must not assert it."""
    reference = json.loads(harness.REFERENCE_PATH.read_text())
    partial = [r for r in rows if r["benchmark"] == "febrl_person"]

    rendered = harness.render(partial, reference)

    assert "this is a **partial** render" in rendered
    assert "4 of 6" not in rendered
    assert "98.82" not in rendered


def test_full_render_keeps_the_verdict(harness: ModuleType, rows: list[dict[str, Any]]) -> None:
    """The complement of the test above -- otherwise it could pass by never firing."""
    reference = json.loads(harness.REFERENCE_PATH.read_text())

    rendered = harness.render(rows, reference)

    assert "this is a **partial** render" not in rendered
    assert "4 of 6" in rendered


def test_a_cell_remeasured_at_a_reduced_k_does_not_count_as_present(
    harness: ModuleType, rows: list[dict[str, Any]]
) -> None:
    """Every key present is not the same as every measurement present.

    ``--k-max 20`` on one cell keeps all 16 keys while destroying the k=150
    number the verdict quotes, so the gate has to look at the measurement.
    """
    reference = json.loads(harness.REFERENCE_PATH.read_text())
    truncated = [
        dict(r, k_max=20, pc=r["pc"][:20], pq=r["pq"][:20])
        if r["benchmark"] == "dblp_scholar"
        else r
        for r in rows
    ]

    rendered = harness.render(truncated, reference)

    assert "this is a **partial** render" in rendered
    assert "98.82" not in rendered


def test_k_at_pc90_reports_the_depth_actually_searched(harness: ModuleType) -> None:
    """A row capped below the UniBlocker budget must not claim k=100 was searched."""
    assert harness._searched_to({"k_max": 150}) == ">100"
    assert harness._searched_to({"k_max": 100}) == ">100"
    assert harness._searched_to({"k_max": 20}) == ">20"


def test_every_number_the_verdict_quotes_is_checked_against_the_rows(
    harness: ModuleType, rows: list[dict[str, Any]]
) -> None:
    """The registry must describe the committed rows, or it gates nothing."""
    for benchmark, model, k, field, value in harness.VERDICT_CLAIMS:
        assert harness._claim_holds(rows, benchmark, model, k, field, value), (
            f"{benchmark} x {model} @k={k} {field} is no longer {value}"
        )


def test_a_moved_measurement_drops_the_verdict(
    harness: ModuleType, rows: list[dict[str, Any]]
) -> None:
    """A re-pin that shifts a quoted number must not leave the prose asserting it."""
    reference = json.loads(harness.REFERENCE_PATH.read_text())
    moved = [
        dict(r, pc=[p * 0.5 for p in r["pc"]])
        if (r["benchmark"], r["model"]) == ("dblp_scholar", harness._MPNET)
        else r
        for r in rows
    ]

    rendered = harness.render(moved, reference)

    assert "this is a **partial** render" in rendered
    assert "98.82" not in rendered


def test_the_named_crosschecks_are_required_not_just_any(
    harness: ModuleType, rows: list[dict[str, Any]]
) -> None:
    """The verdict cites abt_buy and amazon_google by name, so dblp_acm alone is not enough."""
    reference = json.loads(harness.REFERENCE_PATH.read_text())
    checks = json.loads(harness.CROSSCHECK_PATH.read_text())
    assert {c["benchmark"] for c in checks} >= set(harness.VERDICT_CROSSCHECKS)

    # Drop the two named cells from the rendered rows: admission then keeps only the
    # dblp_acm check, which is a surviving check but not a cited one.
    without = [r for r in rows if (r["benchmark"], r["model"]) != ("abt_buy", harness._MPNET)]

    rendered = harness.render(without, reference)

    assert "this is a **partial** render" in rendered


def test_a_cell_with_two_unpinned_generations_is_dropped_not_guessed(
    harness: ModuleType, rows: list[dict[str, Any]]
) -> None:
    """Rows are additive across pins, so render must choose deliberately."""
    one = rows[0]
    ambiguous = [dict(one, model_revision="a" * 40), dict(one, model_revision="b" * 40)]

    selected = harness._select_generation(ambiguous, "probe")

    assert selected == []


def test_the_current_pin_wins_over_a_legacy_row(
    harness: ModuleType, rows: list[dict[str, Any]]
) -> None:
    one = next(r for r in rows if r["model"] in harness.MODEL_REVISIONS)
    pin = harness.MODEL_REVISIONS[one["model"]]
    legacy = dict(one)
    pinned = dict(one, model_revision=pin)

    selected = harness._select_generation([legacy, pinned], "probe")

    assert selected == [pinned]


def test_an_unpinned_model_is_not_disk_cached(harness: ModuleType, tmp_path: Path) -> None:
    """A namespace keyed on the name alone would outlive the weights it holds."""
    embedder, revision = harness._build_embedder("hf-internal-testing/tiny-random-gpt2", tmp_path)

    assert revision is None
    assert type(embedder).__name__ == "SentenceTransformerEmbedder"


def test_committed_crosscheck_cells_are_unique(harness: ModuleType) -> None:
    """The merge keys on this tuple, so a duplicate would mean silent cell loss."""
    checks = json.loads(harness.CROSSCHECK_PATH.read_text())
    keys = [
        (c["benchmark"], c["model"], c.get("model_revision"), c["k"], c.get("metric_revision"))
        for c in checks
    ]

    assert len(keys) == len(set(keys))


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
