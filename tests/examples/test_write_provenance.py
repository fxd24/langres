"""Behaviour tests for the sweep-provenance sidecar writer.

Scoped to the one property a caller cannot check for itself: a window that
closed COMPLETE must never be relabelled partial. Everything else in this module
(`--start`, the tree digests, the stability verdict) reads git and is exercised
by the drivers.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[2]
MODULE_PATH = ROOT / "examples" / "research" / "write_provenance.py"


def _load() -> ModuleType:
    name = "example_write_provenance"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROV = _load()


def _sidecar(tmp_path: Path, finished: str | None = None, **overrides: object) -> Path:
    doc: dict[str, object] = {
        "blobs": {},
        "tree_digests": {},
        "driver_blobs": {},
        "driver_digest": "abc",
        "environment_blobs": {},
        "studies_measured": ["a", "b"],
        "measurement_window": {
            "started": "2026-07-29T05:00:00+02:00",
            "finished": finished,
            "head_when_sweep_started": "deadbeef",
        },
    }
    doc.update(overrides)
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(doc))
    return path


CLOSED = "2026-07-29T06:00:00+02:00"


class TestAClosedWindowStaysClosed:
    """Both re-finish directions rewrite history, and both were reachable.

    The driver's abort path re-invokes ``--finish --partial``, so a failure AFTER
    a successful ``--finish`` — a render error, say — flipped ``window_complete``
    to false over rows that prove otherwise. The mirror image is worse: a plain
    ``--finish`` over a partial sidecar sets ``window_complete`` true while
    KEEPING the ``partial_note`` saying the sweep aborted. Either way ``finished``
    is restamped and the stability verdict is re-derived from whatever the tree
    looks like now.
    """

    def test_a_complete_window_refuses_the_partial_downgrade(self, tmp_path: Path) -> None:
        path = _sidecar(tmp_path, finished=CLOSED, window_complete=True)

        with pytest.raises(SystemExit, match="already closed"):
            PROV.finish(path, partial=True)

    def test_a_partial_window_refuses_the_complete_upgrade(self, tmp_path: Path) -> None:
        """The direction the first version of this guard left open."""
        path = _sidecar(
            tmp_path,
            finished=CLOSED,
            window_complete=False,
            partial_note="The sweep ABORTED before finishing every planned study.",
        )

        with pytest.raises(SystemExit, match="already closed"):
            PROV.finish(path, partial=False)

    def test_the_refusal_leaves_the_sidecar_untouched(self, tmp_path: Path) -> None:
        """Refusing after writing would be no better than not refusing."""
        path = _sidecar(tmp_path, finished=CLOSED, window_complete=True)
        before = path.read_text()

        with pytest.raises(SystemExit):
            PROV.finish(path, partial=True)

        assert path.read_text() == before

    def test_an_open_window_can_still_be_marked_partial(self, tmp_path: Path) -> None:
        """The genuine abort path — no `--finish` has run — must keep working."""
        path = _sidecar(tmp_path)

        PROV.finish(path, partial=True)

        doc = json.loads(path.read_text())
        assert doc["window_complete"] is False
        assert "ABORTED" in doc["partial_note"]

    def test_a_missing_sidecar_says_to_run_start(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="run --start"):
            PROV.finish(tmp_path / "absent.json", partial=True)


class TestStartDoesNotDestroyUncommittedProvenance:
    """``start()``'s job is to overwrite a tracked file. That is the hazard.

    A run killed between ``--finish`` writing the closed record and the driver
    committing it leaves the sidecar on disk as the ONLY description of rows
    ``run_ladder.sh`` already committed. Opening the next window replaced those
    bytes, and nothing in git can bring them back.
    """

    def test_a_sidecar_with_uncommitted_changes_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _sidecar(tmp_path, finished=CLOSED, window_complete=True)
        before = path.read_text()
        monkeypatch.setattr(PROV, "_uncommitted", lambda _p: ["docs/research/prov.json"])

        with pytest.raises(SystemExit, match="uncommitted changes"):
            PROV.start(path)

        assert path.read_text() == before

    def test_force_overwrites_deliberately(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refusal with no way past it is a dead end, not a safeguard."""
        path = _sidecar(tmp_path, finished=CLOSED, window_complete=True)
        monkeypatch.setattr(PROV, "_uncommitted", lambda _p: ["docs/research/prov.json"])

        PROV.start(path, force=True)

        assert json.loads(path.read_text())["measurement_window"]["finished"] is None

    def test_a_clean_sidecar_opens_normally(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _sidecar(tmp_path, finished=CLOSED, window_complete=True)
        monkeypatch.setattr(PROV, "_uncommitted", lambda _p: [])

        PROV.start(path)

        assert json.loads(path.read_text())["measurement_window"]["finished"] is None

    def test_a_path_outside_the_repo_is_not_blocked_by_git(self, tmp_path: Path) -> None:
        """`_uncommitted` must not explode on a pathspec git cannot resolve."""
        assert PROV._uncommitted(tmp_path / "elsewhere.json") == []


class TestDirtyScopeCoversEveryTrackedFile:
    """The dirty check ran over directories, so a root-level file escaped it.

    ``pyproject.toml`` is tracked individually at the repo root, outside both
    ``TRACKED_TREES`` scopes. An uncommitted dependency edit present at
    ``--start`` and untouched during the sweep therefore passed the endpoint
    comparison AND left ``dirty_at_start`` empty — the report naming the last
    committed revision with no modified-tree warning, over a checkout that
    cannot reproduce the measured environment.
    """

    def test_every_tracked_file_falls_inside_a_dirty_scope(self) -> None:
        for path in PROV.TRACKED:
            assert any(
                path == scope or path.startswith(f"{scope}/") for scope in PROV.DIRTY_SCOPES
            ), f"{path} is hashed but no dirty scope would report it modified"

    def test_the_root_level_file_is_the_one_that_used_to_escape(self) -> None:
        """Named explicitly: a passing derived rule over an empty set proves nothing."""
        assert "pyproject.toml" in PROV.TRACKED
        assert "pyproject.toml" in PROV.DIRTY_SCOPES
        assert not any("pyproject.toml".startswith(f"{t}/") for t in PROV.TRACKED_TREES)

    def test_the_scope_is_a_valid_pathspec_git_will_honour(self) -> None:
        """A scope git rejects or silently ignores would report clean forever.

        The dynamic proof — modify `pyproject.toml`, observe it appear — lives in
        `tmp/probe_dirty_scope.py`, which cannot run here without dirtying the
        tree the rest of the suite reads.
        """
        assert PROV._dirty(("pyproject.toml",)) == []
        assert "pyproject.toml" not in PROV._dirty(PROV.TRACKED_TREES)


class TestEnvironmentIsObserved:
    """Source identity is not measurement identity.

    Every cell is a fresh ``uv run`` with neither ``--no-sync`` nor ``--locked``,
    so the dependency environment can move between cells while every source hash
    compares equal and the run is certified "unchanged".
    """

    def test_pyproject_is_hashed_as_measurement_code(self) -> None:
        """Fatal, like the sources: it does not move on its own."""
        assert "pyproject.toml" in PROV.TRACKED

    def test_the_lockfile_is_recorded_without_being_fatal(self, tmp_path: Path) -> None:
        """`uv.lock` is gitignored and uv rewrites it unprompted."""
        assert PROV.ENVIRONMENT == ("uv.lock",)

        path = _sidecar(tmp_path, environment_blobs={"uv.lock": "0" * 40})
        PROV.finish(path)  # must NOT raise

        doc = json.loads(path.read_text())
        assert doc["environment_unchanged_during_run"] is False
        assert doc["environment_changed_during_run"] == ["uv.lock"]

    def test_a_window_predating_the_field_reports_not_observed(self, tmp_path: Path) -> None:
        """`null` means "not observed", never "unchanged" and never "changed"."""
        path = _sidecar(tmp_path)
        json_doc = json.loads(path.read_text())
        del json_doc["environment_blobs"]
        path.write_text(json.dumps(json_doc))

        PROV.finish(path)

        assert json.loads(path.read_text())["environment_unchanged_during_run"] is None

    def test_an_unchanged_environment_is_recorded_as_such(self, tmp_path: Path) -> None:
        path = _sidecar(tmp_path, environment_blobs=PROV._environment_blobs())

        PROV.finish(path)

        assert json.loads(path.read_text())["environment_unchanged_during_run"] is True
