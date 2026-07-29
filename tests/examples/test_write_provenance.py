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

    def test_a_nonexistent_path_has_nothing_to_lose(self, tmp_path: Path) -> None:
        assert PROV._uncommitted(tmp_path / "absent.json") == []

    def test_an_existing_file_outside_the_repo_counts_as_pending(self, tmp_path: Path) -> None:
        """The free pass went to exactly the file the refusal tells you to make.

        ``git status`` says nothing about a path outside the repository, and the
        first version read that silence as "clean" — so ``--output`` pointed at
        the operator's recovery copy was overwritten without ``--force``.
        """
        outside = tmp_path / "recovery_copy.json"
        outside.write_text("{}")

        pending = PROV._uncommitted(outside)

        assert pending and "outside this repository" in pending[0]

    def test_an_existing_ignored_file_counts_as_pending(self, tmp_path: Path) -> None:
        """`git status` omits ignored paths entirely, so silence is not safety."""
        ignored = Path(PROV.REPO_ROOT) / "tmp" / "provenance_guard_probe.json"
        ignored.parent.mkdir(parents=True, exist_ok=True)
        ignored.write_text("{}")
        try:
            pending = PROV._uncommitted(ignored)
        finally:
            ignored.unlink()

        assert pending and "ignored by git" in pending[0]

    def test_a_committed_unmodified_file_is_not_pending(self) -> None:
        """The guard must not refuse every ordinary run."""
        assert PROV._uncommitted(Path(PROV.REPO_ROOT) / "pyproject.toml") == []


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


class TestADeletedInputIsRecordedNotFatal:
    """The one event most worth recording used to be the one guaranteed to be lost.

    ``git hash-object`` exits 128 on a missing file. With ``check=True`` that
    propagated out of ``--finish`` *while the snapshot was being built* — before
    ``finished`` or ``verified_unchanged_during_run`` had been written — so the
    driver committed measured rows beside a still-open start sidecar. Deleting a
    tracked input mid-sweep is still fatal; it is now fatal **after** being
    written down.
    """

    def test_a_missing_file_hashes_to_the_sentinel(self) -> None:
        assert PROV._worktree_blob("does/not/exist.toml") == "absent"

    def test_a_present_file_still_hashes_normally(self) -> None:
        blob = PROV._worktree_blob("pyproject.toml")

        assert blob != "absent"
        assert len(blob) == 40

    def test_the_deletion_is_persisted_as_a_change(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The start snapshot holds a real blob; the finish snapshot holds "absent"."""
        monkeypatch.setattr(PROV, "TRACKED", ("docs/deleted-by-this-test.py",))
        path = _sidecar(
            tmp_path,
            blobs={"docs/deleted-by-this-test.py": {"blob": "0" * 40, "last_commit": "abc"}},
        )

        with pytest.raises(SystemExit):
            PROV.finish(path)

        doc = json.loads(path.read_text())
        assert doc["measurement_window"]["finished"] is not None
        assert doc["verified_unchanged_during_run"] is False
        assert doc["changed_during_run"] == ["docs/deleted-by-this-test.py"]

    def test_a_real_git_failure_is_not_mapped_to_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fail-open is the failure mode this repo keeps hitting; the sentinel is not one."""
        import subprocess

        class _Broken:
            returncode = 1
            stdout = ""
            stderr = "fatal: not a git repository"

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Broken())

        with pytest.raises(RuntimeError, match="git hash-object failed"):
            PROV._worktree_blob("pyproject.toml")
