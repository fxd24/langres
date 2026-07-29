"""Behaviour tests for the sweep-provenance sidecar writer.

Scoped to the one property a caller cannot check for itself: a window that
closed COMPLETE must never be relabelled partial. Everything else in this module
(`--start`, the tree digests, the stability verdict) reads git and is exercised
by the drivers.
"""

from __future__ import annotations

import importlib.util
import json
import re
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
        assert "does NOT cover every row" in doc["partial_note"]

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

    def test_a_pending_deletion_of_a_tracked_sidecar_is_pending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``exists()`` is false for a staged deletion too — the same blind spot.

        Returning "nothing to lose" let ``--start`` recreate and commit the file
        over a deletion the operator had staged, without ``--force``. It is the
        existence test the shell artifact guards had already dropped for exactly
        this case, still living in the Python one.
        """
        tracked = Path(PROV.REPO_ROOT) / "docs" / "research" / "20260729_lfm25_provenance.json"
        monkeypatch.setattr(PROV, "_dirty", lambda _paths: [" D docs/research/x.json"])
        monkeypatch.setattr(Path, "exists", lambda _self: False)

        assert PROV._uncommitted(tracked) == [" D docs/research/x.json"]

    def test_a_path_that_was_never_created_is_still_free(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Asking git first must not make a first run refuse itself."""
        monkeypatch.setattr(PROV, "_dirty", lambda _paths: [])

        assert PROV._uncommitted(Path(PROV.REPO_ROOT) / "docs" / "research" / "never.json") == []


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


class TestAVanishedSidecarFailsVerification:
    """ "No sidecar" meant two different things through one empty test.

    The standing portfolio ladder runs this driver with no provenance at all and
    must not be blocked. But an LFM run that OPENED a window and then lost the
    sidecar hit the same branch: ``--verify`` passed, ``run_ladder.sh`` committed
    and pushed every later model's rows against no start snapshot, and
    ``--finish`` could neither close nor reconstruct the window.
    """

    def test_the_standalone_ladder_keeps_its_no_op(self, tmp_path: Path) -> None:
        PROV.verify(tmp_path / "absent.json")  # must NOT raise

    def test_a_run_that_opened_a_window_refuses_a_missing_sidecar(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="is GONE"):
            PROV.verify(tmp_path / "absent.json", required=True)

    def test_a_present_unchanged_sidecar_still_passes_when_required(self, tmp_path: Path) -> None:
        path = _sidecar(tmp_path)

        PROV.verify(path, required=True)  # must NOT raise

    def test_a_changed_tracked_file_is_still_caught(self, tmp_path: Path) -> None:
        path = _sidecar(tmp_path, blobs={"pyproject.toml": {"blob": "0" * 40}})

        with pytest.raises(SystemExit, match="changed mid-sweep"):
            PROV.verify(path, required=True)

    def test_a_key_absent_from_the_finish_snapshot_does_not_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``after["blobs"][p]`` assumed both snapshots share a key set.

        ``finish()`` was fixed for this last round and ``verify()`` was not —
        the same line, twice, one of them patched. A KeyError here aborts while
        BUILDING the verdict, which is the defect the sentinel exists to remove.
        """
        monkeypatch.setattr(PROV, "TRACKED", ())
        path = _sidecar(tmp_path, blobs={"gone/from/tracked.py": {"blob": "0" * 40}})

        with pytest.raises(SystemExit, match="changed mid-sweep"):
            PROV.verify(path)

    def test_the_driver_asks_for_it_and_the_flag_exists(self) -> None:
        """A flag the driver never passes is a guard that never runs."""
        driver = (Path(PROV.REPO_ROOT) / "examples" / "research" / "run_lfm25.sh").read_text()
        ladder = (Path(PROV.REPO_ROOT) / "examples" / "research" / "run_ladder.sh").read_text()

        assert "export LADDER_PROVENANCE_REQUIRED=1" in driver
        assert 'LADDER_PROVENANCE_REQUIRED:-0}" = "1" ] && required_arg=(--required)' in ladder


class TestThePreflightCoversWhatTheRendererReads:
    """``LFM25_STUDY=a`` still renders and COMMITS a write-up built from both studies.

    The preflight checked only the studies being re-measured, so an unselected
    study holding uncommitted rows or a hand edit had its bytes published as
    numbers in a document whose own header says every figure is read from the
    committed artifacts. Executable proof in ``tmp/probe_round19.sh``; this pins
    the driver so the loop cannot narrow again.
    """

    @staticmethod
    def _driver() -> str:
        return (Path(PROV.REPO_ROOT) / "examples" / "research" / "run_lfm25.sh").read_text()

    def test_the_dirty_check_iterates_both_studies_unconditionally(self) -> None:
        driver = self._driver()

        assert "for study in a b; do" in driver
        assert "for study in $PROVENANCE_STUDIES; do" not in driver

    def test_provenance_still_records_only_what_was_measured(self) -> None:
        """The two lists are different questions: what is read vs what is re-measured."""
        driver = self._driver()

        assert "PROVENANCE_STUDIES" in driver
        assert "--studies $PROVENANCE_STUDIES" in driver or "$PROVENANCE_STUDIES" in driver


class TestAClosedWindowIsNotAPassForAnActiveRun:
    """The escape hatch and the guard were the same branch.

    A closed window is a legitimate no-op for a caller that never opened one. For
    a run still measuring, an accidental or concurrent ``--finish`` closed the
    record early, and returning here made every later per-model check skip the
    snapshot comparison -- so rows measured after that timestamp, under code that
    may have moved, were pushed as though the closed window described them.
    """

    def test_a_closed_window_still_no_ops_for_a_non_participating_ladder(
        self, tmp_path: Path
    ) -> None:
        path = _sidecar(tmp_path, finished=CLOSED, window_complete=True)

        PROV.verify(path)  # must NOT raise

    def test_a_closed_window_refuses_when_the_run_opened_one(self, tmp_path: Path) -> None:
        path = _sidecar(tmp_path, finished=CLOSED, window_complete=True)

        with pytest.raises(SystemExit, match="already CLOSED"):
            PROV.verify(path, required=True)

    def test_the_refusal_names_the_finish_time(self, tmp_path: Path) -> None:
        """Which is the one fact that lets an operator work out what happened."""
        path = _sidecar(tmp_path, finished=CLOSED, window_complete=True)

        # re.escape: the offset's `+` is a regex quantifier, and `match=` is a regex.
        with pytest.raises(SystemExit, match=re.escape(CLOSED)):
            PROV.verify(path, required=True)


class TestThePreflightCoversWhatTheDriverOverwrites:
    """The per-study loop never named the two files this driver writes directly.

    The load probe is refreshed and committed before the render; the combined
    write-up is rendered and committed after it. A hand-edited write-up or an
    uncommitted probe was replaced with no refusal, by the script whose guard
    exists to prevent exactly that.
    """

    @staticmethod
    def _driver() -> str:
        return (Path(PROV.REPO_ROOT) / "examples" / "research" / "run_lfm25.sh").read_text()

    def test_both_direct_outputs_are_in_the_guarded_list(self) -> None:
        driver = self._driver()

        assert 'GUARDED_ARTIFACTS="$LOAD_PROBE_JSON $REPORT_MD"' in driver

    def test_the_paths_are_declared_once(self) -> None:
        """Two declarations is how the preflight and the writer drift apart."""
        driver = self._driver()

        assert driver.count('LOAD_PROBE_JSON="docs/research/') == 1
        assert driver.count('REPORT_MD="docs/research/') == 1

    def test_the_guarded_list_is_declared_before_the_preflight_uses_it(self) -> None:
        driver = self._driver()

        assert driver.index('LOAD_PROBE_JSON="docs/research/') < driver.index(
            'GUARDED_ARTIFACTS="$LOAD_PROBE_JSON'
        )


class TestDataIsCodeInTheDigest:
    """Every subprocess in a granular sweep reloads the tracked benchmark CSVs.

    A ``.py``-only digest certified a sweep unchanged across an edited corpus,
    and the report's population guard compares record COUNTS -- so a same-sized
    replacement passed there too. There was no check anywhere that could see it.
    """

    def test_the_benchmark_corpora_are_inside_a_tracked_tree(self) -> None:
        datasets = Path(PROV.REPO_ROOT) / "src" / "langres" / "data" / "datasets"

        assert datasets.exists()
        assert any(
            str(datasets).startswith(str(Path(PROV.REPO_ROOT) / t)) for t in PROV.TRACKED_TREES
        )

    def test_csv_inputs_are_digested(self) -> None:
        assert ".csv" in PROV.DIGESTED_SUFFIXES
        assert ".py" in PROV.DIGESTED_SUFFIXES

    def test_the_snapshot_records_what_the_digest_covers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The digest's SCOPE travels with the digest.

        The renderer used to hard-code "over every `.py`", which was true when
        written and false the moment data files joined DIGESTED_SUFFIXES. With
        nothing in the sidecar saying which contract applied, one of the two was
        always going to be described wrongly -- and the retrospective window
        committed here really is `.py`-only.
        """
        path = _sidecar(tmp_path)
        # A tmp_path sidecar sits outside the repo, which `start()` correctly
        # treats as unestablishable and refuses; that guard has its own tests.
        monkeypatch.setattr(PROV, "_uncommitted", lambda _p: [])

        PROV.start(path)

        assert json.loads(path.read_text())["tree_digest_suffixes"] == list(PROV.DIGESTED_SUFFIXES)

    def test_editing_a_corpus_changes_the_digest(self, tmp_path: Path) -> None:
        """The property the whole mechanism rests on, exercised rather than assumed."""
        corpus = (
            Path(PROV.REPO_ROOT)
            / "src"
            / "langres"
            / "data"
            / "datasets"
            / "fodors_zagat"
            / "fodors.csv"
        )
        assert corpus.exists(), "the corpus this test edits must exist, or it proves nothing"
        original = corpus.read_bytes()
        before = PROV._tree_digest()["src/langres"]
        try:
            corpus.write_bytes(original + b"\n")
            after = PROV._tree_digest()["src/langres"]
        finally:
            corpus.write_bytes(original)

        assert after != before
        assert PROV._tree_digest()["src/langres"] == before, "the test must leave no trace"

    def test_hashing_is_batched(self) -> None:
        """~500 subprocesses per snapshot, and verify() snapshots before every model."""
        paths = [Path(PROV.REPO_ROOT) / "pyproject.toml", Path(PROV.REPO_ROOT) / "README.md"]

        hashes = PROV._hash_many(paths)

        assert len(hashes) == 2
        assert hashes[0] == PROV._worktree_blob("pyproject.toml")

    def test_an_empty_batch_is_not_a_git_call(self) -> None:
        assert PROV._hash_many([]) == []


class TestTheResumeOwnsAProvenanceWindow:
    """A closed window is a deliberate no-op for --verify, which is how the standing
    portfolio ladder publishes with no provenance at all.

    A resume runs AFTER the original sweep closed, so routing it through the
    guarded driver was not enough: the re-measured row would be committed and
    pushed with the only window describing it having ended before it was
    produced.
    """

    @staticmethod
    def _resume() -> str:
        return (
            Path(PROV.REPO_ROOT) / "examples" / "research" / "resume_lfm25_study_a.sh"
        ).read_text()

    def test_it_opens_and_closes_its_own_window(self) -> None:
        resume = self._resume()

        assert "write_provenance.py --start --studies a" in resume
        assert "write_provenance.py --finish --partial" in resume

    def test_it_requires_the_window_to_be_open(self) -> None:
        assert "export LADDER_PROVENANCE_REQUIRED=1" in self._resume()

    def test_it_commits_the_window_at_both_ends(self) -> None:
        """An open window that dies with the worktree describes nothing."""
        resume = self._resume()

        assert resume.count("commit_only ") == 2
        assert resume.count('"$PROVENANCE_JSON"') >= 2

    def test_it_commits_only_the_paths_it_names(self) -> None:
        """A bare `git commit` takes the whole INDEX.

        Anything the operator had staged before starting -- unrelated WIP, a
        secret in a scratch file -- would ride along inside a provenance commit
        and then be pushed by the child driver. Every other driver in this study
        already uses `--only`; this script was the one site that did not.
        """
        resume = self._resume()

        assert "git commit -q --only" in resume
        commits = [
            line
            for line in resume.splitlines()
            if "git commit" in line and not line.lstrip().startswith("#")
        ]
        assert commits, "no commit at all"
        assert all("--only" in line for line in commits), commits

    def test_a_window_that_cannot_be_committed_fails_the_run(self) -> None:
        """Warning and exiting 0 tells automation a resume it cannot evidence succeeded."""
        resume = self._resume()

        tail = resume[resume.index("close the study-A resume") :]
        assert "exit 1" in tail
        assert "FATAL" in tail

    def test_it_rerenders_the_combined_write_up(self) -> None:
        """`run_ladder.sh` regenerates only the tuned study's own report.

        The combined write-up states on its face that it is generated from BOTH
        rows files, and its noise-floor table subtracts across them -- so a
        changed study-A cell can move it. Left unrendered, the resume publishes a
        document contradicting the row it just committed.
        """
        resume = self._resume()

        assert "lfm25_report.py" in resume
        assert '"$REPORT_MD"' in resume
        # After --finish, matching run_lfm25.sh: the report quotes the window.
        assert resume.index("--finish --partial") < resume.index("lfm25_report.py")

    def test_the_partial_note_does_not_assert_an_abort(self, tmp_path: Path) -> None:
        """A resume is a deliberate, successful run; only its SCOPE is partial."""
        path = _sidecar(tmp_path)

        PROV.finish(path, partial=True)

        note = json.loads(path.read_text())["partial_note"]
        assert "does NOT cover every row" in note
        assert "ABORTED" not in note


class TestTheProbeGuardsItsOwnOutput:
    """`run_lfm25.sh` refuses to start over a dirty load probe.

    But this module's docstring AND the generated write-up's "Reproduce" block
    both advertise running `lfm25_load_probe.py` STANDALONE, and that path
    reached an unconditional `write_text`. So the guard protected the artifact
    only from the caller that already had one -- the sixth time on this branch a
    fix landed on one of two sites that write the same file.
    """

    @staticmethod
    def _probe() -> ModuleType:
        name = "example_lfm25_load_probe"
        path = ROOT / "examples" / "research" / "lfm25_load_probe.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def test_it_refuses_when_the_artifact_holds_uncommitted_work(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        probe = self._probe()
        stranded = tmp_path / "probe.json"
        stranded.write_text('{"measured": "and never committed"}')
        monkeypatch.setattr(probe, "OUTPUT_PATH", stranded)
        monkeypatch.delenv(probe.FORCE_ENV, raising=False)

        with pytest.raises(SystemExit, match="REFUSING to overwrite"):
            probe._refuse_to_overwrite_uncommitted()

    def test_the_force_flag_is_the_documented_escape(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        probe = self._probe()
        stranded = tmp_path / "probe.json"
        stranded.write_text("{}")
        monkeypatch.setattr(probe, "OUTPUT_PATH", stranded)
        monkeypatch.setenv(probe.FORCE_ENV, "1")

        probe._refuse_to_overwrite_uncommitted()

    def test_a_clean_tracked_output_does_not_block_an_ordinary_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control. A guard that refuses everything is not a guard.

        Same reason as the write-up's control for using a committed unrelated
        file: the real probe is legitimately dirty mid-refresh.
        """
        probe = self._probe()
        monkeypatch.setattr(probe, "OUTPUT_PATH", Path(PROV.REPO_ROOT) / "LICENSE")
        monkeypatch.delenv(probe.FORCE_ENV, raising=False)

        probe._refuse_to_overwrite_uncommitted()

    def test_it_runs_before_the_measurement_not_beside_the_write(self) -> None:
        """Refusing after the work is done trains operators to reach for --force."""
        source = (ROOT / "examples" / "research" / "lfm25_load_probe.py").read_text()
        body = source[source.index("def main()") :]

        assert body.index("_refuse_to_overwrite_uncommitted()") < body.index("_run_isolated")

    def test_it_reuses_the_hardened_check(self) -> None:
        """Two copies of a safety check are two things that drift apart."""
        source = (ROOT / "examples" / "research" / "lfm25_load_probe.py").read_text()

        assert "from write_provenance import _uncommitted" in source


class TestTheCombinedWriteUpGuardsItsOwnWriter:
    """Third site of one defect, and the resume is what made it reachable.

    `run_lfm25.sh` refuses to start over a dirty write-up, but `run_ladder.sh`
    guards only the per-study artifacts -- and both the documented standalone
    render and the study-A resume (which round 22 taught to call it) reach this
    writer with no protection. Guarding the CALLER protects the callers you
    remembered; guarding the WRITER protects the file.
    """

    @staticmethod
    def _report() -> ModuleType:
        name = "example_lfm25_report_guard"
        path = ROOT / "examples" / "research" / "lfm25_report.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def test_it_refuses_over_uncommitted_edits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = self._report()
        stranded = tmp_path / "encoders.md"
        stranded.write_text("a hand edit no commit holds")
        monkeypatch.setattr(report, "OUTPUT", stranded)
        monkeypatch.delenv(report.FORCE_ENV, raising=False)

        with pytest.raises(SystemExit, match="REFUSING to overwrite"):
            report._refuse_to_overwrite_uncommitted()

    def test_the_force_flag_is_the_documented_escape(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = self._report()
        stranded = tmp_path / "encoders.md"
        stranded.write_text("x")
        monkeypatch.setattr(report, "OUTPUT", stranded)
        monkeypatch.setenv(report.FORCE_ENV, "1")

        report._refuse_to_overwrite_uncommitted()

    def test_a_clean_tracked_output_renders_normally(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The control: a guard that refuses every ordinary render is not a guard.

        Pointed at a committed, unmodified tracked file rather than at the real
        write-up, because the real one is legitimately dirty whenever someone is
        mid-render -- which would make this control fail for a reason that has
        nothing to do with the guard being correct.
        """
        report = self._report()
        monkeypatch.setattr(report, "OUTPUT", Path(PROV.REPO_ROOT) / "LICENSE")
        monkeypatch.delenv(report.FORCE_ENV, raising=False)

        report._refuse_to_overwrite_uncommitted()

    def test_all_three_writers_share_one_force_flag(self) -> None:
        """Three different escape hatches would be three things to remember."""
        report = self._report()
        probe_source = (ROOT / "examples" / "research" / "lfm25_load_probe.py").read_text()

        assert report.FORCE_ENV == "LFM25_FORCE"
        assert 'FORCE_ENV = "LFM25_FORCE"' in probe_source
        assert "LFM25_FORCE" in (ROOT / "examples" / "research" / "run_lfm25.sh").read_text()


class TestTheResumePublishesWhatItCloses:
    """`run_ladder.sh` pushes the re-measured ROW from inside the resume.

    Without a matching push at the end, a successful resume left the remote
    holding that row beside an OPEN provenance window and a stale combined
    write-up -- the published state contradicting itself -- until someone noticed.
    """

    @staticmethod
    def _resume() -> str:
        return (
            Path(PROV.REPO_ROOT) / "examples" / "research" / "resume_lfm25_study_a.sh"
        ).read_text()

    def test_it_pushes_after_closing_the_window(self) -> None:
        resume = self._resume()

        assert "publish_branch" in resume
        assert resume.index('commit_only "results(lfm25): close') < resume.index("publish\n")

    def test_it_refuses_to_push_onto_the_default_branch(self) -> None:
        """`git push origin HEAD` follows whatever is checked out.

        The rule moved into publish_lib.sh in round 26, so this asserts the resume
        USES it rather than re-deriving the branch itself — a second copy here is
        exactly what the shared file exists to prevent.
        """
        resume = self._resume()
        lib = (Path(PROV.REPO_ROOT) / "examples" / "research" / "publish_lib.sh").read_text()

        assert "publish_lib.sh" in resume
        assert "symbolic-ref" not in resume
        assert "default=${default:-main}" in lib
        assert "NOT pushing: on" in lib


class TestTheResumeOrdersItsIrreversibleStepsLast:
    """Four P1s landed on this script across two rounds; ordering was the cause.

    `--start` REPLACES the shared sidecar -- the one describing every row already
    committed -- so doing it before the driver's dirty-artifact refusal meant a
    resume run in exactly the situation it exists for swapped real provenance for
    a no-op window, committed it and pushed it, having measured nothing.
    """

    @staticmethod
    def _resume() -> str:
        return (
            Path(PROV.REPO_ROOT) / "examples" / "research" / "resume_lfm25_study_a.sh"
        ).read_text()

    def test_the_preflight_runs_before_the_window_is_replaced(self) -> None:
        resume = self._resume()

        assert resume.index("LADDER_PREFLIGHT_ONLY=1") < resume.index("--start --studies a")

    def test_the_preflight_is_the_driver_s_own_check_not_a_copy(self) -> None:
        """A second implementation of one safety rule is the drift disease."""
        resume = self._resume()
        driver = (Path(PROV.REPO_ROOT) / "examples" / "research" / "run_ladder.sh").read_text()

        assert "LADDER_PREFLIGHT_ONLY" in driver
        assert "artifact_is_dirty" not in resume

    def test_the_report_is_committed_only_when_this_run_rendered_it(self) -> None:
        """The writer guard refuses a hand-edited write-up.

        Passing REPORT_MD to `git commit --only` regardless would commit that very
        hand edit under a message claiming it was re-rendered -- publishing the
        bytes the guard had just refused to touch.
        """
        resume = self._resume()

        assert 'COMMIT_PATHS=("$PROVENANCE_JSON")' in resume
        assert 'COMMIT_PATHS+=("$REPORT_MD")' in resume
        # ...and the append is on the SUCCESS branch of the render.
        render = resume.index("lfm25_report.py; then")
        assert render < resume.index('COMMIT_PATHS+=("$REPORT_MD")')

    def test_a_rejected_finish_stops_publication(self) -> None:
        """run_ladder.sh withholds its push when --verify fails.

        Swallowing --finish's status re-enabled at the end exactly what the child
        had deliberately refused.
        """
        resume = self._resume()

        assert "REJECTED this run" in resume
        assert '[ "$code" -eq 0 ] && code=1' in resume
        assert "NOT publishing: this resume exited" in resume

    def test_publication_is_conditional_but_committing_is_not(self) -> None:
        """Durability is never what gets withheld -- only publication."""
        resume = self._resume()

        publish_guard = resume.index('if [ "$code" -eq 0 ]; then\n  publish')
        assert resume.index('commit_only "results(lfm25): close') < publish_guard


class TestThePartialWindowWordingDoesNotGuessACause:
    """`--partial` has meant two things since the resume started using it.

    Read as "the sweep aborted", every report generated after a SUCCESSFUL
    one-cell resume announced that the last sweep had failed.
    """

    def test_it_does_not_assert_an_abort(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        name = "example_lfm25_report_partial"
        path = ROOT / "examples" / "research" / "lfm25_report.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        report = importlib.util.module_from_spec(spec)
        sys.modules[name] = report
        spec.loader.exec_module(report)

        sidecar = tmp_path / "prov.json"
        sidecar.write_text(json.dumps({"window_complete": False, "studies_measured": ["a"]}))
        monkeypatch.setattr(report, "PROVENANCE", sidecar)

        warning = "\n".join(report._cross_study_caveat())

        assert "did not cover every stored row" in warning
        assert "aborted before finishing" not in warning


class TestARefusedRenderStillKeepsItsProvenance:
    """The renderer's input refusal is a non-zero exit, and that must not lose the window.

    Round 24 implemented this check in the shell, where the refusal branch called
    `commit_provenance` explicitly. Round 25 moved the check into the renderer --
    so the property now depends on the driver treating ANY render failure as a
    close-and-commit, which is the branch that was already there. Asserted rather
    than assumed, because moving a guard silently changed which code path carries
    this guarantee.
    """

    @staticmethod
    def _driver() -> str:
        return (Path(PROV.REPO_ROOT) / "examples" / "research" / "run_lfm25.sh").read_text()

    def test_a_failed_render_commits_the_closed_window(self) -> None:
        driver = self._driver()

        block = driver[driver.index("uv run python examples/research/lfm25_report.py") :][:500]
        assert "commit_provenance 1" in block


class TestTheRendererRefusesUncommittedInputs:
    """The renderer owns this check because it is the only thing that knows every
    file it reads.

    `run_lfm25.sh` grew it inline in round 24, which left the study-A resume --
    whose preflight covers only the TUNED artifacts -- reaching the same renderer
    unguarded, while it also reads the base rows and the load probe. Two drivers,
    one guarded, is how the earlier findings in this series happened.
    """

    @staticmethod
    def _report() -> ModuleType:
        name = "example_lfm25_report_inputs"
        path = ROOT / "examples" / "research" / "lfm25_report.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def test_an_uncommitted_base_rows_file_stops_the_render(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The input the resume's own preflight cannot see."""
        report = self._report()
        stranded = tmp_path / "base_rows.jsonl"
        stranded.write_text("{}\n")
        # RENDER_INPUTS is the list the guard reads; patching a single constant
        # stopped reaching it once the inputs were declared in one place.
        monkeypatch.setattr(report, "RENDER_INPUTS", (stranded,))
        monkeypatch.delenv(report.FORCE_ENV, raising=False)

        with pytest.raises(SystemExit, match="REFUSING to render"):
            report._refuse_uncommitted_inputs()

    def test_an_uncommitted_load_probe_stops_the_render(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = self._report()
        stranded = tmp_path / "probe.json"
        stranded.write_text("{}")
        monkeypatch.setattr(report, "RENDER_INPUTS", (stranded,))
        monkeypatch.delenv(report.FORCE_ENV, raising=False)

        with pytest.raises(SystemExit, match="REFUSING to render"):
            report._refuse_uncommitted_inputs()

    def test_committed_inputs_render_normally(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The control, on committed tracked files rather than the live artifacts."""
        report = self._report()
        licence = Path(PROV.REPO_ROOT) / "LICENSE"
        monkeypatch.setattr(report, "RENDER_INPUTS", (licence,))
        monkeypatch.delenv(report.FORCE_ENV, raising=False)

        report._refuse_uncommitted_inputs()

    def test_the_driver_does_not_keep_a_second_copy(self) -> None:
        """Round 24 put this in the shell; a copy in both is the drift disease."""
        driver = (Path(PROV.REPO_ROOT) / "examples" / "research" / "run_lfm25.sh").read_text()

        assert "MOVED_INPUTS" not in driver
        assert "REFUSING to render" not in driver


class TestTheResumeOwnsPublication:
    """`run_ladder.sh` pushes each row the moment it is committed.

    So the wrapper's "publish only on success" guard covered only its OWN push: a
    later --finish rejection or render failure still left origin holding the new
    row beside an OPEN provenance window, with the closing evidence withheld.
    """

    @staticmethod
    def _resume() -> str:
        return (
            Path(PROV.REPO_ROOT) / "examples" / "research" / "resume_lfm25_study_a.sh"
        ).read_text()

    def test_the_child_push_is_suppressed(self) -> None:
        assert "LADDER_NO_PUSH=1" in self._resume()

    def test_the_driver_honours_the_suppression(self) -> None:
        driver = (Path(PROV.REPO_ROOT) / "examples" / "research" / "run_ladder.sh").read_text()

        assert "LADDER_NO_PUSH:-0" in driver
        # ...and it withholds only the PUSH; the commit above it still happens.
        block = driver[driver.index("LADDER_NO_PUSH:-0") :][:400]
        assert "return 0" in block

    def test_suppression_is_set_on_the_measuring_call(self) -> None:
        """Set after the preflight call would leave the real run publishing itself."""
        resume = self._resume()
        measuring = resume.index("LADDER_NO_PUSH=1")

        assert resume.index("LADDER_PREFLIGHT_ONLY=1") < measuring
        assert measuring < resume.index("code=$?")


class TestEveryReadInputIsGuarded:
    """The uncommitted-input guard was written against the files I had in mind.

    It checked the two row files and the load probe, and silently skipped the
    licence text and the prior ladder report -- both of which are QUOTED into the
    output, so a render could publish bytes absent from the commit carrying it.
    """

    @staticmethod
    def _report() -> ModuleType:
        name = "example_lfm25_report_inputs_gate"
        path = ROOT / "examples" / "research" / "lfm25_report.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def test_the_licence_and_prior_ladder_are_declared(self) -> None:
        report = self._report()
        declared = {Path(x).name for x in report.RENDER_INPUTS}

        assert "20260729_lfm25_license.txt" in declared
        assert "20260727_embedder_ladder.md" in declared

    def test_the_outputs_are_not_treated_as_inputs(self) -> None:
        """Guarding the file it is about to write would refuse every re-render."""
        report = self._report()
        declared = {Path(x).name for x in report.RENDER_INPUTS}

        assert Path(report.OUTPUT).name not in declared
        assert Path(report.PROVENANCE).name not in declared


class TestOnePublicationRule:
    """Four copies of "push, but never onto the default branch" existed at once.

    They had drifted in what they printed and, twice in this series, in whether
    they ran at all -- which is how rows reached origin while the provenance
    window describing them stayed on the local branch.
    """

    RESEARCH = Path(__file__).parents[2] / "examples" / "research"
    DRIVERS = ("run_ladder.sh", "run_lfm25.sh", "resume_lfm25_study_a.sh")

    def test_the_rule_lives_in_one_file(self) -> None:
        lib = (self.RESEARCH / "publish_lib.sh").read_text()

        assert "symbolic-ref" in lib
        assert "publish_branch()" in lib

    def test_no_driver_keeps_its_own_copy(self) -> None:
        for name in self.DRIVERS:
            source = (self.RESEARCH / name).read_text()
            assert "symbolic-ref" not in source, name
            assert "publish_lib.sh" in source, name

    def test_the_abort_path_publishes_its_closing_window(self) -> None:
        """Rows are pushed per model; the window closing them must follow."""
        driver = (self.RESEARCH / "run_lfm25.sh").read_text()
        block = driver[
            driver.index("commit_provenance() {") : driver.index("abort_with_provenance")
        ]

        assert "publish_branch" in block

    def test_publication_failure_is_never_fatal(self) -> None:
        """Durability is the commit; publication is not worth aborting a sweep for."""
        lib = (self.RESEARCH / "publish_lib.sh").read_text()

        assert "exit " not in lib
        assert "return 1" in lib
