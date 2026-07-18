"""Tests for the dep-free run-identity + persistence layer (Stream S1).

Targets 95-100% on the pure logic: ``RunContext``/``RunRecord``,
``compute_recipe_id`` (the identity split -- recipe fields hashed, provenance
excluded), ``RunStore`` (JSONL round-trip, last-wins-by-attempt_id, flock,
mkdir-parents, ``RunStoreError``), ``resolve_store``, the ``git_sha`` /
``dataset_fingerprint`` / ``_collect_seeds`` helpers, and ``capture_run``
(running->terminal line, the ``current_run`` contextvar set/reset, tracker
fan-out, failure path).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pydantic
import pytest

from langres.tracking import runs
from langres.tracking.runs import (
    RunContext,
    RunRecord,
    RunStore,
    RunStoreError,
    capture_run,
    compute_recipe_id,
    current_run,
    dataset_fingerprint,
    git_sha,
    resolve_store,
)


def _context(**overrides: Any) -> RunContext:
    """A minimal, fully-specified RunContext; override any field per test."""
    base: dict[str, Any] = {
        "experiment": "exp-a",
        "dataset_name": "febrl4",
        "seeds": {"split": 7},
    }
    base.update(overrides)
    return RunContext(**base)


class _SpyTracker:
    name = "spy"

    def __init__(self, *, url: str | None = None) -> None:
        self._url = url
        self.calls: list[tuple[str, Any]] = []

    def start_run(self, context: Any, *, run_name: str | None = None) -> None:
        self.calls.append(("start_run", run_name))

    def log_params(self, params: Any) -> None:
        self.calls.append(("log_params", dict(params)))

    def log_metrics(self, metrics: Any, *, step: int | None = None) -> None:
        self.calls.append(("log_metrics", (dict(metrics), step)))

    def log_artifact(self, key: str, value: str) -> None:
        self.calls.append(("log_artifact", (key, value)))

    def set_tags(self, tags: Any) -> None:
        self.calls.append(("set_tags", dict(tags)))

    def finish(self, *, status: str) -> None:
        self.calls.append(("finish", status))

    @property
    def run_url(self) -> str | None:
        return self._url

    @property
    def native(self) -> Any:
        return self


# ---------------------------------------------------------------------------
# RunContext / RunRecord models
# ---------------------------------------------------------------------------


class TestRunContext:
    def test_minimal_construction_and_defaults(self) -> None:
        ctx = _context()
        assert ctx.experiment == "exp-a"
        assert ctx.dataset_name == "febrl4"
        assert ctx.tags == {}
        assert ctx.git_dirty is False
        assert ctx.tracking_schema_version == 1

    def test_is_frozen(self) -> None:
        ctx = _context()
        with pytest.raises(pydantic.ValidationError):
            ctx.experiment = "mutated"  # type: ignore[misc]

    def test_langres_version_is_populated(self) -> None:
        # default_factory reads importlib.metadata -- present in this editable install.
        assert _context().langres_version is not None


class TestRunRecord:
    def test_round_trips_through_json(self) -> None:
        ctx = _context()
        record = RunRecord(
            attempt_id="abc-2026",
            recipe_id="abc",
            context=ctx,
            started_at="2026-07-08T00:00:00+00:00",
            status="completed",
            metrics={"pair_f1": 0.9},
        )
        reloaded = RunRecord.model_validate_json(record.model_dump_json())
        assert reloaded == record
        assert reloaded.context.experiment == "exp-a"
        assert reloaded.v == 1


# ---------------------------------------------------------------------------
# compute_recipe_id -- the identity split
# ---------------------------------------------------------------------------


class TestComputeRecipeId:
    def test_is_deterministic_16_hex(self) -> None:
        rid = compute_recipe_id(_context())
        assert rid == compute_recipe_id(_context())
        assert len(rid) == 16
        int(rid, 16)  # valid hex

    def test_excludes_git_dirty_and_git_sha(self) -> None:
        # HIGH-1 regression guard: a dirty tree / different commit must NOT
        # mint a new recipe_id.
        clean = _context(git_sha="a" * 40, git_dirty=False)
        dirty = _context(git_sha="b" * 40, git_dirty=True)
        assert compute_recipe_id(clean) == compute_recipe_id(dirty)

    def test_excludes_lockfile_version_and_timing_provenance(self) -> None:
        a = _context(lockfile_hash="h1", langres_version="0.1.0", python_version="3.12")
        b = _context(lockfile_hash="h2", langres_version="9.9.9", python_version="3.13")
        assert compute_recipe_id(a) == compute_recipe_id(b)

    def test_distinct_seeds_give_distinct_recipe_id(self) -> None:
        a = _context(seeds={"split": 1})
        b = _context(seeds={"split": 2})
        assert compute_recipe_id(a) != compute_recipe_id(b)

    def test_distinct_dataset_fingerprint_gives_distinct_recipe_id(self) -> None:
        a = _context(dataset_fingerprint="fp1")
        b = _context(dataset_fingerprint="fp2")
        assert compute_recipe_id(a) != compute_recipe_id(b)

    def test_distinct_config_gives_distinct_recipe_id(self) -> None:
        a = _context(llm_model="gpt-4o-mini", blocking_k=10)
        b = _context(llm_model="gpt-4o-mini", blocking_k=20)
        assert compute_recipe_id(a) != compute_recipe_id(b)

    def test_permuted_config_keys_give_same_recipe_id(self) -> None:
        # L2: the order-stable seeds map (see TestCollectSeeds) feeds the
        # recipe_id, so two key-permuted builds of the same logical experiment
        # content-address identically.
        cfg_a = {
            "alpha": {"type_name": "Judge", "seed": 4},
            "beta": {"type_name": "Judge", "seed": 5},
        }
        cfg_b = {
            "beta": {"type_name": "Judge", "seed": 5},
            "alpha": {"type_name": "Judge", "seed": 4},
        }
        a = _context(resolver_config=cfg_a, seeds=runs._collect_seeds(7, cfg_a))
        b = _context(resolver_config=cfg_b, seeds=runs._collect_seeds(7, cfg_b))
        assert compute_recipe_id(a) == compute_recipe_id(b)


# ---------------------------------------------------------------------------
# RunStore
# ---------------------------------------------------------------------------


def _record(attempt_id: str, *, status: str = "completed", **kw: Any) -> RunRecord:
    return RunRecord(
        attempt_id=attempt_id,
        recipe_id=attempt_id.split("-")[0],
        context=_context(),
        started_at="2026-07-08T00:00:00+00:00",
        status=status,  # type: ignore[arg-type]
        **kw,
    )


class TestRunStore:
    def test_append_read_round_trip(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path / "runs.jsonl")
        r1 = _record("r1-t")
        r2 = _record("r2-t")
        store.append(r1)
        store.append(r2)
        rows = store.read()
        assert [r.attempt_id for r in rows] == ["r1-t", "r2-t"]
        assert rows[0] == r1

    def test_last_wins_by_attempt_id(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path / "runs.jsonl")
        store.append(_record("same-id", status="running"))
        store.append(_record("same-id", status="completed"))
        rows = store.read()
        assert len(rows) == 1
        assert rows[0].status == "completed"

    def test_read_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert RunStore(tmp_path / "nope.jsonl").read() == []

    def test_append_creates_parent_dirs(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path / "deep" / "nested" / "runs.jsonl")
        store.append(_record("r1-t"))
        assert store.path.exists()

    def test_unwritable_path_raises_run_store_error(self, tmp_path: Path) -> None:
        # A path whose parent is a regular file cannot be mkdir'd -> RunStoreError.
        blocker = tmp_path / "afile"
        blocker.write_text("x")
        store = RunStore(blocker / "sub" / "runs.jsonl")
        with pytest.raises(RunStoreError):
            store.append(_record("r1-t"))

    def test_read_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "runs.jsonl"
        store = RunStore(path)
        store.append(_record("r1-t"))
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n")  # a stray blank line must be ignored, not parsed
        store.append(_record("r2-t"))
        assert [r.attempt_id for r in store.read()] == ["r1-t", "r2-t"]

    def test_each_line_is_independently_valid_json(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path / "runs.jsonl")
        store.append(_record("r1-t"))
        store.append(_record("r2-t"))
        lines = (tmp_path / "runs.jsonl").read_text().splitlines()
        assert len(lines) == 2
        for line in lines:
            assert json.loads(line)["v"] == 1

    def test_append_flushes_and_fsyncs_before_releasing_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # HIGH-1 regression: the bytes must reach the OS (flush + fsync) *under*
        # the exclusive lock, so a concurrent writer cannot interleave. Record
        # the syscall order and assert fsync lands before the LOCK_UN.
        events: list[str] = []

        class _RecordingFcntl:
            LOCK_EX = 2
            LOCK_UN = 8

            @staticmethod
            def flock(fileno: int, op: int) -> None:
                events.append(f"flock:{op}")

        real_fsync = runs.os.fsync

        def _spy_fsync(fd: int) -> None:
            events.append("fsync")
            real_fsync(fd)

        monkeypatch.setattr(runs, "fcntl", _RecordingFcntl)
        monkeypatch.setattr(runs.os, "fsync", _spy_fsync)

        store = RunStore(tmp_path / "runs.jsonl")
        store.append(_record("r1-t"))

        assert events == [
            f"flock:{_RecordingFcntl.LOCK_EX}",
            "fsync",
            f"flock:{_RecordingFcntl.LOCK_UN}",
        ]
        assert RunStore(tmp_path / "runs.jsonl").read()[0].attempt_id == "r1-t"

    def test_read_tolerates_torn_trailing_line(self, tmp_path: Path) -> None:
        # A concurrent writer mid-append leaves a partial JSON line with no
        # terminating newline; read() must skip it, not raise.
        path = tmp_path / "runs.jsonl"
        store = RunStore(path)
        store.append(_record("r1-t"))
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"attempt_id": "r2-t", "recipe')  # torn write, no newline
        rows = store.read()
        assert [r.attempt_id for r in rows] == ["r1-t"]

    def test_read_warns_on_malformed_complete_line(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A complete (newline-terminated) garbage line is genuine corruption:
        # skip it but surface a warning instead of silently swallowing it.
        path = tmp_path / "runs.jsonl"
        store = RunStore(path)
        store.append(_record("r1-t"))
        with path.open("a", encoding="utf-8") as fh:
            fh.write("this is not valid json\n")
        store.append(_record("r2-t"))
        with caplog.at_level(logging.WARNING):
            rows = store.read()
        assert [r.attempt_id for r in rows] == ["r1-t", "r2-t"]
        assert any("malformed run record" in r.message for r in caplog.records)

    def test_append_without_fcntl_warns_once_and_still_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Codex#2: on a platform without fcntl (e.g. Windows), append degrades
        # to a lock-free write and warns exactly once -- it never crashes.
        monkeypatch.setattr(runs, "fcntl", None)
        monkeypatch.setattr(runs, "_FLOCK_WARNED", False)
        store = RunStore(tmp_path / "runs.jsonl")
        with caplog.at_level(logging.WARNING):
            store.append(_record("r1-t"))
            store.append(_record("r2-t"))
        assert [r.attempt_id for r in store.read()] == ["r1-t", "r2-t"]
        flock_warnings = [r for r in caplog.records if "advisory file lock" in r.message]
        assert len(flock_warnings) == 1  # warned once, not once per append


def test_import_langres_does_not_eagerly_require_fcntl(tmp_path: Path) -> None:
    """Codex#2: ``import langres`` (which imports ``tracking.runs``) must not need fcntl.

    ``fcntl`` is Unix-only; a hard top-level import would break ``import langres``
    on Windows. Simulate fcntl being unimportable (``sys.modules['fcntl'] = None``
    makes ``import fcntl`` raise ``ImportError``) in a fresh subprocess and assert
    the package still imports and ``RunStore`` still round-trips.
    """
    script = (
        "import sys; sys.modules['fcntl'] = None; "
        "import langres; from langres.tracking import runs; "
        "assert runs.fcntl is None, runs.fcntl; "
        "from langres.tracking.runs import RunStore, RunContext, RunRecord; "
        "store = RunStore(sys.argv[1]); "
        "ctx = RunContext(experiment='e', dataset_name='d'); "
        "rec = RunRecord(attempt_id='a-t', recipe_id='a', context=ctx, "
        "started_at='2026-07-08T00:00:00+00:00', status='completed'); "
        "store.append(rec); "
        "assert [r.attempt_id for r in store.read()] == ['a-t']; "
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "runs.jsonl")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "OK" in result.stdout


class TestResolveStore:
    def test_none_returns_none(self) -> None:
        assert resolve_store(None) is None

    def test_str_becomes_store(self, tmp_path: Path) -> None:
        store = resolve_store(str(tmp_path / "runs.jsonl"))
        assert isinstance(store, RunStore)

    def test_path_becomes_store(self, tmp_path: Path) -> None:
        store = resolve_store(tmp_path / "runs.jsonl")
        assert isinstance(store, RunStore)
        assert store.path == tmp_path / "runs.jsonl"

    def test_store_passed_through(self, tmp_path: Path) -> None:
        original = RunStore(tmp_path / "runs.jsonl")
        assert resolve_store(original) is original


# ---------------------------------------------------------------------------
# git_sha
# ---------------------------------------------------------------------------


class TestGitSha:
    def test_returns_sha_and_dirty_flag_in_a_repo(self) -> None:
        sha, dirty = git_sha()
        # This worktree is a real git repo, so a sha is available.
        assert sha is not None
        assert len(sha) == 40
        assert isinstance(dirty, bool)

    def test_dirty_detection_explicitly_includes_untracked_files(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        commands: list[list[str]] = []

        def _record(command: list[str], **kwargs: Any) -> Any:
            del kwargs
            commands.append(command)
            stdout = "a" * 40 if command[1:3] == ["rev-parse", "HEAD"] else "?? new.txt\n"
            return types.SimpleNamespace(stdout=stdout, returncode=0)

        monkeypatch.setattr(runs.subprocess, "run", _record)

        assert git_sha() == ("a" * 40, True)
        assert [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
        ] in commands

    def test_missing_git_returns_none_and_warns_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def _no_git(*args: Any, **kwargs: Any) -> Any:
            raise FileNotFoundError("git not found")

        monkeypatch.setattr(runs.subprocess, "run", _no_git)
        monkeypatch.setattr(runs, "_GIT_SHA_WARNED", False)
        with caplog.at_level(logging.WARNING):
            assert git_sha() == (None, False)
            git_sha()  # second call: must NOT warn again
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1

    def test_non_repo_returns_none_sha_and_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # git present but cwd is not a repo: empty stdout -> sha None, warn once.
        def _empty(*args: Any, **kwargs: Any) -> Any:
            return types.SimpleNamespace(stdout="", returncode=128)

        monkeypatch.setattr(runs.subprocess, "run", _empty)
        monkeypatch.setattr(runs, "_GIT_SHA_WARNED", False)
        with caplog.at_level(logging.WARNING):
            sha, dirty = git_sha()
        assert sha is None
        assert dirty is False
        assert any(r.levelno == logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# dataset_fingerprint
# ---------------------------------------------------------------------------


class TestDatasetFingerprint:
    def test_is_deterministic(self) -> None:
        corpus = [{"id": "1", "name": "acme"}, {"id": "2", "name": "globex"}]
        gold = [("1", "2")]
        assert dataset_fingerprint(corpus, gold) == dataset_fingerprint(corpus, gold)

    def test_is_order_independent(self) -> None:
        a = [{"id": "1"}, {"id": "2"}]
        b = [{"id": "2"}, {"id": "1"}]
        assert dataset_fingerprint(a, []) == dataset_fingerprint(b, [])

    def test_mutation_changes_fingerprint(self) -> None:
        corpus = [{"id": "1", "name": "acme"}]
        mutated = [{"id": "1", "name": "ACME CORP"}]
        assert dataset_fingerprint(corpus, []) != dataset_fingerprint(mutated, [])

    def test_gold_participates_in_fingerprint(self) -> None:
        corpus = [{"id": "1"}]
        assert dataset_fingerprint(corpus, [("1", "2")]) != dataset_fingerprint(
            corpus, [("1", "3")]
        )

    def test_handles_pydantic_records(self) -> None:
        class Rec(pydantic.BaseModel):
            id: str
            name: str

        recs = [Rec(id="1", name="acme")]
        dicts = [{"id": "1", "name": "acme"}]
        assert dataset_fingerprint(recs, []) == dataset_fingerprint(dicts, [])

    def test_handles_set_gold_clusters(self) -> None:
        # gold as clusters (a set of frozensets) exercises the set/frozenset
        # canonicalization branch, order-independently.
        gold_a = {frozenset({"1", "2"}), frozenset({"3", "4"})}
        gold_b = {frozenset({"3", "4"}), frozenset({"1", "2"})}
        assert dataset_fingerprint([], gold_a) == dataset_fingerprint([], gold_b)

    def test_handles_mapping_gold(self) -> None:
        # a mapping is normalized to its sorted items, deterministically.
        gold = {"1": "cluster-a", "2": "cluster-b"}
        assert dataset_fingerprint([], gold) == dataset_fingerprint([], dict(gold))

    def test_handles_path_and_nested_model_values(self) -> None:
        class Rec(pydantic.BaseModel):
            id: str

        corpus = [{"path": Path("/data/x.csv"), "rec": Rec(id="1")}]
        # deterministic and does not raise on Path / nested-model values.
        assert dataset_fingerprint(corpus, []) == dataset_fingerprint(corpus, [])

    def test_uncanonicalizable_item_raises_type_error(self) -> None:
        class Opaque:
            pass

        with pytest.raises(TypeError, match="cannot canonicalize"):
            dataset_fingerprint([{"x": Opaque()}], [])


# ---------------------------------------------------------------------------
# _collect_seeds
# ---------------------------------------------------------------------------


class TestCollectSeeds:
    def test_none_config_degrades_to_split_seed(self) -> None:
        assert runs._collect_seeds(42, None) == {"split": 42}

    def test_scans_config_for_random_state_and_seed(self) -> None:
        config = {
            "type_name": "root",
            "blocker": {"type_name": "VectorBlocker", "random_state": 5},
            "module": {"type_name": "RandomForestMatcher", "seed": 9},
        }
        seeds = runs._collect_seeds(1, config)
        assert seeds["split"] == 1
        assert 5 in seeds.values()
        assert 9 in seeds.values()

    def test_ignores_bool_and_non_int_seed_values(self) -> None:
        config = {"seed": True, "random_state": "not-an-int"}
        assert runs._collect_seeds(3, config) == {"split": 3}

    def test_scans_list_nested_components(self) -> None:
        # a list of sub-configs (e.g. a CompositeBlocker's children) is walked.
        config = {
            "type_name": "Composite",
            "children": [
                {"type_name": "A", "seed": 11},
                {"type_name": "B", "random_state": 22},
            ],
        }
        seeds = runs._collect_seeds(1, config)
        assert 11 in seeds.values()
        assert 22 in seeds.values()

    def test_duplicate_component_seeds_get_unique_labels(self) -> None:
        # three same-typed components -> three distinct labels (the suffix must
        # keep climbing past the first collision), all values retained.
        config = {
            "a": {"type_name": "Judge", "seed": 4},
            "b": {"type_name": "Judge", "seed": 5},
            "c": {"type_name": "Judge", "seed": 6},
        }
        seeds = runs._collect_seeds(0, config)
        judge_seeds = {label: v for label, v in seeds.items() if label.startswith("Judge.")}
        assert sorted(judge_seeds.values()) == [4, 5, 6]
        assert len(judge_seeds) == 3

    def test_seed_labels_are_order_stable(self) -> None:
        # L2 regression: two semantically identical configs built with permuted
        # dict-key order must yield the SAME seeds map. Disambiguation suffixes
        # are assigned in canonical (sorted) traversal order, not raw
        # dict-insertion order, so the derived recipe_id can't drift.
        config_a = {
            "alpha": {"type_name": "Judge", "seed": 4},
            "beta": {"type_name": "Judge", "seed": 5},
        }
        config_b = {
            "beta": {"type_name": "Judge", "seed": 5},
            "alpha": {"type_name": "Judge", "seed": 4},
        }
        assert runs._collect_seeds(7, config_a) == runs._collect_seeds(7, config_b)


# ---------------------------------------------------------------------------
# capture_run
# ---------------------------------------------------------------------------


class TestCaptureRun:
    def test_store_none_writes_no_files(self, tmp_path: Path) -> None:
        # store=None -> no persistence at all (the headline invariant).
        with capture_run(_context()) as handle:
            handle.log_metrics({"m": 1.0})
        assert list(tmp_path.iterdir()) == []

    def test_sets_and_resets_contextvar(self) -> None:
        assert current_run.get() is None
        with capture_run(_context()) as handle:
            assert current_run.get() == handle.attempt_id
        assert current_run.get() is None

    def test_nested_capture_restores_parent(self) -> None:
        with capture_run(_context(experiment="outer")) as outer:
            assert current_run.get() == outer.attempt_id
            with capture_run(_context(experiment="inner")) as inner:
                assert current_run.get() == inner.attempt_id
            assert current_run.get() == outer.attempt_id

    def test_writes_running_then_terminal_line(self, tmp_path: Path) -> None:
        path = tmp_path / "runs.jsonl"
        with capture_run(_context(), store=RunStore(path)) as handle:
            attempt = handle.attempt_id
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        assert len(lines) == 2
        assert lines[0]["status"] == "running"
        assert lines[1]["status"] == "completed"
        assert lines[0]["attempt_id"] == lines[1]["attempt_id"] == attempt
        # last-wins collapses the pair to one terminal record.
        rows = RunStore(path).read()
        assert len(rows) == 1
        assert rows[0].status == "completed"

    def test_records_metrics_artifacts_and_status_from_handle(self, tmp_path: Path) -> None:
        path = tmp_path / "runs.jsonl"
        with capture_run(_context(), store=RunStore(path)) as handle:
            handle.log_metrics(
                {"pair_f1": 0.9},
                metric_definition="pair_f1@best_threshold",
                per_seed_metrics=[{"seed": 0, "pair_f1": 0.9}],
                headline_metric=0.9,
            )
            handle.log_artifact("report", "runs/report.md")
            handle.record_cost(0.42, budget_exceeded=True)
        record = RunStore(path).read()[0]
        assert record.metrics == {"pair_f1": 0.9}
        assert record.metric_definition == "pair_f1@best_threshold"
        assert record.per_seed_metrics == [{"seed": 0, "pair_f1": 0.9}]
        assert record.headline_metric == 0.9
        assert record.artifacts["report"] == "runs/report.md"
        assert record.spend_usd == 0.42
        assert record.budget_exceeded is True
        assert record.finished_at is not None
        assert record.duration_seconds is not None
        assert record.recipe_id == compute_recipe_id(_context())
        assert record.trace_id == record.attempt_id

    def test_set_status_overrides_terminal_status(self, tmp_path: Path) -> None:
        path = tmp_path / "runs.jsonl"
        with capture_run(_context(), store=RunStore(path)) as handle:
            handle.set_status("budget_exceeded")
        assert RunStore(path).read()[0].status == "budget_exceeded"

    def test_failure_records_failed_status_and_reraises(self, tmp_path: Path) -> None:
        path = tmp_path / "runs.jsonl"
        with pytest.raises(ValueError, match="boom"):
            with capture_run(_context(), store=RunStore(path)):
                raise ValueError("boom")
        record = RunStore(path).read()[0]
        assert record.status == "failed"
        assert record.error_type == "ValueError"
        assert record.error_message is not None and "boom" in record.error_message
        # contextvar still reset after an exception.
        assert current_run.get() is None

    def test_status_override_survives_a_reraised_budget_error(self, tmp_path: Path) -> None:
        path = tmp_path / "runs.jsonl"
        with pytest.raises(RuntimeError, match="budget"):
            with capture_run(_context(), store=RunStore(path)) as handle:
                handle.set_status("budget_exceeded")
                handle.record_cost(0.2, budget_exceeded=True)
                raise RuntimeError("budget")

        record = RunStore(path).read()[0]
        assert record.status == "budget_exceeded"
        assert record.budget_exceeded is True
        assert record.spend_usd == 0.2

    def test_failure_redacts_credentials_from_persisted_error(self, tmp_path: Path) -> None:
        path = tmp_path / "runs.jsonl"
        with pytest.raises(RuntimeError, match="super-secret"):
            with capture_run(_context(), store=RunStore(path)):
                raise RuntimeError(
                    "provider failed api_key=super-secret token: abc123 "
                    "Authorization: Bearer bearer-secret"
                )

        record = RunStore(path).read()[0]
        assert record.error_message is not None
        assert "super-secret" not in record.error_message
        assert "abc123" not in record.error_message
        assert "bearer-secret" not in record.error_message
        assert "Bearer" not in record.error_message
        assert record.error_message.count("<redacted>") == 3

    def test_tracker_start_run_failure_writes_failed_record_and_resets_contextvar(
        self, tmp_path: Path
    ) -> None:
        # HIGH-2 regression: start_run runs inside the protected block, so a
        # failure there still writes a terminal "failed" record for the attempt
        # and restores the contextvar -- the "running" line never dangles.
        class _BoomTracker(_SpyTracker):
            def start_run(self, context: Any, *, run_name: str | None = None) -> None:
                raise RuntimeError("start boom")

        path = tmp_path / "runs.jsonl"
        assert current_run.get() is None
        with pytest.raises(RuntimeError, match="start boom"):
            with capture_run(_context(), store=RunStore(path), tracker=_BoomTracker()):
                pass  # pragma: no cover - start_run raises before the body runs
        record = RunStore(path).read()[0]
        assert record.status == "failed"
        assert record.error_type == "RuntimeError"
        assert record.error_message is not None and "start boom" in record.error_message
        assert current_run.get() is None

    def test_drives_the_tracker(self, tmp_path: Path) -> None:
        spy = _SpyTracker(url="http://mlflow/run/1")
        with capture_run(_context(), tracker=spy) as handle:
            handle.log_metrics({"m": 1.0}, step=2)
            handle.log_artifact("k", "v")
        kinds = [c[0] for c in spy.calls]
        assert kinds[0] == "start_run"
        assert kinds[-1] == "finish"
        assert ("log_metrics", ({"m": 1.0}, 2)) in spy.calls
        assert ("log_artifact", ("k", "v")) in spy.calls
        assert spy.calls[-1] == ("finish", "completed")

    def test_run_url_threaded_into_artifacts(self, tmp_path: Path) -> None:
        path = tmp_path / "runs.jsonl"
        spy = _SpyTracker(url="http://mlflow/run/1")
        with capture_run(_context(), store=RunStore(path), tracker=spy):
            pass
        record = RunStore(path).read()[0]
        assert "http://mlflow/run/1" in record.artifacts.values()
