"""Run identity, provenance capture, and JSONL persistence -- the tracking core.

langres benchmark runs are otherwise *ephemeral*: rich Pydantic results are
dumped or rendered, then lost -- no run id, no config/dataset/split snapshot, no
persistence, no cross-run comparison. This module adds the missing spine while
staying **dependency-free** (stdlib + pydantic only): two frozen models
(:class:`RunContext` -- the recipe; :class:`RunRecord` -- recipe + outcomes),
their content-addressed identity (:func:`compute_recipe_id`), a JSONL
:class:`RunStore`, and the :func:`capture_run` context manager that ties them
together. Result models (``MethodResult`` etc.) are stored as an **opaque dict**
so ``import langres`` never pulls ``ranx``/``benchmark`` in.

Identity split (the subtle part). LLM runs are nondeterministic, so idempotency
is *same recipe -> same* :func:`compute_recipe_id`, **not** same metrics:

* ``recipe_id`` = ``sha256(canonical_json(<recipe fields>))[:16]`` -- a dedup key
  over the *logical experiment* (config + data + seeds). It **excludes** all
  code/env provenance (``git_sha``, ``git_dirty``, ``lockfile_hash``,
  ``langres_version``, timing), so a dirty tree or a ``uv.lock`` bump does NOT
  mint a new id. That provenance is recorded for explaining a metrics move --
  just not part of dedup identity.
* ``attempt_id`` = ``f"{recipe_id}-{started_at}"`` = the record PK. The
  ``running`` line and the terminal line of one attempt share it, so the reader
  does **last-wins-by-attempt_id**. "Already paid this config?" =
  "any record with this ``recipe_id`` and ``status == 'completed'``?".
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
import os
import re
import subprocess
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from decimal import Decimal
from numbers import Real
from pathlib import Path
from typing import Any, Literal, Never, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from langres.tracking.trackers import ExperimentTracker, NoOpTracker

# ``fcntl`` is Unix-only. ``runs`` sits on the bare ``import langres`` path
# (``judgement_log`` imports ``current_run`` from it), so a hard top-level
# import would break importing the package on Windows. Load it lazily: on a
# platform without ``fcntl`` the store degrades to a lock-free append with a
# one-time warning (see :meth:`RunStore.append`) rather than crashing the import.
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows/non-Unix platforms have no fcntl
    fcntl = None  # type: ignore[assignment]

__all__ = [
    "RunContext",
    "RunRecord",
    "RunStore",
    "RunStoreError",
    "capture_run",
    "compute_recipe_id",
    "current_run",
    "dataset_fingerprint",
    "git_sha",
    "mint_attempt_id",
    "resolve_store",
]

logger = logging.getLogger(__name__)

#: The active attempt id, so components deep in a resolve (the LLM judge, the
#: judgement log) can correlate their rows to the enclosing run without the id
#: being threaded through every call. ``None`` when no ``capture_run`` is open.
current_run: ContextVar[str | None] = ContextVar("langres_current_run", default=None)

#: Terminal states a run can finish in. ``"running"`` is written at *start* so a
#: crashed/torn-down run leaves a visible lone ``running`` line.
RunStatus = Literal["running", "completed", "failed", "budget_exceeded"]

_RECIPE_ID_LENGTH = 16
_MAX_ERROR_MESSAGE_LEN = 2000
_GIT_TIMEOUT_SECONDS = 5.0
_SEED_KEYS = ("random_state", "seed")

#: The RunContext fields that DEFINE a run (the hash domain of
#: :func:`compute_recipe_id`). Everything else on the context is provenance,
#: recorded but deliberately excluded from identity (see the module docstring).
_RECIPE_FIELDS = (
    "experiment",
    "resolver_config",
    "llm_model",
    "cascade_band",
    "blocking_k",
    "budget_usd",
    "method",
    "dataset_name",
    "dataset_fingerprint",
    "split_id",
    "seeds",
)

#: One-shot guard so a git-less environment warns exactly once per process.
_GIT_SHA_WARNED = False

#: One-shot guard so a flock-less (non-Unix) environment warns exactly once.
_FLOCK_WARNED = False


def _detect_langres_version() -> str | None:
    """Best-effort installed langres version for run provenance (``None`` if absent)."""
    try:
        return importlib.metadata.version("langres")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - always installed in-repo
        return None


class RunContext(BaseModel):
    """The recipe: everything that determines a run, plus code/env provenance.

    Frozen. The *config/data/seed* fields feed :func:`compute_recipe_id`; the
    code/env fields are provenance only (see :data:`_RECIPE_FIELDS`).
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False, validate_default=True)

    # -- Identity / organization (NOT hashed) --
    experiment: str
    group: str | None = None
    parent_run_id: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)

    # -- Code/env provenance (recorded, NOT hashed) --
    git_sha: str | None = None
    git_dirty: bool = False
    lockfile_hash: str | None = None
    langres_version: str | None = Field(default_factory=_detect_langres_version)
    tracking_schema_version: int = 1
    python_version: str | None = None
    platform: str | None = None
    reproduction_adapter_version: str | None = None

    # -- Config (hashed) --
    resolver_config: dict[str, Any] | None = None
    llm_model: str | None = None
    cascade_band: tuple[float, float] | None = None
    blocking_k: int | None = None
    budget_usd: float | None = None
    method: str | None = None

    # -- Data (hashed) --
    dataset_name: str
    dataset_fingerprint: str | None = None
    split_id: str | None = None

    # -- Seeds (hashed): named union of every source; the split seed lives in
    #    ``seeds["split"]`` (no duplicate scalar). --
    seeds: dict[str, int] = Field(default_factory=dict)

    @field_validator("tags", mode="after")
    @classmethod
    def _freeze_tags(cls, value: dict[str, str]) -> dict[str, str]:
        snapshot = _snapshot_mapping(value)
        assert snapshot is not None
        return snapshot

    @field_validator("resolver_config", mode="after")
    @classmethod
    def _freeze_resolver_config(
        cls,
        value: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return _snapshot_mapping(value)

    @field_validator("seeds", mode="after")
    @classmethod
    def _freeze_seeds(cls, value: dict[str, int]) -> dict[str, int]:
        snapshot = _snapshot_mapping(value)
        assert snapshot is not None
        return snapshot


class _FrozenSnapshotDict(dict[str, Any]):
    """JSON-serializable immutable mapping used for persisted run snapshots."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("run snapshots are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    setdefault = _immutable
    update = _immutable

    def popitem(self) -> tuple[str, Any]:
        self._raise_immutable()

    def __ior__(self, value: Any) -> Self:  # type: ignore[override,misc]
        del value
        self._raise_immutable()

    @staticmethod
    def _raise_immutable() -> Never:
        raise TypeError("run snapshots are immutable")


def _deep_snapshot(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenSnapshotDict({str(key): _deep_snapshot(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_snapshot(item) for item in value)
    if isinstance(value, (set, frozenset)):
        raise ValueError("run snapshots must contain JSON values; sets are not supported")
    if isinstance(value, Decimal) and not value.is_finite():
        raise ValueError("run snapshots must contain finite numeric values")
    if isinstance(value, Real) and not math.isfinite(value):
        raise ValueError("run snapshots must contain finite numeric values")
    return value


def _snapshot_mapping(value: Mapping[str, Any] | None) -> _FrozenSnapshotDict | None:
    if value is None:
        return None
    snapshot = _deep_snapshot(value)
    assert isinstance(snapshot, _FrozenSnapshotDict)
    return snapshot


def _snapshot_measurements(
    values: Iterable[Mapping[str, Any]] | None,
) -> tuple[_FrozenSnapshotDict, ...] | None:
    if values is None:
        return None
    snapshots: list[_FrozenSnapshotDict] = []
    for value in values:
        snapshot = _snapshot_mapping(value)
        assert snapshot is not None
        snapshots.append(snapshot)
    return tuple(snapshots)


class RunRecord(BaseModel):
    """A :class:`RunContext` plus outcomes -- one JSONL line, ``"v": 1`` idiom.

    Frozen. ``metrics`` is an **opaque dict** (a ``MethodResult``/``JudgePairEval``
    ``.model_dump()``) so this module never depends on the result models.
    """

    model_config = ConfigDict(frozen=True, validate_default=True)

    attempt_id: str
    recipe_id: str
    context: RunContext
    v: int = 1

    # -- Experiment protocol identity (optional for v1/back-compat records) --
    evaluation_id: str | None = None
    cache_id: str | None = None
    protocol: dict[str, Any] | None = None
    measurements: tuple[dict[str, Any], ...] | None = None
    experiment_facts: dict[str, Any] | None = None
    partial_judgements: tuple[dict[str, Any], ...] | None = None

    # -- Timing (never hashed) --
    started_at: str
    finished_at: str | None = None
    duration_seconds: float | None = None

    # -- Metrics (opaque; self-labelled so comparisons can't silently mix) --
    metrics: dict[str, Any] | None = None
    metric_definition: str | None = None
    per_seed_metrics: list[dict[str, Any]] | None = None
    headline_metric: float | None = None

    # -- Cost (langres-native) --
    spend_usd: float | None = None
    budget_exceeded: bool = False

    # -- Artifacts --
    judgement_log_path: str | None = None
    trace_id: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)

    # -- Status / failure --
    status: RunStatus
    error_type: str | None = None
    error_message: str | None = None

    @field_validator("protocol", mode="after")
    @classmethod
    def _freeze_protocol(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        return _snapshot_mapping(value)

    @field_validator("measurements", mode="after")
    @classmethod
    def _freeze_measurements(
        cls,
        value: tuple[dict[str, Any], ...] | None,
    ) -> tuple[dict[str, Any], ...] | None:
        return _snapshot_measurements(value)

    @field_validator("experiment_facts", mode="after")
    @classmethod
    def _freeze_experiment_facts(
        cls,
        value: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return _snapshot_mapping(value)

    @field_validator("partial_judgements", mode="after")
    @classmethod
    def _freeze_partial_judgements(
        cls,
        value: tuple[dict[str, Any], ...] | None,
    ) -> tuple[dict[str, Any], ...] | None:
        return _snapshot_measurements(value)

    @field_validator("artifacts", mode="after")
    @classmethod
    def _freeze_artifacts(cls, value: dict[str, str]) -> dict[str, str]:
        snapshot = _snapshot_mapping(value)
        assert snapshot is not None
        return snapshot


# ---------------------------------------------------------------------------
# Identity + fingerprint helpers
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """Canonicalize the non-JSON-native types a corpus/gold may contain."""
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, Decimal):
        if not obj.is_finite():
            raise ValueError("fingerprint values must be finite")
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot canonicalize {type(obj).__name__} for a fingerprint")


def _canonical_json(obj: Any) -> str:
    """Order-stable JSON: sorted keys, no whitespace -- the hashing normal form."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
        allow_nan=False,
    )


def compute_recipe_id(context: RunContext) -> str:
    """Content-address the run's recipe fields (see :data:`_RECIPE_FIELDS`).

    Excludes all code/env provenance and timing, so a dirty tree or dependency
    bump keeps the same id -- ``recipe_id`` is a dedup key over the *logical*
    experiment, stable across code churn.
    """
    payload = {field: getattr(context, field) for field in _RECIPE_FIELDS}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:_RECIPE_ID_LENGTH]


def _to_jsonable(obj: Any) -> Any:
    """Pydantic model -> JSON dict; anything else passes through to ``json.dumps``."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    return obj


def _iter_items(data: Iterable[Any]) -> list[Any]:
    """Normalize a corpus/gold to a list of items (a mapping -> its ``items()``)."""
    if isinstance(data, Mapping):
        return list(data.items())
    return list(data)


def dataset_fingerprint(corpus: Iterable[Any], gold: Iterable[Any]) -> str:
    """sha256 over the ALREADY-LOADED ``corpus`` + ``gold`` (order-independent).

    Each item is canonicalized then sorted, so two loads in a different row
    order fingerprint identically while any content mutation changes the hash.
    Do NOT re-``load()`` here -- pass the in-memory objects the caller already
    holds.
    """
    digest = hashlib.sha256()
    for label, data in (("corpus", corpus), ("gold", gold)):
        digest.update(label.encode("utf-8"))
        for line in sorted(_canonical_json(_to_jsonable(item)) for item in _iter_items(data)):
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def _unique_label(label: str, seeds: Mapping[str, int]) -> str:
    """Disambiguate a seed label that already exists (two same-typed components)."""
    if label not in seeds:
        return label
    suffix = 2
    while f"{label}#{suffix}" in seeds:
        suffix += 1
    return f"{label}#{suffix}"


def _scan_for_seeds(node: Any, seeds: dict[str, int], *, path: str) -> None:
    """Recursively pull ``random_state``/``seed`` ints out of a config tree."""
    if isinstance(node, Mapping):
        type_name = node.get("type_name")
        prefix = type_name if isinstance(type_name, str) else (path or "config")
        for key in _SEED_KEYS:
            value = node.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                seeds[_unique_label(f"{prefix}.{key}", seeds)] = value
        # Traverse mapping children in canonical (sorted-key) order so the
        # disambiguation suffixes ``_unique_label`` hands out -- and therefore
        # the ``seeds`` map and the ``recipe_id`` that hashes it -- do not depend
        # on the caller's raw dict-insertion order. Lists keep their index order
        # (a list is semantically ordered data).
        for child_key in sorted(node, key=str):
            _scan_for_seeds(
                node[child_key], seeds, path=f"{path}.{child_key}" if path else str(child_key)
            )
    elif isinstance(node, (list, tuple)):
        for index, item in enumerate(node):
            _scan_for_seeds(item, seeds, path=f"{path}[{index}]")


def _collect_seeds(split_seed: int, resolver_config: Mapping[str, Any] | None) -> dict[str, int]:
    """``{"split": split_seed}`` unioned with every seed found in the config tree.

    Degrades to just the split seed when ``resolver_config`` is ``None``.
    """
    seeds: dict[str, int] = {"split": int(split_seed)}
    if resolver_config is not None:
        _scan_for_seeds(resolver_config, seeds, path="")
    return seeds


def git_sha() -> tuple[str | None, bool]:
    """``(HEAD sha, dirty)`` for run provenance -- ``(None, False)`` without git.

    ``check=False`` + a short timeout + swallowed ``FileNotFoundError`` so a
    git-less / non-repo environment never raises; it warns once per process.
    """
    global _GIT_SHA_WARNED
    cwd = str(Path(__file__).resolve().parent)
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
            cwd=cwd,
        )
        porcelain = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
            cwd=cwd,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        if not _GIT_SHA_WARNED:
            logger.warning("git unavailable; recording git_sha=None for run provenance")
            _GIT_SHA_WARNED = True
        return (None, False)
    sha = head.stdout.strip() or None
    dirty = bool(porcelain.stdout.strip())
    if sha is None and not _GIT_SHA_WARNED:
        logger.warning("git could not resolve HEAD (not a repo?); recording git_sha=None")
        _GIT_SHA_WARNED = True
    return (sha, dirty)


def mint_attempt_id(recipe_id: str) -> str:
    """Mint a unique concrete execution id before cache identity is computed."""
    started_at = datetime.now(UTC).isoformat()
    return f"{recipe_id}-{started_at}-{uuid.uuid4().hex[:12]}"


_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer|cookie|password|secret|token)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_AUTHORIZATION_HEADER = re.compile(
    r"(?i)(authorization\s*[:=]\s*)(?:(?:bearer|basic)\s+)?([^\s,;]+)"
)


def _sanitize_error_message(message: str) -> str:
    """Bound persisted errors and redact common credential assignments."""
    sanitized = _AUTHORIZATION_HEADER.sub(r"\1<redacted>", message)
    sanitized = _SECRET_ASSIGNMENT.sub(r"\1\2<redacted>", sanitized)
    return sanitized[:_MAX_ERROR_MESSAGE_LEN]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class RunStoreError(RuntimeError):
    """A :class:`RunStore` could not persist a record (unwritable path, etc.)."""


def _warn_flock_unavailable() -> None:
    """Warn once that advisory file locking is unavailable on this platform."""
    global _FLOCK_WARNED
    if not _FLOCK_WARNED:
        logger.warning(
            "fcntl is unavailable on this platform; RunStore.append is writing without an "
            "advisory file lock -- concurrent writers to the same runs file may interleave."
        )
        _FLOCK_WARNED = True


class RunStore:
    """Append-only JSONL store of :class:`RunRecord` lines.

    ``append`` takes an exclusive ``fcntl.flock`` (cross-process safety when
    several agents write the same file) and creates parent dirs on demand
    (``JudgementLog``'s precedent). ``read`` collapses the ``running`` +
    terminal lines of each attempt via **last-wins-by-attempt_id** and tolerates
    a torn trailing line from a concurrent writer. On a platform without
    ``fcntl`` (e.g. Windows) ``append`` degrades to a lock-free write.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: RunRecord) -> None:
        """Append one record durably; wrap any OS failure as :class:`RunStoreError`.

        The bytes are ``flush``ed and ``fsync``ed to the OS *before* the advisory
        lock is released, so the whole line lands under the lock and concurrent
        writers cannot interleave. (A bare ``TextIOWrapper.write`` only fills a
        Python buffer -- the real ``write(2)`` would otherwise happen at
        ``close()``, after the lock was already dropped.) Where ``fcntl`` is
        absent the lock is skipped with a one-time warning.
        """
        line = record.model_dump_json() + "\n"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                use_lock = fcntl is not None
                if use_lock:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                else:
                    _warn_flock_unavailable()
                try:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    if use_lock:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            raise RunStoreError(
                f"could not append a run record to {self.path}: {exc}. Check the path is "
                "writable -- its parent must be a directory (not a file), and the process "
                "must have permission to create/append there."
            ) from exc

    def read(self) -> list[RunRecord]:
        """Every attempt's latest record, in first-seen order (last-wins).

        Tolerant of a concurrent writer: a torn/partial *trailing* line (the file
        being appended right now) is skipped quietly, while a genuinely malformed
        *complete* line is skipped with a ``logging`` warning rather than
        aborting the whole read.
        """
        if not self.path.exists():
            return []
        by_attempt: dict[str, RunRecord] = {}
        with self.path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
        last_index = len(lines) - 1
        for index, raw in enumerate(lines):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = RunRecord.model_validate_json(stripped)
            except ValueError:
                # A trailing line with no terminating newline is almost certainly
                # a torn write from a racing appender -- skip it silently. A
                # complete (newline-terminated) or mid-file bad line is real
                # corruption, worth a warning but not worth aborting the read.
                if index == last_index and not raw.endswith("\n"):
                    logger.debug("skipping a torn trailing line in %s", self.path)
                else:
                    logger.warning("skipping a malformed run record line in %s", self.path)
                continue
            by_attempt[record.attempt_id] = record
        return list(by_attempt.values())


def resolve_store(spec: str | Path | RunStore | None) -> RunStore | None:
    """``None`` -> ``None`` (no persistence); ``str|Path`` -> ``RunStore``; store -> as-is.

    ``None`` means *no files at all* -- the "``store=None`` -> nothing written"
    invariant. Symmetric with :func:`resolve_tracker`.
    """
    if spec is None:
        return None
    if isinstance(spec, RunStore):
        return spec
    return RunStore(spec)


# ---------------------------------------------------------------------------
# capture_run
# ---------------------------------------------------------------------------


class _RunHandle:
    """Yielded inside ``capture_run``: accumulates outcomes for the terminal record.

    ``log_metrics``/``log_artifact`` both forward to the tracker *and* stash the
    value for the finalized :class:`RunRecord`; ``set_status``/``record_cost``
    only affect the record.
    """

    def __init__(self, attempt_id: str, recipe_id: str, tracker: ExperimentTracker) -> None:
        self.attempt_id = attempt_id
        self.recipe_id = recipe_id
        self._tracker = tracker
        self._metrics: dict[str, Any] | None = None
        self._metric_definition: str | None = None
        self._per_seed_metrics: list[dict[str, Any]] | None = None
        self._headline_metric: float | None = None
        self._artifacts: dict[str, str] = {}
        self._status_override: RunStatus | None = None
        self._spend_usd: float | None = None
        self._budget_exceeded: bool = False
        self._measurements: tuple[dict[str, Any], ...] | None = None
        self._experiment_facts: dict[str, Any] | None = None
        self._partial_judgements: tuple[dict[str, Any], ...] | None = None

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        metric_definition: str | None = None,
        per_seed_metrics: list[dict[str, Any]] | None = None,
        headline_metric: float | None = None,
        step: int | None = None,
    ) -> None:
        """Record metrics on the run and forward them to the tracker."""
        self._metrics = {**(self._metrics or {}), **dict(metrics)}
        if metric_definition is not None:
            self._metric_definition = metric_definition
        if per_seed_metrics is not None:
            self._per_seed_metrics = per_seed_metrics
        if headline_metric is not None:
            self._headline_metric = headline_metric
        self._tracker.log_metrics(metrics, step=step)

    def log_artifact(self, key: str, value: str) -> None:
        """Record an artifact path/URL on the run and forward it to the tracker."""
        self._artifacts[key] = value
        self._tracker.log_artifact(key, value)

    def set_status(self, status: RunStatus) -> None:
        """Override the terminal status (e.g. ``"budget_exceeded"``)."""
        self._status_override = status

    def record_cost(
        self,
        spend_usd: float | None,
        *,
        budget_exceeded: bool = False,
    ) -> None:
        """Record the run's total spend (``SpendMonitor.spent``) and cap state."""
        self._spend_usd = spend_usd
        self._budget_exceeded = budget_exceeded

    def record_measurements(self, measurements: Iterable[Mapping[str, Any]]) -> None:
        """Attach import-neutral serialized stage measurements to the terminal record."""
        self._measurements = _snapshot_measurements(measurements)

    def record_experiment_facts(self, facts: Mapping[str, Any]) -> None:
        """Attach an import-neutral typed-report snapshot to the terminal record."""
        self._experiment_facts = _snapshot_mapping(facts)

    def record_partial_judgements(
        self,
        judgements: Iterable[Mapping[str, Any]],
    ) -> None:
        """Retain already-produced local judgements after a failed cell."""
        self._partial_judgements = _snapshot_measurements(judgements)


@contextmanager
def capture_run(
    context: RunContext,
    *,
    store: str | Path | RunStore | None = None,
    tracker: ExperimentTracker = NoOpTracker(),
    recipe_id: str | None = None,
    evaluation_id: str | None = None,
    cache_id: str | None = None,
    protocol: Mapping[str, Any] | None = None,
    attempt_id: str | None = None,
    suppress_error_details: bool = False,
) -> Iterator[_RunHandle]:
    """Capture one run: identity, a ``running`` marker, and a terminal record.

    Uses the supplied experiment ``recipe_id`` or computes the legacy
    :func:`compute_recipe_id`, then mints ``attempt_id``; if ``store`` resolves,
    appends a ``status="running"`` line first (a crash then leaves a visible gap); sets
    the :data:`current_run` contextvar (set/reset token, so a nested capture
    restores the parent on exit); then, *inside* the protected block, starts the
    tracker and yields a :class:`_RunHandle`. Starting the tracker sits inside
    the try/finally on purpose: if ``start_run`` (or anything before the yield)
    raises, the contextvar is still reset and a terminal ``status="failed"``
    record is still written before the error propagates -- the ``running`` line
    never dangles. On exit it finalizes the :class:`RunRecord`
    (status/metrics/cost/timing/artifacts + the tracker's ``run_url``), appends
    the terminal line, and finishes the tracker. ``store=None`` writes NOTHING.
    """
    resolved_store = resolve_store(store)
    context_snapshot = RunContext.model_validate(context.model_dump(mode="python"))
    resolved_recipe_id = recipe_id or compute_recipe_id(context_snapshot)
    started_at = datetime.now(UTC).isoformat()
    resolved_attempt_id = attempt_id or mint_attempt_id(resolved_recipe_id)
    started_perf = time.perf_counter()
    running_record = RunRecord(
        attempt_id=resolved_attempt_id,
        recipe_id=resolved_recipe_id,
        context=context_snapshot,
        evaluation_id=evaluation_id,
        cache_id=cache_id,
        protocol=dict(protocol) if protocol is not None else None,
        started_at=started_at,
        status="running",
    )

    if resolved_store is not None:
        resolved_store.append(running_record)
    handle = _RunHandle(resolved_attempt_id, resolved_recipe_id, tracker)
    token = current_run.set(resolved_attempt_id)
    error_type: str | None = None
    error_message: str | None = None
    try:
        tracker.start_run(context_snapshot, run_name=context_snapshot.experiment)
        yield handle
    except BaseException as exc:
        error_type = type(exc).__name__
        error_message = (
            "run failed; exception details suppressed"
            if suppress_error_details
            else _sanitize_error_message(str(exc))
        )
        raise
    finally:
        current_run.reset(token)
        finished_at = datetime.now(UTC).isoformat()
        duration_seconds = time.perf_counter() - started_perf
        if error_type is not None:
            final_status: RunStatus = (
                "budget_exceeded" if handle._status_override == "budget_exceeded" else "failed"
            )
        else:
            final_status = handle._status_override or "completed"
        artifacts = dict(handle._artifacts)
        if tracker.run_url is not None:
            artifacts.setdefault("run_url", tracker.run_url)
        artifacts_snapshot = _snapshot_mapping(artifacts)
        assert artifacts_snapshot is not None
        if resolved_store is not None:
            resolved_store.append(
                running_record.model_copy(
                    update={
                        "measurements": handle._measurements,
                        "experiment_facts": handle._experiment_facts,
                        "partial_judgements": handle._partial_judgements,
                        "finished_at": finished_at,
                        "duration_seconds": duration_seconds,
                        "metrics": handle._metrics,
                        "metric_definition": handle._metric_definition,
                        "per_seed_metrics": handle._per_seed_metrics,
                        "headline_metric": handle._headline_metric,
                        "spend_usd": handle._spend_usd,
                        "budget_exceeded": handle._budget_exceeded,
                        "trace_id": resolved_attempt_id,
                        "artifacts": artifacts_snapshot,
                        "status": final_status,
                        "error_type": error_type,
                        "error_message": error_message,
                    }
                )
            )
        tracker.finish(status=final_status)
