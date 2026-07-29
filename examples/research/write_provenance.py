"""Record WHICH CODE produced a study's rows, as git blob hashes.

This runs as part of the measurement path, not as a rendering step, because a
provenance line derived at render time names whatever is checked out *now* —
which is not what measured the rows. That distinction is not pedantic: the first
version of this study's provenance read ``git rev-parse HEAD:<path>``, and the
very next commit touching the harness would have reattributed every measured row
to code that never ran. A provenance line that moves with ``HEAD`` does not go
stale, it goes **false**.

Blob hashes, not commit hashes: a blob changes only when file *content* changes,
so an unrelated commit does not invalidate the record.

Usage (called by ``run_lfm25.sh`` around the sweeps)::

    uv run python examples/research/write_provenance.py --start   # before
    uv run python examples/research/write_provenance.py --finish  # after

``--start`` captures the blobs as they are when measurement begins; ``--finish``
stamps the end of the window and refuses to write if the tracked files changed
mid-sweep, because then no single blob describes all the rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "research" / "20260729_lfm25_provenance.json"

#: The files whose content decides what the numbers MEAN, named individually
#: because they are what a reader diffs.
TRACKED = (
    "src/langres/experiments/statistics.py",
    "examples/research/embedder_ladder.py",
    # Source identity is not measurement identity. Every cell is a fresh
    # `uv run`, with neither `--no-sync` nor `--locked`, so uv re-synchronises
    # the environment each time: edit a dependency bound here and the next cell
    # can embed under a different transformers/torch while every .py blob and
    # tree digest compares equal and `--finish` certifies the sweep "unchanged".
    # A hash over the code that cannot see the environment it runs in is a check
    # decoupled from what it claims. Fatal, like the sources: this file does not
    # move on its own. (Cross-model review.)
    "pyproject.toml",
)

#: The RESOLVED environment, recorded but **not** fatal. ``uv.lock`` is what was
#: actually installed, so it is the sharper record — but it is gitignored (see
#: `.gitignore`; `exclude-newer` in `pyproject.toml` is the reproducibility pin
#: here), and `uv run` may rewrite it mid-sweep without any human touching it.
#: Aborting a multi-hour paid sweep on that would kill work that would have
#: succeeded, so it takes the same treatment as the drivers: hashed, disclosed,
#: never silent. Absent is a legitimate state and is recorded as such.
ENVIRONMENT = ("uv.lock",)

#: Naming two files is not the measurement's code identity. ``embedder_ladder.py``
#: executes ``langres.core.embeddings``, ``core.indexes.vector_index``,
#: ``core.blockers.vector``, ``metrics.metrics`` and the dataset registry, any of
#: which changes what a row means. So the whole importable package is digested
#: too: without it, a mid-sweep change to a blocker would still certify the run
#: "unchanged". (Cross-model review.)
TRACKED_TREES = ("src/langres", "examples/research")

#: What the dirty check looks at. **Derived, not a second hand-kept list.** The
#: check ran over ``TRACKED_TREES`` alone, whose scopes are directories — so
#: ``pyproject.toml``, tracked individually at the repo root, fell outside it: an
#: uncommitted dependency edit present at ``--start`` and untouched during the
#: sweep passed the endpoint comparison, and ``dirty_at_start`` stayed empty. The
#: report then named the last committed revision with no modified-tree warning,
#: over a checkout that cannot reproduce the environment that measured the rows.
#: Computing the scope from ``TRACKED`` means the next root-level entry is
#: covered on the day it is added, instead of waiting for someone to remember a
#: parallel list. (Cross-model review.)
DIRTY_SCOPES = TRACKED_TREES + tuple(
    path for path in TRACKED if not any(path.startswith(f"{tree}/") for tree in TRACKED_TREES)
)

#: The shell drivers are hashed SEPARATELY from the Python, because they fail
#: differently. A change to `.py` changes what a number MEANS -- no single hash
#: then describes the rows, so `--finish` refuses. A change to a driver changes
#: which cells RAN (model list, benchmark granularity, the memory guard), which
#: is provenance-relevant but not row-invalidating; drivers legitimately get
#: repaired mid-sweep, and this study's own drivers were, three times, after an
#: OS kill. Collapsing both into one boolean would either abort those repairs or
#: -- as the *.py-only digest did -- certify them as "unchanged". Recorded and
#: surfaced, not silently swallowed and not fatal.
DRIVER_TREE = "examples/research"
DRIVER_SUFFIX = ".sh"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _worktree_blob(path: str) -> str:
    """Hash the file **as it exists on disk**, not as committed; ``"absent"`` if deleted.

    ``run_lfm25.sh`` executes the working tree. Recording ``HEAD:<path>`` meant an
    uncommitted edit was invisible: both snapshots read the same committed blob
    and ``--finish`` certified the run unchanged while the rows came from code
    that was never committed. ``git hash-object`` gives the blob id the working
    file *would* have, so an uncommitted edit changes it. (Cross-model review.)

    **A deleted input is a provenance change, not a crash.** ``git hash-object``
    exits 128 on a missing file, and with ``check=True`` that propagated out of
    ``--verify``/``--finish`` *while the snapshot was being built* — before
    ``finished`` or ``verified_unchanged_during_run=false`` had been written. The
    driver then committed measured rows beside a still-open start sidecar: the
    one event most worth recording, deleting the code mid-run, was the one event
    that guaranteed it went unrecorded. A sentinel makes the deletion a differing
    blob, which is exactly what ``changed_during_run`` already reports.
    (Cross-model review.)

    Non-zero for a file that *does* exist stays fatal. Mapping every failure to
    the sentinel would turn a permissions error or a broken git into "absent" —
    a fail-open guard, which is the failure mode this repo keeps hitting.
    """
    target = REPO_ROOT / path
    result = subprocess.run(
        ["git", "hash-object", "--", str(target)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    if target.exists():
        raise RuntimeError(f"git hash-object failed for {path}: {result.stderr.strip()}")
    return "absent"


def _tree_digest() -> dict[str, str]:
    """One digest per tracked tree, over the working-tree contents of its .py files."""
    digests: dict[str, str] = {}
    for tree in TRACKED_TREES:
        root = REPO_ROOT / tree
        parts = [
            f"{p.relative_to(REPO_ROOT)}:{_worktree_blob(str(p.relative_to(REPO_ROOT)))}"
            for p in sorted(root.rglob("*.py"))
        ]
        digests[tree] = hashlib.sha256("\n".join(parts).encode()).hexdigest()
    return digests


def _driver_blobs() -> dict[str, str]:
    """Per-file blobs of the shell drivers that decide which cells the sweep runs.

    Per FILE, not just one digest over all of them: the digest answers "did
    anything move", but the report discloses *which* schedule produced the rows,
    and that needs names. Keeping only the digest left that disclosure permanently
    empty on live sidecars. (Cross-model review.)
    """
    root = REPO_ROOT / DRIVER_TREE
    return {
        str(p.relative_to(REPO_ROOT)): _worktree_blob(str(p.relative_to(REPO_ROOT)))
        for p in sorted(root.rglob(f"*{DRIVER_SUFFIX}"))
    }


def _driver_digest(blobs: dict[str, str] | None = None) -> str:
    """Digest of the shell drivers that decide which cells the sweep runs."""
    blobs = _driver_blobs() if blobs is None else blobs
    parts = [f"{path}:{blob}" for path, blob in blobs.items()]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _changed_drivers(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Which driver files were added, removed or edited during the window."""
    changed = [f"{p} (added)" for p in sorted(set(after) - set(before))]
    changed += [f"{p} (removed)" for p in sorted(set(before) - set(after))]
    changed += [p for p in sorted(set(before) & set(after)) if before[p] != after[p]]
    return changed


def _environment_blobs() -> dict[str, str]:
    """Blob hash per resolved-environment file, or ``"absent"`` when it is missing."""
    return {path: _worktree_blob(path) for path in ENVIRONMENT}


def _blobs() -> dict[str, dict[str, str]]:
    """Working-tree blob hash and last-touching commit for each tracked file."""
    out: dict[str, dict[str, str]] = {}
    for path in TRACKED:
        out[path] = {
            "blob": _worktree_blob(path),
            "last_commit": _git("log", "-1", "--format=%h", "--", path),
            "last_commit_date": _git("log", "-1", "--format=%cI", "--", path),
        }
    return out


def _dirty(paths: tuple[str, ...]) -> list[str]:
    """Files with uncommitted modifications or untracked content under ``paths``.

    Deliberately NOT routed through ``_git``, which ``.strip()``s: porcelain emits
    a two-column status field, so a modified-but-unstaged file starts with a SPACE
    (``" M path"``). Stripping ate it, and the fixed-width ``line[3:]`` slice then
    cut one character into the path -- the first record came out as
    ``xamples/research/...``. A provenance record that silently mangles the name of
    the file it is warning about is worse than not having one.
    """
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *paths],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line[3:] for line in out.splitlines() if line.strip()]


def _bootstrap_samples() -> int:
    """Read the replicate count from the signature, so it cannot be typed wrong."""
    import re

    src = (REPO_ROOT / "src" / "langres" / "experiments" / "statistics.py").read_text()
    match = re.search(r"^\s*samples:\s*int\s*=\s*(\d+)", src, re.MULTILINE)
    return int(match.group(1)) if match else 0


COMMENT = [
    "The code that produced the committed rows, captured as git BLOB hashes of the",
    "WORKING TREE -- what the sweep actually executes, not what is committed.",
    "Written by examples/research/write_provenance.py as part of the MEASUREMENT",
    "path -- never derived from HEAD at render time, which would name whatever is",
    "checked out now rather than what measured the rows, and would silently",
    "reattribute every row after the next harness commit. Not stale: false.",
    "`tree_digests` covers every .py under the tracked trees, because naming two",
    "files is not the measurement's code identity: the harness executes blockers,",
    "indexes, embedders and metrics that can change what a row means.",
    "`verified_unchanged_during_run` compares the START and FINISH snapshots. It is",
    "an ENDPOINT check, not a continuous one: a file edited during the sweep, used",
    "by some cells, and restored before --finish would leave both endpoints equal",
    "and read as unchanged. It catches a change that PERSISTS, which is the",
    "realistic case (edits are made and kept); it cannot catch edit-and-revert.",
    "Stated here rather than implied, because a guarantee is only worth what its",
    "weakest case is. Per-cell hashing or an immutable checkout is what would close",
    "it; this run did neither and does not claim to.",
    "`driver_digest` covers the *.sh sweep drivers separately, because they decide",
    "which cells RAN rather than what a number means. A driver edit is recorded in",
    "`drivers_unchanged_during_run` and is NOT fatal -- drivers get repaired",
    "mid-sweep -- but it is never silent: the *.py-only digest used to certify such",
    "an edit as 'unchanged'.",
    "Code identity is not measurement identity: every cell is a fresh `uv run`",
    "with neither --no-sync nor --locked, so the ENVIRONMENT can move between",
    "cells while every source hash compares equal. `pyproject.toml` is therefore in",
    "`blobs` and fatal; `uv.lock` -- gitignored, and rewritable by uv without a",
    "human touching it -- is in `environment_blobs` and disclosed via",
    "`environment_unchanged_during_run` without aborting a paid sweep. `null`",
    "there means the window predates the field: not observed, not unchanged.",
]


def _snapshot() -> dict[str, Any]:
    driver_blobs = _driver_blobs()
    return {
        "blobs": _blobs(),
        "tree_digests": _tree_digest(),
        "driver_digest": _driver_digest(driver_blobs),
        "driver_blobs": driver_blobs,
        "environment_blobs": _environment_blobs(),
    }


def _uncommitted(path: Path) -> list[str]:
    """What would be LOST by overwriting ``path``; empty only when nothing is.

    "git reports no changes" is not the same as "git could see a change".
    Returning ``[]`` for anything git cannot describe made the guard hand out a
    free pass to precisely the files most worth protecting: an ``--output``
    outside the repository — which is the recovery copy the refusal message
    itself tells operators to make — and any gitignored in-repo path, both of
    which `git status` omits entirely. A file whose clean tracked state cannot be
    ESTABLISHED is treated as pending. Non-existent is the one safe case.
    (Cross-model review.)
    """
    try:
        rel = str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        # Outside the repository: existence is all there is to go on, and a file
        # git cannot describe cannot be established as saved.
        return (
            [f"{path} (outside this repository — git cannot tell whether it is saved)"]
            if path.exists()
            else []
        )
    # NOT guarded by `path.exists()`. A pending DELETION of a tracked sidecar also
    # fails that test, and returning "nothing to lose" let `--start` recreate and
    # commit the file over a deletion the operator had staged — without asking for
    # `--force`. It is the same existence-test blind spot the artifact guards had
    # dropped one round earlier, still living here in Python. Git is asked first;
    # a path that was never created is silent either way. (Cross-model review.)
    pending = _dirty((rel,))
    if pending:
        return pending
    tracked = subprocess.run(
        ["git", "ls-files", "--", rel], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    if not tracked:
        # Untracked AND unmodified means either "ignored" or "never created". Only
        # the first has contents to lose.
        return (
            [f"{rel} (ignored by git — its contents exist in no commit)"] if path.exists() else []
        )
    return []


def start(output: Path, force: bool = False) -> None:
    # UNCOMMITTED CHANGES ARE SACRED, and this function's whole job is to
    # overwrite a tracked file. The realistic loss: a prior run is killed between
    # `--finish` writing the closed record and the driver committing it, so the
    # sidecar on disk is the ONLY description of rows that run_ladder.sh already
    # committed -- and the next `--start` silently replaced those bytes. Nothing
    # in git can bring them back. Refused, with `--force` as the deliberate
    # escape rather than a dead end. An UNTRACKED sidecar counts as pending too
    # (`_dirty` passes --untracked-files=all): a first-ever run has nothing to
    # lose and is trivially forced, while a killed run's brand-new sidecar is
    # exactly the file worth protecting. (Cross-model review.)
    pending = _uncommitted(output)
    if pending and not force:
        raise SystemExit(
            f"Refusing to open a new window over {output}: it has uncommitted changes "
            f"({', '.join(pending)}). Those bytes may be the only record of rows already "
            "committed by a run that was killed before its provenance was. Commit it, or "
            "copy it somewhere outside this worktree, then re-run. Pass --force to "
            "overwrite it deliberately."
        )
    # A dirty tree is not fatal -- research often measures uncommitted code -- but
    # it must be recorded, because the blob hashes then name content that exists
    # nowhere in history and cannot be recovered by anyone reading this file.
    dirty = _dirty(DIRTY_SCOPES)
    if dirty:
        logger.warning("measuring a DIRTY tree; %d modified path(s) recorded", len(dirty))
    snapshot = _snapshot()
    doc: dict[str, Any] = {
        "_comment": COMMENT,
        "measurement_window": {
            "started": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "finished": None,
            "head_when_sweep_started": _git("rev-parse", "--short", "HEAD"),
        },
        "bootstrap": {
            "function": "langres.experiments.statistics.paired_entity_bootstrap",
            "samples": _bootstrap_samples(),
            "resampled_unit": "gold cluster",
        },
        "dirty_at_start": dirty,
        **snapshot,
        "verified_unchanged_during_run": None,
    }
    output.write_text(json.dumps(doc, indent=2) + "\n")
    logger.info("provenance opened at %s", output)


def finish(output: Path, partial: bool = False) -> None:
    if not output.exists():
        raise SystemExit(f"{output} missing -- run --start before the sweep")
    doc = json.loads(output.read_text())
    # A CLOSED window is closed. Not "closed complete": the first version of this
    # guard refused only the complete->partial direction, which left the mirror
    # image open -- a plain `--finish` over a sidecar closed with `--partial`
    # flipped `window_complete` to true while KEEPING the `partial_note` saying
    # the sweep aborted, a file contradicting itself. Both directions restamp
    # `finished` and recompute the stability verdict against code that may have
    # moved since the run ended, turning a historical record into a statement
    # about now. A second window needs a second `--start`. Refused BEFORE
    # anything is written, so the closed state survives intact.
    # (Cross-model review.)
    if doc["measurement_window"].get("finished") is not None:
        raise SystemExit(
            f"Refusing to re-finish {output}: this window already closed at "
            f"{doc['measurement_window']['finished']} "
            f"(window_complete={doc.get('window_complete')}). Closing it again would "
            "restamp the timestamp and re-derive the stability verdict from whatever the "
            "tree looks like now, not from what measured the rows. The sidecar is "
            "unchanged; commit it as it stands, or run --start to open a new window."
        )
    # `studies_measured` is what was PLANNED at --start. On an abort partway
    # through, merge-commits leave every untouched older row in place, so a
    # sidecar still claiming ["a", "b"] made the report say the window covers
    # "every row in both studies" and attributed stale rows to a measurement that
    # never reached them. The completeness of the window is recorded separately
    # from its intent. (Cross-model review.)
    doc["window_complete"] = not partial
    if partial:
        doc["partial_note"] = (
            "The sweep ABORTED before finishing every planned study. "
            "`studies_measured` is what was planned, not what was reached: rows "
            "left untouched by this run predate the window described here."
        )
    after = _snapshot()
    # Keyed on what the START snapshot actually observed, with the finish lookup
    # DEFAULTED. ``after["blobs"][p]`` assumed the two snapshots carry identical
    # key sets -- true only while ``TRACKED`` itself holds still, and this study's
    # own harness was edited mid-sweep three times. A KeyError here is the same
    # defect as the one the "absent" sentinel fixes: it aborts while BUILDING the
    # verdict, before `finished` is written, so the event that caused it goes
    # unrecorded. Not keyed on the union: a path the start snapshot never recorded
    # has no baseline to differ from, and calling that "changed" would invent a
    # finding out of a missing observation. (Cross-model review.)
    after_blobs = after["blobs"]
    changed = [
        path
        for path, meta in doc["blobs"].items()
        if meta["blob"] != after_blobs.get(path, {}).get("blob", "absent")
    ]
    changed += [
        f"{tree}/**"
        for tree, d in doc.get("tree_digests", {}).items()
        if d != after["tree_digests"].get(tree)
    ]
    doc["measurement_window"]["finished"] = (
        datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    )
    doc["verified_unchanged_during_run"] = not changed
    if changed:
        doc["changed_during_run"] = changed
        logger.error("measurement code changed mid-sweep: %s", changed)
    # Non-fatal, but never silent: which cells ran can differ from what the
    # committed driver would run today.
    doc["drivers_unchanged_during_run"] = doc.get("driver_digest") == after["driver_digest"]
    if not doc["drivers_unchanged_during_run"]:
        # NAME them. Storing only the boolean meant `_provenance_section()` -- which
        # renders a "they did change" list from `drivers_changed_during_run` -- had
        # nothing to render on a live sidecar, so the disclosure printed empty and
        # the reader could not tell which schedule produced the rows. A report field
        # that can only ever be populated by the retroactive backfill is a
        # disclosure decoupled from the thing it discloses.
        doc["drivers_changed_during_run"] = _changed_drivers(
            doc.get("driver_blobs", {}), after["driver_blobs"]
        )
        doc["driver_digest_at_finish"] = after["driver_digest"]
        logger.warning(
            "the sweep DRIVERS (%s/*%s) changed mid-run; rows stand, but the committed "
            "driver is not the one that scheduled every cell",
            DRIVER_TREE,
            DRIVER_SUFFIX,
        )
    # The resolved environment, on the same non-fatal terms as the drivers.
    # `pyproject.toml` moving IS fatal (it is in `blobs`); `uv.lock` moving under
    # a `uv run` nobody asked to re-lock is disclosed instead of aborting a paid
    # sweep. Silent was never an option: without this, a re-sync between cells
    # left every hash equal and the run certified "unchanged".
    # A sidecar opened before this field existed did not RECORD the environment;
    # it did not record it CHANGING. `None` says "not observed", which is the
    # honest answer -- claiming a change that was never measured is the same
    # class of false statement as claiming stability that was never measured.
    before_env = doc.get("environment_blobs")
    doc["environment_unchanged_during_run"] = (
        None if before_env is None else before_env == after["environment_blobs"]
    )
    if doc["environment_unchanged_during_run"] is False:
        doc["environment_changed_during_run"] = _changed_drivers(
            before_env or {}, after["environment_blobs"]
        )
        logger.warning(
            "the resolved ENVIRONMENT (%s) changed mid-run; cells may have run under "
            "different dependency versions",
            ", ".join(ENVIRONMENT),
        )
    output.write_text(json.dumps(doc, indent=2) + "\n")
    if changed:
        raise SystemExit(
            f"Measurement code changed while the sweep was running ({', '.join(changed)}), "
            "so no single hash describes all the rows. Re-measure on a stable tree."
        )
    logger.info("provenance closed at %s", output)


def verify(output: Path, required: bool = False) -> None:
    """Has the measurement code moved since ``--start``? Exit non-zero if so.

    Called from the driver before each model's results are PUBLISHED, because
    ``--finish`` runs once at the end while ``run_ladder.sh`` commits and pushes
    after every model. Without this, a mid-sweep edit to a tracked ``.py`` could
    have its rows already committed and pushed by the time ``--finish`` rejects
    the run — the refusal arriving after publication it was meant to prevent.

    Silent success when there is no sidecar and ``required`` is false: the
    standing portfolio ladder runs this driver with no provenance at all, and
    must not be blocked by a file it never creates.

    ``required`` is how a run that DOES have provenance says so. Absence was
    treated as "not participating" with no way to tell it from "the open sidecar
    was deleted mid-sweep" — and in the second case every later model's rows were
    committed and pushed beside a start snapshot that no longer existed, with
    ``--finish`` unable to close or reconstruct the window. A missing file is a
    failed verification exactly when the caller expected one. The flag is set by
    the LFM driver, which opens the window; the ladder run that does not open one
    never passes it. (Cross-model review.)
    """
    if not output.exists():
        if required:
            raise SystemExit(
                f"Provenance sidecar {output} is GONE, but this run opened one. Rows "
                "measured after it disappeared have no start snapshot to be verified "
                "against and --finish cannot close the window. Not publishing."
            )
        return
    doc = json.loads(output.read_text())
    if doc.get("measurement_window", {}).get("finished") is not None:
        return
    after = _snapshot()
    # Defaulted, like `finish()`. This line had the same KeyError -- an abort
    # while BUILDING the verdict -- and only one of the two copies was fixed last
    # round. A guard that crashes on the state it exists to describe is not a
    # guard. (Cross-model review.)
    after_blobs = after["blobs"]
    changed = [
        path
        for path, meta in doc.get("blobs", {}).items()
        if meta["blob"] != after_blobs.get(path, {}).get("blob", "absent")
    ]
    changed += [
        f"{tree}/**"
        for tree, d in doc.get("tree_digests", {}).items()
        if d != after["tree_digests"].get(tree)
    ]
    if changed:
        raise SystemExit(
            f"Measurement code changed mid-sweep ({', '.join(changed)}). These rows were "
            "not produced by the code the open provenance window describes."
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start", action="store_true", help="open the window before measuring")
    group.add_argument("--finish", action="store_true", help="close it and verify stability")
    group.add_argument(
        "--verify",
        action="store_true",
        help=(
            "check stability WITHOUT closing the window, for use before each "
            "per-model publication. Silent no-op when no open sidecar exists."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--partial",
        action="store_true",
        help=(
            "with --finish: the sweep aborted before reaching every planned "
            "study, so rows it never touched are NOT described by this window."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "with --start: overwrite an existing sidecar that has uncommitted "
            "changes. Those bytes can be the only record of rows a killed run "
            "already committed, so it is refused unless you say so."
        ),
    )
    parser.add_argument(
        "--required",
        action="store_true",
        help=(
            "with --verify: this run OPENED a provenance window, so a missing "
            "sidecar is a failed verification rather than a non-participating "
            "ladder run. Without it, deleting the open sidecar mid-sweep passed "
            "silently and every later model was published against nothing."
        ),
    )
    parser.add_argument(
        "--studies",
        nargs="*",
        default=["a", "b"],
        help=(
            "Which studies this window actually re-measured. A partial run "
            "(LFM25_STUDY=a|b) leaves the other study's rows untouched, and its "
            "provenance must not be overwritten with a claim that covers them."
        ),
    )
    args = parser.parse_args()
    if args.start:
        start(args.output, force=args.force)
        doc = json.loads(args.output.read_text())
        doc["studies_measured"] = list(args.studies)
        args.output.write_text(json.dumps(doc, indent=2) + "\n")
    elif args.verify:
        verify(args.output, required=args.required)
    else:
        finish(args.output, partial=args.partial)


if __name__ == "__main__":
    main()
