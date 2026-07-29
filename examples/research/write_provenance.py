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
)

#: Naming two files is not the measurement's code identity. ``embedder_ladder.py``
#: executes ``langres.core.embeddings``, ``core.indexes.vector_index``,
#: ``core.blockers.vector``, ``metrics.metrics`` and the dataset registry, any of
#: which changes what a row means. So the whole importable package is digested
#: too: without it, a mid-sweep change to a blocker would still certify the run
#: "unchanged". (Cross-model review.)
TRACKED_TREES = ("src/langres", "examples/research")

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
    """Hash the file **as it exists on disk**, not as committed.

    ``run_lfm25.sh`` executes the working tree. Recording ``HEAD:<path>`` meant an
    uncommitted edit was invisible: both snapshots read the same committed blob
    and ``--finish`` certified the run unchanged while the rows came from code
    that was never committed. ``git hash-object`` gives the blob id the working
    file *would* have, so an uncommitted edit changes it. (Cross-model review.)
    """
    return _git("hash-object", str(REPO_ROOT / path))


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
]


def _snapshot() -> dict[str, Any]:
    driver_blobs = _driver_blobs()
    return {
        "blobs": _blobs(),
        "tree_digests": _tree_digest(),
        "driver_digest": _driver_digest(driver_blobs),
        "driver_blobs": driver_blobs,
    }


def start(output: Path) -> None:
    # A dirty tree is not fatal -- research often measures uncommitted code -- but
    # it must be recorded, because the blob hashes then name content that exists
    # nowhere in history and cannot be recovered by anyone reading this file.
    dirty = _dirty(TRACKED_TREES)
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
    changed = [p for p, meta in doc["blobs"].items() if meta["blob"] != after["blobs"][p]["blob"]]
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
    output.write_text(json.dumps(doc, indent=2) + "\n")
    if changed:
        raise SystemExit(
            f"Measurement code changed while the sweep was running ({', '.join(changed)}), "
            "so no single hash describes all the rows. Re-measure on a stable tree."
        )
    logger.info("provenance closed at %s", output)


def verify(output: Path) -> None:
    """Has the measurement code moved since ``--start``? Exit non-zero if so.

    Called from the driver before each model's results are PUBLISHED, because
    ``--finish`` runs once at the end while ``run_ladder.sh`` commits and pushes
    after every model. Without this, a mid-sweep edit to a tracked ``.py`` could
    have its rows already committed and pushed by the time ``--finish`` rejects
    the run — the refusal arriving after publication it was meant to prevent.

    Silent success when there is no sidecar or the window is already closed: the
    standing portfolio ladder runs this driver with no provenance at all, and
    must not be blocked by a file it never creates.
    """
    if not output.exists():
        return
    doc = json.loads(output.read_text())
    if doc.get("measurement_window", {}).get("finished") is not None:
        return
    after = _snapshot()
    changed = [
        p for p, meta in doc.get("blobs", {}).items() if meta["blob"] != after["blobs"][p]["blob"]
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
        start(args.output)
        doc = json.loads(args.output.read_text())
        doc["studies_measured"] = list(args.studies)
        args.output.write_text(json.dumps(doc, indent=2) + "\n")
    elif args.verify:
        verify(args.output)
    else:
        finish(args.output, partial=args.partial)


if __name__ == "__main__":
    main()
