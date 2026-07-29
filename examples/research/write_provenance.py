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
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "research" / "20260729_lfm25_provenance.json"

#: The files whose content decides what the numbers MEAN.
TRACKED = (
    "src/langres/experiments/statistics.py",
    "examples/research/embedder_ladder.py",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _blobs() -> dict[str, dict[str, str]]:
    """Current blob hash and last-touching commit for each tracked file."""
    out: dict[str, dict[str, str]] = {}
    for path in TRACKED:
        out[path] = {
            "blob": _git("rev-parse", f"HEAD:{path}"),
            "last_commit": _git("log", "-1", "--format=%h", "--", path),
            "last_commit_date": _git("log", "-1", "--format=%cI", "--", path),
        }
    return out


def _bootstrap_samples() -> int:
    """Read the replicate count from the signature, so it cannot be typed wrong."""
    import re

    src = (REPO_ROOT / "src" / "langres" / "experiments" / "statistics.py").read_text()
    match = re.search(r"^\s*samples:\s*int\s*=\s*(\d+)", src, re.MULTILINE)
    return int(match.group(1)) if match else 0


COMMENT = [
    "The code that produced the committed rows, captured as git BLOB hashes.",
    "Written by examples/research/write_provenance.py as part of the MEASUREMENT",
    "path -- never derived from HEAD at render time, which would name whatever is",
    "checked out now rather than what measured the rows, and would silently",
    "reattribute every row after the next harness commit. Not stale: false.",
    "A blob hash changes only when file CONTENT changes, which is the property",
    "being recorded. `verified_unchanged_during_run` is the check that no tracked",
    "file was edited mid-sweep; without it no single blob describes all the rows.",
]


def start(output: Path) -> None:
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
        "blobs": _blobs(),
        "verified_unchanged_during_run": None,
    }
    output.write_text(json.dumps(doc, indent=2) + "\n")
    logger.info("provenance opened at %s", output)


def finish(output: Path) -> None:
    if not output.exists():
        raise SystemExit(f"{output} missing -- run --start before the sweep")
    doc = json.loads(output.read_text())
    before = doc["blobs"]
    after = _blobs()
    unchanged = all(before[p]["blob"] == after[p]["blob"] for p in before)
    doc["measurement_window"]["finished"] = (
        datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    )
    doc["verified_unchanged_during_run"] = unchanged
    if not unchanged:
        changed = [p for p in before if before[p]["blob"] != after[p]["blob"]]
        doc["changed_during_run"] = changed
        logger.error("tracked files changed mid-sweep: %s", changed)
    output.write_text(json.dumps(doc, indent=2) + "\n")
    if not unchanged:
        raise SystemExit(
            "A tracked file changed while the sweep was running, so no single blob "
            "describes all the rows. Re-measure on a stable tree."
        )
    logger.info("provenance closed at %s", output)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start", action="store_true", help="open the window before measuring")
    group.add_argument("--finish", action="store_true", help="close it and verify stability")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    (start if args.start else finish)(args.output)


if __name__ == "__main__":
    main()
