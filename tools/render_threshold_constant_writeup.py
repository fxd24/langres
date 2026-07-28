"""Splice the sweep's generated tables into the threshold-constant write-up.

Three PRs shipped factual errors on 2026-07-28 and **every one was in hand-typed
prose while the generated tables beside it were correct**. So this write-up's
tables are not transcribed: the prose lives in
``docs/research/_20260728_threshold_constant.body.md`` with ``@@MARKER@@`` lines,
and this script replaces each marker with the corresponding block printed by
``examples/research/threshold_constant_sweep.py --render``.

Run after a sweep (or after editing the prose)::

    uv run python tools/render_threshold_constant_writeup.py

It is idempotent and rewrites the whole document, so a stale table cannot
survive a regeneration.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

#: Marker -> the ``print_tables`` heading whose block replaces it. The headings
#: are matched EXACTLY, so renaming one in the harness fails this script loudly
#: rather than silently dropping a table from the write-up.
SECTIONS: dict[str, str] = {
    "@@FAMILIES@@": "Score families, and which this study can speak to:",
    "@@CELLS@@": "Per-cell held-out sweep (blocked-gold convention):",
    "@@LOBO@@": "Leave-one-benchmark-out constant, graded on the held-out benchmark:",
    "@@ALLGOLD@@": "The same, under the end-to-end (all-gold) convention:",
    "@@STABILITY@@": "Is the constant stable when a dataset is dropped?",
    "@@LADDER@@": "The ladder: shipped constant -> free constant -> labels -> oracle:",
    "@@VERDICT@@": "Verdict against the pre-registered rule:",
}

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "examples/research/threshold_constant_sweep.py"
ARTIFACT = REPO / "examples/research/results/threshold_constant_sweep.json"
BODY = REPO / "docs/research/_20260728_threshold_constant.body.md"
OUT = REPO / "docs/research/20260728_threshold_constant.md"


def render(artifact: Path) -> str:
    """Return the harness's own table output for ``artifact``."""
    completed = subprocess.run(
        [sys.executable, str(HARNESS), "--render", str(artifact)],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


def split_sections(rendered: str) -> dict[str, str]:
    """Split ``print_tables`` output into ``{heading: block}``.

    Raises:
        RuntimeError: If a heading this script needs is absent -- a renamed
            heading must fail here, not silently leave a marker in the document.
    """
    headings = set(SECTIONS.values())
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in rendered.splitlines():
        if line.strip() in headings:
            current = line.strip()
            blocks[current] = []
            continue
        if current is not None:
            blocks[current].append(line)
    missing = headings - set(blocks)
    if missing:
        raise RuntimeError(f"the harness printed no section for: {sorted(missing)}")
    return {heading: "\n".join(lines).strip("\n") for heading, lines in blocks.items()}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--body", type=Path, default=BODY)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    blocks = split_sections(render(args.artifact))
    text = args.body.read_text()
    for marker, heading in SECTIONS.items():
        if marker not in text:
            raise RuntimeError(f"{args.body} has no {marker} to fill")
        text = text.replace(marker, blocks[heading])
    args.out.write_text(text)
    print(f"[out] {args.out}")  # noqa: T201 - operator tool


if __name__ == "__main__":
    main()
