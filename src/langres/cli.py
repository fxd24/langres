"""The ``langres`` command line: human review surfaces for the flywheel.

:func:`langres.core.review.select_for_review` writes a ``review_queue.jsonl``
snapshot of the judged pairs worth a human's attention; this module is where a
human answers them. Two review surfaces, one contract
(:class:`~langres.core.harvest.Correction`):

- **The primary path is the CSV round-trip.** ``export-csv`` turns a queue into
  a labelable spreadsheet (``left_*``/``right_*`` display columns + an empty
  ``label`` column); a reviewer fills the ``label`` column in Excel/Sheets;
  ``import-csv`` reads that spreadsheet back into a ``corrections.jsonl`` log.
- ``review`` is the quick terminal loop -- a ``y/n/s/q`` prompt per pair for a
  developer who would rather stay in the shell. Each answer is appended to the
  corrections log *immediately*, so quitting (or ctrl-D) never loses answered
  work and a re-run resumes where it left off.

Design constraints (this is a packaged console-script entry point):

- **stdlib + the light langres contracts only.** ``argparse``/``csv``/``re``
  plus the pydantic-only :class:`ReviewItem`/:class:`ReviewQueue`/
  :class:`Correction`/:class:`CorrectionLog` models -- no torch/litellm/faiss,
  and the CLI never makes a paid call.
- **All output flows through an injected stream** (``output_stream``), and all
  input through ``input_stream``, so every path is testable with ``StringIO``
  and no TTY. Ruff bans ``print`` in ``src/`` (T201); this module honors that.
- **Two adversarial-input defenses.** ``export-csv`` prefixes formula-leading
  display cells (``=``/``+``/``-``/``@``) with ``'`` so a crafted record field
  cannot become a spreadsheet formula -- but it *never* touches the id columns
  (escaping an id like ``-42`` would break ``import-csv``'s own pair
  validation). This assumes record **ids are internal/trusted**: a ``-``/``=``
  leading id from an *untrusted* source stays a live formula-injection vector
  when the export is opened in a spreadsheet -- the deliberate tradeoff for a
  lossless round-trip, not a bug. ``review`` strips C0/C1/ANSI control
  characters out of rendered record content so a hostile field cannot clear or
  spoof the reviewer's terminal.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from langres import __version__
from langres.core.harvest import Correction, CorrectionLog
from langres.core.review import ReviewItem, ReviewQueue

_DEFAULT_CORRECTIONS = "corrections.jsonl"

#: Cells whose first character is one of these are treated as a formula by
#: Excel/Google Sheets -- the CSV-injection vector. See :func:`_escape_formula`.
_FORMULA_LEADERS = ("=", "+", "-", "@")

#: Label tokens ``import-csv`` accepts (case-insensitive; blank = skip the row).
_TRUE_TOKENS = frozenset({"y", "yes", "true", "t", "1"})
_FALSE_TOKENS = frozenset({"n", "no", "false", "f", "0"})

#: ANSI escape sequences (CSI ``ESC [ … final`` plus the two-byte Fe escapes).
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-Z\\-_]")
#: C0 controls (incl. a bare ESC), DEL, and C1 controls.
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")

_PROMPT = "[y]es match  [n]o  [s]kip  [q]uit > "


def main(
    argv: Sequence[str] | None = None,
    *,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> int:
    """Dispatch a ``langres`` subcommand. Returns the process exit code.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
        input_stream: Where ``review`` reads answers from (defaults to stdin) --
            injectable so the interactive loop is testable without a TTY.
        output_stream: Where every command writes its output (defaults to
            stdout) -- all user-facing text goes here, never ``print``.
    """
    out_stream = output_stream if output_stream is not None else sys.stdout
    in_stream = input_stream if input_stream is not None else sys.stdin

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        out_stream.write(f"langres {__version__}\n")
        return 0
    if args.command == "review":
        return _review(Path(args.queue), Path(args.out), args.reviewer, in_stream, out_stream)
    if args.command == "export-csv":
        return _export_csv(Path(args.queue), Path(args.out_csv), out_stream)
    if args.command == "import-csv":
        return _import_csv(
            Path(args.in_csv), Path(args.queue), Path(args.out), args.reviewer, out_stream
        )

    parser.print_help(out_stream)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (CSV round-trip framed as the primary path)."""
    parser = argparse.ArgumentParser(
        prog="langres",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Human review tooling for the langres flywheel.\n\n"
            "Primary path -- export a review queue to CSV, label it in a spreadsheet, "
            "then import the labeled CSV back into a corrections log:\n"
            "    uv run langres export-csv review_queue.jsonl to_label.csv\n"
            "    #  ... open to_label.csv, fill the 'label' column (y/n), save ...\n"
            "    uv run langres import-csv to_label.csv review_queue.jsonl\n\n"
            "'review' is a quick terminal labeling loop for developers who prefer "
            "to stay in the shell."
        ),
        epilog=(
            "After `pip install langres`, drop the `uv run` prefix (e.g. `langres export-csv ...`)."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the installed langres version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="{review,export-csv,import-csv}")

    p_review = subparsers.add_parser(
        "review",
        help="Label a review queue interactively in the terminal (quick-loop convenience).",
        description=(
            "Walk a review queue pair by pair with a y/n/s/q prompt, appending each "
            "answer to the corrections log immediately (ctrl-D or 'q' quits; re-run to "
            "resume). The CSV round-trip (export-csv / import-csv) is the primary path."
        ),
    )
    p_review.add_argument("queue", metavar="queue.jsonl", help="The review_queue.jsonl to label.")
    p_review.add_argument(
        "--out",
        default=_DEFAULT_CORRECTIONS,
        metavar="corrections.jsonl",
        help="Corrections log to append to (default: %(default)s). Pre-read to skip answered pairs.",
    )
    p_review.add_argument(
        "--reviewer",
        default=None,
        metavar="NAME",
        help="Optional reviewer name recorded on each correction.",
    )

    p_export = subparsers.add_parser(
        "export-csv",
        help="Write a review queue as a labelable CSV (the primary review path).",
        description=(
            "Write a review queue as a spreadsheet with left_*/right_* display columns "
            "and an empty 'label' column for a human to fill in. Formula-leading display "
            "cells are escaped; the id columns are left byte-for-byte intact so the "
            "import round-trip stays valid."
        ),
    )
    p_export.add_argument("queue", metavar="queue.jsonl", help="The review_queue.jsonl to export.")
    p_export.add_argument("out_csv", metavar="out.csv", help="The CSV file to write.")

    p_import = subparsers.add_parser(
        "import-csv",
        help="Read a labeled CSV back into a corrections log (not a record importer).",
        description=(
            "Read a CSV whose 'label' column has been filled in (y/yes/true/1 or "
            "n/no/false/0; blank rows are skipped) back into a corrections log. Each "
            "row's (left_id, right_id) is validated against the review queue; an "
            "unrecognized label or an unknown pair aborts with the row number and "
            "writes nothing."
        ),
    )
    p_import.add_argument("in_csv", metavar="in.csv", help="The labeled CSV to read.")
    p_import.add_argument(
        "queue", metavar="queue.jsonl", help="The review queue the CSV was exported from."
    )
    p_import.add_argument(
        "--out",
        default=_DEFAULT_CORRECTIONS,
        metavar="corrections.jsonl",
        help="Corrections log to append to (default: %(default)s).",
    )
    p_import.add_argument(
        "--reviewer",
        default=None,
        metavar="NAME",
        help="Optional reviewer name recorded on each correction.",
    )
    return parser


def _review(
    queue_path: Path,
    out_path: Path,
    reviewer: str | None,
    in_stream: TextIO,
    out_stream: TextIO,
) -> int:
    """Interactive terminal labeling loop; appends each answer immediately."""
    if not queue_path.exists():
        out_stream.write(f"error: review queue not found: {queue_path}\n")
        return 1

    items = ReviewQueue(queue_path).read()
    if not items:
        out_stream.write(
            "Review queue is empty -- nothing to review. Regenerate it with select_for_review().\n"
        )
        return 0

    log = CorrectionLog(out_path)
    answered = {_pair_key(c.left_id, c.right_id) for c in log.read()}
    reviewed = 0
    stopped = False
    for index, item in enumerate(items, start=1):
        key = _pair_key(item.left_id, item.right_id)
        if key in answered:
            continue
        out_stream.write(_render_item(item, index, len(items)))
        answer = _prompt(in_stream, out_stream)
        if answer is None or answer == "q":
            stopped = True
            break
        if answer == "s":
            continue
        log.append(
            Correction(
                left_id=item.left_id,
                right_id=item.right_id,
                label=answer == "y",
                reviewer=reviewer,
                original_score=item.score,
                original_verdict=item.verdict,
            )
        )
        answered.add(key)
        reviewed += 1

    if stopped:
        out_stream.write(
            f"\nStopped. Saved {reviewed} correction(s) to {out_path}; re-run to resume.\n"
        )
    else:
        out_stream.write(f"\nDone. Saved {reviewed} correction(s) to {out_path}.\n")
    return 0


def _prompt(in_stream: TextIO, out_stream: TextIO) -> str | None:
    """Prompt until a valid answer; return ``"y"``/``"n"``/``"s"``/``"q"`` or ``None`` on EOF."""
    while True:
        out_stream.write(_PROMPT)
        out_stream.flush()
        line = in_stream.readline()
        if line == "":  # EOF / ctrl-D -- treat as quit; answered work is already durable
            return None
        answer = line.strip().lower()
        if answer in ("y", "yes"):
            return "y"
        if answer in ("n", "no"):
            return "n"
        if answer in ("s", "skip"):
            return "s"
        if answer in ("q", "quit"):
            return "q"
        out_stream.write("Please answer y (yes), n (no), s (skip), or q (quit).\n")


def _render_item(item: ReviewItem, index: int, total: int) -> str:
    """Render one pair side by side for terminal review (record content sanitized)."""
    verdict = "MATCH" if item.verdict else "NO-MATCH"
    return (
        "\n"
        + "-" * 60
        + "\n"
        + f"Pair {index}/{total}  |  reason: {item.reason}"
        + f"  |  score: {item.score:.3f}  |  judge: {verdict}\n"
        + f"  left  [{_sanitize(item.left_id)}]:  {_render_record(item.left_record)}\n"
        + f"  right [{_sanitize(item.right_id)}]:  {_render_record(item.right_record)}\n"
        + "-" * 60
        + "\n"
    )


def _render_record(record: dict[str, Any] | None) -> str:
    """A record's fields as a sanitized ``k=v`` line, or an ids-only fallback."""
    if not record:
        return "(id only -- no record content joined)"
    return "  ".join(f"{_sanitize(str(k))}={_sanitize(str(value))}" for k, value in record.items())


def _sanitize(text: str) -> str:
    """Strip ANSI escape sequences and C0/C1/DEL control characters."""
    return _CONTROL.sub("", _ANSI_ESCAPE.sub("", text))


def _export_csv(queue_path: Path, out_path: Path, out_stream: TextIO) -> int:
    """Write ``queue_path`` as a labelable CSV; formula-escape display cells only."""
    if not queue_path.exists():
        out_stream.write(f"error: review queue not found: {queue_path}\n")
        return 1

    items = ReviewQueue(queue_path).read()
    left_keys = _collect_keys(item.left_record for item in items)
    right_keys = _collect_keys(item.right_record for item in items)
    header = (
        ["left_id", "right_id"]
        + [f"left_{key}" for key in left_keys]
        + [f"right_{key}" for key in right_keys]
        + ["score", "verdict", "reason", "label"]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for item in items:
            row = [item.left_id, item.right_id]  # id columns: never escaped
            row += [_escape_formula(_cell(item.left_record, key)) for key in left_keys]
            row += [_escape_formula(_cell(item.right_record, key)) for key in right_keys]
            row += [
                _escape_formula(str(item.score)),
                _escape_formula(str(item.verdict).lower()),
                _escape_formula(item.reason),
                "",  # label: left blank for the reviewer to fill
            ]
            writer.writerow(row)

    out_stream.write(
        f"Wrote {len(items)} pair(s) to {out_path}. Fill the 'label' column (y/n), then: "
        f"uv run langres import-csv {out_path} {queue_path}\n"
    )
    return 0


def _import_csv(
    csv_path: Path,
    queue_path: Path,
    out_path: Path,
    reviewer: str | None,
    out_stream: TextIO,
) -> int:
    """Read a labeled CSV back into a corrections log; abort (write nothing) on any bad row."""
    if not csv_path.exists():
        out_stream.write(f"error: input CSV not found: {csv_path}\n")
        return 1
    if not queue_path.exists():
        out_stream.write(f"error: review queue not found: {queue_path}\n")
        return 1

    queue_items = {
        _pair_key(item.left_id, item.right_id): item for item in ReviewQueue(queue_path).read()
    }

    corrections: list[Correction] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = [column for column in ("left_id", "right_id", "label") if column not in fields]
        if missing:
            out_stream.write(
                f"error: CSV is missing required column(s): {', '.join(missing)}. "
                "Re-export it with 'langres export-csv'.\n"
            )
            return 1

        for rownum, raw_row in enumerate(reader, start=2):  # row 1 is the header
            label_token = str(raw_row.get("label") or "").strip()
            if not label_token:
                continue  # blank label = the reviewer skipped this pair
            label = _parse_label(label_token)
            if label is None:
                out_stream.write(
                    f"error: row {rownum}: unrecognized label {label_token!r} "
                    "(use y/yes/true/1 or n/no/false/0, or leave blank to skip). "
                    "No corrections were written.\n"
                )
                return 1
            left_id = str(raw_row.get("left_id") or "").strip()
            right_id = str(raw_row.get("right_id") or "").strip()
            item = queue_items.get(_pair_key(left_id, right_id))
            if item is None:
                out_stream.write(
                    f"error: row {rownum}: pair ({left_id!r}, {right_id!r}) is not in the "
                    f"review queue {queue_path}. A stray row must not corrupt the "
                    "correction log; no corrections were written.\n"
                )
                return 1
            corrections.append(
                Correction(
                    left_id=left_id,
                    right_id=right_id,
                    label=label,
                    reviewer=reviewer,
                    original_score=item.score,
                    original_verdict=item.verdict,
                )
            )

    log = CorrectionLog(out_path)
    for correction in corrections:
        log.append(correction)
    out_stream.write(f"Imported {len(corrections)} correction(s) into {out_path}.\n")
    return 0


def _collect_keys(records: Any) -> list[str]:
    """First-seen-ordered union of the keys across a series of (optional) records."""
    keys: list[str] = []
    for record in records:
        if record:
            for key in record:
                if key not in keys:
                    keys.append(key)
    return keys


def _cell(record: dict[str, Any] | None, key: str) -> str:
    """A record's value for ``key`` as a string, or empty when absent."""
    if record is None or key not in record:
        return ""
    return str(record[key])


def _escape_formula(value: str) -> str:
    """Neutralize a spreadsheet-formula-leading cell by prefixing ``'`` (display cells only)."""
    if value and value[0] in _FORMULA_LEADERS:
        return "'" + value
    return value


def _parse_label(token: str) -> bool | None:
    """Parse a label token to a bool, or ``None`` if it is unrecognized."""
    normalized = token.strip().lower()
    if normalized in _TRUE_TOKENS:
        return True
    if normalized in _FALSE_TOKENS:
        return False
    return None


def _pair_key(left_id: str, right_id: str) -> frozenset[str]:
    """Order-independent pair key (matches JudgementLog / Correction id conventions)."""
    return frozenset({left_id, right_id})


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
