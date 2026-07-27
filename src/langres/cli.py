"""The ``langres`` command line: human review surfaces for the flywheel.

:func:`langres.curation.review.select_for_review` writes a ``review_queue.jsonl``
snapshot of the judged pairs worth a human's attention; this module is where a
human answers them. Two review surfaces, one contract
(:class:`~langres.curation.harvest.Correction`):

- **The primary path is the CSV round-trip.** ``export-csv`` turns a queue into
  a labelable spreadsheet (``left_*``/``right_*`` display columns + an empty
  ``label`` column); a reviewer fills the ``label`` column in Excel/Sheets;
  ``import-csv`` reads that spreadsheet back into a ``corrections.jsonl`` log.
- ``review`` is the quick terminal loop -- a ``y/n/s/q`` prompt per pair for a
  developer who would rather stay in the shell. Each answer is appended to the
  corrections log *immediately*, so quitting (or ctrl-D) never loses answered
  work and a re-run resumes where it left off.

``info`` is the odd one out: an install diagnostic, not a review surface. It
answers the first question a new user hits -- *which extras do I actually have,
and which benchmark datasets can I load?* -- because a `pip install langres`
gets neither every extra nor most of the benchmark corpora (they are excluded
from the wheel; see ``[tool.hatch.build]``).

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
import importlib.util
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from langres._version import __version__
from langres.curation.harvest import Correction, CorrectionLog
from langres.curation.review import ReviewItem, ReviewQueue

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

#: Each optional extra with the import(s) that PROVE it -- the modules langres
#: itself imports from that stack (verified against `src/`, not assumed from the
#: dependency list). Probed with :func:`importlib.util.find_spec`, never
#: ``try: import``: importing torch to discover whether torch is installed costs
#: seconds and drags the heavy stack into the one command whose job is to work
#: on a bare install.
#:
#: ``bitsandbytes`` (``[finetune]``) is deliberately absent. It is Linux-only in
#: the extra and langres never imports it by name -- its config arrives via
#: ``transformers.BitsAndBytesConfig`` -- so probing for it would report a false
#: "no" on every macOS install that has the extra correctly installed.
_EXTRA_PROOF_IMPORTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # fastembed backs the sparse/late-interaction embedders in
    # core/embeddings.py, so omitting it would report "[semantic] yes" on an
    # install where those fail -- the mirror of the bitsandbytes false "no".
    ("semantic", ("faiss", "fastembed", "qdrant_client", "sentence_transformers", "torch")),
    ("llm", ("dspy", "litellm", "openai")),
    ("trained", ("sklearn",)),
    ("eval", ("ranx",)),
    ("finetune", ("peft", "trl")),
    ("hub", ("huggingface_hub",)),
)

#: Declared extras `langres info` deliberately does NOT report, each with its
#: reason. Paired with a test that reads `[project.optional-dependencies]` from
#: pyproject: a new extra fails that test until it is either given proof imports
#: or listed here on purpose. Without it the table's only "expectation" was the
#: table itself -- an expectation regenerated from the thing that breaks cannot
#: detect it breaking.
_EXTRAS_NOT_REPORTED: dict[str, str] = {
    "mlflow": "opt-in experiment-tracking backend, not part of the resolve path",
    "wandb": "opt-in experiment-tracking backend, not part of the resolve path",
    "trackio": "opt-in experiment-tracking backend, not part of the resolve path",
}

#: Credentials that can make an inference call cost money, as
#: ``(env var, Settings field)``. Reported as a **boolean only** -- the value is
#: tested for truthiness and discarded, because `langres info` must never put a
#: secret on a terminal, in a CI log, or in a pasted bug report. The two halves
#: are paired here rather than in two hand-synced lists, where adding a key to
#: one and not the other would raise a KeyError inside the one command that must
#: never crash. Reporting a *subset* is worse than useless: an Azure user with a
#: working key read "not set" on every line and would conclude they had no paid
#: path configured.
_PAID_PATH_KEYS: tuple[tuple[str, str], ...] = (
    ("OPENROUTER_API_KEY", "openrouter_api_key"),
    ("OPENAI_API_KEY", "openai_api_key"),
    ("AZURE_API_KEY", "azure_api_key"),
)

#: Matches `NAME=` / `export NAME=` at the head of a dotenv line. Deliberately
#: captures only the NAME: the value is never read, so it can never be printed.
_DOTENV_NAME_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def _dotenv_key_names(path: Path = Path(".env")) -> set[str]:
    """`*_API_KEY` names declared in ./.env, without reading a single value.

    A diagnostic must not die on the broken `.env` it is being run to diagnose,
    so any read failure yields nothing rather than raising.
    """
    try:
        # utf-8-sig, not utf-8: a `.env` saved by a Windows editor starts with a
        # BOM, and under plain utf-8 that BOM sits before the first name so the
        # line does not match -- hiding the FIRST key in the file. Measured. That
        # is the expensive direction of wrong for this scan: reporting "no key"
        # while litellm loads the same file and spends is the exact bug it
        # exists to prevent. utf-8-sig is identical to utf-8 when no BOM present.
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return set()
    names = {m.group(1) for line in text.splitlines() if (m := _DOTENV_NAME_RE.match(line))}
    return {name for name in names if name.endswith("_API_KEY")}


#: `Settings` credentials deliberately absent from the paid-path report, with
#: the reason. These buy a *service*, not inference tokens, so listing them
#: under "paid path" would blur the one question that section answers: can a
#: `dedupe()` call spend money? A new key on `Settings` must land in one of
#: these two tables -- `tests/test_cli_info.py` fails until it does. Their
#: uppercase forms are also excluded from the "other provider key(s)" scan, so
#: a project that merely tracks experiments does not read as having a paid path.
_KEYS_NOT_PAID_PATH: dict[str, str] = {
    "wandb_api_key": "experiment-tracking backend, bills no inference",
    "langfuse_public_key": "tracing backend, bills no inference",
    "langfuse_secret_key": "tracing backend, bills no inference",
    "qdrant_api_key": "vector-store service, bills no inference",
}


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
    if args.command == "info":
        return _info(out_stream)
    if args.command == "review":
        return _review(Path(args.queue), Path(args.out), args.reviewer, in_stream, out_stream)
    if args.command == "export-csv":
        return _export_csv(Path(args.queue), Path(args.out_csv), out_stream)
    if args.command == "import-csv":
        return _import_csv(
            Path(args.in_csv), Path(args.queue), Path(args.out), args.reviewer, out_stream
        )
    if args.command == "experiments" and args.experiments_command == "reproduce":
        from langres.experiments.reproduction import reconstruct_reproduction_bundle

        reconstruct_reproduction_bundle(Path(args.artifact), output=out_stream)
        return 0
    if args.command == "experiments" and args.experiments_command == "verify":
        from langres.experiments.reproduction import verify_reproduction_bundle

        verify_reproduction_bundle(Path(args.artifact), output=out_stream)
        return 0

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
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="{info,review,export-csv,import-csv,experiments}",
    )

    subparsers.add_parser(
        "info",
        help="Report what this install actually has: extras, datasets, credentials.",
        description=(
            "Print the installed langres version, which optional extras resolve (with "
            "the import that proves each), which benchmark datasets this install can "
            "actually load, and whether a paid-path API key is configured. Reports key "
            "presence as yes/no only -- it never prints a key. Runs on a bare core "
            "install and imports no heavy dependency to decide whether one exists."
        ),
    )

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

    p_experiments = subparsers.add_parser(
        "experiments",
        help="Inspect and reproduce saved experiment handoff bundles.",
    )
    experiment_commands = p_experiments.add_subparsers(
        dest="experiments_command",
        metavar="{reproduce,verify}",
    )
    p_reproduce = experiment_commands.add_parser(
        "reproduce",
        help="Reconstruct saved local model artifacts and verify their plans.",
    )
    p_reproduce.add_argument(
        "artifact",
        metavar="reproduction.json",
        help="The reproduction artifact emitted by Experiment.run().",
    )
    p_verify = experiment_commands.add_parser(
        "verify",
        help="Validate a saved experiment bundle without reconstructing models.",
    )
    p_verify.add_argument(
        "artifact",
        metavar="reproduction.json",
        help="The reproduction artifact emitted by Experiment.run().",
    )
    return parser


def _module_installed(name: str) -> bool:
    """Is ``name`` importable, *without* importing it?

    :func:`importlib.util.find_spec` locates a module's spec on the path; it does
    not execute the module. That distinction is the whole point here -- a
    ``try: import torch`` probe would take seconds and pull the heavy stack into
    ``sys.modules``, which ``tests/test_import_budget.py`` exists to prevent.
    """
    try:
        spec = importlib.util.find_spec(name)
        # `spec.loader is None` means a namespace package -- a bare directory
        # with no module in it, which a partial uninstall leaves behind.
        # Measured: an empty `torchzzz/` on sys.path makes find_spec return a
        # spec, so `is not None` alone would report an extra as installed on an
        # environment where importing it raises.
        return spec is not None and spec.loader is not None
    except (ImportError, ValueError):
        # ImportError: a parent package is absent, so the child cannot exist.
        # ValueError: the module is present but has no spec (__spec__ is None).
        return False


def _benchmark_status(name: str) -> str:
    """Actually load benchmark ``name``; return ``""`` on success or a short reason.

    Deliberately behavioral, not a manifest lookup. Whether a dataset is usable
    here depends on facts no static list tracks -- the wheel excludes most of the
    corpora, and some loaders need an extra -- and a diagnostic that consults a
    hand-maintained copy of that answer reports the copy, not the install.
    Loading every bundled dataset costs about a second in a git checkout, and far
    less in a wheel install where most fail immediately.
    """
    from langres.data.registry import get_benchmark

    try:
        get_benchmark(name).load()
    except Exception as exc:
        # Broad on purpose: this is a diagnostic. Any failure to load is the
        # answer the user asked for, so no loader error may abort the report.
        return type(exc).__name__
    return ""


def _info(out_stream: TextIO) -> int:
    """Print what this install actually has: version, extras, datasets, credentials."""
    from langres.clients.settings import Settings
    from langres.data.registry import list_benchmarks

    out_stream.write(f"langres {__version__}\n")
    out_stream.write(f"python  {sys.version.split()[0]}  ({sys.executable})\n")

    out_stream.write("\nExtras\n")
    for extra, proofs in _EXTRA_PROOF_IMPORTS:
        missing = [module for module in proofs if not _module_installed(module)]
        label = f"[{extra}]"
        if missing:
            out_stream.write(
                f"  {label:<11} no   missing {', '.join(missing)}"
                f"  ->  pip install 'langres[{extra}]'\n"
            )
        else:
            out_stream.write(f"  {label:<11} yes  {', '.join(proofs)}\n")

    entries = list_benchmarks()
    statuses = {entry.name: _benchmark_status(entry.name) for entry in entries}
    loadable = sum(1 for reason in statuses.values() if not reason)
    out_stream.write(f"\nBenchmark datasets ({loadable}/{len(entries)} loadable here)\n")
    for entry in entries:
        reason = statuses[entry.name]
        verdict = "no   " + reason if reason else "yes"
        out_stream.write(f"  {entry.name:<16} {verdict}\n")
    if "BenchmarkDataNotFoundError" in statuses.values():
        out_stream.write(
            "  BenchmarkDataNotFoundError = the corpus is excluded from the PyPI wheel\n"
            "  (no redistribution license); install from a git checkout to get it.\n"
        )

    out_stream.write("\nPaid path (presence only -- no key is ever printed)\n")
    try:
        # Presence only: each value is coerced to bool here and never written.
        settings = Settings()
    except Exception as exc:
        # A diagnostic must not die on the broken `.env` it is being run to
        # diagnose. Measured: a binary `.env` raises UnicodeDecodeError and a
        # mode-000 `.env` raises PermissionError, both after the sections above
        # had already printed -- a half report plus a traceback. Only the
        # exception *type* is surfaced: a pydantic ValidationError's str embeds
        # the offending input value, which could be a key.
        out_stream.write(f"  could not read settings ({type(exc).__name__}) -- check ./.env\n")
        return 0
    for key, field in _PAID_PATH_KEYS:
        present = bool(getattr(settings, field))
        out_stream.write(f"  {key:<20} {'set' if present else 'not set'}\n")

    # langres declares three credentials, but a served model is routed by
    # litellm, which knows ~146 providers -- so `anthropic/...` bills with
    # ANTHROPIC_API_KEY, a variable `Settings` never mentions. Listing only the
    # three would reproduce the Azure bug one provider over: every line reads
    # "not set" while a real call spends. Enumerating 146 providers would be
    # noise and would rot, so report what is actually in the environment. Names
    # only -- never a value.
    # `./.env` is scanned too, not just os.environ. A key that lives ONLY in the
    # dotenv and was never exported is invisible to both `os.environ` and
    # `Settings` (which ignores undeclared fields) -- while litellm's own
    # load_dotenv() picks it up and spends it. Reporting nothing in that case
    # would also contradict the footer's claim that this section reads ./.env.
    already_named = {key for key, _ in _PAID_PATH_KEYS}
    # Credentials `_KEYS_NOT_PAID_PATH` classifies as non-billing. Listing
    # WANDB_API_KEY under "Paid path" would raise a false alarm in any project
    # that merely tracks experiments.
    not_paid = {field.upper() for field in _KEYS_NOT_PAID_PATH}
    others = {
        name
        for name, value in os.environ.items()
        if name.endswith("_API_KEY") and value and name not in already_named | not_paid
    }
    others |= {name for name in _dotenv_key_names() if name not in already_named | not_paid}
    if others:
        out_stream.write(f"  other provider key(s) present: {', '.join(sorted(others))}\n")

    out_stream.write(
        "  Read from the environment and from ./.env in the CURRENT directory.\n"
        "  A real LLM call can additionally pick up a .env in a PARENT directory\n"
        "  (litellm runs load_dotenv() on import), so 'not set' here does not\n"
        "  guarantee a paid call would find no key. Any litellm provider key in\n"
        "  the environment can bill, not only the ones named above.\n"
    )
    return 0


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
    # verdict can be None for a score-only row that recorded no verdict/decision;
    # render "?" rather than silently showing an unknown as NO-MATCH.
    verdict = {True: "MATCH", False: "NO-MATCH", None: "?"}[item.verdict]
    # A decider (binary judge) has no score -- render "n/a" rather than crash on
    # None. Surface its credence instead, which is the signal that queued it.
    signal = "n/a" if item.score is None else f"{item.score:.3f}"
    if item.confidence is not None:
        signal += f"  |  confidence: {item.confidence:.3f}"
    return (
        "\n"
        + "-" * 60
        + "\n"
        + f"Pair {index}/{total}  |  reason: {item.reason}"
        + f"  |  score: {signal}  |  judge: {verdict}\n"
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
                # A decider has no score: emit an empty cell, not the literal "None".
                _escape_formula("" if item.score is None else str(item.score)),
                # verdict None (a score-only row) -> empty cell, not the string "none".
                _escape_formula("" if item.verdict is None else str(item.verdict).lower()),
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
