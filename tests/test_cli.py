"""Tests for the ``langres`` CLI (review / export-csv / import-csv).

Every path is driven through :func:`langres.cli.main` with injected
``StringIO`` streams -- no TTY, no subprocess, zero spend. The suite covers the
interactive loop (label / skip / quit / EOF-quit / resume / re-prompt), the CSV
round-trip that feeds :func:`langres.core.harvest.harvest_labeled_pairs`,
command dispatch and exit codes, ``--version``, the unknown-pair and
invalid-label aborts, spreadsheet formula-escaping, the adversarial-id
round-trip (ids are never escaped), unicode round-tripping, and the terminal
control-character sanitization.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from langres import __version__
from langres.cli import main
from langres.core.harvest import Correction, CorrectionLog, harvest_labeled_pairs
from langres.core.review import ReviewItem, ReviewQueue


def _run(argv: list[str], *, stdin: str = "") -> tuple[int, str]:
    """Run ``main`` with StringIO streams; return ``(rc, captured_output)``."""
    out = io.StringIO()
    rc = main(argv, input_stream=io.StringIO(stdin), output_stream=out)
    return rc, out.getvalue()


def _item(
    left_id: str,
    right_id: str,
    *,
    score: float = 0.5,
    verdict: bool = True,
    reason: str = "uncertainty",
    left_record: dict[str, Any] | None = None,
    right_record: dict[str, Any] | None = None,
) -> ReviewItem:
    return ReviewItem(
        left_id=left_id,
        right_id=right_id,
        score=score,
        verdict=verdict,
        reason=reason,  # type: ignore[arg-type]
        left_record=left_record,
        right_record=right_record,
    )


def _write_queue(path: Path, items: list[ReviewItem]) -> Path:
    ReviewQueue(path).write(items)
    return path


# --------------------------------------------------------------------------- #
# Dispatch, --version, no-command                                             #
# --------------------------------------------------------------------------- #


def test_version_flag_prints_version() -> None:
    rc, out = _run(["--version"])
    assert rc == 0
    assert __version__ in out
    assert "langres" in out


def test_no_command_prints_help() -> None:
    rc, out = _run([])
    assert rc == 0
    # CSV round-trip framed as the primary path (gate UC2).
    assert "export-csv" in out
    assert "import-csv" in out
    assert "review" in out


def test_help_frames_csv_as_primary_path() -> None:
    rc, out = _run([])
    assert "primary" in out.lower()
    # Docs/examples use the `uv run langres ...` form.
    assert "uv run langres" in out


# --------------------------------------------------------------------------- #
# review                                                                       #
# --------------------------------------------------------------------------- #


def test_review_empty_queue_is_a_friendly_stop(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path / "q.jsonl", [])
    rc, out = _run(["review", str(queue)])
    assert rc == 0
    assert "empty" in out.lower()


def test_review_missing_queue_errors_instead_of_reading_as_empty(tmp_path: Path) -> None:
    """A missing/typo'd --queue path must not silently look like an empty queue."""
    rc, out = _run(["review", str(tmp_path / "nope.jsonl")])
    assert rc == 1
    assert "not found" in out
    assert "empty" not in out.lower()


def test_review_labels_yes_and_no_appends_corrections(tmp_path: Path) -> None:
    queue = _write_queue(
        tmp_path / "q.jsonl",
        [_item("a", "b", verdict=True), _item("c", "d", verdict=False)],
    )
    out_path = tmp_path / "corrections.jsonl"
    rc, out = _run(
        ["review", str(queue), "--out", str(out_path), "--reviewer", "alice"],
        stdin="y\nn\n",
    )
    assert rc == 0
    corrections = CorrectionLog(out_path).read()
    assert [(c.left_id, c.right_id, c.label) for c in corrections] == [
        ("a", "b", True),
        ("c", "d", False),
    ]
    # Audit context is carried onto the correction.
    assert corrections[0].reviewer == "alice"
    assert corrections[0].original_verdict is True
    assert "Done" in out


def test_review_skip_writes_no_correction_for_that_pair(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path / "q.jsonl", [_item("a", "b"), _item("c", "d")])
    out_path = tmp_path / "corrections.jsonl"
    rc, _ = _run(["review", str(queue), "--out", str(out_path)], stdin="s\ny\n")
    assert rc == 0
    corrections = CorrectionLog(out_path).read()
    assert [(c.left_id, c.right_id) for c in corrections] == [("c", "d")]


def test_review_long_form_answers_are_accepted(tmp_path: Path) -> None:
    queue = _write_queue(
        tmp_path / "q.jsonl",
        [_item("a", "b"), _item("c", "d"), _item("e", "f")],
    )
    out_path = tmp_path / "corrections.jsonl"
    rc, _ = _run(["review", str(queue), "--out", str(out_path)], stdin="yes\nno\nskip\n")
    assert rc == 0
    corrections = CorrectionLog(out_path).read()
    assert [(c.left_id, c.right_id, c.label) for c in corrections] == [
        ("a", "b", True),
        ("c", "d", False),
    ]


def test_review_quit_stops_but_keeps_earlier_answers(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path / "q.jsonl", [_item("a", "b"), _item("c", "d")])
    out_path = tmp_path / "corrections.jsonl"
    rc, out = _run(["review", str(queue), "--out", str(out_path)], stdin="y\nq\n")
    assert rc == 0
    assert "Stopped" in out and "resume" in out
    corrections = CorrectionLog(out_path).read()
    assert [(c.left_id, c.right_id) for c in corrections] == [("a", "b")]


def test_review_eof_quits_without_losing_answered_work(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path / "q.jsonl", [_item("a", "b"), _item("c", "d")])
    out_path = tmp_path / "corrections.jsonl"
    # Answer the first pair, then ctrl-D (EOF) at the second prompt.
    rc, out = _run(["review", str(queue), "--out", str(out_path)], stdin="y\n")
    assert rc == 0
    assert "Stopped" in out
    corrections = CorrectionLog(out_path).read()
    assert [(c.left_id, c.right_id) for c in corrections] == [("a", "b")]


def test_review_reprompts_on_invalid_answer(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path / "q.jsonl", [_item("a", "b")])
    out_path = tmp_path / "corrections.jsonl"
    rc, out = _run(["review", str(queue), "--out", str(out_path)], stdin="maybe\n\ny\n")
    assert rc == 0
    assert "Please answer" in out
    assert [(c.left_id, c.right_id) for c in CorrectionLog(out_path).read()] == [("a", "b")]


def test_review_resumes_skipping_already_answered_pairs(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path / "q.jsonl", [_item("a", "b"), _item("c", "d")])
    out_path = tmp_path / "corrections.jsonl"
    # First session answers the first pair, then quits.
    _run(["review", str(queue), "--out", str(out_path)], stdin="y\nq\n")
    # Second session: the first pair is already answered, so it jumps to ("c","d").
    rc, out = _run(["review", str(queue), "--out", str(out_path)], stdin="n\n")
    assert rc == 0
    corrections = CorrectionLog(out_path).read()
    assert [(c.left_id, c.right_id, c.label) for c in corrections] == [
        ("a", "b", True),
        ("c", "d", False),
    ]


def test_review_renders_record_content_when_present(tmp_path: Path) -> None:
    queue = _write_queue(
        tmp_path / "q.jsonl",
        [_item("a", "b", left_record={"name": "Acme"}, right_record={"name": "ACME Inc"})],
    )
    rc, out = _run(["review", str(queue)], stdin="q\n")
    assert rc == 0
    assert "name=Acme" in out
    assert "name=ACME Inc" in out


def test_review_falls_back_to_ids_when_no_record(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path / "q.jsonl", [_item("a", "b")])
    rc, out = _run(["review", str(queue)], stdin="q\n")
    assert rc == 0
    assert "[a]" in out and "[b]" in out
    assert "id only" in out


def test_review_strips_ansi_and_control_chars_from_records(tmp_path: Path) -> None:
    # A crafted field tries to color/clear the terminal and ring the bell.
    hostile = "\x1b[31mRED\x1b[2J\x1b[0m evil\x07\x00 text"
    queue = _write_queue(
        tmp_path / "q.jsonl",
        [_item("a", "b", left_record={"name": hostile})],
    )
    rc, out = _run(["review", str(queue)], stdin="q\n")
    assert rc == 0
    assert "\x1b" not in out  # no escape sequences survive
    assert "\x07" not in out and "\x00" not in out
    assert "RED evil text" in out  # printable text is preserved


def test_review_strips_ansi_escapes_from_ids(tmp_path: Path) -> None:
    """A hostile natural-key id must not inject terminal escapes either."""
    hostile_id = "\x1b[31mleft-id\x1b[0m"
    queue = _write_queue(tmp_path / "q.jsonl", [_item(hostile_id, "b")])
    rc, out = _run(["review", str(queue)], stdin="q\n")
    assert rc == 0
    assert "\x1b" not in out
    assert "left-id" in out


def test_review_renders_a_decider_without_a_score(tmp_path: Path) -> None:
    """A binary judge's queued item has score=None: render 'n/a' + its confidence, never crash."""
    decider = ReviewItem(
        left_id="a",
        right_id="b",
        score=None,  # a decider carries a decision, not a score
        verdict=True,
        confidence=0.55,
        reason="uncertainty",
    )
    queue = _write_queue(tmp_path / "q.jsonl", [decider])
    rc, out = _run(["review", str(queue)], stdin="q\n")
    assert rc == 0
    assert "score: n/a" in out  # not a crash, not "None"
    assert "confidence: 0.550" in out  # the signal that actually queued it


def test_export_csv_emits_empty_score_for_a_decider(tmp_path: Path) -> None:
    """A score-less (decider) item exports an empty score cell, not the literal 'None'."""
    decider = ReviewItem(
        left_id="a", right_id="b", score=None, verdict=True, reason="uncertainty"
    )
    queue = _write_queue(tmp_path / "q.jsonl", [decider])
    csv_path = tmp_path / "out.csv"
    rc, _ = _run(["export-csv", str(queue), str(csv_path)])
    assert rc == 0
    rows = list(_read_csv(csv_path))
    first = dict(zip(rows[0], rows[1]))
    assert first["score"] == ""  # empty cell, never "None"


# --------------------------------------------------------------------------- #
# export-csv                                                                   #
# --------------------------------------------------------------------------- #


def test_export_csv_missing_queue_errors(tmp_path: Path) -> None:
    rc, out = _run(["export-csv", str(tmp_path / "nope.jsonl"), str(tmp_path / "o.csv")])
    assert rc == 1
    assert "not found" in out


def test_export_csv_writes_display_columns_and_empty_label(tmp_path: Path) -> None:
    queue = _write_queue(
        tmp_path / "q.jsonl",
        [
            _item(
                "a",
                "b",
                score=0.62,
                left_record={"name": "Acme"},
                right_record={"name": "ACME Inc"},
            ),
            _item("c", "d"),  # ids-only item (exercises the empty-record branch)
        ],
    )
    csv_path = tmp_path / "out.csv"
    rc, out = _run(["export-csv", str(queue), str(csv_path)])
    assert rc == 0
    assert "Wrote 2 pair(s)" in out
    rows = list(_read_csv(csv_path))
    header = rows[0]
    assert header == [
        "left_id",
        "right_id",
        "left_name",
        "right_name",
        "score",
        "verdict",
        "reason",
        "label",
    ]
    first = dict(zip(header, rows[1]))
    assert first["left_id"] == "a"
    assert first["left_name"] == "Acme"
    assert first["right_name"] == "ACME Inc"
    assert first["label"] == ""  # empty for the human to fill


def test_export_csv_escapes_formula_leading_display_cells_only(tmp_path: Path) -> None:
    queue = _write_queue(
        tmp_path / "q.jsonl",
        [_item("-42", "+7", left_record={"name": "=cmd()"}, right_record={"name": "@SUM(A1)"})],
    )
    csv_path = tmp_path / "out.csv"
    rc, _ = _run(["export-csv", str(queue), str(csv_path)])
    assert rc == 0
    rows = list(_read_csv(csv_path))
    row = dict(zip(rows[0], rows[1]))
    # Display cells with a formula leader are neutralized with a leading quote.
    assert row["left_name"] == "'=cmd()"
    assert row["right_name"] == "'@SUM(A1)"
    # Id columns are NEVER escaped -- escaping them would break import round-trip.
    assert row["left_id"] == "-42"
    assert row["right_id"] == "+7"


# --------------------------------------------------------------------------- #
# import-csv                                                                   #
# --------------------------------------------------------------------------- #


def test_import_csv_missing_input_errors(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path / "q.jsonl", [_item("a", "b")])
    rc, out = _run(["import-csv", str(tmp_path / "nope.csv"), str(queue)])
    assert rc == 1
    assert "input CSV not found" in out


def test_import_csv_missing_queue_errors(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "in.csv", ["left_id", "right_id", "label"], [["a", "b", "y"]])
    rc, out = _run(["import-csv", str(csv_path), str(tmp_path / "nope.jsonl")])
    assert rc == 1
    assert "review queue not found" in out


def test_import_csv_missing_required_columns_errors(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path / "q.jsonl", [_item("a", "b")])
    csv_path = _write_csv(tmp_path / "in.csv", ["left_id", "right_id"], [["a", "b"]])
    rc, out = _run(["import-csv", str(csv_path), str(queue)])
    assert rc == 1
    assert "missing required column" in out
    assert "label" in out


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("y", True),
        ("yes", True),
        ("true", True),
        ("1", True),
        ("n", False),
        ("no", False),
        ("false", False),
        ("0", False),
    ],
)
def test_import_csv_accepts_label_tokens(tmp_path: Path, token: str, expected: bool) -> None:
    queue = _write_queue(tmp_path / "q.jsonl", [_item("a", "b")])
    out_path = tmp_path / "corrections.jsonl"
    csv_path = _write_csv(
        tmp_path / "in.csv", ["left_id", "right_id", "label"], [["a", "b", token]]
    )
    rc, out = _run(["import-csv", str(csv_path), str(queue), "--out", str(out_path)])
    assert rc == 0
    assert "Imported 1 correction(s)" in out
    corrections = CorrectionLog(out_path).read()
    assert [(c.left_id, c.right_id, c.label) for c in corrections] == [("a", "b", expected)]


def test_import_csv_strips_leading_bom(tmp_path: Path) -> None:
    """Excel/Google Sheets 'CSV UTF-8' export prepends a BOM to the first header;
    reading it as plain utf-8 would leave it glued to 'left_id' and trip the
    missing-required-column check on a column that is visibly present."""
    queue = _write_queue(tmp_path / "q.jsonl", [_item("a", "b")])
    out_path = tmp_path / "corrections.jsonl"
    csv_path = tmp_path / "in.csv"
    csv_path.write_text("﻿left_id,right_id,label\na,b,y\n", encoding="utf-8")
    rc, out = _run(["import-csv", str(csv_path), str(queue), "--out", str(out_path)])
    assert rc == 0
    assert "missing required column" not in out
    corrections = CorrectionLog(out_path).read()
    assert [(c.left_id, c.right_id, c.label) for c in corrections] == [("a", "b", True)]


def test_import_csv_reviewer_flag_is_recorded_on_corrections(tmp_path: Path) -> None:
    """--reviewer attributes CSV-sourced corrections (previously always None)."""
    queue = _write_queue(tmp_path / "q.jsonl", [_item("a", "b"), _item("c", "d")])
    out_path = tmp_path / "corrections.jsonl"
    csv_path = _write_csv(
        tmp_path / "in.csv",
        ["left_id", "right_id", "label"],
        [["a", "b", "y"], ["c", "d", "n"]],
    )
    rc, out = _run(
        ["import-csv", str(csv_path), str(queue), "--out", str(out_path), "--reviewer", "alice"]
    )
    assert rc == 0
    corrections = CorrectionLog(out_path).read()
    assert len(corrections) == 2
    assert all(c.reviewer == "alice" for c in corrections)


def test_import_csv_blank_label_is_skipped(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path / "q.jsonl", [_item("a", "b"), _item("c", "d")])
    out_path = tmp_path / "corrections.jsonl"
    csv_path = _write_csv(
        tmp_path / "in.csv",
        ["left_id", "right_id", "label"],
        [["a", "b", "y"], ["c", "d", "   "]],  # second row left blank
    )
    rc, _ = _run(["import-csv", str(csv_path), str(queue), "--out", str(out_path)])
    assert rc == 0
    corrections = CorrectionLog(out_path).read()
    assert [(c.left_id, c.right_id) for c in corrections] == [("a", "b")]


def test_import_csv_invalid_token_aborts_with_row_number_and_writes_nothing(
    tmp_path: Path,
) -> None:
    queue = _write_queue(tmp_path / "q.jsonl", [_item("a", "b"), _item("c", "d")])
    out_path = tmp_path / "corrections.jsonl"
    csv_path = _write_csv(
        tmp_path / "in.csv",
        ["left_id", "right_id", "label"],
        [["a", "b", "y"], ["c", "d", "maybe"]],  # row 3 in the file
    )
    rc, out = _run(["import-csv", str(csv_path), str(queue), "--out", str(out_path)])
    assert rc == 1
    assert "row 3" in out
    assert "maybe" in out
    # Transactional: not even the valid first row was written.
    assert not out_path.exists()


def test_import_csv_unknown_pair_aborts_with_row_number_and_writes_nothing(
    tmp_path: Path,
) -> None:
    queue = _write_queue(tmp_path / "q.jsonl", [_item("a", "b")])
    out_path = tmp_path / "corrections.jsonl"
    csv_path = _write_csv(
        tmp_path / "in.csv",
        ["left_id", "right_id", "label"],
        [["a", "b", "y"], ["x", "y", "n"]],  # ("x","y") is not in the queue -> row 3
    )
    rc, out = _run(["import-csv", str(csv_path), str(queue), "--out", str(out_path)])
    assert rc == 1
    assert "row 3" in out
    assert "not in the review queue" in out
    assert not out_path.exists()


# --------------------------------------------------------------------------- #
# Round-trips: the primary review path end to end                             #
# --------------------------------------------------------------------------- #


def test_csv_round_trip_feeds_harvest_labeled_pairs(tmp_path: Path) -> None:
    """export-csv -> (label) -> import-csv -> corrections that harvest can merge."""
    queue = _write_queue(
        tmp_path / "q.jsonl",
        [
            _item("a", "b", score=0.9, verdict=True, left_record={"name": "Acme"}),
            _item("c", "d", score=0.2, verdict=False, left_record={"name": "Globex"}),
        ],
    )
    csv_path = tmp_path / "to_label.csv"
    assert _run(["export-csv", str(queue), str(csv_path)])[0] == 0

    # A reviewer fills the label column (flip both verdicts to test overrides).
    _fill_labels(csv_path, {("a", "b"): "n", ("c", "d"): "y"})

    out_path = tmp_path / "corrections.jsonl"
    assert _run(["import-csv", str(csv_path), str(queue), "--out", str(out_path)])[0] == 0

    corrections = CorrectionLog(out_path).read()
    # The corrections merge with the original judgement log via harvest.
    judgement_rows = [
        {"left_id": "a", "right_id": "b", "score": 0.9, "verdict": True},
        {"left_id": "c", "right_id": "d", "score": 0.2, "verdict": False},
    ]
    labeled = harvest_labeled_pairs(judgement_rows, corrections)
    by_pair = {frozenset({p.left_id, p.right_id}): p for p in labeled}
    assert by_pair[frozenset({"a", "b"})].label is False  # human overrode True -> False
    assert by_pair[frozenset({"a", "b"})].source == "correction"
    assert by_pair[frozenset({"c", "d"})].label is True  # human overrode False -> True


def test_adversarial_id_round_trip_is_lossless(tmp_path: Path) -> None:
    """Formula-leading ids survive export->import because id columns are never escaped."""
    queue = _write_queue(
        tmp_path / "q.jsonl",
        [_item("-42", "+7", left_record={"name": "=danger"})],
    )
    csv_path = tmp_path / "to_label.csv"
    assert _run(["export-csv", str(queue), str(csv_path)])[0] == 0
    _fill_labels(csv_path, {("-42", "+7"): "y"})

    out_path = tmp_path / "corrections.jsonl"
    rc, out = _run(["import-csv", str(csv_path), str(queue), "--out", str(out_path)])
    # Had the ids been escaped to '-42 / '+7, this would have aborted as an unknown pair.
    assert rc == 0, out
    corrections = CorrectionLog(out_path).read()
    assert [(c.left_id, c.right_id, c.label) for c in corrections] == [("-42", "+7", True)]


def test_unicode_round_trips_through_csv(tmp_path: Path) -> None:
    queue = _write_queue(
        tmp_path / "q.jsonl",
        [
            _item(
                "id-zürich",
                "id-東京",
                left_record={"city": "Zürich"},
                right_record={"city": "東京"},
            )
        ],
    )
    csv_path = tmp_path / "to_label.csv"
    assert _run(["export-csv", str(queue), str(csv_path)])[0] == 0
    # Unicode content survives the export unchanged.
    assert "Zürich" in csv_path.read_text(encoding="utf-8")
    assert "東京" in csv_path.read_text(encoding="utf-8")

    _fill_labels(csv_path, {("id-zürich", "id-東京"): "y"})
    out_path = tmp_path / "corrections.jsonl"
    rc, _ = _run(["import-csv", str(csv_path), str(queue), "--out", str(out_path)])
    assert rc == 0
    corrections = CorrectionLog(out_path).read()
    assert [(c.left_id, c.right_id) for c in corrections] == [("id-zürich", "id-東京")]


# --------------------------------------------------------------------------- #
# CSV helpers                                                                  #
# --------------------------------------------------------------------------- #


def _read_csv(path: Path) -> list[list[str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _fill_labels(path: Path, labels: dict[tuple[str, str], str]) -> None:
    """Rewrite ``path``'s label column from a ``{(left_id, right_id): token}`` map."""
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else ["left_id", "right_id", "label"]
    for row in rows:
        token = labels.get((row["left_id"], row["right_id"]))
        if token is not None:
            row["label"] = token
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_review_queue_json_lines_are_valid(tmp_path: Path) -> None:
    """Sanity check the fixtures the CLI reads are real ReviewItem JSONL."""
    queue = _write_queue(tmp_path / "q.jsonl", [_item("a", "b")])
    lines = [json.loads(line) for line in queue.read_text().splitlines()]
    assert lines[0]["left_id"] == "a"


def test_correction_model_is_the_written_contract(tmp_path: Path) -> None:
    """The CLI writes exactly Correction lines (not an ad-hoc dict)."""
    queue = _write_queue(tmp_path / "q.jsonl", [_item("a", "b")])
    out_path = tmp_path / "corrections.jsonl"
    _run(["review", str(queue), "--out", str(out_path)], stdin="y\n")
    raw = json.loads(out_path.read_text().splitlines()[0])
    assert Correction.model_validate(raw).label is True
