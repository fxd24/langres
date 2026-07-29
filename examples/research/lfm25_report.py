"""Render the LFM2.5 write-up from the committed artifacts. No number is typed here.

Three PRs on this repo shipped factual errors in one day and **every one was in
hand-typed prose while the generated table beside it was correct**. So this file
contains prose and structure; every measured quantity is read from:

  - ``20260729_lfm25_tuned_rows.jsonl``          (study A: retrieval-tuned models)
  - ``20260729_lfm25_base_encoders_rows.jsonl``  (study B: base masked-LM encoders)
  - ``20260729_lfm25_load_probe.json``           (how the checkpoints load)
  - ``20260729_lfm25_license.txt``               (the LFM Open License v1.0 itself)

Run after both sweeps:

    uv run python examples/research/lfm25_report.py
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH = REPO_ROOT / "docs" / "research"

TUNED_ROWS = RESEARCH / "20260729_lfm25_tuned_rows.jsonl"
BASE_ROWS = RESEARCH / "20260729_lfm25_base_encoders_rows.jsonl"
LOAD_PROBE = RESEARCH / "20260729_lfm25_load_probe.json"
LICENSE = RESEARCH / "20260729_lfm25_license.txt"
OUTPUT = RESEARCH / "20260729_lfm25_encoders.md"

HEADLINE_K = 20
TUNED_BASELINE = "intfloat/e5-base-v2"
BASE_BASELINE = "LiquidAI/LFM2.5-Embedding-350M"

#: Imported from the portfolio stream, not measured here — a single-family sweep
#: cannot tell "every model agrees" from "the benchmark is solved".
SATURATED = {"fodors_zagat"}

#: Blob hashes of the code that produced the rows, captured at MEASUREMENT time
#: and committed. Exists because "were these numbers produced by the same code
#: that is in main?" was asked of this study while it was still running, and was
#: answerable in minutes only because the code was diffable.
PROVENANCE = RESEARCH / "20260729_lfm25_provenance.json"


def _provenance_section() -> list[str]:
    """Name the code that produced these numbers, read from the persisted sidecar.

    Emphatically **not** derived from ``HEAD`` at render time. That was the first
    implementation and it was wrong in the worst way: it named whatever was
    checked out *now*, so the very next commit touching the harness silently
    reattributed every measured row to code that never ran. Not a stale number —
    a false one, and the same shape as every other "gate decoupled from what it
    checks" bug in this repo. The sidecar is written from the measurement window
    and committed beside the rows.
    """
    lines = ["## Provenance — which code produced these numbers", ""]
    if not PROVENANCE.exists():
        lines.append(
            f"**Not recorded.** `{PROVENANCE.name}` is absent, so this run cannot say which "
            "code produced its rows. Deliberately left blank rather than filled in from the "
            "current checkout, which would assert something unverified."
        )
        return lines

    doc = json.loads(PROVENANCE.read_text())
    boot = doc["bootstrap"]
    window = doc["measurement_window"]
    lines += [
        f"The paired intervals were computed by `{boot['function']}` at "
        f"**B={boot['samples']}** replicates, resampling **{boot['resampled_unit']}s**. "
        f"Measured between `{window['started']}` and `{window['finished']}`, with "
        f"`HEAD` at `{window['head_when_sweep_started']}`.",
        "",
        "| file | blob | last commit touching it |",
        "|---|---|---|",
    ]
    for path, meta in doc["blobs"].items():
        lines.append(
            f"| `{path}` | `{meta['blob'][:12]}` | {meta['last_commit']} "
            f"{meta['last_commit_date']} |"
        )
    stable = doc.get("verified_unchanged_during_run")
    # The sidecar can say it was reconstructed AFTER the fact. Printing the
    # measurement-path sentence over such a sidecar would overstate the provenance
    # of exactly the rows it describes -- the document asserting a stronger
    # guarantee than the artifact it reads. (Cross-model review.)
    retrospective = any("CAPTURED RETROACTIVELY" in line for line in doc.get("_comment", []))
    lines += [
        "",
        "A **blob** hash, not a commit: it changes only when the file's *content* does, "
        "which is the property being recorded.",
    ]
    if retrospective:
        lines.append(
            "**Captured retrospectively.** This sweep finished before "
            "`examples/research/write_provenance.py` existed, so its `--start`/`--finish` "
            "hooks did not run over these rows. The hashes are derived from git at the "
            "commit that was `HEAD` when the sweep began, and the stability check below "
            "was reconstructed afterwards from `git log` over the measurement window "
            "rather than observed live. Later runs capture it on the measurement path; "
            "this one did not, and says so."
        )
    else:
        lines.append(
            "Written by `examples/research/write_provenance.py` as part of the "
            "**measurement** path — not derived from `HEAD` at render time, which would "
            "name whatever is checked out now instead of what measured the rows."
        )
    studies = doc.get("studies_measured")
    if stable is True:
        scope = (
            "every row in both studies"
            if studies in (None, ["a", "b"])
            else (
                f"the rows re-measured in study {', '.join(s.upper() for s in studies)} only — "
                "the other study's rows predate this window and their provenance is not "
                "described here"
            )
        )
        if retrospective:
            # The retrospective sidecar reconstructed stability from `git log` and
            # records dirty_at_start: null -- uncommitted edits during the run are
            # unobservable after the fact. Saying "no tracked file changed" over
            # that asserts strictly more than the artifact supports, which is the
            # same overclaim this document already had to retract once.
            lines.append(
                "No tracked **commit** touched these trees during the measurement window, so "
                f"the hashes are the best available description of {scope}. That is weaker "
                "than 'nothing changed': an uncommitted edit made and kept during the sweep "
                "would not appear in `git log`, and this sidecar records "
                "`dirty_at_start: null` precisely because that state cannot be recovered "
                "afterwards. Runs that capture provenance live do observe it."
            )
        else:
            lines.append(
                f"No tracked file changed between the start and end of the sweep, so these "
                f"hashes apply to {scope}. This is an endpoint comparison: it catches a "
                f"change that persists, not one reverted before the sweep ended."
            )
    elif stable is False:
        lines.append(
            "**A tracked file changed mid-sweep** "
            f"({', '.join(f'`{p}`' for p in doc.get('changed_during_run', []))}), so no "
            "single blob describes all the rows. Treat the table above as approximate."
        )
    else:
        lines.append(
            "The mid-sweep stability of these files was **not recorded**, so it is not "
            "claimed here."
        )

    digests = doc.get("tree_digests")
    if digests:
        lines += [
            "",
            "Naming two files is not the measurement's code identity — the harness also "
            "executes the blockers, indexes, embedders and metrics that decide what a row "
            "*means*. So whole trees are digested, over every `.py` they contain:",
            "",
            "| tree | digest (`.py` contents) |",
            "|---|---|",
        ]
        lines += [f"| `{tree}/**` | `{d[:16]}` |" for tree, d in digests.items()]

    drivers_ok = doc.get("drivers_unchanged_during_run")
    if drivers_ok is not None:
        lines.append("")
        if drivers_ok:
            lines.append(
                "The shell **drivers** are digested separately (`driver_digest`), because they "
                "decide which cells *ran* rather than what a number means. They did not change "
                "while the sweep was in flight."
            )
        else:
            changed = doc.get("drivers_changed_during_run", [])
            lines += [
                "The shell **drivers** are digested separately (`driver_digest`), because they "
                "decide which cells *ran* rather than what a number means — and unlike the "
                "measurement code, **they did change while this sweep was in flight**:",
                "",
                *[f"- `{c}`" for c in changed],
                "",
                "The rows stand: no `.py` moved, so every number still means what the table says "
                "it means. What this disclosure costs is the claim that re-running the *committed* "
                "driver reproduces this exact schedule of cells — it does not, because part of "
                "that schedule was a recovery from an OS kill. It is reported rather than "
                "swallowed because the `.py`-only digest this study started with certified these "
                "same edits as “unchanged”.",
            ]
    return lines


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"missing artifact {path} — run examples/research/run_lfm25.sh first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "—"
    return f"[{low:+.4f}, {high:+.4f}]"


def _millions(count: int | None) -> str:
    return "—" if count is None else f"{count / 1e6:.0f}M"


def _benchmarks(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({row["benchmark"] for row in rows})


def _models(rows: list[dict[str, Any]], baseline: str) -> list[str]:
    """Baseline first, then the rest ordered by measured parameter count."""
    size = {row["model"]: row.get("parameter_count") or 0 for row in rows}
    others = sorted(set(size) - {baseline}, key=lambda name: (size[name], name))
    return ([baseline] if baseline in size else []) + others


def _cells(rows: list[dict[str, Any]], **match: Any) -> list[dict[str, Any]]:
    return [row for row in rows if all(row.get(key) == value for key, value in match.items())]


def _best_arm(rows: list[dict[str, Any]], model: str, benchmark: str) -> dict[str, Any] | None:
    """The model's best prompt arm on this benchmark, at the headline k.

    Reporting the best arm rather than a fixed one is the configuration a user
    following the model card would land on, and every arm is in the per-arm table
    below — so this selects, it does not hide.
    """
    candidates = [
        row
        for row in _cells(rows, model=model, benchmark=benchmark, k=HEADLINE_K, status="ok")
        if row.get("candidate_recall") is not None
    ]
    return max(candidates, key=lambda row: row["candidate_recall"]) if candidates else None


def _failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") != "ok"]


def _recall_table(rows: list[dict[str, Any]], baseline: str) -> str:
    """Per-benchmark candidate recall at the headline k. Never averaged across benchmarks."""
    benchmarks = _benchmarks(rows)
    header = "| model | params | " + " | ".join(f"`{b}`" for b in benchmarks) + " |\n"
    header += "|---|---|" + "---|" * len(benchmarks) + "\n"
    body = ""
    for model in _models(rows, baseline):
        best = [_best_arm(rows, model, benchmark) for benchmark in benchmarks]
        params = next((_millions(cell["parameter_count"]) for cell in best if cell), "—")
        marker = " *(baseline)*" if model == baseline else ""
        cells = " | ".join(
            _fmt(cell["candidate_recall"]) if cell else "**failed**" for cell in best
        )
        body += f"| `{model}`{marker} | {params} | {cells} |\n"
    return header + body


def _reachable_table(rows: list[dict[str, Any]], baseline: str) -> str:
    """Recall of what is *reachable* — the model comparison with the gold-set artefact divided out."""
    benchmarks = _benchmarks(rows)
    header = "| model | " + " | ".join(f"`{b}`" for b in benchmarks) + " |\n"
    header += "|---|" + "---|" * len(benchmarks) + "\n"
    body = ""
    for model in _models(rows, baseline):
        cells = []
        for benchmark in benchmarks:
            cell = _best_arm(rows, model, benchmark)
            cells.append(_fmt(cell.get("recall_of_reachable")) if cell else "**failed**")
        body += f"| `{model}` | " + " | ".join(cells) + " |\n"
    ceilings = " · ".join(
        f"`{b}` {_fmt(next((c['reachable_recall_ceiling'] for m in _models(rows, baseline) if (c := _best_arm(rows, m, b))), None))}"
        for b in benchmarks
    )
    # `body` belongs in the output. It was built and dropped, so the rendered
    # table was a header with no rows -- directly under prose calling it "the
    # model comparison". Caught by cross-model review of the committed artifact.
    return header + body + f"\nReachable ceiling per benchmark: {ceilings}.\n"


def _interval_table(rows: list[dict[str, Any]], baseline: str) -> tuple[str, list[str]]:
    """Paired per-record recall deltas vs. the baseline, resampled BY GOLD CLUSTER.

    Returns the table and the list of notes about degenerate bounds.
    """
    benchmarks = _benchmarks(rows)
    table = "| model | benchmark | arm | Δ recall vs baseline | 95% CI | clusters | excludes 0 |\n"
    table += "|---|---|---|---|---|---|---|\n"
    notes: list[str] = []
    for model in _models(rows, baseline):
        if model == baseline:
            continue
        for benchmark in benchmarks:
            for row in sorted(
                _cells(rows, model=model, benchmark=benchmark, k=HEADLINE_K, status="ok"),
                key=lambda r: str(r.get("prompt_arm")),
            ):
                delta = row.get("vs_reference_delta")
                if delta is None:
                    continue
                low, high = row.get("vs_reference_ci_low"), row.get("vs_reference_ci_high")
                excludes = "—"
                if low is not None and high is not None:
                    excludes = "yes" if (low > 0 or high < 0) else "no"
                    # A percentile bootstrap on a DISCRETE statistic can land a
                    # bound exactly on 0.0000. That does not exclude zero, and a
                    # ">= 0" test would read it as if it did.
                    if low == 0.0 or high == 0.0:
                        excludes = "no (bound **exactly** 0)"
                        notes.append(
                            f"`{model}` on `{benchmark}` (arm `{row['prompt_arm']}`) has a "
                            f"bootstrap bound exactly on 0.0000 — the interval does NOT "
                            f"exclude zero."
                        )
                table += (
                    f"| `{model}` | `{benchmark}` | `{row['prompt_arm']}` | "
                    f"{_fmt(delta)} | {_ci(low, high)} | "
                    f"{_fmt(row.get('ci_clusters'))} | {excludes} |\n"
                )
    return table, notes


def _wins(rows: list[dict[str, Any]], model: str, baseline: str) -> tuple[list[str], list[str]]:
    """Benchmarks where ``model``'s interval vs. the baseline excludes zero, by direction.

    Reported **per arm**, because an arm without a counterpart in the baseline has
    no paired interval and is skipped here. Reading the result as a
    checkpoint-level verdict would then assert something broader than was
    measured — see :func:`_unpaired_arms`, which surfaces exactly what this
    function cannot see.
    """
    ahead: list[str] = []
    behind: list[str] = []
    for benchmark in _benchmarks(rows):
        for row in _cells(rows, model=model, benchmark=benchmark, k=HEADLINE_K, status="ok"):
            low, high = row.get("vs_reference_ci_low"), row.get("vs_reference_ci_high")
            if low is None or high is None:
                continue
            label = f"`{benchmark}` ({row['prompt_arm']})"
            if low > 0 and label not in ahead:
                ahead.append(label)
            elif high < 0 and label not in behind:
                behind.append(label)
    return ahead, behind


def _unpaired_arms(
    rows: list[dict[str, Any]], model: str, baseline: str
) -> list[tuple[str, str, float, float]]:
    """Arms of ``model`` with NO paired interval, and how they compare on raw recall.

    An arm the baseline does not run cannot be bootstrapped against it — the
    paired test needs both per-record vectors on the same records. Those arms are
    invisible to :func:`_wins`, and on this study one of them is the checkpoint's
    *best* configuration: on ``abt_buy`` the documented-prompt arm scores above
    the baseline's best while every paired arm scores below it. Left unsaid, the
    headline would read as a checkpoint-level verdict the data cannot support.

    Returns ``(benchmark, arm, model_recall, baseline_best_recall)``.
    """
    out: list[tuple[str, str, float, float]] = []
    for benchmark in _benchmarks(rows):
        base_cells = _cells(rows, model=baseline, benchmark=benchmark, k=HEADLINE_K, status="ok")
        if not base_cells:
            continue
        base_best = max(c["candidate_recall"] for c in base_cells)
        for row in _cells(rows, model=model, benchmark=benchmark, k=HEADLINE_K, status="ok"):
            if row.get("vs_reference_ci_low") is None:
                out.append((benchmark, row["prompt_arm"], row["candidate_recall"], base_best))
    return out


CONTROL = "random-init-control-350M"

#: Below this, the tuned-vs-random gap is small enough that a reported margin can
#: be a large fraction of the benchmark's whole range. It is a **resolution**
#: label, not a significance test — significance comes from the control's paired
#: interval, which is a measured quantity. Keeping the two apart is the point: an
#: earlier version used this threshold AS a significance test and contradicted
#: its own intervals.
NARROW_RANGE = 0.05


def _assert_comparable_cohorts(tuned: list[dict[str, Any]], base: list[dict[str, Any]]) -> str:
    """Refuse to subtract study B's control from study A's best across cohorts.

    The noise floor is the only number in this document computed ACROSS the two
    studies, and ``LFM25_STUDY=a|b`` explicitly supports re-measuring one of them
    alone. So the two rows files can drift apart — a fresh study-A score minus a
    stale study-B control is not a like-for-like gap, and publishing it as one
    would be exactly the cross-cohort error this document already had to retract
    once. ``metric_revision`` is what the harness bumps when the measured
    quantity changes; a mismatch means the subtraction is meaningless and is
    refused rather than rendered with a caveat. (Cross-model review.)

    Returns the shared revision, for the report to state.
    """
    tuned_revs = {row.get("metric_revision") for row in tuned if row.get("status") == "ok"}
    base_revs = {row.get("metric_revision") for row in base if row.get("status") == "ok"}
    if tuned_revs != base_revs:
        raise SystemExit(
            "Refusing to render: the two studies were measured at different metric "
            f"revisions (study A {sorted(map(str, tuned_revs))}, study B "
            f"{sorted(map(str, base_revs))}). The noise floor subtracts study B's control "
            "from study A's best tuned score, which is only meaningful within one cohort. "
            "Re-measure both studies (LFM25_STUDY=both) before generating this report."
        )
    return ", ".join(sorted(str(r) for r in tuned_revs)) or "unrecorded"


def _cross_study_caveat() -> list[str]:
    """Say so when the sidecar shows only one study was re-measured.

    A partial re-run leaves the other study's rows untouched, so the noise floor
    then spans two measurement windows. The provenance section records the scope;
    this surfaces it where the cross-study number is actually used.
    """
    if not PROVENANCE.exists():
        return []
    studies = json.loads(PROVENANCE.read_text()).get("studies_measured")
    if studies is None or sorted(studies) == ["a", "b"]:
        return []
    named = ", ".join(s.upper() for s in studies)
    return [
        "",
        f"⚠️ **This gap spans two measurement windows.** The provenance sidecar records that "
        f"only study {named} was re-measured in the last run, so the tuned score and the "
        f"control below come from rows captured at different times. Both cohorts share a "
        f"metric revision — that is checked, and rendering is refused otherwise — but a "
        f"single re-measured study is still weaker evidence than one sweep over both.",
    ]


def _control_intervals(base: list[dict[str, Any]], benchmark: str) -> list[str]:
    """Every measured paired interval for the control on ``benchmark``, formatted.

    Read from the rows rather than typed into the prose beside them. The
    paragraph these feed used to quote two endpoints literally, which is the
    failure this repo keeps paying for: a re-measure replaces the JSONL and
    regenerates the document, and the hard-coded sentence would then contradict
    the table directly above it while the page still claimed every number came
    from the artifact. (Cross-model review.)
    """
    seen: list[str] = []
    for row in _cells(base, model=CONTROL, benchmark=benchmark, k=HEADLINE_K, status="ok"):
        low, high = row.get("vs_reference_ci_low"), row.get("vs_reference_ci_high")
        if low is None or high is None:
            continue
        formatted = f"`{_ci(low, high)}`"
        if formatted not in seen:
            seen.append(formatted)
    return sorted(seen)


def _correction_paragraph(
    base: list[dict[str, Any]], uninformative: list[str], narrow: list[str]
) -> list[str]:
    """State the retracted claim, with its refuting numbers taken from the rows.

    Kept in the document deliberately: the earlier revision published the
    opposite verdict, and a correction that quietly disappears is how a reader
    ends up trusting the wrong version they already read. Every quantity is
    derived, so a re-measure cannot leave the retraction contradicting the table.
    """
    if not narrow:
        # Nothing was misclassified by the old rule on these rows, so there is
        # nothing to retract -- do not print a correction about a benchmark this
        # run did not measure that way.
        return []
    quoted = []
    for benchmark in narrow:
        intervals = _control_intervals(base, benchmark)
        if intervals:
            quoted.append(f"on `{benchmark}` they are {' and '.join(intervals)}")
    if not quoted:
        return []
    unusable = ", ".join(f"`{b}`" for b in uninformative) if uninformative else "**no benchmark**"
    return [
        "",
        "**A correction, stated plainly because it was published the other way round for "
        "one revision of this document.** An earlier version called a benchmark "
        f"*uninformative* whenever the tuned-vs-random gap was under {NARROW_RANGE:g} recall, "
        f"and on that basis said {', '.join(f'`{b}`' for b in narrow)} could not distinguish "
        "a trained retriever from noise. That was wrong, and this study's own rows say so: "
        f"of the control's paired intervals, {'; '.join(quoted)} — all excluding zero. The "
        "gap between one seeded control and the best tuned score is a point estimate, not a "
        "variance estimate, and it cannot be used as a significance test — least of all "
        f"against measured intervals sitting in the same file. Only {unusable} genuinely "
        "fails to separate.",
    ]


def _noise_floor_table(
    tuned: list[dict[str, Any]], base: list[dict[str, Any]]
) -> tuple[str, list[str], list[str], list[str]]:
    """Every benchmark's tuned-model spread against the random-init control.

    The control is the same 350M architecture with seeded random weights, so the
    gap between it and the best tuned model is the benchmark's whole dynamic
    range. **Two different questions are kept separate here**, because collapsing
    them produced a wrong claim:

    1. *Can this benchmark tell a trained retriever from random weights at all?*
       Answered by the control's **paired interval** — measured uncertainty.
    2. *How much room is there between them?* Answered by the gap, which is a
       point estimate and says nothing about variance.

    A benchmark can separate the two and still have very little room
    (``walmart_amazon``: the control is significantly below the tuned model, and
    the whole range is 0.018). Only a benchmark failing (1) carries no evidence.

    Returns ``(table, uninformative, informative, narrow)``.
    """
    benchmarks = sorted({row["benchmark"] for row in tuned} | {row["benchmark"] for row in base})
    table = (
        "| benchmark | random-init control | best tuned model | observed gap | "
        "control's paired CI | verdict |\n"
    )
    table += "|---|---|---|---|---|---|\n"
    uninformative: list[str] = []
    informative: list[str] = []
    narrow: list[str] = []
    for benchmark in benchmarks:
        control = _best_arm(base, CONTROL, benchmark)
        # sorted(), not the raw set: iteration order over a set of strings varies
        # between processes, and `max` returns the FIRST maximum -- so on a tie
        # (fodors_zagat, where four models all score 1.0000) the model CREDITED
        # here flipped from run to run while the number stayed right. That made
        # the file fail its own reproduce-the-committed-table check, and it is
        # the same shape as attributing a result to whichever row was hashed
        # first. Ties now break on model name, deterministically.
        tuned_cells = [
            cell
            for model in sorted({row["model"] for row in tuned})
            if (cell := _best_arm(tuned, model, benchmark))
        ]
        if control is None or not tuned_cells:
            table += f"| `{benchmark}` | — | — | — | — | — |\n"
            continue
        floor = control["candidate_recall"]
        best = max(tuned_cells, key=lambda c: (c["candidate_recall"], c["model"]))
        margin = best["candidate_recall"] - floor
        # The verdict comes from the control's own PAIRED INTERVAL, not from the
        # size of the gap. An earlier version used a bare `margin < 0.05`
        # point-estimate cutoff and called `walmart_amazon` unable to tell a
        # trained retriever from random weights -- while this study's own
        # intervals for the control there are [-0.0275, -0.0072] and
        # [-0.0295, -0.0099], BOTH excluding zero. The gap between one seeded
        # control and the best tuned score is not a variance estimate, and using
        # it as one contradicted the measured uncertainty sitting in the same
        # rows. (Cross-model review.) A small gap makes a benchmark
        # low-resolution, not blind.
        low = control.get("vs_reference_ci_low")
        high = control.get("vs_reference_ci_high")
        separates = low is not None and high is not None and (low > 0 or high < 0)
        if not separates:
            verdict = "**cannot separate trained from random**"
            uninformative.append(benchmark)
        elif margin < NARROW_RANGE:
            verdict = f"separates, but narrow (<{NARROW_RANGE:.2f})"
            narrow.append(benchmark)
            informative.append(benchmark)
        else:
            verdict = "separates, wide"
            informative.append(benchmark)
        table += (
            f"| `{benchmark}` | {_fmt(floor)} | {_fmt(best['candidate_recall'])} "
            f"(`{best['model']}`) | **{margin:+.4f}** | {_ci(low, high)} | {verdict} |\n"
        )
    return table, uninformative, informative, narrow


#: The control is the 350M architecture. Only the 350M encoder is therefore a
#: matched-backbone comparison against it; the 230M encoder declares a DIFFERENT
#: backbone (LiquidAI/LFM2.5-230M-Base) and a different size, so its gap to the
#: control confounds pretraining with model size and cannot carry a causal claim
#: about pretrained weights. Stated per row rather than left to the reader.
MATCHED_BACKBONE_ENCODER = "LiquidAI/LFM2.5-Encoder-350M"


def _arm_cells(rows: list[dict[str, Any]], model: str, benchmark: str) -> dict[str, dict[str, Any]]:
    """The model's best cell per prompt arm on this benchmark, at the headline k."""
    best: dict[str, dict[str, Any]] = {}
    for row in _cells(rows, model=model, benchmark=benchmark, k=HEADLINE_K, status="ok"):
        if row.get("candidate_recall") is None:
            continue
        arm = str(row.get("prompt_arm"))
        if arm not in best or row["candidate_recall"] > best[arm]["candidate_recall"]:
            best[arm] = row
    return best


def _control_vs_base_encoders(base: list[dict[str, Any]]) -> list[str]:
    """Where the untrained control outscores the *real* base encoders.

    **Compared ARM BY ARM, not best-arm against best-arm.** Both sides wear the
    same untrained mean-pooling head, so pooling is held fixed either way — but
    taking each side's own best arm let the prompt configuration vary too: on
    ``abt_buy`` the control's best arm was ``none`` while the encoder's was
    ``instruct``, and on ``amazon_google`` it was the other way round. A claim
    that *only* pretrained-versus-random weights differ cannot rest on a
    comparison where the prompt also differs. Holding the arm fixed makes the
    remaining difference the weights — for the matched-backbone encoder. For the
    230M encoder the backbone and size differ as well, so that row stays an
    observation. (Cross-model review.)
    """
    encoders = sorted(
        {row["model"] for row in base if "Encoder" in row["model"]},
    )
    benchmarks = sorted({row["benchmark"] for row in base})
    lines: list[str] = []
    for encoder in encoders:
        beaten: list[str] = []
        compared: list[str] = []
        arms_used: set[str] = set()
        for benchmark in benchmarks:
            control_arms = _arm_cells(base, CONTROL, benchmark)
            encoder_arms = _arm_cells(base, encoder, benchmark)
            shared = sorted(set(control_arms) & set(encoder_arms))
            if not shared:
                continue
            compared.append(benchmark)
            arms_used.update(shared)
            if all(
                control_arms[a]["candidate_recall"] >= encoder_arms[a]["candidate_recall"]
                for a in shared
            ):
                beaten.append(benchmark)
        matched = encoder == MATCHED_BACKBONE_ENCODER
        note = (
            " — **matched backbone and matched prompt arm**, so this pair isolates pretraining"
            if matched
            else " — *different backbone and size*, so this pair confounds pretraining "
            "with model size and is an observation only"
        )
        arm_note = (
            f", comparing the same prompt arm on both sides "
            f"({', '.join(f'`{a}`' for a in sorted(arms_used))})"
            if arms_used
            else ""
        )
        lines.append(
            f"- Random weights match or beat `{encoder}` on **{len(beaten)} of "
            f"{len(compared)}** benchmarks where both ran a shared arm"
            + (f" ({', '.join(f'`{b}`' for b in beaten)})" if beaten else "")
            + arm_note
            + note
            + "."
        )
    return lines


#: The earlier portfolio study, whose margins predate any noise-floor control.
PRIOR_LADDER = "docs/research/20260727_embedder_ladder.md"


def _noise_band_warning(tuned: list[dict[str, Any]], base: list[dict[str, Any]]) -> list[str]:
    """Tell a reader of the earlier ladder how to recalibrate its small margins.

    That study reports per-benchmark wins with no noise floor, because none had
    been measured. Its `walmart_amazon` margins are the ones at risk: the whole
    tuned-vs-random range there is smaller than some of the differences it
    reports as results. Computed here rather than asserted, so the threshold
    cannot drift from the measurement.
    """
    lines: list[str] = []
    walmart_band: float | None = None
    for benchmark in sorted({row["benchmark"] for row in base}):
        control = _best_arm(base, CONTROL, benchmark)
        tuned_cells = [
            cell
            for model in sorted({row["model"] for row in tuned})
            if (cell := _best_arm(tuned, model, benchmark))
        ]
        if control is None or not tuned_cells:
            continue
        margin = max(c["candidate_recall"] for c in tuned_cells) - control["candidate_recall"]
        if margin >= 0.05:
            continue
        if margin <= 0:
            lines.append(
                f"- On `{benchmark}`, the best tuned model does not beat random weights "
                f"at all (range {margin:+.4f}) — any ranking read off it is "
                f"uninterpretable."
            )
        else:
            if benchmark == "walmart_amazon":
                walmart_band = margin
            lines.append(
                f"- On `{benchmark}`, the **entire** distance between the best tuned "
                f"model and a network with random weights is **{margin:.4f}**. A margin "
                f"is best read as a fraction of that, not in isolation."
            )
    if not lines:
        return []
    return [
        "",
        f"**Recalibrating the earlier portfolio study.** `{PRIOR_LADDER}` reports "
        "per-benchmark margins that predate any noise-floor control, so it has no way to "
        "say which of them clear noise. Read against this control:",
        "",
        *lines,
        "",
        *_prior_walmart_row(walmart_band),
        "",
        "This study does not refute those margins and does not call them noise — different "
        "models, measured intervals that exclude zero, and nothing here contradicts them. "
        "The recalibration is about **resolution**: quote a `walmart_amazon` margin with "
        "the trained-vs-random range beside it, because a margin that is a large fraction "
        "of everything the benchmark can measure is a fragile basis for ranking even when "
        "it is real. The earlier document is left as it stands; this is a recalibration "
        "note, not a rewrite.",
    ]


def _prior_walmart_row(noise_band: float | None = None) -> list[str]:
    """Quote the earlier study's `walmart_amazon` margins, read from its own table.

    Parsed rather than retyped: every factual error this repo shipped in one day
    was in hand-copied prose sitting beside a correct generated table.
    """
    path = REPO_ROOT / PRIOR_LADDER
    if not path.exists():
        return []
    rows = [
        match
        for line in path.read_text().splitlines()
        if (
            match := re.match(
                r"\|\s*`([^`]+)`\s*\|\s*walmart_amazon\s*\|\s*([+-][\d.]+)\s*\|"
                r"\s*\[([+-][\d.]+),\s*([+-][\d.]+)\]",
                line.strip(),
            )
        )
    ]
    # Only the rows whose interval excludes zero were reported there as wins.
    wins = [m for m in rows if float(m.group(3)) > 0]
    if not wins:
        return []
    band = noise_band if noise_band is not None else 0.0
    below = [m for m in wins if float(m.group(2)) < band]
    return [
        "",
        "Concretely, every `walmart_amazon` margin that study reports as a win, with its "
        f"interval, set against the {band:.4f} gap measured here between the best tuned "
        "model and one seeded random initialisation:",
        "",
        *[
            f"- `{m.group(1)}`: **{m.group(2)}** [{m.group(3)}, {m.group(4)}]"
            + (f"  ← {float(m.group(2)) / band:.0%} of that observed gap" if band > 0 else "")
            for m in sorted(wins, key=lambda m: m.group(1))
        ],
        "",
        f"**{len(below)} of {len(wins)}** are smaller than the gap. Those intervals "
        f"exclude zero, so they are **real effects and this study does not call them "
        f"noise** — `walmart_amazon` does separate trained models from random weights. "
        f"The narrower point is about resolution: on a benchmark where the whole measured "
        f"distance from a random network to the best tuned model is {band:.4f}, a margin "
        f"of that order is a large share of it, so it is a fragile basis for ranking and "
        f"should be quoted with that distance beside it rather than on its own. The "
        f"percentages are ratios to one observed gap, not to a calibrated range — see the "
        f"noise-floor preamble.",
    ]


def _licence_clauses() -> list[str]:
    """The clauses that decide whether this may be a shipped default, quoted from the file."""
    text = LICENSE.read_text()
    wanted = (r'^"Threshold" shall mean', r"^5\. Commercial Use Limitation", r"^\(a\) ", r"^\(b\) ")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and any(re.match(pattern, line.strip()) for pattern in wanted)
    ]


def _load_probe_section(probe: dict[str, Any]) -> str:
    remote = probe["remote_code"]
    table = "| `trust_remote_code` | class actually instantiated | from checkpoint's code | cos(two unrelated records) | max prompt shift |\n"
    table += "|---|---|---|---|---|\n"
    for key in sorted(remote, reverse=True):
        cell = remote[key]
        table += (
            f"| `{cell['trust_remote_code']}` | `{cell['instantiated_class'].split('.')[0]}"
            f"….{cell['instantiated_class'].split('.')[-1]}` | "
            f"{cell['from_checkpoint_code']} | "
            f"{cell['cosine_between_unrelated_records']:.6f} | "
            f"{cell['max_abs_prompt_shift']:.6g} |\n"
        )

    weights = "| checkpoint | `AutoModel` missing keys | deterministic across two loads | `AutoModelForMaskedLM` missing keys |\n"
    weights += "|---|---|---|---|\n"
    for name in sorted(probe["weight_loading"]):
        cell = probe["weight_loading"][name]
        deterministic = cell["AutoModel"]["two_load_max_drift"] == 0.0
        weights += (
            f"| `{name}` | **{cell['AutoModel']['missing_keys']}** | "
            f"{'yes' if deterministic else '**no**'} | "
            f"{cell['AutoModelForMaskedLM']['missing_keys']} |\n"
        )
    return table, weights  # type: ignore[return-value]


def render() -> str:
    tuned = _read_rows(TUNED_ROWS)
    base = _read_rows(BASE_ROWS)
    probe = json.loads(LOAD_PROBE.read_text())
    remote_table, weight_table = _load_probe_section(probe)  # type: ignore[misc]

    tuned_benchmarks = _benchmarks(tuned)
    interval_table, degenerate = _interval_table(tuned, TUNED_BASELINE)
    ahead, behind = _wins(tuned, "LiquidAI/LFM2.5-Embedding-350M", TUNED_BASELINE)
    unpaired = _unpaired_arms(tuned, "LiquidAI/LFM2.5-Embedding-350M", TUNED_BASELINE)
    n_tests = sum(
        1
        for b in tuned_benchmarks
        for row in _cells(
            tuned, model="LiquidAI/LFM2.5-Embedding-350M", benchmark=b, k=HEADLINE_K, status="ok"
        )
        if row.get("vs_reference_ci_low") is not None
    )
    fwer = 1 - 0.95**n_tests if n_tests else 0.0
    #: Expected number of nominal 95% intervals that exclude zero purely by chance
    #: when nothing is really different. Reported alongside the observed count so a
    #: reader can compare "how many hits" against "how many hits are free".
    expected_by_chance = 0.05 * n_tests

    base_interval_table, base_degenerate = _interval_table(base, BASE_BASELINE)
    # Checked BEFORE the table is built: the noise floor is the one figure here
    # computed across the two studies, and a partial re-run can leave them in
    # different cohorts.
    metric_revision = _assert_comparable_cohorts(tuned, base)
    noise_table, uninformative, informative, narrow = _noise_floor_table(tuned, base)

    parts = [
        "# LiquidAI LFM2.5 encoders on ER candidate blocking",
        "",
        "**Generated by `examples/research/lfm25_report.py` from the committed artifacts "
        "beside it. Do not edit by hand — a re-run overwrites this file, and every number "
        "below is read from a rows file rather than typed.**",
        "",
        "Measured with the standing embedder-ladder harness "
        "(`examples/research/embedder_ladder.py`), so these numbers are the same "
        "measurement the 2026-07-27 ladder reports: candidate recall from a "
        "`VectorBlocker` over a FAISS cosine index, per benchmark, at `k=20`, with "
        "cross-source candidates only on the linkage benchmarks.",
        "",
        "## The headline",
        "",
        f"`LiquidAI/LFM2.5-Embedding-350M` vs. `{TUNED_BASELINE}` — langres's current "
        f"`DEFAULT_EMBEDDING_MODEL`, and the baseline every **study A** interval is "
        f"measured against. **Study B uses a different baseline**: its deltas are against "
        f"`{BASE_BASELINE}`, which shares the 350M backbone, so a study-B interval reads as "
        f'"what retrieval tuning bought on this backbone" and must never be compared with '
        f"a study-A one. Each table states its own baseline in the heading.",
        "",
        (
            f"- Ahead with an interval excluding zero on: "
            f"{', '.join(ahead) if ahead else '**no benchmark/arm**'}."
        ),
        (
            f"- Behind with an interval excluding zero on: "
            f"{', '.join(behind) if behind else '**no benchmark/arm**'}."
        ),
        "",
        "**These are per-ARM verdicts, not a checkpoint-level one**, and the distinction "
        "changes the reading. A paired interval needs both models' per-record vectors on "
        "the same records, so an arm the baseline never runs cannot be tested at all — it "
        "is silently absent from the two lines above rather than counted as a loss:",
        "",
        *(
            [
                f"- `{benchmark}` arm `{arm}`: **{recall:.4f}** vs the baseline's best "
                f"**{base:.4f}** — "
                + (
                    "**above the baseline**, and untested."
                    if recall > base
                    else "level with the baseline, and untested."
                    if recall == base
                    else "below the baseline, and untested."
                )
                for benchmark, arm, recall, base in unpaired
            ]
            or ["- None — every arm has a counterpart in the baseline."]
        ),
        "",
        'So on any benchmark listed above, the honest statement is *"behind on the arms '
        'that could be tested"*, not *"behind"*. The untested arms are the checkpoint\'s '
        "own documented prompts, which is where a vendor would expect it to look best; "
        "closing that gap needs the baseline re-run under the same prompts, which this "
        "study did not do.",
        "",
        "**It cannot become langres's default regardless of how it scores** — see the "
        "licence section. That is a legal constraint, not a measurement.",
        "",
        "## Was the checkpoint even the thing measured?",
        "",
        "This has to come before any score. All three checkpoints declare "
        f'`model_type: "lfm2"`, which the installed transformers '
        f"({probe['transformers_version']}) implements natively "
        f"(`lfm2` in `CONFIG_MAPPING_NAMES`: **{probe['lfm2_natively_implemented']}**) as a "
        "**causal decoder**, while pointing `auto_map.AutoModel` at their own "
        "*bidirectional* class. Dropping `trust_remote_code` therefore substitutes a "
        "different architecture in silence — no exception, no warning.",
        "",
        remote_table,
        "",
        'The untrusted load is not "slightly degraded". This checkpoint pools the **CLS** '
        "token (`1_Pooling/config.json`), and under causal attention the first token is a "
        "function of itself alone — so every text in the corpus collapses onto one vector "
        "(cosine between two unrelated products = 1.000000) and the prompt changes nothing "
        "at all (shift exactly 0). A sweep would have published that as a blocking recall "
        "and attributed it to the model.",
        "",
        "Each row above was measured in its **own subprocess**, which is load-bearing: "
        "probing both configurations in one process makes the second load report the "
        "native class while producing the remote class's vectors, and the first version of "
        "this probe recorded them as bit-identical.",
        "",
        "### The base encoders' weights",
        "",
        "The two base encoders store their tensors under the MaskedLM wrapper's `lfm2.` "
        "prefix, so `AutoModel` — the class sentence-transformers uses — matches **none** "
        "of them and randomly initialises the whole backbone. transformers reports this as "
        "a log warning; the model then loads and embeds happily, and two independent loads "
        "simply disagree.",
        "",
        weight_table,
        "",
        "`ModelSpec.backbone_auto_class` recovers the real weights via `base_model`, and "
        "`_preflight_backbone` refuses to measure any checkpoint reporting a missing key. "
        "**The study-B numbers below are measured on the recovered weights.**",
        "",
        "## Study A — retrieval-tuned models (like-for-like)",
        "",
        f"Candidate recall at `k={HEADLINE_K}`, best prompt arm per cell. Reported per "
        "benchmark and **never averaged across them**: the benchmarks differ in size, "
        "difficulty and saturation, so a mean over them is not a quantity.",
        "",
        _recall_table(tuned, TUNED_BASELINE),
        "",
        "Recall of what is *reachable* — candidate recall divided by the ceiling the "
        "cross-source filter leaves. This is the model comparison; raw recall mixes it "
        "with a property of the gold set.",
        "",
        _reachable_table(tuned, TUNED_BASELINE),
        "",
        f"### Paired intervals vs. `{TUNED_BASELINE}`",
        "",
        "Mean **per-record** recall difference, bootstrap-resampled **by gold cluster** "
        "(`langres.experiments.statistics.paired_entity_bootstrap`) — never by pair rows, "
        "which are dependent inside one entity and give intervals that are far too tight. "
        "The point estimate is the statistic the interval bounds, not a difference of "
        "aggregate recalls.",
        "",
        interval_table,
        "",
        "## Study B — base masked-LM encoders, untrained pooling",
        "",
        '> **Not a like-for-like comparison. Do not read a low score here as "this model '
        'is bad".**',
        ">",
        "> `LFM2.5-Encoder-350M` and `LFM2.5-Encoder-230M` are `fill-mask` base models "
        "tagged `masked-lm`/`bidirectional`. They are **not embedding models**: they ship "
        "no pooling configuration, so sentence-transformers attaches an **untrained "
        'mean-pooling head**. Their own card describes them as "designed to be fine-tuned '
        'into task-specific models". A base MLM under untrained pooling scores poorly as a '
        "retriever by construction, and that is a fact about the configuration, not the "
        "backbone.",
        ">",
        f"> These rows are therefore measured into their own artifact against "
        f"`{BASE_BASELINE}`, so the delta reads as **what retrieval tuning bought**, and "
        "they are never ranked against the study-A models.",
        "",
        "**Backbone provenance, read from each model card's `base_model` field rather than "
        "assumed:** `LFM2.5-Encoder-350M` and `LFM2.5-Embedding-350M` both declare "
        "`LiquidAI/LFM2.5-350M-Base`, so that pair *is* a same-backbone comparison. "
        "`LFM2.5-Encoder-230M` declares `LiquidAI/LFM2.5-230M-Base` — a **different** "
        "backbone, so its gap to the 350M tuned model confounds tuning with model size and "
        "is not a retrieval-tuning delta.",
        "",
        _recall_table(base, BASE_BASELINE),
        "",
        f"### Paired intervals vs. `{BASE_BASELINE}`",
        "",
        base_interval_table,
        "",
        "## Statistics — what these intervals do and do not claim",
        "",
        f"- The intervals are **nominal, per-benchmark and uncorrected**. "
        f"`LFM2.5-Embedding-350M` carries {n_tests} interval(s) against the baseline; "
        f"testing that many at a nominal 95% gives a family-wise error rate of about "
        f"**{fwer:.0%}** if they were independent. No multiplicity correction was applied, "
        f'so a claim of the form "it wins on at least one benchmark" is **not** supported '
        f"at 95% family-wise — read each benchmark on its own.",
        f"- **How many hits are free:** if nothing were really different, "
        f"{n_tests} nominal 95% intervals would be expected to exclude zero "
        f"**{expected_by_chance:.2f}** times by chance alone. Compare that against the "
        f"observed counts above before reading any single interval as a finding. "
        f"No Holm or other family-wise correction is claimed here, because none was "
        f"computed — the p-values a step-down procedure needs are not recoverable from "
        f"percentile-interval endpoints at Holm's varying thresholds.",
        "- The bootstrap resamples **gold clusters**, and `clusters` in the table is the "
        "honest denominator — not the record count.",
        "- A percentile bootstrap on a discrete recall statistic can place a bound "
        "**exactly** on 0.0000. That does not exclude zero.",
    ]

    if degenerate or base_degenerate:
        parts += ["", "**Degenerate bounds found in this run:**", ""]
        parts += [f"- {note}" for note in dict.fromkeys(degenerate + base_degenerate)]
    else:
        parts += ["", "No interval in this run has a bound exactly on 0.0000."]

    parts += [
        "",
        "## The noise floor — which benchmarks can separate signal from noise",
        "",
        "**A named finding, not a footnote.** `random-init-control-350M` is the LFM2.5 350M "
        "architecture with **seeded random weights and no training whatsoever**, run through "
        "the identical pipeline. A random transformer is a random feature map: near-duplicate "
        "strings still land near each other, so it retrieves real candidates. Its recall is "
        "therefore roughly what a benchmark hands a model that knows *nothing*.",
        "",
        "**What the gap above it is, stated precisely.** It is the **observed gap** between "
        "the best of the tuned models measured here and **one seeded random initialisation** "
        "— not the benchmark's dynamic range, and neither endpoint is a bound. A different "
        "seed would move the floor by an amount this study did not measure (one control, no "
        "replicates), and a stronger model than any tested here would move the ceiling. Every "
        '"% of the gap" figure below is therefore a ratio to *that* observed interval, not a '
        "calibrated fraction of what the benchmark could in principle resolve. It is still the "
        "right order-of-magnitude check — a margin comparable to the whole gap is fragile "
        "however the endpoints move — but it is an observation, not a bound, and the earlier "
        'wording ("the benchmark\'s entire usable dynamic range") claimed more than one '
        "control can support. (Cross-model review.)",
        "",
        f"The control lives in study B and the tuned scores in study A, so this is the one "
        f"figure here computed **across the two studies**. Both were measured at metric "
        f"revision **{metric_revision}**; a mismatch refuses the render outright rather than "
        f"publishing a cross-cohort subtraction with a footnote.",
        *_cross_study_caveat(),
        "",
        noise_table,
        "",
        (
            f"**Benchmarks that cannot tell a trained retriever from random weights: "
            f"{', '.join(f'`{b}`' for b in uninformative) if uninformative else '**none**'}.** "
            f"The control's paired interval there does not exclude zero, so a model ranking "
            f"read off {'it' if len(uninformative) == 1 else 'them'} is not evidence about any "
            f"embedder."
        ),
        "",
        (
            f"**Benchmarks that do separate the two, but narrowly "
            f"(&lt;{NARROW_RANGE:.2f} of range): "
            f"{', '.join(f'`{b}`' for b in narrow) if narrow else 'none'}.** "
            f"This is a statement about **resolution, not significance**. The control *is* "
            f"significantly below the tuned models here — the separation is real. But the "
            f"whole trained-vs-random range is small, so a reported margin can be a large "
            f"fraction of it, and a ranking built from differences of that size is fragile "
            f"rather than meaningless."
        ),
        "",
        (
            f"Benchmarks that separate the two with room to spare: "
            f"{', '.join(f'`{b}`' for b in informative if b not in narrow) or '**none**'}."
        ),
        *_correction_paragraph(base, uninformative, narrow),
        "",
        "**The control also outscores the real base encoders.** All three wear the same "
        "untrained mean-pooling head, so pooling is held fixed throughout; whether "
        "*pretraining* is the only remaining difference depends on the backbone, and is "
        "stated per row:",
        "",
        *_control_vs_base_encoders(base),
        "",
        f"**What this supports, and what it does not.** On `{MATCHED_BACKBONE_ENCODER}` vs "
        "the control, architecture and pooling are held fixed and only pretrained-vs-random "
        "weights differ. So the *finding* is exactly this: **under this untrained "
        "mean-pooling configuration, the pretrained checkpoint scores worse than random "
        "weights.** The 230M row points the same way but cannot even support that, because "
        "it varies backbone and size as well as pretraining.",
        "",
        "A tempting explanation is that MLM training shapes token representations for a head "
        "this pipeline never attaches, so averaging them destroys more than averaging random "
        "projections does. That is a **hypothesis, not a result** — this experiment holds "
        "pooling *fixed*, so it cannot attribute the gap to pooling rather than to the "
        "checkpoint. Testing it means varying the pooling over the same features (CLS, "
        "last-token, a trained head) and seeing whether the ordering reverses. Not done "
        "here. What is safe to conclude either way is narrower and sufficient: an untuned "
        "encoder must not be ranked against a retrieval-tuned one.",
        "",
        "This reframes `fodors_zagat` specifically. It was already labelled *saturated* — "
        "every usable embedder scoring near the ceiling. The control shows it is stronger "
        "than that: it is **uninformative in the strict sense**, because an untrained network "
        "also scores near the ceiling. Saturation says the models agree; this says the "
        "benchmark cannot tell a trained retriever from noise. It must never be cited as "
        "evidence that one embedder beats another.",
        "",
        *_noise_band_warning(tuned, base),
        "",
        "The control also exists because this harness *shipped* the bug it now guards "
        "against: a substitution that silently left random weights running scored 0.9911 "
        "recall and 0.9971 AUC on `fodors_zagat`, and nothing in the row looked wrong. "
        "A permanent noise-floor arm is how that stays visible instead of being rediscovered.",
        "",
        "## Licence — this model must not become the default",
        "",
        "All three checkpoints declare `license: other`, `license_name: lfm1.0` (LFM Open "
        "License v1.0). The three repos ship a **byte-identical** LICENSE file, verified by "
        "sha256; it is committed beside this file as "
        "`20260729_lfm25_license.txt`. The operative clauses, quoted from it:",
        "",
    ]
    parts += [f"> {clause}" for clause in _licence_clauses()]
    parts += [
        "",
        "**Consequence.** langres is Apache-2.0, which carries no such restriction. Making "
        "any of these the `DEFAULT_EMBEDDING_MODEL` would silently put every langres user "
        "above $10M annual revenue outside the licence on the *default* path — a condition "
        "they never opted into. So:",
        "",
        "- These models **may** be offered as an **opt-in named model with a licence note**.",
        "- They **must not** become `DEFAULT_EMBEDDING_MODEL`.",
        "",
        "`lfm1.0` is deliberately absent from the harness's `OSI_APPROVED_LICENSES` allow "
        'list, so it reads as "not clearly OSI" and requires a human to look — an allow '
        "list rather than a deny list, because an unknown licence must fail closed. "
        "(langres does not redistribute weights, so *referencing* the model is fine.)",
        "",
        "## Failures",
        "",
    ]
    failures = _failure_rows(tuned) + _failure_rows(base)
    if failures:
        parts += ["| model | benchmark | arm | error |", "|---|---|---|---|"]
        parts += [
            f"| `{row['model']}` | `{row['benchmark']}` | `{row['prompt_arm']}` | "
            f"{(row.get('error') or '').replace('|', '/')[:200]} |"
            for row in failures
        ]
    else:
        parts.append("Every model loaded and every cell measured — no failure rows.")

    parts += ["", *_provenance_section()]

    parts += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "bash examples/research/run_lfm25.sh          # both sweeps + this write-up",
        "LFM25_STUDY=b bash examples/research/run_lfm25.sh     # just the base-encoder study",
        "uv run python examples/research/lfm25_load_probe.py   # the loading artifact",
        "```",
        "",
        "There is **no skip-completed logic**: `merge_rows()` replaces a re-measured "
        "cell, and re-measuring the reference model clears every other model's "
        "`vs_reference_*` on the benchmarks it touches. Re-running a finished study is "
        "a re-do, not a resume — use `LFM25_STUDY` to name the part that is missing.",
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUTPUT.write_text(render())
    logger.info("wrote %s", OUTPUT)


if __name__ == "__main__":
    main()
