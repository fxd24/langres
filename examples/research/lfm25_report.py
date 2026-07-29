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
import os
import re
import sys
from datetime import datetime
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

#: The earlier portfolio study, whose margins predate any noise-floor control.
#: Relative, because it is also QUOTED into the prose by name. Declared here
#: rather than beside its reader 1300 lines down: the guard below listed the same
#: path as a second string literal, so changing one would have left the guard
#: silently protecting a file nothing reads -- the same defect the guard itself
#: was added for, one constant over. (Self-audit.)
PRIOR_LADDER = "docs/research/20260727_embedder_ladder.md"

#: Every file this renderer READS. One list, because the uncommitted-input guard
#: was written against the three that were on my mind and silently skipped the
#: licence text and the prior ladder -- both of which are QUOTED into the output,
#: so a render could publish bytes absent from the commit carrying it. Adding an
#: input to the module without adding it here is what
#: `test_every_read_input_is_guarded` exists to catch. (Cross-model review.)
RENDER_INPUTS: tuple[Path, ...] = (
    TUNED_ROWS,
    BASE_ROWS,
    LOAD_PROBE,
    LICENSE,
    REPO_ROOT / PRIOR_LADDER,
)

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
    # `studies_measured` is INTENT; `window_complete` is what was reached. An
    # aborted sweep still lists both studies, and reading that as "every row in
    # both studies" attributes rows the run never touched to this window.
    if doc.get("window_complete") is False:
        studies = None  # force the partial wording below
    if stable is True:
        scope = (
            "every row in both studies"
            if studies in (None, ["a", "b"]) and doc.get("window_complete") is not False
            # NOT "the sweep ABORTED": a resume marks its window partial on
            # purpose, having succeeded. The property is the same and the cause
            # is not this document's to assert. (Cross-model review.)
            else "the rows this window actually re-measured — it did not cover every "
            "planned study, so rows it never reached predate it"
            if doc.get("window_complete") is False
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
            # A sweep started on a modified tree is supported -- research often
            # measures uncommitted code -- but then the blob above exists nowhere
            # in history, while the table beside it prints a last-touching commit
            # that did NOT produce it. Saying the hashes "apply" without saying
            # that is the report claiming reconstructibility it does not have.
            dirty = doc.get("dirty_at_start") or []
            if dirty:
                lines.append(
                    f"**Measured on a MODIFIED tree.** {len(dirty)} path(s) carried "
                    "uncommitted changes when the sweep began "
                    f"({', '.join(f'`{p}`' for p in dirty[:6])}"
                    f"{', …' if len(dirty) > 6 else ''}). The blob hashes above therefore "
                    "name content that exists in no commit, and the *last commit touching "
                    "it* column is **not** the code that ran. These rows cannot be "
                    "reproduced from the named checkout alone."
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
        # Read the covered suffixes from the sidecar rather than naming them
        # here. A sidecar written before this field existed is a Python-only
        # digest, and saying so is the honest default -- inheriting the CURRENT
        # suffix set would silently relabel those older, narrower hashes as
        # covering data they never hashed. (Cross-model review.)
        suffixes = doc.get("tree_digest_suffixes") or [".py"]
        covered = ", ".join(f"`{s}`" for s in suffixes)
        scope = (
            f"over every {covered} file they contain"
            if len(suffixes) == 1
            else f"over every {covered} file they contain — the code AND the data it reads"
        )
        lines += [
            "",
            "Naming two files is not the measurement's code identity — the harness also "
            "executes the blockers, indexes, embedders and metrics that decide what a row "
            f"*means*. So whole trees are digested, {scope}:",
            "",
            f"| tree | digest ({covered} contents) |",
            "|---|---|",
        ]
        lines += [f"| `{tree}/**` | `{d[:16]}` |" for tree, d in digests.items()]
        if len(suffixes) == 1:
            lines += [
                "",
                "This window's digest covers **source only**. A benchmark CSV or a JSON "
                "config could have changed under it and read as an unchanged tree; later "
                "windows digest those too.",
            ]

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
                # A reconstructed `git log` check is not proof that no source moved:
                # the same section says two paragraphs earlier that an uncommitted
                # edit kept through the sweep would be invisible to it, and that the
                # digest covered selected suffixes only. Asserting "every number
                # still means what the table says" off that evidence contradicts the
                # caveat directly above it. Live windows DO observe the working
                # tree, so they keep the stronger sentence. (Cross-model review.)
                (
                    "No tracked COMMIT touched the measurement files either, which is the most "
                    "these reconstructed hashes can say — it is not the same as “no `.py` "
                    "moved”, for the reason given above. "
                    if retrospective
                    else "The rows stand: no `.py` moved, so every number still means what the "
                    "table says it means. "
                )
                + "What this disclosure costs is the claim that re-running the *committed* "
                "driver reproduces this exact schedule of cells — it does not, because part of "
                "that schedule was a recovery from an OS kill. It is reported rather than "
                "swallowed because the `.py`-only digest this study started with certified these "
                "same edits as “unchanged”.",
            ]

    # Source identity is not measurement identity. Every cell is a fresh `uv run`
    # with neither --no-sync nor --locked, so the dependency environment can move
    # between cells while every source hash compares equal. Three states, and
    # "not recorded" is one of them: this sweep's own sidecar predates the field.
    # (Cross-model review.)
    env_ok = doc.get("environment_unchanged_during_run", "missing")
    lines.append("")
    if env_ok is True:
        lines.append(
            "The resolved **environment** (`uv.lock`) is recorded too, because each cell is a "
            "fresh `uv run` that may re-synchronise dependencies between cells — a change no "
            "source hash can see. It did not move while the sweep was in flight."
        )
    elif env_ok is False:
        lines += [
            "The resolved **environment** (`uv.lock`) **changed while this sweep was in "
            "flight**, so cells may have run under different dependency versions:",
            "",
            *[f"- `{c}`" for c in doc.get("environment_changed_during_run", [])],
            "",
            "`pyproject.toml` is hashed as measurement code and did not move — a change there "
            "refuses the close outright. This is the weaker, non-fatal signal: `uv.lock` is "
            "gitignored and `uv run` can rewrite it unprompted, so it is disclosed rather than "
            "used to abort a multi-hour sweep.",
        ]
    else:
        lines.append(
            "The resolved **environment** was **not recorded** for this window: `uv.lock` "
            "hashing was added after these rows were measured. Each cell is a fresh `uv run` "
            "with neither `--no-sync` nor `--locked`, so a dependency re-synchronisation "
            "between cells would have been invisible to every hash above. Nothing here says it "
            "happened — only that this run could not have observed it. Later runs can."
        )
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


def _excludes_zero(low: float | None, high: float | None) -> bool:
    """Whether a bootstrap interval clears zero, in either direction.

    One predicate, because it was written out three times — the per-arm table, the
    control-interval split and the noise-floor verdict — and this PR was adding a
    fourth. A bound landing EXACTLY on 0.0000 does not exclude zero; that is a
    measured case here (``fodors_zagat``), not a theoretical one, so it must not
    depend on which copy of the comparison a reader happens to hit.
    """
    return low is not None and high is not None and (low > 0 or high < 0)


def _tied_best_arms(rows: list[dict[str, Any]], model: str, benchmark: str) -> list[dict[str, Any]]:
    """Every arm sharing the model's best recall here, ordered by arm name.

    Exists because ``max`` returns the FIRST maximum, which on a tie is whichever
    row the JSONL happens to list first — and ``merge_rows`` reorders rows through
    ``kept + fresh``, so a partial re-measure can swap them. Ties are not
    hypothetical: five of twenty study-A cells and three of twenty study-B cells
    have them, including the ``random-init-control-350M`` control on
    ``walmart_amazon`` whose paired interval decides a published verdict.

    Returned as a LIST rather than resolved here, because the two callers need
    different things from a tie and only one of them can be answered by picking a
    row. (Cross-model review.)
    """
    candidates = [
        row
        for row in _cells(rows, model=model, benchmark=benchmark, k=HEADLINE_K, status="ok")
        if row.get("candidate_recall") is not None
    ]
    if not candidates:
        return []
    best = max(row["candidate_recall"] for row in candidates)
    return sorted(
        (row for row in candidates if row["candidate_recall"] == best),
        key=lambda row: str(row.get("prompt_arm")),
    )


def _best_arm(rows: list[dict[str, Any]], model: str, benchmark: str) -> dict[str, Any] | None:
    """The model's best prompt arm on this benchmark, at the headline k.

    Reporting the best arm rather than a fixed one is the configuration a user
    following the model card would land on, and every arm is in the per-arm table
    below — so this selects, it does not hide.

    Ties break on arm NAME, deterministically. The enclosing table already fixed
    this one level up — ``_noise_floor_table`` breaks its model tie on
    ``(recall, model)`` and says why — while the arm chosen inside each of those
    cells was still decided by row order. One of N identical sites, again.
    """
    tied = _tied_best_arms(rows, model, benchmark)
    return tied[0] if tied else None


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
                    excludes = "yes" if _excludes_zero(low, high) else "no"
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
) -> list[tuple[str, str, float, float, bool]]:
    """Arms of ``model`` with NO paired interval, and how they compare on raw recall.

    An arm the baseline does not run cannot be bootstrapped against it — the
    paired test needs both per-record vectors on the same records. Those arms are
    invisible to :func:`_wins`, and on this study one of them is the checkpoint's
    *best* configuration: on ``abt_buy`` the documented-prompt arm scores above
    the baseline's best while every paired arm scores below it. Left unsaid, the
    headline would read as a checkpoint-level verdict the data cannot support.

    **A missing interval is not proof that the baseline lacks the arm**, which is
    what this returned before and what the prose then asserted.
    ``merge_rows()`` deliberately clears every retained challenger's
    ``vs_reference_*`` when the reference is re-measured — the deltas were
    computed against per-record scores that no longer exist — while keeping the
    reference's own rows; an interval is also unavailable when there are too few
    gold clusters to resample. After either, an ordinary ``none``/``instruct``
    cell landed here and was explained as an arm the baseline never ran. The
    baseline's own rows are now consulted, and the two causes are reported as the
    different things they are. (Cross-model review.)

    Returns ``(benchmark, arm, model_recall, baseline_best_recall,
    baseline_ran_this_arm)``.
    """
    out: list[tuple[str, str, float, float, bool]] = []
    for benchmark in _benchmarks(rows):
        base_cells = _cells(rows, model=baseline, benchmark=benchmark, k=HEADLINE_K, status="ok")
        if not base_cells:
            continue
        base_best = max(c["candidate_recall"] for c in base_cells)
        base_arms = {c["prompt_arm"] for c in base_cells}
        for row in _cells(rows, model=model, benchmark=benchmark, k=HEADLINE_K, status="ok"):
            if row.get("vs_reference_ci_low") is None:
                arm = row["prompt_arm"]
                out.append((benchmark, arm, row["candidate_recall"], base_best, arm in base_arms))
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

    **One revision across both files, not two matching sets.** Comparing
    ``tuned_revs != base_revs`` accepted a CROSSED cohort: study A holding
    benchmark 1 at revision 1 and benchmark 2 at revision 2 while study B holds
    the reverse gives equal sets ``{1, 2}`` and passes, yet *every* per-benchmark
    subtraction then crosses revisions. That state is reachable — ``merge_rows()``
    replaces re-measured cells in place, so a partial re-run leaves a mixed file
    — and it is the guard passing while the thing it guards is broken, which is
    the failure shape this repo keeps paying for. The report also states a single
    revision number, which is only well-defined when there is one.
    (Cross-model review.)
    """
    tuned_revs = {row.get("metric_revision") for row in tuned if row.get("status") == "ok"}
    base_revs = {row.get("metric_revision") for row in base if row.get("status") == "ok"}
    if len(tuned_revs | base_revs) > 1:
        raise SystemExit(
            "Refusing to render: these rows carry different metric revisions "
            f"(study A {sorted(map(str, tuned_revs))}, study B "
            f"{sorted(map(str, base_revs))}). The noise floor subtracts study B's control "
            "from study A's best tuned score, which is only meaningful within one cohort — "
            "and a single mixed file is no safer than two mismatched ones, because the "
            "subtraction is per benchmark. Re-measure both studies (LFM25_STUDY=both) "
            "before generating this report."
        )
    _assert_same_populations(tuned, base)
    return ", ".join(sorted(str(r) for r in tuned_revs | base_revs)) or "unrecorded"


def _populations(rows: list[dict[str, Any]]) -> dict[str, set[tuple[Any, ...]]]:
    """Per benchmark, EVERY dataset size seen — not just the first one.

    ``setdefault`` kept only the first row's tuple, which hid the reachable case:
    a partial model re-run after a dataset refresh leaves old and new populations
    side by side *within one study*, and ``_noise_floor_table`` then picks a
    refreshed best-tuned row and subtracts a stale control. A set makes the
    within-study split visible. (Cross-model review.)
    """
    sizes: dict[str, set[tuple[Any, ...]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        sizes.setdefault(row["benchmark"], set()).add(
            (row.get("n_records"), row.get("n_gold_pairs"))
        )
    return sizes


def _assert_same_populations(tuned: list[dict[str, Any]], base: list[dict[str, Any]]) -> None:
    """``metric_revision`` is not cohort identity, so check the cohort itself.

    That field tracks changes to how a metric is *computed*. A benchmark's
    records or gold pairs can change underneath it — a dataset refresh, a
    preprocessing fix, a different split — and both files keep revision 1 while
    describing different populations. The guard above would pass and the noise
    floor would subtract recalls measured over different denominators, in a
    document whose own text says cross-cohort subtraction is refused.

    Checked in both directions, because both are reachable: **within** a study
    (a partial model re-run across a dataset change) and **across** the two.

    **What this does not catch, stated rather than implied.** ``n_records`` and
    ``n_gold_pairs`` are cardinalities, not identities: a same-sized replacement
    of the underlying records passes. Closing that needs a dataset fingerprint
    the rows do not carry, which means a schema change and a re-measure — so it
    is named here as a known limit instead of being papered over by a check that
    sounds stronger than it is. (Cross-model review.)
    """
    for label, rows in (("A", tuned), ("B", base)):
        split = {b: pops for b, pops in _populations(rows).items() if len(pops) > 1}
        if split:
            detail = "; ".join(
                f"`{b}` ({', '.join(f'{n}/{g}' for n, g in sorted(map(tuple, pops)))})"
                for b, pops in sorted(split.items())
            )
            raise SystemExit(
                f"Refusing to render: study {label} holds rows measured over DIFFERENT "
                f"populations for the same benchmark — {detail} (records/gold pairs). A "
                "partial re-run across a dataset change leaves both in one file, and every "
                "comparison drawn from it silently mixes them. Re-measure that study whole."
            )

    tuned_pop = _populations(tuned)
    base_pop = _populations(base)
    mismatched = [
        f"`{b}` (study A {next(iter(tuned_pop[b]))}, study B {next(iter(base_pop[b]))})"
        for b in sorted(set(tuned_pop) & set(base_pop))
        if tuned_pop[b] != base_pop[b]
    ]
    if mismatched:
        raise SystemExit(
            "Refusing to render: the two studies measured different populations "
            f"(records, gold pairs) on {', '.join(mismatched)}. The metric revision "
            "matches, but that field tracks how a metric is COMPUTED, not which records it "
            "was computed over — so the noise floor would subtract recalls with different "
            "denominators. Re-measure both studies (LFM25_STUDY=both) over the same data "
            "before generating this report."
        )


def _installed_transformers() -> str | None:
    """The transformers version in THIS environment, or None if absent."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("transformers")
    except PackageNotFoundError:
        return None


def _outside_window(captured: str, window: tuple[str, str | None]) -> str | None:
    """``"predates"`` / ``"postdates"`` if the probe fell outside the sweep, else ``None``.

    **Compared as instants, not as strings.** The first version tested
    ``started <= captured <= finished`` on the raw ISO-8601 text. That is only
    ordering when every stamp carries the *same* UTC offset, which held here by
    accident — one host, one sweep, all ``+02:00``. A probe written under a
    different offset breaks it in both directions: with this window
    (``01:18:12+02:00`` … ``03:44:51+02:00``), ``2026-07-29T02:00:00+00:00`` is
    04:00 local — half an hour *after* the sweep closed — yet ``"02" < "03"``
    reads it as comfortably inside, and ``2026-07-29T00:30:00+00:00`` is 02:30
    local, inside the window, yet sorts before ``"01:18"`` and would be reported
    as predating it. A wrong diagnosis and a silent pass from the same line.
    (Cross-model review.)

    Unparseable or offset-less stamps return ``"unanchored"`` rather than
    crashing the render or silently passing: comparing a naive datetime with an
    aware one raises, and the point of this function is disclosure.
    """
    stamps = [_instant(value) for value in (captured, window[0], window[1] or captured)]
    if any(stamp is None for stamp in stamps):
        return "unanchored"
    at, started, finished = stamps
    assert at is not None and started is not None and finished is not None  # narrowed above
    if at < started:
        return "predates"
    return "postdates" if at > finished else None


def _instant(value: str) -> datetime | None:
    """Parse an ISO-8601 stamp to an aware datetime, or ``None`` if it is not one."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _probe_staleness(probe: dict[str, Any]) -> list[str]:
    """Disclose a load probe that does not belong to this measurement window.

    Everything in the section below — which class is instantiated, how many keys
    fail to load, whether unrelated records collapse onto one vector — is a
    property of the *installed* transformers and of the checkpoints' remote code,
    not of the checkpoint's scores. The rows and the probe are two artifacts with
    two lifetimes, and this is the link between them.

    **Version equality is not freshness.** The first version of this check
    compared ``transformers_version`` only, so a refresh that failed offline or
    on a rate limit — leaving the OLD probe in place — raised nothing whenever
    the version happened to agree, while remote code and cache contents could
    have moved underneath it. ``captured_at`` against the measurement window is
    what actually answers "did this probe run in this sweep". (Cross-model
    review.)

    Disclosed rather than fatal: the probe is still the best record there is, and
    refusing the render would delete the disclosure along with the report.
    """
    lines: list[str] = []
    captured = probe.get("captured_at")
    if captured is None:
        lines.append(
            "⚠️ **This section's capture time was not recorded**, so this document cannot "
            "say whether the probe ran inside the sweep that produced the scores. Probes "
            "written before `captured_at` existed are in this state; re-running "
            "`uv run python examples/research/lfm25_load_probe.py` fixes it."
        )
    elif (window := _window_bounds()) and (side := _outside_window(captured, window)):
        started, finished = window
        reason = {
            "predates": "So the driver's refresh did not take — offline, a rate limit, or a "
            "failure it logged and continued past — and these loading findings describe an "
            "earlier environment than the scores do.",
            "postdates": "So it was refreshed after the fact rather than inside the sweep: "
            "it describes today's environment, which may not be the one the scores were "
            "measured under.",
            # Not "outside the window": one of the three stamps carries no UTC
            # offset (or does not parse), so there is no instant to place it
            # against and the honest answer is that the question is unanswerable
            # here -- not that the probe is stale.
            "unanchored": "One of those stamps carries no UTC offset, so they name wall "
            "clocks rather than instants and cannot be ordered. This document therefore "
            "cannot say whether the probe ran inside the sweep.",
        }[side]
        headline = (
            "**This section's capture time cannot be placed against the measurement window.**"
            if side == "unanchored"
            else f"**This section {side} the measurement window.**"
        )
        lines.append(
            f"⚠️ {headline} The probe was captured at "
            f"`{captured}`; the sweep ran from `{started}` to `{finished or 'unclosed'}`. " + reason
        )
    recorded = probe.get("transformers_version")
    installed = _installed_transformers()
    if installed is not None and installed != recorded:
        lines.append(
            f"⚠️ **This section was captured under transformers `{recorded}`; the "
            f"environment rendering this document has `{installed}`.** Which class is "
            "instantiated, how many weights fail to load and whether the untrusted load "
            "collapses are all properties of the installed library, so these findings may "
            "no longer describe what a reader would see."
        )
    if not lines:
        return []
    lines.append(
        "The *scores* are unaffected either way: they carry their own provenance and "
        "were measured under the sweep's own environment."
    )
    # One blank line between every note, so two warnings do not run together into
    # a single paragraph that reads as one claim.
    spaced: list[str] = []
    for note in lines:
        spaced += [note, ""]
    return spaced


def _window_bounds() -> tuple[str, str | None] | None:
    """The sweep's measurement window, if there is a sidecar recording one.

    **Both** ends, not just the start. A probe captured *after* the window closed
    was refreshed retroactively — it describes today's environment, not the one
    the scores were measured under — and checking only the lower bound would pass
    it silently, which is the same "it looks fresh so it must be right" mistake
    the version comparison already made once.
    """
    if not PROVENANCE.exists():
        return None
    window = json.loads(PROVENANCE.read_text()).get("measurement_window", {})
    started = window.get("started")
    if not started:
        return None
    finished = window.get("finished")
    return str(started), str(finished) if finished else None


def _cross_study_caveat() -> list[str]:
    """Say so when the sidecar shows only one study was re-measured.

    A partial re-run leaves the other study's rows untouched, so the noise floor
    then spans two measurement windows. The provenance section records the scope;
    this surfaces it where the cross-study number is actually used.

    ``studies_measured`` is **intent**, ``window_complete`` is what was reached.
    An aborted ``LFM25_STUDY=both`` run still lists both studies, so keying only
    on the list suppressed this warning over exactly the rows that need it — the
    caveat going silent in the one case it exists for. ``_provenance_section()``
    already honours ``window_complete``; this now does too. (Cross-model review.)
    """
    if not PROVENANCE.exists():
        return []
    doc = json.loads(PROVENANCE.read_text())
    studies = doc.get("studies_measured")
    if doc.get("window_complete") is False:
        # CAUSE-NEUTRAL. `--partial` has meant two different things since the
        # study-A resume started using it deliberately: "the sweep aborted" and
        # "this run re-measured a subset on purpose". Reading it as an abort made
        # every report generated after a SUCCESSFUL one-cell resume announce that
        # the last sweep had failed. The consequence for this gap is identical
        # either way -- the window does not cover every row -- so the warning
        # states that and stops guessing why. (Cross-model review.)
        return [
            "",
            "⚠️ **This gap may span two measurement windows.** The provenance sidecar "
            "records that the last window **did not cover every stored row** — either the "
            "sweep stopped early or it deliberately re-measured only part of one study — "
            "so rows outside it predate the window described here, and the tuned score "
            "and the control below can come from rows captured at different times. Both "
            "cohorts share a metric revision — that is checked, and rendering is refused "
            "otherwise — but a window covering only some of these rows is weaker evidence "
            "than one covering all of them.",
        ]
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


def _control_intervals(base: list[dict[str, Any]], benchmark: str) -> tuple[list[str], list[str]]:
    """The control's measured intervals on ``benchmark``, split by whether they exclude zero.

    Read from the rows rather than typed into the prose beside them. The
    paragraph these feed used to quote two endpoints literally, which is the
    failure this repo keeps paying for: a re-measure replaces the JSONL and
    regenerates the document, and the hard-coded sentence would then contradict
    the table directly above it while the page still claimed every number came
    from the artifact. (Cross-model review.)

    **Split, because the sentence quoting them says "all excluding zero" and this
    returned every arm.** A benchmark enters ``narrow`` on its BEST arm's
    interval alone; a second arm whose interval spans zero was printed
    immediately before that claim, making the claim false about a number on the
    same line. Both groups are returned so the paragraph can quote the
    refutation and still disclose the arm that does not support it, rather than
    filtering it out invisibly. (Cross-model review.)

    Returns ``(excluding_zero, spanning_zero)``.
    """
    excluding: list[str] = []
    spanning: list[str] = []
    for row in _cells(base, model=CONTROL, benchmark=benchmark, k=HEADLINE_K, status="ok"):
        low, high = row.get("vs_reference_ci_low"), row.get("vs_reference_ci_high")
        if low is None or high is None:
            continue
        formatted = f"`{_ci(low, high)}`"
        bucket = excluding if _excludes_zero(low, high) else spanning
        if formatted not in bucket:
            bucket.append(formatted)
    return sorted(excluding), sorted(spanning)


def _correction_paragraph(
    base: list[dict[str, Any]], blind: list[str], narrow: list[str]
) -> list[str]:
    """State the retracted claim, with its refuting numbers taken from the rows.

    Kept in the document deliberately: the earlier revision published the
    opposite verdict, and a correction that quietly disappears is how a reader
    ends up trusting the wrong version they already read. Every quantity is
    derived, so a re-measure cannot leave the retraction contradicting the table.

    ``blind``, not the full unusable set: the closing sentence says a benchmark
    "genuinely fails to separate", which is false of an *inverted* one — there
    the interval separates cleanly, just the wrong way round.
    """
    if not narrow:
        # Nothing was misclassified by the old rule on these rows, so there is
        # nothing to retract -- do not print a correction about a benchmark this
        # run did not measure that way.
        return []
    quoted = []
    also_spanning: list[str] = []
    for benchmark in narrow:
        excluding, spanning = _control_intervals(base, benchmark)
        if excluding:
            quoted.append(f"on `{benchmark}` they are {' and '.join(excluding)}")
        also_spanning += [f"`{benchmark}` {interval}" for interval in spanning]
    if not quoted:
        return []
    unusable = ", ".join(f"`{b}`" for b in blind) if blind else "**no benchmark**"
    # Disclosed, never dropped: a benchmark enters `narrow` on its best arm, so
    # another arm's interval can span zero while the sentence above says "all
    # excluding zero". Filtering it out silently would make the claim true by
    # hiding its counterexample. (Cross-model review.)
    caveat = (
        (
            " Not every arm clears zero — "
            + ", ".join(also_spanning)
            + f" {'spans' if len(also_spanning) == 1 else 'span'} it, and "
            f"{'that arm is' if len(also_spanning) == 1 else 'those arms are'} not part of "
            "the refutation above; the verdict is read off each benchmark's best arm."
        )
        if also_spanning
        else ""
    )
    return [
        "",
        "**A correction, stated plainly because it was published the other way round for "
        "one revision of this document.** An earlier version called a benchmark "
        f"*uninformative* whenever the tuned-vs-random gap was under {NARROW_RANGE:g} recall, "
        f"and on that basis said {', '.join(f'`{b}`' for b in narrow)} could not distinguish "
        "a trained retriever from noise. That was wrong, and this study's own rows say so: "
        f"of the control's paired intervals, {'; '.join(quoted)} — all excluding zero."
        + caveat
        + " The gap between one seeded control and the best tuned score is a point "
        "estimate, not a variance estimate, and it cannot be used as a significance test — "
        f"least of all against measured intervals sitting in the same file. Only {unusable} "
        "genuinely fails to separate.",
    ]


def _control_reference(base: list[dict[str, Any]]) -> str | None:
    """The one model the control's paired intervals are computed against, if there is one.

    Every ``vs_reference_*`` interval in study B is paired with study B's own
    baseline. The noise-floor table's *gap* column, by contrast, is the distance
    to the best model in **study A** — a different model on some benchmarks
    (``walmart_amazon``: gap to ``intfloat/e5-base-v2``, interval against
    ``LiquidAI/LFM2.5-Embedding-350M``). Reading the interval as a significance
    test for the gap's endpoint is therefore wrong, and the prose said exactly
    that. This names the endpoint the interval *does* cover so both the column
    header and the narrow-benchmark sentence can be about that model and no
    other. (Cross-model review.)

    ``None`` when the rows disagree — in which case nothing is named rather than
    the wrong thing being named.
    """
    references = {
        row["reference_model"]
        for row in base
        if row.get("model") == CONTROL
        and row.get("vs_reference_ci_low") is not None
        and row.get("reference_model")
    }
    return references.pop() if len(references) == 1 else None


def _control_direction(base: list[dict[str, Any]], benchmark: str) -> str | None:
    """``"below"`` / ``"above"`` — which side of its paired baseline the control's CI sits.

    ``separates`` is :func:`_excludes_zero`: the interval clears zero in
    *either* direction. The narrow-benchmark prose then said the control is
    "significantly below" — true of every cell measured here, but not implied by
    the predicate that put it in that bucket. A control significantly ABOVE its
    paired baseline, with the cross-study best still a little higher, satisfies
    both ``separates`` and ``0 < margin < NARROW_RANGE`` and would have been
    described backwards. (Cross-model review.)
    """
    cell = _best_arm(base, CONTROL, benchmark)
    if cell is None:
        return None
    low, high = cell.get("vs_reference_ci_low"), cell.get("vs_reference_ci_high")
    if low is None or high is None:
        return None
    if high < 0:
        return "below"
    return "above" if low > 0 else None


def _noise_floor_table(
    tuned: list[dict[str, Any]], base: list[dict[str, Any]]
) -> tuple[str, list[str], list[str], list[str], list[str]]:
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

    Returns ``(table, uninformative, informative, narrow, inverted)``.

    ``uninformative`` is "no ranking may be read off this benchmark" and holds
    **two different reasons**: the interval fails to exclude zero (blind), or it
    excludes zero the wrong way and the control beats every tuned model
    (inverted). ``inverted`` is a subset, returned separately because the prose
    justified the whole list with "the interval does not exclude zero" — which is
    the *opposite* of what the inverted branch requires. One list, two reasons,
    one sentence: the sentence had to be false for one of them.
    (Cross-model review.)
    """
    benchmarks = sorted({row["benchmark"] for row in tuned} | {row["benchmark"] for row in base})
    # The CI column names its own other endpoint. Headed "control's paired CI"
    # beside a "best tuned model" column, it read as an interval about THAT
    # model; it is not, on any benchmark where the best study-A model is not
    # study B's baseline. (Cross-model review.)
    reference = _control_reference(base)
    ci_header = f"control's paired CI vs `{reference}`" if reference else "control's paired CI"
    table = (
        "| benchmark | random-init control | best tuned model | observed gap | "
        f"{ci_header} | verdict |\n"
    )
    table += "|---|---|---|---|---|---|\n"
    uninformative: list[str] = []
    informative: list[str] = []
    narrow: list[str] = []
    inverted: list[str] = []
    unpaired_rows: list[str] = []
    unmeasured: list[str] = []
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
        # The verdict must not depend on WHICH of several equally-good control arms
        # was picked. `_best_arm` now breaks that tie deterministically, but a
        # deterministic arbitrary choice is still arbitrary: on `walmart_amazon` the
        # control's two arms are tied at 0.8626 with DIFFERENT intervals
        # ([-0.0275, -0.0072] and [-0.0295, -0.0099]), and a scientific claim that
        # changes with a tie-break rule is not a claim about the benchmark.
        #
        # So separation has to hold for EVERY tied-best arm. Today they agree on
        # both tied cells and this changes no verdict in the committed rows — which
        # is the point: it removes the dependency while it is still costless, rather
        # than after a re-measure moves one of them across zero. Unanimity is the
        # conservative direction: a tie can no longer manufacture a stronger claim
        # than the weakest arm supports. (Cross-model review.)
        tied_controls = _tied_best_arms(base, CONTROL, benchmark)
        # THREE outcomes, not two. An interval that was never measured is not an
        # interval that failed to exclude zero, and collapsing them made a
        # missing measurement read as a finding about the benchmark:
        # ``merge_rows()`` clears the retained control's ``vs_reference_*`` when
        # study B's reference is re-measured, and the prose downstream then said
        # of that benchmark that "the control's paired interval does not exclude
        # zero" — about a number nobody has. Unmeasured benchmarks are excluded
        # from every blind/separates conclusion instead. (Cross-model review.)
        # An arm missing its interval is a missing measurement for the whole cell:
        # unanimity cannot be established over an arm nobody measured, and quietly
        # dropping it would let the remaining arm speak for the tie.
        intervals = [
            (arm.get("vs_reference_ci_low"), arm.get("vs_reference_ci_high"))
            for arm in tied_controls
        ]
        if low is None or high is None or any(lo is None or hi is None for lo, hi in intervals):
            verdict = "— **no control interval in these rows** (not measured, not a finding)"
            unmeasured.append(benchmark)
            table += (
                f"| `{benchmark}` | {_fmt(floor)} | {_fmt(best['candidate_recall'])} "
                f"(`{best['model']}`) | **{margin:+.4f}** | — | {verdict} |\n"
            )
            continue
        separates = all(_excludes_zero(lo, hi) for lo, hi in intervals)
        if not separates:
            verdict = "**cannot separate trained from random**"
            uninformative.append(benchmark)
        elif margin <= 0:
            # The control OUTSCORES every tuned model. `margin <= 0` also satisfies
            # `margin < NARROW_RANGE`, so this fell into the narrow branch and the
            # prose there — "the control *is* significantly below the tuned models"
            # — asserted the exact opposite of the rows. An inverted benchmark is
            # not a narrow one; it is a result that invalidates any ranking read
            # off it, and it gets said rather than bucketed. (Cross-model review.)
            verdict = "**control BEATS every tuned model — inverted, unusable for ranking**"
            uninformative.append(benchmark)
            inverted.append(benchmark)
        elif margin < NARROW_RANGE:
            verdict = f"separates, but narrow (<{NARROW_RANGE:.2f})"
            narrow.append(benchmark)
            informative.append(benchmark)
        else:
            verdict = "separates, wide"
            informative.append(benchmark)
        # ‡ marks the rows where the gap and the interval have DIFFERENT other
        # endpoints, which is the whole content of the caveat below.
        unpaired = reference is not None and best["model"] != reference
        if unpaired:
            unpaired_rows.append(benchmark)
        table += (
            f"| `{benchmark}` | {_fmt(floor)} | {_fmt(best['candidate_recall'])} "
            f"(`{best['model']}`){' ‡' if unpaired else ''} | **{margin:+.4f}** | "
            f"{_ci(low, high)} | {verdict} |\n"
        )
    if unpaired_rows:
        table += (
            f"\n‡ On {', '.join(f'`{b}`' for b in unpaired_rows)} the best tuned model is "
            f"not `{reference}`, so **no paired interval was computed between the control "
            f"and the model in that cell**. The gap there is a point estimate spanning the "
            f"two studies; the interval beside it is about `{reference}` alone. Nothing in "
            f"this table is a significance test against the named best model.\n"
        )
    if unmeasured:
        table += (
            f"\n**{', '.join(f'`{b}`' for b in unmeasured)} carry no control interval in "
            f"these rows**, so they appear in none of the verdicts below — neither as "
            f"benchmarks that separate nor as benchmarks that cannot. `merge_rows()` clears "
            f"a retained challenger's `vs_reference_*` when the reference is re-measured, "
            f"and a bootstrap needs enough gold clusters; either way the absence is a "
            f"missing measurement, not a result. Re-run the control on "
            f"{'it' if len(unmeasured) == 1 else 'them'} to restore the evidence.\n"
        )
    return table, uninformative, informative, narrow, inverted


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


def _control_vs_base_comparisons(base: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The control-versus-base-encoder tallies, as data rather than prose.

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

    A pair contributes nothing unless *both* sides produced an ``ok`` cell under
    the same prompt arm, so an encoder or control that died mid-sweep drops out
    of the tally instead of being silently scored zero — which is what lets the
    caller notice there is no evidence rather than publish a conclusion drawn
    from missing rows.
    """
    encoders = sorted(
        {row["model"] for row in base if "Encoder" in row["model"]},
    )
    benchmarks = sorted({row["benchmark"] for row in base})
    comparisons: list[dict[str, Any]] = []
    for encoder in encoders:
        beaten: list[str] = []
        strict: list[str] = []
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
            deltas = [
                control_arms[a]["candidate_recall"] - encoder_arms[a]["candidate_recall"]
                for a in shared
            ]
            # `beaten` is "matches or beats" and is what the bullet list says.
            # `strict` additionally requires the control to be AHEAD somewhere:
            # `>=` alone counts an exact tie as a win, and recall ties are common
            # on saturated cells, so a directional claim ("outscores", "scores
            # WORSE than random") built on `beaten` is false for a tie.
            # (Cross-model review.)
            if all(d >= 0 for d in deltas):
                beaten.append(benchmark)
                if any(d > 0 for d in deltas):
                    strict.append(benchmark)
        if not compared:
            continue
        comparisons.append(
            {
                "encoder": encoder,
                "beaten": beaten,
                "strict": strict,
                "compared": compared,
                "arms": sorted(arms_used),
                "matched_backbone": encoder == MATCHED_BACKBONE_ENCODER,
            }
        )
    return comparisons


def _control_vs_base_encoders(base: list[dict[str, Any]]) -> list[str]:
    """Render :func:`_control_vs_base_comparisons` as one bullet per encoder."""
    lines: list[str] = []
    for comparison in _control_vs_base_comparisons(base):
        note = (
            " — **matched backbone and matched prompt arm**, so this pair isolates pretraining"
            if comparison["matched_backbone"]
            else " — *different backbone and size*, so this pair confounds pretraining "
            "with model size and is an observation only"
        )
        arms = comparison["arms"]
        arm_note = (
            f", comparing the same prompt arm on both sides ({', '.join(f'`{a}`' for a in arms)})"
            if arms
            else ""
        )
        beaten = comparison["beaten"]
        lines.append(
            f"- Random weights match or beat `{comparison['encoder']}` on "
            f"**{len(beaten)} of {len(comparison['compared'])}** benchmarks where both "
            f"ran a shared arm"
            + (f" ({', '.join(f'`{b}`' for b in beaten)})" if beaten else "")
            + arm_note
            + note
            + "."
        )
    return lines


def _pretraining_section(base: list[dict[str, Any]]) -> list[str]:
    """The pretrained-versus-random conclusion, gated on the rows that support it.

    The harness records a failure row and keeps going when a model dies, so the
    control or the matched-backbone encoder can be **absent** from a rendered
    report. The earlier version asserted "the control also outscores the real
    base encoders" unconditionally: with those rows missing, the bullet list
    below it was empty, the Failures table said why — and the claim was still
    printed as a finding. A conclusion is emitted here only when cells exist to
    carry it, and the matched-backbone causal claim only when that specific pair
    both ran and went the way the sentence says. (Cross-model review.)
    """
    comparisons = _control_vs_base_comparisons(base)
    if not comparisons:
        return [
            "**No pretrained-versus-random comparison is available in this run.** It needs "
            f"the control and at least one base encoder to produce an `ok` cell on the same "
            f"benchmark under the same prompt arm, at k={HEADLINE_K}; no pair did. See the "
            "failures table for which cells are missing. Nothing is concluded here about "
            "pretrained versus random weights.",
        ]

    matched = next((c for c in comparisons if c["matched_backbone"]), None)
    # `strict`, not `beaten`: "outscores" is directional and an exact tie is not
    # a win. The bullets below stay on "matches or beats", which ties satisfy.
    if any(c["strict"] for c in comparisons):
        headline = "**The control also outscores the real base encoders.**"
    elif any(c["beaten"] for c in comparisons):
        headline = (
            "**The control matches, but never outscores, the real base encoders in this run.**"
        )
    else:
        headline = "**The control does not outscore the real base encoders in this run.**"
    lines = [
        f"{headline} All measured pairs wear the same untrained mean-pooling head, so "
        "pooling is held fixed throughout; whether *pretraining* is the only remaining "
        "difference depends on the backbone, and is stated per row:",
        "",
        *_control_vs_base_encoders(base),
        "",
    ]

    if matched is None:
        lines.append(
            f"**What this supports, and what it does not.** `{MATCHED_BACKBONE_ENCODER}` — the "
            "only encoder here that shares the control's backbone, and so the only pair that "
            "could isolate pretraining — produced no comparable cell in this run. The rows "
            "above vary backbone and size as well as pretraining, so **no causal claim about "
            "pretrained versus random weights is made**."
        )
        return lines

    fixed = (
        f"**What this supports, and what it does not.** On `{MATCHED_BACKBONE_ENCODER}` vs "
        "the control, architecture and pooling are held fixed and only pretrained-vs-random "
        "weights differ"
    )
    if not matched["beaten"]:
        lines.append(
            f"{fixed} — but the control did **not** match or beat it on any of the "
            f"{len(matched['compared'])} benchmarks where both ran a shared arm. The "
            "pretrained-scores-worse-than-random finding is therefore **not** reproduced here."
        )
        return lines

    if not matched["strict"]:
        lines.append(
            f"{fixed} — but the control never scored *above* it on any shared arm either: "
            f"every one of the {len(matched['beaten'])} benchmarks it matched is an exact "
            "recall tie. A tie is not a direction, so **no claim is made** about pretrained "
            "versus random weights here; the pair is measured and inconclusive."
        )
        return lines

    lines += [
        f"{fixed}. So the *finding* is exactly this: **under this untrained "
        "mean-pooling configuration, the pretrained checkpoint scores worse than random "
        f"weights** on {len(matched['strict'])} of {len(matched['compared'])} benchmarks "
        "where both ran a shared arm and the control was strictly ahead"
        + (
            f" (it also ties on {len(matched['beaten']) - len(matched['strict'])})"
            if len(matched["beaten"]) > len(matched["strict"])
            else ""
        )
        + ". Any other encoder row points the same way but cannot "
        "even support that, because it varies backbone and size as well as pretraining.",
        "",
        "A tempting explanation is that MLM training shapes token representations for a head "
        "this pipeline never attaches, so averaging them destroys more than averaging random "
        "projections does. That is a **hypothesis, not a result** — this experiment holds "
        "pooling *fixed*, so it cannot attribute the gap to pooling rather than to the "
        "checkpoint. Testing it means varying the pooling over the same features (CLS, "
        "last-token, a trained head) and seeing whether the ordering reverses. Not done "
        "here. What is safe to conclude either way is narrower and sufficient: an untuned "
        "encoder must not be ranked against a retrieval-tuned one.",
    ]
    return lines


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
        # NOT "everything the benchmark can measure". The preamble above retracts
        # exactly that reading -- one seeded control and one tested ceiling are
        # not the benchmark's dynamic range, and neither endpoint is a bound --
        # so this sentence was reintroducing the retracted interpretation two
        # paragraphs later, in the same generated document. (Cross-model review.)
        "the observed trained-vs-random gap beside it, because a margin that is a large "
        "fraction of that one measured distance is a fragile basis for ranking even when "
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
    """Section 5 **entire**, plus the terms it turns on, quoted from the file.

    Extracted by SECTION, not by line shape. The first version matched
    ``^\\(a\\) `` / ``^\\(b\\) ``, which is the shape of every lettered
    sub-clause in the document — so it pulled §4(a) and §4(b), redistribution
    conditions with nothing to do with whether this can be a default, and,
    keying on (a) and (b) alone, **silently dropped §5(c)**: the exemption for a
    Qualified Non-Profit Organization's non-commercial or research use. Quoting a
    restriction while omitting its exemption is a misquote, and the prose built
    on it ("every langres user above $10M … outside the licence") was false for
    exactly the readers the exemption is written for. (Cross-model review.)

    The definitions are selected by whether §5's own text uses the defined term,
    so they cannot drift out of step with it — no second hand-kept list.

    **Aliases and inflections are part of the term, not noise.** A first cut
    matched ``^"([^"]+)" shall mean`` and tested the captured term as a literal
    substring, which promised "the terms §5 turns on" while delivering neither
    of the two §5(a)/(b) actually leans on: ``"You" (or "Your") shall mean`` has
    a parenthetical alias between the closing quote and *shall mean*, and
    ``"Derivative Works"`` is defined in the plural while §5 says *a Derivative
    Work*. Every quoted term before *shall mean* now counts as a name for that
    definition, and a plural definition matches its singular use.
    (Cross-model review.)
    """
    lines = [line.strip() for line in LICENSE.read_text().splitlines() if line.strip()]
    start = next(i for i, line in enumerate(lines) if re.match(r"^5\. Commercial Use", line))
    end = next(
        i for i, line in enumerate(lines[start + 1 :], start + 1) if re.match(r"^\d+\. ", line)
    )
    section = lines[start:end]
    defined = [
        line
        for line in lines[:start]
        if any(_term_used(term, section) for term in _defined_terms(line))
    ]
    return defined + section


def _defined_terms(line: str) -> list[str]:
    """Every name a definition line introduces — ``"You" (or "Your")`` gives both."""
    if not line.startswith('"') or " shall mean" not in line:
        return []
    return re.findall(r'"([^"]+)"', line.split(" shall mean", 1)[0])


def _term_used(term: str, section: list[str]) -> bool:
    """Whether the section uses the term, counting a plural definition's singular use."""
    forms = {term, term.removesuffix("s")} if term.endswith("s") else {term}
    return any(re.search(rf"\b{re.escape(form)}\b", clause) for form in forms for clause in section)


def _untrusted(probe: dict[str, Any], field: str) -> str:
    """A value from the ``trust_remote_code=False`` probe cell, formatted for prose.

    The prose beside the probe table asserted ``1.000000`` and "exactly 0" as
    literals while the table was read from the JSON. The driver refreshes that
    JSON before every render, so a checkpoint or dependency change would have
    moved the table and left the paragraph restating a measurement nobody took
    — in the one file whose stated premise is that no measured quantity is typed
    into prose. (Cross-model review.)
    """
    cell = probe["remote_code"]["trust_remote_code=False"]
    return (
        f"{cell[field]:.6f}" if field == "cosine_between_unrelated_records" else f"{cell[field]:g}"
    )


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
    noise_table, uninformative, informative, narrow, inverted = _noise_floor_table(tuned, base)
    # The model the control's intervals are ACTUALLY paired against, so the prose
    # below names it instead of "the tuned models".
    reference = _control_reference(base)
    # Both are unusable for ranking, for OPPOSITE reasons: `blind` fails to
    # exclude zero, `inverted` excludes it the wrong way. One sentence cannot
    # justify both.
    blind = [b for b in uninformative if b not in inverted]

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
        f"`{BASE_BASELINE}`, and a study-B interval must never be compared with a study-A "
        f"one. That baseline shares its 350M backbone with **`{MATCHED_BACKBONE_ENCODER}` "
        f'only**, so the reading "what retrieval tuning bought on this backbone" applies '
        f"to that pair alone. `LiquidAI/LFM2.5-Encoder-230M` declares a different, smaller "
        f"backbone and `{CONTROL}` has no training at all, so their deltas confound tuning "
        f"with model size or with the absence of weights entirely. Each table states its "
        f"own baseline in the heading.",
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
        "the same records, so an arm without one cannot be tested at all — it is silently "
        "absent from the two lines above rather than counted as a loss:",
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
                # Two different reasons an interval is missing, and only one of
                # them is "the baseline never ran this arm". Saying that of a
                # cleared interval would be a fabricated explanation for a cell
                # the baseline demonstrably did measure. (Cross-model review.)
                + (
                    " The baseline **did** run this arm, so the interval is missing rather "
                    "than impossible: `merge_rows()` clears a challenger's `vs_reference_*` "
                    "when the reference is re-measured, and a bootstrap needs enough gold "
                    "clusters. Re-run this cell against the current reference to restore it."
                    if baseline_ran
                    else " The baseline has no `" + arm + "` arm here, so no paired test "
                    "exists to run."
                )
                for benchmark, arm, recall, base, baseline_ran in unpaired
            ]
            or ["- None — every arm has a counterpart in the baseline."]
        ),
        "",
        'So on any benchmark listed above, the honest statement is *"behind on the arms '
        'that could be tested"*, not *"behind"*. Where the cause is a missing counterpart, '
        "the untested arm is the checkpoint's own documented prompt — which is where a "
        "vendor would expect it to look best; closing that gap needs the baseline re-run "
        "under the same prompts, which this study did not do.",
        "",
        "**It cannot become langres's default regardless of how it scores** — see the "
        "licence section. That is a legal constraint, not a measurement.",
        "",
        "## Was the checkpoint even the thing measured?",
        "",
        *_probe_staleness(probe),
        "This has to come before any score. All three checkpoints declare "
        f'`model_type: "lfm2"`, which the transformers the probe ran under '
        f"({probe['transformers_version']}) implements natively "
        f"(`lfm2` in `CONFIG_MAPPING_NAMES`: **{probe['lfm2_natively_implemented']}**) as a "
        "**causal decoder**, while pointing `auto_map.AutoModel` at their own "
        "*bidirectional* class. Dropping `trust_remote_code` therefore substitutes a "
        "different architecture in silence — no exception, no warning.",
        "",
        remote_table,
        "",
        # Interpolated from the same JSON the table above reads, not typed. The
        # driver refreshes the probe before every render, so a hard-coded
        # "1.000000" and "exactly 0" would keep asserting last month's numbers
        # under a table showing this run's -- in the one file whose whole premise
        # is that no measured quantity is typed into prose. (Cross-model review.)
        'The untrusted load is not "slightly degraded". This checkpoint pools the **CLS** '
        "token (`1_Pooling/config.json`), and under causal attention the first token is a "
        "function of itself alone — so every text in the corpus collapses onto one vector "
        f"(cosine between two unrelated products = {_untrusted(probe, 'cosine_between_unrelated_records')}) "
        f"and the prompt changes nothing at all (shift {_untrusted(probe, 'max_abs_prompt_shift')}). "
        "A sweep would have published that as a blocking recall and attributed it to the "
        "model.",
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
            f"{', '.join(f'`{b}`' for b in blind) if blind else '**none**'}.** "
            f"The control's paired interval there does not exclude zero, so a model ranking "
            f"read off {'it' if len(blind) == 1 else 'them'} is not evidence about any "
            f"embedder."
        ),
        "",
        # The OTHER way a benchmark becomes unusable, and the one the sentence
        # above cannot describe: the interval excludes zero and the CROSS-STUDY
        # GAP still runs the wrong way. Filed under the same "no ranking" heading,
        # justified separately. (Cross-model review.)
        #
        # The two facts stay apart. `inverted` is selected by `margin <= 0` — a
        # cross-study POINT gap — while `separates` accepts either interval
        # direction, so the earlier "the interval excludes zero on the wrong side"
        # welded a claim about the point gap onto the paired interval. After a
        # study-A-only rerun the best tuned score can drop below the older control
        # while that control's own interval still runs significantly BELOW its
        # study-B baseline, and the sentence would have called that interval
        # wrong-sided. The direction is read per benchmark now. (Cross-model
        # review.)
        *(
            [
                f"**Benchmarks where the untrained control outscores every tuned model: "
                f"{', '.join(f'`{b}` (its paired interval runs {_control_direction(base, b) or "either side of zero"} the paired baseline)' for b in inverted)}.** "
                f"Not the same failure, and two separate facts: the *cross-study point gap* "
                f"to the best tuned model is negative — that is what puts a benchmark here — "
                f"while the control's own interval is a different comparison against a "
                f"different model, with the direction noted beside each. Either way it is a "
                f"result about the benchmark, not about any embedder: a ranking read off "
                f"{'it' if len(inverted) == 1 else 'them'} is inverted, not merely noisy.",
                "",
            ]
            if inverted
            else []
        ),
        (
            f"**Benchmarks that do separate the two, but narrowly "
            f"(&lt;{NARROW_RANGE:.2f} of range): "
            # Each one annotated with the direction its OWN interval measured.
            # NOT "below the tuned models": the interval this verdict reads is
            # paired with study B's baseline and nothing else, and the best tuned
            # model in the row can be a study-A model with no paired control
            # interval at all -- as it is on `walmart_amazon`, where the gap runs
            # to `intfloat/e5-base-v2` and the interval to
            # `LiquidAI/LFM2.5-Embedding-350M`.
            #
            # And NOT "below" as a constant: the branch is reached whenever the
            # interval excludes zero in EITHER direction, so a rerun where the
            # control sits significantly ABOVE its paired baseline while the
            # cross-study best stays a little higher still lands here -- and the
            # sentence would then assert the exact opposite of the rows. Both
            # halves of this claim are now read off the data. (Cross-model
            # review.)
            + (
                # No inner `**`: this whole line is already inside a bold span,
                # and a nested one breaks it in every renderer.
                ", ".join(
                    f"`{b}` (control {_control_direction(base, b) or 'not separated from'} "
                    f"the paired baseline)"
                    for b in narrow
                )
                if narrow
                else "none"
            )
            + ".** "
            "This is a statement about **resolution, not significance**. Each of those "
            + (
                f"intervals clears zero against `{reference}` — the model it is computed "
                f"against, and the only one this column can speak for — in the direction "
                f"marked. "
                if reference
                else "intervals clears zero against the paired baseline it is computed "
                "against — the only model this column can speak for — in the direction "
                "marked. "
            )
            + "That separation is real. But the "
            "whole trained-vs-random range is small, so a reported margin can be a large "
            "fraction of it, and a ranking built from differences of that size is fragile "
            "rather than meaningless."
        ),
        "",
        (
            f"Benchmarks that separate the two with room to spare: "
            f"{', '.join(f'`{b}`' for b in informative if b not in narrow) or '**none**'}."
        ),
        # `blind`, not `uninformative`: "genuinely fails to separate" is false of
        # an inverted benchmark, whose interval separates the two perfectly well.
        *_correction_paragraph(base, blind, narrow),
        "",
        *_pretraining_section(base),
        "",
        # Named benchmark, conditional claim: the sentence is only true while the
        # measured intervals actually put fodors_zagat in `blind`. Printing it
        # unconditionally would restate a previous run's result as this one's, and
        # `blind` rather than `uninformative` because the paragraph's reason --
        # "cannot tell a trained retriever from noise" -- is the interval failing
        # to exclude zero, not the inverted case.
        *(
            [
                "This reframes `fodors_zagat` specifically. It was already labelled "
                "*saturated* — every usable embedder scoring near the ceiling. The control "
                "shows it is stronger than that: it is **uninformative in the strict "
                "sense**, because an untrained network also scores near the ceiling. "
                "Saturation says the models agree; this says the benchmark cannot tell a "
                "trained retriever from noise. It must never be cited as evidence that one "
                "embedder beats another.",
                "",
            ]
            if "fodors_zagat" in blind
            else []
        ),
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
        "`20260729_lfm25_license.txt`. Section 5 in full — the section that decides this "
        "— with the terms it turns on, quoted from the file:",
        "",
    ]
    parts += [f"> {clause}" for clause in _licence_clauses()]
    parts += [
        "",
        "**Consequence.** langres is Apache-2.0, which carries no such restriction. The "
        "restriction is on **Commercial Use**, not on all use, and §5(c) exempts a "
        "Qualified Non-Profit Organization using the model for non-commercial or research "
        "purposes — but neither narrows the problem for a *default*, because a default is "
        "the path taken by users who never read this section. Making any of these the "
        "`DEFAULT_EMBEDDING_MODEL` would put the commercial use of every langres user "
        "whose legal entity is at or above $10M annual revenue outside the licence, "
        "silently, on the path they did not choose. So:",
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
        "bash examples/research/run_lfm25.sh          # both sweeps, load probe + this write-up",
        "LFM25_STUDY=b bash examples/research/run_lfm25.sh     # just the base-encoder study",
        "uv run python examples/research/lfm25_load_probe.py   # the loading artifact, on its own",
        "```",
        "",
        # Prospective ("a sweep run this way produces"), never retrospective
        # ("these artifacts are"). Stated as fact it contradicted THIS document:
        # the loading section above already warns that the committed probe
        # postdates the measurement window, and eight paragraphs later the same
        # file asserted the two describe one environment. A generated document
        # that argues with itself is worse than one that admits the gap.
        # (Cross-model review.)
        "Run that way, the driver refreshes the load probe **before** rendering and inside "
        "the same measurement window as the rows, so the loading section and the scores "
        "describe one environment. **Whether that holds for the artifacts committed here is "
        "a separate question, answered by the loading section itself** — it compares the "
        "probe's `captured_at` against both ends of the window and says so when they do not "
        "line up. The standalone command exists for re-checking a checkpoint without "
        "re-measuring; listing it *after* the render, as this block used to, told readers "
        "to refresh the probe once the document quoting it had already been written.",
        "",
        "There is **no skip-completed logic**: `merge_rows()` replaces a re-measured "
        "cell, and re-measuring the reference model clears every other model's "
        "`vs_reference_*` on the benchmarks it touches. Re-running a finished study is "
        "a re-do, not a resume — use `LFM25_STUDY` to name the part that is missing.",
        "",
    ]
    return "\n".join(parts)


#: Discard the tracked write-up deliberately, matching `run_lfm25.sh`.
FORCE_ENV = "LFM25_FORCE"

#: Set by a driver that has just written the provenance sidecar and will commit it
#: in the SAME commit as the report. Narrow on purpose: it exempts exactly one
#: input, and only for the caller that owns both halves of that commit. A
#: standalone render does not set it and is therefore held to the same rule as
#: every other input the report quotes.
PROVENANCE_PENDING_ENV = "LFM25_PROVENANCE_PENDING"


def _refuse_to_overwrite_uncommitted() -> None:
    """Stop before destroying uncommitted edits to the generated write-up.

    Third site of one defect. ``run_lfm25.sh`` refuses to start over a dirty
    write-up, but ``run_ladder.sh`` guards only the per-study artifacts, and both
    this module's documented standalone invocation and the study-A resume — which
    now calls it — reach this writer with no protection at all. Guarding the
    CALLER protects the callers you remembered; guarding the WRITER protects the
    file. (Cross-model review.)

    Reuses ``write_provenance._uncommitted`` rather than reimplementing it: it
    already treats a path git cannot describe as pending, and a second copy of a
    safety check is a second thing to drift.
    """
    if os.environ.get(FORCE_ENV) == "1":
        logger.warning("%s=1: overwriting %s without checking", FORCE_ENV, OUTPUT)
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from write_provenance import _uncommitted

    lost = _uncommitted(OUTPUT)
    if not lost:
        return
    raise SystemExit(
        f"REFUSING to overwrite {OUTPUT}: it holds uncommitted changes ({', '.join(lost)}).\n"
        "  If those are a HAND EDIT, keep them: commit, or copy the file outside this\n"
        "  worktree, then re-render.\n"
        "  If they are simply the PREVIOUS RENDER's own output — re-rendering twice\n"
        "  before committing is the ordinary way to check this file is stable — that is\n"
        "  what the force flag is for. This guard cannot tell the two apart, so it asks:\n"
        f"    {FORCE_ENV}=1 uv run python examples/research/lfm25_report.py"
    )


def _refuse_uncommitted_inputs() -> None:
    """Refuse to quote bytes that no commit holds.

    This lives HERE, not in the drivers, for the reason the writer guard does:
    the renderer is the one thing that knows every file it reads. `run_lfm25.sh`
    grew this check inline, and the study-A resume then reached the same renderer
    without it — its preflight covers only the tuned artifacts, while this reads
    the base rows and the load probe too. Two drivers, one of them guarded, is
    how the first five of these findings happened. (Cross-model review.)

    The failure it prevents is quiet: the generated document quotes uncommitted
    numbers, and the commit carrying it holds only the document, so the published
    file cannot be regenerated from the commit it ships in.
    """
    if os.environ.get(FORCE_ENV) == "1":
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from write_provenance import _uncommitted

    # The sidecar is a READ input -- `_provenance_section`, `_window_bounds` and
    # `_cross_study_caveat` all quote it -- but it is the one input a driver
    # legitimately writes moments before rendering, then commits together with the
    # report. Guarding it unconditionally would refuse every real sweep; exempting
    # it unconditionally (which is what the first version of this list did, with a
    # test asserting the exemption) lets a STANDALONE render publish provenance
    # text derived from bytes no commit holds. So the allowance is explicit, and
    # the driver has to claim it. (Cross-model review.)
    guarded = list(RENDER_INPUTS)
    if os.environ.get(PROVENANCE_PENDING_ENV) != "1":
        guarded.append(PROVENANCE)
    moved = [name for path in guarded for name in _uncommitted(path)]
    if not moved:
        return
    raise SystemExit(
        "REFUSING to render: these inputs hold uncommitted changes right now "
        f"({', '.join(moved)}).\n"
        "  The write-up would quote bytes no commit holds, so it could not be "
        "regenerated from\n"
        "  the commit that carries it. Commit them, then re-render. To render anyway:\n"
        f"    {FORCE_ENV}=1 uv run python examples/research/lfm25_report.py"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _refuse_uncommitted_inputs()
    _refuse_to_overwrite_uncommitted()
    OUTPUT.write_text(render())
    logger.info("wrote %s", OUTPUT)


if __name__ == "__main__":
    main()
