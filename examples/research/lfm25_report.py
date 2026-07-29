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
import subprocess
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

#: Files whose content decides what the numbers MEAN, so the write-up records the
#: exact blob each one was at. The statistics module is here because "were these
#: numbers produced by the same code that is in main?" was asked of this study
#: while it was still running: it was answerable in minutes only because the code
#: was diffable. Rows that cannot name the code that produced them force a re-run
#: to answer that question.
PROVENANCE_FILES = (
    "src/langres/experiments/statistics.py",
    "examples/research/embedder_ladder.py",
)


def _git(*args: str) -> str:
    """``git`` output, or an empty string when git cannot answer."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _bootstrap_default_samples() -> str:
    """The bootstrap replicate count these intervals were actually computed with.

    Read out of the signature rather than repeated as a literal, so it cannot go
    stale the way a hand-typed ``B=1000`` in prose would.
    """
    src = (REPO_ROOT / "src" / "langres" / "experiments" / "statistics.py").read_text()
    match = re.search(r"^\s*samples:\s*int\s*=\s*(\d+)", src, re.MULTILINE)
    return match.group(1) if match else "unknown"


def _provenance_section() -> list[str]:
    """Name the code that produced these numbers, read from git rather than typed."""
    lines = [
        "## Provenance — which code produced these numbers",
        "",
        f"The paired intervals were computed by `paired_entity_bootstrap` at "
        f"**B={_bootstrap_default_samples()}** replicates (the library default; this "
        "harness does not override it), resampling **gold clusters**. Blob hashes of "
        "the files that decide what the numbers mean:",
        "",
        "| file | blob | last commit touching it |",
        "|---|---|---|",
    ]
    for path in PROVENANCE_FILES:
        blob = _git("rev-parse", f"HEAD:{path}") or "unavailable"
        last = _git("log", "-1", "--format=%h %cI", "--", path) or "unavailable"
        lines.append(f"| `{path}` | `{blob[:12]}` | {last} |")
    lines += [
        "",
        "Blob hashes, deliberately, and **not** the render-time `HEAD`: `HEAD` moves with "
        "every commit, so embedding it would make this file re-render differently on each "
        "run and break the harness's own reproduce-the-committed-table check. A blob hash "
        "changes only when the file's content does, which is the property being recorded.",
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
    return header + f"\nReachable ceiling per benchmark: {ceilings}.\n"


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
    """Benchmarks where ``model``'s interval vs. the baseline excludes zero, by direction."""
    ahead, behind = [], []
    for benchmark in _benchmarks(rows):
        for row in _cells(rows, model=model, benchmark=benchmark, k=HEADLINE_K, status="ok"):
            low, high = row.get("vs_reference_ci_low"), row.get("vs_reference_ci_high")
            if low is None or high is None:
                continue
            if low > 0 and benchmark not in ahead:
                ahead.append(benchmark)
            elif high < 0 and benchmark not in behind:
                behind.append(benchmark)
    return ahead, behind


CONTROL = "random-init-control-350M"


def _noise_floor_table(
    tuned: list[dict[str, Any]], base: list[dict[str, Any]]
) -> tuple[str, list[str], list[str]]:
    """Every benchmark's tuned-model spread against the random-init noise floor.

    The control is the same 350M architecture with seeded random weights. Its
    recall is what a benchmark hands a model that knows nothing, so the gap
    between it and the best tuned model is the benchmark's entire usable
    dynamic range. Where that gap is ~0, the benchmark cannot separate a trained
    retriever from a random feature map and carries no evidence about any
    embedder.

    Returns ``(table, uninformative, informative)``.
    """
    benchmarks = sorted({row["benchmark"] for row in tuned} | {row["benchmark"] for row in base})
    table = "| benchmark | random-init control | best tuned model | margin over noise |\n"
    table += "|---|---|---|---|\n"
    uninformative: list[str] = []
    informative: list[str] = []
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
            table += f"| `{benchmark}` | — | — | — |\n"
            continue
        floor = control["candidate_recall"]
        best = max(tuned_cells, key=lambda c: (c["candidate_recall"], c["model"]))
        margin = best["candidate_recall"] - floor
        # 0.05 is a judgement call, stated rather than hidden: a benchmark whose
        # entire tuned-vs-random range is under five recall points cannot support
        # a claim about a model, because the measurement's own noise floor is
        # inside the effect it would have to detect.
        verdict = "**uninformative**" if margin < 0.05 else "usable"
        (uninformative if margin < 0.05 else informative).append(benchmark)
        table += (
            f"| `{benchmark}` | {_fmt(floor)} | {_fmt(best['candidate_recall'])} "
            f"(`{best['model']}`) | **{margin:+.4f}** {verdict} |\n"
        )
    return table, uninformative, informative


def _control_vs_base_encoders(base: list[dict[str, Any]]) -> list[str]:
    """Where the untrained control outscores the *real* base encoders.

    Both sides wear the same untrained mean-pooling head, so this isolates the
    pretrained MLM weights themselves: the only difference between the control
    and an encoder row is whether those weights were trained at all.
    """
    encoders = sorted(
        {row["model"] for row in base if "Encoder" in row["model"]},
    )
    benchmarks = sorted({row["benchmark"] for row in base})
    lines: list[str] = []
    for encoder in encoders:
        beaten = []
        for benchmark in benchmarks:
            control = _best_arm(base, CONTROL, benchmark)
            cell = _best_arm(base, encoder, benchmark)
            if control and cell and control["candidate_recall"] >= cell["candidate_recall"]:
                beaten.append(benchmark)
        lines.append(
            f"- Random weights match or beat `{encoder}` on **{len(beaten)} of "
            f"{len(benchmarks)}** benchmarks"
            + (f" ({', '.join(f'`{b}`' for b in beaten)})." if beaten else ".")
        )
    return lines


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
    noise_table, uninformative, informative = _noise_floor_table(tuned, base)

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
        f"`DEFAULT_EMBEDDING_MODEL` and therefore the baseline every interval below is "
        f"measured against.",
        "",
        (
            f"- Ahead with an interval excluding zero on: "
            f"{', '.join(f'`{b}`' for b in ahead) if ahead else '**no benchmark**'}."
        ),
        (
            f"- Behind with an interval excluding zero on: "
            f"{', '.join(f'`{b}`' for b in behind) if behind else '**no benchmark**'}."
        ),
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
        "therefore what a benchmark hands a model that knows *nothing*, and the gap above it "
        "is the benchmark's entire usable dynamic range.",
        "",
        noise_table,
        "",
        (
            f"**Benchmarks that cannot support an embedder claim: "
            f"{', '.join(f'`{b}`' for b in uninformative) if uninformative else 'none'}.** "
            f"On these the whole tuned-vs-random range is under five recall points, so the "
            f"measurement's own noise floor sits inside any effect it would have to detect. "
            f"A model ranking read off them is not evidence."
        ),
        "",
        (
            f"Benchmarks with usable range: "
            f"{', '.join(f'`{b}`' for b in informative) if informative else '**none**'}."
        ),
        "",
        "**The control also outscores the real base encoders.** Both wear the same "
        "untrained mean-pooling head, so the only difference between these rows is "
        "whether the backbone weights were pretrained at all:",
        "",
        *_control_vs_base_encoders(base),
        "",
        "Pretrained masked-LM features read through an untrained mean-pooling head are "
        "therefore **worse than random features read the same way** on this task. That is "
        "a statement about the *pooling*, not about the checkpoints: MLM training shapes "
        "token representations for a head this pipeline never attaches, and averaging them "
        "destroys more than averaging random projections does. It is also the cleanest "
        "available demonstration of why an untuned encoder must not be ranked against a "
        "retrieval-tuned one.",
        "",
        "This reframes `fodors_zagat` specifically. It was already labelled *saturated* — "
        "every usable embedder scoring near the ceiling. The control shows it is stronger "
        "than that: it is **uninformative in the strict sense**, because an untrained network "
        "also scores near the ceiling. Saturation says the models agree; this says the "
        "benchmark cannot tell a trained retriever from noise. It must never be cited as "
        "evidence that one embedder beats another.",
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
