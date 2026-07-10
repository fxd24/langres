"""Peeters, Steiner & Bizer (EDBT 2025) LLM-EM replication — offline replay + live paid run.

Three modes over the SAME abt-buy `domain-complex-force` slice (arXiv 2310.11244
v4 Table 2), sharing one prompt template + Peeters serializer so what the paid
run pays for is exactly what the `$0` replay validated:

* ``--mode replay`` (default, **$0**, no key): replays the authors' *archived*
  raw model answers through langres — downloads their public answer archive to a
  temp cache, parses each stored answer with our unified parser, aligns to the
  gold labels of our regenerated pair-set slice, and scores pairwise P/R/F1 with
  ``langres.core.metrics.classify_pairs`` (validating *our* metric code). It also
  verifies the prompt round-trip byte-for-byte. Target: ``abt-buy`` /
  ``gpt-4-0613`` / ``domain-complex-force`` → **F1 95.15**.

* ``--mode dry-run`` (**$0**, no key): renders every one of the 1206 pairs
  through the *live* path's template + serializer and reports the token counts +
  a cost estimate — **zero API calls**. Use it to preview cost and confirm the
  rendered prompt matches the archived one before spending anything.

* ``--mode live`` (**PAID**, off by default): runs a real ``LLMJudge`` (the
  paper's ``domain-complex-force`` template, the Peeters per-dataset
  ``record_serializer``, ``response_parser=parse_binary_yes_no``,
  ``temperature=0.0``) over the 1206 regenerated Abt-Buy pairs, under a hard
  :class:`~langres.clients.openrouter.SpendMonitor` cap. It is guarded three
  ways: an explicit ``--yes-spend-money`` flag, a priced-model assertion, and the
  cap. It reports F1 + the aggregated :class:`~langres.core.usage.LLMUsage`
  vector + the **real OpenRouter-billed** cost (``cost_is_real``) per model.

The paid run races exactly two dated snapshots (Abt-Buy domain-complex-force):

* ``openrouter/openai/gpt-4o-mini-2024-07-18`` — the paper's "GPT-mini", published
  F1 **90.95** (P=89.25, R=92.72).
* ``openrouter/openai/gpt-4o-2024-08-06`` — the paper's "GPT-4o", published F1
  **90.47** (P=83.27, R=99.03). (An earlier draft of this harness carried a wrong
  89.33 for this cell; arXiv v4 Table 2 and the authors' ``results.xlsx`` agree on
  90.47.)

``gpt-4-0613`` (the F1 **95.15** cell the offline replay reproduces) would cost
~$3.15 to run live and was **deliberately declined** — not worth the spend, and
it retires 2026-10-23. ``gpt-3.5-turbo-0613`` / ``-0301`` were shut down
2024-09-13, so neither is a live option.

Spend safety (read twice):

* ``--mode live`` makes **no** network call until AFTER it prints the cost
  estimate and sees ``--yes-spend-money``. Each pair's real cost is charged to
  one :class:`SpendMonitor` and ``check()``\\ ed, so cumulative spend cannot
  cross ``--budget`` (default **$1.00** for both models combined; measured total
  ≈ $0.29, a ~3.4x margin).
* Every model MUST be priced in
  :data:`~langres.clients.openrouter.PRICES_PER_1M` — an unpriced model silently
  contributes $0 to the cap, so the script refuses to start without a price entry.
* ``OPENROUTER_API_KEY`` is **required** for ``--mode live`` and is **NOT present
  in this environment**; the paid path fails fast with a clear message rather
  than silently falling back to another provider or an unpriced model.
* The one deviation from the paper's setup is that we route the same dated model
  snapshot through **OpenRouter** instead of calling OpenAI directly. The live
  judge therefore pins :data:`LIVE_PROVIDER`
  (``{"order": ["OpenAI"], "allow_fallbacks": False}``) so OpenRouter must serve
  the request from OpenAI's own backend and cannot silently substitute a
  different provider/quantization of the model.

Cheaper trials + per-pair agreement against the archive:

* ``--limit N`` runs a **stratified** subset of ``N`` pairs (preserving the
  17.1% positive ratio, deterministic under ``--seed``, default 0) instead of all
  1206 — so a live trial can be sized to a few cents. Applies to
  ``dry-run``/``live``/``replay``.
* ``--compare-archived`` (``--mode live`` only) judges each pair live **and**
  compares our parsed verdict to the authors' archived answer for the same model,
  reporting the per-pair agreement rate, a 2x2 confusion of ours-vs-theirs, up to
  10 concrete disagreeing pairs, our F1/P/R on the judged subset next to *their*
  F1/P/R recomputed on that same subset (and the published full-set number), plus
  the usage vector + real billed cost. It fails loudly if our rendered prompt does
  not match the archived one (a mismatch means the alignment is off).

Crash-safe & resumable (paid runs never lose a billed call):

* Every judged pair is appended — ``flush`` + ``os.fsync`` — as one JSON line to a
  per-(model, dataset, prompt-design) JSONL under ``--results-dir`` (default the
  gitignored ``tmp/peeters/``) BEFORE the next paid call, so a ``SIGKILL`` cannot
  lose an already-billed pair (the earlier run that persisted only at the very end
  lost ~$0.187). Each row carries ids, gold, our raw answer + parsed verdict, the
  :class:`LLMUsage` vector, and ``cost_usd``/``cost_is_real``/``provider``/``model``.
* **Resume** re-reads that JSONL and **skips already-judged pairs** — re-running a
  completed model costs **$0** and makes ZERO API calls. The hard spend cap accounts
  for spend already recorded (:meth:`PeetersResultStore.spent`), so the aggregate cap
  holds across resumes and a resumed run cannot exceed it.
* The final report is computed **from the JSONL**, so the numbers are identical
  whether the run finished in one pass or several. ``--report-only`` recomputes and
  prints the full report (agreement/confusion/disagreements + F1 with
  ``--compare-archived``) from existing results with zero API calls — the way to get
  the final table after a paid run.
* Progress is printed every ``--progress-every`` pairs (running spend + running
  archive-agreement). Drive the script with ``python -u`` (or it line-buffers stdout
  itself) so that progress is not lost on a kill.

Credence probe (``--logprobs``):

* ``--logprobs`` (``--mode live``) runs the SAME live judge with
  ``LLMJudge(confidence="logprob")`` — it requests first-token logprobs and records
  a P(Yes) credence (``p_yes`` / ``leaked_mass`` / ``p_yes_is_bound``, plus
  ``correct = verdict == gold``) in each **v2** row, so "does the model's own
  first-token credence predict its errors?" is answerable from the rows alone. It
  is an evidence-gathering probe: **nothing is added to ``PairwiseJudgement``**. On
  the single-token "Yes"/"No" answer, ``top_logprobs`` adds zero output tokens, so
  the probe re-runs at ~the replication cost. Probe rows land in a distinct
  ``…__logprobs.jsonl`` (a contamination firewall — it cannot overwrite the
  committed replication rows) and ``--results-dir`` defaults to the **committed**
  ``examples/research/results/peeters`` so the paid probe's rows are durable.

Data licensing: MatchGPT ships no LICENSE (``license: null``); langres is
Apache-2.0. Nothing from MatchGPT is vendored — the ~186 MB answer archive is
downloaded transiently to a cache dir (``--cache-dir``) and never committed. Our
pair-set slice is regenerated from our own already-vendored DeepMatcher CSVs.

Usage::

    # $0, no key:
    uv run python examples/research/peeters_llm_em_replication.py
    uv run python examples/research/peeters_llm_em_replication.py --mode dry-run
    uv run python examples/research/peeters_llm_em_replication.py --mode dry-run --limit 150
    # PAID (run with the sandbox disabled — OpenRouter is a network call). ``-u`` +
    # the per-pair JSONL make it crash-safe: a kill loses nothing, just re-run to resume.
    python -u examples/research/peeters_llm_em_replication.py --mode live \\
        --compare-archived --yes-spend-money
    # ...killed partway? Re-run the SAME command — judged pairs are skipped ($0), the
    # cap counts prior spend, and it picks up where it left off.
    # Get the final table from the persisted results, zero API calls:
    python examples/research/peeters_llm_em_replication.py --report-only --compare-archived
    # PAID, sized to a 150-pair stratified subset, with per-pair archive agreement:
    python -u examples/research/peeters_llm_em_replication.py --mode live \\
        --model openrouter/openai/gpt-4o-mini-2024-07-18 --limit 150 \\
        --compare-archived --yes-spend-money
    # PAID credence probe over both models (writes …__logprobs.jsonl to the committed dir):
    python -u examples/research/peeters_llm_em_replication.py --mode live --logprobs \\
        --yes-spend-money

``print`` is allowed in examples (this is an operator tool).
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import random
import sys
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langres.clients.openrouter import (
    PRICES_PER_1M,
    BudgetExceeded,
    SpendMonitor,
)
from langres.core.metrics import classify_pairs
from langres.core.models import PairwiseJudgement
from langres.core.modules.llm_judge import LLMJudge, parse_binary_yes_no
from langres.core.usage import LLMUsage
from langres.data.peeters import (
    PeetersReplicationSpec,
    build_candidates,
    build_llm_prompt_template,
    get_peeters_replication,
    gold_match_pairs,
    judgements_from_answers,
    list_peeters_replications,
    load_peeters_sample,
    make_record_serializer,
    parse_binary_answer,
    render_sample_prompts,
)

logger = logging.getLogger("peeters_llm_em")

# ---------------------------------------------------------------------------
# Offline replay (archived answers) — constants + helpers
# ---------------------------------------------------------------------------

#: The authors' Git-LFS answer archive (real bytes, not an LFS pointer).
ARCHIVE_URL = (
    "https://media.githubusercontent.com/media/wbsg-uni-mannheim/MatchGPT/"
    "main/LLMForEM/prompts-and-answers/prompts_and_answers.zip"
)

#: Published arXiv v4 Table 2 F1 (%) per (dataset, model, prompt-design) — the
#: *offline replay* cells this harness can assert against (their archived answers).
PUBLISHED_F1 = {
    ("abt-buy", "gpt-4-0613", "domain-complex-force"): 95.15,
}

#: 2-decimal reporting tolerance for the offline replay: our exact F1 (e.g.
#: 95.1456) must round to the published 2-dp value; 0.05 absorbs that rounding.
F1_TOLERANCE = 0.05


def member_name(dataset: str, prompt_design: str, model: str) -> str:
    """Archive member for one (dataset, prompt-design, model) run."""
    return f"{dataset}-sampled-gs_{prompt_design}_default_{model}_run-1.jsonl"


def ensure_answers(cache_dir: Path, dataset: str, prompt_design: str, model: str) -> Path:
    """Return the extracted JSONL for a run, downloading/extracting on demand.

    The 186 MB archive is downloaded once (skipped if already cached), and each
    requested member is extracted once.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    member = member_name(dataset, prompt_design, model)
    extracted = cache_dir / member
    if extracted.exists():
        print(f"[cache] using extracted answers: {extracted}")
        return extracted

    archive = cache_dir / "prompts_and_answers.zip"
    if not archive.exists():
        print(f"[download] {ARCHIVE_URL}\n           -> {archive} (~186 MB, one time)")
        urllib.request.urlretrieve(ARCHIVE_URL, archive)  # noqa: S310 (trusted host)
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        if member not in names:
            available = "\n  ".join(n for n in names if dataset in n) or "(none for dataset)"
            raise SystemExit(
                f"member {member!r} not in archive. Available for {dataset}:\n  {available}"
            )
        print(f"[extract] {member}")
        zf.extract(member, cache_dir)
    return extracted


def replay(
    dataset: str,
    model: str,
    prompt_design: str,
    cache_dir: Path,
    *,
    limit: int | None = None,
    seed: int = 0,
) -> None:
    """Replay one archived run and report prompt round-trip + pairwise F1 ($0).

    ``limit`` runs a stratified subset (via :func:`stratified_subset_indices`) —
    handy for a quick look; the published-F1 assertion is a full-set claim, so it
    is reported (not asserted) when the run is limited.
    """
    spec = get_peeters_replication(dataset)
    prompts = render_sample_prompts(spec)  # our records + serializer, in sample order

    answers_path = ensure_answers(cache_dir, dataset, prompt_design, model)
    archived = [json.loads(line) for line in answers_path.read_text().splitlines() if line.strip()]

    if len(archived) != len(prompts):
        raise SystemExit(
            f"archive has {len(archived)} lines but our sample has {len(prompts)} pairs — "
            "alignment would be wrong; aborting."
        )

    if limit is not None:
        indices = stratified_subset_indices([p.label for p in prompts], limit, seed)
        prompts = [prompts[i] for i in indices]
        archived = [archived[i] for i in indices]

    # --- Prompt round-trip: our rendered prompt vs their archived prompt -------
    exact = sum(1 for p, rec in zip(prompts, archived, strict=True) if p.prompt == rec["prompt"])
    round_trip = 100.0 * exact / len(prompts)

    # --- Replay: parse their answers, score with OUR metric code ---------------
    raw_answers = [rec["answer"] for rec in archived]
    judgements = judgements_from_answers(prompts, raw_answers)
    metrics = classify_pairs(judgements, gold_match_pairs(prompts), threshold=0.5)
    f1_pct = metrics.f1 * 100.0

    print("\n" + "=" * 72)
    print(f"Peeters LLM-EM replay — {dataset} / {model} / {prompt_design}")
    print("=" * 72)
    print(f"pairs                : {len(prompts)}  ({len(gold_match_pairs(prompts))} positive)")
    print(f"prompt round-trip    : {exact}/{len(prompts)} = {round_trip:.2f}% byte-exact")
    print(
        f"pairwise (via classify_pairs): P={metrics.precision * 100:.2f}  "
        f"R={metrics.recall * 100:.2f}  F1={f1_pct:.2f}"
    )
    print(f"  tp={metrics.tp} fp={metrics.fp} fn={metrics.fn}")

    target = PUBLISHED_F1.get((dataset, model, prompt_design))
    if target is not None and limit is not None:
        print(
            f"published Table 2 F1 : {target:.2f}  (full-set claim — not asserted on a --limit subset)"
        )
    elif target is not None:
        delta = abs(f1_pct - target)
        ok = delta <= F1_TOLERANCE
        print(f"published Table 2 F1 : {target:.2f}  (Δ={delta:.4f}, tol={F1_TOLERANCE})")
        if not ok:
            raise SystemExit(f"FAIL: replayed F1 {f1_pct:.2f} != published {target:.2f}")
        print("REPRODUCED ✓")
    else:
        print("(no published-F1 assertion registered for this cell — reported only)")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Live (paid) run — constants + the testable core
# ---------------------------------------------------------------------------

#: The dated snapshots the paid run races, keyed by OpenRouter routing id (each
#: MUST be a key of PRICES_PER_1M). Value = the published Abt-Buy
#: domain-complex-force F1 (arXiv v4 Table 2), shown as a column so the delta is
#: visible at a glance.
PAID_MODELS: dict[str, float] = {
    "openrouter/openai/gpt-4o-mini-2024-07-18": 90.95,  # paper "GPT-mini"
    "openrouter/openai/gpt-4o-2024-08-06": 90.47,  # paper "GPT-4o" (corrected from 89.33)
}

#: OpenRouter provider-routing block pinned on the live judge. Our sole deviation
#: from the paper's setup is routing the same dated snapshot through OpenRouter
#: rather than calling OpenAI directly; pinning ``order: ["OpenAI"]`` +
#: ``allow_fallbacks: False`` forces OpenRouter to serve the request from OpenAI's
#: own backend, so a different provider/quantization can't silently be substituted
#: and change the F1/cost we report. Sent as ``LLMJudge(provider=...)`` ->
#: ``extra_body["provider"]``.
LIVE_PROVIDER: dict[str, Any] = {"order": ["OpenAI"], "allow_fallbacks": False}

#: Hard spend cap (USD) for the two-model run combined. Measured total ≈ $0.29,
#: so $1.00 is a ~3.4x margin. ``main`` refuses anything above the ceiling.
DEFAULT_BUDGET_USD = 1.00
BUDGET_CEILING_USD = 2.00

#: Estimated output tokens per pair for the dry-run cost estimate: the binary
#: protocol answers with a single "Yes"/"No" word (~1–2 o200k tokens). The live
#: run meters the REAL output cost; this only sizes the pre-flight estimate.
_EST_OUTPUT_TOKENS_PER_PROMPT = 2

#: Default per-pair progress cadence (print a line every N judged pairs).
DEFAULT_PROGRESS_EVERY = 50

#: Prompt design pinned for the live/compare paid runs (arXiv v4 Table 2 target).
LIVE_PROMPT_DESIGN = "domain-complex-force"

#: JSONL result-row schema version (independent of ``JudgementLog``'s ``v``).
#: v2 adds ``correct`` (verdict==gold) always and, on a ``--logprobs`` run, the
#: first-token credence columns (``p_yes``/``leaked_mass``/``p_yes_is_bound``).
#: v1 rows (no credence, no ``correct``) still load — the readers use ``.get()``
#: and never gate on ``v``, so ``--report-only`` on the committed v1 rows is
#: byte-for-byte unaffected.
_RESULT_SCHEMA_VERSION = 2

#: Filename ``variant`` token for the ``--logprobs`` credence probe. It is the
#: contamination firewall: probe rows land in ``…__logprobs.jsonl``, a DIFFERENT
#: file from the committed replication rows, so the probe is physically incapable
#: of overwriting them even when pointed at the same ``--results-dir``.
LOGPROBS_VARIANT = "logprobs"

#: Default results dir for the plain (v1-style) live run — gitignored tmp.
DEFAULT_RESULTS_DIR = Path("tmp/peeters")

#: Results dir the ``--logprobs`` probe defaults to: the COMMITTED results tree
#: (never gitignored ``tmp/``), so the paid probe's rows are durable + committable
#: right beside the replication rows they sit next to (distinct filename).
COMMITTED_RESULTS_DIR = Path("examples/research/results/peeters")


# ---------------------------------------------------------------------------
# Crash-safe, resumable per-pair results store (mirrors m3_race's durability).
#
# Why not reuse ``langres.core.judgement_log.JudgementLog``? It is the closest
# in-repo sink and we DO borrow its shape (a ``v``-versioned JSONL, one row per
# judgement, a ``read()`` round-trip). But it is structurally a judge-CALL log:
# it has no notion of the GOLD label (it never sees ground truth), and it keeps
# ``cost_is_real``/``provider`` only inside the raw ``provenance`` dict behind
# ``features=True`` (which also logs PII record content), not as first-class
# columns. The final F1/agreement table here must be recomputable from the rows
# ALONE, which needs ``gold`` per row — so bending ``JudgementLog`` to fit would
# mean a post-hoc gold join plus digging into nested provenance. It also does not
# ``fsync``. A tiny purpose-built sink with exactly the columns the report needs
# (``gold`` included), an ``fsync`` on every append, and truncation-tolerant reads
# is simpler and keeps this operator tool decoupled from a core class whose
# contract (a privacy-conscious flywheel inlet keyed to ``capture_run``) is a
# different concern.
# ---------------------------------------------------------------------------


def results_path_for(
    results_dir: str | Path,
    dataset: str,
    prompt_design: str,
    model: str,
    *,
    limit: int | None = None,
    seed: int = 0,
    variant: str = "",
) -> Path:
    """The per-(model, dataset, prompt-design, subset, variant) JSONL path under ``results_dir``.

    The ``openrouter/...`` model id's slashes are flattened so nothing creates a
    stray subdirectory; the fields keep each race cell in its own file, so a crash
    in one never touches another (and resume/report-only target one file).

    ``limit``/``seed`` are part of the identity because they *select a different
    pair set*. A ``--limit 150`` trial and the full run judge different pairs, and
    both resume and report-only consume every row in the file — so sharing one path
    would let a trial's rows leak into the full report (wrong ``n_judged``, cost and
    F1) and let unrelated prior spend eat the budget cap. A full run (``limit=None``)
    keeps the plain three-field name.

    ``variant`` (e.g. :data:`LOGPROBS_VARIANT`) is the **contamination firewall**:
    it appends ``__{variant}`` to the filename so the ``--logprobs`` credence probe
    writes to ``…__logprobs.jsonl`` — a physically distinct file from the committed
    replication rows — and therefore cannot overwrite them even when both share a
    ``--results-dir``. Empty (the default) keeps the plain replication filename.
    """
    slug = model.replace("/", "_")
    subset = "" if limit is None else f"__limit{limit}-seed{seed}"
    variant_token = f"__{variant}" if variant else ""
    return Path(results_dir) / f"{dataset}__{prompt_design}__{slug}{subset}{variant_token}.jsonl"


class PeetersResultStore:
    """Append-only JSONL sink for one paid race cell — durable, resumable, atomic-ish.

    Each judged pair is one JSON line, ``flush``-ed and ``os.fsync``-ed before the
    next paid call, so a ``SIGKILL`` cannot lose an already-billed call (the kernel
    holds the bytes even if the process dies). Reads tolerate a truncated trailing
    line (a kill mid-write) by skipping it; :meth:`append` first repairs a missing
    final newline so a later resume never fuses a leftover partial fragment onto a
    fresh row. :meth:`judged_pairs` drives skip-if-committed resume and
    :meth:`spent` seeds the cross-resume budget ledger.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, row: dict[str, Any]) -> None:
        """Append one result row as a JSON line, flushed + fsync'd for durability."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_trailing_newline()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _ensure_trailing_newline(self) -> None:
        """If the file ends mid-line (an interrupted write), close that line first.

        Turns a leftover partial fragment into its own (skippable) line so the next
        append lands on a clean line instead of being concatenated onto the
        fragment — which would make BOTH unparseable and silently lose the new row.
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with self.path.open("rb") as fh:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) == b"\n":
                return
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())

    def rows(self) -> list[dict[str, Any]]:
        """Every intact row in write order; an unparseable (truncated) line is skipped."""
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                out.append(json.loads(stripped))
            except json.JSONDecodeError:
                logger.warning(
                    "skipping unparseable JSONL line in %s (crash mid-write?)", self.path
                )
        return out

    def judged_pairs(self) -> set[frozenset[str]]:
        """The set of ``frozenset({left_id, right_id})`` already committed (resume skip set)."""
        return {frozenset({r["left_id"], r["right_id"]}) for r in self.rows()}

    def spent(self) -> float:
        """Sum of the recorded ``cost_usd`` across committed rows (the resume-safe ledger)."""
        return sum(float(r.get("cost_usd") or 0.0) for r in self.rows())


def stratified_subset_indices(labels: Sequence[int], limit: int, seed: int = 0) -> list[int]:
    """Pick ``limit`` sample-order indices, preserving the positive ratio, deterministically.

    The Peeters pair set is a *concatenation* of the positive block followed by the
    negative block (Abt-Buy: 206 positives at indices 0–205, then 1000 negatives),
    so naively taking the first ``N`` rows would return an all-positive (or, past
    206, positive-heavy) subset. This instead keeps ``round(limit * pos_ratio)``
    positives and the rest negatives — reproducing the full set's ~17.1% positive
    ratio — sampling each class with a seeded :class:`random.Random` so the same
    ``(limit, seed)`` always yields the same subset. Indices are returned in
    **ascending** (sample) order, so they can subset any sample-aligned sequence
    (prompts, candidates, archived answers) without breaking alignment.

    Args:
        labels: The full pair set's gold labels, in sample order (``1`` = match).
        limit: Target subset size. ``>= len(labels)`` returns every index.
        seed: RNG seed (default ``0``) making the choice reproducible.

    Returns:
        The chosen indices, ascending. Its length is ``limit`` unless a class is
        exhausted first (not the case for the shipped Abt-Buy/Amazon-Google sets
        at any ``limit <= 1206``).
    """
    n = len(labels)
    if limit >= n:
        return list(range(n))
    positives = [i for i, label in enumerate(labels) if label == 1]
    negatives = [i for i, label in enumerate(labels) if label == 0]
    n_pos = min(round(limit * len(positives) / n), len(positives), limit)
    n_neg = min(limit - n_pos, len(negatives))
    rng = random.Random(seed)
    return sorted(rng.sample(positives, n_pos) + rng.sample(negatives, n_neg))


def _tokenizer_model(model: str) -> str:
    """Bare model id for tokenization (strip routing/provider prefixes).

    ``litellm.token_counter`` only resolves the correct **o200k_base** encoding
    from the bare OpenAI id (``gpt-4o-mini-2024-07-18``); the ``openrouter/…`` and
    ``openai/…`` prefixed forms silently fall back to cl100k_base and over-count
    (~1.5% high). Taking the last path segment recovers the id that bills
    correctly — verified to match a direct o200k_base count (100,256 tokens over
    the 1206 abt-buy prompts).
    """
    return model.split("/")[-1]


def _litellm_token_counter(model: str) -> Callable[[str], int]:
    """A ``prompt -> input_token_count`` closure for ``model`` (local tiktoken, $0)."""
    import litellm

    tok_model = _tokenizer_model(model)

    def count(prompt: str) -> int:
        return int(
            litellm.token_counter(model=tok_model, messages=[{"role": "user", "content": prompt}])
        )

    return count


def dry_run(
    spec: PeetersReplicationSpec,
    model: str,
    *,
    prices: dict[str, tuple[float, float]] = PRICES_PER_1M,
    count_tokens: Callable[[str], int] | None = None,
    indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Render every pair through the LIVE path and count tokens — ZERO API calls.

    Renders each candidate with the exact ``build_llm_prompt_template`` +
    ``make_record_serializer`` the live ``LLMJudge`` uses, so the input-token
    total is the true billed input. Output tokens are estimated (the single-word
    binary answer). ``count_tokens`` is injectable for a dependency-free test;
    ``None`` uses the real litellm/tiktoken counter. ``indices`` (from
    :func:`stratified_subset_indices`) prices only that ``--limit`` subset;
    ``None`` prices all pairs.

    Returns a dict with the pair count, input/output token totals, mean/max input,
    and a cost estimate priced against ``prices[model]``.
    """
    template = build_llm_prompt_template(spec)
    serializer = make_record_serializer(spec)
    candidates = build_candidates(spec)
    if indices is not None:
        candidates = [candidates[i] for i in indices]
    counter = count_tokens if count_tokens is not None else _litellm_token_counter(model)

    input_tokens = 0
    max_input = 0
    for candidate in candidates:
        prompt = template.replace("{left}", serializer(candidate.left)).replace(
            "{right}", serializer(candidate.right)
        )
        n = counter(prompt)
        input_tokens += n
        max_input = max(max_input, n)

    n_pairs = len(candidates)
    output_tokens = n_pairs * _EST_OUTPUT_TOKENS_PER_PROMPT
    in_per_1m, out_per_1m = prices[model]
    est_usd = input_tokens * in_per_1m / 1_000_000.0 + output_tokens * out_per_1m / 1_000_000.0
    return {
        "model": model,
        "n_pairs": n_pairs,
        "input_tokens": input_tokens,
        "output_tokens_est": output_tokens,
        "mean_input_tokens": input_tokens / n_pairs if n_pairs else 0.0,
        "max_input_tokens": max_input,
        "est_usd": est_usd,
    }


def _aggregate_usage(judgements: Sequence[Any], model: str) -> LLMUsage:
    """Sum the per-judgement ``provenance["usage"]`` vectors into one ``LLMUsage``."""
    totals = [0, 0, 0, 0, 0]
    provider: str | None = None
    for judgement in judgements:
        raw = judgement.provenance.get("usage") or {}
        usage = LLMUsage(**raw) if raw else LLMUsage()
        totals[0] += usage.input_tokens
        totals[1] += usage.output_tokens
        totals[2] += usage.cache_read_input_tokens
        totals[3] += usage.cache_creation_input_tokens
        totals[4] += usage.reasoning_tokens
        provider = provider or usage.provider
    return LLMUsage(
        input_tokens=totals[0],
        output_tokens=totals[1],
        cache_read_input_tokens=totals[2],
        cache_creation_input_tokens=totals[3],
        reasoning_tokens=totals[4],
        provider=provider,
        model=model,
    )


def _build_live_judge(
    spec: PeetersReplicationSpec,
    model: str,
    client: Any = None,
    *,
    confidence: str = "none",
) -> LLMJudge[Any]:
    """The live ``LLMJudge`` for a Peeters slice — the single build site.

    Wires the paper's ``domain-complex-force`` template, the Peeters per-dataset
    ``record_serializer``, ``response_parser=parse_binary_yes_no`` and
    ``temperature=0.0``, and pins :data:`LIVE_PROVIDER` so OpenRouter routes to
    OpenAI's own backend (our only deviation from the paper is the OpenRouter
    hop). Shared by :func:`run_live` and :func:`run_compare_archived` so the two
    paid paths judge with a byte-identical judge. ``client`` is injectable (a
    fake) so the whole flow is exercised at **$0** in tests.

    ``confidence="logprob"`` (the ``--logprobs`` probe) additionally requests
    first-token logprobs and records a P(Yes) credence in provenance. It is the
    ONLY difference from the replication judge, so the probe stays byte-identical
    to the replication run apart from the logprob request (which, on the single
    "Yes"/"No" output token, adds zero output tokens and so ~zero cost).
    """
    return LLMJudge(
        client=client,
        model=model,
        temperature=0.0,
        prompt_template=build_llm_prompt_template(spec),
        record_serializer=make_record_serializer(spec),
        response_parser=parse_binary_yes_no,
        provider=LIVE_PROVIDER,
        confidence="logprob" if confidence == "logprob" else "none",
    )


def _row_from_judgement(
    judgement: Any,
    *,
    model: str,
    dataset: str,
    prompt_design: str,
    gold: int,
) -> dict[str, Any]:
    """The durable JSONL row for one judged pair (everything the report needs).

    Carries ``gold`` (so F1 is recomputable from rows alone) plus the raw model
    answer (``response_text``), our parsed ``verdict``, the per-call ``LLMUsage``
    vector and ``cost_usd``/``cost_is_real``/``provider``/``model`` — the columns
    ``JudgementLog`` either lacks (``gold``) or buries in ``provenance``.

    v2 also records ``correct`` (``verdict == gold``) on every row and, when the
    ``--logprobs`` credence probe was on (``provenance`` carries ``p_yes``), the
    first-token credence columns ``p_yes`` / ``leaked_mass`` / ``p_yes_is_bound``.
    Together they make "does the model's own first-token credence predict its
    errors?" answerable from the rows ALONE. The credence keys are omitted (not
    written as ``null``) when the probe was off, keeping a plain live run's row
    identical to v1 apart from ``v`` and ``correct``.
    """
    prov = judgement.provenance
    verdict = int(judgement.score)
    row: dict[str, Any] = {
        "v": _RESULT_SCHEMA_VERSION,
        "model": model,
        "dataset": dataset,
        "prompt_design": prompt_design,
        "left_id": judgement.left_id,
        "right_id": judgement.right_id,
        "gold": int(gold),
        "response_text": judgement.reasoning or "",
        "verdict": verdict,
        "score": float(judgement.score),
        "correct": int(verdict == int(gold)),
        "cost_usd": float(prov.get("cost_usd") or 0.0),
        "cost_is_real": bool(prov.get("cost_is_real")),
        "provider": prov.get("provider"),
        "usage": prov.get("usage"),
    }
    if "p_yes" in prov:  # the credence probe was on for this judgement
        row["p_yes"] = prov.get("p_yes")
        row["leaked_mass"] = prov.get("confidence_leaked_mass")
        row["p_yes_is_bound"] = prov.get("p_yes_is_bound")
    return row


def _judge_stream(
    spec: PeetersReplicationSpec,
    model: str,
    candidates: Sequence[Any],
    gold_by_pair: dict[frozenset[str], int],
    budget_usd: float,
    *,
    client: Any,
    store: PeetersResultStore | None,
    progress_every: int,
    prior_spent: float,
    dataset: str,
    prompt_design: str,
    their_verdict_by_pair: dict[frozenset[str], int] | None = None,
    confidence: str = "none",
) -> tuple[list[dict[str, Any]], bool]:
    """Judge ``candidates`` under an AGGREGATE cap, persisting each row as it lands.

    The :class:`SpendMonitor` is seeded with ``prior_spent`` (spend already recorded
    in ``store``), so the hard cap holds across resumes — a resumed run cannot push
    cumulative spend past ``budget_usd``, and one that is already at/over the cap
    makes **zero** API calls. Every judgement is appended to ``store`` (flush +
    ``fsync``) *before* the next paid call, so a kill cannot lose a billed call. An
    incremental progress line (unbuffered) prints every ``progress_every`` pairs —
    including running archive-agreement when ``their_verdict_by_pair`` is given.

    Returns ``(rows_this_pass, budget_hit)``; ``rows_this_pass`` is what THIS pass
    judged (the report is computed from the whole ``store``, not this list).
    """
    rows: list[dict[str, Any]] = []
    monitor = SpendMonitor(budget_usd=budget_usd)
    monitor.add(prior_spent)
    try:
        monitor.check()
    except BudgetExceeded:
        logger.warning(
            "prior spend $%.4f already at/over cap $%.2f — making no calls", prior_spent, budget_usd
        )
        return rows, True
    if not candidates:
        return rows, False

    judge = _build_live_judge(spec, model, client, confidence=confidence)
    running_cost = prior_spent
    agree = 0
    n_done = 0
    budget_hit = False
    total = len(candidates)
    for judgement in judge.forward(iter(candidates)):
        key = frozenset({judgement.left_id, judgement.right_id})
        row = _row_from_judgement(
            judgement,
            model=model,
            dataset=dataset,
            prompt_design=prompt_design,
            gold=gold_by_pair.get(key, 0),
        )
        if store is not None:
            store.append(row)  # durable BEFORE the next paid call
        rows.append(row)
        n_done += 1
        running_cost += row["cost_usd"]
        monitor.add(row["cost_usd"])
        if their_verdict_by_pair is not None and int(judgement.score) == their_verdict_by_pair.get(
            key
        ):
            agree += 1
        if progress_every and n_done % progress_every == 0:
            msg = f"[live] {model}  judged {n_done}/{total}  spend ${running_cost:.4f}"
            if their_verdict_by_pair is not None:
                msg += f"  archive-agree {agree}/{n_done} = {agree / n_done * 100:.1f}%"
            print(msg, flush=True)
        try:
            monitor.check()
        except BudgetExceeded:
            budget_hit = True
            logger.warning("budget cap hit after %d/%d pairs", n_done, total)
            break
    return rows, budget_hit


def _metrics_from_rows(rows: Sequence[dict[str, Any]]) -> Any:
    """Pairwise P/R/F1 (via ``classify_pairs``) from persisted rows' verdict vs. gold."""
    judgements = [
        PairwiseJudgement(
            left_id=r["left_id"],
            right_id=r["right_id"],
            score=float(r["verdict"]),
            score_type="prob_llm",
            decision_step="peeters_live",
            provenance={},
        )
        for r in rows
    ]
    gold = {frozenset({r["left_id"], r["right_id"]}) for r in rows if r.get("gold") == 1}
    return classify_pairs(judgements, gold, threshold=0.5)


def _aggregate_usage_from_rows(rows: Sequence[dict[str, Any]], model: str) -> LLMUsage:
    """Sum the per-row ``usage`` vectors into one :class:`LLMUsage` (reuses ``_aggregate_usage``)."""
    ns = [SimpleNamespace(provenance={"usage": r.get("usage")}) for r in rows]
    return _aggregate_usage(ns, model)


def _live_report_from_rows(
    rows: Sequence[dict[str, Any]], *, model: str, n_pairs: int
) -> dict[str, Any]:
    """The live report computed PURELY from persisted rows (identical across resumes).

    ``budget_hit`` is derived as ``n_judged < n_pairs`` — for these runs the only
    reason the stream stops short of the full pair set is the spend cap firing.
    """
    metrics = _metrics_from_rows(rows)
    usage = _aggregate_usage_from_rows(rows, model)
    n_judged = len(rows)
    real_cost = sum(float(r.get("cost_usd") or 0.0) for r in rows)
    cost_is_real = n_judged > 0 and all(bool(r.get("cost_is_real")) for r in rows)
    return {
        "model": model,
        "published_f1": PAID_MODELS.get(model),
        "n_pairs": n_pairs,
        "n_judged": n_judged,
        "budget_hit": n_judged < n_pairs,
        "f1": metrics.f1 * 100.0,
        "precision": metrics.precision * 100.0,
        "recall": metrics.recall * 100.0,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "usage": usage.model_dump(),
        "real_cost_usd": real_cost,
        "cost_is_real": cost_is_real,
        "usd_per_1k_pairs": (real_cost / n_judged * 1000.0) if n_judged else 0.0,
    }


def _n_pairs_for(spec: PeetersReplicationSpec, limit: int | None, seed: int) -> int:
    """The full pair count a (limit, seed) run judges — the report's ``n_pairs`` denominator."""
    sample = load_peeters_sample(spec)
    if limit is None:
        return len(sample)
    return len(stratified_subset_indices([label for _l, _r, label in sample], limit, seed))


def report_live_from_store(
    store: PeetersResultStore,
    *,
    spec: PeetersReplicationSpec,
    model: str,
    limit: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Recompute the live report from an existing results JSONL — ZERO API calls."""
    return _live_report_from_rows(
        store.rows(), model=model, n_pairs=_n_pairs_for(spec, limit, seed)
    )


def run_live(
    spec: PeetersReplicationSpec,
    model: str,
    *,
    budget_usd: float,
    prices: dict[str, tuple[float, float]] = PRICES_PER_1M,
    client: Any = None,
    indices: Sequence[int] | None = None,
    store: PeetersResultStore | None = None,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
    confidence: str = "none",
) -> dict[str, Any]:
    """Judge every sampled pair with a live ``LLMJudge`` under a hard spend cap.

    Builds the judge via :func:`_build_live_judge` (paper template, Peeters
    serializer, ``parse_binary_yes_no``, ``temperature=0.0``, provider pinned) and
    streams it under a :class:`SpendMonitor` via :func:`_judge_stream`, computing
    pairwise P/R/F1 at threshold 0.5 (no sweep — the protocol is binary) over the
    pairs judged. ``indices`` (from :func:`stratified_subset_indices`) restricts the
    run to a ``--limit`` subset; ``None`` judges all pairs.

    When ``store`` is given the run is **crash-safe and resumable**: pairs already
    committed to the JSONL are skipped (a completed model re-runs at $0 with zero
    API calls), the cap accounts for spend already recorded (the aggregate ledger),
    each new pair is durably persisted before the next paid call, and the returned
    report is computed from the WHOLE store — so the numbers are identical whether
    the run finished in one pass or several. ``store=None`` keeps the classic
    in-memory path (writes nothing to disk).

    ``client`` is injectable (a fake returning "Yes"/"No") so the whole flow is
    verified at **$0** in tests; ``None`` lets ``LLMJudge`` build a real litellm
    client from the environment (the paid path).
    """
    candidates = build_candidates(spec)
    sample = load_peeters_sample(spec)
    if indices is not None:
        candidates = [candidates[i] for i in indices]
        sample = [sample[i] for i in indices]
    n_pairs = len(candidates)
    gold_by_pair = {frozenset({left, right}): label for left, right, label in sample}

    prior_spent = 0.0
    if store is not None:
        judged = store.judged_pairs()
        prior_spent = store.spent()
        remaining = [c for c in candidates if frozenset({c.left.id, c.right.id}) not in judged]
        if len(remaining) < n_pairs:
            print(
                f"[resume] {model}: skipping {n_pairs - len(remaining)}/{n_pairs} already-judged "
                f"pairs (prior spend ${prior_spent:.4f})",
                flush=True,
            )
        candidates = remaining

    rows_this_pass, _budget_hit = _judge_stream(
        spec,
        model,
        candidates,
        gold_by_pair,
        budget_usd,
        client=client,
        store=store,
        progress_every=progress_every,
        prior_spent=prior_spent,
        dataset=spec.name,
        prompt_design=LIVE_PROMPT_DESIGN,
        confidence=confidence,
    )

    report_rows = store.rows() if store is not None else rows_this_pass
    return _live_report_from_rows(report_rows, model=model, n_pairs=n_pairs)


# ---------------------------------------------------------------------------
# Per-pair agreement against the authors' archived answers (--compare-archived).
#
# The paid run above scores *our* F1 against the paper's published number. This
# goes finer-grained: for the exact model we run, the authors archived their raw
# per-pair answer, so we can check pair-by-pair whether our live verdict matches
# theirs — the rows a human reads when the aggregate F1 diverges from the paper.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparedPair:
    """One pair judged both ways: our live verdict vs. the authors' archived verdict.

    Attributes:
        left_id / right_id: Source-prefixed record ids.
        left / right: The serialized records the model actually saw.
        label: Gold label (``1`` = match, ``0`` = non-match).
        our_answer / our_verdict: Our raw model answer and its
            :func:`parse_binary_yes_no` verdict (``1``/``0``).
        their_answer / their_verdict: The authors' archived raw answer and its
            :func:`parse_binary_yes_no` verdict — both sides go through the one
            canonical parser so the comparison is apples-to-apples.
    """

    left_id: str
    right_id: str
    left: str
    right: str
    label: int
    our_answer: str
    our_verdict: int
    their_answer: str
    their_verdict: int


def load_archived_answers(
    spec: PeetersReplicationSpec,
    model: str,
    prompt_design: str,
    cache_dir: Path,
) -> list[dict[str, Any]]:
    """Fetch + parse the authors' archived ``{prompt, answer}`` rows for ``model``.

    Reuses the offline replay's :func:`ensure_answers` download/extract/cache path
    (no duplication) — mapping the OpenRouter routing id to the bare model id the
    archive member is keyed on (``openrouter/openai/gpt-4o-mini-2024-07-18`` →
    ``gpt-4o-mini-2024-07-18``). Rows are returned in the archive's line order,
    which matches the sampled pair set (hence :func:`stratified_subset_indices`
    can subset both consistently).
    """
    archived_model = model.split("/")[-1]  # archive members use the bare OpenAI id
    path = ensure_answers(cache_dir, spec.name, prompt_design, archived_model)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_compared_pairs(
    prompts: Sequence[Any],
    candidates: Sequence[Any],
    our_judgements: Sequence[Any],
    archived: Sequence[dict[str, Any]],
    serializer: Callable[[Any], str],
) -> list[ComparedPair]:
    """Zip the four sample-aligned streams into :class:`ComparedPair`\\ s, failing loud on drift.

    For each pair it asserts our rendered prompt equals the archived ``prompt``
    field — a mismatch means the alignment between our pair set and their archive
    is off, which would make *every* downstream comparison meaningless, so it
    raises :class:`SystemExit` with a unified diff rather than reporting garbage.
    Both verdicts come from :func:`parse_binary_yes_no` (ours already applied by
    the live judge; theirs via :func:`parse_binary_answer`).

    Raises:
        ValueError: If the four inputs are not the same length.
        SystemExit: On the first prompt that does not match its archived prompt.
    """
    n = len(prompts)
    if not (len(candidates) == n and len(our_judgements) == n and len(archived) == n):
        raise ValueError(
            f"compare inputs must align 1:1 (prompts={n}, candidates={len(candidates)}, "
            f"judgements={len(our_judgements)}, archived={len(archived)})"
        )
    compared: list[ComparedPair] = []
    for i in range(n):
        rendered = prompts[i].prompt
        archived_prompt = str(archived[i]["prompt"])
        if rendered != archived_prompt:
            diff = "\n".join(
                difflib.unified_diff(
                    archived_prompt.splitlines(),
                    rendered.splitlines(),
                    fromfile="archived",
                    tofile="ours",
                    lineterm="",
                )
            )
            raise SystemExit(
                f"[fatal] prompt mismatch at compared row {i} "
                f"({prompts[i].left_id} vs {prompts[i].right_id}) — alignment is off, so "
                f"every downstream comparison would be meaningless. Aborting.\n{diff}"
            )
        their_answer = str(archived[i]["answer"])
        our_answer = our_judgements[i].reasoning or ""
        compared.append(
            ComparedPair(
                left_id=prompts[i].left_id,
                right_id=prompts[i].right_id,
                left=serializer(candidates[i].left),
                right=serializer(candidates[i].right),
                label=prompts[i].label,
                our_answer=our_answer,
                our_verdict=int(our_judgements[i].score),
                their_answer=their_answer,
                their_verdict=parse_binary_answer(their_answer),
            )
        )
    return compared


def confusion_and_agreement(compared: Sequence[ComparedPair]) -> dict[str, Any]:
    """The per-pair agreement rate + the 2x2 confusion of our verdicts vs. theirs."""
    both_yes = sum(1 for c in compared if c.our_verdict == 1 and c.their_verdict == 1)
    both_no = sum(1 for c in compared if c.our_verdict == 0 and c.their_verdict == 0)
    we_yes_they_no = sum(1 for c in compared if c.our_verdict == 1 and c.their_verdict == 0)
    we_no_they_yes = sum(1 for c in compared if c.our_verdict == 0 and c.their_verdict == 1)
    n = len(compared)
    return {
        "n_compared": n,
        "agreement_rate": (both_yes + both_no) / n if n else 0.0,
        "confusion": {
            "both_yes": both_yes,
            "both_no": both_no,
            "we_yes_they_no": we_yes_they_no,
            "we_no_they_yes": we_no_they_yes,
        },
    }


def run_compare_archived(
    spec: PeetersReplicationSpec,
    model: str,
    *,
    prompt_design: str = LIVE_PROMPT_DESIGN,
    budget_usd: float,
    client: Any = None,
    archived: Sequence[dict[str, Any]] | None = None,
    cache_dir: Path | None = None,
    limit: int | None = None,
    seed: int = 0,
    store: PeetersResultStore | None = None,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
    confidence: str = "none",
) -> dict[str, Any]:
    """Judge a (optionally ``--limit``-ed) subset live and compare pair-by-pair to the archive.

    Runs the live judge over the stratified subset (:func:`_build_live_judge` +
    :func:`_judge_stream`), loads the authors' archived answers for the same
    model (:func:`load_archived_answers`, or the injected ``archived`` in tests),
    and reports (via :func:`_compare_report_from_rows`): the per-pair agreement
    rate, the 2x2 confusion of ours vs. theirs, up to 10 concrete disagreeing
    pairs, our F1/P/R on the judged subset next to *their* F1/P/R recomputed on
    that same subset (plus the published full-set number), and the usage vector +
    real billed cost.

    When ``store`` is given the run is **crash-safe and resumable** exactly as in
    :func:`run_live` (skip-if-committed, cross-resume budget ledger, per-pair
    ``fsync``); progress lines additionally show the running archive-agreement. The
    report is recomputed from the persisted rows + the archive, so ``--report-only``
    (:func:`report_compare_from_store`) reproduces it with zero API calls.

    The archived JSONL row count is asserted equal to the *full* pair-set count
    before any subsetting; per-compared-pair the rendered prompt is asserted equal
    to the archived one (:func:`build_compared_pairs` raises on a mismatch). The
    ``archived`` / ``client`` seams let the whole path run at **$0** in tests.

    Raises:
        SystemExit: On an archived-vs-pair-set row-count mismatch (or, via
            :func:`build_compared_pairs`, a per-pair prompt mismatch).
        ValueError: If ``archived is None`` and no ``cache_dir`` was given.
    """
    full_prompts = render_sample_prompts(spec)
    if archived is None:
        if cache_dir is None:
            raise ValueError("run_compare_archived needs either `archived` or a `cache_dir`")
        archived = load_archived_answers(spec, model, prompt_design, cache_dir)
    if len(archived) != len(full_prompts):
        raise SystemExit(
            f"[fatal] archived JSONL has {len(archived)} rows but the {spec.name} pair set "
            f"has {len(full_prompts)} — alignment would be wrong; aborting."
        )

    labels = [p.label for p in full_prompts]
    indices = (
        stratified_subset_indices(labels, limit, seed)
        if limit is not None
        else list(range(len(labels)))
    )
    full_candidates = build_candidates(spec)
    sub_prompts = [full_prompts[i] for i in indices]
    sub_candidates = [full_candidates[i] for i in indices]
    sub_archived = [archived[i] for i in indices]
    n_pairs = len(sub_candidates)

    gold_by_pair = {frozenset({p.left_id, p.right_id}): p.label for p in sub_prompts}
    # Their archived verdict per pair — powers the running agreement in progress lines.
    their_verdict_by_pair = {
        frozenset({p.left_id, p.right_id}): parse_binary_answer(str(a["answer"]))
        for p, a in zip(sub_prompts, sub_archived, strict=True)
    }

    prior_spent = 0.0
    candidates = sub_candidates
    if store is not None:
        judged = store.judged_pairs()
        prior_spent = store.spent()
        candidates = [c for c in sub_candidates if frozenset({c.left.id, c.right.id}) not in judged]
        if len(candidates) < n_pairs:
            print(
                f"[resume] {model}: skipping {n_pairs - len(candidates)}/{n_pairs} already-judged "
                f"pairs (prior spend ${prior_spent:.4f})",
                flush=True,
            )

    _rows_this_pass, _budget_hit = _judge_stream(
        spec,
        model,
        candidates,
        gold_by_pair,
        budget_usd,
        client=client,
        store=store,
        progress_every=progress_every,
        prior_spent=prior_spent,
        dataset=spec.name,
        prompt_design=prompt_design,
        their_verdict_by_pair=their_verdict_by_pair,
        confidence=confidence,
    )

    report_rows = store.rows() if store is not None else _rows_this_pass
    return _compare_report_from_rows(
        report_rows, spec=spec, model=model, archived=archived, n_pairs=n_pairs
    )


def _compare_report_from_rows(
    rows: Sequence[dict[str, Any]],
    *,
    spec: PeetersReplicationSpec,
    model: str,
    archived: Sequence[dict[str, Any]],
    n_pairs: int,
) -> dict[str, Any]:
    """The archive-agreement report computed PURELY from persisted rows + the archive.

    Reconstructs each judged pair's rendered prompt, candidate and archived answer
    (from ``spec`` + ``archived``, both free/local) keyed on ``(left_id, right_id)``,
    then reuses :func:`build_compared_pairs` / :func:`confusion_and_agreement` so a
    ``--report-only`` run reproduces the exact agreement/confusion/disagreement table
    and F1 with zero API calls.
    """
    full_prompts = render_sample_prompts(spec)
    if len(archived) != len(full_prompts):
        raise SystemExit(
            f"[fatal] archived JSONL has {len(archived)} rows but the {spec.name} pair set "
            f"has {len(full_prompts)} — alignment would be wrong; aborting."
        )
    full_candidates = build_candidates(spec)
    serializer = make_record_serializer(spec)
    prompt_by_pair = {frozenset({p.left_id, p.right_id}): p for p in full_prompts}
    cand_by_pair = {frozenset({c.left.id, c.right.id}): c for c in full_candidates}
    archived_by_pair = {
        frozenset({full_prompts[i].left_id, full_prompts[i].right_id}): archived[i]
        for i in range(len(full_prompts))
    }

    prompts_sub: list[Any] = []
    cands_sub: list[Any] = []
    our_js: list[Any] = []
    arch_sub: list[dict[str, Any]] = []
    for r in rows:
        key = frozenset({r["left_id"], r["right_id"]})
        prompts_sub.append(prompt_by_pair[key])
        cands_sub.append(cand_by_pair[key])
        our_js.append(
            SimpleNamespace(score=float(r["verdict"]), reasoning=r.get("response_text") or "")
        )
        arch_sub.append(archived_by_pair[key])

    compared = build_compared_pairs(prompts_sub, cands_sub, our_js, arch_sub, serializer)
    summary = confusion_and_agreement(compared)
    gold = {frozenset({p.left_id, p.right_id}) for p in prompts_sub if p.label == 1}
    our_metrics = _metrics_from_rows(rows)
    their_judgements = judgements_from_answers(prompts_sub, [str(a["answer"]) for a in arch_sub])
    their_metrics = classify_pairs(their_judgements, gold, threshold=0.5)
    usage = _aggregate_usage_from_rows(rows, model)
    disagreements = [c for c in compared if c.our_verdict != c.their_verdict][:10]
    n_judged = len(rows)
    real_cost = sum(float(r.get("cost_usd") or 0.0) for r in rows)
    cost_is_real = n_judged > 0 and all(bool(r.get("cost_is_real")) for r in rows)

    return {
        "model": model,
        "n_pairs": n_pairs,
        "n_judged": n_judged,
        "budget_hit": n_judged < n_pairs,
        "agreement_rate": summary["agreement_rate"],
        "confusion": summary["confusion"],
        "disagreements": [
            {
                "left_id": c.left_id,
                "right_id": c.right_id,
                "left": c.left,
                "right": c.right,
                "gold_label": c.label,
                "their_answer": c.their_answer,
                "our_answer": c.our_answer,
            }
            for c in disagreements
        ],
        "ours": {
            "f1": our_metrics.f1 * 100.0,
            "precision": our_metrics.precision * 100.0,
            "recall": our_metrics.recall * 100.0,
            "tp": our_metrics.tp,
            "fp": our_metrics.fp,
            "fn": our_metrics.fn,
        },
        "theirs_subset": {
            "f1": their_metrics.f1 * 100.0,
            "precision": their_metrics.precision * 100.0,
            "recall": their_metrics.recall * 100.0,
            "tp": their_metrics.tp,
            "fp": their_metrics.fp,
            "fn": their_metrics.fn,
        },
        "published_f1": PAID_MODELS.get(model),
        "usage": usage.model_dump(),
        "real_cost_usd": real_cost,
        "cost_is_real": cost_is_real,
        "usd_per_1k_pairs": (real_cost / n_judged * 1000.0) if n_judged else 0.0,
    }


def report_compare_from_store(
    store: PeetersResultStore,
    *,
    spec: PeetersReplicationSpec,
    model: str,
    archived: Sequence[dict[str, Any]] | None = None,
    cache_dir: Path | None = None,
    prompt_design: str = LIVE_PROMPT_DESIGN,
    limit: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Recompute the archive-agreement report from an existing JSONL — ZERO API calls.

    ``archived`` is injectable (tests); otherwise the authors' answers are loaded
    from ``cache_dir`` (the same cache the live ``--compare-archived`` run filled).
    """
    if archived is None:
        if cache_dir is None:
            raise ValueError("report_compare_from_store needs either `archived` or a `cache_dir`")
        archived = load_archived_answers(spec, model, prompt_design, cache_dir)
    return _compare_report_from_rows(
        store.rows(),
        spec=spec,
        model=model,
        archived=archived,
        n_pairs=_n_pairs_for(spec, limit, seed),
    )


def _print_archived_comparison(report: dict[str, Any]) -> None:
    """Print the per-pair archive-agreement report (agreement, confusion, disagreements)."""
    conf = report["confusion"]
    ours = report["ours"]
    theirs = report["theirs_subset"]
    pub = report["published_f1"]
    print("\n" + "=" * 96)
    print(f"Peeters LLM-EM ARCHIVE AGREEMENT — {report['model']}")
    print("=" * 96)
    print(
        f"pairs judged         : {report['n_judged']}/{report['n_pairs']}"
        + ("  (budget cap hit — partial)" if report["budget_hit"] else "")
    )
    print(f"per-pair agreement   : {report['agreement_rate'] * 100:.2f}%  (ours vs. theirs)")
    print("confusion (ours×theirs):")
    print(f"  both YES : {conf['both_yes']:5d}    both NO       : {conf['both_no']:5d}")
    print(
        f"  we-Y they-N: {conf['we_yes_they_no']:3d}    we-N they-Y   : {conf['we_no_they_yes']:5d}"
    )
    print("-" * 96)
    print(
        f"OURS   (judged subset): F1={ours['f1']:.2f}  P={ours['precision']:.2f}  R={ours['recall']:.2f}"
    )
    print(
        f"THEIRS (same subset)  : F1={theirs['f1']:.2f}  P={theirs['precision']:.2f}  "
        f"R={theirs['recall']:.2f}"
    )
    pub_s = f"{pub:.2f}" if pub is not None else "—"
    print(f"THEIRS (published, full set) : F1={pub_s}")
    usage = report["usage"]
    print(
        f"usage: in={usage['input_tokens']} out={usage['output_tokens']}  "
        f"real_cost=${report['real_cost_usd']:.4f}  cost_is_real={report['cost_is_real']}  "
        f"$/1k={report['usd_per_1k_pairs']:.4f}"
    )
    disagreements = report["disagreements"]
    print("-" * 96)
    print("disagreeing pairs (up to 10 of the divergences):")
    if not disagreements:
        print("  (none — every judged pair agreed with the archive)")
    for d in disagreements:
        print(
            f"  [{d['left_id']} vs {d['right_id']}] gold={d['gold_label']}  "
            f"theirs={d['their_answer']!r}  ours={d['our_answer']!r}"
        )
        print(f"      left : {d['left']}")
        print(f"      right: {d['right']}")
    print("=" * 96)


def _print_comparison(results: Sequence[dict[str, Any]]) -> None:
    """Print the per-model comparison table (F1 vs published, usage, real cost)."""
    print("\n" + "=" * 96)
    print("Peeters LLM-EM LIVE — abt-buy / domain-complex-force")
    print("=" * 96)
    header = (
        f"{'model':40} {'F1':>7} {'pub':>7} {'Δ':>6} {'P':>7} {'R':>7} "
        f"{'in_tok':>9} {'out_tok':>8} {'cost$':>8} {'$/1k':>8} real"
    )
    print(header)
    print("-" * 96)
    for r in results:
        pub = r["published_f1"]
        delta = f"{r['f1'] - pub:+.2f}" if pub is not None else "—"
        pub_s = f"{pub:.2f}" if pub is not None else "—"
        usage = r["usage"]
        print(
            f"{r['model']:40} {r['f1']:7.2f} {pub_s:>7} {delta:>6} "
            f"{r['precision']:7.2f} {r['recall']:7.2f} "
            f"{usage['input_tokens']:9d} {usage['output_tokens']:8d} "
            f"{r['real_cost_usd']:8.4f} {r['usd_per_1k_pairs']:8.4f} {str(r['cost_is_real']):>5}"
        )
        if r["budget_hit"]:
            print(f"  ! budget cap hit after {r['n_judged']}/{r['n_pairs']} pairs (partial)")
    print("=" * 96)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_paid_models(requested: str | None) -> list[str]:
    """Resolve the ``--model`` selection to a list of PAID_MODELS ids ('both' = all)."""
    if requested in (None, "both"):
        return list(PAID_MODELS)
    if requested not in PAID_MODELS:
        raise SystemExit(
            f"[fatal] {requested!r} is not a paid-run model. Choose one of "
            f"{sorted(PAID_MODELS)} or 'both'."
        )
    return [requested]


def _assert_priced(models: Sequence[str]) -> None:
    """Refuse to start unless every model has a PRICES_PER_1M entry (else the cap is blind)."""
    unpriced = [m for m in models if m not in PRICES_PER_1M]
    if unpriced:
        raise SystemExit(
            f"[fatal] no PRICES_PER_1M entry for {unpriced} — an unpriced model silently "
            f"contributes $0 to the spend cap. Pin its price in "
            f"langres.clients.openrouter.PRICES_PER_1M first. Known: {sorted(PRICES_PER_1M)}"
        )


def _subset_indices(spec: PeetersReplicationSpec, limit: int | None, seed: int) -> list[int] | None:
    """The ``--limit`` stratified sample indices for ``spec`` (``None`` = all pairs)."""
    if limit is None:
        return None
    labels = [label for _left, _right, label in load_peeters_sample(spec)]
    return stratified_subset_indices(labels, limit, seed)


def _run_replay_mode(args: argparse.Namespace) -> int:
    print("Offline replay — NO API key, NO LLM call, $0 spend. Replaying archived answers.\n")
    replay(
        args.dataset,
        args.model or "gpt-4-0613",
        args.prompt_design,
        args.cache_dir,
        limit=args.limit,
        seed=args.seed,
    )
    return 0


def _run_dry_run_mode(args: argparse.Namespace) -> int:
    spec = get_peeters_replication(args.dataset)
    models = _resolve_paid_models(args.model)
    _assert_priced(models)
    indices = _subset_indices(spec, args.limit, args.seed)
    print("Dry run — NO API key, NO LLM call, $0 spend. Rendering prompts + counting tokens.\n")
    total_est = 0.0
    for model in models:
        report = dry_run(spec, model, indices=indices)
        total_est += report["est_usd"]
        print(
            f"{model:40}  pairs={report['n_pairs']}  "
            f"input_tokens={report['input_tokens']}  "
            f"(mean {report['mean_input_tokens']:.1f}, max {report['max_input_tokens']})  "
            f"output_tokens≈{report['output_tokens_est']}  est=${report['est_usd']:.4f}"
        )
    print(
        f"\nestimated total for {len(models)} model(s): ${total_est:.4f}  "
        f"(budget ${args.budget:.2f})"
    )
    return 0


def _store_for(args: argparse.Namespace, dataset: str, model: str) -> PeetersResultStore:
    """The durable results store for this race cell, partitioned by its pair subset.

    A ``--logprobs`` run carries the :data:`LOGPROBS_VARIANT` token so its rows land
    in a file physically distinct from the committed replication rows (firewall).
    """
    return PeetersResultStore(
        results_path_for(
            args.results_dir,
            dataset,
            args.prompt_design,
            model,
            limit=args.limit,
            seed=args.seed,
            variant=LOGPROBS_VARIANT if args.logprobs else "",
        )
    )


def _run_report_only_mode(args: argparse.Namespace) -> int:
    """Recompute + print the full report from existing JSONL results — ZERO API calls."""
    spec = get_peeters_replication(args.dataset)
    models = _resolve_paid_models(args.model)
    print("Report-only — reading persisted results, NO API key, NO LLM call, $0 spend.\n")
    reports: list[dict[str, Any]] = []
    for model in models:
        store = _store_for(args, spec.name, model)
        if not store.rows():
            print(f"[warn] no results at {store.path} for {model}; skipping.")
            continue
        print(f"[report-only] {model}: {len(store.rows())} rows from {store.path}")
        if args.compare_archived:
            report = report_compare_from_store(
                store,
                spec=spec,
                model=model,
                cache_dir=args.cache_dir,
                prompt_design=args.prompt_design,
                limit=args.limit,
                seed=args.seed,
            )
            _print_archived_comparison(report)
        else:
            report = report_live_from_store(
                store, spec=spec, model=model, limit=args.limit, seed=args.seed
            )
        reports.append(report)
    if not args.compare_archived and reports:
        _print_comparison(reports)
    return 0


def _run_live_mode(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv

    spec = get_peeters_replication(args.dataset)
    models = _resolve_paid_models(args.model)
    _assert_priced(models)
    indices = _subset_indices(spec, args.limit, args.seed)

    # Print the estimate BEFORE any network call, so the operator sees the cost.
    print("Estimating cost (dry run, $0) before any spend...\n")
    total_est = 0.0
    for model in models:
        report = dry_run(spec, model, indices=indices)
        total_est += report["est_usd"]
        print(
            f"  {model:40}  pairs={report['n_pairs']}  "
            f"input_tokens={report['input_tokens']}  est=${report['est_usd']:.4f}"
        )
    print(f"\nestimated total: ${total_est:.4f}  |  hard cap: ${args.budget:.2f}\n")
    print(f"results dir: {args.results_dir}  (per-pair JSONL; resume + --report-only read it)\n")

    if not args.yes_spend_money:
        print(
            "[refused] --mode live is a PAID run. Re-run with --yes-spend-money to proceed "
            "(no network call was made)."
        )
        return 1

    load_dotenv(".env")  # OPENROUTER_API_KEY lives in .env, not Settings.
    if "OPENROUTER_API_KEY" not in os.environ:
        print(
            "[fatal] OPENROUTER_API_KEY is not set. The paid path fails fast rather than "
            "falling back to another provider or an unpriced model. Set it and retry."
        )
        return 1

    results: list[dict[str, Any]] = []
    per_model_budget = args.budget / len(models)
    try:
        for model in models:
            store = _store_for(args, spec.name, model)
            if args.compare_archived:
                print(
                    f"[live+archive] judging {spec.name} with {model} "
                    f"(per-model cap ${per_model_budget:.2f}) -> {store.path}"
                )
                report = run_compare_archived(
                    spec,
                    model,
                    prompt_design=args.prompt_design,
                    budget_usd=per_model_budget,
                    cache_dir=args.cache_dir,
                    limit=args.limit,
                    seed=args.seed,
                    store=store,
                    progress_every=args.progress_every,
                    confidence="logprob" if args.logprobs else "none",
                )
                _print_archived_comparison(report)
            else:
                print(
                    f"[live] judging {spec.name} with {model} "
                    f"(per-model cap ${per_model_budget:.2f}) -> {store.path}"
                )
                report = run_live(
                    spec,
                    model,
                    budget_usd=per_model_budget,
                    indices=indices,
                    store=store,
                    progress_every=args.progress_every,
                    confidence="logprob" if args.logprobs else "none",
                )
            results.append(report)
    except BudgetExceeded as exc:
        print(f"[stopped] budget cap fired: {exc}")
        return 2

    if not args.compare_archived:
        _print_comparison(results)
    if args.results_path:
        out = Path(args.results_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"[report] wrote {out}")
    return 0


def main() -> int:
    # Line-buffer stdout so an incremental progress line survives a SIGKILL rather
    # than dying in the buffer (also pass ``python -u`` when driving this script).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--mode",
        choices=["replay", "dry-run", "live"],
        default="replay",
        help="replay archived answers ($0), dry-run the live path ($0), or run live (PAID).",
    )
    parser.add_argument(
        "--dataset",
        default="abt-buy",
        choices=list_peeters_replications(),
        help="Which replication slice (default: abt-buy).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "replay: archived model id (default gpt-4-0613). live/dry-run: one of "
            f"{sorted(PAID_MODELS)} or 'both' (default both)."
        ),
    )
    parser.add_argument(
        "--prompt-design",
        default="domain-complex-force",
        help="Archived prompt design (replay only; default domain-complex-force = Table 2 target).",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=DEFAULT_BUDGET_USD,
        help=f"Hard spend cap (USD) for live mode (default ${DEFAULT_BUDGET_USD:.2f}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Run a stratified subset of N pairs (preserving the ~17.1%% positive ratio, "
            "deterministic under --seed) instead of all 1206. Applies to dry-run/live/replay."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the --limit stratified subset (default 0).",
    )
    parser.add_argument(
        "--compare-archived",
        action="store_true",
        help=(
            "(--mode live) also compare each judged pair to the authors' archived answer for "
            "the same model: per-pair agreement, a 2x2 confusion, up to 10 disagreements, and "
            "our vs. their F1/P/R on the judged subset."
        ),
    )
    parser.add_argument(
        "--yes-spend-money",
        action="store_true",
        help="Required to actually spend in --mode live (off by default).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "langres-peeters-replication",
        help="Where to download/extract the answer archive (replay/compare-archived; gitignored).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the crash-safe per-pair JSONL results (one file per model). "
            "A live run appends here as it goes; resume + --report-only read it. Defaults to "
            "the gitignored tmp/peeters, EXCEPT a --logprobs run defaults to the committed "
            "examples/research/results/peeters (so the paid probe's rows are durable)."
        ),
    )
    parser.add_argument(
        "--logprobs",
        action="store_true",
        help=(
            "Credence probe: run the live judge with LLMJudge(confidence='logprob'), which "
            "requests first-token logprobs and records a P(Yes) credence per pair "
            "(p_yes/leaked_mass/p_yes_is_bound, plus correct=verdict==gold) in the v2 rows. "
            "Writes to a distinct …__logprobs.jsonl (never overwrites the replication rows) and "
            "defaults --results-dir to the committed tree. On the 1-token Yes/No answer it adds "
            "zero output tokens, so cost ≈ the plain replication run."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help=f"Print a progress line every N judged pairs (default {DEFAULT_PROGRESS_EVERY}).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Recompute + print the report from existing --results-dir JSONL with ZERO API "
            "calls (add --compare-archived for the agreement report). No spend, no key."
        ),
    )
    parser.add_argument(
        "--results-path",
        default=None,
        help="Optional JSON path to also dump the final aggregate report(s) to (live mode).",
    )
    args = parser.parse_args()

    # Resolve the results dir: the --logprobs probe defaults to the COMMITTED tree
    # (durable paid rows), everything else to the gitignored tmp default. An
    # explicit --results-dir always wins.
    if args.results_dir is None:
        args.results_dir = COMMITTED_RESULTS_DIR if args.logprobs else DEFAULT_RESULTS_DIR

    if args.limit is not None and args.limit <= 0:
        print(f"[fatal] --limit must be a positive integer (got {args.limit}).")
        return 1

    if args.logprobs and args.mode != "live" and not args.report_only:
        print(
            "[fatal] --logprobs is a paid live probe: use it with --mode live (or "
            "--report-only to read a probe's rows). It is a no-op for --mode replay/dry-run."
        )
        return 1

    # Report-only reads persisted results ($0) — it is orthogonal to --mode and may
    # carry --compare-archived, so it is handled before the paid-mode guards below.
    if args.report_only:
        return _run_report_only_mode(args)

    if args.mode == "live" and args.budget > BUDGET_CEILING_USD:
        print(f"[fatal] --budget ${args.budget:.2f} exceeds the ${BUDGET_CEILING_USD:.2f} ceiling.")
        return 1

    if args.compare_archived and args.mode != "live":
        print("[fatal] --compare-archived only applies to --mode live (it runs a paid judge).")
        return 1

    if args.mode == "replay":
        return _run_replay_mode(args)
    if args.mode == "dry-run":
        return _run_dry_run_mode(args)
    return _run_live_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
