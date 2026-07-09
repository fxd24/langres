"""Offline ($0) replication of Peeters, Steiner & Bizer (EDBT 2025) LLM-EM F1.

Reproduces a cell of arXiv 2310.11244 v4 **Table 2** (per-dataset, per-prompt,
single fixed prompt — *not* the best-of-prompt Table 4) by **replaying the
authors' archived raw model answers** through langres. No API key, no LLM call,
**no spend**: the harness only downloads their public answer archive, parses the
stored answers with our parser, aligns them to the gold labels of our
regenerated pair-set slice, and scores pairwise precision/recall/F1 with
``langres.core.metrics.classify_pairs`` (validating *our* metric code).

It also verifies the **prompt round-trip**: it renders each prompt itself from
our vendored records + serializer and diffs it against the ``prompt`` field the
authors archived, reporting the byte-exact match rate.

Target: ``abt-buy`` / ``gpt-4-0613`` / ``domain-complex-force`` → F1 **95.15**.

Data licensing: MatchGPT ships no LICENSE (``license: null``); langres is
Apache-2.0. Nothing from MatchGPT is vendored — the ~186 MB answer archive is
downloaded transiently to a cache dir (``--cache-dir``, default the system temp
dir) and never committed. Our pair-set slice is regenerated from our own
already-vendored DeepMatcher CSVs.

Usage::

    uv run python examples/research/peeters_llm_em_replication.py
    uv run python examples/research/peeters_llm_em_replication.py --dataset amazon-google
    uv run python examples/research/peeters_llm_em_replication.py --model gpt-4o-2024-08-06
"""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from langres.core.metrics import classify_pairs
from langres.data.peeters import (
    get_peeters_replication,
    gold_match_pairs,
    judgements_from_answers,
    list_peeters_replications,
    render_sample_prompts,
)

#: The authors' Git-LFS answer archive (real bytes, not an LFS pointer).
ARCHIVE_URL = (
    "https://media.githubusercontent.com/media/wbsg-uni-mannheim/MatchGPT/"
    "main/LLMForEM/prompts-and-answers/prompts_and_answers.zip"
)

#: Published arXiv v4 Table 2 F1 (%) per (dataset, model, prompt-design) — the
#: cells this harness can assert against. Extend as more are verified.
PUBLISHED_F1 = {
    ("abt-buy", "gpt-4-0613", "domain-complex-force"): 95.15,
}

#: 2-decimal reporting tolerance: our exact F1 (e.g. 95.1456) must round to the
#: published 2-dp value; 0.05 absorbs that rounding without hiding a real gap.
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


def replay(dataset: str, model: str, prompt_design: str, cache_dir: Path) -> None:
    """Replay one archived run and report prompt round-trip + pairwise F1."""
    spec = get_peeters_replication(dataset)
    prompts = render_sample_prompts(spec)  # our records + serializer, in sample order

    answers_path = ensure_answers(cache_dir, dataset, prompt_design, model)
    archived = [json.loads(line) for line in answers_path.read_text().splitlines() if line.strip()]

    if len(archived) != len(prompts):
        raise SystemExit(
            f"archive has {len(archived)} lines but our sample has {len(prompts)} pairs — "
            "alignment would be wrong; aborting."
        )

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
    if target is not None:
        delta = abs(f1_pct - target)
        ok = delta <= F1_TOLERANCE
        print(f"published Table 2 F1 : {target:.2f}  (Δ={delta:.4f}, tol={F1_TOLERANCE})")
        if not ok:
            raise SystemExit(f"FAIL: replayed F1 {f1_pct:.2f} != published {target:.2f}")
        print("REPRODUCED ✓")
    else:
        print("(no published-F1 assertion registered for this cell — reported only)")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--dataset",
        default="abt-buy",
        choices=list_peeters_replications(),
        help="Which replication slice to replay (default: abt-buy).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4-0613",
        help="Archived model id, e.g. gpt-4-0613, gpt-4o-2024-08-06, gpt-3.5-turbo-0613.",
    )
    parser.add_argument(
        "--prompt-design",
        default="domain-complex-force",
        help="Archived prompt design (default: domain-complex-force = Table 2 target).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "langres-peeters-replication",
        help="Where to download/extract the answer archive (gitignored, not committed).",
    )
    args = parser.parse_args()

    print("Offline replay — NO API key, NO LLM call, $0 spend. Replaying archived answers.\n")
    replay(args.dataset, args.model, args.prompt_design, args.cache_dir)


if __name__ == "__main__":
    main()
