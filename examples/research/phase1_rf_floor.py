"""Phase 1 (#80): the honest, $0 RandomForestJudge pair-level floor on AG + Abt-Buy.

"Prove the seam" Phase 1 asks a blunt question: *before* spending a cent on an
LLM judge, how far does a purely local, supervised baseline get on the standard
Amazon-Google and Abt-Buy literature pair splits — and how far is that from the
Ditto SOTA band (pairwise F1 0.756 AG / 0.893 Abt-Buy)?

This script answers it with **zero spend** (rapidfuzz + scikit-learn only):

    1. :class:`~langres.data.fixed_split_pair_benchmark.FixedSplitPairBenchmark`
       turns each dataset's fixed ``(id_a, id_b, label)`` splits into
       ``ERCandidate`` objects carrying a
       :class:`~langres.core.comparator.StringComparator` comparison vector.
    2. :class:`~langres.core.modules.random_forest_judge.RandomForestJudge` is
       fit on the FULL train split's candidates + labels.
    3. :func:`~langres.data.fixed_split_pair_benchmark.evaluate_fixed_split_honest`
       grades it on the FULL test split at a threshold DERIVED ON TRAIN (Youden),
       and *also* reports the leaky "argmax-F1-on-test" number so the honesty
       delta is explicit.

**Honest framing (important).** This is a *single-metric* floor: the
StringComparator emits ONE rapidfuzz ``token_sort_ratio`` similarity per string
field (title/manufacturer/price for AG; name/description/price for Abt-Buy). It
is **not** a multi-metric "Magellan-class" replication (which would add
Jaccard/overlap/numeric/edit features per field). So the gap to Ditto below is
the gap of a deliberately-thin local baseline — a floor to beat, not a ceiling
to celebrate.

Artifacts (JSON per dataset + a combined Markdown) are written to
``data/benchmarks/phase1/``.

Run ($0, no API key, no ``--env-file .env``):
    uv run python examples/research/phase1_rf_floor.py
"""

import os

# Pin OpenMP / FAISS threading BEFORE importing the dataset loaders (which pull
# torch/faiss transitively via VectorBlocker), matching the other research
# scripts — keeps the run deterministic and dodges the macOS libomp crash.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json  # noqa: E402
import logging  # noqa: E402
from collections.abc import Callable, Sequence  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from pydantic import BaseModel  # noqa: E402

from langres.core.metrics import PairMetrics  # noqa: E402
from langres.core.modules.random_forest_judge import RandomForestJudge  # noqa: E402
from langres.data.abt_buy import (  # noqa: E402
    AbtBuySchema,
    load_abt_buy,
    load_abt_buy_pair_splits,
)
from langres.data.amazon_google import (  # noqa: E402
    ProductSchema,
    load_amazon_google,
    load_amazon_google_pair_splits,
)
from langres.data.fixed_split_pair_benchmark import (  # noqa: E402
    FixedSplitPairBenchmark,
    HonestPairEval,
    evaluate_fixed_split_honest,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#: Ditto pairwise-F1 SOTA band per dataset (the gap target for this floor).
DITTO_F1: dict[str, float] = {"amazon_google": 0.756, "abt_buy": 0.893}

#: Where the JSON + Markdown artifacts land (repo-root ``data/benchmarks``).
_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "phase1"


def _build_benchmark(
    name: str,
    schema: type[BaseModel],
    corpus_loader: Callable[[], tuple[Sequence[Any], object, object]],
    pair_split_loader: Callable[[], dict[str, list[tuple[str, str, int]]]],
) -> FixedSplitPairBenchmark[Any]:
    """Assemble a FixedSplitPairBenchmark for one dataset from its loaders."""
    return FixedSplitPairBenchmark.from_loaders(
        name=name,
        schema=schema,
        corpus_loader=corpus_loader,
        pair_split_loader=pair_split_loader,
    )


def _metrics_dict(metrics: PairMetrics) -> dict[str, float | int]:
    """Flatten a PairMetrics into a JSON-friendly dict (threshold included)."""
    return {
        "threshold": metrics.threshold,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
    }


def run_rf_floor(benchmark: FixedSplitPairBenchmark[Any]) -> tuple[HonestPairEval, dict[str, Any]]:
    """Fit RandomForestJudge on train and grade it honestly on the full test split.

    Args:
        benchmark: The dataset's fixed-split adapter.

    Returns:
        ``(result, artifact)`` — the :class:`HonestPairEval` and a JSON-ready
        artifact dict (shapes, honest vs. leaky metrics, gap to Ditto).
    """
    train = benchmark.build("train")
    test = benchmark.build("test")

    judge: RandomForestJudge[Any] = RandomForestJudge(feature_specs=benchmark.feature_specs)
    judge.fit(iter(train.candidates), train.labels)

    result = evaluate_fixed_split_honest(judge, benchmark, derive_on="train")

    ditto = DITTO_F1[benchmark.name]
    artifact: dict[str, Any] = {
        "dataset": benchmark.name,
        "method": "RandomForestJudge",
        "comparator": (
            "StringComparator — one rapidfuzz token_sort_ratio similarity per "
            "string field (single-metric floor, NOT multi-metric Magellan-class)"
        ),
        "features": [spec.name for spec in benchmark.feature_specs],
        "n_train": len(train.candidates),
        "n_train_pos": sum(train.labels),
        "n_test": len(test.candidates),
        "n_test_pos": len(test.gold),
        "threshold_method": result.threshold_method,
        "derive_on": result.derive_on,
        "derived_threshold": result.derived_threshold,
        "honest": _metrics_dict(result.honest),
        "argmax_on_test": _metrics_dict(result.argmax_on_test),
        "honesty_delta_f1": result.honesty_delta_f1,
        "ditto_f1": ditto,
        "gap_to_ditto_f1": ditto - result.honest.f1,
        "notes": (
            "Zero-spend (rapidfuzz + scikit-learn). HONEST f1 uses a threshold "
            "derived on TRAIN via Youden and applied to the FULL test split; "
            "argmax_on_test is the leaky ceiling that tunes the threshold on the "
            "test set itself (honesty_delta_f1 = argmax_on_test.f1 - honest.f1). "
            "gap_to_ditto_f1 measures the honest number against the Ditto SOTA "
            "band; the gap is expected — this is a single-metric StringComparator "
            "floor, not a multi-feature replication."
        ),
    }
    return result, artifact


def format_report(artifacts: list[dict[str, Any]]) -> str:
    """Render the combined Markdown summary for all datasets."""
    lines = [
        "# Phase 1 — RandomForestJudge honest pair-level floor ($0)",
        "",
        "Single-metric `StringComparator` (one rapidfuzz `token_sort_ratio` per "
        "field) + `RandomForestJudge`, graded on the **full standard test split** "
        "at a threshold **derived on train** (Youden). `argmax_on_test` is the "
        "leaky ceiling (threshold tuned on test) shown only to expose the honesty "
        "delta. This is a floor to beat, not a Magellan-class multi-feature "
        "replication.",
        "",
        "| dataset | honest P | honest R | honest F1 | argmax-on-test F1 | honesty Δ | threshold | Ditto F1 | gap to Ditto |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for art in artifacts:
        honest = art["honest"]
        lines.append(
            f"| {art['dataset']} "
            f"| {honest['precision']:.4f} | {honest['recall']:.4f} | {honest['f1']:.4f} "
            f"| {art['argmax_on_test']['f1']:.4f} | {art['honesty_delta_f1']:.4f} "
            f"| {art['derived_threshold']:.4f} | {art['ditto_f1']:.3f} "
            f"| {art['gap_to_ditto_f1']:.4f} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- **honest F1** is the number that matters: no test-label peeking.",
        "- **honesty Δ** = how much an argmax-on-test report would have inflated "
        "F1 over the honest cut — the exact leakage this Phase 1 seam removes.",
        "- **gap to Ditto** is the distance a $0, single-metric local baseline "
        "leaves for the paid/multi-feature judges the later phases add.",
        "",
        "Per-dataset detail (shapes, tp/fp/fn, features) is in the sibling "
        "`phase1_rf_floor_<dataset>.json` files.",
    ]
    return "\n".join(lines)


def main() -> None:
    """Run the AG + Abt-Buy honest floor, print it, and write the artifacts."""
    datasets: list[tuple[str, type[BaseModel], Any, Any]] = [
        ("amazon_google", ProductSchema, load_amazon_google, load_amazon_google_pair_splits),
        ("abt_buy", AbtBuySchema, load_abt_buy, load_abt_buy_pair_splits),
    ]

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []

    print("=" * 78)
    print("Phase 1 — RandomForestJudge honest pair-level floor (seed-free, $0)")
    print("=" * 78)

    for name, schema, corpus_loader, pair_split_loader in datasets:
        benchmark = _build_benchmark(name, schema, corpus_loader, pair_split_loader)
        result, artifact = run_rf_floor(benchmark)
        artifacts.append(artifact)

        json_path = _OUTPUT_DIR / f"phase1_rf_floor_{name}.json"
        json_path.write_text(json.dumps(artifact, indent=2))

        print(f"\n## {name}")
        print(f"features:            {artifact['features']}")
        print(f"train/test pairs:    {artifact['n_train']} / {artifact['n_test']}")
        print(
            f"HONEST (train-derived thr={result.derived_threshold:.4f}): "
            f"P={result.honest.precision:.4f} R={result.honest.recall:.4f} "
            f"F1={result.honest.f1:.4f}"
        )
        print(
            f"argmax-on-test (leaky):   F1={result.argmax_on_test.f1:.4f} "
            f"(honesty delta {result.honesty_delta_f1:+.4f})"
        )
        print(
            f"Ditto F1={artifact['ditto_f1']:.3f}  ->  gap to Ditto "
            f"{artifact['gap_to_ditto_f1']:+.4f}"
        )

    report = format_report(artifacts)
    md_path = _OUTPUT_DIR / "PHASE1_RESULTS.md"
    md_path.write_text(report + "\n")

    print("\n" + "=" * 78)
    print(report)
    print("\nWrote artifacts to", _OUTPUT_DIR)


if __name__ == "__main__":
    main()
