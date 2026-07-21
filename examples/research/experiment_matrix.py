"""Expand recipes across local synthetic benchmarks, splits, and seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

from langres.experiments import EvaluationProtocol, Experiment, ExperimentReport

from _research_foundation import retrieve_factory


def build_protocol() -> EvaluationProtocol:
    """Declare the complete statistical question before any work starts."""
    return EvaluationProtocol(
        benchmark_ids=("local_companies", "local_products"),
        split_ids=("train", "test"),
        fixed_test_set_id="langres-local-research-matrix-v1",
        split_seeds=(0, 1),
        threshold_split_id="train",
        test_split_id="test",
        threshold_grid=(0.3, 0.5, 0.7),
        confidence_interval_method="none",
        bootstrap_samples=1,
        hardware_cohort="cpu-local",
        benchmark_version="1",
    )


def run_matrix(output_dir: Path) -> ExperimentReport:
    """Run two local architectures over the declared 16-cell matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return Experiment(
        architectures=(
            retrieve_factory(),
            retrieve_factory(name="RetrieveRerank", rerank=True),
        ),
        protocol=build_protocol(),
        store=output_dir / "runs.jsonl",
        cache_dir=output_dir / "stage-cache",
        budget_usd=0.0,
    ).run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/research-matrix"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run all 16 local cells; without this flag only print the plan",
    )
    args = parser.parse_args()
    protocol = build_protocol()
    cell_count = (
        2 * len(protocol.benchmark_ids) * len(protocol.split_ids) * len(protocol.split_seeds)
    )
    print(
        f"{cell_count} cells: 2 recipes × {len(protocol.benchmark_ids)} benchmarks "
        f"× {len(protocol.split_ids)} splits × {len(protocol.split_seeds)} seeds"
    )
    if args.execute:
        print(run_matrix(args.output_dir).to_markdown())


if __name__ == "__main__":
    main()
