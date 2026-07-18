"""Run the real experiment runner on a bundled tiny fixture, offline and at $0."""

from __future__ import annotations

import argparse
from pathlib import Path

from langres.experiments import EvaluationProtocol, Experiment, ExperimentReport

from _research_foundation import retrieve_factory


def run_first_experiment(output_dir: Path) -> ExperimentReport:
    """Execute one local recipe × benchmark × test-split cell."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return Experiment(
        architectures=(retrieve_factory(),),
        protocol=EvaluationProtocol.smoke(seed=0),
        store=output_dir / "runs.jsonl",
        cache_dir=output_dir / "stage-cache",
        budget_usd=0.0,
    ).run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/first-experiment"))
    args = parser.parse_args()
    report = run_first_experiment(args.output_dir)
    print(report.to_markdown())


if __name__ == "__main__":
    main()
