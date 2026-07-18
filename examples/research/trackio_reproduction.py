"""Re-run one protocol with local-only Trackio publication and no network."""

from __future__ import annotations

import argparse
from pathlib import Path

from langres.experiments import EvaluationProtocol, Experiment, ExperimentReport

from _research_foundation import retrieve_factory


def reproduce_with_trackio(output_dir: Path) -> ExperimentReport:
    """Re-execute the same protocol while mirroring metrics to local Trackio."""
    from langres.tracking.trackers import TrackioTracker

    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = TrackioTracker(project="langres-reproduction", space_id=None)
    return Experiment(
        architectures=(retrieve_factory(),),
        protocol=EvaluationProtocol.smoke(seed=0),
        tracker=tracker,
        store=output_dir / "runs.jsonl",
        cache_dir=output_dir / "stage-cache",
        budget_usd=0.0,
        resume=False,
    ).run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/trackio-reproduction"))
    args = parser.parse_args()
    print(reproduce_with_trackio(args.output_dir).to_markdown())


if __name__ == "__main__":
    main()
