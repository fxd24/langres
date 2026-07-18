"""Save and reload locally; optionally push the same allowlisted bundle privately."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import BaseModel

from langres.architectures import FuzzyString


class HubLifecycleCompany(BaseModel):
    """Example-specific schema name that cannot collide with generic test fixtures."""

    id: str
    name: str


def local_round_trip(path: Path) -> FuzzyString:
    """Create a validated local bundle and reconstruct its exact architecture."""
    model = FuzzyString(schema=HubLifecycleCompany, threshold=0.7)
    model.save_pretrained(
        path,
        measurement_summary={
            "protocol_id": "tiny-offline-v1",
            "evaluation_id": "tiny-offline-test",
            "dataset_ids": ("tiny_fixture:test",),
            "quality": {"pair_f1": 1.0},
            "cost": {"usd": 0.0},
        },
    )
    return FuzzyString.from_pretrained(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("artifacts/fuzzy-company-v1"))
    parser.add_argument(
        "--push-to",
        metavar="ORG/REPO",
        help="explicitly opt into a network upload; the repository is always private",
    )
    args = parser.parse_args()
    loaded = local_round_trip(args.path)
    print(f"loaded {type(loaded).__name__} at threshold={loaded.clusterer.threshold}")
    if args.push_to:
        result = loaded.push_to_hub(
            args.push_to,
            private=True,
            revision="main",
            commit_message="Publish langres reproduction bundle",
        )
        pinned = FuzzyString.from_pretrained(args.push_to, revision=result.commit_oid)
        print(f"resolved Hub revision: {pinned.pretrained_source_.resolved_revision}")


if __name__ == "__main__":
    main()
