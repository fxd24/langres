"""Measure whether an embedding resource separates known matches from non-matches."""

from collections.abc import Sequence

import numpy as np

from langres.resources import Embedder, FakeEmbedder

TEXTS = (
    "Acme Corporation",
    "ACME Corp",
    "Globex",
    "Globex LLC",
    "Umbrella Health",
)
MATCHES = ((0, 1), (2, 3))
NON_MATCHES = ((0, 2), (0, 4), (2, 4))


def cosine_scores(
    embedder: Embedder,
    pairs: Sequence[tuple[int, int]],
) -> tuple[float, ...]:
    """Embed once, then return one cosine score per declared pair."""
    vectors = embedder.embed(TEXTS).vectors
    return tuple(float(np.dot(vectors[left], vectors[right])) for left, right in pairs)


def separability_margin(embedder: Embedder) -> float:
    """Return mean(match cosine) minus mean(non-match cosine)."""
    match_scores = cosine_scores(embedder, MATCHES)
    non_match_scores = cosine_scores(embedder, NON_MATCHES)
    return float(np.mean(match_scores) - np.mean(non_match_scores))


def main() -> None:
    """Run the measurement without a download or network call."""
    embedder = FakeEmbedder(dimension=32)
    margin = separability_margin(embedder)
    print(f"mean match-minus-non-match cosine: {margin:.3f}")
    print("This fake-resource result tests the measurement, not semantic model quality.")


if __name__ == "__main__":
    main()
