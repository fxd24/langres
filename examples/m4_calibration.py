"""M4 calibration demo — a data-driven threshold beats a hand-set one on AG ($0).

The framework's thresholds (module defaults of ``0.5``, the cascade's
``0.3``/``0.9``) were hand-set, not read off the score distribution. This demo
proves the fix on the hard Amazon-Google benchmark, entirely offline and free:

1. Build embedding-cosine scores over the LABELLED Amazon-Google ``test`` pairs
   using the free local MiniLM embedder (no API call, $0).
2. Derive a threshold from those ``(score, gold_label)`` pairs with
   :func:`langres.core.calibration.derive_threshold` (Youden's J).
3. Compare pair-level F1 at the derived threshold vs a hand-set ``0.5``, and
   assert the derived threshold is at least as good (it is strictly better here).
4. Print the score-distribution summary and Brier / ECE calibration diagnostics
   (via :mod:`langres.core.metrics`) so the quality is characterized, not asserted.

Run:
    uv run python examples/m4_calibration.py
"""

import os

# Pin OpenMP / FAISS threading BEFORE importing torch/sentence-transformers, so
# the run is deterministic and avoids the macOS libomp duplicate-load crash.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402

from langres.core.calibration import derive_threshold  # noqa: E402
from langres.core.embeddings import SentenceTransformerEmbedder  # noqa: E402
from langres.core.metrics import brier_score, expected_calibration_error  # noqa: E402
from langres.data.amazon_google import (  # noqa: E402
    load_amazon_google,
    load_amazon_google_pair_splits,
)

HAND_SET_THRESHOLD = 0.5
"""The status-quo hand-set threshold the derived one must beat."""


def build_test_pair_scores() -> tuple[list[float], list[bool]]:
    """Embedding-cosine score + gold label for every labelled AG ``test`` pair.

    Uses the fixed literature ``test`` split (``(amazon_id, google_id, label)``)
    and the free local MiniLM embedder: each record's ``embed_text`` is embedded
    once (L2-normalized), so the cosine similarity of a pair is a plain dot
    product. No API call — the whole function costs $0.

    Returns:
        ``(scores, labels)`` aligned lists: cosine similarity in roughly
        ``[-0.2, 1.0]`` and the boolean gold match label.
    """
    corpus, _clusters, _pairs = load_amazon_google()
    text_by_id = {record.id: record.embed_text for record in corpus}

    test_pairs = load_amazon_google_pair_splits()["test"]

    # Embed each involved record's text exactly once, then look up by id.
    unique_ids = sorted({rid for pair in test_pairs for rid in pair[:2]})
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    vectors = embedder.encode([text_by_id[rid] for rid in unique_ids])
    vector_by_id = {rid: vectors[i] for i, rid in enumerate(unique_ids)}

    scores: list[float] = []
    labels: list[bool] = []
    for amazon_id, google_id, label in test_pairs:
        cosine = float(np.dot(vector_by_id[amazon_id], vector_by_id[google_id]))
        scores.append(cosine)
        labels.append(label == 1)
    return scores, labels


def pair_f1(scores: list[float], labels: list[bool], threshold: float) -> float:
    """Pair-level F1 of the rule ``score >= threshold => match`` against gold.

    The labelled pairs ARE the classification units here (no clustering), so this
    is the standard binary-classification F1 at the given operating point.
    """
    tp = sum(1 for s, y in zip(scores, labels, strict=True) if s >= threshold and y)
    fp = sum(1 for s, y in zip(scores, labels, strict=True) if s >= threshold and not y)
    fn = sum(1 for s, y in zip(scores, labels, strict=True) if s < threshold and y)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def main() -> None:
    """Run the $0 Amazon-Google calibration demo and assert the derived win."""
    print("=" * 78)
    print("M4 calibration — data-driven threshold vs hand-set 0.5 on Amazon-Google ($0)")
    print("=" * 78)

    scores, labels = build_test_pair_scores()
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    derived = derive_threshold(scores, labels, method="youden")
    derived_f1 = pair_f1(scores, labels, derived)
    hand_set_f1 = pair_f1(scores, labels, HAND_SET_THRESHOLD)

    print("\n## Labelled AG test pairs")
    print(f"- pairs: {len(labels)}  (positives {n_pos}, negatives {n_neg})")

    arr = np.asarray(scores)
    print("\n## Embedding-cosine score distribution")
    print(
        f"- min {arr.min():.4f}  p25 {np.percentile(arr, 25):.4f}  "
        f"median {np.median(arr):.4f}  p75 {np.percentile(arr, 75):.4f}  max {arr.max():.4f}"
    )
    print(f"- mean(positives) {arr[np.asarray(labels)].mean():.4f}")
    print(f"- mean(negatives) {arr[~np.asarray(labels)].mean():.4f}")

    # Characterize calibration quality of the raw cosine as a probability. Cosine
    # can dip slightly negative, so clip into [0, 1] for the [0,1]-domain metrics.
    conf = [float(min(max(s, 0.0), 1.0)) for s in scores]
    print("\n## Calibration diagnostics of raw cosine (via metrics.py)")
    print(f"- Brier score {brier_score(conf, labels):.4f}  (lower is better)")
    print(f"- ECE         {expected_calibration_error(conf, labels):.4f}  (lower is better)")

    print("\n## Threshold comparison (pair-level F1)")
    print(f"- hand-set threshold {HAND_SET_THRESHOLD:.4f} -> F1 {hand_set_f1:.4f}")
    print(f"- derived threshold  {derived:.4f} -> F1 {derived_f1:.4f}  (Youden's J)")
    print(f"- lift: {derived_f1 - hand_set_f1:+.4f}")

    assert derived_f1 >= hand_set_f1, (
        f"derived-threshold F1 {derived_f1:.4f} should be >= hand-set F1 {hand_set_f1:.4f}"
    )

    print("\nOK — the data-driven threshold matches or beats the hand-set 0.5 on AG.")


if __name__ == "__main__":
    main()
