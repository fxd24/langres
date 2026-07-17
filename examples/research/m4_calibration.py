"""M4 calibration demo — a data-driven threshold beats a hand-set one on AG ($0).

The framework's thresholds (module defaults of ``0.5``, the cascade's
``0.3``/``0.9``) were hand-set, not read off the score distribution. This demo
proves the fix on the hard Amazon-Google benchmark, entirely offline and free —
and honestly, on a held-out split (derive on ``train``, report on ``test``) so
the reported lift is not tuned on the pairs it is measured on:

1. Build embedding-cosine scores over the LABELLED Amazon-Google ``train`` and
   ``test`` pairs using the free local MiniLM embedder (no API call, $0).
2. Derive a threshold from the ``train`` ``(score, gold_label)`` pairs with
   :func:`langres.training.calibration.derive_threshold` (Youden's J) — the ``test``
   labels are never seen during derivation.
3. Compare pair-level F1 at the derived threshold vs a hand-set ``0.5`` on the
   held-out ``test`` split, and assert the derived threshold is at least as good.
4. Print the score-distribution summary and Brier / ECE calibration diagnostics
   (via :mod:`langres.core.metrics`) so the quality is characterized, not asserted.

Run:
    uv run python examples/research/m4_calibration.py
"""

import os

# Pin OpenMP / FAISS threading BEFORE importing torch/sentence-transformers, so
# the run is deterministic and avoids the macOS libomp duplicate-load crash.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402

from langres.training.calibration import derive_threshold  # noqa: E402
from langres.core.embeddings import SentenceTransformerEmbedder  # noqa: E402
from langres.core.metrics import brier_score, expected_calibration_error  # noqa: E402
from langres.data.amazon_google import (  # noqa: E402
    load_amazon_google,
    load_amazon_google_pair_splits,
)

HAND_SET_THRESHOLD = 0.5
"""The status-quo hand-set threshold the derived one must beat."""


def build_split_pair_scores() -> dict[str, tuple[list[float], list[bool]]]:
    """Embedding-cosine score + gold label for the labelled AG ``train`` and ``test`` pairs.

    Uses the fixed literature pair splits (``(amazon_id, google_id, label)``) and
    the free local MiniLM embedder: each involved record's ``embed_text`` is
    embedded once (L2-normalized), so a pair's cosine similarity is a plain dot
    product. No API call — the whole function costs $0. Returning both splits from
    one embedding pass keeps derivation (``train``) and reporting (``test``)
    strictly separated while embedding each record at most once.

    Returns:
        ``{"train": (scores, labels), "test": (scores, labels)}`` — cosine
        similarities in roughly ``[-0.2, 1.0]`` aligned with boolean gold labels.
    """
    corpus, _clusters, _pairs = load_amazon_google()
    text_by_id = {record.id: record.embed_text for record in corpus}

    splits = load_amazon_google_pair_splits()
    train_pairs, test_pairs = splits["train"], splits["test"]

    # Embed each record involved in either split exactly once, then look up by id.
    unique_ids = sorted({rid for pair in (*train_pairs, *test_pairs) for rid in pair[:2]})
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    vectors = embedder.encode([text_by_id[rid] for rid in unique_ids])
    vector_by_id = {rid: vectors[i] for i, rid in enumerate(unique_ids)}

    def score_split(pairs: list[tuple[str, str, int]]) -> tuple[list[float], list[bool]]:
        scores: list[float] = []
        labels: list[bool] = []
        for amazon_id, google_id, label in pairs:
            cosine = float(np.dot(vector_by_id[amazon_id], vector_by_id[google_id]))
            scores.append(cosine)
            labels.append(label == 1)
        return scores, labels

    return {"train": score_split(train_pairs), "test": score_split(test_pairs)}


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

    splits = build_split_pair_scores()
    train_scores, train_labels = splits["train"]
    test_scores, test_labels = splits["test"]

    # Derive on TRAIN only; the test labels are never seen during derivation.
    derived = derive_threshold(train_scores, train_labels, method="youden")

    # Report the lift on the held-out TEST split.
    derived_f1 = pair_f1(test_scores, test_labels, derived)
    hand_set_f1 = pair_f1(test_scores, test_labels, HAND_SET_THRESHOLD)

    n_pos = sum(test_labels)
    n_neg = len(test_labels) - n_pos
    print("\n## Splits")
    print(f"- train pairs (derivation): {len(train_labels)}")
    print(f"- test pairs (reporting):   {len(test_labels)}  (positives {n_pos}, negatives {n_neg})")

    arr = np.asarray(test_scores)
    print("\n## Held-out test embedding-cosine score distribution")
    print(
        f"- min {arr.min():.4f}  p25 {np.percentile(arr, 25):.4f}  "
        f"median {np.median(arr):.4f}  p75 {np.percentile(arr, 75):.4f}  max {arr.max():.4f}"
    )
    print(f"- mean(positives) {arr[np.asarray(test_labels)].mean():.4f}")
    print(f"- mean(negatives) {arr[~np.asarray(test_labels)].mean():.4f}")

    # Characterize calibration quality of the raw cosine as a probability. Cosine
    # can dip slightly negative, so clip into [0, 1] for the [0,1]-domain metrics.
    conf = [float(min(max(s, 0.0), 1.0)) for s in test_scores]
    print("\n## Calibration diagnostics of raw cosine on test (via metrics.py)")
    print(f"- Brier score {brier_score(conf, test_labels):.4f}  (lower is better)")
    print(f"- ECE         {expected_calibration_error(conf, test_labels):.4f}  (lower is better)")

    print("\n## Held-out threshold comparison (pair-level F1 on test)")
    print(f"- hand-set threshold {HAND_SET_THRESHOLD:.4f} -> F1 {hand_set_f1:.4f}")
    print(
        f"- derived threshold  {derived:.4f} -> F1 {derived_f1:.4f}  (Youden's J, tuned on train)"
    )
    print(f"- lift: {derived_f1 - hand_set_f1:+.4f}")

    assert derived_f1 >= hand_set_f1, (
        f"held-out derived-threshold F1 {derived_f1:.4f} should be >= hand-set F1 {hand_set_f1:.4f}"
    )

    print("\nOK — the train-derived threshold matches or beats the hand-set 0.5 on held-out test.")


if __name__ == "__main__":
    main()
