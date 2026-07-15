"""RandomForestMatcher: sklearn RandomForest over ComparisonVector.similarities (W1.2, S2).

Magellan-style supervised judge (``SupervisedFitMixin.fit(candidates,
labels)``): a feature vector per candidate (one entry per declared
``FeatureSpec``, drawn from ``candidate.comparison.similarities``, with a
missing-feature sentinel for MISSING/None) trains an
``sklearn.ensemble.RandomForestClassifier``.

**Serialization never pickles or joblib-dumps the fitted estimator** (joblib
IS pickle, and a fitted forest cannot be rebuilt from its constructor
params) -- see ``docs/ROADMAP.md`` / ``serialization.py``'s no-pickle artifact
contract. Instead, the fitted forest is extracted into a strict, plain-JSON
per-tree array representation using the same public arrays
``sklearn.tree._tree.Tree`` exposes (``children_left``, ``children_right``,
``feature``, ``threshold``, ``value``) -- this is enough to walk every tree
and average class probabilities (exactly what
``RandomForestClassifier.predict_proba`` does internally) without ever
reconstructing a real sklearn estimator object. Persisted behind
:class:`~langres.core.serialization.SerializableState` (a ``forest.json``
sidecar, like ``DSPyMatcher``'s ``program.json``) with an sklearn-version guard
on load.
"""

import json
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import ClassVar, cast

import sklearn
from sklearn.ensemble import RandomForestClassifier

from langres.core.feature import ComparisonVector, FeatureSpec
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.matcher import Matcher, SchemaT
from langres.core.registry import register
from langres.core.reports import ScoreInspectionReport, _inspect_scores_impl

logger = logging.getLogger(__name__)

#: Feature value substituted for a MISSING (absent) comparator feature. Chosen
#: outside the valid similarity range ([0, 1]) so a tree split can always
#: separate "missing" from any real similarity value with a single threshold,
#: without adding a second "is_missing" indicator column per feature.
_MISSING_SENTINEL = -1.0

#: sklearn's ``TREE_LEAF`` sentinel: ``children_left[node] == _TREE_LEAF``
#: marks a leaf (this is the same convention ``sklearn.tree._tree`` uses).
_TREE_LEAF = -1

_STATE_FILENAME = "forest.json"


class _ForestState:
    """Inference-only representation of a fitted forest: plain per-tree arrays.

    Deliberately NOT an sklearn estimator -- this is the one canonical
    representation used both right after :meth:`RandomForestMatcher.fit` and after a
    :meth:`RandomForestMatcher.load_state`, so a freshly-fit judge and a reloaded judge
    score identically.
    """

    def __init__(self, trees: list[dict[str, list[object]]], positive_class_index: int | None):
        self.trees = trees
        self.positive_class_index = positive_class_index

    def predict_match_proba(self, x: list[float]) -> float:
        """Average, over every tree, the fraction voting for the positive class."""
        if self.positive_class_index is None:
            return 0.0  # the forest never saw a positive (match) label during fit
        probs = []
        for tree in self.trees:
            children_left = cast("list[int]", tree["children_left"])
            children_right = cast("list[int]", tree["children_right"])
            feature = cast("list[int]", tree["feature"])
            threshold = cast("list[float]", tree["threshold"])
            value = cast("list[list[float]]", tree["value"])

            node = 0
            while children_left[node] != _TREE_LEAF:
                node = (
                    children_left[node]
                    if x[feature[node]] <= threshold[node]
                    else children_right[node]
                )
            counts = value[node]
            total = sum(counts)
            probs.append(counts[self.positive_class_index] / total if total > 0 else 0.0)
        return sum(probs) / len(probs) if probs else 0.0


def _extract_forest_state(fitted: RandomForestClassifier) -> _ForestState:
    """Extract a fitted forest's per-tree arrays into a JSON-serializable state."""
    classes = list(fitted.classes_)
    positive_index = classes.index(1) if 1 in classes else None
    trees = []
    for estimator in fitted.estimators_:
        tree = estimator.tree_
        trees.append(
            {
                "children_left": tree.children_left.tolist(),
                "children_right": tree.children_right.tolist(),
                "feature": tree.feature.tolist(),
                "threshold": tree.threshold.tolist(),
                "value": [row[0].tolist() for row in tree.value],
            }
        )
    return _ForestState(trees=trees, positive_class_index=positive_index)


def _check_sklearn_version_guard(saved_version: str) -> None:
    """Refuse to load an artifact fit with a different scikit-learn minor version.

    The per-tree JSON arrays extracted here mirror a stable public sklearn API
    (``Tree.children_left`` etc. have been unchanged for many releases), but
    this guard errs on the side of caution rather than risk a silently
    mis-decoded tree on a future breaking change.
    """
    saved_major_minor = ".".join(saved_version.split(".")[:2])
    current_version = sklearn.__version__
    current_major_minor = ".".join(current_version.split(".")[:2])
    if saved_major_minor != current_major_minor:
        raise ValueError(
            f"RandomForestMatcher artifact was fit with scikit-learn {saved_version}, but the "
            f"current environment has scikit-learn {current_version}. Refusing to "
            "load a forest artifact across a scikit-learn minor-version boundary "
            "— re-fit the judge in this environment, or install "
            f"scikit-learn~={saved_major_minor} to load this artifact."
        )


@register("random_forest")
class RandomForestMatcher(Matcher[SchemaT]):
    """Supervised sklearn RandomForest judge over declared comparator features."""

    type_name: ClassVar[str] = "random_forest"

    def __init__(
        self,
        feature_specs: list[FeatureSpec],
        *,
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int = 0,
    ) -> None:
        """Initialize an (unfit) RandomForestMatcher.

        Args:
            feature_specs: The features to build the sklearn feature vector
                from, in order. Should match the pipeline's Comparator
                features (mirrors ``WeightedAverageMatcher``).
            n_estimators: Number of trees in the forest.
            max_depth: Maximum tree depth (``None`` = unbounded, sklearn's
                default).
            random_state: sklearn RNG seed for deterministic fits.
        """
        self.feature_specs = feature_specs
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._forest_state: _ForestState | None = None

    def _feature_vector(self, candidate: ERCandidate[SchemaT]) -> list[float]:
        vector: ComparisonVector | None = candidate.comparison
        if vector is None:
            raise ValueError(
                "RandomForestMatcher requires candidates carrying a comparison vector — add "
                "a Comparator to the pipeline."
            )
        return [
            vector.similarities.get(spec.name, _MISSING_SENTINEL) for spec in self.feature_specs
        ]

    # ------------------------------------------------------------------
    # Fitting (SupervisedFitMixin)
    # ------------------------------------------------------------------

    def fit(self, candidates: Iterator[ERCandidate[SchemaT]], labels: Sequence[bool]) -> None:
        """Fit the RandomForest from labeled candidate pairs.

        Args:
            candidates: The blocked, comparison-attached candidates to learn
                from.
            labels: Gold match/non-match labels, positionally aligned with
                ``candidates``.

        Raises:
            ValueError: If ``len(labels)`` does not match the number of
                candidates, or a candidate carries no comparison vector.
        """
        materialized = list(candidates)
        if len(materialized) != len(labels):
            raise ValueError(
                f"RandomForestMatcher.fit received {len(materialized)} candidates but "
                f"{len(labels)} labels — they must be positionally aligned "
                "and equal length."
            )
        x = [self._feature_vector(candidate) for candidate in materialized]
        y = [int(label) for label in labels]

        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        clf.fit(x, y)
        self._forest_state = _extract_forest_state(clf)

    # ------------------------------------------------------------------
    # Scoring (Matcher)
    # ------------------------------------------------------------------

    def forward(self, candidates: Iterator[ERCandidate[SchemaT]]) -> Iterator[PairwiseJudgement]:
        """Score each candidate with the forest's match-class probability.

        Yields:
            One PairwiseJudgement per candidate, ``score_type="prob_rf"``.

        Raises:
            ValueError: If the judge has not been fit yet, or a candidate
                carries no comparison vector.
        """
        if self._forest_state is None:
            raise ValueError(
                "RandomForestMatcher must be fit before forward(): call fit(candidates, "
                "labels) directly, or resolver.fit(records, labels=...) on a "
                "Resolver whose module is this judge."
            )
        for candidate in candidates:
            x = self._feature_vector(candidate)
            score = self._forest_state.predict_match_proba(x)
            yield PairwiseJudgement(
                left_id=candidate.left.id,  # type: ignore[attr-defined]
                right_id=candidate.right.id,  # type: ignore[attr-defined]
                score=score,
                score_type="prob_rf",
                decision_step="random_forest",
                provenance={"n_estimators": self.n_estimators},
            )

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        """Explore scores without ground truth (shared Matcher utility)."""
        return _inspect_scores_impl(judgements, sample_size)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @property
    def config(self) -> dict[str, object]:
        """Construction-only config (never the fitted forest — see save_state)."""
        return {
            "feature_specs": [spec.model_dump() for spec in self.feature_specs],
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "RandomForestMatcher[SchemaT]":
        """Reconstruct a fresh, UNFIT judge from :attr:`config`."""
        specs = [
            FeatureSpec.model_validate(s) for s in cast("list[object]", config["feature_specs"])
        ]
        return cls(
            feature_specs=specs,
            n_estimators=cast("int", config["n_estimators"]),
            max_depth=cast("int | None", config["max_depth"]),
            random_state=cast("int", config["random_state"]),
        )

    def save_state(self, state_dir: Path) -> None:
        """Persist the fitted forest as plain per-tree JSON arrays.

        Writes nothing when the judge has never been fit (mirrors
        ``DSPyMatcher``'s "nothing to save" behavior for an uncompiled judge) —
        the Resolver drops an empty sidecar directory.
        """
        if self._forest_state is None:
            return
        payload = {
            "sklearn_version": sklearn.__version__,
            "positive_class_index": self._forest_state.positive_class_index,
            "trees": self._forest_state.trees,
        }
        (state_dir / _STATE_FILENAME).write_text(json.dumps(payload))

    def load_state(self, state_dir: Path) -> None:
        """Restore the fitted forest from ``forest.json`` written by :meth:`save_state`.

        Raises:
            ValueError: If the artifact was fit with a different scikit-learn
                minor version than the current environment.
        """
        path = state_dir / _STATE_FILENAME
        if not path.exists():
            return  # never fitted
        payload = json.loads(path.read_text())
        _check_sklearn_version_guard(cast("str", payload["sklearn_version"]))
        self._forest_state = _ForestState(
            trees=cast("list[dict[str, list[object]]]", payload["trees"]),
            positive_class_index=cast("int | None", payload["positive_class_index"]),
        )
