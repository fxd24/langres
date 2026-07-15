"""Tests for RandomForestMatcher — a Magellan-style sklearn RandomForest judge (W1.2, S2).

RandomForestMatcher is supervised (``SupervisedFitMixin.fit(candidates, labels)``) over the
same ``ComparisonVector.similarities`` seam as ``WeightedAverageMatcher`` and
``FellegiSunterMatcher``. It never pickles/joblib-dumps the fitted sklearn
estimator (a hard requirement — see the branch spec and E7): fit state is
extracted into a strict, plain-JSON per-tree array representation
(``children_left``/``children_right``/``feature``/``threshold``/``value``,
the same public arrays :class:`sklearn.tree._tree.Tree` exposes) behind a
:class:`~langres.core.serialization.SerializableState` sidecar, with an
sklearn-version guard on load.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

from langres.core.comparator import StringComparator
from langres.core.feature import ComparisonLevel, ComparisonVector, FeatureSpec
from langres.core.models import CompanySchema, ERCandidate
from langres.core.matchers.random_forest_judge import RandomForestMatcher
from langres.core.registry import get_component
from langres.core.serialization import SerializableState


class ProductSchema(BaseModel):
    """Second schema for schema-agnostic verification."""

    id: str
    title: str
    brand: str | None = None


def _company(id: str, name: str, address: str | None = None) -> CompanySchema:
    return CompanySchema(id=id, name=name, address=address)


def _candidate(
    left: CompanySchema,
    right: CompanySchema,
    comparison: ComparisonVector | None = None,
) -> ERCandidate[CompanySchema]:
    return ERCandidate(left=left, right=right, blocker_name="test", comparison=comparison)


def _compared(
    comparator: StringComparator[CompanySchema],
    left: CompanySchema,
    right: CompanySchema,
) -> ERCandidate[CompanySchema]:
    return _candidate(left, right, comparison=comparator.compare(left, right))


def _company_comparator() -> StringComparator[CompanySchema]:
    return StringComparator.from_schema(CompanySchema)


def _labeled_dataset(
    comparator: StringComparator[CompanySchema], n_matches: int = 15, n_nonmatches: int = 15
) -> tuple[list[ERCandidate[CompanySchema]], list[bool]]:
    """A synthetic, separable labeled dataset (mirrors the FS judge's fixture)."""
    candidates = []
    labels = []
    for i in range(n_matches):
        left = _company(f"m{i}L", f"Acme Corporation {i}", f"{i} Main Street")
        right = _company(f"m{i}R", f"Acme Corporation {i}", f"{i} Main Street")
        candidates.append(_compared(comparator, left, right))
        labels.append(True)
    for i in range(n_nonmatches):
        left = _company(f"n{i}L", f"Zephyr Holdings {i}", f"{i} Ocean Avenue")
        right = _company(f"n{i}R", f"Quasar Industries {i}", f"{i} Mountain Road")
        candidates.append(_compared(comparator, left, right))
        labels.append(False)
    return candidates, labels


# ---------------------------------------------------------------------------
# fit() basics
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_then_forward_separates_matches(self) -> None:
        comparator = _company_comparator()
        candidates, labels = _labeled_dataset(comparator, n_matches=20, n_nonmatches=20)
        judge: RandomForestMatcher[CompanySchema] = RandomForestMatcher(
            feature_specs=comparator.feature_specs, n_estimators=20, random_state=0
        )
        judge.fit(iter(candidates), labels)

        judgements = list(judge.forward(iter(candidates)))
        match_scores = [j.score for j, label in zip(judgements, labels, strict=True) if label]
        nonmatch_scores = [
            j.score for j, label in zip(judgements, labels, strict=True) if not label
        ]
        assert min(match_scores) > max(nonmatch_scores)

    def test_fit_raises_on_length_mismatch(self) -> None:
        comparator = _company_comparator()
        candidates, _ = _labeled_dataset(comparator, n_matches=2, n_nonmatches=2)
        judge: RandomForestMatcher[CompanySchema] = RandomForestMatcher(
            feature_specs=comparator.feature_specs
        )
        with pytest.raises(ValueError, match="labels"):
            judge.fit(iter(candidates), [True, False])  # 4 candidates, 2 labels

    def test_fit_raises_without_comparison_vector(self) -> None:
        judge: RandomForestMatcher[CompanySchema] = RandomForestMatcher(
            feature_specs=[FeatureSpec(name="name")]
        )
        bare = _candidate(_company("a", "Acme"), _company("b", "Acme Inc"), comparison=None)
        with pytest.raises(ValueError, match="comparison vector"):
            judge.fit(iter([bare]), [True])

    def test_fit_is_schema_agnostic_with_product_schema(self) -> None:
        comparator: StringComparator[ProductSchema] = StringComparator.from_schema(ProductSchema)
        candidates = []
        labels = []
        for i in range(6):
            left = ProductSchema(id=f"{i}L", title=f"Widget {i}", brand="Acme")
            right = ProductSchema(id=f"{i}R", title=f"Widget {i}", brand="Acme")
            candidates.append(
                ERCandidate(
                    left=left,
                    right=right,
                    blocker_name="test",
                    comparison=comparator.compare(left, right),
                )
            )
            labels.append(i % 2 == 0)
        judge: RandomForestMatcher[ProductSchema] = RandomForestMatcher(
            feature_specs=comparator.feature_specs, n_estimators=5, random_state=0
        )
        judge.fit(iter(candidates), labels)
        [judgement] = list(judge.forward(iter(candidates[:1])))
        assert judgement.score_type == "prob_rf"

    def test_fit_with_single_feature_no_crash(self) -> None:
        specs = [FeatureSpec(name="name")]
        comparator: StringComparator[CompanySchema] = StringComparator(specs, schema=CompanySchema)
        candidates, labels = _labeled_dataset(comparator, n_matches=5, n_nonmatches=5)
        judge: RandomForestMatcher[CompanySchema] = RandomForestMatcher(
            feature_specs=specs, n_estimators=5, random_state=0
        )
        judge.fit(iter(candidates), labels)
        judgements = list(judge.forward(iter(candidates)))
        assert len(judgements) == 10

    def test_fit_all_positive_labels_no_crash(self) -> None:
        """A degenerate single-class training set (RF never saw a non-match)."""
        comparator = _company_comparator()
        candidates, _ = _labeled_dataset(comparator, n_matches=6, n_nonmatches=0)
        judge: RandomForestMatcher[CompanySchema] = RandomForestMatcher(
            feature_specs=comparator.feature_specs, n_estimators=5, random_state=0
        )
        judge.fit(iter(candidates), [True] * 6)
        judgements = list(judge.forward(iter(candidates)))
        for judgement in judgements:
            assert 0.0 <= judgement.score <= 1.0

    def test_fit_all_negative_labels_no_crash(self) -> None:
        """A degenerate single-class training set (RF never saw a match)."""
        comparator = _company_comparator()
        candidates, _ = _labeled_dataset(comparator, n_matches=0, n_nonmatches=6)
        judge: RandomForestMatcher[CompanySchema] = RandomForestMatcher(
            feature_specs=comparator.feature_specs, n_estimators=5, random_state=0
        )
        judge.fit(iter(candidates), [False] * 6)
        judgements = list(judge.forward(iter(candidates)))
        for judgement in judgements:
            assert judgement.score == pytest.approx(0.0)

    def test_missing_feature_uses_sentinel_not_crash(self) -> None:
        comparator = _company_comparator()
        candidates, labels = _labeled_dataset(comparator, n_matches=5, n_nonmatches=5)
        # A candidate with an entirely empty comparison vector (all MISSING).
        empty = _candidate(
            _company("x", "X"),
            _company("y", "Y"),
            comparison=ComparisonVector(levels={}, similarities={}),
        )
        judge: RandomForestMatcher[CompanySchema] = RandomForestMatcher(
            feature_specs=comparator.feature_specs, n_estimators=5, random_state=0
        )
        judge.fit(iter(candidates), labels)
        [judgement] = list(judge.forward(iter([empty])))
        assert 0.0 <= judgement.score <= 1.0


# ---------------------------------------------------------------------------
# forward() basics
# ---------------------------------------------------------------------------


class TestForward:
    def test_forward_raises_before_fit(self) -> None:
        judge: RandomForestMatcher[CompanySchema] = RandomForestMatcher(
            feature_specs=[FeatureSpec(name="name")]
        )
        candidate = _candidate(
            _company("a", "Acme"),
            _company("b", "Acme Inc"),
            comparison=ComparisonVector(
                levels={"name": ComparisonLevel.PRESENT}, similarities={"name": 0.9}
            ),
        )
        with pytest.raises(ValueError, match="fit"):
            list(judge.forward(iter([candidate])))

    def test_forward_raises_without_comparison_vector(self) -> None:
        comparator = _company_comparator()
        candidates, labels = _labeled_dataset(comparator, n_matches=3, n_nonmatches=3)
        judge: RandomForestMatcher[CompanySchema] = RandomForestMatcher(
            feature_specs=comparator.feature_specs, n_estimators=5, random_state=0
        )
        judge.fit(iter(candidates), labels)
        bare = _candidate(_company("a", "Acme"), _company("b", "Acme Inc"), comparison=None)
        with pytest.raises(ValueError, match="comparison vector"):
            list(judge.forward(iter([bare])))

    def test_forward_left_id_right_id(self) -> None:
        comparator = _company_comparator()
        candidates, labels = _labeled_dataset(comparator, n_matches=5, n_nonmatches=5)
        judge: RandomForestMatcher[CompanySchema] = RandomForestMatcher(
            feature_specs=comparator.feature_specs, n_estimators=5, random_state=0
        )
        judge.fit(iter(candidates), labels)
        one = _compared(comparator, _company("left-1", "Acme"), _company("right-1", "Acme"))
        [judgement] = list(judge.forward(iter([one])))
        assert judgement.left_id == "left-1"
        assert judgement.right_id == "right-1"

    def test_inspect_scores_delegates_to_shared_util(self) -> None:
        comparator = _company_comparator()
        candidates, labels = _labeled_dataset(comparator, n_matches=5, n_nonmatches=5)
        judge: RandomForestMatcher[CompanySchema] = RandomForestMatcher(
            feature_specs=comparator.feature_specs, n_estimators=5, random_state=0
        )
        judge.fit(iter(candidates), labels)
        judgements = list(judge.forward(iter(candidates)))
        report = judge.inspect_scores(judgements, sample_size=5)
        assert report.total_judgements == len(judgements)


# ---------------------------------------------------------------------------
# Serialization: config (construction-only) / SerializableState (fitted state)
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_is_registered_with_type_name(self) -> None:
        assert get_component("random_forest") is RandomForestMatcher
        assert RandomForestMatcher.type_name == "random_forest"

    def test_implements_serializable_state(self) -> None:
        judge: RandomForestMatcher[CompanySchema] = RandomForestMatcher(
            feature_specs=[FeatureSpec(name="name")]
        )
        assert isinstance(judge, SerializableState)

    def test_config_excludes_fitted_forest(self) -> None:
        comparator = _company_comparator()
        candidates, labels = _labeled_dataset(comparator, n_matches=5, n_nonmatches=5)
        judge: RandomForestMatcher[CompanySchema] = RandomForestMatcher(
            feature_specs=comparator.feature_specs, n_estimators=7, max_depth=3, random_state=0
        )
        judge.fit(iter(candidates), labels)
        config = judge.config
        json.dumps(config)  # pure, JSON-serializable construction data
        assert config["n_estimators"] == 7
        assert config["max_depth"] == 3
        assert config["random_state"] == 0
        assert "forest" not in config
        assert "trees" not in config

    def test_from_config_builds_fresh_unfit_judge(self) -> None:
        comparator = _company_comparator()
        original: RandomForestMatcher[CompanySchema] = RandomForestMatcher(
            feature_specs=comparator.feature_specs, n_estimators=9, random_state=2
        )
        rebuilt = RandomForestMatcher.from_config(original.config)
        assert rebuilt.n_estimators == 9
        assert rebuilt.random_state == 2
        with pytest.raises(ValueError, match="fit"):
            list(
                rebuilt.forward(
                    iter([_compared(comparator, _company("a", "X"), _company("b", "Y"))])
                )
            )

    def test_save_state_before_fit_writes_nothing(self, tmp_path: Path) -> None:
        judge: RandomForestMatcher[CompanySchema] = RandomForestMatcher(
            feature_specs=[FeatureSpec(name="name")]
        )
        judge.save_state(tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_save_state_load_state_round_trips_predictions(self, tmp_path: Path) -> None:
        comparator = _company_comparator()
        candidates, labels = _labeled_dataset(comparator, n_matches=10, n_nonmatches=10)
        judge: RandomForestMatcher[CompanySchema] = RandomForestMatcher(
            feature_specs=comparator.feature_specs, n_estimators=10, random_state=0
        )
        judge.fit(iter(candidates), labels)
        original_scores = [j.score for j in judge.forward(iter(candidates))]

        judge.save_state(tmp_path)
        assert (tmp_path / "forest.json").exists()
        json.loads((tmp_path / "forest.json").read_text())  # valid JSON

        fresh = RandomForestMatcher.from_config(judge.config)
        fresh.load_state(tmp_path)
        reloaded_scores = [j.score for j in fresh.forward(iter(candidates))]

        assert reloaded_scores == pytest.approx(original_scores)

    def test_load_state_without_saved_file_stays_unfit(self, tmp_path: Path) -> None:
        judge: RandomForestMatcher[CompanySchema] = RandomForestMatcher(
            feature_specs=[FeatureSpec(name="name")]
        )
        judge.load_state(tmp_path)  # empty dir, no forest.json
        candidate = _candidate(
            _company("a", "Acme"),
            _company("b", "Acme Inc"),
            comparison=ComparisonVector(
                levels={"name": ComparisonLevel.PRESENT}, similarities={"name": 0.9}
            ),
        )
        with pytest.raises(ValueError, match="fit"):
            list(judge.forward(iter([candidate])))

    def test_load_state_rejects_sklearn_version_mismatch(self, tmp_path: Path) -> None:
        comparator = _company_comparator()
        candidates, labels = _labeled_dataset(comparator, n_matches=5, n_nonmatches=5)
        judge: RandomForestMatcher[CompanySchema] = RandomForestMatcher(
            feature_specs=comparator.feature_specs, n_estimators=5, random_state=0
        )
        judge.fit(iter(candidates), labels)
        judge.save_state(tmp_path)

        payload = json.loads((tmp_path / "forest.json").read_text())
        payload["sklearn_version"] = "0.1.0"  # a version that cannot match the installed one
        (tmp_path / "forest.json").write_text(json.dumps(payload))

        fresh = RandomForestMatcher.from_config(judge.config)
        with pytest.raises(ValueError, match="scikit-learn"):
            fresh.load_state(tmp_path)

    def test_resolver_with_random_forest_saves_and_loads(self, tmp_path: Path) -> None:
        from langres.core import AllPairsBlocker, Clusterer, Resolver

        comparator = _company_comparator()
        candidates, labels = _labeled_dataset(comparator, n_matches=10, n_nonmatches=10)
        judge: RandomForestMatcher[CompanySchema] = RandomForestMatcher(
            feature_specs=comparator.feature_specs, n_estimators=10, random_state=0
        )
        judge.fit(iter(candidates), labels)
        resolver = Resolver(
            blocker=AllPairsBlocker(schema=CompanySchema),
            comparator=comparator,
            matcher=judge,
            clusterer=Clusterer(threshold=0.5),
        )
        resolver.save(tmp_path)

        manifest = json.loads((tmp_path / "resolver.json").read_text())
        module_spec = next(c for c in manifest["components"] if c["slot"] == "module")
        assert module_spec["type_name"] == "random_forest"
        assert (tmp_path / "module" / "forest.json").exists()

        reloaded = Resolver.load(tmp_path)
        assert isinstance(reloaded.module, RandomForestMatcher)

    @pytest.mark.slow
    def test_resolver_load_random_forest_in_fresh_process(self, tmp_path: Path) -> None:
        """Fresh-process save/load round trip (the M2 lesson — E12)."""
        from langres.core import AllPairsBlocker, Clusterer, Resolver

        comparator = _company_comparator()
        candidates, labels = _labeled_dataset(comparator, n_matches=10, n_nonmatches=10)
        judge: RandomForestMatcher[CompanySchema] = RandomForestMatcher(
            feature_specs=comparator.feature_specs, n_estimators=10, random_state=0
        )
        judge.fit(iter(candidates), labels)
        resolver = Resolver(
            blocker=AllPairsBlocker(schema=CompanySchema),
            comparator=comparator,
            matcher=judge,
            clusterer=Clusterer(threshold=0.5),
        )
        resolver.save(tmp_path)

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from langres.core import Resolver; "
                    f"r = Resolver.load(r'{tmp_path}'); "
                    "assert type(r.module).__name__ == 'RandomForestMatcher'; "
                    "j = r.predict([{'id': 'a', 'name': 'Acme'}, {'id': 'b', 'name': 'Acme Inc'}]); "
                    "assert len(j) == 1 and j[0].score_type == 'prob_rf'; "
                    "print('OK')"
                ),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            "fresh-process Resolver.load failed for an random_forest artifact.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        assert "UnknownComponentType" not in result.stderr

    def test_never_imports_pickle_or_joblib(self) -> None:
        """Hard guard: the module must not IMPORT pickle/joblib (no-pickle artifact contract).

        Checks actual import statements (not the docstring's prose explaining
        *why* it avoids them — a plain substring check would false-positive on
        that explanation).
        """
        import ast

        import langres.core.matchers.random_forest_judge as module

        source = Path(str(module.__file__)).read_text()
        tree = ast.parse(source)
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert "pickle" not in imported_names
        assert "joblib" not in imported_names
