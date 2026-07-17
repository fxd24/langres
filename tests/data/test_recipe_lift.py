"""Behavior test for the recipe-lift claim: curation beats random at an equal budget.

The seeded, synthetic-and-fast twin of ``examples/recipe_lift_proof.py``. It builds
a heavily imbalanced fuzzy pair pool (near-duplicate matched clusters drowned in
random non-matches -- the shape a real all-pairs blocked ER pool has), then checks
that a *curated* training set (denoise + hard positives + attribute augmentation +
2:1 balance, the four Wave-A miners) trains a matcher with a materially higher
held-out pair-F1 than the **mean over several equal-size random draws** from the
same pool.

Why the mean-of-draws baseline: a random draw from a ~1:hundreds pool is almost
all non-matches and usually contains *no* match at all, so its matcher never
learns what a match is and its F1 collapses. A single lucky draw can catch a match
and approach the recipe, so the honest comparison is the average draw, not the
worst. This mirrors the example's mechanism exactly, at a size that runs in a
couple of seconds on CPU with no network or spend (needs the ``[trained]`` extra
for the RandomForest behind the featurizing miners + the served matcher).
"""

from __future__ import annotations

import random

import pytest
from pydantic import BaseModel

from langres.core.benchmark import evaluate_judge_on_candidates
from langres.core.comparators import StringComparator
from langres.training.finetune import LabeledCandidate
from langres.core.matchers.random_forest_judge import RandomForestMatcher
from langres.core.models import ERCandidate
from langres.core.resolver import Resolver
from langres.data.mining import (
    augment_by_attribute,
    denoise_pairs,
    mine_misclassified_pairs,
    sample_negative_pairs,
)

SEED = 0
GRID = [i / 20 for i in range(1, 21)]  # 0.05 .. 1.00 best-F1 threshold sweep
POS_CAP = 30
N_BASELINE_DRAWS = 5

_CONSONANTS = "bcdfghjklmnpqrstvwxz"
_VOWELS = "aeiou"


class _Org(BaseModel):
    """A tiny two-content-field entity so the comparator emits >1 feature."""

    id: str
    name: str
    city: str


def _word(rng: random.Random, syllables: int = 4) -> str:
    """A distinct pronounceable pseudo-word (consonant-vowel syllables)."""
    return "".join(rng.choice(_CONSONANTS) + rng.choice(_VOWELS) for _ in range(syllables))


def _typo(rng: random.Random, word: str) -> str:
    """A near-duplicate of ``word``: one character substituted (a high-similarity dup)."""
    i = rng.randrange(len(word))
    return word[:i] + rng.choice(_CONSONANTS + _VOWELS) + word[i + 1 :]


def _make_corpus(
    n_clusters: int, n_singletons: int, seed: int
) -> tuple[list[dict[str, str]], set[frozenset[str]]]:
    """A fuzzy corpus: ``n_clusters`` near-duplicate pairs + ``n_singletons`` loners.

    Each matched cluster is two records whose name and city are one-character typos
    of a shared, distinct base (high within-pair similarity); every singleton and
    every base is a fresh pseudo-word, so cross-entity similarity stays low. Returns
    the records (as field dicts) and the gold positive pairs (matched id-sets).
    """
    rng = random.Random(seed)
    records: list[dict[str, str]] = []
    gold: set[frozenset[str]] = set()
    for c in range(n_clusters):
        name, city = _word(rng), _word(rng)
        left = {"id": f"m{c}a", "name": name.title(), "city": city.title()}
        right = {"id": f"m{c}b", "name": _typo(rng, name).title(), "city": _typo(rng, city).title()}
        records += [left, right]
        gold.add(frozenset({left["id"], right["id"]}))
    for s in range(n_singletons):
        records.append({"id": f"s{s}", "name": _word(rng).title(), "city": _word(rng).title()})
    return records, gold


def _build_pool(
    resolver: Resolver, records: list[dict[str, str]], gold: set[frozenset[str]]
) -> list[LabeledCandidate]:
    """Block records all-pairs (comparison attached) and label each pair against gold."""
    return [(c, frozenset({c.left.id, c.right.id}) in gold) for c in resolver.candidates(records)]


def _assemble_recipe(
    train_pool: list[LabeledCandidate], comparator: StringComparator[object]
) -> list[LabeledCandidate]:
    """The four-miner curation recipe (identical composition to the example)."""
    clean, _flagged = denoise_pairs(train_pool, seed=SEED)
    hard = mine_misclassified_pairs(clean, cap=POS_CAP, seed=SEED)
    augmented = [
        (c.model_copy(update={"comparison": comparator.compare(c.left, c.right)}), label)
        for c, label in augment_by_attribute(hard, cap=POS_CAP, seed=SEED)
    ]
    positives = list(hard) + augmented
    negatives = sample_negative_pairs(
        positives + [pair for pair in clean if not pair[1]], ratio=2.0, seed=SEED
    )
    return positives + negatives


def _fit_and_score(
    labeled: list[LabeledCandidate],
    feature_specs: list[object],
    test_candidates: list[ERCandidate[object]],
    gold: set[frozenset[str]],
) -> float:
    """Fit a RandomForest on ``labeled`` and return its best-F1 on the held-out set."""
    matcher: RandomForestMatcher[object] = RandomForestMatcher(
        feature_specs=feature_specs,  # type: ignore[arg-type]
        random_state=SEED,
    )
    matcher.fit(iter([c for c, _ in labeled]), [label for _, label in labeled])
    result, _ = evaluate_judge_on_candidates(matcher, test_candidates, gold, GRID)
    return result.pair.f1


@pytest.fixture(scope="module")
def _lift() -> dict[str, float]:
    """Run the proof once (module-scoped: the RF fits are the expensive part)."""
    train_records, train_gold = _make_corpus(n_clusters=12, n_singletons=70, seed=SEED)
    test_records, test_gold = _make_corpus(n_clusters=8, n_singletons=40, seed=SEED + 1)

    resolver = Resolver.from_schema(_Org)  # AllPairsBlocker + StringComparator
    comparator = resolver.comparator
    feature_specs = list(comparator.feature_specs)

    train_pool = _build_pool(resolver, train_records, train_gold)
    test_candidates = [c for c, _ in _build_pool(resolver, test_records, test_gold)]

    n_pos = sum(1 for _, label in train_pool if label)
    n_neg = len(train_pool) - n_pos

    recipe = _assemble_recipe(train_pool, comparator)
    budget = len(recipe)
    recipe_f1 = _fit_and_score(recipe, feature_specs, test_candidates, test_gold)

    draws = [
        _fit_and_score(
            random.Random(1000 + k).sample(train_pool, min(budget, len(train_pool))),
            feature_specs,
            test_candidates,
            test_gold,
        )
        for k in range(N_BASELINE_DRAWS)
    ]
    baseline_mean = sum(draws) / len(draws)
    return {
        "recipe_f1": recipe_f1,
        "baseline_mean": baseline_mean,
        "beats": sum(1 for f1 in draws if recipe_f1 > f1),
        "imbalance": n_neg / max(n_pos, 1),
        "budget": float(budget),
    }


class TestRecipeLift:
    def test_pool_is_heavily_imbalanced(self, _lift: dict[str, float]) -> None:
        # The mechanism needs the genuine blocked-pool imbalance; guard it holds.
        assert _lift["imbalance"] > 20.0

    def test_recipe_learns_a_real_matcher(self, _lift: dict[str, float]) -> None:
        # Curation trains a matcher that actually separates matches on held-out data.
        assert _lift["recipe_f1"] > 0.5

    def test_recipe_beats_random_mean_by_a_clear_margin(self, _lift: dict[str, float]) -> None:
        # The headline claim: curated pairs beat the mean random draw at an equal budget.
        assert _lift["recipe_f1"] > _lift["baseline_mean"] + 0.15

    def test_recipe_beats_the_majority_of_draws(self, _lift: dict[str, float]) -> None:
        # Tolerates a couple of lucky draws that happen to catch a match; the point
        # is that most equal-budget random draws contain none and score zero.
        assert _lift["beats"] >= N_BASELINE_DRAWS // 2 + 1
