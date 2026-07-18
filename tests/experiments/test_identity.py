from __future__ import annotations

import pytest
from pydantic import ValidationError

from langres.experiments import (
    AttemptIdentity,
    CacheIdentityInput,
    EvaluationProtocol,
    SourceState,
    compute_cache_identity,
    compute_evaluation_identity,
    compute_recipe_identity,
)
from langres.tracking.runs import RunContext, RunRecord


def _protocol(**updates: object) -> EvaluationProtocol:
    values: dict[str, object] = {
        "benchmark_ids": ("dataset",),
        "split_ids": ("fixed",),
        "fixed_test_set_id": "dataset:test:v1",
        "split_seeds": (3,),
        "threshold_split_id": "validation",
        "test_split_id": "test",
        "pair_metrics": ("pair_f1",),
        "cluster_metrics": ("bcubed_f1",),
        "hardware_cohort": "cpu-a",
        "benchmark_version": "1",
    }
    values.update(updates)
    return EvaluationProtocol(**values)


def _source(*, dirty: bool = False, dirty_hash: str | None = None) -> SourceState:
    return SourceState(
        git_sha="a" * 40,
        git_dirty=dirty,
        dirty_tree_hash=dirty_hash,
        lockfile_hash="lock-a",
    )


def test_recipe_identity_reuses_existing_run_context_hash() -> None:
    clean = RunContext(
        experiment="race",
        dataset_name="dataset",
        seeds={"split": 3},
        git_sha="a" * 40,
        git_dirty=False,
    )
    dirty = clean.model_copy(update={"git_sha": "b" * 40, "git_dirty": True})

    assert compute_recipe_identity(clean) == compute_recipe_identity(dirty)


def test_evaluation_identity_tracks_statistical_question_not_budget() -> None:
    base = compute_evaluation_identity(_protocol(budget_usd=None))
    capped = compute_evaluation_identity(_protocol(budget_usd=9.0))
    changed_metric = compute_evaluation_identity(_protocol(pair_metrics=("precision",)))

    assert base == capped
    assert base != changed_metric


def test_dirty_source_requires_diff_hash_and_cannot_claim_official() -> None:
    with pytest.raises(ValidationError, match="dirty_tree_hash"):
        _source(dirty=True)

    with pytest.raises(ValidationError, match="clean source"):
        CacheIdentityInput(
            stage_id="rerank",
            source=_source(dirty=True, dirty_hash="diff-a"),
            semantics="deterministic",
            input_fingerprint="rows-a",
            official=True,
        )

    with pytest.raises(ValidationError, match="clean source commit"):
        CacheIdentityInput(
            stage_id="rerank",
            source=SourceState(git_dirty=False),
            semantics="deterministic",
            input_fingerprint="rows-a",
            official=True,
        )


def test_deterministic_cache_is_reusable_for_same_canonical_inputs() -> None:
    first = compute_cache_identity(
        CacheIdentityInput(
            stage_id="embed",
            source=_source(),
            semantics="deterministic",
            input_fingerprint="rows-a",
            resource_revisions={"embedder": "rev-1"},
            runtime_config={"batch_size": 32},
        )
    )
    reordered = compute_cache_identity(
        CacheIdentityInput(
            stage_id="embed",
            source=_source(),
            semantics="deterministic",
            input_fingerprint="rows-a",
            resource_revisions={"embedder": "rev-1"},
            runtime_config={"batch_size": 32},
        )
    )

    assert first == reordered
    assert first.reusable is True
    assert first.counts_as_independent_repeat is True
    assert first.source_claim == "clean"


def test_publication_claim_does_not_change_output_cache_identity() -> None:
    common = {
        "stage_id": "embed",
        "source": _source(),
        "semantics": "deterministic",
        "input_fingerprint": "rows-a",
    }

    exploratory = compute_cache_identity(CacheIdentityInput(**common))
    official = compute_cache_identity(CacheIdentityInput(**common, official=True))

    assert exploratory.cache_id == official.cache_id
    assert exploratory.official is False
    assert official.official is True


def test_seeded_and_stochastic_cache_semantics_are_explicit() -> None:
    with pytest.raises(ValidationError, match="seed"):
        CacheIdentityInput(
            stage_id="rerank",
            source=_source(),
            semantics="seeded",
            input_fingerprint="rows-a",
        )

    seeded_a = compute_cache_identity(
        CacheIdentityInput(
            stage_id="rerank",
            source=_source(),
            semantics="seeded",
            seed=1,
            input_fingerprint="rows-a",
        )
    )
    seeded_b = compute_cache_identity(
        CacheIdentityInput(
            stage_id="rerank",
            source=_source(),
            semantics="seeded",
            seed=2,
            input_fingerprint="rows-a",
        )
    )
    stochastic = compute_cache_identity(
        CacheIdentityInput(
            stage_id="llm",
            source=_source(),
            semantics="stochastic",
            repeat_index=2,
            attempt_id="attempt-2",
            input_fingerprint="rows-a",
        )
    )

    assert seeded_a != seeded_b
    assert stochastic.reusable is False
    assert stochastic.counts_as_independent_repeat is False


def test_dirty_source_hash_participates_in_cache_identity() -> None:
    a = compute_cache_identity(
        CacheIdentityInput(
            stage_id="embed",
            source=_source(dirty=True, dirty_hash="diff-a"),
            semantics="deterministic",
            input_fingerprint="rows-a",
        )
    )
    b = compute_cache_identity(
        CacheIdentityInput(
            stage_id="embed",
            source=_source(dirty=True, dirty_hash="diff-b"),
            semantics="deterministic",
            input_fingerprint="rows-a",
        )
    )

    assert a != b
    assert a.source_claim == "dirty"
    assert a.official is False


def test_attempt_identity_reads_the_existing_run_record() -> None:
    context = RunContext(experiment="race", dataset_name="dataset")
    record = RunRecord(
        attempt_id="recipe-when",
        recipe_id="recipe",
        evaluation_id="evaluation",
        cache_id="cache",
        context=context,
        started_at="2026-07-18T12:00:00+00:00",
        status="completed",
    )

    identity = AttemptIdentity.from_record(record)

    assert identity.attempt_id == "recipe-when"
    assert identity.recipe_id == "recipe"
    assert identity.evaluation_id == "evaluation"
    assert identity.cache_id == "cache"
