from __future__ import annotations

import pytest
from pydantic import ValidationError

from langres.core.model_ref import ModelRef
from langres.experiments import (
    AttemptIdentity,
    CacheIdentityInput,
    EvaluationProtocol,
    ResourceSlotIdentity,
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
        environment_hash="env-a",
    )


def _cache_input(**updates: object) -> CacheIdentityInput:
    values: dict[str, object] = {
        "stage_id": "rerank",
        "execution_plan_id": "plan-a",
        "operation_identity": {"type": "Rerank", "version": 1, "top_k": 20},
        "resource_slots": (
            ResourceSlotIdentity(
                slot="reranker",
                base="org/model",
                kind="hf",
                revision="revision-a",
                runtime_config={"batch_size": 16},
            ),
        ),
        "source": _source(),
        "semantics": "deterministic",
        "input_fingerprint": "rows-a",
    }
    values.update(updates)
    return CacheIdentityInput(**values)


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


def test_experiment_recipe_excludes_budget_but_legacy_hash_remains_available() -> None:
    low = RunContext(
        experiment="race",
        dataset_name="dataset",
        budget_usd=1.0,
        resolver_config={"model": "same", "budget_usd": 1.0, "max_retries": 1},
    )
    high = RunContext(
        experiment="race",
        dataset_name="dataset",
        budget_usd=20.0,
        resolver_config={"model": "same", "budget_usd": 20.0, "max_retries": 5},
    )

    low_identity = compute_recipe_identity(low)
    high_identity = compute_recipe_identity(high)

    assert low_identity.recipe_id == high_identity.recipe_id
    assert low_identity.legacy_recipe_id != high_identity.legacy_recipe_id


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
            execution_plan_id="plan-a",
            operation_identity={"type": "Rerank"},
            resource_slots=(),
            source=_source(dirty=True, dirty_hash="diff-a"),
            semantics="deterministic",
            input_fingerprint="rows-a",
            official=True,
        )

    with pytest.raises(ValidationError, match="clean source commit"):
        CacheIdentityInput(
            stage_id="rerank",
            execution_plan_id="plan-a",
            operation_identity={"type": "Rerank"},
            resource_slots=(),
            source=SourceState(git_dirty=False),
            semantics="deterministic",
            input_fingerprint="rows-a",
            official=True,
        )


def test_deterministic_cache_is_reusable_for_same_canonical_inputs() -> None:
    first = compute_cache_identity(_cache_input())
    reordered = compute_cache_identity(
        _cache_input(operation_identity={"top_k": 20, "version": 1, "type": "Rerank"})
    )

    assert first == reordered
    assert first.reusable is True
    assert first.counts_as_independent_repeat is True
    assert first.source_claim == "clean"


def test_resource_slot_order_is_canonical_and_model_ref_projection_is_secret_safe() -> None:
    endpoint_ref = ModelRef(
        base="served/model",
        kind="endpoint",
        api_base="https://user:secret@host.example/v1?token=secret",
    )
    endpoint = ResourceSlotIdentity.from_model_ref(
        "llm",
        endpoint_ref,
        provider="vllm",
        runtime_config={"temperature": 0, "api_key": "secret"},
    )
    reranker = ResourceSlotIdentity(
        slot="reranker",
        base="org/model",
        kind="hf",
        revision="revision-a",
    )

    first = compute_cache_identity(_cache_input(resource_slots=(endpoint, reranker)))
    second = compute_cache_identity(_cache_input(resource_slots=(reranker, endpoint)))

    assert first.cache_id == second.cache_id
    assert endpoint.endpoint == "https://host.example/v1"
    assert endpoint.runtime_config["api_key"] == "<redacted>"


def test_publication_claim_does_not_change_output_cache_identity() -> None:
    common = {
        "stage_id": "embed",
        "execution_plan_id": "plan-a",
        "operation_identity": {"type": "Embed"},
        "resource_slots": (),
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
        _cache_input(semantics="seeded")

    seeded_a = compute_cache_identity(_cache_input(semantics="seeded", seed=1))
    seeded_b = compute_cache_identity(_cache_input(semantics="seeded", seed=2))
    stochastic = compute_cache_identity(
        _cache_input(
            stage_id="llm",
            semantics="stochastic",
            repeat_index=2,
            attempt_id="attempt-2",
        )
    )

    assert seeded_a != seeded_b
    assert stochastic.reusable is False
    assert stochastic.counts_as_independent_repeat is False


def test_dirty_source_hash_participates_in_cache_identity() -> None:
    a = compute_cache_identity(_cache_input(source=_source(dirty=True, dirty_hash="diff-a")))
    b = compute_cache_identity(_cache_input(source=_source(dirty=True, dirty_hash="diff-b")))

    assert a != b
    assert a.source_claim == "dirty"
    assert a.official is False


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("execution_plan_id", "plan-b"),
        ("operation_identity", {"type": "Rerank", "version": 2, "top_k": 20}),
        ("input_fingerprint", "rows-b"),
        ("source", SourceState(git_sha="b" * 40, lockfile_hash="lock-a", environment_hash="env-a")),
        ("source", SourceState(git_sha="a" * 40, lockfile_hash="lock-b", environment_hash="env-a")),
        ("source", SourceState(git_sha="a" * 40, lockfile_hash="lock-a", environment_hash="env-b")),
        (
            "resource_slots",
            (
                ResourceSlotIdentity(
                    slot="reranker",
                    base="org/other",
                    kind="hf",
                    revision="revision-a",
                    runtime_config={"batch_size": 16},
                ),
            ),
        ),
        (
            "resource_slots",
            (
                ResourceSlotIdentity(
                    slot="reranker",
                    base="org/model",
                    kind="hf",
                    revision="revision-b",
                    runtime_config={"batch_size": 16},
                ),
            ),
        ),
        (
            "resource_slots",
            (
                ResourceSlotIdentity(
                    slot="reranker",
                    base="org/model",
                    kind="hf",
                    revision="revision-a",
                    adapter="org/adapter",
                    runtime_config={"batch_size": 16},
                ),
            ),
        ),
        (
            "resource_slots",
            (
                ResourceSlotIdentity(
                    slot="reranker",
                    base="org/model",
                    kind="hf",
                    revision="revision-a",
                    runtime_config={"batch_size": 32},
                ),
            ),
        ),
    ],
)
def test_every_output_affecting_cache_fact_invalidates_identity(
    field: str, replacement: object
) -> None:
    base = compute_cache_identity(_cache_input())
    changed = compute_cache_identity(_cache_input(**{field: replacement}))
    assert changed.cache_id != base.cache_id


def test_endpoint_and_secret_runtime_config_are_safely_canonicalized() -> None:
    first = ResourceSlotIdentity(
        slot="llm",
        base="served/model",
        kind="endpoint",
        provider="vllm",
        endpoint="https://user:secret@host.example/v1?api_key=one",
        runtime_config={"temperature": 0, "api_key": "one", "nested": {"token": "one"}},
    )
    second = ResourceSlotIdentity(
        slot="llm",
        base="served/model",
        kind="endpoint",
        provider="vllm",
        endpoint="https://other:different@host.example/v1?api_key=two",
        runtime_config={"temperature": 0, "api_key": "two", "nested": {"token": "two"}},
    )
    assert first == second
    dumped = first.model_dump(mode="json")
    assert "secret" not in str(dumped)
    assert dumped["endpoint"] == "https://host.example/v1"
    assert dumped["runtime_config"]["api_key"] == "<redacted>"
    assert dumped["runtime_config"]["nested"]["token"] == "<redacted>"


@pytest.mark.parametrize(
    "replacement",
    [
        ResourceSlotIdentity(
            slot="llm",
            base="served/model",
            kind="endpoint",
            provider="other-provider",
            endpoint="https://host.example/v1",
        ),
        ResourceSlotIdentity(
            slot="llm",
            base="served/model",
            kind="endpoint",
            provider="vllm",
            endpoint="https://other.example/v1",
        ),
        ResourceSlotIdentity(
            slot="llm",
            base="served/model",
            kind="endpoint",
            provider="vllm",
            endpoint="https://host.example/v1?api-version=2",
        ),
        ResourceSlotIdentity(
            slot="llm",
            base="served/model",
            kind="api",
            provider="vllm",
        ),
    ],
)
def test_served_provider_endpoint_and_kind_invalidate_cache(
    replacement: ResourceSlotIdentity,
) -> None:
    baseline = ResourceSlotIdentity(
        slot="llm",
        base="served/model",
        kind="endpoint",
        provider="vllm",
        endpoint="https://host.example/v1",
    )
    first = compute_cache_identity(_cache_input(resource_slots=(baseline,)))
    changed = compute_cache_identity(_cache_input(resource_slots=(replacement,)))
    assert first.cache_id != changed.cache_id


def test_official_cache_requires_publishable_provenance_and_pinned_resources() -> None:
    with pytest.raises(ValidationError, match="lockfile_hash"):
        _cache_input(
            source=SourceState(git_sha="a" * 40, environment_hash="env-a"),
            official=True,
        )
    with pytest.raises(ValidationError, match="adapter_revision"):
        _cache_input(
            resource_slots=(
                ResourceSlotIdentity(
                    slot="reranker",
                    base="org/model",
                    kind="hf",
                    revision="revision-a",
                    adapter="org/adapter",
                ),
            ),
            official=True,
        )
    with pytest.raises(ValidationError, match="pinned revision"):
        _cache_input(
            resource_slots=(ResourceSlotIdentity(slot="reranker", base="org/model", kind="hf"),),
            official=True,
        )


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
