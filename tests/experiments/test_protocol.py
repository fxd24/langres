from __future__ import annotations

import pytest
from pydantic import ValidationError

from langres.experiments import EvaluationProtocol, expand_official_proof_matrix


def test_smoke_protocol_is_zero_network_and_uncapped() -> None:
    protocol = EvaluationProtocol.smoke(seed=7)

    assert protocol.version == 1
    assert protocol.benchmark_ids == ("tiny_fixture",)
    assert protocol.split_seeds == (7,)
    assert protocol.stochastic_repeats == 1
    assert protocol.budget_usd is None
    assert protocol.publication_profile == "exploratory"


def test_ordinary_protocol_may_omit_budget() -> None:
    protocol = EvaluationProtocol(
        benchmark_ids=("a",),
        split_ids=("fixed",),
        fixed_test_set_id="a:test:v1",
        split_seeds=(0,),
        threshold_split_id="validation",
        test_split_id="test",
        hardware_cohort="cpu-local",
        benchmark_version="1",
    )

    assert protocol.budget_usd is None


def test_official_zero_cost_protocol_may_be_uncapped_but_requires_dataset_provenance() -> None:
    kwargs = {
        "benchmark_ids": ("a",),
        "dataset_fingerprints": {"a": "sha256:a"},
        "split_ids": ("fixed",),
        "fixed_test_set_id": "a:test:v1",
        "split_seeds": (0,),
        "threshold_split_id": "validation",
        "test_split_id": "test",
        "hardware_cohort": "cpu-local",
        "benchmark_version": "1",
        "publication_profile": "official",
    }

    assert EvaluationProtocol(**kwargs).budget_usd is None


def test_official_paid_proof_requires_exact_twenty_dollar_cap() -> None:
    kwargs = {
        "benchmark_ids": ("a",),
        "dataset_revisions": {"a": "revision-a"},
        "split_ids": ("fixed",),
        "fixed_test_set_id": "a:test:v1",
        "split_seeds": (0,),
        "threshold_split_id": "validation",
        "test_split_id": "test",
        "hardware_cohort": "cpu-local",
        "benchmark_version": "1",
        "publication_profile": "official",
        "paid_proof": True,
    }

    with pytest.raises(ValidationError, match="exactly USD 20"):
        EvaluationProtocol(**kwargs, budget_usd=19.0)
    assert EvaluationProtocol(**kwargs, budget_usd=20.0).budget_usd == 20.0


def test_official_protocol_requires_data_and_test_set_identity() -> None:
    common = {
        "benchmark_ids": ("a",),
        "split_ids": ("fixed",),
        "split_seeds": (0,),
        "threshold_split_id": "validation",
        "test_split_id": "test",
        "hardware_cohort": "cpu-local",
        "benchmark_version": "1",
        "publication_profile": "official",
    }
    with pytest.raises(ValidationError, match="dataset fingerprint or revision"):
        EvaluationProtocol(**common, fixed_test_set_id="composite:test")
    with pytest.raises(ValidationError, match="test-set identity"):
        EvaluationProtocol(**common, dataset_fingerprints={"a": "sha256:a"})
    with pytest.raises(ValidationError, match="dataset fingerprint or revision"):
        EvaluationProtocol(
            **common,
            dataset_fingerprints={"a": ""},
            fixed_test_set_id="composite:test",
        )


def test_threshold_and_test_splits_must_be_distinct() -> None:
    with pytest.raises(ValidationError, match="untouched"):
        EvaluationProtocol(
            benchmark_ids=("a",),
            split_ids=("fixed",),
            fixed_test_set_id="a:test:v1",
            split_seeds=(0,),
            threshold_split_id="test",
            test_split_id="test",
            hardware_cohort="cpu-local",
            benchmark_version="1",
        )


def test_unknown_protocol_version_is_rejected() -> None:
    with pytest.raises(ValidationError, match="version"):
        EvaluationProtocol(
            version=2,
            benchmark_ids=("a",),
            split_ids=("fixed",),
            fixed_test_set_id="a:test:v1",
            split_seeds=(0,),
            threshold_split_id="validation",
            test_split_id="test",
            hardware_cohort="cpu-local",
            benchmark_version="1",
        )


def test_ci_protocol_requires_enough_bootstrap_samples() -> None:
    with pytest.raises(ValidationError, match="at least 100"):
        EvaluationProtocol(
            benchmark_ids=("a",),
            split_ids=("fixed",),
            fixed_test_set_id="a:test:v1",
            split_seeds=(0,),
            threshold_split_id="validation",
            test_split_id="test",
            hardware_cohort="cpu-local",
            benchmark_version="1",
            confidence_interval_method="paired_entity_bootstrap",
            bootstrap_samples=20,
        )


def test_official_proof_expands_to_exactly_18_cells_before_retries() -> None:
    protocol = EvaluationProtocol.official_proof(
        benchmark_ids=("dataset-a", "dataset-b"),
        dataset_fingerprints={"dataset-a": "sha256:a", "dataset-b": "sha256:b"},
        fixed_test_set_id="composite:test:v1",
        split_seed=11,
    )

    cells = expand_official_proof_matrix(protocol)

    assert len(cells) == 18
    assert len({cell.cell_id for cell in cells}) == 18
    assert {cell.retry_index for cell in cells} == {0}
    counts = {
        topology: sum(cell.topology == topology for cell in cells)
        for topology in {cell.topology for cell in cells}
    }
    assert counts == {
        "Retrieve": 2,
        "RetrieveRerank": 2,
        "RetrieveLLM": 6,
        "RetrieveRerankLLM": 6,
        "CustomTopology": 2,
    }
    assert protocol.budget_usd == 20.0
    assert protocol.publication_profile == "official"
    assert protocol.paid_proof is True


def test_official_proof_expander_rejects_non_paid_official_protocol() -> None:
    protocol = EvaluationProtocol(
        benchmark_ids=("dataset-a", "dataset-b"),
        dataset_fingerprints={"dataset-a": "sha256:a", "dataset-b": "sha256:b"},
        split_ids=("official",),
        fixed_test_set_id="composite:test:v1",
        split_seeds=(11,),
        stochastic_repeats=3,
        threshold_split_id="validation",
        test_split_id="test",
        hardware_cohort="official-declared",
        benchmark_version="1",
        publication_profile="official",
    )

    with pytest.raises(ValueError, match="paid_proof=True"):
        expand_official_proof_matrix(protocol)


def test_official_proof_expander_defends_against_empty_provenance_values() -> None:
    protocol = EvaluationProtocol.official_proof(
        benchmark_ids=("dataset-a", "dataset-b"),
        dataset_fingerprints={"dataset-a": "sha256:a", "dataset-b": "sha256:b"},
        fixed_test_set_id="composite:test:v1",
    )

    missing_data = protocol.model_copy(
        update={"dataset_fingerprints": {"dataset-a": "", "dataset-b": ""}}
    )
    with pytest.raises(ValueError, match="non-empty dataset provenance"):
        expand_official_proof_matrix(missing_data)

    missing_test = protocol.model_copy(
        update={"fixed_test_set_id": None, "test_set_identities": {}}
    )
    with pytest.raises(ValueError, match="non-empty composite"):
        expand_official_proof_matrix(missing_test)


def test_protocol_mappings_reject_non_json_sets() -> None:
    with pytest.raises(ValidationError, match="sets are not valid JSON"):
        EvaluationProtocol(
            benchmark_ids=("a",),
            split_ids=("fixed",),
            fixed_test_set_id="a:test:v1",
            split_seeds=(0,),
            deterministic_resources={"devices": {"cpu", "gpu"}},
            threshold_split_id="validation",
            test_split_id="test",
            hardware_cohort="cpu-local",
            benchmark_version="1",
        )
