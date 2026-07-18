from __future__ import annotations

import os
import subprocess
import sys
from decimal import Decimal

import numpy as np
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


def test_default_protocol_identity_mappings_are_immutable() -> None:
    protocol = EvaluationProtocol.smoke()
    with pytest.raises(TypeError, match="immutable"):
        protocol.dataset_fingerprints["tiny_fixture"] = "changed"
    with pytest.raises(TypeError, match="immutable"):
        protocol.deterministic_resources["embedder"] = {"batch_size": 16}


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


def test_exploratory_protocol_without_composite_test_id_requires_every_dataset_identity() -> None:
    common = {
        "benchmark_ids": ("a", "b"),
        "split_ids": ("fixed",),
        "split_seeds": (0,),
        "threshold_split_id": "validation",
        "test_split_id": "test",
        "hardware_cohort": "cpu-local",
        "benchmark_version": "1",
    }
    with pytest.raises(ValidationError, match="every benchmark"):
        EvaluationProtocol(**common, test_set_identities={"a": "a:test"})

    protocol = EvaluationProtocol(
        **common,
        test_set_identities={"a": "a:test", "b": "b:test"},
    )
    assert protocol.fixed_test_set_id is None


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
    assert protocol.architecture_repeats == {
        "Retrieve": 1,
        "RetrieveRerank": 1,
        "RetrieveLLM": 3,
        "RetrieveRerankLLM": 3,
        "CustomTopology": 1,
    }


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


def test_official_proof_expander_revalidates_budget_after_unvalidated_copy() -> None:
    protocol = EvaluationProtocol.official_proof(
        benchmark_ids=("dataset-a", "dataset-b"),
        dataset_fingerprints={"dataset-a": "sha256:a", "dataset-b": "sha256:b"},
        fixed_test_set_id="composite:test:v1",
    )
    bypassed = protocol.model_copy(update={"budget_usd": 19.0})

    with pytest.raises(ValueError, match="exactly USD 20"):
        expand_official_proof_matrix(bypassed)


@pytest.mark.parametrize(
    ("field", "unordered"),
    [
        ("benchmark_ids", {"a", "b"}),
        ("split_ids", frozenset({"fixed", "other"})),
        ("split_seeds", {0, 1}),
        ("threshold_grid", {0.4, 0.6}),
        ("pair_metrics", {"pair_f1", "precision"}),
        ("cluster_metrics", frozenset({"bcubed_f1", "ari"})),
    ],
)
def test_protocol_tuple_identity_fields_reject_unordered_inputs(
    field: str,
    unordered: object,
) -> None:
    values: dict[str, object] = {
        "benchmark_ids": ("a", "b"),
        "split_ids": ("fixed",),
        "fixed_test_set_id": "composite:test",
        "split_seeds": (0,),
        "threshold_split_id": "validation",
        "test_split_id": "test",
        "hardware_cohort": "cpu",
        "benchmark_version": "1",
    }
    values[field] = unordered

    with pytest.raises(ValidationError, match="ordered sequence"):
        EvaluationProtocol(**values)


def test_unordered_protocol_input_rejection_is_hash_seed_independent() -> None:
    script = """
from pydantic import ValidationError
from langres.experiments import EvaluationProtocol
try:
    EvaluationProtocol(
        benchmark_ids={"dataset-a", "dataset-b"},
        split_ids=("fixed",),
        fixed_test_set_id="composite:test",
        split_seeds=(0,),
        threshold_split_id="validation",
        test_split_id="test",
        hardware_cohort="cpu",
        benchmark_version="1",
    )
except ValidationError:
    print("rejected")
else:
    print("accepted")
"""
    outcomes = []
    for hash_seed in ("1", "777"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        outcomes.append(result.stdout.strip())

    assert outcomes == ["rejected", "rejected"]


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


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_protocol_mappings_recursively_reject_non_finite_values(non_finite: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        EvaluationProtocol(
            benchmark_ids=("a",),
            split_ids=("fixed",),
            fixed_test_set_id="a:test:v1",
            split_seeds=(0,),
            deterministic_resources={"runtime": {"temperature": non_finite}},
            threshold_split_id="validation",
            test_split_id="test",
            hardware_cohort="cpu-local",
            benchmark_version="1",
        )


@pytest.mark.parametrize("non_finite", [Decimal("NaN"), Decimal("Infinity"), np.float32("nan")])
def test_protocol_mappings_reject_non_finite_non_float_numeric_types(
    non_finite: object,
) -> None:
    with pytest.raises(ValidationError, match="finite"):
        EvaluationProtocol(
            benchmark_ids=("a",),
            split_ids=("fixed",),
            fixed_test_set_id="a:test:v1",
            split_seeds=(0,),
            deterministic_resources={"runtime": {"value": non_finite}},
            threshold_split_id="validation",
            test_split_id="test",
            hardware_cohort="cpu-local",
            benchmark_version="1",
        )
