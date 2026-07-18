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


def test_official_profile_requires_positive_budget_cap() -> None:
    kwargs = {
        "benchmark_ids": ("a",),
        "split_ids": ("fixed",),
        "fixed_test_set_id": "a:test:v1",
        "split_seeds": (0,),
        "threshold_split_id": "validation",
        "test_split_id": "test",
        "hardware_cohort": "cpu-local",
        "benchmark_version": "1",
        "publication_profile": "official",
    }

    with pytest.raises(ValidationError, match="budget_usd"):
        EvaluationProtocol(**kwargs)

    assert EvaluationProtocol(**kwargs, budget_usd=20.0).budget_usd == 20.0


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


def test_official_proof_expands_to_exactly_18_cells_before_retries() -> None:
    protocol = EvaluationProtocol.official_proof(
        benchmark_ids=("dataset-a", "dataset-b"),
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
