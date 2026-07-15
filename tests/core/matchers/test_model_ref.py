"""Unit tests for the weightless ``ModelRef`` normalizer (PR-E serve path).

Pure, fast, dependency-free: no torch/transformers, no network. Locks the three
model-reference surface forms (HF id / local dir string, base+adapter dict,
``ModelRef``), the ``config`` round-trip shape (plain string stays a string; only
base+adapter widens to a dict), and the validation errors.
"""

from __future__ import annotations

import json

import pytest

from langres.core.matchers.model_ref import ModelRef, normalize_model_ref, to_config


def test_string_normalizes_to_base_only_ref() -> None:
    assert normalize_model_ref("your-org/your-ft-model") == ModelRef(
        base="your-org/your-ft-model", adapter=None
    )


def test_local_dir_and_api_names_are_just_base_strings() -> None:
    # The normalizer does NOT route; it only carves a canonical shape. An API
    # name and a local path both become base-only refs.
    assert normalize_model_ref("gpt-5-mini") == ModelRef(base="gpt-5-mini")
    assert normalize_model_ref("/models/my-ft") == ModelRef(base="/models/my-ft")


def test_dict_with_base_and_adapter_normalizes() -> None:
    ref = normalize_model_ref({"base": "meta-llama/Llama-3.1-8B", "adapter": "/ad/lora"})
    assert ref == ModelRef(base="meta-llama/Llama-3.1-8B", adapter="/ad/lora")


def test_dict_with_only_base_has_no_adapter() -> None:
    assert normalize_model_ref({"base": "org/model"}) == ModelRef(base="org/model", adapter=None)


def test_model_ref_passthrough_is_idempotent() -> None:
    ref = ModelRef(base="org/model", adapter="org/adapter")
    assert normalize_model_ref(ref) is ref


def test_empty_string_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        normalize_model_ref("")


def test_dict_missing_base_rejected() -> None:
    with pytest.raises(ValueError, match="'base'"):
        normalize_model_ref({"adapter": "org/adapter"})


def test_dict_empty_base_rejected() -> None:
    with pytest.raises(ValueError, match="'base'"):
        normalize_model_ref({"base": ""})


def test_dict_non_string_adapter_rejected() -> None:
    with pytest.raises(ValueError, match="adapter"):
        normalize_model_ref({"base": "org/model", "adapter": 123})


def test_non_ref_type_rejected() -> None:
    with pytest.raises(TypeError, match="str, dict, or ModelRef"):
        normalize_model_ref(42)  # type: ignore[arg-type]


def test_to_config_of_base_only_is_the_bare_string() -> None:
    # Byte-identical to the pre-model_ref string config -> old artifacts unchanged.
    assert to_config(ModelRef(base="gpt-5-mini")) == "gpt-5-mini"


def test_to_config_of_base_plus_adapter_is_a_dict() -> None:
    cfg = to_config(ModelRef(base="org/base", adapter="org/adapter"))
    assert cfg == {"base": "org/base", "adapter": "org/adapter"}


@pytest.mark.parametrize(
    "model",
    ["gpt-5-mini", "your-org/your-ft-model", {"base": "org/b", "adapter": "org/a"}],
)
def test_config_round_trip_is_weightless_and_stable(model: str | dict[str, str]) -> None:
    ref = normalize_model_ref(model)
    config_value = to_config(ref)
    # Weightless: the serialized form is a plain JSON string/dict of reference
    # strings -- no bytes -- and re-normalizes to the identical ref.
    json.dumps(config_value)
    assert normalize_model_ref(config_value) == ref
