"""Unit tests for the weightless ``ModelRef`` — the ONE backbone concept (W3).

Pure, fast, dependency-free: no torch/transformers, no litellm, no network.
Locks the surface forms, the ``kind`` discriminator's inference rules, the
CWD-independence of routing (B17), the ``revision`` field (B16), the validation
matrix (now enforced in ``__post_init__``, not only in the normalizer), and the
``config`` round-trip invariant.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from langres.core.matchers.model_ref import ModelRef, normalize_model_ref, to_config
from langres.core.model_ref import (
    IN_PROCESS_KINDS,
    LITELLM_ROUTABLE_KINDS,
    SERVED_KINDS,
    InvalidModelRefError,
    UnsupportedBackboneError,
    backend_for,
    infer_kind,
    require_litellm_routable,
)

# --------------------------------------------------------------------------- #
# Kind inference — by syntax alone, never by touching the filesystem.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("gpt-5-mini", "api"),  # bare litellm id
        ("openai/gpt-4o", "api"),  # known provider prefix
        ("openrouter/openai/gpt-4o-mini", "api"),  # multi-slash, known prefix
        ("hosted_vllm/my-model", "api"),
        ("your-org/your-ft-model", "hf"),  # org/name -> Hub id
        ("BAAI/bge-small-en-v1.5", "hf"),
        ("./my-ft", "local"),
        ("../models/my-ft", "local"),
        ("/models/my-ft", "local"),
        ("~/models/my-ft", "local"),
    ],
)
def test_infer_kind(base: str, expected: str) -> None:
    assert infer_kind(base) == expected
    assert normalize_model_ref(base).kind == expected


def test_api_base_makes_it_an_endpoint() -> None:
    ref = normalize_model_ref("my-served-model", api_base="http://localhost:8000/v1")
    assert ref.kind == "endpoint"
    assert ref.api_base == "http://localhost:8000/v1"
    assert backend_for(ref.kind) == "litellm"


def test_unknown_multi_slash_form_raises() -> None:
    """The one 'unknown form' that is unambiguous — so it is the one that raises.

    ``foo/bar/baz`` is neither a Hub id (exactly one slash) nor a known provider
    id, so guessing would only defer the failure to a 404.
    """
    with pytest.raises(InvalidModelRefError, match="cannot infer a backbone kind"):
        infer_kind("foo/bar/baz")
    with pytest.raises(InvalidModelRefError, match="Name the kind explicitly"):
        normalize_model_ref("foo/bar/baz")


def test_an_unknown_multi_slash_id_is_reachable_by_naming_its_kind() -> None:
    ref = normalize_model_ref({"base": "foo/bar/baz", "kind": "api"})
    assert ref.kind == "api"
    assert backend_for(ref.kind) == "litellm"


def test_inference_never_touches_the_filesystem(tmp_path: Any) -> None:
    """**B17.** Directories named like model ids must not move the decision.

    The predecessor probed ``os.path.isdir(ref.base)``, so a local ``./openai``
    directory silently flipped routing litellm -> transformers — the same saved
    config resolving differently per working directory.
    """
    (tmp_path / "openai").mkdir()
    (tmp_path / "gpt-5-mini").mkdir()
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        assert infer_kind("openai/gpt-4o") == "api"
        assert infer_kind("gpt-5-mini") == "api"
    finally:
        os.chdir(cwd)
    assert infer_kind("openai/gpt-4o") == "api"
    assert infer_kind("gpt-5-mini") == "api"


# --------------------------------------------------------------------------- #
# Routing.
# --------------------------------------------------------------------------- #


def test_backend_for_is_total_over_the_kinds() -> None:
    assert {backend_for(k) for k in SERVED_KINDS} == {"litellm"}
    assert {backend_for(k) for k in IN_PROCESS_KINDS} == {"transformers"}


# --------------------------------------------------------------------------- #
# Surface forms.
# --------------------------------------------------------------------------- #


def test_dict_with_base_and_adapter_normalizes() -> None:
    ref = normalize_model_ref({"base": "meta-llama/Llama-3.1-8B", "adapter": "/ad/lora"})
    assert ref == ModelRef(base="meta-llama/Llama-3.1-8B", kind="hf", adapter="/ad/lora")


def test_an_adapter_forces_an_in_process_kind() -> None:
    """A bare-name base + adapter is in-process, though the base alone reads as `api`.

    Inference sees only ``base``, so it cannot know an adapter is coming; the
    normalizer corrects for it (litellm can never assemble an unmerged adapter).
    """
    ref = normalize_model_ref({"base": "some-base", "adapter": "org/lora"})
    assert ref.kind == "hf"
    assert backend_for(ref.kind) == "transformers"


def test_dict_with_only_base_has_no_adapter() -> None:
    assert normalize_model_ref({"base": "org/model"}) == ModelRef(
        base="org/model", kind="hf", adapter=None
    )


def test_explicit_kind_in_a_dict_overrides_inference() -> None:
    # `org/name` reads as `hf`, but a caller who means a provider id says so.
    assert normalize_model_ref({"base": "opeani/gpt-4o", "kind": "api"}).kind == "api"


def test_model_ref_passthrough_is_idempotent() -> None:
    ref = ModelRef(base="org/model", kind="hf", adapter="org/adapter")
    assert normalize_model_ref(ref) is ref


# --------------------------------------------------------------------------- #
# Validation — enforced in __post_init__, so direct construction cannot bypass it.
# --------------------------------------------------------------------------- #


def test_empty_string_rejected() -> None:
    with pytest.raises(InvalidModelRefError, match="non-empty"):
        normalize_model_ref("")


def test_dict_missing_base_rejected() -> None:
    with pytest.raises(InvalidModelRefError, match="'base'"):
        normalize_model_ref({"adapter": "org/adapter"})


def test_dict_empty_base_rejected() -> None:
    with pytest.raises(InvalidModelRefError, match="'base'"):
        normalize_model_ref({"base": ""})


def test_dict_non_string_adapter_rejected() -> None:
    with pytest.raises(InvalidModelRefError, match="adapter"):
        normalize_model_ref({"base": "org/model", "adapter": 123})  # type: ignore[dict-item]


def test_non_ref_type_rejected() -> None:
    with pytest.raises(TypeError, match="str, dict, or ModelRef"):
        normalize_model_ref(42)  # type: ignore[arg-type]


def test_direct_construction_validates_too() -> None:
    """The dataclass is public, so ``__post_init__`` is the only guard that cannot
    be bypassed — a caller need not go through ``normalize_model_ref``."""
    with pytest.raises(InvalidModelRefError, match="non-empty string"):
        ModelRef(base="", kind="api")
    with pytest.raises(InvalidModelRefError, match="kind must be one of"):
        ModelRef(base="x", kind="bogus")  # type: ignore[arg-type]


def test_an_adapter_on_a_served_kind_is_rejected() -> None:
    with pytest.raises(InvalidModelRefError, match="cannot carry an adapter"):
        ModelRef(base="org/b", kind="api", adapter="org/a")


def test_endpoint_requires_api_base_and_others_forbid_it() -> None:
    with pytest.raises(InvalidModelRefError, match="requires api_base"):
        ModelRef(base="m", kind="endpoint")
    with pytest.raises(InvalidModelRefError, match="cannot carry api_base"):
        ModelRef(base="m", kind="api", api_base="http://x/v1")


def test_conflicting_api_base_is_rejected() -> None:
    with pytest.raises(InvalidModelRefError, match="conflicting api_base"):
        normalize_model_ref(
            {"base": "m", "kind": "endpoint", "api_base": "http://a/v1"},
            api_base="http://b/v1",
        )
    with pytest.raises(InvalidModelRefError, match="conflicting api_base"):
        normalize_model_ref(
            ModelRef(base="m", kind="endpoint", api_base="http://a/v1"),
            api_base="http://b/v1",
        )


# --------------------------------------------------------------------------- #
# revision (B16) — in the v1 schema NOW, before artifacts depend on it.
# --------------------------------------------------------------------------- #


def test_revision_pins_an_hf_ref_and_survives_the_round_trip() -> None:
    ref = normalize_model_ref({"base": "org/model", "revision": "abc123"})
    assert ref.kind == "hf"
    assert ref.revision == "abc123"
    assert to_config(ref) == {"base": "org/model", "kind": "hf", "revision": "abc123"}
    assert normalize_model_ref(to_config(ref)) == ref


def test_two_revisions_of_one_base_are_different_refs() -> None:
    """The point of B16: without ``revision``, ``org/name`` drifts as the Hub moves,
    so an 'identical versioned config' is not identical across time."""
    a = normalize_model_ref({"base": "org/model", "revision": "v1"})
    b = normalize_model_ref({"base": "org/model", "revision": "v2"})
    assert a != b
    assert to_config(a) != to_config(b)


def test_revision_is_hf_only() -> None:
    with pytest.raises(InvalidModelRefError, match="cannot carry a revision"):
        ModelRef(base="gpt-4o", kind="api", revision="abc")
    with pytest.raises(InvalidModelRefError, match="cannot carry a revision"):
        ModelRef(base="/models/m", kind="local", revision="abc")


def test_non_string_revision_rejected() -> None:
    with pytest.raises(InvalidModelRefError, match="revision"):
        normalize_model_ref({"base": "org/model", "revision": 7})  # type: ignore[dict-item]


# --------------------------------------------------------------------------- #
# The litellm-routable guard (B10).
# --------------------------------------------------------------------------- #


def test_require_litellm_routable_admits_served_and_hf() -> None:
    for base in ("gpt-5-mini", "openai/gpt-4o", "unknown-provider/some-model"):
        ref = normalize_model_ref(base)
        assert require_litellm_routable(ref, slot="DSPyMatcher") is ref
    served = normalize_model_ref("m", api_base="http://x/v1")
    assert require_litellm_routable(served, slot="DSPyMatcher") is served


def test_require_litellm_routable_rejects_local_and_adapters() -> None:
    with pytest.raises(UnsupportedBackboneError, match="no in-process route"):
        require_litellm_routable(normalize_model_ref("./my-ft"), slot="DSPyMatcher")
    with pytest.raises(UnsupportedBackboneError, match="unmerged base\\+adapter"):
        require_litellm_routable(
            normalize_model_ref({"base": "org/b", "adapter": "org/a"}), slot="DSPyMatcher"
        )


def test_litellm_routable_includes_hf_deliberately() -> None:
    """Measured, not stylistic: litellm knows 146 providers, the prefix table 26,
    so 120 real provider ids infer as ``hf``. Rejecting ``hf`` here would reject
    working code. See LITELLM_ROUTABLE_KINDS for the alternative."""
    assert "hf" in LITELLM_ROUTABLE_KINDS
    assert "local" not in LITELLM_ROUTABLE_KINDS
    assert SERVED_KINDS <= LITELLM_ROUTABLE_KINDS


# --------------------------------------------------------------------------- #
# config round-trip.
# --------------------------------------------------------------------------- #


def test_to_config_of_an_inferable_ref_is_the_bare_string() -> None:
    # Byte-identical to the pre-`kind` string config -> old artifacts unchanged.
    assert to_config(ModelRef(base="gpt-5-mini", kind="api")) == "gpt-5-mini"
    assert to_config(ModelRef(base="org/model", kind="hf")) == "org/model"
    assert to_config(ModelRef(base="./m", kind="local")) == "./m"


def test_to_config_widens_when_inference_could_not_reproduce_the_ref() -> None:
    # `org/base` alone would infer as `hf`; the adapter must be carried.
    assert to_config(ModelRef(base="org/base", kind="hf", adapter="org/adapter")) == {
        "base": "org/base",
        "kind": "hf",
        "adapter": "org/adapter",
    }
    # A kind inference would NOT have guessed must be explicit or it misroutes.
    assert to_config(ModelRef(base="org/model", kind="api")) == {
        "base": "org/model",
        "kind": "api",
    }
    assert to_config(ModelRef(base="foo/bar/baz", kind="api")) == {
        "base": "foo/bar/baz",
        "kind": "api",
    }


@pytest.mark.parametrize(
    "ref",
    [
        ModelRef(base="gpt-5-mini", kind="api"),
        ModelRef(base="org/model", kind="hf"),
        ModelRef(base="org/model", kind="api"),  # explicit kind beats inference
        ModelRef(base="org/model", kind="hf", revision="abc123"),
        ModelRef(base="./m", kind="local"),
        ModelRef(base="/abs/m", kind="local"),
        ModelRef(base="org/b", kind="hf", adapter="org/a"),
        ModelRef(base="foo/bar/baz", kind="api"),
        ModelRef(base="m", kind="endpoint", api_base="http://localhost:8000/v1"),
    ],
)
def test_config_round_trip_is_weightless_and_lossless(ref: ModelRef) -> None:
    """The invariant: ``normalize_model_ref(to_config(ref)) == ref`` for EVERY ref.

    This is what lets ``to_config`` emit the compact string form as an
    optimization without it ever being a lossy one.
    """
    config_value = to_config(ref)
    json.dumps(config_value)  # weightless: reference strings only, no bytes
    assert normalize_model_ref(config_value) == ref


def test_old_matchers_path_re_exports_the_same_objects() -> None:
    """The back-compat shim is a pure re-export -- not a second, divergent copy.

    ``ModelRef`` now lives in ``langres.core.model_ref`` (it is a weightless
    contract, not a matcher). Everything above imports it through the old
    ``langres.core.matchers.model_ref`` path, so this asserts the two paths hand
    back the *identical* objects -- an ``==``-only check would pass even if the
    shim redefined its own class, which would break isinstance/round-trips.
    """
    from langres.core import model_ref as new_path
    from langres.core.matchers import model_ref as shim

    assert shim.ModelRef is new_path.ModelRef
    assert shim.normalize_model_ref is new_path.normalize_model_ref
    assert shim.to_config is new_path.to_config
