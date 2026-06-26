"""Tests for GLinkerAdapter contract-conformance stub (M0 Wave 2c).

Covers:
- Import sanity
- isinstance checks against Blocker and Module ABCs
- NotImplementedError for all four abstract methods
- Config round-trip (GLinkerConfig -> dict -> GLinkerAdapter)
- Default config instantiation
- Registry lookup
"""

import pytest

from langres.core.adapters.glinker import GLinkerAdapter, GLinkerConfig
from langres.core.blocker import Blocker
from langres.core.module import Module
from langres.core.registry import get_component


class TestGLinkerImport:
    def test_import_works(self) -> None:
        """Adapter and config can be imported from the adapters package."""
        assert GLinkerAdapter is not None
        assert GLinkerConfig is not None


class TestGLinkerIsInstanceChecks:
    def test_is_instance_of_blocker(self) -> None:
        """GLinkerAdapter satisfies isinstance(adapter, Blocker)."""
        adapter = GLinkerAdapter()
        assert isinstance(adapter, Blocker)

    def test_is_instance_of_module(self) -> None:
        """GLinkerAdapter satisfies isinstance(adapter, Module)."""
        adapter = GLinkerAdapter()
        assert isinstance(adapter, Module)

    def test_is_both_blocker_and_module(self) -> None:
        """GLinkerAdapter satisfies both ABC checks simultaneously."""
        adapter = GLinkerAdapter()
        assert isinstance(adapter, Blocker) and isinstance(adapter, Module)


class TestGLinkerNotImplementedMethods:
    def test_stream_raises_not_implemented(self) -> None:
        """stream([]) raises NotImplementedError (stub body)."""
        adapter = GLinkerAdapter()
        with pytest.raises(NotImplementedError):
            list(adapter.stream([]))

    def test_forward_raises_not_implemented(self) -> None:
        """forward(iter([])) raises NotImplementedError (stub body)."""
        adapter = GLinkerAdapter()
        with pytest.raises(NotImplementedError):
            list(adapter.forward(iter([])))

    def test_inspect_candidates_raises_not_implemented(self) -> None:
        """inspect_candidates([], []) raises NotImplementedError (stub body)."""
        adapter = GLinkerAdapter()
        with pytest.raises(NotImplementedError):
            adapter.inspect_candidates([], [])

    def test_inspect_scores_raises_not_implemented(self) -> None:
        """inspect_scores([]) raises NotImplementedError (stub body)."""
        adapter = GLinkerAdapter()
        with pytest.raises(NotImplementedError):
            adapter.inspect_scores([])


class TestGLinkerConfig:
    def test_default_config_instantiation(self) -> None:
        """GLinkerAdapter() uses sensible default config."""
        adapter = GLinkerAdapter()
        assert adapter._config.model_name == "urchade/gliner_medium-v2.1"
        assert adapter._config.threshold == 0.5

    def test_explicit_config_stored(self) -> None:
        """GLinkerAdapter(config) stores the provided config on _config."""
        cfg = GLinkerConfig(model_name="my-model", threshold=0.8)
        adapter = GLinkerAdapter(config=cfg)
        assert adapter._config is cfg

    def test_config_property_returns_dict(self) -> None:
        """The public config property returns a plain dict (component convention)."""
        cfg = GLinkerConfig(model_name="my-model", threshold=0.8)
        adapter = GLinkerAdapter(config=cfg)
        assert adapter.config == {"model_name": "my-model", "threshold": 0.8}

    def test_config_round_trip_via_from_config(self) -> None:
        """from_config(cfg.model_dump()) round-trips to equal config."""
        cfg = GLinkerConfig(model_name="x", threshold=0.7)
        adapter = GLinkerAdapter.from_config(cfg.model_dump())
        assert adapter._config == cfg

    def test_from_config_with_defaults(self) -> None:
        """from_config({}) uses field defaults."""
        adapter = GLinkerAdapter.from_config({})
        assert adapter._config == GLinkerConfig()

    def test_config_threshold_bounds(self) -> None:
        """threshold must be in [0.0, 1.0]."""
        with pytest.raises(Exception):  # pydantic ValidationError
            GLinkerConfig(threshold=1.5)

        with pytest.raises(Exception):
            GLinkerConfig(threshold=-0.1)


class TestGLinkerRegistry:
    def test_registered_as_glinker_adapter(self) -> None:
        """get_component('glinker_adapter') returns GLinkerAdapter class."""
        assert get_component("glinker_adapter") is GLinkerAdapter
