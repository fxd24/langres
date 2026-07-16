"""Tests for langres.clients.tracking module."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from langres.clients.settings import Settings
from langres.clients.tracking import create_trackio_tracker, create_wandb_tracker
from langres.core.trackers.trackio_tracker import TrackioTracker


class TestCreateWandbTracker:
    """Tests for create_wandb_tracker factory function."""

    def test_create_wandb_tracker_with_settings(self):
        """Test create_wandb_tracker with explicit settings."""
        settings = Settings(
            openai_api_key="sk-test",
            wandb_api_key="wb-test",
            wandb_project="test-project",
            wandb_entity="test-team",
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
            azure_api_key="azure-key",
            azure_api_endpoint="https://test.openai.azure.com",
        )

        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            mock_run = MagicMock()
            mock_wandb.init.return_value = mock_run

            run = create_wandb_tracker(settings, job_type="test-job")

            # Verify wandb.init called with correct params
            mock_wandb.init.assert_called_once_with(
                project="test-project", entity="test-team", job_type="test-job"
            )

            # Verify run object returned
            assert run is mock_run

    def test_create_wandb_tracker_without_settings_loads_from_env(self):
        """Test create_wandb_tracker loads settings from environment."""
        # Patch both os.environ AND the .env file to prevent leakage
        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "sk-env",
                    "WANDB_API_KEY": "wb-env",
                    "WANDB_PROJECT": "env-project",
                    "LANGFUSE_PUBLIC_KEY": "pk-lf-env",
                    "LANGFUSE_SECRET_KEY": "sk-lf-env",
                    "AZURE_API_KEY": "azure-key",
                    "AZURE_API_ENDPOINT": "https://test.openai.azure.com",
                },
                clear=True,
            ),
            patch("pydantic_settings.sources.DotEnvSettingsSource.__call__", return_value={}),
        ):
            mock_wandb = MagicMock()
            with patch.dict(sys.modules, {"wandb": mock_wandb}):
                mock_run = MagicMock()
                mock_wandb.init.return_value = mock_run

                run = create_wandb_tracker()

                # Verify wandb.init called with env settings
                mock_wandb.init.assert_called_once_with(
                    project="env-project", entity=None, job_type="optimization"
                )

    def test_create_wandb_tracker_default_job_type(self):
        """Test create_wandb_tracker uses default job_type."""
        settings = Settings(
            openai_api_key="sk-test",
            wandb_api_key="wb-test",
            wandb_project="test-project",
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
            azure_api_key="azure-key",
            azure_api_endpoint="https://test.openai.azure.com",
        )

        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            create_wandb_tracker(settings)

            # Verify default job_type is "optimization"
            call_kwargs = mock_wandb.init.call_args.kwargs
            assert call_kwargs["job_type"] == "optimization"

    def test_create_wandb_tracker_with_entity_none(self):
        """Test create_wandb_tracker handles entity=None correctly."""
        settings = Settings(
            openai_api_key="sk-test",
            wandb_api_key="wb-test",
            wandb_project="test-project",
            wandb_entity=None,  # Explicitly None
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
            azure_api_key="azure-key",
            azure_api_endpoint="https://test.openai.azure.com",
        )

        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            create_wandb_tracker(settings)

            # Verify entity=None passed to wandb.init
            call_kwargs = mock_wandb.init.call_args.kwargs
            assert call_kwargs["entity"] is None

    def test_create_wandb_tracker_raises_error_if_api_key_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Online (default) mode still requires WANDB_API_KEY -- the guard is intact."""
        monkeypatch.delenv("WANDB_MODE", raising=False)
        settings = Settings(wandb_api_key=None)
        with pytest.raises(ValueError, match="WANDB_API_KEY environment variable is required"):
            create_wandb_tracker(settings)

    def test_create_wandb_tracker_raises_branded_import_error_without_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The now-lazy wandb import surfaces the branded langres[wandb] fix at call time."""
        monkeypatch.setitem(sys.modules, "wandb", None)  # a fresh `import wandb` now fails
        with pytest.raises(ImportError, match=r"langres\[wandb\]"):
            create_wandb_tracker(Settings(wandb_api_key="k"))

    @pytest.mark.parametrize("mode", ["offline", "disabled", "OFFLINE", " offline "])
    def test_create_wandb_tracker_skips_api_key_when_offline(
        self, monkeypatch: pytest.MonkeyPatch, mode: str
    ):
        """WANDB_MODE offline/disabled needs no key -- the requirement is skipped (F2)."""
        monkeypatch.setenv("WANDB_MODE", mode)
        settings = Settings(wandb_api_key=None, wandb_project="offline-proj")

        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            mock_run = MagicMock()
            mock_wandb.init.return_value = mock_run

            run = create_wandb_tracker(settings)  # must NOT raise without a key

            assert run is mock_run
            mock_wandb.init.assert_called_once_with(
                project="offline-proj", entity=None, job_type="optimization"
            )


class TestCreateTrackioTracker:
    """Tests for create_trackio_tracker factory function.

    Unlike create_wandb_tracker (which calls wandb.init eagerly and returns the
    live run), create_trackio_tracker builds and returns an UNSTARTED
    TrackioTracker -- trackio.init is deferred to .start_run(), so these tests
    never touch the network/trackio.init and don't need to mock trackio.
    """

    def test_returns_unstarted_trackio_tracker(self) -> None:
        tracker = create_trackio_tracker(Settings())
        assert isinstance(tracker, TrackioTracker)
        assert tracker.native is None

    def test_default_settings_loaded_when_none_given(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TRACKIO_SPACE_ID", raising=False)
        monkeypatch.delenv("TRACKIO_DATASET_ID", raising=False)
        tracker = create_trackio_tracker()
        assert isinstance(tracker, TrackioTracker)

    def test_needs_no_wandb_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """create_trackio_tracker must work in a trackio-only (wandb-less) install.

        Co-located with create_wandb_tracker, but the wandb import is lazy inside
        that factory's body -- so importing/calling create_trackio_tracker never
        needs the wandb extra. Simulate wandb's absence and prove the call still
        succeeds (while a fresh ``import wandb`` genuinely fails).
        """
        import importlib

        monkeypatch.setitem(sys.modules, "wandb", None)
        with pytest.raises(ImportError):
            importlib.import_module("wandb")  # the simulation is real

        tracker = create_trackio_tracker(Settings())  # must NOT raise
        assert isinstance(tracker, TrackioTracker)

    def test_explicit_project_and_config_passed_through(self) -> None:
        tracker = create_trackio_tracker(Settings(), project="ar-smoke", config={"k": "v"})
        assert tracker._project == "ar-smoke"
        assert tracker._extra_config == {"k": "v"}

    def test_space_id_and_dataset_id_from_explicit_kwargs(self) -> None:
        tracker = create_trackio_tracker(
            Settings(), space_id="acme/runs", dataset_id="acme/runs_dataset"
        )
        assert tracker._space_id == "acme/runs"
        assert tracker._dataset_id == "acme/runs_dataset"

    def test_space_id_and_dataset_id_fall_back_to_settings(self) -> None:
        settings = Settings(trackio_space_id="acme/runs", trackio_dataset_id="acme/ds")
        tracker = create_trackio_tracker(settings)
        assert tracker._space_id == "acme/runs"
        assert tracker._dataset_id == "acme/ds"

    def test_explicit_space_id_overrides_settings(self) -> None:
        settings = Settings(trackio_space_id="settings/space")
        tracker = create_trackio_tracker(settings, space_id="explicit/space")
        assert tracker._space_id == "explicit/space"

    def test_local_first_when_nothing_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No space_id anywhere (kwarg or settings) -> a pure local tracker.

        Hermetic: drop any ambient TRACKIO_SPACE_ID/TRACKIO_DATASET_ID so the
        assertion doesn't depend on the runner's environment.
        """
        monkeypatch.delenv("TRACKIO_SPACE_ID", raising=False)
        monkeypatch.delenv("TRACKIO_DATASET_ID", raising=False)
        tracker = create_trackio_tracker(Settings())
        assert tracker._space_id is None
        assert tracker._dataset_id is None
