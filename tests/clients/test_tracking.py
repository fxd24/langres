"""Tests for langres.clients.tracking module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from langres.clients.settings import Settings
from langres.clients.tracking import create_wandb_tracker


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

        with patch("langres.clients.tracking.wandb") as mock_wandb:
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
            with patch("langres.clients.tracking.wandb") as mock_wandb:
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

        with patch("langres.clients.tracking.wandb") as mock_wandb:
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

        with patch("langres.clients.tracking.wandb") as mock_wandb:
            create_wandb_tracker(settings)

            # Verify entity=None passed to wandb.init
            call_kwargs = mock_wandb.init.call_args.kwargs
            assert call_kwargs["entity"] is None

    def test_create_wandb_tracker_raises_error_if_api_key_missing(self):
        """Test create_wandb_tracker raises ValueError if WANDB_API_KEY missing."""
        settings = Settings(wandb_api_key=None)
        with pytest.raises(ValueError, match="WANDB_API_KEY environment variable is required"):
            create_wandb_tracker(settings)
