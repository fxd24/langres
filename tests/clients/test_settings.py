"""Tests for langres.clients.settings module."""

import os
from unittest.mock import patch

import pytest

from langres.clients.settings import Settings


class TestSettings:
    """Tests for Settings class."""

    def test_settings_with_all_required_fields(self):
        """Test Settings initialization with all required fields."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test123",
                "WANDB_API_KEY": "wandb-test123",
                "LANGFUSE_PUBLIC_KEY": "pk-lf-test123",
                "LANGFUSE_SECRET_KEY": "sk-lf-test123",
                "AZURE_API_KEY": "azure-key",
                "AZURE_API_BASE": "https://test.openai.azure.com",
            },
            clear=True,
        ):
            settings = Settings()
            assert settings.openai_api_key == "sk-test123"
            assert settings.wandb_api_key == "wandb-test123"
            assert settings.langfuse_public_key == "pk-lf-test123"
            assert settings.langfuse_secret_key == "sk-lf-test123"
            assert settings.azure_api_key == "azure-key"
            assert settings.azure_api_base == "https://test.openai.azure.com"

    def test_settings_default_values(self):
        """Test Settings default values for optional fields."""
        # Patch both os.environ AND the .env file to prevent leakage
        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "sk-test123",
                    "WANDB_API_KEY": "wandb-test123",
                    "LANGFUSE_PUBLIC_KEY": "pk-lf-test123",
                    "LANGFUSE_SECRET_KEY": "sk-lf-test123",
                    "AZURE_API_KEY": "azure-key",
                    "AZURE_API_BASE": "https://test.openai.azure.com",
                },
                clear=True,
            ),
            patch("pydantic_settings.sources.DotEnvSettingsSource.__call__", return_value={}),
        ):
            settings = Settings()
            assert settings.wandb_project == "langres"
            assert settings.wandb_entity is None
            assert settings.langfuse_host == "https://cloud.langfuse.com"
            assert settings.langfuse_project == "langres"

    def test_settings_custom_optional_values(self):
        """Test Settings with custom optional values."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test123",
                "WANDB_API_KEY": "wandb-test123",
                "LANGFUSE_PUBLIC_KEY": "pk-lf-test123",
                "LANGFUSE_SECRET_KEY": "sk-lf-test123",
                "AZURE_API_KEY": "azure-key",
                "AZURE_API_BASE": "https://test.openai.azure.com",
                "WANDB_PROJECT": "custom-project",
                "WANDB_ENTITY": "my-team",
                "LANGFUSE_HOST": "https://custom.langfuse.com",
                "LANGFUSE_PROJECT": "custom-langfuse-project",
            },
            clear=True,
        ):
            settings = Settings()
            assert settings.wandb_project == "custom-project"
            assert settings.wandb_entity == "my-team"
            assert settings.langfuse_host == "https://custom.langfuse.com"
            assert settings.langfuse_project == "custom-langfuse-project"

    def test_settings_with_no_env_vars(self):
        """Test that Settings can be initialized with explicit None values (all optional)."""
        settings = Settings(
            openai_api_key=None,
            wandb_api_key=None,
            langfuse_public_key=None,
            langfuse_secret_key=None,
            azure_api_key=None,
            azure_api_base=None,
        )
        assert settings.openai_api_key is None
        assert settings.wandb_api_key is None
        assert settings.langfuse_public_key is None
        assert settings.langfuse_secret_key is None
        assert settings.azure_api_key is None
        assert settings.azure_api_base is None

    def test_settings_with_partial_fields(self):
        """Test that Settings works with only some fields set."""
        settings = Settings(
            openai_api_key="sk-test123",
            wandb_api_key="wandb-test123",
        )
        assert settings.openai_api_key == "sk-test123"
        assert settings.wandb_api_key == "wandb-test123"
        # Other fields loaded from .env or None
        assert settings.wandb_project == "langres"  # default

    def test_settings_with_azure_openai_fields(self):
        """Test Settings with Azure OpenAI configuration."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test123",
                "WANDB_API_KEY": "wandb-test123",
                "LANGFUSE_PUBLIC_KEY": "pk-lf-test123",
                "LANGFUSE_SECRET_KEY": "sk-lf-test123",
                "AZURE_API_KEY": "azure-key-test",
                "AZURE_API_BASE": "https://my-resource.openai.azure.com",
                "AZURE_API_VERSION": "2025-01-01-preview",
            },
            clear=True,
        ):
            settings = Settings()
            assert settings.azure_api_key == "azure-key-test"
            assert settings.azure_api_base == "https://my-resource.openai.azure.com"
            assert settings.azure_api_version == "2025-01-01-preview"

    def test_settings_azure_default_version(self):
        """Test that Azure API version has sensible default."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test123",
                "WANDB_API_KEY": "wandb-test123",
                "LANGFUSE_PUBLIC_KEY": "pk-lf-test123",
                "LANGFUSE_SECRET_KEY": "sk-lf-test123",
                "AZURE_API_KEY": "azure-key-test",
                "AZURE_API_BASE": "https://my-resource.openai.azure.com",
            },
            clear=True,
        ):
            settings = Settings()
            assert settings.azure_api_version == "2025-01-01-preview"
