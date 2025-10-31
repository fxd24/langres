"""Tests for langres.clients.llm module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from langres.clients.llm import create_llm_client
from langres.clients.settings import Settings


@pytest.fixture(autouse=True)
def clean_langfuse_env():
    """Clean Langfuse environment variables after each test."""
    yield
    # Cleanup after test
    for key in ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]:
        os.environ.pop(key, None)


class TestCreateLLMClient:
    """Tests for create_llm_client factory function."""

    def test_create_llm_client_with_settings(self):
        """Test create_llm_client configures litellm with Langfuse."""
        settings = Settings(
            openai_api_key="sk-test",
            wandb_api_key="wb-test",
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
            langfuse_host="https://custom.langfuse.com",
            azure_api_key="azure-key",
            azure_api_endpoint="https://test.openai.azure.com",
        )

        with patch("langres.clients.llm.litellm") as mock_litellm:
            client = create_llm_client(settings)

            # Verify litellm callbacks are configured
            assert "langfuse" in mock_litellm.success_callback
            assert "langfuse" in mock_litellm.failure_callback

            # Verify litellm module is returned
            assert client is mock_litellm

    def test_create_llm_client_without_settings_loads_from_env(self):
        """Test create_llm_client without settings loads from env vars."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-env",
                "WANDB_API_KEY": "wb-env",
                "LANGFUSE_PUBLIC_KEY": "pk-lf-env",
                "LANGFUSE_SECRET_KEY": "sk-lf-env",
                "AZURE_API_KEY": "azure-env",
                "AZURE_API_ENDPOINT": "https://env.openai.azure.com",
            },
            clear=True,
        ):
            with patch("langres.clients.llm.litellm") as mock_litellm:
                client = create_llm_client()

                # Verify callbacks configured
                assert "langfuse" in mock_litellm.success_callback
                assert "langfuse" in mock_litellm.failure_callback

    def test_create_llm_client_uses_default_host(self):
        """Test create_llm_client with Settings using default Langfuse host."""
        settings = Settings(
            openai_api_key="sk-test",
            wandb_api_key="wb-test",
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
            azure_api_key="azure-key",
            azure_api_endpoint="https://test.openai.azure.com",
            # langfuse_host not provided, should use default
        )

        with patch("langres.clients.llm.litellm") as mock_litellm:
            client = create_llm_client(settings)

            # Verify callbacks configured
            assert "langfuse" in mock_litellm.success_callback
            assert "langfuse" in mock_litellm.failure_callback

            # Verify Settings has default host
            assert settings.langfuse_host == "https://cloud.langfuse.com"

            # Verify litellm module is returned
            assert client is mock_litellm

    def test_create_llm_client_validates_azure_settings(self):
        """Test create_llm_client with Azure OpenAI settings."""
        settings = Settings(
            openai_api_key="sk-test",
            wandb_api_key="wb-test",
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
            azure_api_key="azure-key-123",
            azure_api_endpoint="https://my-resource.openai.azure.com",
            azure_api_version="2024-02-15-preview",
        )

        with patch("langres.clients.llm.litellm") as mock_litellm:
            client = create_llm_client(settings)

            # Verify callbacks configured
            assert "langfuse" in mock_litellm.success_callback
            assert "langfuse" in mock_litellm.failure_callback

            # Verify Settings has Azure configuration
            assert settings.azure_api_key == "azure-key-123"
            assert settings.azure_api_endpoint == "https://my-resource.openai.azure.com"
            assert settings.azure_api_version == "2024-02-15-preview"

            # Verify litellm module is returned
            assert client is mock_litellm

    def test_create_llm_client_without_langfuse(self):
        """Test create_llm_client with Langfuse disabled."""
        with patch("langres.clients.llm.litellm") as mock_litellm:
            client = create_llm_client(enable_langfuse=False)

            # Verify callbacks NOT configured
            assert mock_litellm.success_callback != ["langfuse"]
            assert mock_litellm.failure_callback != ["langfuse"]

            # Verify litellm module is returned
            assert client is mock_litellm

    def test_create_llm_client_raises_error_if_langfuse_keys_missing(self):
        """Test create_llm_client raises ValueError if Langfuse keys missing."""
        settings = Settings(
            langfuse_public_key=None,
            langfuse_secret_key=None,
        )
        with pytest.raises(
            ValueError, match="LANGFUSE_PUBLIC_KEY environment variable is required"
        ):
            create_llm_client(settings, enable_langfuse=True)

    def test_create_llm_client_raises_error_if_langfuse_secret_missing(self):
        """Test create_llm_client raises ValueError if Langfuse secret missing."""
        settings = Settings(
            langfuse_public_key="pk-test",
            langfuse_secret_key=None,
        )
        with pytest.raises(
            ValueError, match="LANGFUSE_SECRET_KEY environment variable is required"
        ):
            create_llm_client(settings, enable_langfuse=True)
