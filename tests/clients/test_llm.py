"""Tests for langres.clients.llm module."""

import os
import sys
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
        """Test create_llm_client configures litellm with Langfuse (explicit opt-in)."""
        settings = Settings(
            openai_api_key="sk-test",
            wandb_api_key="wb-test",
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
            langfuse_host="https://custom.langfuse.com",
            azure_api_key="azure-key",
            azure_api_base="https://test.openai.azure.com",
        )

        with (
            patch("langres.clients.llm.litellm") as mock_litellm,
            patch("langfuse.Langfuse") as mock_langfuse_class,
        ):
            mock_langfuse_instance = MagicMock()
            mock_langfuse_instance.auth_check.return_value = True
            mock_langfuse_class.return_value = mock_langfuse_instance

            client = create_llm_client(settings, enable_langfuse=True)

            # Verify Langfuse client was initialized
            mock_langfuse_class.assert_called_once_with(
                public_key="pk-lf-test",
                secret_key="sk-lf-test",
                host="https://custom.langfuse.com",
            )
            mock_langfuse_instance.auth_check.assert_called_once()

            # Verify litellm Langfuse OpenTelemetry callback is configured
            assert mock_litellm.callbacks == ["langfuse_otel"]

            # Verify litellm module is returned
            assert client is mock_litellm

    def test_create_llm_client_without_settings_loads_from_env(self):
        """Test create_llm_client without settings loads from env vars (Langfuse opt-in)."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-env",
                "WANDB_API_KEY": "wb-env",
                "LANGFUSE_PUBLIC_KEY": "pk-lf-env",
                "LANGFUSE_SECRET_KEY": "sk-lf-env",
                "AZURE_API_KEY": "azure-env",
                "AZURE_API_BASE": "https://env.openai.azure.com",
            },
            clear=True,
        ):
            with (
                patch("langres.clients.llm.litellm") as mock_litellm,
                patch("langfuse.Langfuse") as mock_langfuse_class,
            ):
                mock_langfuse_instance = MagicMock()
                mock_langfuse_instance.auth_check.return_value = True
                mock_langfuse_class.return_value = mock_langfuse_instance

                client = create_llm_client(enable_langfuse=True)

                # Verify Langfuse initialized with env vars
                mock_langfuse_class.assert_called_once()
                mock_langfuse_instance.auth_check.assert_called_once()

                # Verify callback configured
                assert mock_litellm.callbacks == ["langfuse_otel"]

    def test_create_llm_client_uses_default_host(self):
        """Test create_llm_client with Settings using default Langfuse host."""
        settings = Settings(
            openai_api_key="sk-test",
            wandb_api_key="wb-test",
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
            azure_api_key="azure-key",
            azure_api_base="https://test.openai.azure.com",
            # langfuse_host not provided, should use default
        )

        with (
            patch("langres.clients.llm.litellm") as mock_litellm,
            patch("langfuse.Langfuse") as mock_langfuse_class,
        ):
            mock_langfuse_instance = MagicMock()
            mock_langfuse_instance.auth_check.return_value = True
            mock_langfuse_class.return_value = mock_langfuse_instance

            client = create_llm_client(settings, enable_langfuse=True)

            # Verify Langfuse initialized with default host
            mock_langfuse_class.assert_called_once_with(
                public_key="pk-lf-test",
                secret_key="sk-lf-test",
                host="https://cloud.langfuse.com",
            )
            mock_langfuse_instance.auth_check.assert_called_once()

            # Verify callback configured
            assert mock_litellm.callbacks == ["langfuse_otel"]

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
            azure_api_base="https://my-resource.openai.azure.com",
            azure_api_version="2024-02-15-preview",
        )

        with (
            patch("langres.clients.llm.litellm") as mock_litellm,
            patch("langfuse.Langfuse") as mock_langfuse_class,
        ):
            mock_langfuse_instance = MagicMock()
            mock_langfuse_instance.auth_check.return_value = True
            mock_langfuse_class.return_value = mock_langfuse_instance

            client = create_llm_client(settings, enable_langfuse=True)

            # Verify Langfuse initialized
            mock_langfuse_class.assert_called_once()
            mock_langfuse_instance.auth_check.assert_called_once()

            # Verify callback configured
            assert mock_litellm.callbacks == ["langfuse_otel"]

            # Verify Settings has Azure configuration
            assert settings.azure_api_key == "azure-key-123"
            assert settings.azure_api_base == "https://my-resource.openai.azure.com"
            assert settings.azure_api_version == "2024-02-15-preview"

            # Verify litellm module is returned
            assert client is mock_litellm

    def test_create_llm_client_without_langfuse(self):
        """Test create_llm_client with Langfuse explicitly disabled."""
        with patch("langres.clients.llm.litellm") as mock_litellm:
            client = create_llm_client(enable_langfuse=False)

            # Verify the Langfuse OTel callback is NOT configured
            assert mock_litellm.callbacks != ["langfuse_otel"]

            # Verify litellm module is returned
            assert client is mock_litellm

    def test_create_llm_client_default_does_not_require_langfuse_installed(self, monkeypatch):
        """The [llm]-only contract: enable_langfuse now defaults to False, so
        create_llm_client(Settings()) -- exactly what LLMMatcher.from_env()/
        LLMMatcher._get_client() call internally -- must succeed even when
        langfuse isn't installed (it's dev-group only, not part of [llm]).

        Simulates absence by making ``langfuse`` unimportable (the standard
        "set sys.modules[name] = None" trick forces the next `import`/`from
        ... import` to raise ImportError), since langfuse IS installed in
        this dev environment and can't be genuinely uninstalled for the test.
        """
        monkeypatch.setitem(sys.modules, "langfuse", None)
        with patch("langres.clients.llm.litellm") as mock_litellm:
            client = create_llm_client(Settings())

            assert client is mock_litellm
            assert mock_litellm.callbacks != ["langfuse_otel"]

    def test_create_llm_client_no_args_default_does_not_require_langfuse_installed(
        self, monkeypatch
    ):
        """Same contract as above, but via the no-Settings call LLMMatcher's
        ``forward_async`` path (langres/core/modules/llm_judge.py:370) uses."""
        monkeypatch.setitem(sys.modules, "langfuse", None)
        with patch("langres.clients.llm.litellm") as mock_litellm:
            client = create_llm_client()

            assert client is mock_litellm
            assert mock_litellm.callbacks != ["langfuse_otel"]

    def test_create_llm_client_enable_langfuse_raises_actionable_import_error_when_absent(
        self, monkeypatch
    ):
        """Explicitly requesting tracing without langfuse installed must raise
        a helpful ImportError (what/why/fix), not a raw ModuleNotFoundError."""
        monkeypatch.setitem(sys.modules, "langfuse", None)
        with pytest.raises(ImportError, match=r"langres\[dev\]"):
            create_llm_client(Settings(), enable_langfuse=True)

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

    def test_create_llm_client_without_azure_endpoint(self):
        """Test create_llm_client without Azure endpoint (standard OpenAI)."""
        settings = Settings(
            openai_api_key="sk-test",
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
            azure_api_base=None,  # No Azure endpoint
        )

        with (
            patch("langres.clients.llm.litellm") as mock_litellm,
            patch("langfuse.Langfuse") as mock_langfuse_class,
        ):
            mock_langfuse_instance = MagicMock()
            mock_langfuse_instance.auth_check.return_value = True
            mock_langfuse_class.return_value = mock_langfuse_instance

            client = create_llm_client(settings, enable_langfuse=True)

            # Verify Langfuse initialized
            mock_langfuse_class.assert_called_once()
            mock_langfuse_instance.auth_check.assert_called_once()

            # Verify callback configured
            assert mock_litellm.callbacks == ["langfuse_otel"]

            # Verify litellm module is returned
            assert client is mock_litellm

    def test_create_llm_client_raises_error_on_langfuse_init_failure(self):
        """Test create_llm_client raises ValueError if Langfuse initialization fails."""
        settings = Settings(
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
        )

        with patch("langfuse.Langfuse") as mock_langfuse_class:
            # Simulate Langfuse initialization failure
            mock_langfuse_class.side_effect = Exception("Connection failed")

            with pytest.raises(ValueError, match="Langfuse initialization failed"):
                create_llm_client(settings, enable_langfuse=True)

    def test_create_llm_client_raises_error_on_failed_auth_check(self):
        """Test create_llm_client raises ValueError if Langfuse auth_check fails."""
        settings = Settings(
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
        )

        with patch("langfuse.Langfuse") as mock_langfuse_class:
            mock_langfuse_instance = MagicMock()
            # auth_check returns False -> invalid credentials/host
            mock_langfuse_instance.auth_check.return_value = False
            mock_langfuse_class.return_value = mock_langfuse_instance

            with pytest.raises(ValueError, match="Langfuse initialization failed"):
                create_llm_client(settings, enable_langfuse=True)
