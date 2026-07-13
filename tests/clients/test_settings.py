"""Tests for langres.clients.settings module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from langres.clients.settings import Settings


class TestSettings:
    """Tests for Settings class."""

    def test_settings_openrouter_api_key_from_env(self):
        """OPENROUTER_API_KEY loads into settings.openrouter_api_key (judge="auto" seam)."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test123"}, clear=True):
            settings = Settings()
            assert settings.openrouter_api_key == "or-test123"

    def test_settings_openrouter_api_key_defaults_to_none(self):
        """Absent OPENROUTER_API_KEY leaves the field None (judge="auto" then
        raises NoJudgeAvailableError -- fail fast, never a silent fallback)."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("pydantic_settings.sources.DotEnvSettingsSource.__call__", return_value={}),
        ):
            settings = Settings()
            assert settings.openrouter_api_key is None

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


class TestKeyDiscoveryContract:
    """The documented key-discovery order (Settings docstring): constructor
    kwargs > process env (empty string counts as set) > CWD ``.env`` (no
    walk-up). These tests lock the contract that makes a keyless run
    forceable at all -- they run against a THROWAWAY .env in tmp_path, never
    the repo's, and use fake keys only (zero network)."""

    def test_dotenv_file_in_cwd_fills_unset_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The legit dotenv-user path: a key in the project .env is picked up."""
        (tmp_path / ".env").write_text("OPENROUTER_API_KEY=fake-from-dotenv\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert Settings().openrouter_api_key == "fake-from-dotenv"

    def test_env_var_beats_dotenv_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / ".env").write_text("OPENROUTER_API_KEY=fake-from-dotenv\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-from-env")
        assert Settings().openrouter_api_key == "fake-from-env"

    def test_empty_env_var_beats_dotenv_and_counts_as_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The per-key keyless mechanism: OPENROUTER_API_KEY="" must WIN over
        the .env value (env_ignore_empty stays False -- see model_config) and
        read as falsy, so judge="auto" treats it as no key. If this test ever
        fails, no environment manipulation can force a keyless run inside a
        repo whose .env carries a real key."""
        (tmp_path / ".env").write_text("OPENROUTER_API_KEY=fake-from-dotenv\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        settings = Settings()
        assert settings.openrouter_api_key == ""
        assert not settings.openrouter_api_key

    def test_dotenv_lookup_is_cwd_relative_not_walk_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Settings' env_file=".env" reads the CURRENT directory only --
        unlike litellm's import-time load_dotenv(), it never walks up to a
        parent directory's .env."""
        (tmp_path / ".env").write_text("OPENROUTER_API_KEY=fake-from-parent\n")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        monkeypatch.chdir(subdir)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert Settings().openrouter_api_key is None


class TestLangresOfflineFlag:
    """LANGRES_OFFLINE parsing: truthy bool strings turn it on; empty string,
    "0"/"false", and unset mean off (an explicitly empty variable must never
    crash Settings -- consistent with empty-means-absent for keys)."""

    @pytest.mark.parametrize("raw", ["1", "true", "True", "yes"])
    def test_truthy_values_enable_offline(self, raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGRES_OFFLINE", raw)
        assert Settings().langres_offline is True

    @pytest.mark.parametrize("raw", ["", "0", "false", "no"])
    def test_falsy_values_disable_offline(self, raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGRES_OFFLINE", raw)
        assert Settings().langres_offline is False

    def test_unset_defaults_to_off(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # chdir to an empty tmp dir so the repo .env can't feed the field.
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("LANGRES_OFFLINE", raising=False)
        assert Settings().langres_offline is False
