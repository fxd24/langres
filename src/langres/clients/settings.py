"""Central configuration for external services."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for all external services.

    This class loads configuration from environment variables.
    All fields are optional - validation happens when services are actually used.

    Environment variables:
        OPENAI_API_KEY: OpenAI API key
        WANDB_API_KEY: Weights & Biases API key
        WANDB_PROJECT: W&B project name (default: "langres")
        WANDB_ENTITY: W&B entity/team name (optional)
        LANGFUSE_PUBLIC_KEY: Langfuse public API key
        LANGFUSE_SECRET_KEY: Langfuse secret API key
        LANGFUSE_HOST: Langfuse host URL (default: "https://cloud.langfuse.com")
        LANGFUSE_PROJECT: Langfuse project name (default: "langres")
        AZURE_API_BASE: Azure OpenAI endpoint URL
        AZURE_API_KEY: Azure OpenAI API key
        AZURE_API_VERSION: Azure OpenAI API version (default: "2024-02-15-preview")
        QDRANT_URL: Qdrant vector database URL (optional)
        QDRANT_API_KEY: Qdrant API key (optional)

    Example:
        # Load from environment variables
        settings = Settings()

        # Access configuration (though components read from env directly)
        print(settings.openai_api_key)
        print(settings.azure_api_base)

    Example (.env file):
        # Create .env file:
        OPENAI_API_KEY=sk-...
        WANDB_API_KEY=...
        LANGFUSE_PUBLIC_KEY=pk-lf-...
        LANGFUSE_SECRET_KEY=sk-lf-...
        LANGFUSE_PROJECT=langres
        AZURE_API_BASE=https://my-resource.openai.azure.com
        AZURE_API_KEY=...
        AZURE_API_VERSION=2024-02-15-preview

        # Settings will automatically load
        settings = Settings()
    """

    # OpenAI / LLM
    openai_api_key: str | None = None

    # wandb (experiment tracking)
    wandb_api_key: str | None = None
    wandb_project: str = "langres"
    wandb_entity: str | None = None

    # Langfuse (LLM observability)
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_project: str = "langres"

    # Azure OpenAI (LiteLLM reads these directly from environment)
    azure_api_base: str | None = None
    azure_api_key: str | None = None
    azure_api_version: str = "2025-01-01-preview"

    # Qdrant vector database (optional, for future use)
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Case sensitive to match env vars exactly
        case_sensitive=False,
    )
