"""Central configuration for external services."""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for all external services.

    This class loads configuration from environment variables.
    All fields are optional - validation happens when services are actually used.

    Discovery order (per field, highest priority first):
        1. Constructor kwargs -- ``Settings(openrouter_api_key="sk-...")``.
        2. Process environment (``os.environ``). A variable set to the EMPTY
           string counts as *set* and wins over the ``.env`` file -- and an
           empty key is treated as absent by ``matcher="auto"``, so
           ``OPENROUTER_API_KEY="" OPENAI_API_KEY=""`` is the per-key way to
           force a keyless run. (Merely *unsetting* the variable does NOT: the
           ``.env`` file below refills it.)
        3. ``.env`` in the CURRENT WORKING DIRECTORY (``env_file=".env"`` is
           CWD-relative; pydantic-settings does not walk up parent
           directories). This is the conventional project-``.env`` pickup.

    Note: litellm separately runs ``load_dotenv()`` at import time, which DOES
    walk up the directory tree from its install location and loads the nearest
    ``.env`` into ``os.environ`` (without overriding already-set variables).
    That side effect never influences ``matcher="auto"``'s key discovery -- the
    auto decision is made from this class BEFORE litellm is ever imported.

    Environment variables:
        LANGRES_OFFLINE: When truthy ("1"/"true"), ``matcher="auto"`` treats
            every API key as absent and raises ``NoMatcherAvailableError``
            deterministically -- the process-wide switch to force keyless
            behavior (empty string / "0" / "false" / unset mean off).
            Scoped to auto-discovery only: an explicit ``matcher=`` choice in
            code is unaffected. See langres.core.presets.choose_auto_judge.
        OPENAI_API_KEY: OpenAI API key
        OPENROUTER_API_KEY: OpenRouter API key (drives matcher="auto" model
            selection; see langres.core.presets.choose_auto_judge)
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
        RUN_STORE_PATH: JSONL path for persisted run records (optional). NOTE:
            not auto-wired yet -- capture_run(store=...) does NOT default to this
            setting, so a caller must pass store= explicitly today (e.g.
            store=Settings().run_store_path). Threading it as the zero-config
            default store is deferred to the benchmark/harness wrap; until then
            capture_run(store=None) writes nothing. See langres.core.runs.
        MLFLOW_TRACKING_URI: MLflow tracking server URI (optional). Read by the
            MLflow tracker (MlflowTracker / resolve_tracker("mlflow")), NOT by
            capture_run directly -- it only takes effect when you use the MLflow
            backend. Unset lets MLflow pick its own default backend (a local
            file store; MlflowTracker enables it out of the box).
        MLFLOW_EXPERIMENT: MLflow experiment name (default: "langres"). Read by
            the MLflow tracker (MlflowTracker.start_run), not by capture_run.

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

    # Offline switch: matcher="auto" treats every API key as absent when true.
    # Deterministic because the process env beats the .env file (see the
    # discovery order in the class docstring) -- setting LANGRES_OFFLINE=1
    # forces the keyless fail-fast path even when a .env in the CWD carries a
    # real key. Scoped to auto-discovery; explicit matcher= choices bypass it.
    langres_offline: bool = False

    # OpenAI / LLM
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None

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

    # Experiment tracking (S1): run-record persistence + MLflow backend config.
    run_store_path: str | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "langres"

    @field_validator("langres_offline", mode="before")
    @classmethod
    def _empty_offline_is_off(cls, value: object) -> object:
        """Map ``LANGRES_OFFLINE=""`` to off instead of a bool ValidationError.

        Consistent with the empty-string-means-absent key contract (class
        docstring, discovery order step 2): an explicitly empty variable means
        "not set", never a crash.
        """
        return False if value == "" else value

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Case sensitive to match env vars exactly
        case_sensitive=False,
        # NOTE: env_ignore_empty deliberately stays at its default (False).
        # The keyless-run contract depends on it: an env var set to "" must
        # WIN over the .env file (and then read as absent), or no environment
        # manipulation could ever force a keyless run inside a repo whose
        # .env carries a real key. See the class docstring, discovery step 2.
    )
