"""Central configuration for external services."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for all external services.

    This class loads configuration from environment variables.
    All fields are optional - validation happens when services are actually used.

    Nothing here decides whether langres spends money. W4 deleted the
    key-sniffing front door (``matcher="auto"``, ``core.presets``): a key
    present in the environment no longer causes a paid call, because the paid
    path only runs when the caller *names* a paid architecture
    (``VectorLLMCascade(llm=...)``). These settings supply credentials to a
    model the user already chose; they never choose one.

    Discovery order (per field, highest priority first):
        1. Constructor kwargs -- ``Settings(openrouter_api_key="sk-...")``.
        2. Process environment (``os.environ``). A variable set to the EMPTY
           string counts as *set* and wins over the ``.env`` file. (Merely
           *unsetting* the variable does NOT: the ``.env`` file below refills
           it.)
        3. ``.env`` in the CURRENT WORKING DIRECTORY (``env_file=".env"`` is
           CWD-relative; pydantic-settings does not walk up parent
           directories). This is the conventional project-``.env`` pickup.

    Note: litellm separately runs ``load_dotenv()`` at import time, which DOES
    walk up the directory tree from its install location and loads the nearest
    ``.env`` into ``os.environ`` (without overriding already-set variables).
    This is why environment scrubbing was never a spend guard: the cure is
    structural (construct ``FuzzyString`` and there is nothing that could bill
    you), not a variable you unset.

    Environment variables:
        OPENAI_API_KEY: OpenAI API key
        OPENROUTER_API_KEY: OpenRouter API key
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
        HF_TOKEN: Hugging Face write token. Only needed by the Trackio tracker
            (TrackioTracker / resolve_tracker("trackio")) when syncing to an HF
            Space (TRACKIO_SPACE_ID set) -- a purely local Trackio run needs no
            token. huggingface_hub itself also reads HF_TOKEN (and the legacy
            HUGGING_FACE_HUB_TOKEN) directly, so this field is mainly for
            explicit construction (Settings(hf_token=...)); either source
            satisfies TrackioTracker's credential guard.
        TRACKIO_SPACE_ID: HF Space ("user/space") to sync Trackio runs to
            (optional). Unset -> local-first, no credentials/network. Read by
            TrackioTracker.start_run, not by capture_run.
        TRACKIO_DATASET_ID: HF Dataset to additionally sync Trackio metrics to
            (optional; requires TRACKIO_SPACE_ID). Read by TrackioTracker.start_run.

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

    # Trackio (experiment tracking, local-first): HF Space/Dataset sync is opt-in.
    hf_token: str | None = None
    trackio_space_id: str | None = None
    trackio_dataset_id: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Case sensitive to match env vars exactly
        case_sensitive=False,
        # NOTE: env_ignore_empty deliberately stays at its default (False), so
        # an env var set to "" WINS over the .env file rather than falling
        # through to it. This used to be load-bearing for the keyless-run
        # contract (forcing matcher="auto" to find no key); W4 deleted that
        # path, so it is now just the least surprising precedence rule --
        # explicitly empty means empty, not "consult the file".
    )
