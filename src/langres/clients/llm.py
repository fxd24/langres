"""LiteLLM client factory with Langfuse tracing."""

import logging
from typing import Any

import litellm

from langres.clients.settings import Settings

logger = logging.getLogger(__name__)


def create_llm_client(settings: Settings | None = None, enable_langfuse: bool = True) -> Any:
    """Create LiteLLM client with optional Langfuse tracing.

    This function configures LiteLLM with the Langfuse OpenTelemetry callback
    (``langfuse_otel``) for tracing. langfuse 4.x is the OpenTelemetry-based SDK,
    so LiteLLM emits traces via its ``langfuse_otel`` integration rather than the
    legacy v2 ``langfuse`` callback. LiteLLM's callback reads ``LANGFUSE_*``
    credentials directly from environment variables.

    Args:
        settings: Optional Settings object. If None, loads from environment.
        enable_langfuse: If True, configure Langfuse tracing (requires LANGFUSE_* env vars).
                        If False, no tracing is configured.

    Returns:
        The litellm module configured with optional Langfuse callbacks.

    Raises:
        ValueError: If enable_langfuse=True but required Langfuse env vars are missing.

    Environment variables (when enable_langfuse=True):
        LANGFUSE_PUBLIC_KEY: Langfuse public API key (required)
        LANGFUSE_SECRET_KEY: Langfuse secret API key (required)
        LANGFUSE_HOST: Langfuse host URL (default: https://cloud.langfuse.com)
        LANGFUSE_PROJECT: Langfuse project name (default: langres)

    Environment variables (when using Azure OpenAI):
        AZURE_API_BASE: Azure OpenAI endpoint URL
        AZURE_API_KEY: Azure OpenAI API key
        AZURE_API_VERSION: Azure OpenAI API version

    Example:
        # With Langfuse tracing (requires LANGFUSE_* env vars)
        client = create_llm_client(enable_langfuse=True)

        # Without tracing (no env vars required)
        client = create_llm_client(enable_langfuse=False)

        # Use with Azure OpenAI (reads from AZURE_API_* env vars)
        response = client.completion(
            model="azure/gpt-5-mini",  # Azure deployment name
            messages=[...]
        )

    Note:
        The litellm module itself acts as the client - it's not
        a class instance but a module with configuration.

    Note:
        For Azure OpenAI, use model names with "azure/" prefix:
        - "azure/gpt-5-mini" (your deployment name)
        - LiteLLM reads AZURE_API_BASE, AZURE_API_KEY, AZURE_API_VERSION from environment
    """
    if settings is None:
        settings = Settings()  # Loads from env vars

    # Configure Langfuse callbacks if enabled
    if enable_langfuse:
        # W0.4: langfuse moved to the dev dependency group (experiment-tracking
        # tooling, not part of the [llm] extra's install promise) -- imported
        # here, lazily, so `create_llm_client(enable_langfuse=False)` works
        # with just [llm] installed and no langfuse present.
        from langfuse import Langfuse

        # Validate Langfuse env vars are present
        if not settings.langfuse_public_key:
            raise ValueError(
                "LANGFUSE_PUBLIC_KEY environment variable is required when enable_langfuse=True"
            )
        if not settings.langfuse_secret_key:
            raise ValueError(
                "LANGFUSE_SECRET_KEY environment variable is required when enable_langfuse=True"
            )

        # Log configuration (masked for security)
        logger.info(
            "Initializing Langfuse: project=%s, host=%s, public_key=%s...",
            settings.langfuse_project,
            settings.langfuse_host,
            settings.langfuse_public_key[:10] if settings.langfuse_public_key else "None",
        )

        # Explicitly initialize the Langfuse client (v4 / OpenTelemetry SDK)
        # to validate credentials before LiteLLM starts emitting traces.
        try:
            langfuse_client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            # Verify credentials and connectivity (returns False on failure).
            if not langfuse_client.auth_check():
                raise ValueError("Langfuse auth_check failed (invalid credentials or host)")
            logger.info("Langfuse client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Langfuse client: %s", e)
            raise ValueError(f"Langfuse initialization failed: {e}") from e

        # Configure LiteLLM to use the Langfuse OpenTelemetry callback.
        # langfuse 4.x is OTel-based, so LiteLLM's "langfuse_otel" integration
        # (not the legacy v2 "langfuse" callback) is the correct one. It reads
        # LANGFUSE_* env vars for its own exporter.
        litellm.callbacks = ["langfuse_otel"]

        # Suppress verbose LiteLLM logging (prevents "Langfuse Layer Logging - logging success" spam)
        # LiteLLM logs every single API call at INFO level, which pollutes logs with thousands of messages
        litellm_logger = logging.getLogger("LiteLLM")
        litellm_logger.setLevel(logging.WARNING)  # Only show warnings and errors

        logger.info("LiteLLM configured with Langfuse callbacks")
    else:
        logger.info("LiteLLM client configured without tracing")

    if settings.azure_api_base:
        logger.info("Azure OpenAI endpoint: %s", settings.azure_api_base)

    return litellm
