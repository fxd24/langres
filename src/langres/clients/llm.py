"""LiteLLM client factory with Langfuse tracing."""

import logging
import os
from typing import Any

import litellm

from langres.clients.settings import Settings

logger = logging.getLogger(__name__)


def create_llm_client(settings: Settings | None = None) -> Any:
    """Create LiteLLM client with Langfuse tracing enabled.

    This function configures LiteLLM callbacks for Langfuse tracing.
    LiteLLM and Langfuse read credentials directly from environment variables.

    Args:
        settings: Optional Settings object for validation. If None, loads from environment.

    Returns:
        The litellm module configured with Langfuse callbacks.

    Environment variables required:
        LANGFUSE_PUBLIC_KEY: Langfuse public API key
        LANGFUSE_SECRET_KEY: Langfuse secret API key
        LANGFUSE_HOST: Langfuse host URL (default: https://cloud.langfuse.com)
        AZURE_API_ENDPOINT: Azure OpenAI endpoint URL
        AZURE_API_KEY: Azure OpenAI API key
        AZURE_API_VERSION: Azure OpenAI API version

    Example:
        # Ensure environment variables are set
        settings = Settings()  # Validates env vars exist
        client = create_llm_client(settings)

        # Use with Azure OpenAI (reads from AZURE_API_* env vars)
        response = client.completion(
            model="azure/gpt-4o-mini",  # Azure deployment name
            messages=[...]
        )

    Note:
        The litellm module itself acts as the client - it's not
        a class instance but a module with configuration.

    Note:
        All LLM calls made via the returned client will be automatically
        traced in Langfuse, including:
        - Prompts and completions
        - Model, temperature, and other parameters
        - Latency and token usage
        - Costs

    Note:
        For Azure OpenAI, use model names with "azure/" prefix:
        - "azure/gpt-4o-mini" (your deployment name)
        - LiteLLM reads AZURE_API_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION from environment
    """
    if settings is None:
        settings = Settings()  # type: ignore[call-arg]  # Loads from env vars

    # Configure Langfuse callbacks
    # Langfuse reads LANGFUSE_* env vars directly
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

    logger.info("LiteLLM client configured with Langfuse tracing")
    logger.info("Azure OpenAI endpoint: %s", settings.azure_api_endpoint)

    return litellm
