"""LiteLLM client factory with Langfuse tracing."""

import logging
import os
from typing import Any

import litellm

from langres.clients.settings import Settings

logger = logging.getLogger(__name__)


def create_llm_client(settings: Settings | None = None, enable_langfuse: bool = True) -> Any:
    """Create LiteLLM client with optional Langfuse tracing.

    This function configures LiteLLM with optional Langfuse callbacks for tracing.
    LiteLLM and Langfuse read credentials directly from environment variables.

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
        # Validate Langfuse env vars are present
        if not settings.langfuse_public_key:
            raise ValueError(
                "LANGFUSE_PUBLIC_KEY environment variable is required when enable_langfuse=True"
            )
        if not settings.langfuse_secret_key:
            raise ValueError(
                "LANGFUSE_SECRET_KEY environment variable is required when enable_langfuse=True"
            )

        # Langfuse reads LANGFUSE_* env vars directly
        litellm.success_callback = ["langfuse"]
        litellm.failure_callback = ["langfuse"]
        logger.info("LiteLLM client configured with Langfuse tracing")
    else:
        logger.info("LiteLLM client configured without tracing")

    if settings.azure_api_base:
        logger.info("Azure OpenAI endpoint: %s", settings.azure_api_base)

    return litellm
