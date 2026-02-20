"""
langres.clients: Client configuration and factories for external services.

This module provides centralized configuration and client factories for:
- LLM providers (OpenAI, LiteLLM with Langfuse tracing)
- Experiment tracking (wandb)
"""

from langres.clients.llm import create_llm_client
from langres.clients.settings import Settings
from langres.clients.tracking import create_wandb_tracker

__all__ = [
    "Settings",
    "create_llm_client",
    "create_wandb_tracker",
]
