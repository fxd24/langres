"""
langres.clients: Client configuration and factories for external services.

This module provides centralized configuration and client factories for:
- LLM providers (OpenAI, LiteLLM with Langfuse tracing)
- Experiment tracking (wandb)

W0.4: ``create_llm_client``/``create_wandb_tracker``/``create_trackio_tracker``
are resolved lazily (PEP 562) -- ``langres.clients.llm`` imports ``litellm`` at
module level, and litellm's own import runs ``load_dotenv()`` as a side effect,
silently populating ``OPENROUTER_API_KEY``/etc. from any ``.env`` on the path
(the SPEND-SAFETY footgun this branch closes). Eager-importing it here (a side
effect of importing ANY submodule of this package, e.g.
``langres.clients.settings``) meant plain ``import langres`` triggered that
side effect unconditionally. ``create_wandb_tracker`` (``wandb``, dev-only) and
``create_trackio_tracker`` (``trackio``) get the same treatment for consistency
and import weight -- both resolve through ``langres.clients.tracking``, but
``create_trackio_tracker`` only imports ``trackio`` inside its own function
body (see that module's docstring), so accessing either name here never
requires the other backend's dependency. ``Settings`` stays eager -- it's
pydantic-settings only.
"""

import importlib
from typing import TYPE_CHECKING, Any

from langres.clients.settings import Settings

if TYPE_CHECKING:
    from langres.clients.llm import create_llm_client
    from langres.clients.tracking import create_trackio_tracker, create_wandb_tracker

__all__ = [
    "Settings",
    "create_llm_client",
    "create_trackio_tracker",
    "create_wandb_tracker",
]

_LAZY: dict[str, str] = {
    "create_llm_client": "langres.clients.llm",
    "create_wandb_tracker": "langres.clients.tracking",
    "create_trackio_tracker": "langres.clients.tracking",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value
