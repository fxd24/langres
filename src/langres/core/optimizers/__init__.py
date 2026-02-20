"""
langres.core.optimizers: Optimization components for hyperparameter tuning.

This module provides optimizers for tuning entity resolution pipelines:
- BlockerOptimizer: Optimize blocker hyperparameters (embedding models, k_neighbors)
- PromptOptimizer (future): Optimize LLM prompts using DSPy
"""

from langres.core.optimizers.blocker_optimizer import BlockerOptimizer

__all__ = [
    "BlockerOptimizer",
]
