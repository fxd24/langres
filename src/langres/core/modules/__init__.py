"""Module implementations for entity comparison."""

from langres.core.modules.cascade import CascadeModule
from langres.core.modules.llm_judge import LLMJudgeModule
from langres.core.modules.rapidfuzz import RapidfuzzModule

__all__ = ["RapidfuzzModule", "LLMJudgeModule", "CascadeModule"]
