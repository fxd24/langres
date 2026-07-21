"""Reusable model resources and their additive Op adapters.

The package exports weightless contracts, deterministic fakes, and lazy
production implementations. Importing it does not load torch, Transformers,
Sentence Transformers, or LiteLLM.
"""

from langres.resources.base import (
    ChatMessage,
    Embedder,
    EmbeddingBatch,
    EmbeddingFacts,
    GenerationBatch,
    GenerationEnvelope,
    GenerationRequest,
    GenerationUsage,
    LLM,
    LLMRuntimeConfig,
    RerankBatch,
    RerankRequest,
    Reranker,
    RerankerRuntimeConfig,
    ResourceRuntimeConfig,
    SentenceTransformerRuntimeConfig,
    UnknownGenerationCostError,
)
from langres.resources.fakes import FakeEmbedder, FakeLLM, FakeReranker
from langres.resources.llm import LiteLLM, TransformersLLM, llm_from_model_ref
from langres.resources.op_adapters import (
    Generate,
    LLMMatcherAdapter,
    Parse,
    ParsedGeneration,
    Rerank,
    parse_binary_response,
    parse_score_response,
)
from langres.resources.retrieve import Retrieve
from langres.resources.sentence_transformers import (
    CrossEncoderReranker,
    SentenceTransformer,
)

__all__ = [
    "ChatMessage",
    "CrossEncoderReranker",
    "Embedder",
    "EmbeddingBatch",
    "EmbeddingFacts",
    "FakeEmbedder",
    "FakeLLM",
    "FakeReranker",
    "Generate",
    "GenerationBatch",
    "GenerationEnvelope",
    "GenerationRequest",
    "GenerationUsage",
    "LLM",
    "LLMMatcherAdapter",
    "LLMRuntimeConfig",
    "LiteLLM",
    "Parse",
    "ParsedGeneration",
    "Rerank",
    "RerankBatch",
    "RerankRequest",
    "Reranker",
    "RerankerRuntimeConfig",
    "Retrieve",
    "ResourceRuntimeConfig",
    "SentenceTransformer",
    "SentenceTransformerRuntimeConfig",
    "TransformersLLM",
    "UnknownGenerationCostError",
    "llm_from_model_ref",
    "parse_binary_response",
    "parse_score_response",
]
