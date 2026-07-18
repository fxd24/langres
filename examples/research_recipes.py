"""Try the Retrieve/Rerank/LLM recipe vocabulary offline with fake resources."""

from langres.architectures import RetrieveRerankLLM
from langres.resources import FakeEmbedder, FakeLLM, FakeReranker

records = [
    {"id": "a", "name": "Acme"},
    {"id": "b", "name": "ACME"},
    {"id": "c", "name": "Globex"},
]

architecture = RetrieveRerankLLM(
    embedder=FakeEmbedder(),
    reranker=FakeReranker(
        scores={
            '["a","b"]': 0.95,
            '["a","c"]': 0.10,
            '["b","c"]': 0.20,
        }
    ),
    llm=FakeLLM(
        responses={
            '["a","b"]': "MATCH",
            '["a","c"]': "NO_MATCH",
            '["b","c"]': "NO_MATCH",
        }
    ),
    retrieve_k=2,
    llm_k=2,
)

print(architecture.resources)
print(architecture.dedupe(records))
