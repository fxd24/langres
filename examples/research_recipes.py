"""Run all four research recipes with deterministic, zero-network resources."""

from langres.architectures import Retrieve, RetrieveLLM, RetrieveRerank, RetrieveRerankLLM
from langres.resources import FakeEmbedder, FakeLLM, FakeReranker

RECORDS = [
    {"id": "a", "name": "Acme"},
    {"id": "b", "name": "ACME"},
    {"id": "c", "name": "Globex"},
]
PAIR_SCORES = {
    '["a","b"]': 0.95,
    '["a","c"]': 0.10,
    '["b","c"]': 0.20,
}
LLM_RESPONSES = {
    '["a","b"]': "MATCH",
    '["a","c"]': "NO_MATCH",
    '["b","c"]': "NO_MATCH",
}


def build_recipes() -> dict[str, Retrieve | RetrieveRerank | RetrieveLLM | RetrieveRerankLLM]:
    """Return the four built-in recipes over local fake resources."""
    embedder = FakeEmbedder()
    reranker = FakeReranker(scores=PAIR_SCORES)
    llm = FakeLLM(responses=LLM_RESPONSES)
    return {
        "Retrieve": Retrieve(
            embedder=embedder,
            retrieve_k=2,
            threshold=-1.0,
        ),
        "RetrieveRerank": RetrieveRerank(
            embedder=embedder,
            reranker=reranker,
            retrieve_k=2,
            threshold=0.8,
        ),
        "RetrieveLLM": RetrieveLLM(
            embedder=embedder,
            llm=llm,
            retrieve_k=2,
            llm_k=2,
        ),
        "RetrieveRerankLLM": RetrieveRerankLLM(
            embedder=embedder,
            reranker=reranker,
            llm=llm,
            retrieve_k=2,
            llm_k=2,
        ),
    }


def main() -> None:
    """Print each recipe's resource slots and duplicate clusters."""
    for name, recipe in build_recipes().items():
        result = recipe.dedupe(RECORDS)
        print(f"{name}: resources={sorted(recipe.resources)} clusters={list(result)}")


if __name__ == "__main__":
    main()
