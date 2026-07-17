"""Serve *your own* model through langres — over a vLLM/Ollama endpoint, or in-process.

Once you have a fine-tuned (or any local/HF) entity-matching model, langres runs
it through the **same** ``LLMMatcher`` — no new component. You choose *where* the
model runs with two knobs on the matcher:

- ``api_base=`` — point at a served, OpenAI-compatible endpoint (vLLM, Ollama, a
  gateway). ``model`` is the served model id. This is the production recipe.
- (omit ``api_base``) with an HF Hub id, a local directory, or a base+adapter ref
  — langres runs the model **in-process** via a lazily-loaded transformers
  backend, no server. Handy for eval/CI where standing up a server is overkill.

Either way the calibrated match score comes from the same first-token yes/no
logprob step, so ``confidence="logprob"`` + the ``binary_yes_no`` parser give you
a ranking score in [0, 1] from your own weights.

Serving is a *deployment recipe*, not a langres component: you run the server, and
point the matcher at it.


Recipe A — serve with vLLM, then point langres at it
----------------------------------------------------
Serve your model (one line, in its own shell)::

    # your fine-tuned model, or any HF causal LM
    vllm serve your-org/your-ft-matcher --port 8000

    # or serve a base model with a LoRA/QLoRA adapter, unmerged:
    vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --enable-lora --lora-modules ft=your-org/your-adapter --port 8000

Ollama works the same way (``ollama serve``; its OpenAI-compatible API is at
``http://localhost:11434/v1``).

Then run this file::

    uv run python examples/serve_your_model.py


Recipe B — in-process, no server (see ``main`` below)
-----------------------------------------------------
Skip ``api_base`` and pass a local/HF model; generation runs in-process::

    matcher = LLMMatcher(
        model="your-org/your-ft-matcher",   # HF id or a local dir
        confidence="logprob",
        response_parser="binary_yes_no",
    )
    # or a base + QLoRA adapter served unmerged (needs `pip install langres[finetune]`):
    matcher = LLMMatcher(model={"base": "...", "adapter": "..."}, confidence="logprob")
"""

from __future__ import annotations

from pydantic import BaseModel

from langres.core.matchers.llm_judge import LLMMatcher
from langres.core.resolver import ERModel


class Company(BaseModel):
    """The entity shape. Naming it explicitly is the production path.

    ``dedupe(records, matcher=my_matcher)`` used to infer this from the records'
    keys. That convenience is gone with the verbs: ``ERModel.from_schema`` wants
    a real schema, because an *inferred* one is an ephemeral class a fresh
    process cannot import back — so it could never round-trip through
    ``save()``/``load()``, which is precisely what you want for a model you
    fine-tuned and intend to serve. Six lines, once.
    """

    id: str
    name: str | None = None
    city: str | None = None


# A tiny batch to resolve. Every record needs a unique "id".
RECORDS = [
    {"id": "1", "name": "Acme Corporation", "city": "New York"},
    {"id": "2", "name": "Acme Corp", "city": "New York"},
    {"id": "3", "name": "Totally Different Co", "city": "Chicago"},
    {"id": "4", "name": "Totally Different Company", "city": "Chicago"},
    {"id": "5", "name": "Unrelated Bakery", "city": "Miami"},
]


def build_served_matcher(
    model: str = "your-org/your-ft-matcher",
    api_base: str = "http://localhost:8000/v1",
) -> LLMMatcher:
    """Point ``LLMMatcher`` at a served, OpenAI-compatible endpoint (Recipe A).

    ``model`` is the served model id; ``api_base`` is where your vLLM/Ollama
    server listens. ``confidence="logprob"`` + ``binary_yes_no`` give a calibrated
    [0, 1] score from your model's first-token yes/no logprobs. The matcher (incl.
    ``api_base``) is weightlessly serializable — a saved Resolver reloads pointed
    at the same endpoint.
    """
    return LLMMatcher(
        model=model,
        api_base=api_base,
        confidence="logprob",
        response_parser="binary_yes_no",
    )


def build_inprocess_matcher(model: str = "your-org/your-ft-matcher") -> LLMMatcher:
    """Run a local/HF model in-process via transformers, no server (Recipe B).

    Pass an HF Hub id, a local model directory, or a base+adapter dict; langres
    routes it to the in-process backend automatically (no ``api_base``).
    """
    return LLMMatcher(
        model=model,
        confidence="logprob",
        response_parser="binary_yes_no",
    )


def main() -> None:
    # Recipe A: served endpoint. Edit ``model`` / ``api_base`` for your server.
    matcher = build_served_matcher()

    # Recipe B: swap in the in-process backend (no server) by uncommenting:
    # matcher = build_inprocess_matcher("your-org/your-ft-matcher")

    # DedupeResult is a list[set[str]] of entity-id clusters.
    model = ERModel.from_schema(Company, matcher=matcher, threshold=0.6)
    clusters = model.dedupe(RECORDS)
    for cluster in clusters:
        print(cluster)


if __name__ == "__main__":
    main()
