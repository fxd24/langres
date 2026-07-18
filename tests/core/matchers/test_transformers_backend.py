"""Fast unit tests for the in-process ``TransformersBackend`` response construction.

torch is available under ``--all-extras``, so these build **real** logits tensors
and drive the backend's pure logic — the logprobs→``top_logprobs`` conversion, the
LiteLLM-shaped response assembly, and the want_logprobs on/off branches — WITHOUT
downloading a model (the model/tokenizer are injected fakes, so ``_ensure_loaded``
is a no-op). The real end-to-end download+generate path is the ``@pytest.mark.slow``
smoke in ``test_llm_judge_serve_smoke.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="requires the [semantic] extra (torch)")

from langres.core.matchers.model_ref import ModelRef
from langres.core.matchers.transformers_backend import TransformersBackend


class _FakeTokenizer:
    """Minimal tokenizer: no chat template, id i -> ``"<i>"``, callable encode."""

    chat_template = None
    pad_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt") -> SimpleNamespace:
        # Two prompt tokens, arbitrary ids.
        return SimpleNamespace(input_ids=torch.tensor([[7, 8]]))

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(f"<{int(i)}>" for i in ids)


class _FakeModel:
    """Stub causal LM: returns canned ``sequences`` + per-step ``scores``."""

    device = torch.device("cpu")

    def __init__(self, sequences: Any, scores: Any) -> None:
        self._sequences = sequences
        self._scores = scores
        self.kwargs: dict[str, Any] = {}

    def generate(self, input_ids: Any, **kwargs: Any) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(sequences=self._sequences, scores=self._scores)


def _backend_with(model: _FakeModel) -> TransformersBackend:
    backend = TransformersBackend(ModelRef(base="fake/model", kind="hf"))
    # Pre-seed so _ensure_loaded() early-returns (no transformers download).
    backend._tokenizer = _FakeTokenizer()
    backend._model = model
    return backend


def test_complete_builds_litellm_shaped_response_with_logprobs() -> None:
    # prompt = [7, 8]; one generated token id 1 (the argmax below).
    sequences = torch.tensor([[7, 8, 1]])
    # vocab=4 logits favouring id 1, then id 2.
    scores = (torch.tensor([[1.0, 3.0, 0.5, 0.2]]),)
    backend = _backend_with(_FakeModel(sequences, scores))

    resp = backend.complete(
        [{"role": "user", "content": "hi"}], temperature=0.0, want_logprobs=True
    )

    # message content decodes the one new token id (1).
    assert resp.choices[0].message.content == "<1>"
    # usage: 2 prompt tokens, 1 completion token.
    assert resp.usage.prompt_tokens == 2
    assert resp.usage.completion_tokens == 1
    # logprobs.content has one position; its chosen token + top_logprobs.
    content = resp.choices[0].logprobs.content
    assert len(content) == 1
    assert content[0].token == "<1>"
    # top_logprobs are the full vocab (<=20), ranked by logprob; id 1 is top.
    tokens = [alt.token for alt in content[0].top_logprobs]
    assert tokens[0] == "<1>"
    # logprobs equal log_softmax of the logits at those ids.
    expected = torch.log_softmax(torch.tensor([1.0, 3.0, 0.5, 0.2]), dim=-1)
    assert content[0].top_logprobs[0].logprob == pytest.approx(float(expected[1]))
    assert content[0].logprob == pytest.approx(float(expected[1]))


def test_complete_without_logprobs_omits_the_logprobs_block() -> None:
    sequences = torch.tensor([[7, 8, 2]])
    backend = _backend_with(_FakeModel(sequences, None))

    resp = backend.complete(
        [{"role": "user", "content": "hi"}], temperature=0.0, want_logprobs=False
    )

    assert resp.choices[0].logprobs is None
    assert resp.choices[0].message.content == "<2>"
    assert resp.usage.completion_tokens == 1


def test_complete_honors_temperature_and_seed_for_sampling() -> None:
    sequences = torch.tensor([[7, 8, 2]])
    model = _FakeModel(sequences, None)
    backend = TransformersBackend(ModelRef(base="fake/model", kind="hf"), seed=17)
    backend._tokenizer = _FakeTokenizer()
    backend._model = model

    backend.complete(
        [{"role": "user", "content": "hi"}],
        temperature=0.7,
        want_logprobs=False,
    )

    assert model.kwargs["do_sample"] is True
    assert model.kwargs["temperature"] == pytest.approx(0.7)
    assert torch.initial_seed() == 17


def test_complete_uses_greedy_decoding_at_zero_temperature() -> None:
    sequences = torch.tensor([[7, 8, 2]])
    model = _FakeModel(sequences, None)
    backend = _backend_with(model)

    backend.complete(
        [{"role": "user", "content": "hi"}],
        temperature=0.0,
        want_logprobs=False,
    )

    assert model.kwargs["do_sample"] is False
    assert "temperature" not in model.kwargs


def test_top_logprobs_capped_at_twenty() -> None:
    # A 50-token vocab must yield exactly 20 alternatives (the API-max mirror).
    sequences = torch.tensor([[7, 8, 3]])
    scores = (torch.arange(50, dtype=torch.float).unsqueeze(0),)
    backend = _backend_with(_FakeModel(sequences, scores))

    resp = backend.complete(
        [{"role": "user", "content": "hi"}], temperature=0.0, want_logprobs=True
    )

    assert len(resp.choices[0].logprobs.content[0].top_logprobs) == 20


def test_response_feeds_the_matchers_shared_pyes_step() -> None:
    """The backend's response is consumed by LLMMatcher._confidence_from_response.

    Closes the loop: a first token whose top_logprobs carry yes/no mass yields the
    same p_yes the served path would — proving the response shape is what the
    shared logprob→score step reads.
    """
    from langres.core.matchers.llm_judge import LLMMatcher
    from langres.core.models import CompanySchema

    # A tokenizer whose token 1 decodes to "Yes" and token 2 to "No".
    class _YesNoTokenizer(_FakeTokenizer):
        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            mapping = {1: "Yes", 2: "No"}
            return "".join(mapping.get(int(i), f"<{int(i)}>") for i in ids)

    sequences = torch.tensor([[7, 8, 1]])
    # P(Yes) logit high, P(No) lower; other ids negligible.
    scores = (torch.tensor([[-10.0, 2.0, 0.5, -10.0]]),)
    backend = TransformersBackend(ModelRef(base="fake/model", kind="hf"))
    backend._tokenizer = _YesNoTokenizer()
    backend._model = _FakeModel(sequences, scores)

    resp = backend.complete(
        [{"role": "user", "content": "hi"}], temperature=0.0, want_logprobs=True
    )

    judge = LLMMatcher[CompanySchema](client=object(), model="fake/model", confidence="logprob")
    fragment = judge._confidence_from_response(resp)
    assert fragment is not None
    probs = torch.softmax(torch.tensor([2.0, 0.5]), dim=-1)  # yes, no over the 2-way subspace
    assert fragment["p_yes"] == pytest.approx(float(probs[0]), abs=1e-4)
    assert 0.0 <= fragment["p_yes"] <= 1.0
