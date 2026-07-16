"""In-process HF causal-LM backend for :class:`~langres.core.matchers.llm_judge.LLMMatcher`.

Runs generation **and first-token logprobs locally via transformers** (no server),
returning a **LiteLLM-shaped response** so ``LLMMatcher``'s existing parse +
confidence path (:meth:`LLMMatcher._map_verdict` /
:meth:`LLMMatcher._confidence_from_response`) consumes it *identically* — the
calibrated :attr:`PairwiseJudgement.score` comes from the same token-logprob step
whether the model is served over an API (litellm) or run in-process. There is one
logprob→score computation, not two: this backend only reproduces the *response
shape* that computation already reads.

``torch`` / ``transformers`` are imported **lazily on first generation**, never at
module load, so importing this module (and a bare ``import langres``) stays free of
the heavy stack (guarded by ``tests/test_import_budget.py``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from langres.core.model_ref import ModelRef

# ---------------------------------------------------------------------------
# Minimal LiteLLM/OpenAI-shaped response objects.
#
# Only the attributes LLMMatcher actually reads are modelled:
#   response.choices[0].message.content
#   response.choices[0].logprobs.content[i].token / .top_logprobs[j].token/.logprob
#   response.usage.prompt_tokens / .completion_tokens
# Billing (``parse_openrouter_billing``) and ``LLMUsage.from_response`` read the
# rest via ``getattr(..., None)``, so the absence of ``cost`` / ``provider`` /
# ``*_tokens_details`` cleanly yields "no real cost, all-zero subsets".
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TopLogprob:
    """One alternative token at a position, with its log-probability."""

    token: str
    logprob: float


@dataclass(frozen=True)
class _TokenLogprobs:
    """The generated token at a position plus its top-k alternatives."""

    token: str
    logprob: float
    top_logprobs: list[_TopLogprob]


@dataclass(frozen=True)
class _Logprobs:
    content: list[_TokenLogprobs]


@dataclass(frozen=True)
class _Message:
    content: str


@dataclass(frozen=True)
class _Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class _Choice:
    message: _Message
    logprobs: _Logprobs | None


@dataclass(frozen=True)
class _Response:
    choices: list[_Choice]
    usage: _Usage


class TransformersBackend:
    """Lazily-loaded local causal-LM completion backend (one model per instance).

    Construct with a :class:`ModelRef`; the model + tokenizer load on the first
    :meth:`complete` call and are cached for the instance's lifetime. Generation
    is greedy (``do_sample=False``) — for the yes/no + first-token-logprob probe a
    deterministic single step is what we score, and it keeps runs reproducible.
    """

    #: Mirror the API-max ``top_logprobs`` the served (litellm) path requests, so
    #: the two-way yes/no subspace this backend can attribute mass to is as wide.
    _TOP_LOGPROBS = 20

    def __init__(self, model_ref: ModelRef, *, max_new_tokens: int = 8):
        self._ref = model_ref
        self._max_new_tokens = max_new_tokens
        self._model: Any = None
        self._tokenizer: Any = None
        # Generation on one shared model is serialized: the async path runs
        # ``complete`` in a worker thread (``asyncio.to_thread``), and concurrent
        # ``generate`` calls on a single model object are not safe.
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        """Load the base model + tokenizer (and optional PEFT adapter) once."""
        if self._model is not None:
            return
        # Lazy, first-use only: torch/transformers must never import at module load.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self._ref.base)
        model = AutoModelForCausalLM.from_pretrained(self._ref.base)
        if self._ref.adapter is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:  # pragma: no cover - exercised in PR-F
                raise ImportError(
                    "Serving a base+adapter model_ref in-process needs PEFT. "
                    "Install it with `pip install langres[finetune]` (or `pip install peft`)."
                ) from exc
            model = PeftModel.from_pretrained(model, self._ref.adapter)
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer
        self._model = model

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        want_logprobs: bool,
    ) -> _Response:
        """Generate a completion for ``messages``, returning a LiteLLM-shaped response.

        ``temperature`` is accepted for signature parity with the served path but
        the probe decodes greedily (deterministic). When ``want_logprobs`` is set,
        the response carries per-token ``top_logprobs`` for the generated tokens so
        the shared first-token yes/no credence step can compute ``p_yes``.
        """
        import torch

        with self._lock:
            self._ensure_loaded()
            tokenizer, model = self._tokenizer, self._model
            input_ids = self._encode(messages)
            prompt_len = int(input_ids.shape[1])
            # One sequence, no padding -> an all-ones mask is exact; passing it
            # explicitly silences transformers' "attention mask not set" warning
            # (pad_token == eos_token) and its "unexpected behavior" caveat.
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                    output_scores=want_logprobs,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_ids = generated.sequences[0][prompt_len:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            logprobs = (
                self._logprobs_from_scores(generated.scores, new_ids) if want_logprobs else None
            )
            usage = _Usage(prompt_tokens=prompt_len, completion_tokens=int(len(new_ids)))
            return _Response(
                choices=[_Choice(message=_Message(content=text), logprobs=logprobs)],
                usage=usage,
            )

    def _encode(self, messages: list[dict[str, str]]) -> Any:
        """Tokenize ``messages`` into model-device input ids.

        Uses the tokenizer's chat template when present (instruct models) so the
        yes/no prompt is framed the way the model expects; otherwise falls back to
        concatenating the message contents (base models with no chat template).
        """
        tokenizer, model = self._tokenizer, self._model
        if getattr(tokenizer, "chat_template", None):
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            text = "\n\n".join(message["content"] for message in messages)
            input_ids = tokenizer(text, return_tensors="pt").input_ids
        return input_ids.to(model.device)

    def _logprobs_from_scores(self, scores: Any, new_ids: Any) -> _Logprobs:
        """Convert transformers' per-step ``scores`` into LiteLLM ``logprobs.content``.

        For each generated step, log-softmax the vocabulary logits, take the top-k
        alternatives (:attr:`_TOP_LOGPROBS`), and record the actually-generated
        token's own logprob — the exact ``token`` / ``top_logprobs`` shape
        :meth:`LLMMatcher._confidence_from_response` reads.
        """
        import torch

        content: list[_TokenLogprobs] = []
        for step, step_logits in enumerate(scores or ()):
            row = torch.log_softmax(step_logits[0].float(), dim=-1)
            k = min(self._TOP_LOGPROBS, int(row.shape[-1]))
            topk = torch.topk(row, k=k)
            top = [
                _TopLogprob(token=self._tokenizer.decode([int(tid)]), logprob=float(logprob))
                for tid, logprob in zip(topk.indices.tolist(), topk.values.tolist())
            ]
            chosen = int(new_ids[step])
            content.append(
                _TokenLogprobs(
                    token=self._tokenizer.decode([chosen]),
                    logprob=float(row[chosen]),
                    top_logprobs=top,
                )
            )
        return _Logprobs(content=content)
