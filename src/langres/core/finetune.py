"""``finetune()``: QLoRA fine-tune a small LM on labeled pairs → a weightless ``model_ref``.

The standalone training primitive of the training surface: given labeled candidate
pairs, fine-tune a small base LM (LoRA / 4-bit QLoRA) to answer the yes/no match
prompt, and return a :class:`~langres.core.matchers.model_ref.ModelRef` (base +
adapter) that the existing :class:`~langres.core.matchers.llm_judge.LLMMatcher`
serves in-process — no new matcher class, no Resolver. ``Resolver.fit(...,
method=QLoRA(...))`` wraps this (see ``Resolver._fit_finetune``).

**Import-light by construction.** The heavy training stack (``peft`` / ``trl`` /
``bitsandbytes`` / ``torch`` / ``transformers``) is imported **lazily inside the
trainer's ``train()``**, never at module load — so ``import langres`` and even
``import langres.core.finetune`` never pull it (locked by
``tests/test_import_budget.py``). The :class:`QLoRA` method object is plain
config; the training mechanics live behind the injectable :class:`FinetuneTrainer`
seam, so the orchestration (rendering, ref/report assembly, cost) is fully
testable with a fake trainer and no GPU.

Prompt/target rendering matches serving: each pair becomes a chat conversation
``[{user: <the LLMMatcher prompt>}, {assistant: "Yes"|"No"}]`` rendered with the
SAME template + record serializer the served :class:`LLMMatcher` uses, so what the
model is trained on is exactly what it sees at inference.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import Field

from langres.core.matchers.model_ref import ModelRef, normalize_model_ref
from langres.core.methods_api import Method

if TYPE_CHECKING:
    from langres.core.models import ERCandidate

#: One rendered training example: a chat conversation the trainer tokenizes with
#: the base model's chat template (``[{"role": "user", ...}, {"role":
#: "assistant", "content": "Yes"|"No"}]``). Kept as plain dicts so the trainer
#: seam has no langres-specific coupling.
Conversation = list[dict[str, str]]

#: The default fine-tune prompt: a yes/no-eliciting template (so the ``"Yes"`` /
#: ``"No"`` assistant target is natural, and the served matcher's first-token
#: logprob credence reads real yes/no mass). Used by BOTH training and serving so
#: they match; ``{left}`` / ``{right}`` are filled with the two rendered records.
#: Entity-noun-neutral on purpose -- one template serves every domain.
FINETUNE_YES_NO_PROMPT = (
    "Do these two records refer to the same real-world entity?\n\n"
    "Record A:\n{left}\n\n"
    "Record B:\n{right}\n\n"
    "Answer with a single word: Yes or No."
)

#: A labeled candidate pair: ``(candidate, is_match)``. The training input shape,
#: produced by the caller (``Resolver._fit_finetune`` zips ``align_pairs`` output).
LabeledCandidate = tuple["ERCandidate[Any]", bool]


class QLoRA(Method):
    """Fine-tune ``base`` with (Q)LoRA — the ``method=`` object for ``kind="finetune"``.

    Carries WHICH base to fine-tune plus the LoRA/QLoRA hyperparameters and the
    cost knobs. ``base`` lives here (not as a separate ``finetune()`` argument) so
    one object fully specifies the training for BOTH surfaces — the standalone
    :func:`finetune` and ``Resolver.fit(method=...)`` (which has no other natural
    place to name the base). It is plain, import-light config; the heavy training
    is in :class:`QLoRATrainer`.

    Attributes:
        base: The model to fine-tune — an HF Hub id or local dir.
        r / lora_alpha / lora_dropout / target_modules: Standard LoRA knobs
            (``target_modules=None`` lets peft pick the attention/MLP projections
            for the architecture).
        epochs / learning_rate / max_seq_len / batch_size: SFT training knobs.
        load_in_4bit: Request 4-bit QLoRA quantization. Honored only where CUDA +
            bitsandbytes are available; on CPU/MPS the trainer falls back to a
            non-quantized LoRA fine-tune (so the path still runs locally).
        merge_adapter: Merge the trained adapter into the base weights and return a
            single local-dir ``model_ref`` (heavier, self-contained) instead of a
            ``{base, adapter}`` ref (lighter, needs peft to serve).
        budget_gpu_hours: Advisory GPU-hour budget, surfaced in :meth:`describe`.
        gpu_hourly_usd: $/GPU-hour used to derive the dollar cost from wall-clock
            training seconds (``0.0`` — the honest default for local training —
            yields ``$0``, mirroring the in-process serve path's cost).
    """

    kind = "finetune"

    base: str
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] | None = None
    epochs: int = 3
    learning_rate: float = 2e-4
    max_seq_len: int = 1024
    batch_size: int = 8
    load_in_4bit: bool = True
    merge_adapter: bool = False
    budget_gpu_hours: float | None = None
    gpu_hourly_usd: float = Field(default=0.0, ge=0.0)

    def describe(self) -> str:
        """One-liner: quantization, rank, and the advisory GPU-hour budget."""
        quant = "4-bit QLoRA" if self.load_in_4bit else "LoRA"
        budget = f", ~{self.budget_gpu_hours} GPU-hours" if self.budget_gpu_hours else ""
        return f"fine-tune {self.base} ({quant} r={self.r}{budget})"


@dataclass(frozen=True)
class TrainOutcome:
    """What a :class:`FinetuneTrainer` reports back after training.

    ``adapter_dir`` is where the LoRA adapter (or merged model) was written;
    ``merged`` says which — an adapter to load on top of ``base`` (``False``) or a
    standalone merged model directory (``True``). ``train_seconds`` is the
    wall-clock training time (the GPU-seconds cost fact); ``device`` records where
    it ran (``"cuda"`` / ``"mps"`` / ``"cpu"``) for honest reporting.
    """

    adapter_dir: str
    train_seconds: float
    n_train: int
    merged: bool
    device: str


class FinetuneTrainer(Protocol):
    """The injectable training mechanics seam (kept out of the orchestration).

    A trainer turns ``(base, conversations, method)`` into a saved adapter/model
    under ``output_dir`` and reports a :class:`TrainOutcome`. The default
    :class:`QLoRATrainer` runs peft/trl; tests inject a fake so the orchestration
    is exercised with no GPU and no training stack imported.
    """

    def train(
        self,
        base: str,
        conversations: list[Conversation],
        method: QLoRA,
        output_dir: str,
    ) -> TrainOutcome: ...


@dataclass(frozen=True)
class FinetuneOutcome:
    """The result of :func:`run_finetune`: the served ref plus its cost digest.

    :func:`finetune` returns just the ``model_ref``; ``run_finetune`` returns this
    fuller outcome so a caller (and ``Resolver._fit_finetune``) can read the
    GPU-seconds, derived dollars, and merge status for a :class:`FitReport`.
    """

    model_ref: ModelRef
    base: str
    method: str
    gpu_seconds: float
    dollars: float
    n_train: int
    merged: bool
    device: str


def _yes_no(label: bool) -> str:
    """The assistant target for the binary yes/no match protocol."""
    return "Yes" if label else "No"


def _render_conversation(
    candidate: ERCandidate[Any],
    label: bool,
    *,
    prompt_template: str,
    record_serializer: Any | str | None,
) -> Conversation:
    """Render one labeled pair to a chat conversation matching the served prompt.

    Reuses the LLMMatcher's own ``{left}``/``{right}`` substitution + record
    serializer (lazily imported) so the training text is byte-identical to what the
    served matcher sends, then appends the ``"Yes"``/``"No"`` assistant target.
    Building a throwaway ``LLMMatcher`` (client-free) is the single source of truth
    for the rendering — no drift.
    """
    # Lazy: importing llm_judge pulls litellm ([llm]); keep it out of module load.
    from langres.core.matchers.llm_judge import LLMMatcher

    renderer: LLMMatcher[Any] = LLMMatcher(
        client=object(),
        prompt_template=prompt_template,
        record_serializer=record_serializer,
    )
    user_prompt = renderer._render_prompt(
        renderer._serialize(candidate.left), renderer._serialize(candidate.right)
    )
    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": _yes_no(label)},
    ]


def run_finetune(
    pairs: Sequence[LabeledCandidate],
    method: QLoRA,
    *,
    output_dir: str | Path | None = None,
    trainer: FinetuneTrainer | None = None,
    prompt_template: str | None = None,
    record_serializer: Any | str | None = None,
) -> FinetuneOutcome:
    """Fine-tune ``method.base`` on labeled ``pairs`` and return the served-ref digest.

    Orchestration only — the heavy training is delegated to ``trainer`` (default
    :class:`QLoRATrainer`, lazy peft/trl). Renders each pair to a yes/no chat
    conversation (matching serving), trains, and assembles the
    :class:`FinetuneOutcome` (ref + GPU-seconds + derived dollars + merge status).

    Args:
        pairs: Labeled candidate pairs ``(candidate, is_match)``.
        method: The :class:`QLoRA` spec (base + hyperparameters + cost knobs).
        output_dir: Where to write the adapter/model. Defaults to a fresh temp
            directory (it must outlive this call so the ref can be served).
        trainer: Injected training mechanics; defaults to :class:`QLoRATrainer`.
        prompt_template: The ``{left}``/``{right}`` template; defaults to
            :data:`FINETUNE_YES_NO_PROMPT`. Serve the resulting model with the SAME
            template (``Resolver._fit_finetune`` does).
        record_serializer: How each record renders into the prompt (forwarded to
            the throwaway renderer so training matches serving).

    Raises:
        ValueError: If ``pairs`` is empty (nothing to train on).
    """
    materialized = list(pairs)
    if not materialized:
        raise ValueError("finetune requires at least one labeled pair to train on")

    template = prompt_template if prompt_template is not None else FINETUNE_YES_NO_PROMPT
    conversations = [
        _render_conversation(
            candidate,
            label,
            prompt_template=template,
            record_serializer=record_serializer,
        )
        for candidate, label in materialized
    ]

    if output_dir is None:
        import tempfile

        output_dir = tempfile.mkdtemp(prefix="langres-finetune-")
    output_dir = str(output_dir)

    active_trainer = trainer if trainer is not None else QLoRATrainer()
    outcome = active_trainer.train(method.base, conversations, method, output_dir)

    model_ref = (
        normalize_model_ref(outcome.adapter_dir)
        if outcome.merged
        else ModelRef(base=method.base, adapter=outcome.adapter_dir)
    )
    dollars = outcome.train_seconds / 3600.0 * method.gpu_hourly_usd
    return FinetuneOutcome(
        model_ref=model_ref,
        base=method.base,
        method=method.describe(),
        gpu_seconds=outcome.train_seconds,
        dollars=dollars,
        n_train=outcome.n_train,
        merged=outcome.merged,
        device=outcome.device,
    )


def finetune(
    pairs: Sequence[LabeledCandidate],
    method: QLoRA,
    *,
    output_dir: str | Path | None = None,
    trainer: FinetuneTrainer | None = None,
    prompt_template: str | None = None,
    record_serializer: Any | str | None = None,
) -> ModelRef:
    """Fine-tune ``method.base`` on labeled ``pairs`` → a weightless ``model_ref``.

    The standalone primitive (plan example 3): returns a
    :class:`~langres.core.matchers.model_ref.ModelRef` you serve through
    ``LLMMatcher(model=ref)`` (in-process) or ``vllm serve`` — NOT a Resolver, NOT
    a new component. Use :func:`run_finetune` instead when you also want the cost
    digest (GPU-seconds / dollars / merge status); ``Resolver.fit(...,
    method=QLoRA(...))`` builds the full :class:`FitReport`.

    ``base`` lives on the ``QLoRA`` method (not a separate argument) so one object
    fully specifies the training for both this primitive and the ``Resolver.fit``
    surface. See :func:`run_finetune` for the arguments (identical) and errors.
    """
    return run_finetune(
        pairs,
        method,
        output_dir=output_dir,
        trainer=trainer,
        prompt_template=prompt_template,
        record_serializer=record_serializer,
    ).model_ref


class QLoRATrainer:
    """Default :class:`FinetuneTrainer`: peft LoRA + trl SFT, 4-bit where CUDA allows.

    All heavy imports (``torch`` / ``transformers`` / ``peft`` / ``trl`` /
    ``bitsandbytes``) happen **inside** :meth:`train`, first-use only — importing
    this class costs nothing. 4-bit QLoRA is used only when CUDA + bitsandbytes are
    present; on CPU/MPS it falls back to a non-quantized LoRA fine-tune so the path
    runs locally (the ``[finetune]`` extra scopes bitsandbytes to Linux for this
    reason). Trains completion-only (the ``"Yes"``/``"No"`` assistant span) on the
    base model's chat template, matching serving.
    """

    def train(  # pragma: no cover - real training runs in the test-finetune job / on GPU
        self,
        base: str,
        conversations: list[Conversation],
        method: QLoRA,
        output_dir: str,
    ) -> TrainOutcome:
        # Lazy, first-use only: never import the training stack at module load.
        try:
            import torch
            from datasets import Dataset
            from peft import LoraConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from trl import SFTConfig, SFTTrainer
        except ImportError as exc:
            raise ImportError(
                "QLoRA fine-tuning needs the training stack (peft + trl + "
                "transformers/torch). Install it with `pip install "
                "'langres[semantic,llm,finetune]'` (or `uv add ...`)."
            ) from exc

        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        use_4bit = method.load_in_4bit and device == "cuda"

        tokenizer = AutoTokenizer.from_pretrained(base)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {}
        if use_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(base, **model_kwargs)

        lora = LoraConfig(
            r=method.r,
            lora_alpha=method.lora_alpha,
            lora_dropout=method.lora_dropout,
            target_modules=list(method.target_modules) if method.target_modules else None,
            task_type="CAUSAL_LM",
        )
        dataset = Dataset.from_dict({"messages": conversations})
        sft_config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=method.epochs,
            per_device_train_batch_size=method.batch_size,
            learning_rate=method.learning_rate,
            max_length=method.max_seq_len,
            assistant_only_loss=True,
            report_to=[],
            logging_steps=1,
        )
        sft_trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            peft_config=lora,
            processing_class=tokenizer,
        )

        start = time.perf_counter()
        sft_trainer.train()
        train_seconds = time.perf_counter() - start

        adapter_dir = output_dir
        merged = False
        if method.merge_adapter:
            merged_model = sft_trainer.model.merge_and_unload()
            merged_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            merged = True
        else:
            sft_trainer.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        return TrainOutcome(
            adapter_dir=adapter_dir,
            train_seconds=train_seconds,
            n_train=len(conversations),
            merged=merged,
            device=device,
        )
