"""Unit tests for ``langres.core.finetune`` -- the standalone QLoRA primitive.

These lock the ORCHESTRATION (prompt/target rendering, ``model_ref`` + cost
assembly, the :class:`QLoRA` method object) with an *injected fake trainer*, so
they run with no GPU and never import peft/trl/torch. The real peft/trl training
path (:class:`QLoRATrainer`) is exercised by the CPU dry-run under the
``finetune`` marker (the ``test-finetune`` CI job) at the bottom of this file,
never in the fast suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.finetune import (
    FINETUNE_YES_NO_PROMPT,
    Conversation,
    FinetuneOutcome,
    QLoRA,
    TrainOutcome,
    finetune,
    run_finetune,
)
from langres.core.matchers.model_ref import ModelRef
from langres.core.models import CompanySchema, ERCandidate

RECORDS = [
    {"id": "a", "name": "Acme Corp"},
    {"id": "b", "name": "Acme Corporation"},
    {"id": "c", "name": "Beta Inc"},
]


def _labeled_pairs() -> list[tuple[Any, bool]]:
    """Real ``(ERCandidate, is_match)`` pairs (as ``run_finetune`` receives them).

    Blocks the 3 records into all-pairs candidates and labels only the ``{a, b}``
    pair a match, so the rendered assistant targets carry both a ``"Yes"`` and a
    ``"No"``.
    """
    candidates = list(AllPairsBlocker(schema=CompanySchema).stream(RECORDS))

    def is_match(cand: Any) -> bool:
        return {str(cand.left.id), str(cand.right.id)} == {"a", "b"}

    return [(cand, is_match(cand)) for cand in candidates]


class _FakeTrainer:
    """A :class:`~langres.core.finetune.FinetuneTrainer` that records its call.

    Writes a stub adapter dir (so the produced ref points at a real directory) and
    returns a canned :class:`TrainOutcome` -- no GPU, no peft/trl. Captures the
    ``(base, conversations, method, output_dir)`` it was handed so tests can assert
    what the orchestration rendered and forwarded.
    """

    def __init__(
        self, *, train_seconds: float = 12.0, merged: bool = False, device: str = "cpu"
    ) -> None:
        self.train_seconds = train_seconds
        self.merged = merged
        self.device = device
        self.calls: list[dict[str, Any]] = []

    def train(
        self,
        base: str,
        conversations: list[Conversation],
        method: QLoRA,
        output_dir: str,
    ) -> TrainOutcome:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "adapter_config.json").write_text("{}")
        self.calls.append(
            {
                "base": base,
                "conversations": conversations,
                "method": method,
                "output_dir": output_dir,
            }
        )
        return TrainOutcome(
            adapter_dir=output_dir,
            train_seconds=self.train_seconds,
            n_train=len(conversations),
            merged=self.merged,
            device=self.device,
        )


# --- QLoRA method object: kind identity, describe(), config surface -----------


def test_qlora_kind_is_finetune_classvar() -> None:
    """``QLoRA.kind`` is the ``"finetune"`` dispatch identity, not a serialized field."""
    assert QLoRA.kind == "finetune"
    assert "kind" not in QLoRA.model_fields
    assert "base" in QLoRA.model_fields


def test_qlora_describe_reports_quant_rank_and_budget() -> None:
    """``describe()`` is the 'what + cost' one-liner: base, quantization, rank, budget."""
    d = QLoRA(base="tiny/model", r=8, budget_gpu_hours=2.5).describe()
    assert "tiny/model" in d
    assert "4-bit QLoRA" in d
    assert "r=8" in d
    assert "2.5 GPU-hours" in d


def test_qlora_describe_drops_quant_and_budget_when_off() -> None:
    """No 4-bit / no budget -> plain "LoRA" with no GPU-hours clause."""
    d = QLoRA(base="tiny/model", load_in_4bit=False).describe()
    assert "LoRA" in d and "4-bit" not in d
    assert "GPU-hours" not in d


def test_qlora_gpu_hourly_usd_rejects_negative() -> None:
    """The $/GPU-hour cost knob is guarded non-negative (Field ge=0)."""
    with pytest.raises(ValueError):
        QLoRA(base="tiny/model", gpu_hourly_usd=-1.0)


# --- run_finetune: rendering matches serving ---------------------------------


def test_conversations_render_user_prompt_plus_yes_no_target() -> None:
    """Each pair -> ``[{user: <finetune prompt>}, {assistant: "Yes"|"No"}]``.

    The user turn carries the yes/no prompt + the serialized records; the
    assistant turn is the binary target, so training text == what the served
    matcher sees.
    """
    pairs = _labeled_pairs()
    trainer = _FakeTrainer()

    run_finetune(pairs, QLoRA(base="tiny/model"), trainer=trainer)

    prompt_header = FINETUNE_YES_NO_PROMPT.split("{left}")[0]  # the default template's lead-in
    conversations = trainer.calls[0]["conversations"]
    assert len(conversations) == len(pairs)
    for (cand, label), convo in zip(pairs, conversations, strict=True):
        assert [turn["role"] for turn in convo] == ["user", "assistant"]
        assert convo[1]["content"] == ("Yes" if label else "No")
        assert prompt_header in convo[0]["content"]  # rendered from the default prompt
    # The matched pair's records are actually rendered into the user turn.
    match_convo = next(
        convo for (cand, label), convo in zip(pairs, conversations, strict=True) if label
    )
    assert "Acme Corp" in match_convo[0]["content"]
    assert match_convo[1]["content"] == "Yes"


def test_prompt_template_override_flows_into_rendering() -> None:
    """A custom ``{left}``/``{right}`` template is what gets trained on (train==serve seam)."""
    pairs = _labeled_pairs()
    trainer = _FakeTrainer()

    run_finetune(
        pairs,
        QLoRA(base="tiny/model"),
        trainer=trainer,
        prompt_template="CUSTOM {left} <=> {right}",
    )

    user_turn = trainer.calls[0]["conversations"][0][0]["content"]
    assert user_turn.startswith("CUSTOM ")
    assert FINETUNE_YES_NO_PROMPT.split("{left}")[0] not in user_turn  # not the default


# --- run_finetune: model_ref shape + cost digest -----------------------------


def test_run_finetune_unmerged_returns_base_plus_adapter_ref() -> None:
    """Unmerged QLoRA -> a ``ModelRef(base, adapter=<dir>)`` served without merging."""
    pairs = _labeled_pairs()
    trainer = _FakeTrainer(merged=False, device="cpu")

    outcome = run_finetune(pairs, QLoRA(base="tiny/model"), trainer=trainer, output_dir="/tmp/x")

    assert isinstance(outcome, FinetuneOutcome)
    assert outcome.model_ref == ModelRef(base="tiny/model", adapter="/tmp/x")
    assert outcome.base == "tiny/model"
    assert outcome.merged is False
    assert outcome.device == "cpu"
    assert outcome.n_train == len(pairs)
    assert "tiny/model" in outcome.method  # describe() string


def test_run_finetune_merged_returns_self_contained_ref() -> None:
    """``merge_adapter`` -> a single local-dir ref (``adapter is None``)."""
    pairs = _labeled_pairs()
    trainer = _FakeTrainer(merged=True)

    outcome = run_finetune(
        pairs, QLoRA(base="tiny/model", merge_adapter=True), trainer=trainer, output_dir="/tmp/m"
    )

    assert outcome.model_ref == ModelRef(base="/tmp/m", adapter=None)
    assert outcome.merged is True


def test_gpu_seconds_and_dollars_derive_from_train_seconds() -> None:
    """GPU-seconds are the wall-clock fact; dollars = seconds/3600 * $/GPU-hour."""
    pairs = _labeled_pairs()
    trainer = _FakeTrainer(train_seconds=3600.0)

    outcome = run_finetune(pairs, QLoRA(base="tiny/model", gpu_hourly_usd=2.0), trainer=trainer)

    assert outcome.gpu_seconds == 3600.0
    assert outcome.dollars == pytest.approx(2.0)


def test_default_gpu_hourly_usd_is_free() -> None:
    """The honest local-training default ($0/GPU-hour) yields $0 regardless of time."""
    pairs = _labeled_pairs()
    trainer = _FakeTrainer(train_seconds=999.0)

    outcome = run_finetune(pairs, QLoRA(base="tiny/model"), trainer=trainer)

    assert outcome.dollars == 0.0
    assert outcome.gpu_seconds == 999.0


# --- run_finetune: output dir + empty-input guard ----------------------------


def test_output_dir_is_respected() -> None:
    """A caller-supplied ``output_dir`` is where the adapter lands (and the ref points)."""
    pairs = _labeled_pairs()
    trainer = _FakeTrainer()

    outcome = run_finetune(pairs, QLoRA(base="tiny/model"), trainer=trainer, output_dir="/tmp/here")

    assert trainer.calls[0]["output_dir"] == "/tmp/here"
    assert outcome.model_ref.adapter == "/tmp/here"


def test_default_output_dir_is_a_fresh_temp_dir() -> None:
    """Without ``output_dir`` a temp dir is created (it must outlive the call to serve)."""
    pairs = _labeled_pairs()
    trainer = _FakeTrainer()

    outcome = run_finetune(pairs, QLoRA(base="tiny/model"), trainer=trainer)

    adapter = outcome.model_ref.adapter
    assert adapter is not None
    assert "langres-finetune-" in adapter
    assert Path(adapter).is_dir()


def test_run_finetune_rejects_empty_pairs() -> None:
    """Nothing to train on is a caller error, not a silent empty adapter."""
    with pytest.raises(ValueError, match="at least one labeled pair"):
        run_finetune([], QLoRA(base="tiny/model"), trainer=_FakeTrainer())


# --- finetune(): the standalone primitive returns just the ref ----------------


def test_finetune_returns_the_model_ref() -> None:
    """``finetune()`` is ``run_finetune().model_ref`` -- the weightless served handle."""
    pairs = _labeled_pairs()
    trainer = _FakeTrainer()

    ref = finetune(pairs, QLoRA(base="tiny/model"), trainer=trainer, output_dir="/tmp/f")

    assert ref == ModelRef(base="tiny/model", adapter="/tmp/f")


# --- The real peft/trl path: CPU dry-run (test-finetune job / --all-extras) ---


@pytest.mark.finetune
def test_qlora_trainer_cpu_dry_run(tmp_path: Path) -> None:
    """Real peft+trl LoRA on a tiny model, CPU (no 4-bit) -- the ``[finetune]`` smoke.

    Runs the DEFAULT :class:`QLoRATrainer` end-to-end: imports peft/trl/transformers,
    fine-tunes a tiny instruct LM on a handful of yes/no pairs on CPU
    (``load_in_4bit`` is ignored off CUDA), and asserts it produced a servable
    adapter dir plus honest GPU-seconds. Marked ``finetune`` so it runs only in the
    dedicated ``test-finetune`` CI job / locally with ``--all-extras`` -- never the
    fast suite.
    """
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    pytest.importorskip("torch")

    pairs = _labeled_pairs()
    method = QLoRA(
        base="HuggingFaceTB/SmolLM2-135M-Instruct",
        epochs=1,
        batch_size=1,
        max_seq_len=256,
    )

    outcome = run_finetune(pairs, method, output_dir=tmp_path)

    assert outcome.n_train == len(pairs)
    assert outcome.gpu_seconds > 0.0
    assert outcome.device in {"cpu", "mps", "cuda"}
    assert outcome.model_ref.base == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert outcome.model_ref.adapter is not None
    assert Path(outcome.model_ref.adapter).is_dir()


# --- The full go/no-go: train -> save -> reload -> serve in-process -> evaluate ---

_MATCH_NAMES = [
    ("Acme Corp", "Acme Corporation"),
    ("Globex Inc", "Globex Incorporated"),
    ("Initech LLC", "Initech Limited"),
    ("Umbrella Co", "Umbrella Company"),
    ("Soylent Corp", "Soylent Corporation"),
    ("Stark Industries", "Stark Ind."),
    ("Wayne Enterprises", "Wayne Enterprise"),
    ("Wonka Ltd", "Wonka Limited"),
    ("Cyberdyne Systems", "Cyberdyne Sys"),
    ("Tyrell Corp", "Tyrell Corporation"),
]
_NON_NAMES = [
    ("Acme Corp", "Globex Inc"),
    ("Initech LLC", "Umbrella Co"),
    ("Soylent Corp", "Stark Industries"),
    ("Wayne Enterprises", "Wonka Ltd"),
    ("Cyberdyne Systems", "Tyrell Corp"),
    ("Acme Corp", "Umbrella Co"),
    ("Globex Inc", "Stark Industries"),
    ("Initech LLC", "Wonka Ltd"),
    ("Soylent Corp", "Tyrell Corp"),
    ("Wayne Enterprises", "Cyberdyne Systems"),
]


def _balanced_pairs() -> list[tuple[ERCandidate[CompanySchema], bool]]:
    """10 matching name-variant pairs + 10 non-matching pairs (balanced overfit set)."""
    pairs: list[tuple[ERCandidate[CompanySchema], bool]] = []
    for i, (left, right) in enumerate(_MATCH_NAMES):
        c = ERCandidate(
            left=CompanySchema(id=f"m{i}L", name=left),
            right=CompanySchema(id=f"m{i}R", name=right),
            blocker_name="test",
        )
        pairs.append((c, True))
    for i, (left, right) in enumerate(_NON_NAMES):
        c = ERCandidate(
            left=CompanySchema(id=f"n{i}L", name=left),
            right=CompanySchema(id=f"n{i}R", name=right),
            blocker_name="test",
        )
        pairs.append((c, False))
    return pairs


def _serve_scores(model_cfg: Any, pairs: list[tuple[ERCandidate[CompanySchema], bool]]) -> Any:
    """Serve ``model_cfg`` in-process on the yes/no prompt; return (F1, p_yes separation)."""
    from langres.core.matchers.llm_judge import LLMMatcher
    from langres.core.metrics import classify_pairs

    matcher: LLMMatcher[Any] = LLMMatcher(
        model=model_cfg,
        confidence="logprob",
        response_parser="binary_yes_no",
        prompt_template=FINETUNE_YES_NO_PROMPT,
    )
    judgements = list(matcher.forward(iter([c for c, _ in pairs])))
    gold = {frozenset({str(c.left.id), str(c.right.id)}) for c, y in pairs if y}
    metrics = classify_pairs(judgements, gold, 0.5)
    pos = [j.provenance["p_yes"] for (_, y), j in zip(pairs, judgements, strict=True) if y]
    neg = [j.provenance["p_yes"] for (_, y), j in zip(pairs, judgements, strict=True) if not y]
    separation = sum(pos) / len(pos) - sum(neg) / len(neg)
    return metrics.f1, separation


@pytest.mark.slow
@pytest.mark.finetune
def test_finetune_overfit_train_serve_evaluate() -> None:
    """The whole loop learns: real QLoRA train -> reload -> in-process serve beats the base.

    The go/no-go for the fine-tune surface. Fine-tunes SmolLM2-135M (real peft LoRA +
    trl completion-only SFT) on 20 balanced yes/no pairs, reloads the produced
    base+adapter ``model_ref``, serves it IN-PROCESS with the same finetune prompt +
    logprob probe, and asserts the fine-tuned model scores the overfit set *and*
    separates match/non-match ``p_yes`` **better than the untrained base** -- i.e.
    training actually moved the served decisions (not just that the plumbing runs).
    Marked ``slow`` + ``finetune`` so it runs only locally / on demand (real training
    + model download), never in the fast suite or the CPU-dry-run CI job.
    """
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    pytest.importorskip("torch")
    from langres.core.matchers.model_ref import to_config

    pairs = _balanced_pairs()
    base_f1, base_sep = _serve_scores("HuggingFaceTB/SmolLM2-135M-Instruct", pairs)

    method = QLoRA(base="HuggingFaceTB/SmolLM2-135M-Instruct", epochs=8, batch_size=4)
    outcome = run_finetune(pairs, method)
    assert outcome.gpu_seconds > 0.0
    assert outcome.model_ref.adapter is not None

    tuned_f1, tuned_sep = _serve_scores(to_config(outcome.model_ref), pairs)

    # Training must both raise overfit F1 and widen match/non-match p_yes separation
    # over the untrained base (the base collapses to all-positive at threshold 0.5).
    assert tuned_f1 > base_f1
    assert tuned_sep > base_sep
    assert tuned_f1 >= 0.8  # overfits the 20-pair set (observed ~0.93)
