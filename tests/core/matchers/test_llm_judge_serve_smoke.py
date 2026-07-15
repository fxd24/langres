"""Real in-process serving smoke (PR-E go/no-go slice) — ``@pytest.mark.slow``.

Downloads a TINY real instruct model and runs it **in-process** (no server) through
``LLMMatcher``, asserting the calibrated score comes from first-token yes/no
logprobs — a value in [0, 1], NOT the silent-0.5 parse-miss fallback (which no
longer exists: a parse miss abstains, it never fabricates 0.5). This is the
end-to-end proof that a served/local model resolves through the existing judge
path, and it exercises ``TransformersBackend`` (download → generate → logprobs).

Marked ``slow`` (downloads weights + CPU generation), so it is excluded from the
fast CI gate; run it explicitly: ``uv run pytest -m slow -k serve_smoke``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="requires the [semantic] extra (torch)")
pytest.importorskip("transformers", reason="requires the [semantic] extra (transformers)")

from langres.core.matchers.llm_judge import LLMMatcher
from langres.core.models import CompanySchema, ERCandidate

# Smallest real instruct model with a chat template that puts usable mass on
# yes/no first tokens; ~135M params, CPU-runnable, downloaded from the Hub.
_TINY_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

_YES_NO_PROMPT = (
    "Do these two records describe the same company?\n"
    "Record A: {left}\n"
    "Record B: {right}\n"
    "Answer with a single word: Yes or No."
)


@pytest.mark.slow
def test_inprocess_served_model_produces_a_calibrated_logprob_score() -> None:
    matcher: LLMMatcher[CompanySchema] = LLMMatcher(
        model=_TINY_MODEL,  # HF id, no api_base -> in-process transformers backend
        confidence="logprob",
        response_parser="binary_yes_no",
        prompt_template=_YES_NO_PROMPT,
    )
    # It routed to the in-process backend, not litellm.
    assert matcher._backend_kind == "transformers"

    candidate = ERCandidate(
        left=CompanySchema(id="c1", name="Acme Corporation"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="smoke",
    )
    judgement = next(iter(matcher.forward([candidate])))

    # The score is a real first-token yes/no credence, in [0, 1].
    assert judgement.score is not None, "no p_yes computed — logprobs path did not run"
    assert 0.0 <= judgement.score <= 1.0
    assert judgement.confidence_source == "logprob"
    # NOT the silent-0.5 parse-miss fallback: a real p_yes was found + promoted.
    assert judgement.provenance.get("parse_error") is None
    assert judgement.provenance.get("p_yes") is not None
