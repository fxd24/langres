"""S5: LLMMatcher run correlation -- litellm ``metadata`` passthrough.

The judge stamps the active ``capture_run`` attempt id (plus pair identity and
decision step) into litellm's first-class ``metadata`` param, so a Langfuse/OTel
trace joins the ``RunRecord`` and ``JudgementLog`` on ``langres_attempt_id``.
This holds on **both** paths -- the sync ``forward()`` (``completion``) and the
async ``forward_async()`` (``acompletion``, via the retry helper) -- which share
``_run_correlation_metadata`` so they cannot drift.

Two invariants are load-bearing and locked here for each path:
1. ``metadata`` is added ONLY on the litellm path (``client is litellm``) and
   ONLY inside an open run (``current_run`` set).
2. With no run open, the completion call is **byte-identical** to before
   -- no ``metadata`` kwarg at all. This is the key regression guard.

The client is mocked throughout -- no network, no real spend.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

# The litellm-path gate is identity (``client is litellm``), so these tests need
# the real litellm module. It lives in the optional ``[llm]`` extra and is not
# guaranteed installed (the bare-core CI job) -- skip the whole file if absent.
litellm = pytest.importorskip("litellm")

from langres.core.models import CompanySchema, ERCandidate  # noqa: E402
from langres.core.matchers.llm_judge import LLMMatcher  # noqa: E402
from langres.core.runs import RunContext, capture_run  # noqa: E402


def _pair() -> ERCandidate[CompanySchema]:
    return ERCandidate(
        left=CompanySchema(id="c1", name="Acme Corporation"),
        right=CompanySchema(id="c2", name="Acme Corp"),
        blocker_name="test",
    )


def _response(content: str = "MATCH\nScore: 0.9\nReasoning: same company") -> Mock:
    """A minimal completion response (mirrors test_llm_judge.py's shape)."""
    resp = Mock()
    resp.choices = [Mock()]
    resp.choices[0].message.content = content
    resp.usage = Mock()
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 50
    return resp


def _context() -> RunContext:
    return RunContext(experiment="s5-llm-correlation", dataset_name="fake")


def test_metadata_stamped_on_litellm_path_inside_capture_run(mocker: Any) -> None:
    """Inside a run on the litellm path, ``completion()`` receives ``metadata``
    with the attempt id, both pair ids, and the decision step."""
    mocker.patch.object(litellm, "completion", return_value=_response())
    module = LLMMatcher(client=litellm, model="gpt-4o-mini")

    with capture_run(_context(), store=None) as handle:
        list(module.forward([_pair()]))

    metadata = litellm.completion.call_args.kwargs["metadata"]
    assert metadata == {
        "langres_attempt_id": handle.attempt_id,
        "left_id": "c1",
        "right_id": "c2",
        "decision_step": "llm_judgment",
    }


def test_no_metadata_kwarg_when_no_capture_run(mocker: Any) -> None:
    """Byte-identical invariant: with no open run, ``completion()`` is called
    exactly as before -- NO ``metadata`` kwarg (key regression guard)."""
    mocker.patch.object(litellm, "completion", return_value=_response())
    module = LLMMatcher(client=litellm, model="gpt-4o-mini")

    list(module.forward([_pair()]))  # not inside any capture_run

    assert "metadata" not in litellm.completion.call_args.kwargs


def test_direct_client_never_receives_metadata_even_inside_capture_run() -> None:
    """A user-supplied direct (non-litellm) client -- one that would 400 on an
    unknown ``metadata`` kwarg -- never receives it, even inside a run."""
    client = Mock()  # NOT the litellm module
    client.completion.return_value = _response()
    module = LLMMatcher(client=client, model="gpt-4o-mini")

    with capture_run(_context(), store=None):
        list(module.forward([_pair()]))

    assert "metadata" not in client.completion.call_args.kwargs


# --- Async path (forward_async) -- same three invariants on ``acompletion``. ---


@pytest.mark.asyncio
async def test_async_metadata_stamped_on_litellm_path_inside_capture_run(mocker: Any) -> None:
    """Async mirror: inside a run on the litellm path, ``acompletion()`` receives
    ``metadata`` with the attempt id, both pair ids, and the async decision step."""
    mocker.patch.object(litellm, "acompletion", AsyncMock(return_value=_response()))
    module = LLMMatcher(client=litellm, model="gpt-4o-mini")

    with capture_run(_context(), store=None) as handle:
        await module.forward_async([_pair()])

    metadata = litellm.acompletion.call_args.kwargs["metadata"]
    assert metadata == {
        "langres_attempt_id": handle.attempt_id,
        "left_id": "c1",
        "right_id": "c2",
        "decision_step": "llm_judgment_async",
    }


@pytest.mark.asyncio
async def test_async_no_metadata_kwarg_when_no_capture_run(mocker: Any) -> None:
    """Async byte-identical invariant: with no open run, ``acompletion()`` is
    called with NO ``metadata`` kwarg (the async regression guard)."""
    mocker.patch.object(litellm, "acompletion", AsyncMock(return_value=_response()))
    module = LLMMatcher(client=litellm, model="gpt-4o-mini")

    await module.forward_async([_pair()])  # not inside any capture_run

    assert "metadata" not in litellm.acompletion.call_args.kwargs


@pytest.mark.asyncio
async def test_async_direct_client_never_receives_metadata_even_inside_capture_run() -> None:
    """A user-supplied direct (non-litellm) client never receives ``metadata`` on
    the async path either, even inside a run."""
    client = Mock()  # NOT the litellm module
    client.acompletion = AsyncMock(return_value=_response())
    module = LLMMatcher(client=client, model="gpt-4o-mini")

    with capture_run(_context(), store=None):
        await module.forward_async([_pair()])

    assert "metadata" not in client.acompletion.call_args.kwargs
