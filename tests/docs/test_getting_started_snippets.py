"""Snippet-rot guard for docs/GETTING_STARTED.md.

Extracts the FIRST fenced ``python`` block from the getting-started guide and
runs it verbatim, so the page's opening example can never silently drift from
the code it documents.

SPEND SAFETY: this test runs ONLY the keyless ``judge="string"`` snippet, which
makes no network call and costs $0. It hard-refuses to execute a block that is
not pinned to ``judge="string"`` (a keyed/``"auto"`` snippet would make a paid
call), so a future edit that puts a paid snippet first fails loudly instead of
spending money.
"""

from __future__ import annotations

import re
from pathlib import Path

_GUIDE = Path(__file__).resolve().parents[2] / "docs" / "GETTING_STARTED.md"

_FIRST_PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _first_python_snippet() -> str:
    """Return the first fenced ``python`` block in the getting-started guide."""
    text = _GUIDE.read_text(encoding="utf-8")
    match = _FIRST_PYTHON_BLOCK.search(text)
    assert match is not None, "docs/GETTING_STARTED.md has no ```python code block"
    return match.group(1)


def test_getting_started_exists() -> None:
    """The guide the doc ladder points at as 'start here' must exist."""
    assert _GUIDE.is_file()


def test_first_snippet_is_the_keyless_lane() -> None:
    """Spend-safety guard: the first snippet must be the $0 keyless lane."""
    snippet = _first_python_snippet()
    assert 'judge="string"' in snippet, (
        "The first GETTING_STARTED.md python snippet must be the keyless "
        'judge="string" lane -- this test runs it verbatim and must never make '
        "a paid call."
    )
    assert '"auto"' not in snippet, (
        'The first snippet must not use judge="auto" (it would make a paid '
        "call); keep the keyless lane first."
    )


def test_first_snippet_runs_verbatim() -> None:
    """The opening keyless snippet must execute and produce the documented result."""
    snippet = _first_python_snippet()
    assert 'judge="string"' in snippet  # spend-safety, re-checked before exec
    namespace: dict[str, object] = {}
    exec(snippet, namespace)  # noqa: S102 -- deliberate: run the doc snippet as written

    result = namespace["result"]
    # The documented outcome: Acme Corporation / Acme Corp merge; the singleton drops.
    assert result == [{"1", "2"}]
    assert getattr(result, "judge_used") == "string"
    assert getattr(result, "score_type") == "heuristic"
