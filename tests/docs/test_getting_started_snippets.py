"""Snippet-rot guard for docs/GETTING_STARTED.md.

Extracts the FIRST fenced ``python`` block from the getting-started guide and
runs it verbatim, so the page's opening example can never silently drift from
the code it documents.

SPEND SAFETY: this test **executes** the first snippet, so the guard on what that
snippet may contain is a money guard, not a style rule.

The guard used to be ``'matcher="string"' in snippet`` -- a *string literal*
standing in for "this cannot spend", back when the front door was
``dedupe(records, matcher=...)`` and the neighbouring value (``"auto"``) would
sniff an API key and bill you. W4 deleted that door. The check is now structural
instead of lexical: the snippet must construct one of the architectures that has
**no paid model slot at all**, which is a property of the class rather than a
promise about a string. `FuzzyString` cannot make a paid call for the same reason
`2 + 2` cannot -- there is nothing in it that could.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_GUIDE = Path(__file__).resolve().parents[2] / "docs" / "GETTING_STARTED.md"

_FIRST_PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)

#: Architectures with no paid backbone slot -- structurally incapable of spending.
#: `VectorLLMCascade` is deliberately absent: it takes an `llm=` and would bill.
_ZERO_SPEND_ARCHITECTURES = frozenset({"FuzzyString"})


def _first_python_snippet() -> str:
    """Return the first fenced ``python`` block in the getting-started guide."""
    text = _GUIDE.read_text(encoding="utf-8")
    match = _FIRST_PYTHON_BLOCK.search(text)
    assert match is not None, "docs/GETTING_STARTED.md has no ```python code block"
    return match.group(1)


def _constructed_names(snippet: str) -> set[str]:
    """Every bare name the snippet CALLS -- read from the AST, not by grepping.

    A mention in a comment or a string is not a construction; only the parse tree
    can tell the difference, and this decides whether we execute the block.
    """
    return {
        node.func.id
        for node in ast.walk(ast.parse(snippet))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_getting_started_exists() -> None:
    """The guide the doc ladder points at as 'start here' must exist."""
    assert _GUIDE.is_file()


def test_first_snippet_is_the_zero_spend_lane() -> None:
    """Spend-safety guard: the first snippet must construct a $0 architecture.

    This test's sibling EXECUTES the block. If a future edit puts a paid model
    first, this fails loudly instead of billing whoever ran the test suite.
    """
    snippet = _first_python_snippet()
    constructed = _constructed_names(snippet)

    assert constructed & _ZERO_SPEND_ARCHITECTURES, (
        "The first GETTING_STARTED.md python snippet must construct a zero-spend "
        f"architecture ({', '.join(sorted(_ZERO_SPEND_ARCHITECTURES))}) -- this test "
        f"runs it verbatim and must never make a paid call. It constructs: "
        f"{sorted(constructed) or 'nothing'}."
    )
    assert "VectorLLMCascade" not in constructed, (
        "The first snippet must not construct VectorLLMCascade -- it takes an llm= "
        "backbone and would make PAID calls when this test executes the block. Keep "
        "the $0 architecture first and show the paid one later on the page."
    )
    assert '"auto"' not in snippet, (
        'matcher="auto" no longer exists (W4 deleted the key-sniffing path). A '
        "snippet using it documents an API that is gone."
    )


def test_first_snippet_runs_verbatim() -> None:
    """The opening zero-spend snippet must execute and produce the documented result."""
    snippet = _first_python_snippet()
    # Spend-safety, re-checked immediately before exec: the two tests are
    # independent, and this one is the one that actually runs the code.
    assert _constructed_names(snippet) & _ZERO_SPEND_ARCHITECTURES

    namespace: dict[str, object] = {}
    exec(snippet, namespace)  # noqa: S102 -- deliberate: run the doc snippet as written

    result = namespace["result"]
    # The documented outcome: Acme Corporation / Acme Corp merge; the singleton drops.
    assert result == [{"1", "2"}]
    # The result names the model that produced it (was `judge_used == "string"`,
    # a preset name; it is now the architecture class the reader constructed).
    assert getattr(result, "architecture") == "FuzzyString"
    assert getattr(result, "score_type") == "heuristic"
    # Nothing with weights ran, and the result says so rather than leaving it blank.
    assert getattr(result, "backbone") is None
