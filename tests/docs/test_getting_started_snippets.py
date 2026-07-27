"""Snippet-rot guard for docs/GETTING_STARTED.md.

Runs the guide's opening ``python`` block verbatim, so the page's first example
can never silently drift from the code it documents, and separately runs the
`FuzzyString` demo and checks the outcome the page prints beside it.

SPEND SAFETY: these tests **execute** doc snippets, so the guard on what a
snippet may contain is a money guard, not a style rule.

The guard used to be ``'matcher="string"' in snippet`` -- a *string literal*
standing in for "this cannot spend", back when the front door was
``dedupe(records, matcher=...)`` and the neighbouring value (``"auto"``) would
sniff an API key and bill you. W4 deleted that door. The check became structural
instead of lexical: the snippet must construct something with **no paid model
slot at all**, which is a property of the class rather than a promise about a
string. `FuzzyString` cannot make a paid call for the same reason `2 + 2`
cannot -- there is nothing in it that could.

It is now an **allow list over the langres symbols a snippet calls**, for a
reason worth writing down. The guard used to say "the first block must construct
`FuzzyString`", which silently bundled two different invariants: *this block
cannot spend* and *this block is the FuzzyString demo*. They came apart the
moment the page opened with a different $0 example, and the failure mode of
re-coupling them is the one this repo keeps hitting -- a check that stops
observing the thing it exists to catch. So: the money guard applies to whatever
block gets executed, and the FuzzyString rot guard finds its block by content.

Two properties of the allow list, deliberate:

- It is an **allow** list (rots closed): an unknown langres symbol fails the
  guard rather than passing it. Only names proven incapable of spending are on
  it, each with its reason.
- It requires the called langres symbols to be a **subset**, not merely to
  intersect. The old intersection form would have passed a snippet that
  constructed `FuzzyString` *and* `VectorLLMCascade`.

Non-langres calls are ignored: builtins and the stdlib have no billing path into
this library. `import langres.x` style is rejected outright, because the calls it
enables are attribute calls that the bare-name check cannot see.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_GUIDE = Path(__file__).resolve().parents[2] / "docs" / "GETTING_STARTED.md"

_FIRST_PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)

#: langres symbols a snippet may call under `exec`, and why each cannot bill:
#: `FuzzyString` has no paid model slot (rapidfuzz only); `get_benchmark` reads a
#: labeled dataset that ships inside the wheel. `VectorLLMCascade` is deliberately
#: absent -- it takes an `llm=` and would bill.
_ZERO_SPEND_SYMBOLS = frozenset({"FuzzyString", "get_benchmark"})


def _first_python_snippet() -> str:
    """Return the first fenced ``python`` block in the getting-started guide."""
    text = _GUIDE.read_text(encoding="utf-8")
    match = _FIRST_PYTHON_BLOCK.search(text)
    assert match is not None, "docs/GETTING_STARTED.md has no ```python code block"
    return match.group(1)


def _python_snippets() -> list[str]:
    """Every fenced ``python`` block in the guide, in page order."""
    return _FIRST_PYTHON_BLOCK.findall(_GUIDE.read_text(encoding="utf-8"))


def _snippet_constructing(name: str) -> str:
    """The first snippet that CALLS ``name`` -- located by content, not position.

    A rot guard that asserts one snippet's documented output must find that
    snippet by what it does. Keyed on position, reordering the page would leave
    the test passing against a block it was never written for.
    """
    for snippet in _python_snippets():
        if name in _constructed_names(snippet):
            return snippet
    raise AssertionError(f"docs/GETTING_STARTED.md no longer has a python block calling {name}()")


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


def _langres_names(snippet: str) -> set[str]:
    """Names the snippet imports FROM langres -- the only ones that could spend."""
    return {
        alias.asname or alias.name
        for node in ast.walk(ast.parse(snippet))
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "langres"
        for alias in node.names
    }


def _assert_cannot_spend(snippet: str) -> None:
    """Refuse to execute a doc snippet that could make a paid call."""
    assert not any(
        alias.name.split(".")[0] == "langres"
        for node in ast.walk(ast.parse(snippet))
        if isinstance(node, ast.Import)
        for alias in node.names
    ), (
        "A snippet this test executes uses `import langres...`, whose calls are "
        "attribute calls the bare-name money guard below cannot see. Use "
        "`from langres... import ...` so the guard can read what is called."
    )
    called = _constructed_names(snippet) & _langres_names(snippet)
    assert called, (
        "A snippet this test executes calls no langres symbol at all, so there is "
        "nothing for the money guard to check -- it is documenting something other "
        f"than this library. It calls: {sorted(_constructed_names(snippet)) or 'nothing'}."
    )
    assert called <= _ZERO_SPEND_SYMBOLS, (
        f"This test runs the snippet verbatim and must never make a paid call. It "
        f"calls {sorted(called - _ZERO_SPEND_SYMBOLS)}, which is not on the zero-spend "
        f"allow list ({', '.join(sorted(_ZERO_SPEND_SYMBOLS))}). Add it only if it is "
        "structurally incapable of billing -- and say why."
    )
    assert '"auto"' not in snippet, (
        'matcher="auto" no longer exists (W4 deleted the key-sniffing path). A '
        "snippet using it documents an API that is gone."
    )


def test_getting_started_exists() -> None:
    """The guide the doc ladder points at as 'start here' must exist."""
    assert _GUIDE.is_file()


def test_first_snippet_is_the_zero_spend_lane() -> None:
    """Spend-safety guard: the opening snippet must be structurally unable to bill.

    This test's siblings EXECUTE doc blocks. If a future edit puts a paid model
    first, this fails loudly instead of billing whoever ran the test suite.
    """
    _assert_cannot_spend(_first_python_snippet())


def test_the_money_guard_rejects_a_paid_snippet() -> None:
    """The guard must be able to FAIL -- a check never seen failing is a hypothesis.

    Every real snippet on the page passes, so nothing here would notice if the
    guard stopped looking. This feeds it the snippet it exists to stop.
    """
    paid = (
        "from langres.architectures import VectorLLMCascade\n"
        'VectorLLMCascade(llm="openrouter/openai/gpt-4o-mini").dedupe([])\n'
    )
    with pytest.raises(AssertionError, match="VectorLLMCascade"):
        _assert_cannot_spend(paid)

    # ...and an `import langres` form, whose attribute calls it cannot inspect.
    with pytest.raises(AssertionError, match="attribute calls"):
        _assert_cannot_spend("import langres.architectures\nlangres.architectures.FuzzyString()\n")


def test_first_snippet_runs_verbatim() -> None:
    """The guide's opening example must execute as written."""
    snippet = _first_python_snippet()
    # Spend-safety, re-checked immediately before exec: the tests are independent,
    # and this is one of the two that actually runs the code.
    _assert_cannot_spend(snippet)

    exec(snippet, {})  # noqa: S102 -- deliberate: run the doc snippet as written


def test_the_fuzzystring_snippet_runs_verbatim() -> None:
    """The $0 architecture demo must produce the result documented beside it."""
    snippet = _snippet_constructing("FuzzyString")
    _assert_cannot_spend(snippet)

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
