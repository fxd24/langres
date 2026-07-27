"""The clean-install documentation gate must be able to go red.

`tools/doc_snippets.py` builds the real wheel, installs it into an extras-free
virtualenv, and executes the user-facing docs' python blocks with that
interpreter. This file is the evidence that it *works* -- that an injected
snippet needing `[semantic]` genuinely turns the job red, and that every
"exemption" path fails closed rather than open.

That distinction is the point. This repo's rules name "gates decoupled from what
they check" as a recurring failure -- a CI step named "95% coverage gate" that
enforced 90, a cancelled run read as green, a review bot reporting `pass` in 5s
without fetching the diff. **A check nobody has watched fail is a hypothesis.**
So the proof tests below do not assert that the gate *would* catch a regression;
they inject one and watch it caught, against a real wheel in a real empty venv.

`tools` is on the path via `[tool.pytest.ini_options].pythonpath`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_snippets import (
    DOC_PATHS,
    DirectiveError,
    Snippet,
    _snippet_env,
    assert_gate_is_observing,
    build_clean_install,
    check_examples_reference,
    collect,
    declared_extras,
    extract,
    run,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A `[semantic]`-only import that fails at IMPORT time on a bare install
#: (`langres/core/indexes/vector_index.py` does `import faiss` at module scope).
#: Measured, not assumed -- see `test_semantic_only_snippet_turns_the_gate_red`.
SEMANTIC_ONLY_SNIPPET = "from langres.core.indexes import FAISSIndex\nprint(FAISSIndex)\n"


def _write(tmp_path: Path, text: str) -> Path:
    doc = tmp_path / "doc.md"
    doc.write_text(text, encoding="utf-8")
    return doc


def _extract(tmp_path: Path, text: str) -> list[Snippet]:
    return extract(_write(tmp_path, text), doc="doc.md", extras=declared_extras(REPO_ROOT))


# ---------------------------------------------------------------------------
# The exemption vocabulary fails CLOSED: every ambiguity is an error, never a
# silent skip. A typo'd exemption that reads as an exemption is the whole bug.
# ---------------------------------------------------------------------------


def test_unmarked_python_block_is_executed_by_default(tmp_path: Path) -> None:
    """The rots-closed property: nobody has to opt a new snippet IN to checking."""
    (snippet,) = _extract(tmp_path, "```python\nx = 1\n```\n")
    assert snippet.exemption is None
    assert snippet.language == "python"


def test_directive_exempts_the_block_that_follows_it(tmp_path: Path) -> None:
    (snippet,) = _extract(
        tmp_path, "<!-- docs-gate: requires-extra=semantic -->\n```python\nimport torch\n```\n"
    )
    assert snippet.exemption == "requires-extra=semantic"


def test_unknown_directive_token_is_an_error(tmp_path: Path) -> None:
    """A misspelled reason must not read as a valid exemption."""
    with pytest.raises(DirectiveError, match="unknown docs-gate directive"):
        _extract(tmp_path, "<!-- docs-gate: requires-repoo -->\n```python\nx = 1\n```\n")


def test_requires_extra_must_name_an_extra_pyproject_declares(tmp_path: Path) -> None:
    """`requires-extra=` is validated against the real extras, not free text."""
    with pytest.raises(DirectiveError, match="unknown extra 'quantum'"):
        _extract(tmp_path, "<!-- docs-gate: requires-extra=quantum -->\n```python\nx = 1\n```\n")


def test_directive_attached_to_nothing_is_an_error(tmp_path: Path) -> None:
    """A stray exemption silently exempting nothing is a decoupled gate."""
    with pytest.raises(DirectiveError, match="not followed by a fenced code block"):
        _extract(
            tmp_path, "<!-- docs-gate: requires-repo -->\n\nSome prose.\n\n```python\nx=1\n```\n"
        )


def test_directive_at_end_of_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(DirectiveError, match="end of the file"):
        _extract(tmp_path, "```python\nx = 1\n```\n\n<!-- docs-gate: requires-repo -->\n")


def test_unrecognized_fence_language_is_an_error(tmp_path: Path) -> None:
    """An unclassified language must not pass as 'not executed, therefore fine'."""
    with pytest.raises(DirectiveError, match="unrecognized fence language 'rust'"):
        _extract(tmp_path, "```rust\nfn main() {}\n```\n")


def test_blank_lines_between_directive_and_fence_are_allowed(tmp_path: Path) -> None:
    (snippet,) = _extract(tmp_path, "<!-- docs-gate: illustrative -->\n\n```python\nx = 1\n```\n")
    assert snippet.exemption == "illustrative"


def test_reported_line_number_points_at_the_opening_fence(tmp_path: Path) -> None:
    """The report is only actionable if the location is right."""
    (snippet,) = _extract(tmp_path, "intro\n\nmore\n\n```python\nx = 1\n```\n")
    assert snippet.line == 5
    assert snippet.location == "doc.md:5"


# ---------------------------------------------------------------------------
# The `examples/` rule -- the shape of the defect that motivated this stream.
# ---------------------------------------------------------------------------


def _snippet(code: str, *, exemption: str | None = None) -> Snippet:
    return Snippet(doc="d.md", line=1, language="bash", code=code, exemption=exemption)


def test_examples_reference_fails_when_the_wheel_omits_examples() -> None:
    detail = check_examples_reference(
        _snippet("uv run python examples/research/first_experiment.py\n"),
        examples_shipped=False,
    )
    assert detail is not None
    assert "examples/research/first_experiment.py" in detail


def test_examples_reference_passes_once_the_doc_declares_requires_repo() -> None:
    assert (
        check_examples_reference(
            _snippet("python examples/foo.py\n", exemption="requires-repo"),
            examples_shipped=False,
        )
        is None
    )


def test_examples_rule_switches_itself_off_if_examples_ever_ship() -> None:
    """The rule is observed from the built wheel, not asserted as folklore."""
    assert (
        check_examples_reference(_snippet("python examples/foo.py\n"), examples_shipped=True)
        is None
    )


@pytest.mark.parametrize(
    "exemption", ["requires-repo", "illustrative", "requires-network", "requires-extra=semantic"]
)
def test_any_declared_exemption_satisfies_the_examples_rule(exemption: str) -> None:
    """The rule polices one claim: "a bare install can run this".

    A block that declares *any* exemption has stopped making that claim. Demanding
    `requires-repo` specifically forced a mislabel -- an `illustrative` fragment
    that merely mentions an example path could only be silenced by tagging it
    `requires-repo`, which says something false about it.
    """
    assert (
        check_examples_reference(
            _snippet("python examples/foo.py\n", exemption=exemption), examples_shipped=False
        )
        is None
    )


# ---------------------------------------------------------------------------
# The gate must not be able to pass by observing nothing.
# ---------------------------------------------------------------------------


def test_gate_refuses_to_pass_when_no_python_block_is_executable(tmp_path: Path) -> None:
    """0 executable snippets is an ERROR, not a green run.

    The failure shape this blocks: a change to _FENCE_RE / DOC_PATHS /
    EXECUTED_LANGUAGES stops matching, everything is skipped, the job prints
    "0 failed" and goes green forever -- the same silent-match-nothing failure
    the `[tool.hatch.build]` path literals carry.
    """
    only_exempted = _extract(
        tmp_path,
        "<!-- docs-gate: illustrative -->\n```python\nx = 1\n```\n\n```bash\nls\n```\n",
    )
    with pytest.raises(DirectiveError, match="0 of them are executable python"):
        assert_gate_is_observing(only_exempted)


def test_gate_refuses_to_pass_on_an_empty_snippet_set() -> None:
    with pytest.raises(DirectiveError, match="passes by finding nothing"):
        assert_gate_is_observing([])


def test_a_document_matching_no_fenced_block_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A doc the extractor cannot see is a doc it cannot check."""
    (tmp_path / "README.md").write_text("no code here at all\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.setattr("doc_snippets.DOC_PATHS", ("README.md",))
    with pytest.raises(DirectiveError, match="contributed 0 fenced code blocks"):
        collect(tmp_path)


def test_the_real_docs_still_have_executable_python() -> None:
    """The guard above, applied to the actual repo -- this gate is observing."""
    assert_gate_is_observing(collect(REPO_ROOT))


# ---------------------------------------------------------------------------
# The gate must not be defeatable by the developer's own environment.
# ---------------------------------------------------------------------------


def test_snippet_env_drops_pythonpath(monkeypatch: pytest.MonkeyPatch) -> None:
    """pytest puts `src` on PYTHONPATH; inheriting it would let the gate lie.

    A snippet that imported langres from the source tree would pass while the
    wheel was broken -- exactly the failure a clean-install gate exists to catch.
    """
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT / "src"))
    assert "PYTHONPATH" not in _snippet_env()


def test_snippet_env_drops_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """A documentation snippet must never be able to bill anyone."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-should-not-reach-a-snippet")
    assert "OPENROUTER_API_KEY" not in _snippet_env()


def test_real_docs_directives_all_parse() -> None:
    """Every `docs-gate` directive in the real docs is well-formed.

    Cheap, needs no wheel, and runs on every PR -- so a typo'd exemption is named
    here rather than surfacing as a mysteriously skipped snippet later.
    """
    snippets = collect(REPO_ROOT)
    assert snippets, f"no fenced blocks found in {DOC_PATHS}"


# ---------------------------------------------------------------------------
# PROOF: inject a regression and watch the gate go red against a real wheel in a
# real empty virtualenv. ~5s for the build, paid once for the session.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def clean_install(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The interpreter a `pip install langres` user actually gets."""
    return build_clean_install(tmp_path_factory.mktemp("docs-gate-proof"))[0]


def test_semantic_only_snippet_turns_the_gate_red(
    clean_install: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inject a `[semantic]`-only snippet into a doc; the gate must fail on it."""
    monkeypatch.delenv("PYTHONPATH", raising=False)
    snippets = _extract(tmp_path, f"```python\n{SEMANTIC_ONLY_SNIPPET}```\n")
    workdir = tmp_path / "work"
    workdir.mkdir()

    (result,) = run(snippets, interpreter=clean_install, workdir=workdir, examples_shipped=False)

    assert result.status == "fail", f"gate did NOT catch the regression: {result}"
    assert "faiss" in result.detail


def test_the_same_snippet_is_skipped_once_the_doc_declares_the_extra(
    clean_install: Path, tmp_path: Path
) -> None:
    """The exemption is what makes it green -- declared in the doc, not in the gate."""
    snippets = _extract(
        tmp_path,
        f"<!-- docs-gate: requires-extra=semantic -->\n```python\n{SEMANTIC_ONLY_SNIPPET}```\n",
    )
    workdir = tmp_path / "work"
    workdir.mkdir()

    (result,) = run(snippets, interpreter=clean_install, workdir=workdir, examples_shipped=False)

    assert result.status == "skip"
    assert result.detail == "declared requires-extra=semantic"


def test_a_core_only_snippet_passes_against_the_wheel(clean_install: Path, tmp_path: Path) -> None:
    """The gate is not merely always-red: the $0 offline path really works."""
    snippets = _extract(
        tmp_path,
        "```python\n"
        "from langres.architectures import FuzzyString\n"
        'records = [{"id": "1", "name": "Acme Corp"}, {"id": "2", "name": "Acme Corporation"}]\n'
        "assert FuzzyString(threshold=0.6).dedupe(records)\n"
        "```\n",
    )
    workdir = tmp_path / "work"
    workdir.mkdir()

    (result,) = run(snippets, interpreter=clean_install, workdir=workdir, examples_shipped=False)

    assert result.status == "pass", result.detail


def test_snippets_run_cumulatively_within_one_document(clean_install: Path, tmp_path: Path) -> None:
    """A page builds up a session; block two may use what block one defined.

    Without this, every continuation block in README/GETTING_STARTED reports a
    `NameError` the document does not actually have -- a wall of false failures
    that buries the real ones.
    """
    snippets = _extract(
        tmp_path,
        "```python\nshared = 41\n```\n\nprose\n\n```python\nassert shared + 1 == 42\n```\n",
    )
    workdir = tmp_path / "work"
    workdir.mkdir()

    first, second = run(
        snippets, interpreter=clean_install, workdir=workdir, examples_shipped=False
    )

    assert first.status == "pass", first.detail
    assert second.status == "pass", second.detail


def test_a_failing_block_is_not_carried_into_later_blocks(
    clean_install: Path, tmp_path: Path
) -> None:
    """A block that raised did not bind its names, so it must not join the prefix.

    Otherwise one broken snippet re-raises inside every later snippet's run and
    the report blames blocks that are fine.
    """
    snippets = _extract(
        tmp_path,
        "```python\nraise RuntimeError('boom')\n```\n\n```python\nassert True\n```\n",
    )
    workdir = tmp_path / "work"
    workdir.mkdir()

    first, second = run(
        snippets, interpreter=clean_install, workdir=workdir, examples_shipped=False
    )

    assert first.status == "fail"
    assert "boom" in first.detail
    assert second.status == "pass", second.detail
