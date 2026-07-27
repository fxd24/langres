"""Clean-install documentation gate: does a `pip install langres` user's copy-paste work?

The defect this exists for: `docs/index.md` presented
`uv run python examples/research/first_experiment.py` directly above
`pip install langres`. A pip user fails twice over -- `examples/` is not in the
wheel (`[tool.hatch.build]` ships `src/langres` only), and that script builds a
`Retrieve`/`QdrantDenseIndex` pipeline whose `qdrant-client` lives in the
`[semantic]` extra, not in core `dependencies`. Nothing in CI could see it: every
test job runs `uv sync --all-extras` against the *source tree*, where both the
examples directory and every extra are present. A human reading the docs found it.

So this gate reproduces the user's environment instead of the developer's:

1. build the real wheel (`uv build`),
2. install it into a **fresh, empty virtualenv** -- no extras, not editable, no
   repo on `sys.path`,
3. execute every ```python block in the user-facing docs with that interpreter,
   verbatim, in a scratch working directory.

An editable install or a repo-relative `PYTHONPATH` would silently defeat all of
it, so `_snippet_env` clears `PYTHONPATH` and the runner never runs a snippet
with the repo as its working directory.

## Snippets are executed unless the DOC declares an exemption

Some snippets are *meant* to need more than a bare install: an extra, a git
checkout, a network call. That is fine -- but the exemption is declared in the
document, next to the snippet, not in a list this file keeps privately. A private
list drifts from the docs and the gate then verifies the list.

    <!-- docs-gate: requires-extra=semantic -->
    ```python
    from langres.core.indexes import QdrantDenseIndex
    ```

**`illustrative` is the one exemption no evidence can contradict, and it will
absorb every hard case unless someone is watching.** `requires-extra=` is
validated against pyproject's real extras. `requires-repo` and
`requires-network` each name a condition you could go and check. `illustrative`
asserts "this is not runnable code" -- a claim about intent, which nothing here
can falsify. That asymmetry is not hypothetical: a block this gate had *watched
execute* on a bare wheel was later marked `illustrative`, one line below prose
reading "This cell is copy-paste complete". Prefer the specific reason whenever
one is true -- a fragment whose binding block needs the network is
`requires-network`, not `illustrative` -- and read the per-document executed
ratio that :func:`report` prints, which is what makes the drift visible.

**A one-hop probe cannot catch a two-hop leak.** The clean-venv check originally
inspected only the interpreter this file launches directly, and passed. Meanwhile
`PATH` still began with the repo's own `.venv/bin` -- CI runs this gate under
`uv run` -- so a snippet that shelled out to `python` or the `langres` console
script would have run against the source tree and its dev dependencies, reached
the sentinel, and reported PASS while failing for a pip user. Isolation has to be
checked at every hop a snippet can take, not at the first one; and "first in
PATH" is a *preference*, not an exclusion, so the repo's entries are removed
rather than merely outranked (see :func:`_snippet_path`).

The direction of rot is the whole design. **Default is "execute".** A newly added
snippet nobody classified is *run*, and fails if it does not work on a bare
install -- the gate rots **closed** (a new snippet fails until someone looks at
it) rather than **open** (a new broken snippet passes by default). Everything
else here follows from that: an unrecognized directive token, an unrecognized
fence language, and a directive that does not precede a fence are all **errors**,
never silent skips. A typo'd exemption must not read as an exemption.

## Non-python blocks

Only ```python blocks execute -- running arbitrary ```bash from documentation
(`pip install ...`, `git clone ...`) would exercise the network, not the wheel.
But the reported defect lived in a bash block, so one static rule covers every
fenced block regardless of language: **a block that references an `examples/...`
path must declare an exemption** (`requires-repo` is the apt one), because the
built wheel does not ship that directory. That claim is *observed* from the wheel
this run built, not assumed (see `examples_shipped` in :func:`main`) -- if
`examples/` ever starts shipping, the rule correctly stops firing.

Note the rule exists because *importing* is not enough to catch this class.
Measured on a bare wheel install: `from langres.core.indexes import
QdrantDenseIndex` succeeds, and so does constructing one -- `qdrant_client` is
only reached in use. A gate that scanned imports would have missed the very
defect it was built for. This one runs the code.

## Finding nothing is a failure, not a pass

:func:`assert_gate_is_observing` refuses a run with zero executable snippets, and
:func:`collect` refuses a document that yields zero fenced blocks. A gate that
can go green by matching nothing is this repo's recurring bug, not a corner case.

## Running it

The same one command in CI and locally, so there is no CI-only path to drift:

    uv run python tools/doc_snippets.py                    # build wheel + clean venv
    uv run python tools/doc_snippets.py --interpreter PATH # reuse an interpreter

Exits non-zero if any snippet fails or any directive is malformed.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The user-facing documents whose snippets a bare `pip install langres` must run.
#: These are the pages a new user actually lands on; deeper reference docs
#: (TECHNICAL_OVERVIEW, EXPERIMENTS, ...) address readers who already have a
#: working install and are deliberately out of scope.
DOC_PATHS: tuple[str, ...] = ("README.md", "docs/index.md", "docs/GETTING_STARTED.md")

#: Fence languages whose blocks are EXECUTED against the clean install.
EXECUTED_LANGUAGES = frozenset({"python"})

#: Fence languages that are legal but not executed (shell transcripts, sample
#: output, config). Kept as an explicit allow-list: an unknown language is an
#: error, so a block written in a language nobody classified cannot slip through
#: as "not executed, therefore fine".
NON_EXECUTED_LANGUAGES = frozenset(
    {"", "bash", "console", "shell", "sh", "text", "json", "yaml", "toml", "csv", "diff"}
)

#: Exemption reasons a document may declare. `requires-extra=<name>` is validated
#: against pyproject's real extras, so a snippet cannot be excused by an extra
#: that does not exist.
_REASON_REQUIRES_REPO = "requires-repo"
_REASON_REQUIRES_NETWORK = "requires-network"
_REASON_ILLUSTRATIVE = "illustrative"
BARE_EXEMPTION_REASONS = frozenset(
    {_REASON_REQUIRES_REPO, _REASON_REQUIRES_NETWORK, _REASON_ILLUSTRATIVE}
)

_DIRECTIVE_RE = re.compile(r"^\s*<!--\s*docs-gate:\s*(?P<token>[^\s>]+)\s*-->\s*$")
#: A fenced block. Both fence characters, and CommonMark's legal 0-3 spaces of
#: indentation, are matched **because not matching them is a silent skip** --
#: the gate's worst failure mode. `mkdocs.yml` enables `pymdownx.superfences`,
#: which renders ``~~~python`` and list-indented fences exactly like ```` ```python ````,
#: so a doc could show a python block this gate never saw. Measured before the
#: fix: `~~~python`, a 3-space-indented fence, and a list-nested fence all
#: extracted to nothing, with no error and no skip line.
_FENCE_RE = re.compile(r"^(?P<indent>[ \t]{0,3})(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
#: A fence this parser deliberately does NOT support. Matching it raises rather
#: than ignoring it: failing closed on a construct we cannot read is the point.
_UNSUPPORTED_FENCE_RE = re.compile(r"^(?:\s*>[\s>]*|[ \t]{4,})(?:`{3,}|~{3,})")
#: `mkdocs.yml` enables `pymdownx.snippets`, so `--8<-- "other.md"` pulls fences
#: this extractor never reads. Documenting that as a known blind spot is weaker
#: than closing it: the remaining blocks on the page still satisfy the
#: observability check, so the include is invisible in exactly the way that
#: reports green. Fail closed instead, consistent with every other unknown here.
_SNIPPET_INCLUDE_RE = re.compile(r"^\s*(?:-{2,}8<-{2,}|;{0,1}--8<--)")

#: One import proving each optional extra is present. The clean venv must have
#: NONE of them: `pip install langres` installs only the core dependencies, so
#: any of these resolving means the environment under test is not the one being
#: advertised. Checking a single module (faiss) let a venv carrying
#: `[llm]`/`[trained]`/`[eval]` pass the cleanliness assertion unchallenged.
_EXTRA_PROOF_MODULES = frozenset(
    {
        "faiss",
        "sentence_transformers",
        "qdrant_client",
        "torch",
        "litellm",
        "dspy",
        "sklearn",
        "ranx",
        "peft",
        "trl",
    }
)
#: A repo-relative path under `examples/` -- the directory the wheel does not ship.
_EXAMPLES_REF_RE = re.compile(r"examples/[\w./-]+\.py")

#: Proof a snippet reached its own last line. See :func:`run_snippet`.
_SENTINEL = "__langres_docs_gate_reached_end__"
_SENTINEL_EPILOGUE = f'\nimport sys as _dg_sys\n_dg_sys.stderr.write("{_SENTINEL}")\n'


class DirectiveError(Exception):
    """A `docs-gate` directive is malformed, unknown, or attached to nothing."""


@dataclass(frozen=True)
class Snippet:
    """One fenced block, with the exemption its document declared for it."""

    doc: str
    """Repo-relative path of the document (e.g. ``docs/index.md``)."""
    line: int
    """1-based line number of the opening fence -- what the report points at."""
    language: str
    """First token of the fence info string (``""`` for a bare fence)."""
    code: str
    """The block's body, verbatim."""
    exemption: str | None
    """The declared reason this block is not executed, or ``None``."""

    @property
    def location(self) -> str:
        return f"{self.doc}:{self.line}"


@dataclass(frozen=True)
class SnippetResult:
    """The outcome of one snippet: ``pass`` / ``fail`` / ``skip``."""

    snippet: Snippet
    status: str
    detail: str


def declared_extras(repo_root: Path = REPO_ROOT) -> frozenset[str]:
    """The extras `pyproject.toml` actually declares (what `requires-extra=` may name)."""
    config = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    return frozenset(config["project"]["optional-dependencies"])


def _validate_token(token: str, extras: frozenset[str]) -> str:
    """Return ``token`` if it is a legal exemption; otherwise raise."""
    if token in BARE_EXEMPTION_REASONS:
        return token
    if token.startswith("requires-extra="):
        extra = token.removeprefix("requires-extra=")
        if extra not in extras:
            raise DirectiveError(
                f"unknown extra {extra!r} in docs-gate directive {token!r}. "
                f"pyproject declares: {', '.join(sorted(extras))}"
            )
        return token
    raise DirectiveError(
        f"unknown docs-gate directive {token!r}. Valid: "
        + ", ".join(sorted(BARE_EXEMPTION_REASONS))
        + ", requires-extra=<extra>"
    )


def extract(path: Path, *, doc: str, extras: frozenset[str]) -> list[Snippet]:
    """Parse every fenced block in ``path``, attaching any preceding directive.

    A directive applies to the next fenced block, skipping blank lines. A
    directive that reaches something other than a fence raises -- a stray or
    typo'd exemption that quietly does nothing is exactly the failure mode this
    gate exists to prevent.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    snippets: list[Snippet] = []
    pending: str | None = None
    pending_line = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        directive = _DIRECTIVE_RE.match(line)
        if directive is not None:
            if pending is not None:
                raise DirectiveError(
                    f"{doc}:{pending_line}: docs-gate directive is followed by another "
                    "directive instead of a fenced block; one directive exempts one block."
                )
            pending = _validate_token(directive.group("token"), extras)
            pending_line = index + 1
            index += 1
            continue

        fence = _FENCE_RE.match(line)
        if fence is None:
            if _SNIPPET_INCLUDE_RE.match(line):
                raise DirectiveError(
                    f"{doc}:{index + 1}: this page transcludes another file with "
                    "pymdownx.snippets (`--8<--`), whose fenced blocks MkDocs renders to "
                    "readers but this extractor never sees. The other python blocks on the "
                    "page would still satisfy the observability check, so a broken included "
                    "example would ship green. Inline the content into this page, or move the "
                    "include out of the gated documents."
                )
            if _UNSUPPORTED_FENCE_RE.match(line):
                raise DirectiveError(
                    f"{doc}:{index + 1}: this fenced block is indented 4+ spaces or nested in "
                    "a blockquote, which this extractor does not read -- so it would be "
                    "silently skipped rather than checked. Unindent it (0-3 spaces) or move "
                    "it out of the quote."
                )
            if pending is not None and line.strip():
                raise DirectiveError(
                    f"{doc}:{pending_line}: docs-gate directive is not followed by a fenced "
                    f"code block (next content is line {index + 1}). Move it directly above "
                    "the block it exempts, or delete it."
                )
            index += 1
            continue

        fence_line = index + 1  # 1-based line number of the opening fence
        indent = len(fence.group("indent"))
        marker = fence.group("fence")
        info = fence.group("info").strip()
        language = info.split()[0].lower() if info else ""

        if language not in EXECUTED_LANGUAGES and language not in NON_EXECUTED_LANGUAGES:
            raise DirectiveError(
                f"{doc}:{fence_line}: unrecognized fence language {language!r}. "
                "Add it to EXECUTED_LANGUAGES or NON_EXECUTED_LANGUAGES in "
                "tools/doc_snippets.py so it is classified deliberately."
            )

        body: list[str] = []
        index += 1
        closed = False
        while index < len(lines):
            closing = _FENCE_RE.match(lines[index])
            # CommonMark: the closer must use the SAME character, be at least as
            # long, and carry no info string. Matching on length alone let a
            # second ```python read as a closer, swallowing every block after it.
            if (
                closing is not None
                and closing.group("fence")[0] == marker[0]
                and len(closing.group("fence")) >= len(marker)
                and not closing.group("info").strip()
            ):
                closed = True
                break
            # An opener for an EXECUTED language inside another block's body is
            # almost always a missing closing fence upstream. CommonMark (and
            # this parser) then read the python block as bash *content*, so it
            # silently leaves the gate -- rendered as code on the site, never
            # executed here. Raise instead of swallowing it.
            nested = _FENCE_RE.match(lines[index])
            if (
                nested is not None
                and nested.group("info").strip().split()[:1]
                and (nested.group("info").strip().split()[0].lower() in EXECUTED_LANGUAGES)
            ):
                raise DirectiveError(
                    f"{doc}:{index + 1}: a '{language or 'bare'}' block opened at line "
                    f"{fence_line} contains what looks like the start of a python block. "
                    "Almost certainly the block above is missing its closing fence -- as "
                    "written, that python is treated as text and is NEVER executed by this "
                    "gate."
                )
            body.append(_dedent_body_line(lines[index], indent))
            index += 1
        if not closed:
            raise DirectiveError(
                f"{doc}:{fence_line}: fenced block is never closed (reached end of file). "
                "An unclosed fence swallows every block after it, so those snippets would "
                "vanish from this gate silently. Close it."
            )
        index += 1  # step past the closing fence

        snippets.append(
            Snippet(
                doc=doc,
                line=fence_line,
                language=language,
                code="\n".join(body) + "\n",
                exemption=pending,
            )
        )
        pending = None

    if pending is not None:
        raise DirectiveError(
            f"{doc}:{pending_line}: docs-gate directive is at the end of the file with no "
            "fenced block after it."
        )
    return snippets


def _dedent_body_line(line: str, indent: int) -> str:
    """Strip the fence's own indentation from a body line (list-nested fences)."""
    if indent and not line[:indent].strip():
        return line[indent:]
    return line


def collect(repo_root: Path = REPO_ROOT) -> list[Snippet]:
    """Every fenced block across :data:`DOC_PATHS`, in document order.

    Raises if any document contributes **zero** blocks -- see
    :func:`assert_gate_is_observing` for why finding nothing must be loud.
    """
    extras = declared_extras(repo_root)
    snippets: list[Snippet] = []
    for doc in DOC_PATHS:
        found = extract(repo_root / doc, doc=doc, extras=extras)
        if not found:
            raise DirectiveError(
                f"{doc} contributed 0 fenced code blocks. Either the file moved (fix "
                "DOC_PATHS), or the fence syntax changed and _FENCE_RE no longer matches "
                "it. A document this gate cannot see is a document it cannot check."
            )
        snippets.extend(found)
    return snippets


def assert_gate_is_observing(snippets: list[Snippet]) -> None:
    """Fail loudly if there is nothing left to execute.

    A gate that can pass by matching nothing is this repo's recurring failure --
    the `[tool.hatch.build]` path literals that "fail silently" when a directory
    is renamed, the step named "95% coverage gate" that enforced 90, the
    cancelled run read as green. Every one of them reported all-clear while
    observing nothing.

    The shape here would be: a change to ``_FENCE_RE``, ``DOC_PATHS`` or
    ``EXECUTED_LANGUAGES`` stops matching python blocks, every remaining block is
    skipped, the job prints "0 failed" and goes green forever. So an empty
    executable set is an error, not a pass -- and the exemptions cannot cause it
    either: if every python block in the docs were exempted, this gate would be
    running nothing and should say so out loud.
    """
    # `code.strip()` matters: an empty ```python``` fence is "executable", runs,
    # and passes -- so counting fences rather than content would let this
    # assertion be satisfied by a block that proves nothing.
    executable = [
        s
        for s in snippets
        if s.language in EXECUTED_LANGUAGES and s.exemption is None and s.code.strip()
    ]
    if not executable:
        total = len(snippets)
        raise DirectiveError(
            f"found {total} fenced block(s) across {', '.join(DOC_PATHS)} but 0 of them are "
            "executable python, so this run would check nothing and pass. Either every "
            "python block is now exempted (make one runnable on a bare install), or the "
            "extractor stopped recognizing them (check _FENCE_RE / EXECUTED_LANGUAGES). "
            "A gate that passes by finding nothing is worse than no gate."
        )

    # Per DOCUMENT, not just overall. A repo-wide count is satisfied by one
    # healthy page while another erodes to nothing -- which is exactly what
    # happened: a page went from 2 executed / 13 failing to 1 executed / 17
    # exempted, reporting ZERO failures while checking less than before. The
    # repo-wide assertion above stayed comfortably satisfied throughout.
    #
    # The floor is deliberately "at least one", not a ratio: a threshold would be
    # an arbitrary number to argue with, and the real defence is the
    # checked/total ratio that `report` prints for every document.
    #
    # KNOWN LIMIT, measured rather than assumed: this floor does NOT catch the
    # regression that motivated it. A page keeping one trivial checked block
    # (`pass`) while exempting all 17 behavioural examples satisfies it -- the
    # 1-checked/17-exempted state passes, verified directly against this
    # function. The ratio is printed but compared to nothing, so erosion is
    # visible only to a reader who happens to look at a green job's log.
    #
    # Closing it properly needs a committed per-document baseline that fails on
    # a DECREASE (the shape of `tests/test_import_tangle.py`, this repo's
    # existing ratchet). That is deliberately not done here: a baseline pinned
    # to today's numbers would fail the moment the in-flight documentation
    # branches legitimately move them, so it belongs in a change that can update
    # the baseline and the docs together, as one reviewable diff.
    for doc in DOC_PATHS:
        in_doc = [s for s in snippets if s.doc == doc]
        has_python = any(s.language in EXECUTED_LANGUAGES and s.code.strip() for s in in_doc)
        if has_python and not any(s in executable for s in in_doc):
            raise DirectiveError(
                f"{doc}: every python block on this page is exempted, so the page is "
                "documented but unchecked. At least one block per page must run on a bare "
                "install -- that is the claim the page makes to a reader who pip-installed "
                "langres. Make one cell runnable, or move the page out of DOC_PATHS if it "
                "is genuinely not about running code. (A page with no python blocks at all "
                "is unaffected.)"
            )


#: The ONLY variables a snippet's interpreter inherits. An **allow-list**, not a
#: subtraction of known-bad names: a deny-list of credential suffixes rots open,
#: and this one measurably did -- `AWS_SECRET_ACCESS_KEY`, `AZURE_OPENAI_KEY`,
#: `GOOGLE_APPLICATION_CREDENTIALS` and `LANGFUSE_SECRET_KEY` all survived a
#: `endswith(("_API_KEY", "_TOKEN"))` filter. Every new credential a provider
#: invents would have survived it too.
#:
#: `PYTHONPATH` is deliberately absent: pytest puts `src` and `tools` on it
#: (`[tool.pytest.ini_options].pythonpath`), and inheriting that would let a
#: snippet import langres from the SOURCE TREE and pass while the wheel is
#: broken -- the precise way this gate would lie. `PYTHONHOME`/`PYTHONSTARTUP`
#: are absent for the same reason.
#:
#: `PATH` is inherited but **rewritten** -- see :func:`_snippet_path`. Passing it
#: through untouched was a real hole: CI runs this gate under `uv run`, so `PATH`
#: begins with the repo's own `.venv/bin`. A snippet that shells out to `python`
#: or the `langres` console script would have resolved the SOURCE TREE and its
#: dev dependencies, reached the sentinel, and passed -- while failing for a pip
#: user. The cleanliness probe only inspects the interpreter we launch directly,
#: so it could not see that.
_INHERITED_ENV_VARS = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT")


def _snippet_path(interpreter: Path, *, repo_root: Path = REPO_ROOT) -> str:
    """`PATH` with the clean venv first and every repo-local directory removed.

    Two moves, because either alone leaks. Putting the venv first makes a bare
    `python` resolve to the wheel install. Dropping repo-local entries means no
    *later* entry can serve the source tree either -- notably `.venv/bin`, which
    `uv run` puts at the front of our own `PATH`.
    """
    resolved_repo = repo_root.resolve()
    kept = [
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry and not _is_within(entry, resolved_repo)
    ]
    return os.pathsep.join([str(interpreter.parent), *kept])


def _is_within(entry: str, root: Path) -> bool:
    """Is `entry` inside `root`? A malformed PATH entry is treated as suspect."""
    try:
        return root in Path(entry).resolve().parents or Path(entry).resolve() == root
    except OSError:  # pragma: no cover - unresolvable PATH entry
        return True


def _snippet_env(interpreter: Path) -> dict[str, str]:
    """The child environment: no repo on the path, no credentials, nothing else."""
    env = {name: os.environ[name] for name in _INHERITED_ENV_VARS if name in os.environ}
    env["PATH"] = _snippet_path(interpreter)
    return env


def check_examples_reference(snippet: Snippet, *, examples_shipped: bool) -> str | None:
    """Reject an unexempted reference to `examples/...`; return the failure detail.

    The wheel ships ``src/langres`` only, so a documented command that runs a
    file from ``examples/`` cannot work for a pip user. ``examples_shipped`` is
    measured from the wheel this run built, so the rule disappears by itself if
    that ever changes.

    **ANY declared exemption satisfies this rule, not just ``requires-repo``.**
    The rule polices one claim -- "a bare install can run this" -- and a block
    that declares *any* exemption has stopped making that claim. Requiring
    `requires-repo` specifically forced a mislabel: an `illustrative` block that
    merely mentions an example path was unfixable except by tagging it
    `requires-repo`, which says something false about it. `requires-repo` stays
    the suggestion in the message because it is the apt reason for the common
    case.
    """
    if examples_shipped or snippet.exemption is not None:
        return None
    referenced = sorted(set(_EXAMPLES_REF_RE.findall(snippet.code)))
    if not referenced:
        return None
    return (
        f"references {', '.join(referenced)}, which the wheel does not ship "
        f"(it contains src/langres only), so a `pip install langres` user cannot run it. "
        f"Point at an installed API instead, or declare "
        f"`<!-- docs-gate: {_REASON_REQUIRES_REPO} -->` above the block."
    )


def run_snippet(
    snippet: Snippet, *, interpreter: Path, workdir: Path, prefix: str = ""
) -> SnippetResult:
    """Execute one snippet verbatim with ``interpreter`` inside ``workdir``.

    ``prefix`` is the code of the earlier passing python blocks in the same
    document -- see :func:`run` for why the page, not the block, is the unit.
    """
    script = workdir / "snippet.py"
    # The sentinel is appended AFTER the snippet (which still runs verbatim) and
    # is what "passed" actually means. Exit code 0 alone is not evidence the body
    # ran: measured, a prefix block containing `sys.exit(0)` / `raise SystemExit`
    # / `os._exit(0)` made a snippet whose body was `import totally_missing_module`
    # report PASS. One `sys.exit` in a documented error-handling example would
    # have turned every later block on that page green without executing it.
    script.write_text(prefix + snippet.code + _SENTINEL_EPILOGUE, encoding="utf-8")
    try:
        # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
        # Evaluated, not waved through. This is the LIST form with no `shell=True`,
        # so no shell parses it: `argv` reaches execve as separate words and
        # nothing in `interpreter` or `script` can become a command. (The rule
        # suggests `shlex.escape()`, which does not exist -- `shlex.quote` does --
        # and quoting is for the string form we deliberately do not use.)
        # Running this code is also the entire point of the file: the input is
        # this repo's own documentation, and anyone who can edit it can already
        # put whatever they like in a ```python block.
        proc = subprocess.run(
            [str(interpreter), str(script)],
            cwd=workdir,
            capture_output=True,
            text=True,
            env=_snippet_env(interpreter),
            # Generous on purpose: a doc snippet should be quick, but a cold model
            # download or a slow import must not be reported as a broken snippet.
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        # Caught, not propagated: an escaping exception would discard the whole
        # report, losing every failure already determined.
        return SnippetResult(snippet, "fail", "timed out after 600s")
    if proc.returncode == 0 and _SENTINEL in proc.stderr:
        return SnippetResult(snippet, "pass", "")
    if proc.returncode == 0:
        return SnippetResult(
            snippet,
            "fail",
            "the snippet exited 0 without reaching the end of the block (something in it "
            "terminated the interpreter early, e.g. sys.exit/SystemExit/os._exit), so this "
            "block was not actually executed to completion.",
        )
    tail = (proc.stderr or proc.stdout).replace(_SENTINEL, "").strip().splitlines()
    return SnippetResult(snippet, "fail", "\n".join(tail[-12:]))


def build_clean_install(workdir: Path, *, repo_root: Path = REPO_ROOT) -> tuple[Path, Path]:
    """Build the wheel and install it into a fresh, extras-free virtualenv.

    Returns ``(interpreter, wheel)``. Deliberately the real artifact and a real
    empty environment: an editable install or `uv sync` would put the source tree
    and every extra back, which is the developer's environment, not the user's.
    """
    dist = workdir / "dist"
    subprocess.run(
        ["uv", "build", "--out-dir", str(dist)], cwd=repo_root, check=True, capture_output=True
    )
    # Name the build failure instead of letting it surface as a tuple-unpack
    # ValueError: `wheel` decides both what gets installed AND whether the
    # examples rule applies, so an ambiguous build must not be guessed at.
    wheels = sorted(dist.glob("*.whl"))
    if len(wheels) != 1:
        raise DirectiveError(
            f"`uv build` produced {len(wheels)} wheels in {dist}, expected exactly 1: "
            f"{[w.name for w in wheels]}"
        )
    wheel = wheels[0]
    venv = workdir / "venv"
    subprocess.run(["uv", "venv", str(venv)], cwd=repo_root, check=True, capture_output=True)
    interpreter = venv / "bin" / "python"
    subprocess.run(
        ["uv", "pip", "install", "--python", str(interpreter), str(wheel)],
        # cwd=repo_root so `[tool.uv].exclude-newer` (the 7-day supply-chain
        # quarantine) governs this resolution too.
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    _assert_environment_is_really_clean(interpreter, repo_root=repo_root)
    return interpreter, wheel


def _assert_environment_is_really_clean(interpreter: Path, *, repo_root: Path) -> None:
    """Prove the premise this whole gate rests on, instead of assuming it.

    Everything here is worthless if the "clean" venv can still see the source
    tree or an extra: every snippet would pass against the developer's full
    environment and the job would be green forever with no signal -- a check
    nobody has watched fail. So check it, every run, before running any snippet.

    Three properties, all cheap: langres must resolve *inside the venv*, a
    `[semantic]`-only import must fail, and -- the one this probe originally
    missed -- a *nested* `python` / `langres` must resolve there too. Checking
    only the interpreter we launch directly left a snippet that shells out
    running against the repo's own `.venv/bin`, passing on dev dependencies a
    pip user does not have. A probe that inspects one hop cannot see a two-hop
    leak, so it now walks `PATH` the way a subprocess would.
    """
    probe = (
        "import langres, sys, pathlib, shutil;"
        "here = pathlib.Path(langres.__file__).resolve();"
        f"venv = pathlib.Path({str(interpreter.parent.parent)!r}).resolve();"
        f"repo = pathlib.Path({str(repo_root)!r}).resolve();"
        "assert venv in here.parents, f'langres resolved OUTSIDE the clean venv: {here}';"
        "assert repo not in here.parents, f'langres resolved from the SOURCE TREE: {here}';"
        "import importlib.util as u;"
        # Every extra, not just faiss. Probing one module let an ordinary
        # external venv installed with [llm]/[trained]/[eval] satisfy the whole
        # assertion -- a "clean install" report from an environment carrying the
        # dependencies whose absence is the entire thing being tested.
        f"extras = {sorted(_EXTRA_PROOF_MODULES)!r};"
        "present = [m for m in extras if u.find_spec(m) is not None];"
        "assert not present, f'optional extras {present} are installed, so this is not "
        "the extras-free environment a `pip install langres` user gets';"
        # The nested-resolution check: what a snippet's own subprocess would get.
        "nested = [(n, shutil.which(n)) for n in ('python', 'python3', 'langres')];"
        "bad = [(n, p) for n, p in nested if p and repo in pathlib.Path(p).resolve().parents];"
        "assert not bad, f'PATH resolves {bad} inside the SOURCE TREE, so a snippet that "
        "shells out would run against the repo, not the wheel'"
    )
    # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
    # Same evaluation as `run_snippet`: list form, no `shell=True`, no shell
    # involved. `probe` is a fixed template whose only interpolations are two
    # paths inserted with `!r` (a Python repr, so they cannot terminate the
    # string literal), and it is handed to `-c` as one argv element.
    proc = subprocess.run(
        [str(interpreter), "-c", probe],
        capture_output=True,
        text=True,
        env=_snippet_env(interpreter),
        timeout=120,
    )
    if proc.returncode != 0:
        raise DirectiveError(
            "the 'clean install' this gate runs snippets against is NOT clean, so every "
            "result it produces would be meaningless:\n" + (proc.stderr or proc.stdout).strip()
        )


def run(
    snippets: list[Snippet],
    *,
    interpreter: Path,
    workdir: Path,
    examples_shipped: bool,
) -> list[SnippetResult]:
    """Check and execute ``snippets``, returning one result each.

    **The page is the unit, not the block.** A tutorial legitimately builds up a
    session -- README defines ``records`` in one block and uses it in the next
    three -- so each python block runs with the code of the earlier *passing*
    python blocks in the same document prepended. That is what a reader who
    follows the page top to bottom actually executes.

    Running each block in isolation instead would report ``NameError: records``
    on every continuation block: a wall of failures the document does not have,
    which buries the failures it does. A failing block is not added to the
    prefix (its bindings did not happen), and an exempted block is excluded --
    the prefix is only what a bare install actually ran.
    """
    results: list[SnippetResult] = []
    prefixes: dict[str, str] = {}
    for number, snippet in enumerate(snippets):
        path_failure = check_examples_reference(snippet, examples_shipped=examples_shipped)
        if path_failure is not None:
            results.append(SnippetResult(snippet, "fail", path_failure))
            continue
        if snippet.exemption is not None:
            results.append(SnippetResult(snippet, "skip", f"declared {snippet.exemption}"))
            continue
        if snippet.language not in EXECUTED_LANGUAGES:
            results.append(SnippetResult(snippet, "skip", f"{snippet.language or 'bare'} block"))
            continue
        # One scratch directory per snippet: snippets that write files (a saved
        # resolver, a corrections log) must not see each other's leftovers, and
        # none of them may write into the repo.
        cell = workdir / f"snippet-{number:03d}"
        cell.mkdir()
        prefix = prefixes.get(snippet.doc, "")
        result = run_snippet(snippet, interpreter=interpreter, workdir=cell, prefix=prefix)
        if result.status == "pass":
            prefixes[snippet.doc] = prefix + snippet.code
        results.append(result)
    return results


def report(results: list[SnippetResult]) -> str:
    """Render the per-snippet table plus the failure detail; return the text."""
    lines = []
    for result in results:
        note = result.detail if result.status == "skip" else ""
        lines.append(
            f"{result.status.upper():5s} {result.snippet.location:32s} "
            f"{result.snippet.language or 'bare':7s} {note}".rstrip()
        )
    failures = [r for r in results if r.status == "fail"]
    passed = sum(1 for r in results if r.status == "pass")
    skipped = sum(1 for r in results if r.status == "skip")

    # How much of each page this run actually EXECUTED. Printed always, not only
    # on failure, because the number that matters is the one nobody asked for:
    # a page can reach "0 failed" by exempting its way there, and pass/fail/skip
    # totals render that identical to a page that genuinely works. Measured
    # once: a page went 2-executed/13-failing -> 1-executed/17-exempted and the
    # summary line improved. A ratio makes that visible in a diff, with no
    # threshold to argue about.
    #
    # The denominator is PYTHON blocks, not all fenced blocks. Counting bash and
    # text in it would pad every ratio with blocks that can never run and make an
    # eroding page look better than it is -- a metric drifting from the thing it
    # measures, which is the failure this whole line exists to expose.
    lines.append("")
    lines.append("Python blocks checked per document (a low ratio means exempted, not proven)")
    for doc in DOC_PATHS:
        in_doc = [r for r in results if r.snippet.doc == doc and r.snippet.language == "python"]
        if not in_doc:
            continue
        # "Checked", not "executed": a block rejected by the `examples/` rule
        # never ran, but it was observed and reported -- the opposite of exempted.
        checked = sum(1 for r in in_doc if r.status != "skip")
        lines.append(f"  {doc:32s} {checked}/{len(in_doc)} python blocks checked")

    lines.append("")
    lines.append(f"{passed} passed, {len(failures)} failed, {skipped} skipped")
    for failure in failures:
        lines.append("")
        lines.append(f"--- FAIL {failure.snippet.location} ---")
        lines.append(failure.detail)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point: build (or reuse) a clean install, run the docs, print, exit."""
    import tempfile

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--interpreter",
        default=None,
        help="Python to run snippets with. Default: build the wheel and install it "
        "into a fresh extras-free virtualenv (what a `pip install langres` user gets).",
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="langres-docs-gate-") as tmp:
        workdir = Path(tmp)
        if args.interpreter is None:
            interpreter, wheel = build_clean_install(workdir)
            with zipfile.ZipFile(wheel) as zf:
                examples_shipped = any(n.startswith("examples/") for n in zf.namelist())
        else:
            # No wheel to read, so we cannot KNOW whether examples ship -- and
            # the only honest default for an unknown is the strict one. False
            # can produce a spurious failure on a dev's own checkout; True would
            # let a real `examples/` reference through unseen. This gate exists
            # because a check that passes when it cannot observe is worthless,
            # so the unknown fails closed. (CI never takes this branch.)
            interpreter = Path(args.interpreter)
            examples_shipped = False
            # The SAME cleanliness assertion the built venv gets. Skipping it
            # here meant `--interpreter .venv/bin/python` ran every snippet
            # against the editable source tree with all dev extras present:
            # a green "clean install" report from an environment that is the
            # opposite of one -- and the unexempted paid snippets could reach
            # LiteLLM and bill for real. A convenience flag must not be able to
            # buy a greener result than the real thing.
            _assert_environment_is_really_clean(interpreter, repo_root=REPO_ROOT)
        snippets = collect()
        assert_gate_is_observing(snippets)
        results = run(
            snippets,
            interpreter=interpreter,
            workdir=workdir,
            examples_shipped=examples_shipped,
        )

    sys.stdout.write(report(results) + "\n")
    return 1 if any(r.status == "fail" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
