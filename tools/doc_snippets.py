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
path must declare `requires-repo`**, because the built wheel does not ship that
directory. That claim is *observed* from the wheel this run built, not assumed
(see `examples_shipped` in :func:`run`) -- if `examples/` ever starts shipping,
the rule correctly stops firing.

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
_FENCE_RE = re.compile(r"^(?P<ticks>`{3,})(?P<info>.*)$")
#: A repo-relative path under `examples/` -- the directory the wheel does not ship.
_EXAMPLES_REF_RE = re.compile(r"examples/[\w./-]+\.py")


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
            if pending is not None and line.strip():
                raise DirectiveError(
                    f"{doc}:{pending_line}: docs-gate directive is not followed by a fenced "
                    f"code block (next content is line {index + 1}). Move it directly above "
                    "the block it exempts, or delete it."
                )
            index += 1
            continue

        fence_line = index + 1  # 1-based line number of the opening fence
        ticks = fence.group("ticks")
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
        while index < len(lines):
            closing = _FENCE_RE.match(lines[index])
            if (
                closing is not None
                and len(closing.group("ticks")) >= len(ticks)
                and not closing.group("info").strip()
            ):
                break
            body.append(lines[index])
            index += 1
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


def collect(repo_root: Path = REPO_ROOT) -> list[Snippet]:
    """Every fenced block across :data:`DOC_PATHS`, in document order."""
    extras = declared_extras(repo_root)
    snippets: list[Snippet] = []
    for doc in DOC_PATHS:
        snippets.extend(extract(repo_root / doc, doc=doc, extras=extras))
    return snippets


def _snippet_env() -> dict[str, str]:
    """The child environment: no repo on the path, no credentials.

    ``PYTHONPATH`` is cleared because pytest puts ``src`` and ``tools`` on it
    (`[tool.pytest.ini_options].pythonpath`); inheriting that would let a snippet
    import langres from the source tree and pass while the wheel is broken --
    the precise way this gate would lie.

    API keys are stripped so a documentation snippet can never bill anyone. That
    is not a complete spend guard (litellm's import runs ``load_dotenv()``), but
    the snippet also runs in a scratch directory with no ``.env`` in it, so there
    is nothing for that to find.
    """
    env = {k: v for k, v in os.environ.items() if not k.endswith(("_API_KEY", "_TOKEN"))}
    env.pop("PYTHONPATH", None)
    return env


def check_examples_reference(snippet: Snippet, *, examples_shipped: bool) -> str | None:
    """Reject an unexempted reference to `examples/...`; return the failure detail.

    The wheel ships ``src/langres`` only, so a documented command that runs a
    file from ``examples/`` cannot work for a pip user. ``examples_shipped`` is
    measured from the wheel this run built, so the rule disappears by itself if
    that ever changes.
    """
    if examples_shipped or snippet.exemption == _REASON_REQUIRES_REPO:
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


def run_snippet(snippet: Snippet, *, interpreter: Path, workdir: Path) -> SnippetResult:
    """Execute one snippet verbatim with ``interpreter`` inside ``workdir``."""
    script = workdir / "snippet.py"
    script.write_text(snippet.code, encoding="utf-8")
    proc = subprocess.run(
        [str(interpreter), str(script)],
        cwd=workdir,
        capture_output=True,
        text=True,
        env=_snippet_env(),
        # Generous on purpose: a doc snippet should be quick, but a cold model
        # download or a slow import must not be reported as a broken snippet.
        timeout=600,
    )
    if proc.returncode == 0:
        return SnippetResult(snippet, "pass", "")
    tail = (proc.stderr or proc.stdout).strip().splitlines()
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
    (wheel,) = dist.glob("*.whl")
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
    return interpreter, wheel


def run(
    snippets: list[Snippet],
    *,
    interpreter: Path,
    workdir: Path,
    examples_shipped: bool,
) -> list[SnippetResult]:
    """Check and execute ``snippets``, returning one result each."""
    results: list[SnippetResult] = []
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
        results.append(run_snippet(snippet, interpreter=interpreter, workdir=cell))
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
            interpreter = Path(args.interpreter)
            examples_shipped = False
        snippets = collect()
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
