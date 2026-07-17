"""The training-surface DoD examples import + run with the clean top-level names.

Acceptance for the public-surface wiring (task #13): every example the plan
promises imports with the names it *shows* -- ``from langres import ...`` -- and
the two fully-offline ones RUN end to end at ``$0``:

- **#1 supervised RandomForest fit** and **#5 describe() + calibrate fit** run
  FULLY OFFLINE here (deterministic, no LM, no network).
- **#2 DSPy/MIPRO prompt-tune** and **#4 vLLM serve** need a served LLM endpoint,
  so they are IMPORT+CONSTRUCT-validated only: the method / matcher objects build
  with the clean imports, but the paid/endpoint call is never executed.
- **#3 QLoRA capstone** has its own end-to-end MPS run in
  ``tests/core/test_finetune_capstone.py``; here we only confirm its public
  imports resolve.

Also guards the wiring itself: the surfaced symbols are importable from
``langres`` (and the method objects from ``langres.core``) and are in ``__all__``.
"""

from __future__ import annotations

from typing import Any

import pytest

from langres.core.harvest import LabeledPair

# Clean top-level imports -- the whole point of this PR. If any fails to resolve,
# collection fails loudly (that IS the regression guard for the wiring).
from langres import (
    GEPA,
    MIPRO,
    Bootstrap,
    CompanySchema,
    FitReport,
    Isotonic,
    Method,
    Platt,
    Resolver,
    align_pairs,
)


def test_every_exported_exception_is_actually_raised_somewhere() -> None:
    """No name in ``__all__`` may advertise an exception nothing can produce.

    **The gate W4 added, and the hole it closes.** Every other assertion in this
    file is a *subset* check (``root_names <= __all__``): it proves the names we
    care about are present, and is structurally incapable of noticing a name that
    should have LEFT. So when W4 deleted ``choose_auto_judge`` -- the only code
    that raised ``NoMatcherAvailableError`` -- the whole suite stayed green while
    ``langres.__all__`` still advertised a documented, root-exported exception
    that no code path could ever throw. A user could import it and write
    ``except NoMatcherAvailableError:`` forever, and never catch anything.

    Exceptions are gated rather than all exports because the check is only honest
    for them. "Referenced somewhere in ``src/``" is the wrong proxy for a general
    export: ``MIPRO``, ``Platt``, ``CompanySchema`` are *meant* to be constructed
    by users and may legitimately appear nowhere else in the library. But an
    exception the library never raises is dead by definition -- there is no
    caller-side story that redeems it. Zero false positives, and it catches
    exactly the failure that shipped.

    AST-based, deliberately: it reads ``raise X`` / ``raise X(...)`` from the
    parse tree rather than grepping, because a docstring mentioning the name --
    and there are several -- is not a producer. (Grep-as-import-graph has already
    produced multiple verified errors in this repo.)
    """
    import ast
    import pathlib

    import langres

    exported_exceptions = {
        name
        for name in langres.__all__
        if isinstance(getattr(langres, name, None), type)
        and issubclass(getattr(langres, name), BaseException)
    }
    assert exported_exceptions, "expected the root to export at least one exception"

    raised: set[str] = set()
    for path in pathlib.Path(langres.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(exc, ast.Name):
                raised.add(exc.id)
            elif isinstance(exc, ast.Attribute):
                raised.add(exc.attr)

    orphans = sorted(exported_exceptions - raised)
    assert not orphans, (
        f"langres.__all__ exports {len(orphans)} exception(s) that NO code in src/ raises: "
        f"{orphans}.\nAn exception nobody throws is a promise to the user that cannot be kept: "
        "they can import it and `except` it forever and never catch a thing.\nEither delete the "
        "export (its producer is gone -- this is what happened to NoMatcherAvailableError when "
        "W4 deleted choose_auto_judge), or restore the code that raises it."
    )


def test_surfaced_symbols_are_in_public_all() -> None:
    """Every newly surfaced name is exported (in ``__all__``) from where it should be."""
    import langres
    import langres.core as core

    root_names = {
        "Method",
        "Bootstrap",
        "MIPRO",
        "GEPA",
        "Platt",
        "Isotonic",
        "align_pairs",
        "FitReport",
        "LLMMatcher",
    }
    assert root_names <= set(langres.__all__)
    # LLMMatcher stays LAZY at the root ([llm]); the light method/fit symbols are
    # also in langres.core.__all__.
    assert (root_names - {"LLMMatcher"}) <= set(core.__all__)
    # Method-object identity is shared between the two surfaces (one class, two names).
    assert langres.MIPRO is core.MIPRO and langres.Platt is core.Platt
    assert (Method, Bootstrap, GEPA, Isotonic, FitReport) == (
        core.Method,
        core.Bootstrap,
        core.GEPA,
        core.Isotonic,
        core.FitReport,
    )


def test_dod1_random_forest_supervised_fit_runs_offline() -> None:
    """DoD #1: from_schema(matcher="random_forest") -> fit(pairs=...) -> resolve, at $0."""
    pytest.importorskip("sklearn")

    resolver = Resolver.from_schema(CompanySchema, matcher="random_forest")
    # Two entity-disjoint components so a 0.5 split holds a whole component out.
    records = [
        {"id": "a1", "name": "Acme"},
        {"id": "a2", "name": "Acme"},
        {"id": "a3", "name": "Aardvark"},
        {"id": "b1", "name": "Beta"},
        {"id": "b2", "name": "Beta"},
        {"id": "b3", "name": "Bumble"},
    ]
    pairs = [
        LabeledPair(left_id="a1", right_id="a2", score=None, label=True, source="correction"),
        LabeledPair(left_id="a1", right_id="a3", score=None, label=False, source="correction"),
        LabeledPair(left_id="b1", right_id="b2", score=None, label=True, source="correction"),
        LabeledPair(left_id="b1", right_id="b3", score=None, label=False, source="correction"),
    ]

    resolver.fit(records, pairs=pairs, split=0.5, seed=0)

    report = resolver.fit_report_
    assert isinstance(report, FitReport)
    assert report.trained is True
    assert report.n_train > 0
    assert isinstance(resolver.resolve(records), list)


# Six entity-disjoint groups, each with two positives + one negative, so an
# entity-disjoint split keeps both classes on both sides -- the shape a calibrate
# fit and its held-out delta need (mirrors tests/core/test_resolver_fit_calibrate).
_CAL_BASES = [
    ("Acme", "Beta"),
    ("Gamma", "Delta"),
    ("Epsilon", "Zeta"),
    ("Eta", "Theta"),
    ("Iota", "Kappa"),
    ("Lambda", "Mu"),
]


def _calibration_dataset() -> tuple[list[dict[str, str]], list[LabeledPair]]:
    records: list[dict[str, str]] = []
    pairs: list[LabeledPair] = []
    for g, (x, y) in enumerate(_CAL_BASES):
        x0, x1, y0, y1 = f"g{g}x0", f"g{g}x1", f"g{g}y0", f"g{g}y1"
        records += [
            {"id": x0, "name": f"{x} Corp"},
            {"id": x1, "name": f"{x} Corporation"},
            {"id": y0, "name": f"{y} Inc"},
            {"id": y1, "name": f"{y} Incorporated"},
        ]
        pairs += [
            LabeledPair(left_id=x0, right_id=x1, score=None, label=True, source="correction"),
            LabeledPair(left_id=y0, right_id=y1, score=None, label=True, source="correction"),
            LabeledPair(left_id=x0, right_id=y0, score=None, label=False, source="correction"),
        ]
    return records, pairs


def test_dod5_describe_and_calibrate_fit_runs_offline() -> None:
    """DoD #5: describe() shows the trainable slots; fit(method=Platt()) trains a calibrator, at $0."""
    pytest.importorskip("sklearn")

    resolver = Resolver.from_schema(CompanySchema, matcher="string", threshold=0.5)

    # describe() pre-fit: the calibrator slot exists and is frozen/empty.
    pre = next(line for line in resolver.describe().splitlines() if line.startswith("calibrator"))
    assert "<none>" in pre and "frozen" in pre

    records, pairs = _calibration_dataset()
    resolver.fit(records, pairs=pairs, method=Platt())

    post = next(line for line in resolver.describe().splitlines() if line.startswith("calibrator"))
    assert "Calibrator" in post and "TRAINABLE" in post
    assert isinstance(resolver.fit_report_, FitReport)


def test_dod2_mipro_method_constructs_with_clean_import() -> None:
    """DoD #2 (import+construct only -- running MIPRO needs a served LLM, not executed here)."""
    method = MIPRO(auto="light", budget_usd=5.0)
    assert method.kind == "prompt"
    assert "MIPROv2" in method.describe()
    # Bootstrap/GEPA are the sibling prompt optimizers on the same clean surface.
    assert Bootstrap().kind == "prompt" and GEPA().kind == "prompt"


def test_dod4_vllm_llmmatcher_constructs_with_clean_import() -> None:
    """DoD #4 (import+construct only -- forward() would hit the vLLM endpoint, not called)."""
    pytest.importorskip("litellm")
    from langres import LLMMatcher

    matcher: LLMMatcher[Any] = LLMMatcher(
        model="hosted_vllm/my-finetuned-model",
        api_base="http://localhost:8000/v1",
        response_parser="binary_yes_no",
        confidence="logprob",
    )
    # Constructed, not called: the client is built lazily on first forward().
    assert matcher is not None
    assert isinstance(Isotonic(), Method)  # the other calibrate method, same surface


def test_dod3_capstone_public_imports_resolve() -> None:
    """DoD #3: the QLoRA capstone example's public imports all resolve (run lives elsewhere)."""
    pytest.importorskip("litellm")  # LLMMatcher access pulls litellm ([llm])
    from langres import LLMMatcher, QLoRA, run_finetune
    from langres.training.finetune import FINETUNE_YES_NO_PROMPT
    from langres.core.matchers.model_ref import to_config
    from langres.eval import candidates_for, evaluate, get_benchmark

    assert all(
        obj is not None
        for obj in (
            LLMMatcher,
            QLoRA,
            run_finetune,
            FINETUNE_YES_NO_PROMPT,
            to_config,
            candidates_for,
            evaluate,
            get_benchmark,
        )
    )
    # align_pairs (the pairs->candidates bridge) is now a clean top-level import too.
    assert callable(align_pairs)
