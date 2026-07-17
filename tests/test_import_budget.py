"""Import weight + spend-safety tests (W0.4: extras split + lazy heavy imports).

Two concerns, both closed by making ``langres/__init__.py``'s import chain --
``langres.core``, ``langres.clients``, ``langres.core.blockers``,
``langres.core.matchers`` -- resolve heavy/optional-dependency symbols lazily
(PEP 562 ``__getattr__``) instead of eager top-level imports:

1. **Import weight**: a bare ``import langres`` must not pull torch, litellm,
   faiss, sentence-transformers, or scikit-learn into ``sys.modules`` (the
   ``[semantic]``/``[llm]``/``[trained]`` extras) -- these are now optional
   dependencies, not installed in a core-only environment, and even when
   installed should not slow down or bloat a plain import.
2. **Spend safety**: ``litellm`` calls ``load_dotenv()`` as an import side
   effect, silently populating ``OPENROUTER_API_KEY``/etc. from any ``.env``
   on the path -- independent of whether a judge is ever chosen. A bare
   ``import langres`` must never trigger this (this already cost real
   OpenRouter spend once this session, before this fix).

Subprocess-based (mirrors the existing dspy import-safety pattern in
``tests/test_verbs.py``/``tests/core/modules/test_dspy_judge.py``) since the
whole point is a *fresh* process/import state -- ``sys.modules`` in the
current pytest process is already polluted by other tests.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _import_ok(module_name: str) -> bool:
    """True if ``module_name`` is importable, without actually importing it."""
    return importlib.util.find_spec(module_name) is not None


# ---------------------------------------------------------------------------
# Import weight: heavy/optional deps must stay out of sys.modules.
# ---------------------------------------------------------------------------

_HEAVY_MODULES = [
    "torch",
    "transformers",
    "litellm",
    "faiss",
    "sentence_transformers",
    "sklearn",
    # Tracking backends (S1): the ExperimentTracker adapters must load mlflow/
    # wandb/trackio lazily, never on a bare `import langres`. huggingface_hub
    # is trackio's own transitive dependency and was absent from the eager
    # import graph before trackio_tracker.py -- listed here so it can't
    # silently become eager either.
    "mlflow",
    "wandb",
    "trackio",
    "huggingface_hub",
    # Fine-tune stack ([finetune], PR-F): peft/trl/bitsandbytes import lazily
    # inside core.finetune's QLoRATrainer.train, never on a bare `import langres`.
    "peft",
    "trl",
    "bitsandbytes",
]

_CHECK_SCRIPT = (
    "import sys; import langres; "
    "leaked = [m for m in {modules!r} if m in sys.modules]; "
    "assert not leaked, f'heavy modules leaked into sys.modules: {{leaked}}'; "
    "print('OK')"
).format(modules=_HEAVY_MODULES)


def test_import_langres_excludes_heavy_modules_from_sys_modules() -> None:
    """Plain ``import langres`` must not import torch/litellm/faiss/sentence-transformers."""
    result = subprocess.run(
        [sys.executable, "-c", _CHECK_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"import-budget check failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


# The autoresearch facade (``langres.optimize`` / ``langres.score_blocking``,
# PR P-C) is eager-exported from ``langres/__init__.py`` -- so a bare
# ``import langres`` must expose both as callables WHILE staying import-light:
# ``langres/optimize.py``'s module top is stdlib/typing only (every factory /
# data / metrics / faiss import is lazy inside a function body), so pulling the
# module into the eager graph must not drag torch / faiss / sentence-transformers
# / litellm / scikit-learn into ``sys.modules``. Fresh-process for an unpolluted
# import state (same pattern as the checks above).
_OPTIMIZE_FACADE_HEAVY_DEPS = [
    "torch",
    "faiss",
    "sentence_transformers",
    "litellm",
    "sklearn",
]

_OPTIMIZE_FACADE_SCRIPT = (
    "import sys; import langres; "
    "assert callable(langres.optimize), 'langres.optimize missing or not callable'; "
    "assert callable(langres.score_blocking), 'langres.score_blocking missing or not callable'; "
    "leaked = [m for m in {modules!r} if m in sys.modules]; "
    "assert not leaked, f'optimize facade leaked heavy modules on bare import: {{leaked}}'; "
    "print('OK')"
).format(modules=_OPTIMIZE_FACADE_HEAVY_DEPS)


def test_optimize_facade_is_eager_and_import_light() -> None:
    """``langres.optimize``/``score_blocking`` are callable on bare import, pulling no heavy dep."""
    result = subprocess.run(
        [sys.executable, "-c", _OPTIMIZE_FACADE_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"optimize-facade import-budget check failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


# The prompt-optimize ``Method`` objects (``Bootstrap`` / ``MIPRO`` / ``GEPA``)
# are pure config a caller constructs at the fit call site -- constructing one
# (even ``GEPA(reflection_model=...)``, which names a DSPy optimizer) must never
# pull ``dspy`` into ``sys.modules``. The heavy ``dspy`` import stays lazy inside
# ``dspy_judge``, reached only when a fit actually compiles. Fresh-process so the
# check is not masked by another test having already imported dspy.
_METHODS_PROMPT_IMPORT_LIGHT_SCRIPT = (
    "import sys; "
    "from langres.core.methods_prompt import GEPA, MIPRO, Bootstrap; "
    "Bootstrap(); MIPRO(auto='heavy'); GEPA(reflection_model='x', max_metric_calls=10); "
    "assert 'dspy' not in sys.modules, 'methods_prompt pulled dspy on construct'; "
    "print('OK')"
)


def test_methods_prompt_stays_import_light() -> None:
    """Constructing a prompt-optimize ``Method`` must not import ``dspy`` (config only)."""
    result = subprocess.run(
        [sys.executable, "-c", _METHODS_PROMPT_IMPORT_LIGHT_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"methods_prompt import-budget check failed.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


# The eval harness (``core.metrics`` / ``core.benchmark``) must be importable
# without the ``[eval]`` extra: ``ranx`` (ranking metrics MRR/NDCG/MAP) is now
# imported lazily inside ``evaluate_blocking_with_ranking`` only, so importing
# the modules must never pull ``ranx`` into ``sys.modules``. Subprocess-based for
# a fresh import state (this pytest process loads ranx via the ranking-metric
# test). The curated ``langres.eval`` facade gets the same assertion in
# ``tests/test_eval.py``.
_RANX_DECOUPLE_SCRIPT = (
    "import sys; "
    "import langres.core.metrics; import langres.core.benchmark; "
    "assert 'ranx' not in sys.modules, "
    "'ranx leaked into sys.modules without calling the ranking metrics'; "
    "print('OK')"
)


def test_core_metrics_and_benchmark_do_not_import_ranx() -> None:
    """core.metrics/core.benchmark must import without the [eval] extra.

    Locks in the ranx decoupling: ``ranx`` is imported lazily only when
    ``evaluate_blocking_with_ranking`` (MRR/NDCG/MAP) actually runs, so the
    module imports stay ranx-free.
    """
    result = subprocess.run(
        [sys.executable, "-c", _RANX_DECOUPLE_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"ranx-decoupling check failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


# ``langres.testing`` (the public ``ScriptedJudge`` test/example double) must
# stay out of the core import graph: a bare ``import langres`` should never
# pull it in (it's an explicit ``from langres.testing import ScriptedJudge``
# opt-in), and importing it directly must stay light -- no
# torch/litellm/faiss/sentence-transformers/scikit-learn/dspy/ranx. Same
# subprocess pattern as the ranx-decoupling check above, since the whole point
# is a fresh, unpolluted ``sys.modules``.
_TESTING_MODULE_NOT_EAGER_SCRIPT = (
    "import sys; import langres; "
    "assert 'langres.testing' not in sys.modules, "
    "'langres.testing was eagerly imported by a bare import langres'; "
    "print('OK')"
)


def test_import_langres_does_not_eagerly_import_testing_module() -> None:
    """A bare ``import langres`` must not pull in ``langres.testing``."""
    result = subprocess.run(
        [sys.executable, "-c", _TESTING_MODULE_NOT_EAGER_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"langres.testing eager-import check failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


_TESTING_MODULE_HEAVY_DEPS = [
    "torch",
    "litellm",
    "faiss",
    "sentence_transformers",
    "sklearn",
    "dspy",
    "ranx",
]

_TESTING_MODULE_IMPORT_LIGHT_SCRIPT = (
    "import sys; import langres.testing; "
    "leaked = [m for m in {modules!r} if m in sys.modules]; "
    "assert not leaked, f'langres.testing leaked heavy modules: {{leaked}}'; "
    "print('OK')"
).format(modules=_TESTING_MODULE_HEAVY_DEPS)


def test_import_langres_testing_stays_import_light() -> None:
    """``import langres.testing`` alone must not pull the heavy/optional stack."""
    result = subprocess.run(
        [sys.executable, "-c", _TESTING_MODULE_IMPORT_LIGHT_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"langres.testing import-budget check failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


# The EvalReport tearsheet (``langres.report.eval_report``) and its SVG backend
# (``langres.report._svg``) render entirely from stdlib + numpy (a core dep). They
# must NEVER pull the heavy/optional stack -- that is the permanent guarantee a
# $0 report can always be built on a bare core-only install (no torch, no
# matplotlib, no litellm). Same fresh-process subprocess pattern as above.
_EVAL_REPORT_HEAVY_DEPS = [
    "torch",
    "litellm",
    "faiss",
    "sentence_transformers",
    "sklearn",
    "ranx",
    "mlflow",
    "wandb",
    "matplotlib",
    "dspy",
]

_EVAL_REPORT_IMPORT_LIGHT_SCRIPT = (
    "import sys; import langres.report.eval_report; import langres.report._svg; "
    "leaked = [m for m in {modules!r} if m in sys.modules]; "
    "assert not leaked, f'eval_report/_svg leaked heavy modules: {{leaked}}'; "
    "print('OK')"
).format(modules=_EVAL_REPORT_HEAVY_DEPS)


def test_eval_report_stays_import_light() -> None:
    """``import langres.report.eval_report`` (+ ``_svg``) must not pull a heavy dep.

    The tearsheet is dependency-free by construction: inline SVG, no matplotlib,
    no ML stack. This locks it so a future edit can never regress the $0,
    core-only path.
    """
    result = subprocess.run(
        [sys.executable, "-c", _EVAL_REPORT_IMPORT_LIGHT_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"eval_report import-budget check failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


# The data-profile report (``langres.data.data_profile``) and its shared render
# scaffold (``langres.report._report_html``) render entirely from stdlib + numpy (a
# core dep) + the import-light ``core.metrics``. Like the EvalReport tearsheet
# they must NEVER pull the heavy/optional stack -- the plan's load-bearing
# guarantee that a ``$0`` data profile is buildable on a bare core-only install
# (no torch, no sentence-transformers, no matplotlib). The report *consumes*
# precomputed embeddings; it never generates them, so it carries no [semantic]
# dep. Same fresh-process subprocess pattern as the eval_report check above.
_DATA_PROFILE_IMPORT_LIGHT_SCRIPT = (
    "import sys; import langres.data.data_profile; import langres.report._report_html; "
    "leaked = [m for m in {modules!r} if m in sys.modules]; "
    "assert not leaked, f'data_profile/_report_html leaked heavy modules: {{leaked}}'; "
    "print('OK')"
).format(modules=_EVAL_REPORT_HEAVY_DEPS)


def test_data_profile_stays_import_light() -> None:
    """``import langres.data.data_profile`` (+ ``_report_html``) must not pull a heavy dep.

    The data profile is dependency-free by construction: inline SVG, no
    matplotlib, no ML stack, no embedding generation (embeddings are consumed
    precomputed). This locks it so a future edit can never regress the $0,
    core-only path.
    """
    result = subprocess.run(
        [sys.executable, "-c", _DATA_PROFILE_IMPORT_LIGHT_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"data_profile import-budget check failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


# ``langres.data.registry.list_methods`` is a public, import-light discovery API
# (exported from ``langres.data``): it must return the method NAMES without
# pulling ``langres.methods`` — which imports VectorBlocker / RandomForestMatcher /
# EmbeddingScoreMatcher at module scope, dragging in faiss / scikit-learn /
# sentence-transformers. The names live in the ``langres._method_names`` leaf so
# a core-only (or ``[semantic]``-only) user can list them. Subprocess-based for a
# fresh import state (this pytest process is already polluted by other tests).
_LIST_METHODS_SCRIPT = (
    "import sys; import langres.data.registry as r; r.list_methods(); "
    "assert 'langres.methods' not in sys.modules, "
    "'list_methods pulled langres.methods (the heavy dispatch module)'; "
    "leaked = [m for m in ['faiss', 'sklearn', 'sentence_transformers'] if m in sys.modules]; "
    "assert not leaked, f'list_methods leaked heavy modules: {leaked}'; "
    "print('OK')"
)


def test_registry_list_methods_stays_import_light() -> None:
    """``registry.list_methods()`` must not pull ``langres.methods`` or the heavy stack.

    Guards the P2 fix: method NAMES come from the import-light
    ``langres._method_names`` leaf, so name-listing is safe in a core-only /
    partial-extras install even though ``langres.methods`` (dispatch) imports the
    ``[semantic]``/``[trained]`` stack at module scope.
    """
    result = subprocess.run(
        [sys.executable, "-c", _LIST_METHODS_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"list_methods import-budget check failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


_TRACKING_MODULES = ["ranx", "mlflow", "wandb"]

_TRACKING_CHECK_SCRIPT = (
    "import sys; import langres; "
    "leaked = [m for m in {modules!r} if m in sys.modules]; "
    "assert not leaked, f'tracking deps leaked into sys.modules: {{leaked}}'; "
    "print('OK')"
).format(modules=_TRACKING_MODULES)


def test_import_langres_excludes_tracking_deps_from_sys_modules() -> None:
    """The S1 tracking layer must not pull ranx/mlflow/wandb on a bare import.

    ``core/runs.py`` refs the result models (which need ``ranx``) only under
    ``TYPE_CHECKING``, and the ``ExperimentTracker`` adapters import
    ``mlflow``/``wandb`` lazily -- so eager ``import langres`` stays clean.
    """
    result = subprocess.run(
        [sys.executable, "-c", _TRACKING_CHECK_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"tracking import-budget check failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


# The flywheel's back half is root-exported, but its three lazy names
# (``EvalReport``, ``gold_pairs_from_clusters``, ``derive_threshold``) must not
# drag their owning modules into a bare ``import langres``: ``report.eval_report``
# pulls ``core.benchmark``/``core.metrics`` (kept out of the eager graph on
# purpose), and ``core.calibration`` imports scikit-learn ([trained]) at module
# scope. Same fresh-process subprocess pattern as above.
#
# !! THIS LIST IS A DENY LIST, SO IT ROTS OPEN !! Every entry asserts a module is
# ABSENT from ``sys.modules``, which a module that no longer exists satisfies
# trivially. A stale path here does not fail -- it silently stops guarding, and
# the test stays green while checking nothing. When a module named here moves,
# the ONLY signal is this comment. Re-point it; never just delete the entry.
# (Caught for real: the `report` extraction left `langres.core.eval_report` here,
# passing vacuously against a module that had ceased to exist.)
_ROOT_LAZY_MODULES = [
    "langres.report.eval_report",
    "langres.core.benchmark",
    "langres.core.calibration",
    "langres.data.data_profile",
]

_ROOT_LAZY_SCRIPT = (
    "import sys; import langres; "
    "leaked = [m for m in {modules!r} if m in sys.modules]; "
    "assert not leaked, f'lazy root-export modules were eagerly imported: {{leaked}}'; "
    "print('OK')"
).format(modules=_ROOT_LAZY_MODULES)


def test_import_langres_does_not_eagerly_import_lazy_root_export_modules() -> None:
    """The lazy root exports must stay lazy: bare import pulls none of their modules."""
    result = subprocess.run(
        [sys.executable, "-c", _ROOT_LAZY_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"root-export laziness check failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


# The finetune surface ([finetune], PR-F) must be import-light: resolving the
# ``langres.finetune``/``QLoRA`` symbols and importing ``langres.core.finetune``
# must NOT pull the training stack (peft/trl/bitsandbytes) or torch/transformers --
# they load lazily only inside ``QLoRATrainer.train``. So a core+[llm] user can
# reference the symbols, build a ``QLoRA(...)`` spec, and inject a custom trainer
# without the (Linux-only, heavy) QLoRA deps installed. Fresh-process subprocess
# for an unpolluted ``sys.modules``.
_FINETUNE_MODULES = ["peft", "trl", "bitsandbytes", "torch", "transformers"]

_FINETUNE_IMPORT_LIGHT_SCRIPT = (
    "import sys; import langres; "
    "langres.finetune; langres.QLoRA; langres.run_finetune; "
    "import langres.core.finetune; "
    "leaked = [m for m in {modules!r} if m in sys.modules]; "
    "assert not leaked, f'finetune surface leaked the training stack: {{leaked}}'; "
    "print('OK')"
).format(modules=_FINETUNE_MODULES)


def test_finetune_surface_stays_import_light() -> None:
    """``langres.finetune``/``QLoRA`` + ``core.finetune`` must not pull peft/trl/torch."""
    result = subprocess.run(
        [sys.executable, "-c", _FINETUNE_IMPORT_LIGHT_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"finetune import-budget check failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_import_langres_is_fast() -> None:
    """Soft timing budget: a warm ``import langres`` should be well under a second.

    Not a hard CI gate (machine-dependent) -- records the actual number on
    failure so a regression is visible rather than silently tolerated.
    """
    script = (
        "import time; t0 = time.perf_counter(); import langres; print(time.perf_counter() - t0)"
    )
    # Warm-up run (module bytecode cache, filesystem cache) so the measured
    # run reflects steady-state import cost, not first-ever compilation.
    subprocess.run([sys.executable, "-c", "import langres"], capture_output=True, text=True)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    elapsed = float(result.stdout.strip())
    assert elapsed < 2.0, f"import langres took {elapsed:.3f}s (budget: 2.0s)"


# ---------------------------------------------------------------------------
# Spend safety: import must never populate OPENROUTER_API_KEY/etc. from a
# nearby .env (the litellm load_dotenv() footgun).
# ---------------------------------------------------------------------------


def test_import_langres_does_not_leak_env_from_dotenv(tmp_path: Path) -> None:
    """A ``.env`` with a real-looking key must not leak into ``os.environ``.

    Reproduces the actual footgun: litellm's import runs ``load_dotenv()``,
    which walks up the directory tree from cwd looking for a ``.env`` file
    and populates ``os.environ`` from it -- independent of any matcher= choice.
    Runs the subprocess with cwd set to a directory containing exactly that
    kind of ``.env``; since litellm must stay out of sys.modules (previous
    test), the key must never appear.

    Explicitly strips ``OPENROUTER_API_KEY`` from the child's inherited
    environment first: an *earlier* test in this suite legitimately importing
    litellm (e.g. to test ``LLMMatcher`` directly) can itself trigger
    ``load_dotenv()`` in THIS pytest process, populating the real key from the
    repo's own ``.env`` a few directories up -- that's a pre-existing fact of
    running this suite locally with a real key configured, not a regression
    this test is checking for. Isolating it here keeps the assertion about
    ``import langres`` specifically, not about the ambient dev environment.
    """
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-test-canary-should-not-leak\n")
    child_env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    script = (
        "import os; import langres; "
        "assert 'OPENROUTER_API_KEY' not in os.environ, "
        "'OPENROUTER_API_KEY leaked into os.environ via import langres'; "
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=child_env,
    )
    assert result.returncode == 0, (
        f"spend-safety check failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Lazy attribute resolution (PEP 562 __getattr__) -- in-process, exercising
# the actual resolution/caching/error paths of each __getattr__.
# ---------------------------------------------------------------------------


#: Implementations ``langres.core`` deliberately no longer re-exports, and the
#: package that owns each one now. Data for
#: ``TestCoreLazyGetattr::test_implementations_are_not_re_exported``.
_MOVED_OFF_THE_FACADE: dict[str, str] = {
    "AllPairsBlocker": "langres.core.blockers",
    "CompositeBlocker": "langres.core.blockers",
    "KeyBlocker": "langres.core.blockers",
    "VectorBlocker": "langres.core.blockers",
    "StringComparator": "langres.core.comparators",
    "AnchorStore": "langres.core.anchor_store",
    "Canonicalizer": "langres.core.canonicalizer",
    "CorrelationClusterer": "langres.core.clusterers",
    "CascadeMatcher": "langres.core.matchers",
    "EmbeddingScoreMatcher": "langres.core.matchers",
    "WeightedAverageMatcher": "langres.core.matchers",
    "LLMMatcher": "langres.core.matchers",
    "RandomForestMatcher": "langres.core.matchers.random_forest_judge",
    "SelectMatcher": "langres.core.matchers.select_judge",
    "SentenceTransformerEmbedder": "langres.core.embeddings",
    "FakeEmbedder": "langres.core.embeddings",
    "FAISSIndex": "langres.core.indexes",
    "VectorIndex": "langres.core.indexes",
    "PipelineDebugger": "langres.core.debugging",
}


class TestCoreLazyGetattr:
    """``langres.core.__getattr__`` for the optional-extra symbols left on the facade.

    ``langres.core`` carries **contracts** only, so the lazy names here are the
    contract-adjacent handful that still need an extra: ``Calibrator``
    (scikit-learn, ``[trained]``) and the ``MlflowTracker``/``WandbTracker``
    adapters. The *implementations* keep their own lazy seams in the packages
    that own them -- covered by ``TestBlockersPackageLazyGetattr`` and
    ``TestModulesPackageLazyGetattr`` below, which assert exactly the
    resolve/cache/actionable-ImportError behaviour this class used to assert
    for the same classes via the facade.
    """

    def test_calibrator_resolves_and_caches(self) -> None:
        pytest.importorskip("sklearn", reason="requires the [trained] extra")
        import langres.core as core

        cal = core.Calibrator
        from langres.core.calibration import Calibrator

        assert cal is Calibrator
        # Cached on the module namespace -- a second access must not re-hit
        # __getattr__ (it's now a plain module attribute).
        assert core.__dict__["Calibrator"] is Calibrator

    def test_implementations_are_not_re_exported(self) -> None:
        """The facade is contracts-only: an implementation must NOT resolve here.

        This is the regression guard for the facade-emptying wave -- re-adding
        any of these to a ``core/_exports`` fragment puts ``langres.core`` back
        above the components it sits beneath and re-knots the import graph
        (``tests/test_import_tangle.py`` is the ratchet that measures the cost).
        """
        import langres.core as core

        for name, owner in _MOVED_OFF_THE_FACADE.items():
            assert name not in core.__all__, (
                f"{name} is back in langres.core.__all__ -- it is an implementation "
                f"and belongs to {owner}"
            )
            with pytest.raises(AttributeError, match=name):
                getattr(core, name)

    def test_contracts_are_still_re_exported(self) -> None:
        """The other half of the split: the contracts stay on the facade."""
        import langres.core as core

        for name in (
            "Blocker",
            "Comparator",
            "Matcher",
            "GroupwiseMatcher",
            "Clusterer",
            "Resolver",
            "ERCandidate",
            "PairwiseJudgement",
            "register",
        ):
            assert name in core.__all__, f"{name} is a contract and must stay on langres.core"
            assert getattr(core, name) is not None

    def test_calibrator_raises_actionable_import_error_when_trained_absent(self) -> None:
        """Core-only install (no [trained]): a real, un-simulated ImportError.

        Unlike ``test_missing_dependency_raises_actionable_import_error``
        below (which *simulates* absence via monkeypatching so it also runs
        when the extra IS installed), this exercises the genuine failure path
        -- meaningful only in a core-only environment, so it skips itself when
        scikit-learn is actually importable.
        """
        if _import_ok("sklearn"):
            pytest.skip("scikit-learn is installed ([trained] extra present) -- nothing to observe")
        import langres.core as core

        with pytest.raises(ImportError, match=r"langres\.core\.Calibrator.*langres\[trained\]"):
            core.Calibrator  # noqa: B018

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        import langres.core as core

        with pytest.raises(AttributeError, match="not_a_real_attribute"):
            core.not_a_real_attribute  # noqa: B018

    def test_missing_dependency_raises_actionable_import_error(self, monkeypatch) -> None:
        """A missing [trained] package surfaces a 'pip install' hint, not a raw traceback."""
        import importlib

        import langres.core as core

        # __getattr__ caches a successful resolution onto the module namespace
        # (see its docstring) -- an earlier test in this file may have already
        # resolved and cached Calibrator, which would make this access skip
        # __getattr__ entirely. Clear it so the patched import is actually hit.
        monkeypatch.delitem(vars(core), "Calibrator", raising=False)

        real_import_module = importlib.import_module

        def _fail_for_sklearn(name: str, *args: object, **kwargs: object) -> object:
            if name == "langres.core.calibration":
                raise ModuleNotFoundError("No module named 'sklearn'")
            return real_import_module(name, *args, **kwargs)

        monkeypatch.setattr(core.importlib, "import_module", _fail_for_sklearn)
        with pytest.raises(ImportError, match=r"langres\.core\.Calibrator.*langres\[trained\]"):
            core.Calibrator  # noqa: B018


class TestRootLazyGetattr:
    """``langres.__getattr__``: the flywheel root exports (paved road).

    The loop's back half is importable from ``langres`` directly -- eagerly
    where free (the harvest surface is already in the eager import graph via
    ``langres.core``), lazily where it would add import weight (see
    ``test_import_langres_does_not_eagerly_import_lazy_root_export_modules``).
    """

    def test_harvest_surface_is_eagerly_root_exported(self) -> None:
        import langres
        from langres.core import harvest

        assert langres.Correction is harvest.Correction
        assert langres.CorrectionLog is harvest.CorrectionLog
        assert langres.harvest_labeled_pairs is harvest.harvest_labeled_pairs
        assert langres.derive_threshold_from_pairs is harvest.derive_threshold_from_pairs

    def test_eval_report_resolves_and_caches(self) -> None:
        import langres
        from langres.report.eval_report import EvalReport

        assert langres.EvalReport is EvalReport
        # Cached on the module namespace -- a second access must not re-hit
        # __getattr__ (it's now a plain module attribute).
        assert vars(langres)["EvalReport"] is EvalReport

    def test_gold_pairs_from_clusters_resolves(self) -> None:
        import langres
        from langres.core.benchmark import gold_pairs_from_clusters

        assert langres.gold_pairs_from_clusters is gold_pairs_from_clusters

    def test_derive_threshold_resolves(self) -> None:
        pytest.importorskip("sklearn", reason="requires the [trained] extra")
        import langres

        from langres.core.calibration import derive_threshold

        assert langres.derive_threshold is derive_threshold

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        import langres

        with pytest.raises(AttributeError, match="not_a_real_attribute"):
            langres.not_a_real_attribute  # noqa: B018

    def test_derive_threshold_missing_sklearn_raises_actionable_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing [trained] extra surfaces a 'pip install' hint, not a raw traceback."""
        import langres

        # Clear any cached resolution from an earlier test so __getattr__ runs.
        monkeypatch.delitem(vars(langres), "derive_threshold", raising=False)

        def _fail(name: str) -> object:
            raise ModuleNotFoundError("No module named 'sklearn'")

        monkeypatch.setattr(langres.importlib, "import_module", _fail)
        with pytest.raises(ImportError, match=r"langres\.derive_threshold.*langres\[trained\]"):
            langres.derive_threshold  # noqa: B018

    def test_no_extra_symbol_propagates_import_error_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """EvalReport needs no extra: an ImportError from it is a genuine bug
        and must propagate as-is, never dressed up as a 'pip install' hint."""
        import langres

        monkeypatch.delitem(vars(langres), "EvalReport", raising=False)

        def _fail(name: str) -> object:
            raise ModuleNotFoundError("No module named 'nonsense_dep'")

        monkeypatch.setattr(langres.importlib, "import_module", _fail)
        with pytest.raises(ModuleNotFoundError, match="nonsense_dep"):
            langres.EvalReport  # noqa: B018


class TestClientsLazyGetattr:
    """``langres.clients.__getattr__`` for ``create_llm_client``/``create_wandb_tracker``."""

    def test_create_llm_client_resolves(self) -> None:
        pytest.importorskip("litellm", reason="requires the [llm] extra")
        import langres.clients as clients

        from langres.clients.llm import create_llm_client

        assert clients.create_llm_client is create_llm_client

    def test_create_wandb_tracker_resolves(self) -> None:
        import langres.clients as clients

        from langres.clients.tracking import create_wandb_tracker

        assert clients.create_wandb_tracker is create_wandb_tracker

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        import langres.clients as clients

        with pytest.raises(AttributeError, match="not_a_real_attribute"):
            clients.not_a_real_attribute  # noqa: B018


class TestBlockersPackageLazyGetattr:
    """``langres.core.blockers.__getattr__`` for ``VectorBlocker``."""

    def test_vector_blocker_resolves_via_package_path(self) -> None:
        pytest.importorskip("faiss", reason="requires the [semantic] extra")
        import langres.core.blockers as blockers

        from langres.core.blockers.vector import VectorBlocker

        assert blockers.VectorBlocker is VectorBlocker
        # Cached on the module namespace -- a second access must not re-hit
        # __getattr__ (matches core/clients/modules' lazy loaders).
        assert blockers.__dict__["VectorBlocker"] is VectorBlocker

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        import langres.core.blockers as blockers

        with pytest.raises(AttributeError, match="not_a_real_attribute"):
            blockers.not_a_real_attribute  # noqa: B018


class TestModulesPackageLazyGetattr:
    """``langres.core.matchers.__getattr__`` for LLMMatcher/LLMMatcher/CascadeChainMatcher."""

    def test_llm_judge_resolves_via_package_path(self) -> None:
        pytest.importorskip("litellm", reason="requires the [llm] extra")
        import langres.core.matchers as modules_pkg

        from langres.core.matchers.llm_judge import LLMMatcher

        assert modules_pkg.LLMMatcher is LLMMatcher

    def test_llm_judge_module_resolves_via_package_path(self) -> None:
        pytest.importorskip("litellm", reason="requires the [llm] extra")
        import langres.core.matchers as modules_pkg

        from langres.core.matchers.llm_judge import LLMMatcher

        assert modules_pkg.LLMMatcher is LLMMatcher

    def test_cascade_module_resolves_via_package_path(self) -> None:
        pytest.importorskip("litellm", reason="requires the [llm] extra")
        pytest.importorskip("sentence_transformers", reason="requires the [semantic] extra")
        import langres.core.matchers as modules_pkg

        from langres.core.matchers.cascade import CascadeChainMatcher

        assert modules_pkg.CascadeChainMatcher is CascadeChainMatcher

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        import langres.core.matchers as modules_pkg

        with pytest.raises(AttributeError, match="not_a_real_attribute"):
            modules_pkg.not_a_real_attribute  # noqa: B018

    def test_missing_dependency_raises_actionable_import_error(self, monkeypatch) -> None:
        import langres.core.matchers as modules_pkg

        # See the analogous comment in TestCoreLazyGetattr: clear any cached
        # resolution from an earlier test so __getattr__ actually runs.
        monkeypatch.delitem(vars(modules_pkg), "LLMMatcher", raising=False)

        def _fail(name: str) -> object:
            raise ModuleNotFoundError("No module named 'litellm'")

        monkeypatch.setattr(modules_pkg.importlib, "import_module", _fail)
        with pytest.raises(ImportError, match=r"langres\.core\.matchers\.LLMMatcher.*llm"):
            modules_pkg.LLMMatcher  # noqa: B018
