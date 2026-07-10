"""Import weight + spend-safety tests (W0.4: extras split + lazy heavy imports).

Two concerns, both closed by making ``langres/__init__.py``'s import chain --
``langres.core``, ``langres.clients``, ``langres.core.blockers``,
``langres.core.modules`` -- resolve heavy/optional-dependency symbols lazily
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
    "litellm",
    "faiss",
    "sentence_transformers",
    "sklearn",
    # Tracking backends (S1): the ExperimentTracker adapters must load mlflow/
    # wandb lazily, never on a bare `import langres`.
    "mlflow",
    "wandb",
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


# The EvalReport tearsheet (``langres.core.eval_report``) and its SVG backend
# (``langres.core._svg``) render entirely from stdlib + numpy (a core dep). They
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
    "import sys; import langres.core.eval_report; import langres.core._svg; "
    "leaked = [m for m in {modules!r} if m in sys.modules]; "
    "assert not leaked, f'eval_report/_svg leaked heavy modules: {{leaked}}'; "
    "print('OK')"
).format(modules=_EVAL_REPORT_HEAVY_DEPS)


def test_eval_report_stays_import_light() -> None:
    """``import langres.core.eval_report`` (+ ``_svg``) must not pull a heavy dep.

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


# ``langres.data.registry.list_methods`` is a public, import-light discovery API
# (exported from ``langres.data``): it must return the method NAMES without
# pulling ``langres.methods`` — which imports VectorBlocker / RandomForestJudge /
# EmbeddingScoreJudge at module scope, dragging in faiss / scikit-learn /
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
    and populates ``os.environ`` from it -- independent of any judge= choice.
    Runs the subprocess with cwd set to a directory containing exactly that
    kind of ``.env``; since litellm must stay out of sys.modules (previous
    test), the key must never appear.

    Explicitly strips ``OPENROUTER_API_KEY`` from the child's inherited
    environment first: an *earlier* test in this suite legitimately importing
    litellm (e.g. to test ``LLMJudge`` directly) can itself trigger
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


class TestCoreLazyGetattr:
    """``langres.core.__getattr__`` for the [semantic]/[llm] symbols + submodules."""

    def test_vector_blocker_resolves_and_caches(self) -> None:
        pytest.importorskip("faiss", reason="requires the [semantic] extra")
        import langres.core as core

        vb = core.VectorBlocker
        from langres.core.blockers.vector import VectorBlocker

        assert vb is VectorBlocker
        # Cached on the module namespace -- a second access must not re-hit
        # __getattr__ (it's now a plain module attribute).
        assert core.__dict__["VectorBlocker"] is VectorBlocker

    def test_llm_judge_resolves(self) -> None:
        pytest.importorskip("litellm", reason="requires the [llm] extra")
        import langres.core as core

        from langres.core.modules.llm_judge import LLMJudge

        assert core.LLMJudge is LLMJudge

    def test_random_forest_judge_resolves(self) -> None:
        pytest.importorskip("sklearn", reason="requires the [trained] extra")
        import langres.core as core

        from langres.core.modules.random_forest_judge import RandomForestJudge

        assert core.RandomForestJudge is RandomForestJudge

    def test_select_judge_resolves(self) -> None:
        pytest.importorskip("dspy", reason="requires the [llm] extra")
        import langres.core as core

        from langres.core.modules.select_judge import SelectJudge

        assert core.SelectJudge is SelectJudge

    def test_embeddings_and_indexes_symbols_resolve(self) -> None:
        pytest.importorskip("sentence_transformers", reason="requires the [semantic] extra")
        pytest.importorskip("faiss", reason="requires the [semantic] extra")
        import langres.core as core

        from langres.core.embeddings import SentenceTransformerEmbedder
        from langres.core.indexes.vector_index import FAISSIndex

        assert core.SentenceTransformerEmbedder is SentenceTransformerEmbedder
        assert core.FAISSIndex is FAISSIndex

    def test_vector_blocker_raises_actionable_import_error_when_semantic_absent(self) -> None:
        """Core-only install (no [semantic]): a real, un-simulated ImportError.

        Unlike ``test_missing_dependency_raises_actionable_import_error``
        below (which *simulates* absence via monkeypatching so it also runs
        when the extra IS installed), this exercises the genuine failure path
        -- meaningful only in a core-only environment, so it skips itself when
        faiss is actually importable.
        """
        if _import_ok("faiss"):
            pytest.skip("faiss is installed ([semantic] extra present) -- nothing to observe")
        import langres.core as core

        with pytest.raises(ImportError, match=r"langres\.core\.VectorBlocker.*langres\[semantic\]"):
            core.VectorBlocker  # noqa: B018

    def test_llm_judge_raises_actionable_import_error_when_llm_absent(self) -> None:
        """Core-only install (no [llm]): a real, un-simulated ImportError (see above)."""
        if _import_ok("litellm"):
            pytest.skip("litellm is installed ([llm] extra present) -- nothing to observe")
        import langres.core as core

        with pytest.raises(ImportError, match=r"langres\.core\.LLMJudge.*langres\[llm\]"):
            core.LLMJudge  # noqa: B018

    def test_random_forest_judge_raises_actionable_import_error_when_trained_absent(self) -> None:
        """Core-only install (no [trained]): a real, un-simulated ImportError (see above)."""
        if _import_ok("sklearn"):
            pytest.skip("scikit-learn is installed ([trained] extra present) -- nothing to observe")
        import langres.core as core

        with pytest.raises(
            ImportError, match=r"langres\.core\.RandomForestJudge.*langres\[trained\]"
        ):
            core.RandomForestJudge  # noqa: B018

    def test_select_judge_raises_actionable_import_error_when_llm_absent(self) -> None:
        """Core-only install (no [llm]): a real, un-simulated ImportError (see above)."""
        if _import_ok("dspy"):
            pytest.skip("dspy is installed ([llm] extra present) -- nothing to observe")
        import langres.core as core

        with pytest.raises(ImportError, match=r"langres\.core\.SelectJudge.*langres\[llm\]"):
            core.SelectJudge  # noqa: B018

    def test_submodules_resolve_to_the_module_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Attribute access resolves each submodule via ``__getattr__``.

        Python's import machinery binds ``core.benchmark`` as a plain
        attribute the moment *anything* does ``import langres.core.benchmark``
        directly, bypassing ``__getattr__`` entirely -- and the wider test
        suite legitimately does exactly that elsewhere (its own tests import
        these submodules directly). Clearing the cached attribute first (same
        pattern as the missing-dependency tests below) forces this access to
        actually go through ``__getattr__``'s ``_LAZY_SUBMODULES`` branch
        regardless of what already ran earlier in the suite.
        """
        import langres.core as core
        import langres.core.benchmark as benchmark_mod
        import langres.core.metrics as metrics_mod
        import langres.core.optimizers as optimizers_mod

        monkeypatch.delitem(vars(core), "benchmark", raising=False)
        monkeypatch.delitem(vars(core), "metrics", raising=False)
        monkeypatch.delitem(vars(core), "optimizers", raising=False)

        assert core.benchmark is benchmark_mod
        assert core.metrics is metrics_mod
        assert core.optimizers is optimizers_mod

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        import langres.core as core

        with pytest.raises(AttributeError, match="not_a_real_attribute"):
            core.not_a_real_attribute  # noqa: B018

    def test_missing_dependency_raises_actionable_import_error(self, monkeypatch) -> None:
        """A missing [semantic]/[llm] package surfaces a 'pip install' hint, not a raw traceback."""
        import importlib

        import langres.core as core

        # __getattr__ caches a successful resolution onto the module namespace
        # (see its docstring) -- an earlier test in this file may have already
        # resolved and cached FAISSIndex, which would make this access skip
        # __getattr__ entirely. Clear it so the patched import is actually hit.
        monkeypatch.delitem(vars(core), "FAISSIndex", raising=False)

        real_import_module = importlib.import_module

        def _fail_for_faiss(name: str, *args: object, **kwargs: object) -> object:
            if name == "langres.core.indexes":
                raise ModuleNotFoundError("No module named 'faiss'")
            return real_import_module(name, *args, **kwargs)

        monkeypatch.setattr(core.importlib, "import_module", _fail_for_faiss)
        with pytest.raises(ImportError, match=r"langres\.core\.FAISSIndex.*langres\[semantic\]"):
            core.FAISSIndex  # noqa: B018


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
    """``langres.core.modules.__getattr__`` for LLMJudge/LLMJudgeModule/CascadeModule."""

    def test_llm_judge_resolves_via_package_path(self) -> None:
        pytest.importorskip("litellm", reason="requires the [llm] extra")
        import langres.core.modules as modules_pkg

        from langres.core.modules.llm_judge import LLMJudge

        assert modules_pkg.LLMJudge is LLMJudge

    def test_llm_judge_module_resolves_via_package_path(self) -> None:
        pytest.importorskip("litellm", reason="requires the [llm] extra")
        import langres.core.modules as modules_pkg

        from langres.core.modules.llm_judge import LLMJudgeModule

        assert modules_pkg.LLMJudgeModule is LLMJudgeModule

    def test_cascade_module_resolves_via_package_path(self) -> None:
        pytest.importorskip("litellm", reason="requires the [llm] extra")
        pytest.importorskip("sentence_transformers", reason="requires the [semantic] extra")
        import langres.core.modules as modules_pkg

        from langres.core.modules.cascade import CascadeModule

        assert modules_pkg.CascadeModule is CascadeModule

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        import langres.core.modules as modules_pkg

        with pytest.raises(AttributeError, match="not_a_real_attribute"):
            modules_pkg.not_a_real_attribute  # noqa: B018

    def test_missing_dependency_raises_actionable_import_error(self, monkeypatch) -> None:
        import langres.core.modules as modules_pkg

        # See the analogous comment in TestCoreLazyGetattr: clear any cached
        # resolution from an earlier test so __getattr__ actually runs.
        monkeypatch.delitem(vars(modules_pkg), "LLMJudge", raising=False)

        def _fail(name: str) -> object:
            raise ModuleNotFoundError("No module named 'litellm'")

        monkeypatch.setattr(modules_pkg.importlib, "import_module", _fail)
        with pytest.raises(ImportError, match=r"langres\.core\.modules\.LLMJudge.*llm"):
            modules_pkg.LLMJudge  # noqa: B018
