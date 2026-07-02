"""Regression: KeyBlocker/CompositeBlocker/CorrelationClusterer register on
plain ``import langres.core`` (W1.3).

``@register(...)`` only fires when a component's OWNING MODULE is imported
(see ``langres.core.registry``'s module docstring). Every registrable
component must therefore be eager-imported by either ``core/blockers/
__init__.py`` + ``core/__init__.py`` (or ``core/clusterers/__init__.py`` +
``core/__init__.py`` for clusterer variants) -- mirroring the existing
``AllPairsBlocker``/``VectorBlocker``/``LLMJudge`` pattern -- or listed in
``registry._LAZY_COMPONENT_MODULES``. Without that wiring, a fresh process
that does ``from langres.core import Resolver`` and then ``Resolver.load()``
on an artifact referencing ``"key_blocker"``/``"composite_blocker"``/
``"correlation_clusterer"`` raises ``UnknownComponentType`` -- even though
each test module for these three classes imports them directly at module
scope, which trivially (and misleadingly) registers them as a side effect
before any test body runs. This test spawns a genuinely fresh subprocess that
does NOT import any of the three submodules directly, so it can't be fooled
by that side effect.
"""

import subprocess
import sys

import pytest


@pytest.mark.slow
def test_new_w1_3_components_register_via_langres_core_alone() -> None:
    """A fresh process resolves key_blocker/composite_blocker/correlation_clusterer
    via ``from langres.core import get_component`` alone -- no submodule imports.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from langres.core import get_component\n"
            "for name in ('key_blocker', 'composite_blocker', 'correlation_clusterer'):\n"
            "    get_component(name)\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
