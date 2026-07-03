"""Regression: FellegiSunterJudge/RFJudge register on plain ``import langres.core``
(W1.2) AND resolve via the lazy registry path.

``@register(...)`` only fires when a component's OWNING MODULE is imported
(see ``langres.core.registry``'s module docstring). Every registrable
component must therefore be eager-imported by ``core/__init__.py`` (mirroring
the existing ``AllPairsBlocker``/``VectorBlocker``/``LLMJudge`` pattern) *or*
listed in ``registry._LAZY_COMPONENT_MODULES``. Without either, a fresh
process that does ``from langres.core import Resolver`` and then
``Resolver.load()`` on an artifact referencing ``"fellegi_sunter_judge"``/
``"rf_judge"`` raises ``UnknownComponentType``.

Mirrors ``tests/core/test_w1_3_registry_wiring.py`` (the identical P2 review
finding that landed for KeyBlocker/CompositeBlocker/CorrelationClusterer):
today's ``core/__init__.py`` happens to eager-import both judges, so
``get_component`` already resolves them regardless of ``_LAZY_COMPONENT_MODULES``.
But that dict is the load-bearing mechanism once W0.4 (packaging-dx) makes
those eager imports lazy -- mirroring the existing ``dspy_judge``/
``select_judge``/``key_blocker``/``composite_blocker``/``correlation_clusterer``
entries, which are the only reason those five survive a leaner import surface.
Dedicated E12 fresh-process save/load round-trip tests for each judge already
live in ``tests/core/judges/test_fellegi_sunter_judge.py`` and
``tests/core/modules/test_rf_judge.py`` -- this file only covers the
``get_component``-alone and ``_LAZY_COMPONENT_MODULES`` claims, not repeated
here.
"""

import subprocess
import sys

import pytest


@pytest.mark.slow
def test_new_w1_2_components_register_via_langres_core_alone() -> None:
    """A fresh process resolves fellegi_sunter_judge/rf_judge via
    ``from langres.core import get_component`` alone -- no submodule imports.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from langres.core import get_component\n"
            "for name in ('fellegi_sunter_judge', 'rf_judge'):\n"
            "    get_component(name)\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_lazy_component_modules_includes_new_w1_2_components() -> None:
    """fellegi_sunter_judge/rf_judge resolve via the lazy registry path
    (``registry._LAZY_COMPONENT_MODULES``), not only via ``core/__init__.py``'s
    current eager imports (review finding, P2 -- same class as W1.3's).

    Without an entry here, a saved artifact referencing
    ``"fellegi_sunter_judge"``/``"rf_judge"`` would raise
    ``UnknownComponentType`` on ``Resolver.load`` post-W0.4.
    """
    from langres.core.registry import _LAZY_COMPONENT_MODULES

    assert _LAZY_COMPONENT_MODULES["fellegi_sunter_judge"] == "langres.core.judges.fellegi_sunter"
    assert _LAZY_COMPONENT_MODULES["rf_judge"] == "langres.core.modules.rf_judge"
