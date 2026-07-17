"""Regression: KeyBlocker/CompositeBlocker/CorrelationClusterer register on
plain ``import langres.core`` (W1.3).

``@register(...)`` only fires when a component's OWNING MODULE is imported
(see ``langres.core.registry``'s module docstring). Every registrable
component must therefore be eager-imported by either ``core/blockers/
__init__.py`` + ``core/__init__.py`` (or ``core/clusterers/__init__.py`` +
``core/__init__.py`` for clusterer variants) -- mirroring the existing
``AllPairsBlocker``/``VectorBlocker``/``LLMMatcher`` pattern -- or listed in
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
from pathlib import Path

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


def test_lazy_component_modules_includes_new_w1_3_components() -> None:
    """key_blocker/composite_blocker/correlation_clusterer resolve via the lazy
    registry path (``registry._LAZY_COMPONENT_MODULES``), not only via
    ``core/__init__.py``'s current eager imports (review finding, P2).

    Today's ``core/__init__.py`` happens to eager-import all three, so
    ``get_component`` already resolves them regardless of this dict. But
    ``_LAZY_COMPONENT_MODULES`` is the load-bearing mechanism once those eager
    imports go away (planned in W0.4, packaging-dx) -- mirroring the existing
    ``dspy_judge``/``select_judge`` entries, which are the only reason those two
    survive a leaner import surface. Without an entry here, a saved artifact
    referencing ``"key_blocker"``/``"composite_blocker"``/``"correlation_clusterer"``
    would raise ``UnknownComponentType`` on ``Resolver.load`` post-W0.4.
    """
    from langres.core.registry import _LAZY_COMPONENT_MODULES

    assert _LAZY_COMPONENT_MODULES["key_blocker"] == "langres.core.blockers.key"
    assert _LAZY_COMPONENT_MODULES["composite_blocker"] == "langres.core.blockers.composite"
    assert _LAZY_COMPONENT_MODULES["correlation_clusterer"] == "langres.core.clusterers.correlation"


@pytest.mark.slow
def test_resolver_fresh_process_roundtrip_with_composite_key_vector_blocker(
    tmp_path: Path,
) -> None:
    """Fresh-process save/load/resolve for a CompositeBlocker(KeyBlocker, VectorBlocker).

    Closes two things at once:

    - Portability (P2): the subprocess imports ONLY ``from langres.core import
      Resolver`` (never the blocker submodules directly) then ``Resolver.load``s
      an artifact whose blocker slot is ``"composite_blocker"`` wrapping a
      ``"key_blocker"`` and a ``"vector_blocker"`` child -- it must not raise
      ``UnknownComponentType``.
    - The nested-index sidecar sub-claim from the P1 finding: the reloaded
      Resolver's nested VectorBlocker starts with an unbuilt (freshly
      deserialized-config, no sidecar restore for a composite child) index, and
      ``resolve()`` must still succeed by building it transparently via
      ``Resolver._ensure_index_built``'s recursive traversal -- not just at the
      top level.
    """
    from langres.core import Clusterer, Resolver
    from langres.core.comparators import StringComparator
    from langres.core.blockers import CompositeBlocker, KeyBlocker, VectorBlocker
    from langres.core.embeddings import FakeEmbedder
    from langres.core.indexes import FAISSIndex
    from langres.core.matchers import WeightedAverageMatcher
    from langres.core.models import CompanySchema
    from tests.fixtures.companies import COMPANY_RECORDS

    index = FAISSIndex(embedder=FakeEmbedder(embedding_dim=32), metric="cosine")
    vector_blocker: VectorBlocker[CompanySchema] = VectorBlocker(
        vector_index=index, schema=CompanySchema, text_field="name", k_neighbors=5
    )
    key_blocker: KeyBlocker[CompanySchema] = KeyBlocker(schema=CompanySchema, key_field="address")
    composite: CompositeBlocker[CompanySchema] = CompositeBlocker(
        children=[key_blocker, vector_blocker], op="union"
    )
    comparator = StringComparator.from_schema(CompanySchema)
    resolver = Resolver(
        blocker=composite,
        comparator=comparator,
        matcher=WeightedAverageMatcher(feature_specs=comparator.feature_specs),
        clusterer=Clusterer(threshold=0.7),
    )
    resolver.resolve(COMPANY_RECORDS)  # builds the nested index before saving
    resolver.save(tmp_path)

    script = (
        "from langres.core import Resolver\n"
        "from tests.fixtures.companies import COMPANY_RECORDS\n"
        f"reloaded = Resolver.load({str(tmp_path)!r})\n"
        "clusters = reloaded.resolve(COMPANY_RECORDS)\n"
        "assert isinstance(clusters, list)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"fresh-process composite-blocker roundtrip failed.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "OK" in result.stdout


@pytest.mark.slow
def test_resolver_fresh_process_roundtrip_with_key_blocker_alone(tmp_path: Path) -> None:
    """Fresh-process save/load/resolve for a Resolver whose *sole* blocker is
    ``KeyBlocker`` (not nested inside a ``CompositeBlocker``).

    The composite roundtrip test above only exercises ``key_blocker`` as a
    composite CHILD -- ``Resolver.load`` reconstructs a composite's children
    via the same generic ``rebuild_component``, so that already proves
    ``get_component("key_blocker")`` resolves. This test additionally proves
    ``key_blocker`` resolves when it's the Resolver's top-level blocker slot
    (W0.4 exit check: every ``_LAZY_COMPONENT_MODULES`` entry must survive a
    fresh-process ``Resolver.load``, not just as an incidental nested case).
    """
    from langres.core import Clusterer, Resolver
    from langres.core.comparators import StringComparator
    from langres.core.blockers import KeyBlocker
    from langres.core.matchers import WeightedAverageMatcher
    from langres.core.models import CompanySchema
    from tests.fixtures.companies import COMPANY_RECORDS

    blocker: KeyBlocker[CompanySchema] = KeyBlocker(schema=CompanySchema, key_field="address")
    comparator = StringComparator.from_schema(CompanySchema)
    resolver = Resolver(
        blocker=blocker,
        comparator=comparator,
        matcher=WeightedAverageMatcher(feature_specs=comparator.feature_specs),
        clusterer=Clusterer(threshold=0.7),
    )
    resolver.resolve(COMPANY_RECORDS)
    resolver.save(tmp_path)

    script = (
        "from langres.core import Resolver\n"
        "from tests.fixtures.companies import COMPANY_RECORDS\n"
        f"reloaded = Resolver.load({str(tmp_path)!r})\n"
        "clusters = reloaded.resolve(COMPANY_RECORDS)\n"
        "assert isinstance(clusters, list)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"fresh-process key-blocker roundtrip failed.\nSTDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "OK" in result.stdout


@pytest.mark.slow
def test_resolver_fresh_process_roundtrip_with_correlation_clusterer(tmp_path: Path) -> None:
    """Fresh-process save/load/resolve for a Resolver whose clusterer slot is
    ``CorrelationClusterer`` (the merge-resistant pivot-algorithm variant).

    No existing test round-trips ``correlation_clusterer`` as an actual
    Resolver slot through a fresh subprocess -- only ``get_component`` alone
    (above) and an in-process ``config``/``from_config`` round trip
    (``tests/core/clusterers/test_correlation_clusterer.py``). This closes
    that gap for the W0.4 exit check.
    """
    from langres.core import Resolver
    from langres.core.blockers import AllPairsBlocker
    from langres.core.clusterers import CorrelationClusterer
    from langres.core.comparators import StringComparator
    from langres.core.matchers.weighted_average import WeightedAverageMatcher
    from langres.core.models import CompanySchema
    from tests.fixtures.companies import COMPANY_RECORDS

    comparator = StringComparator.from_schema(CompanySchema)
    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=comparator,
        matcher=WeightedAverageMatcher(feature_specs=comparator.feature_specs),
        clusterer=CorrelationClusterer(threshold=0.7),
    )
    resolver.resolve(COMPANY_RECORDS)
    resolver.save(tmp_path)

    script = (
        "from langres.core import Resolver\n"
        "from tests.fixtures.companies import COMPANY_RECORDS\n"
        f"reloaded = Resolver.load({str(tmp_path)!r})\n"
        "clusters = reloaded.resolve(COMPANY_RECORDS)\n"
        "assert isinstance(clusters, list)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"fresh-process correlation-clusterer roundtrip failed.\nSTDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "OK" in result.stdout
