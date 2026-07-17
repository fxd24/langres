"""W0 legacy save/load parity: the frozen old-format artifact still loads and runs.

The committed ``legacy_artifact_v1/`` directory holds two models saved in the
CURRENT (pre-#193) on-disk format -- a base ``Resolver.from_schema`` string
pipeline and a named ``FuzzyString`` architecture. A later wave changes that
format; its legacy-load adapter MUST still reconstruct these artifacts and
reproduce their output. Because the bytes cannot be regenerated after the format
changes (that is the whole point), they are committed as-is and this test only
*reads* them -- see :mod:`tests.parity._build_legacy_artifact` for how they were
produced.

Importing :mod:`tests.parity._fixture_records` registers the artifact's schema
(``ParityBusinessW0``) under the name the ``AllPairsBlocker`` config stores, so
``Resolver.load`` can rebuild the blocker in a fresh process.

Regenerate the *expected output* goldens (not the artifacts) with::

    LANGRES_PARITY_UPDATE=1 uv run pytest tests/parity --no-cov
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from langres.core.resolver import Resolver
from langres.metrics.metrics import calculate_bcubed_metrics

# Import for its registration side effect (schema) as well as the records/gold.
from tests.parity import _fixture_records  # noqa: F401  (registers ParityBusinessW0)
from tests.parity._fixture_records import GOLD_CLUSTERS, RECORDS
from tests.parity._golden import canonical_clusters, check_golden

_ARTIFACT_ROOT = Path(__file__).parent / "legacy_artifact_v1"

# (artifact subdir, expected loaded class name, expected model_class in manifest).
_ARTIFACTS = [
    ("resolver_string", "ERModel", None),
    ("fuzzy_string", "FuzzyString", "fuzzy_string"),
]
_REQUIRED_SLOTS = {"blocker", "comparator", "module", "clusterer"}


@pytest.mark.parametrize(("subdir", "expected_class", "expected_model_class"), _ARTIFACTS)
def test_legacy_manifest_shape(
    subdir: str, expected_class: str, expected_model_class: str | None
) -> None:
    """The frozen resolver.json carries the version envelope + slot-tagged components.

    These are exactly the invariants a later wave's legacy-load adapter reads to
    map an old artifact onto the new layout, so pin them directly (not via a
    golden): an ``artifact_version``, and one component per required slot.
    """
    manifest = json.loads((_ARTIFACT_ROOT / subdir / "resolver.json").read_text())
    assert manifest["artifact_version"], "legacy artifact must carry an artifact_version"
    assert manifest["model_class"] == expected_model_class
    slots = {component["slot"] for component in manifest["components"]}
    assert _REQUIRED_SLOTS <= slots, f"missing slots: {_REQUIRED_SLOTS - slots}"


@pytest.mark.parametrize(("subdir", "expected_class", "expected_model_class"), _ARTIFACTS)
def test_legacy_load_and_dedupe(
    subdir: str, expected_class: str, expected_model_class: str | None
) -> None:
    """Load the frozen artifact and snapshot the dedupe it produces on the fixture.

    Proves old -> (current) load parity now, and gives a later wave the golden its
    old -> new legacy adapter must reproduce byte-for-byte.
    """
    loaded = Resolver.load(_ARTIFACT_ROOT / subdir)
    assert type(loaded).__name__ == expected_class
    result = loaded.dedupe(RECORDS)
    payload = {
        "loaded_class": type(loaded).__name__,
        "architecture": result.architecture,
        "backbone": result.backbone,
        "score_type": result.score_type,
        "threshold": result.threshold,
        "clusters": canonical_clusters(result),
        "bcubed": calculate_bcubed_metrics(list(result), GOLD_CLUSTERS),
    }
    check_golden(f"legacy_load_{subdir}", payload)
