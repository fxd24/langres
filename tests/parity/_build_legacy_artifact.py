"""Build the frozen ``legacy_artifact_v1/`` fixtures. **Run once, in W0.**

These artifacts capture the CURRENT (pre-#193-rewrite) on-disk ``save`` format:
``resolver.json`` (an ``ArtifactManifest`` with ``artifact_version`` + slot-tagged
``blocker``/``comparator``/``module``/``clusterer`` components) plus any sidecars.
A later wave changes that format; this is the "old format" artifact its
legacy-load adapter MUST still load. It CANNOT be regenerated once the format
changes, so it is committed as bytes and this script exists only as the record of
how it was produced.

Do **not** re-run this after the format has changed -- doing so would overwrite
the frozen old-format bytes with new-format ones and defeat the entire net.

Run: ``uv run python tests/parity/_build_legacy_artifact.py``
"""

from __future__ import annotations

import shutil
from pathlib import Path

from langres.architectures import FuzzyString
from langres.core.resolver import Resolver

from tests.parity._fixture_records import ParityBusinessW0

_HERE = Path(__file__).parent
_ROOT = _HERE / "legacy_artifact_v1"


def build() -> None:
    """(Re)write both committed legacy artifacts from the frozen fixture schema."""
    # Resolver.from_schema string path: a full slot-based manifest with a real
    # comparator + weighted-average module sidecar-free pipeline. The primary
    # legacy-load target -- an explicit schema (not inferred) so a fresh process
    # can rebuild the AllPairsBlocker from its ``schema_type_name``.
    resolver_dir = _ROOT / "resolver_string"
    if resolver_dir.exists():
        shutil.rmtree(resolver_dir)
    Resolver.from_schema(ParityBusinessW0, matcher="string", threshold=0.7).save(resolver_dir)

    # FuzzyString: the named-architecture path -- its manifest additionally
    # stamps ``model_class="fuzzy_string"`` so load() reconstructs a FuzzyString,
    # not a bare Resolver. save() works for FuzzyString (unlike VectorLLMCascade).
    fuzzy_dir = _ROOT / "fuzzy_string"
    if fuzzy_dir.exists():
        shutil.rmtree(fuzzy_dir)
    FuzzyString(schema=ParityBusinessW0, threshold=0.7).save(fuzzy_dir)


if __name__ == "__main__":
    build()
    print(f"Wrote legacy artifacts under {_ROOT}")  # noqa: T201  (one-off build script)
