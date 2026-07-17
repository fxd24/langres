"""Canonical golden read/assert/update helper for the W0 parity net (#193).

One job: turn a plain-Python snapshot payload into deterministic, canonical JSON
and compare it byte-for-byte against a committed golden in ``goldens/`` -- or
regenerate that golden when ``LANGRES_PARITY_UPDATE=1``. Shared by every parity
test so the canonicalization rules (float rounding, cluster ordering, key
sorting) are defined once and cannot drift between tests.

Determinism rules, applied to every payload:

- **Floats round to 6 decimals** (scores, metrics, thresholds) so the 16th
  decimal of a rapidfuzz score never fails an otherwise-identical run.
- **Object keys are sorted** and JSON is indented, so the on-disk golden is a
  readable, diffable, stable artifact.
- Payloads must carry **no** timestamps, absolute paths, or other run-varying
  fields -- the tests are responsible for not putting them in.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_GOLDENS_DIR = Path(__file__).parent / "goldens"

#: Set ``LANGRES_PARITY_UPDATE=1`` to (re)write goldens instead of asserting.
#: A deliberate re-baseline, never the default -- CI always asserts.
UPDATE = os.getenv("LANGRES_PARITY_UPDATE") == "1"


def round_floats(obj: Any, ndigits: int = 6) -> Any:
    """Recursively round every float in ``obj`` to ``ndigits`` decimals.

    ``bool`` is checked first and passed through untouched: ``bool`` is a
    subclass of ``int`` in Python, and a match ``decision`` must serialize as
    ``true``/``false``, never as a number.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {key: round_floats(value, ndigits) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(value, ndigits) for value in obj]
    return obj


def canonical_clusters(clusters: Any) -> list[list[str]]:
    """Order-independent cluster form: ``sorted([sorted(list(c)) for c in clusters])``.

    Entity resolution output is a *set* of *sets*; both the outer grouping and
    the intra-cluster id order are meaningless, so we impose a total order on
    both before snapshotting. Ids are stringified for a stable sort.
    """
    return sorted([sorted(str(x) for x in cluster) for cluster in clusters])


def check_golden(name: str, payload: dict[str, Any]) -> None:
    """Assert ``payload`` equals ``goldens/<name>.json``; regenerate under ``UPDATE``.

    The payload is canonicalized (floats rounded, keys sorted, trailing newline)
    into the exact text that lives on disk, so the comparison is a plain string
    equality -- no float-tolerance fuzz, no ordering surprises. That same
    canonical text is what ``UPDATE`` writes, so a regenerated golden re-asserts
    identically on the next run.
    """
    text = json.dumps(round_floats(payload), indent=2, sort_keys=True) + "\n"
    path = _GOLDENS_DIR / f"{name}.json"
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return
    assert path.exists(), (
        f"parity golden {path} is missing; generate it once with "
        "LANGRES_PARITY_UPDATE=1 uv run pytest tests/parity"
    )
    assert text == path.read_text(), (
        f"BEHAVIOR PARITY DRIFT vs {path.name}: the current pipeline no longer "
        "reproduces the frozen W0 output. If this change is intentional, "
        "regenerate with LANGRES_PARITY_UPDATE=1 and review the golden diff; "
        "otherwise a refactor changed behavior it was meant to preserve."
    )
