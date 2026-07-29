"""Behaviour tests for the sweep-provenance sidecar writer.

Scoped to the one property a caller cannot check for itself: a window that
closed COMPLETE must never be relabelled partial. Everything else in this module
(`--start`, the tree digests, the stability verdict) reads git and is exercised
by the drivers.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[2]
MODULE_PATH = ROOT / "examples" / "research" / "write_provenance.py"


def _load() -> ModuleType:
    name = "example_write_provenance"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROV = _load()


def _sidecar(tmp_path: Path, **overrides: object) -> Path:
    doc: dict[str, object] = {
        "blobs": {},
        "tree_digests": {},
        "driver_blobs": {},
        "driver_digest": "abc",
        "studies_measured": ["a", "b"],
        "measurement_window": {
            "started": "2026-07-29T05:00:00+02:00",
            "finished": "2026-07-29T06:00:00+02:00",
            "head_when_sweep_started": "deadbeef",
        },
    }
    doc.update(overrides)
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(doc))
    return path


class TestPartialCannotOverwriteComplete:
    """The driver's abort path re-invokes ``--finish --partial``.

    A failure AFTER a successful ``--finish`` — a render error, say — would have
    rewritten ``window_complete`` to false, stamped a new finished timestamp, and
    added a note claiming the sweep never reached every planned study, over rows
    that prove otherwise.
    """

    def test_a_closed_complete_window_refuses_the_downgrade(self, tmp_path: Path) -> None:
        path = _sidecar(tmp_path, window_complete=True)

        with pytest.raises(SystemExit, match="already closed"):
            PROV.finish(path, partial=True)

    def test_the_refusal_leaves_the_sidecar_untouched(self, tmp_path: Path) -> None:
        """Refusing after writing would be no better than not refusing."""
        path = _sidecar(tmp_path, window_complete=True)
        before = path.read_text()

        with pytest.raises(SystemExit):
            PROV.finish(path, partial=True)

        assert path.read_text() == before

    def test_an_open_window_can_still_be_marked_partial(self, tmp_path: Path) -> None:
        """The genuine abort path — no `--finish` has run — must keep working."""
        path = _sidecar(tmp_path)

        PROV.finish(path, partial=True)

        doc = json.loads(path.read_text())
        assert doc["window_complete"] is False
        assert "ABORTED" in doc["partial_note"]

    def test_a_missing_sidecar_says_to_run_start(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="run --start"):
            PROV.finish(tmp_path / "absent.json", partial=True)
