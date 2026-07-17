"""Deterministic proof that the M1 bootstrap pipeline runs end-to-end (real embeddings).

Drives the importable core of ``examples/research/m1_bootstrap_fodors_zagat.py``
(``run_bootstrap``) over the real Fodors-Zagat corpus with real sentence-transformer
embeddings and the zero-spend :class:`FakeLabeler`. Asserts the M1-critical
blocking signal (cross-source Pair-Completeness >= 0.95), that a non-empty gold set
and a report with non-trivial agreement + calibration are produced, and that the
gold set round-trips through ``save``/``load``.

Marked ``slow`` (it loads an embedding model + builds a FAISS index) but
network-free, so it runs in CI alongside the Person strong-path test. The gated
real-GLM branch needs no key and is never exercised here.
"""

from __future__ import annotations

import pytest

from examples.research.m1_bootstrap_fodors_zagat import run_bootstrap
from langres.curation.models import GoldSet
from langres.curation.report import BootstrapReport

pytestmark = pytest.mark.slow


def test_bootstrap_end_to_end_is_deterministic_and_clears_gate(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Pair-completeness clears 0.95; a non-empty gold set + report round-trip."""
    results = run_bootstrap(tmp_path)

    gold: GoldSet = results["gold"]
    report: BootstrapReport = results["report"]

    # 1. Blocking pair-completeness — the cross-source matches must survive blocking.
    assert report.blocking.pair_completeness >= 0.95

    # 2. A non-empty, labeled gold set was produced.
    assert isinstance(gold, GoldSet)
    assert len(gold.pairs) > 0
    assert all(p.source == "fake" for p in gold.pairs)
    assert gold.metadata["total_cost_usd"] == 0.0  # FakeLabeler never spends
    # The fake teacher produced a real mix (not an all-positive / all-negative
    # collapse), so the agreement numbers below are non-degenerate.
    assert gold.metadata["matches"] > 0
    assert gold.metadata["non_matches"] > 0

    # 3. The report carries non-trivial agreement + calibration (FakeLabeler
    #    disagrees with truth on the hard band, so neither is degenerate).
    assert report.agreement is not None
    assert report.agreement.n_evaluated > 0
    assert report.calibration is not None
    assert 0.0 <= report.calibration.brier <= 1.0

    # 4. Gold-set save/reload round-trip is identical.
    assert results["roundtrip_ok"] is True
    assert results["gold_path"].exists()
    assert results["reloaded"].model_dump() == gold.model_dump()


def test_bootstrap_is_repeatable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Two independent runs produce identical labels (determinism)."""
    first = run_bootstrap(tmp_path / "a")
    second = run_bootstrap(tmp_path / "b")
    assert [(p.left_id, p.right_id, p.label, p.confidence) for p in first["gold"].pairs] == [
        (p.left_id, p.right_id, p.label, p.confidence) for p in second["gold"].pairs
    ]
