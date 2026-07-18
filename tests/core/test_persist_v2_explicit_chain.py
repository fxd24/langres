"""Persist v2: an explicit Op-chain model (``from_topology``) round-trips through
``save``/``load`` — while the classic four-slot path stays byte-identical (#193 PR-B).

The crown invariant is the *classic* byte-parity pinned by
``tests/parity/test_behavior_parity_*`` + ``tests/test_resolver_config_dict.py`` +
``tests/parity/test_legacy_load_w0.py``. This file adds the v2 half: a synthetic,
``$0`` explicit chain (reusing ``tests/parity/_explicit_chain_fixture.py``) saves,
reloads in a fresh interpreter state, and reproduces its dedupe — with its paid
Score re-secured on load and its spend cap unwrapped (never persisted) on save.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from langres.core._artifacts import op_spec, rebuild_op
from langres.core.blocker import Blocker
from langres.core.clusterer import Clusterer
from langres.core.groups import ERCandidateGroup
from langres.core.matcher import GroupwiseMatcher
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.op import Finalize, Stage, ThresholdSelect
from langres.core.op_adapters import (
    BlockerSource,
    ClustererStage,
    GroupwiseMatcherScore,
    MatcherScore,
)
from langres.core.registry import register
from langres.core.resolver import ERModel
from langres.core.results import DedupeResult
from langres.core.serialization import ArtifactManifest, ComponentSpec, OpSpec
from langres.core.spend_cap import SpendCappedMatcher
from tests.parity._explicit_chain_fixture import (
    RECORDS,
    THRESHOLD,
    CostedNameMatcher,
    build_explicit_chain_model,
    build_score_after_select_model,
    chain_ops,
)


def _canonical(result: DedupeResult) -> list[list[str]]:
    return sorted(sorted(cluster) for cluster in result)


def _matcher_score(model: ERModel) -> MatcherScore[Any]:
    assert model._ops is not None
    return next(op for op in model._ops if isinstance(op, MatcherScore))


# ----------------------------------------------------------------------------------
# 1. v2 round-trip: save -> load -> identical dedupe + re-secured chain
# ----------------------------------------------------------------------------------


def test_v2_round_trip_reproduces_dedupe_and_metadata(tmp_path: Path) -> None:
    """A ``from_topology`` chain saves, reloads, and dedupes IDENTICALLY —
    clusters, ``score_type``, ``threshold`` and ``backbone`` all match."""
    model, _monitor, _matcher = build_explicit_chain_model(model="test/backbone-x")
    before = model.dedupe(RECORDS)

    model.save(tmp_path)
    loaded = ERModel.load(tmp_path)
    after = loaded.dedupe(RECORDS)

    assert _canonical(before) == _canonical(after) == [["a1", "a2"]]
    assert before.score_type == after.score_type == "prob_llm"
    assert before.threshold == after.threshold == THRESHOLD
    assert before.backbone == after.backbone == "test/backbone-x"
    # resolve() (the low-level exit) reproduces too.
    assert sorted(sorted(c) for c in loaded.resolve(RECORDS)) == [["a1", "a2"]]


def test_v2_round_trip_reestablishes_the_spend_cap(tmp_path: Path) -> None:
    """The loaded chain's ``MatcherScore`` is re-secured: its matcher is a
    ``SpendCappedMatcher`` on the LOADED model's fresh ledger — the door re-wrapped
    the raw matcher that was serialized (budget/monitor deliberately not persisted)."""
    model, _monitor, _matcher = build_explicit_chain_model()
    model.save(tmp_path)
    loaded = ERModel.load(tmp_path)

    capped = _matcher_score(loaded).matcher
    assert isinstance(capped, SpendCappedMatcher)
    # Shares the loaded model's ONE ledger (a fresh default-budget monitor).
    assert capped.monitor is loaded._spend_monitor
    # The wrapped inner matcher is the RAW fixture matcher, faithfully rebuilt.
    assert isinstance(capped._module, CostedNameMatcher)


def test_v2_round_trip_multi_score_chain(tmp_path: Path) -> None:
    """A Score-after-Select chain (two MatcherScores + a TopKSelect) round-trips:
    every stage returns, in order, and every paid Score is re-secured."""
    model = build_score_after_select_model()
    before = model.dedupe(RECORDS)

    model.save(tmp_path)
    loaded = ERModel.load(tmp_path)

    assert loaded._ops is not None
    assert [type(op).__name__ for op in loaded._ops] == [type(op).__name__ for op in model._ops]
    assert _canonical(loaded.dedupe(RECORDS)) == _canonical(before)
    # BOTH matcher Scores re-wrapped against the loaded model's one ledger.
    for op in loaded._ops:
        if isinstance(op, MatcherScore):
            assert isinstance(op.matcher, SpendCappedMatcher)
            assert op.matcher.monitor is loaded._spend_monitor


# ----------------------------------------------------------------------------------
# 2. Unwrap-on-save: the artifact carries the RAW matcher, no cap/monitor/budget leak
# ----------------------------------------------------------------------------------


def test_saved_json_carries_raw_matcher_and_leaks_no_cap(tmp_path: Path) -> None:
    """The saved ``resolver.json`` records the RAW matcher's ``type_name`` under the
    ``matcher_score`` op — never the ``SpendCappedMatcher`` wrapper — and no
    monitor/budget leaks into the artifact."""
    model, _monitor, _matcher = build_explicit_chain_model()
    model.save(tmp_path)

    text = (tmp_path / "resolver.json").read_text()
    manifest = json.loads(text)

    assert manifest["artifact_version"] == "2"
    assert "components" not in manifest  # explicit artifact drops the classic key
    ms_op = next(op for op in manifest["ops"] if op["role"] == "matcher_score")
    assert ms_op["component"]["type_name"] == "costed_name_matcher"
    assert ms_op["params"] == {"out_space": "prob_llm"}

    # The live-monitor wrapper and the budget are NEVER serialized.
    for leaked in ("SpendCappedMatcher", "spend_capped", "monitor", "budget"):
        assert leaked not in text, f"artifact leaked {leaked!r}"


# ----------------------------------------------------------------------------------
# 3. Classic byte-parity (the crown invariant) is untouched
# ----------------------------------------------------------------------------------


def _classic_resolver() -> ERModel:
    from langres.core.blockers import AllPairsBlocker
    from langres.core.comparators import StringComparator
    from langres.core.matchers import WeightedAverageMatcher
    from langres.core.models import CompanySchema

    comparator = StringComparator.from_schema(CompanySchema)
    return ERModel(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=comparator,
        matcher=WeightedAverageMatcher(feature_specs=comparator.feature_specs),
        clusterer=Clusterer(threshold=0.7),
    )


def test_classic_save_has_no_ops_key_and_stamps_v1(tmp_path: Path) -> None:
    """A classic four-slot save is byte-unchanged: it stamps ``"1"`` and its
    ``resolver.json`` carries NO ``ops`` key (F1 + the crown byte-invariant)."""
    _classic_resolver().save(tmp_path)
    manifest = json.loads((tmp_path / "resolver.json").read_text())

    assert manifest["artifact_version"] == "1"
    assert "ops" not in manifest
    assert [c["slot"] for c in manifest["components"]] == [
        "blocker",
        "comparator",
        "module",
        "clusterer",
    ]


def test_classic_config_dict_shape_unchanged() -> None:
    """A classic ``config_dict`` still returns ``{"components": ...}`` only — no
    ``ops`` key — so its ``recipe_id`` never forks (F2)."""
    snapshot = _classic_resolver().config_dict()
    assert set(snapshot.keys()) == {"components"}


def test_explicit_config_dict_keys_on_ops() -> None:
    """An explicit-chain ``config_dict`` keys on ``ops`` (never ``components``)."""
    model, _m, _mt = build_explicit_chain_model()
    snapshot = model.config_dict()
    assert set(snapshot.keys()) == {"ops"}
    assert isinstance(snapshot["ops"], list)


# ----------------------------------------------------------------------------------
# 4. Version window: v1 reads on v2, v3 rejected, explicit stamps v2
# ----------------------------------------------------------------------------------


def test_explicit_save_stamps_v2_classic_stamps_v1(tmp_path: Path) -> None:
    """Explicit save stamps ``"2"``; classic save stamps ``"1"`` — from the written json."""
    build_explicit_chain_model()[0].save(tmp_path / "explicit")
    _classic_resolver().save(tmp_path / "classic")

    explicit = json.loads((tmp_path / "explicit" / "resolver.json").read_text())
    classic = json.loads((tmp_path / "classic" / "resolver.json").read_text())
    assert explicit["artifact_version"] == "2"
    assert classic["artifact_version"] == "1"


def test_v1_artifact_loads_on_v2_langres(tmp_path: Path) -> None:
    """A v1 (classic) artifact loads on this v2-reader build unchanged — v2 is an
    additive layout, so v1 is inside the readable window (the #193 gate change)."""
    _classic_resolver().save(tmp_path)
    assert json.loads((tmp_path / "resolver.json").read_text())["artifact_version"] == "1"
    reloaded = ERModel.load(tmp_path)
    assert reloaded.clusterer.threshold == 0.7


def test_v3_artifact_is_rejected(tmp_path: Path) -> None:
    """A strictly-newer layout (``"3"``) is a hard 'upgrade langres' error."""
    build_explicit_chain_model()[0].save(tmp_path)
    manifest_path = tmp_path / "resolver.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_version"] = "3"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="newer than this langres build"):
        ERModel.load(tmp_path)


# ----------------------------------------------------------------------------------
# 5. F3: op_spec fails loud on a stage it cannot faithfully round-trip
# ----------------------------------------------------------------------------------


class _SubMatcherScore(MatcherScore[Any]):
    """A MatcherScore SUBCLASS — rebuilding it as a base MatcherScore would drop its
    (hypothetical) own state, so op_spec must reject it rather than write a lossy spec."""


class _FakeGroupwise(GroupwiseMatcher[Any]):
    def forward_groups(
        self, groups: Iterator[ERCandidateGroup[Any]]
    ) -> Iterator[PairwiseJudgement]:
        return iter([])  # never runs; op_spec rejects the stage before any scoring


class _FakeFinalize(Finalize):
    def forward(self, clusters: list[set[str]]) -> list[set[str]]:
        return clusters


def test_op_spec_rejects_matcher_score_subclass() -> None:
    stage = _SubMatcherScore(CostedNameMatcher(), out_space="prob_llm")
    with pytest.raises(TypeError, match="cannot serialize"):
        op_spec(stage)


def test_op_spec_rejects_groupwise_matcher_score() -> None:
    with pytest.raises(TypeError, match="cannot serialize"):
        op_spec(GroupwiseMatcherScore(_FakeGroupwise()))


def test_op_spec_rejects_finalize() -> None:
    with pytest.raises(TypeError, match="cannot serialize"):
        op_spec(_FakeFinalize())


def test_op_spec_rejects_unregistered_inner_matcher() -> None:
    """An unregistered raw matcher (no ``type_name``) fails loud at save via the
    nested ``component_spec`` — the same fail-loud contract as the classic path."""

    class _Unregistered(CostedNameMatcher):
        type_name = None  # type: ignore[assignment]  # shadow the registered name

    with pytest.raises(TypeError, match="not serializable"):
        op_spec(MatcherScore(_Unregistered(), out_space="prob_llm"))


# ----------------------------------------------------------------------------------
# 6. rebuild_op malformed-spec guards
# ----------------------------------------------------------------------------------


def test_rebuild_op_rejects_unknown_role(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown OpSpec role"):
        rebuild_op(OpSpec(role="not_a_role"), state_dir=tmp_path)


def test_rebuild_op_rejects_missing_component(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a nested component"):
        rebuild_op(
            OpSpec(role="matcher_score", params={"out_space": "prob_llm"}), state_dir=tmp_path
        )


# ----------------------------------------------------------------------------------
# 7. F5: an explicit-chain sidecar (VectorBlocker-style stateful Source) round-trips
# ----------------------------------------------------------------------------------


@register("persist_v2_marker_blocker")
class _MarkerBlocker(Blocker[Any]):
    """A minimal ``SerializableState`` blocker standing in for a built ``VectorBlocker``
    Source: it persists a marker file (its 'index') to a sidecar and restores it, so
    the ordinal-``op{i}`` sidecar path is exercised without faiss/torch."""

    type_name = "persist_v2_marker_blocker"

    def __init__(self, *, marker: str = "built-index") -> None:
        self._marker = marker
        self.restored: str | None = None

    def stream(self, data: list[Any]) -> Iterator[ERCandidate[Any]]:
        return iter([])  # unused in this save/load-only test

    def inspect_candidates(
        self, candidates: list[ERCandidate[Any]], entities: list[Any], sample_size: int = 10
    ) -> Any:
        raise NotImplementedError  # unused in this save/load-only test

    @property
    def config(self) -> dict[str, object]:
        return {}

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "_MarkerBlocker":
        return cls()

    def save_state(self, state_dir: Path) -> None:
        # An empty marker persists NOTHING — mirrors a VectorBlocker whose index
        # was never built (a SerializableState owner with nothing to write).
        if self._marker:
            (state_dir / "index.marker").write_text(self._marker)

    def load_state(self, state_dir: Path) -> None:
        self.restored = (state_dir / "index.marker").read_text()


def test_explicit_sidecar_state_round_trips_by_ordinal(tmp_path: Path) -> None:
    """A stateful Source writes/reads its sidecar under ``op0`` (ordinal, not slot
    name); load restores its out-of-band state (F5)."""
    ops: list[Stage] = [
        BlockerSource(_MarkerBlocker()),
        MatcherScore(CostedNameMatcher(), out_space="prob_llm"),
        ThresholdSelect(THRESHOLD),
        ClustererStage(Clusterer(threshold=0.0)),
    ]
    ERModel.from_topology(ops=ops).save(tmp_path)

    # The sidecar is keyed by ordinal, not slot name.
    assert (tmp_path / "op0" / "index.marker").read_text() == "built-index"

    loaded = ERModel.load(tmp_path)
    assert loaded._ops is not None
    source = loaded._ops[0]
    assert isinstance(source, BlockerSource)
    assert isinstance(source.blocker, _MarkerBlocker)
    assert source.blocker.restored == "built-index"  # out-of-band state restored


def test_explicit_empty_sidecar_is_dropped(tmp_path: Path) -> None:
    """A stateful Source that persists NOTHING (e.g. an unbuilt VectorBlocker index)
    leaves no ``op{i}`` dir behind — the empty sidecar is removed, so load never
    tries to read a missing state file."""
    ops: list[Stage] = [
        BlockerSource(_MarkerBlocker(marker="")),  # writes no state file
        MatcherScore(CostedNameMatcher(), out_space="prob_llm"),
        ThresholdSelect(THRESHOLD),
        ClustererStage(Clusterer(threshold=0.0)),
    ]
    ERModel.from_topology(ops=ops).save(tmp_path)

    assert not (tmp_path / "op0").exists()  # empty sidecar dropped
    loaded = ERModel.load(tmp_path)
    assert loaded._ops is not None
    source = loaded._ops[0]
    assert isinstance(source, BlockerSource)
    assert isinstance(source.blocker, _MarkerBlocker)
    assert source.blocker.restored is None  # no state to restore


def test_stateless_explicit_chain_writes_no_sidecar(tmp_path: Path) -> None:
    """The shipped-style chain (AllPairsBlocker Source) owns no state, so save
    writes only ``resolver.json`` — no ``op{i}`` sidecar dirs."""
    ops, _matcher = chain_ops()
    ERModel.from_topology(ops=ops).save(tmp_path)

    children = sorted(p.name for p in tmp_path.iterdir())
    assert children == ["resolver.json"]


# ----------------------------------------------------------------------------------
# 8. op_spec / rebuild_op agree on every round-trippable stage (spec-level parity)
# ----------------------------------------------------------------------------------


def test_op_spec_round_trips_every_stage_kind(tmp_path: Path) -> None:
    """Each stage of the canonical chain survives op_spec -> serialize -> rebuild_op,
    and every produced OpSpec validates through the manifest (json boundary)."""
    ops, _matcher = chain_ops()
    specs = [op_spec(stage) for stage in ops]

    # Roles, in order, match the canonical chain.
    assert [s.role for s in specs] == [
        "blocker_source",
        "comparator_score",
        "matcher_score",
        "threshold_select",
        "clusterer_stage",
    ]
    # Round-trips through the manifest json (the OpSpec data contract).
    manifest = ArtifactManifest(artifact_version="2", langres_version="test", ops=specs)
    restored = ArtifactManifest.model_validate_json(manifest.model_dump_json())
    assert restored.ops == specs
    assert isinstance(restored.ops[0].component, ComponentSpec)

    rebuilt = [rebuild_op(spec, state_dir=tmp_path / f"op{i}") for i, spec in enumerate(specs)]
    assert [type(stage).__name__ for stage in rebuilt] == [type(stage).__name__ for stage in ops]
    threshold_stage = rebuilt[3]
    assert isinstance(threshold_stage, ThresholdSelect)
    assert threshold_stage.threshold == THRESHOLD
