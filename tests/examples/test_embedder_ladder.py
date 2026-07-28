"""Offline behaviour tests for the embedder-ladder harness's measurement rules.

$0 and model-free on purpose: these cover the three places the harness can
publish a *plausible but wrong* number — the reachable-recall ceiling, the
direction of the separability score, and the unit a confidence interval is
resampled over. Each of them was wrong at some point in this harness's history,
and none of them would have shown up as a crash.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).parents[2]
LADDER_PATH = ROOT / "examples" / "research" / "embedder_ladder.py"


def _load() -> ModuleType:
    name = "example_embedder_ladder"
    sys.path.insert(0, str(LADDER_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, LADDER_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # Registered BEFORE exec: `@dataclass` resolves its own module out of
        # `sys.modules` while processing the class, and an unregistered module
        # fails there with an opaque AttributeError.
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(LADDER_PATH.parent))


LADDER = _load()


def _record(record_id: str, source: str) -> SimpleNamespace:
    return SimpleNamespace(id=record_id, source=source)


class TestReachableCeiling:
    """The cross-source filter is NOT recall-neutral — the correction under test."""

    def test_a_three_record_gold_cluster_puts_a_gold_pair_out_of_reach(self) -> None:
        # Two of the three are on source "a", so their gold pair is intra-source
        # and no cross-source candidate set can ever contain it.
        corpus = [_record("a1", "a"), _record("a2", "a"), _record("b1", "b")]
        gold_pairs = {
            frozenset({"a1", "a2"}),
            frozenset({"a1", "b1"}),
            frozenset({"a2", "b1"}),
        }
        assert LADDER._reachable_ceiling(corpus, gold_pairs) == 2 / 3

    def test_a_purely_one_to_one_gold_set_reaches_everything(self) -> None:
        corpus = [_record("a1", "a"), _record("b1", "b")]
        assert LADDER._reachable_ceiling(corpus, {frozenset({"a1", "b1"})}) == 1.0

    def test_no_gold_pairs_is_a_ceiling_of_one_not_a_division_by_zero(self) -> None:
        assert LADDER._reachable_ceiling([_record("a1", "a")], set()) == 1.0


class TestSeparabilityDirection:
    """A positive must be found whichever side is the query."""

    def test_a_pair_scored_only_in_the_reverse_direction_still_separates(self) -> None:
        # Positives come out of a sorted() id tuple, so on a source-prefixed
        # corpus every positive is A-query -> B-document while the sampled
        # negatives are mixed. Scoring one direction only would compare the two
        # classes under different direction distributions. Here the positive
        # scores 0.0 forward and 1.0 backward: a forward-only implementation
        # returns 0.5 (chance), the symmetric one returns 1.0.
        ids = ["a1", "a2", "b1", "b2"]
        documents = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        queries = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        auc = LADDER.separability_auc(documents, queries, ids, {frozenset({"a1", "b1"})})
        assert auc == 1.0

    def test_no_gold_pair_in_the_corpus_reports_nothing_rather_than_a_number(self) -> None:
        vectors = np.eye(3)
        assert LADDER.separability_auc(vectors, vectors, ["x", "y", "z"], set()) is None


class TestPairedInterval:
    """The interval must resample gold clusters, never the dependent record rows."""

    def test_the_resampled_unit_is_the_cluster_not_the_record(self) -> None:
        baseline = {f"r{i}": (0.0, f"c{i // 3}") for i in range(6)}
        candidate = {f"r{i}": (1.0, f"c{i // 3}") for i in range(6)}
        interval = LADDER.paired_interval(baseline, candidate)
        assert interval is not None
        assert interval.n_entities == 6
        assert interval.n_clusters == 2
        assert interval.observed_difference == 1.0

    def test_a_single_shared_record_yields_no_interval_rather_than_a_fake_one(self) -> None:
        assert LADDER.paired_interval({"r0": (0.0, "c0")}, {"r0": (1.0, "c0")}) is None

    def test_only_records_measured_by_both_models_are_compared(self) -> None:
        baseline = {"r0": (0.0, "c0"), "r1": (0.0, "c1"), "gone": (1.0, "c2")}
        candidate = {"r0": (1.0, "c0"), "r1": (1.0, "c1")}
        interval = LADDER.paired_interval(baseline, candidate)
        assert interval is not None
        assert interval.n_entities == 2


class TestDocumentedPromptArm:
    """A checkpoint's OWN prompt recipe is a third arm, and it can be asymmetric.

    The generic-instruction arm answers "does any instruction help". The question
    a user actually faces is "should I follow the model card", and for
    EmbeddingGemma the model card is asymmetric: queries get
    ``"task: search result | query: "`` and documents get
    ``"title: none | text: "`` — NOT a bare document. Measuring a query prefix
    against un-prefixed documents is a different configuration from the
    documented one, so the arm has to carry both sides.
    """

    def test_arms_are_document_and_query_prompt_pairs(self) -> None:
        assert LADDER.PROMPT_ARMS["none"] == (None, None)
        assert LADDER.PROMPT_ARMS["instruct"] == (None, LADDER.INSTRUCTION)

    def test_a_model_without_a_documented_recipe_gets_only_the_base_arms(self) -> None:
        spec = LADDER.ModelSpec("all-MiniLM-L6-v2")
        assert spec.documented_arm is None
        assert LADDER.arms_for(spec, LADDER.PROMPT_ARMS) == LADDER.PROMPT_ARMS

    def test_an_instruction_trained_model_gains_a_third_asymmetric_arm(self) -> None:
        spec = LADDER.MODELS_BY_NAME["google/embeddinggemma-300m"]
        assert spec.documented_arm is not None
        arms = LADDER.arms_for(spec, LADDER.PROMPT_ARMS)

        assert set(arms) == {"none", "instruct", "documented"}
        document_prompt, query_prompt = arms["documented"]
        # Both sides prefixed, and differently -- that is what makes it the
        # documented recipe rather than the generic arm with different wording.
        assert document_prompt == "title: none | text: "
        assert query_prompt == "task: search result | query: "

    def test_the_documented_arm_is_never_added_to_uninstructed_models(self) -> None:
        """Cheap to add, but it would measure a question already answered.

        Every model here that was trained without a query-side instruction
        already has its answer from the generic arm; spending queue time on a
        second flavour of the same negative is the wrong trade.
        """
        for name in ("all-MiniLM-L6-v2", "all-mpnet-base-v2", "BAAI/bge-base-en-v1.5"):
            assert LADDER.MODELS_BY_NAME[name].documented_arm is None


class _CountingEmbedder:
    """Deterministic $0 embedder that records every text it is asked to encode."""

    embedding_dim = 8

    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def encode(self, texts: list[str], prompt: str | None = None) -> np.ndarray:
        rendered = [(prompt or "") + text for text in texts]
        self.seen.append(rendered)
        # Any deterministic function of the string works; the assertions are
        # about WHICH strings arrive, never about the geometry.
        return np.array([[float(len(t) + i) for i in range(8)] for t in rendered], dtype=np.float32)

    def cache_info(self) -> dict[str, int]:
        return {"misses": sum(len(batch) for batch in self.seen)}


class TestDocumentSidePrefix:
    """The document prefix must not leak onto the query side.

    ``create_index`` snapshots the texts it is given and
    ``search_all(query_prompt=...)`` re-encodes *those*, so a documented arm
    built from prefixed documents would query with
    ``query_prompt + document_prompt + text`` — a recipe no model card
    describes. The arm would still produce a full table of plausible recall
    numbers, measured under a configuration nobody chose.
    """

    def test_documents_are_prefixed_and_the_query_side_is_not(self) -> None:
        embedder = _CountingEmbedder()
        texts = ["Apple Inc.", "Apple Computer", "Microsoft Corp"]

        index, _vectors, _seconds, _encoded = LADDER.build_prompted_index(
            embedder, texts, "title: none | text: "
        )

        # What the index was BUILT from carries the document prefix...
        assert embedder.seen[0] == ["title: none | text: " + text for text in texts]
        # ...and what it will RE-ENCODE as queries does not.
        assert index._corpus_texts == texts

    def test_a_prompted_search_never_double_prefixes(self) -> None:
        embedder = _CountingEmbedder()
        texts = ["Apple Inc.", "Apple Computer", "Microsoft Corp"]

        index, _vectors, _seconds, _encoded = LADDER.build_prompted_index(
            embedder, texts, "title: none | text: "
        )
        embedder.seen.clear()
        index.search_all(k=2, query_prompt="task: search result | query: ")

        assert embedder.seen == [["task: search result | query: " + text for text in texts]]

    def test_an_unprefixed_arm_indexes_the_text_verbatim(self) -> None:
        embedder = _CountingEmbedder()
        texts = ["Apple Inc.", "Microsoft Corp"]

        index, _vectors, _seconds, _encoded = LADDER.build_prompted_index(embedder, texts, None)

        assert embedder.seen[0] == texts
        assert index._corpus_texts == texts


class _SwappableEmbedder:
    """A $0 embedder whose vectors change when ``weights`` is reassigned.

    Stands in for a checkpoint re-uploaded under the same Hub name: same object
    identity, same namespace, different numbers out.
    """

    embedding_dim = 4

    def __init__(self, weights: float = 1.0) -> None:
        self.weights = weights

    def encode(self, texts: list[str], prompt: str | None = None) -> np.ndarray:
        rendered = [(prompt or "") + text for text in texts]
        return np.array(
            [[self.weights * (len(t) + i) for i in range(4)] for t in rendered], dtype=np.float32
        )


class TestStaleCacheCanary:
    """The cache namespace omits the Hub revision; this is what covers that.

    A namespace keyed on model name + dtype still hits after an upstream
    re-upload, so a warm re-run would read the NEW checkpoint's metadata while
    serving the OLD checkpoint's vectors — publishing a row that mixes two
    checkpoints with nothing in the row to reveal it.
    """

    def _cached(self, embedder: Any, tmp_path: Path) -> Any:
        from langres.core.embeddings import DiskCachedEmbedder

        return DiskCachedEmbedder(embedder, cache_dir=tmp_path, namespace="canary_ns")

    def test_a_cold_cache_agrees_with_the_checkpoint(self, tmp_path: Path) -> None:
        """Control: the check must not fire on the ordinary first run."""
        embedder = _SwappableEmbedder()

        LADDER._assert_cache_matches_checkpoint(
            embedder, self._cached(embedder, tmp_path), "canary_ns", tmp_path
        )

    def test_a_warm_cache_from_the_same_checkpoint_still_agrees(self, tmp_path: Path) -> None:
        """Control: a *warm* cache must stay silent, or the check is unusable.

        Without this, a check that simply always raised on a populated cache
        would pass the test below and fail every real re-run.
        """
        embedder = _SwappableEmbedder()
        cached = self._cached(embedder, tmp_path)
        LADDER._assert_cache_matches_checkpoint(embedder, cached, "canary_ns", tmp_path)

        LADDER._assert_cache_matches_checkpoint(
            embedder, self._cached(embedder, tmp_path), "canary_ns", tmp_path
        )

    def test_a_re_uploaded_checkpoint_aborts_the_run(self, tmp_path: Path) -> None:
        """The failure this exists to catch, actually caught."""
        embedder = _SwappableEmbedder(weights=1.0)
        LADDER._assert_cache_matches_checkpoint(
            embedder, self._cached(embedder, tmp_path), "canary_ns", tmp_path
        )

        embedder.weights = 2.0  # the same name now serves different weights

        with pytest.raises(LADDER.StaleEmbeddingCacheError) as excinfo:
            LADDER._assert_cache_matches_checkpoint(
                embedder, self._cached(embedder, tmp_path), "canary_ns", tmp_path
            )

        # The message has to be actionable on its own: whoever hits this is
        # mid-sweep and needs the path, not a description of the hazard.
        assert "canary_ns.db" in str(excinfo.value)

    def test_a_cache_written_before_the_canary_existed_is_refused(self, tmp_path: Path) -> None:
        """The hole the canary would otherwise have on its very first run.

        On a pre-existing cache the canary simply *misses*: it is encoded fresh
        from whatever checkpoint is loaded now, written into the unvouched
        database, and then compared against another fresh encoding of itself. It
        always matches, while every corpus vector beside it may belong to a
        different checkpoint — the same defect the check exists to close,
        reintroduced one level up. (Caught by cross-model review.)
        """
        embedder = _SwappableEmbedder()
        # A cache with real entries and no canary: exactly what every namespace
        # written before this check looks like.
        legacy = self._cached(embedder, tmp_path)
        legacy.encode(["some corpus text", "another corpus text"])

        with pytest.raises(LADDER.StaleEmbeddingCacheError) as excinfo:
            LADDER._assert_cache_matches_checkpoint(
                embedder, self._cached(embedder, tmp_path), "canary_ns", tmp_path
            )

        assert "before this check existed" in str(excinfo.value)
        assert "canary_ns.db" in str(excinfo.value)


class TestPersistence:
    def test_rerunning_a_model_replaces_its_rows_instead_of_appending(self) -> None:
        stale = LADDER.LadderRow(
            model="m", benchmark="b", prompt_arm="none", k=20, status="ok", candidate_recall=0.1
        )
        other = LADDER.LadderRow(model="other", benchmark="b", prompt_arm="none", k=20, status="ok")
        fresh = LADDER.LadderRow(
            model="m", benchmark="b", prompt_arm="none", k=20, status="ok", candidate_recall=0.9
        )
        merged = LADDER.merge_rows([stale, other], [fresh])
        assert [row.candidate_recall for row in merged if row.model == "m"] == [0.9]
        assert any(row.model == "other" for row in merged)

    def test_a_partial_rerun_keeps_the_cells_it_did_not_measure(self) -> None:
        """`--k` / `--prompts` make partial re-runs normal; they must not delete.

        Keyed on `(model, benchmark)` the re-run of one arm wiped every other arm
        and every other k for that benchmark. The file shrank and the report
        rendered. Found by cross-model review.
        """
        other_arm = LADDER.LadderRow(
            model="m", benchmark="b", prompt_arm="instruct", k=20, status="ok", candidate_recall=0.5
        )
        other_k = LADDER.LadderRow(
            model="m", benchmark="b", prompt_arm="none", k=50, status="ok", candidate_recall=0.7
        )
        fresh = LADDER.LadderRow(
            model="m", benchmark="b", prompt_arm="none", k=20, status="ok", candidate_recall=0.9
        )
        merged = LADDER.merge_rows([other_arm, other_k], [fresh])

        assert {(r.prompt_arm, r.k) for r in merged} == {
            ("instruct", 20),
            ("none", 50),
            ("none", 20),
        }

    def test_a_failed_rerun_removes_the_rows_it_invalidates(self) -> None:
        """A load failure must not leave the previous run's numbers beside it."""
        stale = LADDER.LadderRow(
            model="m", benchmark="b", prompt_arm="none", k=20, status="ok", candidate_recall=0.9
        )
        failure = LADDER.LadderRow(
            model="m", benchmark="b", prompt_arm="-", k=0, status="failed", error="OSError: gone"
        )
        merged = LADDER.merge_rows([stale], [failure])

        assert [row.status for row in merged] == ["failed"]

    def test_an_arm_that_fails_voids_only_that_arm(self) -> None:
        kept = LADDER.LadderRow(
            model="m", benchmark="b", prompt_arm="none", k=20, status="ok", candidate_recall=0.9
        )
        doomed = LADDER.LadderRow(
            model="m",
            benchmark="b",
            prompt_arm="documented",
            k=20,
            status="ok",
            candidate_recall=0.4,
        )
        failure = LADDER.LadderRow(
            model="m",
            benchmark="b",
            prompt_arm="documented",
            k=0,
            status="failed",
            error="ValueError: bad prompt",
        )
        merged = LADDER.merge_rows([kept, doomed], [failure])

        assert {(r.prompt_arm, r.status) for r in merged} == {
            ("none", "ok"),
            ("documented", "failed"),
        }

    def test_remeasuring_the_reference_clears_deltas_it_no_longer_backs(self) -> None:
        # Those deltas were computed against per-record scores the sidecar has
        # just overwritten. Keeping them publishes a comparison the file can no
        # longer reproduce, and nothing would look wrong.
        other = LADDER.LadderRow(
            model="other",
            benchmark="b",
            prompt_arm="none",
            k=20,
            status="ok",
            vs_reference_delta=0.05,
            vs_reference_ci_low=0.01,
            vs_reference_ci_high=0.09,
        )
        untouched = LADDER.LadderRow(
            model="other",
            benchmark="elsewhere",
            prompt_arm="none",
            k=20,
            status="ok",
            vs_reference_delta=0.07,
        )
        fresh_reference = LADDER.LadderRow(
            model=LADDER.REFERENCE_MODEL, benchmark="b", prompt_arm="none", k=20, status="ok"
        )
        merged = LADDER.merge_rows([other, untouched], [fresh_reference])

        cleared = next(r for r in merged if r.model == "other" and r.benchmark == "b")
        assert cleared.vs_reference_delta is None
        assert cleared.vs_reference_ci_low is None
        assert cleared.vs_reference_ci_high is None
        kept = next(r for r in merged if r.benchmark == "elsewhere")
        assert kept.vs_reference_delta == 0.07

    def test_a_reference_cell_that_failed_does_not_survive_in_the_sidecar(self) -> None:
        """The sidecar must never outlive the rows that back it.

        A failed re-measurement produces no update for that cell, so a plain
        merge keeps the previous per-record recall. A later model then publishes
        a `vs_reference_*` interval whose baseline exists nowhere in the current
        rows — an interval that cannot be reproduced from the data it ships with.
        """
        existing = {
            "ref|b|none": {"r1": (1.0, "c1")},
            "ref|b|instruct": {"r1": (0.5, "c1")},
            "ref|elsewhere|none": {"r2": (0.25, "c2")},
        }
        touched = {"ref|b|none", "ref|b|instruct"}
        updates = {"ref|b|none": {"r1": (0.75, "c1")}}

        refreshed = LADDER.refresh_reference(existing, updates, touched)

        # Re-measured: replaced. Attempted and failed: gone. Untouched: kept.
        assert refreshed["ref|b|none"] == {"r1": (0.75, "c1")}
        assert "ref|b|instruct" not in refreshed
        assert refreshed["ref|elsewhere|none"] == {"r2": (0.25, "c2")}

    def test_a_reference_run_where_everything_failed_still_voids_its_cells(self) -> None:
        """The empty-updates case is the one that most needs voiding, not the one to skip."""
        existing = {"ref|b|none": {"r1": (1.0, "c1")}}
        assert LADDER.refresh_reference(existing, {}, {"ref|b|none"}) == {}

    def test_rows_round_trip_through_the_tracked_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.jsonl"
        row = LADDER.LadderRow(
            model="m",
            benchmark="b",
            prompt_arm="instruct",
            k=20,
            status="ok",
            candidate_recall=0.5,
            reachable_recall_ceiling=0.8,
            recall_of_reachable=0.625,
            index_build_encoded=0,
            metric_revision=LADDER.METRIC_REVISION,
        )
        LADDER.write_rows(path, [row])
        assert LADDER.read_rows(path) == [row]

    def test_reference_per_record_recall_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "reference.json"
        store = {LADDER._reference_key("bench", "none"): {"r0": (0.5, "c0"), "r1": (1.0, "c1")}}
        LADDER.write_reference(path, store)
        assert LADDER.read_reference(path) == store

    def test_the_reference_key_names_the_model_and_k_it_was_measured_at(self) -> None:
        # Both are one-line constants. If either changed while the file stayed
        # put, an unkeyed sidecar would silently baseline against the wrong model.
        key = LADDER._reference_key("bench", "none")
        assert LADDER.REFERENCE_MODEL in key
        assert f"k{LADDER.CI_K}" in key

    def test_a_missing_reference_file_is_no_reference_not_a_crash(self, tmp_path: Path) -> None:
        assert LADDER.read_reference(tmp_path / "absent.json") == {}


def _cell(model: str, arm: str, **overrides: object) -> Any:
    defaults: dict[str, object] = dict(
        model=model,
        benchmark="bench",
        prompt_arm=arm,
        k=20,
        status="ok",
        parameter_count=1_000_000,
        embedding_dim=8,
        n_records=10,
        n_gold_pairs=4,
        candidate_recall=0.5,
        reachable_recall_ceiling=0.8,
        recall_of_reachable=0.625,
        total_candidates=100,
        candidates_per_unit_recall=200.0,
        separability_auc=0.99,
        index_build_seconds=0.2,
        index_build_encoded=0,
        metric_revision=LADDER.METRIC_REVISION,
        saturation="not saturated",
    )
    defaults.update(overrides)
    return LADDER.LadderRow(**defaults)


def _full_grid() -> list[Any]:
    """Every model x benchmark x arm x k cell the ladder declares."""
    return [
        _cell(spec.name, arm, benchmark=benchmark, k=k)
        for spec in LADDER.MODELS
        for arm in LADDER.arms_for(spec, LADDER.PROMPT_ARMS)
        for benchmark in LADDER.BENCHMARKS
        for k in LADDER.K_VALUES
    ]


class TestReport:
    def test_the_report_flags_a_warm_cache_build_and_an_inconclusive_interval(self) -> None:
        rows = [
            _cell("ref", "none"),
            _cell(
                "ref",
                "instruct",
                prompt_delta=0.004,
                prompt_delta_ci_low=-0.01,
                prompt_delta_ci_high=0.02,
            ),
        ]
        report = LADDER.render_report(rows)

        assert "recall/ceil" in report
        assert "(spans 0)" in report
        # `enc` = 0 is what makes the build seconds a cache read, not an encode.
        assert "| 0.2 | 0 |" in report
        assert "not saturated" in report
        # Size is reported as the measured count, never as a t-shirt label.
        assert "| 1.0M |" in report

    def test_the_published_prompt_delta_is_the_statistic_its_interval_bounds(self) -> None:
        # The aggregate difference is 0.0 here while the per-record mean is
        # +0.0400: printing only the aggregate beside the interval would show a
        # point estimate outside its own CI.
        rows = [
            _cell("ref", "none", candidate_recall=0.5),
            _cell(
                "ref",
                "instruct",
                candidate_recall=0.5,
                prompt_delta=0.04,
                prompt_delta_ci_low=0.01,
                prompt_delta_ci_high=0.07,
            ),
        ]
        report = LADDER.render_report(rows)
        assert "Δ per-record" in report
        assert "+0.0400" in report
        assert "[+0.0100, +0.0700]" in report

    def test_an_exactly_zero_interval_is_not_called_inconclusive(self) -> None:
        assert LADDER._ci(0.0, 0.0) == "[+0.0000, +0.0000] (exactly 0)"
        assert "(spans 0)" in LADDER._ci(-0.01, 0.02)

    def test_a_model_without_an_interval_is_shown_not_dropped(self) -> None:
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none"),
            _cell("uncompared", "none", parameter_count=2_000_000),
        ]
        report = LADDER.render_report(rows)
        section = report.split("## Is it better than what ships today?")[1]
        assert "uncompared" in section
        assert "not measured against the current" in section

    def test_rows_from_an_older_metric_revision_are_excluded_and_named(self) -> None:
        # Two definitions of the same metric must never share a column.
        rows = [
            _cell("current", "none"),
            _cell(
                "legacy", "none", metric_revision=LADDER.METRIC_REVISION - 1, separability_auc=0.5
            ),
        ]
        report = LADDER.render_report(rows)
        assert "legacy" in report.split("## Models that were measured")[0]
        assert "`legacy`" not in report.split("## Models that were measured")[1]
        assert "older metric definition" in report

    def test_a_failed_model_is_a_row_in_the_report_not_a_silent_gap(self) -> None:
        rows = [
            LADDER.LadderRow(
                model="broken",
                benchmark="bench",
                prompt_arm="-",
                k=0,
                status="failed",
                metric_revision=LADDER.METRIC_REVISION,
                error="OSError: gated repo",
            )
        ]
        report = LADDER.render_report(rows)
        assert "broken" in report
        assert "gated repo" in report

    def test_every_measured_arm_reaches_the_prompt_table(self) -> None:
        """A measured arm missing from the arm table is a silent skip."""
        rows = [
            _cell("m", "none"),
            _cell("m", "instruct", prompt_delta=-0.01),
            _cell("m", "documented", prompt_delta=0.02, candidate_recall=0.6),
        ]
        section = LADDER.render_report(rows).split("## Does an instruction prompt help?")[1]
        assert "| documented |" in section
        assert "| instruct |" in section
        assert "+0.0200" in section

    def test_the_bigger_is_not_better_claim_cites_a_row_the_report_publishes(self) -> None:
        """The claim must come from the tables, not from an excluded measurement."""
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none", parameter_count=20_000_000),
            _cell(
                "big-and-worse",
                "none",
                parameter_count=200_000_000,
                vs_reference_delta=-0.03,
                vs_reference_ci_low=-0.04,
                vs_reference_ci_high=-0.02,
            ),
        ]
        report = LADDER.render_report(rows)
        assert "Parameter count is not the axis" in report
        assert "10x the parameters" in report
        assert "200.0M" in report

    def test_no_bigger_is_not_better_claim_without_a_conclusive_row(self) -> None:
        """A delta whose interval straddles 0 does not establish the claim."""
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none", parameter_count=20_000_000),
            _cell(
                "big-and-unclear",
                "none",
                parameter_count=200_000_000,
                vs_reference_delta=-0.03,
                vs_reference_ci_low=-0.09,
                vs_reference_ci_high=0.02,
            ),
        ]
        assert "Parameter count is not the axis" not in LADDER.render_report(rows)

    def test_the_headline_is_computed_from_the_rows_and_refuses_to_average(self) -> None:
        """A model that wins one benchmark and loses another must not be summarised."""
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none", benchmark="wins"),
            _cell(LADDER.REFERENCE_MODEL, "none", benchmark="loses"),
            _cell(
                "contender",
                "none",
                benchmark="wins",
                parameter_count=2_000_000,
                vs_reference_delta=0.2,
                vs_reference_ci_low=0.18,
                vs_reference_ci_high=0.23,
            ),
            _cell(
                "contender",
                "none",
                benchmark="loses",
                parameter_count=2_000_000,
                vs_reference_delta=-0.03,
                vs_reference_ci_low=-0.04,
                vs_reference_ci_high=-0.02,
            ),
        ]
        headline = LADDER.render_report(rows).split("## Headline")[1].split("## How to read")[0]

        assert "+0.2000" in headline
        assert "-0.0300" in headline
        assert "worse" in headline
        assert "publishes no cross-benchmark mean" in headline
        # An interval clear of zero must not be labelled inconclusive here.
        assert "(spans 0)" not in headline

    def test_the_headline_does_not_call_a_noisy_sign_flip_a_disagreement(self) -> None:
        """Same signs, but both intervals straddle 0 -- that is not evidence."""
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none", benchmark="a"),
            _cell(LADDER.REFERENCE_MODEL, "none", benchmark="b"),
            _cell(
                "contender",
                "none",
                benchmark="a",
                parameter_count=2_000_000,
                vs_reference_delta=0.2,
                vs_reference_ci_low=-0.05,
                vs_reference_ci_high=0.4,
            ),
            _cell(
                "contender",
                "none",
                benchmark="b",
                parameter_count=2_000_000,
                vs_reference_delta=-0.03,
                vs_reference_ci_low=-0.2,
                vs_reference_ci_high=0.1,
            ),
        ]
        headline = LADDER.render_report(rows).split("## Headline")[1].split("## How to read")[0]

        assert "worse" not in headline
        assert "spread across benchmarks" in headline

    def test_a_single_benchmark_model_gets_no_headline_claim(self) -> None:
        """One benchmark is not a spread -- there is nothing to disagree with."""
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none"),
            _cell("contender", "none", parameter_count=2_000_000, vs_reference_delta=0.2),
        ]
        assert "## Headline" not in LADDER.render_report(rows)

    def test_a_model_that_never_ran_is_named_not_quietly_absent(self) -> None:
        """A partial sweep must not render as a complete one."""
        # One real cell: benchmark and arm both exist in the declared grid, so
        # the gap list has to name the rest of that grid rather than shrug.
        report = LADDER.render_report(
            [_cell(LADDER.REFERENCE_MODEL, "none", benchmark=LADDER.BENCHMARKS[0], k=LADDER.CI_K)]
        )
        section = report.split("## What did not run")[1]

        # Every other model in the ladder is missing from this one-row sweep.
        for spec in LADDER.MODELS:
            if spec.name != LADDER.REFERENCE_MODEL:
                assert f"`{spec.name}`" in section
        assert "not run" in section
        # ...and the model that DID run is listed with the grid it is missing,
        # rather than reading as fully measured on one benchmark and one arm.
        assert f"`{LADDER.REFERENCE_MODEL}`" in section
        assert "instruct" in section

    def test_a_complete_sweep_says_so_instead_of_printing_an_empty_gap_table(self) -> None:
        section = LADDER.render_report(_full_grid()).split("## What did not run")[1]
        assert "Every model, benchmark and prompt arm in the ladder was measured." in section

    def test_one_missing_grid_cell_is_not_called_a_complete_sweep(self) -> None:
        """Marginals lie: every benchmark and every arm can appear with a hole left."""
        spec = LADDER.MODELS_BY_NAME["google/embeddinggemma-300m"]
        hole = (spec.name, "amazon_google", "documented", LADDER.CI_K)
        rows = [
            row for row in _full_grid() if (row.model, row.benchmark, row.prompt_arm, row.k) != hole
        ]
        section = LADDER.render_report(rows).split("## What did not run")[1]

        assert "Every model, benchmark and prompt arm in the ladder was measured." not in section
        assert "`amazon_google`/`documented`/k=20" in section

    def test_a_model_whose_only_rows_failed_is_not_filed_as_never_run(self) -> None:
        """ "Failed" and "never reached" are different facts about a model."""
        broken = LADDER.MODELS[1].name
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none"),
            LADDER.LadderRow(
                model=broken,
                benchmark="abt_buy",
                prompt_arm="-",
                k=0,
                status="failed",
                metric_revision=LADDER.METRIC_REVISION,
                error="OSError: gated repo",
            ),
        ]
        section = LADDER.render_report(rows).split("## What did not run")[1]
        row = next(line for line in section.splitlines() if f"`{broken}`" in line)

        assert "**failed**" in row
        assert "not run" not in row


class TestRecommendationSplitsOnLicence:
    """The recommendation must not let a use-restricted checkpoint read as a default.

    langres is Apache-2.0. ``google/embeddinggemma-300m`` is the best-measured
    model on some benchmarks *and* the only non-OSI licence in the ladder, so a
    recommendation that ranked purely on recall would name it as the default
    candidate. The split is the whole point of the section, and a bug in it looks
    exactly like a correct recommendation.
    """

    @staticmethod
    def _rows() -> list[Any]:
        """Gemma clearly ahead of the reference; one OSI model modestly ahead."""
        return [
            _cell(LADDER.REFERENCE_MODEL, "none"),
            _cell(
                "google/embeddinggemma-300m",
                "none",
                vs_reference_delta=0.20,
                vs_reference_ci_low=0.17,
                vs_reference_ci_high=0.23,
            ),
            _cell(
                "BAAI/bge-small-en-v1.5",
                "none",
                vs_reference_delta=0.03,
                vs_reference_ci_low=0.01,
                vs_reference_ci_high=0.05,
            ),
        ]

    def _section(self) -> str:
        report = LADDER.render_report(self._rows())
        start = report.index("## Recommendation")
        return report[start : report.index("## The recall/cost frontier")]

    def test_the_restricted_model_is_not_in_the_default_candidate_table(self) -> None:
        osi_table = self._section().split("### Use-restricted")[0]

        assert "`BAAI/bge-small-en-v1.5` | mit" in osi_table
        assert "google/embeddinggemma-300m" not in osi_table

    def test_the_best_osi_candidate_is_named_and_is_not_the_best_model(self) -> None:
        """The larger delta belongs to the restricted model; it must not win here."""
        section = self._section()

        assert "**Best OSI-licensed candidate: `BAAI/bge-small-en-v1.5`**" in section

    def test_the_restricted_model_is_named_with_its_licence_and_as_an_opt_in(self) -> None:
        restricted = self._section().split("### Use-restricted")[1]

        assert "`google/embeddinggemma-300m` — licence `gemma`" in restricted
        assert "NOT OSI-approved" in restricted
        assert "documented opt-in" in restricted
        # The measurement is still reported -- excluding it from the default
        # table must not hide that it won.
        assert "+0.2000" in restricted

    def test_an_unknown_licence_is_not_treated_as_OSI(self) -> None:
        """The allow list must fail closed: absence is not approval.

        A deny list would let a checkpoint added tomorrow, with a licence nobody
        classified, walk into the default-candidate table.
        """
        assert not LADDER._is_osi(LADDER.ModelSpec("someone/new-model"))
        assert LADDER.ModelSpec("someone/new-model").license == "unknown"

    def test_the_coverage_denominator_counts_the_whole_ladder(self) -> None:
        """ "3 of 14", never "3 of 3" -- a partial field must read as partial."""
        section = self._section()

        assert f"of the {len(LADDER.MODELS)} models in the ladder" in section
        assert "**3 of the" in section
