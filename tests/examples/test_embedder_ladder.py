"""Offline behaviour tests for the embedder-ladder harness's measurement rules.

$0 and model-free on purpose: these cover the three places the harness can
publish a *plausible but wrong* number — the reachable-recall ceiling, the
direction of the separability score, and the unit a confidence interval is
resampled over. Each of them was wrong at some point in this harness's history,
and none of them would have shown up as a crash.
"""

from __future__ import annotations

import importlib.util
import re
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


class _PromptSensitiveEmbedder:
    """Bare and prompted encodes drift independently.

    Models the real hazard: a change to explicit prompt handling that leaves
    un-prompted embeddings byte-identical.
    """

    embedding_dim = 4

    def __init__(self) -> None:
        self.prompt_scale = 1.0

    def encode(self, texts: list[str], prompt: str | None = None) -> np.ndarray:
        scale = 1.0 if prompt is None else self.prompt_scale
        return np.array([[scale * (len(t) + i) for i in range(4)] for t in texts], dtype=np.float32)


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


class _RetunableEmbedder:
    """Only LONG inputs change when ``max_seq_length`` moves.

    Models the input-selective hazard a single short probe cannot see: the
    checkpoint truncates at ``max_seq_length``, so shortening it leaves a
    five-token text bit-identical while every long record is re-cut.
    """

    embedding_dim = 4

    def __init__(self, max_seq_length: int = 512) -> None:
        self.max_seq_length = max_seq_length

    def encode(self, texts: list[str], prompt: str | None = None) -> np.ndarray:
        cut = [((prompt or "") + text)[: self.max_seq_length] for text in texts]
        return np.array([[float(len(t) + i) for i in range(4)] for t in cut], dtype=np.float32)


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

    def test_shortening_max_seq_length_is_caught(self, tmp_path: Path) -> None:
        """A short probe alone cannot see this, so the check must not rest on one.

        The knob rides in the canary TEXT, so a changed value changes the cache
        key: the partition then holds vectors with no canary under the new key
        and the legacy gate refuses it — the same refusal, reused.
        """
        embedder = _RetunableEmbedder(max_seq_length=512)
        cached = self._cached(embedder, tmp_path)
        LADDER._assert_cache_matches_checkpoint(embedder, cached, "canary_ns", tmp_path)
        cached.encode(["a corpus record that is long enough to be affected " * 40])

        embedder.max_seq_length = 128

        with pytest.raises(LADDER.StaleEmbeddingCacheError):
            LADDER._assert_cache_matches_checkpoint(embedder, cached, "canary_ns", tmp_path)

    def test_a_dead_checkpoint_is_a_model_failure_not_a_cache_refusal(self) -> None:
        """A checkpoint that cannot load must not be reported as a corrupt cache.

        Swallowing the load error substitutes ``max_seq_length=None``, which is
        a DIFFERENT canary key, so a model with a populated cache is refused as
        stale. That exits with the integrity code, and the shell driver treats
        that as "abort the whole sweep" rather than "record one model failure"
        -- so a missing checkpoint or a broken dependency would stop every other
        model and blame the cache for it.
        """

        class _DeadCheckpoint:
            def _get_model(self) -> object:
                raise OSError("checkpoint not found on disk")

        with pytest.raises(OSError, match="checkpoint not found"):
            LADDER._vector_affecting_runtime(_DeadCheckpoint())

    def test_an_embedder_with_no_max_seq_length_still_yields_a_fingerprint(self) -> None:
        """Control: an ABSENT knob is not a failure and must stay duck-typed.

        The propagation above must not turn "this embedder does not expose
        ``max_seq_length``" into an error -- that is the case the long probe
        exists for, and every fake in this file is in it.
        """

        class _NoKnob:
            pass

        assert LADDER._vector_affecting_runtime(_NoKnob()) == "max_seq_length=None"

    def test_an_unchanged_max_seq_length_still_passes(self, tmp_path: Path) -> None:
        """Control: folding the knob into the key must not refuse every warm run."""
        embedder = _RetunableEmbedder(max_seq_length=512)
        cached = self._cached(embedder, tmp_path)
        LADDER._assert_cache_matches_checkpoint(embedder, cached, "canary_ns", tmp_path)
        cached.encode(["a corpus record"])

        LADDER._assert_cache_matches_checkpoint(embedder, cached, "canary_ns", tmp_path)

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

    def test_the_refusal_is_not_bypassed_by_running_it_again(self, tmp_path: Path) -> None:
        """A refused run must leave the namespace exactly as it found it.

        The first draft asked "is the canary here?" by *encoding* it, which
        answers the question by changing it: the refused run deposited a canary,
        so the second run found one present and sailed through. A refusal you get
        past by running it twice is not a refusal.
        """
        embedder = _SwappableEmbedder()
        legacy = self._cached(embedder, tmp_path)
        legacy.encode(["some corpus text"])
        entries_before = LADDER._cache_entry_count(tmp_path / "canary_ns.db")

        for _attempt in range(2):
            with pytest.raises(LADDER.StaleEmbeddingCacheError):
                LADDER._assert_cache_matches_checkpoint(
                    embedder, self._cached(embedder, tmp_path), "canary_ns", tmp_path
                )

        assert LADDER._cache_entry_count(tmp_path / "canary_ns.db") == entries_before

    def test_a_non_finite_canary_is_maximal_drift_not_agreement(self, tmp_path: Path) -> None:
        """``NaN > tolerance`` is **False**, so a NaN would be read as agreement.

        Unstable half precision on some devices and a truncated cached blob both
        produce this. Without the explicit finiteness branch the guard accepts a
        cache it could not compare, and the ladder publishes rows computed from
        non-finite vectors.
        """
        embedder = _SwappableEmbedder()
        LADDER._assert_cache_matches_checkpoint(
            embedder, self._cached(embedder, tmp_path), "canary_ns", tmp_path
        )

        embedder.weights = float("nan")

        with pytest.raises(LADDER.StaleEmbeddingCacheError):
            LADDER._assert_cache_matches_checkpoint(
                embedder, self._cached(embedder, tmp_path), "canary_ns", tmp_path
            )

    def test_adopting_a_legacy_cache_vouches_once_and_keeps_checking(self, tmp_path: Path) -> None:
        """``--trust-existing-cache`` is a one-time assertion, not an off switch.

        The refusal above is retroactive: every namespace written before the
        canary existed would demand a full re-measure of a cache the operator may
        know is current. So adoption exists — but if it also disabled the check
        from then on, it would be an off switch wearing a one-time label, and the
        namespace would go unverified forever. This is the test that tells those
        two apart.
        """
        embedder = _SwappableEmbedder(weights=1.0)
        legacy = self._cached(embedder, tmp_path)
        legacy.encode(["some corpus text"])

        LADDER._assert_cache_matches_checkpoint(
            embedder, self._cached(embedder, tmp_path), "canary_ns", tmp_path, adopt_legacy=True
        )

        # Adoption pinned the canary, so a later run needs no flag...
        LADDER._assert_cache_matches_checkpoint(
            embedder, self._cached(embedder, tmp_path), "canary_ns", tmp_path
        )

        # ...and a checkpoint swap after adoption is still caught.
        embedder.weights = 2.0
        with pytest.raises(LADDER.StaleEmbeddingCacheError):
            LADDER._assert_cache_matches_checkpoint(
                embedder, self._cached(embedder, tmp_path), "canary_ns", tmp_path
            )


class TestIntegrityRefusalIsNotAResult:
    """A cache refusal must abort, never become a ``status="failed"`` row.

    Every other exception in that handler is a fact about the *model* — it did
    not load, it ran out of memory — and recording it as a failure row is honest.
    A cache-integrity refusal is a fact about the *harness*, and turning it into a
    row is destructive: ``main()`` persists the row and ``merge_rows()`` voids
    every previously recorded cell for that (model, benchmark), so the refusal
    would DELETE good measurements from the tracked jsonl — and ``run_ladder.sh``
    would see exit 0 and commit the deletion.
    """

    def _evaluate(self, monkeypatch: pytest.MonkeyPatch, exc: Exception) -> Any:
        def _boom(*_args: Any, **_kwargs: Any) -> Any:
            raise exc

        monkeypatch.setattr(LADDER, "_load_benchmark", lambda _name: _boom())
        return LADDER.evaluate_model_on_benchmark(
            LADDER.ModelSpec("irrelevant"),
            "fodors_zagat",
            k_values=[1],
            prompt_arms={"none": (None, None)},
            cache_dir=Path("unused"),
            device=None,
            batch_size=1,
        )

    def test_a_stale_cache_error_escapes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(LADDER.StaleEmbeddingCacheError):
            self._evaluate(monkeypatch, LADDER.StaleEmbeddingCacheError("stale"))

    def test_an_ordinary_model_failure_is_still_recorded_as_a_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control: without this, "re-raise everything" would pass the test above.

        A model that genuinely fails to load must still produce a failure row —
        that is the harness's whole no-silent-skip rule.
        """
        rows, _updates = self._evaluate(monkeypatch, RuntimeError("no such checkpoint"))

        assert [row.status for row in rows] == ["failed"]
        assert "no such checkpoint" in (rows[0].error or "")


class TestIntegrityRefusalReachesTheDriver:
    """The abort has to survive the process boundary, or the deletion just moves.

    ``run_ladder.sh`` treats any non-zero exit as "the process died" and calls
    ``record_process_failure``, which removes every existing row for the model and
    commits the replacement. Re-raising inside Python therefore was not enough on
    its own: the refusal exited non-zero and the driver deleted the good rows
    anyway. The exit CODE is the whole signal.
    """

    def test_the_reserved_code_is_not_one_the_interpreter_already_uses(self) -> None:
        # 1 is an uncaught exception, 2 is argparse. Colliding with either would
        # make the driver treat a genuine crash as an integrity refusal.
        assert LADDER.EXIT_CACHE_INTEGRITY not in (0, 1, 2)

    def test_the_driver_special_cases_that_exact_code(self) -> None:
        """The two halves are in different languages; nothing else pins them together.

        A test on the Python side alone would keep passing if someone changed
        ``EXIT_CACHE_INTEGRITY`` and left the shell comparing against 3.
        """
        driver = (ROOT / "examples" / "research" / "run_ladder.sh").read_text()

        assert f"if [ $code -eq {LADDER.EXIT_CACHE_INTEGRITY} ]" in driver
        # And it must bail before the row-deleting path, not after it.
        assert driver.index(f"$code -eq {LADDER.EXIT_CACHE_INTEGRITY}") < driver.index(
            'record_process_failure "$model"'
        )


class TestPromptedCachePartitions:
    """The cache key is (text, prompt); an unprompted canary vouches for one slice.

    The ``instruct`` and ``documented`` arms read prompt-keyed entries. A change
    that touches only explicit prompt handling — ``include_prompt``, prompt-token
    pooling — leaves bare embeddings byte-identical, so a canary that only ever
    asks for the bare vector passes while the prompted vectors it did not look at
    came from the old behaviour.
    """

    def _cached(self, embedder: Any, tmp_path: Path) -> Any:
        from langres.core.embeddings import DiskCachedEmbedder

        return DiskCachedEmbedder(embedder, cache_dir=tmp_path, namespace="canary_ns")

    def test_drift_confined_to_the_prompted_path_is_caught(self, tmp_path: Path) -> None:
        embedder = _PromptSensitiveEmbedder()
        LADDER._assert_cache_matches_checkpoint(
            embedder,
            self._cached(embedder, tmp_path),
            "canary_ns",
            tmp_path,
            prompts=[None, "query: "],
        )

        # Bare embeddings unchanged; only the prompted path moves.
        embedder.prompt_scale = 5.0

        with pytest.raises(LADDER.StaleEmbeddingCacheError) as excinfo:
            LADDER._assert_cache_matches_checkpoint(
                embedder,
                self._cached(embedder, tmp_path),
                "canary_ns",
                tmp_path,
                prompts=[None, "query: "],
            )
        assert "query: " in str(excinfo.value)

    def test_a_populated_prompt_partition_without_its_canary_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Reachable even on a namespace whose UNPROMPTED canary is present.

        That canary was pinned before per-prompt canaries existed and vouches
        only for its own partition. Treating a populated prompt partition as
        merely "cold" would pin a fresh canary over stale prompted vectors and
        accept them forever — the vacuous-canary defect again, one partition
        over. (Cross-model review.)
        """
        embedder = _PromptSensitiveEmbedder()
        # An upgraded namespace: unprompted canary pinned, prompted corpus
        # vectors present from an older run, no prompted canary.
        LADDER._assert_cache_matches_checkpoint(
            embedder, self._cached(embedder, tmp_path), "canary_ns", tmp_path, prompts=[None]
        )
        self._cached(embedder, tmp_path).encode(["corpus text"], prompt="query: ")

        with pytest.raises(LADDER.StaleEmbeddingCacheError) as excinfo:
            LADDER._assert_cache_matches_checkpoint(
                embedder,
                self._cached(embedder, tmp_path),
                "canary_ns",
                tmp_path,
                prompts=[None, "query: "],
            )
        assert "query: " in str(excinfo.value)

    def test_an_empty_prompt_partition_is_cold_and_simply_pinned(self, tmp_path: Path) -> None:
        """Control: adding an arm must not be mistaken for a stale partition.

        Without this, "refuse any partition lacking a canary" would pass the test
        above and make every newly added prompt arm unrunnable.
        """
        embedder = _PromptSensitiveEmbedder()
        LADDER._assert_cache_matches_checkpoint(
            embedder, self._cached(embedder, tmp_path), "canary_ns", tmp_path, prompts=[None]
        )

        LADDER._assert_cache_matches_checkpoint(
            embedder,
            self._cached(embedder, tmp_path),
            "canary_ns",
            tmp_path,
            prompts=[None, "query: "],
        )

    def test_counting_the_unprompted_partition_uses_IS_not_equals(self, tmp_path: Path) -> None:
        """``prompt = NULL`` is NULL in SQL, never true.

        With ``=`` the unprompted partition — the largest one — counts zero and
        every cache reports as empty, silently disabling the gate above.
        """
        embedder = _PromptSensitiveEmbedder()
        self._cached(embedder, tmp_path).encode(["a", "b"])

        assert LADDER._cache_entry_count(tmp_path / "canary_ns.db", None) == 2

    def test_the_bare_canary_alone_does_not_notice(self, tmp_path: Path) -> None:
        """Control: this is the hole, demonstrated.

        Without it the test above could pass for the wrong reason — e.g. if the
        fake's bare output moved too — and would not show that the prompted
        partition is what added the coverage.
        """
        embedder = _PromptSensitiveEmbedder()
        LADDER._assert_cache_matches_checkpoint(
            embedder, self._cached(embedder, tmp_path), "canary_ns", tmp_path, prompts=[None]
        )

        embedder.prompt_scale = 5.0

        LADDER._assert_cache_matches_checkpoint(
            embedder, self._cached(embedder, tmp_path), "canary_ns", tmp_path, prompts=[None]
        )


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
        # Two definitions of the same metric must never share a column. The
        # property is that no stale NUMBER reaches a measurement table -- not
        # that the name is unmentionable: "What did not run" naming it, with the
        # reason, is the accounting that keeps the exclusion visible.
        rows = [
            _cell("current", "none"),
            _cell(
                "legacy", "none", metric_revision=LADDER.METRIC_REVISION - 1, separability_auc=0.5
            ),
        ]
        report = LADDER.render_report(rows)
        assert "legacy" in report.split("## Models that were measured")[0]
        assert "older metric definition" in report

        body = report.split("## Models that were measured")[1]
        tables, did_not_run = body.split("## What did not run")
        # No stale row in any measurement table.
        assert "`legacy`" not in tables
        # ...and it is still accounted for, with the reason.
        assert "| `legacy` | measured under an older metric revision" in did_not_run

    def test_a_failed_custom_checkpoint_is_counted_and_then_named(self) -> None:
        """The denominator and the enumeration must describe the same ladder.

        ``--models`` accepts a checkpoint outside ``MODELS``. The coverage
        denominator counted it while "What did not run" iterated ``MODELS``, so
        the report said "N of the N+1 models" above a table that could not name
        the extra one — a promise the section could not keep.
        """
        rows = [
            _cell("all-MiniLM-L6-v2", "none"),
            LADDER.LadderRow(
                model="someone/custom-checkpoint",
                benchmark="bench",
                prompt_arm="-",
                k=0,
                status="failed",
                metric_revision=LADDER.METRIC_REVISION,
                error="OSError: gated repo",
            ),
        ]
        report = LADDER.render_report(rows)
        did_not_run = report.split("## What did not run")[1]

        assert "| `someone/custom-checkpoint` | **failed**" in did_not_run
        # The two counts are the same ladder: 14 fixed + this one.
        assert "of the 15 models in the ladder have a row" in report
        assert "of the 15 models in the ladder have no usable row" in did_not_run

    def test_a_partially_measured_custom_checkpoint_is_grid_checked(self) -> None:
        """The grid check iterated ``MODELS`` too, so a custom model escaped it.

        Without this, a custom checkpoint measured on one benchmark could leave
        the report concluding "every model, benchmark and prompt arm was
        measured".
        """
        rows = [_cell("someone/custom-checkpoint", "none")]
        report = LADDER.render_report(rows)
        did_not_run = report.split("## What did not run")[1]

        assert "Every model, benchmark and prompt arm" not in did_not_run
        assert "| `someone/custom-checkpoint` | benchmarks" in did_not_run

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

    def test_a_restricted_model_with_no_win_is_not_recommended(self) -> None:
        """A licence classification is not a performance finding.

        "Recommended as a documented opt-in" fired off the licence bucket alone,
        so a checkpoint that was compared and beat the reference nowhere still
        read as recommended. Documented opt-in is the exposure MECHANISM its
        licence requires; whether anything is recommended is a question only the
        measurement answers.
        """
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none"),
            _cell(
                "google/embeddinggemma-300m",
                "none",
                vs_reference_delta=-0.05,
                vs_reference_ci_low=-0.08,
                vs_reference_ci_high=-0.02,
            ),
        ]
        report = LADDER.render_report(rows)
        section = report[
            report.index("## Recommendation") : report.index("## The recall/cost frontier")
        ]
        restricted = section.split("### Use-restricted")[1]

        assert "Recommended as a documented opt-in" not in restricted
        assert "measured no benchmark where it beats" in restricted
        assert "required exposure mechanism" in restricted
        # The licence statement itself is unchanged -- this is about the verdict.
        assert "NOT OSI-approved" in restricted

    def test_a_restricted_model_that_was_never_compared_says_so(self) -> None:
        """Control on the other side: no interval is not a measured loss either."""
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none"),
            _cell("google/embeddinggemma-300m", "none", vs_reference_delta=None),
        ]
        report = LADDER.render_report(rows)
        restricted = report[
            report.index("## Recommendation") : report.index("## The recall/cost frontier")
        ].split("### Use-restricted")[1]

        assert "carries no interval" in restricted
        assert "missing measurement, not a verdict" in restricted
        assert "Recommended as a documented opt-in" not in restricted

    def test_a_restricted_model_that_did_win_is_still_recommended(self) -> None:
        """Control: gating on evidence must not suppress a real recommendation."""
        restricted = self._section().split("### Use-restricted")[1]

        assert "**Recommended as a documented opt-in**" in restricted

    def test_a_restricted_model_is_not_described_as_merely_unverified(self) -> None:
        """The two buckets must not collapse into each other.

        A licence that WAS read and is not OSI is a finding; keeping it in the
        hedged "someone should look at this" section would understate it.
        """
        section = self._section()

        assert "### Unclassified licences" not in section
        assert "google/embeddinggemma-300m" in section.split("### Use-restricted")[1]

    def test_an_unread_licence_is_reported_as_unverified_not_as_restricted(self) -> None:
        """``--models`` accepts a checkpoint outside ``MODELS``; it gets ``"unknown"``.

        Failing an allow list means "not shown to be OSI". Printing that as
        "licence `unknown`, which is NOT OSI-approved" under a heading reading
        *use-restricted* asserts two things the run never measured: that the
        terms were read, and that they restrict use.
        """
        rows = [
            *self._rows(),
            _cell(
                "someone/brand-new-model",
                "none",
                vs_reference_delta=0.07,
                vs_reference_ci_low=0.04,
                vs_reference_ci_high=0.10,
            ),
        ]
        report = LADDER.render_report(rows)
        section = report[
            report.index("## Recommendation") : report.index("## The recall/cost frontier")
        ]
        osi_table, rest = section.split("### Use-restricted")
        restricted, unclassified = rest.split("### Unclassified licences")

        # Kept out of the default candidates -- the allow list fails closed.
        assert "someone/brand-new-model" not in osi_table
        # ...but never accused.
        assert "someone/brand-new-model" not in restricted
        assert "`someone/brand-new-model` — licence not recorded." in unclassified
        assert "NOT OSI-approved" not in unclassified
        # The measurement still survives the hedge.
        assert "+0.0700" in unclassified

    def test_an_uncompared_unclassified_model_is_not_reported_as_measured(self) -> None:
        """The unclassified section had the SAME defect as the restricted one.

        It said "Measured ahead of `<ref>` on: no benchmark" whether the model
        was compared and did not win or was never compared at all. Fixing one
        section and not its sibling is the failure mode that made the two share
        one phrasing helper.
        """
        rows = [
            *self._rows(),
            _cell("someone/brand-new-model", "none", vs_reference_delta=None),
        ]
        report = LADDER.render_report(rows)
        unclassified = report[
            report.index("## Recommendation") : report.index("## The recall/cost frontier")
        ].split("### Unclassified licences")[1]

        assert "carries no interval" in unclassified
        assert "missing measurement, not a verdict" in unclassified
        assert "Measured ahead" not in unclassified

    def test_a_compared_unclassified_model_that_never_wins_says_so(self) -> None:
        """The middle state: compared, beat nothing. Not the same as uncompared."""
        rows = [
            *self._rows(),
            _cell(
                "someone/brand-new-model",
                "none",
                vs_reference_delta=-0.04,
                vs_reference_ci_low=-0.07,
                vs_reference_ci_high=-0.01,
            ),
        ]
        report = LADDER.render_report(rows)
        unclassified = report[
            report.index("## Recommendation") : report.index("## The recall/cost frontier")
        ].split("### Unclassified licences")[1]

        assert "measured no benchmark where it beats" in unclassified
        assert "carries no interval" not in unclassified

    def test_an_unknown_licence_is_not_treated_as_OSI(self) -> None:
        """The allow list must fail closed: absence is not approval.

        A deny list would let a checkpoint added tomorrow, with a licence nobody
        classified, walk into the default-candidate table.
        """
        assert not LADDER._is_osi(LADDER.ModelSpec("someone/new-model"))
        assert LADDER.ModelSpec("someone/new-model").license == "unknown"

    def test_a_missing_interval_is_not_reported_as_a_tie(self) -> None:
        """``merge_rows`` clears ``vs_reference_*`` when the reference is remeasured.

        Every challenger then has zero wins for the same reason it has zero
        losses: nothing was compared. Saying "the measurement cannot tell them
        apart" turns that gap into a finding of equivalence, and then into a
        reason to keep the default.
        """
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none"),
            _cell(
                "BAAI/bge-small-en-v1.5",
                "none",
                vs_reference_delta=None,
                vs_reference_ci_low=None,
                vs_reference_ci_high=None,
            ),
        ]
        report = LADDER.render_report(rows)
        section = report[
            report.index("## Recommendation") : report.index("## The recall/cost frontier")
        ]

        assert "cannot rank them at all" in section
        assert "missing measurement, not a tie" in section
        assert "cannot tell them apart" not in section

    def test_a_real_zero_win_field_still_recommends_keeping_the_default(self) -> None:
        """The control: a challenger that WAS compared and did not win.

        Without this, a check that always reported "missing measurement" would
        pass the test above while destroying the recommendation it guards.
        """
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none"),
            _cell(
                "BAAI/bge-small-en-v1.5",
                "none",
                vs_reference_delta=0.01,
                vs_reference_ci_low=-0.02,
                vs_reference_ci_high=0.04,
            ),
        ]
        report = LADDER.render_report(rows)
        section = report[
            report.index("## Recommendation") : report.index("## The recall/cost frontier")
        ]

        assert "cannot tell them apart" in section
        assert "cannot rank them at all" not in section

    def test_an_uncompared_challenger_does_not_pad_the_field(self) -> None:
        """A model with no intervals is 'not compared', never '0 of 5'.

        Zero wins has two causes -- beaten on every benchmark, or never
        compared -- and they render identically unless the table says so. The
        second silently enlarges the field a 'best candidate' is declared best
        of, which is a stronger claim than the data supports.
        """
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none"),
            _cell(
                "BAAI/bge-small-en-v1.5",
                "none",
                vs_reference_delta=0.03,
                vs_reference_ci_low=0.01,
                vs_reference_ci_high=0.05,
            ),
            _cell(
                "intfloat/e5-base-v2",
                "none",
                vs_reference_delta=None,
                vs_reference_ci_low=None,
                vs_reference_ci_high=None,
            ),
        ]
        report = LADDER.render_report(rows)
        section = report[
            report.index("## Recommendation") : report.index("## The recall/cost frontier")
        ]
        table = section.split("### Use-restricted")[0]

        uncompared_row = next(line for line in table.splitlines() if "e5-base-v2" in line)
        assert "not compared" in uncompared_row
        assert "0 of" not in uncompared_row
        # The winner is still named -- withholding the finding would be the
        # opposite error -- but the claim is scoped to what was measured.
        assert "**Best OSI-licensed candidate: `BAAI/bge-small-en-v1.5`**" in section
        assert "Best of 1 of the 2 OSI models" in section

    def test_the_winner_is_chosen_on_the_shared_benchmark_set(self) -> None:
        """More coverage must not win by having had more chances.

        A model with 2 wins from 5 attempts outranks one with 1 from 1 on raw
        counts alone, which ranks the size of the experiment rather than the
        model. The ranking therefore runs on the benchmarks every compared model
        shares.
        """
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none", benchmark="a"),
            _cell(LADDER.REFERENCE_MODEL, "none", benchmark="b"),
            # Broad: two wins, but only ONE of them on the shared benchmark.
            _cell(
                "BAAI/bge-small-en-v1.5",
                "none",
                benchmark="a",
                vs_reference_delta=0.02,
                vs_reference_ci_low=0.01,
                vs_reference_ci_high=0.03,
            ),
            _cell(
                "BAAI/bge-small-en-v1.5",
                "none",
                benchmark="b",
                vs_reference_delta=0.02,
                vs_reference_ci_low=0.01,
                vs_reference_ci_high=0.03,
            ),
            # Narrow: compared only on "a", and wins it by more.
            _cell(
                "intfloat/e5-base-v2",
                "none",
                benchmark="a",
                vs_reference_delta=0.30,
                vs_reference_ci_low=0.25,
                vs_reference_ci_high=0.35,
            ),
        ]
        report = LADDER.render_report(rows)
        section = report[
            report.index("## Recommendation") : report.index("## The recall/cost frontier")
        ]

        # On the shared set {"a"} both have one win, and e5 wins it by more.
        assert "**Best OSI-licensed candidate: `intfloat/e5-base-v2`**" in section
        assert "1 of the 1 benchmark(s) all 2 compared models share" in section

    def test_no_shared_benchmark_means_no_winner_is_named(self) -> None:
        """Disjoint coverage is not a ranking problem to solve, it is no ranking."""
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none", benchmark="a"),
            _cell(LADDER.REFERENCE_MODEL, "none", benchmark="b"),
            _cell(
                "BAAI/bge-small-en-v1.5",
                "none",
                benchmark="a",
                vs_reference_delta=0.02,
                vs_reference_ci_low=0.01,
                vs_reference_ci_high=0.03,
            ),
            _cell(
                "intfloat/e5-base-v2",
                "none",
                benchmark="b",
                vs_reference_delta=0.30,
                vs_reference_ci_low=0.25,
                vs_reference_ci_high=0.35,
            ),
        ]
        report = LADDER.render_report(rows)
        section = report[
            report.index("## Recommendation") : report.index("## The recall/cost frontier")
        ]

        assert "share no common benchmark, so no winner is named" in section
        assert "Best OSI-licensed candidate" not in section

    def test_a_win_outside_the_shared_set_blocks_the_keep_the_default_verdict(self) -> None:
        """The shared-set ranking must not erase a CI-clear win from the table.

        ``best_count`` counts only the shared benchmarks, so a challenger winning
        on a benchmark the others were never measured on scores zero — and the
        old sentence then said no model beats the reference "on any benchmark",
        contradicting the row directly above it.
        """
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none", benchmark="a"),
            _cell(LADDER.REFERENCE_MODEL, "none", benchmark="b"),
            # Shared benchmark "a": no one wins it.
            _cell(
                "BAAI/bge-small-en-v1.5",
                "none",
                benchmark="a",
                vs_reference_delta=0.01,
                vs_reference_ci_low=-0.01,
                vs_reference_ci_high=0.03,
            ),
            _cell(
                "intfloat/e5-base-v2",
                "none",
                benchmark="a",
                vs_reference_delta=0.01,
                vs_reference_ci_low=-0.01,
                vs_reference_ci_high=0.03,
            ),
            # ...but e5 wins "b" outright, and bge was never measured there.
            _cell(
                "intfloat/e5-base-v2",
                "none",
                benchmark="b",
                vs_reference_delta=0.30,
                vs_reference_ci_low=0.25,
                vs_reference_ci_high=0.35,
            ),
        ]
        report = LADDER.render_report(rows)
        section = report[
            report.index("## Recommendation") : report.index("## The recall/cost frontier")
        ]

        assert "does win" in section and "outside the shared set" in section
        assert "`intfloat/e5-base-v2`" in section
        assert "keep the current default" not in section

    def test_an_exact_tie_names_no_winner(self) -> None:
        """``name`` is in the sort key for byte-stability, not to break ties.

        Letting ``max`` resolve an exact tie reports the sort order as if it were
        a measurement.
        """
        tied = dict(
            vs_reference_delta=0.05,
            vs_reference_ci_low=0.03,
            vs_reference_ci_high=0.07,
        )
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none"),
            _cell("BAAI/bge-small-en-v1.5", "none", **tied),
            _cell("intfloat/e5-base-v2", "none", **tied),
        ]
        report = LADDER.render_report(rows)
        section = report[
            report.index("## Recommendation") : report.index("## The recall/cost frontier")
        ]

        assert "No single best OSI-licensed candidate" in section
        assert "`BAAI/bge-small-en-v1.5`" in section
        assert "`intfloat/e5-base-v2`" in section
        assert "**Best OSI-licensed candidate:" not in section

    def test_a_clear_loss_is_not_called_indistinguishable(self) -> None:
        """An interval entirely below zero distinguished the models perfectly.

        It just did so in the incumbent's favour. Reporting it as "the
        measurement cannot tell them apart" states the opposite of the finding,
        while reaching the same recommendation -- which is how a wrong reason
        survives review.
        """
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none"),
            _cell(
                "BAAI/bge-small-en-v1.5",
                "none",
                vs_reference_delta=-0.10,
                vs_reference_ci_low=-0.12,
                vs_reference_ci_high=-0.08,
            ),
        ]
        report = LADDER.render_report(rows)
        section = report[
            report.index("## Recommendation") : report.index("## The recall/cost frontier")
        ]

        assert "**measured loss**" in section
        assert "`BAAI/bge-small-en-v1.5`" in section
        assert "cannot tell them apart" not in section
        # The recommendation itself is unchanged -- only its reason.
        assert "keep the current default" in section

    def test_an_interval_spanning_zero_is_still_called_inconclusive(self) -> None:
        """The control: without it, always saying 'measured loss' would pass."""
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none"),
            _cell(
                "BAAI/bge-small-en-v1.5",
                "none",
                vs_reference_delta=-0.01,
                vs_reference_ci_low=-0.03,
                vs_reference_ci_high=0.02,
            ),
        ]
        report = LADDER.render_report(rows)
        section = report[
            report.index("## Recommendation") : report.index("## The recall/cost frontier")
        ]

        assert "cannot tell them apart" in section
        assert "**measured loss**" not in section

    def test_an_exact_zero_interval_is_a_result_not_a_gap(self) -> None:
        """``[0, 0]`` is certainty of a zero effect, which ``_ci`` already says.

        ``fodors_zagat`` produces it for real -- both arms hit recall 1.0 on
        every record -- so calling it "the measurement cannot tell them apart"
        gives the opposite evidential reading of a benchmark that measured the
        models as exactly equal.
        """
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none"),
            _cell(
                "BAAI/bge-small-en-v1.5",
                "none",
                vs_reference_delta=0.0,
                vs_reference_ci_low=0.0,
                vs_reference_ci_high=0.0,
            ),
        ]
        report = LADDER.render_report(rows)
        section = report[
            report.index("## Recommendation") : report.index("## The recall/cost frontier")
        ]

        assert "resolved them as exactly equal" in section
        assert "cannot tell them apart" not in section
        assert "keep the current default" in section

    def test_an_exact_tie_survives_a_clear_loser_in_the_same_field(self) -> None:
        """The three states are a partition, not a priority order.

        Reporting them as mutually exclusive branches meant one model with an
        interval below zero swept every exact tie into "intervals spanning
        zero" -- asserting uncertainty about the one comparison the measurement
        actually resolved.
        """
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none"),
            _cell(
                "BAAI/bge-small-en-v1.5",
                "none",
                vs_reference_delta=0.0,
                vs_reference_ci_low=0.0,
                vs_reference_ci_high=0.0,
            ),
            _cell(
                "all-MiniLM-L12-v2",
                "none",
                vs_reference_delta=-0.08,
                vs_reference_ci_low=-0.11,
                vs_reference_ci_high=-0.05,
            ),
        ]
        report = LADDER.render_report(rows)
        section = report[
            report.index("## Recommendation") : report.index("## The recall/cost frontier")
        ]

        assert "measured loss" in section
        assert "`all-MiniLM-L12-v2`" in section
        assert "resolved them as exactly equal" in section
        assert "`BAAI/bge-small-en-v1.5`" in section
        # The exact tie must NOT be described as inconclusive.
        assert "cannot tell them apart" not in section

    def test_a_wide_interval_around_zero_is_still_inconclusive(self) -> None:
        """Control: the exact-tie branch must not swallow real uncertainty."""
        rows = [
            _cell(LADDER.REFERENCE_MODEL, "none"),
            _cell(
                "BAAI/bge-small-en-v1.5",
                "none",
                vs_reference_delta=0.0,
                vs_reference_ci_low=-0.05,
                vs_reference_ci_high=0.05,
            ),
        ]
        report = LADDER.render_report(rows)
        section = report[
            report.index("## Recommendation") : report.index("## The recall/cost frontier")
        ]

        assert "cannot tell them apart" in section
        assert "resolved them as exactly equal" not in section

    def test_the_coverage_denominator_counts_the_whole_ladder(self) -> None:
        """ "3 of 14", never "3 of 3" -- a partial field must read as partial."""
        section = self._section()

        assert f"of the {len(LADDER.MODELS)} models in the ladder" in section
        assert "**3 of the" in section


# ---------------------------------------------------------------------------
# Silently-wrong checkpoints
#
# Every case below was MEASURED on the LFM2.5 checkpoints on 2026-07-29, and
# every one of them loads, embeds, and returns finite unit-norm vectors. None
# raises without these guards, so the failure mode is a published number rather
# than a crash -- which is why they are tested instead of trusted.
# ---------------------------------------------------------------------------


def _fake_auto_model(qualname: str, module: str, auto_map: dict[str, str] | None) -> Any:
    """An object shaped like a loaded ``auto_model``, with a controllable class identity."""
    cls = type(qualname, (), {"__module__": module})
    cls.__qualname__ = qualname
    instance = cls()
    instance.config = SimpleNamespace(auto_map=auto_map, model_type="lfm2")
    return instance


class TestDeclaredArchitecture:
    """``trust_remote_code`` + a natively implemented ``model_type`` is the trap."""

    def test_the_native_class_winning_over_the_declared_one_is_refused(self) -> None:
        # Measured: LFM2.5-Embedding-350M without trust_remote_code loads the
        # native CAUSAL Lfm2Model. It pools the CLS token, causal attention makes
        # that token a function of itself alone, and every text in the corpus
        # collapses to one vector -- cos(two unrelated products) == 1.0000.
        auto_model = _fake_auto_model(
            "Lfm2Model",
            "transformers.models.lfm2.modeling_lfm2",
            {"AutoModel": "modeling_lfm2_bidirectional.Lfm2BidirectionalModel"},
        )
        spec = LADDER.ModelSpec("LiquidAI/LFM2.5-Embedding-350M", trust_remote_code=True)

        with pytest.raises(LADDER.SilentlyWrongCheckpointError, match="auto_map.AutoModel"):
            LADDER._assert_declared_architecture(spec, auto_model)

    def test_the_declared_remote_class_passes(self) -> None:
        auto_model = _fake_auto_model(
            "Lfm2BidirectionalModel",
            "transformers_modules.LiquidAI.LFM2_5.abc123.modeling_lfm2_bidirectional",
            {"AutoModel": "modeling_lfm2_bidirectional.Lfm2BidirectionalModel"},
        )
        spec = LADDER.ModelSpec("LiquidAI/LFM2.5-Embedding-350M", trust_remote_code=True)

        LADDER._assert_declared_architecture(spec, auto_model)

    def test_a_right_named_class_from_the_wrong_place_is_still_refused(self) -> None:
        """Name-matching alone would pass a native class that happened to agree."""
        auto_model = _fake_auto_model(
            "Lfm2BidirectionalModel",
            "transformers.models.lfm2.modeling_lfm2",
            {"AutoModel": "modeling_lfm2_bidirectional.Lfm2BidirectionalModel"},
        )

        with pytest.raises(LADDER.SilentlyWrongCheckpointError):
            LADDER._assert_declared_architecture(
                LADDER.ModelSpec("x", trust_remote_code=True), auto_model
            )

    def test_a_checkpoint_declaring_no_auto_map_is_not_constrained(self) -> None:
        """The ordinary case: BertModel for e5-base-v2, no auto_map, nothing to check."""
        auto_model = _fake_auto_model("BertModel", "transformers.models.bert.modeling_bert", None)

        LADDER._assert_declared_architecture(LADDER.ModelSpec("intfloat/e5-base-v2"), auto_model)


class _FakeAutoClass:
    """Stands in for ``transformers.AutoModel`` / ``AutoModelForMaskedLM``."""

    def __init__(self, missing: list[str], backbone: Any | None) -> None:
        self._missing = missing
        self._backbone = backbone
        self.__name__ = "FakeAutoClass"

    def from_pretrained(self, name: str, **kwargs: Any) -> tuple[Any, dict[str, list[str]]]:
        model = SimpleNamespace(base_model_prefix="lfm2")
        model.base_model = model if self._backbone is None else self._backbone
        return model, {"missing_keys": self._missing, "unexpected_keys": []}


class TestPreflightBackbone:
    """``from_pretrained`` reports a total key mismatch as a WARNING and carries on."""

    @staticmethod
    def _install(monkeypatch: pytest.MonkeyPatch, **classes: Any) -> None:
        monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(**classes))

    def test_randomly_initialised_weights_stop_the_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Measured: AutoModel on LFM2.5-Encoder-350M matched ZERO of the
        # checkpoint's 148 tensors (missing=148, unexpected=148) because they are
        # stored under the MaskedLM wrapper's `lfm2.` prefix. The model then
        # embeds happily; two independent loads simply disagree.
        self._install(monkeypatch, AutoModel=_FakeAutoClass(["layers.0.conv.conv.weight"], None))
        spec = LADDER.ModelSpec("LiquidAI/LFM2.5-Encoder-350M", trust_remote_code=True)

        with pytest.raises(LADDER.SilentlyWrongCheckpointError, match="randomly initialised"):
            LADDER._preflight_backbone(spec)

    def test_a_clean_load_returns_the_recovered_backbone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backbone = SimpleNamespace(name="the real weights")
        self._install(monkeypatch, AutoModelForMaskedLM=_FakeAutoClass([], backbone))
        spec = LADDER.ModelSpec(
            "LiquidAI/LFM2.5-Encoder-350M",
            trust_remote_code=True,
            backbone_auto_class="AutoModelForMaskedLM",
        )

        assert LADDER._preflight_backbone(spec) is backbone

    def test_a_wrapper_exposing_no_distinct_backbone_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Substituting the wrapper for itself would leave the random weights in place."""
        self._install(monkeypatch, AutoModelForMaskedLM=_FakeAutoClass([], None))
        spec = LADDER.ModelSpec(
            "x", trust_remote_code=True, backbone_auto_class="AutoModelForMaskedLM"
        )

        with pytest.raises(LADDER.SilentlyWrongCheckpointError, match="no distinct base_model"):
            LADDER._preflight_backbone(spec)

    def test_an_ordinary_checkpoint_pays_no_extra_load(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Custom loading code is what gets the prefix wrong; the rest need no probe."""
        self._install(monkeypatch)  # any attribute access would raise AttributeError

        assert LADDER._preflight_backbone(LADDER.ModelSpec("intfloat/e5-base-v2")) is None


class _PropertyBackedTransformer:
    """A sentence-transformers ``Transformer`` as it actually is in 5.6.0.

    ``auto_model`` is a PROPERTY over a registered child named ``model``, so
    ``transformer.auto_model = x`` does not replace the child that runs.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    @property
    def auto_model(self) -> Any:
        return self.model

    def named_children(self) -> Any:
        return iter([("model", self.model)])


class _FakeBackbone:
    def __init__(self, signature: float) -> None:
        self.signature = signature

    def to(self, device: Any) -> "_FakeBackbone":
        return self

    def eval(self) -> "_FakeBackbone":
        return self


class _SwapAwareEmbedder:
    """Encodes whatever backbone the transformer currently holds."""

    def __init__(self, transformer: Any) -> None:
        self._transformer = transformer

    def encode(self, texts: Any, prompt: str | None = None) -> np.ndarray:
        return np.array([[self._transformer.auto_model.signature]] * len(texts))


class _FakeSTModel:
    """A ``SentenceTransformer`` stand-in: indexable, with a device."""

    def __init__(self, transformer: Any) -> None:
        self._transformer = transformer
        self.device = "cpu"

    def __getitem__(self, index: int) -> Any:
        return self._transformer


class TestSubstituteBackbone:
    """The fix for a silent failure was itself silently ineffective. Measured."""

    def test_the_registered_child_is_replaced_not_the_property(self) -> None:
        transformer = _PropertyBackedTransformer(_FakeBackbone(1.0))
        base = _SwapAwareEmbedder(transformer)
        backbone = _FakeBackbone(2.0)

        LADDER._substitute_backbone(
            LADDER.ModelSpec("x"), base, _FakeSTModel(transformer), backbone
        )

        # Assigning to `auto_model` would have left `model` in place; this asserts
        # the CHILD moved, which is the thing the forward pass calls.
        assert transformer.model is backbone
        assert transformer.auto_model is backbone

    def test_a_substitution_that_changes_no_vector_is_refused(self) -> None:
        """The exact failure that shipped: bit-identical vectors after the swap."""
        transformer = _PropertyBackedTransformer(_FakeBackbone(1.0))
        base = _SwapAwareEmbedder(transformer)
        # Same signature => encoding is unchanged => the swap did nothing that runs.
        backbone = _FakeBackbone(1.0)

        with pytest.raises(LADDER.SilentlyWrongCheckpointError, match="exactly 0"):
            LADDER._substitute_backbone(
                LADDER.ModelSpec("x"), base, _FakeSTModel(transformer), backbone
            )

    def test_a_backbone_held_by_no_registered_child_is_refused(self) -> None:
        transformer = SimpleNamespace(
            auto_model=_FakeBackbone(1.0), named_children=lambda: iter([])
        )

        with pytest.raises(LADDER.SilentlyWrongCheckpointError, match="registered child"):
            LADDER._substitute_backbone(
                LADDER.ModelSpec("x"),
                _SwapAwareEmbedder(transformer),
                _FakeSTModel(transformer),
                _FakeBackbone(2.0),
            )


class _ConstantEmbedder:
    """Returns the same vector whatever the prompt -- the shipped-bug signature."""

    def __init__(self, *, responds_to_prompt: bool) -> None:
        self._responds = responds_to_prompt

    def encode(self, texts: Any, prompt: str | None = None) -> np.ndarray:
        shift = 0.25 if (prompt and self._responds) else 0.0
        return np.array([[1.0 + shift, 2.0, 3.0]] * len(texts))


class TestPromptIsLive:
    """langres has shipped a silently discarded ``query_prompt`` before."""

    def test_a_prompt_that_moves_nothing_stops_the_run(self) -> None:
        with pytest.raises(LADDER.SilentlyWrongCheckpointError, match="reached nothing"):
            LADDER._assert_prompt_is_live(
                _ConstantEmbedder(responds_to_prompt=False), [None, "query: "]
            )

    def test_a_prompt_that_moves_the_vector_passes(self) -> None:
        LADDER._assert_prompt_is_live(_ConstantEmbedder(responds_to_prompt=True), [None, "query: "])

    def test_an_unprompted_arm_has_nothing_to_assert(self) -> None:
        LADDER._assert_prompt_is_live(_ConstantEmbedder(responds_to_prompt=False), [None])


class TestOneReferenceModel:
    """``--reference-model`` makes the baseline a property of the RUN, the delta of the ROW."""

    def test_two_baselines_in_one_file_are_refused(self) -> None:
        rows = [
            _cell("a", "none", vs_reference_delta=0.01, reference_model="all-MiniLM-L6-v2"),
            _cell("b", "none", vs_reference_delta=0.02, reference_model="intfloat/e5-base-v2"),
        ]

        with pytest.raises(ValueError, match="2 different baselines"):
            LADDER._assert_one_reference_model(rows)

    def test_rendering_under_a_different_baseline_is_refused(self) -> None:
        """The numbers would not change -- only the label above them."""
        row = _cell("a", "none", vs_reference_delta=0.01, reference_model="intfloat/e5-base-v2")

        with pytest.raises(ValueError, match="relabel the delta column"):
            LADDER._assert_one_reference_model([row])

    def test_rows_predating_the_field_are_read_as_the_default_baseline(self) -> None:
        """The 300 committed 2026-07-27 rows carry None and were measured against it."""
        row = _cell("a", "none", vs_reference_delta=0.01)
        assert row.reference_model is None

        LADDER._assert_one_reference_model([row])

    def test_a_delta_of_exactly_zero_still_names_its_baseline(self) -> None:
        """The guard must not fail open on the rows it exists to police.

        Regression: the baseline set was built with a truthiness filter, and
        ``0.0`` is falsy. Exact-zero deltas are not exotic -- every model ties
        on a saturated benchmark, and the committed rows carry ``+0.0000``
        there. So a file whose only comparisons were zeros produced an EMPTY
        set, returned early, and rendered under whatever baseline was passed,
        relabelling the delta column without recomputing it.
        """
        row = _cell("a", "none", vs_reference_delta=0.0, reference_model="intfloat/e5-base-v2")

        with pytest.raises(ValueError, match="relabel the delta column"):
            LADDER._assert_one_reference_model([row])

    def test_rows_without_a_delta_constrain_nothing(self) -> None:
        LADDER._assert_one_reference_model([_cell("a", "none")])


class TestLfm25Specs:
    """The shipped-default blocker, asserted rather than left to a doc sentence."""

    def test_every_lfm_checkpoint_is_recorded_as_non_osi(self) -> None:
        # LFM Open License v1.0 §5(a)-(b) condition Commercial Use on not
        # exceeding a $10,000,000 annual-revenue Threshold (§3). Apache-2.0 has no
        # such restriction, so these must never read as safe for a default.
        lfm = [spec for spec in LADDER.EXTRA_SPECS if spec.name.startswith("LiquidAI/")]

        assert len(lfm) == 3
        assert all(spec.license == "lfm1.0" for spec in lfm)
        assert not any(LADDER._is_osi(spec) for spec in lfm)

    def test_the_base_encoders_are_flagged_as_not_retrieval_tuned(self) -> None:
        by_name = {spec.name: spec for spec in LADDER.EXTRA_SPECS}

        assert by_name["LiquidAI/LFM2.5-Embedding-350M"].tuned_for_retrieval
        assert not by_name["LiquidAI/LFM2.5-Encoder-350M"].tuned_for_retrieval
        assert not by_name["LiquidAI/LFM2.5-Encoder-230M"].tuned_for_retrieval

    def test_the_base_encoders_declare_the_class_that_owns_their_weights(self) -> None:
        by_name = {spec.name: spec for spec in LADDER.EXTRA_SPECS}

        assert by_name["LiquidAI/LFM2.5-Encoder-350M"].backbone_auto_class == "AutoModelForMaskedLM"
        assert by_name["LiquidAI/LFM2.5-Encoder-230M"].backbone_auto_class == "AutoModelForMaskedLM"
        # The tuned checkpoint loads correctly through the normal path.
        assert by_name["LiquidAI/LFM2.5-Embedding-350M"].backbone_auto_class is None

    def test_the_embedding_checkpoint_carries_its_own_trained_prefixes(self) -> None:
        """Read from its config_sentence_transformers.json, not from the card prose."""
        spec = LADDER.MODELS_BY_NAME["LiquidAI/LFM2.5-Embedding-350M"]

        assert spec.documented_arm == ("document: ", "query: ")

    def test_the_extra_specs_stay_out_of_the_standing_ladder(self) -> None:
        """Adding them to MODELS would enlarge the 2026-07-27 ladder's denominator."""
        assert not {spec.name for spec in LADDER.EXTRA_SPECS} & {s.name for s in LADDER.MODELS}
        assert "LiquidAI/LFM2.5-Embedding-350M" in LADDER.MODELS_BY_NAME


class TestGeneratedReproduceCommands:
    """A published command that cannot run is worse than no command.

    ``run_ladder.sh`` refuses (exit 2) when a custom ``LADDER_ARTIFACT`` arrives
    without ``LADDER_ALL_MODELS`` -- the coverage denominator. That guard and the
    generated "How to reproduce" block were added in the same PR and the second
    was never re-checked against the first, so every committed reproduce command
    exited 2 before measuring anything. Verified against the real committed
    reports, because that is where a reader copies from.
    """

    REPORTS = (
        "docs/research/20260727_embedder_ladder.md",
        "docs/research/20260729_lfm25_tuned.md",
        "docs/research/20260729_lfm25_base_encoders.md",
    )

    @staticmethod
    def _driver_requires_all_models() -> bool:
        """Read the requirement off the driver, so this test cannot drift from it."""
        driver = (ROOT / "examples" / "research" / "run_ladder.sh").read_text()
        return "LADDER_ARTIFACT is set but LADDER_ALL_MODELS is not" in driver

    def test_every_command_setting_a_custom_artifact_also_sets_the_denominator(self) -> None:
        assert self._driver_requires_all_models(), (
            "the driver no longer enforces this; update or delete this test rather "
            "than letting it pass vacuously"
        )

        offenders: list[str] = []
        for name in self.REPORTS:
            path = ROOT / name
            if not path.exists():  # pragma: no cover - artifact not generated yet
                continue
            for block in re.findall(r"```bash\n(.*?)```", path.read_text(), re.S):
                for command in block.split("\n\n"):
                    if "LADDER_ARTIFACT=" not in command:
                        continue
                    if "LADDER_ALL_MODELS=" not in command:
                        offenders.append(f"{name}: {command.splitlines()[0]}")

        assert not offenders, "reproduce commands that would exit 2:\n" + "\n".join(offenders)


class TestArtifactPathsMustShareOnePrefix:
    """A report that mis-states which rows it came from is unreproducible.

    ``--rows``/``--report``/``--reference`` are independent flags, but the
    reproduce block derives all three from ``--report``'s prefix and
    ``run_ladder.sh`` takes a single ``LADDER_ARTIFACT``. Mismatched paths did
    not merely render an odd command: the report cited a rows file the run never
    read, and the two ``run_ladder.sh`` commands beside it could not be made
    correct at all -- there is no way to express divergent paths in them.
    """

    @staticmethod
    def _argv(rows: Path, report: Path, reference: Path) -> list[str]:
        return [
            "--render-only",
            "--rows",
            str(rows),
            "--report",
            str(report),
            "--reference",
            str(reference),
        ]

    def test_the_defaults_agree(self) -> None:
        """The relation has to hold for the shipped defaults or nothing can run."""
        stem = str(LADDER.DEFAULT_REPORT_PATH).removesuffix(".md")

        assert str(LADDER.DEFAULT_ROWS_PATH) == f"{stem}_rows.jsonl"
        assert str(LADDER.DEFAULT_REFERENCE_PATH) == f"{stem}_reference_recall.json"

    def test_a_rows_path_from_another_study_is_refused(self, tmp_path: Path) -> None:
        report = tmp_path / "study.md"
        argv = self._argv(
            tmp_path / "other_rows.jsonl", report, tmp_path / "study_reference_recall.json"
        )

        with pytest.raises(SystemExit) as excinfo:
            LADDER.main(argv)

        assert excinfo.value.code == 2
        assert not report.exists(), "the refusal must precede every write"

    def test_a_reference_path_from_another_study_is_refused(self, tmp_path: Path) -> None:
        argv = self._argv(
            tmp_path / "study_rows.jsonl", tmp_path / "study.md", tmp_path / "other_reference.json"
        )

        with pytest.raises(SystemExit) as excinfo:
            LADDER.main(argv)

        assert excinfo.value.code == 2

    def test_matching_paths_are_accepted(self, tmp_path: Path) -> None:
        """The guard must not fire on the combination the driver actually passes."""
        rows = tmp_path / "study_rows.jsonl"
        rows.write_text("")
        report = tmp_path / "study.md"

        LADDER.main(self._argv(rows, report, tmp_path / "study_reference_recall.json"))

        assert report.exists()


class TestTheArtifactGuardDoesNotTrustGitSilence:
    """``git status`` says nothing for three different reasons; one is safe.

    ``LADDER_ARTIFACT`` takes any prefix, and ``tmp/my_study`` is an obvious
    thing to try. For an ignored or external path ``git status --porcelain``
    prints nothing at all, so an existing artifact read as CLEAN and the harness
    rewrote it -- destroying the only copy, since an ignored file's contents are
    in no commit. Executable proof lives in ``tmp/probe_round19.sh``; this pins
    the guard so it cannot quietly go back to trusting silence.
    """

    @staticmethod
    def _driver() -> str:
        return (ROOT / "examples" / "research" / "run_ladder.sh").read_text()

    def test_untracked_existing_artifacts_are_treated_as_dirty(self) -> None:
        driver = self._driver()

        assert 'git ls-files -- "$artifact"' in driver
        assert "artifact_is_dirty()" in driver

    def test_a_path_git_refuses_to_describe_is_not_called_clean(self) -> None:
        """Outside the repository, ``git status`` exits non-zero rather than empty."""
        assert 'if ! status=$(git status --porcelain --untracked-files=all -- "$artifact"' in (
            self._driver()
        )


class TestHistoryStaysWithItsOwnDocument:
    """Two blocks in this generator are history, not measurement.

    The correction to the merged #239 PR body and the pilot claim that motivated
    the ladder are both about the 2026-07-27 portfolio ladder. Emitted for every
    artifact they interpolated the CURRENT study's headline into a sentence about
    a different PR -- the study-B report told readers that #239 had described
    ``LFM2.5-Encoder-350M``'s ``-0.0536`` as query-only, and claimed to re-measure
    two models absent from its own rows.
    """

    STUDIES = (
        "docs/research/20260729_lfm25_tuned.md",
        "docs/research/20260729_lfm25_base_encoders.md",
    )
    PORTFOLIO = "docs/research/20260727_embedder_ladder.md"

    def test_the_gate_is_derived_from_the_module_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not a typed date string, which would drift from the paths it names.

        ``ARTIFACT_PREFIX`` is a module global that ``main()`` rebinds, so this
        sets it explicitly rather than relying on whatever an earlier test in the
        session left behind.
        """
        default = str(LADDER.DEFAULT_REPORT_PATH.relative_to(LADDER.REPO_ROOT)).removesuffix(".md")
        monkeypatch.setattr(LADDER, "ARTIFACT_PREFIX", default)

        assert LADDER._is_portfolio_ladder() is True

        monkeypatch.setattr(LADDER, "ARTIFACT_PREFIX", "docs/research/20260729_lfm25_tuned")

        assert LADDER._is_portfolio_ladder() is False

    def test_no_study_report_carries_the_239_correction(self) -> None:
        for name in self.STUDIES:
            path = ROOT / name
            if not path.exists():  # pragma: no cover - artifact not generated yet
                continue
            assert "supersedes the merged #239" not in path.read_text(), name

    def test_no_study_report_claims_to_remeasure_the_pilot_models(self) -> None:
        for name in self.STUDIES:
            path = ROOT / name
            if not path.exists():  # pragma: no cover - artifact not generated yet
                continue
            assert "The claim that started this sweep" not in path.read_text(), name

    def test_the_portfolio_ladder_keeps_both(self) -> None:
        """Scoping must not silently delete the correction from the document it corrects."""
        path = ROOT / self.PORTFOLIO
        if not path.exists():  # pragma: no cover - artifact not generated yet
            pytest.skip("portfolio ladder report not generated")
        text = path.read_text()

        assert "supersedes the merged #239" in text
        assert "The claim that started this sweep" in text


class TestTheResumeGoesThroughTheGuardedDriver:
    """A resume writes into rows a killed sweep may have left uncommitted.

    Calling ``embedder_ladder.py`` directly skipped the dirty-artifact refusal,
    the provenance ``--verify`` and -- worst -- the COMMIT, leaving an expensive
    re-measured cell to die with the worktree.
    """

    @staticmethod
    def _resume() -> str:
        return (ROOT / "examples" / "research" / "resume_lfm25_study_a.sh").read_text()

    def test_it_does_not_invoke_the_harness_directly(self) -> None:
        """The COMMAND, not the prose: the comment explains why it no longer does."""
        commands = [
            line
            for line in self._resume().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

        assert not any("embedder_ladder.py" in line for line in commands)

    def test_it_calls_the_driver(self) -> None:
        resume = self._resume()

        assert "bash examples/research/run_ladder.sh" in resume
        assert 'LADDER_BENCHMARKS="walmart_amazon"' in resume

    def test_the_driver_honours_the_benchmark_override(self) -> None:
        """A flag the driver ignores would silently re-run the whole study."""
        driver = (ROOT / "examples" / "research" / "run_ladder.sh").read_text()

        assert 'BENCHMARKS="${LADDER_BENCHMARKS:-' in driver

    def test_the_coverage_denominator_is_still_the_whole_study(self) -> None:
        """Collapsing it to the measured model is a regression this harness shipped once."""
        resume = self._resume()

        assert 'LADDER_ALL_MODELS="$STUDY_A_MODELS"' in resume
        assert "BAAI/bge-base-en-v1.5" in resume


def _caveat(report: str) -> str:
    """Just the instruction-axis blockquote.

    Slicing to the end of the report swept in later sections that legitimately
    name the baseline model, so the assertion passed or failed on unrelated text.
    """
    start = report.index("Do not read this table as")
    return report[start : report.index("by a different route.", start)]


class TestTheProseNamesOnlyModelsItMeasured:
    """The instruction-axis caveat explained each study using a hand-typed model list.

    Rendered into the base-encoder study that list named `all-MiniLM-*`,
    `all-mpnet-base-v2` and BGE -- none of which have a row there -- while naming
    none of the models that do. The worked examples in the paragraph immediately
    above it did the same with `google/embeddinggemma-300m` and
    `Qwen/Qwen3-Embedding-*`: the SECOND site of one defect, which is how five
    earlier findings on this branch got half-fixed.
    """

    @staticmethod
    def _specs(*names: str) -> tuple[Any, ...]:
        return tuple(LADDER.MODELS_BY_NAME[name] for name in names)

    @staticmethod
    def _rows(specs: tuple[Any, ...]) -> list[Any]:
        return [
            _cell(spec.name, arm)
            for spec in specs
            for arm in LADDER.arms_for(spec, LADDER.PROMPT_ARMS)
        ]

    def test_the_lists_come_from_the_ladder_being_rendered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base_study = self._specs(
            "LiquidAI/LFM2.5-Embedding-350M",
            "LiquidAI/LFM2.5-Encoder-350M",
            "LiquidAI/LFM2.5-Encoder-230M",
        )
        monkeypatch.setattr(LADDER, "LADDER", base_study)

        report = LADDER.render_report(self._rows(base_study))

        caveat = _caveat(report)
        assert "LiquidAI/LFM2.5-Encoder-350M" in caveat
        for absent in ("all-MiniLM-L6-v2", "all-mpnet-base-v2", "BAAI/bge-base-en-v1.5"):
            assert absent not in caveat, absent

    def test_it_makes_no_claim_about_how_a_checkpoint_was_TRAINED(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`documented_arm` records what THIS HARNESS configured, nothing more.

        Read as "was not trained with a query-side instruction" it was false for
        `intfloat/e5-base-v2` (native `query:`/`passage:`) and for the
        Qwen3-Embedding rows -- which the paragraph above simultaneously
        described as HAVING a documented instruction, two paragraphs apart in one
        generated document. Deriving a claim beats typing it only when the field
        means what the sentence says.
        """
        portfolio = self._specs("intfloat/e5-base-v2", "google/embeddinggemma-300m")
        monkeypatch.setattr(LADDER, "LADDER", portfolio)

        caveat = _caveat(LADDER.render_report(self._rows(portfolio)))

        assert "intfloat/e5-base-v2" in caveat
        assert "were not trained with one" not in caveat
        assert "never asked for one" not in caveat
        # And it says outright that absence here is not absence of training.
        assert "not about how a" in caveat

    def test_the_worked_examples_are_dropped_when_their_model_is_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The twin site: the paragraph above the blockquote."""
        base_study = self._specs(
            "LiquidAI/LFM2.5-Embedding-350M",
            "LiquidAI/LFM2.5-Encoder-350M",
        )
        monkeypatch.setattr(LADDER, "LADDER", base_study)

        report = LADDER.render_report(self._rows(base_study))

        assert "prefixes documents with" not in report
        assert "Qwen/Qwen3-Embedding-*`'s documented instruction" not in report

    def test_the_portfolio_ladder_keeps_its_worked_examples(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control: those examples are correct where the models were measured."""
        portfolio = self._specs("google/embeddinggemma-300m", "all-MiniLM-L6-v2")
        monkeypatch.setattr(LADDER, "LADDER", portfolio)

        report = LADDER.render_report(self._rows(portfolio))

        assert "prefixes documents with" in report
        assert "all-MiniLM-L6-v2" in _caveat(report)


class TestTheLicenceSentenceDescribesTheActualLicence:
    """ "in Gemma's case a prohibited-use policy" was rendered for EVERY restricted model.

    So the LiquidAI `lfm1.0` entries described the wrong licence entirely, in the
    one paragraph a reader consults to make a legal decision.
    """

    def test_an_lfm_checkpoint_gets_the_lfm_restriction(self) -> None:
        sentence = LADDER._licence_restriction("lfm1.0")

        assert "LFM Open License" in sentence
        assert "$10M" in sentence
        assert "Gemma" not in sentence

    def test_a_gemma_checkpoint_still_gets_gemma_s(self) -> None:
        assert "prohibited-use policy" in LADDER._licence_restriction("gemma")

    def test_an_unknown_licence_invents_no_specifics(self) -> None:
        """Silent about a licence it does not know, never wrong about one."""
        sentence = LADDER._licence_restriction("some-new-licence-2.0")

        assert "read the checkpoint's own LICENSE" in sentence
        assert "$10M" not in sentence
        assert "prohibited-use policy" not in sentence
