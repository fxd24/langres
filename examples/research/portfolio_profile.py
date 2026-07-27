"""B2 -- is the benchmark portfolio trustworthy? Profile every registered dataset.

Every claim langres makes is measured on this portfolio. If a benchmark is
**saturated** (a free method already ties the literature, so it has no headroom
to rank anything) or **structurally unrepresentative** (an artifact in the gold
labels makes the metric measure the labeling rather than the task), then the
numbers we report are about the benchmark, not the method. This script is the
calibration of our own instrument.

It iterates :func:`~langres.data.registry.list_benchmarks` -- whatever is
registered, no per-dataset special-casing -- and for each loadable entry emits:

* the **distribution facts**, straight from
  :class:`~langres.data.data_profile.DataProfileReport` (class balance /
  prevalence / imbalance, duplicate-cluster-size distribution, field
  sparsity + string length, a rapidfuzz separability AUC as the pair-difficulty
  proxy, and the two-source **vocabulary overlap**);
* the **saturation measurement** -- ``rapidfuzz`` graded on the dataset's own
  fixed *literature* train/valid/test split at a threshold derived from train
  (:func:`~langres.data.fixed_split_pair_benchmark.evaluate_fixed_split_honest`),
  which is the only pair-F1 that is apples-to-apples with a published
  DeepMatcher/Ditto number;
* the **reproduction status** -- whether the dataset's files survive the PyPI
  wheel's ``[tool.hatch.build] exclude`` list, computed from ``pyproject.toml``
  rather than restated, so it cannot drift. A benchmark that exists only in a
  git checkout has a different reproduction status from one a ``pip`` user can
  load, and "excluded from this install" is a different failure from "broken".

Two verdict rules, **fixed before looking** (per the research agenda):

* **Saturated** := the free ``rapidfuzz`` method scores within
  :data:`SATURATION_MARGIN` of the published SOTA on the same split. If a free
  method ties the literature, the set cannot rank methods.
* **Structurally caveated** := the gold labels carry an artifact
  (:func:`structural_caveats`) that makes the metric describe the labeling.

Nothing is retired on this basis -- the deliverable is an **annotation**. A
saturated set is still a fine regression test; it just cannot be evidence for a
method.

Everything is **$0**: rapidfuzz + the profiler are core deps; no paid call, no
network (the vocabulary/label/field sections never touch an embedder).

Run::

    uv run python examples/research/portfolio_profile.py
    uv run python examples/research/portfolio_profile.py --only fodors_zagat

``print`` is allowed in examples (this is an operator tool).
"""

import os

# Pin OpenMP / FAISS threading BEFORE importing anything that may pull torch/faiss.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import fnmatch  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import tomllib  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from pydantic import BaseModel  # noqa: E402

from langres.core.method_registry import get_method  # noqa: E402
from langres.data.data_profile import DataProfileReport  # noqa: E402
from langres.data.fixed_split_pair_benchmark import (  # noqa: E402
    FixedSplitPairBenchmark,
    HonestPairEval,
    evaluate_fixed_split_honest,
)
from langres.data.registry import BenchmarkEntry, get_benchmark, list_benchmarks  # noqa: E402

logger = logging.getLogger("portfolio_profile")

#: How close a free method must get to the published SOTA for the set to count as
#: saturated. The research agenda's rule, fixed before any number was looked at.
SATURATION_MARGIN = 0.02

#: A gold set this small cannot separate methods: at the ~100-pair sets this rule
#: is aimed at, one pair is worth ~1% of F1 (``fodors_zagat``: 1/112 = 0.9%).
TINY_GOLD_PAIRS = 200

#: A gold "entity" this large in a two-source linkage set is a transitive-closure
#: artifact of the labeling long before it is 10+ genuinely identical records.
LARGE_COMPONENT = 10

#: Below this shared-token coverage the two sides barely share a vocabulary, so a
#: lexical method is measuring the encoding gap rather than entity resolution.
LEXICAL_GAP_COVERAGE = 0.5

#: Below this share of gold pairs being cross-source, a cross-source pipeline is
#: chasing a recall ceiling it cannot reach: recall and Pair Completeness are
#: bounded by the labeling rather than by the method. F1 is bounded too, but at a
#: HIGHER number -- with perfect precision, ``r`` caps F1 at ``2r/(1+r)``, so a
#: 0.8396 recall ceiling still permits F1 0.9128. Do not read this value as an F1
#: bound. 0.95 -- i.e. flag it as soon as more than one gold pair in twenty is
#: structurally unreachable.
CAPPED_RECALL_CEILING = 0.95

#: Repo root, resolved from this file, so the script runs from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]


class PublishedResult(BaseModel):
    """A published SOTA number this repo actually records, with its provenance.

    Never a remembered number: ``source`` must name the file (and line) in this
    repository that states it, so the saturation verdict is auditable and a stale
    entry is findable. A benchmark with no recorded number simply has no
    saturation verdict -- which is itself reported, not silently treated as
    "unsaturated".

    Attributes:
        f1: The published pair-level F1 on the dataset's standard test split.
        system: The system that reported it (e.g. ``"Ditto"``).
        source: Where in this repository that number is written down.
    """

    f1: float
    system: str
    source: str


#: Published pair-F1 per benchmark, populated ONLY from numbers recorded in this
#: repository (see :class:`PublishedResult`). Absent = not recorded here, so that
#: benchmark simply has no saturation verdict -- which is reported, never silently
#: read as "unsaturated". Four of the nine loadable sets are in that position:
#: ``dblp_scholar``, ``walmart_amazon``, ``wdc_computers`` and ``febrl_person``
#: have **no** published F1 written down anywhere in this repo, and
#: ``tiny_fixture`` is explicitly not a benchmark
#: (``datasets/tiny_fixture/ATTRIBUTION.md``).
PUBLISHED_SOTA: dict[str, PublishedResult] = {
    "abt_buy": PublishedResult(
        f1=0.893,
        system="Ditto (Li et al., VLDB 2021)",
        source="data/benchmarks/phase1/PHASE1_RESULTS.md:8 (Ditto F1 column)",
    ),
    "amazon_google": PublishedResult(
        f1=0.756,
        system="Ditto (Li et al., VLDB 2021)",
        source="data/benchmarks/phase1/PHASE1_RESULTS.md:7 (Ditto F1 column)",
    ),
    # The cited line reads "reported SOTA pairwise F1 ~0.98" and names NO system,
    # so neither does this entry -- attributing it would be an invented citation.
    # The "~" also makes 0.98 an approximation used as an exact comparand; the
    # verdict survives it comfortably (measured gap 0.0895 vs a 0.02 margin).
    "dblp_acm": PublishedResult(
        f1=0.98,
        system="reported SOTA (~0.98; source names no system)",
        source="src/langres/data/datasets/dblp_acm/ATTRIBUTION.md:8",
    ),
    "fodors_zagat": PublishedResult(
        f1=1.00,
        system="ZeroER (Wu et al., SIGMOD 2020)",
        source="docs/research/20260701_er_seam_audit.md:41",
    ),
}


class BenchmarkProfile(BaseModel):
    """One registered benchmark's distribution facts, saturation, and verdicts.

    Attributes:
        name: Registry name.
        task: ``"linkage"`` / ``"dedup"`` -- the portfolio-level gap this exposes.
        domain: Free-text entity domain.
        loadable: Whether the dataset ships in-repo at all.
        loaded: Whether ``load()`` actually succeeded here.
        load_error: The failure, when it did not.
        wheel_loadable: Whether a ``pip install langres`` can load this dataset at
            all -- ``True`` only when the dataset vendors data files and
            ``pyproject.toml`` excludes **none** of them. Reproduction status, not
            a quality signal. ``abt_buy``/``amazon_google`` ship their
            ``peeters_sampled_test.csv`` pair set but not their corpus tables, so
            they are partially shipped and NOT loadable.
        n_files_shipped: Dataset data files that survive the exclude list.
        excluded_files: The dataset files ``pyproject.toml`` drops from the wheel.
        n_records: Corpus size.
        n_clusters: Gold equivalence classes (singletons included).
        n_singletons: Gold clusters of size 1.
        max_cluster_size: Largest gold cluster -- the closure-artifact tell.
        positive_pairs: Within-cluster gold pairs.
        prevalence: Positive-pair prevalence (class balance).
        imbalance_ratio: Negatives per positive.
        entropy_bits: Cluster-size distribution entropy.
        size_distribution: ``(cluster_size, n_clusters)`` rows.
        separability_auc: rapidfuzz string separability AUC -- the pair-difficulty
            proxy the profiler already computes.
        n_fields: Fields in the corpus.
        min_non_null_rate: The sparsest field's completeness.
        mean_field_len: Mean string length across the profiled fields.
        vocab_jaccard: Type-level vocabulary overlap between the two sources.
        vocab_min_coverage: The lower of the two occurrence-weighted token
            coverages -- what a string comparator actually experiences.
        cross_source_ceiling: The share of gold pairs a cross-source candidate
            set can ever contain (see :func:`cross_source_ceiling`) -- a hard
            recall ceiling no blocker can lift.
        rapidfuzz_f1: Honest pair F1 on the dataset's fixed literature test split
            (threshold derived on train). ``None`` when no fixed split ships --
            or when the measurement failed, which ``saturation_error``
            distinguishes.
        saturation_error: The failure, when the measurement raised. ``None``
            alongside a ``None`` ``rapidfuzz_f1`` means "no fixed split ships",
            not "we could not tell".
        rapidfuzz_threshold: The threshold that F1 was measured at.
        rapidfuzz_argmax_f1: The leaky argmax-on-test F1, for the honesty delta.
        published_f1: The recorded published number, when there is one.
        published_system: Which system reported it.
        saturated: The saturation verdict; ``None`` when it cannot be decided.
        caveats: Structural caveats (see :func:`structural_caveats`).
    """

    name: str
    task: str
    domain: str
    loadable: bool
    loaded: bool
    load_error: str | None = None
    wheel_loadable: bool
    n_files_shipped: int
    excluded_files: list[str]
    n_records: int | None = None
    n_clusters: int | None = None
    n_singletons: int | None = None
    max_cluster_size: int | None = None
    positive_pairs: int | None = None
    prevalence: float | None = None
    imbalance_ratio: float | None = None
    entropy_bits: float | None = None
    size_distribution: list[tuple[int, int]] = []
    separability_auc: float | None = None
    n_fields: int | None = None
    min_non_null_rate: float | None = None
    mean_field_len: float | None = None
    vocab_jaccard: float | None = None
    vocab_min_coverage: float | None = None
    cross_source_ceiling: float | None = None
    rapidfuzz_f1: float | None = None
    saturation_error: str | None = None
    rapidfuzz_threshold: float | None = None
    rapidfuzz_argmax_f1: float | None = None
    published_f1: float | None = None
    published_system: str | None = None
    saturated: bool | None = None
    caveats: list[str] = []


def wheel_exclusions() -> list[str]:
    """The ``[tool.hatch.build] exclude`` path globs, read from ``pyproject.toml``.

    Read rather than restated: the patterns are documented in ``pyproject.toml``
    as path literals that fail silently when a directory is renamed, so a copy
    here would be a second thing to keep in sync.
    """
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        config = tomllib.load(fh)
    excludes = config.get("tool", {}).get("hatch", {}).get("build", {}).get("exclude", [])
    return [str(pattern) for pattern in excludes]


def wheel_status(name: str, patterns: list[str]) -> tuple[bool, int, list[str]]:
    """Which of a dataset's data files survive the wheel's exclude list.

    ``.md`` files (ATTRIBUTION / SOURCE) are ignored: they always ship and never
    make a dataset loadable.

    Args:
        name: Registry name; the dataset directory is
            ``src/langres/data/datasets/<name>``.
        patterns: The exclude globs from :func:`wheel_exclusions`.

    Returns:
        ``(wheel_loadable, n_shipped, excluded_files)``. ``wheel_loadable`` is
        ``True`` only when the dataset vendors data files and **none** are
        excluded -- a partially-shipped dataset still raises
        ``BenchmarkDataNotFoundError`` on ``load()``, so "some files ship" is not
        the useful predicate. A dataset with no directory (``opensanctions``)
        reports ``(False, 0, [])``: nothing ships because nothing is vendored.
    """
    directory = REPO_ROOT / "src" / "langres" / "data" / "datasets" / name
    if not directory.is_dir():
        return False, 0, []
    excluded: list[str] = []
    shipped = 0
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix == ".md":
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in patterns):
            excluded.append(relative)
        else:
            shipped += 1
    return shipped > 0 and not excluded, shipped, excluded


def measure_saturation(entry: BenchmarkEntry, schema: type[Any]) -> HonestPairEval | None:
    """Grade ``rapidfuzz`` on the dataset's fixed literature split, honestly.

    Uniform probe, not per-dataset wiring: a loader module is asked for
    ``load_<name>`` and ``load_<name>_pair_splits`` by convention. A dataset with
    no literature split (``fodors_zagat`` / ``febrl_person`` ship a perfect
    mapping instead) simply has no comparable measurement, and this returns
    ``None`` rather than inventing a split of its own -- a self-made split is not
    comparable to a published number, which is the whole point of the exercise.

    Args:
        entry: The registry entry (its ``module_path`` holds the loaders).
        schema: The record schema, for the auto-derived comparator + matcher.

    Returns:
        A ``HonestPairEval``, or ``None`` when the dataset ships no fixed split.
    """
    module = importlib.import_module(entry.module_path)
    corpus_loader = getattr(module, f"load_{entry.name}", None)
    split_loader = getattr(module, f"load_{entry.name}_pair_splits", None)
    if corpus_loader is None or split_loader is None:
        return None
    fixed = FixedSplitPairBenchmark.from_loaders(
        name=entry.name,
        schema=schema,
        corpus_loader=corpus_loader,
        pair_split_loader=split_loader,
    )
    matcher = get_method("rapidfuzz").build(schema)
    return evaluate_fixed_split_honest(matcher, fixed)


def structural_caveats(profile: BenchmarkProfile) -> list[str]:
    """Flag the label-structure artifacts that make a metric describe the labels.

    Each rule is a measured predicate over numbers the profiler already computed,
    stated before any dataset was looked at:

    * ``tiny-gold`` -- fewer than :data:`TINY_GOLD_PAIRS` positive pairs, so one
      pair moves F1 by ~1% and the set cannot resolve a method difference.
    * ``one-to-one`` -- every gold cluster is exactly a pair and nothing is a
      singleton: matching is 1:1 by construction, which is an assignment problem,
      not the many-to-many ER task.
    * ``large-component`` -- a gold cluster of :data:`LARGE_COMPONENT`+ records in
      a two-source linkage set. B1's diagnostic in the gold labels: the label
      itself is a transitive closure, so the metric rewards chaining.
    * ``lexical-gap`` -- the two sources share less than
      :data:`LEXICAL_GAP_COVERAGE` of their token occurrences, so a string method
      is measuring the encoding gap, not entity resolution.

    Args:
        profile: A populated profile (loaded; unloaded ones get no caveats).

    Returns:
        The caveat tags, sorted.
    """
    caveats: list[str] = []
    if profile.positive_pairs is not None and profile.positive_pairs < TINY_GOLD_PAIRS:
        caveats.append("tiny-gold")
    if (
        profile.max_cluster_size == 2
        and profile.n_singletons == 0
        and profile.n_clusters is not None
        and profile.n_clusters > 0
    ):
        caveats.append("one-to-one")
    # Scoped to linkage on purpose: the rule reads a big cluster as a closure
    # artifact of two-source labeling. In a DEDUP corpus a 10-record entity is
    # ordinary, so firing there would brand a healthy dataset. No registered entry
    # is a dedup set today (that is itself a portfolio gap) -- this keeps the rule
    # honest for the one that eventually is.
    if (
        profile.task == "linkage"
        and profile.max_cluster_size is not None
        and profile.max_cluster_size >= LARGE_COMPONENT
    ):
        caveats.append("large-component")
    if profile.vocab_min_coverage is not None and profile.vocab_min_coverage < LEXICAL_GAP_COVERAGE:
        caveats.append("lexical-gap")
    if (
        profile.cross_source_ceiling is not None
        and profile.cross_source_ceiling < CAPPED_RECALL_CEILING
    ):
        caveats.append("capped-recall")
    return sorted(caveats)


def _section(report: DataProfileReport, kind: str) -> Any | None:
    """The first section of ``kind`` in ``report``, or ``None`` if absent."""
    return next((s for s in report.sections if s.kind == kind), None)


def cross_source_ceiling(corpus: Sequence[Any], gold_clusters: Sequence[set[str]]) -> float | None:
    """The share of gold pairs a **cross-source** candidate set can ever contain.

    A two-source linkage pipeline pairs A-records against B-records, so it can
    never emit a pair whose two records come from the same side. Any gold cluster
    spanning 3+ records emits such pairs, and every one of them is unreachable --
    a **recall ceiling below 1.0 that no blocker, however good, can lift**. It
    caps recall and Pair Completeness directly, so it belongs beside the
    saturation verdict rather than being rediscovered per experiment.

    **It is not an F1 ceiling.** F1 is bounded too, but higher: with perfect
    precision and recall capped at ``r``, F1 reaches ``2r/(1+r)`` -- so
    ``amazon_google``'s 0.8396 still permits F1 0.9128. Reading this number as an
    F1 bound would mark achievable scores impossible.

    This is the same phenomenon as ``dblp_scholar``'s 37-record component (§5),
    measured portfolio-wide: there it is severe enough (0.3977) to have been
    mistaken for a blocking failure.

    Args:
        corpus: The loaded records (each carrying a ``source`` field).
        gold_clusters: The gold partition.

    Returns:
        ``cross_source_gold_pairs / all_gold_pairs``, or ``None`` for a corpus
        with no ``source`` field (nothing two-sided to bound) or no gold pairs.
    """
    source_of = {
        record.id: getattr(record, "source", None)
        for record in corpus
        if getattr(record, "source", None) is not None
    }
    if not source_of:
        return None
    total = cross = 0
    for cluster in gold_clusters:
        members = sorted(cluster)
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                total += 1
                if source_of.get(left) != source_of.get(right):
                    cross += 1
    return cross / total if total else None


def profile_entry(entry: BenchmarkEntry, patterns: list[str]) -> BenchmarkProfile:
    """Profile one registry entry end to end (never raises on a broken loader).

    Args:
        entry: The registry entry.
        patterns: Wheel exclude globs from :func:`wheel_exclusions`.

    Returns:
        A :class:`BenchmarkProfile`. A load failure is *recorded* on the profile
        (``loaded=False`` + ``load_error``) rather than raised, so one broken
        dataset cannot hide the other nine -- and so "excluded from this install"
        stays distinguishable from "actually broken" via ``ships_in_wheel``.
    """
    wheel_loadable, n_shipped, excluded = wheel_status(entry.name, patterns)
    profile = BenchmarkProfile(
        name=entry.name,
        task=entry.task,
        domain=entry.domain,
        loadable=entry.loadable,
        loaded=False,
        wheel_loadable=wheel_loadable,
        n_files_shipped=n_shipped,
        excluded_files=excluded,
    )
    if not entry.loadable:
        profile.load_error = "external-only: not vendored (see the registry fetch hint)"
        return profile

    try:
        bench = get_benchmark(entry.name)
        corpus, gold_clusters, _gold_pairs = bench.load()
        report = DataProfileReport.from_benchmark(bench)
    except Exception as exc:  # noqa: BLE001 - one broken loader must not hide the rest
        profile.load_error = f"{type(exc).__name__}: {exc}"
        return profile
    profile.loaded = True
    profile.cross_source_ceiling = cross_source_ceiling(corpus, gold_clusters)

    label = _section(report, "label_structure")
    if label is not None:
        profile.n_records = label.n_records
        profile.n_clusters = label.n_clusters
        profile.n_singletons = label.n_singletons
        profile.max_cluster_size = label.max_cluster_size
        profile.positive_pairs = label.positive_pairs
        profile.prevalence = label.prevalence
        profile.imbalance_ratio = label.imbalance_ratio
        profile.entropy_bits = label.entropy_bits
        profile.size_distribution = label.size_distribution

    separability = _section(report, "separability")
    if separability is not None:
        profile.separability_auc = separability.auc

    fields = _section(report, "corpus_field")
    if fields is not None and fields.fields:
        profile.n_fields = fields.n_fields_total
        profile.min_non_null_rate = min(f.non_null_rate for f in fields.fields)
        lengths = [f.mean_len for f in fields.fields if f.mean_len is not None]
        profile.mean_field_len = sum(lengths) / len(lengths) if lengths else None

    vocabulary = _section(report, "vocabulary_overlap")
    if vocabulary is not None:
        profile.vocab_jaccard = vocabulary.jaccard
        coverages = [
            c
            for c in (vocabulary.left_token_coverage, vocabulary.right_token_coverage)
            if c is not None
        ]
        profile.vocab_min_coverage = min(coverages) if coverages else None

    # ``schema`` is on every registered loader but not on the ``Benchmark``
    # protocol, so it is probed rather than assumed.
    schema = getattr(bench, "schema", None)
    evaluation: HonestPairEval | None = None
    if schema is not None:
        try:
            evaluation = measure_saturation(entry, schema)
        except Exception as exc:  # noqa: BLE001 - a missing split must not kill the row
            # Recorded, not just logged. Without this a CRASHED measurement and a
            # dataset that legitimately ships no literature split are the same
            # blank cell in the published table -- the artifact could not tell
            # "not applicable" from "broken", which is the reading the annotation
            # asks a reader to make.
            profile.saturation_error = f"{type(exc).__name__}: {exc}"
            logger.warning("%s: saturation measurement failed: %s", entry.name, exc)
    if evaluation is not None:
        profile.rapidfuzz_f1 = evaluation.honest.f1
        profile.rapidfuzz_threshold = evaluation.derived_threshold
        profile.rapidfuzz_argmax_f1 = evaluation.argmax_on_test.f1

    published = PUBLISHED_SOTA.get(entry.name)
    if published is not None:
        profile.published_f1 = published.f1
        profile.published_system = published.system
        if profile.rapidfuzz_f1 is not None:
            profile.saturated = profile.rapidfuzz_f1 >= published.f1 - SATURATION_MARGIN

    profile.caveats = structural_caveats(profile)
    return profile


def to_markdown(profiles: list[BenchmarkProfile]) -> str:
    """The annotation table: one row per registered benchmark."""
    header = (
        "| benchmark | task | wheel | records | gold pairs | prevalence | max cluster | "
        "sep AUC | vocab J | min cover | x-src ceiling | rapidfuzz F1 | published F1 | "
        "saturated | caveats |"
    )
    lines = [header, "|" + "---|" * 15]
    for p in profiles:
        lines.append(
            "| "
            + " | ".join(
                [
                    p.name,
                    p.task,
                    "yes" if p.wheel_loadable else "no",
                    _num(p.n_records, ",d"),
                    _num(p.positive_pairs, ",d"),
                    _num(p.prevalence, ".2e"),
                    _num(p.max_cluster_size, "d"),
                    _num(p.separability_auc, ".3f"),
                    _num(p.vocab_jaccard, ".3f"),
                    _num(p.vocab_min_coverage, ".3f"),
                    _num(p.cross_source_ceiling, ".4f"),
                    _num(p.rapidfuzz_f1, ".4f"),
                    _num(p.published_f1, ".4f"),
                    _verdict(p.saturated),
                    ", ".join(p.caveats) or "-",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _num(value: float | int | None, spec: str) -> str:
    """Format a number, or ``n/a`` when it is absent."""
    return "n/a" if value is None else format(value, spec)


def _verdict(saturated: bool | None) -> str:
    """Render the saturation verdict; ``?`` means "cannot be decided", never "no"."""
    if saturated is None:
        return "?"
    return "YES" if saturated else "no"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", help="explicit benchmark names")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/research/results/portfolio_profile.json"),
        help="where to write the machine-readable profiles (a TRACKED path)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    patterns = wheel_exclusions()
    wanted = set(args.only) if args.only else None
    profiles: list[BenchmarkProfile] = []
    for entry in list_benchmarks():
        if wanted is not None and entry.name not in wanted:
            continue
        print(f"[profile] {entry.name} ...")
        profile = profile_entry(entry, patterns)
        if not profile.loaded:
            print(f"          NOT PROFILED: {profile.load_error}")
        profiles.append(profile)

    print("\n" + to_markdown(profiles))
    tasks = sorted({p.task for p in profiles})
    print(f"\nregistered tasks: {tasks} ({len(profiles)} entries)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps([p.model_dump() for p in profiles], indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
