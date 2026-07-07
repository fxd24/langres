"""T8 (b) -- Amazon-Google economics: the closed loop on hard data, <=$10 capped.

Runs the SAME closed loop (:func:`examples.flywheel_closed_loop.run_closed_loop`)
as the FZ wiring smoke, but on **Amazon-Google** -- a hard, unsaturated benchmark
with a real teacher/student gap -- so the numbers it produces are the economics
GETTING_STARTED cites (escalation rate, frontier-call reduction, pairwise F1 for
teacher/student/cascade, real $ spent). A real frontier teacher (``gpt-4o-mini``
via OpenRouter) bootstraps silver labels; a cheap RFJudge student is trained; a
cascade runs the student everywhere and escalates only the uncertain band.

Reuse, not duplication: ``run_closed_loop`` is Fodors-Zagat-shaped (its fixtures
are :class:`~examples.flywheel_closed_loop.FZRecord`), so this script
**materializes a bounded Amazon-Google subset into that same fixture format** and
points ``run_closed_loop`` at it -- the loop is untouched. AG products map onto
``FZRecord`` fields as ``name=title`` (the dominant signal) and ``addr=manufacturer``
(``city``/``phone`` left empty); the field *values* are what the teacher reads and
the comparator matches, so the cosmetic field labels don't affect the result.

The AG data is read straight from the vendored CSVs
(``langres.data.datasets.amazon_google``) via stdlib ``csv`` -- deliberately NOT
through ``langres.data.amazon_google``, which imports the ``[semantic]`` embedding
stack at module load. So this script (and its ``simulated=True`` $0 verification)
run in a lean env with no torch/faiss/dspy.

Spend safety (read twice): the whole run is metered by ONE ``SpendMonitor``
wrapping the OUTSIDE of the teacher (bootstrap + cascade escalations share it), so
cumulative spend can never cross ``--budget`` (default $8, ceiling $10; sized so a
full run finishes well inside budget -- the cap is a backstop, not the operating
point). ``--model`` must be priced in
:data:`~langres.clients.openrouter.PRICES_PER_1M` and resolve once before ANY
spend. A breach raises :class:`~langres.clients.openrouter.BudgetExceeded`
(caught in ``main`` -> exit 2; the report is not produced -- raise ``--budget`` or
lower ``--max-pairs`` and re-run).

Verified at **$0** with ``SimulatedFrontierJudge`` in
``tests/examples/test_flywheel_paid.py`` (``run_ag_economics(simulated=True)``).
The orchestrator runs the single paid execution (``-m`` module form -- the script
cross-imports the ``examples`` package)::

    uv run python -m examples.research.flywheel_amazon_google --budget 8.0

``OPENROUTER_API_KEY`` is loaded from ``.env``. ``print`` is allowed in examples.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import warnings
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from examples.flywheel_closed_loop import (
    SimulatedFrontierJudge,
    format_report,
    run_closed_loop,
)
from examples.research._flywheel_paid_common import (
    DEFAULT_MODEL,
    RESULTS_DIR,
    build_real_teacher,
    preflight_real_model,
    report_to_results,
    write_result_docs,
)
from langres.clients.openrouter import BudgetExceeded, dspy_price_per_1k

if TYPE_CHECKING:
    from examples.flywheel_closed_loop import ClosedLoopReport

logger = logging.getLogger("flywheel_amazon_google")

#: Hard ceiling for this economics run (task: <=$10). Default budget is below it.
BUDGET_CEILING_USD = 10.0
DEFAULT_BUDGET_USD = 8.0
#: Amazon-Google records are products -> the teacher's entity noun.
ENTITY_NOUN = "product"
_SEED = 7

#: Vendored Amazon-Google CSVs (read directly -- see module docstring for why).
_DATASET_PACKAGE = "langres.data.datasets.amazon_google"
_TABLE_A_FILE = "tableA.csv"  # Amazon
_TABLE_B_FILE = "tableB.csv"  # Google
_TEST_PAIRS_FILE = "test.csv"  # fixed literature split; label 1 = match

#: Default number of candidate pairs to score (bounds real spend; the $10 cap is a
#: backstop). ~2000 pairs on gpt-4o-mini is roughly $0.6-1.0 of teacher bootstrap.
DEFAULT_MAX_PAIRS = 2000
#: Floor on positives so both the RFJudge train half and the held-out metric split
#: reliably span both label classes (all-True labels crash ``RFJudge.fit``).
_MIN_POSITIVES = 30
#: Rough per-pair cost estimate for the pre-flight heads-up (CoT output tokens
#: dominate; the live cap meters and enforces the REAL cost).
_EST_TOKENS_PER_PAIR = 700


def _read_table(filename: str) -> dict[str, dict[str, str]]:
    """Read a vendored AG product table (``id,title,manufacturer,price``) by id."""
    with resources.files(_DATASET_PACKAGE).joinpath(filename).open(encoding="utf-8") as fh:
        return {row["id"]: row for row in csv.DictReader(fh)}


def _read_test_pairs() -> list[tuple[str, str, int]]:
    """Read the fixed AG test split as ``(ltable_id, rtable_id, label)`` tuples."""
    with resources.files(_DATASET_PACKAGE).joinpath(_TEST_PAIRS_FILE).open(encoding="utf-8") as fh:
        return [(row["ltable_id"], row["rtable_id"], int(row["label"])) for row in csv.DictReader(fh)]


def materialize_ag_fixtures(out_dir: Path, *, max_pairs: int, seed: int) -> Path:
    """Write a bounded AG subset as FZRecord-shaped ``records.json`` + ``gold_pairs.json``.

    Reads the vendored AG tables + test split, takes a deterministic subset of
    ``max_pairs`` cross-source pairs (guaranteeing at least :data:`_MIN_POSITIVES`
    positives so both split halves span both label classes), maps each product
    onto ``FZRecord`` fields (``name=title``, ``addr=manufacturer``), and writes
    the two files :func:`run_closed_loop` reads.

    Args:
        out_dir: Directory to write ``records.json`` + ``gold_pairs.json`` into.
        max_pairs: Number of candidate pairs to keep (bounds teacher spend).
        seed: Deterministic subset-selection seed.

    Returns:
        ``out_dir`` (the ``data_dir`` to hand to :func:`run_closed_loop`).

    Raises:
        ValueError: If the vendored split has fewer than :data:`_MIN_POSITIVES`
            positives (it does not -- guards against a truncated dataset).
    """
    amazon = _read_table(_TABLE_A_FILE)
    google = _read_table(_TABLE_B_FILE)
    all_pairs = _read_test_pairs()

    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    positives = [p for p in all_pairs if p[2] == 1]
    negatives = [p for p in all_pairs if p[2] == 0]
    if len(positives) < _MIN_POSITIVES:
        raise ValueError(
            f"only {len(positives)} positive AG test pairs available (<{_MIN_POSITIVES}); "
            "the vendored dataset looks truncated."
        )
    # Keep roughly the natural positive rate, but never fewer than the floor.
    n_pos = min(len(positives), max(_MIN_POSITIVES, round(max_pairs * len(positives) / len(all_pairs))))
    n_neg = min(len(negatives), max(0, max_pairs - n_pos))
    subset = positives[:n_pos] + negatives[:n_neg]
    rng.shuffle(subset)

    candidate_pairs: list[dict[str, object]] = []
    referenced: set[tuple[str, str]] = set()  # (source, id)
    for ltable_id, rtable_id, label in subset:
        left_id, right_id = f"amazon-{ltable_id}", f"google-{rtable_id}"
        candidate_pairs.append({"left_id": left_id, "right_id": right_id, "label": bool(label)})
        referenced.add(("amazon", ltable_id))
        referenced.add(("google", rtable_id))

    records: list[dict[str, str]] = []
    for source, raw_id in sorted(referenced):
        table = amazon if source == "amazon" else google
        row = table[raw_id]
        records.append(
            {
                "id": f"{source}-{raw_id}",
                "name": row.get("title", ""),  # dominant signal
                "addr": row.get("manufacturer", ""),  # secondary attribute
                "city": "",
                "phone": "",
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (out_dir / "gold_pairs.json").write_text(
        json.dumps({"candidate_pairs": candidate_pairs}, indent=2), encoding="utf-8"
    )
    logger.info(
        "materialized %d AG pairs (%d positives / %d negatives) over %d records into %s",
        len(candidate_pairs),
        n_pos,
        n_neg,
        len(records),
        out_dir,
    )
    return out_dir


def run_ag_economics(
    *,
    model: str = DEFAULT_MODEL,
    budget_usd: float = DEFAULT_BUDGET_USD,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    simulated: bool = False,
    seed: int = _SEED,
    work_dir: Path | None = None,
) -> ClosedLoopReport:
    """Materialize a bounded AG subset, then run the closed loop under a spend cap.

    Args:
        model: Frontier teacher model id (ignored when ``simulated``).
        budget_usd: Hard spend cap wrapping the whole teacher.
        max_pairs: Number of AG candidate pairs to score (bounds spend).
        simulated: When ``True``, use the deterministic $0
            :class:`SimulatedFrontierJudge` -- the zero-network verification path.
        seed: Deterministic seed for subset selection + loop splits.
        work_dir: Where to materialize fixtures + write loop artifacts; a
            temporary directory is used when omitted.

    Returns:
        The finished :class:`ClosedLoopReport`.

    Raises:
        BudgetExceeded: If cumulative (real) teacher spend crosses ``budget_usd``.
    """
    if work_dir is None:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="flywheel_ag_") as tmp:
            return run_ag_economics(
                model=model,
                budget_usd=budget_usd,
                max_pairs=max_pairs,
                simulated=simulated,
                seed=seed,
                work_dir=Path(tmp),
            )

    data_dir = materialize_ag_fixtures(work_dir / "data", max_pairs=max_pairs, seed=seed)
    teacher = (
        SimulatedFrontierJudge(seed=seed)
        if simulated
        else build_real_teacher(model, entity_noun=ENTITY_NOUN)
    )
    return run_closed_loop(
        data_dir, seed=seed, work_dir=work_dir / "loop", teacher=teacher, spend_cap_usd=budget_usd
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="T8(b) Amazon-Google economics: the closed loop on hard data, <=$10 capped."
    )
    parser.add_argument(
        "--budget", type=float, default=DEFAULT_BUDGET_USD, help="Hard spend cap (USD, <=10)."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="A PRICES_PER_1M-pinned OpenRouter model."
    )
    parser.add_argument(
        "--max-pairs", type=int, default=DEFAULT_MAX_PAIRS, help="AG candidate pairs to score."
    )
    parser.add_argument(
        "--simulated",
        action="store_true",
        help="Use the deterministic $0 SimulatedFrontierJudge (no key, no network).",
    )
    args = parser.parse_args()

    model = "simulated-frontier" if args.simulated else args.model
    if not args.simulated:
        reason = preflight_real_model(
            args.model, budget_usd=args.budget, ceiling_usd=BUDGET_CEILING_USD
        )
        if reason is not None:
            print(f"[fatal] {reason}")
            return 1
        est = args.max_pairs * (_EST_TOKENS_PER_PAIR / 1000.0) * dspy_price_per_1k(args.model)
        print(
            f"[estimate] bootstrap scores ~{args.max_pairs} pairs on {args.model!r}; "
            f"rough est ${est:.4f} (+ cascade escalations), hard cap ${args.budget:.2f}."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = run_ag_economics(
                model=args.model,
                budget_usd=args.budget,
                max_pairs=args.max_pairs,
                simulated=args.simulated,
            )
    except BudgetExceeded as exc:
        print(
            f"[stopped] spend cap fired: {exc} "
            f"(partial judgements recovered: {len(exc.partial_judgements)}). "
            "Raise --budget or lower --max-pairs and re-run."
        )
        return 2

    print(format_report(report))
    results = report_to_results(
        report,
        dataset="amazon-google",
        model=model,
        budget_usd=args.budget,
        simulated=args.simulated,
        notes=(
            f"Economics on a bounded Amazon-Google subset ({report.n_candidates} candidate "
            f"pairs, seed {_SEED}). AG products mapped onto FZRecord as name=title, "
            "addr=manufacturer (cosmetic field labels; values drive the result). "
            "The positive rate is kept near the natural ~10% with a floor so both split "
            "halves span both label classes. These are THE economics numbers "
            "GETTING_STARTED cites; the FZ/simulated runs are wiring/plumbing only."
        ),
    )
    json_path, md_path = write_result_docs(
        results,
        out_dir=RESULTS_DIR,
        stem="flywheel_amazon_google_results" + ("_simulated" if args.simulated else ""),
        title="Flywheel closed loop -- Amazon-Google economics (T8b)",
    )
    print(f"[report] wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
