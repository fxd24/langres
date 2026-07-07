"""T8 (a) -- Fodors-Zagat wiring smoke: the closed loop with a REAL teacher, <=$2.

Runs :func:`examples.flywheel_closed_loop.run_closed_loop` once on the committed
Fodors-Zagat fixture (``examples/data/flywheel_loop/``) with a REAL frontier
teacher (``gpt-4o-mini`` via OpenRouter) under a hard
:class:`~langres.clients.openrouter.SpendMonitor` cap (default $2, ceiling $2;
est. ~$0.1-0.6 on the 120-pair fixture). This proves the example works with a
real model end to end; it is a **wiring smoke, NOT an economics claim** -- the FZ
fixture is easy and the student already resolves it, so the cascade's value here
is the frontier-call reduction, not an F1 gain (the real teacher/student gap
lives in the Amazon-Google run, ``flywheel_amazon_google.py``).

Spend safety (read twice): the whole run is metered by ONE ``SpendMonitor``
wrapping the OUTSIDE of the teacher (bootstrap batch + cascade escalations share
it), so cumulative spend can never cross ``--budget``. ``--model`` must be priced
in :data:`~langres.clients.openrouter.PRICES_PER_1M` (else the cap is blind) and
must resolve once before ANY spend. A breach raises
:class:`~langres.clients.openrouter.BudgetExceeded` (caught in ``main`` -> exit 2).

The whole flow is verified at **$0** with the deterministic ``SimulatedFrontierJudge``
in ``tests/examples/test_flywheel_paid.py`` (``run_fz_smoke(simulated=True)``) --
that test never makes a real call. The orchestrator runs the single paid execution
(``-m`` module form -- the script cross-imports the ``examples`` package, so the
repo root must be on ``sys.path``)::

    uv run python -m examples.research.flywheel_fz_smoke --budget 2.0

``OPENROUTER_API_KEY`` is loaded from ``.env``. ``print`` is allowed in examples.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from typing import TYPE_CHECKING

from examples.flywheel_closed_loop import (
    DATA_DIR,
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
from langres.clients.openrouter import BudgetExceeded

if TYPE_CHECKING:
    from examples.flywheel_closed_loop import ClosedLoopReport

logger = logging.getLogger("flywheel_fz_smoke")

#: Hard ceiling for this wiring smoke (task: <=$2).
BUDGET_CEILING_USD = 2.0
DEFAULT_BUDGET_USD = 2.0
#: Fodors-Zagat records are restaurants -> the teacher's entity noun.
ENTITY_NOUN = "restaurant"
_SEED = 7


def run_fz_smoke(
    *, model: str = DEFAULT_MODEL, budget_usd: float = DEFAULT_BUDGET_USD, simulated: bool = False
) -> ClosedLoopReport:
    """Run the FZ closed loop under a spend cap; simulated=True keeps it at $0.

    Args:
        model: Frontier teacher model id (ignored when ``simulated``).
        budget_usd: Hard spend cap wrapping the whole teacher.
        simulated: When ``True``, use the deterministic $0
            :class:`SimulatedFrontierJudge` instead of a real DSPy judge -- the
            zero-network path the verification test drives.

    Returns:
        The finished :class:`ClosedLoopReport`.

    Raises:
        BudgetExceeded: If cumulative (real) teacher spend crosses ``budget_usd``.
    """
    teacher = (
        SimulatedFrontierJudge(seed=_SEED)
        if simulated
        else build_real_teacher(model, entity_noun=ENTITY_NOUN)
    )
    return run_closed_loop(DATA_DIR, seed=_SEED, teacher=teacher, spend_cap_usd=budget_usd)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="T8(a) FZ wiring smoke: the closed loop with a real teacher, <=$2 capped."
    )
    parser.add_argument(
        "--budget", type=float, default=DEFAULT_BUDGET_USD, help="Hard spend cap (USD, <=2)."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="A PRICES_PER_1M-pinned OpenRouter model."
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

    try:
        # The BEFORE-arm silver-only circularity warning + the uncompiled-DSPy
        # warning are expected narration here, not errors.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = run_fz_smoke(model=args.model, budget_usd=args.budget, simulated=args.simulated)
    except BudgetExceeded as exc:
        print(
            f"[stopped] spend cap fired: {exc} "
            f"(partial judgements recovered: {len(exc.partial_judgements)})"
        )
        return 2

    print(format_report(report))
    results = report_to_results(
        report,
        dataset="fodors-zagat",
        model=model,
        budget_usd=args.budget,
        simulated=args.simulated,
        notes=(
            "WIRING SMOKE, not an economics claim: Fodors-Zagat is easy, so the cheap "
            "student already resolves the held-out split and the cascade's value here is "
            "the frontier-call reduction, not an F1 gain. Real teacher/student economics "
            "live in flywheel_amazon_google.py."
        ),
    )
    json_path, md_path = write_result_docs(
        results,
        out_dir=RESULTS_DIR,
        stem="flywheel_fz_smoke_results" + ("_simulated" if args.simulated else ""),
        title="Flywheel closed loop -- Fodors-Zagat wiring smoke (T8a)",
    )
    print(f"[report] wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
