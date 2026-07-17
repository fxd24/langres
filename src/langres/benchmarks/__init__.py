"""``langres.benchmarks``: the entity-resolution benchmark **harness**.

The generic, dataset-agnostic machinery for benchmarking resolution *methods*
across *datasets*: race named methods on a benchmark into a table
(:func:`~langres.benchmarks.runner.run_method` / ``run_methods``), or score one
judge over a fixed candidate set against gold pairs
(:func:`~langres.benchmarks.judge_eval.evaluate`), spend-capped and honest by
construction. It sits *beside* ``langres.core`` (which is ER *modelling*, not
benchmarking) and depends on it — and on ``langres.data`` (the benchmark **spec**,
:mod:`langres.data.benchmark`), ``langres.metrics`` and ``langres.methods`` —
one-way; nothing in ``core``/``data`` imports back into here at module top.

This package is internal plumbing: users reach the harness through the curated
facade, ``langres.eval`` (``langres.evaluate`` / ``langres.eval.evaluate``) after
loading a dataset via ``langres.data``. Like ``langres.autoresearch``'s
``__init__``, this one exports **nothing** — which is also what keeps it
import-light. Import the pieces by dotted path::

    from langres.benchmarks.runner import run_method, run_methods, BenchmarkTable
    from langres.benchmarks.judge_eval import evaluate, evaluate_judge_on_candidates
"""
