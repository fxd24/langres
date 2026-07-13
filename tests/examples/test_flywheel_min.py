"""Smoke test for ``examples/flywheel_min.py`` -- the paved-road flywheel loop.

Behavior-focused (harness tier): the script must run end to end at $0 from a
plain install (string judge only, CLI CSV round-trip included) and the tuned
threshold must VISIBLY change the outcome -- a strictly higher pair F1 against
the gold sample, and different clusters. Runs the example exactly as a user
would (a subprocess), so the ``langres`` console-script calls inside it are
exercised for real.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "flywheel_min.py"

_F1 = re.compile(r"\[6\] (BEFORE|AFTER) .*F1=([0-9.]+)")


def test_flywheel_min_runs_and_tuned_threshold_improves_f1(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(_EXAMPLE), str(tmp_path / "work")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"example failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    f1 = {m.group(1): float(m.group(2)) for m in _F1.finditer(result.stdout)}
    assert set(f1) == {"BEFORE", "AFTER"}, f"missing summary lines in:\n{result.stdout}"
    assert f1["AFTER"] > f1["BEFORE"], (
        f"the tuned threshold must visibly improve pair F1 against gold "
        f"(before={f1['BEFORE']}, after={f1['AFTER']})"
    )

    # The clusters themselves changed: the default cut's over-merges split.
    clusters = {
        line.split(": ", 1)[1]
        for line in result.stdout.splitlines()
        if line.startswith(("[1] clusters", "[5] clusters"))
    }
    assert len(clusters) == 2, f"expected two distinct cluster printouts in:\n{result.stdout}"

    # The loop's artifacts landed where the script says they do.
    work = tmp_path / "work"
    for name in ("judgements.jsonl", "review_queue.jsonl", "corrections.jsonl"):
        assert (work / name).is_file(), f"missing flywheel artifact {name}"
    tearsheet = (work / "tearsheet.html").read_text(encoding="utf-8")
    assert "<svg" in tearsheet  # a real rendered report, not an empty shell
