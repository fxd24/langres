"""Test for progressive pipeline building example.

This test verifies that the progressive_pipeline_building.py example runs
successfully and demonstrates the exploratory workflow for parameter tuning.
"""

import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
def test_progressive_pipeline_building_runs() -> None:
    """Test that progressive pipeline building example runs without errors."""
    example_path = Path(__file__).parent.parent / "examples" / "progressive_pipeline_building.py"

    result = subprocess.run(
        ["uv", "run", "python", str(example_path)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=Path(__file__).parent.parent,
    )

    assert result.returncode == 0, f"Example failed: {result.stderr}"

    # Note: logging output goes to stderr, not stdout
    output = result.stderr

    # Verify key output appears
    assert "STEP 1: EXPLORE BLOCKING" in output
    assert "STEP 2: TUNE BLOCKING" in output
    assert "STEP 3: EXPLORE SCORING" in output
    assert "STEP 4: EXPLORE CLUSTERING" in output
    assert "STEP 5: TUNE CLUSTERING" in output
    assert "SUMMARY: Calibrated Parameters" in output

    # Verify that inspection methods were called
    assert "Candidate Inspection Report" in output
    assert "Score Inspection Report" in output
    assert "Cluster Inspection Report" in output

    # Verify that parameter tuning is demonstrated
    assert "COMPARISON:" in output
    assert "k=" in output  # Blocking parameter comparison
    assert "threshold=" in output  # Clustering parameter comparison

    # Verify recommendations are present
    assert "Next steps:" in output
    assert "WITHOUT ground truth labels" in output
