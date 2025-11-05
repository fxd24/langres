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

    # Verify key output appears in stdout
    assert "STEP 1: EXPLORE BLOCKING" in result.stdout
    assert "STEP 2: TUNE BLOCKING" in result.stdout
    assert "STEP 3: EXPLORE SCORING" in result.stdout
    assert "STEP 4: EXPLORE CLUSTERING" in result.stdout
    assert "STEP 5: TUNE CLUSTERING" in result.stdout
    assert "SUMMARY: Calibrated Parameters" in result.stdout

    # Verify that inspection methods were called
    assert "Candidate Inspection Report" in result.stdout
    assert "Score Distribution Report" in result.stdout
    assert "Cluster Inspection Report" in result.stdout

    # Verify that parameter tuning is demonstrated
    assert "COMPARISON:" in result.stdout
    assert "k=" in result.stdout  # Blocking parameter comparison
    assert "threshold=" in result.stdout  # Clustering parameter comparison

    # Verify recommendations are present
    assert "Next steps:" in result.stdout
    assert "WITHOUT ground truth labels" in result.stdout
