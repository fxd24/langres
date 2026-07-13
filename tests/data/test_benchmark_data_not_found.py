"""The missing-benchmark-data path fails with a helpful, actionable error.

The PyPI wheel/sdist exclude the large third-party benchmark corpora (see
``[tool.hatch.build].exclude`` in ``pyproject.toml``), so on an installed
package every excluded-CSV read must raise
:class:`~langres.data.BenchmarkDataNotFoundError` telling the user the data is
git-checkout-only — not a bare ``FileNotFoundError`` deep inside
``importlib.resources``. In this repo the files exist, so the tests exercise
the same code path via names that are absent in *every* install.
"""

import pytest

from langres.data import BenchmarkDataNotFoundError
from langres.data._benchmark_utils import read_csv_rows


class TestBenchmarkDataNotFound:
    def test_missing_file_in_present_dataset_raises_helpful_error(self) -> None:
        """A missing file inside an existing dataset package (the wheel keeps
        e.g. peeters_sampled_test.csv but drops tableA.csv) raises the helpful
        error, chained from the original FileNotFoundError."""
        with pytest.raises(BenchmarkDataNotFoundError) as excinfo:
            read_csv_rows("langres.data.datasets.abt_buy", "no_such_file.csv")
        message = str(excinfo.value)
        assert "not bundled in the PyPI package" in message
        assert "git clone https://github.com/fxd24/langres" in message
        assert "tiny_fixture" in message  # names what DOES work without a checkout
        assert isinstance(excinfo.value.__cause__, FileNotFoundError)

    def test_missing_dataset_package_raises_helpful_error(self) -> None:
        """A dataset directory that is entirely absent (all files excluded from
        the wheel -> the namespace package is gone) raises the same error,
        chained from ModuleNotFoundError."""
        with pytest.raises(BenchmarkDataNotFoundError) as excinfo:
            read_csv_rows("langres.data.datasets.does_not_exist", "tableA.csv")
        assert "git clone" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)

    def test_is_a_file_not_found_error(self) -> None:
        """Callers already catching FileNotFoundError keep working."""
        with pytest.raises(FileNotFoundError):
            read_csv_rows("langres.data.datasets.abt_buy", "no_such_file.csv")

    def test_exported_from_langres_data(self) -> None:
        """The exception is importable from the public ``langres.data`` namespace."""
        import langres.data

        assert langres.data.BenchmarkDataNotFoundError is BenchmarkDataNotFoundError
        assert "BenchmarkDataNotFoundError" in langres.data.__all__
