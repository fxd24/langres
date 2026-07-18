from __future__ import annotations

import subprocess
import sys


def test_experiments_contract_imports_without_heavy_backends() -> None:
    script = (
        "import sys; import langres.experiments; "
        "heavy = {'torch', 'transformers', 'litellm', 'faiss', "
        "'sentence_transformers', 'sklearn', 'trackio', 'huggingface_hub'}; "
        "leaked = sorted(heavy.intersection(sys.modules)); "
        "assert not leaked, leaked"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
