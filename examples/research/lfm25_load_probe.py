"""Record how the LFM2.5 checkpoints load, as a re-runnable artifact.

The blocking-recall sweep answers "how good is this model". This answers the
question that has to come first: **is the thing being measured the checkpoint at
all.** Three ways these checkpoints load wrong without raising, all measured:

1. ``model_type: "lfm2"`` is implemented natively by transformers as a CAUSAL
   decoder, while every one of these checkpoints points ``auto_map.AutoModel``
   at its own bidirectional class. Drop ``trust_remote_code`` and the native
   class wins in silence.
2. The base encoders store their tensors under the MaskedLM wrapper's ``lfm2.``
   prefix, so ``AutoModel`` matches none of them and randomises the backbone --
   reported by transformers as a log warning, not an error.
3. A checkpoint loaded under the wrong attention can ignore the prompt exactly,
   which is indistinguishable from langres's own shipped ``query_prompt`` bug.

Each is recorded as a number a later reader can re-derive rather than a sentence
they have to trust. Run:

    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
        uv run python examples/research/lfm25_load_probe.py

$0, keyless, CPU-only, and ~2 minutes once the checkpoints are cached.

**Every configuration runs in its own subprocess, and that is load-bearing, not
tidiness.** Measured while writing this file: probing ``trust_remote_code=True``
and then ``False`` *in one process* makes the second load report the native class
while producing the remote class's vectors exactly — importing the checkpoint's
modelling code leaves global state behind that survives into a later
non-trusting load. The first version of this probe did exactly that and recorded
the two configurations as bit-identical, which is the opposite of the finding.
A fresh process reports ``cos = 1.000000`` and a prompt shift of ``0``; a
contaminated one reports the healthy numbers under the broken class's name. So
the isolation is what makes the artifact true, and an in-process probe here is
not a weaker check — it is a wrong one.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone

# Set before torch/faiss import, not in `.env`: `.env` is gitignored and absent
# from a fresh worktree, and without OMP_NUM_THREADS the process DEADLOCKS in
# __kmp_join_barrier at 0% CPU with no error (measured: 3.5h of silence).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "docs" / "research" / "20260729_lfm25_load_probe.json"

EMBEDDING_MODEL = "LiquidAI/LFM2.5-Embedding-350M"
ENCODER_MODELS = ("LiquidAI/LFM2.5-Encoder-350M", "LiquidAI/LFM2.5-Encoder-230M")

#: Two genuinely different product records. A working encoder must not score
#: them as the same point.
PROBE_TEXTS = (
    "Sony STR-DH550 5.2 Channel 4K AV Receiver",
    "Cuisinart DCC-3200 14-cup programmable coffee maker",
)


#: Discard the tracked probe deliberately, the same escape hatch `run_lfm25.sh`
#: offers for the study artifacts.
FORCE_ENV = "LFM25_FORCE"


def _refuse_to_overwrite_uncommitted() -> None:
    """Stop before destroying uncommitted measurements in the probe artifact.

    ``run_lfm25.sh`` refuses to start when this file is dirty, but this module's
    own docstring — and the generated write-up's "Reproduce" block — advertise
    running it STANDALONE, and that path reached an unconditional
    ``write_text``. The guard therefore protected the artifact only from the
    caller that already had a guard. Same defect shape as five earlier findings
    on this branch: the fix landed on one of two sites that write the same file.
    (Cross-model review.)

    The check is ``write_provenance._uncommitted``, not a second implementation:
    it already handles the cases a naive ``git status`` gets wrong (a gitignored
    or out-of-repo path reads as clean), and two copies of a safety check are
    two things that drift.
    """
    if os.environ.get(FORCE_ENV) == "1":
        logger.warning("%s=1: overwriting %s without checking", FORCE_ENV, OUTPUT_PATH)
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from write_provenance import _uncommitted

    lost = _uncommitted(OUTPUT_PATH)
    if not lost:
        return
    raise SystemExit(
        f"REFUSING to overwrite {OUTPUT_PATH}: it holds uncommitted changes "
        f"({', '.join(lost)}).\n"
        "  A probe run costs ~2 minutes; the measurements already in that file may "
        "not be reproducible.\n"
        "  If those are measurements or a hand edit, keep them: commit, or copy the file\n"
        "  outside this worktree, then re-run. If they are just the previous probe's own\n"
        "  output, that is what the force flag is for — this guard cannot tell the two\n"
        "  apart, so it asks:\n"
        f"    {FORCE_ENV}=1 uv run python examples/research/lfm25_load_probe.py"
    )


def _sentence_transformer(name: str, *, trust: bool) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, trust_remote_code=trust, device="cpu")


def probe_remote_code(name: str, *, trust: bool) -> dict[str, Any]:
    """What sentence-transformers actually instantiates, and what it then encodes."""
    model = _sentence_transformer(name, trust=trust)
    auto_model = model[0].auto_model
    cls = type(auto_model)

    bare = np.asarray(model.encode(list(PROBE_TEXTS), normalize_embeddings=True), dtype=np.float64)
    prompted = np.asarray(
        model.encode(list(PROBE_TEXTS), prompt="query: ", normalize_embeddings=True),
        dtype=np.float64,
    )
    return {
        "trust_remote_code": trust,
        "instantiated_class": f"{cls.__module__}.{cls.__qualname__}",
        "from_checkpoint_code": "transformers_modules" in cls.__module__,
        "declared_auto_model": (getattr(auto_model.config, "auto_map", None) or {}).get(
            "AutoModel"
        ),
        # 1.0 means the two unrelated records collapsed onto the same vector.
        "cosine_between_unrelated_records": float(bare[0] @ bare[1]),
        # Exactly 0 means the prompt reached nothing.
        "max_abs_prompt_shift": float(np.abs(prompted - bare).max()),
    }


def probe_weight_loading(name: str) -> dict[str, Any]:
    """Whether each auto class actually uses the checkpoint's tensors."""
    import torch
    from transformers import AutoModel, AutoModelForMaskedLM

    result: dict[str, Any] = {}
    for auto_class in (AutoModel, AutoModelForMaskedLM):
        model, info = auto_class.from_pretrained(
            name, trust_remote_code=True, output_loading_info=True, dtype=torch.float32
        )
        result[auto_class.__name__] = {
            "class": type(model).__name__,
            "missing_keys": len(info.get("missing_keys", [])),
            "unexpected_keys": len(info.get("unexpected_keys", [])),
        }

    # A correctly loaded checkpoint is identical across two independent loads.
    # A randomised one is not, and this is its only externally visible symptom.
    first = AutoModel.from_pretrained(name, trust_remote_code=True, dtype=torch.float32)
    second = AutoModel.from_pretrained(name, trust_remote_code=True, dtype=torch.float32)
    state_a, state_b = first.state_dict(), second.state_dict()
    result["AutoModel"]["two_load_max_drift"] = max(
        float((state_a[key] - state_b[key]).abs().max()) for key in state_a
    )
    return result


def _run_isolated(kind: str, argument: str) -> dict[str, Any]:
    """Run one probe in a fresh interpreter and return its JSON result.

    The isolation is the measurement's precondition — see the module docstring.
    ``sys.executable`` is this venv's interpreter and the child does no ``uv``
    work of its own, so this does not become a second concurrent ``uv run``.
    """
    completed = subprocess.run(
        [sys.executable, __file__, "--single", kind, argument],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(json.loads(completed.stdout))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # BEFORE the probing, not beside the write: refusing after the work is done
    # spends the two minutes and the model loads to reach the same refusal, and
    # an operator who then reaches for the force flag is doing so having already
    # been made to wait -- which is how a guard trains people to bypass it.
    _refuse_to_overwrite_uncommitted()
    import transformers

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    report: dict[str, Any] = {
        # WHEN, not just under-what. The write-up used to infer this probe's
        # freshness from `transformers_version` matching the installed one, so an
        # offline or rate-limited refresh that left the OLD file in place raised
        # no warning whenever the version happened to agree -- while checkpoint
        # remote code, cache contents and every other dependency could have moved.
        # Freshness is now stated by the artifact and compared against the
        # measurement window. (Cross-model review.)
        "captured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "transformers_version": transformers.__version__,
        # The trap's precondition: transformers implements this model_type
        # itself, so a missing trust_remote_code is a silent substitution rather
        # than an ImportError.
        "lfm2_natively_implemented": "lfm2" in CONFIG_MAPPING_NAMES,
        "probe_texts": list(PROBE_TEXTS),
        "remote_code": {},
        "weight_loading": {},
    }

    for trust in ("true", "false"):
        logger.info("probing %s trust_remote_code=%s", EMBEDDING_MODEL, trust)
        report["remote_code"][f"trust_remote_code={trust == 'true'}"] = _run_isolated(
            "remote_code", trust
        )

    for name in ENCODER_MODELS:
        logger.info("probing weight loading for %s", name)
        report["weight_loading"][name] = _run_isolated("weight_loading", name)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    logger.info("wrote %s", OUTPUT_PATH)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        kind, argument = sys.argv[2], sys.argv[3]
        if kind == "remote_code":
            payload = probe_remote_code(EMBEDDING_MODEL, trust=argument == "true")
        else:
            payload = probe_weight_loading(argument)
        sys.stdout.write(json.dumps(payload))
    else:
        main()
