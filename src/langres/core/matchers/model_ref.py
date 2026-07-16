"""Back-compat shim: ``ModelRef`` moved to :mod:`langres.core.model_ref`.

It is a weightless *contract* (stdlib-only, round-trips as JSON config), not a
matcher -- so it now lives in ``core/`` beside the other contracts. This shim
keeps the old import path working; it is removed in the architecture refactor's
final wave.
"""

from langres.core.model_ref import ModelRef, normalize_model_ref, to_config

__all__ = ["ModelRef", "normalize_model_ref", "to_config"]
