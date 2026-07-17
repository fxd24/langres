"""Back-compat shim: ``langres.core.methods_prompt`` moved to ``langres.training.methods_prompt``.

# TEMPORARY: deleted by the W2 sweep

The prompt-optimization ``Method`` objects (``Bootstrap`` / ``MIPRO`` / ``GEPA``)
configure how a matcher is *fitted*, not entity-resolution modelling itself, so
they now live in ``langres.training`` beside ``core``. Import from
``langres.training.methods_prompt`` (or the ``langres`` / ``langres.core``
facades, which still re-export them). ``dspy`` stays lazy in the real fit path,
never imported by this shim.
"""

from langres.training.methods_prompt import GEPA, MIPRO, Bootstrap, PromptMethod

__all__ = ["Bootstrap", "GEPA", "MIPRO", "PromptMethod"]
