"""A registered Resolver subclass, standing in for a W4 named architecture.

Lives in ``tests.fixtures`` (not in a test module) so a **fresh subprocess** can
import it by name and thereby fire its ``@register_model`` — which is exactly
what the save/load identity round-trip has to prove. W4 lands the real
architectures (``FuzzyString``, ``VectorLLMCascade``); this is the minimum shape
needed to test that the seam carries a class identity across a process boundary.
"""

from typing import ClassVar

from langres.core.registry import register_model
from langres.core.resolver import Resolver


@register_model("fixture_fuzzy_string")
class FixtureFuzzyString(Resolver):
    """A registered architecture: string-only, and it accepts no ``method=``.

    Doubles as the B4/B5 crossover check — it carries both an
    ``accepted_method_kinds`` declaration and a registered model identity, the
    two things a W4 architecture is made of.
    """

    accepted_method_kinds: ClassVar[frozenset[str] | None] = frozenset({"calibrate"})


class UnregisteredArchitecture(Resolver):
    """A Resolver subclass nobody registered: must save/load exactly as today."""
