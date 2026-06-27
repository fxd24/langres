"""Cold-start gold-set bootstrapping (M1).

Public data contract for labeled record pairs used to seed entity resolution.
"""

from langres.bootstrap.models import GoldPair, GoldSet

__all__ = ["GoldPair", "GoldSet"]
