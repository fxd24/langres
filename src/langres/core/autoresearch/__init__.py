"""Autoresearch loop for entity resolution: propose → run → evaluate → keep-if-better.

The immutable Objective (the un-gameable scorer) and the pluggable proposer over a
declarative SearchSpace sit under here; the public entry point is ``langres.optimize``.
See epic #145. Submodules are imported by dotted path — this package intentionally
exports nothing until the public facade wires it up.
"""
