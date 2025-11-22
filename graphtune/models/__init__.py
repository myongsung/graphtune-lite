"""Model factory API (v1-compatible).

Right now we re-export legacy build_model so your CLI stays identical.
Later you can swap to registry-based build_model here.
"""
from ..legacy.__init__ import build_model  # noqa: F401
