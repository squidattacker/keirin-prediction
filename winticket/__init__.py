"""winticket package bootstrap."""

from __future__ import annotations

# Re-export config so callers can rely on ``winticket.config`` regardless of
# package loading order.
from . import config  # noqa: F401

