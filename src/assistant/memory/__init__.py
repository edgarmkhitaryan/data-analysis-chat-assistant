"""Memory: per-manager preference profiles + the automatic learning loop."""

from assistant.memory.feedback import (
    Candidate,
    GateResult,
    decide,
    promote_candidate,
    promote_if_qualified,
)
from assistant.memory.profiles import ProfileStore, UserPrefs

__all__ = [
    "Candidate",
    "GateResult",
    "decide",
    "promote_candidate",
    "promote_if_qualified",
    "ProfileStore",
    "UserPrefs",
]
