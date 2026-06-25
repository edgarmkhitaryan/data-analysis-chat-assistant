"""Deterministic safety controls enforced in code, at boundaries (plan/007).

Phase 5 ships PII masking (:mod:`assistant.safety.pii`); Phase 6 adds the input
guard and the read-only SQL validator alongside it.
"""

from assistant.safety.pii import mask_rows, scan_text

__all__ = ["mask_rows", "scan_text"]
