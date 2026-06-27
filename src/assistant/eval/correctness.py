"""Objective result-correctness check against a per-case reference query (plan/011 §2).

Where a golden case provides a ``reference_sql``, the harness runs it to get the
*authoritative* aggregates, then verifies the agent reproduced every one of them
(within a relative tolerance) in the rows its report stands on. This is the objective
cross-check that keeps the LLM-judge honest: if the judge says "great answer" but the
numbers don't match the reference, that disagreement is the most valuable failure to
surface (plan/011 §2).

Convention for authoring a ``reference_sql``: select the **measure(s)** as numeric
columns and any **dimension** (category, month, product name) as a *string* column.
Only numeric cells are compared, so string dimensions are ignored automatically and we
compare like-for-like figures regardless of how the agent shaped/aliased its own query.

This module is pure (no network) so it is unit-tested without quota; the harness injects
a ``reference_fn`` that actually runs the SQL.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReferenceCheck:
    """The outcome of comparing an agent result to its reference aggregates."""

    ran: bool  # did the reference query execute (False = it errored)?
    ok: bool | None  # did every reference aggregate appear in the agent's rows?
    matched: int
    total: int
    detail: str = ""


def _numbers(rows: list[dict]) -> list[float]:
    """Every numeric cell across the rows (booleans excluded — they aren't aggregates)."""
    out: list[float] = []
    for row in rows:
        for value in row.values():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                out.append(float(value))
    return out


def _matches(expected: float, candidates: list[float], tolerance: float) -> bool:
    """Is ``expected`` reproduced by some candidate within a relative tolerance?"""
    denom = max(abs(expected), 1e-9)
    return any(abs(expected - got) / denom <= tolerance for got in candidates)


def compare_aggregates(
    reference_rows: list[dict],
    agent_rows: list[dict],
    *,
    tolerance: float = 0.01,
    min_magnitude: float = 0.5,
) -> ReferenceCheck:
    """Check that every reference aggregate is reproduced in the agent's rows.

    ``min_magnitude`` drops near-zero incidental values so a stray ``0`` doesn't count
    as a key aggregate. Returns ``ran=True`` (the comparison was performed) with ``ok``
    True only when *all* reference aggregates were matched.
    """
    expected = [v for v in _numbers(reference_rows) if abs(v) >= min_magnitude]
    candidates = _numbers(agent_rows)
    if not expected:
        return ReferenceCheck(
            ran=True, ok=False, matched=0, total=0, detail="reference returned no numeric aggregates"
        )
    matched = sum(1 for v in expected if _matches(v, candidates, tolerance))
    total = len(expected)
    return ReferenceCheck(
        ran=True,
        ok=matched == total,
        matched=matched,
        total=total,
        detail=f"{matched}/{total} reference aggregates reproduced (±{tolerance:.0%})",
    )
