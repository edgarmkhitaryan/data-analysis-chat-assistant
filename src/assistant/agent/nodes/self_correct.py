"""Self-correction node: prepare repair context before regenerating SQL (plan/008 §1).

Sits on the retry edge of the analysis loop (``validate_sql``/``execute_sql`` ->
``self_correct`` -> ``generate_sql``). It does not call the LLM; it just shapes the
feedback the next ``generate_sql`` will act on:

- after a **validation or execution error**, ``last_error`` already holds the exact
  message, so this is a pass-through (the next generation repairs that specific error);
- after an **empty result** (0 rows, no error), it injects a one-time guided hint to
  reconsider over-narrow filters and marks ``empty_retried`` so we re-prompt **once**,
  then accept and report honestly rather than loop on a genuinely empty result.

Every pass is a bounded, inspectable state transition (capped by ``MAX_SQL_ATTEMPTS``).
"""

import logging

from assistant.agent.state import AgentState

logger = logging.getLogger(__name__)

_EMPTY_HINT = (
    "The previous query executed successfully but returned 0 rows. Reconsider whether a "
    "date range, a status filter (e.g. excluding Cancelled/Returned), or another predicate "
    "is too narrow or mismatched, and broaden or correct it if appropriate."
)


def self_correct(state: AgentState) -> dict:
    """Shape the repair context for the next ``generate_sql`` attempt."""
    attempt = state.get("sql_attempts", 0)
    if state.get("last_error"):
        logger.info(
            "Self-correction (attempt %d) repairing error: %s", attempt, state["last_error"]
        )
        return {}
    logger.info("Self-correction (attempt %d): empty result -> one guided reformulation", attempt)
    return {"last_error": _EMPTY_HINT, "empty_retried": True}
