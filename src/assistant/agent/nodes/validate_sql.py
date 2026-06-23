"""Basic SQL validation node (Phase 2).

A lightweight read-only gate: it confirms the statement is a single SELECT/CTE
and rejects obvious DML/DDL. Phase 6 replaces this with a robust ``sqlglot``
AST-based validator (table allow-list, LIMIT injection, no regex fragility); the
graph wiring stays the same, only this node's body gets stronger.
"""

import re

from assistant.agent.state import AgentState

_FORBIDDEN_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "merge",
    "truncate",
    "grant",
    "replace",
)


def validate_sql(state: AgentState) -> dict:
    """Validate the generated SQL; set ``last_error`` if it is not safe to run."""
    sql = (state.get("generated_sql") or "").strip().rstrip(";").strip()
    if not sql:
        return {"last_error": "No SQL was generated."}

    lowered = sql.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return {"last_error": "Only SELECT queries are allowed."}
    if ";" in sql:
        return {"last_error": "Only a single statement is allowed."}
    for keyword in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", lowered):
            return {"last_error": f"Disallowed keyword in query: {keyword.upper()}."}

    return {"generated_sql": sql, "last_error": None}
