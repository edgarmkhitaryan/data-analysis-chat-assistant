"""SQL validation node (plan/007 §2).

Thin wrapper over the pure :func:`assistant.safety.sql_validator.validate_select`
(sqlglot AST: single statement, read-only, table allow-list, LIMIT injection/clamp).
On success it stores the *normalized* SQL (with the enforced LIMIT); on failure it
sets ``last_error`` so the graph degrades (and, from Phase 8, self-corrects). The
dry-run cost guard runs later, at the execution boundary.
"""

from assistant.agent.dependencies import AgentDeps
from assistant.agent.state import AgentState
from assistant.safety.sql_validator import validate_select


def validate_sql(state: AgentState, deps: AgentDeps) -> dict:
    """Validate + normalize the generated SQL; set ``last_error`` if unsafe."""
    result = validate_select(
        state.get("generated_sql") or "",
        dataset=deps.settings.bq_dataset,
        max_limit=deps.settings.sql_max_limit,
    )
    if not result.ok:
        return {"last_error": result.error}
    return {"generated_sql": result.sql, "last_error": None}
