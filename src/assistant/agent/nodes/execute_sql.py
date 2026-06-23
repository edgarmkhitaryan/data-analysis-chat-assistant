"""Execution node: runs the validated SQL via the cost-guarded BigQuery runner.

On any failure it records a human-readable ``last_error`` and empties the rows,
so the graph can route to graceful degradation instead of crashing. Rows are
normalized to JSON-safe natives here, at the boundary into state.
"""

from assistant.agent.dependencies import AgentDeps
from assistant.agent.nodes.common import json_safe_row
from assistant.agent.state import AgentState
from assistant.bigquery import QueryCostError


def execute_sql(state: AgentState, deps: AgentDeps) -> dict:
    """Execute the query and store JSON-safe rows, or capture the error."""
    sql = state["generated_sql"]
    try:
        result = deps.runner.execute_query(sql)
    except QueryCostError as exc:
        return {"last_error": str(exc), "raw_rows": [], "row_count": 0}
    except Exception as exc:  # noqa: BLE001 — turn any execution failure into safe state
        return {"last_error": f"Query execution failed: {exc}", "raw_rows": [], "row_count": 0}

    rows = [json_safe_row(row) for row in result.rows]
    return {"raw_rows": rows, "row_count": result.row_count, "last_error": None}
