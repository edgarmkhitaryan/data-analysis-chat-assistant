"""PII masking node (plan/007 §3): the only edge from execute_sql to the report.

Masks the raw query results so the report LLM — and therefore the user — only ever
sees ``masked_rows``. Entirely deterministic: no LLM call, no PII in any prompt.
Sits inside the analysis subgraph, so it runs for every (sub-)question.
"""

from assistant.agent.dependencies import AgentDeps
from assistant.agent.state import AgentState
from assistant.safety.pii import mask_rows


def mask_pii(state: AgentState, deps: AgentDeps) -> dict:
    """Replace ``raw_rows`` with ``masked_rows`` and record how many cells were masked."""
    rows = state.get("raw_rows", [])
    masked_rows, masked_count = mask_rows(
        rows,
        deps.settings.pii_mask_columns,
        deps.settings.pii_mask_style,
    )
    return {"masked_rows": masked_rows, "pii_masked_count": masked_count}
