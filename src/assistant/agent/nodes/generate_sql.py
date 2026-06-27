"""SQL generation node: turns the question + schema context into a SELECT query.

If a previous attempt left an error (or a "0 rows" hint from ``self_correct``) on
the state, it is fed back to the model so the regenerated query can fix it. The
bounded loop that drives the retries lives in the graph (plan/008 §1); this node
just produces (or re-produces) SQL and increments ``sql_attempts``.
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage

from assistant.agent.dependencies import AgentDeps
from assistant.agent.nodes.common import as_text
from assistant.agent.state import AgentState
from assistant.llm import get_chat_model, resilient_invoke

_SYSTEM_PROMPT = (
    "You are an expert analytics engineer who writes BigQuery Standard SQL for a "
    "retail company. Given the database schema and a business question, return "
    "exactly ONE SQL SELECT query that answers it. Return only the SQL — no "
    "explanation and no markdown fences.\n"
    "For questions about the database structure itself (what tables or columns exist, "
    "their types), query the dataset's INFORMATION_SCHEMA — e.g. SELECT table_name, "
    "column_name, data_type FROM `<project.dataset>`.INFORMATION_SCHEMA.COLUMNS — using "
    "the same project/dataset as the tables in the schema above. Restrict it to ONLY the "
    "tables listed in that schema with WHERE table_name IN (...), and order by "
    "table_name, ordinal_position, so the result describes exactly those tables."
)


def _exemplars_block(state: AgentState) -> str:
    """Format retrieved Trios as question->SQL few-shot exemplars, if any."""
    trios = state.get("retrieved_trios") or []
    if not trios:
        return ""
    blocks = [
        "Here are examples of how analysts answered similar questions. Use them as "
        "guidance for table choice, joins, and business conventions — adapt the "
        "logic to the current question; do not copy them blindly.",
    ]
    for i, trio in enumerate(trios, start=1):
        blocks.append(f"\nExample {i}:\nQuestion: {trio.question}\nSQL:\n{trio.sql}")
    return "\n".join(blocks) + "\n\n"


def _extract_sql(content: object) -> str:
    """Pull a clean SQL statement out of the model's reply.

    Tolerates fenced code blocks (```sql ... ```) and trims a trailing semicolon.
    """
    text = as_text(content).strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    return text.strip().rstrip(";").strip()


def generate_sql(state: AgentState, deps: AgentDeps) -> dict:
    """Generate SQL for the current question (incorporating any prior error)."""
    chat = get_chat_model(temperature=0.0, settings=deps.settings)
    human = (
        f"{state['schema_context']}\n\n"
        f"{_exemplars_block(state)}"
        f"Question: {state['question']}\n\nSQL:"
    )
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human),
    ]
    if state.get("last_error"):
        messages.append(
            HumanMessage(
                content=(
                    f"The previous attempt did not succeed: {state['last_error']}\n"
                    "Revise the query to address this and return only the corrected SQL."
                )
            )
        )
    reply = resilient_invoke(chat, messages, settings=deps.settings)
    return {
        "generated_sql": _extract_sql(reply.content),
        "sql_attempts": state.get("sql_attempts", 0) + 1,
        "last_error": None,
    }
