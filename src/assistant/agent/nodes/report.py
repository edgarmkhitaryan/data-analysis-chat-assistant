"""Report synthesis node: turns the query results into an analyst-grade answer.

In Phase 2 the report uses a sensible default analyst voice. Phase 4 layers the
org persona (tone) and the user's format preference (tables vs. bullets) on top,
and Phase 5 ensures only masked rows ever reach this node.
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from assistant.agent.dependencies import AgentDeps
from assistant.agent.nodes.common import as_text
from assistant.agent.state import AgentState
from assistant.llm import get_chat_model

_SYSTEM_PROMPT = (
    "You are a data analyst assistant for a retail company's non-technical "
    "executives. Write a clear, accurate answer to the question using ONLY the "
    "query results provided — never invent numbers. Lead with the key finding, "
    "then the supporting detail. Be concise and business-friendly. Format numbers "
    "for a business reader: monetary values with a currency symbol and two decimals, "
    "and large numbers with thousands separators."
)

# Cap how many rows we put in the prompt to keep token cost bounded; the model
# still receives the true total row count for context.
_MAX_ROWS_IN_PROMPT = 100


def synthesize_report(state: AgentState, deps: AgentDeps) -> dict:
    """Produce the written report and append it to the conversation."""
    chat = get_chat_model(temperature=0.3, settings=deps.settings)

    rows = state.get("raw_rows", [])
    shown = rows[:_MAX_ROWS_IN_PROMPT]
    row_count = state.get("row_count", len(rows))

    if rows:
        data_block = json.dumps(shown, indent=2, ensure_ascii=False, default=str)
        truncation = (
            f"\n(Showing the first {len(shown)} of {row_count} rows.)"
            if row_count > len(shown)
            else ""
        )
        results_section = f"Query results ({row_count} rows):\n{data_block}{truncation}"
    else:
        results_section = "Query results: the query returned no rows."

    human = f"Question: {state['question']}\n\n{results_section}"
    reply = chat.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human)])
    report = as_text(reply.content).strip()
    return {"report": report, "messages": [AIMessage(content=report)]}
