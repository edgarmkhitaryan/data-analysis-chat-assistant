"""Synthesize node: merge sub-question reports into one briefing (plan/005 §3).

For a single question this is a passthrough — the one sub-result's report becomes
the answer. For a compound question it merges the sub-reports into one cohesive
executive briefing, preserving each part's figures verbatim and noting any
sub-question that failed (partial-failure tolerant).
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from assistant.agent.dependencies import AgentDeps
from assistant.agent.nodes.common import as_text, compose_system_prompt
from assistant.agent.state import AgentState
from assistant.llm import get_chat_model

logger = logging.getLogger(__name__)

_MERGE_BASE = (
    "You are a data analyst assistant for a retail company's non-technical executives. "
    "You are merging several separate analyses into ONE cohesive executive briefing. "
    "Preserve every figure exactly as provided; never invent data. Address each part "
    "clearly under its own heading, and add a short overall takeaway."
)


def _succeeded(result: dict) -> bool:
    return bool(result.get("report")) and not result.get("error")


def synthesize(state: AgentState, deps: AgentDeps) -> dict:
    """Pass a single report through, or merge multiple sub-reports into one briefing."""
    sub_results = state.get("sub_results", [])

    # Single question: surface the one report unchanged.
    if not state.get("is_compound"):
        only = sub_results[0] if sub_results else {}
        report = only.get("report") or "I wasn't able to complete that analysis."
        return {
            "report": report,
            "generated_sql": only.get("sql"),
            "row_count": only.get("row_count", 0),
            "last_error": only.get("error"),
            "messages": [AIMessage(content=report)],
        }

    # Compound question: merge the parts that succeeded.
    successful = [r for r in sub_results if _succeeded(r)]
    failed = [r for r in sub_results if not _succeeded(r)]

    if not successful:
        message = (
            "I wasn't able to answer any part of that question. "
            "Please try rephrasing or narrowing it."
        )
        return {"report": message, "messages": [AIMessage(content=message)]}

    parts = "\n\n".join(
        f"### Part {i}: {r['sub_question']}\n{r['report']}"
        for i, r in enumerate(successful, start=1)
    )
    note = ""
    if failed:
        note = (
            "\n\nThese parts could not be answered: "
            + "; ".join(r["sub_question"] for r in failed)
            + "."
        )
        logger.info("Synthesize: %d/%d sub-questions succeeded", len(successful), len(sub_results))

    system = compose_system_prompt(state, _MERGE_BASE)
    human = (
        f"The user's overall question was: {state['question']}\n\n"
        f"Findings for each part:\n\n{parts}{note}"
    )
    chat = get_chat_model(temperature=0.3, settings=deps.settings)
    report = as_text(
        chat.invoke([SystemMessage(content=system), HumanMessage(content=human)]).content
    ).strip()
    return {"report": report, "messages": [AIMessage(content=report)], "generated_sql": None}
