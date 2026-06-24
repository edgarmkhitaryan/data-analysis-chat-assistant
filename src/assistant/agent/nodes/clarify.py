"""Clarify node: ask one targeted question instead of guessing (plan/005 §3).

A terminal node reached when ``contextualize`` could not resolve a follow-up
confidently. It surfaces the single clarifying question (already produced by
contextualize) as the turn's reply; the user's next message answers it and,
now in history, lets contextualize resolve the original intent.
"""

from langchain_core.messages import AIMessage

from assistant.agent.state import AgentState


def clarify(state: AgentState) -> dict:
    """Emit the clarifying question as this turn's response."""
    question = (
        state.get("clarifying_question") or "Could you clarify exactly what you'd like to analyze?"
    )
    return {"report": question, "messages": [AIMessage(content=question)]}
