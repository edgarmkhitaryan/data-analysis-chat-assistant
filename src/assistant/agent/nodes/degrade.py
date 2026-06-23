"""Graceful-degradation node: an honest, non-crashing answer when analysis fails.

This is the Phase 2 fallback. Phase 8 expands the resilience story (bounded
self-correction before reaching here, empty-result reformulation, retries); this
node remains the final safe exit when those are exhausted.
"""

from langchain_core.messages import AIMessage

from assistant.agent.state import AgentState


def degrade(state: AgentState) -> dict:
    """Return a clear message explaining that the analysis could not be completed."""
    reason = state.get("last_error") or "an unexpected error"
    message = (
        f"I wasn't able to complete that analysis ({reason}). "
        "Please try rephrasing or narrowing your question."
    )
    return {"report": message, "messages": [AIMessage(content=message)]}
