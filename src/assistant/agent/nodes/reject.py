"""Refusal node: a graceful, deterministic decline for rejected turns (plan/007 §1).

Reached when the guard classifies a turn as off-topic or an attempted
manipulation. The refusal is **deterministic — we never feed flagged/adversarial
input back to an LLM** (that is the whole point of catching it before the model);
instead we return a clear, on-brand message explaining what the assistant *can*
do. The safety event itself is logged upstream in the guard.
"""

from langchain_core.messages import AIMessage

from assistant.agent.state import AgentState

_REFUSAL = (
    "I can't help with that one. I'm a retail data-analysis assistant — I can answer "
    "questions about sales, products, customers, orders, and trends, and manage your "
    'saved reports. For example: *"What were the top 10 products by revenue last month?"*'
)


def respond_reject(state: AgentState) -> dict:
    """Return the graceful refusal as this turn's response."""
    return {"report": _REFUSAL, "messages": [AIMessage(content=_REFUSAL)]}
