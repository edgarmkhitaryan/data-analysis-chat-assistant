"""Contextualize node: rewrite a follow-up into a standalone question.

The first node in the graph (plan/005 §3, 010 §1). On the first turn (no history)
it is a no-op passthrough with **no LLM cost**. On later turns it uses the thread
history to resolve references ("break that down", "what about last year") into a
self-contained question, so every downstream node operates on an explicit
question rather than implicit cross-turn state. If the follow-up is too ambiguous
to resolve confidently, it flags clarification instead of guessing — the graph
then routes to ``clarify``.
"""

import logging
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from assistant.agent.dependencies import AgentDeps
from assistant.agent.nodes.common import as_text
from assistant.agent.state import AgentState
from assistant.llm import get_chat_model

logger = logging.getLogger(__name__)


class Contextualization(BaseModel):
    """Structured result of rewriting a follow-up into a standalone question."""

    standalone_question: str = Field(description="the latest message rewritten to stand alone")
    needs_clarification: bool = Field(
        default=False, description="true if the message is too ambiguous to resolve"
    )
    clarifying_question: str | None = Field(
        default=None, description="one concise question to ask when clarification is needed"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


_SYSTEM = (
    "You rewrite a retail manager's latest chat message into a STANDALONE analytical "
    'question, using the conversation so far to resolve references like "that", "it", '
    '"them", or elided context ("break it down by month", "what about last year").\n\n'
    "Rules:\n"
    "- If the latest message is already self-contained (a complete question, or a clear "
    "instruction/preference like 'from now on use tables'), return it UNCHANGED with high "
    "confidence.\n"
    "- If it depends on prior context, resolve the references into an explicit, "
    "self-contained question.\n"
    "- Resolve relative dates against today's date ({today}) when possible.\n"
    "- Set `confidence` to how certain the user's intent is. If you must GUESS a referent, "
    "dimension, metric, or comparison target the user did not state or clearly imply "
    "(e.g. 'break it down' without saying by what, or 'compare' without a target), set "
    "confidence to 0.4 or lower.\n"
    "- When confidence is low, set needs_clarification=true and provide ONE concise "
    "clarifying_question instead of a guessed standalone_question.\n"
    "- Never invent specifics the user did not imply."
)


def _format_history(history: list, limit: int) -> str:
    lines = []
    for message in history[-limit:]:
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        lines.append(f"{role}: {as_text(message.content)}")
    return "\n".join(lines)


def contextualize(state: AgentState, deps: AgentDeps) -> dict:
    """Resolve the latest message into a standalone question, or flag clarification."""
    raw_question = state.get("raw_question") or ""
    messages = state.get("messages", [])
    history = messages[:-1] if messages else []

    # First turn / empty history: nothing to resolve against — passthrough, no LLM call.
    if not history:
        return {
            "question": raw_question,
            "history_used": False,
            "needs_clarification": False,
            "clarifying_question": None,
        }

    chat = get_chat_model(temperature=0.0, settings=deps.settings)
    system = _SYSTEM.format(today=date.today().isoformat())
    human = (
        f"Conversation so far:\n{_format_history(history, deps.settings.max_history_messages)}\n\n"
        f"Latest user message: {raw_question}"
    )
    try:
        result: Contextualization = chat.with_structured_output(Contextualization).invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
    except Exception as exc:  # noqa: BLE001 — never break the turn on a rewrite failure
        logger.warning("Contextualize failed (%s); passing the question through", exc)
        return {"question": raw_question, "history_used": False, "needs_clarification": False}

    if (
        result.needs_clarification
        or result.confidence < deps.settings.contextualize_confidence_floor
    ):
        clarifying = (
            result.clarifying_question or "Could you clarify exactly what you'd like to analyze?"
        )
        logger.info("Contextualize -> clarify (confidence %.2f)", result.confidence)
        return {
            "question": raw_question,
            "history_used": True,
            "needs_clarification": True,
            "clarifying_question": clarifying,
        }

    standalone = result.standalone_question.strip() or raw_question
    if standalone != raw_question:
        logger.info("Contextualize rewrote: %r -> %r", raw_question, standalone)
    return {
        "question": standalone,
        "history_used": standalone != raw_question,
        "needs_clarification": False,
        "clarifying_question": None,
    }
