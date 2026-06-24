"""Input guard: the first node, classifies intent with the fast LLM (plan/007 §1).

Phase 4 scope: distinguish a normal **analysis** question from a **standing
preference update** (e.g. "from now on use tables"), extracting the preference.
Phase 6 extends this same node with a rule-based prompt-injection pre-filter and
the remaining intents (manage_reports, rejected).

Only a *standing* preference ("from now on", "always", "by default") becomes
``update_preference``; a one-off formatting aside ("show this as a table") stays
an analysis question. Classification failures fall back to analysis so a turn is
never broken by the guard.
"""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from assistant.agent.dependencies import AgentDeps
from assistant.agent.state import AgentState
from assistant.llm import get_chat_model

logger = logging.getLogger(__name__)


class IntentDecision(BaseModel):
    """Structured result of the guard's classification."""

    intent: Literal["analysis", "update_preference"]
    pref_format: Literal["table", "bullets", "prose"] | None = None
    pref_verbosity: Literal["concise", "detailed"] | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reason: str = ""


_GUARD_SYSTEM = (
    "You classify a retail manager's chat message for a data-analysis assistant.\n\n"
    'Return intent = "update_preference" ONLY when the user states a STANDING '
    "preference for how their reports should be formatted from now on (cues like "
    '"from now on", "always", "by default", "going forward"). In that case set:\n'
    "- pref_format: table, bullets, or prose (only if they specify a layout)\n"
    "- pref_verbosity: concise or detailed (only if they specify length)\n\n"
    'Return intent = "analysis" for everything else, including a one-off formatting '
    'request inside a data question (e.g. "show this as a table just this once").'
)


def guard_input(state: AgentState, deps: AgentDeps) -> dict:
    """Classify the turn's intent and, for preferences, extract the change."""
    question = state.get("question", "")
    chat = get_chat_model(temperature=0.0, settings=deps.settings)
    try:
        decision: IntentDecision = chat.with_structured_output(IntentDecision).invoke(
            [SystemMessage(content=_GUARD_SYSTEM), HumanMessage(content=question)]
        )
    except Exception as exc:  # noqa: BLE001 — never let the guard break a turn
        logger.warning("Guard classification failed (%s); defaulting to analysis", exc)
        return {"intent": "analysis", "pref_update": None}

    if decision.intent == "update_preference":
        pref: dict[str, str] = {}
        if decision.pref_format:
            pref["format"] = decision.pref_format
        if decision.pref_verbosity:
            pref["verbosity"] = decision.pref_verbosity
        if pref:
            logger.info("Intent=update_preference %s (%.2f)", pref, decision.confidence)
            return {"intent": "update_preference", "pref_update": pref}

    return {"intent": "analysis", "pref_update": None}
