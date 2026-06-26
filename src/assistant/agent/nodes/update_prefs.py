"""Persist a standing user preference (the write path), then confirm.

Preferences are a single **compact free-form description** ([010 §2](.)). When the
user states a new standing preference we *merge* it into their existing description
with a small LLM call (override what it contradicts, keep the rest, stay concise),
then persist synchronously — so it applies on the next question and survives restarts.

The preference is a **side-effect** (plan/005 §3, plan/010 §2): a *preference-only*
message persists and ends here, while a **combined** "set preference + ask analysis"
message (``also_analysis``) persists and then continues into the analysis with the
new preference applied — the analysis is never dropped. In the combined case we emit
no message here; the ack is prepended to the report by ``synthesize`` via
``pref_saved_note``.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from assistant.agent.dependencies import AgentDeps
from assistant.agent.nodes.common import as_text
from assistant.agent.state import AgentState
from assistant.llm import get_chat_model, resilient_invoke

_MERGE_SYSTEM = (
    "You maintain a COMPACT, plain-text description of a retail manager's standing "
    "preferences for how their data-analysis reports should be written and what they "
    "care about. Given the CURRENT description and a NEW instruction the user just gave, "
    "return the UPDATED description: merge the new instruction in, override anything it "
    "contradicts, keep everything still relevant, and stay concise (a few short sentences "
    "or a compact list). Output ONLY the updated description text — no preamble."
)


def _fallback_merge(current: str, instruction: str) -> str:
    """Deterministic merge if the LLM is unavailable: append the new instruction."""
    return f"{current}\n{instruction}".strip() if current else instruction.strip()


def _merge_preferences(current: str, instruction: str, deps: AgentDeps) -> str:
    """Fold ``instruction`` into the ``current`` compact preference text (LLM, with fallback)."""
    chat = get_chat_model(temperature=0.0, settings=deps.settings)
    human = f"Current preferences: {current or '(none yet)'}\n\nNew instruction: {instruction}"
    try:
        reply = resilient_invoke(
            chat,
            [SystemMessage(content=_MERGE_SYSTEM), HumanMessage(content=human)],
            settings=deps.settings,
        )
        merged = as_text(reply.content).strip()
        return merged or _fallback_merge(current, instruction)
    except Exception:  # noqa: BLE001 — never lose the preference if the merge call fails
        return _fallback_merge(current, instruction)


def update_prefs(state: AgentState, deps: AgentDeps) -> dict:
    """Merge + persist the standing preference; acknowledge it (or hand off to the analysis)."""
    instruction = (state.get("pref_update") or "").strip()
    if not instruction:
        message = (
            "I couldn't tell which preference to set. Try e.g. "
            '"from now on use tables" or "always include % change vs last quarter".'
        )
        return {"report": message, "messages": [AIMessage(content=message)]}

    current = deps.profiles.get(state["user_id"]).preferences
    merged = _merge_preferences(current, instruction, deps)
    updated = deps.profiles.set_preferences(state["user_id"], merged)

    # Combined intent: persist, then let the analysis run; surface the save as a note
    # prepended to the eventual report rather than a separate terminal message.
    if state.get("also_analysis"):
        return {"user_prefs": updated, "pref_saved_note": "_Saved your preference._"}

    message = f"Done — I've updated your preferences:\n\n{updated.preferences}"
    return {"report": message, "messages": [AIMessage(content=message)], "user_prefs": updated}
