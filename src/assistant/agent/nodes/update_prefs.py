"""Persist a standing user preference (the write path), then confirm.

This is a short path that does NOT run the SQL pipeline: it writes the parsed
preference synchronously to the profile store and acknowledges. Because the write
is synchronous and the profile is reloaded at the top of every turn, the change
takes effect on the next question and survives restarts.
"""

from langchain_core.messages import AIMessage

from assistant.agent.dependencies import AgentDeps
from assistant.agent.state import AgentState


def update_prefs(state: AgentState, deps: AgentDeps) -> dict:
    """Save the standing preference in ``pref_update`` and acknowledge it."""
    pref = state.get("pref_update") or {}
    if not pref:
        message = (
            "I couldn't tell which preference to set. Try e.g. "
            '"from now on use tables" or "always keep it brief".'
        )
        return {"report": message, "messages": [AIMessage(content=message)]}

    updated = deps.profiles.update(state["user_id"], **pref)
    changes = ", ".join(f"{key} = {value}" for key, value in pref.items())
    message = f"Done — from now on I'll use {changes} for your reports."
    return {"report": message, "messages": [AIMessage(content=message)], "user_prefs": updated}
