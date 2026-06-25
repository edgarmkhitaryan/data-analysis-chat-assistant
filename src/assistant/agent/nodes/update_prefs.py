"""Persist a standing user preference (the write path), then confirm.

This writes the parsed preference synchronously to the profile store. Because the
write is synchronous and the profile is reloaded at the top of every turn, the
change takes effect on the next question and survives restarts.

The preference is a **side-effect** (plan/005 §3, plan/010 §2): a *preference-only*
message persists and ends here, but a **combined** "set preference + ask analysis"
message (``also_analysis``) persists and then continues into the analysis with the
new preference applied — the analysis is never dropped. In the combined case we
emit no message here; the persisted ack is prepended to the analysis report by the
``synthesize`` node via ``pref_saved_note``.
"""

from langchain_core.messages import AIMessage

from assistant.agent.dependencies import AgentDeps
from assistant.agent.state import AgentState


def update_prefs(state: AgentState, deps: AgentDeps) -> dict:
    """Persist the standing preference; acknowledge it (or hand off to the analysis)."""
    pref = state.get("pref_update") or {}
    if not pref:
        message = (
            "I couldn't tell which preference to set. Try e.g. "
            '"from now on use tables" or "always keep it brief".'
        )
        return {"report": message, "messages": [AIMessage(content=message)]}

    updated = deps.profiles.update(state["user_id"], **pref)
    changes = ", ".join(f"{key} = {value}" for key, value in pref.items())

    # Combined intent: persist, then let the analysis run; surface the save as a note
    # prepended to the eventual report rather than a separate terminal message.
    if state.get("also_analysis"):
        return {
            "user_prefs": updated,
            "pref_saved_note": f"_Saved your preference ({changes})._",
        }

    message = f"Done — from now on I'll use {changes} for your reports."
    return {"report": message, "messages": [AIMessage(content=message)], "user_prefs": updated}
