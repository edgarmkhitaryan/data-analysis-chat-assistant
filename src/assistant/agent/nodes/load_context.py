"""Load the org persona and the user's preferences for this turn (the read path).

Runs at the start of the analysis branch so report synthesis can compose the
org's tone (persona) with the manager's format/verbosity preference. Reading
both fresh every turn is what makes persona hot-edits and preference changes take
effect on the very next question.
"""

from assistant.agent.dependencies import AgentDeps
from assistant.agent.state import AgentState
from assistant.persona import load_persona


def load_context(state: AgentState, deps: AgentDeps) -> dict:
    """Populate ``persona`` and ``user_prefs`` from config + the profile store."""
    persona = load_persona(deps.settings.default_persona, deps.settings.personas_dir)
    prefs = deps.profiles.get(state["user_id"])
    return {"persona": persona, "user_prefs": prefs}
