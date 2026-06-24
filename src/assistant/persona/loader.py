"""Persona loading with hot-reload (Requirement 8 — Agility).

Tone/voice is config, not code: personas live in editable YAML files. The loader
caches per persona and watches the file's mtime, so editing a YAML changes the
agent's tone on the **next turn** with no restart. A malformed or missing file
falls back to the last-known-good persona (or a built-in default), so a bad edit
never takes the agent down.
"""

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Persona(BaseModel):
    """An org-level voice/tone definition (set by the CEO, changed weekly)."""

    name: str
    display_name: str = ""
    tone: str = ""
    audience: str = ""
    style_rules: list[str] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)
    version: int = 1


# Built-in safety net if no YAML can be loaded at all.
_FALLBACK_PERSONA = Persona(
    name="default",
    display_name="Default",
    tone="Clear, professional, and concise.",
    guardrails=["Never claim data you did not query."],
)

# Cache: persona name -> (file mtime, parsed Persona).
_cache: dict[str, tuple[float, Persona]] = {}


def load_persona(name: str, personas_dir: str) -> Persona:
    """Load a persona by name, honoring hot edits and degrading gracefully."""
    path = Path(personas_dir) / f"{name}.yaml"
    if not path.exists():
        logger.warning("Persona file %s not found; using fallback", path)
        return _cache.get(name, (0.0, _FALLBACK_PERSONA))[1]

    mtime = path.stat().st_mtime
    cached = _cache.get(name)
    if cached and cached[0] == mtime:
        return cached[1]

    try:
        data = yaml.safe_load(path.read_text()) or {}
        persona = Persona(**data)
    except Exception as exc:  # noqa: BLE001 — a bad edit must not break the agent
        logger.warning("Failed to load persona %s (%s); keeping last-known-good", path, exc)
        return cached[1] if cached else _FALLBACK_PERSONA

    _cache[name] = (mtime, persona)
    return persona
