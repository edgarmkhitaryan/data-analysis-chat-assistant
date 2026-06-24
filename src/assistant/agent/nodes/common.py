"""Small shared helpers used by multiple nodes."""

import datetime
import decimal
from typing import Any

import pandas as pd


def as_text(content: Any) -> str:
    """Normalize a LangChain message ``content`` (str or content blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "".join(parts)
    return str(content)


def to_json_safe(value: Any) -> Any:
    """Convert one cell value to a JSON-serializable Python native.

    Handles the types BigQuery/pandas return that ``json`` cannot serialize:
    ``NaN``/``NaT``/``NA`` -> ``None``, timestamps -> ISO strings, ``Decimal`` ->
    ``float``, and numpy scalars -> their Python equivalents.
    """
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return value


def json_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    """Apply :func:`to_json_safe` across a row's values."""
    return {key: to_json_safe(value) for key, value in row.items()}


FORMAT_INSTRUCTIONS = {
    "table": "Present the core results as a Markdown table.",
    "bullets": "Present the answer as concise bullet points.",
    "prose": "Write the answer in short prose paragraphs.",
}

VERBOSITY_INSTRUCTIONS = {
    "concise": "Keep it brief: lead with the headline and only the figures that matter.",
    "detailed": "Provide thorough detail and useful context around the numbers.",
}


def compose_system_prompt(state: dict, base: str) -> str:
    """Compose a report system prompt: base + org persona (tone) + user format/verbosity.

    Shared by the per-question report node and the compound-question synthesizer so
    both honor the same persona and preferences.
    """
    parts = [base]

    persona = state.get("persona")
    if persona is not None:
        if persona.tone:
            parts.append(f"Voice and tone: {persona.tone}")
        if persona.style_rules:
            parts.append("Style rules:\n- " + "\n- ".join(persona.style_rules))
        if persona.guardrails:
            parts.append("Guardrails:\n- " + "\n- ".join(persona.guardrails))

    prefs = state.get("user_prefs")
    if prefs is not None:
        parts.append(FORMAT_INSTRUCTIONS.get(prefs.format, ""))
        parts.append(VERBOSITY_INSTRUCTIONS.get(prefs.verbosity, ""))

    return "\n\n".join(part for part in parts if part)
