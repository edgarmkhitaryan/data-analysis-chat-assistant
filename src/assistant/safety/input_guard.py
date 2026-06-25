"""Rule-based input guardrail: a cheap, pre-LLM prompt-injection filter (plan/007 §1).

Runs *before* the LLM intent classifier as defense-in-depth: classic injection /
jailbreak / prompt-extraction patterns and blatant non-SELECT SQL attempts are
caught deterministically, so we never spend a model call engaging with them and
never risk the model being talked out of its role. A hit routes the turn straight
to a polite refusal (and is logged as a safety event).

Deliberately conservative — these patterns target manipulation, not data
questions. Note that asking for customer emails/addresses is a *normal* analysis
question here: PII is masked downstream (plan/007 §3), so such asks are answered
(with masked values), never rejected. We only flag attempts to subvert the agent.
"""

import re

# (compiled pattern, short machine-readable reason). Reasons feed logs + the
# rejection_reason metric; the user sees a generic graceful refusal, not these.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Prompt-injection / instruction override.
    (
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\b.{0,30}\b"
            r"(?:instruction|instructions|prompt|prompts|rule|rules|directive|directives)\b",
            re.IGNORECASE,
        ),
        "instruction_override",
    ),
    # System-prompt / instruction extraction.
    (
        re.compile(
            r"\b(?:reveal|show|print|repeat|expose|output|tell me|give me|what(?:'s| is| are))\b"
            r".{0,30}\b(?:system\s+prompt|your\s+(?:prompt|instructions|rules))\b",
            re.IGNORECASE,
        ),
        "prompt_extraction",
    ),
    (re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE), "prompt_extraction"),
    # Jailbreak / role-subversion cues.
    (re.compile(r"\bjailbreak\b", re.IGNORECASE), "jailbreak"),
    (re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE), "jailbreak"),
    (re.compile(r"\bdo\s+anything\s+now\b", re.IGNORECASE), "jailbreak"),
    # Blatant non-SELECT / write SQL against the warehouse. (Note: "delete my
    # reports" is report-management, not this — it has no FROM/TABLE here.)
    (re.compile(r"\b(?:drop|truncate)\s+(?:table|database|schema)\b", re.IGNORECASE), "write_sql"),
    (re.compile(r"\bdelete\s+from\b", re.IGNORECASE), "write_sql"),
    (re.compile(r"\binsert\s+into\b", re.IGNORECASE), "write_sql"),
    (re.compile(r"\bupdate\s+\w+\s+set\b", re.IGNORECASE), "write_sql"),
    (re.compile(r"\b(?:alter|create)\s+(?:table|database|schema)\b", re.IGNORECASE), "write_sql"),
    (re.compile(r"\bgrant\s+(?:all|select|insert|update|delete)\b", re.IGNORECASE), "write_sql"),
]


def injection_check(text: str) -> str | None:
    """Return a short reason if ``text`` matches a known attack pattern, else None."""
    if not text:
        return None
    for pattern, reason in _PATTERNS:
        if pattern.search(text):
            return reason
    return None
