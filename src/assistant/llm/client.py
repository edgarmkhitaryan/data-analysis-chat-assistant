"""Factory for the Gemini chat model used across the agent's nodes.

Centralizing model construction here means model choice, credentials, and retry
policy are configured in exactly one place. Nodes ask for ``get_chat_model()``
(fast default) or ``get_chat_model(heavy=True)`` (the escalation model for hard
reasoning) and never touch credentials or provider details.
"""

from langchain_google_genai import ChatGoogleGenerativeAI

from assistant.config import Settings, get_settings

# Conservative built-in handling of transient API errors (rate limits, 5xx).
# Phase 8 layers tenacity backoff + a circuit breaker on top of this baseline.
_REQUEST_TIMEOUT_S = 60.0
_MAX_RETRIES = 3


def get_chat_model(
    *,
    heavy: bool = False,
    temperature: float = 0.0,
    settings: Settings | None = None,
) -> ChatGoogleGenerativeAI:
    """Return a configured Gemini chat model.

    Args:
        heavy: Use the heavier escalation model (``LLM_MODEL_HEAVY``) instead of
            the fast default (``LLM_MODEL``).
        temperature: Sampling temperature; defaults to 0.0 for deterministic
            SQL/report generation.
        settings: Optional settings override (mainly for tests).
    """
    settings = settings or get_settings()
    model_name = settings.llm_model_heavy if heavy else settings.llm_model
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.gemini_api_key.get_secret_value(),
        temperature=temperature,
        timeout=_REQUEST_TIMEOUT_S,
        max_retries=_MAX_RETRIES,
    )
