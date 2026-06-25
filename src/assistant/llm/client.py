"""Factory for the Gemini chat model used across the agent's nodes.

Centralizing model construction here means model choice, credentials, and retry
policy are configured in exactly one place. Nodes ask for ``get_chat_model()``
(fast default) or ``get_chat_model(heavy=True)`` (the escalation model for hard
reasoning) and never touch credentials or provider details.

LLM calls go through :func:`resilient_invoke`, which adds tenacity backoff + a
shared circuit breaker (plan/008 §3). The model's own built-in retry is therefore
disabled (``max_retries=0``) so we have a single, observable retry layer rather
than two multiplying each other.
"""

from typing import Any

from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from assistant.config import Settings, get_settings
from assistant.resilience import CircuitBreaker, resilient_call

_REQUEST_TIMEOUT_S = 60.0

# One process-wide breaker for the Gemini dependency (built lazily from settings).
_llm_breaker: CircuitBreaker | None = None


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
        max_retries=0,  # tenacity owns retries (see resilient_invoke)
    )


def _breaker(settings: Settings) -> CircuitBreaker:
    global _llm_breaker
    if _llm_breaker is None:
        _llm_breaker = CircuitBreaker(
            "gemini",
            threshold=settings.circuit_breaker_threshold,
            cooldown_s=settings.circuit_breaker_cooldown_seconds,
        )
    return _llm_breaker


def resilient_invoke(runnable: Runnable, messages: Any, *, settings: Settings | None = None) -> Any:
    """Invoke a chat runnable with retry-on-transient + circuit breaker (plan/008 §3).

    Works for a plain chat model *and* a ``with_structured_output`` runnable — it
    just wraps ``.invoke``. Permanent errors (auth/4xx) fail fast; an open breaker
    fails fast so callers can degrade gracefully.
    """
    settings = settings or get_settings()
    return resilient_call(
        lambda: runnable.invoke(messages),
        breaker=_breaker(settings),
        max_attempts=settings.llm_max_retries,
        base_delay=settings.llm_retry_base_delay,
    )
