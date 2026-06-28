"""Unit tests for model-tier selection in the chat-model factory (cost optimization).

No network: ChatGoogleGenerativeAI construction is lazy, so we can assert which model
name each tier resolves to without spending quota.
"""

from assistant.config import Settings
from assistant.llm import get_chat_model


def _settings() -> Settings:
    return Settings(
        GEMINI_API_KEY="x",
        GOOGLE_CLOUD_PROJECT="y",
        LLM_MODEL="gemini-main",
        LLM_MODEL_CHEAP="gemini-cheap",
    )


def test_default_uses_main_model():
    assert "gemini-main" in get_chat_model(settings=_settings()).model


def test_cheap_uses_cheap_model():
    assert "gemini-cheap" in get_chat_model(cheap=True, settings=_settings()).model
