"""Gemini client factories (chat + embeddings)."""

from assistant.llm.client import get_chat_model
from assistant.llm.embeddings import get_embedder

__all__ = ["get_chat_model", "get_embedder"]
