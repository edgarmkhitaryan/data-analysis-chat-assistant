"""Factory for the Gemini embedding model used by the Golden Bucket retriever.

``task_type`` matters for retrieval quality: documents (the stored Trios) should
be embedded as ``RETRIEVAL_DOCUMENT`` and live queries as ``RETRIEVAL_QUERY`` so
the vectors live in a comparable space. The default is left unset for general use;
the Golden Bucket (Phase 3) passes the appropriate type at index and query time.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from assistant.config import Settings, get_settings


def get_embedder(
    *,
    task_type: str | None = None,
    settings: Settings | None = None,
) -> GoogleGenerativeAIEmbeddings:
    """Return a configured Gemini embedding model.

    Args:
        task_type: Optional embedding task hint, e.g. ``"RETRIEVAL_DOCUMENT"`` or
            ``"RETRIEVAL_QUERY"``. Left unset for general-purpose embeddings.
        settings: Optional settings override (mainly for tests).
    """
    settings = settings or get_settings()
    kwargs: dict[str, object] = {
        "model": settings.embedding_model,
        "google_api_key": settings.gemini_api_key.get_secret_value(),
    }
    if task_type is not None:
        kwargs["task_type"] = task_type
    return GoogleGenerativeAIEmbeddings(**kwargs)
