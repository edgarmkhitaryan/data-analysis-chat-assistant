"""Query-time retrieval over the Golden Bucket.

Embeds the (already standalone) question and returns the most similar Trios above
a configurable floor. If nothing clears the floor the caller gets an empty list —
a "cold" retrieval, which is a useful learning-loop signal (plan/006 §5).
"""

import numpy as np

from assistant.config import Settings, get_settings
from assistant.golden.index import load_or_build_store
from assistant.golden.models import Trio
from assistant.golden.store import GoldenStore, ScoredTrio
from assistant.llm import get_embedder


class GoldenRetriever:
    """Embeds questions and ranks Trios by cosine similarity."""

    def __init__(self, store: GoldenStore, settings: Settings) -> None:
        self._store = store
        self._settings = settings
        self._embedder = get_embedder(task_type="RETRIEVAL_QUERY", settings=settings)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "GoldenRetriever":
        """Build a retriever, loading (or building) the index from configured paths."""
        settings = settings or get_settings()
        store = load_or_build_store(settings.golden_trios_dir, settings.golden_index_dir, settings)
        return cls(store, settings)

    def __len__(self) -> int:
        return len(self._store)

    def retrieve(
        self, question: str, k: int | None = None, floor: float | None = None
    ) -> list[ScoredTrio]:
        """Return the top Trios for a question (empty list = cold retrieval)."""
        if len(self._store) == 0:
            return []
        k = k or self._settings.golden_top_k
        floor = self._settings.golden_sim_floor if floor is None else floor
        vector = np.asarray(self._embedder.embed_query(question), dtype=np.float32)
        return self._store.search(vector, k=k, floor=floor)

    def _embed_document(self, text: str) -> np.ndarray:
        embedder = get_embedder(task_type="RETRIEVAL_DOCUMENT", settings=self._settings)
        return np.asarray(embedder.embed_documents([text])[0], dtype=np.float32)

    def max_document_similarity(self, text: str) -> float:
        """Highest cosine similarity of ``text`` to any stored Trio (0 if empty).

        Embeds ``text`` as a *document* (same task type as the stored Trios) so an
        identical question scores ~1.0 — the right signal for the learning loop's
        deduplication gate (a query-type embedding of identical text only scores ~0.9).
        """
        if len(self._store) == 0:
            return 0.0
        hits = self._store.search(self._embed_document(text), k=1, floor=0.0)
        return hits[0].score if hits else 0.0

    def add_trio(self, trio: Trio) -> None:
        """Embed and append a Trio to the in-memory store (immediately retrievable)."""
        self._store.add(trio, self._embed_document(trio.embedding_text))
