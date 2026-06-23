"""Query-time retrieval over the Golden Bucket.

Embeds the (already standalone) question and returns the most similar Trios above
a configurable floor. If nothing clears the floor the caller gets an empty list —
a "cold" retrieval, which is a useful learning-loop signal (plan/006 §5).
"""

import numpy as np

from assistant.config import Settings, get_settings
from assistant.golden.index import load_or_build_store
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
