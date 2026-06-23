"""In-process vector store for the Golden Bucket (the prototype implementation).

Cosine similarity over a small NumPy matrix — zero-ops and fast enough for tens
of Trios. It sits behind a narrow ``search`` interface so the production swap to
Vertex AI Vector Search is mechanical (see plan/006 §7).
"""

from dataclasses import dataclass

import numpy as np

from assistant.golden.models import Trio


@dataclass(frozen=True)
class ScoredTrio:
    """A retrieved Trio with its cosine similarity to the query."""

    trio: Trio
    score: float


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row so a dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class GoldenStore:
    """Holds Trios and their (normalized) embeddings; ranks them by cosine."""

    def __init__(self, trios: list[Trio], embeddings: np.ndarray) -> None:
        if len(trios) != len(embeddings):
            raise ValueError("trios and embeddings must have the same length")
        self._trios = trios
        self._matrix = _normalize(embeddings.astype(np.float32)) if len(trios) else embeddings

    def __len__(self) -> int:
        return len(self._trios)

    def search(self, query_vector: np.ndarray, k: int = 3, floor: float = 0.0) -> list[ScoredTrio]:
        """Return the top-k Trios with similarity >= ``floor``, highest first."""
        if not self._trios:
            return []
        query = _normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
        similarities = self._matrix @ query
        ranked = np.argsort(-similarities)[:k]
        return [
            ScoredTrio(self._trios[i], float(similarities[i]))
            for i in ranked
            if similarities[i] >= floor
        ]
