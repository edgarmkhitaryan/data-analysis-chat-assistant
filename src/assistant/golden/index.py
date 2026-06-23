"""Building, caching, and loading the Golden Bucket embedding index.

The index is a built artifact (embeddings + a copy of the Trios + a manifest) so
we never re-embed on every start. A content fingerprint detects when the source
Trios changed, so the index rebuilds itself automatically when stale — and
``scripts/ingest_golden.py`` can force a rebuild explicitly.
"""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from assistant.config import Settings, get_settings
from assistant.golden.models import Trio
from assistant.golden.store import GoldenStore
from assistant.llm import get_embedder

logger = logging.getLogger(__name__)

_EMBEDDINGS_FILE = "embeddings.npy"
_TRIOS_FILE = "trios.json"
_MANIFEST_FILE = "manifest.json"


def load_trios(trios_dir: str) -> list[Trio]:
    """Load and validate every Trio JSON in a directory, ordered by id."""
    paths = sorted(Path(trios_dir).glob("*.json"))
    return [Trio.model_validate_json(path.read_text()) for path in paths]


def _fingerprint(trios: list[Trio]) -> str:
    """A stable hash of the Trios' identity + embedded text, to detect changes."""
    digest = hashlib.sha256()
    for trio in trios:
        digest.update(trio.id.encode())
        digest.update(b"\0")
        digest.update(trio.embedding_text.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def build_index(trios_dir: str, index_dir: str, settings: Settings | None = None) -> int:
    """Embed all Trios and write the index to ``index_dir``. Returns the count."""
    settings = settings or get_settings()
    trios = load_trios(trios_dir)
    if not trios:
        raise FileNotFoundError(f"No Trio JSON files found in {trios_dir}")

    embedder = get_embedder(task_type="RETRIEVAL_DOCUMENT", settings=settings)
    vectors = embedder.embed_documents([trio.embedding_text for trio in trios])
    matrix = np.asarray(vectors, dtype=np.float32)

    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / _EMBEDDINGS_FILE, matrix)
    (out / _TRIOS_FILE).write_text(
        json.dumps([trio.model_dump() for trio in trios], indent=2, ensure_ascii=False)
    )
    (out / _MANIFEST_FILE).write_text(
        json.dumps(
            {
                "count": len(trios),
                "dim": int(matrix.shape[1]),
                "model": settings.embedding_model,
                "fingerprint": _fingerprint(trios),
            },
            indent=2,
        )
    )
    logger.info("Built Golden Bucket index: %d trios -> %s", len(trios), index_dir)
    return len(trios)


def _is_stale(trios_dir: str, index_dir: str) -> bool:
    manifest = Path(index_dir) / _MANIFEST_FILE
    if not manifest.exists() or not (Path(index_dir) / _EMBEDDINGS_FILE).exists():
        return True
    try:
        recorded = json.loads(manifest.read_text()).get("fingerprint")
    except (json.JSONDecodeError, OSError):
        return True
    return recorded != _fingerprint(load_trios(trios_dir))


def load_store(index_dir: str) -> GoldenStore:
    """Load a :class:`GoldenStore` from a previously built index."""
    out = Path(index_dir)
    matrix = np.load(out / _EMBEDDINGS_FILE)
    trios = [Trio.model_validate(d) for d in json.loads((out / _TRIOS_FILE).read_text())]
    return GoldenStore(trios, matrix)


def load_or_build_store(
    trios_dir: str, index_dir: str, settings: Settings | None = None
) -> GoldenStore:
    """Load the index, rebuilding it first if missing or out of date."""
    if _is_stale(trios_dir, index_dir):
        logger.info("Golden Bucket index missing or stale; rebuilding from %s", trios_dir)
        build_index(trios_dir, index_dir, settings)
    return load_store(index_dir)
