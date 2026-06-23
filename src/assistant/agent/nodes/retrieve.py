"""Golden Bucket retrieval node (Hybrid Intelligence, Req #1).

Fetches the analyst Trios most similar to the current question. These are
injected downstream into SQL generation (as question->SQL exemplars) and report
synthesis (as question->report style exemplars). Retrieval failures degrade to a
"cold" retrieval rather than breaking the turn.
"""

import logging

from assistant.agent.dependencies import AgentDeps
from assistant.agent.state import AgentState

logger = logging.getLogger(__name__)


def retrieve_golden(state: AgentState, deps: AgentDeps) -> dict:
    """Retrieve relevant Trios for the question; mark a cold retrieval if none."""
    try:
        scored = deps.retriever.retrieve(state["question"])
    except Exception as exc:  # noqa: BLE001 — retrieval must never break the turn
        logger.warning("Golden retrieval failed: %s", exc)
        scored = []

    trios = [hit.trio for hit in scored]
    if trios:
        logger.info(
            "Retrieved %d trios: %s",
            len(trios),
            ", ".join(f"{hit.trio.id}({hit.score:.2f})" for hit in scored),
        )
    return {"retrieved_trios": trios, "retrieval_cold": len(trios) == 0}
