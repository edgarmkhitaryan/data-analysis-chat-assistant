"""Automatic learning loop (plan/010 §3): capture → gate → promote, in the background.

Every completed analysis turn becomes a *candidate* Trio. A fully automatic gate —
**deterministic metrics → deduplication/novelty → LLM-as-judge faithfulness** —
decides promotion. There is **no user feedback, no manual trigger, and no human in
the loop**; the gate runs on a worker thread after the answer is shown, so it never
adds latency. Approved candidates become ``source="learned"`` Trios (reversible by
id) and are retrievable immediately (in-memory) and on the next start (on disk).

The decision is split out as the pure :func:`decide` so it is unit-tested without
quota; :func:`promote_if_qualified` wires in the real retriever (dedup) and judge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from assistant.eval.judge import judge_report
from assistant.golden.models import Trio

logger = logging.getLogger(__name__)

# A neutral rubric for the loop's judge (no per-case rubric here). It must demand
# completeness as well as faithfulness so a faithful-but-incomplete answer (e.g. a
# comparison that silently dropped one side) scores low on intent-satisfaction.
_LEARNING_RUBRIC = (
    "A good answer fully addresses every part of the question with concrete figures, leaves "
    "no requested part unanswered or marked 'not available'/'no data', and makes no "
    "fabricated or unsupported claims."
)


@dataclass
class Candidate:
    """A captured turn that *could* become a learned Trio (no user input involved)."""

    run_id: str
    question: str
    sql: str
    report: str
    succeeded: bool
    attempts: int
    row_count: int
    pii_leak_prevented: int
    # The PII-masked rows the report stood on, so the judge can ground faithfulness in the
    # actual data (not just internal consistency). Defaults empty for back-compat.
    rows: list[dict] = field(default_factory=list)


@dataclass
class GateResult:
    """The automatic gate's verdict."""

    approved: bool
    reasons: list[str] = field(default_factory=list)
    trio_id: str | None = None


def _check_metrics(candidate: Candidate, max_attempts: int) -> GateResult:
    """Stage 1 — deterministic metrics (no LLM, no embeddings)."""
    reasons: list[str] = []
    if not candidate.succeeded:
        reasons.append("turn did not succeed")
    if not (candidate.sql and candidate.report):
        reasons.append("missing SQL or report")
    if candidate.row_count <= 0:
        reasons.append("query returned no rows")
    if candidate.attempts > max_attempts:
        reasons.append(f"too many self-corrections ({candidate.attempts} > {max_attempts})")
    if candidate.pii_leak_prevented > 0:
        reasons.append("a PII leak was prevented (safety incident)")
    return GateResult(approved=not reasons, reasons=reasons)


def decide(
    candidate: Candidate,
    max_similarity: float,
    faithfulness: int,
    intent_satisfaction: int,
    *,
    dedup_threshold: float,
    faithfulness_bar: int,
    intent_bar: int,
    max_attempts: int = 2,
) -> GateResult:
    """Pure promotion decision given precomputed dedup similarity + judge scores.

    Order matters (cheapest, most decisive first): metrics, then novelty, then the
    LLM-judge bars — BOTH faithfulness AND intent-satisfaction, so a faithful but
    incomplete answer (one that didn't actually answer the question) is not learned.
    """
    metrics = _check_metrics(candidate, max_attempts)
    if not metrics.approved:
        return metrics
    if max_similarity >= dedup_threshold:
        return GateResult(
            False, [f"near-duplicate of an existing Trio (similarity {max_similarity:.2f})"]
        )
    if faithfulness < faithfulness_bar:
        return GateResult(False, [f"faithfulness {faithfulness} below bar {faithfulness_bar}"])
    if intent_satisfaction < intent_bar:
        return GateResult(
            False, [f"intent-satisfaction {intent_satisfaction} below bar {intent_bar}"]
        )
    return GateResult(True)


def promote_candidate(candidate: Candidate, deps) -> str:
    """Write the learned Trio (persisted for next start) and add it to the live index."""
    settings = deps.settings
    trio_id = f"learned_{candidate.run_id}"
    trio = Trio(
        id=trio_id,
        question=candidate.question,
        sql=candidate.sql,
        report=candidate.report,
        tags=["learned"],
        created_by="learning_loop",
        created_at=datetime.now(UTC).date().isoformat(),
        quality="approved",
        source="learned",
    )
    directory = Path(settings.golden_trios_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{trio_id}.json").write_text(trio.model_dump_json(indent=2), encoding="utf-8")
    deps.retriever.add_trio(trio)  # retrievable this session; disk index rebuilds next start
    return trio_id


def promote_if_qualified(candidate: Candidate, deps, judge_fn=judge_report) -> GateResult:
    """Run the full automatic gate and promote if it passes (the background entry point)."""
    settings = deps.settings

    # Stage 1 — deterministic metrics (no LLM, no embeddings).
    metrics = _check_metrics(candidate, max_attempts=settings.learning_max_attempts)
    if not metrics.approved:
        logger.info("learning: discard %s — %s", candidate.run_id, "; ".join(metrics.reasons))
        return metrics

    # Stage 2 — deduplication (one embedding). Short-circuits BEFORE the judge so the
    # expensive LLM call only runs on genuinely novel questions (plan/010 §3).
    max_similarity = deps.retriever.max_document_similarity(candidate.question)
    if max_similarity >= settings.learning_dedup_similarity:
        reason = f"near-duplicate of an existing Trio (similarity {max_similarity:.2f})"
        logger.info("learning: discard %s — %s", candidate.run_id, reason)
        return GateResult(False, [reason])

    # Stage 3 — LLM-as-judge: must clear BOTH faithfulness AND intent-satisfaction
    # (a faithful but incomplete answer must not be learned). Faithfulness is grounded in
    # the actual masked rows the report stood on, so a fabricated figure is caught.
    score = judge_fn(
        candidate.question,
        candidate.report,
        _LEARNING_RUBRIC,
        settings,
        data=candidate.rows,
        sql=candidate.sql,
    )
    if score.faithfulness < settings.learning_faithfulness_bar:
        reason = f"faithfulness {score.faithfulness} below bar {settings.learning_faithfulness_bar}"
        logger.info("learning: discard %s — %s", candidate.run_id, reason)
        return GateResult(False, [reason])
    if score.intent_satisfaction < settings.learning_intent_bar:
        reason = (
            f"intent-satisfaction {score.intent_satisfaction} below bar "
            f"{settings.learning_intent_bar}"
        )
        logger.info("learning: discard %s — %s", candidate.run_id, reason)
        return GateResult(False, [reason])

    trio_id = promote_candidate(candidate, deps)
    logger.info(
        "learning: promoted %s -> %s (similarity %.2f, faithfulness %d, intent %d)",
        candidate.run_id,
        trio_id,
        max_similarity,
        score.faithfulness,
        score.intent_satisfaction,
    )
    return GateResult(True, trio_id=trio_id)
