"""Unit tests for the automatic learning-loop gate (plan/010 §3, plan/011 §1.1).

Pure decision logic — no network. The gate is metrics -> dedup -> LLM-judge; the
pure :func:`decide` takes precomputed similarity + faithfulness so the whole policy
is testable without quota. (Live dedup/judge/promote are verified separately.)
"""

from assistant.memory.feedback import Candidate, decide

DEDUP = 0.93
BAR = 4


def _candidate(**overrides) -> Candidate:
    base = {
        "run_id": "r1",
        "question": "top products by revenue",
        "sql": "SELECT 1",
        "report": "Top product is X at $5,000.",
        "succeeded": True,
        "attempts": 1,
        "row_count": 5,
        "pii_leak_prevented": 0,
    }
    base.update(overrides)
    return Candidate(**base)


def _verdict(candidate=None, max_similarity=0.10, faithfulness=5, intent_satisfaction=5):
    return decide(
        candidate or _candidate(),
        max_similarity,
        faithfulness,
        intent_satisfaction,
        dedup_threshold=DEDUP,
        faithfulness_bar=BAR,
        intent_bar=BAR,
    )


def test_promotes_clean_novel_faithful():
    assert _verdict().approved


def test_metrics_failures_block_before_dedup_or_judge():
    # Even with perfect similarity/faithfulness, bad metrics veto.
    assert not _verdict(_candidate(succeeded=False)).approved
    assert not _verdict(_candidate(row_count=0)).approved
    assert not _verdict(_candidate(attempts=3)).approved
    assert not _verdict(_candidate(pii_leak_prevented=1)).approved
    assert not _verdict(_candidate(sql="")).approved


def test_duplicate_is_not_promoted():
    verdict = _verdict(max_similarity=0.97)  # >= dedup threshold
    assert not verdict.approved
    assert "duplicate" in verdict.reasons[0].lower()


def test_just_below_dedup_threshold_is_novel():
    assert _verdict(max_similarity=0.92).approved


def test_low_faithfulness_is_not_promoted():
    verdict = _verdict(faithfulness=3)  # below bar of 4
    assert not verdict.approved
    assert "faithfulness" in verdict.reasons[0].lower()


def test_faithfulness_at_bar_is_promoted():
    assert _verdict(faithfulness=4).approved


def test_low_intent_satisfaction_is_not_promoted():
    # Faithful but incomplete (e.g. a comparison that silently dropped one side).
    verdict = _verdict(intent_satisfaction=3)  # below bar of 4
    assert not verdict.approved
    assert "intent" in verdict.reasons[0].lower()
