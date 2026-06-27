"""Unit tests for the objective reference-aggregate cross-check (plan/011 §2).

Pure logic — proves the correctness check works without BigQuery or Gemini quota.
"""

from assistant.eval.correctness import compare_aggregates
from assistant.eval.harness import CaseResult, summarize


def test_exact_match_all_reproduced():
    ref = [{"cat": "Jeans", "revenue": 1000.0}, {"cat": "Sweaters", "revenue": 800.0}]
    agent = [{"category": "Jeans", "total": 1000.0}, {"category": "Sweaters", "total": 800.0}]
    check = compare_aggregates(ref, agent)
    assert check.ran and check.ok and check.matched == 2 and check.total == 2


def test_within_tolerance_counts_as_match():
    ref = [{"x": 1000.0}]
    agent = [{"y": 1005.0}]  # 0.5% off
    assert compare_aggregates(ref, agent, tolerance=0.01).ok
    assert not compare_aggregates(ref, agent, tolerance=0.001).ok


def test_mismatch_when_a_figure_is_wrong():
    ref = [{"a": 100.0}, {"b": 200.0}]
    agent = [{"a": 100.0}, {"b": 999.0}]
    check = compare_aggregates(ref, agent)
    assert not check.ok and check.matched == 1 and check.total == 2


def test_string_dimensions_and_booleans_are_ignored():
    # Only numeric measures are compared; the string dimension must not become an aggregate.
    ref = [{"category": "Jeans", "revenue": 500.0, "flag": True}]
    agent = [{"name": "Jeans", "revenue": 500.0}]
    check = compare_aggregates(ref, agent)
    assert check.ok and check.total == 1


def test_empty_reference_is_not_ok():
    check = compare_aggregates([{"label": "x"}], [{"revenue": 1.0}])
    assert check.ran and not check.ok and check.total == 0


def test_summarize_surfaces_disagreements():
    """Judge happy (intent>=4) but reference mismatch -> flagged as a disagreement."""
    results = [
        CaseResult(
            "good",
            "analysis",
            "analysis",
            True,
            5,
            True,
            True,
            intent_satisfaction=5,
            faithfulness=5,
            reference_attempted=True,
            reference_ran=True,
            reference_ok=True,
            reference_detail="2/2 reference aggregates reproduced (±2%)",
        ),
        CaseResult(
            "sneaky",
            "analysis",
            "analysis",
            True,
            5,
            True,
            True,
            intent_satisfaction=5,
            faithfulness=5,
            reference_attempted=True,
            reference_ran=True,
            reference_ok=False,
            reference_detail="1/2 reference aggregates reproduced (±2%)",
        ),
    ]
    text, _ = summarize(results)
    assert "Disagreements" in text
    assert "sneaky" in text
    assert "ref=MISMATCH" in text
    assert "Reference correctness: 1/2" in text
